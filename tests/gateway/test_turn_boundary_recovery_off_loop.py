"""PRX-5780: the two remaining SQLite-on-the-loop recovery sites run on a worker thread.

Both were sampled in PRX-5524's blocked-loop dumps after the first fixes landed:

- ``session_authority._drain`` → ``restore_local_session``: a local session's receipt, session row,
  lineage tip, admissions and frozen policy are re-read at EVERY turn boundary, on the loop, for
  every local session — a per-turn stall, not a per-tick one.
- ``serve_profile_runtime`` → the API-turn recovery pass: the media sweeps, the queued-target query
  and a restore per target, all synchronous SQLite, inside the 30 s profile reconcile tick.

The contract each test pins: the store work runs off the owner loop, and whatever needs the loop
(``authority._schedule`` creates tasks) still runs on it.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from gateway import session_api_turn
from gateway import session_local_recovery
from gateway.session_authority import SessionAuthority


def test_turn_boundary_restore_runs_on_a_worker(monkeypatch):
    """``_restore_local_turn`` hands the receipt read to a worker, and refuses on the loop's behalf.

    A refusal (a receipt that is not this owner's) must surface to the drain unchanged: it is what
    pauses the session with a named reason, so the hop must not swallow it.
    """
    seen: list[int] = []

    def fake_restore(authority, sid):
        seen.append(threading.get_ident())
        if sid == 'refused':
            raise session_local_recovery._ReceiptRefused('storage_unavailable')

    monkeypatch.setattr(session_local_recovery, 'restore_local_session', fake_restore)

    async def _scenario():
        loop_thread = threading.get_ident()
        # A real instance with no initialisation: the method only forwards ``self`` to the restore.
        authority = object.__new__(SessionAuthority)
        await authority._restore_local_turn(SimpleNamespace(session_id='ok'))
        try:
            await authority._restore_local_turn(SimpleNamespace(session_id='refused'))
        except session_local_recovery._ReceiptRefused:
            refused = True
        else:
            refused = False
        return loop_thread, refused

    loop_thread, refused = asyncio.run(_scenario())
    assert refused, 'a refused receipt must reach the drain, not be swallowed by the thread hop'
    assert seen and all(thread != loop_thread for thread in seen), (
        'the receipt read ran on the event loop')


def test_serve_path_api_recovery_runs_on_a_worker_and_schedules_on_the_loop(monkeypatch):
    """The hot-serve pass reads its store off-loop and only schedules on the loop's thread."""
    refs = [SimpleNamespace(name='one'), SimpleNamespace(name='two')]
    store_threads: list[int] = []

    def fake_store(authority):
        store_threads.append(threading.get_ident())
        return refs

    monkeypatch.setattr(session_api_turn, '_recoverable_api_sessions', fake_store)
    scheduled: list = []

    async def _scenario():
        loop_thread = threading.get_ident()

        def _schedule(ref):
            assert threading.get_ident() == loop_thread, (
                'authority._schedule creates tasks: it must run on the event loop')
            scheduled.append(ref)

        authority = SimpleNamespace(_schedule=_schedule)
        await session_api_turn.recover_authority_api_turns_offloop(object(), authority)
        return loop_thread

    loop_thread = asyncio.run(_scenario())
    assert scheduled == refs
    assert store_threads and store_threads[0] != loop_thread, (
        'the API-turn recovery read the store on the event loop')


def test_inline_flavour_still_schedules_every_ref(monkeypatch):
    """Boot's own-thread pass keeps its old behaviour: every prepared ref is scheduled."""
    refs = [SimpleNamespace(name='one')]
    scheduled: list = []
    monkeypatch.setattr(session_api_turn, '_recoverable_api_sessions', lambda authority: refs)
    session_api_turn.recover_authority_api_turns(
        object(), SimpleNamespace(_schedule=scheduled.append))
    assert scheduled == refs
