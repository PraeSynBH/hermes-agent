"""PRX-5717: a profile's local-session recovery must not run its store reads on the event loop.

Restoring one local receipt costs several synchronous SQLite reads (the receipt, its session row,
the lineage tip, its admissions). ``initialize_session_authority`` therefore read one query per
local receipt on the loop, and every profile's ``state.db`` carries thousands of receipts: on a
loaded host this stalled every other profile's turns for 45-122 s while a ``rescan-profiles`` added
five profiles (measured 09:57-10:08Z, 2026-10-20).

The contract: the store work hops to a worker, and only the task scheduling stays on the loop —
``authority._schedule`` creates tasks, so it may not run on a thread.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import session_local_recovery as recovery


def _authority(scheduled: list, *, loop_thread: int | None = None) -> SimpleNamespace:
    def _schedule(ref):
        assert loop_thread is None or threading.get_ident() == loop_thread, (
            "authority._schedule creates tasks: it must run on the event loop")
        scheduled.append(ref)

    return SimpleNamespace(profile_id="p", db=object(), _schedule=_schedule)


def test_store_reads_do_not_block_the_loop(monkeypatch):
    """While a receipt read is in flight the loop keeps running other work.

    The read is parked on an event (no sleeping): if the recovery runs it inline, the loop never
    reaches the assertion below and the read only unblocks on its own timeout — the test then fails
    on ``task.done()``. With the store work on a worker, the loop is free the whole time.
    """
    started = threading.Event()
    release = threading.Event()

    async def _scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()

        def slow_receipt_ids(_authority):
            loop.call_soon_threadsafe(entered.set)
            release.wait(timeout=5.0)
            return []

        monkeypatch.setattr(recovery, "local_receipt_ids", slow_receipt_ids)
        # The reconciliation (PRX-6097) is a store read of its own, not what this test measures.
        monkeypatch.setattr(recovery, "drop_unrestorable_receipts", lambda _authority: [])
        monkeypatch.setattr(recovery, "drop_unreconstructable_receipts", lambda _authority: [])
        task = asyncio.create_task(recovery.recover_local_sessions_offloop(_authority([])))
        task.add_done_callback(lambda _t: started.set())
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        assert not task.done(), "the recovery was still reading, so the loop was free"
        release.set()
        await asyncio.wait_for(task, timeout=5.0)
        return True

    assert asyncio.run(_scenario()) is True
    assert started.is_set()


def test_queued_turns_are_scheduled_on_the_loop_after_the_store_work(monkeypatch):
    """Splitting the read out must not drop the drain scheduling: every prepared ref is scheduled,
    and on the loop's own thread."""
    refs = [SimpleNamespace(name="one"), SimpleNamespace(name="two")]
    threads: list[int] = []

    def prepare(_authority, schedule):
        assert schedule is True
        threads.append(threading.get_ident())
        return refs, []

    monkeypatch.setattr(recovery, "_recover_local_store", prepare)
    scheduled: list = []

    async def _scenario():
        loop_thread = threading.get_ident()
        await recovery.recover_local_sessions_offloop(
            _authority(scheduled, loop_thread=loop_thread), schedule=True)
        return loop_thread

    loop_thread = asyncio.run(_scenario())
    assert scheduled == refs
    assert threads and threads[0] != loop_thread, "the store work ran on the event loop"
