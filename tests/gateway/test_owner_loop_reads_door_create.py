"""The three remaining session-create doors reach no writer-locked read on the owner loop.

Regression for PRX-6024, the sibling class of PRX-6015. ``session.create`` (PRA-6517) and the Bot
Chat mint (``session_bot._bot_chat_step``) already read and stage in a worker; these doors kept the
shape -- ``get_session`` / ``get_session_by_title`` / ``stage_local_session`` in an ``async def`` on
the owner loop. With the read pool offering nothing and another thread holding ``SessionDB._lock``,
the documented degradation (``hermes_state._read_ctx``) takes the WRITER connection under that lock,
so the whole profile's loop -- sessions, timers, sockets -- freezes for the length of the hold:

* ``gateway/session_hosted_rpc.py`` ``_create`` (hosted room / group-chat member create)
* ``gateway/session_a2a.py`` ``forward`` (peer forward into its canonical context session)
* ``gateway/session_kanban.py`` ``run_task`` (dispatcher card attempt)

The contract for all three: the create must not stall the loop behind a held writer, and must still
return (and durably commit) its session. Real SessionStore/SessionAuthority; no endpoint contacted.
"""
import asyncio
import threading
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest


def _release_read_pool(authority):
    """The state that sends a read to the writer connection: the pool offers nothing to borrow,
    and on the loop the pool refuses to open one."""
    while authority.db._evict_one_idle_read_conn():
        pass


def _hold_writer(authority, held, release):
    def hold():
        authority.db._lock.acquire()
        held.set()
        release.wait(5)  # bounded: a stalled loop must never deadlock the probe's own release
        authority.db._lock.release()
    threading.Thread(target=hold, daemon=True).start()


async def _max_loop_gap(create, authority):
    """Run *create* on this loop while another thread holds the store's writer lock."""
    _release_read_pool(authority)
    held, release = threading.Event(), threading.Event()
    _hold_writer(authority, held, release)
    assert await asyncio.to_thread(held.wait, 10)
    gaps = []

    async def ticker():
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.02)  # health: allow HX013 -- the ticker's period IS the loop-stall measurement
            now = time.monotonic()
            gaps.append(now - last)
            last = now
    tick = asyncio.create_task(ticker())
    task = asyncio.create_task(create())
    try:
        await asyncio.sleep(1.0)  # health: allow HX013 -- the writer hold IS the stall window the ticker measures
        release.set()
        result = await asyncio.wait_for(task, 60)
    finally:
        release.set()
        tick.cancel()
    return max(gaps), result


def test_hosted_room_member_create_reads_off_the_owner_loop(hosted_owner):
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC

    authority, loop, principal, _agent = hosted_owner
    coords = {'profile': 'default', 'source': 'bot_room'}

    def rpc(room):
        return HostedRoomAuthorityRPC(authority, loop, room_id=room, member_id='one', profile='default',
                                      principal=principal, authorize=lambda *a: True)
    created = {}

    async def probe():
        await rpc('warm')._dispatch('create', dict(coords, title='Group: warm'))  # cold imports are not the read under test
        member = rpc('room')

        async def create():
            try:
                return await member._dispatch('create', dict(coords, title='Group: room'))
            except BaseException as exc:  # the stall is the finding; never mask it as a failure
                created['error'] = exc
                raise
        return await _max_loop_gap(create, authority)
    gap, result = asyncio.run_coroutine_threadsafe(probe(), loop).result(120)
    assert 'error' not in created, created
    assert result['session_id'] == rpc('room').ref.session_id
    assert authority.db.get_session(result['session_id'])['title'] == 'Group: room'
    assert gap < 0.5, f'the owner loop stalled {gap:.2f}s behind a held writer'


def test_a2a_forward_create_reads_off_the_owner_loop(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {'default': 'fixture'}, 'platform_toolsets': {'cli': []}})
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda config: 'fixture')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = run.GatewayRunner.__new__(run.GatewayRunner)
    runner.adapters, runner.session_store, runner._session_db, runner._draining = {}, store, store._db, False
    params = dict(agent='dev', tenant='team', peer='alice', context_id='ctx', text='hello')

    async def probe():
        authority = await initialize_session_authority(runner, profile_id='target', instance_id='test')
        authority._schedule = lambda ref: None
        conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'producer'})
        try:
            warm = await conn.dispatch({'id': 0, 'method': 'a2a.forward',
                                        'params': dict(params, context_id='warm', input_id='w')})
            assert 'result' in warm, warm  # cold imports are not the read under test
            gap, response = await _max_loop_gap(
                lambda: conn.dispatch({'id': 1, 'method': 'a2a.forward',
                                       'params': dict(params, input_id='first')}), authority)
            assert 'result' in response, response
            assert authority.db.get_session(response['result']['session_id']) is not None
            return gap
        finally:
            await conn.close()
            authority.db.close()
    gap = asyncio.run(probe())
    assert gap < 0.5, f'the owner loop stalled {gap:.2f}s behind a held writer'


class _Admitted(Exception):
    """The card attempt reached admission: the create under test is already committed."""


def _card_owner(home, profile):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    import hermes_state_runtime as rt
    store = SessionStore(home / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _handle_message=None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id=profile, instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    runner.session_authority = authority
    return store, authority


def test_kanban_task_create_reads_off_the_owner_loop(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_kanban import run_task
    from gateway.session_contract import Principal

    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(home))
    monkeypatch.setattr('gateway.run._load_gateway_config',
                        lambda: {'model': {'default': 'fixture'}, 'platform_toolsets': {'cli': []}})
    monkeypatch.setattr('gateway.run._resolve_gateway_model', lambda config: 'fixture')
    workspace = home / 'workspace'
    workspace.mkdir()
    (workspace / 'pyproject.toml').write_text('[project]\nname = "task"\n')
    kb.create_board('owned')
    store, authority = _card_owner(home, str(home))
    actor = Principal('owner', str(home), frozenset({'session:create'}), 'transport')
    admitted = {}

    async def submit(_actor, request, **_kwargs):
        # The create under test is the door; the turn admission behind it is not this probe's
        # subject (and would start a real turn). A card attempt is admitted only after its session
        # is committed, so reaching here proves the create ran.
        admitted['session_id'] = request.ref.session_id
        raise _Admitted
    authority.submit = submit

    def claim(n):
        with closing(connect(board='owned')) as conn:
            task_id = kb.create_task(conn, title=f'Card {n}', body='Do the thing', assignee='default',
                                     workspace_kind='dir', workspace_path=str(workspace))
            kb.recompute_ready(conn)
            task = kb.claim_task(conn, task_id)
            return dict(board='owned', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)

    def connection():
        return SimpleNamespace(authority=authority, actor=actor, native_owner=True,
                               resume=lambda ref, params: asyncio.sleep(0))

    async def probe():
        with pytest.raises(_Admitted):  # cold imports are not the read under test
            await run_task(connection(), await asyncio.to_thread(claim, 0))

        async def create():
            try:
                return await run_task(connection(), await asyncio.to_thread(claim, 1))
            except _Admitted:
                return 'admitted'
        return await _max_loop_gap(create, authority)
    try:
        gap, result = asyncio.run(probe())
    finally:
        authority.db.close()
    assert result == 'admitted', result
    assert authority.db.get_session(admitted['session_id']) is not None
    assert gap < 0.5, f'the owner loop stalled {gap:.2f}s behind a held writer'
    store.close_all_db_handles()
