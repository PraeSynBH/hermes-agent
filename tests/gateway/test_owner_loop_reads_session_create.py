"""The owner loop's ``session.create`` reaches no writer-locked read.

Regression for PRA-6517. A session create staged its session on the event loop:
``session_controls._create`` -> ``session_local.stage_local_session:288`` ->
``SessionDB.get_session`` -> ``hermes_state._read_ctx``. On a loop thread the read pool refuses
to open or reclaim a connection (``acquire()``'s loop-thread refusal), so a read that finds the
pool empty takes ``_read_ctx``'s documented degradation -- the WRITER connection under
``SessionDB._lock``. ``gateway.error.log`` 2026-10-22 07:47:16 on the shared gateway (pid 48855)
names exactly that frame:

    hermes_state.py:971 _read_ctx              (`with self._lock:`)
      <- contextlib.py:141 __enter__
      <- hermes_state.py:1180 _read_retrying_ioerr
      <- hermes_state.py:1162 _read_one
      <- hermes_state_sessions.py:835 get_session
      <- gateway/session_local.py:288 stage_local_session
      <- gateway/session_controls.py:541 _create
      <- tui_gateway/ws.py:362 _dispatch_request

and the loop reported ``stalled 40.1s``. The contract: with the read pool offering nothing and
another writer holding the store's lock, a create must not stall the loop -- and must still return
its session.
"""
import asyncio
import threading
import time

from tests.gateway.test_session_follow_profile_config import _owner


def test_session_create_reads_off_the_owner_loop(tmp_path, monkeypatch):
    from gateway import run

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config',
                        lambda: {'model': {'default': 'fixture'}, 'platform_toolsets': {'cli': []}})
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda config: 'fixture')
    store, _runner, authority, owner = _owner(tmp_path)
    authority._schedule = lambda ref: None

    def params(request_id):
        return {'request_id': request_id, 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': []}

    async def probe():
        warm = await owner.dispatch({'id': 0, 'method': 'session.create', 'params': params('warm')})
        assert 'result' in warm, warm  # cold imports are not the read under test
        # The state that sends a read to the writer connection: the pool offers nothing to
        # borrow, and on the loop the pool refuses to open one.
        while authority.db._evict_one_idle_read_conn():
            pass
        held, release = threading.Event(), threading.Event()

        def hold_writer():  # a slow commit holding the store's writer lock
            authority.db._lock.acquire()
            held.set()
            release.wait(5)
            authority.db._lock.release()
        threading.Thread(target=hold_writer, daemon=True).start()
        assert await asyncio.to_thread(held.wait, 5)
        gaps = []

        async def ticker():
            last = time.monotonic()
            while True:
                await asyncio.sleep(0.02)  # health: allow HX013 -- the ticker's period IS the loop-stall measurement
                now = time.monotonic()
                gaps.append(now - last)
                last = now
        tick = asyncio.create_task(ticker())
        create = asyncio.create_task(owner.dispatch({'id': 1, 'method': 'session.create', 'params': params('first')}))
        try:
            await asyncio.sleep(1.0)  # health: allow HX013 -- the writer hold IS the stall window the ticker measures
            release.set()
            response = await asyncio.wait_for(create, 30)
        finally:
            release.set()
            tick.cancel()
        assert 'result' in response, response
        assert max(gaps) < 0.5, f'the owner loop stalled {max(gaps):.2f}s behind a held writer'
        return response['result']['session_id']

    try:
        session_id = asyncio.run(probe())
        assert authority.db.get_session(session_id) is not None
    finally:
        store.close_all_db_handles()
