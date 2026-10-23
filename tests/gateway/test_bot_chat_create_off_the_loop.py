"""The bot-chat create door reaches no writer-locked read on the owner loop.

Regression for PRX-6015's sibling class. ``session.create`` was moved off the loop
(``session_controls._stage_create``, PRA-6517); the Bot Chat mint behind
``gateway.session_bot._create_bot_chat`` kept the same shape -- ``get_session_by_title('Bot Chat')``
and then ``stage_local_session`` -> ``SessionDB.get_session`` -> ``hermes_state._read_ctx`` -- in an
``async def`` on the owner loop. With the read pool offering nothing and another thread holding
``SessionDB._lock``, the documented degradation takes the WRITER connection under that lock, so the
whole profile's loop (sessions, timers, sockets) froze for the length of the hold: measured at 20.3 s
of loop stall against a probe that held the writer for exactly 20 s.

The contract: a bot-chat mint must not stall the loop behind a held writer, and must still return
its session. Real SessionStore/SessionAuthority and a real config.yaml; no endpoint is contacted.
"""
import asyncio
import threading
import time

from tests.gateway.test_follow_profile_followups import _profile
from tests.gateway.test_session_follow_profile_config import _owner


def test_bot_chat_create_reads_off_the_owner_loop(tmp_path, monkeypatch):
    from gateway import run
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)
    _profile(home, 'fixture', 1, 'k-fixture')
    store, _runner, authority, owner = _owner(home)
    authority._schedule = lambda ref: None
    from gateway.session_bot import _create_bot_chat

    def retire(sid):
        return authority.db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET title='Retired', title_source='user' WHERE id=?", (sid,)).rowcount)

    async def probe():
        # Cold opens are not the read under test: one mint warms the imports and the read pool.
        first = await _create_bot_chat(authority, owner.actor)
        assert first is not None
        # A renamed Bot Chat is a spent creation identity, so this mint stages a new session (the
        # lookup finds the title free) instead of resuming the retired one.
        assert await asyncio.to_thread(retire, first.session_id) == 1
        # The state that sends a read to the writer connection: the pool offers nothing to
        # borrow, and on the loop the pool refuses to open one.
        while authority.db._evict_one_idle_read_conn():
            pass
        held, release = threading.Event(), threading.Event()

        def hold_writer():  # a slow commit holding the store's writer lock
            authority.db._lock.acquire()
            held.set()
            release.wait(30)
            authority.db._lock.release()
        threading.Thread(target=hold_writer, daemon=True).start()
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
        mint = asyncio.create_task(_create_bot_chat(authority, owner.actor))
        try:
            # The staging (lookup + create) must complete while the writer is held: only the
            # commit waits for it. A loop-side read would have the loop here for the whole hold.
            await asyncio.sleep(1.0)  # health: allow HX013 -- the writer hold IS the stall window the ticker measures
            release.set()
            ref = await asyncio.wait_for(mint, 30)
        finally:
            release.set()
            tick.cancel()
        assert ref is not None
        assert max(gaps) < 0.5, f'the owner loop stalled {max(gaps):.2f}s behind a held writer'
        return ref

    try:
        ref = asyncio.run(probe())
        assert authority.db.get_session(ref.session_id) is not None
    finally:
        store.close_all_db_handles()
