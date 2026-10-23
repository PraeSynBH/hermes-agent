"""Cron ``status``/``cancel`` polls read the admission row off the owner loop.

``session_cron.operation``'s poll is the most frequent ledger read a gateway runs — a firer asks for
``status`` every 0.1-2 s for the whole length of a cron run. A read that misses the read pool
degrades to the writer connection under ``SessionDB._lock`` (``hermes_state._read_ctx``'s documented
fallback, since ``acquire()`` refuses descriptor work on a loop thread), so a held writer surfaced as
a stalled loop: ``gateway.shutdown_watchdog`` recorded a 200.6 s stall at 2026-10-22 13:47:13 and the
stack captured inside it was ``cron.scheduler_authority.observe`` -> ``session_cron.operation`` ->
``get_session_admission`` -> ``_read_ctx`` waiting on ``_lock``.

Same remedy as the creation path beside it (N50 / the retry-exists read): the row read runs in a
worker, so a held writer is a wait in that worker instead of a frozen loop.
"""
import asyncio
import threading
import time
from types import SimpleNamespace

from cron import jobs
from gateway import run, session_cron
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import SessionAuthority
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, list_session_admissions


def test_cron_status_poll_reads_the_admission_row_off_the_owner_loop(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local', model='fixture')
        db = SessionDB(tmp_path / 'state.db')
        runner = SimpleNamespace(_draining=False, session_store=SessionStore(tmp_path / 'sessions', GatewayConfig()),
                                 adapters={})
        authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='test', db=db,
                                     epoch=begin_runtime_epoch(db, instance_id='test'))
        monkeypatch.setattr(authority, '_schedule', lambda ref: None)
        runner.session_authority = authority

        async def probe():
            receipt = await session_cron.operation(
                authority, 'submit', {'job_id': job['id'], 'request_id': 'fire', 'extra_prompt': None})
            # The state that makes a ledger read take the writer connection instead of a pooled
            # reader: the pool offers nothing to borrow, and on the loop it refuses to open one.
            while db._evict_one_idle_read_conn():
                pass
            held, release = threading.Event(), threading.Event()

            def hold_writer():  # a writer holding the store's lock, as a slow commit does
                db._lock.acquire()
                held.set()
                release.wait(5)
                db._lock.release()
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
            poll = asyncio.create_task(session_cron.operation(
                authority, 'status', {'session_id': receipt['session_id'], 'admission_id': receipt['admission_id']}))
            try:
                await asyncio.sleep(1.0)  # health: allow HX013 -- the writer hold IS the stall window the ticker measures
                release.set()
                state = await asyncio.wait_for(poll, 30)
            finally:
                release.set()
                tick.cancel()
                await asyncio.gather(tick, return_exceptions=True)
            assert state['status'] in {'queued', 'started'}, state
            assert max(gaps) < 0.5, max(gaps)
        try:
            asyncio.run(probe())
        finally:
            db.close()
