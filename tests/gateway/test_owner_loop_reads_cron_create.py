"""Cron fire creation reads its retry-exists row off the owner loop: a ledger read that misses the
read pool degrades to the writer connection under ``SessionDB._lock`` (``hermes_state._read_ctx``'s
documented fallback, since ``acquire()`` refuses to do descriptor work on a loop thread), so a held
writer must not surface as a stalled loop while ``cron.submit`` admits the fire."""
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


def test_cron_submit_retry_exists_read_waits_off_the_owner_loop(tmp_path, monkeypatch):
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
            # Warm the creation path (cold imports are not the read under test).
            await session_cron.operation(authority, 'submit', {'job_id': job['id'], 'request_id': 'warm',
                                                               'extra_prompt': None})
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
            submit = asyncio.create_task(session_cron.operation(
                authority, 'submit', {'job_id': job['id'], 'request_id': 'fire', 'extra_prompt': None}))
            try:
                await asyncio.sleep(1.0)  # health: allow HX013 -- the writer hold IS the stall window the ticker measures
                release.set()
                receipt = await asyncio.wait_for(submit, 30)
            finally:
                release.set()
                tick.cancel()
            assert len(list_session_admissions(db, session_id=receipt['session_id'], pending_only=False)) == 1
            assert max(gaps) < 0.5, max(gaps)
        try:
            asyncio.run(probe())
        finally:
            db.close()
