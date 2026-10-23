"""PRX-6075: `session_housekeeping` needs a per-tick ceiling, and the ceiling needs a number.

The watcher's own arms are milliseconds (two dict walks over the agent cache). The hourly
SessionStore prune is the exception: it is one atomic full-index rewrite plus an fsync, measured at
4.3 s on a quiet host and 51 681 ms on the live one, and PRX-5790's tick profile attributed it to
`session_housekeeping` as a single unbounded tick. So the prune runs as its own task and the tick
waits at most `_SESSION_HOUSEKEEPING_TICK_BUDGET_S` for it: the work still happens, the tick stops
being an outage, and the shortfall is logged with the phase split instead of disappearing.

These use a store that parks on an `threading.Event` rather than a sleep: the assertions are about
which of the two (tick, prune) makes progress, never about wall-clock scheduling.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from gateway import loop_tick_timing as timing
from gateway import run_watchers
from gateway.run_watchers import GatewaySessionWatchersMixin


class _ParkedStore:
    """Async facade stand-in whose prune blocks until released."""

    def __init__(self) -> None:
        self.calls = 0
        self.removed = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    async def prune_old_entries(self, max_age_days: int) -> int:
        self.calls += 1
        self.started.set()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.release.wait)
        self.finished.set()
        return self.removed


class _Runner(GatewaySessionWatchersMixin):
    """The watcher mixin with the two agent-cache arms stubbed to their real cost (nothing)."""

    def _sweep_idle_cached_agents(self) -> int:
        return 0

    def _sweep_agent_cache_under_pressure(self) -> int:
        return 0


def _runner(store: _ParkedStore, *, max_age_days: int = 90) -> _Runner:
    runner = _Runner()
    runner.config = SimpleNamespace(session_store_max_age_days=max_age_days)
    runner.async_session_store = store
    runner._last_session_store_prune_ts = 0.0
    return runner


@pytest.fixture(autouse=True)
def _fresh_counters():
    timing.reset()
    yield
    timing.reset()


def test_a_slow_prune_does_not_hold_the_tick_past_the_budget(monkeypatch):
    """The tick returns while the prune is still running — that is the whole ceiling."""
    monkeypatch.setattr(run_watchers, '_SESSION_HOUSEKEEPING_TICK_BUDGET_S', 0.05)
    store = _ParkedStore()
    runner = _runner(store)

    async def _scenario() -> float:
        started = time.monotonic()
        await runner._session_housekeeping()
        elapsed = time.monotonic() - started
        assert store.started.is_set(), 'the prune was started, not skipped'
        assert not store.finished.is_set(), 'the prune is still parked: the tick did not wait'
        store.release.set()
        await asyncio.wait_for(runner._store_prune_task, timeout=10)
        return elapsed

    assert asyncio.run(_scenario()) < 2.0


def test_the_prune_logs_its_own_outcome_after_the_tick_returned(monkeypatch, caplog):
    """Returning early must not lose the account of what the rewrite did."""
    monkeypatch.setattr(run_watchers, '_SESSION_HOUSEKEEPING_TICK_BUDGET_S', 0.05)
    caplog.set_level(logging.INFO, logger='gateway.run')
    store = _ParkedStore()
    store.removed = 3
    runner = _runner(store)

    async def _scenario() -> None:
        await runner._session_housekeeping()
        store.release.set()
        await asyncio.wait_for(runner._store_prune_task, timeout=10)

    asyncio.run(_scenario())
    assert store.finished.is_set()
    messages = [r.getMessage() for r in caplog.records]
    assert any('SessionStore prune: dropped 3 entries' in m for m in messages), messages


def test_the_tick_warns_with_the_phase_split_when_it_overruns(monkeypatch, caplog):
    """`budget_used` says a tick was expensive; the split says which arm was."""
    monkeypatch.setattr(run_watchers, '_SESSION_HOUSEKEEPING_TICK_BUDGET_S', 0.0)
    caplog.set_level(logging.WARNING, logger='gateway.run')
    store = _ParkedStore()
    store.release.set()  # the prune completes at once: the split still names both arms
    runner = _runner(store)

    asyncio.run(runner._session_housekeeping())

    over = [r for r in caplog.records if 'over its' in r.getMessage()]
    assert over, [r.getMessage() for r in caplog.records]
    assert 'agent-cache=' in over[0].getMessage() and 'prune=' in over[0].getMessage()


def test_a_second_prune_is_not_started_while_one_is_in_flight(monkeypatch):
    """Two concurrent full-index rewrites would serialise on the save lock for no gain."""
    monkeypatch.setattr(run_watchers, '_SESSION_HOUSEKEEPING_TICK_BUDGET_S', 0.05)
    store = _ParkedStore()
    runner = _runner(store)

    async def _scenario() -> None:
        await runner._prune_session_store(90)
        await runner._prune_session_store(90)
        assert store.calls == 1
        store.release.set()
        await asyncio.wait_for(runner._store_prune_task, timeout=10)
        # Once the rewrite is done a later hour may prune again.
        store.release.clear()
        await runner._prune_session_store(90)
        store.release.set()
        await asyncio.wait_for(runner._store_prune_task, timeout=10)
        assert store.calls == 2

    asyncio.run(_scenario())


def test_the_watcher_tick_carries_the_budget_into_its_counters(monkeypatch):
    """The ceiling is only real if the per-minute summary reports what was spent against it."""
    monkeypatch.setattr(run_watchers, '_SESSION_HOUSEKEEPING_TICK_BUDGET_S', 0.0)
    store = _ParkedStore()
    store.release.set()
    runner = _runner(store)
    runner._running = True

    async def _scenario() -> None:
        with timing.timed('session_housekeeping',
                          budget_s=run_watchers._SESSION_HOUSEKEEPING_TICK_BUDGET_S):
            await runner._session_housekeeping()

    asyncio.run(_scenario())
    view = timing.snapshot()['session_housekeeping']
    assert view['budget_s'] == 0.0
    assert view['over_budget_total'] == 1
    assert view['budget_used_s'] > 0.0
