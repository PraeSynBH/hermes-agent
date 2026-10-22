"""Per-tick gateway pollers must cost ONE bounded-pool hop, never one thread per profile.

Regression for PRA-6660: the handoff watcher, the loop-wakeup scan and heartbeat restore gated every
served profile through ``_run_in_executor_with_context`` — the UNBOUNDED turn executor, which starts a
fresh OS thread per work item. On a 16-profile gateway that was ~9 of the ~21 thread spawns/s measured
(1262 submissions/min), and under load (PRA-6502) the same rate produced hundreds of live threads,
because live thread count was ``arrival rate × service time`` with nothing to cap it. The polls do no
agent work at all: they open each profile's store and answer "is there anything to do?".

The contracts here are behaviour, not source shape:

* a poll tick costs the same number of offload hops whether the gateway serves 2 profiles or 12, and
  the sweep probes every profile inside that one hop;
* polls route to the bounded, worker-reusing pool when the runner has one — the turn pool is never
  touched by a poll;
* a home whose probe cannot answer stays in the busy set (fail open: its scope is entered as before).
"""

from __future__ import annotations

import asyncio
import threading
import types
from pathlib import Path

import pytest

from gateway import run
from gateway.run import GatewayRunner, _HOUSEKEEPING_MAX_WORKERS
from gateway.run_idle_gates import off_loop_busy_homes, offload_for_poll


class _ProbeDB:
    def __init__(self, pending):
        self._pending = pending
        self.probes = 0

    def has_pending_handoffs(self):
        self.probes += 1
        return self._pending


class _RecordingDB:
    async def list_pending_handoffs(self):
        return []


class _PollHopRunner:
    """Runner double that records WHICH hop each poll takes."""

    def __init__(self, scopes):
        self._scopes = scopes
        self.poll_hops: list = []
        self.turn_hops: list = []
        self._session_db = _RecordingDB()

        async def _process_handoff(_row, _profile_name=None):
            return None

        self._process_handoff = _process_handoff
        states = iter([True, False])

        class _Running:
            def __bool__(_self):
                return next(states, False)

        self._running = _Running()

    async def _run_housekeeping_in_executor(self, func, *args):
        self.poll_hops.append(func)
        return await asyncio.to_thread(func, *args)

    async def _run_in_executor_with_context(self, func, *args):
        self.turn_hops.append(func)
        return await asyncio.to_thread(func, *args)


def _scopes(count: int) -> list:
    scopes = [(None, None)]
    scopes += [(f"profile{i}", Path(f"/h/profiles/profile{i}")) for i in range(count)]
    return scopes


def _drive_handoff_tick(monkeypatch, scopes, pending_by_home):
    """Run exactly one handoff-watcher tick over *scopes*; return the recording runner."""
    from hermes_cli import goals

    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _r: scopes)

    class _SpyScope:
        def __init__(self, home):
            self.home = home

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(run, "_async_profile_runtime_scope", _SpyScope)

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(run.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(
        goals, "_DB_CACHE",
        {str(h): _ProbeDB(pending_by_home.get(h, False)) for _n, h in scopes[1:]})
    return _PollHopRunner(scopes)


@pytest.mark.asyncio
async def test_handoff_tick_costs_the_same_hops_for_any_profile_count(monkeypatch):
    """One sweep per tick: 2 profiles and 12 profiles cost the same offload hops."""
    hops = []
    for count in (2, 12):
        runner = _drive_handoff_tick(monkeypatch, _scopes(count), {})
        await asyncio.wait_for(
            run.GatewayRunner._handoff_watcher(runner, interval=0.0), timeout=10)
        hops.append(len(runner.poll_hops))
    assert hops[0] == hops[1], (
        f"a poll tick must not cost a hop per profile (2 profiles: {hops[0]} hops, "
        f"12 profiles: {hops[1]} hops)")


@pytest.mark.asyncio
async def test_handoff_poll_never_touches_the_turn_pool(monkeypatch):
    """Every poll hop goes to the bounded pool; the one-thread-per-item turn pool sees nothing."""
    runner = _drive_handoff_tick(monkeypatch, _scopes(6), {})
    await asyncio.wait_for(run.GatewayRunner._handoff_watcher(runner, interval=0.0), timeout=10)
    assert runner.poll_hops, "the watcher must offload its store reads, not block the loop"
    assert runner.turn_hops == [], (
        "a per-tick poller submitted work to the turn executor, which spawns an OS thread per "
        "submission (PRA-6660)")


@pytest.mark.asyncio
async def test_sweep_probes_every_home_in_its_single_hop(monkeypatch):
    """The one hop must still ask EVERY profile's store, not just the first."""
    from hermes_cli import goals

    homes = [Path(f"/h/profiles/p{i}") for i in range(5)]
    dbs = {}
    for index, home in enumerate(homes):
        dbs[str(home)] = _ProbeDB(pending=index == 3)
    monkeypatch.setattr(goals, "_DB_CACHE", dbs)

    runner = types.SimpleNamespace(_run_housekeeping_in_executor=asyncio.to_thread)

    busy = await off_loop_busy_homes(runner, homes, lambda db: db.has_pending_handoffs())
    assert busy == {homes[3]}, f"exactly the profile with work must be busy; got {busy}"
    assert [db.probes for db in dbs.values()] == [1, 1, 1, 1, 1], "every store probed exactly once"


@pytest.mark.asyncio
async def test_sweep_fails_open_on_a_probe_that_cannot_answer(monkeypatch):
    """An unreadable store must keep its scope entered, never silently skip the profile."""
    from hermes_cli import goals

    broken = Path("/h/profiles/broken")

    def _explode():
        raise RuntimeError("state.db unreadable")

    monkeypatch.setattr(goals, "_get_session_db", _explode)
    runner = types.SimpleNamespace(_run_housekeeping_in_executor=asyncio.to_thread)

    busy = await off_loop_busy_homes(runner, [broken], lambda db: db.has_pending_handoffs())
    assert busy == {broken}


def test_offload_for_poll_prefers_the_bounded_pool():
    """The bounded pool wins whenever the runner has it; the turn helper is only a fallback."""
    def poll_hop(*_args):
        raise AssertionError("identity only; never called")

    def turn_hop(*_args):
        raise AssertionError("identity only; never called")

    runner = types.SimpleNamespace(
        _run_housekeeping_in_executor=poll_hop,
        _run_in_executor_with_context=turn_hop,
    )
    assert offload_for_poll(runner) is poll_hop
    assert offload_for_poll(types.SimpleNamespace(_run_in_executor_with_context=turn_hop)) is turn_hop
    assert offload_for_poll(types.SimpleNamespace()) is None


def test_offload_for_poll_prefers_the_poll_lane():
    """PRA-6791: the poll lane wins over the housekeeping pool, which is seconds long per item."""
    def lane_hop(*_args):
        raise AssertionError("identity only; never called")

    def housekeeping_hop(*_args):
        raise AssertionError("identity only; never called")

    def turn_hop(*_args):
        raise AssertionError("identity only; never called")

    runner = types.SimpleNamespace(
        _run_poll_in_executor=lane_hop,
        _run_housekeeping_in_executor=housekeeping_hop,
        _run_in_executor_with_context=turn_hop,
    )
    assert offload_for_poll(runner) is lane_hop, (
        "a poll hop queued behind best-effort housekeeping reports the queue, not the store: it "
        "waited 1970 ms p50 behind four busy housekeeping workers and 10 ms on a lane of its own")


def _pooled_runner():
    """A real GatewayRunner (no __init__) with all three lanes, as boot creates them."""
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._executor_lock = threading.Lock()
    runner._executor = None
    runner._housekeeping_executor = None
    runner._poll_executor = None
    runner._executor_closing = False
    return runner


def test_a_poll_hop_runs_beside_a_saturated_housekeeping_pool():
    """The lane pin: with EVERY housekeeping worker wedged for seconds, a poll still runs now."""
    wedge = threading.Event()
    housekeeping_started = threading.Event()
    ran_on: list[str] = []

    def wedged_housekeeping():
        housekeeping_started.set()
        assert wedge.wait(60), "wedged housekeeping item never released"

    runner = _pooled_runner()
    try:
        housekeeping_pool = runner._get_housekeeping_executor()
        for _ in range(_HOUSEKEEPING_MAX_WORKERS):
            housekeeping_pool.submit(wedged_housekeeping)
        assert housekeeping_started.wait(10), "housekeeping workers never started"

        async def poll():
            return await runner._run_poll_in_executor(
                lambda: ran_on.append(threading.current_thread().name))

        asyncio.run(asyncio.wait_for(poll(), timeout=10))
        assert ran_on and ran_on[0].startswith("hermes-gateway-poll"), (
            f"the poll hop did not run on its own lane: {ran_on}")
    finally:
        wedge.set()
        runner._shutdown_executor()


def test_loop_scan_warms_the_store_on_the_poll_lane(monkeypatch):
    """The warm hop is per-tick polling too: it must not mint a turn-pool thread per scan."""
    from gateway import run_goals
    from hermes_cli import goals

    hops: list[str] = []

    async def _lane(func, *args):
        hops.append("lane")
        return func(*args)

    async def _turn(func, *args):  # pragma: no cover - asserted never taken
        hops.append("turn")
        return func(*args)

    monkeypatch.setattr(goals, "_get_session_db", lambda: object())
    runner = _pooled_runner()
    asyncio.run(GatewayRunner._warm_goals_session_db(runner, "test", offload=_lane))
    assert hops == ["lane"], hops
    assert _turn not in hops
