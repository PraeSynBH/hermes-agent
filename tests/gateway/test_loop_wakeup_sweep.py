"""PRA-6806: one loop-wakeup sweep is ONE poll hop, fanned out over the homes inside it.

The watcher used to submit two hops per served profile per tick (warm + read) and await them in
order. On the 17-profile host one tick queued 34 hops behind itself on the 2-worker poll lane, so
``loop_wakeup_scan`` still measured p50 236-717 ms / p95 6.2-7.0 s against ~2 ms of warm work per
home (PRA-6794): the queue had moved off the housekeeping pool instead of going away.

Inside the single hop the reads are independent, so they all go to ONE bounded, process-lifetime
pool (cap 4). The first cut read open stores on the calling thread instead, on a load1~110
measurement where a warm read was 1.4 ms; at load1 155-186 the same 18 warm reads serially cost
p50 118.9 ms / p95 1640.8 ms against p50 20.0 ms / p95 101.0 ms on four workers, and the live
summaries on the booted pid agreed (p50 112-182 ms). The serial sum, not the hop, is the tick.

Contracts a future edit can break silently, in order of how much they cost:

* a tick submits ONE sweep hop, whatever the profile count;
* every home of a multi-home sweep is read on the one bounded pool; a single-home sweep stays on
  the calling thread and creates no pool (there is nothing to overlap);
* a home with no loop, or an unreadable home, costs the sweep nothing but its own read;
* the READ no longer enters the per-profile runtime scope (config + ``.env`` + terminal policy per
  home per tick), while FIRING still does — it writes and injects a turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading

import pytest

from gateway import run_goals, run_idle_gates
from gateway.run import GatewayRunner


def _homes(tmp_path, names):
    return [(name, tmp_path / "profiles" / name) for name in names]


def test_the_sweep_reports_only_homes_that_hold_an_active_loop(tmp_path, monkeypatch):
    """A quiet home is one read and no entry; the fire path must not be handed empties."""
    scopes = _homes(tmp_path, ("alpha", "beta", "gamma"))
    monkeypatch.setattr(
        run_idle_gates, "_active_loops_in_home",
        lambda home: [("session-b", "state")] if str(home).endswith("beta") else [])

    swept = run_idle_gates.loop_sweep([(None, None)] + scopes)

    assert swept == [("beta", scopes[1][1], [("session-b", "state")])]


def test_an_unreadable_home_does_not_lose_the_other_homes(tmp_path, monkeypatch):
    """Best effort, same as the per-home scan before it: one bad store is a debug line."""
    scopes = _homes(tmp_path, ("alpha", "beta", "gamma"))

    def _read(home):
        if str(home).endswith("alpha"):
            raise RuntimeError("database is locked")
        return [("session", "state")]

    monkeypatch.setattr(run_idle_gates, "_active_loops_in_home", _read)

    swept = run_idle_gates.loop_sweep(scopes)

    assert [name for name, _home, _hits in swept] == ["beta", "gamma"]


def test_every_home_is_read_on_the_one_bounded_reused_pool(tmp_path, monkeypatch):
    """The serial 18-home sum is the tick (p50 119 ms) — the reads overlap instead, cap 4."""
    scopes = _homes(tmp_path, ("alpha", "beta", "gamma", "delta"))
    caller = threading.get_ident()
    readers = []
    monkeypatch.setattr(run_idle_gates, "_LOOP_SWEEP_POOL", None)
    monkeypatch.setattr(
        run_idle_gates, "_active_loops_in_home",
        lambda home: readers.append(threading.get_ident()) or [("session", "state")])

    swept = run_idle_gates.loop_sweep(scopes)
    pool = run_idle_gates._loop_sweep_pool()
    run_idle_gates.loop_sweep(scopes)

    assert sorted(name for name, _home, _hits in swept) == ["alpha", "beta", "delta", "gamma"]
    assert len(readers) == 2 * len(scopes), "every home is read on every sweep"
    assert caller not in readers, "a multi-home sweep does not read on the calling thread"
    assert pool._max_workers == run_idle_gates._LOOP_SWEEP_FAN_OUT
    assert run_idle_gates._loop_sweep_pool() is pool, "the pool is process-lifetime, not per tick"
    run_idle_gates._LOOP_SWEEP_POOL.shutdown(wait=False)
    monkeypatch.setattr(run_idle_gates, "_LOOP_SWEEP_POOL", None)


def test_a_single_home_sweep_stays_on_the_calling_thread(tmp_path, monkeypatch):
    """Nothing to overlap: one home costs its own read, never a pool thread plus a hop."""
    scopes = _homes(tmp_path, ("alpha",))
    caller = threading.get_ident()
    readers = []
    monkeypatch.setattr(run_idle_gates, "_LOOP_SWEEP_POOL", None)
    monkeypatch.setattr(
        run_idle_gates, "_active_loops_in_home",
        lambda home: readers.append(threading.get_ident()) or [])

    run_idle_gates.loop_sweep(scopes)

    assert readers == [caller]
    assert run_idle_gates._LOOP_SWEEP_POOL is None, "a one-home sweep creates no pool"


def test_a_read_that_raises_is_not_re_raised_to_the_tick(tmp_path, monkeypatch):
    """A sweep failure must never take the watcher's ``while`` loop down with it."""
    scopes = _homes(tmp_path, ("alpha",))

    def _read(_home):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(run_idle_gates, "_active_loops_in_home", _read)

    assert run_idle_gates.loop_sweep(scopes) == []


def test_a_raising_read_on_the_pool_does_not_lose_the_other_homes(tmp_path, monkeypatch):
    """The fan-out arm swallows the same failure the serial arm does."""
    scopes = _homes(tmp_path, ("alpha", "beta"))
    monkeypatch.setattr(run_idle_gates, "_LOOP_SWEEP_POOL", None)

    def _read(home):
        if str(home).endswith("alpha"):
            raise RuntimeError("database is locked")
        return [("session", "state")]

    monkeypatch.setattr(run_idle_gates, "_active_loops_in_home", _read)

    swept = run_idle_gates.loop_sweep(scopes)

    assert [name for name, _home, _hits in swept] == ["beta"]
    run_idle_gates._LOOP_SWEEP_POOL.shutdown(wait=False)
    monkeypatch.setattr(run_idle_gates, "_LOOP_SWEEP_POOL", None)


def _watcher(swept):
    """A GatewayRunner stand-in: a running flag, and a poll lane that really runs the work."""
    runner = object.__new__(GatewayRunner)
    runner.config = None
    runner._running = True
    runner.hops = []
    runner.fired = []
    runner.scopes_entered = []
    runner.swept = swept

    async def _hop(func, *args):
        runner.hops.append(func)
        return await asyncio.to_thread(func, *args)

    async def _fire(sid, state, now, warned_no_route, profile_name):
        runner.fired.append((profile_name, sid))

    runner._run_poll_in_executor = _hop
    runner._loop_wakeup_fire_one = _fire
    return runner


@pytest.mark.asyncio
async def test_a_tick_submits_the_whole_scan_as_one_poll_hop(monkeypatch):
    """One sweep hop for N profiles (the 34-hop queue cannot come back without failing this)."""
    scopes = [(None, None), ("alpha", "/h/alpha"), ("beta", "/h/beta"), ("gamma", "/h/gamma")]
    watcher = _watcher([])

    async def _poll_plan(_runner, _gate_body):
        return scopes, set()

    def _loop_sweep_stub(_targets):
        return []

    monkeypatch.setattr(run_idle_gates, "poll_plan", _poll_plan)
    monkeypatch.setattr(run_idle_gates, "loop_sweep", _loop_sweep_stub)
    monkeypatch.setattr(asyncio, "sleep", _one_tick(watcher))

    await watcher._loop_wakeup_watcher()

    assert watcher.hops == [_loop_sweep_stub], "the scan must be exactly one poll-lane hop"


@pytest.mark.asyncio
async def test_quiet_homes_are_not_scanned_and_firing_still_enters_the_profile_scope(monkeypatch):
    """The idle gate still prunes; the read drops the runtime scope, the fire keeps it."""
    scopes = [(None, None), ("alpha", "/h/alpha"), ("beta", "/h/beta")]
    watcher = _watcher([("alpha", "/h/alpha", [("session-a", "state"), ("session-b", "state")])])
    swept_targets = []

    async def _poll_plan(_runner, _gate_body):
        return scopes, {"/h/alpha"}

    def _loop_sweep(targets):
        swept_targets.append(list(targets))
        return watcher.swept

    @contextlib.asynccontextmanager
    async def _scope(profile_home):
        watcher.scopes_entered.append(profile_home)
        yield

    monkeypatch.setattr(run_idle_gates, "poll_plan", _poll_plan)
    monkeypatch.setattr(run_idle_gates, "loop_sweep", _loop_sweep)
    monkeypatch.setattr(run_goals, "_wakeup_scan_scope", _scope)
    monkeypatch.setattr(asyncio, "sleep", _one_tick(watcher))

    await watcher._loop_wakeup_watcher()

    assert swept_targets == [[(None, None), ("alpha", "/h/alpha")]], "the gate prunes before the hop"
    assert watcher.fired == [("alpha", "session-a"), ("alpha", "session-b")]
    assert watcher.scopes_entered == ["/h/alpha"], "firing keeps the profile's own runtime scope"


def _one_tick(watcher):
    """``asyncio.sleep`` stand-in: the watcher's startup wait and its inter-tick wait, then stop."""
    calls = []

    async def _sleep(_seconds, *_args, **_kwargs):
        calls.append(_seconds)
        if len(calls) > 1:
            watcher._running = False

    return _sleep
