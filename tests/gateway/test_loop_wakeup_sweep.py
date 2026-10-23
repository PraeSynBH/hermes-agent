"""PRA-6806: one loop-wakeup sweep is ONE poll hop, read serially inside it.

The watcher used to submit two hops per served profile per tick (warm + read) and await them in
order. On the 17-profile host one tick queued 34 hops behind itself on the 2-worker poll lane, so
``loop_wakeup_scan`` still measured p50 236-717 ms / p95 6.2-7.0 s against ~2 ms of warm work per
home (PRA-6794): the queue had moved off the housekeeping pool instead of going away.

A bounded fan-out INSIDE the hop was the suggested shape. Measured here: warm, four workers cost
p50 24.8-95 ms / p95 67-142 ms for the same 18 reads that serially cost p50 1.9-58 ms (a warm read
is ~0.1-1.4 ms — cheaper than the hop); cold, the fan-out opened the whole served set in 0.78 s
where serial opens cost up to 5.5 s for ONE store. So the shipped shape splits by coldness: open
stores are read on the calling thread, not-yet-open ones on a bounded, process-lifetime pool.

Contracts a future edit can break silently, in order of how much they cost:

* a tick submits ONE sweep hop, whatever the profile count;
* warm homes are read on the calling thread and cold homes concurrently, cap 4, on ONE pool;
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
    monkeypatch.setattr(run_idle_gates, "_home_store_open", lambda _home: True)
    monkeypatch.setattr(
        run_idle_gates, "_active_loops_in_home",
        lambda home: [("session-b", "state")] if str(home).endswith("beta") else [])

    swept = run_idle_gates.loop_sweep([(None, None)] + scopes)

    assert swept == [("beta", scopes[1][1], [("session-b", "state")])]


def test_an_unreadable_home_does_not_lose_the_other_homes(tmp_path, monkeypatch):
    """Best effort, same as the per-home scan before it: one bad store is a debug line."""
    scopes = _homes(tmp_path, ("alpha", "beta", "gamma"))
    monkeypatch.setattr(run_idle_gates, "_home_store_open", lambda _home: True)

    def _read(home):
        if str(home).endswith("alpha"):
            raise RuntimeError("database is locked")
        return [("session", "state")]

    monkeypatch.setattr(run_idle_gates, "_active_loops_in_home", _read)

    swept = run_idle_gates.loop_sweep(scopes)

    assert [name for name, _home, _hits in swept] == ["beta", "gamma"]


def test_warm_homes_are_read_on_the_calling_thread(tmp_path, monkeypatch):
    """An open store's read is cheaper than a thread hop: the steady-state tick takes none."""
    scopes = _homes(tmp_path, ("alpha", "beta", "gamma", "delta"))
    caller = threading.get_ident()
    readers = []
    monkeypatch.setattr(run_idle_gates, "_home_store_open", lambda _home: True)
    monkeypatch.setattr(
        run_idle_gates, "_active_loops_in_home",
        lambda home: readers.append(threading.get_ident()) or [])

    run_idle_gates.loop_sweep(scopes)

    assert len(readers) == len(scopes)
    assert set(readers) == {caller}
    assert run_idle_gates._LOOP_SWEEP_POOL is None, "no pool is created when nothing is cold"


def test_cold_homes_are_opened_concurrently_on_a_bounded_reused_pool(tmp_path, monkeypatch):
    """Cold opens are the one place concurrency pays (5.5 s/home serial vs 0.78 s for the fan-out)."""
    scopes = _homes(tmp_path, ("alpha", "beta", "gamma", "delta"))
    monkeypatch.setattr(run_idle_gates, "_LOOP_SWEEP_POOL", None)
    monkeypatch.setattr(run_idle_gates, "_home_store_open", lambda home: str(home).endswith("alpha"))
    monkeypatch.setattr(run_idle_gates, "_active_loops_in_home", lambda home: [("session", "state")])

    swept = run_idle_gates.loop_sweep(scopes)
    pool = run_idle_gates._loop_sweep_pool()
    run_idle_gates.loop_sweep(scopes)

    assert sorted(name for name, _home, _hits in swept) == ["alpha", "beta", "delta", "gamma"]
    assert len(swept) == len(scopes), "a cold home's hits are reported too"
    assert pool._max_workers == run_idle_gates._LOOP_SWEEP_FAN_OUT
    assert run_idle_gates._loop_sweep_pool() is pool, "the pool is process-lifetime, not per tick"
    run_idle_gates._LOOP_SWEEP_POOL.shutdown(wait=False)
    monkeypatch.setattr(run_idle_gates, "_LOOP_SWEEP_POOL", None)


def test_a_read_that_raises_is_not_re_raised_to_the_tick(tmp_path, monkeypatch):
    """A sweep failure must never take the watcher's ``while`` loop down with it."""
    scopes = _homes(tmp_path, ("alpha",))
    monkeypatch.setattr(run_idle_gates, "_home_store_open", lambda _home: True)

    def _read(_home):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(run_idle_gates, "_active_loops_in_home", _read)

    assert run_idle_gates.loop_sweep(scopes) == []


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
