"""PRA-6850: a quiet reconcile tick costs one off-loop pass per half, and says which half is which.

The tick was measured at 1 138 ms with every loop-thread sample inside it idle in ``select``: the
loop was waiting on the worker, so the work — not the loop — had to be named. Warm, that work is
tiny (a signature pass over 17 homes is ~0.3 ms, the demand refresh and served enumeration are tens
of milliseconds), while each ``asyncio.to_thread`` is a hop onto the executor every turn shares. So
the invariants here are (a) each half records its own tick-timing counter, (b) the read pass is ONE
call carrying the three reads a cycle needs in dependency order, and (c) the WHOLE tick is one
poll-lane hop — both halves ride one lane slot, and the counters decompose it instead of inferring
it (``hop - hooks - read`` is the lane wait; round 4 measured the hook half's p50 as all hop).
"""
import asyncio
import inspect
from pathlib import Path

import pytest

import gateway.run_profile_reconcile as reconcile_mod
import hermes_cli.gateway_multiplex_capacity as capacity_mod
import hermes_cli.gateway_multiplex_demand as demand_mod
import hermes_cli.profiles as profiles_mod
from gateway import loop_tick_timing
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from tests.waiting import async_wait_until


def _runner(tmp_path, monkeypatch, *, homes):
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = True
    runner._primary_profile_name = "default"
    runner._served_profile_homes = dict(homes)
    runner._served_profile_signatures = {
        name: reconcile_mod.profile_serve_signature(home_) for name, home_ in homes.items()}

    async def _unserve(name, home_, *, reason=""):
        return None

    runner._unserve_profile = _unserve
    runner._live_resource_claims = lambda active: {}
    runner._record_served_profiles = lambda active, profile_homes: runner._note_served_profiles(profile_homes)
    return runner


def _two_served(tmp_path):
    """Launch home + one served profile, each on disk and in the served set (a quiet tick)."""
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    for h in (home, profile):
        (h / "config.yaml").write_text("model: {default: test}\n", encoding="utf-8")
    return home, profile, {"default": home, "worker": profile}


@pytest.mark.asyncio
async def test_a_quiet_tick_records_the_hook_and_served_halves_separately(tmp_path, monkeypatch):
    """The tick-timing line has to say which half owns the second, not just that there was one."""
    _home, _profile, homes = _two_served(tmp_path)
    runner = _runner(tmp_path, monkeypatch, homes=homes)

    async def _quiet_reconcile(**_kwargs):
        return {"added": [], "removed": [], "rescanned": [], "parked": []}

    runner.reconcile_served_profiles = _quiet_reconcile
    loop_tick_timing.reset()
    watcher = asyncio.create_task(runner._profile_reconcile_watcher(interval=0.05))
    try:
        await async_wait_until(lambda: loop_tick_timing.snapshot().get("profile_reconcile", {})
                               .get("count", 0) >= 2, timeout=10, what="two watcher ticks")
    finally:
        runner._running = False
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)

    snapshot = loop_tick_timing.snapshot()
    try:
        ticks = snapshot["profile_reconcile"]["count"]
        assert ticks >= 2
        # One record per half per tick, all three counters from the same pass.
        assert snapshot["profile_reconcile_hooks"]["count"] == ticks
        assert snapshot["profile_reconcile_served"]["count"] == ticks
        # ... and a half is never charged more than the tick that contains it.
        assert snapshot["profile_reconcile_hooks"]["max_s"] <= snapshot["profile_reconcile"]["max_s"]
        assert snapshot["profile_reconcile_served"]["max_s"] <= snapshot["profile_reconcile"]["max_s"]
    finally:
        loop_tick_timing.reset()


@pytest.mark.asyncio
async def test_the_served_half_of_a_quiet_tick_is_one_thread_hop(tmp_path, monkeypatch):
    """Demand, enumeration and the signature sweep ride in one ``to_thread``, not three."""
    _home, _profile, homes = _two_served(tmp_path)
    runner = _runner(tmp_path, monkeypatch, homes=homes)

    calls = []
    real_read = reconcile_mod.GatewayProfileReconcileMixin._live_profiles_snapshot

    def spy(self):
        calls.append(1)
        return real_read(self)

    monkeypatch.setattr(reconcile_mod.GatewayProfileReconcileMixin, "_live_profiles_snapshot", spy)
    await runner.reconcile_served_profiles(reason="watcher")

    assert calls == [1], f"a quiet reconcile enumerated once, got {len(calls)} passes"
    assert not inspect.iscoroutinefunction(real_read), (
        "the read half is handed to a worker as one blocking call")


@pytest.mark.asyncio
async def test_the_read_half_rides_the_poll_lane_not_the_shared_default_executor(tmp_path, monkeypatch):
    """The hop is what the tick paid: on the default pool it measured p50 57 ms for 0.3 ms of work."""
    _home, _profile, homes = _two_served(tmp_path)
    runner = _runner(tmp_path, monkeypatch, homes=homes)

    lanes = []
    shared = []

    async def poll_lane(func, *args):
        lanes.append(getattr(func, "__name__", repr(func)))
        return func(*args)

    async def default_pool(func, *args, **kwargs):
        shared.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    runner._run_poll_in_executor = poll_lane
    monkeypatch.setattr(asyncio, "to_thread", default_pool)
    await runner.reconcile_served_profiles(reason="watcher")
    await runner._sync_hooks_on_consent_change()

    assert lanes == ["_live_profiles_snapshot", "_resync_changed_hook_consent"], lanes
    assert shared == [], f"the tick went back to the shared default executor: {shared!r}"


@pytest.mark.asyncio
async def test_a_quiet_tick_takes_one_poll_lane_hop_for_both_halves(tmp_path, monkeypatch):
    """Both halves ride ONE lane slot, and the counters say which body the hop carried.

    The tick used to submit the hook sync and the read pass as two hops awaited in order, on a
    2-worker lane every other poller shares. Measured on the live host the hook half's whole p50
    (53 ms) was that hop for two ``stat`` per home, so the second hop was pure queue. One hop is
    also one lane slot, whatever N is (PRA-6794).
    """
    _home, _profile, homes = _two_served(tmp_path)
    runner = _runner(tmp_path, monkeypatch, homes=homes)

    lanes = []

    async def poll_lane(func, *args):
        lanes.append(getattr(func, "__name__", repr(func)))
        return await asyncio.get_running_loop().run_in_executor(None, func, *args)

    runner._run_poll_in_executor = poll_lane
    loop_tick_timing.reset()
    watcher = asyncio.create_task(runner._profile_reconcile_watcher(interval=0.05))
    try:
        await async_wait_until(lambda: loop_tick_timing.snapshot().get("profile_reconcile", {})
                               .get("count", 0) >= 2, timeout=10, what="two watcher ticks")
    finally:
        runner._running = False
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)

    snapshot = loop_tick_timing.snapshot()
    try:
        ticks = snapshot["profile_reconcile"]["count"]
        assert ticks >= 2
        assert set(lanes) == {"_tick_off_loop"}, f"the tick hopped more than once: {lanes!r}"
        assert len(lanes) == ticks, f"{len(lanes)} hops for {ticks} ticks"
        assert snapshot["profile_reconcile_hop"]["count"] == ticks
        # The bodies are timed inside the one hop they ride, so neither can outlast it, and the
        # loop-side apply plus the hop together stay inside the tick that contains them.
        assert snapshot["profile_reconcile_hooks"]["max_s"] <= snapshot["profile_reconcile_hop"]["max_s"]
        assert snapshot["profile_reconcile_read"]["max_s"] <= snapshot["profile_reconcile_hop"]["max_s"]
        assert (snapshot["profile_reconcile_hop"]["max_s"] + snapshot["profile_reconcile_served"]["max_s"]
                <= snapshot["profile_reconcile"]["max_s"] + 0.05)
    finally:
        loop_tick_timing.reset()


@pytest.mark.asyncio
async def test_a_standalone_tick_syncs_hooks_in_one_hop_and_reads_no_profiles(tmp_path, monkeypatch):
    """A standalone gateway has no served set to enumerate, so its tick is hooks only — one hop."""
    _home, _profile, homes = _two_served(tmp_path)
    runner = _runner(tmp_path, monkeypatch, homes=homes)
    runner.config = GatewayConfig(multiplex_profiles=False)

    async def _no_migration():
        return False

    runner._multiplex_for_added_profile = _no_migration

    lanes = []
    reads = []

    async def poll_lane(func, *args):
        lanes.append(getattr(func, "__name__", repr(func)))
        return await asyncio.get_running_loop().run_in_executor(None, func, *args)

    async def _reconcile(**_kwargs):
        reads.append("reconcile")
        return {"added": [], "removed": [], "rescanned": [], "parked": []}

    runner._run_poll_in_executor = poll_lane
    runner.reconcile_served_profiles = _reconcile
    loop_tick_timing.reset()
    watcher = asyncio.create_task(runner._profile_reconcile_watcher(interval=0.05))
    try:
        await async_wait_until(lambda: loop_tick_timing.snapshot().get("profile_reconcile", {})
                               .get("count", 0) >= 2, timeout=10, what="two watcher ticks")
    finally:
        runner._running = False
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)

    snapshot = loop_tick_timing.snapshot()
    try:
        assert reads == [], "a standalone tick reconciled a served set it does not have"
        assert set(lanes) == {"_tick_off_loop"}, lanes
        assert "profile_reconcile_read" not in snapshot, "a standalone tick read the profile roster"
        assert snapshot["profile_reconcile_hooks"]["count"] >= 2
    finally:
        loop_tick_timing.reset()


def test_the_tick_walks_the_profile_roster_once_per_read_pass(monkeypatch):
    """The roster walk is ~130 syscalls of identity probes; demand and enumeration share one.

    Measured on the live host (PRA-6850): one read pass made **566 syscalls, 557 of them `stat`**,
    and each `stat` costs ~0.49 ms inside the gateway process (279 ms live p50 against 7 ms of the
    same work in a quiet one). Walking `profiles/` twice — once for demand's roster, once for the
    serving enumeration — was ~260 of those.
    """
    walks = []

    def counting_iter(*args, **kwargs):
        walks.append(1)
        return []

    monkeypatch.setattr(profiles_mod, "_iter_named_profile_dirs", counting_iter)
    monkeypatch.setattr(capacity_mod, "profile_ceiling", lambda: 4)

    reconcile_mod.GatewayProfileReconcileMixin()._live_profiles_snapshot()

    assert len(walks) == 1, f"the read pass walked profiles/ {len(walks)} times, not once"


def test_the_demand_read_still_precedes_the_served_enumeration(tmp_path, monkeypatch):
    """The ceiling ranks by the demand snapshot, so the same worker refreshes it FIRST."""
    order = []
    real_refresh = demand_mod.refresh
    real_serve = profiles_mod.profiles_to_serve

    def spy_refresh(*args, **kwargs):
        order.append("demand")
        return real_refresh(*args, **kwargs)

    def spy_serve(*args, **kwargs):
        order.append("serve")
        return real_serve(*args, **kwargs)

    monkeypatch.setattr(demand_mod, "refresh", spy_refresh)
    monkeypatch.setattr(profiles_mod, "profiles_to_serve", spy_serve)
    # A ceiling is what turns the ranking on; without one the read is skipped entirely.
    monkeypatch.setattr(capacity_mod, "profile_ceiling", lambda: 4)

    homes, signatures = reconcile_mod.GatewayProfileReconcileMixin()._live_profiles_snapshot()

    assert order[:2] == ["demand", "serve"], order
    assert set(signatures) == set(homes), "every enumerated home gets a signature in the same pass"
