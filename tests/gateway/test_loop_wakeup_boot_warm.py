"""PRX-5989: the loop-wakeup watcher's first scan must not open every profile's state.db in a tick.

Opening one home's goals SessionDB costs 200-650 ms on this host (5.3 s for a 17-profile fan-out)
against ~5 ms once warm, and it is what the first ``Tick cost loop_wakeup_scan`` line recorded as
p95 1.3-17.1 s. ``_prime_loop_wakeup_stores`` pays it during startup, over exactly the scopes the
scan walks and under each profile's own scope.
"""
from pathlib import Path

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner, _handoff_watch_scopes


def _runner(monkeypatch, homes):
    """GatewayRunner stand-in: multiplexed, serving *homes*, with a warm recorder attached."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    warmed = []

    async def _warm(label, offload=None):
        from hermes_constants import get_hermes_home
        warmed.append((label, str(Path(get_hermes_home()).resolve())))

    runner._warm_goals_session_db = _warm
    monkeypatch.setattr("gateway.run._multiplex_profile_homes", lambda _config: list(homes.items()))
    return runner, warmed


def _homes(tmp_path, names):
    launch = tmp_path / ".hermes"
    homes = {}
    for name in names:
        home = launch / "profiles" / name
        home.mkdir(parents=True)
        homes[name] = home
    return launch, homes


@pytest.mark.asyncio
async def test_boot_warm_opens_every_scanned_profiles_store_under_its_own_scope(tmp_path, monkeypatch):
    """One warm per scanned home, each inside that home's scope (a wrong scope warms another cache)."""
    launch, homes = _homes(tmp_path, ("alpha", "beta", "gamma"))
    monkeypatch.setenv("HERMES_HOME", str(launch))

    runner, warmed = _runner(monkeypatch, homes)
    await runner._prime_loop_wakeup_stores()

    # The launch home is the watcher's unscoped root poll; every secondary gets its own scope.
    # Order is not part of the contract: the homes are warmed concurrently.
    assert sorted(home for _label, home in warmed) == sorted(
        [str(launch.resolve())] + [str(home.resolve()) for home in homes.values()])
    assert {label for label, _home in warmed} == {"loop wakeup boot"}


@pytest.mark.asyncio
async def test_the_warm_covers_exactly_the_scopes_the_first_scan_walks(tmp_path, monkeypatch):
    """The warm set IS the scan set: a home left out is a cold first tick (and only that home)."""
    launch, homes = _homes(tmp_path, ("alpha", "beta"))
    monkeypatch.setenv("HERMES_HOME", str(launch))

    runner, warmed = _runner(monkeypatch, homes)
    await runner._prime_loop_wakeup_stores()

    scanned = [str(Path(home).resolve()) if home is not None else str(launch.resolve())
               for _name, home in _handoff_watch_scopes(runner)]
    assert sorted(home for _label, home in warmed) == sorted(scanned)


@pytest.mark.asyncio
async def test_a_store_that_will_not_open_does_not_fail_the_warm(tmp_path, monkeypatch):
    """Best-effort: one unreadable home leaves the others warm and the boot on its feet."""
    launch, homes = _homes(tmp_path, ("alpha", "beta"))
    monkeypatch.setenv("HERMES_HOME", str(launch))

    runner, _warmed = _runner(monkeypatch, homes)
    seen = []

    async def _warm(label, offload=None):
        from hermes_constants import get_hermes_home
        home = str(Path(get_hermes_home()).resolve())
        if home.endswith("alpha"):
            raise RuntimeError("database is locked")
        seen.append(home)

    runner._warm_goals_session_db = _warm
    await runner._prime_loop_wakeup_stores()

    assert seen, "the healthy homes still warmed"
    assert not any(home.endswith("alpha") for home in seen)
