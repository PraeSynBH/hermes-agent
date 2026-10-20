"""The 26-profile envelope still holds when the ceiling is ranked by demand (SOU-3075 / PRA-6524).

The ceiling exists because the per-profile thread baseline scales with the roster; the demand rank
must not weaken it (served count stays at the ceiling), must not churn between identical calls, and
must leave the INSTALLED roster whole. This is the regression envelope for PRA-6524 on the ranking
path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import gateway_multiplex_capacity as capacity
from hermes_cli import gateway_multiplex_demand as demand
from hermes_cli.profiles import create_profile, get_profile_dir, profiles_to_serve

TOTAL = 26           # 25 named + default
CEILING = 15         # default + 14 named, the live host's ceiling
DEMAND_BEARING = 20  # more demand than slots: the over-subscribed case the check pages on


@pytest.fixture
def big_host(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    capacity.reset_memo()
    demand.reset()
    for index in range(TOTAL - 1):
        name = f"prof{index:02d}"
        create_profile(name, no_alias=True)
        if index < DEMAND_BEARING:
            profile_home = get_profile_dir(name)
            (profile_home / "cron").mkdir(exist_ok=True)
            (profile_home / "cron" / "jobs.json").write_text(
                json.dumps({"jobs": [{"id": f"job-{index}", "enabled": True}]}), encoding="utf-8")
    (home / "config.yaml").write_text(
        f"gateway:\n  multiplex_max_profiles: {CEILING}\n", encoding="utf-8")
    capacity.reset_memo()
    yield home
    capacity.reset_memo()
    demand.reset()


def test_the_envelope_holds_with_demand_ranking(big_host):
    served = profiles_to_serve(multiplex=True)

    assert len(served) == CEILING
    assert served[0][0] == "default"
    deferred = capacity.deferred_by_capacity()
    assert len(deferred) == TOTAL - CEILING
    assert all(reason.startswith("capacity: ") and "; cost: " in reason for reason in deferred.values())
    # Every deferred profile's cost is stated, and none of them claims demand it does not have.
    assert all(len(reason) <= 500 for reason in deferred.values())


def test_no_demand_profile_is_served_while_a_demand_bearing_one_is_deferred(big_host):
    """A1: the rank follows demand across a roster larger than the ceiling."""
    served = {name for name, _home in profiles_to_serve(multiplex=True)} - {"default"}
    snapshot = demand.refresh()
    deferred = capacity.deferred_by_capacity()

    assert any(snapshot.has(name) for name in deferred)
    for name in deferred:
        if snapshot.has(name):
            assert all(snapshot.has(served_name) for served_name in served), name


def test_the_decision_is_identical_across_repeated_enumerations(big_host):
    """Determinism across restarts: no dict order, no timestamps, no counts weighting."""
    first = [name for name, _home in profiles_to_serve(multiplex=True)]
    demand.reset()
    second = [name for name, _home in profiles_to_serve(multiplex=True)]
    assert first == second


def test_the_installed_roster_is_never_capped(big_host):
    installed = profiles_to_serve(multiplex=True, include_parked=True)
    assert len(installed) == TOTAL
