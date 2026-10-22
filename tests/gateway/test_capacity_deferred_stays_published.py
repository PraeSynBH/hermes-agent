"""A profile the ``gateway.multiplex_max_profiles`` ceiling defers stays published as parked (PRX-5734).

The ceiling parks a profile outside the served set — no adapters, no session authority — and that
verdict is what a client and an operator read: admission sees the ``capacity:`` reason as the
PROVISIONAL ``profile_serving`` wait and only reports ``profile_not_served`` once the rank has had
its re-rank window (PRX-5963).
Two writers used to fight over it: the reconcile's deleted-profile sweep treated every parked name
missing from the CAPPED roster as deleted and unparked it, and the capacity fold (which re-parks
them) only ran when a tick actually applied a change. So on any quiet tick the host published
``parked_profiles: {}`` while 28 profiles were in fact unserved: the outage was named in a log line
and nowhere a human looks.

These pin the durable invariant: the parked map (and ``gateway_state.json``) names every
ceiling-deferred profile after EVERY reconcile tick, changed or not.
"""
import json

import pytest
from unittest.mock import patch

from hermes_cli import gateway_multiplex_capacity as capacity
from gateway.status import flush_runtime_status
from tests.gateway.test_multiplex_hot_serve import _mkprofile, _runner

CAPACITY_REASON = "capacity: 3 profiles per gateway (gateway.multiplex_max_profiles)"


def _parked_state(home):
    flush_runtime_status()
    return json.loads((home / "gateway_state.json").read_text(encoding="utf-8")).get("parked_profiles")


@pytest.fixture
def capped(tmp_path, monkeypatch):
    monkeypatch.setattr(capacity, "profile_ceiling", lambda *_a, **_k: 3)
    monkeypatch.setattr(capacity, "serve_priority", lambda *_a, **_k: ())
    capacity.reset_memo()
    yield
    capacity.reset_memo()


@pytest.mark.asyncio
async def test_quiet_tick_keeps_ceiling_deferred_profiles_parked(tmp_path, monkeypatch, capped):
    runner, home = _runner(tmp_path, monkeypatch)
    for name in ("alpha", "bravo", "charlie", "delta"):
        _mkprofile(home, name, f"DISCORD_BOT_TOKEN={name}-token\n")

    with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        await runner._start_secondary_profile_adapters()

    assert set(runner._served_profile_homes) == {"default", "alpha", "bravo"}
    assert set(runner.session_runtime_descriptor["parked_profiles"]) == {"charlie", "delta"}
    assert set(_parked_state(home)) == {"charlie", "delta"}

    # A watcher tick that changes nothing applies no diff — and must not retract the verdict.
    await runner.reconcile_served_profiles(reason="watcher")

    assert set(runner.session_runtime_descriptor["parked_profiles"]) == {"charlie", "delta"}
    assert set(_parked_state(home)) == {"charlie", "delta"}
    # Since SOU-3075 the reason also states the trade (counts only); the prefix and the ceiling
    # sentence this test pins are unchanged, and the whole map is still published every tick.
    reasons = set(_parked_state(home).values())
    assert len(reasons) == 1
    (reason,) = reasons
    assert reason.startswith(f"{CAPACITY_REASON} — this profile is deferred")
    assert "; cost: " in reason


@pytest.mark.asyncio
async def test_raised_ceiling_unparks_on_the_next_tick(tmp_path, monkeypatch):
    """The verdict follows the config: once the roster fits, nothing stays parked by capacity."""
    runner, home = _runner(tmp_path, monkeypatch)
    for name in ("alpha", "bravo", "charlie"):
        _mkprofile(home, name, f"DISCORD_BOT_TOKEN={name}-token\n")

    monkeypatch.setattr(capacity, "profile_ceiling", lambda *_a, **_k: 3)
    monkeypatch.setattr(capacity, "serve_priority", lambda *_a, **_k: ())
    capacity.reset_memo()
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        await runner._start_secondary_profile_adapters()
    assert set(runner._served_profile_homes) == {"default", "alpha", "bravo"}
    assert set(runner.session_runtime_descriptor["parked_profiles"]) == {"charlie"}

    monkeypatch.setattr(capacity, "profile_ceiling", lambda *_a, **_k: None)
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        result = await runner.reconcile_served_profiles(reason="watcher")

    assert result["added"] == ["charlie"]
    assert runner.session_runtime_descriptor["parked_profiles"] == {}
    assert _parked_state(home) == {}
