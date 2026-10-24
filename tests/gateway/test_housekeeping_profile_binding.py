"""One bind per served profile per housekeeping tick (PRX-6137).

Every per-profile arm used to re-enter each served profile's scope (re-hydrating its secret scope)
and re-read its ``config.yaml``. On a 17-profile host that fan-out measured ~1.6 s per pass, and a
tick paid it once per arm — ``plugin-update-check`` and ``mcp-config-reconcile`` are both cadence 1
and both fan out.

These are the invariants of the fix: one bind per profile per tick, shared by every arm on that
tick; the arms read ``config.yaml`` through the bind instead of loading it again; and the bind is
dropped when the tick ends, so a config edited mid-run is still seen by the next tick's DRIFT
reconcile (#112445).
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway import run_housekeeping_chores, run_profile_reconcile
from gateway.run_profile_reconcile import (
    begin_housekeeping_tick,
    bound_profile_chore,
    end_housekeeping_tick,
    profile_scoped_chore,
)

HOMES = [("default", "root"), ("secondary", "secondary")]


@pytest.fixture
def served_profiles(tmp_path, monkeypatch):
    """A multiplexed runner over two fake served homes, with the bind's seams counted."""
    counts = {"scopes": [], "hydrations": [], "configs": []}
    homes = [(name, tmp_path / dirname) for name, dirname in HOMES]
    configs = {str(home): {"home": str(home)} for _name, home in homes}

    monkeypatch.setattr(gateway_run, "_multiplex_profile_homes", lambda _config: homes)
    monkeypatch.setattr(gateway_run, "_load_profile_secret_scope",
                        lambda home: counts["hydrations"].append(home) or {"home": str(home)})

    @contextmanager
    def fake_scope(home, prepared_secret_scope=None, **_kwargs):
        counts["scopes"].append(home)
        yield

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)

    def fake_load_config_readonly():
        # The read must happen inside the profile's scope: the home the scope seam recorded is the
        # profile whose config the arm just asked for.
        home = str(counts["scopes"][-1])
        counts["configs"].append(home)
        return configs[home]

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", fake_load_config_readonly)
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    return runner, counts, configs, homes


def test_two_arms_on_one_tick_bind_each_profile_once(served_profiles):
    runner, counts, _configs, homes = served_profiles
    seen = []

    def read_config(binding):
        seen.append((binding.label, binding.load_config()["home"]))

    quiet, bound = profile_scoped_chore(runner, lambda: None), bound_profile_chore(runner, read_config)

    begin_housekeeping_tick()
    try:
        quiet()
        bound()
    finally:
        end_housekeeping_tick()

    # Two arms, four profile visits, one bind each: the secret scope is hydrated once per profile
    # and the second arm's read is free.
    assert counts["hydrations"] == [home for _name, home in homes]
    assert counts["configs"] == [str(home) for _name, home in homes]
    assert seen == [("default", str(homes[0][1])), ("secondary", str(homes[1][1]))]


def test_the_bind_does_not_survive_the_tick(served_profiles):
    """A config edited mid-run must be visible on the next tick — no process-scoped cache."""
    runner, counts, configs, homes = served_profiles
    arm = bound_profile_chore(runner, lambda binding: binding.load_config())
    first_home = str(homes[0][1])

    begin_housekeeping_tick()
    try:
        arm()
    finally:
        end_housekeeping_tick()
    assert counts["configs"] == [str(home) for _name, home in homes]

    configs[first_home] = {"home": "edited"}
    begin_housekeeping_tick()
    try:
        arm()
    finally:
        end_housekeeping_tick()

    assert counts["configs"].count(first_home) == 2
    assert counts["hydrations"].count(homes[0][1]) == 2


def test_an_arm_outside_a_tick_binds_without_sharing(served_profiles):
    """``run_runtime_init`` fans out at startup, off the tick thread: fresh bindings each call."""
    runner, counts, _configs, homes = served_profiles
    arm = profile_scoped_chore(runner, lambda: None)
    arm()
    arm()
    assert len(counts["hydrations"]) == 2 * len(homes)


def test_mcp_reconcile_reads_the_bound_config(served_profiles, monkeypatch):
    """The set compare runs on the tick's config instead of loading ``config.yaml`` again."""
    runner, _counts, configs, homes = served_profiles
    reconciled = []

    @contextmanager
    def no_oauth():
        yield

    monkeypatch.setattr("tools.mcp_oauth.suppress_interactive_oauth", no_oauth)
    monkeypatch.setattr("tools.mcp_tool_discovery.reconcile_mcp_servers_with_config",
                        lambda config=None: reconciled.append(config) or
                        {"removed": [], "added": [], "pending": []})

    arm = run_profile_reconcile._mcp_config_reconciler(runner)
    for _tick in range(2):  # the first tick primes: startup discovery already reflects this config
        begin_housekeeping_tick()
        try:
            arm()
        finally:
            end_housekeeping_tick()

    assert reconciled == [configs[str(home)] for _name, home in homes]


def test_load_mcp_config_honours_a_preloaded_config(monkeypatch):
    """The bind's dict is used as-is: no ``load_config()`` call, no second deepcopy."""
    from tools.mcp_tool_config import _load_mcp_config

    def explode():
        raise AssertionError("a bound config must not be re-read")

    monkeypatch.setattr("hermes_cli.config.load_config", explode)
    servers = _load_mcp_config({"mcp_servers": {"alpha": {"command": "true"}, "off": {"enabled": False}}})
    assert set(servers) == {"alpha", "off"}


def test_plugin_update_check_reads_the_bound_config(monkeypatch):
    """``plugins.*`` comes off the bound config; the cadence never loads ``config.yaml`` itself."""
    binding = run_profile_reconcile.ServedProfile("secondary", None)
    binding.config = {"plugins": {"auto_update_check_hours": 6, "auto_apply": True}}
    seen = {}

    def fake_check(*, log=None, config_get=None):
        assert config_get is not None, "the arm must hand the cadence its bound config reader"
        seen["hours"] = config_get("plugins", "auto_update_check_hours")
        seen["apply"] = config_get("plugins", "auto_apply")

    monkeypatch.setattr("hermes_cli.plugins_cadence.maybe_run_gateway_check", fake_check)
    run_housekeeping_chores._housekeeping_plugin_update_check(binding)

    assert seen == {"hours": 6, "apply": True}


def test_run_due_chores_opens_and_closes_the_binding_window():
    """The tick owns the window — and closes it even when a chore raises."""
    observed = []

    def arm():
        observed.append(getattr(run_profile_reconcile._tick_profiles, "open", False))

    run_housekeeping_chores._run_due_chores([(1, "Probe arm", arm)], 1)

    assert observed == [True]
    assert getattr(run_profile_reconcile._tick_profiles, "open", False) is False
