"""Serving a multiplexed secondary profile must not read its config or walk plugins on the loop.

PRX-5524 traced the gateway's continuous loop starvation to inline I/O on the loop, and the
2026-10-19 stall dump named the exact shape: ``task_step`` -> ``gen_iternext`` -> ``_io_open``
inside ``importlib.metadata.entry_points`` reached from
``_load_secondary_profile_config`` -> ``discover_plugins``. One 10 s healthz probe plus a 13 s p50
is what a profile-add tick cost the whole fleet.

The contract here is twofold: the profile's config load and plugin discovery run in a worker
thread, AND they still run under that profile's scope (``asyncio.to_thread`` copies the context,
which is what makes the hop safe). Gateway-state mutations stay on the loop.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import gateway.config as config_module
import gateway.run as run_module
import hermes_cli.plugins as plugins_module
from gateway.run_adapters import GatewayAdapterLifecycleMixin
from gateway.run_profile_reconcile import _load_profile_serving_config


class _Runner:
    """The gateway state ``_load_secondary_profile_config`` mutates — recorded, never real."""

    config = SimpleNamespace(multiplex_profiles=True)

    def __init__(self):
        self.rewired = []
        self.hooks = 0

    def _subscribe_plugin_rewire(self, manager, profile_name, profile_home):
        self.rewired.append((profile_name, Path(profile_home)))

    def _register_config_hooks(self, message, profile_name, **kwargs):
        self.hooks += 1

    def _snapshot_profile_busy_modes(self, profile_name, config):
        self.snapshotted = (profile_name, config)


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    home = root / "profiles" / "bot2"
    home.mkdir(parents=True)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n")
    (home / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n"
        "telegram:\n  require_mention: false\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def test_profile_config_and_discovery_run_off_the_loop_in_the_profile_scope(
    profile_home, monkeypatch,
):
    loop_thread = threading.get_ident()
    seen = {}

    def _discover_plugins():
        from hermes_constants import get_hermes_home
        seen["discover_thread"] = threading.get_ident()
        seen["discover_home"] = Path(get_hermes_home()).resolve()
        seen["discover_ran"] = True

    def _load_gateway_config():
        seen["config_thread"] = threading.get_ident()
        seen["config_ran"] = True
        return {"profile": "bot2"}

    monkeypatch.setattr(plugins_module, "discover_plugins", _discover_plugins)
    monkeypatch.setattr(run_module, "_load_gateway_config", _load_gateway_config)

    runner = _Runner()
    profile_cfg = asyncio.run(
        GatewayAdapterLifecycleMixin._load_secondary_profile_config(
            runner, "bot2", profile_home))

    assert seen.get("discover_ran"), "plugin discovery never ran — the profile's plugins were skipped"
    assert seen["discover_thread"] != loop_thread, (
        "plugin discovery ran on the event loop; that filesystem walk is the PRX-5524 stall")
    assert seen["config_thread"] != loop_thread, (
        "the profile's gateway config was read on the event loop")
    assert seen["discover_home"] == profile_home.resolve(), (
        "discovery escaped the served profile's scope across the thread hop")
    assert seen.get("config_ran"), "the profile's gateway config was never loaded"
    assert profile_cfg is not None
    # The gateway-state mutations still happen, on the loop, after the off-loop reads.
    assert runner.rewired == [("bot2", profile_home)]
    assert runner.hooks == 1


def test_helper_returns_both_configs(profile_home, monkeypatch):
    """The worker-thread body hydrates the runtime config first, then the gateway config."""
    order = []
    monkeypatch.setattr(run_module, "_load_gateway_config", lambda: order.append("runtime") or {})
    monkeypatch.setattr(config_module, "load_gateway_config", lambda: order.append("gateway") or {"cfg": 1})

    runtime_cfg, cfg, violation = _load_profile_serving_config(profile_home)

    assert order == ["runtime", "gateway"]
    assert runtime_cfg == {}
    assert cfg == {"cfg": 1}
    assert violation is None
