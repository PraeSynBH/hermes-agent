"""#132547 fixed the own-gateway probe by moving it off the loop; the directory enumeration and the
signature stat sweep around it were left inline on the 30 s reconcile watcher. Under a wide
multiplexer that is a full ``profiles/`` walk plus two ``os.stat`` calls per served profile running
on the event loop while turns execute — the same stall class the probe fix removed.

Invariant: no reconcile cycle performs ``profiles_to_serve`` or ``profile_serve_signature`` on the
event loop thread.
"""
import asyncio
import threading
from pathlib import Path

import pytest

import gateway.run_profile_reconcile as reconcile_mod
import hermes_cli.profiles as profiles_mod
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_reconcile_sweeps_never_run_on_the_event_loop(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("model: {default: test}\n", encoding="utf-8")
    ghost = home / "profiles" / "ghost"  # in the served set but absent on disk: forces a removal
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")

    calls = []

    def record(name):
        calls.append((name, threading.get_ident()))

    real_profiles_to_serve = profiles_mod.profiles_to_serve
    real_signature = reconcile_mod.profile_serve_signature

    def spy_profiles_to_serve(*args, **kwargs):
        record("profiles_to_serve")
        return real_profiles_to_serve(*args, **kwargs)

    def spy_signature(home_):
        record("profile_serve_signature")
        return real_signature(home_)

    # profiles_to_serve is imported inside the methods from its module; profile_serve_signature is
    # read as a module global by _serve_signatures.
    monkeypatch.setattr(profiles_mod, "profiles_to_serve", spy_profiles_to_serve)
    monkeypatch.setattr(reconcile_mod, "profile_serve_signature", spy_signature)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = True
    runner._primary_profile_name = "default"
    runner._served_profile_homes = {"default": home, "worker": profile, "ghost": ghost}
    runner._served_profile_signatures = {
        "default": real_signature(home), "worker": real_signature(profile), "ghost": real_signature(ghost)}

    async def _unserve(name, home):
        return None

    runner._unserve_profile = _unserve
    runner._live_resource_claims = lambda active: {}
    runner._record_served_profiles = lambda active, profile_homes: runner._note_served_profiles(profile_homes)

    loop_thread = threading.get_ident()
    await runner.reconcile_served_profiles(reason="watcher")

    names = [name for name, _ident in calls]
    # Both enumeration sites ran (the served-set diff and the post-change rescan) ...
    assert names.count("profiles_to_serve") == 2, names
    # ... and the per-profile signature sweep ran for the served profile.
    assert "profile_serve_signature" in names
    # None of them on the loop thread.
    on_loop = [name for name, ident in calls if ident == loop_thread]
    assert on_loop == [], f"ran on the event loop thread: {on_loop}"
