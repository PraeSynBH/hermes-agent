"""Admission verdicts name the wait that failed them (PRX-5725).

PRX-5625 taught ``admit_owner`` to wait a boot out, and mapped two different faults onto the word
"boot" while doing it: ``discover_gateway_endpoint`` turned a ``control_timeout`` (a live owner that
did not answer ``identify``) into ``starting: deadline``, and any in-loop ``TimeoutError`` into the
same. The result was unchanged in size and worse in detail — 183 runs in 5 h, every one
``phase=runtime-boot`` at ``elapsed=120.0-120.1s``, on a daemon that admitted a lone caller in
0.5-1.3 s, with 35 of them on profiles the multiplexer never served at all (they could not have been
admitted at any budget).

The contracts pinned here:

* one observation may spend at most ``PROBE_TIMEOUT`` of the budget, so a silent owner costs a probe
  instead of the run;
* a silent owner is reported as ``control_timeout``, naming the socket and how long ``identify`` went
  unanswered — never as a boot;
* a profile the live multiplexer does not serve, with no owner of its own, fails in seconds with
  ``profile_not_served`` and the served list, and never walks the boot path or spawns a daemon;
* a ``standalone: true`` profile the policy never serves keeps its own start path.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest


def _dead_owner_lock(home: Path) -> None:
    """The devops shape: a lock naming a pid that is gone, so the home has no owner of its own."""
    home.mkdir(parents=True, mode=0o700, exist_ok=True)
    (home / "gateway.lock").write_text(json.dumps({
        "pid": 999_999_999, "kind": "hermes-gateway", "start_time": 1, "hermes_home": str(home),
    }), encoding="utf-8")


@pytest.mark.platforms("posix")
def test_a_silent_owner_is_a_control_timeout_not_a_boot(tmp_path):
    """An owner that takes the request and never answers keeps its own verdict, named (PRX-5725)."""
    from gateway.control_socket import GatewayControlServer
    from hermes_cli import gateway_runtime as runtime

    async def probe():
        home = tmp_path / "profile"
        home.mkdir(mode=0o700)
        stall = threading.Event()  # set at teardown so the stalled handler threads exit

        def identify():
            stall.wait(30)  # outlives the caller's deadline
            return {"runtime_protocol": 1, "state": "ready", "served_profiles": []}

        server = GatewayControlServer(home, verb_handlers={"identify": identify})
        assert await server.start()
        try:
            before = time.monotonic()
            result = await asyncio.to_thread(runtime.ensure_gateway_runtime, home, timeout=3.0)
            elapsed = time.monotonic() - before
        finally:
            stall.set()
            await server.stop()
        return result, elapsed

    result, elapsed = asyncio.run(probe())
    assert (result.state, result.reason_code) == ("inaccessible", "control_timeout")
    assert "did not answer identify" in (result.detail or "")
    assert elapsed < 8, f"the budget must bound the wait, took {elapsed:.1f}s"


def test_one_silent_observe_cannot_spend_the_whole_budget(tmp_path, monkeypatch):
    """Each observation is capped at PROBE_TIMEOUT and retried, so the budget buys attempts.

    A single ``identify`` inheriting ``remaining(deadline)`` is what turned 183 runs into
    ``elapsed=120.0-120.1s`` runtime-boot losses: the one call that could have reported a control
    timeout instead spent the entire admission.
    """
    from hermes_cli import gateway_runtime as runtime

    budget, probe_cap = 0.4, 0.05
    monkeypatch.setattr(runtime, "PROBE_TIMEOUT", probe_cap)
    observed: list[float | None] = []

    def discover(home, *, timeout=None):
        observed.append(timeout)
        return runtime.GatewayDiscovery("inaccessible", reason_code="control_timeout",
                                        detail="silent")

    monkeypatch.setattr(runtime, "discover_gateway_endpoint", discover)
    result = runtime.ensure_gateway_runtime(tmp_path / "profile", timeout=budget)

    assert len(observed) >= 3, f"a bounded probe is retried, not spent once: {observed}"
    assert all(cap is not None and cap <= probe_cap for cap in observed), observed
    assert (result.state, result.reason_code) == ("inaccessible", "control_timeout"), (
        "a silent owner must not be relabelled as a boot when the budget runs out")


@pytest.mark.platforms("posix")
def test_a_profile_the_live_multiplexer_does_not_serve_fails_in_seconds(tmp_path, monkeypatch):
    """`general`/`devops`: policy says served, the live set says no, no owner — a wait that cannot end."""
    from gateway.control_socket import GatewayControlServer
    from hermes_cli import gateway_runtime as runtime
    from hermes_cli import gateway_runtime_start as start

    root = tmp_path / ".hermes"
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    other = root / "profiles" / "alpha"
    (other / "config.yaml").parent.mkdir(parents=True, mode=0o700)
    (other / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    home = root / "profiles" / "devops"
    _dead_owner_lock(home)
    (home / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    spawned: list[Path] = []
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", lambda target, **k: spawned.append(Path(target)))

    async def probe():
        payload = {"runtime_protocol": 1, "state": "ready",
                   "served_profiles": [{"profile_id": str(root), "home": str(root)},
                                       {"profile_id": str(other), "home": str(other)}]}
        server = GatewayControlServer(root, verb_handlers={"identify": lambda: payload})
        assert await server.start()
        try:
            before = time.monotonic()
            result = await asyncio.to_thread(runtime.ensure_gateway_runtime, home, timeout=10.0)
            return result, time.monotonic() - before
        finally:
            await server.stop()

    result, elapsed = asyncio.run(probe())
    assert (result.state, result.reason_code) == ("inaccessible", "profile_not_served")
    assert elapsed < 5, f"an unserved profile must not walk the boot path, took {elapsed:.1f}s"
    detail = result.detail or ""
    assert "devops" in detail and "2 profiles" in detail
    assert ".hermes" in detail and "alpha" in detail, "the verdict names the served set"
    assert spawned == [], "an unserved profile must never spawn a competing daemon"


@pytest.mark.platforms("posix")
def test_a_standalone_profile_still_starts_its_own_owner(tmp_path, monkeypatch):
    """The unserved verdict is for the policy's served set only: `standalone: true` keeps its path."""
    from gateway.control_socket import GatewayControlServer
    from hermes_cli import gateway_runtime as runtime
    from hermes_cli import gateway_runtime_service as service
    from hermes_cli import gateway_runtime_start as start

    root = tmp_path / ".hermes"
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    home = root / "profiles" / "sioux"
    _dead_owner_lock(home)
    (home / "config.yaml").write_text("gateway:\n  standalone: true\n", encoding="utf-8")
    spawned: list[Path] = []
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", lambda target, **k: spawned.append(Path(target)))

    async def probe():
        payload = {"runtime_protocol": 1, "state": "ready",
                   "served_profiles": [{"profile_id": str(root), "home": str(root)}]}
        server = GatewayControlServer(root, verb_handlers={"identify": lambda: payload})
        assert await server.start()
        try:
            return await asyncio.to_thread(runtime.ensure_gateway_runtime, home, timeout=1.0)
        finally:
            await server.stop()

    result = asyncio.run(probe())
    assert result.reason_code != "profile_not_served"
    assert spawned == [home.resolve()]


def test_boot_evidence_names_create_time_and_start_time(tmp_path):
    """A boot verdict can say how long this host's boot took, from the lifecycle ledger (AC2)."""
    from hermes_cli import gateway_runtime as runtime

    home = tmp_path / "profile"
    (home / "state").mkdir(parents=True, mode=0o700)
    ledger = home / "state" / "gateway.lifecycle.json"
    ledger.write_text(json.dumps({"phase": "running", "pid": os.getpid(),
                                  "create_time": 1792519657.06,
                                  "started_at": "2026-10-20T18:08:03.918183+00:00"}),
                      encoding="utf-8")

    evidence = runtime.boot_evidence(home)
    assert evidence is not None
    assert "create_time 2026-10-20T18:07:37+00:00" in evidence
    assert "start_time 2026-10-20T18:08:03.918183+00:00" in evidence
    assert "= 26.9s" in evidence

    ledger.write_text("not json", encoding="utf-8")
    assert runtime.boot_evidence(home) is None
    assert runtime.boot_evidence(tmp_path / "never-booted") is None
