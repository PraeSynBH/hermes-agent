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


_CAPACITY_PARK = ("capacity: 16 profiles per gateway (gateway.multiplex_max_profiles) — this profile "
                  "is deferred; cost: 1 Paperclip agents, 0 enabled cron jobs, 0 adapter credentials, "
                  "work: agents — raise the ceiling or move it up gateway.multiplex_serve_priority")


def _parked_payload(root: Path, other: Path, reason: str) -> dict:
    """``identify`` from a READY multiplexer that serves ``other`` and parks ``devops`` by ``reason``."""
    return {"runtime_protocol": 1, "state": "ready",
            "served_profiles": [{"profile_id": str(root), "home": str(root)},
                                {"profile_id": str(other), "home": str(other)}],
            "parked_profiles": {"devops": reason}}


def _multiplexer_fixture(tmp_path):
    """``(root, other, home)``: a host whose policy serves a named profile it does not list."""
    root = tmp_path / ".hermes"
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    other = root / "profiles" / "alpha"
    (other / "config.yaml").parent.mkdir(parents=True, mode=0o700)
    (other / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    home = root / "profiles" / "devops"
    _dead_owner_lock(home)
    (home / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    return root, other, home


@pytest.mark.platforms("posix")
def test_a_ceiling_deferral_is_a_wait_not_a_permanent_verdict(tmp_path, monkeypatch):
    """PRX-5963: ``capacity:`` in ``parked_profiles`` is PROVISIONAL — the rank is rebuilt every tick.

    devops was parked by the ceiling during boot (the rank was still cold) and served 2 m 47 s later
    by the first warm tick; three runs dispatched in between read the park as "can never complete"
    and died. A ceiling deferral therefore reports ``profile_serving`` (a retryable wait), while
    every other park reason keeps the terminal ``profile_parked``.
    """
    from gateway.control_socket import GatewayControlServer
    from hermes_cli import gateway_runtime as runtime

    root, other, home = _multiplexer_fixture(tmp_path)

    async def probe(payload):
        server = GatewayControlServer(root, verb_handlers={"identify": lambda: payload})
        assert await server.start()
        try:
            return await asyncio.to_thread(runtime.ensure_gateway_runtime, home, timeout=10.0)
        finally:
            await server.stop()

    deferred = asyncio.run(probe(_parked_payload(root, other, _CAPACITY_PARK)))
    assert (deferred.state, deferred.reason_code) == ("inaccessible", "profile_serving")
    assert deferred.retry_after_s is not None, "a provisional verdict must say when to re-observe"
    assert _CAPACITY_PARK[:40] in (deferred.detail or ""), deferred.detail

    # A park that survives ticks (unusable store, home owned elsewhere) stays terminal.
    settled = asyncio.run(probe(_parked_payload(root, other, "session store unusable: locked")))
    assert (settled.state, settled.reason_code) == ("inaccessible", "profile_parked")
    assert "session store unusable" in (settled.detail or "")


@pytest.mark.platforms("posix")
def test_admission_waits_a_ceiling_deferral_out_and_files_itself_as_demand(monkeypatch):
    """The wait is bounded by the ceiling's own re-rank, and it files the sighting that moves it."""
    from types import SimpleNamespace

    from hermes_cli import gateway_client, gateway_multiplex_served, gateway_runtime

    endpoint = SimpleNamespace(api_origin="http://127.0.0.1:1", profile_id="p", instance_id="i",
                              code_sha="0b10", control_home=None)
    queued = [gateway_runtime.GatewayDiscovery(
        "inaccessible", reason_code="profile_serving", detail=_CAPACITY_PARK, retry_after_s=0.02),
        gateway_runtime.GatewayDiscovery("ready", endpoint=endpoint)]
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime",
                        lambda *a, **k: queued.pop(0) if queued else gateway_runtime.GatewayDiscovery(
                            "ready", endpoint=endpoint))
    monkeypatch.setattr(gateway_runtime, "gateway_profile_home", lambda home: Path("/homes/devops"))
    wanted: list[str] = []
    monkeypatch.setattr(gateway_multiplex_served, "note_profile_wanted",
                        lambda name, **k: wanted.append(name) or True)

    def attach(*args, **kwargs):
        raise gateway_client.GatewayClientError("ATTACHED")

    monkeypatch.setattr(gateway_client, "_session_ticket", attach)

    async def run():
        async with gateway_client.connect_gateway():
            raise AssertionError("the stub always refuses")

    with pytest.raises(gateway_client.GatewayClientError) as caught:
        asyncio.run(run())
    assert "ATTACHED" in str(caught.value), "a deferred profile must be waited for, not failed at once"
    assert wanted == ["devops"], "the waiting run has to be filed as live demand for the next tick"


@pytest.mark.platforms("posix")
def test_a_deferral_that_outlives_the_window_names_the_ceiling(monkeypatch):
    """No slot inside the window: the run is booked as a DEFERRAL, not as an unusable profile.

    PRX-5972: the window expiring used to raise ``GatewayUnavailableError("Gateway inaccessible:
    profile_not_served …")`` — the verdict for a home that can never be served — so the Paperclip
    adapter booked ``adapter_failed`` onto the agent's card for a plain ceiling decision (run
    ``979b4fe2``: 75 s, exit 1). The settled deferral now carries its own reason code, a retry
    schedule, and the ceiling's own numbers.
    """
    from hermes_cli import gateway_client, gateway_multiplex_served, gateway_runtime

    monkeypatch.setattr(gateway_client, "PROFILE_SERVE_WAIT_S", 0.25)
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime",
                        lambda *a, **k: gateway_runtime.GatewayDiscovery(
                            "inaccessible", reason_code="profile_serving", detail=_CAPACITY_PARK,
                            retry_after_s=0.02))
    monkeypatch.setattr(gateway_runtime, "gateway_profile_home", lambda home: Path("/homes/devops"))
    monkeypatch.setattr(gateway_multiplex_served, "note_profile_wanted", lambda name, **k: True)

    async def run():
        async with gateway_client.connect_gateway():
            raise AssertionError("a settled deferral must not admit the run")

    before = time.monotonic()
    with pytest.raises(gateway_client.GatewayProfileDeferredError) as caught:
        asyncio.run(run())
    elapsed = time.monotonic() - before
    error, line = caught.value, str(caught.value)
    assert error.reason_code == "profile_deferred_capacity", line
    assert error.retryable is True and error.retry_after_s is not None, line
    assert "retryAfterS=" in line and "retryable=true" in line and "profile=devops" in line, line
    assert "profile_not_served" not in line, "a capacity booking must not borrow the unserved verdict"
    assert "no slot among" in line and "ceiling 16" in line, line
    assert "capacity: 16 profiles" in line, "the park's own cost/remedy clause has to survive"
    assert "boot" not in line and "starving" not in line, line
    assert elapsed < 5, f"the deferral window bounds the wait, took {elapsed:.1f}s"


@pytest.mark.platforms("posix")
def test_a_deferral_a_reconcile_already_re_ranked_fails_fast(monkeypatch):
    """A park that already carries a waiting client's sighting cannot be won by waiting again.

    ``cost_clause`` appends the live wanted-sighting to the park reason, so a reason that carries it
    was published by a tick that read the strongest demand signal there is and still kept the profile
    outside the ceiling. 16 agents sit behind parked profiles (PRX-5972), and each of their runs used
    to pay the whole window for a rank that had already been asked to move.
    """
    from hermes_cli import gateway_client, gateway_multiplex_demand, gateway_multiplex_served
    from hermes_cli import gateway_runtime

    seen = gateway_multiplex_demand.wanted_signal("devops")
    park = _CAPACITY_PARK.replace(" — raise the ceiling", f", {seen} — raise the ceiling")
    monkeypatch.setattr(gateway_client, "PROFILE_SERVE_WAIT_S", 30.0)
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime",
                        lambda *a, **k: gateway_runtime.GatewayDiscovery(
                            "inaccessible", reason_code="profile_serving", detail=park,
                            retry_after_s=5.0))
    monkeypatch.setattr(gateway_runtime, "gateway_profile_home", lambda home: Path("/homes/devops"))
    sightings: list[str] = []
    monkeypatch.setattr(gateway_multiplex_served, "note_profile_wanted",
                        lambda name, **k: sightings.append(name) or True)

    async def run():
        async with gateway_client.connect_gateway():
            raise AssertionError("a settled deferral must not admit the run")

    before = time.monotonic()
    with pytest.raises(gateway_client.GatewayProfileDeferredError) as caught:
        asyncio.run(run())
    elapsed = time.monotonic() - before
    line = str(caught.value)
    assert "profile_deferred_capacity" in line and "already deferred" in line, line
    assert sightings == [], "a run must not join a queue the reconcile has already answered"
    assert elapsed < 5, f"the window is not re-spent on a rank that already saw the sighting ({elapsed:.1f}s)"


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


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("aliased", [True, False], ids=["lock-aliases-multiplexer", "home-owns-its-owner"])
def test_runtime_unserved_is_not_fooled_by_a_borrowed_lock(tmp_path, monkeypatch, aliased):
    """The `agent`/`designer` window: a served profile gets *unserved* while work is queued.

    Its ``gateway.lock`` names the multiplexer's own pid (hermes_home rewritten, flock held by the
    owner), so "the lock is held" looks like a boot in progress for as long as the unserving lasts —
    144 runs on 2026-10-20 (07:09-10:05, `agent`) burned 240 s each on it. A lock that names the
    multiplexer is not an owner of its own, so this must be `profile_not_served`; a home with a real
    owner of its own must keep its control-timeout/start path.
    """
    from gateway.control_socket import GatewayControlServer
    from hermes_cli import gateway_runtime as runtime

    root = tmp_path / ".hermes"
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    home = root / "profiles" / "agent"
    home.mkdir(parents=True, mode=0o700)
    (home / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    borrowed, genuine = os.getpid(), os.getpid() + 1_000_000
    (root / "gateway.pid").write_text(json.dumps({"pid": borrowed}), encoding="utf-8")
    # The seam: a live gateway process cannot be synthesized here, so the identity probe is replaced
    # and only the pid it reports is exercised (the ownership decision itself is real).
    monkeypatch.setattr(runtime, "_owner_pid", lambda path: borrowed if aliased else genuine)
    from hermes_cli import gateway_runtime_service as service, gateway_runtime_start as start
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
    if aliased:
        assert (result.state, result.reason_code) == ("inaccessible", "profile_not_served")
        assert "agent" in (result.detail or "")
        assert spawned == [], "a borrowed lock must not send this home down the boot/start path"
    else:
        assert result.reason_code != "profile_not_served", "this owner is wedged, not unserved"


def test_a_garbage_ticket_grant_is_an_admission_error_not_a_traceback(monkeypatch):
    """The owner answering the ticket verb with garbage must not kill the run with a bare traceback.

    11 fleet runs on 2026-10-20 died as ``Traceback ... DiscoveryError: invalid_control_response``:
    no phase, no class, so no retry chain could classify them (PRX-5725).
    """
    from types import SimpleNamespace

    from hermes_cli import gateway_client, gateway_runtime
    from hermes_cli.gateway_runtime_discovery import DiscoveryError

    endpoint = SimpleNamespace(api_origin="http://127.0.0.1:1", profile_id="p", instance_id="i",
                               code_sha="0b10", control_home=None)
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime",
                        lambda *a, **k: gateway_runtime.GatewayDiscovery("ready", endpoint=endpoint))

    def garbage(*args, **kwargs):
        raise DiscoveryError("invalid_control_response")

    monkeypatch.setattr(gateway_client, "_session_ticket", garbage)

    async def run():
        async with gateway_client.connect_gateway():
            raise AssertionError("a garbage grant must not admit the run")

    with pytest.raises(gateway_client.GatewayClientError) as caught:
        asyncio.run(run())
    line = str(caught.value)
    assert "phase=session-ticket" in line and "invalid_control_response" in line, line


def _draining_discovery(runtime):
    """A restart's after-turn wait as ``ensure`` hands it out: draining, but dialable (#139041)."""
    from types import SimpleNamespace

    endpoint = SimpleNamespace(api_origin="http://127.0.0.1:1", profile_id="p", instance_id="waiting-owner",
                               code_sha="0b10", control_home=None)
    return runtime.GatewayDiscovery("draining", endpoint=endpoint, reason_code="restart_wait")


@pytest.mark.platforms("posix")
def test_new_work_waits_out_the_restart_wait_owner_instead_of_attaching(monkeypatch):
    """A spawn must not attach to a restart's after-turn wait (PRX-5791).

    The wait owner is attachable on purpose — the turn parked on approval/clarify has to be
    answerable (#139041) — but PRX-5787 closed its claim path from T=0. A finite run that dialled it
    booted, printed ``Error: runtime_draining`` at its first ``session.create`` and exited 1: 101
    heartbeat runs across the 21:38 and 23:05-23:25 drains, every one a spawn spent on a wait. A run
    with new work waits for the owner that replaces the wait, inside its one admission budget.
    """
    from hermes_cli import gateway_client, gateway_runtime

    observed: list[float | None] = []
    monkeypatch.setattr(gateway_client, "DRAIN_POLL_INTERVAL_S", 0.02)
    monkeypatch.setattr(gateway_client, "admission_budget_seconds", lambda: 0.3)

    def ensure(*args, **kwargs):
        observed.append(kwargs.get("timeout"))
        return _draining_discovery(gateway_runtime)

    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", ensure)

    def attach(*args, **kwargs):
        raise AssertionError("a run with new work must not attach to a restart-wait owner")

    monkeypatch.setattr(gateway_client, "_session_ticket", attach)

    async def run():
        async with gateway_client.connect_gateway(new_work=True):
            raise AssertionError("nothing may be submitted to a shutting-down owner")

    with pytest.raises(gateway_client.GatewayAdmissionTimeout) as caught:
        asyncio.run(run())
    line = str(caught.value)
    assert "phase=runtime-drain" in line and "restart_wait" in line, line
    assert len(observed) >= 3, f"the drain is re-observed inside the budget, not spent by one call: {observed}"


@pytest.mark.platforms("posix")
def test_an_interactive_client_still_attaches_to_the_restart_wait_owner(monkeypatch):
    """Why the wait stays reachable at all: its own turn may be parked on a human (#139041)."""
    from hermes_cli import gateway_client, gateway_runtime

    discovery = _draining_discovery(gateway_runtime)
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", lambda *a, **k: discovery)

    def attach(*args, **kwargs):
        raise gateway_client.GatewayClientError("ATTACHED")

    monkeypatch.setattr(gateway_client, "_session_ticket", attach)

    async def run():
        async with gateway_client.connect_gateway():
            raise AssertionError("the stub always refuses")

    with pytest.raises(gateway_client.GatewayClientError) as caught:
        asyncio.run(run())
    assert "ATTACHED" in str(caught.value)


def test_a_finite_run_is_the_client_that_has_new_work(monkeypatch):
    """The one-shot marker (``-q``/``-z``) is what selects the wait; an interactive viewer does not."""
    from hermes_cli.gateway_connection import GatewayConnection

    for key in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    assert GatewayConnection().new_work is True
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION")
    assert GatewayConnection().new_work is False


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


def test_a_settled_deferral_leaves_the_process_a_retry_verdict(monkeypatch, capsys):
    """The deferral's exit status is EX_TEMPFAIL, not the 1 that books a failed run (PRX-5972).

    A reason code on the exception is only visible to a reader inside this process. What a dispatcher
    and the Paperclip adapter act on is the process EXIT STATUS (PRX-5964 gave the never-admitted
    class that same contract); run ``979b4fe2`` spent 75 s on a ceiling wait and exited 1, so the
    agent's card carried a failure for a capacity decision. EX_TEMPFAIL says the one true thing:
    nothing was submitted, and re-dispatch is safe.
    """
    from hermes_cli import gateway_chat, gateway_chat_startup
    from hermes_cli.gateway_client import GatewayProfileDeferredError
    from hermes_cli.turn_exit import TURN_NOT_ADMITTED_EXIT_CODE

    deferral = GatewayProfileDeferredError(
        "Gateway deferred: profile_deferred_capacity (retryAfterS=300, retryable=true, profile=cmo): "
        "no slot among 17 served of 43 installed (ceiling 17)",
        300.0, profile_name="cmo")

    async def run(args, emitter=None):
        raise deferral

    monkeypatch.setattr(gateway_chat_startup, "ensure_launch_provider", lambda args: True)
    monkeypatch.setattr(gateway_chat, "run_gateway_chat", run)
    assert gateway_chat.launch_from_kwargs({"q": "literal"}) == TURN_NOT_ADMITTED_EXIT_CODE
    err = capsys.readouterr().err
    assert "profile_deferred_capacity" in err and "profile_not_served" not in err, err

    # The record a machine reader parses (``--format stream-json``) closes with the same verdict.
    assert gateway_chat.launch_from_kwargs(
        {"q": "literal", "output_format": "stream-json"}) == TURN_NOT_ADMITTED_EXIT_CODE
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    result = records[-1]
    assert result["type"] == "result" and result["exit_code"] == TURN_NOT_ADMITTED_EXIT_CODE, result
    assert "profile_deferred_capacity" in result["error"], result
    assert "profile_not_served" not in result["error"], result
