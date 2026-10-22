"""A declared restart window is not a boot (PRA-6692).

On 2026-10-21 six heartbeat runs were dispatched inside a 187 s deliberate restart
(``restart-gateway.sh``) and every one burned its whole 120 s admission budget: the window is
longer than the budget, so no admission-side wait can survive it, and the verdict they got
(``starting: deadline``) named a boot that was never happening. These tests pin the three
contracts the fix owes:

* the lease is read, and only a live one counts (an unrefreshed lease is a dead owner);
* an open window is a RETRYABLE verdict with ``retryAfterS``, returned in seconds, and nothing is
  spawned while it is open;
* a boot observed from another process that dies without answering is a start this invocation
  owes the client, not a latch that spends the budget watching a lock nobody holds.
"""
import asyncio
import os
from pathlib import Path
import time

import pytest

from hermes_cli.gateway_restart_window import (
    RESTART_WINDOW_REASON,
    WINDOW_MARKER_RELPATH,
    declared_restart_window,
)


def _write_lease(home: Path, until: float, token: str = "") -> Path:
    marker = home / WINDOW_MARKER_RELPATH
    marker.parent.mkdir(parents=True, exist_ok=True)
    body = f"{int(until)}\n" + (f"token={token}\n" if token else "")
    marker.write_text(body, encoding="utf-8")
    return marker


def test_only_a_live_lease_counts(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    assert declared_restart_window([home]) is None

    marker = _write_lease(home, time.time() - 5)
    # An unrefreshed lease is a dead owner: it must never fence a launch.
    assert declared_restart_window([home]) is None

    _write_lease(home, time.time() + 90)
    live = declared_restart_window([home])
    assert live is not None and 80 <= live.remaining_s <= 90
    assert str(marker) == live.marker

    marker.write_text("not-an-epoch\n", encoding="utf-8")
    assert declared_restart_window([home]) is None


def test_gateway_start_is_refused_inside_the_window_unless_it_is_the_owner(tmp_path, monkeypatch):
    from hermes_cli.gateway_restart_window import refuse_outside_declared_window

    home = Path(os.environ["HERMES_HOME"])
    _write_lease(home, time.time() + 120, token="tok-9f2c")
    monkeypatch.delenv("HERMES_RESTART_WINDOW_TOKEN", raising=False)
    with pytest.raises(SystemExit) as refused:
        refuse_outside_declared_window("start", [home])
    assert refused.value.code == 1
    # The owning job proves itself with the lease's token: its own start is the only one allowed.
    monkeypatch.setenv("HERMES_RESTART_WINDOW_TOKEN", "tok-9f2c")
    refuse_outside_declared_window("start", [home])
    # ... and an aged-out lease never blocks recovery.
    _write_lease(home, time.time() - 5, token="tok-9f2c")
    monkeypatch.delenv("HERMES_RESTART_WINDOW_TOKEN", raising=False)
    refuse_outside_declared_window("start", [home])


def test_ensure_answers_the_window_in_seconds_and_starts_nothing(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime as runtime
    from hermes_cli import gateway_runtime_start

    home = tmp_path / "profile"
    home.mkdir()
    _write_lease(home, time.time() + 120)
    starts = []
    monkeypatch.setattr(gateway_runtime_start, "spawn_unmanaged_gateway",
                        lambda *a, **k: starts.append(a) or pytest.fail("spawned inside a window"))

    before = time.monotonic()
    result = runtime.ensure_gateway_runtime(home, timeout=30.0)
    elapsed = time.monotonic() - before

    assert result.reason_code == RESTART_WINDOW_REASON
    assert result.retry_after_s and result.retry_after_s > 0
    assert "retryAfterS=" in (result.detail or "")
    # Seconds, not the budget: the window is read before any observation or start.
    assert elapsed < 10
    assert starts == []


def test_connect_gateway_raises_a_retryable_restarting_verdict_fast(tmp_path, monkeypatch):
    from hermes_cli import gateway_client

    home = Path(os.environ["HERMES_HOME"])
    _write_lease(home, time.time() + 120)

    async def attach():
        async with gateway_client.connect_gateway(new_work=True):
            pytest.fail("a declared window is not an owner to attach to")

    started = time.monotonic()
    with pytest.raises(gateway_client.GatewayRestartingError) as caught:
        asyncio.run(attach())
    elapsed = time.monotonic() - started

    error = caught.value
    assert error.state == "restarting"
    assert error.retryable is True
    assert error.retry_after_s and 0 < error.retry_after_s <= 120
    assert "retryAfterS=" in str(error) and "retryable=true" in str(error)
    # A run must not spend its 120 s budget here (acceptance 4).
    assert elapsed < 10


def test_the_window_outranks_a_parkable_owner_for_new_work(tmp_path, monkeypatch):
    """A restart's after-turn wait is attachable, but it admits no NEW work (PRX-5787/PRX-5791).

    Inside a declared window there is no replacement owner to wait for until the owning job starts
    one — longer than the whole admission budget — so a run with a turn to place must be answered
    ``restarting`` instead of parked. An interactive client (no new work) keeps the owner: answering
    a prompt parked on approval/clarify is its job (#139041).
    """
    from hermes_cli import gateway_runtime as runtime
    from gateway.drain_control import RESTART_WAIT_REASON

    home = tmp_path / "profile"
    home.mkdir()
    _write_lease(home, time.time() + 120)
    parked = runtime.GatewayDiscovery(
        "draining", reason_code=RESTART_WAIT_REASON,
        endpoint=runtime.GatewayEndpoint(
            profile_id=str(home), instance_id="i-1", authority_epoch=1, runtime_protocol=1,
            api_origin="", supervisor="none", capabilities=frozenset()))
    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: parked)

    started = time.monotonic()
    verdict = runtime.ensure_gateway_runtime(home, timeout=30.0, new_work=True)
    assert time.monotonic() - started < 10
    assert verdict.reason_code == RESTART_WINDOW_REASON
    assert verdict.retry_after_s and verdict.retry_after_s > 0

    # The interactive client still reaches the owner to answer its parked prompt.
    assert runtime.ensure_gateway_runtime(home, timeout=30.0) is parked


def test_a_boot_that_dies_unanswered_is_started_by_the_waiting_client(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime as runtime
    from hermes_cli import gateway_runtime_service, gateway_runtime_start

    home = tmp_path / "profile"
    home.mkdir()
    verdicts = iter([
        runtime.GatewayDiscovery("starting", reason_code="pending"),
        runtime.GatewayDiscovery("absent"),
    ])
    monkeypatch.setattr(runtime, "_observe_replacing_stale", lambda *a, **k: next(verdicts, None)
                        or runtime.GatewayDiscovery("ready"))
    monkeypatch.setattr(runtime, "discover_gateway_endpoint",
                        lambda *a, **k: runtime.GatewayDiscovery("absent"))
    monkeypatch.setattr(gateway_runtime_service, "discover_existing_gateway_service",
                        lambda *a, **k: None)
    spawned = []

    class _Child:
        pid = 4242

        def poll(self):
            return None

    def spawn(target, **kwargs):
        spawned.append(target)
        return _Child()

    monkeypatch.setattr(gateway_runtime_start, "spawn_unmanaged_gateway", spawn)

    result = runtime.ensure_gateway_runtime(home, timeout=20.0)
    assert result.state == "ready"
    assert spawned == [home], "the lock was free and nothing was booting — the client owns the start"
