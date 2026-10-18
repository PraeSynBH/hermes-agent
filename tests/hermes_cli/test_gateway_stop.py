"""Stopping the gateway on purpose must not read as an unclean death (PRA-6427).

A stop requested on purpose can exit the gateway without its own teardown running: launchd
SIGKILLs a job still draining at ``ExitTimeOut``, and the stop path's bounded force kills one
that outlasted its grace. ``mark_exited`` lives inside the gateway process, so the sentinel then
stays ``phase=running`` and the next boot reports a phantom ``gateway.previous_unclean_exit`` —
for a stop someone asked for, which also buried the real death signal in PRA-6426's forensics.
"""
from __future__ import annotations

import json
import subprocess

import pytest

import hermes_cli.gateway as gateway_cli
from gateway.lifecycle_ledger import get_lifecycle_sentinel_path
from hermes_cli import gateway_stop


def _sentinel(fields: dict) -> None:
    path = get_lifecycle_sentinel_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields))


def _sentinel_after() -> dict:
    return json.loads(get_lifecycle_sentinel_path().read_text())


# ---------------------------------------------------------------------------
# the stop path closes the sentinel it emptied
# ---------------------------------------------------------------------------


def test_forced_stop_records_a_planned_exit_so_the_next_boot_is_clean(monkeypatch):
    _sentinel({"phase": "running", "pid": 42, "start_time": 1000.0})
    monkeypatch.setenv(gateway_stop.PLANNED_STOP_REASON_ENV, "storage_window_stop")

    kills = []
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: None if kills else 42)
    monkeypatch.setattr(
        gateway_cli, "terminate_pid",
        lambda pid, force=False, **kw: kills.append((pid, force)),
    )
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda pid: None)
    monkeypatch.setattr("time.sleep", lambda _: None)

    assert gateway_cli._wait_for_gateway_exit(timeout=10.0, force_after=0.0) is True

    assert kills == [(42, True)]  # it really was the forced path that emptied the PID
    sentinel = _sentinel_after()
    assert sentinel["phase"] == "exited"
    assert sentinel["exit_reason"] == "storage_window_stop"
    assert sentinel["pid"] == 42


def test_stop_that_never_finishes_records_nothing(monkeypatch):
    """The gateway is still up: writing a clean exit here would invert the false verdict."""
    _sentinel({"phase": "running", "pid": 42, "start_time": 1000.0})
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: 42)
    monkeypatch.setattr(gateway_cli, "terminate_pid", lambda pid, force=False, **kw: None)

    assert gateway_cli._wait_for_gateway_exit(timeout=0.0, force_after=None) is False
    assert _sentinel_after()["phase"] == "running"


def test_graceful_stop_keeps_the_exit_the_gateway_recorded(monkeypatch):
    """The process already said ``exited``; the stopper must not overwrite that record."""
    _sentinel({"phase": "exited", "pid": 42, "exit_code": 0, "exit_reason": "graceful_shutdown"})
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: None)

    assert gateway_cli._wait_for_gateway_exit(timeout=1.0, force_after=5.0) is True
    assert _sentinel_after()["exit_reason"] == "graceful_shutdown"


def test_replacement_life_is_never_clobbered(monkeypatch):
    """A stop racing a restart: the replacement claimed ``phase=running`` for its own PID first."""
    _sentinel({"phase": "running", "pid": 4242, "start_time": 1000.0})
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: None)

    assert gateway_cli._wait_for_gateway_exit(timeout=1.0, force_after=5.0) is True
    assert _sentinel_after()["pid"] == 4242


# ---------------------------------------------------------------------------
# launchd: wait out the gateway's own drain budget instead of pre-empting it
# ---------------------------------------------------------------------------


@pytest.fixture
def launchd_stop_env(monkeypatch):
    """``launchd_stop`` with a no-op launchctl and a recorder for the bounded wait."""
    waits: list[dict] = []
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(
        gateway_cli.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""),
    )
    monkeypatch.setattr(gateway_cli, "_mark_planned_stop", lambda *a, **k: None)
    monkeypatch.setattr(gateway_cli, "_wait_for_gateway_exit", lambda **kw: waits.append(kw) or True)
    return waits


def test_stop_waits_out_the_launchd_drain_budget_before_sigkill(launchd_stop_env, monkeypatch):
    """The gateway sizes its SIGTERM drain to launchd's live ``ExitTimeOut``; a stopper that
    SIGKILLs at the old 5 s pre-empts that budget, killing in-flight turns mid-teardown and
    skipping the gateway's own ``mark_exited``."""
    monkeypatch.setattr("gateway.restart.read_launchd_exit_timeout_s", lambda label=None: 60.0)

    gateway_cli.launchd_stop()

    assert launchd_stop_env and launchd_stop_env[0]["force_after"] == 60.0
    assert launchd_stop_env[0]["timeout"] > launchd_stop_env[0]["force_after"]


def test_stop_keeps_the_bounded_force_when_no_budget_is_readable(launchd_stop_env, monkeypatch):
    """Fail-open: not launchd-owned, or ``launchctl print`` unavailable → the historical 5 s."""
    monkeypatch.setattr("gateway.restart.read_launchd_exit_timeout_s", lambda label=None: None)

    gateway_cli.launchd_stop()

    assert launchd_stop_env and launchd_stop_env[0]["force_after"] == 5.0
