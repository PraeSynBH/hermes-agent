"""The pre-exec guard's identity probe is memoized, and its budget is bounded (PRA-6500).

One failure produced both invariants. On a 43-profile multiplex gateway the guard's
process-identity probe — a lock read, a pidfile stat and a kernel process query — is
syscall-bound and was re-run on every terminal call. Under the gateway's ~300-thread GIL
convoy it stretched past the caller's whole deadline: 1169
``process-identity probe wedged`` refusals, each burning up to 180 s of a turn and
returning no output at all.

So: a guard that answers a *process-lifetime* question must not re-ask the kernel per
call, and a guard that has genuinely wedged must cost a retry rather than inherit the
command's 180 s deadline.

The latch is deliberately **positive-only**. Caching a negative verdict would permanently
un-guard a gateway that asked before its pid file existed — and would keep a macOS gateway
from ever getting its launchd shim — so a miss is always re-probed.
"""

from __future__ import annotations

import json
import os
import threading
from types import SimpleNamespace

import pytest

import tools.terminal_tool as terminal_module
from tools import process_registry, process_registry_pidfile as pidfile


@pytest.fixture
def gateway_env(monkeypatch):
    """This process looks like the live supervised gateway, with a counting pid probe."""
    calls: list[bool] = []

    def _counting_get_running_pid(*, cleanup_stale: bool = True, **_kw):
        calls.append(cleanup_stale)
        return os.getpid()

    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setattr("gateway.restart.is_supervised_gateway_launch", lambda *_a, **_k: True)
    monkeypatch.setattr("gateway.status.get_running_pid", _counting_get_running_pid)
    monkeypatch.setattr(pidfile, "_owns_pidfile", False)
    return calls


def test_a_proven_gateway_identity_is_latched(gateway_env):
    """N terminal calls must not become N kernel process queries."""
    verdicts = [process_registry._is_supervised_gateway_process() for _ in range(25)]

    assert verdicts == [True] * 25, "the latch must not change the verdict"
    assert len(gateway_env) == 1, f"the pid file was probed {len(gateway_env)} times, expected 1"
    assert gateway_env == [False], "the probe must keep cleanup_stale=False (never unlink)"


def test_a_negative_verdict_is_never_cached(gateway_env, monkeypatch):
    """A gateway that asks before its pid file exists must be recognised on the NEXT call.

    Caching the miss would permanently un-guard the very process these guards protect, and
    would keep a macOS gateway from ever getting its launchd shim.
    """
    monkeypatch.setattr("gateway.status.get_running_pid", lambda **_k: os.getpid() + 1)

    assert process_registry._is_supervised_gateway_process() is False
    assert process_registry._is_supervised_gateway_process() is False

    # The pid file lands: the next call must see it rather than a cached miss.
    monkeypatch.setattr("gateway.status.get_running_pid", lambda **_k: os.getpid())
    assert process_registry._is_supervised_gateway_process() is True


def test_a_latched_probe_is_still_gated_by_the_launch_marker(gateway_env, monkeypatch):
    """Only the pid-file half is latched: losing the launch marker must still disarm the guards.

    Latching the whole verdict would leave the self-kill guards armed for a process that is no
    longer the supervised gateway (a plain foreground `hermes gateway run` has no launch marker).
    """
    assert process_registry._is_supervised_gateway_process() is True
    assert len(gateway_env) == 1

    monkeypatch.setattr("gateway.restart.is_supervised_gateway_launch", lambda *_a, **_k: False)

    assert process_registry._is_supervised_gateway_process() is False, (
        "a latched pid-file verdict must not outlive the launch marker that gated it"
    )


def test_identity_probe_short_circuits_without_the_gateway_marker(monkeypatch):
    """A non-gateway process never probes at all."""
    calls: list[bool] = []
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda **_k: calls.append(True))
    monkeypatch.setattr(pidfile, "_owns_pidfile", False)

    assert process_registry._is_supervised_gateway_process() is False
    assert calls == [], "no marker means no probe"


# --- the guard's budget -------------------------------------------------------------


def _plan(timeout: float) -> SimpleNamespace:
    return SimpleNamespace(
        config={},
        env_type="local",
        effective_task_id="pra-6500-budget",
        cwd="/tmp",
        effective_timeout=timeout,
        promoted_from_foreground_timeout=None,
    )


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    """Stub planning/env/approval/execution; returns the executions that happened."""
    ran: list[str] = []
    monkeypatch.setattr(terminal_module, "_plan_execution", lambda *_a, **_k: _plan(180))
    monkeypatch.setattr(terminal_module, "_acquire_env", lambda *_a, **_k: object())
    monkeypatch.setattr(
        terminal_module, "_run_approval_guards", lambda *_a, **_k: terminal_module._ApprovalVerdict(),
    )
    monkeypatch.setattr(
        terminal_module, "_run_foreground", lambda *_a, **_k: ran.append("foreground") or "ran",
    )
    return ran


@pytest.fixture
def hanging_guard(monkeypatch):
    """A guard that never renders a verdict; released at teardown so no worker is left blocked."""
    release = threading.Event()
    monkeypatch.setattr(terminal_module, "_pre_exec_block", lambda *_a, **_k: release.wait())
    yield release
    release.set()


def test_a_long_command_deadline_does_not_become_the_guards_budget(monkeypatch, stubbed_pipeline):
    """A 180 s command must hand the guard its own ceiling, not the command's deadline."""
    budgets: list[float] = []
    monkeypatch.setattr(
        "agent.deadline.run_bounded_sync",
        lambda _fn, timeout, label=None: budgets.append(timeout) or SimpleNamespace(timed_out=False),
    )

    terminal_module.terminal_tool("echo ok")

    assert budgets == [terminal_module._PRE_EXEC_GUARD_MAX_TIMEOUT_S], (
        f"the guard was handed {budgets}; a wedged guard must be bounded by its own ceiling"
    )
    assert budgets[0] < 180, "the guard must not inherit the command's own default deadline"


def test_a_wedged_guard_fails_closed_and_reports_once_as_gateway_health(
    monkeypatch, caplog, stubbed_pipeline, hanging_guard,
):
    """The wedge is a process-wide health event: one line, not one per refused command."""
    monkeypatch.setattr(terminal_module, "_PRE_EXEC_GUARD_MIN_TIMEOUT_S", 0)
    monkeypatch.setattr(terminal_module, "_PRE_EXEC_GUARD_MAX_TIMEOUT_S", 0.1)
    monkeypatch.setattr(terminal_module, "_pre_exec_guard_wedge_reported", False)

    with caplog.at_level("WARNING", logger="tools.terminal_tool"):
        results = [json.loads(terminal_module.terminal_tool("echo ok")) for _ in range(2)]

    from agent.monitoring.gateway_health import classify_gateway_error

    assert [r["status"] for r in results] == ["error", "error"]
    assert stubbed_pipeline == [], "a guard with no verdict must not fail open into execution"
    wedges = [r for r in caplog.records if "pre-execution guard timed out" in r.getMessage()]
    assert len(wedges) == 1, f"two refusals logged {len(wedges)} wedge events; expected 1"
    assert classify_gateway_error(wedges[0].getMessage()) == "timeout", (
        "the wedge must classify as a gateway-health timeout, not an anonymous tool error"
    )
