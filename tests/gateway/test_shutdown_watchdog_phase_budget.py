"""The shutdown watchdog bounds a SEQUENCE of phases, so its delay must fit their sum (PRX-5828).

At 2026-10-21T15:55:19Z a signal-driven restart force-exited pid 86032 after 58 s with the event loop
parked in ``select()``. Notify 1.21 s + a 50 s drain cap + an 11 s teardown is 62.2 s against a 58 s
leash, so the turn that answered 47 s into the drain was amputated 8 s into teardown (7 ``interrupted``
heartbeat rows for that window). Every phase was individually bounded; only their sum was not. The
dump then claimed "asyncio drain path appears wedged" for a drain that had already finished.

Invariants pinned here: (1) notify + drain cap + post-drain cleanup fits the leash the watchdog is
armed with, and that leash still fires before launchd's SIGKILL; (2) a fire names the phase it
interrupted and says whether the drain had finished; (3) a genuinely wedged drain still force-exits
inside ``ExitTimeOut`` — the watchdog is resized, never removed.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import hermes_thread_dump
import pytest

from gateway.restart import (
    CRON_DRAIN_CLEANUP_RESERVE_S,
    LAUNCHD_GUI_EXIT_TIMEOUT_CLAMP_S,
    LAUNCHD_NOTIFY_BUDGET_S,
    LAUNCHD_WATCHDOG_DUMP_MARGIN_S,
    describe_launchd_drain_cap,
    effective_stop_drain_timeout,
    effective_stop_watchdog_delay,
    resolve_launchd_capped_drain,
)
from gateway.shutdown_watchdog import arm_shutdown_watchdog, resolve_shutdown_watchdog_delay
from gateway.shutdown_watchdog_report import (
    describe_shutdown_watchdog_firing,
    shutdown_watchdog_snapshot,
)

_EXIT_TIMEOUT = float(LAUNCHD_GUI_EXIT_TIMEOUT_CLAMP_S)
# Measured, not guessed: notify took 1.21 s at the 15:54 restart and 5.57 s at 14:07; teardown was
# `Gateway stopped (total teardown 11.00s)` at the 05:59 restart.
_MEASURED_NOTIFY_S = 5.57
_MEASURED_TEARDOWN_S = 11.0


def _runner(*, drain: float = 180.0, launchd: float | None = _EXIT_TIMEOUT, by_signal: bool = True):
    return SimpleNamespace(
        _restart_drain_timeout=drain, _launchd_exit_timeout_s=launchd, _stop_requested_by_signal=by_signal,
    )


def _leash(runner) -> float:
    """The leash this stop arms: effective drain + grace, clamped under launchd's SIGKILL."""
    return effective_stop_watchdog_delay(
        runner, resolve_shutdown_watchdog_delay(effective_stop_drain_timeout(runner))
    )


def test_the_drain_cap_leaves_room_for_the_phases_around_it():
    """notify + drain + teardown must fit the leash, and the leash must beat launchd's SIGKILL."""
    runner = _runner()
    cap, leash = effective_stop_drain_timeout(runner), _leash(runner)
    assert LAUNCHD_NOTIFY_BUDGET_S + cap + CRON_DRAIN_CLEANUP_RESERVE_S <= leash
    assert leash <= _EXIT_TIMEOUT - LAUNCHD_WATCHDOG_DUMP_MARGIN_S


def test_a_turn_answering_in_the_last_seconds_of_the_drain_still_gets_its_teardown():
    """PRX-5828's exact shape: work finishing late in the window, then an 11 s teardown."""
    runner = _runner()
    cap, leash = effective_stop_drain_timeout(runner), _leash(runner)
    for answered_at in (cap - 10.0, cap - 3.0, cap - 0.5, cap):
        assert LAUNCHD_NOTIFY_BUDGET_S + answered_at + _MEASURED_TEARDOWN_S <= leash


def test_the_1555_timeline_overran_the_old_cap_and_fits_the_derived_one():
    old_cap = _EXIT_TIMEOUT - 10.0  # the bare reserve the defect collapsed every phase into
    new_cap = resolve_launchd_capped_drain(180.0, _EXIT_TIMEOUT)
    leash = _EXIT_TIMEOUT - LAUNCHD_WATCHDOG_DUMP_MARGIN_S
    assert _MEASURED_NOTIFY_S + old_cap + _MEASURED_TEARDOWN_S > leash  # the forced exit
    assert new_cap < old_cap
    assert _MEASURED_NOTIFY_S + new_cap + _MEASURED_TEARDOWN_S < leash  # the fix


def test_a_wedged_drain_still_force_exits_inside_launchd_exit_timeout():
    """Resizing must not remove the backstop: a drain that never answers is still killed in budget."""
    runner = _runner()
    cap, leash = effective_stop_drain_timeout(runner), _leash(runner)
    assert 0.0 < cap < leash < _EXIT_TIMEOUT


def test_the_cap_audit_line_names_every_phase_budget():
    line = describe_launchd_drain_cap(38.0, 60.0, _EXIT_TIMEOUT, 58.0)
    assert "capped to 38s" in line
    for term in ("notify 6s", f"teardown {CRON_DRAIN_CLEANUP_RESERVE_S:.0f}s", "58s watchdog leash"):
        assert term in line, f"{term!r} missing from the cap audit line: {line}"


def _snapshot(**overrides) -> dict:
    """The 15:55:19Z dump's shape: drain capped at 50 s, finished at +50.2 s, teardown 7.8 s old."""
    snapshot = {
        "phase": "teardown",
        "phase_age_s": 7.8,
        "phase_elapsed_s": 58.02,
        "drain_cap_s": 50.0,
        "drain_elapsed_s": 47.2,
        "drain_finished_at_s": 50.22,
        "drain_timed_out": False,
    }
    snapshot.update(overrides)
    return snapshot


def test_a_fire_after_the_drain_names_the_phase_and_withdraws_the_wedge_claim():
    verdict = describe_shutdown_watchdog_firing(_snapshot())
    assert "'teardown' 7.8s in" in verdict
    assert "drain finished at +50.2s" in verdict
    assert "was not wedged" in verdict
    assert "appears wedged" not in verdict


def test_a_fire_inside_the_drain_past_its_cap_is_reported_as_the_drain_stalling():
    verdict = describe_shutdown_watchdog_firing(
        _snapshot(phase="drain", phase_age_s=44.5, phase_elapsed_s=50.1, drain_cap_s=38.0,
                  drain_finished_at_s=None),
    )
    assert "the drain overran its own 38s cap" in verdict
    assert "stalled inside the drain" in verdict


def test_a_fire_inside_the_drain_before_its_cap_is_not_called_a_lease_overrun():
    verdict = describe_shutdown_watchdog_firing(
        _snapshot(phase="drain", phase_age_s=20.0, phase_elapsed_s=26.0, drain_cap_s=38.0,
                  drain_finished_at_s=None),
    )
    assert "before its 38s cap" in verdict
    assert "not a lease overrun" in verdict


def test_a_fire_before_the_drain_started_says_the_drain_was_not_the_phase():
    verdict = describe_shutdown_watchdog_firing(
        _snapshot(phase="notify_active_sessions", phase_age_s=6.4, phase_elapsed_s=6.4, drain_cap_s=0.0,
                  drain_finished_at_s=None),
    )
    assert "before the drain started" in verdict


def test_an_unrecorded_phase_is_reported_as_unattributed_not_as_a_wedge():
    verdict = describe_shutdown_watchdog_firing({"snapshot_error": "RuntimeError('bare double')"})
    assert "no phase was recorded" in verdict
    assert "appears wedged" not in verdict


def test_snapshot_carries_the_phase_drain_cap_and_drain_end():
    from gateway.run_shutdown import GatewayShutdownMixin

    ctx = GatewayShutdownMixin._StopContext(deferred_count=lambda: 0)
    ctx.started_at = time.monotonic()
    ctx.begin_phase("teardown")
    ctx.drain_cap_s = 38.0
    ctx.drain_finished_at_s = 43.6
    runner = SimpleNamespace(
        _restart_requested=False, _draining=True, _running=False,
        _running_agent_count=lambda: 5, _active_cron_job_count=lambda: 0,
        _active_api_run_count=lambda: 0, _restart_drain_timeout=60.0,
        _launchd_exit_timeout_s=_EXIT_TIMEOUT, _stop_requested_by_signal=True,
    )
    snapshot = shutdown_watchdog_snapshot(runner, ctx, watchdog_delay_s=58.0)
    assert snapshot["phase"] == "teardown"
    assert snapshot["drain_cap_s"] == 38.0
    assert snapshot["drain_finished_at_s"] == 43.6
    assert snapshot["watchdog_delay_s"] == 58.0
    assert snapshot["active_agents"] == 5


@pytest.fixture()
def _fresh_thread_dump_window():
    """The all-thread dump is rate-limited per process; each test is its own incident."""
    hermes_thread_dump.reset_dump_rate_limit()
    yield
    hermes_thread_dump.reset_dump_rate_limit()


def test_the_watchdog_dump_and_log_record_the_phase_verdict(tmp_path, caplog):
    """The file and the log both carry the phase: a "wedged" claim in either misleads the reader."""
    done, fired = threading.Event(), threading.Event()
    dump = tmp_path / "logs" / "watchdog.log"
    with caplog.at_level(logging.CRITICAL, logger="gateway.shutdown_watchdog"), \
            patch("gateway.shutdown_watchdog.os._exit", side_effect=lambda _code: fired.set()):
        arm_shutdown_watchdog(
            0.15, done_event=done, dump_path=dump, exit_code=9,
            snapshot_fn=_snapshot, verdict_fn=describe_shutdown_watchdog_firing,
        )
        assert fired.wait(timeout=5.0), "watchdog did not fire"

    header = json.loads(dump.read_text(encoding="utf-8").splitlines()[0])
    assert header["snapshot"]["phase"] == "teardown"
    assert "was not wedged" in header["verdict"]
    fired_lines = [r.getMessage() for r in caplog.records if "Shutdown watchdog fired" in r.getMessage()]
    assert fired_lines and "was not wedged" in fired_lines[0]
    assert "appears wedged" not in caplog.text
