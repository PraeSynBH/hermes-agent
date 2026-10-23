"""Unit tests for in-band restart after-turn deferral helpers (#77184)."""

import asyncio
import logging
import math
from pathlib import Path
from types import SimpleNamespace

from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT,
    parse_restart_after_turn_timeout,
    resolve_restart_exit_wait_budget,
    resolve_systemd_timeout_stop_sec,
)
from gateway.run import GatewayRunner
from gateway.run_shutdown import GatewayShutdownMixin


def test_parse_restart_after_turn_timeout_defaults_and_clamps():
    assert parse_restart_after_turn_timeout("") == DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT
    assert parse_restart_after_turn_timeout(None) == DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT
    assert parse_restart_after_turn_timeout("bogus") == DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT
    assert parse_restart_after_turn_timeout(0) == 0.0
    assert parse_restart_after_turn_timeout("-5") == 0.0
    assert parse_restart_after_turn_timeout("120") == 120.0


def test_restart_exit_wait_budget_outlasts_deferral_plus_stop_envelope():
    for chat, after_turn, cron in ((0, 0, 0), (0, 1800, 30), (2, 3, 80), (80, 3, 2), (180, 21600, 0)):
        # The observer must never give up before the supervisor's own stop deadline would.
        assert resolve_restart_exit_wait_budget(chat, after_turn, cron) > after_turn + resolve_systemd_timeout_stop_sec(chat, cron)
    # A non-finite drain waits indefinitely instead of crashing the integer envelope.
    assert resolve_restart_exit_wait_budget(float("inf"), 0, 0) == math.inf
    assert resolve_restart_exit_wait_budget(0, 0, float("inf")) == math.inf


def test_cli_restart_wait_covers_configured_cron_drain(tmp_path, monkeypatch):
    import hermes_cli.gateway as gateway_cli

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in ("HERMES_RESTART_DRAIN_TIMEOUT", "HERMES_RESTART_AFTER_TURN_TIMEOUT", "HERMES_CRON_DRAIN_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    config = tmp_path / "config.yaml"
    config.write_text("agent:\n  restart_drain_timeout: 2\n  restart_after_turn_timeout: 3\n  cron_drain_timeout: 80\n")
    with_cron = gateway_cli._get_restart_exit_wait_budget()
    config.write_text("agent:\n  restart_drain_timeout: 2\n  restart_after_turn_timeout: 3\n  cron_drain_timeout: 0\n")
    # The configured cron drain reaches the CLI wait, which outlasts the stop it can take.
    assert with_cron > 3 + resolve_systemd_timeout_stop_sec(2, 80)
    assert with_cron > gateway_cli._get_restart_exit_wait_budget()


def test_load_restart_after_turn_timeout_preserves_zero(tmp_path, monkeypatch):
    """Config/env ``0`` must disable after-turn wait, not fall back to default."""
    import gateway.run as gateway_run

    monkeypatch.delenv("HERMES_RESTART_AFTER_TURN_TIMEOUT", raising=False)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "agent:\n  restart_after_turn_timeout: 0\n",
        encoding="utf-8",
    )
    assert GatewayRunner._load_restart_after_turn_timeout() == 0.0

    monkeypatch.setenv("HERMES_RESTART_AFTER_TURN_TIMEOUT", "0")
    assert GatewayRunner._load_restart_after_turn_timeout() == 0.0


def test_the_deferral_names_the_fleet_wide_cron_cost_at_t_zero(monkeypatch, caplog):
    """PRA-6833: the wait closes admission process-wide, so it must say what it costs cron.

    Both fleet-wide cron claim holes were 33-36 min restart deferrals; the only lines they produced
    were the work-unit dump, which never mentioned that every ticked profile's due slots were held
    for the whole wait. The deferral therefore announces the cron cost when it OPENS (not 40 min
    later) and repeats the count in the 30 s status line.
    """
    from cron import scheduler_drain_gate, scheduler_ownership

    monkeypatch.setattr(scheduler_ownership, "ticked_homes", lambda: {"/h/a": Path("/h/a")})
    monkeypatch.setattr(scheduler_drain_gate, "due_slot_summary", lambda homes: (7, 3))

    runner = SimpleNamespace(
        _draining=True, _running=True, session_runtime_descriptor={},
        _restart_after_turn_timeout=0.3,
        _active_work_count=lambda: 1, _awaitable_work_count=lambda: 1,
        _wedged_agent_count=lambda: 0, _restart_safe_cron_count=lambda: 0,
        _describe_active_work=list, _scale_to_zero_status=lambda *a, **k: None,
    )
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        assert asyncio.run(GatewayShutdownMixin._await_active_work_before_restart(runner)) is False

    messages = [r.getMessage() for r in caplog.records]
    opening = [m for m in messages if "Restart deferral (cap" in m]
    assert len(opening) == 1, "the cron cost is named once, when the deferral opens"
    assert "7 due cron slot(s)" in opening[0] and "3 profile(s)" in opening[0]
    assert "held fleet-wide until the drain releases" in opening[0]

    status = [m for m in messages if "Restart deferred: waiting on 1 active work unit(s)" in m]
    assert status, "the 30 s status line still reports the wait"
    assert "cron dispatch is held fleet-wide" in status[0], "the count rides the status line too"


def test_an_unreadable_cron_count_still_names_the_hold_and_never_breaks_the_wait(monkeypatch, caplog):
    """The announcement is a diagnostic inside a shutdown path: a broken count must not abort it."""
    from cron import scheduler_drain_gate

    def broken(homes):
        raise OSError("store gone")

    monkeypatch.setattr(scheduler_drain_gate, "due_slot_summary", broken)

    runner = SimpleNamespace(
        _draining=True, _running=True, session_runtime_descriptor={},
        _restart_after_turn_timeout=0.2,
        _active_work_count=lambda: 1, _awaitable_work_count=lambda: 1,
        _wedged_agent_count=lambda: 0, _restart_safe_cron_count=lambda: 0,
        _describe_active_work=list, _scale_to_zero_status=lambda *a, **k: None,
    )
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        assert asyncio.run(GatewayShutdownMixin._await_active_work_before_restart(runner)) is False

    opening = [r.getMessage() for r in caplog.records if "Restart deferral (cap" in r.getMessage()]
    assert len(opening) == 1
    assert "cron dispatch is held fleet-wide until the drain releases" in opening[0]
