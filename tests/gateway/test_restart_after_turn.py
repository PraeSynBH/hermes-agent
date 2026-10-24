"""Unit tests for in-band restart after-turn deferral helpers (#77184)."""

import asyncio
import json
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


def _stuck_runner(cap: float):
    """A runner whose in-flight work never finishes, so only the bound under test ends the wait."""
    return SimpleNamespace(
        _draining=True, _running=True, session_runtime_descriptor={},
        _restart_after_turn_timeout=cap,
        _active_work_count=lambda: 1, _awaitable_work_count=lambda: 1,
        _wedged_agent_count=lambda: 0, _restart_safe_cron_count=lambda: 0,
        _describe_active_work=list, _scale_to_zero_status=lambda *a, **k: None,
    )


def _cron_cost(monkeypatch, due: int, homes: int, bound):
    """Pin the fleet-wide due-slot scan, the ticked homes and the cron bound (seconds or None)."""
    from cron import scheduler_drain_gate, scheduler_ownership

    monkeypatch.setattr(scheduler_ownership, "ticked_homes", lambda: {f"/h/{i}": Path(f"/h/{i}") for i in range(homes)})
    monkeypatch.setattr(scheduler_drain_gate, "due_slot_summary", lambda _homes: (due, homes))

    def _bound():
        if isinstance(bound, Exception):
            raise bound
        return bound

    monkeypatch.setattr(scheduler_drain_gate, "hold_bound_seconds", _bound)


def test_a_waiting_cron_slot_ends_the_deferral_at_the_cron_bound(monkeypatch, caplog):
    """PRA-6925: the hold's cost is the slots it leaves unissued, so a waiting slot bounds the wait.

    Measured on this host: three of four deferrals held 1-3 due slots across 1-3 profiles for the
    FULL 1800 s cap while the drain-done line recorded ``cron_at_start=0`` — 30 minutes of
    fleet-wide cron silence protecting no cron work at all. Once a slot is waiting, the deferral
    therefore ends within the cron inactivity watchdog, and the slot fires as a catch-up.
    """
    _cron_cost(monkeypatch, due=2, homes=2, bound=0.25)
    started = asyncio.run(_timed_wait(_stuck_runner(5.0), caplog))
    elapsed, result = started

    assert result is False, "the wait ends by forcing stop()/drain, exactly as the cap does"
    assert elapsed < 3.0, "a 0.25 s cron bound must not run for the 5 s cap"

    messages = [r.getMessage() for r in caplog.records]
    ended = [m for m in messages if "Restart deferral ends at the cron bound" in m]
    assert ended, "the early end names its reason, or a post-mortem reads it as a cap timeout"
    assert "HERMES_CRON_TIMEOUT" in ended[0]
    armed = [m for m in messages if "are waiting, so this wait now ends at most" in m]
    assert armed and "2 due cron slot(s) across 2 profile(s)" in armed[0]
    assert not [m for m in messages if "Restart after-turn wait timed out" in m], \
        "the cap timeout line would misreport why the wait ended"


def test_without_a_waiting_slot_the_full_cap_still_protects_in_flight_turns(monkeypatch, caplog):
    """Nothing due means the hold costs cron nothing yet: the wait keeps protecting turns."""
    _cron_cost(monkeypatch, due=0, homes=2, bound=0.25)
    elapsed, result = asyncio.run(_timed_wait(_stuck_runner(2.5), caplog))

    assert result is False
    assert elapsed >= 2.4, "the cron bound must only start once a slot is actually waiting"
    messages = [r.getMessage() for r in caplog.records]
    assert [m for m in messages if "Restart after-turn wait timed out after" in m]
    assert not [m for m in messages if "Restart deferral ends at the cron bound" in m]


def test_an_unreadable_cron_bound_leaves_the_deferral_exactly_as_it_was(monkeypatch, caplog):
    """A diagnostic that can only shorten a shutdown fails toward the old behaviour, never into it."""
    _cron_cost(monkeypatch, due=3, homes=3, bound=ImportError("no cron"))
    elapsed, result = asyncio.run(_timed_wait(_stuck_runner(2.5), caplog))

    assert result is False
    assert elapsed >= 2.4, "no bound derivable means the configured cap is the only bound"
    assert [m for m in caplog.records if "Restart after-turn wait timed out after" in m.getMessage()]


def test_the_cron_bound_comes_from_the_inactivity_watchdog(monkeypatch):
    """The yardstick is HERMES_CRON_TIMEOUT: the longest a healthy run may go quiet, so a held SLOT
    waiting longer than that has outlived the scheduler's own patience."""
    from cron.scheduler_drain_gate import hold_bound_seconds

    monkeypatch.delenv("HERMES_CRON_TIMEOUT", raising=False)
    assert hold_bound_seconds() == 600.0
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "120")
    assert hold_bound_seconds() == 120.0
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
    assert hold_bound_seconds() is None, "0 disables the watchdog, so it cannot bound anything"


async def _timed_wait(runner, caplog):
    """Drive the deferral with the clock on, returning (wall seconds, drained-to-zero verdict)."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        drained = await GatewayShutdownMixin._await_active_work_before_restart(runner)
    return loop.time() - started, drained


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


# --- PRX-6133 (CEO ruling rev 2 on PRX-6109, R7): the T=0 bound event and its consumer ---------
#
# The surface control had to subtract a restart hold from its measurement gap by parsing the prose
# of "Restart deferral (cap 1800s): cron dispatch is held fleet-wide … N due cron slot(s) across M
# profile(s)", and the only escalation in the box arrived at T+26 of a T+30 hold, to a file nothing
# reads. So the deferral now writes what it knows at T=0 as data, and closes it with the width the
# hold actually had.

def _hold_ledger(tmp_path, monkeypatch):
    """Point the ledger at tmp_path and keep the requester lookup off the real home."""
    import gateway.restart_hold_ledger as ledger

    path = tmp_path / "restart-holds.jsonl"
    monkeypatch.setattr(ledger, "hold_ledger_path", lambda home=None: path)
    monkeypatch.setattr(ledger, "_requester", lambda: None)
    return path


def _hold_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_the_deferral_writes_the_bound_event_at_t_zero(tmp_path, monkeypatch, caplog):
    """R7.1: one structured record when the hold opens, carrying the four fields."""
    _cron_cost(monkeypatch, due=7, homes=3, bound=None)
    path = _hold_ledger(tmp_path, monkeypatch)
    _elapsed, result = asyncio.run(_timed_wait(_stuck_runner(0.3), caplog))

    assert result is False, "this runner never drains, so the cap ends the wait"
    begin, end = _hold_records(path)
    assert begin["event"] == "dispatch_hold_begin"
    assert begin["started_at"].endswith("Z") and end["ended_at"].endswith("Z")
    assert begin["cap_s"] == 0.3, "the cap the wait is actually honouring, not the default"
    assert begin["due_slots"] == 7 and begin["profiles"] == 3, \
        "the fleet-wide cost, from the same scan the WARNING quotes"
    assert begin["cron_cost_readable"] is True
    assert begin["predicted_end_at"] > begin["started_at"], "the window the control predicts"
    assert end["event"] == "dispatch_hold_end" and end["hold_id"] == begin["hold_id"]
    assert end["reason"] == "cap"
    assert end["held_s"] >= 0.25, "the END record carries the real width, not the cap"
    assert end["due_slots"] == 7 and end["profiles"] == 3


def test_the_bound_event_names_the_cron_bound_as_how_the_hold_ended(tmp_path, monkeypatch, caplog):
    """A hold that ends early must not be recorded as one that spent the cap (PRA-6925)."""
    _cron_cost(monkeypatch, due=2, homes=2, bound=0.25)
    path = _hold_ledger(tmp_path, monkeypatch)
    elapsed, result = asyncio.run(_timed_wait(_stuck_runner(5.0), caplog))

    assert result is False and elapsed < 3.0
    _begin, end = _hold_records(path)
    assert end["reason"] == "cron_bound"
    assert end["held_s"] < 5.0, "the 5 s cap was not spent, and the record says so"


def test_the_bound_event_records_a_successful_drain(tmp_path, monkeypatch, caplog):
    """The common ending: the work finished, so the hold cost cron only its own width."""
    _cron_cost(monkeypatch, due=1, homes=1, bound=None)
    path = _hold_ledger(tmp_path, monkeypatch)
    work = {"awaitable": 1}
    statuses = {"n": 0}

    def _finishes_during_the_wait(*_a, **_k):
        # The 30 s status mark is written once before the loop and once per status tick; work that
        # drains between two ticks is the common ending, and it must be recorded as one.
        statuses["n"] += 1
        if statuses["n"] >= 2:
            work["awaitable"] = 0

    runner = SimpleNamespace(
        _draining=True, _running=True, session_runtime_descriptor={},
        _restart_after_turn_timeout=5.0,
        _active_work_count=lambda: work["awaitable"],
        _awaitable_work_count=lambda: work["awaitable"],
        _wedged_agent_count=lambda: 0, _restart_safe_cron_count=lambda: 0,
        _describe_active_work=list, _scale_to_zero_status=_finishes_during_the_wait,
    )
    assert asyncio.run(GatewayShutdownMixin._await_active_work_before_restart(runner)) is True

    _begin, end = _hold_records(path)
    assert end["reason"] == "drained"
    assert end["held_s"] < 5.0


def test_an_unreadable_cron_scan_is_reported_not_recorded_as_zero(tmp_path, monkeypatch, caplog):
    """"Nothing was due yet" and "the scan failed" are different facts to a consumer."""
    from cron import scheduler_drain_gate

    def broken(homes):
        raise OSError("store gone")

    monkeypatch.setattr(scheduler_drain_gate, "due_slot_summary", broken)
    path = _hold_ledger(tmp_path, monkeypatch)

    runner = SimpleNamespace(
        _draining=True, _running=True, session_runtime_descriptor={},
        _restart_after_turn_timeout=0.2,
        _active_work_count=lambda: 1, _awaitable_work_count=lambda: 1,
        _wedged_agent_count=lambda: 0, _restart_safe_cron_count=lambda: 0,
        _describe_active_work=list, _scale_to_zero_status=lambda *a, **k: None,
    )
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        assert asyncio.run(GatewayShutdownMixin._await_active_work_before_restart(runner)) is False

    begin, _end = _hold_records(path)
    assert begin["due_slots"] is None and begin["profiles"] is None
    assert begin["cron_cost_readable"] is False
    assert begin["cap_s"] == 0.2, "the record still carries the bound the wait honoured"


# --- PRX-6173: the cron bound must not release a drain that cannot finish the work --------------
#
# Measured on 2026-10-24T05:42:26Z: the deferral ended at the cron bound after 600 s with 14 units
# active, the drain it released into got the launchd-clamped 60 s budget (``active_at_start=13``) and
# interrupted 12 of them, and 6 runs admitted at the release instant waited 133.5 s for the
# replacement to listen and settled ``interrupted``. The released slot was not served any sooner by
# the bound firing: admission stays closed through stop(), exit and the next boot, so the slot fires
# when the replacement listens either way.

def _retiring_runner(cap: float, script):
    """A runner whose awaitable population follows *script* ``[(seconds_after_arm, count), ...]``."""
    started = []

    def _awaitable() -> int:
        if not started:
            started.append(asyncio.get_running_loop().time())
        elapsed = asyncio.get_running_loop().time() - started[0]
        count = script[0][1]
        for at, value in script:
            if elapsed >= at:
                count = value
        return count

    return SimpleNamespace(
        _draining=True, _running=True, session_runtime_descriptor={},
        _restart_after_turn_timeout=cap,
        _active_work_count=_awaitable, _awaitable_work_count=_awaitable,
        _wedged_agent_count=lambda: 0, _restart_safe_cron_count=lambda: 0,
        _describe_active_work=list, _scale_to_zero_status=lambda *a, **k: None,
    )


def test_a_bound_that_is_still_retiring_work_extends_instead_of_releasing_the_drain(
        tmp_path, monkeypatch, caplog):
    """The ending the loss needs: work still finishing at the bound is not handed to stop()/drain.

    Twelve of the thirteen units alive at the release were interrupted because a 60 s drain cannot
    retire turns that have been running for minutes. Extending the hold while the population is
    still falling lets those turns finish inside the wait the cap already allowed.
    """
    _cron_cost(monkeypatch, due=2, homes=2, bound=0.25)
    path = _hold_ledger(tmp_path, monkeypatch)

    runner = _retiring_runner(5.0, [(0, 3), (0.2, 2), (0.45, 1), (0.65, 0)])
    elapsed, result = asyncio.run(_timed_wait(runner, caplog))

    assert result is True, "the work drained inside the wait instead of being interrupted by stop()"
    assert elapsed >= 0.6, "the wait outlived the first cron bound it hit"
    assert elapsed < 3.0, "the cap still bounds it"
    messages = [r.getMessage() for r in caplog.records]
    assert [m for m in messages if "the cron bound elapsed with" in m and "re-armed" in m], \
        "the extension names itself, or a post-mortem reads a 3x-bound hold as a cap timeout"
    assert not [m for m in messages if "Restart deferral ends at the cron bound" in m], \
        "this hold never handed the remainder to the drain"

    _begin, end = _hold_records(path)
    assert end["reason"] == "drained"
    assert end["cron_bound_extensions"] >= 1


def test_a_bound_that_retires_nothing_still_ends_the_wait(tmp_path, monkeypatch, caplog):
    """PRA-6925 preserved: a hold protecting nothing is pure fleet-wide cron silence."""
    _cron_cost(monkeypatch, due=2, homes=2, bound=0.25)
    path = _hold_ledger(tmp_path, monkeypatch)
    elapsed, result = asyncio.run(_timed_wait(_stuck_runner(5.0), caplog))

    assert result is False and elapsed < 3.0
    assert [m for m in caplog.records if "Restart deferral ends at the cron bound" in m.getMessage()]
    assert not [m for m in caplog.records if "re-armed" in m.getMessage()]
    _begin, end = _hold_records(path)
    assert end["reason"] == "cron_bound"
    assert "cron_bound_extensions" not in end, "an unextended hold keeps its record shape"


def test_the_cap_still_ends_an_extended_wait(tmp_path, monkeypatch, caplog):
    """The extension spends time the cap already allowed; it never outlives the cap."""
    _cron_cost(monkeypatch, due=1, homes=1, bound=0.25)
    path = _hold_ledger(tmp_path, monkeypatch)
    # Always one fewer unit than the previous window: productive forever, so only the cap ends it.
    script = [(i * 0.2, 100 - i) for i in range(40)]
    elapsed, result = asyncio.run(_timed_wait(_retiring_runner(1.0, script), caplog))

    assert result is False
    assert 0.9 <= elapsed < 1.6, f"the 1 s cap ended the extended wait, not the 0.25 s bound ({elapsed:.2f}s)"
    assert [m for m in caplog.records if "Restart after-turn wait timed out after" in m.getMessage()]
    _begin, end = _hold_records(path)
    assert end["reason"] == "cap"
    assert end["cron_bound_extensions"] >= 1
