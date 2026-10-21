"""A gateway drain must never mute cron quietly, and must never outlive what it silences (SOU-3252).

Measured 2026-10-21: a leaked ``.drain_request.json`` from an acceptance harness muted every ``*/10``
job for 68 min, and the only on-disk trace was a DEBUG line — so each liveness detector reported the
symptom (``last_run_at`` aging) and the healer's own catch-up pass was already fresh when the page
arrived. The gate now names the held jobs, stamps why, alerts when the drain drags, and the slot
still fires as a catch-up once the drain releases.
"""
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import cron.jobs as J
import cron.scheduler as S
from cron import scheduler_drain_gate as G


@pytest.fixture
def cron_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    (home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(J, "HERMES_DIR", home)
    monkeypatch.setattr(J, "CRON_DIR", home / "cron")
    monkeypatch.setattr(J, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(J, "OUTPUT_DIR", home / "cron" / "output")
    monkeypatch.setattr(S, "_hermes_home", home)
    monkeypatch.setattr(S, "_sweep_mcp_orphans", lambda: None)
    S._running_job_ids.clear()
    G.note_dispatch_resumed()  # episode dedup state is module-global
    return home


def _due_job(name: str = "pager", **extra):
    return J.create_job(prompt="summarise inbox", schedule="every 1h", name=name,
                        deliver="local", **extra)


def _make_due(job_id: str, *, minutes_ago: int = 1) -> str:
    stored = J.load_jobs()
    slot = (J._hermes_now() - timedelta(minutes=minutes_ago)).replace(microsecond=0).isoformat()
    next(r for r in stored if r["id"] == job_id)["next_run_at"] = slot
    J.save_jobs(stored)
    return slot


def _record_dispatches(monkeypatch) -> list[str]:
    """Stand in for the run itself; the tick's dispatch DECISION is what these tests are about."""
    seen: list[str] = []

    def fake_run_one_job(job, **_kwargs):
        seen.append(job["id"])
        return True

    monkeypatch.setattr(S, "run_one_job", fake_run_one_job)
    return seen


def _held(detail: str) -> bool:
    return str(detail).startswith(G.DRAIN_HOLD_REASON)


def test_drain_hold_names_the_job_stamps_the_reason_and_holds_the_slot(cron_home, monkeypatch):
    seen = _record_dispatches(monkeypatch)
    job = _due_job()
    due_before = _make_due(job["id"])

    ran = S.tick(verbose=False, can_dispatch=lambda: False)

    assert ran == 0 and seen == [], "a draining gateway dispatches nothing"
    stored = J.load_jobs()[0]
    # Held, not consumed: no run, the due instant untouched, and nothing for the next tick to
    # "restore" or "reclaim" — the hold scans read-only, so it must not write dispatch marks.
    assert stored["last_run_at"] is None and stored["next_run_at"] == due_before
    assert "pending_slot" not in stored and "last_dispatch" not in stored
    # The durable answer to "why is this late?" lives on the job, not only in the gateway log.
    assert _held(stored["last_fire_error"]["detail"])
    assert "held, not consumed" in stored["last_fire_error"]["detail"]


def test_drain_hold_warns_once_per_boundary_naming_every_held_job(cron_home, monkeypatch, caplog):
    # The ticker polls every 60 s; a 60-min drain must not write 60 identical lines. A NEW job
    # coming due is a new boundary and warns again, naming everything currently held.
    _record_dispatches(monkeypatch)
    first = _due_job("first")
    _make_due(first["id"])

    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        S.tick(verbose=False, can_dispatch=lambda: False)
        S.tick(verbose=False, can_dispatch=lambda: False)
    warnings = [r for r in caplog.records if G.DRAIN_HOLD_REASON in r.getMessage()]
    assert len(warnings) == 1
    assert first["id"] in warnings[0].getMessage()

    second = _due_job("second")
    _make_due(second["id"])
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        S.tick(verbose=False, can_dispatch=lambda: False)

    warnings = [r for r in caplog.records if G.DRAIN_HOLD_REASON in r.getMessage()]
    assert len(warnings) == 1, "a new boundary warns exactly once"
    assert first["id"] in warnings[0].getMessage()
    assert second["id"] in warnings[0].getMessage()


def test_the_stamped_reason_names_the_drain_when_the_marker_is_readable(cron_home, monkeypatch):
    # `hermes cron list` / the liveness checks read last_fire_error; "why" has to include the drain
    # that caused it — the SOU-3252 defect was a page that could only say "last_run_at is old".
    from gateway import drain_control as dc

    _record_dispatches(monkeypatch)
    marker = dc.write_drain_request(principal="dashboard", home=cron_home)
    assert dc.drain_requested(home=cron_home) is True, "a live trusted writer keeps the drain active"
    job = _due_job()
    _make_due(job["id"])

    assert S.tick(verbose=False, can_dispatch=lambda: not dc.drain_requested()) == 0

    detail = J.load_jobs()[0]["last_fire_error"]["detail"]
    assert _held(detail)
    assert "gateway.draining=true" in detail
    assert marker["requested_at"] in detail and "principal=dashboard" in detail


def test_a_drain_past_the_detector_tolerance_is_an_error_once_per_episode(
        cron_home, monkeypatch, caplog):
    # A drain open longer than the tightest served detector (the lead sweep: 40 min) holds a page's
    # worth of work, so the DRAIN is the incident. The marker's own max-age bounds a leaked one;
    # this watchdog catches one kept alive by re-writes. One ERROR per episode — not per tick, and
    # not per marker refresh (a keep-alive moves requested_at forever).
    from gateway import drain_control as dc

    _record_dispatches(monkeypatch)
    opened = dc.write_drain_request(principal="dashboard", home=cron_home)
    assert dc.drain_requested(home=cron_home) is True
    job = _due_job()
    _make_due(job["id"])

    clock = {"now": datetime.now(UTC)}
    monkeypatch.setattr(G, "_now", lambda: clock["now"])

    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        S.tick(verbose=False, can_dispatch=lambda: not dc.drain_requested())
    assert not [r for r in caplog.records if r.levelno == logging.ERROR]

    caplog.clear()
    clock["now"] += timedelta(seconds=G.DRAIN_ALERT_SECONDS + 300)
    with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
        S.tick(verbose=False, can_dispatch=lambda: not dc.drain_requested())
        S.tick(verbose=False, can_dispatch=lambda: not dc.drain_requested())

    errors = [r for r in caplog.records if r.levelno == logging.ERROR and "drain" in r.getMessage()]
    assert len(errors) == 1, "one ERROR per drain episode, not one per tick"
    assert "principal=dashboard" in errors[0].getMessage()
    assert opened["requested_at"] in J.load_jobs()[0]["last_fire_error"]["detail"]


def test_release_mid_window_fires_the_missed_slot_as_a_late_catch_up(cron_home, monkeypatch):
    # The other half of the contract: held is not dropped. Once dispatch is allowed again the slot
    # fires once, stamped with the lateness a liveness reader needs.
    job = _due_job()
    held_slot = _make_due(job["id"], minutes_ago=180)  # 3 h late: beyond the 30 min grace

    assert S.tick(verbose=False, can_dispatch=lambda: False) == 0
    seen = _record_dispatches(monkeypatch)
    assert S.tick(verbose=False, can_dispatch=lambda: True) == 1
    assert seen == [job["id"]], "the held slot fires exactly once on the first allowed tick"

    stored = J.load_jobs()[0]
    assert stored["next_run_at"] != held_slot, "the fired slot advances past the occurrence"
    stamp = stored["last_dispatch"]
    assert stamp["kind"] == "catch_up"
    assert stamp["lateness_seconds"] >= 180 * 60 - 60


def test_marker_written_by_a_writer_that_exited_never_mutes_the_tick(cron_home, monkeypatch):
    # The acceptance-harness leak that caused SOU-3252: the writer exits without cancelling. The
    # marker must die with it (gateway/drain_control.py, SOU-3249) so the very next tick dispatches
    # instead of waiting out the max age — no 68-minute pager gap.
    from gateway import drain_control as dc

    writer = subprocess.run(
        [sys.executable, "-c",
         "from gateway.drain_control import write_drain_request;"
         "write_drain_request(principal='dashboard')"],
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, "HERMES_HOME": str(cron_home)},
        capture_output=True, text=True, timeout=180, check=False)
    assert writer.returncode == 0, writer.stderr
    assert dc.drain_request_path(cron_home).exists(), "the writer left its marker behind"
    assert dc.drain_requested(home=cron_home) is False, "a dead writer's drain is already over"

    seen = _record_dispatches(monkeypatch)
    job = _due_job()
    _make_due(job["id"])

    assert S.tick(verbose=False, can_dispatch=lambda: not dc.drain_requested()) == 1
    assert seen == [job["id"]]
