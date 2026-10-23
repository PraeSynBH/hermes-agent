"""Attribution for a managed restart: the requester must be named, and it must survive.

Regression for PRX-5423 §A. The 2026-10-22T19:55:39Z restart closed admission for 31 min 37 s and
interrupted a run, and the operator-facing logs said nothing about who asked: the sender's line had
gone into the requesting PROFILE's home (invisible to the gateway's own logs) and named a pid and
argv, never the run. A signal carries no payload, so the sender's line plus the gateway's handler
are the entire record — these tests pin both halves and the hand-off between them.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from gateway import lifecycle_ledger as ledger
from gateway import restart_attribution as attribution

RUN, AGENT, ISSUE = "run-1107f057", "agent-a1053376", "issue-0836adc4"


class _Runner:
    """Stands in for GatewayRunner: the handler's only demand is the restart request."""

    def __init__(self):
        self.calls: list[dict] = []

    def request_restart(self, **kwargs):
        self.calls.append(kwargs)
        return True


def _claim_sentinel() -> Path:
    path = ledger.get_lifecycle_sentinel_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"phase": "running", "pid": os.getpid(), "started_at": "x"}))
    return path


def _managed_line(caplog) -> str:
    lines = [record.getMessage() for record in caplog.records
             if record.getMessage().startswith("managed restart requested")]
    assert lines, "the restart must leave exactly one attribution line"
    return lines[-1]


def test_the_request_line_lands_in_the_gateway_home_with_the_run_that_asked(tmp_path, monkeypatch):
    """The line exists where the GATEWAY reads it, and names the run/agent/issue — not just a pid."""
    from gateway import host_rendezvous
    from hermes_cli import gateway as gw

    caller_home = tmp_path / "profiles" / "cto"
    gateway_home = tmp_path / "host-gateway"
    monkeypatch.setattr(attribution, "process_home", lambda: caller_home)
    monkeypatch.setattr(host_rendezvous, "read_record", lambda *a, **k: type(
        "Record", (), {"home": str(gateway_home), "pid": 4321})())
    monkeypatch.setenv("PAPERCLIP_RUN_ID", RUN)
    monkeypatch.setenv("PAPERCLIP_AGENT_ID", AGENT)
    monkeypatch.setenv("PAPERCLIP_TASK_ID", ISSUE)

    gw._record_restart_request("graceful_sigusr1", 4321, "drain_timeout=60s")

    line = attribution.request_log_path(gateway_home).read_text(encoding="utf-8")
    assert f"requestedByRunId={RUN}" in line
    assert f"agentId={AGENT}" in line and f"issueId={ISSUE}" in line
    assert "target_pid=4321" in line and "caller_pid=" in line
    assert not attribution.request_log_path(caller_home).exists(), (
        "a profile home's log is invisible to the host gateway — the record must not go there")


def test_the_line_is_written_before_the_signal(tmp_path, monkeypatch):
    """The handler reads the file the instant the signal lands; a later write loses the race."""
    from hermes_cli import gateway as gw

    monkeypatch.setattr(attribution, "process_home", lambda: tmp_path)
    seen: dict = {}

    def _kill(pid, sig):
        seen["log"] = attribution.request_log_path(tmp_path).read_text(encoding="utf-8")

    monkeypatch.setattr(gw.os, "kill", _kill)
    monkeypatch.setattr(gw, "_is_pid_ancestor_of_current_process", lambda pid: True)

    assert gw._request_gateway_self_restart(4321) is True
    assert "path=self_restart_sigusr1" in seen["log"]


def test_the_handler_names_the_requesting_run_and_stamps_the_lifecycle(caplog):
    sentinel = _claim_sentinel()
    attribution.record_request(
        "self_restart_sigusr1", os.getpid(), "ancestor=self", home=attribution.process_home(),
        identity={"requestedByRunId": RUN, "agentId": AGENT, "issueId": ISSUE})
    runner = _Runner()

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        attribution.sigusr1_restart_handler(runner)()

    line = _managed_line(caplog)
    assert f"requestedByRunId={RUN}" in line
    assert f"agentId={AGENT}" in line and f"issueId={ISSUE}" in line
    assert "source=sigusr1" in line and "drainRequired=true" in line
    assert runner.calls == [{"detached": False, "via_service": True}], "attribution never replaces the restart"
    stored = json.loads(sentinel.read_text())
    assert stored["restart_request"]["requestedByRunId"] == RUN


def test_the_next_life_can_name_the_restart_that_produced_it():
    """`jq .prior_restart_request state/gateway.lifecycle.json` answers after the fact."""
    sentinel = _claim_sentinel()
    ledger.stamp_restart_request(
        {"requestedByRunId": RUN, "agentId": AGENT, "issueId": ISSUE, "path": "self_restart_sigusr1"},
        source="sigusr1")

    ledger.mark_exited(75, "gateway_restart")
    assert json.loads(sentinel.read_text())["restart_request"]["requestedByRunId"] == RUN

    ledger.record_startup(defer_integrity_check=True)
    assert json.loads(sentinel.read_text())["prior_restart_request"]["requestedByRunId"] == RUN


def test_an_unrecorded_signal_is_reported_as_such(caplog):
    """No line ⇒ provably outside the instrumented senders, not merely unexplained."""
    _claim_sentinel()
    runner = _Runner()

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        attribution.sigusr1_restart_handler(runner)()

    line = _managed_line(caplog)
    assert "requestedByRunId=unrecorded" in line
    assert "instrumented CLI paths" in line
    stored = json.loads(ledger.get_lifecycle_sentinel_path().read_text())
    assert stored["restart_request"]["attributed"] is False


def test_a_stale_line_is_not_this_signals_requester(tmp_path):
    """A request from long ago cannot claim a signal it did not send."""
    log = attribution.request_log_path(tmp_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3600))
    log.write_text(f"{stale} path=graceful_sigusr1 target_pid={os.getpid()} caller_pid=1", encoding="utf-8")

    assert attribution.latest_request_for(os.getpid(), home=tmp_path) is None


def test_a_host_record_for_another_pid_does_not_capture_the_line(tmp_path, monkeypatch):
    """The rendezvous record is host-wide; only a record naming OUR target may choose the home."""
    from gateway import host_rendezvous
    from hermes_cli import gateway as gw

    caller_home = tmp_path / "profiles" / "cto"
    monkeypatch.setattr(attribution, "process_home", lambda: caller_home)
    monkeypatch.setattr(host_rendezvous, "read_record", lambda *a, **k: type(
        "Record", (), {"home": str(tmp_path / "someone-elses-gateway"), "pid": 9999})())

    gw._record_restart_request("graceful_sigusr1", 4321)

    assert attribution.request_log_path(caller_home).exists()
    assert not (tmp_path / "someone-elses-gateway").exists()


def test_another_processes_request_does_not_attribute_ours(tmp_path):
    log = attribution.request_log_path(tmp_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    attribution.record_request("graceful_sigusr1", os.getpid() + 1, home=tmp_path,
                               identity={"requestedByRunId": "someone-else"})

    assert attribution.latest_request_for(os.getpid(), home=tmp_path) is None
