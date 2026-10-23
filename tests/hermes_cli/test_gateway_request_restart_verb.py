"""`hermes gateway request-restart`: the drain-first handoff an in-session caller may ask for.

The gateway's own restart verbs refuse from inside (`_refuse_from_inside_gateway`: their fallbacks
SIGTERM/kickstart-kill the caller), and the terminal pre-exec guard refused the one restart path
that was built for this situation, so an agent session in a supervised gateway could never deploy a
Python module change (PRX-5768 blocked on it). The new verb does one thing — SIGUSR1 to the running
gateway, which drains and exits back to its supervisor — and refuses whenever that handoff cannot
be taken: no gateway, no supervisor, or a caller that is not the gateway's descendant (PRX-5771).
"""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import status as gateway_status
from hermes_cli import gateway as gw
from hermes_cli import gateway_supervised_restart as supervised

GATEWAY_PID = 4321


def _run(capsys):
    """Drive the verb; a refusal raises SystemExit, a success returns (exit status 0)."""
    code = 0
    try:
        supervised.cmd_request_restart(SimpleNamespace())
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr()


@pytest.fixture
def ancestor_signal(monkeypatch):
    """A supervised gateway the caller IS a descendant of; SIGUSR1 is recorded, never sent."""
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda *a, **k: GATEWAY_PID)
    monkeypatch.setattr(supervised, "gateway_declares_external_supervisor", lambda pid, home=None: True)
    monkeypatch.setattr(gw, "_is_pid_ancestor_of_current_process", lambda pid: True)
    monkeypatch.setattr(gw.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    return sent


def test_the_handoff_signals_the_gateway_and_returns_immediately(ancestor_signal, capsys):
    """Exit 0 + the acceptance line; the CLI never waits for the drain (it has already exited)."""
    code, captured = _run(capsys)

    assert code == 0
    assert "Service restart requested" in captured.out
    assert ancestor_signal == [(GATEWAY_PID, signal.SIGUSR1)], "the handoff is ONE SIGUSR1, never a SIGTERM"


def test_the_audit_line_carries_this_callers_pid(ancestor_signal, capsys):
    """The gateway cannot see who signalled it, so the CLI's own line is the only attribution."""
    from hermes_constants import get_hermes_home

    _run(capsys)

    lines = (Path(get_hermes_home()) / "logs" / "gateway-restart-requests.log").read_text(
        encoding="utf-8").splitlines()
    assert lines, "a restart request must leave an audit line"
    assert "path=self_restart_sigusr1" in lines[-1]
    assert f"target_pid={GATEWAY_PID}" in lines[-1]
    assert f"caller_pid={os.getpid()}" in lines[-1]


def test_an_unsupervised_gateway_is_refused(monkeypatch, capsys):
    """A drain-first exit with nothing to relaunch it leaves the gateway down — never do that."""
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda *a, **k: GATEWAY_PID)
    monkeypatch.setattr(supervised, "gateway_declares_external_supervisor", lambda pid, home=None: False)
    monkeypatch.setattr(gw.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    code, captured = _run(capsys)

    assert code == 1
    assert "declares no supervisor" in captured.out
    assert sent == [], "a refused handoff must not signal the gateway"


def test_no_running_gateway_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda *a, **k: None)

    code, captured = _run(capsys)

    assert code == 1
    assert "No running gateway" in captured.out


def _a_real_pid_that_is_not_an_ancestor() -> int:
    """Walk this process's real parent chain and return a pid outside it (the identity seam's own test)."""
    seen: set[int] = set()
    pid = os.getpid()
    while pid and pid not in seen:
        seen.add(pid)
        pid = gw._get_parent_pid(pid) or 0
    candidate = 2
    while candidate in seen:
        candidate += 1
    return candidate


def test_a_real_non_ancestor_pid_refuses_through_the_real_predicate(monkeypatch, capsys):
    """No stubs on the identity seam: this process is nobody's ancestor, so nothing is signalled."""
    sent: list[tuple[int, int]] = []
    stranger = _a_real_pid_that_is_not_an_ancestor()
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda *a, **k: stranger)
    monkeypatch.setattr(supervised, "gateway_declares_external_supervisor", lambda pid, home=None: True)
    monkeypatch.setattr(gw.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    code, captured = _run(capsys)

    assert code == 1
    assert "not a descendant" in captured.out
    assert sent == []


def _sentinel(home: Path, **body) -> None:
    """Write the lifecycle sentinel the request gate reads (the gateway's own durable record)."""
    from gateway.lifecycle_ledger import get_lifecycle_sentinel_path

    path = get_lifecycle_sentinel_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"phase": "running", "pid": GATEWAY_PID, "started_at": _iso(-3600), **body}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _iso(offset_s: float) -> str:
    from datetime import UTC, datetime, timedelta

    return (datetime.now(UTC) + timedelta(seconds=offset_s)).isoformat()


@pytest.fixture
def gateway_home(monkeypatch):
    """The gate reads the TARGET gateway's home, which is not the caller's under ``-p``."""
    from hermes_constants import get_hermes_home
    from gateway import restart_attribution

    home = Path(get_hermes_home())
    monkeypatch.setattr(restart_attribution, "target_gateway_home", lambda pid: home)
    return home


def test_a_request_that_arrives_during_an_open_deferral_joins_it(ancestor_signal, gateway_home, capsys):
    """PRA-6925: the requester cannot see a restart is already draining, so it asks again.

    Measured: agent a1053376 asked twice for one issue (runs 226a6bd8 and a2efccb4), and both
    requests were honoured with the full 30-minute fleet-wide cron hold. A second SIGUSR1 commands
    the same drain, so the verb now answers from the sentinel and signals nothing.
    """
    _sentinel(gateway_home, restart_request={
        "requested_at": _iso(-240), "source": "sigusr1",
        "requestedByRunId": "226a6bd8-3589-421b-b400-900a8a7321f1",
        "agentId": "a1053376-5764-483e-8d70-fff68b841589",
        "issueId": "e1d697f2-619a-44ec-bd0e-770c3fe0d263",
    })

    code, captured = _run(capsys)

    assert code == 0, "the caller's intent is already in flight — never a retryable failure"
    assert "already in progress" in captured.out and "joins it" in captured.out
    assert "226a6bd8-3589-421b-b400-900a8a7321f1" in captured.out, "name who is already draining"
    assert "e1d697f2-619a-44ec-bd0e-770c3fe0d263" in captured.out
    assert ancestor_signal == [], "joining must not queue a second drain"


def test_a_request_right_after_a_completed_restart_is_answered_from_the_record(
        ancestor_signal, gateway_home, capsys):
    """The gateway this process is talking to was BORN from a managed restart minutes ago."""
    _sentinel(gateway_home, started_at=_iso(-180), prior_restart_request={
        "requested_at": _iso(-240), "source": "sigusr1",
        "requestedByRunId": "48833a85-ce04-4078-8306-889adde2191d",
        "issueId": "e10d8627-b591-403f-af95-3176fbd5c8a2",
    })

    code, captured = _run(capsys)

    assert code == 0
    assert "already restarted" in captured.out
    assert "48833a85-ce04-4078-8306-889adde2191d" in captured.out
    assert ancestor_signal == []


def test_an_old_restart_record_never_blocks_a_new_request(ancestor_signal, gateway_home, capsys):
    """A floor that outlives the record's meaning would block honest deploys — and the cap bounds it."""
    _sentinel(gateway_home, started_at=_iso(-86400), prior_restart_request={
        "requested_at": _iso(-86500), "source": "sigusr1",
    })

    code, captured = _run(capsys)

    assert code == 0
    assert "Service restart requested" in captured.out
    assert ancestor_signal == [(GATEWAY_PID, signal.SIGUSR1)]


def test_a_sentinel_for_another_pid_never_suppresses_this_gateways_restart(
        ancestor_signal, gateway_home, capsys):
    """One host, many gateways: a record about a different process must not answer for ours."""
    _sentinel(gateway_home, restart_request={"requested_at": _iso(-60), "source": "sigusr1"})
    _sentinel(gateway_home, pid=GATEWAY_PID + 1,
              restart_request={"requested_at": _iso(-60), "source": "sigusr1"})

    code, captured = _run(capsys)

    assert code == 0
    assert "Service restart requested" in captured.out
    assert ancestor_signal == [(GATEWAY_PID, signal.SIGUSR1)]


def test_the_verb_is_wired_with_no_flags():
    from hermes_cli._parser import build_top_level_parser
    from hermes_cli.subcommands.gateway import build_gateway_parser

    parser, subparsers, _chat = build_top_level_parser()
    build_gateway_parser(subparsers, cmd_gateway=lambda a: None, cmd_proxy=lambda a: None,
                         cmd_gateway_enroll=lambda a: None)

    args = parser.parse_args(["gateway", "request-restart"])

    assert args.gateway_command == "request-restart"
    assert gw._GATEWAY_SUBCOMMANDS["request-restart"] is supervised.cmd_request_restart
