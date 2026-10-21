"""`hermes gateway request-restart`: the drain-first handoff an in-session caller may ask for.

The gateway's own restart verbs refuse from inside (`_refuse_from_inside_gateway`: their fallbacks
SIGTERM/kickstart-kill the caller), and the terminal pre-exec guard refused the one restart path
that was built for this situation, so an agent session in a supervised gateway could never deploy a
Python module change (PRX-5768 blocked on it). The new verb does one thing — SIGUSR1 to the running
gateway, which drains and exits back to its supervisor — and refuses whenever that handoff cannot
be taken: no gateway, no supervisor, or a caller that is not the gateway's descendant (PRX-5771).
"""

from __future__ import annotations

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


def test_the_verb_is_wired_with_no_flags():
    from hermes_cli._parser import build_top_level_parser
    from hermes_cli.subcommands.gateway import build_gateway_parser

    parser, subparsers, _chat = build_top_level_parser()
    build_gateway_parser(subparsers, cmd_gateway=lambda a: None, cmd_proxy=lambda a: None,
                         cmd_gateway_enroll=lambda a: None)

    args = parser.parse_args(["gateway", "request-restart"])

    assert args.gateway_command == "request-restart"
    assert gw._GATEWAY_SUBCOMMANDS["request-restart"] is supervised.cmd_request_restart
