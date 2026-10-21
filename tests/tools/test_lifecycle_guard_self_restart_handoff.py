"""The ONE restart an agent inside the supervised gateway may run is the drain-first handoff.

`hermes gateway request-restart` only sends SIGUSR1 to the running gateway, which refuses new
turns, drains in-flight ones (the caller's turn ends normally), exits and is relaunched by its
supervisor. The pre-exec guard refused every lifecycle spelling unconditionally on a rationale that
does not hold on this path ("the gateway would kill this command before it could complete"), and
the remedy it named — a shell outside the gateway — is unreachable from an agent session, so a
Python module change could only be deployed by a human shell or the daily launchd job (PRX-5771).

This file pins BOTH directions: the exact argv passes, and the allowance opens nothing else.
"""

from __future__ import annotations

import shlex

import pytest

from tools import process_registry
from tools.terminal_tool_guards import gateway_lifecycle_block

HANDOFF = "hermes gateway request-restart"

# Every shape the guard still refuses from inside the gateway. The handoff noun is deliberately
# close to these (`request-restart` contains `restart`), so a careless whitelist shows up here.
STILL_REFUSED = [
    "hermes gateway restart",
    "hermes gateway stop",
    "hermes gateway uninstall",
    "launchctl kickstart -k gui/501/ai.hermes.gateway",
    "launchctl bootout gui/501/ai.hermes.gateway",
    "systemctl --user restart hermes-gateway.service",
    "sh -c 'hermes gateway restart'",
    "sh -c 'hermes gateway stop; launchctl bootout gui/501/ai.hermes.gateway'",
    f"{HANDOFF}; hermes gateway restart",
    f"{HANDOFF} && launchctl kickstart -k gui/501/ai.hermes.gateway",
    f"{HANDOFF}\nhermes gateway restart",
    f"hermes gateway restart && {HANDOFF}",
]


class _Env:
    """A local terminal backend: nothing in this file needs a remote script read."""

    def __init__(self, cwd: str):
        self.cwd = cwd
        self.env = {}
        self.executed: list[str] = []

    def execute(self, command, **kwargs):
        self.executed.append(command)
        return {"output": "", "returncode": 1}


@pytest.fixture(autouse=True)
def inside_gateway(monkeypatch):
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr("tools.terminal_tool.get_session_cwd", lambda key: None)


def _guard(tmp_path, command: str):
    return gateway_lifecycle_block(
        command=command, env=_Env(str(tmp_path)), env_type="local", cwd=str(tmp_path),
        workdir=None, session_key=f"prx-5771-{id(command)}",
    )


def test_the_handoff_verb_passes_the_guard(tmp_path):
    assert _guard(tmp_path, HANDOFF) is None


def test_extra_whitespace_is_the_same_argv(tmp_path):
    """A shell normalizes whitespace before exec, so this IS the handoff argv — not a new route."""
    assert _guard(tmp_path, "  hermes   gateway\trequest-restart  ") is None


def test_the_whitelist_covers_no_second_command(tmp_path):
    """An exact argv: a handoff followed by a real restart is judged by the scan, not the shortcut."""
    assert _guard(tmp_path, f"{HANDOFF} & hermes gateway restart") is not None


@pytest.mark.parametrize("command", STILL_REFUSED)
def test_every_other_lifecycle_shape_still_refuses(tmp_path, command):
    blocked = _guard(tmp_path, command)
    assert blocked is not None, f"{command!r} must stay refused from inside the gateway"
    assert "Blocked" in blocked


def test_a_script_that_spells_a_restart_still_refuses(tmp_path):
    """The allowance is on the executed argv, never on a script that could hide the real verb."""
    script = tmp_path / "deploy.sh"
    script.write_text("#!/bin/bash\nhermes gateway restart\n", encoding="utf-8")

    blocked = _guard(tmp_path, f"bash {shlex.quote(str(script))}")

    assert blocked is not None, "a wrapper script must not launder the blocked verb"


def test_the_refusal_names_the_in_session_route(tmp_path):
    """The old message's only remedy was unreachable from here, which is why the model looped."""
    blocked = _guard(tmp_path, "hermes gateway restart")

    assert "hermes gateway request-restart" in blocked
