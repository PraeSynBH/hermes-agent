"""The pre-exec guard asks a backend to read a referenced script only when it IS another machine.

A local backend's "remote" read is a shell command against the very filesystem the walk just read,
so a routine script that shells out to the toolchain (every path it names is a candidate the walk
finds nothing at) turned one guard call into ~50 shell round trips, inside the bounded worker whose
budget is the whole reason the command gets refused. Backend bytes are also never latchable, so the
round trips disabled the clean-verdict memo for any command that named a missing path: the 45 s
ceiling was never the guard's real budget once the box was busy (PRX-5758).
"""

from __future__ import annotations

import shlex

import pytest

from tools import process_registry
from tools.terminal_tool_guards import gateway_lifecycle_block

RESTART = "hermes gateway restart"


class _Env:
    """A terminal backend that records the shell reads the guard sends it."""

    def __init__(self, cwd: str, output: str = ""):
        self.cwd = cwd
        self.env = {}
        self.output = output
        self.executed: list[str] = []

    def execute(self, command, **kwargs):
        self.executed.append(command)
        if self.output:
            return {"output": self.output, "returncode": 0}
        return {"output": "", "returncode": 1}


@pytest.fixture(autouse=True)
def inside_gateway(monkeypatch):
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)


def _guard(monkeypatch, tmp_path, *, env_type: str, command: str, output: str = ""):
    env = _Env(str(tmp_path), output=output)
    monkeypatch.setattr("tools.terminal_tool.get_session_cwd", lambda key: None)
    return env, gateway_lifecycle_block(
        command=command, env=env, env_type=env_type, cwd=str(tmp_path), workdir=None,
        session_key=f"prx-5758-{env_type}-{id(env)}",
    )


def test_a_local_backend_is_never_asked_to_read_a_path(monkeypatch, tmp_path):
    """Nothing runs on the host unless TERMINAL_ENV=local: the backend sees the same file."""
    missing = tmp_path / "not-created-yet.sh"

    env, blocked = _guard(monkeypatch, tmp_path, env_type="local", command=f"bash {shlex.quote(str(missing))} --dry-run")

    assert blocked is None, "a script that does not exist is nothing to scan, not a refusal"
    assert env.executed == [], f"the walk shelled out {len(env.executed)} times for paths a local env reads itself"


def test_a_local_binary_the_script_names_costs_no_round_trip(monkeypatch, tmp_path):
    """A compiled binary is not a script; asking the local backend re-reads those same bytes."""
    binary = tmp_path / "tool"
    binary.write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 64)  # Mach-O 64-bit
    script = tmp_path / "routine.sh"
    script.write_text(f"#!/bin/bash\n{shlex.quote(str(binary))} --version\n", encoding="utf-8")

    env, blocked = _guard(monkeypatch, tmp_path, env_type="local", command=f"bash {shlex.quote(str(script))}")

    assert blocked is None
    assert env.executed == [], f"the walk shelled out {len(env.executed)} times for a binary it had already read"


def test_a_remote_backend_still_answers_for_a_path_the_host_does_not_have(monkeypatch, tmp_path):
    """The skip is a property of the LOCAL backend, not of the walk: a sandbox is elsewhere."""
    missing = tmp_path / "only-in-the-sandbox.sh"

    env, blocked = _guard(
        monkeypatch, tmp_path, env_type="ssh",
        command=f"bash {shlex.quote(str(missing))}",
        output=f"#!/usr/bin/env bash\n{RESTART}\n",
    )

    assert env.executed, "a remote backend is the only place a path missing on the host can be found"
    assert blocked is not None and "Blocked" in blocked, (
        "a lifecycle command in the sandbox's copy of the script must still refuse the call"
    )
