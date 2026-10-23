"""Session close reaps the process groups its turn spawned (PRA-6928).

PRA-6594 item 1: a session-scoped process-group reap on gateway session end. Per-call cleanup
(PRA-6601's owned daemon + group kill) runs when the *call* returns; it cannot run when the turn is
killed, the CLI is SIGKILLed, or the interpreter dies. The spawn ledger records the owning session
and the child's process group, and the session close path kills the whole group.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from gateway import session_policy

pytestmark = pytest.mark.platforms("posix")


def _spawn_sleeper() -> subprocess.Popen:
    """A real child in its own process group."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        start_new_session=True,
    )


def _wait_gone(proc: subprocess.Popen, timeout: float = 10.0) -> bool:
    """Wait until *proc* exits, polling in short increments."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    return proc.poll() is not None


@pytest.fixture
def isolated_ledger(tmp_path, monkeypatch):
    """A spawn ledger scoped to *tmp_path* so tests never touch the real ~/.hermes."""
    from hermes_cli import process_identity as pi

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pi, "install_id", lambda *a, **k: "install-test")
    return pi


def test_release_session_workspaces_kills_recorded_process_groups(isolated_ledger, monkeypatch):
    """A session's recorded groups are SIGKILLed when the session closes."""
    monkeypatch.setattr("tools.terminal_tool.get_session_cwd", lambda sid: None)

    child = _spawn_sleeper()
    try:
        isolated_ledger.register_child(child.pid, "test-child", session_id="s1")

        entries = isolated_ledger.ledger_entries_for_sessions(["s1"])
        assert len(entries) == 1
        assert entries[0]["pgid"] == child.pid  # start_new_session=True: pid == pgid

        session_policy.release_session_workspaces(None, ["s1"])

        assert _wait_gone(child), "child process group was not reaped"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=15)
