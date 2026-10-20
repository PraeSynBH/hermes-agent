"""A browser-harness daemon Hermes STARTS is recorded in the spawn ledger — and one it merely
reuses is not.

``browser_harness.daemon`` detaches into its own session and reparents to ``ppid=1`` the moment the
CLI that started it exits, so no teardown hook reaches it when Hermes is SIGKILLed or its run
scratch is torn down: PRA-6594 found 15 of them still holding LISTEN sockets. The fix is ownership
data, not another heuristic: ``hermes_cli.process_identity.register_child`` writes
``(pid, create_time, spawner pid/create_time, purpose)``, and the startup reconciler reaps a helper
ONLY on a provably dead spawner.

The pid-file path mirror is asserted against the INSTALLED package (``browser_harness._ipc``) rather
than against a copy of its rules, so an upstream path change surfaces here instead of silently
turning every registration into a no-op.

Regression for PRA-6601 / PRA-6594.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import browser_harness_dirs as bhd

pytestmark = pytest.mark.platforms("posix")


def _package_pid_path(env_overrides: dict[str, str], name: str = "default") -> str:
    """``browser_harness._ipc.pid_path(name)`` as the harness itself computes it for this env."""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    env.update(env_overrides)
    code = f"from browser_harness import _ipc; print(_ipc.pid_path({name!r}))"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, encoding="utf-8", timeout=60, check=True)
    return str(Path(out.stdout.strip()).resolve())


@pytest.mark.parametrize("env_overrides,name", [
    ({"BH_RUNTIME_DIR": "/tmp/hbh-isolated"}, "default"),            # single-client dir: bare stem
    ({"BH_RUNTIME_DIR": "/tmp/hbh-shared", "BH_RUNTIME_DIR_SHARED": "1"}, "r7k2"),
    ({"BH_TMP_DIR": "/tmp/hbh-tmp-only"}, "default"),                # legacy dir, still its own
    ({"BH_HOME": "/tmp/hbh-home", "BH_RUNTIME_DIR_SHARED": "1"}, "sess-1"),
])
def test_pid_path_mirror_matches_the_installed_package(env_overrides, name):
    """Hermes reads the daemon's pid file from the parent: the path must be the package's own."""
    env = dict(env_overrides)
    if name != "default":
        env["BU_NAME"] = name
    mirrored = bhd.daemon_pid_path(env)
    assert mirrored is not None
    assert str(mirrored.resolve()) == _package_pid_path(env_overrides, name)


def test_pid_path_is_none_without_a_resolvable_runtime_dir():
    """No dir and no home to derive one from: record nothing (never guess a path)."""
    assert bhd.daemon_pid_path({"BU_NAME": "default"}) is None


def test_pid_path_rejects_an_invalid_daemon_name():
    """``BU_NAME`` reaches a filesystem path; the package's own charset is the fence."""
    assert bhd.daemon_pid_path({"BH_RUNTIME_DIR": "/tmp/x", "BU_NAME": "../evil"}) is None


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    from hermes_cli import process_identity as pi

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pi, "install_id", lambda *a, **k: "install-test")
    path = tmp_path / pi.LEDGER_FILENAME

    def entries() -> list[dict]:
        return json.loads(path.read_text()) if path.exists() else []

    return entries


def _sleeping_daemon(pid_file: Path) -> subprocess.Popen:
    """A real child standing in for the daemon, publishing its pid the way the harness does."""
    proc = subprocess.Popen(["/bin/sleep", "120"])
    pid_file.write_text(str(proc.pid), encoding="utf-8")
    return proc


def test_a_daemon_this_call_started_is_registered_under_its_own_purpose(ledger, tmp_path, monkeypatch):
    """The receipt the reconciler needs: pid + create_time + spawner, purpose `browser-harness-daemon`."""
    import time

    from tools.browser_use_cli import _register_started_daemon

    pid_file = tmp_path / "bu-daemon.pid"
    started = time.time()
    proc = _sleeping_daemon(pid_file)
    try:
        _register_started_daemon({"BU_NAME": "default"}, pid_file, None, started)
        entries = [e for e in ledger() if e["pid"] == proc.pid]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["purpose"] == "browser-harness-daemon"
        assert entry["spawner_pid"] == os.getpid()
        assert entry["create_time"] is not None
    finally:
        proc.kill()
        proc.wait(timeout=15)


def test_a_reused_daemon_is_never_recorded_as_this_process_child(ledger, tmp_path):
    """A daemon already running (``before_pid``) belongs to whoever started it — recording it here
    would let this process's death reap a healthy daemon another Hermes still drives."""
    import time

    from tools.browser_use_cli import _register_started_daemon

    pid_file = tmp_path / "bu-daemon.pid"
    proc = _sleeping_daemon(pid_file)
    try:
        _register_started_daemon({"BU_NAME": "default"}, pid_file, proc.pid, time.time())
        assert [e for e in ledger() if e["purpose"] == "browser-harness-daemon"] == []
    finally:
        proc.kill()
        proc.wait(timeout=15)


def test_a_daemon_that_predates_the_call_is_not_ours(ledger, tmp_path):
    """Two Hermes processes racing the spawn lock: the loser sees a daemon it did not start."""
    import time

    from tools.browser_use_cli import _register_started_daemon

    pid_file = tmp_path / "bu-daemon.pid"
    proc = _sleeping_daemon(pid_file)
    try:
        _register_started_daemon({"BU_NAME": "default"}, pid_file, None, time.time() + 60)
        assert [e for e in ledger() if e["purpose"] == "browser-harness-daemon"] == []
    finally:
        proc.kill()
        proc.wait(timeout=15)


def test_an_unpublished_pid_is_simply_not_recorded(ledger, tmp_path):
    """No pid file yet (a CLI that never started a daemon): nothing to own, nothing to write."""
    import time

    from tools.browser_use_cli import _register_started_daemon

    _register_started_daemon({"BU_NAME": "default"}, tmp_path / "missing.pid", None, time.time())
    assert ledger() == []
