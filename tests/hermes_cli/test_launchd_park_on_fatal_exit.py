"""SOU-4222 — live macOS proof of the launchd park-on-exit-78 contract.

launchd has no ``RestartPreventExitStatus``. What replaces it is a pair: the installed plist's
``KeepAlive.SuccessfulExit=false`` (a clean exit is a deliberate stop) and the
``hermes_cli.stderr_timestamp`` wrapper's 78 -> 0 mapping (a fatal gateway verdict becomes that
clean exit). Each half passes its own unit test while the pair can still loop, because the
wrapper's decision is only real once launchd acts on the exit status. So this file bootstraps
throwaway LaunchAgents in exactly the shape ``generate_launchd_plist`` emits, whose child is a
fake ``hermes gateway run`` that exits 78 (park) or 75 (retry), and reads launchd's own ``runs``
/ ``last exit code`` back.

Needs a launchd session (a logged-in Mac, the macOS CI lane). Where ``launchctl bootstrap`` is
refused the test skips with launchctl's own message rather than pretending.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest

from hermes_cli.gateway_launchd import launchd_program_arguments
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("macos")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ThrottleInterval=1 keeps a REGRESSION cheap to observe (launchd's floor is 10s, but the plist
# value is honoured for a fresh job) and SETTLE_S is comfortably longer than one respawn.
_THROTTLE_S = 1
_SETTLE_S = 6.0


def _uid() -> int:
    return os.getuid()


def _fake_hermes(root: Path, exit_code: int) -> Path:
    """A stand-in ``hermes`` whose ``gateway run`` exits *exit_code* immediately."""
    exe = root / "bin" / "hermes"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text(f"#!/bin/sh\nexit {exit_code}\n", encoding="utf-8")
    exe.chmod(0o755)
    return exe


def _job_plist(root: Path, exit_code: int) -> tuple[Path, str]:
    label = f"com.nousresearch.hermes-park-{exit_code}-{uuid.uuid4().hex[:8]}"
    home, logs = root / "home", root / "home" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    wrapper = [
        sys.executable,
        "-m",
        "hermes_cli.stderr_timestamp",
        "--error-log",
        str(logs / "gateway.error.log"),
        "--",
        str(_fake_hermes(root, exit_code)),
        "gateway",
        "run",
        "--external-supervisor",
    ]
    spec = {
        "Label": label,
        "ProgramArguments": launchd_program_arguments(
            wrapper, logs / "gateway.log", logs / "gateway.error.log"
        ),
        "WorkingDirectory": str(home),
        "EnvironmentVariables": {
            "HERMES_HOME": str(home),
            "PATH": "/usr/bin:/bin",
            # The wrapper resolves the canonical argv matcher by importing the tree it ships from.
            "PYTHONPATH": str(PROJECT_ROOT),
        },
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": _THROTTLE_S,
        "StandardOutPath": str(logs / "launchd.out.log"),
        "StandardErrorPath": str(logs / "launchd.err.log"),
    }
    plist_path = root / f"{label}.plist"
    plist_path.write_bytes(plistlib.dumps(spec))
    return plist_path, label


def _domain() -> str:
    return f"gui/{_uid()}"


def _bootstrap(plist_path: Path, label: str) -> None:
    done = subprocess.run(
        ["launchctl", "bootstrap", _domain(), str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if done.returncode != 0:
        pytest.skip(f"launchctl cannot load a LaunchAgent on this host: {done.stderr.strip()}")


def _bootout(label: str) -> None:
    subprocess.run(
        ["launchctl", "bootout", f"{_domain()}/{label}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _job_state(label: str) -> dict:
    """launchd's own record for *label*: ``runs``, ``last exit code``, ``state``."""
    done = subprocess.run(
        ["launchctl", "print", f"{_domain()}/{label}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    info: dict = {"loaded": done.returncode == 0}
    for line in done.stdout.splitlines():
        key, _, value = line.strip().partition(" = ")
        value = value.strip()
        if key == "runs":
            info["runs"] = int(value)
        elif key == "last exit code":
            info["last_exit"] = value
        elif key == "state":
            info.setdefault("state", value)
    return info


def _run_job(tmp_path: Path, exit_code: int) -> dict:
    plist_path, label = _job_plist(tmp_path, exit_code)
    _bootstrap(plist_path, label)
    try:
        state = wait_until(
            lambda: (_state := _job_state(label)).get("last_exit") is not None and _state,
            timeout=60,
            what=f"the {label} job to exit once",
        )
        # A regression (78 relayed to launchd) respawns; give launchd room for at least one
        # respawn before deciding the job parked. The wait IS the behaviour under test.
        threading.Event().wait(_SETTLE_S)  # health: allow HX013 -- "stays parked" is a statement about elapsed time
        return _job_state(label) or state
    finally:
        _bootout(label)


def test_live_job_parks_after_a_fatal_verdict(tmp_path):
    """Exit 78 reaches launchd as a clean exit: one run, then nothing."""
    final = _run_job(tmp_path, 78)

    assert final.get("last_exit", "").startswith("0"), (
        f"the wrapper relayed a fatal verdict to launchd instead of parking it: {final}"
    )
    assert final.get("runs") == 1, f"launchd relaunched a parked job: {final}"
    assert "not running" in final.get("state", ""), f"job is still scheduled: {final}"


def test_live_job_relaunches_on_a_transient_verdict(tmp_path):
    """Exit 75 must still come back — parking must not become 'never restart'."""
    final = _run_job(tmp_path, 75)

    assert final.get("runs", 0) >= 2, f"launchd did not relaunch a retryable verdict: {final}"
    assert final.get("last_exit", "").startswith("75"), final
