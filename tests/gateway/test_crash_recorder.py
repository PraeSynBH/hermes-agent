"""Fatal-signal crash recorder — a death by SIGBUS/SIGSEGV/SIGABRT must leave a record.

PRA-6669: pid 12392 died of SIGBUS on 2026-10-21 with nothing in
``gateway-exit-diag.log`` — ``atexit`` and the ``except SystemExit`` arm in ``run_gateway``
are unreachable on a fatal signal — so the next boot's ``gateway.previous_unclean_exit``
carried no cause and the soak windows gated on it silently restarted.

These are behaviour contracts, not shape checks: a real child process takes a real fatal
signal, and the next boot's diag rows are read back from the exit-diag log.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from gateway import crash_recorder
from gateway.lifecycle_ledger import record_startup


_CRASHING_CHILD = """
import signal, sys
sys.path.insert(0, sys.argv[2])
from gateway.crash_recorder import arm
assert arm(sys.argv[1]), "recorder refused to arm"

def _die():
    signal.raise_signal(getattr(signal, sys.argv[3]))

print("armed", flush=True)
_die()
"""

#: pid_max on Linux; a pid this large is never alive, so the sentinel reads as a dead life.
_DEAD_PID = 2 ** 22 + 12345

_ARM_FAILS_CHILD = """
import sys
sys.path.insert(0, sys.argv[2])
import faulthandler
from gateway.crash_recorder import arm, is_armed
armed = arm(sys.argv[1])
print(f"armed={armed} is_armed={is_armed()} faulthandler={faulthandler.is_enabled()}", flush=True)
"""


def _crash_once(tmp_path: Path, signal_name: str = "SIGBUS") -> tuple[int, Path]:
    """Run a child that arms the recorder and then kills itself with ``signal_name``."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = tmp_path / "logs"
    child = subprocess.Popen(
        [sys.executable, "-c", _CRASHING_CHILD, str(log_dir), repo_root, signal_name],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "armed"
        child.wait(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    return child.returncode, log_dir / crash_recorder.CRASH_LOG_NAME


def _exit_diag_records(home: Path) -> list[dict]:
    path = home / "logs" / "gateway-exit-diag.log"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.platforms("posix")
def test_fatal_signal_leaves_a_parseable_record(tmp_path: Path) -> None:
    """A death by fatal signal writes a header + the faulting stack, and the next boot can
    name the signal without probing the dead process."""
    returncode, crash_log = _crash_once(tmp_path, "SIGBUS")
    assert returncode == -signal.SIGBUS, f"child did not die of SIGBUS (rc={returncode})"

    lines = crash_log.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["tag"] == crash_recorder.ARM_TAG
    assert header["python"] and header["pid"]
    assert "SIGBUS" in header["signals"]

    banner = [i for i, line in enumerate(lines) if line.startswith("Fatal Python error:")]
    assert banner, f"no fatal banner in the crash record:\n{crash_log.read_text()[:2000]}"
    assert lines[banner[0]] == "Fatal Python error: Bus error"
    assert any(line.startswith('  File "') for line in lines[banner[0]:]), "no traceback followed the banner"

    crash = crash_recorder.prior_crash(path=crash_log)
    assert crash is not None
    assert crash["signal"] == "SIGBUS"
    assert crash["signal_number"] == signal.SIGBUS
    assert crash["reason"] == "Bus error"
    assert crash["pid"] == header["pid"]
    assert crash["traceback"] is True


@pytest.mark.platforms("posix")
def test_abort_is_reported_as_SIGABRT(tmp_path: Path) -> None:
    """The reason -> signal table covers more than one entry: an abort is not mistaken for
    a vehicle for SIGBUS."""
    returncode, crash_log = _crash_once(tmp_path, "SIGABRT")
    assert returncode == -signal.SIGABRT

    crash = crash_recorder.prior_crash(path=crash_log)
    assert crash is not None
    assert crash["signal"] == "SIGABRT"
    assert crash["signal_number"] == signal.SIGABRT


@pytest.mark.platforms("posix")
def test_next_boot_row_carries_the_signal_and_resets_the_soak_window(tmp_path: Path) -> None:
    """``gateway.previous_unclean_exit`` says which signal, and one distinct
    ``gateway.soak_window_reset`` line carries reason=crash_signal for the gates."""
    _, crash_log = _crash_once(tmp_path, "SIGBUS")
    prior = crash_recorder.prior_crash(path=crash_log)
    assert prior is not None and prior["pid"]
    crashed_pid = prior["pid"]

    sentinel = tmp_path / "state" / "gateway.lifecycle.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({
        "phase": "running", "pid": crashed_pid, "start_time": 1000.0,
        "started_at": "2026-10-21T09:21:37+00:00",
    }), encoding="utf-8")

    evidence = record_startup(home=tmp_path)
    assert evidence is not None
    assert evidence["prior_crash_signal"] == "SIGBUS"

    records = _exit_diag_records(tmp_path)
    unclean = [r for r in records if r["tag"] == "gateway.previous_unclean_exit"]
    assert len(unclean) == 1
    assert unclean[0]["prior_crash_signal"] == "SIGBUS"
    assert unclean[0]["prior_crash_signal_number"] == signal.SIGBUS
    assert unclean[0]["prior_crash_ts"], "the crash row must date the death"
    assert unclean[0]["prior_pid"] == crashed_pid

    reset = [r for r in records if r["tag"] == "gateway.soak_window_reset"]
    assert len(reset) == 1
    assert reset[0]["reason"] == "crash_signal"
    assert reset[0]["prior_crash_signal"] == "SIGBUS"


def test_a_clean_life_is_not_reported_as_a_crash(tmp_path: Path) -> None:
    """Every boot appends a header; only a record with a fatal banner is a crash, so a
    clean life must not reset a soak window."""
    log_dir = tmp_path / "logs"
    assert crash_recorder.arm(log_dir) is True
    assert crash_recorder.arm(log_dir) is True  # idempotent: one descriptor per life

    lines = (log_dir / crash_recorder.CRASH_LOG_NAME).read_text(encoding="utf-8").splitlines()
    headers = [json.loads(line) for line in lines if crash_recorder.ARM_TAG in line]
    assert len(headers) == 1
    assert crash_recorder.prior_crash(path=log_dir / crash_recorder.CRASH_LOG_NAME) is None


@pytest.mark.platforms("posix")
def test_recorder_keeps_working_with_exit_diag_opted_out(tmp_path: Path, monkeypatch) -> None:
    """``HERMES_GATEWAY_EXIT_DIAG=0`` must not create a new failure mode: the record is still
    written and still parseable (it is not an exit-diag row)."""
    monkeypatch.setenv("HERMES_GATEWAY_EXIT_DIAG", "0")
    _, crash_log = _crash_once(tmp_path, "SIGBUS")

    crash = crash_recorder.prior_crash(path=crash_log)
    assert crash is not None and crash["signal"] == "SIGBUS"


def test_a_crash_from_an_older_life_is_not_attributed_to_this_one(tmp_path: Path) -> None:
    """The crash row names the signal only when the record is the sentinel owner's: a stale
    record from an earlier life must not be blamed for a SIGKILL death (or re-reset a soak
    window on every boot)."""
    from gateway.lifecycle_ledger import detect_unclean_exit

    _crash_once(tmp_path, "SIGBUS")  # leaves a record from the crashed child
    sentinel = tmp_path / "state" / "gateway.lifecycle.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({
        "phase": "running", "pid": _DEAD_PID, "start_time": 1000.0,
        "started_at": "2026-10-21T09:30:00+00:00",
    }), encoding="utf-8")

    evidence = detect_unclean_exit(home=tmp_path)
    assert evidence is not None
    assert evidence["prior_pid"] == _DEAD_PID
    assert "prior_crash_signal" not in evidence


def test_an_unwritable_crash_log_leaves_faulthandler_enabled(tmp_path: Path) -> None:
    """A crash log that cannot be opened must not take the diagnostic (or the gateway) with
    it: arming fails soft and faulthandler still runs on stderr, as before PRA-6669.

    Run in a child: ``arm`` is once-per-process, so the pytest worker's own recorder would
    mask a fresh life's first arm.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")

    out = subprocess.run(
        [sys.executable, "-c", _ARM_FAILS_CHILD, str(blocker), repo_root],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "armed=False is_armed=False faulthandler=True"
