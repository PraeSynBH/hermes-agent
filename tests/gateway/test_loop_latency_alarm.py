"""The alarm on the gateway owner's loop latency (PRX-5598).

One process serves every profile on one asyncio loop, so a stalled loop is a fleet-wide outage of
session admission. On 2026-10-19 the loop's own loopback health probe ran 10-25 s under load 38 and
past 40 s at load 100; three agents sat reported as errored for ~90 minutes and nothing fired.
The contracts pinned here:

* the rule is p95 over a rolling window, above the threshold for three consecutive samples, and one
  warning per stall episode;
* a loopback probe that is not answered reads as a floor on the stall, never as a healthy sample;
* the warning names the pid, the thread count and the 1-minute load — the three facts that tell a
  starved loop apart from a loaded host.
"""
from __future__ import annotations

import shutil
import socket
import tempfile
import threading
from pathlib import Path

import pytest

# Short enough for AF_UNIX's ~104-byte path limit once "state/gateway.loop-tick.<pid>.sock" is
# appended; a pytest tmp_path is not, which is the same reason the owner has a TCP fallback.
_SOCKET_HOME = "/tmp"


def test_a_fresh_owner_alarms_on_the_third_consecutive_sample():
    """The 12:24Z shape: a run admitted into a stall. Three samples above the threshold, one warn."""
    from gateway.loop_latency import LoopLatencyAlarm

    alarm = LoopLatencyAlarm()
    assert [alarm.observe(21.8) for _ in range(3)] == [False, False, True]
    assert alarm.p95() == pytest.approx(21.8)


def test_one_slow_sample_in_a_full_window_is_not_a_stall():
    """A full window's nearest-rank p95 is its second-largest sample: a single spike forgives itself."""
    from gateway.loop_latency import LoopLatencyAlarm

    alarm = LoopLatencyAlarm()
    for _ in range(20):
        assert alarm.observe(0.004) is False
    assert alarm.observe(30.0) is False
    assert alarm.p95() == pytest.approx(0.004)


def test_one_warning_per_stall_episode_and_a_rearmed_alarm_after_it():
    from gateway.loop_latency import LoopLatencyAlarm

    alarm = LoopLatencyAlarm()
    fired = [alarm.observe(20.0) for _ in range(8)]
    # One episode, one line: strikes 4+ are the same stall, not new news.
    assert fired.count(True) == 1 and fired[2] is True
    # Recovery re-arms it, so the NEXT stall warns again. The full window's p95 absorbs the first
    # slow sample (rank 19 of 20), so the streak starts one sample later there.
    for _ in range(30):
        alarm.observe(0.01)
    assert [alarm.observe(20.0) for _ in range(4)] == [False, False, False, True]


def test_the_warning_names_pid_threads_and_load():
    from gateway.loop_latency import LoopLatencyAlarm, alarm_warning

    alarm = LoopLatencyAlarm()
    for _ in range(3):
        alarm.observe(21.8)
    line = alarm_warning(alarm, pid=63813, threads=412, load=100.25)
    assert "p95=21.8s" in line
    assert "pid=63813" in line and "threads=412" in line and "load1=100.25" in line
    assert "admission budget" in line
    assert "load1=n/a" in alarm_warning(alarm, pid=1, threads=1, load=None)


@pytest.mark.platforms("posix")
def test_a_probe_measures_the_witness_and_reads_silence_as_a_floor():
    """Answered: a real number. Silent: the timeout, as a floor. No node: None, not a zero."""
    from gateway.loop_latency import describe_probe, probe_loop

    home = Path(tempfile.mkdtemp(prefix="lat-", dir=_SOCKET_HOME))
    try:
        (home / "state").mkdir()
        answered = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        answered.bind(str(home / "state" / "gateway.loop-tick.111.sock"))
        answered.listen(1)

        def answer():
            conn, _address = answered.accept()
            with conn:
                conn.recv(1)
                conn.sendall(b"1")

        thread = threading.Thread(target=answer, daemon=True)
        thread.start()
        try:
            latency = probe_loop([home], pid=111, timeout=5.0)
        finally:
            thread.join(10)
            answered.close()

        assert latency is not None and 0.0 <= latency < 5.0
        assert str(describe_probe(latency, timeout=5.0)).startswith("owner_loop=")

        # A node that accepts (the kernel completes the connect from the accept backlog) but never
        # answers is a stalled loop: report the timeout as a floor, not a fast sample.
        silent = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        silent.bind(str(home / "state" / "gateway.loop-tick.222.sock"))
        silent.listen(1)
        try:
            stall = probe_loop([home], pid=222, timeout=0.2)
        finally:
            silent.close()
        assert stall == 0.2
        assert describe_probe(stall, timeout=0.2) == "owner_loop>0.2s"

        # A node nothing is listening on is not a measurement at all.
        assert probe_loop([home], pid=333, timeout=0.2) is None
    finally:
        shutil.rmtree(home, ignore_errors=True)
