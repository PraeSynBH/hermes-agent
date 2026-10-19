"""Loopback latency of the gateway owner's asyncio loop, and the alarm on it.

One process serves every profile on a single asyncio loop (``hermes gateway run
--external-supervisor``), so a stall in that loop is a fleet-wide outage of session admission that
no run can report: on 2026-10-19 a trivial ``GET /healthz`` on the owner's own loopback ran
10-25 s under 1-min load 38 and 40+ s under load 100, every run admitted in those windows died at
its admission budget, and nothing alarmed — /healthz degrading from 3 ms to 25 s was observable
only by hand, ~90 minutes after three agents had already been reported as errored.

The probe here is the loop-tick socket the liveness watchdog already serves
(``gateway.shutdown_watchdog.get_loop_tick_socket_path``). Its answer is written by a coroutine on
the loop, so the round trip times how long the loop took to dispatch — the quantity the external
``/healthz`` probe measures, without the HTTP stack, and from any process (the owner's own alarm
thread, or a client describing a stall in its failure line). :class:`LoopLatencyAlarm` turns those
samples into one warning: p95 above the threshold for three consecutive samples, naming the
owner's pid, its thread count and the 1-minute load average.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import socket
import threading
import time
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# A stuck loop normally answers nothing at all, so a sample that times out is a lower bound on the
# stall rather than a missing sample. 15 s sits above this host's documented stall tail (~25 s on
# the ``identify`` verb, ~10 s answered) only in the sense that a longer wait would hold the alarm
# thread — the answer is a floor either way.
PROBE_TIMEOUT_S = 15.0
# The alarm rule: p95 over the last WINDOW_SAMPLES loopback probes above LATENCY_THRESHOLD_S for
# CONSECUTIVE_SAMPLES samples in a row. 5 s is a tenth of the admission budget's headroom against a
# normal call (``session.create`` p50 21.8 s end to end, ~5.4 s p95 for admission itself), so it
# warns while runs are still getting through rather than after the first death.
LATENCY_THRESHOLD_S = 5.0
CONSECUTIVE_SAMPLES = 3
# 20 samples is ~200 s at the cadence below, and the nearest-rank p95 of a full window is its
# second-largest sample: ONE slow sample in the window forgives itself, two (a loop that stalled
# twice inside 200 s) hold the p95 up, and a sustained stall holds it up for as long as it lasts.
WINDOW_SAMPLES = 20
# Sampling cadence: 10 s means an episode is named within ~30-40 s, well inside the window
# the p95 is taken over.
ALARM_INTERVAL_S = 10.0


def load_average_1m() -> Optional[float]:
    """The host's 1-minute load average, or None where the platform has none (Windows)."""
    getloadavg = getattr(os, "getloadavg", None)
    if getloadavg is None:
        return None
    try:
        return float(getloadavg()[0])
    except (OSError, ValueError):
        return None


def _homes(homes: Optional[Iterable[Path]]) -> list[Path]:
    if homes is None:
        return [None]  # the process home, resolved by the watchdog's path helper
    return [Path(home) for home in homes]


def _witness_paths(home: Optional[Path], pid: Optional[int]) -> list[Path]:
    """Candidate loop-tick socket nodes for ``home``: one named pid, or every live-looking node."""
    from gateway.shutdown_watchdog import get_loop_tick_socket_path

    if pid is not None:
        return [get_loop_tick_socket_path(home, pid)]
    try:
        state = get_loop_tick_socket_path(home, 0).parent
        return sorted(state.glob("gateway.loop-tick.*.sock"))
    except OSError:
        return []


def _round_trip(target: str, timeout: float) -> Optional[float]:
    """Seconds until the loop-tick witness answered, ``timeout`` when it did not, None if unsocketable."""
    started = time.monotonic()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(timeout)
            peer.connect(target)
            # The kernel completes the connect from the accept backlog; only the coroutine that
            # writes this byte proves the loop dispatched, so the first recv is the measurement.
            peer.sendall(b"1")
            if peer.recv(1) != b"1":
                return None
            return time.monotonic() - started
    except TimeoutError:
        return float(timeout)
    except OSError:
        return None


def probe_loop(
    homes: Optional[Iterable[Path]] = None, *, pid: Optional[int] = None,
    timeout: float = PROBE_TIMEOUT_S,
) -> Optional[float]:
    """Seconds the owner's loop took to answer one loopback ping, or None when no witness answers.

    ``homes`` are the candidate homes to look for the witness in (default: this process's home);
    ``pid`` names the owner (default: every ``gateway.loop-tick.<pid>.sock`` node in ``homes``).
    A returned value of ``timeout`` means the loop did not answer inside it — a lower bound on the
    stall, not a measurement of it. None means there was no witness at all (no owner, a Windows
    owner whose witness is on TCP, a stale node), so "could not measure" never reads as "fast".
    """
    for home in _homes(homes):
        for target in _witness_paths(home, pid):
            latency = _round_trip(str(target), timeout)
            if latency is not None:
                return latency
    return None


def describe_probe(latency: Optional[float], *, timeout: float = PROBE_TIMEOUT_S) -> Optional[str]:
    """``owner_loop=0.4s`` / ``owner_loop>15s`` for a failure line, or None when there was no witness."""
    if latency is None:
        return None
    return f"owner_loop>{latency:g}s" if latency >= timeout else f"owner_loop={latency:.1f}s"


def describe_load(load: Optional[float]) -> Optional[str]:
    return None if load is None else f"load1={load:.2f}"


class LoopLatencyAlarm:
    """The alarm rule over a rolling window of loopback samples: pure, no clock, no I/O.

    ``observe`` returns True exactly once per stall episode — on the sample where the streak
    reaches ``consecutive`` — so a twenty-minute wedge warns once instead of every sample, and
    re-arms as soon as the p95 falls back under the threshold.
    """

    def __init__(self, *, threshold_s: float = LATENCY_THRESHOLD_S,
                 consecutive: int = CONSECUTIVE_SAMPLES, window: int = WINDOW_SAMPLES) -> None:
        self.threshold_s, self.consecutive, self.window = float(threshold_s), int(consecutive), int(window)
        self._samples: list[float] = []
        self._streak = 0

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def samples(self) -> int:
        return len(self._samples)

    def p95(self) -> Optional[float]:
        """Nearest-rank p95 of the window, None before the first sample."""
        if not self._samples:
            return None
        ordered = sorted(self._samples)
        rank = max(1, -(-len(ordered) * 95 // 100))  # ceil(n * 0.95), nearest rank
        return ordered[rank - 1]

    def observe(self, latency_s: float) -> bool:
        self._samples.append(float(latency_s))
        del self._samples[:-self.window]
        percentile = self.p95()
        self._streak = self._streak + 1 if percentile is not None and percentile > self.threshold_s else 0
        return self._streak == self.consecutive


def alarm_warning(alarm: LoopLatencyAlarm, *, pid: int, threads: int, load: Optional[float]) -> str:
    """The one line a fired alarm logs: the rule that fired, then who and how loaded the owner is.

    pid, thread count and the 1-minute load are what separate the three ways a loop starves —
    one GIL shared by 400 threads, filesystem scans inside the loop, and a host at load 100 — and
    they were the three facts recovered by hand on 2026-10-19.
    """
    return (f"Gateway event loop is stalling session admission: loopback health probe "
            f"p95={(alarm.p95() or 0.0):.1f}s over the last {alarm.samples} samples "
            f"(threshold {alarm.threshold_s:.1f}s, {alarm.streak} consecutive) "
            f"pid={pid} threads={threads} load1={'n/a' if load is None else f'{load:.2f}'} — "
            "runs admitted now die at their admission budget")


class _AlarmHandle:
    """The alarm thread's lifecycle handle (mirrors ``_LoopLivenessWatchdogHandle``)."""

    def __init__(self, stop_event: threading.Event, thread: threading.Thread) -> None:
        self._stop_event, self._thread = stop_event, thread
        self.stop, self.join, self.is_alive = stop_event.set, thread.join, thread.is_alive


def start_loop_latency_alarm(
    home: Optional[Path] = None, *, pid: Optional[int] = None,
    interval_s: float = ALARM_INTERVAL_S, threshold_s: float = LATENCY_THRESHOLD_S,
    consecutive: int = CONSECUTIVE_SAMPLES, probe_timeout: float = PROBE_TIMEOUT_S,
) -> Optional[_AlarmHandle]:
    """Start the out-of-loop alarm on this owner's loop latency. Never raises.

    The thread runs beside the loop (it waits on its own socket, so it is not starved by the GIL
    stall it is measuring beyond the granularity of a wakeup), and a no-answer sample counts as a
    ``probe_timeout`` second floor: by then the loop is far past the threshold anyway.
    """
    alarm = LoopLatencyAlarm(threshold_s=threshold_s, consecutive=consecutive)
    stop_event = threading.Event()

    def _watch() -> None:
        while True:
            latency = probe_loop([home] if home is not None else None, pid=pid, timeout=probe_timeout)
            if latency is not None and alarm.observe(latency):
                logger.warning("%s", alarm_warning(
                    alarm, pid=os.getpid(), threads=threading.active_count(), load=load_average_1m()))
            if stop_event.wait(interval_s):
                return

    try:
        from agent.memory_provider import spawn_context_thread

        # Under the spawner's contextvars, like every other gateway background thread: the samples
        # themselves are host facts, but the log line must land in the owner's profile scope.
        thread = spawn_context_thread(_watch, name="gateway-loop-latency-alarm")
        thread.start()
    except Exception:
        logger.debug("Failed to start gateway loop latency alarm", exc_info=True)
        return None
    return _AlarmHandle(stop_event, thread)
