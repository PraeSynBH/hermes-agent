"""Per-tick wall time of the gateway loop's periodic watchers, and its periodic summary.

The blocked-loop dump (``gateway/shutdown_watchdog.py``) proves a stall happened but samples ONE
frame of one tick: it names the site that happened to be running when the 10 s probe expired, not
how long a tick costs or whether a fix moved the work. PRX-5524/PRX-5780 removed the synchronous
SQLite each ticker used to run inline (control snapshot, hosted-room scope, local and native
recovery, the process watcher), and the acceptance for that is a TREND: the same tick got cheaper.

What is recorded here is the wall time inside one tick, from entry to exit — a bound on what the
tick costs its caller, not a measure of loop occupancy: a tick that hands its store work to
``asyncio.to_thread`` and awaits it still spends that time inside the tick while the loop is free.
So the summary is advisory evidence beside the dumps and the loopback p95, never a substitute for
them: a tick's wall time falling from seconds to milliseconds is the fix landing; a tick still at
seconds with no dump is the worker hop doing its job. Every counter is process-local and monotonic.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import time

logger = logging.getLogger(__name__)

# One INFO line per interval naming every ticker's count / p50 / max. 10 minutes keeps the trend
# readable in ``hermes logs`` through an 8 h shift (~48 lines) without touching a hot path's cost.
SUMMARY_INTERVAL_S = 600.0
# Ring size per ticker: the last 64 ticks, enough for a nearest-rank p50 that follows a change in
# load within minutes on the 1 s ticker and within half an hour on the 30 s one.
WINDOW = 64

_counters: dict[str, dict] = {}
_last_summary = 0.0


def _counter(name: str) -> dict:
    rec = _counters.get(name)
    if rec is None:
        rec = _counters[name] = {'count': 0, 'total': 0.0, 'max': 0.0, 'last': 0.0,
                                 'window': [0.0] * WINDOW, 'at': 0}
    return rec


def _nearest_rank_p50(samples) -> float:
    ordered = sorted(samples)
    return ordered[max(0, (len(ordered) - 1) // 2)] if ordered else 0.0


def snapshot() -> dict[str, dict]:
    """One read-only view of every ticker's counters (``count`` ticks start at zero on import)."""
    view = {}
    for name, rec in _counters.items():
        filled = min(rec['count'], WINDOW)
        window = rec['window'][:filled]
        view[name] = {
            'count': rec['count'],
            'last_s': rec['last'],
            'mean_s': (rec['total'] / rec['count']) if rec['count'] else 0.0,
            'p50_s': _nearest_rank_p50(window),
            'max_s': rec['max'],
        }
    return view


def reset() -> None:
    """Drop every counter (tests, and a supervisor that wants a clean window after a restart)."""
    global _last_summary
    _counters.clear()
    _last_summary = 0.0


def log_summary(now: float | None = None) -> str | None:
    """Emit the trend line if the interval elapsed; returns the line it logged (else None)."""
    global _last_summary
    now = time.monotonic() if now is None else now
    if not _counters or (now - _last_summary) < SUMMARY_INTERVAL_S:
        return None
    _last_summary = now
    parts = []
    for name, value in sorted(snapshot().items()):
        parts.append(f"{name} n={value['count']} p50={value['p50_s'] * 1000.0:.0f}ms"
                     f" max={value['max_s'] * 1000.0:.0f}ms")
    line = 'Loop tick timing (wall time inside the tick): ' + '; '.join(parts)
    logger.info(line)
    return line


def record(name: str, seconds: float, *, now: float | None = None) -> None:
    """Account one tick of *name* (its wall time) and log the periodic summary when it is due."""
    rec = _counter(name)
    rec['count'] += 1
    rec['total'] += seconds
    rec['last'] = seconds
    rec['max'] = max(rec['max'], seconds)
    rec['window'][rec['at'] % WINDOW] = seconds
    rec['at'] += 1
    log_summary(now)


@contextmanager
def timed(name: str):
    """Time one pass (or one unit of work) of ticker *name*; recorded even when it raises."""
    start = time.monotonic()
    try:
        yield
    finally:
        record(name, time.monotonic() - start)
