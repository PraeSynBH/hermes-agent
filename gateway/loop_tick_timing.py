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

The cold-start-vs-load question on PRX-5524 (the first ~2 min after boot, or only under load1 > 100)
is settled by reading one line per watcher per minute against ``uptime`` at the same timestamps:
``p95`` over the window is the load-coupled reading, ``boot_max`` keeps the boot spike visible after
it has rolled out of the window.

Cost: two ``time.monotonic()`` calls per tick plus a dict write — no timers, no threads, no
per-tick allocation beyond the float. ``sorted()`` runs once per watcher per interval, in the
interval's own tick, and the ring sorts at most ``WINDOW`` floats.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import math
import time

logger = logging.getLogger(__name__)

# One line per watcher per minute: the interval has to be fine enough to line the numbers up
# against ``uptime`` (load1) at the same timestamp, which is the whole point of the card.
SUMMARY_INTERVAL_S = 60.0
# Ring size per watcher. At one line a minute this covers the last ~60 ticks of the 1 s watcher (a
# minute of load) and the last 32 min of the 30 s one; a nearest-rank p95 over it is 3 ticks off the
# worst, so the line reports the tail instead of hiding it behind a mean.
WINDOW = 64
# One tick this long is a stall candidate on its own (the watchdog dumps at 10 s), so it gets a
# WARNING immediately — but at most one per watcher per interval, or a starved loop floods the log
# with a warning per second and buries everything else.
SLOW_TICK_S = 1.0

_counters: dict[str, dict] = {}
_last_summary = 0.0


def _counter(name: str) -> dict:
    rec = _counters.get(name)
    if rec is None:
        rec = _counters[name] = {'count': 0, 'total': 0.0, 'max': 0.0, 'last': 0.0,
                                 'window': [0.0] * WINDOW, 'at': 0,
                                 'slow': 0, 'slow_total': 0, 'warned_at': None,
                                 'budget': 0.0, 'used': 0.0, 'over': 0, 'over_total': 0}
    return rec


def _nearest_rank(samples, q: float) -> float:
    """Nearest-rank percentile of *samples* (0 < q <= 1); 0.0 for an empty window."""
    ordered = sorted(samples)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def snapshot() -> dict[str, dict]:
    """One read-only view of every watcher's counters (``count`` starts at zero on import).

    Window statistics (``window_count``, ``p50_s``, ``p95_s``, ``max_window_s``) describe the last
    ``WINDOW`` ticks; ``max_s`` / ``mean_s`` / ``count`` are since import, because a boot spike has
    to stay readable after it rolls out of the ring.
    """
    view = {}
    for name, rec in _counters.items():
        filled = min(rec['count'], WINDOW)
        window = rec['window'][:filled]
        view[name] = {
            'count': rec['count'],
            'window_count': filled,
            'last_s': rec['last'],
            'mean_s': (rec['total'] / rec['count']) if rec['count'] else 0.0,
            'p50_s': _nearest_rank(window, 0.50),
            'p95_s': _nearest_rank(window, 0.95),
            'max_s': rec['max'],
            'max_window_s': max(window) if window else 0.0,
            'slow_since_summary': rec['slow'],
            'slow_total': rec['slow_total'],
            # Per-tick ceiling the watcher enforces, and what the last tick spent against it:
            # a watcher that slices its work reports here whether the slice held (PRX-6075).
            'budget_s': rec['budget'],
            'budget_used_s': rec['used'],
            'over_budget_since_summary': rec['over'],
            'over_budget_total': rec['over_total'],
        }
    return view


def reset() -> None:
    """Drop every counter (tests, and a supervisor that wants a clean window after a restart)."""
    global _last_summary
    _counters.clear()
    _last_summary = 0.0


def _ms(seconds: float) -> str:
    """Milliseconds, keeping a digit where the value has one: ``0.02ms`` and ``1250ms`` both read."""
    millis = seconds * 1000.0
    if millis < 1.0:
        return f"{millis:.2f}"
    if millis < 10.0:
        return f"{millis:.1f}"
    return f"{millis:.0f}"


def _line(name: str, value: dict) -> str:
    line = (f"Tick cost {name} n={value['window_count']}"
            f" p50={_ms(value['p50_s'])}ms"
            f" p95={_ms(value['p95_s'])}ms"
            f" max={_ms(value['max_window_s'])}ms"
            f" boot_max={_ms(value['max_s'])}ms")
    if value['slow_since_summary']:
        line += f" slow={value['slow_since_summary']}/{value['slow_total']}"
    if value['budget_s']:
        # ``budget_used`` closes the loop the issue asked for: the ceiling is only real if a
        # sweep can read what the tick spent against it, beside the slow= counter.
        line += (f" budget_used={_ms(value['budget_used_s'])}ms/{_ms(value['budget_s'])}ms"
                 f" over={value['over_budget_since_summary']}/{value['over_budget_total']}")
    return line


def log_summary(now: float | None = None) -> list[str] | None:
    """Emit one line per watcher if the interval elapsed; returns the lines it logged (else None).

    Each line stands alone under the module's logger (``gateway.loop_tick_timing``) so a sweep can
    grep one watcher across a boot, and so the record's own timestamp is the timestamp to compare
    against load1.
    """
    global _last_summary
    now = time.monotonic() if now is None else now
    if not _counters:
        return None
    if _last_summary <= 0.0:
        # The first tick opens the window instead of closing an empty one: a line at boot would
        # carry n=1, while the first line a minute later carries the whole boot minute.
        _last_summary = now
        return None
    if (now - _last_summary) < SUMMARY_INTERVAL_S:
        return None
    _last_summary = now
    lines = []
    for name, value in sorted(snapshot().items()):
        line = _line(name, value)
        logger.info(line)
        lines.append(line)
        # Both counts are "since the last summary" in the line (`slow=` / `over=`), each beside its
        # own running total. Resetting only one of them makes the other read `<total>/<total>` on
        # every line, so a sweep cannot tell an overrun in THIS window from one at boot — which is
        # the only question a per-hour claim asks (PRX-6075 ask 3, PRX-6091).
        _counters[name]['slow'] = 0
        _counters[name]['over'] = 0
    return lines


def record(name: str, seconds: float, *, budget_s: float | None = None,
           now: float | None = None) -> None:
    """Account one tick of *name* (its wall time) and log the periodic summary when it is due.

    *budget_s* is the per-tick ceiling the watcher slices its work to. Recorded so the summary can
    report ``budget_used`` and an over-budget count: without a number beside the ceiling, "the tick
    is bounded" is a claim no sweep can check (PRX-6075).
    """
    rec = _counter(name)
    rec['count'] += 1
    rec['total'] += seconds
    rec['last'] = seconds
    rec['max'] = max(rec['max'], seconds)
    rec['window'][rec['at'] % WINDOW] = seconds
    rec['at'] += 1
    if budget_s is not None:
        rec['budget'] = budget_s
        rec['used'] = seconds
        if seconds > budget_s:
            rec['over'] += 1
            rec['over_total'] += 1
    if seconds >= SLOW_TICK_S:
        rec['slow'] += 1
        rec['slow_total'] += 1
        at = time.monotonic() if now is None else now
        warned_at = rec['warned_at']
        if warned_at is None or (at - warned_at) >= SUMMARY_INTERVAL_S:
            rec['warned_at'] = at
            logger.warning("Tick %s took %.0fms (>= %.0fms); further slow ticks of this watcher are"
                           " counted, not logged, until the next summary",
                           name, seconds * 1000.0, SLOW_TICK_S * 1000.0)
    log_summary(now)


@contextmanager
def timed(name: str, *, budget_s: float | None = None, per_tick_debug: bool = True):
    """Time one pass (or one unit of work) of watcher *name*; recorded even when it raises.

    ``budget_s`` is the per-tick ceiling this watcher enforces; pass it so the summary reports
    ``budget_used`` and the over-budget count.
    ``per_tick_debug=False`` for the fast watchers (the 1 s hosted-room worker): their per-tick line
    is 86 400 records a day at DEBUG, and the per-minute summary already carries count/p50/p95/max.
    """
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        record(name, elapsed, budget_s=budget_s)
        if per_tick_debug:
            logger.debug("tick %s %.1fms", name, elapsed * 1000.0)
