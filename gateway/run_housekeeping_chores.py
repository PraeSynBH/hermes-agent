"""The chores a gateway housekeeping tick runs.

Split out of ``gateway/run.py`` (which keeps the scheduler, ``_start_gateway_housekeeping``, and the
per-profile state.db/FTS/checkpoint jobs). Each chore opens its own dependency lazily so the
housekeeping thread never pays a facade import cycle, and ``_housekeeping_chore`` is the single
failure boundary: a broken chore costs one debug line, never the tick loop.
"""

from __future__ import annotations

import logging
import re
import threading
import time

from agent.async_utils import safe_schedule_threadsafe

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

# Per-tick ceiling for the whole tick (PRX-6104). The tick runs on its own thread, so its wall time is
# not loop occupancy — but an arm that schedules onto the gateway loop and waits spends that wait
# inside the tick, which is how one arm reached 16-17 s while the loop was busy with another watcher.
# Measured on this host before the ceiling: p50 4.8 s of every 60 s and a 17.8 s steady-state tail
# (PRX-6095, PRX-6112). Chores that no longer fit are carried to the next tick instead of extending
# this one, so the tick's cost stops being the sum of every due arm.
HOUSEKEEPING_TICK_BUDGET_S = 2.0
# A nearly-spent tick still starts a chore: the warm arms are ms-scale, so the floor only defers work
# that could not have finished anyway. Whatever overruns it is what the split names.
_MIN_CHORE_S = 0.05
# Arms whose whole job is to prove the process is alive run even on an already-spent tick: a carried
# heartbeat would age gateway_state.json past the 2x-interval warning in `hermes gateway status`
# while the tick is behaving exactly as designed.
_LIVENESS_CHORES = frozenset({"Runtime heartbeat", "Cron ticker supervisor"})
# The split shows the dearest arms and folds the rest into one figure: enough to name the arm on any
# over-budget tick, short enough to stay one log line.
_SPLIT_ARMS = 6
# An under-budget tick logs its split at INFO only above this; a quiet tick stays at DEBUG.
_SLOW_TICK_LOG_S = 0.5
# The channel-directory refresh is a Slack web fan-out on the loop and the tick waits for it: at most
# this much of the tick's budget, then the build finishes on the loop alone. A cache refresh with a
# 5-minute cadence is never worth more — and the pre-fix ceiling here was 30 s.
_CHANNEL_DIRECTORY_WAIT_S = 1.0

# The running tick's deadline, read by arms that wait on the gateway loop. Thread-local because the
# tick is one thread and its arms run on it.
_tick_deadline = threading.local()
# A directory build still running on the loop from a tick that stopped waiting for it. Scheduling a
# fresh build every 5 ticks would stack Slack fan-outs behind each other on the same busy loop.
_channel_directory_lock = threading.RLock()
_channel_directory_inflight = None


def remaining_tick_budget(default_s: float) -> float:
    """Seconds the running tick can still spend, capped at *default_s*; *default_s* outside a tick.

    An arm that schedules onto the gateway loop and blocks on the future converts the loop's latency
    into the tick's cost. Bounding that wait by what is left of the tick's own ceiling is what stops
    one busy loop from turning a cache refresh into the whole tick (PRX-6104 ask 2).
    """
    deadline = getattr(_tick_deadline, "deadline", None)
    if deadline is None:
        return default_s
    return max(_MIN_CHORE_S, min(default_s, deadline - time.monotonic()))


def _chore_slug(label: str) -> str:
    """Log-field form of a chore label: ``Channel directory refresh`` -> ``channel-directory-refresh``."""
    return "-".join(re.findall(r"[a-z0-9]+", label.lower()))


def _log_tick_split(split, total_s: float, budget_s: float, carried: int) -> None:
    """One line per tick naming the arms that cost it, at WARNING when the ceiling was passed.

    Without it the tick is a single number: "which chore" is a guess, and a per-chore budget cannot be
    sized from a guess (PRX-6104 ask 1). The over-budget form is the one a sweep greps:
    ``housekeeping_tick over its 2000ms budget: 2410ms total (channel-directory-refresh=1900ms ...)``.
    """
    ordered = sorted(split, key=lambda arm: arm[1], reverse=True)
    shown = ordered[:_SPLIT_ARMS]
    fields = [f"{name}={seconds * 1000.0:.0f}ms" for name, seconds in shown]
    if len(ordered) > len(shown):
        fields.append(f"other={sum(s for _n, s in ordered[len(shown):]) * 1000.0:.0f}ms")
    if carried:
        fields.append(f"carried={carried}")
    detail = ", ".join(fields)
    if total_s > budget_s:
        logger.warning("housekeeping_tick over its %.0fms budget: %.0fms total (%s)",
                       budget_s * 1000.0, total_s * 1000.0, detail)
    elif total_s > _SLOW_TICK_LOG_S:
        logger.info("housekeeping_tick %.0fms (%s)", total_s * 1000.0, detail)
    else:
        logger.debug("housekeeping_tick %.0fms (%s)", total_s * 1000.0, detail)


def _housekeeping_chore(label: str, fn, *args, **kwargs) -> None:
    """Run one housekeeping chore; failures log at debug (a persistent failure such as a broken
    import after a partial update would otherwise warn every tick forever) and never stop the loop."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        logger.debug("%s error: %s", label, exc)


def _run_due_chores(chores, tick_count: int, carried: dict | None = None,
                    budget_s: float | None = None) -> None:
    """Run the chores due on this tick under one ceiling, carrying the ones that do not fit.

    The tick runs on its own thread, but its cost is still a loop-starvation probe: a chore that
    schedules onto the gateway loop and waits (channel directory refresh, the low-disk prompt)
    inflates the tick exactly when the loop is busy, while a chore wedged in a blocking read inflates
    it without touching the loop at all. The number is what separates those two readings, and the
    per-chore split is what names the arm (PRX-5790, PRX-6104).

    *carried* is the caller's ``{chore index: consecutive ticks deferred}``, and it is what makes the
    ceiling a rotation instead of a starvation: the most-deferred chore runs first on the next tick,
    so no arm can sit behind a slow one forever. Cadences stay ticks of ``interval`` (an inner gate
    owns any real-time schedule), so a carried chore only delays work whose own marker decides when it
    is due — the checkpoint prune and the state.db maintenance are both gated that way.
    """
    from gateway.loop_tick_timing import timed
    budget = HOUSEKEEPING_TICK_BUDGET_S if budget_s is None else budget_s
    carried = {} if carried is None else carried
    due = [i for i, (every, _label, _fn) in enumerate(chores) if tick_count % every == 0]
    # Most-starved first; ties keep table order, which is how the heartbeat stays ahead of everything
    # in its defer class. A tick that carries exactly one chore is the tick whose split names it.
    queue = sorted(set(due) | set(carried),
                   key=lambda i: (-carried.get(i, 0), due.index(i) if i in due else len(chores)))
    split: list[tuple[str, float]] = []
    started = time.monotonic()
    _tick_deadline.deadline = started + budget
    try:
        with timed('housekeeping_tick', budget_s=budget, per_tick_debug=False):
            spent = 0.0
            for index in queue:
                label = chores[index][1]
                if spent >= budget and label not in _LIVENESS_CHORES:
                    carried[index] = carried.get(index, 0) + 1
                    continue
                chore_start = time.monotonic()
                _housekeeping_chore(label, chores[index][2])
                elapsed = time.monotonic() - chore_start
                spent += elapsed
                carried.pop(index, None)
                split.append((_chore_slug(label), elapsed))
            _log_tick_split(split, time.monotonic() - started, budget, len(carried))
    finally:
        _tick_deadline.deadline = None


def _channel_directory_done(fut) -> None:
    """Release the refresh slot and observe the outcome of a build the tick stopped waiting for."""
    global _channel_directory_inflight
    with _channel_directory_lock:
        if _channel_directory_inflight is fut:
            _channel_directory_inflight = None
    try:
        exc = fut.exception()
    except Exception:  # health: allow BLE001 -- a cancelled future has no outcome to report
        return
    if exc is not None:
        logger.debug("Channel directory refresh failed: %s", exc)


def _housekeeping_channel_directory(adapters, loop) -> None:
    """Refresh the channel directory, waiting no longer than this tick can afford.

    ``build_channel_directory`` is async (Slack web calls) and this is a background thread, so the
    build is scheduled onto the gateway loop and awaited — but the wait is bounded by the tick's
    remaining budget (hard cap ``_CHANNEL_DIRECTORY_WAIT_S``), not by a 30 s ceiling of its own: on a
    loop busy with another watcher that ceiling made the loop's latency the tick's cost (15.9 s of a
    16.4 s tick, PRX-6104). Past the deadline the tick returns and the build finishes on the loop; a
    later tick that finds a build still in flight skips the refresh rather than stacking a second
    fan-out behind it. The directory is a cache, so a skipped refresh costs one cadence of staleness.
    """
    global _channel_directory_inflight
    from gateway.channel_directory import build_channel_directory
    if loop is None:
        return
    with _channel_directory_lock:
        pending = _channel_directory_inflight
        if pending is not None:
            if not pending.done():
                logger.debug("Channel directory refresh still in flight from an earlier tick; skipping")
                return
            _channel_directory_inflight = None
        fut = safe_schedule_threadsafe(
            build_channel_directory(adapters), loop, logger=logger,
            log_message="Channel directory refresh scheduling error")
        if fut is None:
            return
        _channel_directory_inflight = fut
        fut.add_done_callback(_channel_directory_done)
    wait_s = remaining_tick_budget(_CHANNEL_DIRECTORY_WAIT_S)
    try:
        fut.result(timeout=wait_s)
    except TimeoutError:
        logger.debug("Channel directory refresh still running after %.0fms; leaving it to the loop",
                     wait_s * 1000.0)
    except Exception as exc:  # health: allow BLE001 -- one tick's refresh failure is a debug line
        logger.debug("Channel directory refresh failed: %s", exc)


def _housekeeping_paste_sweep() -> None:
    from hermes_cli.debug import _sweep_expired_pastes
    deleted, remaining = _sweep_expired_pastes()
    if deleted:
        logger.info("Paste sweep: deleted %d expired paste(s), %d pending", deleted, remaining)


def _housekeeping_misfire_catch_up(cron_provider, adapters, loop) -> None:
    """External cron providers only: fire jobs whose time passed with no external fire delivered (dead
    loopback hop). No-op for the built-in ticker; enforces misfire_grace_minutes; CAS claim de-dupes."""
    from cron.scheduler_provider import fire_overdue_jobs
    caught_up = fire_overdue_jobs(cron_provider, adapters=adapters, loop=loop)
    if caught_up:
        logger.info("Misfire catch-up: fired %d overdue job(s)", caught_up)


def _housekeeping_curator() -> None:
    """maybe_run_curator() is gated by config.interval_hours (7 days default); this is the poll."""
    from agent.curator import maybe_run_curator
    maybe_run_curator(idle_for_seconds=float("inf"), on_summary=lambda msg: logger.info("curator: %s", msg))


def _housekeeping_plugin_update_check() -> None:
    """Plugin update-check cadence (plugins_cadence): due-gated by
    plugins.auto_update_check_hours, read-only, receipt-surfaced; the
    opt-in auto-apply rides the manual update pipeline. A network error
    costs one warning and a stamped marker — never an apply."""
    from hermes_cli.plugins_cadence import maybe_run_gateway_check

    maybe_run_gateway_check(log=logger)


def _housekeeping_deferred_fts_retry() -> None:
    """A SessionDB opened while another process held the rebuild lock fails closed onto the LIKE fallback
    and the gateway stays up for days. Non-blocking, rate-limited inside SessionDB; no-op when not stale."""
    # Retry here, on the existing tick, against the shared instances this process already holds:
    # non-blocking admission, no new thread, rate-limited inside SessionDB. No-op when nothing is stale (one
    # attribute read per instance). See #100108.
    from hermes_state_registry import borrow_live_shared_session_dbs
    with borrow_live_shared_session_dbs() as _session_dbs:
        for _sdb in _session_dbs:
            _retry = getattr(_sdb, "retry_deferred_fts_recovery", None)
            if callable(_retry) and _retry():
                logger.info(
                    "Deferred state.db FTS rebuild completed in-process for %s; full-text search restored.",
                    getattr(_sdb, "db_path", "state.db"))


def _housekeeping_memory_trim() -> None:
    """Messaging-gateway counterpart to the TUI idle reaper; config-gated and rate-limited inside."""
    from hermes_cli.mem_trim import trim_memory
    trim_memory(reason="messaging gateway housekeeping")


def _housekeeping_checkpoint_prune() -> None:
    """Checkpoint store retention + size cap on a live timer; ``auto_prune_from_config`` gates on
    ``checkpoints.auto_prune`` and the 24h ``.last_prune`` marker. Off the startup path because its
    ``git gc`` can block for tens of seconds on a large store — it is the last chore of the tick for
    that reason, and a tick that spends its ceiling on it carries the rest rather than waiting."""
    from tools.checkpoint_maintenance import auto_prune_from_config
    auto_prune_from_config()
