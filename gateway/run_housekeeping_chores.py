"""The chores a gateway housekeeping tick runs.

Split out of ``gateway/run.py`` (which keeps the scheduler, ``_start_gateway_housekeeping``, and the
per-profile state.db/FTS/checkpoint jobs). Each chore opens its own dependency lazily so the
housekeeping thread never pays a facade import cycle, and ``_housekeeping_chore`` is the single
failure boundary: a broken chore costs one debug line, never the tick loop.
"""

from __future__ import annotations

import logging

from agent.async_utils import safe_schedule_threadsafe

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


def _housekeeping_chore(label: str, fn, *args, **kwargs) -> None:
    """Run one housekeeping chore; failures log at debug (a persistent failure such as a broken
    import after a partial update would otherwise warn every tick forever) and never stop the loop."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        logger.debug("%s error: %s", label, exc)


def _run_due_chores(chores, tick_count: int) -> None:
    """Run the chores due on this tick, as one timed unit (``housekeeping_tick`` in the tick-cost log).

    The housekeeping tick runs on its own thread, but its cost is still a loop-starvation probe:
    a chore that schedules onto the gateway loop and waits (channel directory refresh, the low-disk
    prompt) inflates the tick exactly when the loop is busy, while a chore wedged in a blocking read
    inflates it without touching the loop at all. The number is what separates those two readings,
    and it separates them from a slow disk, which only a per-chore figure would (PRX-5790).
    """
    from gateway.loop_tick_timing import timed
    with timed('housekeeping_tick'):
        for every, label, fn in chores:
            if tick_count % every == 0:
                _housekeeping_chore(label, fn)


def _housekeeping_channel_directory(adapters, loop) -> None:
    from gateway.channel_directory import build_channel_directory
    if loop is not None:
        # build_channel_directory is async (Slack web calls) and this is a background thread:
        # schedule onto the gateway loop and wait briefly so refresh failures still log.
        fut = safe_schedule_threadsafe(
            build_channel_directory(adapters), loop, logger=logger,
            log_message="Channel directory refresh scheduling error")
        if fut is not None:
            fut.result(timeout=30)


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
