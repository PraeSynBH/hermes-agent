"""What the shutdown watchdog dumps, and what it is allowed to claim about the phase it interrupted.

``gateway/run_shutdown.py`` owns the stop phases; this sibling owns the watchdog's *view* of them.

The leash is counted from ``stop()`` and bounds a SEQUENCE — notify the active sessions, drain them,
tear down — while the dump used to report one fixed verdict ("asyncio drain path appears wedged") for
all of it. On 2026-10-21T15:55:19Z that sentence described a teardown 8 s old on an event loop parked
in ``select()``: the drain *had* finished, the lease simply ran out (PRX-5828, 7 amputated turns). The
verdict here names the phase, its age and whether the drain had already returned, so the next reader
weighs a slow teardown against a real hang instead of chasing a wedge that is not there.
"""

from __future__ import annotations

from typing import Any, Optional

from gateway.restart import effective_stop_drain_timeout, effective_stop_watchdog_delay


def effective_stop_watchdog_leash(runner: Any) -> float:
    """Leash the shutdown watchdog is armed with: drain + grace, clamped under launchd's SIGKILL.

    The grace and drain come from the phases the stop actually gets (``effective_stop_drain_timeout``
    = the launchd-capped drain), so the leash moves with the cap instead of being a second opinion —
    PRX-5828's 58 s leash against a 50 s cap + 11 s teardown was exactly two opinions.
    ``resolve_shutdown_watchdog_delay`` is read off ``gateway.run_shutdown`` on purpose: tests and
    callers patch it there, and this module must not become a second binding of that seam.
    """
    from gateway import run_shutdown

    return effective_stop_watchdog_delay(
        runner,
        run_shutdown.resolve_shutdown_watchdog_delay(effective_stop_drain_timeout(runner)),
    )


def _optional_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def shutdown_watchdog_snapshot(runner: Any, ctx: Any, *, watchdog_delay_s: float) -> dict:
    """State the thread-based shutdown watchdog dumps when ``stop()`` overruns its leash.

    ``ctx`` is the stop's ``_StopContext``: it carries the phase the leash ran out on and how much of
    ``drain_cap_s`` the drain had already spent (``drain_finished_at_s`` is set the moment the drain
    returns, which is what distinguishes a slow teardown from a wedged drain).
    """
    return {
        "restart_requested": bool(runner._restart_requested),
        "draining": bool(runner._draining),
        "running": bool(runner._running),
        "active_agents": runner._running_agent_count(),
        "active_cron_jobs": runner._active_cron_job_count(),
        "active_api_runs": runner._active_api_run_count(),
        "active_deferred_agent_workers": ctx.deferred_count(),
        "restart_drain_timeout": runner._restart_drain_timeout,
        "effective_drain_timeout": effective_stop_drain_timeout(runner),
        "launchd_exit_timeout_s": getattr(runner, "_launchd_exit_timeout_s", None),
        "watchdog_delay_s": watchdog_delay_s,
        "phase": ctx.phase,
        "phase_age_s": ctx.phase_age(),
        "phase_elapsed_s": ctx.elapsed() if ctx.started_at is not None else None,
        "drain_cap_s": ctx.drain_cap_s,
        "drain_elapsed_s": ctx.drain_elapsed,
        "drain_finished_at_s": ctx.drain_finished_at_s,
        "drain_timed_out": bool(ctx.timed_out),
    }


def describe_shutdown_watchdog_firing(snapshot: dict) -> str:
    """Name the shutdown phase a forced exit interrupted, instead of asserting a wedge.

    Only the drain can make an idle loop look hung; a fire during or after teardown is the lease
    running out on a process that was still making progress. Say which phase, how old it is, and
    whether the drain had finished — that is the difference between "fix the budget" and "find the
    hang", and the 15:55:19Z dump sent its reader to the wrong one.
    """
    phase = str(snapshot.get("phase") or "unknown")
    age = _optional_float(snapshot.get("phase_age_s"))
    total = _optional_float(snapshot.get("phase_elapsed_s"))
    cap = _optional_float(snapshot.get("drain_cap_s")) or 0.0
    drain_done = _optional_float(snapshot.get("drain_finished_at_s"))
    at = f"'{phase}' {age:.1f}s in" if age is not None else f"phase '{phase}'"
    if total is not None:
        at += f" at +{total:.1f}s"
    if phase == "drain":
        if age is not None and age >= cap:
            return f"{at} — the drain overran its own {cap:.0f}s cap: stalled inside the drain"
        return f"{at} — inside the drain before its {cap:.0f}s cap: a stall, not a lease overrun"
    if drain_done is not None:
        after = f"{total - drain_done:.1f}s after " if total is not None else "after "
        return (f"{at} — {after}the drain finished at +{drain_done:.1f}s (cap {cap:.0f}s), so the "
                "drain was not wedged; post-drain work overran the lease")
    if phase == "unknown":
        return (f"{at} — no phase was recorded, so nothing here attributes the exit; treat it as an "
                "unexplained stall and read the thread dump")
    return f"{at} — before the drain started, so the drain did not consume the lease"
