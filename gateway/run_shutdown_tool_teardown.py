"""Tool-subprocess teardown for the stop path, and the best-effort step wrapper it runs on.

Split out of ``gateway/run_shutdown.py`` (its growth ratchet, SOU-3966). Nothing here decides WHEN
to tear down — the phase order lives in the stop sequence; this module owns what one sweep does and
how a failing step is swallowed. ``kill_all`` fans out into per-target ``kill_process`` calls that
do blocking work (registry checkpoint disk I/O, ``subprocess.run`` for systemd scopes, sandbox
exec), so a caller on the loop hops off it first.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


def quiet_step(label: str, fn: Callable[[], Any]) -> Any:
    """Run one best-effort teardown step; a failure is debug-logged as ``"<label>: <exc>"``."""
    try:
        return fn()
    except Exception as _e:  # health: allow BLE001 -- one failing teardown step must not abort the rest; the label names it and the shutdown continues
        logger.debug("%s: %s", label, _e)
        return None


def stop_kill_tool_subprocesses(phase: str) -> list:
    """Kill tool subprocesses + terminal envs + browsers; returns cron job IDs marked interrupted.

    Called twice: after a drain timeout (reclaim children before systemd SIGKILLs) and as a final
    catch-all. Best-effort; one failing subsystem cannot block the rest.
    """

    def _step(label: str, fn: Callable[[], Any]) -> Any:
        return quiet_step(f"{label} ({phase}) error", fn)

    def _count_step(fmt: str, fn: Callable[[], int]) -> None:
        n = fn()
        if n:
            logger.info(fmt, phase, n)

    def _kill_processes() -> None:
        from tools.process_registry import process_registry
        # Host shutdown: kill even persist_on_release jobs or they become
        # PPID=1 orphans (#41225/#46778); an explicit source reaches them.
        _count_step(
            "Shutdown (%s): killed %d tool subprocess(es)",
            lambda: process_registry.kill_all(source="gateway_shutdown"))

    def _mark_cron_interrupted() -> list:
        # kill_all() is global: a cron job mid-dispatch lost its tool subprocess and its agent thread may
        # still emit a plausible response from truncated output — mark it interrupted, never success.
        # Any cron job still dispatched at this instant just had its tool subprocess killed above
        # (kill_all() has no per-job-ID targeting — it's a global sweep). No-op when no cron job is in
        # flight. See #60432.
        from cron.scheduler import mark_running_jobs_interrupted
        _interrupted = mark_running_jobs_interrupted(
            f"Gateway shutdown ({phase}) killed the job's tool subprocess before the run finished."
        )
        if _interrupted:
            logger.warning(
                "Shutdown (%s): marked %d in-flight cron job(s) interrupted: %s",
                phase, len(_interrupted), ", ".join(_interrupted),
            )
        return _interrupted

    def _interrupt_delegations() -> None:
        from tools.async_delegation import interrupt_all as _interrupt_async
        _count_step(
            "Shutdown (%s): interrupted %d background delegation(s)",
            lambda: _interrupt_async(reason=f"gateway shutdown ({phase})"),
        )

    _step("process_registry.kill_all", _kill_processes)
    _marked_cron_jobs = _step("mark_running_jobs_interrupted", _mark_cron_interrupted) or []
    _step("async interrupt_all", _interrupt_delegations)
    def _cleanup_environments() -> None:
        from tools.terminal_tool_lifecycle import cleanup_all_environments
        cleanup_all_environments()

    def _cleanup_browsers() -> None:
        from tools.browser_session_lifecycle import cleanup_all_browsers
        cleanup_all_browsers()

    _step("cleanup_all_environments", _cleanup_environments)
    _step("cleanup_all_browsers", _cleanup_browsers)
    return _marked_cron_jobs


async def stop_kill_tool_subprocesses_off_loop(phase: str) -> list:
    """Run ``stop_kill_tool_subprocesses`` in a worker thread; returns cron job IDs marked interrupted.

    ``kill_all`` fans out into per-target ``kill_process`` calls that do blocking work
    (registry checkpoint disk I/O, ``subprocess.run`` for systemd scopes, sandbox exec),
    so running the sweep inline would monopolize the gateway event loop (#116327).
    Offloaded with ``asyncio.to_thread`` — the loop's default executor, deliberately NOT
    the gateway-owned ``self._executor``, which ``_stop_quiesce_and_close_session_dbs``
    drains right after this phase. Phase order is preserved: callers await this before
    cron notices / adapter teardown. If the surrounding stop task is cancelled while the
    worker runs, the thread is left to finish on its own; the thread-based shutdown
    watchdog remains the hard backstop.
    """
    return await asyncio.to_thread(stop_kill_tool_subprocesses, phase)
