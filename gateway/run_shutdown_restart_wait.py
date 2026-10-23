"""The restart's after-turn wait: what it waits for, how long, and what that costs cron.

Split out of ``gateway/run_shutdown.py`` (PRA-6925): the wait is the one place in the shutdown path
whose LENGTH is a fleet-wide decision, because admission closes process-wide while it runs. Cron
dispatch closed for the whole wait is what made a 30-minute bounce look like a stalled cron ticker;
the bound added here (``_cron_hold_bound``) is the counterweight, and it belongs next to the work
accounting it is derived from rather than in the teardown body.

``gateway.run`` internals are imported lazily inside method bodies (import cycle), so
``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import suppress
from typing import Any, Optional

from gateway.run_runtime import managed_turn_count

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


def _cron_hold_cost() -> Optional[tuple[int, int]]:
    """``(due slots, profiles holding them)`` across every ticked profile, or ``None`` if unreadable.

    One scan for both consumers below: the opening/status CLAUSE (``_cron_dispatch_hold_note``) and
    the deferral's cron bound, which needs the number rather than the sentence.
    """
    try:
        from cron.scheduler_drain_gate import due_slot_summary
        from cron.scheduler_ownership import ticked_homes

        return due_slot_summary(ticked_homes().values())
    except Exception as exc:  # health: allow BLE001 -- a diagnostic inside a shutdown, logged with its cause
        logger.debug("Restart deferral: cron hold cost unavailable: %s", exc)
        return None


def _cron_hold_bound() -> Optional[float]:
    """``cron.scheduler_drain_gate.hold_bound_seconds``; ``None`` (no bound) if cron can't say."""
    try:
        from cron.scheduler_drain_gate import hold_bound_seconds

        return hold_bound_seconds()
    except Exception:  # health: allow BLE001 -- failing here must leave the deferral exactly as it was
        return None


def _cron_dispatch_hold_note(cost: Optional[tuple[int, int]] = None) -> str:
    """The cron cost of THIS deferral, as one clause: what the closed admission is holding.

    Admission is process-wide (``can_dispatch``, cron/scheduler_drain_gate.py), so a restart's
    after-turn wait holds every ticked profile's due slots for its whole duration — 33-36 min of
    fleet-wide cron silence twice on 2026-10-22/23, and the deferral line never said so (PRA-6833).
    Counting here rather than only in the ticker's per-profile WARNINGs is what makes the cost
    nameable at T=0 in ONE place.

    Reads each ticked home's store directly (no store open, no scope switch, no lock) and never
    raises: this runs on the loop during a shutdown, and a diagnostic must not abort a wait that
    exists to protect in-flight work. *cost* is a scan the caller already took (so the deferral
    loop can arm its cron bound off the same read); omit it and the scan happens here.
    """
    cost = _cron_hold_cost() if cost is None else cost
    if cost is None:
        return "cron dispatch is held fleet-wide until the drain releases"
    due, holding = cost
    if due:
        return (f"cron dispatch is held fleet-wide until the drain releases: {due} due cron slot(s)"
                f" across {holding} profile(s) are waiting and fire as catch-up")
    return ("cron dispatch is held fleet-wide until the drain releases (no cron slot is due yet; "
            "any that comes due is held and fires as catch-up)")


class GatewayRestartWaitMixin:
    """Wait for in-flight work before ``stop()``, and name what that wait costs.

    Bound onto ``GatewayShutdownMixin`` (and so onto ``GatewayRunner``) via the MRO.
    """

    def _wedged_agent_count(self) -> int:
        """Work units the restart wait may skip: chat agents idle past ``agent.gateway_timeout`` and
        cron runs older than the scheduler's stale-inflight allowance (#115469).

        API work has no activity clock and pending sentinels are brand-new, so neither counts.
        """
        return self._wedged_chat_agent_count() + self._wedged_cron_job_count()

    def _restart_wait_cron_counts(self) -> dict:
        """``cron.scheduler.get_restart_wait_cron_counts``, counted per profile-scoped run.

        Fail-soft toward waiting: if the split can't be read, every active cron run stays
        awaitable and nothing is excluded.
        """
        try:
            from cron.scheduler import get_restart_wait_cron_counts
            return get_restart_wait_cron_counts()
        except Exception:
            return {"awaitable": self._active_cron_job_count(), "wedged": 0, "restart_safe": 0}

    def _wedged_cron_job_count(self) -> int:
        """Cron runs past ``cron.scheduler.get_wedged_job_ids``'s allowance; 0 if cron can't import."""
        return self._restart_wait_cron_counts()["wedged"]

    def _restart_safe_cron_count(self) -> int:
        """Cron runs whose worker owns a restart-safe systemd scope; 0 if cron can't import.

        Such a worker runs outside the gateway cgroup, so neither the tool-process sweep nor
        ``mark_running_jobs_interrupted`` reaches it, and its final send rides the durable delivery
        queue for whichever gateway is live next. The restart after-turn wait must therefore not
        hold the gateway in ``draining`` for it: waiting buys the run nothing and refuses new turns
        for up to the whole cap (observed live: a 100-minute job held the gateway ~30 minutes).
        Degraded (no user bus) workers are NOT in this set — they share the cgroup and die with a
        systemd stop, so they keep holding the wait.
        """
        return self._restart_wait_cron_counts()["restart_safe"]

    def _wedged_chat_agent_count(self) -> int:
        """Running chat agents with no activity for ``agent.gateway_timeout`` (0 when disabled);
        an unreadable activity summary means "not wedged".
        """
        from gateway.run import _AGENT_PENDING_SENTINEL, _float_env
        timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800)
        if timeout <= 0:
            return 0

        def _idle_seconds(agent: Any) -> Optional[float]:
            summary_fn = getattr(agent, "get_activity_summary", None)
            if not callable(summary_fn):
                return None
            try:
                summary = summary_fn()
                return float(summary.get("seconds_since_activity", 0.0)) if isinstance(summary, dict) else None
            except Exception:
                return None

        return sum(
            1
            for agent in list((getattr(self, "_running_agents", None) or {}).values())
            if agent is not None and agent is not _AGENT_PENDING_SENTINEL
            and (idle := _idle_seconds(agent)) is not None and idle >= timeout
        )

    def _awaitable_work_count(self) -> int:
        """Active work minus the units the restart wait must not hold for.

        Two disjoint exclusions: wedged turns (idle past ``agent.gateway_timeout`` / past the cron
        in-flight allowance — restart is their remedy, #115469) and cron runs executing in a
        restart-safe external worker (``_restart_safe_cron_count``), which outlives this process
        either way. Cron runs are counted per profile-scoped run by the scheduler, not by
        subtracting from the bare-ID active count: two profiles can run the same job ID, and one
        excluded run must not hide the other.
        """
        non_cron = (
            self._running_agent_count()
            + managed_turn_count(self)
            + self._active_api_run_count()
            + self._active_deferred_agent_worker_count()
        )
        return (
            max(0, non_cron - self._wedged_chat_agent_count())
            + self._restart_wait_cron_counts()["awaitable"]
        )

    def _describe_active_work(self) -> list:
        """One dict per in-flight work unit the restart wait is holding for, so an observer
        (``hermes update``, ``hermes gateway status``) can name it instead of printing a bare count.

        ``kind`` ∈ ``chat`` (session turn), ``cron`` (job id + external worker pid when the run was
        handed to a restart-safe scope), ``managed`` / ``api`` / ``deferred`` (count only — those sources expose
        no identity). Best-effort: a source that can't be read is omitted, never raises.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL
        now = time.time()
        units: list = []
        for key, state in list(self._sessions_map().items()):
            agent = state.turn.agent
            if agent is None:
                continue
            unit: dict = {"kind": "chat", "session": key, "pid": os.getpid()}
            if state.turn.started_ts:
                unit["elapsed_s"] = round(now - state.turn.started_ts, 1)
            if agent is not _AGENT_PENDING_SENTINEL:
                unit["model"] = getattr(agent, "model", None)
                summary_fn = getattr(agent, "get_activity_summary", None)
                if callable(summary_fn):
                    with suppress(Exception):
                        summary = summary_fn()
                        unit["current_tool"] = summary.get("current_tool")
                        unit["idle_s"] = summary.get("seconds_since_activity")
            units.append(unit)
        with suppress(Exception):
            from cron.scheduler import get_running_job_details, get_wedged_job_ids
            wedged = get_wedged_job_ids()
            for job in get_running_job_details():
                units.append({"kind": "cron", "job_id": job["job_id"], "elapsed_s": job["elapsed_s"],
                              "pid": job["worker_pid"] or os.getpid(), "external": bool(job["worker_pid"]),
                              "wedged": job["job_id"] in wedged,
                              "restart_safe": bool(job.get("restart_safe"))})
        for kind, count in (("managed", managed_turn_count(self)), ("api", self._active_api_run_count()),
                            ("deferred", self._active_deferred_agent_worker_count())):
            units.extend({"kind": kind, "pid": os.getpid()} for _ in range(count))
        return units

    async def _await_active_work_before_restart(self) -> bool:
        """Wait for in-flight work before ``stop()`` so the requesting turn isn't force-interrupted.

        Wedged turns are excluded (restart is their remedy). True when drained to zero, False when a
        bound elapsed or only wedged work remains (caller proceeds to ``stop()``).

        Two bounds, not one (PRA-6925): ``agent.restart_after_turn_timeout`` while cron has nothing
        waiting, and — from the moment a due slot is waiting — one cron inactivity watchdog later,
        because past that the wait is pure fleet-wide cron silence.
        """
        active = self._active_work_count()
        if active <= 0:
            return True
        if self._awaitable_work_count() <= 0:
            logger.warning(
                "Restart requested with %d active work unit(s), none awaitable "
                "(%d wedged past the inactivity timeout, %d in restart-safe external cron "
                "workers that outlive this process); skipping the after-turn wait and "
                "proceeding to stop()/drain", active,
                self._wedged_agent_count(), self._restart_safe_cron_count(),
            )
            return False
        timeout = float(getattr(self, "_restart_after_turn_timeout", 0.0) or 0.0)
        if timeout <= 0:
            logger.info(
                "Restart requested with %d active work unit(s); "
                "restart_after_turn_timeout=0 — entering stop()/drain immediately", active,
            )
            return False
        logger.info(
            "Restart requested with %d active work unit(s); "
            "deferring stop() until they finish (cap=%.0fs) so in-flight "
            "turns are not amputated (#77184)", active, timeout,
        )
        # Announce the cron cost at T=0 (PRA-6833): the wait closes admission for its whole
        # duration, so every ticked profile's due slots are held from this instant. Saying it here,
        # once, is what the 33-36 min holes were missing — the first per-profile WARNING a ticker
        # wrote named one home and one boundary, never the fleet-wide cost of the wait.
        logger.warning(
            "Restart deferral (cap %.0fs): %s", timeout, _cron_dispatch_hold_note(),
        )
        from gateway.drain_control import open_restart_wait
        # Admission stays closed for the whole wait: it closed at request time (``request_restart``)
        # and this wait only holds what was already in flight. Reopening it here would start turns
        # that stop() then amputates — 101 of them across two restarts on 2026-10-21 (PRX-5787).
        open_restart_wait(self)
        self._scale_to_zero_status("draining", "restart wait: status mark failed")
        loop = asyncio.get_running_loop()
        started = loop.time()
        cap_deadline = started + timeout
        # The cron bound (PRA-6925): the wait's cron cost is the due slots it leaves unissued, and
        # until a slot is actually waiting there is no cost to bound. So the cap above is the
        # bound only while cron has nothing waiting; the first scan that finds a due slot starts
        # the watchdog, and the wait then ends within it rather than holding that slot — and every
        # profile's admission — for the rest of the cap. Chosen over reopening cron dispatch
        # during the wait: a run started now is killed by the stop() this wait precedes (a
        # permanent failure in jobs.json), and on a host without systemd user scopes there is no
        # restart-safe worker to hand it to (cron/scheduler.py::_launch_external_cron_worker).
        hold_bound = _cron_hold_bound()
        hold_started: Optional[float] = None
        last_status_at = 0.0
        while self._awaitable_work_count() > 0:
            now = loop.time()
            deadline = cap_deadline
            if hold_started is not None and hold_bound is not None:
                deadline = min(cap_deadline, hold_started + hold_bound)
            if now >= deadline:
                if deadline < cap_deadline:
                    logger.warning(
                        "Restart deferral ends at the cron bound after %.0fs: a due cron slot has "
                        "been held %.0fs (bound %.0fs = HERMES_CRON_TIMEOUT) with %d work unit(s) "
                        "still active; proceeding to stop()/drain, which lets the held slot fire "
                        "as catch-up, instead of keeping admission closed for the remaining %.0fs "
                        "of the %.0fs cap",
                        now - started, now - hold_started, hold_bound,
                        self._active_work_count(), cap_deadline - now, timeout,
                    )
                else:
                    logger.warning(
                        "Restart after-turn wait timed out after %.0fs with %d "
                        "still active; proceeding to stop()/drain which may "
                        "interrupt remaining work (#77184)", timeout, self._active_work_count(),
                    )
                return False
            if (now - last_status_at) >= 30.0:
                cost = _cron_hold_cost()
                if hold_started is None and hold_bound is not None and cost and cost[0]:
                    hold_started = now
                    logger.warning(
                        "Restart deferral: %d due cron slot(s) across %d profile(s) are waiting, "
                        "so this wait now ends at most %.0fs later (cron bound) instead of the "
                        "%.0fs cap — a held slot is fleet-wide cron silence, and %d work unit(s) "
                        "have not finished in the meantime",
                        cost[0], cost[1], hold_bound, timeout, self._awaitable_work_count(),
                    )
                logger.info(
                    "Restart deferred: waiting on %d active work unit(s) "
                    "(%d wedged and excluded, %d restart-safe and excluded; "
                    "%.0fs remaining before force drain, admission closed); %s: %s",
                    self._awaitable_work_count(), self._wedged_agent_count(),
                    self._restart_safe_cron_count(), deadline - now,
                    _cron_dispatch_hold_note(cost), self._describe_active_work(),
                )
                self._scale_to_zero_status("draining", "restart wait: status mark failed")
                last_status_at = now
            await asyncio.sleep(0.1)
        if self._active_work_count() > 0:
            logger.warning(
                "Restart deferred wait: %d excluded work unit(s) remain "
                "(%d wedged, %d restart-safe external cron); proceeding to stop()/drain",
                self._active_work_count(), self._wedged_agent_count(),
                self._restart_safe_cron_count(),
            )
            return False
        logger.info("Restart deferred wait complete — active work drained; proceeding to stop()")
        return True
