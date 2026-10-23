"""The drain gate must never silence cron quietly, and must never outlive what it silences.

``can_dispatch`` (``gateway/run_bootstrap.py``) is False while the gateway drains in-flight work, and
a held tick leaves every due job's ``next_run_at`` intact so the next allowed tick catches up. That
part is right. What was missing is everything else, measured on 2026-10-21 (SOU-3252):

* the skip logged ``logger.debug("Cron dispatch paused while gateway drains existing work")`` — a
  ``*/10`` pager job went 68 min without firing and NOTHING on disk said why;
* the pause had no bound relative to the things it silences: a drain left open for an hour drops
  every occurrence of a ``*/10`` job outside the catch-up window (half the period, clamped
  120 s–2 h), while each downstream liveness detector reports the symptom — its own ``last_run_at``
  aging — not the cause.

So a held boundary now (a) WARNs once with the due job ids and the drain's own identity, (b) stamps
``last_fire_error`` on each held job, which is what ``hermes cron list`` and the liveness checks read,
and (c) logs an ERROR once when the drain has been open longer than the tightest served detector's
tolerance. The marker's own max-age bounds a LEAKED drain (900 s, gateway/drain_control.py); this
watchdog is for one that is legitimately kept alive — every second of it is pager silence, and past a
detector's tolerance the drain itself is the incident, not the jobs it holds.

Two corrections after the fleet-wide claim holes of 2026-10-22/23 (PRA-6830/PRA-6833), both of which
had to be fixed for (c) to be able to fire at all:

* **the episode clock is per profile home.** This process ticks every served profile, so one
  module-global episode slot was overwritten on every profile hop: each home's ``first_seen``
  reset once per cycle, "gateway drain opened" was re-WARNed every minute per home (visible in the
  2026-10-22T19:39Z logs), and the observable pause was capped at one tick interval — the ERROR
  was unreachable for EVERY drain class, not only the cap-bounded one.
* **the threshold is derived from the deferral cap.** A graceful restart's after-turn wait is
  capped by ``agent.restart_after_turn_timeout`` and the process EXITS at its cap, so a threshold
  above that cap can never be reached: the ERROR was dead code for exactly the class that froze
  the fleet twice (33–36 min deferrals, zero ERROR). See :func:`drain_alert_seconds`.

The due scan here is READ-ONLY on purpose: ``get_due_jobs()`` persists a ``last_dispatch`` stamp and
a ``pending_slot`` for everything it returns (cron/jobs.py::_evaluate_due_job), and a held slot must
not be recorded as dispatched.

One more correction after the 2026-10-23 cards (PRA-6925): the *cost* of the hold is now bounded,
not just reported. A restart deferral may keep a waiting slot for one cron inactivity watchdog
(:func:`hold_bound_seconds`) and no longer, so the ERROR above is a backstop for drain classes that
are not cap-bounded rather than the only trace of a deferral that never had to last that long.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from cron.scheduler_refused import stamp_fire_error
from hermes_constants import get_hermes_home

logger = logging.getLogger("cron.scheduler")

# Prefix of the stamped detail. Machine-readable on purpose: a liveness check greps for it rather
# than pattern-matching a sentence.
DRAIN_HOLD_REASON = "gateway_draining"

# The tightest served detector's tolerance: the lead sweep fires every 10 min and pages after 4
# missed passes, so a drain that outlives it is holding a page's worth of work.
DETECTOR_TOLERANCE_SECONDS = 2400.0

# ``agent.restart_after_turn_timeout``'s default (gateway/restart.py), used when the cap cannot be
# read. The cap is the lifetime of a restart deferral, and therefore the hard ceiling on how long
# this watchdog can ever observe a pause of that class.
DEFAULT_RESTART_DEFERRAL_CAP_SECONDS = 1800.0

# Headroom between the alert and the cap, so the alert is OBSERVABLE rather than merely reachable.
# The episode clock starts at the first held TICK, which trails the drain's T=0 by up to one ticker
# period (60 s), and the episode ends when the process exits — 300 s covers the tick offset (1 min)
# plus four more ticks at the 60 s cadence, i.e. the ERROR is written minutes before the cap even
# with heavy tick jitter. Capped at half a short cap: a deliberately tiny deferral budget shortens
# the silence it can cause in the same proportion.
DEFERRAL_CAP_ALERT_MARGIN_SECONDS = 300.0

# One episode per PROFILE HOME (the key is the home path). It carries the first held instant (the
# watchdog's clock: a keep-alive re-write moves the marker's ``requested_at`` forever, so the
# marker's own age can never measure how long dispatch has been paused), the due set already warned
# about (one WARNING per boundary), and whether the pageable ERROR has fired. Entries are dropped
# when dispatch resumes for that home; one that leaves the tick set mid-episode keeps a single small
# entry, bounded by the roster size.
_episodes: dict[str, dict[str, Any]] = {}


def note_dispatch_resumed(home: Optional[str] = None) -> None:
    """Close the current episode after a tick that was allowed to dispatch."""
    _episodes.pop(home or str(get_hermes_home()), None)


def _now() -> datetime:
    """Episode clock. A seam, so a test can age an episode without sleeping through it."""
    return datetime.now(UTC)


def _restart_deferral_cap() -> float:
    """The configured cap (seconds) on a restart's after-turn wait — the deferral's own lifetime.

    Behavioral read through the merged config, so a profile's ``agent.restart_after_turn_timeout``
    is what this profile's gateway honored. Unreadable/garbage config falls back to the default
    rather than removing the alert entirely.
    """
    try:
        from gateway.restart import parse_restart_after_turn_timeout
        from hermes_cli.config import cfg_get, load_config_readonly

        raw = cfg_get(load_config_readonly(), "agent", "restart_after_turn_timeout", default=None)
        return parse_restart_after_turn_timeout(raw)
    except (ImportError, OSError, ValueError, TypeError):
        return DEFAULT_RESTART_DEFERRAL_CAP_SECONDS


def drain_alert_seconds(restart_cap: Optional[float] = None) -> float:
    """Held-dispatch age at which the drain itself is the pageable incident — the smaller of two bounds.

    * :data:`DETECTOR_TOLERANCE_SECONDS` — past it the drain holds a page's worth of work, whatever
      keeps it open;
    * the deferral cap less :data:`DEFERRAL_CAP_ALERT_MARGIN_SECONDS` — a graceful restart's
      after-turn wait is capped and the process exits at the cap, so a restart deferral can never
      be observed older than the cap (PRA-6833: 2400 s against an 1800 s cap made this ERROR dead
      code for the class that froze the fleet twice).

    ``cap <= 0`` (``restart_after_turn_timeout: 0`` enters ``stop()`` at once) means no cap-bounded
    deferral class exists at all, so the detector's tolerance is the only bound left.
    """
    cap = _restart_deferral_cap() if restart_cap is None else restart_cap
    if cap is None or cap <= 0:
        return DETECTOR_TOLERANCE_SECONDS
    # A cap-bounded deferral is a KNOWN-bounded silence whose end is the process exiting, not an
    # idle drain: by `cap - margin` a */10 pager has missed two or three passes and the fleet's
    # cron is down for the rest of this process's life, so the lower bound is already an incident
    # rather than a "routine restart is minutes" false positive.
    margin = min(DEFERRAL_CAP_ALERT_MARGIN_SECONDS, cap / 2.0)
    return min(DETECTOR_TOLERANCE_SECONDS, cap - margin)


def hold_bound_seconds() -> Optional[float]:
    """How long a restart deferral may keep ONE due slot held, or ``None`` when there is no bound.

    The watchdog is the yardstick, not a new policy: ``HERMES_CRON_TIMEOUT``
    (``cron.scheduler._cron_inactivity_seconds``, default 600 s) is already the longest a healthy
    run may go quiet before the scheduler cuts it off, so a *slot* held past it has waited longer
    than the scheduler's own patience with a running job. Measured 2026-10-23 (PRA-6925): four
    deferrals in twelve hours, three of them holding 1-3 due slots across 1-3 profiles for the full
    1800 s cap, and every drain-done line recorded ``cron_at_start=0`` — the hold protected no
    in-flight cron work at all while it closed admission fleet-wide.

    ``0`` (unlimited) removes the yardstick, and an unreadable setting removes the bound rather
    than inventing one: returning ``None`` keeps the deferral exactly as it was, which is the
    fail-safe direction for a diagnostic that can only shorten a shutdown.
    """
    try:
        from cron.scheduler import _cron_inactivity_seconds

        bound = float(_cron_inactivity_seconds())
    except (ImportError, OSError, ValueError, TypeError):
        return None
    return bound if bound > 0 else None


def _drain_marker() -> Optional[dict[str, Any]]:
    """The ACTIVE drain marker for THIS profile home, or None (no marker, or none readable).

    Lazy import: ``gateway.drain_control`` pulls the gateway's config/psutil machinery, and the
    ticker imports this module on every boot. The marker is per ``HERMES_HOME``, and the ticker
    enters the served profile's scope before ticking, so this reads the same home the gateway
    quiesced — not the launch profile's.
    """
    try:
        from gateway.drain_control import active_drain

        return active_drain()
    except (ImportError, OSError, ValueError, TypeError) as exc:  # a diagnostic read never breaks the tick
        logger.warning("Cron drain hold: drain marker unreadable: %s", exc)
        return None


def _marker_age(marker: Optional[dict[str, Any]]) -> Optional[float]:
    """Seconds since the marker's ``requested_at`` (the drain's last keep-alive), or None."""
    if not marker:
        return None
    try:
        from gateway.drain_control import drain_age_seconds

        return drain_age_seconds(marker)
    except (ImportError, OSError, ValueError, TypeError):
        return None


def drain_detail(marker: Optional[dict[str, Any]]) -> str:
    """The stamped/logged description of the pause, naming its cause when one is readable."""
    detail = (f"{DRAIN_HOLD_REASON}: cron dispatch paused while the gateway drains in-flight work"
              " — the due slot is held, not consumed; it fires (as a catch-up) at the first tick "
              "after the drain releases")
    if not marker:
        # can_dispatch is process-wide, so a multiplex host pauses every served profile while the
        # marker lives in the DRAINING gateway's home. Say so instead of implying a cause we cannot see.
        return detail + "; no drain marker is readable for this profile"
    since = marker.get("requested_at") or "unknown"
    parts = [f"gateway.draining=true (since {since}, principal={marker.get('principal')}"]
    age = _marker_age(marker)
    if age is not None:
        parts.append(f", age {age / 60.0:.0f} min")
    parts.append(")")
    return detail + "; " + "".join(parts)


def _is_due(job: dict[str, Any], now: datetime) -> bool:
    """Whether *job* has a due slot at *now* — one rule for this home and for the whole fleet."""
    if not job.get("enabled", True) or job.get("paused_at") or job.get("state") == "completed":
        return False
    raw = job.get("next_run_at")
    if not raw:
        return False
    from cron.jobs import _ensure_aware

    try:
        due_at = _ensure_aware(datetime.fromisoformat(str(raw)))
    except (TypeError, ValueError):
        return False
    return due_at <= now


def due_job_ids_now() -> list[str]:
    """Ids of enabled jobs whose stored ``next_run_at`` has passed — a read-only scan.

    Deliberately NOT ``get_due_jobs()``: that one persists dispatch marks for what it returns, and
    the tick dispatches nothing. The list is for naming what is being held.
    """
    from cron.jobs import _hermes_now, load_jobs

    now = _hermes_now()
    return [str(job["id"]) for job in load_jobs() if _is_due(job, now)]


def due_slot_summary(homes: Iterable[os.PathLike | str]) -> tuple[int, int]:
    """``(due slots, homes holding them)`` across every one of *homes*, read straight off disk.

    Fleet-wide on purpose: admission is closed process-wide, so a restart deferral holds every
    ticked profile's cron, and the cost of the wait has to be nameable in ONE line on the gateway's
    side (PRA-6833) rather than only per-profile in the ticker's own WARNINGs. ``load_jobs()`` reads
    the ACTIVE home's store, so this reads each ``cron/jobs.json`` directly — no store open, no
    scope switch, no lock, and nothing persisted. An unreadable or half-written store is skipped:
    this is a diagnostic taken during a shutdown and must never abort the deferral it describes.
    """
    from cron.jobs import _hermes_now

    now = _hermes_now()
    due_total = 0
    holding = 0
    for home in homes:
        try:
            with open(Path(home) / "cron" / "jobs.json", encoding="utf-8") as fh:
                stored = json.load(fh)
        except (OSError, ValueError, TypeError):
            continue
        jobs = stored.get("jobs") if isinstance(stored, dict) else stored
        if not isinstance(jobs, list):
            continue
        due = sum(1 for job in jobs if isinstance(job, dict) and _is_due(job, now))
        if due:
            due_total += due
            holding += 1
    return due_total, holding


def hold_for_drain() -> int:
    """Declare the held tick: stamp the due jobs, warn once per boundary, alert when the drain drags.

    Returns 0 — the tick executed nothing — so a caller can ``return hold_for_drain()`` from its gate.
    Best-effort by design: this is the diagnostic path for a tick that is already doing no work, and
    an unwritable store here must not be booked as a failed tick (that would arm the EMFILE backoff
    and mislabel a healthy ticker as failing).
    """
    home_tag = str(get_hermes_home())
    now = _now()
    episode = _episodes.get(home_tag)
    if episode is None:
        episode = _episodes[home_tag] = {"first_seen": now, "held": (), "alerted": False}
        logger.warning("Cron dispatch held for %s: gateway drain opened", home_tag)
    marker = _drain_marker()
    detail = drain_detail(marker)

    try:
        held = due_job_ids_now()
    except (OSError, ValueError, TypeError) as exc:  # store unreadable: the pause still gets said out loud
        logger.warning("Cron dispatch held (%s); due set unreadable: %s", detail, exc)
        return 0

    if not held:
        # Nothing was due, so nothing was silenced at this boundary (the episode keeps its clock).
        logger.debug("Cron dispatch held (%s); no job was due at this boundary", detail)
        return 0

    boundary = tuple(sorted(held))
    if episode["held"] != boundary:
        episode["held"] = boundary
        logger.warning(
            "Cron dispatch held %d due job(s) (%s): %s",
            len(held), ", ".join(boundary), detail,
        )
    try:
        stamp_fire_error(held, detail)
    except (OSError, ValueError, TypeError) as exc:  # the WARNING above already names the cause
        logger.warning("Cron dispatch held (%s); could not stamp last_fire_error: %s", detail, exc)

    paused_for = (now - episode["first_seen"]).total_seconds()
    cap = _restart_deferral_cap()
    limit = drain_alert_seconds(cap)
    if paused_for >= limit and not episode["alerted"]:
        episode["alerted"] = True
        logger.error(
            "Cron dispatch has been paused for %.0f min by a gateway drain (limit %.0f min — the "
            "shorter of the %.0f min detector tolerance and a deferral cap of %s) — the drain "
            "itself is the incident, not the %d job(s) it holds for %s (%s). %s",
            paused_for / 60.0, limit / 60.0, DETECTOR_TOLERANCE_SECONDS / 60.0,
            f"{cap / 60.0:.0f} min" if cap > 0 else "0 (no after-turn wait)",
            len(held), home_tag, ", ".join(boundary), detail,
        )
    return 0
