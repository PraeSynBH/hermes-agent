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

The due scan here is READ-ONLY on purpose: ``get_due_jobs()`` persists a ``last_dispatch`` stamp and
a ``pending_slot`` for everything it returns (cron/jobs.py::_evaluate_due_job), and a held slot must
not be recorded as dispatched.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Optional

from cron.scheduler_refused import stamp_fire_error

logger = logging.getLogger("cron.scheduler")

# Prefix of the stamped detail. Machine-readable on purpose: a liveness check greps for it rather
# than pattern-matching a sentence.
DRAIN_HOLD_REASON = "gateway_draining"

# A drain open this long is itself the pageable condition. The tightest served detector tolerates
# 40 min of silence (the lead sweep fires every 10 min and pages after 4 missed passes), so a drain
# that outlives it is holding a page's worth of work. Deliberately the detector tolerance and not
# lower: an ordinary update/restart drain is minutes and must never page.
DRAIN_ALERT_SECONDS = 2400.0

# One episode = one uninterrupted stretch of held ticks in this home. It carries the first held
# instant (the watchdog's clock: a keep-alive re-write moves the marker's ``requested_at`` forever,
# so the marker's own age can never measure how long dispatch has been paused), the due set already
# warned about (one WARNING per boundary), and whether the pageable ERROR has fired.
_episode: Optional[dict[str, Any]] = None


def note_dispatch_resumed() -> None:
    """Close the current episode after a tick that was allowed to dispatch."""
    global _episode
    _episode = None


def _now() -> datetime:
    """Episode clock. A seam, so a test can age an episode without sleeping through it."""
    return datetime.now(UTC)


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


def due_job_ids_now() -> list[str]:
    """Ids of enabled jobs whose stored ``next_run_at`` has passed — a read-only scan.

    Deliberately NOT ``get_due_jobs()``: that one persists dispatch marks for what it returns, and
    the tick this gate holds dispatches nothing. The list is for naming what is being held.
    """
    from cron.jobs import _ensure_aware, _hermes_now, load_jobs

    now = _hermes_now()
    held: list[str] = []
    for job in load_jobs():
        if not job.get("enabled", True) or job.get("paused_at") or job.get("state") == "completed":
            continue
        raw = job.get("next_run_at")
        if not raw:
            continue
        try:
            due_at = _ensure_aware(datetime.fromisoformat(str(raw)))
        except (TypeError, ValueError):
            continue
        if due_at <= now:
            held.append(str(job["id"]))
    return held


def hold_for_drain() -> int:
    """Declare the held tick: stamp the due jobs, warn once per boundary, alert when the drain drags.

    Returns 0 — the tick executed nothing — so a caller can ``return hold_for_drain()`` from its gate.
    Best-effort by design: this is the diagnostic path for a tick that is already doing no work, and
    an unwritable store here must not be booked as a failed tick (that would arm the EMFILE backoff
    and mislabel a healthy ticker as failing).
    """
    global _episode
    from hermes_constants import get_hermes_home

    home_tag = str(get_hermes_home())
    now = _now()
    if _episode is None or _episode["home"] != home_tag:
        _episode = {"home": home_tag, "first_seen": now, "held": (), "alerted": False}
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
    if _episode["held"] != boundary:
        _episode["held"] = boundary
        logger.warning(
            "Cron dispatch held %d due job(s) (%s): %s",
            len(held), ", ".join(boundary), detail,
        )
    try:
        stamp_fire_error(held, detail)
    except (OSError, ValueError, TypeError) as exc:  # the WARNING above already names the cause
        logger.warning("Cron dispatch held (%s); could not stamp last_fire_error: %s", detail, exc)

    paused_for = (now - _episode["first_seen"]).total_seconds()
    if paused_for >= DRAIN_ALERT_SECONDS and not _episode["alerted"]:
        _episode["alerted"] = True
        logger.error(
            "Cron dispatch has been paused for %.0f min by a gateway drain (limit %.0f min) — the "
            "drain itself is the incident, not the %d job(s) it holds (%s). %s",
            paused_for / 60.0, DRAIN_ALERT_SECONDS / 60.0, len(held), ", ".join(boundary), detail,
        )
    return 0
