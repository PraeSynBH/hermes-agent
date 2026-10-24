"""A restart deferral's cron cost as a machine-readable record, written at T=0.

The deferral announces itself in prose (``Restart deferral (cap 1800s): cron dispatch is held
fleet-wide until the drain releases: N due cron slot(s) across M profile(s)``) and, at T+26 of a
T+30 hold, in ``errors.log``. Both are for a human reading a log. The surface control that has to
SUBTRACT the hold from its measurement gap (PRX-6044's trip-wire, CEO ruling R2 on PRX-6109) had
only that prose to parse, and the cap-bounded escalation arrived four minutes before the hold ended
by itself — a postmortem, not an alert (CEO ruling rev 2, R7: PRX-6133).

So each deferral appends one JSON object per phase to ``<launch home>/logs/restart-holds.jsonl``:

    {"schema": 1, "event": "dispatch_hold_begin", "hold_id": "...", "started_at": "...Z",
     "cap_s": 1800.0, "predicted_end_at": "...Z", "due_slots": 3, "profiles": 2,
     "cron_cost_readable": true, "pid": 49093, "requester": {...}}
    {"schema": 1, "event": "dispatch_hold_end", "hold_id": "...", "started_at": "...",
     "ended_at": "...", "held_s": 1234.5, "reason": "drained|cap|cron_bound|aborted",
     "cap_s": 1800.0, "due_slots": 3, "profiles": 2}

The four fields the ruling requires (``started_at``, ``cap_s``, ``due_slots``, ``profiles``) are on
BOTH records, and a field the process could not read is ``null`` with ``cron_cost_readable: false``
rather than absent — a consumer keys on the schema, not on which shutdown path wrote the line. The
end record's ``due_slots``/``profiles`` are the LAST scan the wait took (the T=0 one when it took no
other), which is what the hold actually cost by the time it ended.

Why the begin/end pair rather than one line: the window the control subtracts is the hold's
LIFETIME, and it is not knowable at T=0. The cap is the ceiling (the process exits at it), the
cron bound (``cron.scheduler_drain_gate.hold_bound_seconds``) can end it sooner once a due slot is
waiting, and a SIGTERM ends it sooner still — a second instance of the same shape as the 21.5 min
hold on 2026-10-23. Guessing the width from the cap would over-attribute whenever a hold ended
early, which is the fail-open direction for a control whose job is to notice a real stall.

Ownership and durability:

* the path is the LAUNCH home (``gateway.restart_attribution.process_home``), the same home
  ``gateway.log`` lives in — a per-session ``HERMES_HOME`` override would hide the record from the
  host control, which is exactly the hole PRX-5423 closed for restart attribution;
* one line per event, appended with a single ``write`` while the file is open for append; the
  lines are ~300 bytes, far below ``PIPE_BUF``, so concurrent writers cannot interleave halves;
* best-effort and never raising: this runs inside a shutdown that exists to protect in-flight
  work, and forensics must never abort it.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("gateway.run")

LEDGER_NAME = "restart-holds.jsonl"
SCHEMA_VERSION = 1


def _launch_home() -> Path:
    """The launch profile's home: where the host control reads, and where ``gateway.log`` lives."""
    try:
        from gateway.restart_attribution import process_home

        return Path(process_home())
    except Exception:  # health: allow BLE001 -- a forensic path must not raise inside a shutdown
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())


def hold_ledger_path(home: Optional[Path] = None) -> Path:
    """``<home>/logs/restart-holds.jsonl`` (launch home when *home* is omitted)."""
    return (Path(home) if home is not None else _launch_home()) / "logs" / LEDGER_NAME


def _utc_iso(ts: Optional[float] = None) -> str:
    moment = datetime.fromtimestamp(ts, UTC) if ts is not None else datetime.now(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _append(record: dict[str, Any], home: Optional[Path]) -> bool:
    """Append one JSONL line. Never raises; True when the line landed."""
    try:
        path = hold_ledger_path(home)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Restart deferral: could not append the cron hold record (%s): %s", record.get("event"), exc)
        return False


def _requester() -> Optional[dict[str, Any]]:
    """Who asked for this restart, from the lifecycle sentinel stamped at request time.

    ``gateway.lifecycle_ledger.stamp_restart_request`` writes it the moment the SIGUSR1 lands (a
    SIGUSR1 carries no payload), so by the deferral it is the only record of the author — and the
    field R8 discipline is graded on. Best-effort: absent means the request was unattributed.
    """
    try:
        from gateway.lifecycle_ledger import _read_json, get_lifecycle_sentinel_path

        sentinel = _read_json(get_lifecycle_sentinel_path()) or {}
        request = sentinel.get("restart_request")
        return request if isinstance(request, dict) else None
    except Exception:  # health: allow BLE001 -- identity is enrichment, never a precondition
        return None


def record_hold_begin(
    *,
    cap_s: float,
    started_at: Optional[float] = None,
    due_slots: Optional[int] = None,
    profiles: Optional[int] = None,
    active_units: Optional[int] = None,
    home: Optional[Path] = None,
) -> Optional[str]:
    """Append the T=0 bound event for a deferral that holds fleet-wide dispatch; return its id.

    ``due_slots`` / ``profiles`` are the T=0 scan of every ticked profile's store
    (``cron.scheduler_drain_gate.due_slot_summary``); ``None`` means the scan could not be read,
    which is reported rather than papered over with a zero — zero is a fact ("nothing was due yet")
    and a consumer subtracting a window needs to tell the two apart.
    """
    started = started_at if started_at is not None else datetime.now(UTC).timestamp()
    readable = due_slots is not None and profiles is not None
    record: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "event": "dispatch_hold_begin",
        "hold_id": uuid.uuid4().hex,
        "started_at": _utc_iso(started),
        "cap_s": float(cap_s),
        "predicted_end_at": _utc_iso(started + float(cap_s)),
        "due_slots": due_slots,
        "profiles": profiles,
        "cron_cost_readable": readable,
        "pid": os.getpid(),
    }
    if active_units is not None:
        record["active_units"] = active_units
    requester = _requester()
    if requester:
        record["requester"] = requester
    return str(record["hold_id"]) if _append(record, home) else None


def record_hold_end(
    *,
    hold_id: Optional[str],
    started_at: float,
    cap_s: float,
    reason: str,
    due_slots: Optional[int] = None,
    profiles: Optional[int] = None,
    ended_at: Optional[float] = None,
    home: Optional[Path] = None,
) -> bool:
    """Close the hold with its real width, so a consumer need not assume the cap was spent.

    ``reason`` is one of ``drained`` (the awaitable work finished), ``cap`` (the after-turn timeout
    elapsed), ``cron_bound`` (a due slot was waiting, so the wait ended one cron inactivity
    watchdog later — ``cron.scheduler_drain_gate.hold_bound_seconds``) or ``aborted`` (the wait was
    cancelled or raised, so it did not end by any of its own bounds). A ``None`` *hold_id* means the
    begin record could not be written, so there is nothing to close.
    """
    if not hold_id:
        return False
    ended = ended_at if ended_at is not None else datetime.now(UTC).timestamp()
    record: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "event": "dispatch_hold_end",
        "hold_id": str(hold_id),
        "started_at": _utc_iso(started_at),
        "ended_at": _utc_iso(ended),
        "held_s": round(max(0.0, ended - started_at), 3),
        "reason": reason,
        "cap_s": float(cap_s),
        "due_slots": due_slots,
        "profiles": profiles,
        "cron_cost_readable": due_slots is not None and profiles is not None,
    }
    return _append(record, home)
