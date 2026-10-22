"""External drain-control marker contract (dashboard → gateway).

No control channel exists into a running gateway, so begin/cancel-drain writes
(or removes) ``{HERMES_HOME}/.drain_request.json`` and a gateway watcher reacts;
an ACTIVE marker means ``gateway_state -> "draining"``.

Four lenient staleness signals (any one suffices): epoch mismatch (HERMES_HOME is a durable
volume on Hermes Cloud, so a marker survives the restart a drain-gated action ends in and would
park the fresh gateway in ``draining`` forever), expiry (any orphan past its own lease — the
``max_age_seconds`` field, clamped to :data:`DRAIN_REQUEST_MAX_AGE_SECONDS` and defaulting to it
when absent; re-writing refreshes it — PRX-5819), a dead writer (the marker is a
file, so the process that wrote it can be SIGKILLed before any cleanup runs — SOU-3249), and an
untrusted principal (a drain silences a whole fleet, so only the authenticated control seams may
ask for one).

Reading never raises: a malformed file reads as ``{}`` — still drain-active (fail-safe toward
quiescing).  Staleness rejects only on a *definite* verdict.
"""
from __future__ import annotations

import contextlib
import functools
import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Optional

from gateway.memory_status import _parse_iso
from hermes_constants import get_hermes_home
from utils import atomic_json_write

_log = logging.getLogger(__name__)

# ``drain_reason`` a graceful restart publishes for its after-turn wait (#139041). The owner still
# runs its in-flight turns, so a client may attach to answer a parked approval/clarify; new input is
# committed queued for the next owner and never claimed here. ``stop()`` withdraws it.
RESTART_WAIT_REASON = "restart_wait"


# Admission closes at T=0 of every drain, and a restart's after-turn wait is not an exception. The
# wait can hold the process for ``restart_after_turn_timeout`` (1800 s by default) and it ends with
# ``stop()``, which does NOT wait for anything admitted during it: measured 2026-10-21, two restarts
# (21:37, 23:05 PDT) started 101 runs inside their waits and killed every one of them
# (``error='runtime_draining'``, mean lifetime 7.2 s and 4.6 s). A run started into a drain is lost
# work, and it is lost *after* the client paid a spawn and a boot — strictly worse than the same
# refusal at T=0, which every client already handles as the retryable "restarting" verdict. So the
# wait admits nothing it did not already own: it only KEEPS what was in flight at T=0 (queue_only
# commits still land, #139041), and it says so once at the top instead of every 30 s (PRX-5787).


def open_restart_wait(runner) -> None:
    """Publish the restart's after-turn wait (``drain_reason``) unless a drain already named its own.

    The reason is what lets an interactive client attach to the owner to answer a turn parked on
    approval/clarify (#139041); it never reopens admission, which the drain closed at T=0.
    """
    descriptor = getattr(runner, "session_runtime_descriptor", None)
    if descriptor is not None and not descriptor.get("drain_reason"):
        descriptor["drain_reason"] = RESTART_WAIT_REASON


def restart_wait_open(runner) -> bool:
    """True while *runner* is in a restart's after-turn wait (draining, not yet stopping)."""
    descriptor = getattr(runner, "session_runtime_descriptor", None) or {}
    return (bool(getattr(runner, "_draining", False)) and bool(getattr(runner, "_running", False))
            and descriptor.get("drain_reason") == RESTART_WAIT_REASON)


# The reads a client needs during that wait to find and open the conversation parked on
# approval/clarify: Desktop's session picker, the session detail and its transcript pages, and the
# health probes. GET only and still authenticated; every write and every other route keeps the
# drain's refusal. The prompt itself (approval.pending / *.respond) rides the WebSocket.
_RESTART_WAIT_READS = re.compile(
    r"/api/(?:health|status|profiles/sessions(?:/sidebar)?"
    r"|sessions/[^/]+(?:/messages(?:/around)?|/timeline)?)")


def restart_wait_read(runner, method, path) -> bool:
    """True for an allowlisted read request while *runner* is in a restart's after-turn wait."""
    return (method == "GET" and bool(_RESTART_WAIT_READS.fullmatch(path or ""))
            and restart_wait_open(runner))

# The WebSocket verbs an interactive client may still call during that wait: attach to and read the
# parked session, answer its prompt (or interrupt the turn so the restart can finish) and commit
# queued input for the next owner. Every other verb (config.set, session.save, rollback.restore,
# process.kill, reload.*, ...) would run on an owner that is shutting down, so it is refused with
# ``runtime_draining`` exactly like the REST writes above.
RESTART_WAIT_WS_METHODS = frozenset({
    "ping", "runtime.describe", "session.resume", "session.detach", "session.events.since",
    "session.info", "session.list", "prompt.submit", "prompt.receipt", "session.interrupt",
    "approval.pending", "approval.received", "approval.respond", "clarify.respond", "clarify.lock",
})


def ws_method_refused_while_draining(runner, method) -> bool:
    """True when an interactive WebSocket verb must be refused because *runner* is draining."""
    return bool(getattr(runner, "_draining", False)) and method not in RESTART_WAIT_WS_METHODS

_DRAIN_REQUEST_FILENAME = ".drain_request.json"

# The only principals a LIVE gateway honours. Both are set by an authenticated seam, never by the
# caller: ``drain-control`` is the bearer token principal of plugins/dashboard_auth/drain, and
# ``dashboard`` is the cookie-gated dashboard control (hermes_cli/web_routers/actions.py). The
# marker is a plain file, so a script or acceptance harness can write anything it likes — measured
# 2026-10-21, a leaked harness marker carrying ``principal: "prx-5802 acceptance v5"`` refused
# admission on the live fleet for 60 min and paged CanaryDeadmanDown (SOU-3249). An unknown
# principal is therefore refused; a marker with NO principal at all (legacy or corrupt body) keeps
# the old leniency, because a half-written file must never un-quiesce a fleet mid-update.
TRUSTED_DRAIN_PRINCIPALS = frozenset({"drain-control", "dashboard"})

# Blast radius of a leaked marker, in seconds — the DEFAULT lease and the CEILING for a marker
# that names its own. A drain-gated lifecycle action completes in minutes, and the off-host canary
# pages CanaryDeadmanDown after 45 min of fleet silence, so this bound must stay strictly under
# that threshold — 3600 s made a single leaked test marker a guaranteed page (SOU-3249). A
# deliberate drain longer than this keeps the marker alive by re-writing it —
# write_drain_request() is idempotent and bumps ``requested_at``; a marker nobody re-writes expires
# here. A writer that knows its own lifetime should name a SHORTER lease in the body
# (``max_age_seconds``, PRX-5819): measured 2026-10-21, an acceptance harness died with the marker
# engaged and held the host for the whole constant — 17 refused spawns and every ``agent``-profile
# cron silent for 64 min.
DRAIN_REQUEST_MAX_AGE_SECONDS = 900.0
# Marker field naming that per-marker lease. Absent, or not a finite positive number, means the
# constant above — byte-for-byte today's behaviour for every existing writer.
DRAIN_LEASE_FIELD = "max_age_seconds"
# Dedup for the expired-marker warning (the watcher re-reads every second); keyed by
# ``requested_at`` so a keep-alive re-write that later expires logs again.
_expiry_logged_for: Optional[str] = None
# Same dedup for the other two staleness verdicts, keyed the same way.
_writer_dead_logged_for: Optional[str] = None
_untrusted_logged_for: Optional[str] = None


@functools.lru_cache(maxsize=1)
def current_instantiation_epoch() -> str:
    """Identity of THIS container / VM / boot instantiation.

    Linux: ``"<boot_id>:<pid1_start>"`` — stable for the life of PID 1 (a gateway-only respawn
    keeps honouring an in-flight drain) but changes when the machine is recreated: boot_id on a VM
    reboot, PID 1's start time on ``docker restart``.  macOS: ``"darwin-boot:<kern.boottime>"``,
    the same restart-surviving-stable stamp (psutil reads ``kern.boottime``): a marker written
    before a reboot is rejected exactly as it is under Docker, which the ``""`` this used to return
    off Linux disabled on the only host we run acceptance harnesses on (SOU-3249).

    ``""`` when the identity is unreadable (no ``/proc`` and no psutil, or Windows) — the epoch
    check is then disabled, never fail-closed.  Note that a *gateway* restart deliberately does not
    change this: the operator's drain must outlive the gateway process it quiesced, or a
    supervisor respawn mid-update reopens admission just before the update kills what it admitted
    (the PRX-5787 class).  A dead *writer* is caught by writer liveness instead, not the epoch.
    """
    boot_id = pid1_start = ""
    with contextlib.suppress(OSError):
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    with contextlib.suppress(OSError, IndexError):
        # "<pid> (<comm>) <state> ...": comm may contain spaces/parens, so split on the
        # LAST ')'. starttime is field 22 (1-indexed) = tail index 19.
        pid1_start = Path("/proc/1/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19]
    if boot_id or pid1_start:
        return f"{boot_id}:{pid1_start}"
    if sys.platform == "darwin":
        with contextlib.suppress(Exception):
            import psutil  # type: ignore

            # kern.boottime (integration seconds): identical in every process on this boot, and a
            # reboot moves it. Not process_birth()'s clock-step-stable clock — this one must be
            # shared across processes, and int() keeps a later read byte-identical.
            return f"darwin-boot:{int(psutil.boot_time())}"
    return ""


def drain_request_path(home: Optional[Path] = None) -> Path:
    """Absolute path to the drain-request marker, respecting HERMES_HOME."""
    return Path(home if home is not None else get_hermes_home()) / _DRAIN_REQUEST_FILENAME


def _process_birth() -> Optional[float]:
    """This process's restart-stable birth second, or ``None`` when this host cannot say.

    ``hermes_cli.process_identity.process_birth`` is the canonical read (on macOS the kernel's
    unadjusted start, so a clock step does not move it).  Only compared against a later read of the
    same pid on this host, so the unit never crosses a machine.
    """
    try:
        import psutil  # type: ignore
        from hermes_cli.process_identity import process_birth

        return round(process_birth(psutil.Process()), 3)
    except Exception:
        return None


def _clamp_lease(raw: Any) -> float:
    """The lease a marker names, clamped to :data:`DRAIN_REQUEST_MAX_AGE_SECONDS`.

    Unusable input (absent, bool, NaN/inf, non-numeric, <= 0) yields the constant, never "expire
    now" and never "no bound": the lease is a blast-radius bound, so a corrupt one must fall back
    to today's behaviour rather than un-quiesce a fleet mid-update. Above the constant clamps DOWN,
    so a marker can ask to be forgotten sooner but never to outlive a legitimate drain's bound.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return DRAIN_REQUEST_MAX_AGE_SECONDS
    lease = float(raw)
    if not math.isfinite(lease) or lease <= 0:
        return DRAIN_REQUEST_MAX_AGE_SECONDS
    return min(lease, DRAIN_REQUEST_MAX_AGE_SECONDS)


def write_drain_request(
    *, principal: str = "drain-control", suppress_notification: bool = False,
    test_only: bool = False, home: Optional[Path] = None,
    max_age_seconds: Optional[float] = None,
) -> dict[str, Any]:
    """Write the begin-drain marker atomically; returns the payload.

    Re-writing refreshes ``requested_at`` (keep-alive past the max-age) AND the writer identity, so
    a long drain stays active by re-writing.

    ``max_age_seconds`` is this marker's own lease (PRX-5819): a writer that knows it will be gone
    in a minute says so instead of leaving the fleet quiesced for the default 900 s, and a writer
    that dies is then harmless — no cancel, no watchdog. Written clamped to
    :data:`DRAIN_REQUEST_MAX_AGE_SECONDS`, so a marker can never outlive the constant; omitted
    (the default) writes no field at all, which reads exactly as it always has.

    ``suppress_notification`` skips ONLY the home-channel "gateway shutting down" broadcast (the
    per-session interrupt ping is never suppressed).  It is refused (written as False) for a
    ``test_only`` / untrusted principal: silencing the one human-visible signal is a privilege of
    the authenticated control seams, never of a harness (SOU-3249).

    ``test_only`` marks a marker only a test gateway should see.  A marker from an untrusted
    principal is already inert on a live gateway and is marked the same way, with a warning at write
    time, so an out-of-tree harness finds out here rather than by paging the fleet.

    Stamped with the instantiation epoch (a copy surviving a machine restart on the durable volume
    reads as stale) and with the writing process's pid/birth (a writer that dies without cancelling
    — including SIGKILL, which runs no ``finally`` — is ignored within one watcher tick). So a drain
    that must outlive its writer needs a live process re-writing it, exactly like the max-age
    keep-alive; a one-shot writer's drain ends when it does.
    """
    trusted = not test_only and principal in TRUSTED_DRAIN_PRINCIPALS
    payload = {
        "action": "drain", "requested_at": datetime.now(UTC).isoformat(), "principal": principal,
        "epoch": current_instantiation_epoch(), "suppress_notification": bool(suppress_notification) and trusted,
        "writer_pid": os.getpid(), "writer_birth": _process_birth(),
    }
    if max_age_seconds is not None:
        payload[DRAIN_LEASE_FIELD] = _clamp_lease(max_age_seconds)
    if not trusted:
        payload["test_only"] = True
        _log.warning(
            "drain-control: marker written by untrusted principal %r is INERT on a live gateway "
            "(trusted: %s); it is kept for test gateways only. suppress_notification=%s.",
            principal, ", ".join(sorted(TRUSTED_DRAIN_PRINCIPALS)), bool(suppress_notification),
        )
    atomic_json_write(drain_request_path(home), payload)
    return payload


def clear_drain_request(*, home: Optional[Path] = None, require_contract_in_sync: bool = True) -> bool:
    """Remove the drain marker (cancel-drain, idempotent). Returns True if one existed.

    Clearing the marker is the deploy's completion claim: the fleet is about to admit turns again.
    A wire-contract change served by a process that booted before it is the SOU-3316 shape, so the
    marker stays (the fleet does not re-open on the old contract), the refusal is an ERROR line
    naming both revisions, and the marker's own lease bounds the damage. An ``unknown`` verdict
    (no sentinel / no contract document) never blocks — this gate catches a proven mismatch only.
    """
    if require_contract_in_sync:
        from gateway.lifecycle_ledger import contract_revision_verdict

        verdict, detail = contract_revision_verdict(home)
        if verdict == "stale":
            _log.error(
                "drain-control: REFUSING to clear the drain marker — the deploy is not complete: %s. "
                "Restart the gateway on the new tree, then cancel the drain.", detail,
            )
            return False
    path = drain_request_path(home)
    try:
        path.unlink()
        return True
    except OSError as e:
        if not isinstance(e, FileNotFoundError):
            _log.warning("drain-control: failed to remove %s: %s", path, e)
        return False


def _marker_is_expired(body: dict[str, Any]) -> bool:
    """True iff ``requested_at`` parses AND is older than this marker's lease.

    The lease is the marker's own ``max_age_seconds`` clamped to
    :data:`DRAIN_REQUEST_MAX_AGE_SECONDS`, or that constant when the field is absent/unusable
    (PRX-5819). Missing/unparseable and future-dated (clock skew) timestamps are honoured.
    Logged once per marker, not per poll — the operator's breadcrumb for a leak.

    See #85433.
    """
    global _expiry_logged_for
    raw = body.get("requested_at")
    lease = _clamp_lease(body.get(DRAIN_LEASE_FIELD))
    age = drain_age_seconds(body)
    if age is None or age <= lease:
        return False
    if _expiry_logged_for != raw:
        _expiry_logged_for = raw
        _log.warning(
            "drain-control: ignoring expired drain marker (requested_at=%s, age=%.0fs > max %.0fs, principal=%s) "
            "— the drain that wrote it was never cancelled; treating as stale so the gateway keeps accepting turns.",
            raw, age, lease, body.get("principal"),
        )
    return True


def drain_age_seconds(body: dict[str, Any]) -> Optional[float]:
    """Seconds since the marker's ``requested_at``, or ``None`` when it carries no parseable clock.

    The one clock reader for the marker, so a consumer that reports "draining since … for N min"
    (the cron drain gate, SOU-3252) cannot drift from the :func:`_marker_is_expired` verdict, which
    compares this same age against the marker's own lease. A tz-naive or future-dated stamp still
    reads: ``_parse_iso`` treats a missing zone as UTC.
    """
    requested_at = _parse_iso(body.get("requested_at"))
    if requested_at is None:
        return None
    return (datetime.now(UTC) - requested_at).total_seconds()


def _marker_epoch_is_stale(body: dict[str, Any]) -> bool:
    """True iff BOTH sides name an instantiation epoch and they differ.

    An empty epoch on either side is unprovable, never stale: legacy markers written before the
    field existed, and hosts where :func:`current_instantiation_epoch` cannot answer, keep the
    leniency the max-age/writer/principal checks provide.
    """
    current, marker_epoch = current_instantiation_epoch(), body.get("epoch")
    return bool(current and marker_epoch and marker_epoch != current)


def _writer_still_alive(pid: Any, birth: Any) -> Optional[bool]:
    """True/False when provable, ``None`` when this host cannot say.

    Same shape as ``hermes_cli.process_identity._pid_alive_matches``: a 2 s birth tolerance (a
    recycled PID is never that close to the original's birth) and a zombie counted as dead, because
    a zombie has already run whatever cleanup it was going to run.  ``None`` on any unreadable
    probe — including another user's process — so an unprovable writer never cancels a real drain;
    the max-age bound covers that case instead.
    """
    if isinstance(pid, bool) or not isinstance(pid, int) or not isinstance(birth, (int, float)):
        return None
    try:
        import psutil  # type: ignore
        from hermes_cli.process_identity import process_birth
    except Exception:
        return None
    try:
        proc = psutil.Process(pid)
        return (abs(process_birth(proc) - float(birth)) < 2.0
                and proc.status() != getattr(psutil, "STATUS_ZOMBIE", "zombie"))
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return None


def _marker_writer_is_dead(body: dict[str, Any]) -> bool:
    """True iff the marker names a writer that is DEFINITELY gone.

    The cancel-on-exit the writer cannot perform itself: a harness killed with SIGKILL runs no
    ``finally`` and no ``atexit``, but the marker it left names its pid and birth, so the next
    watcher tick (<=1 s) concludes the drain is over instead of refusing admission until the
    max-age (SOU-3249: 60 min of fleet silence, 9 cron jobs 2885-3785 s late).

    A marker with no writer identity (hand-written, or written by an older version) gets no verdict
    here and keeps the max-age bound.
    """
    global _writer_dead_logged_for
    if _writer_still_alive(body.get("writer_pid"), body.get("writer_birth")) is not False:
        return False
    raw = body.get("requested_at")
    if _writer_dead_logged_for != raw:
        _writer_dead_logged_for = raw
        _log.warning(
            "drain-control: ignoring drain marker whose writer is gone (writer_pid=%s, "
            "requested_at=%s, principal=%s) — it exited without cancelling the drain; treating as "
            "stale so the gateway keeps accepting turns.",
            body.get("writer_pid"), raw, body.get("principal"),
        )
    return True


def _marker_principal_refused(body: dict[str, Any]) -> bool:
    """True when the marker is not from a principal the live gateway honours.

    Two ways in: an explicit ``test_only`` flag, and any principal outside
    :data:`TRUSTED_DRAIN_PRINCIPALS`.  A body with no principal at all is left to the other
    staleness checks (a corrupt marker must not un-quiesce the fleet).
    """
    global _untrusted_logged_for
    principal = body.get("principal")
    if principal is None or (isinstance(principal, str) and not principal.strip()):
        return False
    why = "test_only marker" if body.get("test_only") else f"untrusted principal {principal!r}"
    if body.get("test_only") or principal not in TRUSTED_DRAIN_PRINCIPALS:
        key = f"{body.get('requested_at')}|{principal}"
        if _untrusted_logged_for != key:
            _untrusted_logged_for = key
            _log.warning(
                "drain-control: refusing drain marker from %s (%s) — a drain refuses admission "
                "fleet-wide, so only the authenticated control seams may ask for one. Ignoring it; "
                "cancel with clear_drain_request().",
                why, ", ".join(sorted(TRUSTED_DRAIN_PRINCIPALS)),
            )
        return True
    return False


def _active_drain_body(home: Optional[Path]) -> Optional[dict[str, Any]]:
    """Marker body if present AND not stale (definite epoch mismatch, expired, dead writer, or
    refused principal), else None.

    Ordered cheapest-first: the epoch read is cached and the body is already in memory, while the
    writer liveness probe is a sysctl/``/proc`` read.  Every verdict is definite-only, so an
    unreadable probe never cancels a drain.
    """
    body = read_drain_request(home=home)
    if body is None:
        return None
    if (_marker_epoch_is_stale(body) or _marker_is_expired(body)
            or _marker_principal_refused(body) or _marker_writer_is_dead(body)):
        return None
    return body


def drain_requested(*, home: Optional[Path] = None) -> bool:
    """True iff an active (present, same-epoch, unexpired, live-writer, trusted) marker exists.

    A marker whose ``epoch`` does not match the current instantiation epoch is treated as absent: it
    survived a container/VM restart (HERMES_HOME is a durable Fly volume on Hermes Cloud) and the lifecycle
    action that triggered the drain has already completed — honouring it would wedge the freshly-restarted
    gateway in ``draining`` (NS-570). A marker whose ``requested_at`` is older than
    :data:`DRAIN_REQUEST_MAX_AGE_SECONDS` is likewise treated as absent: it is an orphan whose
    drain-gated action completed without a restart and was never cancelled (#85433). A marker whose
    recorded writer is provably gone is absent at once (SOU-3249), and so is one from a principal the
    live gateway does not trust. Every check is lenient (see :func:`_marker_epoch_is_stale`,
    :func:`_marker_is_expired`, :func:`_marker_writer_is_dead`, :func:`_marker_principal_refused`): a
    legacy/corrupt marker with no epoch, no timestamp and no writer identity still reads as
    drain-active, and so does one from a host whose epoch or process table cannot be read.
    """
    return _active_drain_body(home) is not None


def active_drain(home: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """``drain_requested()``'s verdict with its evidence: the ACTIVE marker body, or ``None``.

    ``drain_requested()`` answers "may I dispatch?"; a consumer that has to explain a pause (the
    cron drain gate stamps ``last_fire_error`` and the liveness pager names the cause, SOU-3252)
    needs the ``principal``/``requested_at``/``writer_pid`` too. Same activeness rule by
    construction — one predicate, one answer — so a stale or refused marker is ``None`` here
    exactly as it is ``False`` there.
    """
    return _active_drain_body(home)


def drain_notification_suppressed(*, home: Optional[Path] = None) -> bool:
    """True iff an ACTIVE marker asks to suppress the shutdown broadcast.

    Same activeness rule as :func:`drain_requested`, so an orphan can never silence
    a fresh gateway; a marker without the field reads False (fail toward louder).
    A marker from an untrusted/``test_only`` principal is never active, so it can
    never suppress either — the flag is refused at write time too (SOU-3249).

    "Active" means exactly what :func:`drain_requested` means — a marker present AND stamped with the
    current instantiation epoch AND not past its max-age. A stale (other-epoch) marker that survived a
    machine restart on the durable HERMES_HOME volume, or an expired same-epoch orphan (#85433), is ignored
    here just as it is for drain state (NS-570): we must never let an orphaned marker's flag silence a
    *fresh* gateway's legitimate shutdown broadcast.
    """
    body = _active_drain_body(home)
    return bool(body and body.get("suppress_notification"))


def read_drain_request(*, home: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Return the marker payload, ``{}`` if present but unparseable, ``None`` if absent. Never raises."""
    path = drain_request_path(home)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        if not isinstance(e, FileNotFoundError):
            _log.warning("drain-control: failed to read %s: %s", path, e)
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}
