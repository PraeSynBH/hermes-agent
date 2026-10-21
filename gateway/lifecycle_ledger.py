"""Gateway lifecycle ledger — durable termination-reason evidence.

Graceful shutdowns leave forensics (``gateway-exit-diag.log``); an unclean
death (SIGKILL, kernel OOM, VM death) runs no handler.  A sentinel at
``<HERMES_HOME>/state/gateway.lifecycle.json`` closes the gap:
:func:`record_startup` finds ``phase == "running"`` from the previous life →
unclean death, appended to the exit-diag log as ``gateway.previous_unclean_exit``
and logged at WARNING; :func:`mark_exited` rewrites ``phase=exited`` on every
clean exit path.  Best-effort: forensics must never affect the lifecycle.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _process_hermes_home() -> Path:
    """HERMES_HOME for process-level identity files (ignore task overrides)."""
    from hermes_constants import get_hermes_home, get_process_hermes_home

    # get_process_hermes_home expands ``~``/``$VAR`` (python -m gateway.run skips the CLI normalizer).
    return get_process_hermes_home() if os.environ.get("HERMES_HOME", "").strip() else get_hermes_home()


def _home_path(home: Optional[Path], *relative: str) -> Path:
    return (_process_hermes_home() if home is None else home).joinpath(*relative)


def get_lifecycle_sentinel_path(home: Optional[Path] = None) -> Path:
    """Return ``<HERMES_HOME>/state/gateway.lifecycle.json``."""
    return _home_path(home, "state", "gateway.lifecycle.json")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _proc_fields(path: str, wanted: dict[str, str]) -> dict[str, int]:
    """``{dst: int}`` for each ``src: dst`` key found in a ``Key: value`` /proc file."""
    found: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                key = line.split(":", 1)[0]
                if key in wanted:
                    found[wanted[key]] = int(line.split()[1])
                    if len(found) == len(wanted):
                        break
    except (OSError, ValueError, IndexError):
        return {}
    return found


def sample_memory() -> dict[str, Any]:
    """Cheap /proc snapshot (KiB): own RSS + MemTotal/MemAvailable + swap used.  Linux-only
    (``{}`` elsewhere), never raises; the 30s heartbeat embeds it so OOM cycles are classifiable."""
    sample = _proc_fields("/proc/self/status", {"VmRSS": "rss_kib"})
    mem = _proc_fields("/proc/meminfo", {"MemTotal": "mem_total_kib", "MemAvailable": "mem_available_kib",
                                         "SwapTotal": "SwapTotal", "SwapFree": "SwapFree"})
    swap_total, swap_free = mem.pop("SwapTotal", None), mem.pop("SwapFree", None)
    sample.update(mem)
    if swap_total is not None and swap_free is not None:
        sample["swap_used_kib"] = swap_total - swap_free
    return sample


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_sentinel(payload: dict[str, Any], home: Optional[Path]) -> None:
    try:
        from utils import atomic_json_write

        path = get_lifecycle_sentinel_path(home)
        from hermes_constants import mkdir_under_hermes_home

        mkdir_under_hermes_home(path.parent)
        atomic_json_write(path, payload, indent=None)
    except Exception:
        logger.debug("Failed to write lifecycle sentinel", exc_info=True)


def _append_exit_diag(record: dict[str, Any], home: Optional[Path]) -> None:
    """Append a JSON line to gateway-exit-diag.log (same format as the CLI's ``_exit_diag``)."""
    try:
        path = _home_path(home, "logs", "gateway-exit-diag.log")
        from hermes_constants import mkdir_under_hermes_home

        mkdir_under_hermes_home(path.parent)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        logger.debug("Failed to append unclean-exit record", exc_info=True)


def _pid_is_sentinel_owner(pid: Any, start_time: Any, create_time: Any) -> bool:
    """True when ``pid`` is a live process that is the sentinel's incarnation — guards the
    ``--replace`` race: a live matching owner mid-teardown is a handover, not a death.

    Identity is the psutil ``create_time`` the sentinel stamps at claim (epoch seconds, same
    producer on both sides). The sentinel's ``start_time`` is the ledger claim time and is NOT
    comparable with ``gateway.status.get_process_start_time`` (proc ticks on Linux, centiseconds
    elsewhere) — the old comparison never matched, so every ``--replace`` handover read as an
    unclean death. A pre-stamp sentinel (no ``create_time``) still has ``start_time`` in epoch
    seconds: the owner was born BEFORE it claimed, a PID reuser AFTER the owner died, so a birth
    later than the claim is a reuser."""
    try:
        pid_int = int(pid)
        # NOT os.kill(pid, 0): on Windows that sends CTRL_C_EVENT to the target's console group.
        from gateway.status import _pid_exists

        if pid_int <= 0 or not _pid_exists(pid_int):
            return False
    except Exception:
        return False
    from hermes_cli.process_identity import _process_create_time

    actual = _process_create_time(pid_int)
    if actual is None:
        return True  # can't disambiguate PID reuse
    if type(create_time) in (int, float):
        return abs(actual - float(create_time)) <= 2.0
    if type(start_time) in (int, float):
        return actual <= float(start_time) + 2.0
    return True


def _suspected_oom(mem: dict[str, Any]) -> bool:
    """Heuristic only (classification stays with the reader); thresholds are
    memory_status' "critical" tier so a live warning and a post-mortem verdict agree."""
    from gateway.memory_status import _CRITICAL_AVAILABLE_FRACTION, _CRITICAL_AVAILABLE_KIB

    total, avail = mem.get("mem_total_kib"), mem.get("mem_available_kib")
    if not isinstance(avail, int):
        return False
    return avail < _CRITICAL_AVAILABLE_KIB or (
        isinstance(total, int) and total > 0 and avail / total < _CRITICAL_AVAILABLE_FRACTION
    )


def stamp_restart_request(requester: Optional[dict[str, Any]], *, source: str,
                          home: Optional[Path] = None) -> None:
    """Record WHO asked for the restart now in progress in the lifecycle sentinel.

    The sentinel is the file a post-mortem reads, and the requester is knowable ONLY in the moment
    the request lands: a SIGUSR1 carries no payload and the requesting process may be gone by the
    time the drain ends (PRX-5423). :func:`mark_exited` carries this record through the exit and
    :func:`record_startup` re-publishes it as ``prior_restart_request``, so "what put this process
    here, and who asked?" survives the process that could answer it.

    Ownership-guarded: a replacement that already claimed the sentinel is left alone. Best-effort,
    never raises.
    """
    try:
        sentinel = _read_json(get_lifecycle_sentinel_path(home))
        if sentinel is None or sentinel.get("pid") != os.getpid():
            return
        request: dict[str, Any] = {"requested_at": _now_iso(), "source": source}
        if requester:
            for key in ("requestedByRunId", "agentId", "issueId", "path", "caller_pid", "ts",
                        "age_s", "detail"):
                if requester.get(key) not in (None, ""):
                    request[key] = requester[key]
        else:
            # An unattributed restart is itself the finding: say so where the machine reads.
            request["attributed"] = False
        sentinel["restart_request"] = request
        _write_sentinel(sentinel, home)
    except Exception:
        logger.debug("Failed to stamp the restart requester", exc_info=True)


def _crash_evidence(home: Optional[Path], prior_pid: Any = None) -> dict[str, Any]:
    """``prior_crash_*`` fields for a death that left a fatal-signal record, else ``{}``.

    ``gateway-crash.log`` is written by the dying process from signal context
    (``gateway.crash_recorder``); naming the signal here is what separates "died hard" from
    "died hard of SIGBUS" on the row the soak gates key on (PRA-6669). A record whose pid is
    not the sentinel's previous owner belongs to an older life and is not this boot's news.
    """
    try:
        from gateway.crash_recorder import CRASH_LOG_NAME, prior_crash

        crash = prior_crash(path=_home_path(home, "logs", CRASH_LOG_NAME))
    except Exception:
        logger.debug("Could not read the fatal-signal crash record", exc_info=True)
        return {}
    if not crash or (prior_pid is not None and crash.get("pid") != prior_pid):
        return {}
    fields: dict[str, Any] = {"prior_crash_ts": crash.get("ts"), "prior_crash_reason": crash.get("reason"),
                              "prior_crash_pid": crash.get("pid")}
    if crash.get("signal"):
        fields["prior_crash_signal"] = crash["signal"]
        fields["prior_crash_signal_number"] = crash.get("signal_number")
    return fields


def detect_unclean_exit(home: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Evidence dict when the previous life died uncleanly, else ``None``. Read-only."""
    sentinel = _read_json(get_lifecycle_sentinel_path(home))
    if not sentinel or sentinel.get("phase") != "running":
        return None
    if _pid_is_sentinel_owner(sentinel.get("pid"), sentinel.get("start_time"), sentinel.get("create_time")):
        return None  # live owner — planned takeover in flight, not a death
    evidence: dict[str, Any] = {
        "prior_pid": sentinel.get("pid"), "prior_started_at": sentinel.get("started_at"),
        "prior_start_time": sentinel.get("start_time"),
    }
    # Enrich with the last heartbeat: last proven liveness and memory at that moment.
    try:
        from gateway.shutdown_watchdog import get_loop_heartbeat_path

        hb = _read_json(get_loop_heartbeat_path(home)) or {}
    except Exception:
        hb = {}
    if hb:
        evidence["last_heartbeat_at"] = hb.get("updated_at")
    mem = hb.get("mem")
    if isinstance(mem, dict):
        evidence["last_heartbeat_mem"] = mem
        if _suspected_oom(mem):
            evidence["suspected_oom"] = True
    evidence.update(_crash_evidence(home, evidence.get("prior_pid")))
    return evidence


# Startup-watchdog lease held while the unclean-exit integrity check runs (#115542):
# ``PRAGMA quick_check(1)`` is one monolithic SQLite operation and per-call leases
# clamp at 900s, so a single entry lease cannot cover a multi-thousand-second healthy
# check on a huge store — without renewal the watchdog kills the attempt with exit 75
# and the restart loop never reaches the cron ticker. The progress handler below renews
# a phase-owned lease for as long as SQLite is making progress; without a deadline it
# always returns 0 so it never aborts the check, and the ok/absent/first-complaint
# verdicts stay fail-closed. When an explicit ``deadline_at`` is supplied the
# handler returns non-zero once it passes, so SQLite aborts the scan and the caller
# can bound a pathological store instead of pinning the boot path.
_INTEGRITY_CHECK_LEASE_PHASE = "state_db_unclean_integrity_check"
_INTEGRITY_CHECK_LEASE_S = 900.0
# VM instructions between progress-handler invocations; each invocation is a
# monotonic-clock compare, renewing at most every _INTEGRITY_CHECK_LEASE_RENEW_S.
_INTEGRITY_CHECK_PROGRESS_OPS = 100_000
_INTEGRITY_CHECK_LEASE_RENEW_S = 60.0
# Bound on the deferred post-connect scan: a pathological store cannot pin its thread forever.
STATE_DB_INTEGRITY_DEADLINE_S = 1800.0


def _install_integrity_check_lease(conn: sqlite3.Connection, *, deadline_at: Optional[float] = None) -> None:
    """Claim the integrity-check lease and renew it from a SQLite progress handler.

    ``report_startup_progress`` is a no-op when no watchdog is armed and never raises.
    Without ``deadline_at`` the handler always returns 0 -- it must never abort the verdict
    PRAGMA.  With ``deadline_at`` (a ``time.monotonic`` timestamp) it returns non-zero once
    that instant is reached, so SQLite raises ``OperationalError: interrupted`` and the
    caller can bound the scan.  Synchronous by design: no checker worker thread can outlive
    the check.
    """
    from hermes_startup_watchdog import report_startup_progress

    report_startup_progress(_INTEGRITY_CHECK_LEASE_S, phase=_INTEGRITY_CHECK_LEASE_PHASE)
    last_renew = time.monotonic()

    def _on_progress() -> int:
        nonlocal last_renew
        now = time.monotonic()
        if now - last_renew >= _INTEGRITY_CHECK_LEASE_RENEW_S:
            last_renew = now
            report_startup_progress(_INTEGRITY_CHECK_LEASE_S, phase=_INTEGRITY_CHECK_LEASE_PHASE)
        if deadline_at is not None and now >= deadline_at:
            return 1
        return 0

    conn.set_progress_handler(_on_progress, _INTEGRITY_CHECK_PROGRESS_OPS)


def check_state_db_integrity(home: Optional[Path] = None, *, deadline_s: Optional[float] = None) -> str:
    """``"ok"``, ``"absent"``, or the first ``quick_check`` complaint.  Never raises.

    Only after an unclean death — SIGKILL mid-WAL-checkpoint can leave half-written
    b-tree pages.  ``quick_check(1)`` stops at the first problem (~2s on a healthy
    500MB store): cheap once per unclean boot, too costly every boot.  Opened
    normally: a WAL store needs its -shm sidecar for read-only, and the PRAGMA writes nothing.

    Holds a phase-owned startup-watchdog lease for the check duration
    (renewed from a SQLite progress handler, #115542) so a long healthy check
    on a huge store is not mistaken for a parked deadlock.

    ``deadline_s`` (optional): abort the scan once that many seconds elapse and
    return ``"check-incomplete: exceeded <N>s"``.  ``deadline_s=None`` (default) is the
    historic unbounded behaviour -- the progress handler never aborts the PRAGMA.
    """
    path = _home_path(home, "state.db")
    if not path.exists():
        return "absent"
    deadline_at: Optional[float] = None
    if deadline_s is not None:
        try:
            deadline_at = time.monotonic() + max(0.0, float(deadline_s))
        except (TypeError, ValueError):
            deadline_at = None
    try:
        with closing(sqlite3.connect(str(path))) as conn:
            _install_integrity_check_lease(conn, deadline_at=deadline_at)
            row = conn.execute("PRAGMA quick_check(1)").fetchone()
    except Exception as exc:
        if (deadline_at is not None and isinstance(exc, sqlite3.OperationalError)
                and "interrupt" in str(exc).lower()):
            limit = max(0.0, float(deadline_s)) if deadline_s is not None else 0.0
            return f"check-incomplete: exceeded {limit:g}s"
        return f"check-failed: {exc}"
    return "check-failed: no result" if not row or row[0] is None else str(row[0])


def _state_db_journal_mode(path: Path) -> Optional[str]:
    """On-disk journal mode of ``path`` via one ``PRAGMA journal_mode`` read on a short-lived
    connection (no lock outlives it); ``None`` when undeterminable.  Never raises."""
    try:
        from hermes_state_wal import _on_disk_journal_mode

        with closing(sqlite3.connect(str(path), timeout=1.0)) as conn:
            return _on_disk_journal_mode(conn) or None
    except Exception:
        logger.debug("Could not read state.db journal mode", exc_info=True)
        return None


def _report_unclean_exit(evidence: dict[str, Any], home: Optional[Path], *, defer_integrity_check: bool = False) -> None:
    """Persist the exit-diag record and log at WARNING.

    When ``defer_integrity_check`` is set, the forensic store scan is NOT run here (it would
    gate platform startup); the record carries ``state_db_integrity: "deferred"`` and
    :func:`run_deferred_integrity_check` runs the scan post-connect.
    """
    if defer_integrity_check:
        evidence["state_db_integrity"] = "deferred"
    else:
        # The death may have torn the store; this is the only moment we know to look.
        verdict = evidence["state_db_integrity"] = check_state_db_integrity(home=home)
        if verdict not in ("ok", "absent"):
            logger.error(
                "state.db FAILED integrity check after an unclean gateway exit: %s — sessions may read as "
                "missing until it is repaired. Run `hermes doctor`.",
                verdict,
            )
    _append_exit_diag({"ts": _now_iso(), "tag": "gateway.previous_unclean_exit", "pid": os.getpid(), **evidence}, home)
    if evidence.get("prior_crash_signal"):
        # A knob the soak gates and restart watchdogs read: "no deploy occurred, the window
        # still restarted" (PRA-6502 / PRA-6380) must be distinguishable from a clean run of
        # boots. One line, distinct tag, the signal that killed the process (PRA-6669).
        _append_exit_diag({"ts": _now_iso(), "tag": "gateway.soak_window_reset", "pid": os.getpid(),
                           "reason": "crash_signal",
                           "prior_crash_signal": evidence["prior_crash_signal"],
                           "prior_crash_ts": evidence.get("prior_crash_ts"),
                           "prior_pid": evidence.get("prior_pid")}, home)
    logger.warning(
        "Previous gateway life (pid=%s, started_at=%s) exited UNCLEANLY (no exit path ran — SIGKILL / OOM / "
        "VM death, or a process kill issued by the agent or one of its descendants, e.g. a pkill/taskkill of "
        "the host interpreter image; see #113667). last_heartbeat_at=%s last_mem=%s suspected_oom=%s "
        "crash_signal=%s",
        evidence.get("prior_pid"), evidence.get("prior_started_at"), evidence.get("last_heartbeat_at"),
        evidence.get("last_heartbeat_mem"), evidence.get("suspected_oom", False),
        evidence.get("prior_crash_signal"),
    )


# A managed restart that just completed already gave its requester a fresh process, and every
# honoured request costs up to the whole after-turn cap of fleet-wide cron silence. The floor is
# deliberately short: a legitimate second deploy (a module change landed after the first restart)
# must not wait long, and the caller is told exactly which restart already served it.
RESTART_REQUEST_FLOOR_SECONDS = 600.0

# How far past the deferral cap a ``restart_request`` record may be and still describe a restart
# that is in progress: the cap bounds the after-turn wait, then stop()/drain runs before the process
# exits. Past this the process is either gone or wedged, and a new signal is the right move.
RESTART_STOP_SLACK_SECONDS = 300.0


def _instant_age_seconds(stamp: Any) -> Optional[float]:
    """Age of an ISO-8601 stamp as the ledger writes it (tz-aware; ``Z`` accepted), or ``None``."""
    if not isinstance(stamp, str) or not stamp.strip():
        return None
    text = stamp.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (datetime.now(UTC) - parsed).total_seconds()


def _deferral_cap_seconds() -> float:
    """``agent.restart_after_turn_timeout`` — the deferral's own lifetime — or the default."""
    try:
        from cron.scheduler_drain_gate import _restart_deferral_cap

        return float(_restart_deferral_cap())
    except Exception:  # health: allow BLE001 -- an unreadable cap must answer like the default, never raise
        return 1800.0


def restart_request_block(*, target_pid: Optional[int] = None,
                          home: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Why a NEW restart request must not be signalled now, and the record that proves it.

    Two verdicts, both read off the lifecycle sentinel instead of a new state file, because the
    sentinel is already the one durable place a restart's identity survives:
    :func:`stamp_restart_request` writes ``restart_request`` at T=0, and :func:`record_startup`
    carries it onto the next life as ``prior_restart_request``.

    * ``in_progress`` — THIS live gateway has stamped a ``restart_request`` whose deferral can still
      be open, so a second signal buys nothing but makes the requester believe a fresh restart is
      coming. Measured 2026-10-23 (PRA-6925): four agent-requested restarts in twelve hours, three
      of them running the full 1800 s cap while cron had no work in flight.
    * ``recently_completed`` — the process serving this request was itself born from a managed
      restart inside :data:`RESTART_REQUEST_FLOOR_SECONDS`, so the deploy the requester wants is
      already in the running process.

    ``None`` means "no verdict": no sentinel, a sentinel for another pid, an unparseable stamp, or a
    record old enough that the deferral it names cannot still be running. Every one of those lets
    the restart through — a wrong refusal blocks a deploy, which is worse than a redundant restart.
    """
    sentinel = _read_json(get_lifecycle_sentinel_path(home))
    if not sentinel or sentinel.get("phase") != "running":
        return None
    try:
        pid = int(sentinel.get("pid"))
    except (TypeError, ValueError):
        return None
    if target_pid is not None and pid != int(target_pid):
        return None

    in_progress = sentinel.get("restart_request")
    if isinstance(in_progress, dict):
        age = _instant_age_seconds(in_progress.get("requested_at"))
        if age is not None and age <= _deferral_cap_seconds() + RESTART_STOP_SLACK_SECONDS:
            return {"verdict": "in_progress", "pid": pid, "aged_s": round(age, 1),
                    "request": dict(in_progress)}

    prior = sentinel.get("prior_restart_request")
    born = _instant_age_seconds(sentinel.get("started_at"))
    if isinstance(prior, dict) and born is not None and born <= RESTART_REQUEST_FLOOR_SECONDS:
        return {"verdict": "recently_completed", "pid": pid, "aged_s": round(born, 1),
                "request": dict(prior), "started_at": sentinel.get("started_at")}
    return None


def record_startup(home: Optional[Path] = None, *, defer_integrity_check: bool = True) -> Optional[dict[str, Any]]:
    """Boot entry point: report any unclean previous exit (evidence dict, also persisted
    to ``gateway-exit-diag.log`` and logged at WARNING) then claim the sentinel.  Never raises.

    ``defer_integrity_check`` defaults to True: the state.db integrity scan is forensic and
    must NOT gate platform startup (a large store's ``quick_check`` blocked the ``:8644`` fleet
    peer-DM transport for minutes).  In deferred mode the exit-diag record is written
    immediately with ``state_db_integrity: "deferred"`` and the scan runs post-connect via
    :func:`run_deferred_integrity_check`.  Pass False for the historic INLINE scan.
    """
    evidence: Optional[dict[str, Any]] = None
    try:
        evidence = detect_unclean_exit(home)
        if evidence is not None:
            _report_unclean_exit(evidence, home, defer_integrity_check=defer_integrity_check)
    except Exception:
        logger.debug("Unclean-exit detection failed", exc_info=True)
    try:
        # Read the previous life's record BEFORE claiming it: it is the only place the restart
        # requester survives (stamped by the signal handler while that process was still alive).
        prior = _read_json(get_lifecycle_sentinel_path(home)) or {}
        claim: dict[str, Any] = {"phase": "running", "pid": os.getpid(), "start_time": time.time(), "started_at": _now_iso()}
        # The WIRE CONTRACT this life serves, frozen at boot like sys.modules: the deploy check reads
        # it back to prove a contract change actually replaced the process (SOU-3318).
        serving_contract = _serving_contract_revision()
        if serving_contract:
            claim["contract_revision"] = serving_contract
        # Process birth (psutil), distinct from ``start_time`` (the ledger claim, seconds later once
        # imports finish): the Windows start attestation binds PIDs to birth time (#110020 review).
        from hermes_cli.process_identity import _process_create_time

        create_time = _process_create_time(os.getpid())
        if create_time is not None:
            claim["create_time"] = create_time
        # Carry the verdict on the PREVIOUS life on the new sentinel: it is the only
        # machine-readable copy (/api/status reads it to report an OOM restart).
        # Scoped to this life — the next clean exit or boot rewrites the sentinel.
        if evidence is not None:
            claim["prior_unclean_exit"] = True
            if evidence.get("suspected_oom"):
                claim["prior_suspected_oom"] = True
        # WHO asked for the restart that produced this life (PRX-5423): carried off the previous
        # sentinel whether it exited cleanly or died mid-drain, because the requesting CLI and its
        # environment are gone by now.
        prior_request = prior.get("restart_request")
        if isinstance(prior_request, dict):
            claim["prior_restart_request"] = prior_request
        _write_sentinel(claim, home)
    except Exception:
        logger.debug("Failed to claim lifecycle sentinel", exc_info=True)
    return evidence


def run_deferred_integrity_check(
    evidence: Optional[dict[str, Any]], *, home: Optional[Path] = None,
    deadline_s: float = STATE_DB_INTEGRITY_DEADLINE_S,
) -> str:
    """Run the deferred post-connect state.db scan and append a second diag record.

    Companion to :func:`record_startup` with ``defer_integrity_check=True``: the forensic scan
    is moved off the startup path (where, on a multi-GB store, it gated every platform adapter
    — including the ``:8644`` fleet peer-DM transport — for minutes) into a daemon thread
    started only after platforms connected.  ``deadline_s`` bounds a pathological store so the
    thread cannot be pinned forever.

    Appends ``{"tag": "gateway.state_db_integrity", "deferred": true, "state_db_integrity":
    <verdict>, "prior_pid": ...}`` and logs the existing ERROR text on a bad verdict.  NEVER
    raises -- a forensic scan must not disturb a running gateway (the alerting path is the
    bounded per-poll readiness probe ``gateway/readiness.py::_probe_state_db``).

    Only a WAL store is scanned: in rollback-journal mode (``database.journal_mode: delete``,
    or a silent WAL fallback) the scan's SHARED lock would block every live commit for its
    whole run, so the scan is skipped with a ``skipped: ...`` verdict and one WARNING.  The
    on-disk mode decides, not config; an undeterminable mode is skipped too.
    """
    try:
        path = _home_path(home, "state.db")
        mode = _state_db_journal_mode(path) if path.exists() else "wal"
        if mode != "wal":
            verdict = (f"skipped: rollback-journal store (journal_mode={mode or 'unknown'}), "
                       "scan would block live writes")
            logger.warning(
                "Previous gateway exit was unclean, but the state.db integrity check was skipped: %s is a "
                "rollback-journal store (journal_mode=%s) and a background scan would block live writes. "
                "To check it, stop the gateway and run: sqlite3 %s 'PRAGMA quick_check'",
                path, mode or "unknown", path,
            )
        else:
            verdict = check_state_db_integrity(home=home, deadline_s=deadline_s)
        if verdict not in ("ok", "absent") and not verdict.startswith("skipped:"):
            logger.error(
                "state.db FAILED integrity check after an unclean gateway exit: %s — sessions may read as "
                "missing until it is repaired. Run `hermes doctor`.",
                verdict,
            )
        record: dict[str, Any] = {
            "ts": _now_iso(),
            "tag": "gateway.state_db_integrity",
            "deferred": True,
            "state_db_integrity": verdict,
        }
        if isinstance(evidence, dict) and evidence.get("prior_pid") is not None:
            record["prior_pid"] = evidence.get("prior_pid")
        _append_exit_diag(record, home)
        if isinstance(evidence, dict):
            evidence["state_db_integrity"] = verdict
        return verdict
    except Exception:
        logger.debug("Deferred state.db integrity check failed", exc_info=True)
        return "check-failed: deferred scan raised"


def mark_exited(exit_code: Optional[int] = None, reason: str = "graceful_shutdown", home: Optional[Path] = None) -> None:
    """Mark the current life as cleanly exited.  Idempotent, never raises.

    Only rewrites a sentinel provably owned by this process: during ``--replace``
    the replacement claims it before the old process finishes teardown and must not
    be clobbered.  ``pid=None`` / malformed sentinels have unknown ownership → left alone.
    """
    try:
        sentinel = _read_json(get_lifecycle_sentinel_path(home))
        if sentinel is not None and sentinel.get("pid") != os.getpid():
            return
        exited: dict[str, Any] = {"phase": "exited", "pid": os.getpid(), "exit_code": exit_code, "exit_reason": reason,
                                  "exited_at": _now_iso()}
        # Carry the incarnation identity: the Windows start attestation matches a clean exit by
        # PID *and* start time so a reused PID's exit cannot vouch for a different life (#110020).
        for key in ("start_time", "create_time", "restart_request"):
            if sentinel is not None and sentinel.get(key) is not None:
                exited[key] = sentinel[key]
        _write_sentinel(exited, home)
    except Exception:
        logger.debug("Failed to mark lifecycle sentinel exited", exc_info=True)


def mark_planned_stop(target_pid: Optional[int], reason: str = "gateway_stop", home: Optional[Path] = None,
                      *, expected_create_time: Any = None) -> bool:
    """Close the sentinel for a stop that was REQUESTED on purpose, from outside the gateway.

    A deliberate stop can outlive the gateway's own teardown: the supervisor (launchd
    ``ExitTimeOut``) or an impatient stopper SIGKILLs a gateway that is still draining
    in-flight turns, so :func:`mark_exited` — the single funnel inside the process — never
    runs.  The sentinel then stays ``phase=running`` and the next boot reports
    ``gateway.previous_unclean_exit`` for a stop that was asked for: a phantom death that
    buries the real one (PRA-6427, and the forensics of PRA-6426).

    The stopper calls this once its target is gone.  Ownership-guarded twice, so a race can
    never invent a clean exit: the sentinel must still name ``target_pid`` (a replacement
    that already claimed it is left alone), and when both ``expected_create_time`` values
    are known they must agree (a recycled PID's stop cannot vouch for another life).
    A sentinel already marked ``exited`` is left as the process wrote it.  Idempotent,
    never raises; returns True when it wrote the record.
    """
    try:
        sentinel = _read_json(get_lifecycle_sentinel_path(home))
        if sentinel is None or target_pid is None or sentinel.get("pid") != target_pid:
            return False
        if sentinel.get("phase") != "running":  # mark_exited (or an earlier planned stop) got there first
            return False
        stamped = sentinel.get("create_time")
        if isinstance(stamped, (int, float)) and isinstance(expected_create_time, (int, float)):
            if abs(float(stamped) - float(expected_create_time)) > 2.0:
                return False
        exited: dict[str, Any] = {
            "phase": "exited", "pid": target_pid, "exit_code": None, "exit_reason": reason,
            "exited_at": _now_iso(), "recorded_by": "planned_stop", "stopper_pid": os.getpid(),
        }
        for key in ("start_time", "create_time", "started_at"):
            if sentinel.get(key) is not None:
                exited[key] = sentinel[key]
        _write_sentinel(exited, home)
        _append_exit_diag({"ts": _now_iso(), "tag": "gateway.planned_stop", "pid": os.getpid(),
                           "target_pid": target_pid, "exit_reason": reason}, home)
        return True
    except Exception:
        logger.debug("Failed to record a planned stop in the lifecycle sentinel", exc_info=True)
        return False


def _serving_contract_revision() -> Optional[str]:
    """The wire-contract revision THIS process serves (its boot snapshot).

    ``boot_revision`` reads the in-tree contract document once and answers from the snapshot: the
    only failure it can see is an unreadable file, which yields ``None`` there.
    """
    from gateway.contract_revision import boot_revision

    return boot_revision()


def serving_contract_revision(home: Optional[Path] = None) -> Optional[str]:
    """The contract revision the LIVE gateway stamped at boot, or ``None`` when unknown.

    Read from the lifecycle sentinel, never re-hashed from the tree: a gateway imports its param
    models at boot and keeps them, so an old process asked to describe itself from the current
    checkout would report the NEW revision and call a skewed deploy healthy (SOU-3318).
    """
    sentinel = _read_json(get_lifecycle_sentinel_path(home))
    if not sentinel or sentinel.get("phase") != "running":
        return None
    planned = sentinel.get("contract_revision")
    return planned if isinstance(planned, str) and planned else None


def contract_revision_verdict(home: Optional[Path] = None) -> tuple[str, str]:
    """``(verdict, detail)`` for the deploy check: ``in_sync`` / ``stale`` / ``unknown``.

    ``stale`` is the SOU-3316 shape exactly — a wire-contract change shipped while the process
    serving the old contract stayed up — so a deploy in that state is NOT complete and the fleet
    must not be declared ready. ``unknown`` (no sentinel, no contract document) never blocks: the
    check exists to catch a proven mismatch, not to invent one.
    """
    from gateway.contract_revision import revision as tree_revision

    tree, serving = tree_revision(), serving_contract_revision(home)
    if not tree or not serving:
        return "unknown", f"tree={tree or 'unreadable'} serving={serving or 'not stamped'}"
    if tree == serving:
        return "in_sync", f"contract-{tree}"
    return "stale", (
        f"the serving gateway was built against contract-{serving} but the tree declares "
        f"contract-{tree} — a wire-contract change shipped without restarting the gateway")


def read_prior_exit_label(profile_home: Path) -> str:
    """``clean``/``unclean``/``unknown`` for the profile's last gateway life; annotates
    ``container-boot.log``.  ``running`` is unclean: the old PID namespace is gone at container boot."""
    try:
        phase = (_read_json(get_lifecycle_sentinel_path(profile_home)) or {}).get("phase")
        return {"exited": "clean", "running": "unclean"}.get(phase, "unknown")
    except Exception:
        return "unknown"
