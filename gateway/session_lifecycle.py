"""SessionStore explicit suspension, crash-recovery markers, pruning and shared clock/id helpers."""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone, UTC
from typing import TYPE_CHECKING, Optional

from hermes_state_ids import new_session_id

if TYPE_CHECKING:
    from gateway.session import SessionEntry, SessionSource

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.session")


def _now() -> datetime:
    """Return the current local time."""
    return datetime.now()


def _new_session_id(now: datetime) -> str:
    return new_session_id(now, hex_len=8)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _parse_iso(value) -> Optional[datetime]:
    """``datetime.fromisoformat`` that returns None for empty/malformed input."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


# Auto-continue freshness window (1 hour) after the ``resume_pending`` mark; ``gateway/run.py``
# bridges config.yaml ``agent.gateway_auto_continue_freshness`` into the env var at startup.
_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT = 60 * 60


def auto_continue_freshness_window() -> float:
    """Resume-scheduler freshness window; stale automation never discards the transcript."""
    raw = os.environ.get("HERMES_AUTO_CONTINUE_FRESHNESS")
    try:
        return float(raw) if raw else float(_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)
    except (TypeError, ValueError):
        return float(_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)


class SessionLifecycleMixin:
    """SessionStore explicit boundaries and crash-recovery markers."""

    def _is_session_ended_in_db(self, session_id: str, session_key: Optional[str] = None) -> bool:
        """True iff state.db says the session is gone: ended (non-null end_reason) or hard-deleted
        (no row in a readable owning DB). No DB or a DB error -> False (same failure mode as
        ``_prune_stale_sessions_locked``). Lets routing self-heal a session finalized or deleted
        while the gateway stays alive. Store resolved from the owning profile.

        Used by ``get_or_create_session`` to self-heal at routing time: ``_prune_stale_sessions_locked``
        only runs at startup, so a session ended in the DB while the gateway stays alive (any path that
        finalizes the row without clearing sessions.json) would otherwise be reused as a live routing key
        and silently swallow every subsequent message until the next restart (#54878 — the live-gateway
        variant of #52804/FM9). A hard delete is the same shape one step further: the row is GONE, not
        merely ended, and reusing the route makes run_agent's INSERT OR IGNORE resurrect the deleted
        session with its old id (#42422) — so a missing row is treated exactly like an ended one. DB
        errors are non-fatal — never block routing on a failed lookup.
        The store is resolved from the row's owning profile rather than the ambient scope: an unscoped
        background writer keeps its own copy of the same session, and comparing against that copy reports a
        live session as ended (#66887).

        Pass *session_key* when the owning key is known but the id may have left the routing index:
        after the self-heal re-homes the key, the id has no owner and would resolve to the launch
        store, which never holds a routed profile's rows (#118862).
        """
        db = self._db_for_key(session_key) if session_key else self._db_for_session_id(session_id)
        if not db or not session_id:
            return False
        try:
            row = db.get_session(session_id)
        except Exception:
            return False
        return row is None or row.get("end_reason") is not None

    def _local_receipt_target(self, entry: Optional[SessionEntry]) -> Optional[str]:
        """The physical target the authority's receipt names for *entry*'s local route, else None
        (not a local route, no receipt, or the receipt belongs to another route).

        Such a route moves only through the authority (compression and reset advance the receipt
        and the route in one transaction), and its first admitted turn reopens a stamped row
        (``reopen_local_session``). A row stamped ended (a viewer's ``ws_orphan_reap``) is
        therefore never a stale route: dropping it minted a replacement id the authority refuses
        (``admission_conflict``) and stranded the conversation (#140642)."""
        origin = getattr(entry, "origin", None)
        if origin is None or getattr(origin, "platform", None) is None or origin.platform.value != "local":
            return None
        db = self._db_for_key(entry.session_key)
        if db is None or getattr(db, "_read_ctx", None) is None:
            return None
        import sqlite3

        from hermes_state_local import local_receipt
        from hermes_state_runtime import RuntimeStoreError
        try:
            receipt = local_receipt(db, str(origin.chat_id))
            if receipt.get("route") != entry.session_key:
                return None
            return str(receipt["entry"]["session_id"]) or None
        except (RuntimeStoreError, sqlite3.Error, KeyError, TypeError, AttributeError):
            return None

    def _route_reset_reason(self, entry: SessionEntry) -> Optional[str]:
        """Only explicit suspension replaces a routed conversation; time never does."""
        return "suspended" if entry.suspended else None

    def _update_entry(self, session_key: str, mutate) -> bool:
        """Apply ``mutate(entry)`` under ``_lock``, then save after releasing it; False when the
        entry is missing or *mutate* returned False (nothing to persist)."""
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None or mutate(entry) is False:
                return False
        self._save()
        return True

    def _update_all_entries_locked(self, mutate) -> int:
        """Apply ``mutate(entry) -> bool`` to every entry under ``_lock``; save once, outside the
        lock, if any returned True. Returns the count that did."""
        with self._lock:
            self._ensure_loaded_locked()
            changed = sum(1 for entry in self._entries.values() if mutate(entry))
        if changed:
            self._save()
        return changed

    def suspend_session(self, session_key: str) -> bool:
        """Mark a session suspended so it auto-resets on next access (/stop). True if it existed.

        Used by ``/stop`` to prevent stuck sessions from being resumed after a gateway restart (#7536).
        """
        return self._update_entry(session_key, lambda e: setattr(e, "suspended", True))

    def _turn_marker_candidate(self, entry: SessionEntry, token, started_at):
        """``(candidate, touched, observed_token)`` for the active-turn pair. Lock held."""
        candidate = entry.to_dict()
        candidate["active_turn_token"] = token
        candidate["active_turn_started_at"] = _iso(started_at)
        touched = _now() if started_at is not None else None
        if touched is not None:
            # Keeps the legacy 120s startup heuristic working for an older binary during a rolling
            # downgrade/upgrade window.
            candidate["updated_at"] = touched.isoformat()
        return candidate, touched, entry.active_turn_token

    def _persist_turn_marker(self, session_key, entry, prepared, token, started_at, touched) -> None:
        """Persist the active-turn pair BEFORE publishing it in memory, so a failed write can
        neither leak an unowned token nor drop a live one. Never called with ``_lock`` held: the
        single-entry UPSERT is I/O, and this runs per turn (PRX-5789)."""
        observed = prepared[2]
        self._save_entry(session_key, entry_data=prepared[0])
        with self._lock:
            if self._entries.get(session_key) is not entry or entry.active_turn_token != observed:
                return  # the route moved or a newer turn owns the marker meanwhile
            entry.active_turn_token = token
            entry.active_turn_started_at = started_at
            if touched is not None:
                entry.updated_at = touched

    def mark_turn_active(self, session_key: str) -> Optional[str]:
        """Persist exact ownership of the running agent turn; returns the opaque token for
        :meth:`clear_turn_active`. Re-marking replaces the previous token so a stale asynchronous
        unwind cannot clear a newer turn."""
        token = uuid.uuid4().hex
        # Aware UTC, unlike the local wall clock elsewhere: the next process compares it with
        # epoch transcript timestamps and may run in another zone (DST, container vs unit TZ).
        started_at = datetime.now(UTC)
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None:
                return None
            prepared = self._turn_marker_candidate(entry, token, started_at)
        self._persist_turn_marker(session_key, entry, prepared, token, started_at, prepared[1])
        return token

    def clear_turn_active(self, session_key: str, token: str) -> bool:
        """Compare-and-swap clear an active-turn marker; ``False`` when the entry disappeared or a
        newer turn owns it."""
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None or entry.active_turn_token != token:
                return False
            prepared = self._turn_marker_candidate(entry, None, None)
        self._persist_turn_marker(session_key, entry, prepared, None, None, None)
        return True

    def recover_interrupted_turns(self, max_age_seconds: int = 60 * 60) -> int:
        """Promote crash-left turn markers into ``resume_pending`` (unclean startup only).
        Old/invalid markers are cleared without resuming; suspended sessions are never re-armed.
        Returns the number of newly promoted sessions."""
        now, epoch_now = _now(), time.time()
        promoted = 0

        def _promote(entry: SessionEntry) -> bool:
            nonlocal promoted
            if not entry.active_turn_token:
                return False
            started_at = entry.active_turn_started_at
            # Epoch arithmetic: a pre-upgrade naive marker reads as local time, an aware one exactly.
            marker_is_stale = started_at is None or (
                max_age_seconds > 0 and epoch_now - started_at.timestamp() > max_age_seconds
            )
            if not marker_is_stale and not entry.suspended:
                if entry.resume_pending:
                    # A drain-timeout marker is more specific; keep it.
                    if entry.last_resume_marked_at is None:
                        entry.last_resume_marked_at = now
                else:
                    entry.resume_pending = True
                    entry.resume_reason = "restart_interrupted"
                    entry.last_resume_marked_at = now  # freshness starts at discovery
                    promoted += 1
            entry.active_turn_token = None
            entry.active_turn_started_at = None
            return True

        self._update_all_entries_locked(_promote)
        return promoted

    def discard_active_turn_markers(self) -> int:
        """Clear orphan turn markers after a verified clean shutdown."""
        def _discard(entry: SessionEntry) -> bool:
            if not entry.active_turn_token and entry.active_turn_started_at is None:
                return False
            entry.active_turn_token = None
            entry.active_turn_started_at = None
            return True
        return self._update_all_entries_locked(_discard)

    def mark_resume_pending(self, session_key: str, reason: str = "restart_timeout") -> bool:
        """Mark a session resumable after a restart interruption (keeps the session_id/transcript,
        unlike ``suspend_session``). True if marked."""
        def _apply(entry: SessionEntry):
            if entry.suspended:  # never override an explicit ``suspended`` (hard forced-wipe)
                return False
            entry.resume_pending = True
            entry.resume_reason = reason
            entry.last_resume_marked_at = _now()
        return self._update_entry(session_key, _apply)

    def clear_resume_pending(self, session_key: str) -> bool:
        """Clear the resume-pending flag after a successful resumed turn; True if cleared."""
        def _apply(entry: SessionEntry):
            if not entry.resume_pending:
                return False
            entry.resume_pending = False
            entry.resume_reason = None
            entry.last_resume_marked_at = None
        return self._update_entry(session_key, _apply)

    def prune_old_entries(self, max_age_days: int) -> int:
        """Drop routing entries idle (by ``updated_at``) for more than max_age_days; suspended
        entries and entries with active background processes are kept. Only the key -> session_id
        mapping is dropped (the transcript stays). ``max_age_days <= 0`` disables. Returns count.

        The candidate scan and the active-work query happen OUTSIDE ``self._lock`` (PRX-6075): the
        index holds every key the gateway has ever routed — 15 762 rows on this host — and asking
        the process registry about each one under the lock serialised every turn's session write
        behind a scan of the registry for the whole tick. ``_lock`` guards ``_entries``/``_loaded``
        only; entries that moved while it was released are re-read before they are dropped.
        """
        if max_age_days is None or max_age_days <= 0:
            return 0
        cutoff = _now() - timedelta(days=max_age_days)
        with self._lock:
            self._ensure_loaded_locked()
            # Keyed by ``entry.session_key`` — the registry knows session keys, not session_ids.
            candidate_keys = [
                key for key, entry in self._entries.items()
                if not entry.suspended and entry.updated_at < cutoff
            ]
        if not candidate_keys:
            return 0
        active = self._active_session_keys_safe(context="prune")
        if active is None:
            return 0  # cannot tell which sessions are live: drop nothing this hour
        with self._lock:
            removed_keys = []
            for key in candidate_keys:
                entry = self._entries.get(key)
                # Re-read under the lock: a turn that started while the registry was being
                # scanned has to be able to save itself from this prune.
                if entry is None or entry.suspended or entry.updated_at >= cutoff:
                    continue
                if entry.session_key in active:
                    continue
                removed_keys.append(key)
                self._entries.pop(key, None)
        if removed_keys:
            self._save()
            logger.info("SessionStore pruned %d entries older than %d days",
                        len(removed_keys), max_age_days)
        return len(removed_keys)
