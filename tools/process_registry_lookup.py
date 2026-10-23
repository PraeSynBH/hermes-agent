"""Session lookup for the process registry: exact, memory-only, and by unique prefix.

``get`` is the general entry point — two in-memory stores, then the retained-results fallback on
disk, then a unique-prefix scan. ``peek`` is the memory-only read for a caller that already holds
the full id: the gateway's process watcher polls it every tick, and the thread hop it used to pay
for that read (a pool submit, a new OS thread when the pool is saturated, and the GIL hand-back on
the loop) is what starved the loop for 16-50 s at a time (PRA-6825/PRA-6826).

``ProcessLookupMixin`` needs the registry's ``self._lock``, ``self._running``, ``self._finished``
and ``self._refresh_detached_session``. It also owns the active-work queries — one key
(:meth:`has_active_for_session`) and the whole set in one lock pass (:meth:`active_session_keys`).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from tools.process_registry_results import load_completed_results

if TYPE_CHECKING:
    from tools.process_registry import ProcessSession


class ProcessLookupMixin:
    """Look a session up by full id, in memory only, or by a unique id prefix."""

    # Minimum suffix chars for prefix resolution; "p"/"proc_1" are too collision-prone.
    _MIN_PREFIX_CHARS = 4

    def get(self, session_id: str) -> Optional[ProcessSession]:
        """Session by full ID or unique prefix (``proc_4dae`` / bare ``4dae``, like git
        short hashes); ambiguous or too-short prefixes resolve to None, never a guess."""
        if not isinstance(session_id, str) or not session_id:
            return None
        with self._lock:
            session = self._running.get(session_id) or self._finished.get(session_id)
        if session is None:
            session = load_completed_results(session_id).get(session_id)
        return self._refresh_detached_session(session if session is not None else self._resolve_prefix(session_id))

    def peek(self, full_id: str) -> Optional[ProcessSession]:
        """Exact in-memory lookup: the two stores and nothing else — no retained-results scan, no
        prefix resolution, no filesystem. ``None`` means "not in memory", not "gone".

        A caller polling the id it armed wants this one: a dict read with no thread hop in front of
        it. Only a miss needs :meth:`get`, which is the path that touches the filesystem."""
        if not isinstance(full_id, str) or not full_id:
            return None
        with self._lock:
            session = self._running.get(full_id) or self._finished.get(full_id)
        return self._refresh_detached_session(session)

    def has_active_for_session(self, session_key: str, max_active_age: Optional[float] = None) -> bool:
        """Active processes for a gateway session key. Processes older than
        ``max_active_age`` seconds are ignored as stale so a forgotten ``http.server``
        can't freeze session idle/daily reset forever; ``None`` keeps legacy behaviour
        (any running process blocks)."""
        now = time.time()
        return self._any_running(
            lambda s: s.session_key == session_key
            and (max_active_age is None or (now - s.started_at) < max_active_age))

    def active_session_keys(self, max_active_age: Optional[float] = None) -> set[str]:
        """Session keys with a live process — the batch form of :meth:`has_active_for_session`.

        A caller holding thousands of keys must not pay a lock round-trip *and* a scan per key: the
        gateway's routing-index prune walks every session key it has ever routed (18 945 on the host
        that filed PRX-6075) once an hour, and the per-key form made that tick one shared-lock
        acquisition per entry — 39.0 s inside a single tick, while every other tick of the same
        watcher ran in ~2.5 ms. One pass answers all of them.
        """
        now = time.time()
        with self._lock:
            sessions = list(self._running.values())
        for session in sessions:
            self._refresh_detached_session(session)
        with self._lock:
            return {
                s.session_key for s in self._running.values()
                if s.session_key and not s.exited
                and (max_active_age is None or (now - s.started_at) < max_active_age)
            }

    def _resolve_prefix(self, session_id: str) -> Optional[ProcessSession]:
        """Resolve a unique session-ID prefix (a bare hex tail is normalized to
        ``proc_<tail>``); :meth:`get` tries exact first."""
        query = session_id.strip() if isinstance(session_id, str) else ""
        if not query:
            return None
        if not query.startswith("proc_"):
            query = f"proc_{query}"
        if len(query) - len("proc_") < self._MIN_PREFIX_CHARS:
            return None
        matches = load_completed_results(query)
        with self._lock:
            matches.update({
                sid: s for store in (self._running, self._finished)
                for sid, s in store.items() if sid.startswith(query)
            })
        return next(iter(matches.values())) if len(matches) == 1 else None
