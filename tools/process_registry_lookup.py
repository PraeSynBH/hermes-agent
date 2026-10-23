"""Session lookup for the process registry: exact, memory-only, and by unique prefix.

``get`` is the general entry point — two in-memory stores, then the retained-results fallback on
disk, then a unique-prefix scan. ``peek`` is the memory-only read for a caller that already holds
the full id: the gateway's process watcher polls it every tick, and the thread hop it used to pay
for that read (a pool submit, a new OS thread when the pool is saturated, and the GIL hand-back on
the loop) is what starved the loop for 16-50 s at a time (PRA-6825/PRA-6826).

``ProcessLookupMixin`` needs the registry's ``self._lock``, ``self._running``, ``self._finished``
and ``self._refresh_detached_session``.
"""

from __future__ import annotations

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
