"""The registry's active-work queries: one key at a time, and the whole set in one lock pass.

PRX-6075: the gateway's hourly routing-index prune asks "is this session live?" about every key it
has ever routed (18 945 on the reporting host). The per-key form cost one shared-lock acquisition
per entry — 39.0 s inside a single housekeeping tick — so the prune reads ``active_session_keys``
instead. These pin that the batch answer is the same answer, or a live session loses its protection
from the prune.
"""

from __future__ import annotations

import time

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture
def registry() -> ProcessRegistry:
    return ProcessRegistry()


def _session(sid: str, *, session_key=None, started_at=None, exited=False) -> ProcessSession:
    s = ProcessSession(
        id=sid,
        command="echo hi",
        task_id="t1",
        started_at=started_at or time.time(),
        exited=exited,
        exit_code=0 if exited else None,
        output_buffer="",
    )
    s.session_key = session_key
    return s


def test_active_session_keys_matches_the_per_key_answer(registry):
    """Key for key, the batch set must agree with ``has_active_for_session``."""
    rows = [
        _session("proc_live", session_key="gw_live"),
        _session("proc_stale", session_key="gw_stale", started_at=time.time() - 90000),
        _session("proc_exited", session_key="gw_exited", exited=True),
        _session("proc_unkeyed"),  # a process no session owns
    ]
    for s in rows:
        registry._running[s.id] = s

    assert registry.active_session_keys() == {"gw_live", "gw_stale"}
    assert registry.active_session_keys(max_active_age=86400) == {"gw_live"}
    for key in ("gw_live", "gw_stale", "gw_exited", "gw_unkeyed"):
        assert (key in registry.active_session_keys(max_active_age=86400)) \
            is registry.has_active_for_session(key, max_active_age=86400), key


def test_active_session_keys_is_one_pass_not_one_lock_per_key(registry):
    """A thousand keys must not become a thousand lock acquisitions against the live loop."""
    for i in range(20):
        registry._running[f"proc_{i}"] = _session(f"proc_{i}", session_key=f"gw_{i}")

    acquisitions = []
    original = registry._lock

    class _Counting:
        def __enter__(self):
            acquisitions.append(1)
            return original.__enter__()

        def __exit__(self, *exc):
            return original.__exit__(*exc)

    registry._lock = _Counting()
    keys = registry.active_session_keys()
    assert keys == {f"gw_{i}" for i in range(20)}
    assert len(acquisitions) == 2, 'one pass to snapshot, one to answer — never one per key'
