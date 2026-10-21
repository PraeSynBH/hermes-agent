"""Regression: blocking I/O must not run while session_store._lock is held.

``get_or_create_session`` previously held the store lock during SQLite
SELECTs (``_is_session_ended_in_db``), a full routing-index rewrite +
``os.fsync`` (``_save``), and a recovery DB query
(``_recover_session_from_db``) -- all on every inbound message.

These tests assert those three I/O calls are invoked *outside* the lock.
They follow the mock-DB idiom from ``test_session_store_runtime_stale_guard``.
"""
import json
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _TrackedLock:
    """Drop-in replacement for ``threading.Lock`` that tracks hold state.

    Used to assert that blocking I/O runs only when the lock is released.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._held = False

    def acquire(self, *a, **kw):
        r = self._lock.acquire(*a, **kw)
        if r:
            self._held = True
        return r

    def release(self):
        self._held = False
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *a):
        self.release()

    @property
    def held(self) -> bool:
        return self._held


def _db_with_rows(rows: dict) -> MagicMock:
    """Mock SessionDB where ``get_session`` maps session_id -> row dict."""
    db = MagicMock()
    db.get_session.side_effect = lambda sid: rows.get(sid)
    db.find_latest_gateway_session_for_peer.return_value = None
    db.reopen_session.return_value = None
    db.create_session.return_value = None
    # Identity compression tip (no child session).
    db.get_compression_tip.side_effect = lambda sid: sid
    return db


def _make_store(tmp_path, db_mock=None) -> SessionStore:
    """Build a SessionStore with a ``_TrackedLock``, bypassing disk load."""
    config = GatewayConfig()
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    if db_mock is not None:
        store._db = db_mock
    store._loaded = True
    store._lock = _TrackedLock()
    return store


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="12345",
    )


def _seed_entry(store, key, session_id) -> SessionEntry:
    now = datetime.now()
    entry = SessionEntry(
        session_key=key,
        session_id=session_id,
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=1),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    store._entries[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStaleCheckOutsideLock:
    def test_is_session_ended_not_holding_lock(self, tmp_path):
        """``_is_session_ended_in_db`` must run with the lock released."""
        source = _source()
        db = _db_with_rows({
            "sid_alive": {"end_reason": None, "id": "sid_alive"},
        })
        store = _make_store(tmp_path, db)
        key = store._generate_session_key(source)
        _seed_entry(store, key, "sid_alive")

        lock = store._lock
        calls_under_lock = []

        orig = store._is_session_ended_in_db

        def tracking(sid):
            if lock.held:
                calls_under_lock.append(sid)
            return orig(sid)

        store._is_session_ended_in_db = tracking  # type: ignore[method-assign]

        store.get_or_create_session(source)

        assert not calls_under_lock, (
            f"_is_session_ended_in_db called {len(calls_under_lock)} "
            f"time(s) while lock was held"
        )


class TestSaveOutsideLock:
    def test_save_not_holding_lock(self, tmp_path):
        """``_save`` must run with the lock released."""
        source = _source()
        db = _db_with_rows({})
        store = _make_store(tmp_path, db)

        lock = store._lock
        save_calls_under_lock = []

        orig_save = store._save_entries

        def tracking_save():
            if lock.held:
                save_calls_under_lock.append(True)
            orig_save()

        store._save_entries = tracking_save  # type: ignore[method-assign]

        # force_new bypasses the existing-entry path, goes straight to create.
        store.get_or_create_session(source, force_new=True)

        assert not save_calls_under_lock, (
            f"_save called {len(save_calls_under_lock)} time(s) "
            f"while lock was held"
        )


class TestRecoverOutsideLock:
    def test_recover_not_holding_lock(self, tmp_path):
        """``_recover_session_from_db`` must run with the lock released."""
        source = _source()
        db = _db_with_rows({})
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_recovered",
            "started_at": datetime.now().timestamp(),
        }
        store = _make_store(tmp_path, db)
        # No entry seeded -- forces the recovery path.

        lock = store._lock
        recover_calls_under_lock = []

        orig = store._query_recoverable_session

        def tracking(**kw):
            if getattr(lock, "held", False):
                recover_calls_under_lock.append(True)
            return orig(**kw)

        store._query_recoverable_session = tracking  # type: ignore[method-assign]

        store.get_or_create_session(source)

        assert not recover_calls_under_lock, (
            f"_recover_session_from_db called "
            f"{len(recover_calls_under_lock)} time(s) while lock was held"
        )


def test_concurrent_same_key_returns_one_published_session(tmp_path):
    """Concurrent first messages for one routing key must converge on one ID."""
    source = _source()
    db = _db_with_rows({})
    store = _make_store(tmp_path, db)
    owner_started = threading.Event()
    release_owner = threading.Event()
    original_query = store._query_recoverable_session

    def synchronized_query(**kwargs):
        owner_started.set()
        assert release_owner.wait(timeout=10)
        return original_query(**kwargs)

    store._query_recoverable_session = synchronized_query  # type: ignore[method-assign]
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(store.get_or_create_session, source)
        assert owner_started.wait(timeout=10)
        follower = pool.submit(store.get_or_create_session, source)
        release_owner.set()
        entries = [owner.result(timeout=10), follower.result(timeout=10)]

    key = store._generate_session_key(source)
    assert entries[0] is entries[1]
    assert entries[0].session_id == store._entries[key].session_id
    created_ids = {call.kwargs["session_id"] for call in db.create_session.call_args_list}
    assert created_ids == {entries[0].session_id}


def test_auto_reset_does_not_recover_session_being_ended(tmp_path):
    source = _source()
    db = _db_with_rows({})
    store = _make_store(tmp_path, db)
    key = store._generate_session_key(source)
    old = _seed_entry(store, key, "old-session")
    old.suspended = True
    db.find_latest_gateway_session_for_peer.return_value = {
        "id": old.session_id,
        "session_key": key,
        "started_at": old.created_at.timestamp(),
    }

    entry = store.get_or_create_session(source)

    assert entry.session_id != old.session_id
    assert entry.was_auto_reset is True
    db.reopen_session.assert_not_called()
    # Auto-reset now writes through promote_to_session_reset (upgrades
    # accidental agent_close/ws_orphan_reap ends) with the specific
    # auditable reason — a suspended session resets as "suspended".
    db.promote_to_session_reset.assert_called_once_with(
        old.session_id, "suspended"
    )
    db.end_session.assert_not_called()


class TestWholeIndexSaveOutsideLock:
    """PRX-5789: the whole-index write left ``_lock``.

    ``_save`` serializes every entry, writes ``gateway_routing`` and fsyncs sessions.json. Held
    under ``_lock`` on a loaded host that is an unbounded hold, and the loop's
    ``restore_local_session`` (a turn boundary, per local session per turn) waits on that lock.
    """

    def test_a_full_save_persists_with_the_store_lock_released(self, tmp_path):
        store = _make_store(tmp_path, _db_with_rows({}))
        source = _source()
        store.get_or_create_session(source, force_new=True)
        key = store._generate_session_key(source)
        held_during_persist = []

        def persisting(data, generation):
            held_during_persist.append(store._lock.held)

        store._persist_routing_data = persisting  # type: ignore[method-assign]

        assert store.set_session_metadata(key, "probe", "saved") is True

        assert held_during_persist == [False], (
            "the index was persisted while SessionStore._lock was held"
        )

    @pytest.mark.asyncio
    @pytest.mark.allow_real_home_io  # importing gateway.run activates the checkout's install root; this test writes only tmp_path
    async def test_a_loop_side_restore_does_not_wait_for_a_persisting_save(self, tmp_path, monkeypatch):
        """The frame PRX-5780 could not close: a restore runs on the owner loop, inside the claim
        gate, and must not wait on ``_lock`` even while another thread's save is persisting."""
        from gateway import run
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        from gateway.session_authority import initialize_session_authority
        from gateway.session_contract import Principal
        from gateway.session_local import create_local_session
        from gateway.session_local_recovery import restore_local_session

        monkeypatch.setattr(run, "_load_gateway_config", lambda: {"platform_toolsets": {"cli": []}})
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
        authority = await initialize_session_authority(runner, profile_id="fixture", instance_id="first")
        actor = Principal("owner", "fixture", frozenset({"session:create", "session:read"}), "socket")
        ref = create_local_session(authority, actor, {
            "request_id": "r", "source": "gui", "cwd": str(tmp_path), "model": "frozen", "toolsets": []})
        route = authority.sessions[ref.session_id].route

        persisting = threading.Event()
        release = threading.Event()
        real_persist = store._persist_routing_data

        def blocked(data, generation):
            persisting.set()
            release.wait(timeout=10)
            real_persist(data, generation)

        monkeypatch.setattr(store, "_persist_routing_data", blocked)
        saver = threading.Thread(target=store.set_session_metadata, args=(route, "probe", "saving"), daemon=True)
        # Safety valve: on a store that persists under ``_lock`` the restore below cannot return
        # until this lets go, and the assertion after it says so instead of hanging the file.
        valve = threading.Timer(5.0, release.set)
        valve.start()
        saver.start()
        try:
            assert persisting.wait(timeout=10), "the save never reached its persistence step"
            started = time.monotonic()

            async def turn_boundary():
                restore_local_session(authority, ref.session_id)

            await asyncio.create_task(turn_boundary())
            elapsed = time.monotonic() - started
            # The event is the assertion; the bound only catches a restore that hung some other way.
            assert not release.is_set(), (
                "the restore waited for a save that was still persisting: it held SessionStore._lock"
            )
            assert elapsed < 5.0, f"the loop's restore took {elapsed:.2f}s behind a persisting save"
            assert store.get_session_metadata(route, "probe") == "saving"
        finally:
            valve.cancel()
            release.set()
            saver.join(timeout=10)

    def test_a_cold_load_prunes_and_persists_with_the_lock_released(self, tmp_path):
        """A load that decides the index must be rewritten (a stale route pruned) performs that write
        with ``_lock`` released. A cold load runs inside a caller's critical section — the same
        ``restore_local_session`` block that PRX-5789's dump names — and the whole-index rewrite it
        owes is the same unbounded I/O as any full save."""
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore

        now = datetime.now()
        stale = SessionEntry(
            session_key="agent:main:telegram:dm:1", session_id="sid_ended",
            created_at=now - timedelta(hours=2), updated_at=now - timedelta(hours=1),
            platform=Platform.TELEGRAM, chat_type="dm",
        )
        db = MagicMock()
        db.load_gateway_routing_entries.return_value = {stale.session_key: json.dumps(stale.to_dict())}
        db.get_session.return_value = {"id": "sid_ended", "end_reason": "session_reset"}
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        store._db = db  # pins the routing handle: no real state.db is touched

        held_at_write = []
        real_persist = store._persist_routing_data

        def checking(data, generation):
            held_at_write.append(store._lock.locked())
            real_persist(data, generation)

        store._persist_routing_data = checking  # type: ignore[method-assign]
        store._ensure_loaded()

        assert stale.session_key not in store._entries, "the ended route was not pruned"
        assert held_at_write, "the load's rewrite never ran"
        assert not any(held_at_write), "a load-time save ran while SessionStore._lock was held"
