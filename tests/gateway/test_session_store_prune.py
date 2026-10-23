"""Tests for SessionStore.prune_old_entries and the gateway watcher that calls it.

The SessionStore in-memory dict (and its backing sessions.json) grew
unbounded — every unique (platform, chat_id, thread_id, user_id) tuple
ever seen was kept forever, regardless of how stale it became.  These
tests pin the prune behaviour:

  * Entries older than max_age_days (by updated_at) are removed
  * Entries marked ``suspended`` are preserved (user-paused)
  * Entries with an active process attached are preserved
  * max_age_days <= 0 disables pruning entirely
  * sessions.json is rewritten with the post-prune dict
  * The ``updated_at`` field — not ``created_at`` — drives the decision
    (so a long-running-but-still-active session isn't pruned)
"""

import json
import threading
from datetime import datetime, timedelta
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionStore

def test_session_store_default_db_uses_runtime_hermes_home(tmp_path, monkeypatch):
    """SessionStore must honor runtime HERMES_HOME when opening the default DB.

    Regression for the import-time DEFAULT_DB_PATH freeze: importing
    hermes_state before a fixture redirected HERMES_HOME used to pin every
    default SessionDB() at the developer's real ~/.hermes/state.db.
    """
    config = GatewayConfig()
    fake_home = tmp_path / "alt_hermes_home"
    fake_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)

    try:
        assert store._db is not None
        assert store._db.db_path == fake_home / "state.db"
    finally:
        if store._db is not None:
            store._db.close()

def _make_store(tmp_path, max_age_days: int = 90, active_session_keys_fn=None):
    """Build a SessionStore bypassing SQLite/disk-load side effects."""
    config = GatewayConfig(

        session_store_max_age_days=max_age_days,
    )
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(
            sessions_dir=tmp_path,
            config=config,
            active_session_keys_fn=active_session_keys_fn,
        )
    store._db = None
    store._loaded = True
    return store

def _entry(key: str, age_days: float, *, suspended: bool = False,
           session_id: str | None = None) -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=key,
        session_id=session_id or f"sid_{key}",
        created_at=now - timedelta(days=age_days + 30),  # arbitrary older
        updated_at=now - timedelta(days=age_days),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        suspended=suspended,
    )

class TestPruneBasics:

    def test_prune_uses_updated_at_not_created_at(self, tmp_path):
        """A session created long ago but updated recently must be kept."""
        store = _make_store(tmp_path)
        now = datetime.now()
        entry = SessionEntry(
            session_key="long-lived",
            session_id="sid",
            created_at=now - timedelta(days=365),   # ancient
            updated_at=now - timedelta(days=3),     # but just chatted
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
        store._entries["long-lived"] = entry

        removed = store.prune_old_entries(max_age_days=30)

        assert removed == 0
        assert "long-lived" in store._entries

    def test_prune_skips_entries_with_active_processes(self, tmp_path):
        """Sessions with active bg processes aren't pruned even if old.

        The query answers in session_keys — matching what
        process_registry.active_session_keys() returns.  Prior to the fix the
        store was handed one call per entry and an implementation bug passed a
        session_id; real-world usage (via process_registry) keys on the
        session_key and never matched, so active sessions were still pruned.
        """
        active_session_keys = {"active"}

        def _active() -> set[str]:
            return active_session_keys

        store = _make_store(tmp_path, active_session_keys_fn=_active)
        store._entries["active"] = _entry(
            "active", age_days=1000, session_id="sid_active"
        )
        store._entries["idle"] = _entry(
            "idle", age_days=1000, session_id="sid_idle"
        )

        removed = store.prune_old_entries(max_age_days=90)

        assert removed == 1
        assert "active" in store._entries
        assert "idle" not in store._entries

    def test_prune_active_check_uses_session_key_not_session_id(self, tmp_path):
        """Regression guard: a query that only recognises session_ids must
        NOT protect entries during prune.  This pins the fix so a future
        refactor can't silently revert to matching session_id again.
        """
        def _recognises_only_ids() -> set[str]:
            return {"sid_active"}

        store = _make_store(tmp_path, active_session_keys_fn=_recognises_only_ids)
        store._entries["active"] = _entry(
            "active", age_days=1000, session_id="sid_active"
        )

        removed = store.prune_old_entries(max_age_days=90)

        # Entry is pruned because the callback receives "active" (session_key),
        # not "sid_active" (session_id), so _recognises_only_ids returns False.
        assert removed == 1
        assert "active" not in store._entries

    def test_prune_is_thread_safe(self, tmp_path):
        """Prune acquires _lock internally; concurrent update_session is safe."""
        store = _make_store(tmp_path)
        for i in range(20):
            age = 1000 if i % 2 == 0 else 1
            store._entries[f"s{i}"] = _entry(f"s{i}", age_days=age)

        results = []

        def _pruner():
            results.append(store.prune_old_entries(max_age_days=90))

        def _reader():
            # Mimic a concurrent update_session reader iterating under lock.
            with store._lock:
                list(store._entries.keys())

        threads = [threading.Thread(target=_pruner)]
        threads += [threading.Thread(target=_reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()

        # Exactly one pruner ran; removed exactly the 10 stale entries.
        assert results == [10]
        assert len(store._entries) == 10
        for i in range(20):
            if i % 2 == 1:  # fresh
                assert f"s{i}" in store._entries

class TestPruneCostShape:
    """PRX-6075: the hourly prune walked every routed key twice over the process registry.

    The index holds every key the gateway has ever routed (15 762 rows on the host that filed the
    card), and the prune asked the registry about each one *inside* `SessionStore._lock`. One
    hourly tick therefore cost O(keys) lock round-trips against the live process registry while
    holding the lock a turn needs to persist itself: 51 681 ms for a tick whose other arms are
    milliseconds. These pin the shape, not the number.
    """

    def test_prune_queries_the_registry_once_for_the_whole_index(self, tmp_path):
        calls = []

        def _active() -> set[str]:
            calls.append(1)
            return set()

        store = _make_store(tmp_path, active_session_keys_fn=_active)
        for i in range(2000):
            store._entries[f"s{i}"] = _entry(f"s{i}", age_days=1000)

        assert store.prune_old_entries(max_age_days=90) == 2000
        assert len(calls) == 1, 'one registry pass per prune, not one per routed key'

    def test_prune_queries_the_registry_without_holding_the_store_lock(self, tmp_path):
        """A turn persisting itself needs `_lock`; a registry scan must not hold it."""
        held = []

        def _active() -> set[str]:
            held.append(store._lock.locked())
            return set()

        store = _make_store(tmp_path, active_session_keys_fn=_active)
        store._entries["stale"] = _entry("stale", age_days=1000)

        store.prune_old_entries(max_age_days=90)
        assert held == [False]

    def test_prune_keeps_an_entry_that_became_active_during_the_registry_scan(self, tmp_path):
        """The scan happens outside the lock, so the decision is re-taken on the live entry."""
        def _active() -> set[str]:
            store._entries["stale"].updated_at = datetime.now()  # a turn just persisted
            return set()

        store = _make_store(tmp_path, active_session_keys_fn=_active)
        store._entries["stale"] = _entry("stale", age_days=1000)
        store._entries["old"] = _entry("old", age_days=1000)

        assert store.prune_old_entries(max_age_days=90) == 1
        assert "stale" in store._entries
        assert "old" not in store._entries

    def test_prune_drops_nothing_when_the_registry_cannot_answer(self, tmp_path):
        """A registry error must fail closed: keeping every entry beats pruning live sessions."""
        def _boom() -> set[str]:
            raise RuntimeError("registry unavailable")

        store = _make_store(tmp_path, active_session_keys_fn=_boom)
        store._entries["stale"] = _entry("stale", age_days=1000)

        assert store.prune_old_entries(max_age_days=90) == 0
        assert "stale" in store._entries


class TestPrunePersistsToDisk:
    def test_prune_rewrites_sessions_json(self, tmp_path):
        """After prune, sessions.json on disk reflects the new dict."""
        config = GatewayConfig(

            session_store_max_age_days=90,
        )
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = None
        # Force-populate without calling get_or_create to avoid DB side-effects
        store._entries["stale"] = _entry("stale", age_days=500)
        store._entries["fresh"] = _entry("fresh", age_days=1)
        store._loaded = True
        store._save()

        # Verify pre-prune state on disk. Filter out metadata sentinels
        # (e.g. the "_README" note) so we assert on session keys only.
        saved_pre = json.loads((tmp_path / "sessions.json").read_text())
        assert {k for k in saved_pre if not k.startswith("_")} == {"stale", "fresh"}

        # Prune and check disk.
        store.prune_old_entries(max_age_days=90)
        saved_post = json.loads((tmp_path / "sessions.json").read_text())
        assert {k for k in saved_post if not k.startswith("_")} == {"fresh"}

class TestGatewayConfigSerialization:

    def test_session_store_max_age_days_roundtrips(self):
        cfg = GatewayConfig(session_store_max_age_days=30)
        restored = GatewayConfig.from_dict(cfg.to_dict())
        assert restored.session_store_max_age_days == 30
