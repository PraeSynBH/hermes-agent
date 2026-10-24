"""``maybe_auto_vacuum``: the prune-less VACUUM arm (PRX-6168).

A host whose state.db freelist was freed by something OTHER than the gateway's own prune (an
orphan GC, a manual delete) can never reclaim it: every external VACUUM is refused while a
long-lived gateway holds all the stores, and the gateway's own VACUUM arm needed ``pruned > 0``.
``maybe_auto_vacuum`` is that arm with no prune — reachable with ``sessions.auto_prune: false``,
because deleting session history stays opt-in. These tests drive REAL stores (real PRAGMAs, real
rewrite) so the ratio gate and the size delta are measured, not mocked.
"""

import subprocess
import sys

import pytest

from hermes_state_common import AUTO_VACUUM_MIN_FREELIST_RATIO

pytestmark = pytest.mark.platforms("posix")  # the holder scan is unavailable on Windows

_HOLDER = (
    "import sqlite3, sys\n"
    "conn = sqlite3.connect(sys.argv[1])\n"
    "conn.execute('SELECT count(*) FROM sqlite_master')\n"
    "print('ready', flush=True)\n"
    "sys.stdin.readline()\n"
)


def _fill(db, sessions: int = 6, rows: int = 20) -> None:
    db.create_session(session_id="keep", source="cli")
    db.append_message(session_id="keep", role="user", content="hi")
    for i in range(sessions):
        sid = f"bulk{i}"
        db.create_session(session_id=sid, source="cli")
        for _ in range(rows):
            db.append_message(session_id=sid, role="assistant", content="z" * 4000)
        db.end_session(sid, "done")
    db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _dense_store(tmp_path):
    """A store with almost nothing reclaimable: the ratio gate must keep VACUUM away."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    _fill(db)
    return db


def _sparse_store(tmp_path):
    """The host's shape: most pages freed by someone else, so only VACUUM gives them back."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    _fill(db)
    for i in range(6):
        db.delete_session(f"bulk{i}")
    db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return db


def test_vacuum_runs_with_no_prune_and_shrinks_the_store(tmp_path):
    """auto_prune off / pruned == 0 is not a reason to skip: the freelist is already there."""
    db = _sparse_store(tmp_path)
    try:
        before = db.logical_size_bytes()
        ratio = db._freelist_ratio()
        assert ratio is not None and ratio > AUTO_VACUUM_MIN_FREELIST_RATIO

        result = db.maybe_auto_vacuum(min_vacuum_interval_days=0)

        assert result["vacuumed"] is True
        assert result["pruned"] == 0
        assert result["closed"] == 0
        after = db.logical_size_bytes()
        assert after is not None and before is not None and after < before
        assert db._freelist_ratio() < ratio
        # Nothing was deleted on the way: history stays opt-in.
        assert db.get_session("keep") is not None
        assert db.get_meta("last_vacuum") is not None
        # No prune ran, so the prune throttle marker must stay untouched.
        assert db.get_meta("last_auto_prune") is None

        # The 30-day throttle then holds: a second startup does not rewrite again.
        again = db.maybe_auto_vacuum(min_vacuum_interval_days=30)
        assert again["vacuumed"] is False
    finally:
        db.close()


def test_vacuum_skips_while_a_foreign_process_holds_the_store(tmp_path):
    """Same admission as the prune path: a rewrite under a live holder is what refuses turns."""
    db = _sparse_store(tmp_path)
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(db.db_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        before = db.logical_size_bytes()

        result = db.maybe_auto_vacuum(min_vacuum_interval_days=0)

        assert result["vacuumed"] is False
        assert result["vacuum_skipped_holders"]
        assert db.logical_size_bytes() == before
        assert db.get_meta("last_vacuum") is None
    finally:
        if holder.poll() is None:
            holder.stdin.close()
            holder.wait(timeout=10)
        db.close()


def test_a_store_below_the_freelist_ratio_is_not_rewritten(tmp_path):
    db = _dense_store(tmp_path)
    try:
        before = db.logical_size_bytes()

        result = db.maybe_auto_vacuum(min_vacuum_interval_days=0)

        assert result["vacuumed"] is False
        assert result["freelist_ratio"] is not None
        assert result["freelist_ratio"] <= AUTO_VACUUM_MIN_FREELIST_RATIO
        assert "vacuum_skipped_holders" not in result
        assert db.logical_size_bytes() == before
        assert db.get_meta("last_vacuum") is None
    finally:
        db.close()


def test_a_failed_rewrite_is_reported_never_raised_and_not_throttled(tmp_path, monkeypatch):
    """Maintenance must never block startup, and a failure must not burn the 30-day window."""
    db = _sparse_store(tmp_path)
    try:
        def _boom():
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(db, "vacuum", _boom)

        result = db.maybe_auto_vacuum(min_vacuum_interval_days=0)

        assert result["vacuumed"] is False
        assert "disk on fire" in result["error"]
        assert db.get_meta("last_vacuum") is None
    finally:
        db.close()
