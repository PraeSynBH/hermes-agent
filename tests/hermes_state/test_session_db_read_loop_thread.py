"""The gateway's event loop must never take a read-pool permit.

Regression for PRA-6517. The permit path is synchronous descriptor work: it lists
``/dev/fd`` for headroom, opens a read-only connection when a permit is free, and on a
permit miss closes an idle pooled connection to reclaim one. On 2026-10-19 the shared
gateway's loop thread was sampled inside exactly that last path --

    hermes_state.py:938 _read_ctx -> :876 _get_read_conn -> readpool:207 acquire
      -> :222 _acquire_process_permit -> :137 _reclaim_idle_read_conn_anywhere
      -> :235 reclaim_idle -> hermes_state.py:909 _evict_one_idle_read_conn
      -> :578 _close_conn_logged -> sqlite_safe_read.py:113 close

-- and the loop reported ``stalled 46.4s`` from 23:08Z. ``sqlite3.close()`` also runs
under ``sqlite_safe_read._live_lock``, a process-global lock, so a slow close on the loop
serialises every other thread's connection open/close as well.

The contract: on a loop thread ``acquire()`` refuses before any of that work happens, so
``_get_read_conn()`` returns None and the caller takes its documented degradation (the
writer connection under the lock). Off a loop thread the pool is unchanged -- still
reclaims an idle peer and still opens. Asserting both directions keeps the fix from
degenerating into "the pool is disabled".
"""

import asyncio
import threading

import pytest

import hermes_state_readpool as readpool
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session(session_id="s1", source="cli", model="m")
    yield d
    d.close()


def _on_a_loop() -> bool:
    """The predicate the fix keys on, resolved from asyncio rather than from the module
    under test, so the test can be run against a tree where the helper does not exist
    yet (red-before) and still assert behaviour."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _warm_pool(db) -> None:
    """Leave one idle read connection in the pool, holding one process permit."""
    with db._read_ctx():
        pass
    assert db._read_pool.qsize() == 1


def _seize_every_permit() -> int:
    """Take all free process permits so the next acquire must reclaim. Returns the count
    taken (the caller must give them back before closing the SessionDB)."""
    taken = 0
    while readpool._process_read_permits.acquire(blocking=False):
        taken += 1
    return taken


def _release(n: int) -> None:
    for _ in range(n):
        readpool._process_read_permits.release()


@pytest.mark.requires_wal
def test_a_read_on_the_event_loop_refuses_instead_of_reclaiming_inline(db, monkeypatch):
    monkeypatch.setattr(readpool, "_fd_headroom_ok", lambda: True)
    _warm_pool(db)

    closed = []
    real_close = db._close_conn_logged

    def spy(conn, why):
        closed.append(threading.current_thread().name)
        return real_close(conn, why)

    monkeypatch.setattr(db, "_close_conn_logged", spy)

    seized = _seize_every_permit()
    try:
        seen = {}

        async def probe():
            seen["thread"] = threading.current_thread().name
            assert _on_a_loop() is True
            return db._get_read_conn()

        conn = asyncio.run(probe())
    finally:
        _release(seized)

    assert conn is None, "a loop-thread read opened or reclaimed a connection inline"
    assert closed == [], f"a loop thread closed a pooled connection: {closed}"
    # The peer is still pooled: nothing was evicted to make room.
    assert db._read_pool.qsize() == 1


@pytest.mark.requires_wal
def test_a_read_off_the_event_loop_still_reclaims_and_opens(db, monkeypatch):
    """The guard is loop-only — a worker thread still reclaims an idle peer for its permit."""
    monkeypatch.setattr(readpool, "_fd_headroom_ok", lambda: True)
    _warm_pool(db)

    closed = []
    real_close = db._close_conn_logged
    monkeypatch.setattr(
        db, "_close_conn_logged",
        lambda conn, why: (closed.append(threading.current_thread().name), real_close(conn, why))[1],
    )

    seized = _seize_every_permit()
    result = {}
    try:
        def worker():
            assert _on_a_loop() is False
            result["conn"] = db._get_read_conn()

        t = threading.Thread(target=worker, name="pool-reader")
        t.start()
        t.join()
    finally:
        _release(seized)

    assert result.get("conn") is not None, "an off-loop reader was refused a permit it could reclaim"
    assert closed == ["pool-reader"], f"expected the worker to evict one idle peer, saw {closed}"
    db._close_read_conn(result["conn"])
