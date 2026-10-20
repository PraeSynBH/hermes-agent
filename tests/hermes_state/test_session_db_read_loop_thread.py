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

PRA-6546 finishes the other side: a pooled read connection that DOES have to be closed --
``SessionDB.close()`` draining the pool, or a read whose connection was drained under it --
must not run ``sqlite_safe_read.close()`` on the loop either. Those closes go to the read
pool's single closer thread, permit and all (the permit is released after the real close so
the ceilings still bound descriptors); off a loop thread they stay inline.
"""

import asyncio
import threading

import pytest

import hermes_cli.sqlite_safe_read as safe_read
import hermes_state_readpool as readpool
from hermes_state import SessionDB
from tests.waiting import wait_until


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


# ── PRA-6546: the read conns that DO get closed must not close on the loop ──


def _spy_tracked_closes(monkeypatch) -> list:
    """Record ``(id(conn), thread ident, thread name)`` for every tracked connection close,
    the way ``tests/gateway/test_reconcile_sweeps_off_loop.py`` spies the call it cares about."""
    closes: list = []
    real_close = safe_read._TrackingMixin.close

    def spy(self):
        closes.append((id(self), threading.get_ident(), threading.current_thread().name))
        return real_close(self)

    monkeypatch.setattr(safe_read._TrackingMixin, "close", spy)
    return closes


def _close_of(closes, conn):
    for entry in closes:
        if entry[0] == id(conn):
            return entry
    return None


@pytest.mark.requires_wal
def test_a_loop_thread_read_never_evicts_a_peer_on_the_loop(tmp_path, monkeypatch):
    """The sampled shape: a loop-thread miss that would reclaim a peer's idle conn on the same
    file. Nothing may be evicted, so no tracked close runs on the loop."""
    monkeypatch.setattr(readpool, "_fd_headroom_ok", lambda: True)
    path = tmp_path / "state.db"
    peer = SessionDB(db_path=path)
    me = SessionDB(db_path=path)
    try:
        peer.create_session(session_id="peer", source="cli", model="m")
        with peer._read_ctx() as peer_conn:
            pass
        assert peer._read_pool.qsize() == 1

        closes = _spy_tracked_closes(monkeypatch)
        loop_ident = {}
        seized = _seize_every_permit()
        try:
            async def read():
                loop_ident["ident"] = threading.get_ident()
                with me._read_ctx() as conn:
                    return conn is me._conn  # degraded to the writer connection

            served_from_writer = asyncio.run(read())
        finally:
            _release(seized)
        # Snapshot before teardown: the closes below run on this (main) thread, which is also the
        # loop's ident, and would otherwise be read as close-on-the-loop.
        on_loop = [c for c in closes if c[1] == loop_ident["ident"]]
        peer_conn_close = _close_of(closes, peer_conn)
        pooled_after = peer._read_pool.qsize()
    finally:
        me.close()
        peer.close()

    assert served_from_writer, "the loop-thread read did not take its documented degradation"
    assert on_loop == [], f"a loop thread closed a tracked connection: {on_loop}"
    assert peer_conn_close is None, "the peer's idle read conn was evicted for the loop"
    assert pooled_after == 1, "the peer's idle read conn left the pool"


@pytest.mark.requires_wal
def test_close_from_the_event_loop_closes_the_pooled_read_conn_off_the_loop(db, monkeypatch):
    """``SessionDB.close()`` drains the read pool; the drain's closes run on the closer thread."""
    closes = _spy_tracked_closes(monkeypatch)
    with db._read_ctx() as pooled:
        pass
    assert db._read_pool.qsize() == 1

    loop_ident = {}

    async def closer():
        loop_ident["ident"] = threading.get_ident()
        db.close()

    asyncio.run(closer())

    entry = wait_until(
        lambda: _close_of(closes, pooled), timeout=10, what="the pooled read conn's close")
    assert entry is not None
    _, ident, name = entry
    assert ident != loop_ident["ident"], "the pooled read conn was closed on the event loop"
    assert name == readpool.CLOSER_THREAD_NAME, f"closed on {name}, not the read pool's closer thread"


@pytest.mark.requires_wal
def test_a_deferred_close_holds_its_permit_until_the_close_lands(db, monkeypatch):
    """Bookkeeping is synchronous, the close is not — and the permit follows the close, so a
    slow closer cannot let the pool overshoot its ceiling."""
    _warm_pool(db)
    release_close = threading.Event()
    close_started = threading.Event()
    real_close = db._close_conn_logged

    def slow_close(conn, why):
        if why == "read-conn":  # the writer's close stays inline; only the handoff is blocked
            close_started.set()
            assert release_close.wait(10), "the test never released the blocked close"
        return real_close(conn, why)

    monkeypatch.setattr(db, "_close_conn_logged", slow_close)

    async def closer():
        db.close()

    asyncio.run(closer())  # returns while the close is still blocked: the loop was not held

    assert close_started.wait(10), "the deferred close never started"
    assert db._read_pool.qsize() == 0, "the pool was not drained synchronously"
    seized = _seize_every_permit()
    try:
        assert seized == readpool._READ_POOL_PROCESS_MAX - 1, (
            "a deferred close released its permit before the close landed "
            f"({seized} of {readpool._READ_POOL_PROCESS_MAX} free)")
        release_close.set()
        # This test holds every other permit, so the deferred close returning its one shows as 1.
        assert wait_until(
            lambda: _free_permits() == 1, timeout=10,
            what="the deferred close to return its permit")
    finally:
        _release(seized)
    assert _free_permits() == readpool._READ_POOL_PROCESS_MAX


def _free_permits() -> int:
    """How many process permits are free right now (takes and gives each one back)."""
    free = _seize_every_permit()
    _release(free)
    return free
