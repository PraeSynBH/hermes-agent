"""The per-path SessionDB handle census attributes live handles to their opening call site.

PRA-6500 is about a multiplexed gateway holding several ``state.db`` handles per profile. An
``lsof`` count cannot drive that fix: it mixes live handles with SQLite's unix-VFS reuse list (a
closed connection's descriptor can stay parked on the inode while a peer holds POSIX locks on it),
so it overstates duplicates and never names their owner. ``handle_census`` answers both questions
from inside the process, from the same per-path registry that already budgets read connections.

These assert on the census, not on ``lsof``, for the reason
``test_session_db_read_conn_pool.py`` documents: raw descriptor counts lag the real connection
count.
"""

from hermes_state import SessionDB
from hermes_state_readpool import handle_census


def _open_writer_and_reader(path):
    """Two handles on one file; the reader is opened from this frame so the site is this test's."""
    writer = SessionDB(db_path=path)
    reader = SessionDB(db_path=path, read_only=True)
    return writer, reader


def _row_for(path):
    key = str(path.resolve())
    return next((row for row in handle_census()["detail"] if row["path"] == key), None)


def test_census_counts_live_handles_per_path_and_names_the_opening_site(tmp_path):
    path = tmp_path / "state.db"
    writer, reader = _open_writer_and_reader(path)
    try:
        row = _row_for(path)
        assert row is not None, "the census must list a path with live handles"
        assert len(row["handles"]) == 2
        assert sorted(handle["read_only"] for handle in row["handles"]) == [False, True]
        sites = {handle["site"] for handle in row["handles"]}
        assert any("_open_writer_and_reader" in site for site in sites), sites
    finally:
        reader.close()
        writer.close()

    assert _row_for(path) is None, "a closed handle must leave the census"


def test_census_is_process_wide_across_paths(tmp_path):
    first = tmp_path / "one" / "state.db"
    second = tmp_path / "two" / "state.db"
    first.parent.mkdir()
    second.parent.mkdir()
    a = SessionDB(db_path=first)
    b = SessionDB(db_path=second)
    try:
        census = handle_census()
        listed = {row["path"] for row in census["detail"]}
        assert {str(first.resolve()), str(second.resolve())} <= listed
        assert census["handles"] == sum(len(row["handles"]) for row in census["detail"])
        assert census["paths"] == len(census["detail"])
    finally:
        a.close()
        b.close()


def test_census_limit_bounds_the_detail_not_the_totals(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        bounded = handle_census(limit=0)
        assert bounded["detail"] == []
        assert bounded["paths"] >= 1
        assert bounded["handles"] >= 1
    finally:
        db.close()


def test_census_skips_a_handle_whose_open_failed(tmp_path, monkeypatch):
    """Registration happens after a successful open; a raised construction must not be listed."""
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)

    def _boom(_self):
        raise RuntimeError("no connection for you")

    monkeypatch.setattr(SessionDB, "_open_read_only", _boom)
    try:
        try:
            SessionDB(db_path=path, read_only=True)
        except RuntimeError:
            pass
        row = _row_for(path)
        assert row is not None and len(row["handles"]) == 1
    finally:
        db.close()
