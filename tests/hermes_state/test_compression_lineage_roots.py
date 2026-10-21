"""Regression for PRA-6631: many lineage roots in one read, with the per-id walk's answers.

``logical_owner`` resolves a route's admission identity through ``get_compression_lineage``, which
costs two or more synchronous reads per id. A caller with a routing index of thousands of sessions
paid that per entry on the event loop. ``compression_lineage_roots`` answers the same question for
any number of ids from ONE read; this file pins the equivalence on every edge the lineage walk has
(a plain chain, a branch, a reset fork, a tool child, a non-compression parent, an id not in the
store) and pins the statement count.
"""
from __future__ import annotations

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _chain(db: SessionDB, root: str, length: int, *, branch_at=None, reset_at=None, tool_at=None):
    """A compression chain root -> ... (each parent ended by compression, per the lineage edge)."""
    db.create_session(root, source="webui")
    parent = root
    for i in range(1, length):
        sid = f"{root}-c{i}"
        db.end_session(parent, "compression")
        if i == branch_at:
            db.create_session(sid, source="webui", parent_session_id=parent,
                              model_config={"_branched_from": parent})
        elif i == reset_at:
            db.create_session(sid, source="webui", parent_session_id=parent,
                              model_config={"_reset_from": parent})
        elif i == tool_at:
            db.create_session(sid, source="tool", parent_session_id=parent)
        else:
            db.create_session(sid, source="webui", parent_session_id=parent)
        parent = sid


def _tree(db: SessionDB):
    _chain(db, "plain", 4)
    _chain(db, "forked", 4, branch_at=2)
    _chain(db, "resetting", 5, reset_at=3)
    _chain(db, "tooled", 4, tool_at=2)
    # A parent that did NOT end by compression is not a lineage edge: the child keeps its own id.
    db.create_session("open-parent", source="webui")
    db.create_session("open-child", source="cli", parent_session_id="open-parent")
    return ["plain", "plain-c1", "plain-c2", "plain-c3",
            "forked", "forked-c1", "forked-c2", "forked-c3",
            "resetting", "resetting-c1", "resetting-c2", "resetting-c3", "resetting-c4",
            "tooled", "tooled-c1", "tooled-c2", "tooled-c3",
            "open-parent", "open-child", "absent"]


@pytest.mark.parametrize("batch", [1, 3, 1000], ids=["one-at-a-time", "few", "all-at-once"])
def test_roots_match_the_per_id_lineage_for_every_edge(db: SessionDB, batch: int) -> None:
    ids = _tree(db)
    batched = {}
    for start in range(0, len(ids), batch):
        batched.update(db.compression_lineage_roots(ids[start:start + batch]))

    assert batched == {sid: (db.get_compression_lineage(sid) or [sid])[0] for sid in ids}


def test_a_thousands_strong_index_costs_one_read(db: SessionDB, monkeypatch) -> None:
    for i in range(2000):
        sid = f"bulk-{i:04d}"
        db.create_session(sid, source="webui")
        if i % 7 == 0:  # every seventh conversation was compressed once
            db.end_session(sid, "compression")
            db.create_session(f"{sid}-c1", source="webui", parent_session_id=sid)
    index = [f"bulk-{i:04d}" for i in range(2000)] + [f"bulk-{i:04d}-c1" for i in range(0, 2000, 7)]

    reads = []
    real_read = db._read_all

    def counting_read(sql, params=()):
        reads.append(sql)
        return real_read(sql, params)
    monkeypatch.setattr(db, "_read_all", counting_read)

    roots = db.compression_lineage_roots(index)

    assert len(reads) == 1, "the whole routing index must be one statement"
    assert roots["bulk-0007-c1"] == "bulk-0007"
    assert roots["bulk-0001"] == "bulk-0001"
    # The per-id walk this replaced: two or more statements per id, on the caller's thread.
    db.compression_lineage_roots(index[:2])
    assert len(reads) == 2
