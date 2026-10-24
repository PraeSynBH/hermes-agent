"""The gateway's housekeeping VACUUM arm must be reachable with ``sessions.auto_prune: false``.

The host this fixes runs ``auto_prune: false`` (deleting history is opt-in and stays so), which made
``_housekeeping_state_db_maintenance`` return before it opened any store — so the gateway's only
VACUUM path was unreachable, and no external VACUUM can run while that long-lived process holds all
30 stores. ``sessions.auto_vacuum`` opens the arm without opening the prune, and the arm must work
with ``pruned == 0`` (the freelist was freed by an orphan GC, never by this process's prune).
"""

from pathlib import Path

import pytest

import gateway.run as gateway_run
from hermes_state_common import AUTO_VACUUM_MIN_FREELIST_RATIO

pytestmark = pytest.mark.platforms("posix")  # holder scan is unavailable on Windows


def _configure(home: Path, **sessions) -> None:
    body = "".join(f"  {key}: {value}\n" for key, value in sessions.items())
    (home / "config.yaml").write_text(
        "model:\n  provider: nous\nsessions:\n" + body, encoding="utf-8")


def _seed_store(home: Path, *, free_most_of_it: bool) -> None:
    """Real store: 6 × 20 × 4 kB of history, then (optionally) freed by a plain delete."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("keep", "cli")
        db.append_message("keep", "user", "hi")
        for i in range(6):
            sid = f"bulk{i}"
            db.create_session(sid, "cli")
            for _ in range(20):
                db.append_message(sid, "assistant", "z" * 4000)
            db.end_session(sid, "done")
        db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if free_most_of_it:
            for i in range(6):
                db.delete_session(f"bulk{i}")
            db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        db.close()


def _measure(home: Path) -> tuple[int | None, float | None]:
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        return db.logical_size_bytes(), db._freelist_ratio()
    finally:
        db.close()


@pytest.fixture
def home(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    hermes_home = fake_home / ".hermes"
    hermes_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # The hermetic conftest pins DEFAULT_DB_PATH at one sandbox store; restore the import-time
    # sentinel so an argless ``acquire()`` resolves through the profile scope under test.
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    resolved = Path(hermes_state._default_db_path())
    assert resolved.is_relative_to(tmp_path), f"unpinned store escaped the sandbox: {resolved}"
    return hermes_home


def _record_vacuum_arm(monkeypatch) -> list:
    """Run the real arm, recording its report — the ``pruned`` count is the DOD's evidence."""
    from hermes_state import SessionDB

    reports: list = []
    real = SessionDB.maybe_auto_vacuum

    def _wrapped(self, *args, **kwargs):
        report = real(self, *args, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(SessionDB, "maybe_auto_vacuum", _wrapped)
    return reports


def test_auto_vacuum_arm_reclaims_with_auto_prune_off_and_pruned_zero(home, monkeypatch):
    _configure(home, auto_prune=False, auto_vacuum=True, min_vacuum_interval_days=0)
    _seed_store(home, free_most_of_it=True)
    before, ratio = _measure(home)
    assert ratio is not None and ratio > AUTO_VACUUM_MIN_FREELIST_RATIO
    reports = _record_vacuum_arm(monkeypatch)

    gateway_run._housekeeping_state_db_maintenance()

    assert len(reports) == 1
    assert reports[0]["vacuumed"] is True
    assert reports[0]["pruned"] == 0
    after, ratio_after = _measure(home)
    assert after is not None and before is not None and after < before
    assert ratio_after is not None and ratio_after < ratio

    from hermes_state import SessionDB
    db = SessionDB(db_path=home / "state.db")
    try:
        assert db.get_session("keep") is not None, "auto_prune off must not delete history"
    finally:
        db.close()


def test_auto_vacuum_still_reclaims_when_auto_prune_runs_but_frees_nothing(home, monkeypatch):
    """``pruned > 0`` must not be a prerequisite: a prune that deletes nothing leaves the arm to run."""
    _configure(home, auto_prune=True, retention_days=90, auto_vacuum=True, min_vacuum_interval_days=0)
    _seed_store(home, free_most_of_it=True)
    reports = _record_vacuum_arm(monkeypatch)

    gateway_run._housekeeping_state_db_maintenance()

    assert len(reports) == 1
    assert reports[0]["vacuumed"] is True
    assert reports[0]["pruned"] == 0


def test_with_auto_vacuum_off_the_housekeeping_does_no_work(home, monkeypatch):
    """Today's behaviour, pinned: neither VACUUM path is even entered."""
    from hermes_state import SessionDB

    _configure(home, auto_prune=False, auto_vacuum=False)
    _seed_store(home, free_most_of_it=True)
    before, ratio = _measure(home)
    assert ratio is not None and ratio > AUTO_VACUUM_MIN_FREELIST_RATIO
    calls: list = []
    monkeypatch.setattr(SessionDB, "maybe_auto_vacuum", lambda self, **kw: calls.append("vacuum"))
    monkeypatch.setattr(
        SessionDB, "maybe_auto_prune_and_vacuum", lambda self, **kw: calls.append("prune"))

    gateway_run._housekeeping_state_db_maintenance()

    assert calls == []
    assert _measure(home)[0] == before
