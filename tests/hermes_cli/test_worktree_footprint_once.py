"""The worktree footprint audit is a per-process cost, not a per-profile one (PRA-6637).

The shared daemon runs the 6-hourly pruner once per SERVED PROFILE
(``cron/scheduler.py:_maybe_run_worktree_maintenance`` ticks every profile home), so an unmemoised
audit walks the same ``<repo>/.worktrees`` once per profile: 47 walks of 149 trees inside the
2026-10-20 22:08 boot, each a near-timeout ``du`` (~17-22 s on this host, warm cache), all of them
concurrent. That disk pressure is the boot window's only >10 s event-loop stall.

Contract: one ``worktrees_summary`` walk per repo root per PROCESS, and one warning.
"""

import threading

import pytest

from hermes_cli import worktree_gc, worktree_ops


@pytest.fixture
def repo(tmp_path):
    """A repo root with a ``.worktrees/`` dir — all the audit itself inspects."""
    root = tmp_path / "repo"
    (root / ".worktrees" / "one").mkdir(parents=True)
    (root / ".worktrees" / "two").mkdir()
    return root


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch):
    """Each test starts un-audited: the daemon audits a repo root once per process."""
    monkeypatch.setattr(worktree_ops, "_footprint_audited", set(), raising=False)


def _counting_walk(monkeypatch):
    """Patch the walk (``du``) itself; returns the repo roots that were walked."""
    walked = []

    def fake_summary(repo_root):
        walked.append(str(repo_root))
        return 12, 2048

    monkeypatch.setattr(worktree_gc, "worktrees_summary", fake_summary)
    return walked


def test_concurrent_profile_passes_walk_the_repo_once(repo, monkeypatch, caplog):
    walked = _counting_walk(monkeypatch)
    passes = [threading.Thread(target=worktree_ops._warn_worktree_footprint, args=(str(repo),))
              for _ in range(8)]

    for thread in passes:
        thread.start()
    for thread in passes:
        thread.join(timeout=30)

    assert len(walked) == 1, f"{len(walked)} footprint walks of one repo root (want 1 per process)"
    notice = [r for r in caplog.records if ".worktrees/ holds" in r.getMessage()]
    assert len(notice) == 1, f"{len(notice)} footprint warning(s) (want exactly one)"


def test_one_walk_per_repo_root_across_repo_spellings(repo, tmp_path, monkeypatch):
    """A symlinked path is the same repo: the realpath is the memo key."""
    link = tmp_path / "repo-link"
    link.symlink_to(repo, target_is_directory=True)
    walked = _counting_walk(monkeypatch)

    worktree_ops._warn_worktree_footprint(str(repo))
    worktree_ops._warn_worktree_footprint(str(link))

    assert len(walked) == 1, walked


def test_n_served_profiles_walk_once_per_repo(repo, monkeypatch):
    """The daemon's own caller: one cron tick per served profile home, N profile scopes."""
    import hermes_constants
    from cron import scheduler as sched

    walked = _counting_walk(monkeypatch)

    # The pruner's own pass (git sweeps, install-state reclaim) is out of scope here; it reaches
    # the audit first thing, which is the call this contract is about.
    def pass_only_the_audit(repo_root):
        worktree_ops._warn_worktree_footprint(str(repo_root))

    monkeypatch.setattr(worktree_ops, "_prune_stale_worktrees", pass_only_the_audit)
    monkeypatch.setattr(sched, "_last_worktree_maintenance_at", {})
    monkeypatch.setattr(sched, "_worktree_maintenance_repos", lambda: [str(repo)])

    homes = iter([str(repo.parent / f"profile-{i}") for i in range(47)])
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: next(homes))

    spawned = []
    real_thread = threading.Thread

    class RecordingThread(real_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            spawned.append(self)

    monkeypatch.setattr(sched.threading, "Thread", RecordingThread)
    for _ in range(47):
        sched._maybe_run_worktree_maintenance()
    for thread in spawned:
        thread.join(timeout=30)

    assert len(spawned) == 47, "each served profile home must still get its own prune pass"
    assert len(walked) == 1, f"{len(walked)} footprint walks across 47 profiles (want 1 per repo)"
