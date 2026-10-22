"""The footprint size probe measures per tree, so a big tree still gets a size (PRA-6783).

One ``du -sm`` over the whole ``.worktrees`` parent is O(every inode under it) against one fixed
20 s deadline, so on a 29-tree / 11.5 GB host it expired on every boot (17.2 s warm, 22.4 s bare)
and ``size_mb`` came back None — 722 of 746 boot notices read "(unknown size)", which left the size
half of the escalation gate (``count >= 10 or size_mb >= 5120``) dead on exactly the hosts it is
for. Per-tree probes cost about the same total work; a stalled tree now costs its own timeout.

Invariant: a ``.worktrees/`` whose whole-parent walk would exceed the timeout still reports a size,
or says "size probe timed out" in words — never a silent None.
"""

import logging
import os
import subprocess

import pytest

from hermes_cli import worktree_gc, worktree_ops


@pytest.fixture
def repo(tmp_path):
    """A repo root whose ``.worktrees/`` holds three trees with real bytes in them."""
    root = tmp_path / "repo"
    for name in ("one", "two", "three"):
        tree = root / ".worktrees" / name
        tree.mkdir(parents=True)
        (tree / "blob.bin").write_bytes(b"x" * (8 * 1024))
    return root


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch):
    monkeypatch.setattr(worktree_ops, "_footprint_audited", set(), raising=False)


def _record_du_targets(monkeypatch):
    """Every path ``du`` was invoked on, delegating to the real subprocess."""
    targets = []
    real_run = worktree_gc._run

    def recording_run(cmd, timeout, cwd=None, env=None):
        if cmd and cmd[0] == "du":
            targets.append(os.path.realpath(cmd[-1]))
        return real_run(cmd, timeout, cwd, env)

    monkeypatch.setattr(worktree_gc, "_run", recording_run)
    return targets


def _du_that_stalls_for(monkeypatch, name):
    """Make ``du`` on the named tree (or every tree, when ``name`` is None) time out."""
    real_run = worktree_gc._run

    def run(cmd, timeout, cwd=None, env=None):
        if cmd and cmd[0] == "du" and (name is None or os.path.basename(cmd[-1]) == name):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)
        return real_run(cmd, timeout, cwd, env)

    monkeypatch.setattr(worktree_gc, "_run", run)


def test_size_probe_walks_each_tree_not_the_parent(repo, monkeypatch):
    """The unbounded whole-parent walk is the bug: it must never be issued."""
    targets = _record_du_targets(monkeypatch)

    footprint = worktree_gc.worktrees_footprint(str(repo))

    assert os.path.realpath(str(repo / ".worktrees")) not in targets, targets
    assert footprint.count == 3
    assert footprint.size_mb is not None and footprint.size_mb >= 3
    assert footprint.unmeasured == 0
    assert footprint.size_text().endswith("GB")


def test_one_stalled_tree_does_not_blank_the_total(repo, monkeypatch):
    """The old shape reported None whenever the walk expired; one slow tree is not the answer."""
    _du_that_stalls_for(monkeypatch, "two")

    footprint = worktree_gc.worktrees_footprint(str(repo))

    assert (footprint.count, footprint.unmeasured) == (3, 1)
    assert footprint.size_mb is not None, "a stalled tree must not blank every other tree's size"
    assert footprint.size_mb >= 2
    assert "unmeasured" in footprint.size_text()


def test_unmeasurable_tree_says_so_in_words(repo, monkeypatch):
    """Nothing measurable: an explicit timeout, never a silent None."""
    _du_that_stalls_for(monkeypatch, None)

    footprint = worktree_gc.worktrees_footprint(str(repo))

    assert footprint.count == 3
    assert footprint.size_mb is None
    assert footprint.size_text() == "size probe timed out"


def test_size_gate_fires_under_the_count_threshold(repo, monkeypatch, caplog):
    """9 trees at ~1 GB each is 9 GB of reclaimable disk with the count gate still silent."""
    monkeypatch.setattr(worktree_gc, "worktrees_footprint",
                        lambda repo_root: worktree_gc.Footprint(9, 9216))

    with caplog.at_level(logging.WARNING):
        worktree_ops._warn_worktree_footprint(str(repo))

    notice = [r.getMessage() for r in caplog.records if ".worktrees/ holds" in r.getMessage()]
    assert notice, "the size half of the gate must raise the notice on its own"
    assert "9 tree(s)" in notice[0] and "9.0GB" in notice[0]
