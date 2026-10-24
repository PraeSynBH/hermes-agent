"""Admission bound on a repo's worktree population (``worktrees.max_linked``, #PRX-6132).

Arrival, not reclamation, is what has to be bounded: a linked worktree is born on every
``hermes -w`` launch and every isolated subagent, and the archive-first sweep can only act on trees
idle past its age floor. These run against real git repos: at the cap one idle dirty tree is
archived and removed so the new tree fits, a cap with nothing reclaimable refuses the creation and
leaves the population alone, ``reclaim_before_admit: false`` refuses without recycling, and an
unset ``max_linked`` keeps the old unbounded creation path byte for byte.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest


def _git(args, cwd):
    result = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=str(cwd),
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"git {args}: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    _git(["init", "-q", "-b", "main", str(root)], tmp_path)
    (root / ".gitignore").write_text(".worktrees/\n", encoding="utf-8")
    (root / "a.txt").write_text("one\n", encoding="utf-8")
    _git(["add", "-A"], root)
    _git(["commit", "-qm", "init"], root)
    (root / ".worktrees").mkdir()
    return root


def _tree(repo, name, *, dirty=False):
    path = repo / ".worktrees" / name
    _git(["worktree", "add", "-q", "-b", f"wt/{name}", str(path), "HEAD"], repo)
    if dirty:
        (path / "notes.md").write_text("uncommitted\n", encoding="utf-8")
    return path


def _idle(path, hours=100):
    """Age the tree and its activity (files, HEAD, reflog) past the 24 h Hermes-made tier."""
    stamp = time.time() - hours * 3600
    git_dir = Path(_git(["rev-parse", "--absolute-git-dir"], path))
    for item in [*path.rglob("*"), git_dir / "HEAD", git_dir / "logs" / "HEAD", path]:
        if item.exists():
            os.utime(item, (stamp, stamp), follow_symlinks=False)


def _linked(repo):
    out = _git(["worktree", "list", "--porcelain"], repo)
    return len([b for b in out.split("\n\n") if b.strip()]) - 1


def _set_worktrees_config(body):
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(f"worktrees:\n{body}", encoding="utf-8")


def _archives(repo):
    return [line for line in _git(["for-each-ref", "--format=%(refname)",
                                   "refs/hermes/archive/"], repo).splitlines() if line]


@pytest.fixture(autouse=True)
def _isolate_host_process_scan(monkeypatch):
    """The reclaim pass scans every live process's cwd; on a dev host some of them sit in the real
    ``~/.hermes``, which the test harness refuses to read (the pre-existing sweep suite fails the
    same way on such a host). The live-process gate itself is covered by
    ``test_worktree_snapshot_prune``; what these tests bound is admission."""
    from hermes_cli import worktree_sweep

    monkeypatch.setattr(worktree_sweep, "live_process_pids", lambda paths, exclude=(): {})


def test_creation_recycles_one_idle_tree_and_never_exceeds_the_cap(repo, capsys):
    from hermes_cli import worktree_pool
    from hermes_cli.worktree_ops import _setup_worktree

    worktree_pool.reset_cache()
    _set_worktrees_config("  max_linked: 2\n  archive_dirty: true\n")
    reclaimable = _tree(repo, "hermes-idle", dirty=True)
    _tree(repo, "hermes-young")
    _idle(reclaimable)
    assert _linked(repo) == 2

    info = _setup_worktree(str(repo), sync_base=False)

    assert info is not None
    assert not reclaimable.exists(), "the idle dirty tree should have been archived and removed"
    assert len(_archives(repo)) == 1
    assert _linked(repo) == 2, "the population must never exceed the cap"
    out = capsys.readouterr().out
    assert "Worktree created" in out
    assert "(2/2 in this repo)" in out


def test_creation_is_refused_when_the_cap_is_reached_and_nothing_is_reclaimable(repo, capsys):
    from hermes_cli import worktree_pool
    from hermes_cli.worktree_ops import _setup_worktree

    worktree_pool.reset_cache()
    _set_worktrees_config("  max_linked: 1\n")
    busy = _tree(repo, "hermes-busy", dirty=True)  # young: inside its idle tier, never touched
    assert _linked(repo) == 1

    info = _setup_worktree(str(repo), sync_base=False)

    assert info is None
    assert busy.exists() and (busy / "notes.md").exists()
    assert _linked(repo) == 1, "a refused creation must not add a tree"
    assert _archives(repo) == []
    out = capsys.readouterr().out
    assert "Worktree not created" in out
    assert "1/1" in out, "the refusal names the count and the cap"
    assert "hermes worktree prune" in out, "the refusal names the manual lever"


def test_reclaim_before_admit_false_refuses_without_recycling(repo, capsys):
    from hermes_cli import worktree_pool
    from hermes_cli.worktree_ops import _setup_worktree

    worktree_pool.reset_cache()
    _set_worktrees_config("  max_linked: 1\n  reclaim_before_admit: false\n")
    idle = _tree(repo, "hermes-idle", dirty=True)
    _idle(idle)

    assert _setup_worktree(str(repo), sync_base=False) is None

    assert idle.exists(), "reclaim_before_admit: false must not run the sweep"
    assert _archives(repo) == []
    assert "1/1" in capsys.readouterr().out


def test_unset_max_linked_keeps_the_unbounded_creation_path(repo, capsys):
    """No ``max_linked``: no count, no reclaim, and the creation line has no population suffix."""
    from hermes_cli import worktree_pool
    from hermes_cli.worktree_ops import _setup_worktree

    worktree_pool.reset_cache()
    _set_worktrees_config("  archive_dirty: true\n")
    idle = _tree(repo, "hermes-idle", dirty=True)
    _idle(idle)

    info = _setup_worktree(str(repo), sync_base=False)

    assert info is not None
    assert idle.exists(), "the unbounded path never reclaims at admission"
    assert _archives(repo) == []
    assert _linked(repo) == 2
    out = capsys.readouterr().out
    assert f"Worktree created: {info['path']}" in out
    assert "in this repo" not in out


def test_subagent_tree_is_refused_at_the_cap_and_degrades_to_the_parent_cwd(repo):
    """The other creation path (delegation isolation) is bounded by the same gate, and a refusal
    degrades the child rather than failing it."""
    from hermes_cli import worktree_pool
    from tools.subagent_worktree import create_subagent_worktree

    worktree_pool.reset_cache()
    _set_worktrees_config("  max_linked: 1\n")
    _tree(repo, "subagent-busy")

    assert create_subagent_worktree(str(repo), "child-1") is None
    assert _linked(repo) == 1


def test_settings_read_the_registered_keys_and_coerce_them(repo):
    """``sweep_settings`` is the one reader of ``worktrees.*``; both keys answer from config."""
    from hermes_cli.worktree_sweep import sweep_settings

    _set_worktrees_config("  max_linked: '3'\n  reclaim_before_admit: false\n")
    settings = sweep_settings()
    assert settings["max_linked"] == 3 and isinstance(settings["max_linked"], int)
    assert settings["reclaim_before_admit"] is False
    _set_worktrees_config("  archive_dirty: true\n")
    settings = sweep_settings()
    assert settings["max_linked"] == 0, "absent max_linked is unbounded"
    assert settings["reclaim_before_admit"] is True
