"""Per-repo admission caps and the ratchet (``worktrees.max_linked_by_repo``, #PRX-6144).

One global number has to be the number the *largest* repo tolerates, so it refuses nothing in the
small ones. These run against real git repos: a repo's own entry binds while the global number
stays unbounded, an absolute path beats a basename for the same repo, an unnamed repo still falls
back to ``max_linked``, and the ratchet — measured at sweep time, never at admission — brings a
repo's cap down to ``population + ratchet_headroom`` while never coming back up on its own.
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


def _tree(repo, name):
    path = repo / ".worktrees" / name
    _git(["worktree", "add", "-q", "-b", f"wt/{name}", str(path), "HEAD"], repo)
    return path


def _set_worktrees_config(body):
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(f"worktrees:\n{body}", encoding="utf-8")


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    """No cached counts, no snapshot from a previous test in the profile home."""
    from hermes_cli import worktree_pool, worktree_sweep

    monkeypatch.setattr(worktree_sweep, "live_process_pids", lambda paths, exclude=(): {})
    worktree_pool.reset_cache()
    path = worktree_pool.ratchet_path()
    if path.exists():
        path.unlink()
    yield
    worktree_pool.reset_cache()
    for leftover in (path, path.with_name(path.name + ".lock")):
        if leftover.exists():
            leftover.unlink()


def _settings():
    from hermes_cli.worktree_pool import pool_settings

    return pool_settings()


def test_a_repo_cap_binds_while_the_global_number_stays_unbounded(repo):
    """``max_linked: 0`` (no global bound) plus one named repo: only that repo is capped."""
    from hermes_cli import worktree_pool

    _set_worktrees_config(f"  max_linked: 0\n  max_linked_by_repo:\n    {repo.name}: 1\n")
    _tree(repo, "hermes-a")

    verdict = worktree_pool.admit(str(repo))

    assert verdict.cap == 1 and verdict.configured_cap == 1
    assert verdict.count == 1 and not verdict.allowed
    assert "1/1" in worktree_pool.refusal_text(verdict)


def test_an_unnamed_repo_still_falls_back_to_the_global_cap(repo):
    """Naming another repo must not change this one: it keeps ``max_linked`` exactly."""
    from hermes_cli import worktree_pool

    _set_worktrees_config("  max_linked: 1\n  max_linked_by_repo:\n    some-other-repo: 400\n")
    _tree(repo, "hermes-a")

    verdict = worktree_pool.admit(str(repo))

    assert verdict.cap == 1 and verdict.configured_cap == 1
    assert not verdict.allowed


def test_an_absolute_path_key_beats_the_basename_for_the_same_repo(repo):
    """A deliberately named path is not shadowed by a basename entry (nor the other way round)."""
    from hermes_cli import worktree_pool

    _set_worktrees_config(f"  max_linked: 9\n  max_linked_by_repo:\n"
                          f"    {repo.name}: 9\n    {repo}: 1\n")
    _tree(repo, "hermes-a")

    verdict = worktree_pool.admit(str(repo))

    assert verdict.cap == 1, "the path entry must win over the basename entry"
    assert not verdict.allowed


def test_a_per_repo_zero_keeps_its_unbounded_meaning(repo):
    """``0`` in the table is that repo's documented "no cap", not "inherit the global number"."""
    from hermes_cli import worktree_pool

    _set_worktrees_config(f"  max_linked: 1\n  max_linked_by_repo:\n    {repo.name}: 0\n")
    _tree(repo, "hermes-a")

    verdict = worktree_pool.admit(str(repo))

    assert verdict.allowed and verdict.cap == 0 and not verdict.measured


def test_the_ratchet_lowers_the_cap_to_population_plus_headroom(repo):
    """A sweep measures the population; admission then enforces the lowered number."""
    from hermes_cli import worktree_pool

    _set_worktrees_config("  max_linked: 10\n  ratchet: true\n  ratchet_headroom: 3\n")
    _tree(repo, "hermes-a")
    _tree(repo, "hermes-b")
    assert worktree_pool.admit(str(repo)).cap == 10, "no snapshot yet -> the configured number"

    cap = worktree_pool.ratchet_after_sweep(str(repo), 2, _settings())

    assert cap == 5
    verdict = worktree_pool.admit(str(repo))
    assert verdict.cap == 5 and verdict.ratcheted and verdict.configured_cap == 10
    assert verdict.allowed, "2 linked against a cap of 5 must still admit"


def test_a_refusal_says_the_cap_was_ratcheted_and_names_both_levers():
    from hermes_cli import worktree_pool

    verdict = worktree_pool.Admission(False, 5, 5, configured_cap=10, ratcheted=True)

    text = worktree_pool.refusal_text(verdict)
    assert "5/5" in text and "ratcheted down from 10" in text
    assert "max_linked_by_repo" in text and "worktrees.ratchet: false" in text


def test_the_ratchet_never_goes_back_up_and_a_changed_ceiling_resets_it(repo):
    from hermes_cli import worktree_pool

    _set_worktrees_config("  max_linked: 10\n  ratchet: true\n  ratchet_headroom: 1\n")
    assert worktree_pool.ratchet_after_sweep(str(repo), 6, _settings()) == 7
    assert worktree_pool.ratchet_after_sweep(str(repo), 2, _settings()) == 3, "the cap follows down"
    assert worktree_pool.ratchet_after_sweep(str(repo), 8, _settings()) == 3, "and never back up"

    _set_worktrees_config("  max_linked: 12\n  ratchet: true\n  ratchet_headroom: 1\n")
    assert worktree_pool.ratchet_after_sweep(str(repo), 8, _settings()) == 9, \
        "a changed ceiling opens a fresh ratchet at that value"


def test_a_snapshot_is_written_even_with_the_ratchet_off(repo):
    """The disk sweeper's tier-3 line reads the population/cap denominator from the snapshot."""
    from hermes_cli import worktree_pool
    from utils import read_json_or_empty

    _set_worktrees_config("  max_linked: 4\n  ratchet: false\n")
    _tree(repo, "hermes-a")

    assert worktree_pool.ratchet_after_sweep(str(repo), 1, _settings()) == 4
    stored = read_json_or_empty(worktree_pool.ratchet_path())["repos"][os.path.realpath(str(repo))]
    assert stored["cap"] == 4 and stored["configured"] == 4
    assert stored["population"] == 1 and stored["ratcheted"] is False
    assert worktree_pool.admit(str(repo)).cap == 4, "ratchet off -> the configured number binds"


def test_population_summary_names_the_population_and_the_cap(repo):
    from hermes_cli import worktree_pool

    _set_worktrees_config("  max_linked: 10\n  ratchet: true\n  ratchet_headroom: 1\n")
    _tree(repo, "hermes-a")
    worktree_pool.ratchet_after_sweep(str(repo), 1, _settings())

    line = worktree_pool.population_summary(str(repo))

    assert "1/2" in line and repo.name in line
    assert "ratcheted down from 10" in line, "a repo pinned below its ceiling must say so"
