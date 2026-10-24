"""A per-run worktree lock must not outlive the run that took it.

git has no liveness check on ``worktree lock``: while the reason stands, ``git worktree remove``
refuses the tree forever and ``git worktree prune`` will not even drop a registration whose
directory is already deleted. A harness that locks every run worktree and never unlocks it leaves
a pool nothing else can reap (measured: 55 dead-pid locks in one repo, every one held by a pid that
no longer existed).
"""
import os
import subprocess
import sys
from pathlib import Path

from hermes_cli import worktree_ops

GIT = ['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid']


def _repo(tmp_path):
    repo = tmp_path / 'repo'
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(repo)], check=True)
    (repo / '.gitignore').write_text('.worktrees/\n')
    subprocess.run([*GIT, '-C', str(repo), 'add', '.gitignore'], check=True)
    subprocess.run([*GIT, '-C', str(repo), 'commit', '-qm', 'fixture'], check=True)
    return repo


def _dead_pid():
    """A pid that existed and is now gone (never a hardcoded number: pid ranges differ per host)."""
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    proc.wait()
    return proc.pid


def _lock(repo, tree, reason):
    subprocess.run(['git', '-C', str(repo), 'worktree', 'lock', '--reason', reason, str(tree)],
                   check=True)


def _lock_reason(repo, tree):
    admin = repo / '.git' / 'worktrees' / Path(tree).name
    locked = admin / 'locked'
    return locked.read_text().strip() if locked.exists() else None


def _git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=False)


def test_a_run_worktree_lock_is_released_when_the_run_exits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(worktree_ops, '_cprint', lambda *_: None)  # keeps `cli` out of this test
    repo = _repo(tmp_path)
    info = worktree_ops._setup_worktree(str(repo), sync_base=False)
    assert info is not None
    tree = Path(info['path'])
    assert _lock_reason(repo, tree) == f'hermes pid={os.getpid()}'

    worktree_ops._release_worktree_locks()  # the exit net (atexit) firing

    assert _lock_reason(repo, tree) is None
    assert _git(repo, 'worktree', 'remove', str(tree)).returncode == 0


def test_startup_maintenance_clears_a_lock_whose_owner_was_killed(tmp_path):
    repo = _repo(tmp_path)
    tree = repo / '.worktrees' / 'hermes-killed'
    subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '-q', str(tree), '-b', 'hermes/killed'],
                   check=True)
    _lock(repo, tree, f'hermes pid={_dead_pid()} db=/home/someone/state.db')

    assert worktree_ops._release_dead_worktree_locks(str(repo)) == 1

    assert _lock_reason(repo, tree) is None
    assert _git(repo, 'worktree', 'remove', str(tree)).returncode == 0


def test_a_live_owners_lock_is_left_alone(tmp_path):
    repo = _repo(tmp_path)
    tree = repo / '.worktrees' / 'hermes-running'
    subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '-q', str(tree), '-b', 'hermes/running'],
                   check=True)
    _lock(repo, tree, f'hermes pid={os.getpid()}')

    assert worktree_ops._release_dead_worktree_locks(str(repo)) == 0

    assert _lock_reason(repo, tree) == f'hermes pid={os.getpid()}'


def test_a_dead_lock_on_a_deleted_directory_stops_blocking_prune(tmp_path):
    """The documented recovery is ``rm -rf <dir> && git worktree prune``; while the registration
    stays locked, prune keeps it, so the name can never be checked out again."""
    repo = _repo(tmp_path)
    tree = repo / '.worktrees' / 'hermes-deleted'
    subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '-q', str(tree), '-b', 'hermes/deleted'],
                   check=True)
    _lock(repo, tree, f'hermes pid={_dead_pid()}')
    subprocess.run(['rm', '-rf', str(tree)], check=True)
    assert _git(repo, 'worktree', 'prune').returncode == 0
    assert 'hermes-deleted' in _git(repo, 'worktree', 'list', '--porcelain').stdout

    assert worktree_ops._release_dead_worktree_locks(str(repo)) == 1

    assert 'hermes-deleted' not in _git(repo, 'worktree', 'list', '--porcelain').stdout


def test_a_retained_session_lock_survives_the_launcher_exit(tmp_path):
    """``--tui -w`` hands its tree to the session on quit; that lock is meant to outlive the pid
    that created it, so the exit net must not take it back."""
    repo = _repo(tmp_path)
    tree = repo / '.worktrees' / 'hermes-retained'
    subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '-q', str(tree), '-b', 'hermes/retained'],
                   check=True)
    worktree_ops._lock_worktree(str(repo), str(tree), f'hermes pid={os.getpid()}')

    assert worktree_ops._retain_worktree_for_session(str(repo), str(tree), 'session-1', '/state.db')
    worktree_ops._release_worktree_locks()

    assert _lock_reason(repo, tree) == 'hermes session=session-1 db=/state.db'
