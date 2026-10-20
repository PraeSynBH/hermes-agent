"""Dependency state of a deleted checkout is reclaimed; state anything live could still read, or
whose checkout this process merely cannot see, is kept (``pm/install_states.py``). Past the
retention floor, state whose checkout still exists is evicted once unused -- a harness that keeps a
worktree per issue leaves those behind forever."""

from __future__ import annotations

import os
import subprocess
import time

import pm.environments
import pm.install_states
import pytest
from hermes_cli.worktree_ops import _prune_stale_worktrees
from pm.filesystem import lock_fd
from pm.install_states import (
    collect_orphan_install_states,
    collect_stale_install_states,
    orphan_install_states,
    stale_install_states,
)


def _state(installs, key, project_root, *, used_ago=0.0):
    state = installs / key
    (state / "inputs").mkdir(parents=True)
    (state / "inputs" / ".project-root").write_text(str(project_root), encoding="utf-8")
    (state / "environments" / "gen" / "venv").mkdir(parents=True)
    if used_ago:
        stamp = time.time() - used_ago
        os.utime(state / "inputs", (stamp, stamp))
    return state


def test_startup_prune_reclaims_only_provably_deleted_checkouts(tmp_path, monkeypatch):
    # ``tmp_path`` itself lives under the OS temp dir; a temp root this test can point at keeps
    # the mount-boundary cases below outside it, which is what a foreign-host record looks like.
    monkeypatch.setattr(pm.install_states, "system_temp_root", lambda: tmp_path / "sys-temp")
    installs = tmp_path / "home" / "installs"
    repo = tmp_path / "repo"
    (repo / ".worktrees").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    live_root = tmp_path / "home" / "hermes-agent"
    live_root.mkdir(parents=True)
    live = _state(installs, "aaaa", live_root)
    gone_scratch = _state(installs, "bbbb", tmp_path / "home" / "cache" / "scratch" / "clone")
    gone_worktree = _state(installs, "cccc", repo / ".worktrees" / "removed-by-hand")
    # Pre-record state (no .project-root) has no evidence either way: never touched.
    unknown = installs / "dddd"
    (unknown / "environments").mkdir(parents=True)
    # A data root shared across a mount boundary records checkouts this process cannot see:
    # a host install from inside a container, a container install from the host.
    host_install = _state(installs, "eeee", tmp_path / "other-host-home" / ".hermes" / "hermes-agent")
    container_install = _state(installs, "ffff", tmp_path / "opt-hermes")

    assert orphan_install_states(installs) == [gone_scratch, gone_worktree]
    monkeypatch.setattr(pm.environments, "installs_root", lambda: installs)
    # No tree under .worktrees/ is old enough to prune: the reclaim must still run.
    _prune_stale_worktrees(str(repo))
    assert not gone_scratch.exists() and not gone_worktree.exists()
    assert all(p.is_dir() for p in (live, unknown, host_install, container_install))


def test_held_orphan_is_kept(tmp_path):
    installs = tmp_path / "installs"
    locked = _state(installs, "dddd", tmp_path / "gone-a")
    leased = _state(installs, "eeee", tmp_path / "gone-b")
    runtime = _state(installs, "ffff", tmp_path / "gone-c")

    fds = [os.open(locked / ".install.lock", os.O_CREAT | os.O_RDWR, 0o600)]
    for lease_dir in (leased / "environments" / "gen" / ".leases",
                      runtime / "pm-runtime" / "generations" / "gen" / ".leases"):
        lease_dir.mkdir(parents=True)
        fds.append(os.open(lease_dir / "reader", os.O_CREAT | os.O_RDWR, 0o600))
    assert all(lock_fd(fd, wait=False) for fd in fds)
    try:
        assert collect_orphan_install_states(installs) == []
        assert locked.is_dir() and leased.is_dir() and runtime.is_dir()
    finally:
        for fd in fds:
            os.close(fd)
    # Locks released (owner exited): all are reclaimed on the next pass.
    assert sorted(p.name for p in collect_orphan_install_states(installs)) == ["dddd", "eeee", "ffff"]


def test_run_scoped_temp_checkout_is_reclaimed_after_its_worktree_is_deleted(tmp_path, monkeypatch):
    """The harness leaves the run dir (sometimes the whole path) behind, so under the temp dir a
    missing checkout is the proof. A live worktree, and a checkout outside temp, are kept."""
    temp_root = tmp_path / "sys-temp"
    run_dir = temp_root / "paperclip-run-prx-5657-1234"
    (run_dir / "live-wt").mkdir(parents=True)
    installs = tmp_path / "home" / "installs"
    gone = _state(installs, "aaaa", run_dir / "finished-wt")
    live = _state(installs, "bbbb", run_dir / "live-wt")
    outside_temp = _state(installs, "cccc", tmp_path / "paperclip" / "prx-5657-wt")
    monkeypatch.setattr(pm.install_states, "system_temp_root", lambda: temp_root)

    assert orphan_install_states(installs) == [gone]
    assert collect_orphan_install_states(installs) == [gone]
    assert live.is_dir() and outside_temp.is_dir()


@pytest.mark.platforms("posix")
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores directory modes")
def test_partly_removed_orphan_keeps_its_record_until_retried(tmp_path):
    installs = tmp_path / "installs"
    state = _state(installs, "aaaa", tmp_path / "gone")
    stuck = state / "environments" / "gen" / "venv"
    (stuck / "pyvenv.cfg").write_text("", encoding="utf-8")
    stuck.chmod(0o500)  # its entries cannot be unlinked: the removal stops part-way
    try:
        assert collect_orphan_install_states(installs) == []
        assert orphan_install_states(installs) == [state]
    finally:
        stuck.chmod(0o700)
    assert collect_orphan_install_states(installs) == [state] and not state.exists()


def test_stale_states_past_the_retention_floor_are_evicted_oldest_first(tmp_path):
    """A checkout that keeps existing is not evidence of use: past the floor, keep the newest few."""
    installs = tmp_path / "installs"
    # The checkouts still exist: that is the whole case the deletion tiers never fire on.
    states = []
    for n in range(6):
        checkout = tmp_path / "paperclip" / f"issue-{n}-wt"
        checkout.mkdir(parents=True)
        states.append(_state(installs, f"k{n}", checkout, used_ago=72 * 3600 - n * 3600))

    # Ranked by last use: the two that went unused longest ago fall off, the newest four stay.
    assert stale_install_states(installs, keep=4, min_age_seconds=48 * 3600) == [states[1], states[0]]
    evicted = collect_stale_install_states(installs, keep=4, min_age_seconds=48 * 3600)

    assert evicted == [states[1], states[0]]
    assert all(not state.exists() for state in states[:2])
    assert all(state.is_dir() for state in states[2:])
    assert not list(installs.glob(".evicting-*"))


def test_state_used_within_the_floor_survives_the_count_cap(tmp_path):
    """The count cap bounds a host; the age floor is what keeps it from rebuilding live work."""
    installs = tmp_path / "installs"
    for n in range(3):
        _state(installs, f"k{n}", tmp_path / "wt", used_ago=3600)

    assert stale_install_states(installs, keep=1, min_age_seconds=48 * 3600) == []
    assert collect_stale_install_states(installs, keep=1, min_age_seconds=48 * 3600) == []
    assert sorted(p.name for p in installs.iterdir()) == ["k0", "k1", "k2"]


def test_the_state_this_process_runs_from_is_never_evicted(tmp_path):
    installs = tmp_path / "installs"
    checkouts = {}
    for name in ("repo", "wt-a", "wt-b"):
        checkouts[name] = tmp_path / name
        checkouts[name].mkdir()
    active = _state(installs, "k0", checkouts["repo"], used_ago=100 * 3600)
    stale = _state(installs, "k1", checkouts["wt-a"], used_ago=90 * 3600)
    recent = _state(installs, "k2", checkouts["wt-b"], used_ago=80 * 3600)

    evicted = collect_stale_install_states(installs, keep=1, min_age_seconds=48 * 3600,
                                           active=[active])

    assert evicted == [stale]
    assert active.is_dir() and recent.is_dir() and not stale.exists()


def test_held_stale_state_is_kept_until_its_owner_exits(tmp_path):
    installs = tmp_path / "installs"
    for name in ("wt-a", "wt-b", "wt-c"):
        (tmp_path / name).mkdir()
    spare = _state(installs, "k0", tmp_path / "wt-a", used_ago=100 * 3600)
    installing = _state(installs, "k1", tmp_path / "wt-b", used_ago=90 * 3600)
    running = _state(installs, "k2", tmp_path / "wt-c", used_ago=80 * 3600)

    fds = [os.open(installing / ".install.lock", os.O_CREAT | os.O_RDWR, 0o600)]
    lease_dir = running / "environments" / "gen" / ".leases"
    lease_dir.mkdir(parents=True)
    fds.append(os.open(lease_dir / "reader", os.O_CREAT | os.O_RDWR, 0o600))
    assert all(lock_fd(fd, wait=False) for fd in fds)
    try:
        assert collect_stale_install_states(installs, keep=0, min_age_seconds=48 * 3600) == [spare]
        assert installing.is_dir() and running.is_dir()
    finally:
        for fd in fds:
            os.close(fd)
    # Owners gone: the next pass evicts them, and nothing else.
    assert sorted(p.name for p in collect_stale_install_states(installs, keep=0,
                                                              min_age_seconds=48 * 3600)) == ["k1", "k2"]


@pytest.mark.platforms("posix")
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores directory modes")
def test_partly_removed_eviction_keeps_no_install_key(tmp_path):
    """A half-deleted state must not wear an install key: the next launch has to rebuild."""
    installs = tmp_path / "installs"
    (tmp_path / "wt").mkdir()
    state = _state(installs, "k0", tmp_path / "wt", used_ago=100 * 3600)
    stuck = state / "environments" / "gen" / "venv"
    (stuck / "pyvenv.cfg").write_text("", encoding="utf-8")
    stuck.chmod(0o500)
    try:
        assert collect_stale_install_states(installs, keep=0, min_age_seconds=48 * 3600) == [state]
        assert not state.exists()
        assert list(installs.glob(".evicting-*"))
    finally:
        for leftover in installs.glob(".evicting-*"):
            for path in leftover.rglob("*"):
                if path.is_dir():
                    path.chmod(0o700)
    assert collect_stale_install_states(installs, keep=0, min_age_seconds=48 * 3600) == []
    assert not list(installs.glob(".evicting-*"))


def test_startup_prune_evicts_stale_states_and_keeps_the_running_one(tmp_path, monkeypatch):
    """The launch path is the timer: without it the floor only exists in `hermes pm gc`."""
    installs = tmp_path / "home" / "installs"
    repo = tmp_path / "repo"
    (repo / ".worktrees").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    active = _state(installs, pm.environments.install_key(repo), repo, used_ago=200 * 3600)
    stale = []
    for n in range(5):
        checkout = tmp_path / "paperclip" / f"issue-{n}-wt"
        checkout.mkdir(parents=True, exist_ok=True)
        stale.append(_state(installs, f"k{n}", checkout, used_ago=100 * 3600 - n * 3600))
    monkeypatch.setattr(pm.environments, "installs_root", lambda: installs)

    _prune_stale_worktrees(str(repo))

    assert active.is_dir()
    assert not stale[0].exists() and all(state.is_dir() for state in stale[1:])


def test_startup_prune_collects_the_running_states_superseded_generations(tmp_path, monkeypatch):
    """The state a launch runs from is protected from eviction, so its own old generations -- one
    ~650 MB venv per publish -- are the tier nothing else collects on a timer."""
    installs = tmp_path / "home" / "installs"
    repo = tmp_path / "repo"
    (repo / ".worktrees").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.setattr(pm.environments, "installs_root", lambda: installs)
    state = pm.environments.install_state_dir(repo)
    superseded = state / "environments" / "gen-superseded"
    fresh = state / "environments" / "gen-fresh"
    for generation in (superseded, fresh):
        (generation / "venv").mkdir(parents=True)
        (generation / ".lease-managed").write_text("", encoding="utf-8")
    # Past the collector's day floor; the fresh one is inside it and must survive.
    old = time.time() - 2 * 24 * 3600
    os.utime(superseded / ".lease-managed", (old, old))

    _prune_stale_worktrees(str(repo))

    assert not superseded.exists()
    assert fresh.is_dir()
