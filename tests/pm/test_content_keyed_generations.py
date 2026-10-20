"""Dependency trees are keyed by their content, so a checkout that is new *as a path* still starts
warm: the state directory stays per checkout (lock, input stamps, facts.json, .project-root), while
the expensive trees -- the application venv and the test environment -- are stored once per
uv.lock/extras/interpreter identity and referenced by every checkout whose inputs hash to it.

This is what a run-scoped worktree needs. The harness hands every launch a fresh path
(``$TMPDIR/paperclip-run-*/<issue>-wt``, ``.worktrees/<issue>``, ``~/paperclip/<issue>-wt``), and
the install key hashes that path, so every launch used to materialise a ~750 MB environment
(368 MB venv + 381 MB test environment, measured 2026-10-20) that the next launch could not reuse.
"""

from __future__ import annotations

import json
import os
import time

import pm.environment
import pm.environments
import pm.generation_store
import pm.native_build
import pm.operations
import pm.packages
import pm.plugins_state
import pm.workspace
import pytest
from pm.filesystem import lock_fd
from pm.generation_store import COMPLETE_MARKER, collect_unreferenced, store_root
from pm.install_states import collect_stale_install_states


@pytest.fixture
def installs(tmp_path, monkeypatch):
    """A private data root: every tree, state and record lands under ``tmp_path``."""
    root = tmp_path / "home" / "installs"
    root.mkdir(parents=True)
    monkeypatch.setattr(pm.environments, "installs_root", lambda: root)
    return root


def _checkout(path):
    path.mkdir(parents=True)
    (path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (path / "pyproject.toml").write_text("[project]\nname = \"hermes\"\n", encoding="utf-8")
    return path


def _fake_venv_build(monkeypatch, builds):
    """Fake the three seams between ``Venv.apply`` and uv; record every tree it builds."""

    class Environment:
        def __init__(self, destination):
            self.destination = destination
            self.env = {}

        def create(self):
            builds.append(self.destination)
            self.destination.mkdir(parents=True)
            (self.destination / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")

        def check(self):
            pass

    def lock_and_sync(_members, _extras, *, root, **_kwargs):
        root.mkdir(parents=True)
        (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    monkeypatch.setattr(pm.environment, "managed_environment",
                        lambda destination, **_kwargs: Environment(destination))
    monkeypatch.setattr(pm.workspace, "lock_and_sync", lock_and_sync)
    monkeypatch.setattr(pm.plugins_state, "enabled_plugins_ordered", lambda **_kwargs: None)
    monkeypatch.setattr(pm.native_build, "source_build_environment", lambda _project: {})
    monkeypatch.setattr(pm.environments, "flush_before_selecting", lambda: None)


def _stamped(monkeypatch, stamp):
    monkeypatch.setattr(pm.packages.Venv, "expected_stamp", lambda self, extras, **kwargs: stamp)


def test_two_checkouts_with_the_same_lock_install_one_tree(tmp_path, monkeypatch, installs):
    """The whole point: the second checkout does not pay for the first one's install."""
    builds = []
    _fake_venv_build(monkeypatch, builds)
    _stamped(monkeypatch, "same-content")
    first, second = _checkout(tmp_path / "wt-a"), _checkout(tmp_path / "wt-b")

    a = pm.packages.Venv(first).apply(["all"], plugin_dirs=[])
    b = pm.packages.Venv(second).apply(["all"], plugin_dirs=[])

    assert a["environment"] == b["environment"], "identical content must select one tree"
    assert a["environment"].parent.parent == (store_root() / pm.generation_store.VENV)
    assert builds == [a["environment"]], "only the first checkout builds"
    assert (a["environment"].parent / COMPLETE_MARKER).is_file()
    assert a["resolved_lock"].is_file()


def test_a_different_lock_gets_its_own_tree(tmp_path, monkeypatch, installs):
    """Two checkouts on different dependency sets must never share a mutable venv."""
    builds = []
    _fake_venv_build(monkeypatch, builds)
    first, second = _checkout(tmp_path / "wt-a"), _checkout(tmp_path / "wt-b")

    a = pm.packages.Venv(first).apply(["all"], plugin_dirs=[])
    _stamped(monkeypatch, "different-content")
    b = pm.packages.Venv(second).apply(["all"], plugin_dirs=[])

    assert a["environment"] != b["environment"]
    assert len(builds) == 2


def test_a_selected_shared_tree_is_the_install_environment(tmp_path, monkeypatch, installs):
    """``selected_venv`` is the single reader of the selection: it must accept a shared tree."""
    from pm.environments import runtime_facts_path, selected_venv

    shared = store_root() / pm.generation_store.VENV / "digest" / "venv"
    shared.mkdir(parents=True)
    (shared / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    repo = _checkout(tmp_path / "wt")
    state = pm.environments.install_state_dir(repo)
    state.mkdir(parents=True)
    runtime_facts_path(repo).write_text(
        json.dumps({"schema": 1, "packages": {"venv": {"environment": str(shared), "stamp": "digest", "extras": []}}}),
        encoding="utf-8")

    assert selected_venv(repo) == shared.resolve()

    # ... and it must still refuse a path that is neither this state's nor the shared store's.
    outside = tmp_path / "elsewhere" / "venv"
    outside.mkdir(parents=True)
    (outside / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    runtime_facts_path(repo).write_text(
        json.dumps({"schema": 1, "packages": {"venv": {"environment": str(outside), "stamp": "digest", "extras": []}}}),
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="outside this install"):
        selected_venv(repo)


def _state_with_tree(installs, key, checkout, tree, *, used_ago=200 * 3600):
    state = installs / key
    (state / "inputs").mkdir(parents=True)
    (state / "inputs" / ".project-root").write_text(str(checkout), encoding="utf-8")
    (state / "facts.json").write_text(
        json.dumps({"schema": 1, "packages": {"venv": {"environment": str(tree / "venv"),
                                                      "stamp": tree.name, "extras": []}}}),
        encoding="utf-8")
    stamp = time.time() - used_ago
    os.utime(state / "inputs", (stamp, stamp))
    return state


def _tree(kind, key, *, age_hours=200.0, leased=False):
    tree = store_root() / kind / key
    (tree / "venv").mkdir(parents=True)
    (tree / "venv" / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    marker = tree / COMPLETE_MARKER
    marker.write_text("", encoding="utf-8")
    stamp = time.time() - age_hours * 3600
    os.utime(marker, (stamp, stamp))
    if leased:
        (tree / ".leases").mkdir()
        fd = os.open(tree / ".leases" / "reader", os.O_CREAT | os.O_RDWR, 0o600)
        lock_fd(fd, wait=False)
    return tree


def test_a_tree_is_released_when_no_checkout_selects_it(tmp_path, installs):
    """Three worktrees, one tree: it lives while any state names it, and goes when the last does."""
    repo_a, repo_b = _checkout(tmp_path / "wt-a"), _checkout(tmp_path / "wt-b")
    shared = _tree(pm.generation_store.VENV, "shared")
    _state_with_tree(installs, "aaaa", repo_a, shared)
    _state_with_tree(installs, "bbbb", repo_b, shared)
    spare = _tree(pm.generation_store.VENV, "spare")

    assert collect_unreferenced(installs) == [spare], "only the tree nothing selects is released"
    assert shared.is_dir()

    # The consumers go (one deleted run worktree at a time): the tree they were keeping is
    # released on the pass after the last of them, and never before.
    import shutil

    shutil.rmtree(installs / "bbbb")
    assert collect_unreferenced(installs) == []
    assert shared.is_dir(), "one checkout still selects it"
    shutil.rmtree(installs / "aaaa")
    assert collect_unreferenced(installs) == [shared]


def test_a_leased_tree_is_kept_even_when_unreferenced(tmp_path, installs):
    """A process that booted into the tree holds a lease for its lifetime; a publish is not proof."""
    leased = _tree(pm.generation_store.VENV, "running", leased=True)
    legacy = _tree(pm.generation_store.VENV, "legacy", age_hours=1)

    assert collect_unreferenced(installs) == []
    assert leased.is_dir() and legacy.is_dir(), "one is leased, the other is younger than the floor"


@pytest.mark.platforms("posix")
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores directory modes")
def test_a_partly_released_tree_keeps_no_key(tmp_path, installs):
    """A half-deleted tree must not be adoptable: the next launch has to build a fresh one."""
    tree = _tree(pm.generation_store.VENV, "stuck")
    (tree / "venv" / "blocker").write_text("", encoding="utf-8")
    (tree / "venv").chmod(0o500)
    try:
        # The key went first, so the entry is no longer adoptable by name even though its bytes
        # are still on disk; the leftover is swept once it can be removed.
        assert collect_unreferenced(installs) == [tree]
        assert not tree.exists()
        assert list((store_root() / pm.generation_store.VENV).glob(".evicting-*"))
        assert pm.generation_store.complete_entry(pm.generation_store.VENV, "stuck") is None
    finally:
        for leftover in (store_root() / pm.generation_store.VENV).glob(".evicting-*"):
            for path in leftover.rglob("*"):
                if path.is_dir():
                    path.chmod(0o700)


def test_a_state_whose_shared_tree_is_leased_is_not_evicted(tmp_path, installs):
    """Eviction takes the record the next launch reads; a live reader of its tree outranks it."""
    checkout = _checkout(tmp_path / "wt-a")
    _checkout(tmp_path / "wt-b")
    leased = _tree(pm.generation_store.VENV, "running", leased=True)
    _state_with_tree(installs, "aaaa", checkout, leased)
    assert collect_stale_install_states(installs, keep=0) == []
    assert (installs / "aaaa").is_dir()


def _fake_side_build(monkeypatch, builds, base_python):
    from pm.operations import _python

    class Lock:
        def __init__(self, *_args, **_kwargs):
            pass

        def version(self, _name):
            return "3.14.0"

        def artifacts(self, _name, _target):
            return [{"sha256": "a" * 64}]

    def build_environment(*, source, out, python=None, **_kwargs):
        builds.append(out)
        python = _python(out)
        python.parent.mkdir(parents=True)
        python.touch()
        (out / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
        return python

    monkeypatch.setattr("pm._uv._toolchain", lambda **_kwargs: (None, base_python))
    monkeypatch.setattr("pm.lock.Lockfile", Lock)
    monkeypatch.setattr("pm.store.current_target", lambda: "linux-x64")
    monkeypatch.setattr(pm.operations, "build_environment", build_environment)
    monkeypatch.setattr(pm.environments, "flush_before_selecting", lambda: None)


def test_the_test_environment_is_built_once_for_every_checkout_on_one_lock(tmp_path, monkeypatch, installs):
    """The 381 MB suite interpreter is the other half of a launch's cost, and equally content-only."""
    base = tmp_path / "tools" / "python" / "bin" / "python"
    base.parent.mkdir(parents=True)
    base.touch()
    builds = []
    _fake_side_build(monkeypatch, builds, base)
    first, second = _checkout(tmp_path / "wt-a"), _checkout(tmp_path / "wt-b")
    roots = {checkout: pm.environments.install_state_dir(checkout) / "test-environment"
             for checkout in (first, second)}

    a = pm.operations.ensure_project_environment("test-environment", first, extras=["all"],
                                                 groups=["dev"], root=roots[first], explicit=True,
                                                 shared=True)
    b = pm.operations.ensure_project_environment("test-environment", second, extras=["all"],
                                                groups=["dev"], root=roots[second], explicit=True,
                                                shared=True)

    assert a == b, "identical test dependencies must select one interpreter"
    assert len(builds) == 1
    assert pm.operations.environment_python("test-environment", root=roots[first]) == a
    assert pm.operations.environment_python("test-environment", root=roots[second]) == a
    assert a.parent.parent.parent in pm.generation_store.references(installs), "both records name the tree"

    # A different lock is a different identity: the suite gets its own tree.
    (second / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    c = pm.operations.ensure_project_environment("test-environment", second, extras=["all"],
                                                groups=["dev"], root=roots[second], explicit=True,
                                                shared=True)
    assert c != a and len(builds) == 2
