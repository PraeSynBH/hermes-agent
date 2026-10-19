"""Resolving where an installed package lives is memoized per store state.

Regression for PRA-6502: the terminal PATH builder calls ``_installed_location`` on every
spawn, and for chromium that resolution walks the whole browser bundle (``entry.rglob``).
In a multiplexed gateway (one interpreter, one GIL, ~40 profiles' agents spawning terminals
continuously) that repeated walk starved the event loop: control requests answered in 3–18 s
and every model-connection handshake timed out. The contract pinned here is the fix's whole
premise — a second resolution with an unchanged store performs no directory walk, and a
reinstall (facts.json rewritten) is never served from the old resolution.
"""
from __future__ import annotations

import os
from pathlib import Path

import pm.install
import pm.paths as pm_paths
import pytest
from pm.lock import Facts, Lockfile
from pm.packages import Chromium
from pm.store import current_target, tree_digest


def _chromium_store(tmp_path: Path, monkeypatch, *, entry: str = "chromium-1208"):
    """A real store with one installed chromium entry and its facts/lockfile wired in."""
    shipped, writable = tmp_path / "tools", tmp_path / "home" / "tools"
    target, digest = current_target(), "1" * 64
    package, version = Chromium(), "1208+140.0"
    lock, facts = Lockfile(tmp_path / "lock.json"), Facts(shipped / "facts.json")
    lock.set_pin(package.name, version, {target: {"url": "https://test.invalid", "sha256": digest}})
    entry_dir = shipped / entry
    binary = entry_dir / "chrome-mac-arm64" / "Google Chrome for Testing.app" / "Contents" / "MacOS" / "chrome"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"pinned chromium")
    (entry_dir / "chrome-mac-arm64" / "intermediate").mkdir()
    facts.record(package.name, version, entry_dir.name, package.env(entry_dir, target), shipped,
                 target=target, artifacts=[digest], digest=tree_digest(entry_dir))
    lock.save()
    monkeypatch.setattr(pm_paths, "store_root", lambda: shipped)
    monkeypatch.setattr(pm_paths, "writable_store_root", lambda: writable)
    monkeypatch.setattr(pm_paths, "facts_path", lambda: shipped / "facts.json")
    monkeypatch.setattr(pm_paths, "lockfile_path", lambda: lock.path)
    getattr(pm.install, "clear_location_cache", lambda: None)()
    return package, lock, target, entry_dir


@pytest.fixture
def counts_scandir(monkeypatch):
    """Count every directory scan the process performs (rglob/glob/iterdir go through it)."""
    real, seen = os.scandir, {"n": 0}

    def counting(path="."):
        seen["n"] += 1
        return real(path)

    monkeypatch.setattr(os, "scandir", counting)
    return seen


def test_a_second_resolution_of_an_unchanged_store_walks_nothing(tmp_path, monkeypatch, counts_scandir):
    package, lock, target, entry_dir = _chromium_store(tmp_path, monkeypatch)

    first = pm.install._installed_location(package, lock, target)
    assert first is not None and first[1].root == entry_dir.parent
    walks = counts_scandir["n"]
    assert walks > 0, "the first resolution must actually walk the browser bundle"

    for _ in range(3):
        again = pm.install._installed_location(package, lock, target)
        assert again is not None and again[1].root == first[1].root
    assert counts_scandir["n"] == walks


def test_a_reinstall_is_never_served_the_previous_resolution(tmp_path, monkeypatch):
    package, lock, target, entry_dir = _chromium_store(tmp_path, monkeypatch)
    first = pm.install._installed_location(package, lock, target)
    assert first is not None and first[1].root == entry_dir.parent

    # Reinstall into a new entry and rewrite facts.json the way an install does. The pinned
    # version is unchanged, so only the store state can tell the two apart.
    package, lock, target, replacement = _chromium_store(tmp_path, monkeypatch, entry="chromium-1209")
    moved = pm.install._installed_location(package, lock, target)
    assert moved is not None
    facts, store = moved
    fact = facts.get("chromium")
    assert fact is not None
    assert store.entry(fact["entry"]) == replacement
