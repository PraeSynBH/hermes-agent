"""Reclaim per-checkout dependency state whose checkout is gone.

Dependency state lives under ``<home>/installs/<install_key>`` where the key hashes the
checkout path (:func:`pm.environments.install_key`). Every ``hermes -w`` worktree, review tree
or scratch clone that boots the agent therefore commits a full environment (venv, test venv,
PM runtime, ~200 MB each), and nothing removed it when the tree was deleted: one developer
host carried 402 such orphans (80 GB) after a fortnight of worktree campaigns.

The state dir records the checkout it belongs to in ``inputs/.project-root`` (written by
:func:`pm.environments.record_activation_inputs`). A missing path alone does not prove the
checkout is gone: a data root shared with a container (``-v ~/.hermes:/opt/data``) or an
unmounted volume records paths this process cannot see. So a dir is an orphan only when its
checkout sat where this process can see that it was deleted -- under the data root that owns
``installs/`` (scratch clones, the runtime checkout), directly in a ``.worktrees/`` dir that
still exists, or under the OS temp dir. Removal still fails closed on anything a live process
could hold: the install lock, or a dependency / PM runtime generation lease.

The temp tier is what bounds agent hosts: a harness that gives every run a fresh worktree in
``$TMPDIR`` commits a full environment per run, and deleting the worktree at the end of the run
left the state behind forever -- one ~750 MB tree per launch, none of them reclaimable under the
data-root and ``.worktrees`` rules alone. Temp is the one parent that is not evidence: a harness
routinely leaves the run dir (sometimes the whole path) sitting empty, so a missing root there is
the proof, and a live run still holds the lock or a lease.

Deletion evidence does not bound a host whose checkouts *stay*: a harness that keeps one worktree
per issue at a stable path (``~/paperclip/<issue>-wt``), or a developer's long-lived clone outside
the data root, records a checkout that exists forever, so the tiers above never fire and every one
of them pins a full environment. That tail is bounded by age and count instead: keep the newest
:data:`KEPT_INSTALL_STATES` states seen within :data:`STALE_INSTALL_STATE_AGE` and evict the rest.
Evicting a state is safe rather than destructive -- the next launch in that checkout materialises a
fresh one -- and a state only counts as evictable when nothing still runs from it: a process that
booted in the checkout holds a generation lease for its lifetime
(:func:`pm.environments.activate_dependencies`), and an install in flight holds the install lock.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

#: States kept regardless of age (the runtime checkout, the trees a campaign is working in).
KEPT_INSTALL_STATES = 4
#: How long a state must have gone unused before eviction may touch it at all.
STALE_INSTALL_STATE_AGE = 48 * 3600.0


def system_temp_root() -> Path:
    """The OS temp dir run-scoped checkouts record under (resolved, so ``/var`` matches ``/private/var``)."""
    return Path(tempfile.gettempdir()).resolve()


def _run_scoped_temp_checkout(root: Path) -> bool:
    """True when *root* was a checkout under the OS temp dir and no longer exists."""
    try:
        under_temp = root.resolve().is_relative_to(system_temp_root())
    except (OSError, RuntimeError):
        return False
    return under_temp and not root.exists()


def _provably_deleted(root: Path, data_root: Path) -> bool:
    if root.exists():
        return False
    return (root.is_relative_to(data_root)
            or (root.parent.name == ".worktrees" and root.parent.is_dir())
            or _run_scoped_temp_checkout(root))


def orphan_install_states(installs: Path) -> list[Path]:
    """State dirs whose recorded checkout is provably deleted. Dirs without a record are left alone."""
    if not installs.is_dir():
        return []
    data_root = installs.resolve().parent
    orphans: list[Path] = []
    for state in sorted(installs.iterdir()):
        record = state / "inputs" / ".project-root"
        if not state.is_dir() or not record.is_file():
            continue
        try:
            root = record.read_text(encoding="utf-8-sig").strip()
        except OSError:
            continue
        if root and _provably_deleted(Path(root), data_root):
            orphans.append(state)
    return orphans


def _held(state: Path) -> bool:
    """True while a process may still be using this state dir: the install lock is taken or a
    generation lease is held. Any lock we cannot even open counts as held."""
    from hermes_cli.runtime_state import leases_held
    from pm.filesystem import lock_fd

    lock = state / ".install.lock"
    if lock.exists():
        try:
            fd = os.open(lock, os.O_RDWR)
        except OSError:
            return True
        try:
            if not lock_fd(fd, wait=False):
                return True
        finally:
            os.close(fd)
    for generations in (state / "environments", state / "pm-runtime" / "generations"):
        if generations.is_dir():
            for generation in generations.iterdir():
                if generation.is_dir() and (generation / ".leases").is_dir() and leases_held(generation):
                    return True
    return False


def _last_used(state: Path) -> float:
    """When pm last resolved this state, from the stamp a successful sync rewrites.

    ``inputs/`` is rewritten on every install *and* no-op sync, so a cache hit that writes nothing
    else still moves it. A state with no ``inputs/`` is pre-record or half-built; its own mtime is
    the only signal, and a fresh one is young enough to be kept by the age floor either way.
    """
    for marker in (state / "inputs", state / "facts.json", state):
        try:
            return marker.stat().st_mtime
        except OSError:
            continue
    return 0.0


def stale_install_states(installs: Path, *, keep: int = KEPT_INSTALL_STATES,
                         min_age_seconds: float = STALE_INSTALL_STATE_AGE,
                         active: Iterable[Path] = ()) -> list[Path]:
    """States of checkouts that still exist but have gone unused, past the newest *keep*.

    Ordered by last use, so the ties a host accumulates during one campaign fall off oldest first.
    *active* (the state the calling process runs from) is never a candidate: evicting it under a
    live interpreter is the one removal no lease would necessarily prevent.
    """
    if not installs.is_dir():
        return []
    protected = {Path(path).resolve() for path in active}
    states = [state for state in installs.iterdir()
              if state.is_dir() and not state.name.startswith(".")
              and state.resolve() not in protected]
    if len(states) <= keep:
        return []
    now = time.time()
    ranked = sorted(states, key=_last_used, reverse=True)
    return [state for state in ranked[max(keep, 0):] if now - _last_used(state) >= min_age_seconds]


def _sweep_evictions(installs: Path) -> None:
    """Finish a removal whose pass died mid-``rmtree``; the state key is already gone by then."""
    for leftover in installs.glob(".evicting-*"):
        if leftover.is_dir():
            shutil.rmtree(leftover, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                leftover.unlink()


def collect_stale_install_states(installs: Path, *, keep: int = KEPT_INSTALL_STATES,
                                 min_age_seconds: float = STALE_INSTALL_STATE_AGE,
                                 active: Iterable[Path] = ()) -> list[Path]:
    """Evict unheld stale states beyond the retention floor; returns the states evicted.

    Each state is renamed out of its install-key name before it is removed. A removal that stops
    part-way (an open file on Windows, a read-only dir) therefore leaves nothing that still looks
    like a usable state: the next launch in that checkout builds a fresh one instead of reading a
    half-deleted venv, and the leftover is swept by the following pass.
    """
    removed: list[Path] = []
    if not installs.is_dir():
        return removed
    _sweep_evictions(installs)
    for state in stale_install_states(installs, keep=keep, min_age_seconds=min_age_seconds,
                                      active=active):
        if _held(state):
            logger.debug("stale install state %s is still held; skipped", state.name)
            continue
        trash = state.with_name(f".evicting-{state.name}-{os.getpid()}")
        try:
            state.rename(trash)
        except OSError:
            logger.debug("stale install state %s could not be taken; retried next pass", state.name)
            continue
        shutil.rmtree(trash, ignore_errors=True)
        if trash.exists():
            logger.debug("evicted install state %s only partly removed; retried next pass", state.name)
        removed.append(state)
    return removed


def collect_orphan_install_states(installs: Path) -> list[Path]:
    """Remove every unheld orphan state dir under *installs*; returns what was removed."""
    removed: list[Path] = []
    for state in orphan_install_states(installs):
        if _held(state):
            logger.debug("orphan install state %s is still held; skipped", state.name)
            continue
        # The record goes last, and only once everything else is gone: the startup prune runs on
        # a daemon thread and a removal can fail part-way (an open file on Windows, a read-only
        # dir), and a dir that lost its record first would never be recognised again.
        for child in state.iterdir():
            if child.name == "inputs":
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):
                    child.unlink()
        if any(child.name != "inputs" for child in state.iterdir()):
            logger.debug("orphan install state %s only partly removed; retried next pass", state.name)
            continue
        shutil.rmtree(state, ignore_errors=True)
        if not state.exists():
            removed.append(state)
    return removed
