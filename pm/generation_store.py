"""Dependency trees shared by content, not by checkout path.

``pm.environments.install_key`` hashes the checkout *path*, so a harness that gives every run a
fresh worktree (``~/paperclip/<issue>-wt``, ``.worktrees/<issue>``, ``$TMPDIR/paperclip-run-*/``)
never hits warm state and materialises a full environment per launch. Measured on one host:
~750 MB per launch (368 MB ``environments/<gen>/venv`` + 381 MB
``test-environment/gen-*/venv``), one per launch from 23:39 to 01:10.

A tree is a pure function of its inputs -- ``uv.lock``, ``pyproject.toml``, the enabled extras,
the pinned interpreter, the member (plugin) set -- so this module stores one tree per *content*
and lets every checkout whose inputs hash to the same key reference that one tree. Everything
per-checkout stays where it was, in the path-keyed state directory: the install lock, the
input-mtime stamps, ``facts.json``, the publication journal, ``inputs/.project-root`` (which the
orphan tier keys on). A state names its tree here by absolute path, so the record a collector
reads is unchanged in shape.

What this must never weaken, and how each is kept:

* **Different content never shares a mutable tree.** The entry name *is* the content digest, so
  two checkouts on different locks cannot name the same entry. Entries are immutable once
  published: nothing ever syncs into an existing one (a repair replaces it, see
  :func:`build`).
* **A state is unusable by name after a partial removal.** A build marks an entry complete last
  (``.complete``, written after the tree is flushed); builders and readers only ever adopt a
  complete entry, and :func:`collect_unreferenced` renames an entry out of its key before it
  removes anything, exactly as the state eviction does.
* **Provenance stays reclaimable.** ``inputs/.project-root`` continues to record the checkout, so
  the orphan and count tiers still decide a state's fate; an entry survives only while some
  state's record names it or a process still leases it. Removing the state makes the entry
  unreferenced, and the next pass reclaims it -- the tree is collected because its last consumer
  went, not because a sweep guessed.

Nothing in here imports the application: it runs from the bootstrap interpreter too.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

#: Entry kinds. A kind is a namespace, so a test environment can never be mistaken for the
#: application venv that happens to carry the same digest.
VENV = "venv"
TEST_ENV = "testenv"

#: An unreferenced entry is not reclaimed before this: a build publishes its ``.complete``
#: marker before the state record that names it is written, and a crash in between must not cost
#: a finished tree. The window also covers a checkout whose record is being replaced.
MIN_AGE_SECONDS = 86400.0

#: The store's directory name under the data root's ``installs/``.
STORE_DIRECTORY = ".generations"

COMPLETE_MARKER = ".complete"
#: Lease directory prefix every lease-managed generation carries (``hermes_cli.runtime_state``).
LEASE_MARKER = ".lease-managed"


def store_root() -> Path:
    """Where shared trees live: a sibling of the install states, never inside one."""
    from pm.environments import installs_root

    return installs_root() / STORE_DIRECTORY


def entry(kind: str, key: str) -> Path:
    """The one tree for *kind*/*key*. The key is a content digest, not a path."""
    if not kind or not key or "/" in key or "\\" in key or key.startswith("."):
        raise ValueError(f"invalid dependency tree key: {kind!r}/{key!r}")
    return store_root() / kind / key


def is_complete(entry_path: Path) -> bool:
    """True when *entry_path* is a published, adoptable tree."""
    try:
        return ((entry_path / COMPLETE_MARKER).is_file()
                and (entry_path / "venv" / "pyvenv.cfg").is_file())
    except OSError:
        return False


def complete_entry(kind: str, key: str) -> Path | None:
    """The published tree for *kind*/*key*, or ``None`` when nothing may be adopted."""
    path = entry(kind, key)
    return path if is_complete(path) else None


def mark_complete(entry_path: Path) -> None:
    """Publish *entry_path*: after this, any checkout with the same content may adopt it."""
    (entry_path / COMPLETE_MARKER).write_text("", encoding="utf-8")
    # Leases (hermes_cli.runtime_state.lease_directory) only protect a generation that carries
    # this marker; without it every reader leaks a tree no collector will ever take.
    (entry_path / LEASE_MARKER).touch()


def record_origin(entry_path: Path, *, kind: str, key: str, checkout: Path) -> None:
    """Keep enough provenance for a person reading the store to know what made this tree."""
    row = {"kind": kind, "key": key, "checkout": str(checkout), "created": time.time()}
    with contextlib.suppress(OSError):
        (entry_path / "origin.json").write_text(json.dumps(row, indent=2), encoding="utf-8")


@contextlib.contextmanager
def build_lock(kind: str, key: str):
    """Serialize builders of one entry without touching the per-checkout install lock.

    Two checkouts needing the same content at the same moment must not both install into it;
    the loser of this lock finds the winner's complete entry and adopts it.
    """
    from pm.filesystem import lock_fd

    directory = store_root() / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / f"{kind}-{key}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        lock_fd(fd, wait=True)
        yield
    finally:
        os.close(fd)


def build(kind: str, key: str, run: Callable[[Path], None], *, checkout: Path | None = None,
          replace: bool = False) -> Path:
    """Publish the tree for *kind*/*key*, adopting an existing one unless *replace*.

    *replace* is the repair contract: the recorded tree is by definition the damaged one, so the
    replacement is published under a derived key -- a repair hands back a tree of its own instead
    of writing into the one a live reader may still be exec'ing, and the tree it replaced keeps
    its bytes until :func:`collect_unreferenced` sees that no checkout names it any more.
    """
    with build_lock(kind, key):
        target = entry(kind, key)
        existing = complete_entry(kind, key)
        if existing is not None and not replace:
            return existing
        if replace:
            target = entry(kind, f"{key}-repair-{uuid.uuid4().hex[:12]}")
        elif target.exists():
            # An incomplete entry is a builder that died before its complete marker, or a tree that
            # lost a file. Nothing may adopt it, and waiting out the age floor would block every
            # later launch on the same content -- but a process may still be exec'ing it, so its
            # bytes are only taken when no lease says a reader is inside.
            if leases_held(target):
                target = entry(kind, f"{key}-stale-{uuid.uuid4().hex[:12]}")
            else:
                discard(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            run(target)
            from pm.environments import flush_before_selecting

            flush_before_selecting()
            mark_complete(target)
        except BaseException:
            # Leave nothing that could be adopted by name: a half-written venv reads as a
            # missing environment, never as a current one.
            shutil.rmtree(target, ignore_errors=True)
            raise
        record_origin(target, kind=kind, key=key, checkout=checkout or Path.cwd())
        return target


def _entry_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _recorded(path: str | None) -> Path | None:
    if not isinstance(path, str) or not path:
        return None
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def references(installs: Path) -> set[Path]:
    """Every entry some checkout's record names, resolved.

    Read through the records the install already keeps, so the answer needs no refcount to
    maintain (and cannot drift): ``facts.json`` names the selected venv, and a side
    environment's ``active.json`` names its shared tree. A state whose record is unreadable
    contributes nothing -- an unreadable record is the state's own problem, and the entry it
    might have named ages into the reclaim window rather than being taken immediately.
    """
    named: set[Path] = set()
    if not installs.is_dir():
        return named
    store = (installs / STORE_DIRECTORY).resolve()
    for state in installs.iterdir():
        if not state.is_dir() or state.name.startswith(".") or state == store:
            continue
        facts = _entry_json(state / "facts.json")
        packages = facts.get("packages")
        venv = packages.get("venv") if isinstance(packages, dict) else None
        if isinstance(venv, dict):
            environment = _recorded(venv.get("environment"))
            if environment is not None and environment.is_relative_to(store):
                named.add(environment.parent)
        testenv = _recorded(_entry_json(state / "test-environment" / "active.json").get("environment"))
        if testenv is not None and testenv.is_relative_to(store):
            named.add(testenv.parent)
    return named


def leases_held(directory: Path) -> bool:
    """True while any process holds a lease in *directory* (see ``runtime_state.lease_directory``)."""
    from hermes_cli.runtime_state import leases_held as held

    return held(directory)


def _sweep_evictions(store: Path) -> None:
    """Finish a removal whose pass died mid-``rmtree``; the key was already gone by then."""
    for kind_dir in (store / VENV, store / TEST_ENV):
        if not kind_dir.is_dir():
            continue
        for leftover in kind_dir.glob(".evicting-*"):
            if leftover.is_dir():
                shutil.rmtree(leftover, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):
                    leftover.unlink()


def _set_aside(path: Path) -> Path | None:
    """Rename *path* out of its key; the caller removes what it returns.

    A removal that stops part-way (an open file, a read-only dir) therefore leaves nothing that
    still looks like an adoptable tree: the next launch builds a fresh one under the key instead
    of exec'ing a half-deleted environment, and the leftover is swept by the following pass.
    """
    trash = path.with_name(f".evicting-{path.name}-{os.getpid()}")
    try:
        path.rename(trash)
    except OSError:
        return None
    return trash


def discard(path: Path) -> None:
    """Take *path* out of its key before removing it."""
    trash = _set_aside(path)
    if trash is None:
        logger.debug("dependency tree %s could not be taken aside; retried next pass", path)
        return
    shutil.rmtree(trash, ignore_errors=True)


def collect_unreferenced(installs: Path, *, min_age_seconds: float = MIN_AGE_SECONDS) -> list[Path]:
    """Remove every unleased tree no checkout names; returns the entries removed.

    The state tiers decide which *checkouts* survive (a deleted checkout, one past the count
    bound); this decides which *trees* they still need. One entry referenced by three worktrees
    therefore costs one tree and survives, which is the whole point: a fresh worktree reuses
    what the last launch built instead of paying for its own copy.
    """
    removed: list[Path] = []
    store = installs / STORE_DIRECTORY
    if not store.is_dir():
        return removed
    _sweep_evictions(store)
    named = references(installs)
    now = time.time()
    for kind in (VENV, TEST_ENV):
        kind_dir = store / kind
        if not kind_dir.is_dir():
            continue
        for candidate in sorted(kind_dir.iterdir()):
            if not candidate.is_dir() or candidate.name.startswith("."):
                continue
            if candidate.resolve() in named:
                continue
            try:
                age = now - (candidate / COMPLETE_MARKER).stat().st_mtime
            except OSError:
                age = now - candidate.stat().st_mtime if candidate.exists() else 0.0
            if age < min_age_seconds or leases_held(candidate):
                continue
            discard(candidate)
            if not candidate.exists():
                removed.append(candidate)
            else:
                logger.debug("dependency tree %s only partly removed; retried next pass", candidate.name)
    return removed
