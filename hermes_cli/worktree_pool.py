"""Admission bound on a repo's linked-worktree population (``worktrees.max_linked``).

Arrival was otherwise unbounded: every ``hermes -w`` launch and every isolated subagent creates a
linked worktree, and nothing on the creation path asked how many already existed — 893 trees
across 162 repos on 2026-10-23, ~90/day in one repo at ~126 MB each. Reclamation is a policy over
*idle* trees, so it can only ever see the part of the population older than its age floor; a
population mostly younger than any floor it may use is unbounded by construction. The bound
belongs where the tree is born: count this repo's linked worktrees before ``git worktree add``,
recycle the idle ones with the existing archive-first sweep, and refuse past the cap.

Refusing is what makes the number a bound; recycling alone only moves the cliff.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("cli")

# Idle tier one admission reclaim pass may act on: the same 24 h the boot pruner gives a
# Hermes-made tree. Never lower it for admission — an age-0 reclaim would delete trees an agent is
# still working in, and only the age floor keeps "idle" meaningful.
RECLAIM_IDLE_HOURS = 24

# Per-process population cache: a launch asks git once, then keeps its own count current. Another
# process creating or removing a tree is not seen until the next launch — the bound is per repo
# and the overshoot is bounded by the number of processes admitting concurrently.
_count_cache: dict[str, int] = {}
_count_lock = threading.Lock()


@dataclass
class Admission:
    """Verdict for one worktree creation in one repo."""

    allowed: bool
    count: int             # linked worktrees in the repo after any reclaim
    cap: int               # worktrees.max_linked; 0 = unbounded
    reclaimed: int = 0     # trees the reclaim pass freed
    measured: bool = True  # False = git could not answer; the population is unknown


def pool_settings() -> dict:
    """``worktrees.max_linked`` / ``worktrees.reclaim_before_admit`` (0 / True when unset)."""
    from hermes_cli.worktree_sweep import sweep_settings

    settings = sweep_settings()
    return {"max_linked": settings["max_linked"],
            "reclaim_before_admit": settings["reclaim_before_admit"]}


def _porcelain_count(repo_root: str) -> Optional[int]:
    """Linked worktrees registered on *repo_root* — every ``git worktree list`` entry but the main
    checkout. None when git cannot answer."""
    from hermes_cli.worktree_ops import _git  # late: worktree_ops calls into this module

    try:
        result = _git(["worktree", "list", "--porcelain"], repo_root, timeout=10)
    except (OSError, subprocess.SubprocessError):
        logger.debug("worktree count failed for %s", repo_root, exc_info=True)
        return None
    if result.returncode != 0:
        return None
    blocks = [block for block in result.stdout.split("\n\n") if block.strip()]
    return max(0, len(blocks) - 1)


def linked_count(repo_root: str, *, refresh: bool = False) -> Optional[int]:
    """This repo's linked-worktree count (per-process cache; *refresh* re-asks git)."""
    key = os.path.realpath(repo_root)
    with _count_lock:
        if not refresh and key in _count_cache:
            return _count_cache[key]
    count = _porcelain_count(repo_root)
    if count is not None:
        _remember(key, count)
    return count


def _remember(key: str, count: int) -> None:
    with _count_lock:
        _count_cache[key] = count


def reset_cache() -> None:
    """Drop every cached count (tests)."""
    with _count_lock:
        _count_cache.clear()


def note_created(repo_root: str, count: int) -> int:
    """Record one just-added tree -> the repo's population after the add."""
    _remember(os.path.realpath(repo_root), count + 1)
    return count + 1


def reclaim(repo_root: str, need: int) -> Optional[int]:
    """One bounded, synchronous archive-first pass: free at most *need* idle trees.

    ``worktree_sweep.run_sweep`` plans over every tree and applies the whole plan; on the
    admission path that is minutes of archiving to shrink a population by one. This uses the same
    machinery (``plan_sweep`` + ``execute_plan``) and applies only the oldest *need* reclaimable
    decisions, so the archive refs (``refs/hermes/archive/<tree>``) and every safety gate are
    unchanged: a live lock owner, any process inside the tree, a Kanban tree, a tree inside its
    idle tier, and anything an archive cannot carry are never touched. Returns the number of trees
    freed, or None when the pass left the population unverifiable.
    """
    from hermes_cli import worktree_sweep

    settings = worktree_sweep.sweep_settings()
    before = linked_count(repo_root, refresh=True)
    try:
        plan = worktree_sweep.plan_sweep(repo_root, RECLAIM_IDLE_HOURS, settings)
        doomed = sorted((d for d in plan if d.action in worktree_sweep._DESTRUCTIVE),
                        key=lambda d: d.mtime)[:max(1, need)]
        if doomed:
            worktree_sweep.execute_plan(repo_root, doomed, settings)
    except Exception:
        logger.warning("Admission reclaim pass failed in %s", repo_root, exc_info=True)
        return 0
    after = linked_count(repo_root, refresh=True)
    if before is None or after is None:
        return None
    freed = max(0, before - after)
    if freed:
        logger.warning("Reclaimed %d idle worktree(s) in %s to admit a new one", freed, repo_root)
    return freed


def admit(repo_root: str) -> Admission:
    """Decide whether one more linked worktree may be created in *repo_root*.

    ``max_linked`` <= 0 is the pre-bound behaviour exactly: no count, no git call, always admit.
    At the cap one bounded reclaim pass runs first (``reclaim_before_admit``); if it frees
    nothing, the creation is refused and the caller must neither add a tree nor print a success
    line. A count git cannot produce is not a reason to block a launch, so it admits with a
    warning.
    """
    settings = pool_settings()
    cap = settings["max_linked"]
    if cap <= 0:
        return Admission(True, 0, 0, measured=False)
    count = linked_count(repo_root)
    if count is None:
        logger.warning("Could not count linked worktrees in %s — admitting unmeasured", repo_root)
        return Admission(True, 0, cap, measured=False)
    if count < cap:
        return Admission(True, count, cap)
    freed = reclaim(repo_root, need=count - cap + 1) if settings["reclaim_before_admit"] else 0
    if freed is None:
        logger.warning("Population unverifiable after the admission reclaim pass in %s — "
                       "admitting unmeasured", repo_root)
        return Admission(True, count, cap, measured=False)
    verdict = Admission(count - freed < cap, count - freed, cap, freed)
    if not verdict.allowed:
        logger.warning("Refusing to create a worktree in %s: %s", repo_root, refusal_text(verdict))
    return verdict


def refusal_text(verdict: Admission) -> str:
    """One actionable line for a refused creation: the count, the cap, and the manual lever."""
    return (f"this repo is at its worktree cap ({verdict.count}/{verdict.cap} linked) and one "
            "reclaim pass freed nothing it may remove (live, locked, busy and young trees are "
            "never touched). Reclaim by hand with `hermes worktree prune`, or raise "
            "`worktrees.max_linked` in config.yaml (0 = no bound).")
