"""Admission bound on a repo's linked-worktree population (``worktrees.max_linked``).

Arrival was otherwise unbounded: every ``hermes -w`` launch and every isolated subagent creates a
linked worktree, and nothing on the creation path asked how many already existed — 893 trees
across 162 repos on 2026-10-23, ~90/day in one repo at ~126 MB each. Reclamation is a policy over
*idle* trees, so it can only ever see the part of the population older than its age floor; a
population mostly younger than any floor it may use is unbounded by construction. The bound
belongs where the tree is born: count this repo's linked worktrees before ``git worktree add``,
recycle the idle ones with the existing archive-first sweep, and refuse past the cap.

One global integer cannot say both "voyonder runs ~18 at a time" and "alaska_supply needs ~340
trees": the number that refuses nothing in the largest repo tolerates a population 8x the working
set in the smallest. ``worktrees.max_linked_by_repo`` names a repo's own cap and falls back to
``max_linked`` for anything unnamed (``0`` keeps its unbounded meaning in both places). And a
cap set once is a snapshot, not a policy: as the reaper drains a population the cap must come down
with it or the population regrows to the old number. ``worktrees.ratchet`` records each repo's
population at sweep time and lowers its cap to ``population + ratchet_headroom`` — a ratchet that
only ever decreases, leaving room for concurrent launches but not for the cliff the population
just left. A changed configured number starts a fresh ratchet at that value.

Refusing is what makes the number a bound; recycling alone only moves the cliff.
"""

from __future__ import annotations

import datetime
import logging
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
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

# Last refusal per repo, for callers that must tell "at the cap" from "git broke": a refused
# creation is a capacity condition, not a failure of the launch, so the CLI books it as a
# deferral (EX_TEMPFAIL) instead of an error the dispatcher would count against the task.
_last_refusal: dict[str, Admission] = {}


@dataclass
class Admission:
    """Verdict for one worktree creation in one repo."""

    allowed: bool
    count: int             # linked worktrees in the repo after any reclaim
    cap: int               # the cap admission enforced; 0 = unbounded
    reclaimed: int = 0     # trees the reclaim pass freed
    measured: bool = True  # False = git could not answer; the population is unknown
    configured_cap: int = 0  # the configured number before the ratchet lowered it; 0 = same as cap
    ratcheted: bool = False  # True when the ratchet, not the operator, set this cap


# The ratchet leaves this many trees of room above the population it measures: the fleet runs
# ~18 launches at once in its busiest repo, so a smaller cushion would refuse a launch the moment
# a repo sat at its cap. Configurable as ``worktrees.ratchet_headroom``.
RATCHET_HEADROOM_DEFAULT = 20
_RATCHET_NAME = "worktree_ratchet.json"
_ratchet_holder = threading.local()


def pool_settings() -> dict:
    """``worktrees.*`` admission settings (``max_linked``/``max_linked_by_repo``/ratchet)."""
    from hermes_cli.worktree_sweep import sweep_settings

    settings = sweep_settings()
    return {key: settings[key] for key in ("max_linked", "max_linked_by_repo", "ratchet",
                                           "ratchet_headroom", "reclaim_before_admit")}


def _coerce_cap(value) -> int:
    """One cap value as a non-negative int; anything unreadable reads as 0 (unbounded)."""
    try:
        return max(0, int(float(value or 0)))
    except (TypeError, ValueError):
        return 0


def _repo_realpath(repo_root: str) -> str:
    return os.path.realpath(str(repo_root))


def _is_path_key(key: str) -> bool:
    return os.path.isabs(key) or key.startswith("~")


def repo_override_cap(repo_root: str, settings: Optional[dict] = None) -> Optional[int]:
    """``worktrees.max_linked_by_repo[<this repo>]``, or None when the repo is unnamed.

    A key is either an absolute path (``~/src/app``) or a repo basename (``app``). An absolute
    path wins over a basename, so a basename entry cannot shadow a deliberately named path; a
    relative-looking key that is not an existing repo's basename is simply never consulted. ``0``
    keeps its documented "unbounded" meaning for that one repo.
    """
    settings = settings or pool_settings()
    table = settings.get("max_linked_by_repo") or {}
    if not isinstance(table, dict) or not table:
        return None
    real = _repo_realpath(repo_root)
    for key, value in table.items():
        name = str(key).strip().rstrip("/")
        if not name or not _is_path_key(name):
            continue
        try:
            if os.path.realpath(os.path.expanduser(name)) == real:
                return _coerce_cap(value)
        except (OSError, ValueError):
            continue
    base = os.path.basename(real.rstrip(os.sep))
    for key, value in table.items():
        name = str(key).strip().rstrip("/")
        if name and not _is_path_key(name) and name == base:
            return _coerce_cap(value)
    return None


def configured_cap(repo_root: str, settings: Optional[dict] = None) -> int:
    """The operator's cap for this repo: its ``max_linked_by_repo`` entry, else ``max_linked``."""
    settings = settings or pool_settings()
    override = repo_override_cap(repo_root, settings)
    if override is not None:
        return override
    return _coerce_cap(settings.get("max_linked"))


def ratchet_path() -> Path:
    """Per-profile snapshot of each repo's measured population and effective cap."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / _RATCHET_NAME


def _read_ratchet() -> dict:
    """``{realpath: {configured, cap, population, ratcheted, updated_at}}``; {} when absent."""
    from utils import read_json_or_empty
    try:
        raw = read_json_or_empty(ratchet_path())
    except Exception:
        logger.debug("worktree ratchet read failed", exc_info=True)
        return {}
    repos = raw.get("repos")
    return repos if isinstance(repos, dict) else {}


def _mutate_ratchet(mutator) -> dict:
    """Read-modify-write the ratchet file under a cross-process lock; returns the repo map.

    A locked, atomic read-modify-write keeps concurrent launches from clobbering each other's
    measurements. Any failure degrades to "no snapshot" — the bound must never block a launch
    because a state file could not be written.
    """
    try:
        from hermes_cli.auth import _file_lock
        from utils import atomic_json_write
        path = ratchet_path()
        with _file_lock(path.with_name(path.name + ".lock"), _ratchet_holder, 5.0,
                        "Timed out waiting for the worktree ratchet lock"):
            repos = _read_ratchet()
            changed = mutator(repos)
            if changed:
                atomic_json_write(path, {"version": 1, "repos": repos})
            return repos
    except Exception:
        logger.debug("worktree ratchet update failed", exc_info=True)
        return _read_ratchet()


def ratchet_cap(repo_root: str, settings: Optional[dict] = None) -> tuple[int, bool]:
    """The cap admission must enforce here -> ``(cap, ratcheted)``.

    ``cap`` is the configured per-repo/global number, lowered to the ratchet's stored value when
    one exists for this repo and the configured number has not changed since it was taken. ``0``
    means unbounded and is never ratcheted. A repo with no snapshot admits against its configured
    number exactly as before — the ratchet only takes effect once a sweep has measured it.
    """
    settings = settings or pool_settings()
    configured = configured_cap(repo_root, settings)
    if configured <= 0 or not settings.get("ratchet"):
        return configured, False
    entry = _read_ratchet().get(_repo_realpath(repo_root))
    if not isinstance(entry, dict) or _coerce_cap(entry.get("configured")) != configured:
        return configured, False
    stored = _coerce_cap(entry.get("cap"))
    if stored <= 0:
        return configured, False
    cap = min(stored, configured)
    return cap, cap < configured


def ratchet_after_sweep(repo_root: str, population: Optional[int],
                        settings: Optional[dict] = None) -> Optional[int]:
    """Record one sweep measurement and lower this repo's cap toward its population -> cap.

    Called from the sweep, never from admission: a cap recomputed on every creation would simply
    follow the population up and never bind. ``candidate = population + ratchet_headroom`` leaves
    room for concurrent launches while keeping the cap off the cliff the population just left.
    The stored cap only ever decreases; a changed configured value (the operator raised or
    lowered the ceiling) starts a fresh ratchet at that value. ``None`` population (git could not
    count) leaves the state untouched. ``0`` configured means unbounded and is recorded but never
    enforced. Failures are non-fatal.
    """
    settings = settings or pool_settings()
    configured = configured_cap(repo_root, settings)
    if population is None:
        population = linked_count(repo_root, refresh=True)
    if population is None:
        entry = _read_ratchet().get(_repo_realpath(repo_root))
        return _coerce_cap(entry.get("cap")) if isinstance(entry, dict) else configured
    ratchet_on = bool(settings.get("ratchet")) and configured > 0
    headroom = max(1, _coerce_cap(settings.get("ratchet_headroom")) or RATCHET_HEADROOM_DEFAULT)
    candidate = min(configured, population + headroom) if ratchet_on else configured
    now = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()

    def _apply(repos: dict) -> bool:
        key = _repo_realpath(repo_root)
        raw_prev = repos.get(key)
        prev: dict = raw_prev if isinstance(raw_prev, dict) else {}
        prev_configured = _coerce_cap(prev.get("configured"))
        prev_cap = _coerce_cap(prev.get("cap"))
        # A changed ceiling opens a fresh ratchet at the operator's number; otherwise the cap can
        # only come down. 0 stays 0: "unbounded" must not be narrowed by an automatic ratchet.
        new_cap = candidate if (prev_configured != configured or prev_cap <= 0) \
            else min(candidate, prev_cap)
        new_cap = 0 if configured <= 0 else max(1, min(new_cap, configured))
        entry = {"configured": configured, "cap": new_cap, "population": population,
                 "ratcheted": bool(new_cap and new_cap < configured), "updated_at": now}
        if isinstance(prev, dict) and all(prev.get(k) == v for k, v in entry.items()):
            return False
        repos[key] = entry
        return True

    repos = _mutate_ratchet(_apply)
    entry = repos.get(_repo_realpath(repo_root)) if isinstance(repos, dict) else None
    return _coerce_cap(entry.get("cap")) if isinstance(entry, dict) else configured


def population_summary(repo_root: str, settings: Optional[dict] = None) -> str:
    """One operator-facing line: this repo's population against the cap admission enforces."""
    settings = settings or pool_settings()
    count = linked_count(repo_root, refresh=True)
    cap, ratcheted = ratchet_cap(repo_root, settings)
    name = os.path.basename(_repo_realpath(repo_root).rstrip(os.sep)) or str(repo_root)
    if count is None:
        return (f"Worktree population: unknown in {name} (git could not count) — "
                + ("no cap" if cap <= 0 else f"cap {cap}"))
    if cap <= 0:
        return f"Worktree population: {count} linked in {name} (no cap)"
    line = f"Worktree population: {count}/{cap} in {name}"
    if ratcheted:
        configured = configured_cap(repo_root, settings)
        line += (f" (ratcheted down from {configured} as the population fell; raise "
                 "`worktrees.max_linked`/`max_linked_by_repo` or set `worktrees.ratchet: false`)")
    return line


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

    The cap is this repo's ``max_linked_by_repo`` entry when it has one, else ``max_linked``, and
    is lowered further by the ratchet's stored value once a sweep has measured the population. A
    cap <= 0 is the pre-bound behaviour exactly: no count, no git call, always admit. At the cap
    one bounded reclaim pass runs first (``reclaim_before_admit``); if it frees nothing, the
    creation is refused and the caller must neither add a tree nor print a success line. A count
    git cannot produce is not a reason to block a launch, so it admits with a warning.
    """
    settings = pool_settings()
    configured = configured_cap(repo_root, settings)
    cap, ratcheted = ratchet_cap(repo_root, settings)
    if cap <= 0:
        return Admission(True, 0, 0, measured=False, configured_cap=configured)
    count = linked_count(repo_root)
    if count is None:
        logger.warning("Could not count linked worktrees in %s — admitting unmeasured", repo_root)
        return Admission(True, 0, cap, measured=False, configured_cap=configured,
                         ratcheted=ratcheted)
    if count < cap:
        return Admission(True, count, cap, configured_cap=configured, ratcheted=ratcheted)
    freed = reclaim(repo_root, need=count - cap + 1) if settings["reclaim_before_admit"] else 0
    if freed is None:
        logger.warning("Population unverifiable after the admission reclaim pass in %s — "
                       "admitting unmeasured", repo_root)
        return Admission(True, count, cap, measured=False, configured_cap=configured,
                         ratcheted=ratcheted)
    verdict = Admission(count - freed < cap, count - freed, cap, freed,
                        configured_cap=configured, ratcheted=ratcheted)
    if not verdict.allowed:
        logger.warning("Refusing to create a worktree in %s: %s", repo_root, refusal_text(verdict))
        _last_refusal[os.path.realpath(repo_root)] = verdict
    return verdict


def last_refusal(repo_root: str) -> Optional[Admission]:
    """The refusal this process booked for *repo_root*, or None — so a launch whose worktree
    setup returned None can tell "the repo is at its cap" (a capacity deferral) from "git or the
    filesystem failed" (an error). Absent for an admitted creation."""
    return _last_refusal.get(os.path.realpath(repo_root))


def clear_refusal(repo_root: str) -> None:
    """Drop the booked refusal for *repo_root* — called once a creation succeeds."""
    _last_refusal.pop(os.path.realpath(repo_root), None)


def refusal_text(verdict: Admission) -> str:
    """One actionable line for a refused creation: the count, the cap, and the manual lever."""
    base = (f"this repo is at its worktree cap ({verdict.count}/{verdict.cap} linked) and one "
            "reclaim pass freed nothing it may remove (live, locked, busy and young trees are "
            "never touched). Reclaim by hand with `hermes worktree prune`")
    if verdict.ratcheted and verdict.configured_cap > 0:
        return (f"{base}. The cap is ratcheted down from {verdict.configured_cap} as the "
                "population fell: raise `worktrees.max_linked`/`max_linked_by_repo` in "
                "config.yaml (0 = no bound) or set `worktrees.ratchet: false`.")
    return (f"{base}, or raise `worktrees.max_linked`/`max_linked_by_repo` in config.yaml "
            "(0 = no bound).")
