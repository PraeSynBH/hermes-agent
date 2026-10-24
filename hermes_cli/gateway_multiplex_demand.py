"""Which profiles hold LIVE demand — the input to the multiplexer's serve ranking.

``gateway.multiplex_serve_priority`` used to be a hand-maintained list, so it went stale the hour a
profile acquired its first agent or cron job: a profile with real work sat deferred behind one that
held nothing (SOU-3073). Ranking now follows demand, read from three sources on every reconcile tick:

S1 ``profiles/<n>/cron/jobs.json`` — enabled jobs. Owned by core, costs one small JSON read.
S2 ``profiles/<n>/state.db`` ``st_mtime`` — used within :data:`RECENT_USE_WINDOW_SECONDS`. One
   ``stat``, and the only cheap evidence that a profile is *being used*: a human talking to it, with
   no cron, no route and no agent, was previously indistinguishable from a dead profile (PRX-5750).
S3 ``profiles/<n>/.env`` — the adapter credentials that make the profile connect as a bot, via
   :func:`hermes_cli.profile_channels.credential_env_keys` (the same table ``profile clone`` and the
   migration preflight trust; never a second list here). Owned by core, one ``.env`` parse.
   **A credential is a capability, not work** (PRX-5750): every live profile has one, so ranking by
   it made the demand class universal and degenerated to the old priority list. It is kept as a
   *tiebreak below* every work signal — it only separates two profiles that both hold no work.
S4 the demand manifest — ``gateway.multiplex_demand_manifest`` (default
   ``<default home>/gateway_demand.json``), schema 1. Written by whoever knows about agents that
   wake profiles: today a host-side collector, never read from another product's database.

S1–S4 are read from disk. The fifth source is LIVE and lives in this process only: a client that
asked for a profile and is waiting on its admission (``note_wanted``, PRX-5738). S1–S4 say a profile
*could* hold work; a pending admission says somebody is blocked on it right now, which is the
sighting the 2026-10-20 outage never produced — the busiest profile sat deferred while its runs
failed admission. A sighting carries its own TTL and expires, so it cannot pin a slot forever.

**A run that was dispatched but has not started is queued work, not assignment (PRX-6000).** The
manifest's ``queued`` count (``heartbeat_runs.status='queued'`` for the profile's agents) ranks with
``busy`` in class 0, because such a run cannot start while the profile is unserved: it is the same
"somebody is waiting" claim as a blocked admission, and its TTL is the dispatcher's own queue expiry.
The 2026-10-22 boot that evicted ``release`` for 19 minutes had exactly that shape — ``release``
held two queued runs, the collector reported only ``running`` ones, so the profile ranked as merely
assigned (class 1) and lost its slot to a profile pinned by a live sighting. With both sides in
class 0 the operator's ``gateway.multiplex_serve_priority`` decides, which is the one input that can
say "this profile ships production deploys" (see ``apply_ceiling``'s tiebreak).

**A pending WAKE is its own rank class, above assigned work (PRA-6796).** ``agents`` counts cards a
profile *holds*; it does not say anybody is trying to run one. So a profile whose Paperclip agent
was woken a second ago ranked identically to one that merely owns two idle agents, and the tie
fell through to the name — alphabetically last lost the last slot and its every wake was refused
(release: 15 straight ``profile_not_served`` refusals behind an idle cron profile). The manifest's
``busy`` count (queued/running runs for its agents) and ``blocked_admission_15m`` are the two ways
the host says *this wake is waiting*, and both now rank ahead of assignment and recency.

**Trust boundary:** core never reads Paperclip. The manifest is a documented, schema-versioned host
input; any scheduler can write it. A malformed or absent manifest degrades to S1 ∨ S2 and is
*reported*, never silently read as "no demand".

Staleness policy (deliberate, stated in the docs page):

===========================  ==============================  ==============================
manifest state               ranking uses                    the host check reports
===========================  ==============================  ==============================
absent                       S1 ∨ S2 only                    FAIL (agent demand unknown)
older than ``ttl_seconds``   the last known values           WARN, deferred lines tagged stale
malformed / wrong schema     S1 ∨ S2 only                    FAIL (unreadable)
===========================  ==============================  ==============================

Dropping stale demand is the same failure class as the hand-list going stale, so staleness fails
open toward *serving* demand, loudly. A source that could not be READ at all is different: the
profile counts as demand-bearing (fail toward serving it) and is reported ``unreadable`` — treating
an unreadable ``.env`` as "no demand" is the bug class this module exists to remove.

Cost: :func:`snapshot` performs **no I/O at all** — not even the first time. Because the serving path
calls it per inbound route (``/p/<profile>/``) on the event loop, and because the first
:func:`refresh` of a process pays one-time plugin discovery (``credential_env_keys``: 15–22 s on a
loaded host, measured 00:00Z), a cold read must never be paid by a caller that does not own
freshness. With no snapshot, :func:`snapshot` therefore returns a **cold view** —
``facts={}``, ``manifest_state="cold"``, ``captured_at=None`` — in which no profile holds demand, so
the ceiling falls back to the ``demand=None`` order (priority list, then alphabetical).

Cold is not "no demand": :meth:`DemandSnapshot.demand_source` reports ``state="cold"`` with
``age_seconds=None``, and the host check renders that as ``demand=cold (first reconcile pending)``
instead of implying real facts. Only :func:`refresh` reads, and only its two owners call it: the
reconcile tick (``asyncio.to_thread``, every 30 s) and the boot path's own pre-warm (off-loop,
started beside boot so the discovery overlaps the rest of startup, then awaited before boot freezes
the served set — PRX-5963). A read that fails or outlives its bound leaves a priority-only boot set,
which the first tick then corrects — the previous behaviour, and the worse trade for every boot that
can afford the overlap (SOU-3125).
"""

from __future__ import annotations

import json
import logging
import stat
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

MANIFEST_KEY = "multiplex_demand_manifest"
DEFAULT_MANIFEST_NAME = "gateway_demand.json"
SCHEMA_VERSION = 1
DEFAULT_TTL_SECONDS = 900.0

#: How long ONE pending-admission sighting keeps a profile ranked as demand-bearing. Bounded on
#: purpose: a client that asked once and gave up must not pin a ceiling slot forever, so a stale
#: sighting expires by itself (PRX-5738) while a profile that keeps being asked keeps refreshing it.
WANTED_TTL_SECONDS = 300.0

#: How recently ``state.db`` must have been written for a profile to count as *used* (PRX-5750).
#: One ``stat`` per profile per tick, so the window is generous: a profile a human opened this
#: morning still holds a claim, one last touched yesterday does not. Six hours is longer than the
#: daily restart gate, so a profile in use across a restart keeps its claim either side of it.
RECENT_USE_WINDOW_SECONDS = 6 * 3600.0

#: ``name -> monotonic expiry`` for "a client asked for this profile and is waiting on it". Kept
#: beside the snapshot, never inside it: the snapshot is a disk-derived view aged by a cache window,
#: this is LIVE state of THIS process — a route resolved a moment ago — and caching it away would be
#: exactly the silent deferral it exists to prevent.
_wanted: dict[str, float] = {}

MANIFEST_OK = "ok"
MANIFEST_STALE = "stale"
MANIFEST_ABSENT = "absent"
MANIFEST_MALFORMED = "malformed"
#: No snapshot has been taken in this process yet. Published as ``demand_source.state`` so a reader
#: never mistakes the priority-only fallback rank for a considered, demand-ranked decision.
MANIFEST_COLD = "cold"


@dataclass(frozen=True)
class DemandFacts:
    """One profile's demand. ``agents is None`` means *unknown* (no manifest), never zero."""

    agents: Optional[int] = None
    cron_jobs: int = 0
    credentials: tuple[str, ...] = ()
    blocked_admission_15m: int = 0
    #: Runs for this profile's Paperclip agents that are queued or running (manifest ``busy``) — a
    #: wake that has already been dispatched and is waiting for admission. Counted for the label
    #: only: one pending run claims the slot exactly as a thousand would (no weights).
    pending_runs: int = 0
    #: Runs for this profile's agents that have been DISPATCHED but not started (manifest
    #: ``queued``). A queued run cannot start while the profile is unserved, so it is the same
    #: "somebody is waiting" claim as ``pending_runs`` — and the one a profile loses its slot
    #: without ever showing: on 2026-10-22 ``release`` held two queued runs through the boot that
    #: evicted it, and a collector reporting only ``running`` made them invisible (PRX-6000).
    queued_runs: int = 0
    #: Seconds since ``profiles/<n>/state.db`` was last written, or ``None`` when the profile has no
    #: store. ``None`` is "no evidence of use", never "used long ago" — a profile created an hour
    #: ago and never opened must not outrank one whose store is a day old and also idle.
    state_age_seconds: Optional[float] = None
    #: Source ids (``cron``, ``env``) that could not be read for this profile.
    unreadable: tuple[str, ...] = ()

    @property
    def used_recently(self) -> bool:
        """Was this profile's session store written inside :data:`RECENT_USE_WINDOW_SECONDS`?"""
        return (self.state_age_seconds is not None
                and self.state_age_seconds <= RECENT_USE_WINDOW_SECONDS)

    def work_signals(self) -> tuple[str, ...]:
        """Which signals say *this profile holds work right now*, in strength order (PRX-5750).

        The published vocabulary — ``work: cron,recency`` in a deferral reason, ``"work": [...]`` in
        a ``serving_decision`` row — comes from here, so the record and the rank can never disagree.
        ``credentials`` is last on purpose: it is a capability that every live profile has, so it
        only ever separates two profiles that hold no work.
        """
        signals: list[str] = []
        if self.blocked_admission_15m:
            signals.append("admission")
        if self.pending_runs or self.queued_runs:
            signals.append("wake")
        if self.agents:
            signals.append("agents")
        if self.cron_jobs:
            signals.append("cron")
        if self.used_recently:
            signals.append("recency")
        if self.credentials:
            signals.append("credentials")
        signals.extend(self.unreadable)
        return tuple(signals)

    def has_pending_wake(self) -> bool:
        """Is a wake for this profile WAITING right now — blocked admission, or a dispatched run?

        The distinction this class exists for (PRA-6796): *assigned* work says a profile holds
        cards, while a pending wake says somebody dispatched one and it cannot be admitted. Only
        the second one is a client that has already paid for a run and is being refused. A run
        sitting in the QUEUE is that client too (PRX-6000): it is dispatched, it cannot start while
        the profile is unserved, and reporting only ``running`` runs made exactly that claim
        invisible on the boot that evicted ``release``.
        """
        return bool(self.blocked_admission_15m or self.pending_runs or self.queued_runs)

    def has_queued_work(self) -> bool:
        """Work that exists whether or not anybody touched the profile: cron, agents, an admission.

        This is the class a ``multiplex_serve_priority`` entry can never override (PRX-5750) — a
        scheduled job or a blocked client outranks the operator's hand-list, as it always has.
        """
        if self.unreadable:
            return True
        return bool(self.has_pending_wake() or self.agents or self.cron_jobs)

    def has_work(self) -> bool:
        """Any work at all — queued work, or a session store written inside the use window."""
        return self.has_queued_work() or self.used_recently

    def rank_class(self) -> int:
        """The rank's only key, ascending: 0 a pending wake, 1 queued work, 2 recent use,
        3 credentials, 4 nothing.

        Classes, not weights: a profile with 19 agents ties with one that has 1, so editing a job
        cannot flip the served set (SOU-3075's rule). The ladder exists because a cron job is work
        that outlives a human's last click — it must beat a profile that was merely opened an hour
        ago, and both must beat the 40 profiles whose only claim is a bot token (PRX-5750). Class 0
        is separate from assigned work because a refused wake is not a job edit: it is a run that
        already failed, and no tiebreak may let it lose to a profile that is merely configured
        (PRA-6796).
        """
        if self.has_pending_wake():
            return 0
        if self.has_queued_work():
            return 1
        if self.used_recently:
            return 2
        if self.credentials:
            return 3
        return 4

    def has_demand(self) -> bool:
        """Any claim to a slot at all — work first, and an unreadable source fails toward SERVING."""
        if self.unreadable:
            return True
        return self.has_work() or bool(self.credentials)

    def as_record(self) -> dict:
        """The published shape — counts only; agent names stay in the manifest."""
        record: dict = {
            "agents": self.agents,
            "cron_jobs": self.cron_jobs,
            "credentials": list(self.credentials),
            "blocked_admission_15m": self.blocked_admission_15m,
            "pending_runs": self.pending_runs,
            "queued_runs": self.queued_runs,
            "state_age_seconds": (None if self.state_age_seconds is None
                                  else int(self.state_age_seconds)),
            "work": list(self.work_signals()),
        }
        if self.unreadable:
            record["unreadable"] = list(self.unreadable)
        return record


@dataclass(frozen=True)
class DemandSnapshot:
    """An immutable view of every named profile's demand, plus the manifest's own health."""

    facts: Mapping[str, DemandFacts] = field(default_factory=dict)
    manifest_state: str = MANIFEST_ABSENT
    manifest_path: Optional[Path] = None
    manifest_age_seconds: Optional[float] = None
    manifest_error: Optional[str] = None
    unknown_profiles: tuple[str, ...] = ()
    #: ``None`` on the cold view — nothing has been captured, and ``0.0`` would read as 1970.
    captured_at: Optional[float] = None

    def for_profile(self, name: str) -> DemandFacts:
        return self.facts.get(name) or DemandFacts()

    def wanted(self, name: str) -> bool:
        """A LIVE pending-admission sighting for *name* — never part of the cached facts."""
        return wanted(name)

    def has(self, name: str) -> bool:
        """Does *name* hold demand? ``default`` is never ranked; callers keep it served anyway.

        A client waiting on the profile right now counts even when every disk-derived source says
        the profile holds nothing: that client is the work (PRX-5738).
        """
        return self.wanted(name) or self.for_profile(name).has_demand()

    def has_work(self, name: str) -> bool:
        """Does *name* hold WORK — queued work, or a session store inside the use window?"""
        return self.wanted(name) or self.for_profile(name).has_work()

    def holds_credentials(self, name: str) -> bool:
        """Can *name* connect as a bot? True only when it has no stronger claim to a slot."""
        return bool(self.for_profile(name).credentials)

    def has_pending_wake(self, name: str) -> bool:
        """Is a wake for *name* waiting — a live sighting, a blocked admission, a queued run?"""
        return self.wanted(name) or self.for_profile(name).has_pending_wake()

    def rank_class(self, name: str) -> int:
        """How strongly *name* claims a slot: 0 a pending wake · 1 queued work · 2 recent use ·
        3 credentials · 4 none.

        A client waiting on the profile right now is a pending wake with the shortest history, so a
        sighting pins the class at 0 regardless of what the disk says (PRX-5738); so does a wake the
        manifest already counted as blocked or queued (PRA-6796).
        """
        if self.wanted(name):
            return 0
        return self.for_profile(name).rank_class()


    def demand_source(self) -> dict:
        """The ``serving_decision.demand_source`` record (path, state, age)."""
        return {
            "path": str(self.manifest_path) if self.manifest_path is not None else None,
            "state": self.manifest_state,
            "age_seconds": None if self.manifest_age_seconds is None else int(self.manifest_age_seconds),
        }


_LOCK = threading.Lock()
_snapshot: Optional[DemandSnapshot] = None

#: What :func:`snapshot` answers until something refreshes. ``facts={}`` is load-bearing: with no
#: entries every profile reads as demand-free, which reproduces ``apply_ceiling(demand=None)``
#: exactly. A facts map of zeroes would be a different claim ("we looked, nobody has work").
_COLD = DemandSnapshot(manifest_state=MANIFEST_COLD)


def _default_home() -> Path:
    from hermes_constants import get_default_hermes_root
    return Path(get_default_hermes_root())


def manifest_path(default_home: Optional[Path] = None) -> Path:
    """``gateway.multiplex_demand_manifest``, else ``<default home>/gateway_demand.json``.

    The configured value is a plain path (``~`` expanded); a value that is not a string is ignored
    rather than guessed at. The config read is the ceiling reader's ``(mtime, size)`` memo, so this
    costs one ``stat`` on the hot path.
    """
    from hermes_cli import gateway_multiplex_capacity as capacity
    home = Path(default_home) if default_home is not None else _default_home()
    configured = capacity.gateway_section(home).get(MANIFEST_KEY)
    if isinstance(configured, str) and configured.strip():
        return Path(configured.strip()).expanduser()
    return home / DEFAULT_MANIFEST_NAME


def _read_manifest(path: Path):
    """``(state, {name: (agents, blocked_admission_15m, busy, queued)}, age_seconds, error)`` for the
    manifest.

    Absent is its own state (the host runs no collector). Anything read but unusable — torn JSON,
    wrong schema, a non-object entry, a bad ``written_at`` — is ``malformed`` with the reason, so
    the check can name it instead of degrading silently.
    """
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return MANIFEST_ABSENT, {}, None, None
    except OSError as exc:
        return MANIFEST_MALFORMED, {}, None, f"manifest unreadable: {exc}"
    try:
        data = json.loads(raw)
    except ValueError as exc:
        return MANIFEST_MALFORMED, {}, None, f"manifest is not JSON: {exc}"
    if not isinstance(data, dict):
        return MANIFEST_MALFORMED, {}, None, "manifest is not a JSON object"
    if data.get("schema") != SCHEMA_VERSION:
        return MANIFEST_MALFORMED, {}, None, f"manifest schema {data.get('schema')!r} is not {SCHEMA_VERSION}"
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        return MANIFEST_MALFORMED, {}, None, "manifest has no 'profiles' object"
    written_at = data.get("written_at")
    try:
        written = datetime.fromisoformat(str(written_at))  # 3.11+ accepts the trailing Z
    except ValueError:
        return MANIFEST_MALFORMED, {}, None, f"manifest written_at {written_at!r} is not an ISO timestamp"
    if written.tzinfo is None:
        written = written.replace(tzinfo=UTC)
    age = max(0.0, (datetime.now(UTC) - written).total_seconds())
    parsed: dict[str, tuple[int, int, int, int]] = {}
    for name, entry in profiles.items():
        if not isinstance(entry, dict):
            return MANIFEST_MALFORMED, {}, None, f"manifest entry {name!r} is not an object"
        try:
            agents = int(entry.get("agents") or 0)
            blocked = int(entry.get("blocked_admission_15m") or 0)
            # Older collectors wrote no ``busy`` and no ``queued``: absent is 0 (no dispatch), never
            # malformed. Both are "a run was dispatched for this profile" — ``busy`` running,
            # ``queued`` sitting in the queue unable to start (PRX-6000).
            busy = int(entry.get("busy") or 0)
            queued = int(entry.get("queued") or 0)
        except (TypeError, ValueError):
            return MANIFEST_MALFORMED, {}, None, f"manifest entry {name!r} has non-numeric counts"
        if agents < 0 or blocked < 0 or busy < 0 or queued < 0:
            return MANIFEST_MALFORMED, {}, None, f"manifest entry {name!r} has negative counts"
        parsed[str(name)] = (agents, blocked, busy, queued)
    ttl = data.get("ttl_seconds")
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or ttl <= 0:
        ttl = DEFAULT_TTL_SECONDS
    state = MANIFEST_STALE if age > float(ttl) else MANIFEST_OK
    return state, parsed, age, None


# Every 30 s the reconcile tick re-reads each named profile's cron store and ``.env`` to rank the
# ceiling, and each of those used to be open+parse. On a gateway with ~950 threads a read is a GIL
# round trip, so the sweep — not the arithmetic — was the tick's floor (PRA-6850: 42 profiles,
# `profile_reconcile_served` p50 1.3 s). Memoised on the file's ``(mtime_ns, size)``: an unchanged
# store costs one ``stat``. ``_state_age`` stays live — it is a clock reading, not a file fact.
_SOURCE_MEMO: dict[Path, tuple[Any, Any]] = {}


def _source_signature(path: Path) -> Optional[tuple[int, int]]:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _regular_file_signature(path: Path) -> Optional[tuple[int, int]]:
    """``_source_signature`` for a path that must also BE a regular file — one stat, not two."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size) if stat.S_ISREG(st.st_mode) else None


def _memoised(path: Path, signature: Any, compute):
    """``compute()`` for *path*, re-used while *signature* is unchanged (the tick's whole point)."""
    hit = _SOURCE_MEMO.get(path)
    if hit is not None and hit[0] == signature:
        return hit[1]
    value = compute()
    _SOURCE_MEMO[path] = (signature, value)
    return value


def _cron_jobs(home: Path) -> tuple[int, tuple[str, ...]]:
    """S1: enabled jobs in ``<home>/cron/jobs.json``. A corrupt file is unreadable, not zero."""
    path = Path(home) / "cron" / "jobs.json"
    signature = _source_signature(path)
    if signature is None:
        return 0, ()
    return _memoised(path, signature, lambda: _read_cron_jobs(path))


def _read_cron_jobs(path: Path) -> tuple[int, tuple[str, ...]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        logger.debug("demand: cron store at %s is unreadable: %s", path, exc)
        return 0, ("cron",)
    jobs = data.get("jobs") if isinstance(data, dict) else data
    if not isinstance(jobs, list):
        logger.debug("demand: cron store at %s has no jobs list", path)
        return 0, ("cron",)
    return sum(1 for job in jobs if isinstance(job, dict) and job.get("enabled") is True), ()


def _credentials(home: Path, keys: Mapping[str, str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """S2: platforms whose bot credential is present in ``<home>/.env``.

    Reuses ``profile_channels`` (the credential table, and its ``.env`` parser) rather than keeping
    a second key list that would drift from ``hermes profile clone``. A ``.env`` this process cannot
    open is *unreadable* — never "no channels".
    """
    path = Path(home) / ".env"
    # One stat for the regular-file test AND the memo signature (PRA-6850): `is_file()` plus
    # `_source_signature()` was two stats per profile per tick, ×43 profiles, at ~0.5 ms a stat
    # inside this process. A non-regular `.env` reads as "no credentials", exactly as is_file() did.
    signature = _regular_file_signature(path)
    if signature is None:
        return (), ()
    return _memoised(path, (signature, len(keys)), lambda: _read_credentials(path, keys))


def _read_credentials(path: Path, keys: Mapping[str, str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        with open(path, "rb") as fh:
            fh.read(1)
    except OSError as exc:
        logger.debug("demand: %s is unreadable: %s", path, exc)
        return (), ("env",)
    from hermes_cli.profile_channels import _env_values
    values = _env_values(path, wanted=dict(keys))
    return tuple(sorted({platform for key, platform in keys.items() if values.get(key)})), ()


def _state_age(home: Path) -> Optional[float]:
    """S2: seconds since ``<home>/state.db`` was last written, or ``None`` when there is no store.

    One ``stat`` — the only cheap evidence a profile is being *used* rather than merely configured.
    A store we cannot stat, or one with a timestamp in the future (a clock skew on a shared volume),
    is no evidence: reporting a negative age would rank a profile as used forever.
    """
    try:
        mtime = (Path(home) / "state.db").stat().st_mtime
    except OSError:
        return None
    age = time.time() - mtime
    if age < 0:
        logger.debug("demand: %s has an mtime in the future; ignoring the use signal", home)
        return None
    return age


def _roster() -> list[tuple[str, Path]]:
    """Named profiles with an identity marker — the same roster serving uses, ``default`` aside."""
    from hermes_cli.profiles import _iter_named_profile_dirs
    try:
        return [(entry.name, entry) for entry in _iter_named_profile_dirs()]
    except OSError as exc:  # an unreadable profiles/ root is not an empty host
        logger.debug("demand: could not enumerate profiles/: %s", exc)
        raise


def refresh(roster: Optional[list[tuple[str, Path]]] = None) -> DemandSnapshot:
    """The authoritative read: manifest once, then S1 + S2 per profile. Never raises.

    Called off the event loop from the reconcile tick (``asyncio.to_thread``). A per-source failure
    is non-fatal and per-profile; only a failure that would mislabel a profile is recorded.
    ``roster`` is the tick's single enumeration, shared with ``profiles_to_serve`` so one cycle
    walks ``profiles/`` once instead of twice (PRA-6850: the walk is ~130 syscalls).
    """
    global _snapshot
    try:
        path = manifest_path()
        state, manifest, age, error = _read_manifest(path)
    except Exception as exc:  # health: allow BLE001 -- a broken config read must not stop serving
        logger.debug("demand: could not resolve the manifest: %s", exc, exc_info=True)
        path, state, manifest, age, error = None, MANIFEST_MALFORMED, {}, None, f"manifest path unresolvable: {exc}"
    keys: Mapping[str, str] = {}
    keys_unreadable = False
    try:
        from hermes_cli.profile_channels import credential_env_keys
        keys = credential_env_keys()
    except Exception as exc:  # health: allow BLE001 -- reported as env=unreadable per profile
        logger.debug("demand: could not build the credential key table: %s", exc, exc_info=True)
        keys_unreadable = True
    facts: dict[str, DemandFacts] = {}
    manifest_known = state in (MANIFEST_OK, MANIFEST_STALE)
    if roster is None:
        try:
            roster = _roster()
        except OSError:
            roster = []
    for name, home in roster:
        cron_jobs, cron_bad = _cron_jobs(home)
        if keys_unreadable:
            credentials, env_bad = (), ("env",)
        else:
            credentials, env_bad = _credentials(home, keys)
        agents: Optional[int] = None
        blocked = pending = queued = 0
        if manifest_known:
            agents, blocked, pending, queued = manifest.get(name, (0, 0, 0, 0))
        facts[name] = DemandFacts(
            agents=agents,
            cron_jobs=cron_jobs,
            credentials=credentials,
            blocked_admission_15m=blocked,
            pending_runs=pending,
            queued_runs=queued,
            state_age_seconds=_state_age(home),
            unreadable=cron_bad + env_bad,
        )
    known_names = set(facts)
    snapshot = DemandSnapshot(
        facts=facts,
        manifest_state=state,
        manifest_path=path,
        manifest_age_seconds=age,
        manifest_error=error,
        unknown_profiles=tuple(sorted(n for n in manifest if n not in known_names)) if manifest_known else (),
        captured_at=time.time(),
    )
    with _LOCK:
        _snapshot = snapshot
    return snapshot


def snapshot() -> DemandSnapshot:
    """The current demand view: a dict read, and a COLD view when nothing has refreshed yet.

    No I/O, ever — including the first call (SOU-3125). The read this used to do on demand cost
    15–22 s of one-time plugin discovery, and the two callers that hit it cold pay for it in the
    worst places: the boot path (inside the restart SOU-3075 lands) and the event loop (a per-route
    ``profiles_to_serve`` → ``capacity`` → here). Freshness is owned by :func:`refresh` from the
    reconcile tick and the boot pre-warm; this function only reports what they last published.
    """
    with _LOCK:
        current = _snapshot
    return current if current is not None else _COLD


def reset() -> None:
    """Drop the memoised snapshot (tests, and an operator who wants the next read to be fresh)."""
    global _snapshot
    with _LOCK:
        _snapshot = None
    _wanted.clear()
    _SOURCE_MEMO.clear()  # a "fresh read" means the per-file memos too


def note_wanted(name: str, *, ttl: Optional[float] = None, clock=None) -> None:
    """Record that a client asked for *name* and is waiting on its admission (PRX-5738).

    The cheapest demand signal there is, and the only LIVE one: cron jobs, ``.env`` credentials and
    the manifest all say a profile *could* hold work, where this says somebody is blocked on it
    right now. The gateway records it at the seam where an inbound route to an unserved profile is
    refused (``gateway.run_runtime.refuse_parked_scope`` /
    ``gateway.parked_route_refusal.refuse_parked_route``), so the next reconcile tick can serve the
    profile instead of leaving the client to fail fast forever (PRX-5725).

    Evicting something is what makes that possible, so the profile that loses the slot is named in
    the decision's ``pinned`` / ``deferred_for_demand`` records and in the INFO pin line. The TTL
    is the whole safety property: a sighting expires on its own, so a client that asked once and
    walked away cannot hold a ceiling slot for the life of the process.
    """
    if not name:
        return
    lifetime = WANTED_TTL_SECONDS if ttl is None else float(ttl)
    _wanted[str(name)] = (clock or time.monotonic)() + lifetime


def wanted(name: str, *, clock=None) -> bool:
    """Is *name* pinned by a pending-admission sighting that has not expired? O(1), no I/O."""
    key = str(name)
    expiry = _wanted.get(key)
    if expiry is None:
        return False
    if (clock or time.monotonic)() >= expiry:
        _wanted.pop(key, None)  # the sighting went stale: the profile stops being pinned
        return False
    return True


def wanted_signal(name: str) -> str:
    """The work signal a ``wanted`` sighting justifies in the pin line and the deferral reason."""
    return "a client is waiting (pending admission)"


def wake_signal(facts: Any) -> str:
    """*Why* this profile's slot is claimed by a wake that is waiting, or ``""`` (PRA-6796).

    The one sentence a deferral reason carries when the parked profile is not idle-but-configured
    but actively failing to run something. It reads next to ``cost: …`` in ``parked_profiles`` and
    in the refusal the dispatcher books, which is where an operator (or the owning agent) looks.
    """
    if facts.blocked_admission_15m:
        refused = facts.blocked_admission_15m
        return (f"a Paperclip wake is blocked on this profile "
                f"({refused} refused admission{'s' if refused != 1 else ''} in 15 min)")
    runs = int(facts.pending_runs or 0) + int(getattr(facts, "queued_runs", 0) or 0)
    if runs:
        return f"{runs} Paperclip run{'s' if runs != 1 else ''} queued for this profile"
    return ""


__all__ = [
    "DEFAULT_MANIFEST_NAME",
    "DEFAULT_TTL_SECONDS",
    "MANIFEST_ABSENT",
    "MANIFEST_COLD",
    "MANIFEST_KEY",
    "MANIFEST_MALFORMED",
    "MANIFEST_OK",
    "MANIFEST_STALE",
    "RECENT_USE_WINDOW_SECONDS",
    "SCHEMA_VERSION",
    "WANTED_TTL_SECONDS",
    "DemandFacts",
    "DemandSnapshot",
    "manifest_path",
    "note_wanted",
    "refresh",
    "reset",
    "snapshot",
    "wake_signal",
    "wanted",
    "wanted_signal",
]
