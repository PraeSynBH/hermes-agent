"""Which profiles hold LIVE demand — the input to the multiplexer's serve ranking.

``gateway.multiplex_serve_priority`` used to be a hand-maintained list, so it went stale the hour a
profile acquired its first agent or cron job: a profile with real work sat deferred behind one that
held nothing (SOU-3073). Ranking now follows demand, read from three sources on every reconcile tick:

S1 ``profiles/<n>/cron/jobs.json`` — enabled jobs. Owned by core, costs one small JSON read.
S2 ``profiles/<n>/.env`` — the adapter credentials that make the profile connect as a bot, via
   :func:`hermes_cli.profile_channels.credential_env_keys` (the same table ``profile clone`` and the
   migration preflight trust; never a second list here). Owned by core, one ``.env`` parse.
S3 the demand manifest — ``gateway.multiplex_demand_manifest`` (default
   ``<default home>/gateway_demand.json``), schema 1. Written by whoever knows about agents that
   wake profiles: today a host-side collector, never read from another product's database.

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

Cost: :func:`snapshot` is TTL-memoised on a monotonic clock and performs **no I/O** inside the TTL;
:func:`refresh` is the authoritative read and must be called off the event loop by the reconcile
tick (``asyncio.to_thread``), exactly like the served-set signature sweep next door.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

MANIFEST_KEY = "multiplex_demand_manifest"
DEFAULT_MANIFEST_NAME = "gateway_demand.json"
SCHEMA_VERSION = 1
DEFAULT_TTL_SECONDS = 900.0
#: Monotonic lifetime of one :class:`DemandSnapshot`. A 30 s reconcile tick plus this TTL bounds
#: demand staleness at ~35 s — inside the 60 s acceptance — while an inbound route re-reading the
#: snapshot pays no I/O inside the window.
SNAPSHOT_TTL_SECONDS = 5.0

MANIFEST_OK = "ok"
MANIFEST_STALE = "stale"
MANIFEST_ABSENT = "absent"
MANIFEST_MALFORMED = "malformed"


@dataclass(frozen=True)
class DemandFacts:
    """One profile's demand. ``agents is None`` means *unknown* (no manifest), never zero."""

    agents: Optional[int] = None
    cron_jobs: int = 0
    credentials: tuple[str, ...] = ()
    blocked_admission_15m: int = 0
    #: Source ids (``cron``, ``env``) that could not be read for this profile.
    unreadable: tuple[str, ...] = ()

    def has_demand(self) -> bool:
        """An unreadable source fails toward SERVING the profile (F7 in the plan)."""
        if self.unreadable:
            return True
        return self.cron_jobs > 0 or bool(self.credentials) or bool(self.agents)

    def as_record(self) -> dict:
        """The published shape — counts only; agent names stay in the manifest."""
        record: dict = {
            "agents": self.agents,
            "cron_jobs": self.cron_jobs,
            "credentials": list(self.credentials),
            "blocked_admission_15m": self.blocked_admission_15m,
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
    captured_at: float = 0.0

    def for_profile(self, name: str) -> DemandFacts:
        return self.facts.get(name) or DemandFacts()

    def has(self, name: str) -> bool:
        """Does *name* hold demand? ``default`` is never ranked; callers keep it served anyway."""
        return self.for_profile(name).has_demand()

    def demand_source(self) -> dict:
        """The ``serving_decision.demand_source`` record (path, state, age)."""
        return {
            "path": str(self.manifest_path) if self.manifest_path is not None else None,
            "state": self.manifest_state,
            "age_seconds": None if self.manifest_age_seconds is None else int(self.manifest_age_seconds),
        }


_LOCK = threading.Lock()
_snapshot: Optional[DemandSnapshot] = None
_snapshot_at: Optional[float] = None


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
    """``(state, {name: (agents, blocked_admission_15m)}, age_seconds, error)`` for the manifest.

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
    parsed: dict[str, tuple[int, int]] = {}
    for name, entry in profiles.items():
        if not isinstance(entry, dict):
            return MANIFEST_MALFORMED, {}, None, f"manifest entry {name!r} is not an object"
        try:
            agents = int(entry.get("agents") or 0)
            blocked = int(entry.get("blocked_admission_15m") or 0)
        except (TypeError, ValueError):
            return MANIFEST_MALFORMED, {}, None, f"manifest entry {name!r} has non-numeric counts"
        if agents < 0 or blocked < 0:
            return MANIFEST_MALFORMED, {}, None, f"manifest entry {name!r} has negative counts"
        parsed[str(name)] = (agents, blocked)
    ttl = data.get("ttl_seconds")
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or ttl <= 0:
        ttl = DEFAULT_TTL_SECONDS
    state = MANIFEST_STALE if age > float(ttl) else MANIFEST_OK
    return state, parsed, age, None


def _cron_jobs(home: Path) -> tuple[int, tuple[str, ...]]:
    """S1: enabled jobs in ``<home>/cron/jobs.json``. A corrupt file is unreadable, not zero."""
    path = Path(home) / "cron" / "jobs.json"
    if not path.exists():
        return 0, ()
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
    if not path.is_file():
        return (), ()
    try:
        with open(path, "rb") as fh:
            fh.read(1)
    except OSError as exc:
        logger.debug("demand: %s is unreadable: %s", path, exc)
        return (), ("env",)
    from hermes_cli.profile_channels import _env_values
    values = _env_values(path, wanted=dict(keys))
    return tuple(sorted({platform for key, platform in keys.items() if values.get(key)})), ()


def _roster() -> list[tuple[str, Path]]:
    """Named profiles with an identity marker — the same roster serving uses, ``default`` aside."""
    from hermes_cli.profiles import _iter_named_profile_dirs
    try:
        return [(entry.name, entry) for entry in _iter_named_profile_dirs()]
    except OSError as exc:  # an unreadable profiles/ root is not an empty host
        logger.debug("demand: could not enumerate profiles/: %s", exc)
        raise


def refresh() -> DemandSnapshot:
    """The authoritative read: manifest once, then S1 + S2 per profile. Never raises.

    Called off the event loop from the reconcile tick (``asyncio.to_thread``). A per-source failure
    is non-fatal and per-profile; only a failure that would mislabel a profile is recorded.
    """
    global _snapshot, _snapshot_at
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
        blocked = 0
        if manifest_known:
            agents, blocked = manifest.get(name, (0, 0))
        facts[name] = DemandFacts(
            agents=agents,
            cron_jobs=cron_jobs,
            credentials=credentials,
            blocked_admission_15m=blocked,
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
        _snapshot, _snapshot_at = snapshot, time.monotonic()
    return snapshot


def snapshot() -> DemandSnapshot:
    """The current demand view, refreshed only when the 5 s TTL has lapsed (no I/O inside it)."""
    with _LOCK:
        current, captured = _snapshot, _snapshot_at
    if current is not None and captured is not None and (time.monotonic() - captured) < SNAPSHOT_TTL_SECONDS:
        return current
    return refresh()


def reset() -> None:
    """Drop the memoised snapshot (tests, and an operator who wants the next read to be fresh)."""
    global _snapshot, _snapshot_at
    with _LOCK:
        _snapshot, _snapshot_at = None, None


__all__ = [
    "DEFAULT_MANIFEST_NAME",
    "DEFAULT_TTL_SECONDS",
    "MANIFEST_ABSENT",
    "MANIFEST_KEY",
    "MANIFEST_MALFORMED",
    "MANIFEST_OK",
    "MANIFEST_STALE",
    "SCHEMA_VERSION",
    "DemandFacts",
    "DemandSnapshot",
    "manifest_path",
    "refresh",
    "reset",
    "snapshot",
]
