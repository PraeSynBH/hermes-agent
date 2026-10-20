"""Hard ceiling on how many profiles ONE gateway process serves.

Under ``gateway.multiplex_profiles`` one process serves every profile on the host, and the
per-profile baseline is real: measured 286 threads / 43 served profiles = 6.7 threads per profile
on a 12-core macOS node, on top of ~25 process-wide threads (PRA-6524). The baseline is
per *profile*, not per call, so it scales with the profile count no matter how well per-call
fan-out is bounded — the ceiling is therefore a number, enforced here, not a warning.

``gateway.multiplex_max_profiles`` (default: unset = no ceiling) caps the SERVED set. Profiles past
the ceiling are not served by this process at all: no adapters, no session authority, no threads.
They are reported under ``parked_profiles`` with a ``capacity:`` reason, so a client routing to one
gets a terminal verdict instead of waiting out its deadline for a profile nothing is serving.

Which profiles? ``default`` is always served (it is the process's own profile and owns the control
socket). Named profiles are RANKED BY LIVE DEMAND since SOU-3075: a profile that owns Paperclip
agents, an enabled cron job, or adapter credentials is served before one that holds nothing
(:mod:`hermes_cli.gateway_multiplex_demand`). ``gateway.multiplex_serve_priority`` survives as the
tiebreak *inside* a class — it can no longer outrank demand — and the name breaks remaining ties.
No weights and no counts-based ordering: a profile with 19 agents does not outrank one with 1 (that
score would flip the served set on every job edit). When two demand-bearing profiles compete for one
slot the loser is deferred by name, deterministically, and the over-subscription is what the host
check pages on.

``apply_ceiling(..., demand=None)`` keeps the pre-SOU-3075 order bit-for-bit for callers and tests
that have no demand view.

This module owns the decision only. The installed roster (``include_parked``/``include_standalone``
enumerations) is never capped: a profile past the ceiling still exists, is still listed, and is
still started once the ceiling or the priority list makes room.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

MAX_PROFILES_KEY = "multiplex_max_profiles"
PRIORITY_KEY = "multiplex_serve_priority"

#: Prefix every ceiling park reason carries. The reason is how the rest of the gateway tells a
#: PROVISIONAL deferral (this config, this tick — a raised ceiling serves the profile again) from a
#: terminal park (unusable store, home owned elsewhere) that only an explicit rescan retries.
CAPACITY_REASON_PREFIX = "capacity:"

#: ``(path, mtime_ns, size)`` of the config.yaml the cached gateway section was read from. The
#: reader sits on the hot serving path (``profiles_to_serve`` runs per inbound route and per
#: rescan), so a raw config read per call is not affordable; a re-stat per call is.
_CONFIG_MEMO: dict[str, tuple[tuple[int, int], dict]] = {}

_last_deferred: dict[str, str] = {}
#: The last serving decision (ceiling, ranking order, deferrals, demand source), published by
#: ``gateway.run_runtime.publish_serving_decision``.
_last_decision: Optional[dict] = None


def _gateway_section(path: Path) -> dict:
    from hermes_cli.config import read_user_config_raw
    if not path.exists():
        return {}
    cfg = read_user_config_raw(path) or {}
    section = cfg.get("gateway")
    merged: dict = dict(section) if isinstance(section, dict) else {}
    # Top-level aliases exist because users write both spellings; the gateway section wins.
    for key in (MAX_PROFILES_KEY, PRIORITY_KEY):
        if key not in merged and cfg.get(key) is not None:
            merged[key] = cfg[key]
    return merged


def _parse_cap(value: object) -> Optional[int]:
    """A ceiling is a positive integer; anything else (unset, 0, negative, junk) means no ceiling."""
    if isinstance(value, bool):
        return None
    try:
        cap = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return cap if cap > 0 else None


def _parse_priority(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(part) for part in value if str(part).strip())
    return ()


def _read_config_memo(default_home: Path) -> dict:
    path = Path(default_home) / "config.yaml"
    try:
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        signature = (0, 0)
    memo = _CONFIG_MEMO.get(str(path))
    if memo is not None and memo[0] == signature:
        return memo[1]
    try:
        section = _gateway_section(path)
    except Exception:  # an unreadable config must not take the serving path down
        logger.debug("could not read %s for the profile ceiling", path, exc_info=True)
        section = {}
    _CONFIG_MEMO[str(path)] = (signature, section)
    return section


def gateway_section(default_home: Optional[Path] = None) -> dict:
    """The default profile's ``gateway`` section, memoised on ``(mtime, size)``.

    Read-only: returned as the memo holds it, so callers must not mutate. ``hermes_cli
    .gateway_multiplex_demand`` reads ``multiplex_demand_manifest`` through this rather than
    growing a second config memo.
    """
    from hermes_constants import get_default_hermes_root
    return _read_config_memo(Path(default_home) if default_home is not None else get_default_hermes_root())


def profile_ceiling(default_home: Optional[Path] = None) -> Optional[int]:
    """The configured profiles-per-gateway ceiling, or None when the host sets none."""
    from hermes_constants import get_default_hermes_root
    section = _read_config_memo(Path(default_home) if default_home is not None else get_default_hermes_root())
    return _parse_cap(section.get(MAX_PROFILES_KEY))


def serve_priority(default_home: Optional[Path] = None) -> tuple[str, ...]:
    """``gateway.multiplex_serve_priority``: the tiebreak order between profiles of equal demand."""
    from hermes_constants import get_default_hermes_root
    section = _read_config_memo(Path(default_home) if default_home is not None else get_default_hermes_root())
    return _parse_priority(section.get(PRIORITY_KEY))


def apply_ceiling(serve: Sequence[tuple[str, Path]], cap: Optional[int],
                  priority: Iterable[str] = (), *, demand: Any = None) -> tuple[list[tuple[str, Path]], dict[str, str]]:
    """Split *serve* into what this process serves and what the ceiling defers.

    Returns ``(kept, deferred)`` with *deferred* a ``name -> reason`` map (empty when the ceiling
    is unset or covers everyone). ``default`` is always kept, so *cap* 1 means "the default profile
    only" rather than "no gateway".

    *demand* is a :class:`hermes_cli.gateway_multiplex_demand.DemandSnapshot` (anything with
    ``has(name)`` and, for the cost clause, ``for_profile(name)``). ``demand=None`` reproduces the
    pre-SOU-3075 order — priority list, then alphabetical — byte for byte: it is the regression
    guard for every existing caller and test.

    Each deferral reason carries the deferred profile's cost, so the trade is stated where an
    operator reads it (``parked_profiles``, the WARNING, the host check) and not only in a log line
    they never see.
    """
    pairs = list(serve)
    by_name = {name: home for name, home in pairs}
    priority_index = {name: index for index, name in enumerate(n for n in priority if n != "default")}
    if demand is None:
        order = [name for name in priority if name in by_name and name != "default"]
        order += [name for name, _home in pairs if name != "default" and name not in order]
    else:
        order = sorted((name for name in by_name if name != "default"),
                       key=lambda name: (0 if demand.has(name) else 1,
                                         priority_index.get(name, len(priority_index)), name))
    has_default = "default" in by_name
    ranked = (["default"] if has_default else []) + order
    if cap is None or len(pairs) <= cap:
        _remember_decision(cap, ranked, {}, demand)
        return pairs, {}
    keep = (["default"] if has_default else []) + order[:max(0, cap - (1 if has_default else 0))]
    kept_names = set(keep)
    if demand is None:
        # Legacy deferral text: unchanged so a host without a demand view publishes exactly what it
        # published before this module learned about demand.
        reason = f"{CAPACITY_REASON_PREFIX} {cap} profiles per gateway (gateway.{MAX_PROFILES_KEY})"
        deferred = {name: reason for name in ranked if name not in kept_names}
        _remember_decision(cap, ranked, deferred, demand)
    else:
        head = deferral_head(cap)
        deferred = {name: f"{head}; {cost_clause(demand, name)}"
                    for name in ranked if name not in kept_names}
        # Who the demand rank moved across the edge, in both directions: the profiles it KEPT that
        # the pre-SOU-3075 order would have deferred (the pins) and the profiles that order would
        # have kept instead (what the pins cost). PRX-5738 asks for both, by name, at INFO.
        legacy_keep = set(_legacy_order(by_name, priority)[:max(0, cap - (1 if has_default else 0))])
        pinned = {name: demand_signal(demand, name) for name in keep
                  if name != "default" and name not in legacy_keep}
        _remember_decision(cap, ranked, deferred, demand, pinned=pinned,
                           deferred_for_demand=[n for n in legacy_keep if n not in kept_names])
    return [(name, by_name[name]) for name in keep if name in by_name], deferred


def _legacy_order(by_name: dict[str, Path], priority: Iterable[str]) -> list[str]:
    """The pre-SOU-3075 ranking (priority list in written order, then alphabetical).

    Kept as the reference the demand rank is *reported against*, never as a fallback: it is how the
    ceiling would have cut the same roster with no demand view, which is what makes "demand kept
    this one, that one paid for it" a fact rather than a guess.
    """
    names = [name for name in priority if name in by_name and name != "default"]
    return names + sorted(name for name in by_name if name != "default" and name not in names)


def demand_signal(demand: Any, name: str) -> str:
    """One short phrase naming the work that justified keeping *name* (the pin line's evidence)."""
    from hermes_cli import gateway_multiplex_demand as demand_module
    if _demand_flag(demand, "wanted", name):
        return demand_module.wanted_signal(name)
    facts = _demand_facts(demand, name)
    cron_jobs = int(getattr(facts, "cron_jobs", 0) or 0)
    if cron_jobs:
        return f"{cron_jobs} enabled cron job{'s' if cron_jobs != 1 else ''}"
    agents = getattr(facts, "agents", None)
    if agents:
        return f"{int(agents)} Paperclip agents"
    credentials = tuple(getattr(facts, "credentials", ()) or ())
    if credentials:
        return f"connects as a bot ({len(credentials)} adapter credentials)"
    return "demand unreadable" if getattr(facts, "unreadable", ()) else "demand"


def _demand_facts(demand: Any, name: str):
    getter = getattr(demand, "for_profile", None)
    if not callable(getter):
        return None
    try:
        return getter(name)
    except Exception:  # health: allow BLE001 -- a demand view that cannot answer costs a label
        return None


def _demand_flag(demand: Any, flag: str, name: str) -> bool:
    getter = getattr(demand, flag, None)
    if not callable(getter):
        return False
    try:
        return bool(getter(name))
    except Exception:  # health: allow BLE001 -- a label never stops the decision
        return False



def deferral_head(cap: Optional[int]) -> str:
    """The ``capacity: …`` prefix every deferral reason starts with (the check matches on it)."""
    return (f"{CAPACITY_REASON_PREFIX} {cap} profiles per gateway (gateway.{MAX_PROFILES_KEY}) — "
            f"this profile is deferred")


def cost_clause(demand: Any, name: str) -> str:
    """``cost: 12 Paperclip agents, 0 enabled cron jobs, 0 adapter credentials`` — counts only.

    With no manifest read the agent count is ``unknown``, never ``0`` (§3.3): "0 agents" is the one
    claim that would let a deferred profile look like it holds nothing.
    """
    facts = None
    getter = getattr(demand, "for_profile", None)
    if callable(getter):
        try:
            facts = getter(name)
        except Exception:  # health: allow BLE001 -- a demand view that cannot answer costs a label
            facts = None
    agents = getattr(facts, "agents", None)
    agents_text = ("Paperclip agent demand unknown" if agents is None
                   else f"{int(agents)} Paperclip agents")
    cron_jobs = int(getattr(facts, "cron_jobs", 0) or 0)
    credentials = tuple(getattr(facts, "credentials", ()) or ())
    clause = f"cost: {agents_text}, {cron_jobs} enabled cron jobs, {len(credentials)} adapter credentials"
    unreadable = tuple(getattr(facts, "unreadable", ()) or ())
    if unreadable:
        clause += f", demand unreadable: {', '.join(str(source) for source in unreadable)}"
    if _demand_flag(demand, "wanted", name):
        # The one cost an operator must never have to infer: somebody is blocked on this profile.
        from hermes_cli import gateway_multiplex_demand as demand_module
        clause += f", {demand_module.wanted_signal(name)}"
    return clause


def _remember_decision(cap: Optional[int], ranked: Sequence[str], deferred: dict[str, str],
                       demand: Any, *, pinned: Optional[dict[str, str]] = None,
                       deferred_for_demand: Sequence[str] = ()) -> None:
    """Stash the ordered decision for ``last_decision()`` / the runtime-status publisher."""
    global _last_decision
    source = None
    getter = getattr(demand, "demand_source", None)
    if callable(getter):
        try:
            source = dict(getter())
        except Exception:  # health: allow BLE001 -- a missing label never stops the decision
            source = None
    _last_decision = {
        "ceiling": cap,
        "cap_note": f"gateway.{MAX_PROFILES_KEY}",
        "ranked": list(ranked),
        "deferred": dict(deferred),
        "demand_source": source,
        "pinned": dict(pinned or {}),
        "deferred_for_demand": list(deferred_for_demand),
    }


def last_decision() -> Optional[dict]:
    """The last decision ``apply_ceiling`` made, or None before it ran in this process."""
    decision = _last_decision
    if decision is None:
        return None
    return {
        "ceiling": decision["ceiling"],
        "cap_note": decision["cap_note"],
        "ranked": list(decision["ranked"]),
        "deferred": dict(decision["deferred"]),
        "demand_source": dict(decision["demand_source"]) if decision.get("demand_source") else None,
        "pinned": dict(decision.get("pinned") or {}),
        "deferred_for_demand": list(decision.get("deferred_for_demand") or []),
    }



def _deferral_summary(name: str, reason: str) -> str:
    """``cfo (0 Paperclip agents, 3 enabled cron jobs, 0 adapter credentials)`` for the WARNING.

    The counts live in the reason's ``; cost: …`` clause (built once, in ``apply_ceiling``), so the
    log line and the published park reason can never disagree about a profile's cost.
    """
    marker = "; cost: "
    index = reason.find(marker)
    return f"{name} ({reason[index + len(marker):]})" if index >= 0 else name


def remember_deferred(deferred: dict[str, str]) -> None:
    """Stash this process's last capacity decision for the runtime-status publisher."""
    global _last_deferred
    if deferred == _last_deferred:
        return
    if deferred:
        # Name what still works as well as what stops: an operator reading only this line used to
        # conclude cron was dead host-wide and reorder the priority list to "fix" it (PRX-5717).
        # Since SOU-3075 the line also names each deferred profile's cost, so "deferred" reads as a
        # trade rather than a mystery.
        logger.warning(
            "[MULTIPLEX] gateway.%s leaves %d profile(s) unserved on this gateway: %s "
            "(no adapters, no inbound routing; their cron jobs ARE still ticked, delivery falls "
            "back to a routed adapter or fails closed)",
            MAX_PROFILES_KEY, len(deferred),
            ", ".join(_deferral_summary(name, deferred[name]) for name in sorted(deferred)))
    else:
        logger.info("[MULTIPLEX] gateway.%s: every profile fits on this gateway again", MAX_PROFILES_KEY)
    _last_deferred = dict(deferred)
    _log_demand_pins(deferred)


def _log_demand_pins(deferred: dict[str, str]) -> None:
    """INFO: which profiles demand kept at the edge, and which profile paid for each (PRX-5738).

    ``parked_profiles`` answers "who is not served"; it cannot answer the operator's actual question
    after an outage — *why is this one still up and that one down*. So the decision reports itself:
    the profile the pre-demand order would have deferred but demand kept, with the signal that
    justified it, next to the idle profile that lost the slot instead. Only on a changed decision,
    and only when demand moved the edge — a roster where the rank changes nothing logs no extra line.
    """
    decision = _last_decision or {}
    pinned = decision.get("pinned") or {}
    if not deferred or not pinned:
        return
    kept = ", ".join(f"'{name}' ({signal})" for name, signal in sorted(pinned.items()))
    paid = ", ".join(f"'{name}'" for name in sorted(decision.get("deferred_for_demand") or []))
    logger.info("[MULTIPLEX] Pinned by demand: %s kept a slot; %s deferred instead (idle at the "
                "ceiling edge)", kept, paid or "nobody")



def deferred_by_capacity() -> dict[str, str]:
    """Profiles the last serving enumeration deferred because of the ceiling."""
    return dict(_last_deferred)


def reset_memo() -> None:
    """Drop the config + decision caches (tests, and an operator edit observed in-process)."""
    global _last_decision
    _CONFIG_MEMO.clear()
    _last_deferred.clear()
    _last_decision = None


__all__ = [
    "CAPACITY_REASON_PREFIX", "MAX_PROFILES_KEY", "PRIORITY_KEY", "apply_ceiling", "cost_clause",
    "deferral_head", "deferred_by_capacity", "demand_signal", "gateway_section", "last_decision",
    "profile_ceiling", "remember_deferred", "reset_memo", "serve_priority",
]
