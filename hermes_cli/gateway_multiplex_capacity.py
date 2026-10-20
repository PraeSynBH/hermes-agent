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
socket). Named profiles are taken from ``gateway.serve_priority`` first — in the order written —
then alphabetically. Explicit priority is what keeps a ceiling from silently dropping whichever
profile sorts late; without it the choice is at least deterministic and stable across restarts.

This module owns the decision only. The installed roster (``include_parked``/``include_standalone``
enumerations) is never capped: a profile past the ceiling still exists, is still listed, and is
still started once the ceiling or the priority list makes room.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

MAX_PROFILES_KEY = "multiplex_max_profiles"
PRIORITY_KEY = "multiplex_serve_priority"

#: Prefix every ceiling park reason carries. The reason is how the rest of the gateway tells a
#: PROVISIONAL deferral (this config, this tick — a raised ceiling serves the profile again) from a
#: terminal park (unusable store, home owned elsewhere) that only an explicit rescan retries.
CAPACITY_REASON_PREFIX = "capacity:"

#: ``(path, mtime_ns, size)`` of the config.yaml the cached ceiling was read from. The reader sits on
#: the hot serving path (``profiles_to_serve`` runs per inbound route and per rescan), so a raw
#: config read per call is not affordable; a re-stat per call is.
_CONFIG_MEMO: dict[str, tuple[tuple[int, int], Optional[int], tuple[str, ...]]] = {}

_last_deferred: dict[str, str] = {}


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


def _read_config_memo(default_home: Path) -> tuple[Optional[int], tuple[str, ...]]:
    path = Path(default_home) / "config.yaml"
    try:
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        signature = (0, 0)
    memo = _CONFIG_MEMO.get(str(path))
    if memo is not None and memo[0] == signature:
        return memo[1], memo[2]
    try:
        section = _gateway_section(path)
    except Exception:  # an unreadable config must not take the serving path down
        logger.debug("could not read %s for the profile ceiling", path, exc_info=True)
        section = {}
    cap = _parse_cap(section.get(MAX_PROFILES_KEY))
    priority = _parse_priority(section.get(PRIORITY_KEY))
    _CONFIG_MEMO[str(path)] = (signature, cap, priority)
    return cap, priority


def profile_ceiling(default_home: Optional[Path] = None) -> Optional[int]:
    """The configured profiles-per-gateway ceiling, or None when the host sets none."""
    from hermes_constants import get_default_hermes_root
    return _read_config_memo(Path(default_home) if default_home is not None else get_default_hermes_root())[0]


def serve_priority(default_home: Optional[Path] = None) -> tuple[str, ...]:
    """``gateway.multiplex_serve_priority``: named profiles that must survive the ceiling, in order."""
    from hermes_constants import get_default_hermes_root
    return _read_config_memo(Path(default_home) if default_home is not None else get_default_hermes_root())[1]


def apply_ceiling(serve: Sequence[tuple[str, Path]], cap: Optional[int],
                  priority: Iterable[str] = ()) -> tuple[list[tuple[str, Path]], dict[str, str]]:
    """Split *serve* into what this process serves and what the ceiling defers.

    Returns ``(kept, deferred)`` with *deferred* a ``name -> reason`` map (empty when the ceiling
    is unset or covers everyone). ``default`` is always kept, so *cap* 1 means "the default profile
    only" rather than "no gateway".
    """
    pairs = list(serve)
    if cap is None or len(pairs) <= cap:
        return pairs, {}
    by_name = {name: home for name, home in pairs}
    order = [name for name in priority if name in by_name and name != "default"]
    order += [name for name, _home in pairs if name != "default" and name not in order]
    keep = (["default"] if "default" in by_name else []) + order[:max(0, cap - (1 if "default" in by_name else 0))]
    kept_names = set(keep)
    reason = f"{CAPACITY_REASON_PREFIX} {cap} profiles per gateway (gateway.{MAX_PROFILES_KEY})"
    deferred = {name: reason for name, _home in pairs if name not in kept_names}
    return [(name, by_name[name]) for name in keep if name in by_name], deferred


def remember_deferred(deferred: dict[str, str]) -> None:
    """Stash this process's last capacity decision for the runtime-status publisher."""
    global _last_deferred
    if deferred == _last_deferred:
        return
    if deferred:
        # Name what still works as well as what stops: an operator reading only this line used to
        # conclude cron was dead host-wide and reorder the priority list to "fix" it (PRX-5717).
        logger.warning(
            "[MULTIPLEX] gateway.%s leaves %d profile(s) unserved on this gateway: %s "
            "(no adapters, no inbound routing; their cron jobs ARE still ticked, delivery falls "
            "back to a routed adapter or fails closed)",
            MAX_PROFILES_KEY, len(deferred), ", ".join(sorted(deferred)))
    else:
        logger.info("[MULTIPLEX] gateway.%s: every profile fits on this gateway again", MAX_PROFILES_KEY)
    _last_deferred = dict(deferred)


def deferred_by_capacity() -> dict[str, str]:
    """Profiles the last serving enumeration deferred because of the ceiling."""
    return dict(_last_deferred)


def reset_memo() -> None:
    """Drop the config + decision caches (tests, and an operator edit observed in-process)."""
    _CONFIG_MEMO.clear()
    _last_deferred.clear()


__all__ = [
    "CAPACITY_REASON_PREFIX", "MAX_PROFILES_KEY", "PRIORITY_KEY", "apply_ceiling", "deferred_by_capacity",
    "profile_ceiling", "remember_deferred", "reset_memo", "serve_priority",
]
