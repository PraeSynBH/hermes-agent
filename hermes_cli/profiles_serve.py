"""Which profiles a gateway serves: ``profiles_to_serve`` and the capacity ceiling.

Split out of ``hermes_cli/profiles.py`` so the serving decision (and the
``gateway.multiplex_max_profiles`` ceiling it honours) lives in one place instead of growing the
profile-management facade. Callers keep importing ``hermes_cli.profiles.profiles_to_serve`` — the
facade re-exports this one.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


def parked_marker_path(home: Path) -> Path:
    return Path(home) / "gateway.parked"


def profile_is_parked(home: Path) -> bool:
    """Marker contents are deliberately irrelevant, including for provisioning."""
    return parked_marker_path(home).exists()


_parked_default_warned: set[Path] = set()


def _warn_if_default_parked(default: Path) -> None:
    if profile_is_parked(default) and default not in _parked_default_warned:
        logger.warning("Ignoring gateway.parked for the default profile; stop the host gateway instead")
        _parked_default_warned.add(default)


def _live_roster(*, include_standalone: bool, include_parked: bool) -> list[tuple[str, Path]]:
    """``default`` plus every live named profile under ``profiles/``, before any capacity decision."""
    from hermes_cli.profiles import _get_default_hermes_home, _iter_named_profile_dirs, profile_is_standalone
    default = _get_default_hermes_home()
    _warn_if_default_parked(default)
    serve: list[tuple[str, Path]] = [("default", default)]
    serve.extend((entry.name, entry) for entry in _iter_named_profile_dirs()
                 if (include_standalone or not profile_is_standalone(entry))
                 and (include_parked or not profile_is_parked(entry)))
    return serve


def profiles_to_serve(multiplex: bool, *, include_standalone: bool = False,
                      include_parked: bool = False) -> list[tuple[str, Path]]:
    """``(profile_name, hermes_home)`` pairs a gateway should serve — the single chokepoint
    for "which profiles does the inbound gateway handle".

    ``multiplex=False``: exactly one entry for the *active* profile (byte-for-byte the
    historical single-profile behavior; name is ``"default"`` or the named profile's id).
    ``multiplex=True``: default plus every live named profile under ``profiles/`` (tombstoned
    and parked profiles skipped). Pure directory read: never creates a profile dir (#94590).

    Named profiles that authored ``gateway.standalone: true`` are skipped because they opted
    out of the host multiplexer; a ``gateway.parked`` marker (``hermes -p X gateway stop``) skips
    a profile the host would otherwise serve. Callers enumerating INSTALLED profiles pass
    ``include_standalone=True, include_parked=True``; serving/ticking callers pass neither.

    Serving calls also honour ``gateway.multiplex_max_profiles``
    (:mod:`hermes_cli.gateway_multiplex_capacity`): past the ceiling a profile is not served by
    this process at all, so the per-profile thread baseline stops scaling with the roster. The
    ceiling is never applied to an INSTALLED enumeration — a deferred profile still exists and is
    still listed/startable, it just has no adapters on THIS gateway.
    """
    from hermes_cli.profiles import get_active_profile_name, get_profile_dir
    active = get_active_profile_name() or "default"
    if not multiplex:
        from hermes_cli.profiles import _get_default_hermes_home
        _warn_if_default_parked(_get_default_hermes_home())
        return [(active, get_profile_dir(active))]
    serve = _live_roster(include_standalone=include_standalone, include_parked=include_parked)
    if include_parked or include_standalone:
        return serve
    from hermes_cli import gateway_multiplex_capacity as capacity
    cap = capacity.profile_ceiling()
    # Ranking needs live demand (SOU-3075), but a host with no ceiling ranks nothing: building the
    # snapshot there would put a stat sweep of every profile's cron store and .env on a path that
    # runs per inbound route, for a decision that never defers anyone.
    demand = None
    if cap is not None:
        from hermes_cli import gateway_multiplex_demand as demand_module
        demand = demand_module.snapshot()
    serve, deferred = capacity.apply_ceiling(serve, cap, capacity.serve_priority(), demand=demand)
    capacity.remember_deferred(deferred)
    return serve


def standalone_profiles() -> list[tuple[str, Path]]:
    """Live named profiles whose own config opted out of the host multiplexer.

    This is the *config* fact (``gateway.standalone: true``), which is what a status line claiming
    "standalone by config" has to report. It is deliberately not ``roster - served``: the served set
    is capped by ``gateway.multiplex_max_profiles`` and ranked by live demand, so that difference is
    the capacity-deferred set and it moves whenever demand does (PRX-5944).
    """
    from hermes_cli.profiles import _iter_named_profile_dirs, profile_is_standalone
    return [(entry.name, entry) for entry in _iter_named_profile_dirs() if profile_is_standalone(entry)]


def profiles_to_tick(multiplex: bool, *, reserved: Iterable[tuple[str, Path]] = (),
                     active: str = "") -> list[tuple[str, Path]]:
    """``(profile_name, hermes_home)`` pairs a gateway's cron ticker has to visit.

    The ticker's set is NOT the served set: ``gateway.multiplex_max_profiles`` caps what this
    process SERVES (adapters, session authority, the per-profile thread baseline), and a profile
    past the ceiling still keeps its ``cron/`` store. Deriving the tick set from the cap left every
    deferred profile's jobs in a store no ticker visited — they never fired, with no trace beyond a
    ``capacity:`` park reason (PRX-5717) — so this enumeration deliberately skips
    :func:`profiles_to_serve`'s ceiling.

    Parked and standalone profiles stay out, exactly as they do for serving: a ``gateway.parked``
    profile was stopped on purpose, and a standalone one has its own gateway process ticking it
    (``gateway.run._cron_profile_gate`` would stand this ticker down per tick anyway). One process
    ticks every profile either way — ``gateway.multiplex_profiles`` gates adapters, not cron.

    ``reserved`` is what this process reserved (the served set as the gateway took it) and ``active``
    its launch profile: a ``--profile <name>`` gateway's own home can sit outside ``profiles/``, so
    both are added by name when the roster does not already list them.
    """
    if not multiplex:
        from hermes_cli.profiles import get_active_profile_name, get_profile_dir
        name = get_active_profile_name() or "default"
        return [(name, get_profile_dir(name))]
    homes = _live_roster(include_standalone=False, include_parked=False)
    seen = {name for name, _home in homes}
    for name, home in reserved:
        if name not in seen:
            homes.append((name, home))
            seen.add(name)
    if active and active not in seen:
        from hermes_cli.profiles import get_profile_dir
        try:
            homes.append((active, get_profile_dir(active)))
        except ValueError:
            pass  # not a profile id (a custom home): nothing under profiles/ to tick
    return homes
