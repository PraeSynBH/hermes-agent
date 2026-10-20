"""Which profiles a gateway serves: ``profiles_to_serve`` and the capacity ceiling.

Split out of ``hermes_cli/profiles.py`` so the serving decision (and the
``gateway.multiplex_max_profiles`` ceiling it honours) lives in one place instead of growing the
profile-management facade. Callers keep importing ``hermes_cli.profiles.profiles_to_serve`` — the
facade re-exports this one.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parked_marker_path(home: Path) -> Path:
    return Path(home) / "gateway.parked"


def profile_is_parked(home: Path) -> bool:
    """Marker contents are deliberately irrelevant, including for provisioning."""
    return parked_marker_path(home).exists()


_parked_default_warned: set[Path] = set()


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
    from hermes_cli.profiles import (_get_default_hermes_home, _iter_named_profile_dirs,
                                     get_active_profile_name, get_profile_dir,
                                     profile_is_standalone)
    active = get_active_profile_name() or "default"
    default = _get_default_hermes_home()
    if profile_is_parked(default) and default not in _parked_default_warned:
        logger.warning("Ignoring gateway.parked for the default profile; stop the host gateway instead")
        _parked_default_warned.add(default)
    if not multiplex:
        return [(active, get_profile_dir(active))]
    serve: list[tuple[str, Path]] = [("default", default)]
    serve.extend((entry.name, entry) for entry in _iter_named_profile_dirs()
                 if (include_standalone or not profile_is_standalone(entry))
                 and (include_parked or not profile_is_parked(entry)))
    if include_parked or include_standalone:
        return serve
    from hermes_cli import gateway_multiplex_capacity as capacity
    serve, deferred = capacity.apply_ceiling(serve, capacity.profile_ceiling(),
                                             capacity.serve_priority())
    capacity.remember_deferred(deferred)
    return serve
