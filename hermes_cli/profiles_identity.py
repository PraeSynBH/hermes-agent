"""The profile this process's HOME names, derived from that path rather than from the filesystem.

``get_active_profile_name`` is a pure function of the home path, but answering it resolves three
paths (the home, the default root, the profiles root) and each ``resolve`` is a ``realpath`` walk —
pure Python under the GIL. It sits under every runtime-status publish
(``agent.monitoring.gateway_health._safe_profile``), which the profile reconciler calls on the
gateway loop once per tick, so at a few hundred competing threads those walks cost the loop its
grip: 91 of 268 sampled slow-tick frames were inside this chain (PRA-6825/PRA-6826).

The answer only changes when the home path does — both roots are derived from the same
``HERMES_HOME`` — so it is remembered per home path: one entry per profile the process serves,
never a single process-wide value (one process serves many profiles).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

_ACTIVE_PROFILE_NAME: dict[str, str] = {}


def get_active_profile_name() -> str:
    """Profile name inferred from HERMES_HOME: ``"default"`` when unset or ``~/.hermes``, the
    name under ``~/.hermes/profiles/<name>``, ``"custom"`` for any other path."""
    from hermes_constants import get_hermes_home

    from hermes_cli.profiles import _PROFILE_ID_RE, _get_default_hermes_home, _get_profiles_root

    home = str(get_hermes_home())
    cached = _ACTIVE_PROFILE_NAME.get(home)
    if cached is not None:
        return cached
    resolved = Path(home).resolve()
    name = "custom"
    if resolved == _get_default_hermes_home().resolve():
        name = "default"
    else:
        profiles_root = _get_profiles_root().resolve()
        with contextlib.suppress(ValueError):
            parts = resolved.relative_to(profiles_root).parts
            if len(parts) == 1 and _PROFILE_ID_RE.match(parts[0]):
                name = parts[0]
    _ACTIVE_PROFILE_NAME[home] = name
    return name
