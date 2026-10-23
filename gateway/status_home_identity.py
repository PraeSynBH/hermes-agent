"""The canonical form of a HERMES_HOME path, for identity records that get persisted and compared.

A process names a handful of homes at most, and resolving one is a ``realpath`` walk — pure Python
under the GIL. The pid record is rebuilt on every ``publish_runtime_status``, which the profile
reconciler runs on the gateway loop once per tick, so at a few hundred competing threads that walk
costs the loop seconds per publish (PRA-6825/PRA-6826). The resolved form is therefore remembered
per input path string: a different home (the multiplexed per-profile read) is a different key.
"""

from __future__ import annotations

from pathlib import Path

_CANONICAL_HERMES_HOME: dict[str, Path] = {}


def canonical_hermes_home(path: Path | str) -> Path:
    """Stable absolute HERMES_HOME path for persisted identity data."""
    key = str(path)
    cached = _CANONICAL_HERMES_HOME.get(key)
    if cached is None:
        cached = _CANONICAL_HERMES_HOME[key] = Path(path).expanduser().resolve(strict=False)
    return cached
