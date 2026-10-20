"""The tiny JSON file I/O the runtime record is read and written through.

Split out of ``gateway/status.py`` (SOU-3075) so the facade can keep taking on record FIELDS without
taking on file I/O line count: ``status.py`` is already over the code-health target and may only go
down. Behaviour is unchanged — ``gateway.status`` imports these names, so existing callers keep
importing them from there.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Optional

from utils import atomic_json_write


def _read_json_file(path: Path, *, bare_pid_ok: bool = False) -> Optional[dict[str, Any]]:
    """JSON object at ``path``, or None when absent/empty/unreadable/invalid. ``bare_pid_ok`` also
    accepts legacy bare-integer PID files as ``{"pid": N}``."""
    try:
        raw = path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeDecodeError):
        # OSError: file vanished or permission flipped between exists() and
        # read. UnicodeDecodeError: file holds non-UTF-8 / binary garbage
        # (a truncated or clobbered status file). Either way it's unusable.
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
        if bare_pid_ok:
            with contextlib.suppress(ValueError):
                payload = int(raw)
    if bare_pid_ok and isinstance(payload, int):
        return {"pid": payload}
    return payload if isinstance(payload, dict) else None


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    atomic_json_write(path, payload, indent=None, separators=(",", ":"))


def _unlink_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


__all__ = ["_read_json_file", "_unlink_quietly", "_write_json_file"]
