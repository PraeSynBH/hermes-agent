"""Publish what the process knows about its language servers, for instruments outside it.

A census taken from ``ps`` can see a server's pid, its RSS and its cwd — and nothing about *whose*
work it is.  That is why the fleet's reaper cannot answer the one question PRA-6854's acceptance
turns on ("is this root's owning run terminal?"): a worktree path carries no run id, and the newest
worktrees are named by hash.  The service knows the requesting session, so it writes it down:

    <hermes home>/state/lsp_census.json

One entry per live client: server id, workspace root, pid, the owner session that last used it,
seconds since that use, and the process tree's RSS.  Readers join ``owner`` to their own run
records; nothing here is a lifetime rule, only evidence.  Written after every sweep, atomically, and
readable by any process — the gateway's own state, not a control surface.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

CENSUS_FILENAME = "lsp_census.json"


def census_path() -> Path:
    """``<hermes home>/state/lsp_census.json`` for the ACTIVE profile (callers bind the scope)."""
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "state" / CENSUS_FILENAME


def write_census(entries: list[dict[str, Any]], *, path: Optional[Path] = None) -> Optional[Path]:
    """Atomically replace the census file with *entries*; ``None`` if it could not be written.

    Best-effort by construction: a census is an instrument, and an instrument must never fail a
    sweep, so an unwritable home (read-only mount, deleted profile) is reported by returning None.
    """
    target = path or census_path()
    payload = {"written_at": time.time(), "pid": os.getpid(), "clients": entries}
    tmp_name = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=str(target.parent), prefix=".lsp_census-",
                                         suffix=".json", delete=False, encoding="utf-8") as tmp:
            tmp_name = tmp.name
            json.dump(payload, tmp, sort_keys=True)
        os.replace(tmp_name, target)
        return target
    except OSError:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        return None


__all__ = ["CENSUS_FILENAME", "census_path", "write_census"]
