"""The wire-contract revision a process SERVES, versus the one the tree defines.

SOU-3316: ``merge(PRX-5850)`` added ``one_shot`` to the submit allow-list while a gateway
process booted 4h14m earlier still held the old set. Every ``hermes -z`` child spawned from the
new tree declared a field that process did not define, so the refusal was a bare
``invalid_params`` — indistinguishable from a malformed request — for 3h24m and 71 runs.

Two things make that skew nameable, and both live here:

* :func:`revision` — a content digest of the declared contract document
  (``apps/shared/src/gateway-contract.openrpc.json``, the same file the generated TS is rendered
  from). Any edit to the wire contract changes it, so nobody has to remember to bump a constant.
* :func:`record_boot_revision` / :func:`boot_revision` — the digest THIS process imported at
  startup. A gateway keeps serving the module versions it booted with, so the question "what does
  the serving process define?" is answered by the boot snapshot, never by re-reading the file: a
  stale process that re-read the tree would report the new revision and look healthy, which is
  precisely the failure this closes.

Absent or unreadable contract document (an install shipped without ``apps/``) yields ``None``
everywhere: a client then declares nothing and the owner keeps its closed-set ``invalid_params``,
which is today's behaviour. The diagnostic degrades, validation never loosens.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_RELATIVE_PATH = ("apps", "shared", "src", "gateway-contract.openrpc.json")

#: The submit param a client uses to declare the contract revision it was built against. Optional:
#: a client that sends nothing keeps today's behaviour exactly.
REVISION_PARAM = "contract_revision"

#: Compatibility value for a revision digest; long enough to name a change, short enough for a log
#: line and a refusal's ``detail``.
_REVISION_CHARS = 12

_cache: dict[tuple[str, int, int], str | None] = {}
_boot_revision: str | None = None
_boot_recorded = False


def contract_path(root: Path | None = None) -> Path:
    """The declared contract document this install would render from."""
    return Path(root or _PROJECT_ROOT).joinpath(*CONTRACT_RELATIVE_PATH)


def revision(root: Path | None = None) -> str | None:
    """Content digest of the declared wire contract, or ``None`` when the file is unreadable.

    Cached per path+mtime+size: the gateway asks on every refused submit, and the file only
    changes when a deploy lands.
    """
    path = contract_path(root)
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None
    if key in _cache:
        return _cache[key]
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:_REVISION_CHARS]
    except OSError:
        digest = None
    _cache[key] = digest
    return digest


def record_boot_revision(root: Path | None = None) -> str | None:
    """Snapshot the contract this process serves. Idempotent; only the first call counts."""
    global _boot_revision, _boot_recorded
    if not _boot_recorded:
        _boot_revision = revision(root)
        _boot_recorded = True
    return _boot_revision


def boot_revision() -> str | None:
    """The contract this process serves: its boot snapshot, or the tree's when never recorded.

    An un-recorded process (a unit test, a short-lived tool) has no stale modules to speak of, so
    the current digest is the honest answer. A gateway records at startup
    (``gateway/run_bootstrap.py``), which is what makes a mid-flight merge visible.
    """
    return _boot_revision if _boot_recorded else revision()


def reset_boot_revision() -> None:
    """Forget the boot snapshot (tests and in-process re-bootstrap only)."""
    global _boot_revision, _boot_recorded
    _boot_revision, _boot_recorded = None, False


def declaration(root: Path | None = None) -> dict[str, str]:
    """The submit params a freshly-spawned client attaches to declare its contract revision."""
    current = revision(root)
    return {REVISION_PARAM: current} if current else {}


def describe(served: str | None) -> str:
    """A revision as a log line and a refusal's ``detail`` should carry it."""
    return f'contract-{served}' if served else 'unknown-contract'
