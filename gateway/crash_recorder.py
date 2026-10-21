"""Fatal-signal crash recorder — a parseable record for deaths that run no exit path.

A SIGBUS / SIGSEGV / SIGABRT kills the process inside kernel signal delivery, so every
Python-level exit hook in ``run_gateway`` (``atexit``, ``except SystemExit``, the
``gateway.exit_*`` rows) is unreachable. On 2026-10-21 pid 12392 died of SIGBUS with
nothing in ``gateway-exit-diag.log`` — ``Fatal Python error: Bus error`` in
``gateway.error.log``, faulthandler itself dying mid-dump after 23 of 274 threads, and
the next boot's ``gateway.previous_unclean_exit`` row carrying no cause at all. A death
by fatal signal was indistinguishable from "unknown", which silently restarted the soak
windows gated on that row (PRA-6502 / PRA-6380).

``faulthandler`` is the only writer that can run in that signal context, so the recorder
IS a ``faulthandler.enable()`` pointed at its own log:

* On CPython >= 3.14 the fatal signals cannot be ``faulthandler.register()``-ed at all
  (``RuntimeError: signal 10 cannot be registered, use enable() instead``) — they belong
  to the fatal handler ``enable()`` installs, which is why this module wraps ``enable``
  rather than registering a handler per signal.
* The ``O_APPEND`` descriptor is opened once per life and held for its whole lifetime: a
  file handle allocated inside the signal handler is exactly what a dying process cannot
  afford, and ``faulthandler`` keeps only the number, so the handle must stay alive or a
  garbage collection closes the fd the dump is meant to go to.
* Each life writes one JSON header line at arm time (``ts``, ``pid``, ``python``,
  ``platform``, ``signals``), so the file is a sequence of records delimited without
  heuristics and the crash dump is the tail of the newest one.
* ``all_threads=False`` is deliberate. Both crashes this protects against faulted inside
  the all-thread walk from signal context (PRA-6536: pid 67785, 240 threads,
  ``EXC_BAD_ACCESS`` in ``_Py_DumpTracebackThreads``; pid 12392: truncated at 23/274
  threads). The thread that received a fatal signal is the one whose C stack names the
  fault, and dumping only it is a bounded, single-thread walk that completes.

:func:`prior_crash` reads the record back on the next boot; the signal comes from
faulthandler's own fatal banner (``Modules/faulthandler.c`` names the *reason*, never the
number — one table below), and the crash time is the mtime of the write that recorded it,
so no probe of a dead process is needed.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CRASH_LOG_NAME = "gateway-crash.log"
ARM_TAG = "gateway.crash_recorder.armed"

#: faulthandler's fatal-signal wording -> the signal that produced it. The banner is
#: ``Fatal Python error: <reason>``; these are the five signal reasons it can carry.
_REASON_SIGNALS = {
    "segmentation fault": "SIGSEGV",
    "bus error": "SIGBUS",
    "aborted": "SIGABRT",
    "illegal instruction": "SIGILL",
    "floating-point exception": "SIGFPE",
}
_FATAL_SIGNAL_NAMES = tuple(_REASON_SIGNALS.values())
_BANNER = "Fatal Python error:"
#: A 274-thread dump is far below this; the bound keeps a pathological log from being
#: slurped whole on every boot.
_TAIL_BYTES = 2 * 1024 * 1024

_armed: Optional[dict[str, Any]] = None
_prior_cache: dict[Path, Optional[dict[str, Any]]] = {}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def crash_log_path(home: Optional[Path | str] = None) -> Path:
    """``<HERMES_HOME>/logs/gateway-crash.log`` (``home`` overrides for tests/other profiles)."""
    if home is not None:
        return Path(home) / "logs" / CRASH_LOG_NAME
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "logs" / CRASH_LOG_NAME


def is_armed() -> bool:
    """True once this process's recorder owns an open crash-log descriptor."""
    return _armed is not None and not _armed["handle"].closed


def _enable_stderr_fallback() -> None:
    """Last resort when the crash log cannot be opened: keep the pre-PRA-6669 behaviour.

    ``faulthandler.enable()`` to stderr can itself fail when stderr is None (Windows VBS /
    pythonw / detached service) — the gateway must not die for a diagnostic (#71671).
    """
    try:
        import faulthandler

        faulthandler.enable()
    except Exception:
        logger.debug("faulthandler.enable() unavailable", exc_info=True)


def arm(log_dir: Optional[Path | str] = None, *, home: Optional[Path | str] = None) -> bool:
    """Point faulthandler's fatal-signal handler at the crash log. Idempotent; never raises.

    ``log_dir`` is the runner's configured log directory; ``home`` names a profile home for
    callers that only have one. Both default to ``<HERMES_HOME>/logs``.
    """
    global _armed
    path = (Path(log_dir) / CRASH_LOG_NAME) if log_dir is not None else crash_log_path(home)
    armed = _armed
    if armed is not None and is_armed():
        return armed["path"] == path
    try:
        import faulthandler

        path.parent.mkdir(parents=True, exist_ok=True)
        handle = os.fdopen(
            os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644),
            "a",
            encoding="utf-8",
        )
        header = {
            "ts": _now_iso(),
            "tag": ARM_TAG,
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "signals": [name for name in _FATAL_SIGNAL_NAMES if hasattr(signal, name)],
        }
        handle.write(json.dumps(header) + "\n")
        handle.flush()
        faulthandler.enable(file=handle, all_threads=False)
    except Exception:
        logger.debug("Fatal-signal crash recorder could not arm", exc_info=True)
        _enable_stderr_fallback()
        return False
    # Hold the handle for the process lifetime: faulthandler keeps only the descriptor.
    _armed = {"path": path, "handle": handle}
    return True


def prior_crash(home: Optional[Path | str] = None, *, path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """The newest fatal-signal record in the crash log, or ``None``. Cached; never raises.

    ``{"signal", "signal_number", "reason", "ts", "pid", "python", "traceback"}`` — ``ts`` is
    the mtime of the write that recorded the crash, ``traceback`` says whether any stack made
    it out (a dump can be truncated by the fault it is reporting). Callers match ``pid``
    against the previous life's pid: a record from an older life is not this boot's news.
    """
    log_path = path if path is not None else crash_log_path(home)
    if log_path not in _prior_cache:
        try:
            _prior_cache[log_path] = _scan_newest_crash(log_path)
        except Exception:
            logger.debug("Could not read the crash record", exc_info=True)
            _prior_cache[log_path] = None
    return _prior_cache[log_path]


def _scan_newest_crash(path: Path) -> Optional[dict[str, Any]]:
    """Newest record (arm header + its tail) that carries a fatal banner, else ``None``.

    Delimited by the headers themselves, so no line is classified by guessing.
    """
    try:
        with path.open("rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, size - _TAIL_BYTES))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    records: list[tuple[dict[str, Any], list[str]]] = []
    for line in text.splitlines():
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("tag") == ARM_TAG:
                records.append((parsed, []))
                continue
        if records:
            records[-1][1].append(line)
    for header, tail in reversed(records):
        for index, line in enumerate(tail):
            if not line.startswith(_BANNER):
                continue
            reason = line[len(_BANNER):].strip()
            name = _REASON_SIGNALS.get(reason.lower())
            number = getattr(signal, name, None) if name else None
            return {
                "signal": name,
                "signal_number": number if isinstance(number, int) else None,
                "reason": reason or None,
                "ts": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                "pid": header.get("pid"),
                "python": header.get("python"),
                "traceback": any(entry.startswith('  File "') for entry in tail[index:]),
            }
    return None
