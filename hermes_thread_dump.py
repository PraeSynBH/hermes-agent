"""The one bounded, rate-limited all-thread dump for "the process looks wedged" diagnostics.

``faulthandler.dump_traceback(all_threads=True)`` walks every thread state in C
(``_Py_DumpTracebackThreads``). On 2026-10-19 that walk faulted while the gateway's loop was
blocked and killed the process — exit 139 (SIGSEGV) at 89 minutes of uptime, taking six
in-flight runs with it. The diagnostic that exists to make a blocked loop visible had become
the largest single cause of fleet run loss we have measured.

So the full dump is produced here, once, and only here:

* frames come from ``sys._current_frames()`` plus ``traceback.format_stack`` — pure Python,
  no C thread-state walk, and a thread that dies mid-walk is simply absent instead of
  faulting the process;
* fan-out is capped (``max_threads``) and the payload is capped (``max_chars``);
* at most one dump per process per ``min_interval_s`` — a blocked loop repeats (1,102
  deathwatch samples in the 2026-10-19 case needed one dump, not 1,102);
* nothing raises: when the frames cannot be read the caller gets ``""`` and a one-line log,
  because a diagnostic must never abort the request it is diagnosing.

``faulthandler`` stays for the fatal path (``faulthandler.enable``/SIGABRT), where the
process is lost anyway.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
from typing import IO, Optional

logger = logging.getLogger(__name__)

# 200-250 live threads is the gateway's production plateau (PRA-6507), so 64 bounded dumps
# cover the interesting frames (main + the busiest few) without a 250-thread write.
DEFAULT_MAX_THREADS = 64
DEFAULT_MAX_CHARS = 60_000
# One dump per five minutes is already more than a diagnosis needs; the second and later
# fires of the same wedge exist only to be counted.
DEFAULT_MIN_INTERVAL_S = 300.0

_lock = threading.Lock()
_last_dump_at: float = 0.0


def reset_dump_rate_limit() -> None:
    """Clear the per-process dump floor (a fresh process, or a test that dumps twice)."""
    global _last_dump_at
    with _lock:
        _last_dump_at = 0.0


def render_thread_stacks(
    *, max_threads: int = DEFAULT_MAX_THREADS, max_chars: int = DEFAULT_MAX_CHARS
) -> str:
    """Every thread's stack, main thread first, capped by thread count and by characters."""
    frames = sys._current_frames()
    names = {t.ident: t.name for t in threading.enumerate()}

    def order(item: tuple[Optional[int], object]) -> tuple[int, str, int]:
        ident, _frame = item
        name = names.get(ident, "?")
        # MainThread (the loop in the gateway) is the one a blocked-loop diagnostic is about.
        return (0 if name == "MainThread" else 1, name, ident or 0)

    ordered = sorted(frames.items(), key=order)
    out: list[str] = []
    used = 0
    dumped = 0
    for ident, frame in ordered:
        if dumped >= max_threads:
            break
        lines = [f"--- thread {ident} [{names.get(ident, '?')}]"]
        try:
            lines.extend(line.rstrip("\n") for line in traceback.format_stack(frame))
        except Exception:  # a frame whose state is being torn down
            lines.append("    (stack unreadable)")
        block = "\n".join(lines) + "\n"
        if used + len(block) > max_chars:
            out.append(f"... truncated at {max_chars} chars\n")
            break
        out.append(block)
        used += len(block)
        dumped += 1
    omitted = len(ordered) - dumped
    if omitted > 0:
        out.append(
            f"... {omitted} of {len(ordered)} threads omitted "
            f"(cap {max_threads} threads / {max_chars} chars)\n"
        )
    return "".join(out)


def dump_thread_stacks(
    reason: str,
    *,
    file: Optional[IO[str]] = None,
    max_threads: int = DEFAULT_MAX_THREADS,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
) -> str:
    """Write one bounded thread dump to ``file`` (stderr by default) and return it.

    Returns ``""`` when suppressed by the rate limit or when the frames could not be read —
    never raises, and never takes a process-wide fault path.
    """
    global _last_dump_at
    target = sys.stderr if file is None else file
    now = time.monotonic()
    with _lock:
        if min_interval_s > 0 and _last_dump_at and (now - _last_dump_at) < min_interval_s:
            logger.warning(
                "[thread-dump] %s: full dump suppressed — another ran %.0fs ago (floor %.0fs)",
                reason, now - _last_dump_at, min_interval_s,
            )
            return ""
        _last_dump_at = now
    try:
        body = render_thread_stacks(max_threads=max_threads, max_chars=max_chars)
    except Exception:
        logger.warning("[thread-dump] %s: thread frames unreadable, no dump produced", reason,
                       exc_info=True)
        return ""
    payload = f"--- thread dump: {reason} (pid={os.getpid()}) ---\n{body}--- end thread dump ---\n"
    try:
        target.write(payload)
        target.flush()
    except Exception:  # a wedged stderr/file must not raise into the diagnostic path
        logger.debug("[thread-dump] %s: dump could not be written", reason, exc_info=True)
    return payload
