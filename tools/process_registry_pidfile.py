"""Whether this process owns the live gateway pid file, asked cheaply.

Split out of ``tools/process_registry.py`` (a capped facade) so the latch and its reasoning
live next to the code that needs them. ``process_registry._is_supervised_gateway_process``
combines this answer with the launch-marker gate that must stay live on every call.

Why a latch at all: the probe is syscall-bound — a lock read, a pidfile stat and a kernel
process query for the start-time fingerprint, ~10-100 ms a call on an idle host — and it is
asked on every terminal call, every ``execute_code`` call and every child dispatch. In a
43-profile multiplex gateway (~300 threads, GIL convoy) that stretched past the caller's whole
deadline: 1169 "process-identity probe wedged" refusals, each burning up to 180 s of a turn and
returning nothing (PRA-6500). Once this process has PROVEN it owns the live pid file there is
nothing left to learn, so stop asking.

Only a positive is latched. Caching a miss would permanently un-guard a gateway that asked
before its pid file existed — which would also keep a macOS gateway from ever getting its
launchd shim (``tools/environments/local.py`` relies on the next call re-probing) — so a miss
always re-probes.
"""

from __future__ import annotations

import os

# One-way, per-process: set only after a definite "yes, this pid owns the live pid file".
_owns_pidfile = False


def owns_live_gateway_pidfile() -> bool:
    """True when this process owns the live gateway pid file in this home.

    Raises whatever ``get_running_pid`` raises; callers decide what an unanswerable probe means.
    """
    global _owns_pidfile
    if _owns_pidfile:
        return True
    from gateway.status import get_running_pid

    if get_running_pid(cleanup_stale=False) != os.getpid():
        return False
    _owns_pidfile = True
    return True
