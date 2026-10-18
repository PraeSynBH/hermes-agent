"""Stopping the gateway on purpose: wait out its teardown, then close the lifecycle ledger.

Two jobs, one owner, because they are two halves of the same deliberate stop (PRA-6427):

* **Wait honestly.** ``hermes gateway stop`` waits for the gateway's own teardown and only
  force-kills as a backstop.  Under launchd the gateway sizes its SIGTERM drain to the live
  ``ExitTimeOut`` (``gateway.restart.resolve_launchd_capped_drain``), so a stopper that
  SIGKILLs earlier kills in-flight turns mid-teardown — the state.db-write hazard the bounded
  force is meant to avoid.
* **Leave a truthful ledger.** ``gateway.lifecycle_ledger.mark_exited`` lives *inside* the
  gateway process, so a stop that ends in SIGKILL never reaches it: the sentinel stays
  ``phase=running`` and the next boot reports ``gateway.previous_unclean_exit`` for a stop
  someone asked for — a phantom death that buries the real signal (PRA-6426's forensics).
  The stopper therefore records the planned exit itself, with an ownership guard, and appends
  a ``gateway.planned_stop`` record so no planned stop is ever silent.

Kept out of the ``hermes_cli.gateway`` facade (at its line cap): the facade re-exports
``_wait_for_gateway_exit``, and this module reads the facade's ``terminate_pid`` through
``_gw()`` so tests that patch it on ``hermes_cli.gateway`` still intercept the kill.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# A stopper that knows WHY it is stopping the gateway names the stop here, so the ledger reads
# `exit_reason: storage_window_stop` instead of a bare `gateway_stop`. One-shot IPC for the stop
# in flight, not configuration: it never reaches config.yaml/.env.
PLANNED_STOP_REASON_ENV = "HERMES_GATEWAY_STOP_REASON"


def _gw():
    """The facade, late-bound: ``terminate_pid`` is patched there by callers and tests."""
    from hermes_cli import gateway

    return gateway


def stop_reason(default: str) -> str:
    """``$HERMES_GATEWAY_STOP_REASON`` when a caller set one, else ``default``."""
    raw = (os.environ.get(PLANNED_STOP_REASON_ENV) or "").strip()
    return raw[:64] or default


def process_create_time(pid: Optional[int]) -> Optional[float]:
    """Process birth (epoch seconds) for the ledger's incarnation guard, or None."""
    if pid is None:
        return None
    try:
        from hermes_cli.process_identity import _process_create_time
    except ImportError:
        return None
    return _process_create_time(pid)


def record_planned_exit(pid: Optional[int], reason: str, create_time: Optional[float] = None) -> bool:
    """Close the lifecycle sentinel for a gateway this command stopped on purpose.

    No-op when the gateway recorded its own exit, when the sentinel names another life (a
    replacement already claimed it), or when the PID was recycled — the guards live in
    :func:`gateway.lifecycle_ledger.mark_planned_stop`.
    """
    if pid is None:
        return False
    try:
        from gateway.lifecycle_ledger import mark_planned_stop
    except ImportError:
        logger.debug("lifecycle ledger unavailable; planned stop not recorded", exc_info=True)
        return False
    return mark_planned_stop(pid, stop_reason(reason), expected_create_time=create_time)


def wait_for_gateway_exit(timeout: float = 10.0, force_after: float | None = 5.0) -> bool:
    """Wait up to ``timeout`` s for the gateway (by gateway.pid, not launchd labels, so multiple
    HERMES_HOMEs work) to exit; SIGKILL it after ``force_after`` s of graceful waiting."""
    from gateway.status import get_process_start_time, get_running_pid

    terminate_pid = _gw().terminate_pid
    deadline = time.monotonic() + timeout
    force_deadline = (time.monotonic() + force_after) if force_after is not None else None
    force_sent = False
    # Whose stop is this: pinned before the wait, so the planned-exit record below names the
    # life this command stopped even after its PID file is gone.
    target_pid = get_running_pid()
    target_create_time = process_create_time(target_pid)

    try:
        while time.monotonic() < deadline:
            pid = get_running_pid()
            if pid is None:
                return True  # Process exited cleanly.

            if force_deadline is not None and not force_sent and time.monotonic() >= force_deadline:
                # Grace period expired — force-kill the specific PID.
                try:
                    terminate_pid(pid, force=True, expected_start_time=get_process_start_time(pid))
                    print(f"⚠ Gateway PID {pid} did not exit gracefully; sent SIGKILL")
                except (ProcessLookupError, PermissionError, OSError):
                    return True  # Already gone or we can't touch it.
                force_sent = True

            time.sleep(0.3)

        # Timed out even after force-kill.
        remaining_pid = get_running_pid()
        if remaining_pid is not None:
            print(f"⚠ Gateway PID {remaining_pid} still running after {timeout}s — restart may fail")
            return False
        return True
    finally:
        # Gone, and a stop that had to be forced (or was SIGKILLed by the supervisor) never
        # reached the gateway's own mark_exited: close the sentinel here so the next boot does
        # not report this stop as an unclean death.
        if target_pid is not None and get_running_pid() is None:
            record_planned_exit(target_pid, "gateway_stop_forced" if force_sent else "gateway_stop",
                                target_create_time)
