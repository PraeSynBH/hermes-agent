"""Who asked for a managed gateway restart, and where that answer survives the restart.

A SIGUSR1 carries no payload: the kernel tells the gateway THAT a restart was requested and
nothing about who asked. The only trustworthy record is therefore the sender's own line, written
immediately BEFORE the signal (:func:`record_request`), and the only moment to read it is inside
the signal handler — this process exits a drain later (PRX-5396, PRX-5423).

Three holes this closes, all measured on the 2026-10-22T19:55:39Z restart (admission closed
31 min 37 s, 33 turns waited for, 1 amputated):

* The line was written into the CALLER's profile home, so the host gateway's own logs proved
  nothing even though ``profiles/cto/logs/gateway-restart-requests.log`` had it. A profile home
  is invisible to ``~/.hermes/logs`` — the record has to land where the gateway reads.
* It named a pid, cwd and argv — never the Paperclip run, agent or issue that asked. A 31-minute
  admission closure on a 49-deep queue must name its author, not its command line.
* The gateway logged the signal and read none of it, so the restart's own lifecycle file lost the
  requester the moment the requesting process exited.

The lifecycle half (`gateway.lifecycle_ledger.stamp_restart_request`) writes the same identity
into ``state/gateway.lifecycle.json``; :func:`gateway.lifecycle_ledger.record_startup` carries it
onto the next life so ``jq .prior_restart_request`` answers after the fact.
"""

from __future__ import annotations

import calendar
import logging
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

REQUEST_LOG_NAME = "gateway-restart-requests.log"

# How stale a line may be and still belong to the signal being handled. The sender writes
# immediately before ``os.kill``, so a fresh request is milliseconds old; the window only has to
# cover a signal queued behind a stalled event loop, while staying far below the interval between
# two distinct operator restarts (a restart outlives this by 30 minutes at the measured drain).
REQUEST_MATCH_WINDOW_S = 120.0

# Fields in the order the writer emits them. ``argv`` and ``detail`` absorb the rest of the line
# (argv is quoted shell words; detail is free text), so every other field is a single token.
_FIELDS = (
    "path", "target_pid", "caller_pid", "ppid",
    "requestedByRunId", "agentId", "issueId",
    "cwd", "argv", "detail",
)
_KEY_RE = re.compile(r"(?:^|\s)(" + "|".join(_FIELDS) + r")=")

#: Environment variable → record field, for the run that asked for the restart.
_IDENTITY_ENV = (
    ("PAPERCLIP_RUN_ID", "requestedByRunId"),
    ("PAPERCLIP_AGENT_ID", "agentId"),
    ("PAPERCLIP_TASK_ID", "issueId"),
)


def process_home() -> Path:
    """The launch home of THIS process (identity files, never the per-session override).

    ``get_hermes_home()`` honours the task-scoped override, which would misroute a request line
    written by a run inside another profile; ``gateway.lifecycle_ledger`` resolves its sentinel
    with the same rule and both must agree on where ownership lives.
    """
    from hermes_constants import get_hermes_home, get_process_hermes_home

    if os.environ.get("HERMES_HOME", "").strip():
        return get_process_hermes_home()
    return get_hermes_home()


def request_log_path(home: Optional[Path] = None) -> Path:
    """``<HERMES_HOME>/logs/gateway-restart-requests.log`` of the gateway that will read it."""
    return (process_home() if home is None else Path(home)) / "logs" / REQUEST_LOG_NAME


def target_gateway_home() -> Path:
    """The home of the gateway that owns this host, so the line lands where it will be READ.

    A CLI started inside a profile writes from that profile's home; the host gateway reads its
    own. The per-user rendezvous record publishes the owner's launch home, so a request made from
    any profile is still found after the restart. Falls back to this process's home when the
    record is unreadable (single-profile or unsupervised setups, where the two are equal).
    """
    try:
        from gateway import host_rendezvous as hr

        record = hr.read_record(hr.ROLE_GATEWAY, include_stale=True)
        home = getattr(record, "home", "")
        if home:
            return Path(home).expanduser()
    except Exception:
        logger.debug("Host gateway home unavailable; using this process's home", exc_info=True)
    return process_home()


def identity_from_env(environ: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """The requesting run's identity, from the environment the request was made in.

    Empty when the requester is not a Paperclip run (a human shell, systemd's ``reload``): the
    caller pid and argv in the same line are then the attribution.
    """
    env = os.environ if environ is None else environ
    identity: dict[str, str] = {}
    for variable, field in _IDENTITY_ENV:
        value = (env.get(variable) or "").strip()
        if value:
            identity[field] = value
    return identity


def record_request(path: str, target_pid: int, detail: str = "", *, home: Optional[Path] = None,
                   identity: Optional[Mapping[str, str]] = None) -> None:
    """Append one line for a restart request this process is ABOUT to signal.

    Ordering is the contract, not a preference: the gateway reads this file the instant the signal
    lands, and a line written after ``os.kill`` races the restart it describes — the measured
    2026-10-22 request was recorded 0.4 s behind the signal it caused. A line therefore means "an
    attempt was made"; the gateway only attributes signals that arrive while it is fresh.

    Never raises: an audit aid must not be able to abort a restart.
    """
    fields = dict(identity_from_env() if identity is None else identity)
    parts = [
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        f"path={path}", f"target_pid={int(target_pid)}",
        f"caller_pid={os.getpid()}", f"ppid={os.getppid()}",
    ]
    parts += [f"{field}={value}" for field, value in fields.items()]
    parts += [f"cwd={os.getcwd()}", "argv=" + " ".join(shlex.quote(arg) for arg in sys.argv)]
    if detail:
        parts.append(f"detail={detail}")
    try:
        log_path = request_log_path(target_gateway_home() if home is None else home)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(" ".join(parts) + "\n")
    except OSError:
        return


def parse_request_line(line: str) -> dict[str, str]:
    """``{field: value}`` for a written line; ``{}`` when it carries none of our fields."""
    matches = list(_KEY_RE.finditer(line))
    fields: dict[str, str] = {}
    if not matches:
        return fields
    fields["ts"] = line.split(" ", 1)[0]
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        fields[match.group(1)] = line[match.end():end].strip()
    return fields


def _age_seconds(stamp: Any) -> Optional[float]:
    """Age of a ``YYYY-MM-DDTHH:MM:SSZ`` record stamp, or ``None`` when it cannot be dated."""
    try:
        return max(0.0, time.time() - calendar.timegm(time.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError):
        return None


def latest_request_for(target_pid: int, *, home: Optional[Path] = None,
                       within_s: float = REQUEST_MATCH_WINDOW_S) -> Optional[dict[str, Any]]:
    """The newest request line addressed to ``target_pid``, or ``None`` when nothing matches.

    ``None`` is a finding, not a miss: it means the signal did not come from an instrumented
    sender (a raw ``kill``, systemd's ``reload``, a supervisor), which is what makes an
    unattributed restart provable instead of merely unexplained.
    """
    try:
        lines = request_log_path(home).read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None
    # Newest first over a bounded tail: the log is append-only and one line per restart.
    for line in reversed(lines[-50:]):
        fields: dict[str, Any] = parse_request_line(line)
        if fields.get("target_pid") != str(int(target_pid)):
            continue
        age = _age_seconds(fields.get("ts"))
        if age is not None and age > within_s:
            return None
        if age is not None:
            fields["age_s"] = round(age, 1)
        return fields
    return None


def describe_requester(requester: Optional[Mapping[str, Any]]) -> str:
    """The one GREPPABLE line the card asks for, with ``unrecorded`` where identity is unknown."""
    if not requester:
        return ("managed restart requested requestedByRunId=unrecorded agentId=unrecorded "
                "issueId=unrecorded drainRequired=true source=sigusr1 — no request line for this "
                f"pid within {REQUEST_MATCH_WINDOW_S:.0f}s: the signal came from outside the "
                "instrumented CLI paths (operator kill, systemd reload, supervisor).")
    parts = [
        "managed restart requested",
        f"requestedByRunId={requester.get('requestedByRunId') or 'unrecorded'}",
        f"agentId={requester.get('agentId') or 'unrecorded'}",
        f"issueId={requester.get('issueId') or 'unrecorded'}",
        "drainRequired=true",
        "source=sigusr1",
    ]
    for field in ("path", "caller_pid", "age_s", "detail"):
        value = requester.get(field)
        if value not in (None, ""):
            parts.append(f"{field}={value}")
    return " ".join(parts)


def sigusr1_restart_handler(runner, *, logger_: Optional[logging.Logger] = None):
    """Build the SIGUSR1 handler: name the requester, then the drain-aware service restart.

    The requester is readable ONLY at this instant — a signal carries no payload and this process
    exits a drain later — so the name is logged and stamped into the lifecycle file before the
    restart is requested. The signal's own meaning is logged too: systemd's ``reload`` verb
    (``ExecReload=kill -USR1``) lands here, and operators expect ``reload`` to mean an in-process
    config reload, not a drain-and-relaunch (#117267).
    """
    log = logger_ or logging.getLogger("gateway.run")

    def restart_signal_handler():
        requester = latest_request_for(os.getpid())
        log.info(
            "SIGUSR1 received (systemctl reload / hermes gateway restart): performing a graceful "
            "gateway restart — drain active turns, exit, supervisor relaunches. Not an in-process "
            "config reload.")
        log.info("%s", describe_requester(requester))
        from gateway.lifecycle_ledger import stamp_restart_request

        stamp_restart_request(requester, source="sigusr1")
        runner.request_restart(detached=False, via_service=True)

    return restart_signal_handler
