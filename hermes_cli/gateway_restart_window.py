"""The declared restart window: a lease a client reads instead of buying a boot it cannot get.

``restart-gateway.sh`` on this host stops the gateway, rewrites every session store offline and
then starts it again. Neither half is instant — the 2026-10-21T14:06 run spent 59 s draining and
181 s in maintenance, 187 s of declared outage — and the owning job publishes the window while it
is open (``window.open`` in ``deploy-state.json``, and the lease file below for readers that must
not parse a log).

Six heartbeat runs were dispatched inside that window and each burned its whole 120 s admission
budget (four died ``phase=runtime-boot``, two ``phase=session``) because every one of them read the
outage as a slow boot: ``ensure_gateway_runtime`` answers ``starting: deadline`` for any wait that
ran out, whether a daemon was coming up or none would be started for another two minutes (PRA-6692).

So the window is a fence, read before any budget is spent:

* a client that sees an open window returns the bounded ``restart_window`` verdict at once — the
  caller retries it, it is not a boot and not a failure of this host;
* nothing starts an owner while the window is open: the owning job's start is the only one, which
  is why the fence is read before the spawn branch and not after it.

The lease carries its own deadline and its owner refreshes it while it lives, so a crashed or
SIGKILLed pass can never leave this host without gateway recovery: an unrefreshed lease ages out
and the next reader starts the gateway as usual.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

#: Lease file the owning job keeps ahead of the clock: one line, the epoch second it is good until.
WINDOW_MARKER_NAME = ".gateway-maintenance-window"
#: ``logs/`` under the install root (``restart-gateway.sh`` writes ``$HOME/.hermes/logs``).
WINDOW_MARKER_RELPATH = "logs/" + WINDOW_MARKER_NAME

#: Reason code: the bounded name for "a declared restart window owns the gateway right now".
RESTART_WINDOW_REASON = "restart_window"


@dataclass(frozen=True)
class DeclaredRestartWindow:
    """A live declared window: the marker it was read from and the seconds left on it."""

    marker: str
    until_epoch: float
    remaining_s: int
    #: The owning job's token (``token=<hex>`` in the lease). A start that presents it in
    #: ``HERMES_RESTART_WINDOW_TOKEN`` is the owner's own start and is allowed through the fence;
    #: every other start is refused. Empty when the lease carries no token (an older writer).
    owner_token: str = ""

    def describe(self) -> str:
        """One bounded line for a client's failure text: who owns the gateway, for how long."""
        return f"declared restart window open: marker={self.marker} retryAfterS={self.remaining_s}"


def _read_lease(path) -> tuple[float | None, str]:
    """The lease's deadline epoch and owner token, or (None, "") when it cannot be read."""
    try:
        # utf-8-sig: a lease written by Windows tooling may carry a BOM, and a BOM'd epoch would
        # read as unparseable and (correctly but uselessly) close the fence.
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None, ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None, ""
    try:
        until = float(lines[0])
    except ValueError:
        return None, ""
    token = ""
    for line in lines[1:]:
        if line.startswith("token="):
            token = line[len("token="):].strip()
    return until, token


def refuse_outside_declared_window(verb: str, homes) -> None:
    """Exit the CLI when ``verb`` would put a gateway on a host a declared window owns.

    The owning job starts the gateway itself and proves it with the lease's token; every other
    start — a launchd watchdog, a dashboard, an agent running ``hermes gateway start`` — is a second
    owner booted into the middle of an offline store rewrite (PRA-6692 defect 3: a gateway was
    created at 14:08:36 inside the pass, four stores were refused, and hermes-storage died on
    ``--no-gateway expects a stopped gateway ... it is running``).

    Outside a window this returns at once, and a lease that has aged out is not a window at all, so
    recovery is never blocked by a killed pass.

    ``HERMES_RESTART_WINDOW_TOKEN`` is not a setting: it is the per-run owner CAPABILITY the
    declaring job mints into its lease and hands to its own children. It must not be in config.yaml
    (anything could then claim the window) and must not outlive the run — which is what an env var
    is, not config.
    """
    import os
    import sys

    window = declared_restart_window(homes)
    if window is None:
        return
    # health: allow HX002 -- run-scoped owner capability, never configuration (see docstring)
    presented = os.environ.get("HERMES_RESTART_WINDOW_TOKEN", "").strip()
    if window.owner_token and presented == window.owner_token:
        print(f"↻ {window.describe()} — this process owns it, starting as its owner.")
        return
    print(f"⏳ Gateway {verb} refused: {window.describe()}")
    sys.exit(1)


def window_fence_homes() -> set:
    """The homes a window may be declared on: this profile, the launch profile, the install root.

    ``restart-gateway.sh`` writes the lease under the install root (``$HOME/.hermes/logs``), and a
    named profile's own home is where a per-profile reader looks; the process's home covers a shell
    whose ``HERMES_HOME`` is neither.
    """
    from pathlib import Path

    from hermes_constants import get_default_hermes_root, get_hermes_home, get_process_hermes_home

    return {Path(get_hermes_home()), Path(get_process_hermes_home()), Path(get_default_hermes_root())}


def refuse_outside_declared_window_for_host(verb: str) -> None:
    """``refuse_outside_declared_window`` for the homes of the process asking (CLI entrypoints)."""
    refuse_outside_declared_window(verb, window_fence_homes())


def declared_restart_window(homes) -> DeclaredRestartWindow | None:
    """The live declared restart window on any of ``homes``, or None.

    Read-only and cheap (one small file per home): it sits on the admission path of every client.
    A marker that is missing, unreadable, malformed or past its deadline is None — an unrefreshed
    lease is a dead owner, and a dead owner must never fence a launch. The first live one wins;
    the owning job writes exactly one.
    """
    now = time.time()
    seen: set[str] = set()
    for home in homes:
        try:
            path = home / WINDOW_MARKER_RELPATH
        except AttributeError:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        until, token = _read_lease(path)
        if until is None or until <= now:
            continue
        return DeclaredRestartWindow(marker=key, until_epoch=until,
                                     remaining_s=max(1, int(round(until - now))),
                                     owner_token=token)
    return None
