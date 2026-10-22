"""Which profiles does the LIVE default multiplexer serve? One answer for every CLI/dashboard surface.

``gateway/run_adapters.py::_record_served_profiles`` writes ``served_profiles`` into the default
home's ``gateway_state.json`` at startup. That record is the truth about the running process; the
default ``config.yaml`` plus ``GATEWAY_MULTIPLEX_PROFILES`` as seen by the *CLI* process is only a
guess (``hermes -p coder ...`` loads coder's ``.env``, so an env-only opt-in on the default profile
is invisible to it, and an allowlist edited after start flips the guess before the restart).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def live_default_gateway_pid() -> Optional[int]:
    """PID of the default profile's gateway when a VERIFIED live process owns it, else None.

    ``gateway.status.live_gateway_pid_for_home``: pid file + lock, then the runtime record the gateway
    itself writes, each proven against the live process (start time, gateway command line, home). A
    launch-service gateway can be live with no ``gateway.pid`` at all, and a stale record whose PID was
    recycled by an unrelated process must not make its ``served_profiles`` authoritative. Never key this
    off the record's ``updated_at``: an idle gateway never advances it.
    """
    from hermes_constants import get_default_hermes_root
    from gateway.status import live_gateway_pid_for_home
    return live_gateway_pid_for_home(get_default_hermes_root())


def recorded_served_profiles(default_root: Optional[Path] = None) -> Optional[list[str]]:
    """``served_profiles`` the live default gateway recorded, or None when the key is absent (a record
    from before the multiplexer recorded it, or a stopped/absent gateway). Callers fall back to config
    derivation only on None: an empty list is an authoritative "serves nobody else"."""
    from hermes_constants import get_default_hermes_root
    from gateway.status import read_runtime_status
    if live_default_gateway_pid() is None:
        return None
    runtime = read_runtime_status((default_root or get_default_hermes_root()) / "gateway_state.json")
    served = (runtime or {}).get("served_profiles")
    return [str(p) for p in served] if isinstance(served, list) else None


def recorded_unserved_profiles(default_root: Optional[Path] = None) -> dict[str, str]:
    """``{profile: reason}`` the LIVE default multiplexer does not serve — its own published verdict.

    A profile past ``gateway.multiplex_max_profiles`` gets no adapters and no inbound routing on
    that process, so every run dispatched to it fails admission (``profile_not_served``); a profile
    whose session store could not be opened is parked the same way. Empty when the gateway serves
    everything, and for a stopped/absent gateway: a dead record's verdict is history (PRX-5734).
    """
    from hermes_constants import get_default_hermes_root
    from gateway.status import read_runtime_status, runtime_status_pid_is_live
    if live_default_gateway_pid() is None:
        return {}
    runtime = read_runtime_status((default_root or get_default_hermes_root()) / "gateway_state.json")
    if not runtime_status_pid_is_live(runtime or {}):
        return {}
    parked = (runtime or {}).get("parked_profiles")
    return {str(k): str(v) for k, v in parked.items()} if isinstance(parked, dict) else {}


def unserved_profile_lines(parked: dict[str, str]) -> list[str]:
    """Status block for ``parked``: one line per reason, then the remedy for a ceiling deferral.
    Empty when nothing is unserved, so a healthy host prints nothing."""
    if not parked:
        return []
    from hermes_cli.gateway_multiplex_capacity import CAPACITY_REASON_PREFIX
    by_reason: dict[str, list[str]] = {}
    for name, reason in parked.items():
        by_reason.setdefault(str(reason or "not served"), []).append(str(name))
    lines = ["", "Profiles this gateway does NOT serve (runs dispatched to them cannot be admitted):"]
    for reason, names in sorted(by_reason.items()):
        lines.append(f"  ⚠ {len(names)} unserved — {reason}")
        lines.append(f"      {', '.join(sorted(names))}")
    if any(reason.startswith(CAPACITY_REASON_PREFIX) for reason in by_reason):
        lines.append("  Raise gateway.multiplex_max_profiles (each served profile costs ~7 threads)")
        lines.append("  or move the profile up gateway.multiplex_serve_priority to keep it served.")
    return lines


def print_served_set_notices() -> None:
    """``hermes gateway status``: the standalone boot-guard box, then every profile the live
    multiplexer does not serve. Both are read from the live record; neither prints when clean."""
    from hermes_cli.gateway_multiplex_mode import recorded_standalone_warning_lines
    for line in recorded_standalone_warning_lines():
        print(line)
    for line in unserved_profile_lines(recorded_unserved_profiles()):
        print(line)


def multiplexer_served_secondaries() -> list[str]:
    """Named profiles the live default multiplexer serves (excludes ``default``); empty when none."""
    return [p for p in (recorded_served_profiles() or []) if p and p != "default"]


def served_profile_unserved_platforms(profile: str) -> dict[str, str]:
    """``{platform: reason}`` for a served profile's platforms the multiplexer deliberately does not run
    (unpaired WhatsApp or Relay shared ingress; ``gateway.run_adapters`` stamps
    ``<profile>:<platform>`` as ``disabled`` with a reason and remedy)."""
    from hermes_constants import get_default_hermes_root
    from gateway.status import read_runtime_status
    if not profile or live_default_gateway_pid() is None:
        return {}
    runtime = read_runtime_status(get_default_hermes_root() / "gateway_state.json") or {}
    platforms = runtime.get("platforms")
    if not isinstance(platforms, dict):
        return {}
    prefix = f"{profile}:"
    return {
        key[len(prefix):]: str(entry.get("error_message") or "not served under multiplex")
        for key, entry in platforms.items()
        if isinstance(key, str) and key.startswith(prefix) and isinstance(entry, dict)
        and entry.get("error_code") in {"multiplex_shared_ingress", "whatsapp_unpaired"}
    }


def served_profile_ingress_urls(profile: Optional[str] = None) -> dict[str, dict[str, str]]:
    """``{profile: {platform: url}}`` for every secondary inbound-port platform the live multiplexer
    serves on its shared listener (``<profile>:<platform>`` entries carrying ``ingress_url``). This is
    what the user pastes into the vendor console (Twilio, LINE, Teams, ...). ``profile`` narrows the map."""
    from hermes_constants import get_default_hermes_root
    from gateway.status import read_runtime_status, shared_listener_mirror_platforms
    if live_default_gateway_pid() is None:
        return {}
    runtime = read_runtime_status(get_default_hermes_root() / "gateway_state.json") or {}
    platforms = runtime.get("platforms")
    if not isinstance(platforms, dict):
        return {}
    urls: dict[str, dict[str, str]] = {}
    for key, entry in platforms.items():
        if not (isinstance(key, str) and ":" in key and isinstance(entry, dict)):
            continue
        url = entry.get("ingress_url")
        if not url or entry.get("state") in ("fatal", "disconnected", "stopped"):
            continue
        name, platform = key.split(":", 1)
        if profile and name != profile:
            continue
        urls.setdefault(name, {})[platform] = str(url)
    # api_server/webhook are the default's adapters mirrored at /p/<profile>/ (no entry of their own).
    served = [str(p) for p in (runtime.get("served_profiles") or []) if p and p != "default"]
    for name in served if not profile else [p for p in served if p == profile]:
        for platform, entry in shared_listener_mirror_platforms(runtime, name).items():
            if entry.get("ingress_url"):
                urls.setdefault(name, {})[platform] = str(entry["ingress_url"])
    return urls


def format_ingress_url_lines(urls: dict[str, str], indent: str = "  ") -> list[str]:
    """One ``<indent><platform>: <url>`` line per platform, sorted."""
    return [f"{indent}{platform}: {url}" for platform, url in sorted(urls.items())]


def note_profile_wanted(profile_name: str, *, timeout: float = 2.0) -> bool:
    """Tell the live default multiplexer a client is waiting on ``profile_name``'s admission.

    The demand rank has exactly one LIVE reading — a client that asked and is waiting
    (``gateway_multiplex_demand.note_wanted``) — and it lives inside the gateway process, so the
    sighting has to be filed over the control socket. That is what lets a run dispatched to a
    ceiling-deferred profile be ranked IN on the next reconcile tick, evicting a profile that holds
    no work, instead of re-observing a slot nobody was ever going to free (PRX-5963). Returns False
    (never raises) when no multiplexer answers: a single-profile host, a gateway predating the verb,
    or an unreadable socket — the caller then just re-observes the served set.

    Blocking: the answer arrives on the multiplexer's own event loop, so a caller already on a loop
    must hand this to a worker (``asyncio.to_thread``) — called inline it waits out its own timeout
    and reports False.
    """
    try:
        from hermes_constants import get_default_hermes_root
        from gateway.control_socket import note_gateway_profile_wanted
        if live_default_gateway_pid() is None:
            return False
        answer = note_gateway_profile_wanted(get_default_hermes_root(), profile_name, timeout=timeout)
    except Exception:
        logger.debug("multiplexer wanted-sighting failed for %r", profile_name, exc_info=True)
        return False
    return isinstance(answer, dict) and bool(answer.get("wanted"))


def notify_multiplexer_profiles_changed(profile_name: str, *, timeout: float = 8.0) -> Optional[list[str]]:
    """Tell the live default multiplexer that ``profiles/`` changed (``profile_name`` was created or
    deleted) so it hot-serves / unroutes it now instead of at its next periodic rescan. Returns the
    served-profile list the gateway answered with, or None when no multiplexer answered (no live default
    gateway, single-profile gateway, or a gateway predating the verb). Never raises."""
    try:
        from hermes_constants import get_default_hermes_root
        from gateway.control_socket import rescan_gateway_profiles
        if live_default_gateway_pid() is None:
            return None
        answer = rescan_gateway_profiles(get_default_hermes_root(), timeout=timeout)
    except Exception:
        logger.debug("multiplexer rescan notification failed for %r", profile_name, exc_info=True)
        return None
    if not isinstance(answer, dict) or answer.get("multiplex") is False:
        return None
    served = answer.get("served_profiles")
    return [str(p) for p in served] if isinstance(served, list) else None
