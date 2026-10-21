"""Page the on-call board when a transport dies, and again when it recovers.

A dead chat transport cannot report itself over the chat. On 2026-10-21 the adopted WhatsApp
bridge stopped answering at 19:07Z; the poll loop logged 2,571 ``Poll error`` lines over 6.5 h,
every status surface still read "connected", and eight cron deliveries failed with nothing but a
per-job ``delivery error`` line (PRA-6623). The one channel that survives the chat transport is
the board, so transport death and recovery are posted to an on-call webhook in Alertmanager's
format. The receiver owns the card: one issue per (alertname, client) fingerprint while the alert
is firing, and the ``resolved`` notice is what tells the assignee the outage is over (Paperclip's
on-call receiver records it and the issue owner closes the card on its next heartbeat). One
fingerprint for the whole outage, however many failures re-declare it, and a notice that outlives
this process.

Callers run on the shared gateway loop (a fatal error, a cron delivery), so this module never
blocks and never raises: the POST runs on a short-lived daemon thread with a hard timeout, and a
webhook that is down costs a log line, not a stalled loop (PRA-6517 / PRA-6380).

Off unless ``gateway.alert_webhook_url`` is configured — an unconfigured install pages nothing.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Hard ceiling for one page. A pager is a side channel; it must never hold a caller longer than this.
_POST_TIMEOUT_S = 5.0
# One firing page per fingerprint per interval. The receiver already holds one card per fingerprint
# open, so this only bounds how often a long outage re-declares itself.
_RESEND_INTERVAL_S = 900.0

_lock = threading.Lock()
_configured_url: Optional[str] = None
_config_read = False
# fingerprint -> ("firing" | "resolved", monotonic time of the last POST that left this process).
_last_status: dict[str, tuple[str, float]] = {}
# One lock per fingerprint, held across that fingerprint's POST: a recovery can never overtake the
# outage page it closes (a resolved that arrives first would close nothing and strand an open card),
# while different fingerprints still page in parallel.
_fingerprint_locks: dict[str, threading.Lock] = {}
# Fingerprints already reported as unpageable, so a broken webhook is one warning, not one per fatal.
_logged_unreachable: set[str] = set()


def configure(url: Optional[str]) -> None:
    """Pin the webhook URL (gateway startup, tests). ``None`` turns paging off for this process."""
    global _configured_url, _config_read
    with _lock:
        _configured_url = (url or "").strip() or None
        _config_read = True


def webhook_url() -> Optional[str]:
    """The configured on-call webhook, or None when paging is off (the shipped default)."""
    global _configured_url, _config_read
    with _lock:
        if _config_read:
            return _configured_url
        _config_read = True
        # Read the mapping directly: ``hermes_cli.config.get_config_value`` calls sys.exit(1) for a
        # key that is not set, which would kill the gateway on its first unpageable fatal.
        try:
            from hermes_constants import get_routing_process_hermes_home
            import hermes_yaml as _yaml
            config = _yaml.safe_load(
                (get_routing_process_hermes_home() / "config.yaml").read_text(encoding="utf-8-sig")) or {}
            value = (config.get("gateway") or {}).get("alert_webhook_url", config.get("alert_webhook_url"))
        except Exception:
            # The destination is a host-level fact, read from the LAUNCH home: an unreadable or
            # profile-scoped read must disable paging, never raise into a fatal-error path.
            logger.debug("on-call webhook URL could not be read", exc_info=True)
            value = None
        _configured_url = value.strip() if isinstance(value, str) and value.strip() else None
        return _configured_url


def reset_for_tests() -> None:
    """Forget the URL, the throttle state and the log-once set (tests only)."""
    global _configured_url, _config_read
    with _lock:
        _configured_url, _config_read = None, False
        _last_status.clear()
        _fingerprint_locks.clear()
        _logged_unreachable.clear()


def page_firing(
    alertname: str, client: str, *, summary: str, description: str = "",
    severity: str = "high", tier: str = "1",
) -> bool:
    """Page that ``client`` is down. True when a POST was dispatched.

    Re-paging the same fingerprint is throttled to one per ``_RESEND_INTERVAL_S``: the receiver
    keeps the single card open, so a caller may re-declare the same outage freely.
    """
    return _page(alertname, client, "firing", summary=summary, description=description,
                 severity=severity, tier=tier)


def page_resolved(
    alertname: str, client: str, *, summary: str, description: str = "",
    severity: str = "high", tier: str = "1",
) -> bool:
    """Page that ``client`` is back, so the receiver can close the card.

    Sent even when THIS process never saw the outage: a gateway restarted mid-outage has no firing
    state, and only the receiver knows a card is still open. It ignores a resolved notice for a
    fingerprint it holds no card for, so this is idempotent (PRA-6623).
    """
    return _page(alertname, client, "resolved", summary=summary, description=description,
                 severity=severity, tier=tier)


def transport_down(platform: str, *, code: str, message: str, retryable: bool) -> bool:
    """Page a transport (a messaging platform adapter) that went fatal.

    ``retryable`` decides the wording, not whether a human hears about it: a retryable failure is
    the gateway's to fix, and a card that closes itself on recovery is still the only way the
    outage is visible while the chat is down.
    """
    recovery = ("The gateway is retrying in the background and will close this card when the "
                "transport reconnects." if retryable else
                "Nothing will retry this automatically — it needs a human action.")
    return page_firing(
        "TransportDown", platform,
        summary=f"{platform} transport down: {code}",
        description=f"{message}\n\n{recovery}",
        severity="high" if retryable else "critical",
    )


def transport_up(platform: str) -> bool:
    """Close the transport card opened by ``transport_down``."""
    return page_resolved(
        "TransportDown", platform,
        summary=f"{platform} transport recovered",
        description="The transport is connected again; deliveries can reach it.",
    )


def page_platform_fatal(adapter) -> bool:
    """Page for an adapter the runner just declared fatal. Never raises.

    ``relay_disabled`` is a configuration verdict, not an outage: paging it would open a card for a
    platform the operator turned off on purpose.
    """
    try:
        if adapter.fatal_error_code == "relay_disabled":
            return False
        platform = getattr(adapter, "_runtime_status_platform_key", None) or adapter.platform.value
        return transport_down(
            platform, code=adapter.fatal_error_code or "unknown",
            message=adapter.fatal_error_message or "the transport failed", retryable=adapter.fatal_error_retryable,
        )
    except Exception:
        logger.debug("on-call page for a fatal adapter failed", exc_info=True)
        return False


def page_platform_recovered(adapter) -> bool:
    """Close the card ``page_platform_fatal`` opened for this adapter. Never raises."""
    try:
        platform = getattr(adapter, "_runtime_status_platform_key", None) or adapter.platform.value
        return transport_up(platform)
    except Exception:
        logger.debug("on-call recovery page for an adapter failed", exc_info=True)
        return False


def cron_target_failing(platform: str, chat_id: str, *, job_id: str, error: str) -> bool:
    """Page that cron output cannot be delivered to one target.

    The fingerprint is the TARGET, not the job: eight jobs failing against a dead WhatsApp bridge
    are one outage, and the board should hold one card for it (PRA-6623).
    """
    return page_firing(
        "CronDeliveryFailing", f"{platform}:{chat_id}",
        summary=f"cron delivery to {platform}:{chat_id} is failing",
        description=f"Job `{job_id}` could not deliver: {error}\n\n"
                    "Every job targeting this destination fails the same way while the transport is "
                    "down; this card covers the target and closes on the first successful delivery.",
    )


def cron_target_ok(platform: str, chat_id: str) -> bool:
    """Close the cron delivery card for a target that just delivered."""
    return page_resolved(
        "CronDeliveryFailing", f"{platform}:{chat_id}",
        summary=f"cron delivery to {platform}:{chat_id} recovered",
        description="A delivery to this target succeeded again.",
    )


def _page(
    alertname: str, client: str, status: str, *, summary: str, description: str,
    severity: str, tier: str,
) -> bool:
    """Throttle, record and dispatch one page. Never raises."""
    try:
        url = webhook_url()
        if not url:
            return False
        fingerprint = f"{alertname}:{client}"
        with _lock:
            previous, sent_at = _last_status.get(fingerprint, ("", 0.0))
            if status == "resolved":
                # The receiver ignores a resolved notice for a fingerprint it holds no card for, so a
                # process that never saw the outage (a restart mid-outage) can still close the card.
                # Once this process has said "resolved", it stays quiet until something fires again.
                if previous == "resolved":
                    return False
            elif previous == "firing" and (time.monotonic() - sent_at) < _RESEND_INTERVAL_S:
                return False
            _last_status[fingerprint] = (status, time.monotonic())
            fp_lock = _fingerprint_locks.setdefault(fingerprint, threading.Lock())
        payload = {
            "version": "4",
            "status": status,
            "alerts": [{
                "status": status,
                "labels": {"alertname": alertname, "client": client, "severity": severity, "tier": tier},
                "annotations": {"summary": summary, "description": description},
            }],
        }
        threading.Thread(  # health: allow HX012 -- no profile-scoped state is read on the worker; the URL was resolved on the caller's thread
            target=_post, args=(url, fingerprint, payload, fp_lock), name="oncall-page", daemon=True,
        ).start()
        return True
    except Exception:
        logger.debug("on-call page for %s/%s could not be dispatched", alertname, client, exc_info=True)
        return False


def _post(url: str, fingerprint: str, payload: dict, fp_lock: threading.Lock) -> None:
    """Deliver one page on a worker thread; log the first failure per fingerprint, then stay quiet."""
    with fp_lock:
        try:
            import urllib.request
            request = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"), method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=_POST_TIMEOUT_S) as response:
                response.read(64)
            with _lock:
                _logged_unreachable.discard(fingerprint)
        except Exception as exc:
            with _lock:
                first = fingerprint not in _logged_unreachable
                _logged_unreachable.add(fingerprint)
            if first:
                logger.warning(
                    "on-call page for %s failed (%s: %s)", fingerprint, type(exc).__name__, exc,
                    exc_info=True,
                )
            else:
                logger.debug(
                    "on-call page for %s still failing (%s)", fingerprint, type(exc).__name__,
                    exc_info=True,
                )
