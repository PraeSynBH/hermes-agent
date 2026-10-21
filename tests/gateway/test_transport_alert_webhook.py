"""Regression for PRA-6623: a dead transport has to reach the board, not just the log.

An adopted WhatsApp bridge died at 19:07Z and nothing noticed for 6.5 h — no fatal error, no
reconnect, no board card, and eight cron deliveries failed with one ``delivery error`` line each.
These tests pin the escalation channel those failures were missing:

* a transport that goes fatal pages the on-call webhook ONCE and re-uses that card while the
  outage lasts, closing it on recovery;
* cron delivery failures page per TARGET, so eight jobs failing against one dead bridge are one
  card, not eight;
* paging is off unless ``gateway.alert_webhook_url`` is set, and a broken webhook never raises
  into the gateway loop.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.waiting import wait_until


class _Collector:
    """A real HTTP endpoint: the pager posts over real sockets, so a dead code path cannot pass."""

    def __init__(self):
        self.alerts = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                outer.alerts.append(json.loads(self.rfile.read(length) or b"{}"))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def wait_for(self, count: int) -> list:
        wait_until(lambda: len(self.alerts) >= count, timeout=5, what=f"{count} alert(s) posted")
        return self.alerts


@pytest.fixture
def collector():
    from gateway import alert_webhook

    alert_webhook.reset_for_tests()
    sink = _Collector()
    alert_webhook.configure(sink.url)
    try:
        yield sink
    finally:
        sink.close()
        alert_webhook.reset_for_tests()


def _labels(alert) -> dict:
    return alert["alerts"][0]["labels"]


def test_a_fatal_transport_pages_once_and_reuses_the_card(collector):
    """One card per outage: the second fatal of the same transport is a re-fire, not a new issue."""
    from gateway import alert_webhook

    assert alert_webhook.transport_down(
        "whatsapp", code="whatsapp_bridge_exited", message="bridge is gone", retryable=True) is True
    first = collector.wait_for(1)[0]
    assert first["status"] == "firing"
    assert _labels(first) == {
        "alertname": "TransportDown", "client": "whatsapp", "severity": "high", "tier": "1"}

    # Same transport again inside the resend interval: the receiver already holds the card open.
    assert alert_webhook.transport_down(
        "whatsapp", code="whatsapp_bridge_exited", message="still gone", retryable=True) is False
    assert len(collector.alerts) == 1


def test_a_non_retryable_fatal_pages_critical_and_says_a_human_must_act(collector):
    """A logged-out session is not self-healing, so the card must not read as 'recovering'."""
    from gateway import alert_webhook

    alert_webhook.transport_down(
        "whatsapp", code="whatsapp_logged_out", message="re-pair with hermes whatsapp",
        retryable=False)
    alert = collector.wait_for(1)[0]
    assert _labels(alert)["severity"] == "critical"
    assert "human action" in alert["alerts"][0]["annotations"]["description"]


def test_recovery_closes_the_card(collector):
    from gateway import alert_webhook

    alert_webhook.transport_down("whatsapp", code="x", message="down", retryable=True)
    assert alert_webhook.transport_up("whatsapp") is True
    fired, resolved = collector.wait_for(2)
    assert (fired["status"], resolved["status"]) == ("firing", "resolved")
    assert _labels(resolved)["alertname"] == "TransportDown"


def test_a_recovery_after_a_restart_still_closes_the_card(collector):
    """A gateway restarted mid-outage has no firing state; only the receiver knows the card is open."""
    from gateway import alert_webhook

    assert alert_webhook.transport_up("whatsapp") is True
    resolved = collector.wait_for(1)[0]
    assert resolved["status"] == "resolved"
    # A second recovery in the same process does not re-page.
    assert alert_webhook.transport_up("whatsapp") is False
    assert len(collector.alerts) == 1


def test_cron_failures_page_per_target_not_per_job(collector):
    """Eight jobs failing against one dead bridge are one outage and one card (PRA-6623)."""
    from gateway import alert_webhook

    for job_id in ("daily-brief", "disk-report", "inbox-triage"):
        alert_webhook.cron_target_failing(
            "whatsapp", "15551234567", job_id=job_id, error="Cannot connect to host 127.0.0.1:49641")
    alert = collector.wait_for(1)[0]
    assert _labels(alert) == {
        "alertname": "CronDeliveryFailing", "client": "whatsapp:15551234567",
        "severity": "high", "tier": "1"}
    assert "daily-brief" in alert["alerts"][0]["annotations"]["description"]
    assert len(collector.alerts) == 1, "a second job targeting the same destination must not re-page"


def test_a_different_target_is_a_different_card(collector):
    from gateway import alert_webhook

    alert_webhook.cron_target_failing("whatsapp", "111", job_id="a", error="down")
    alert_webhook.cron_target_failing("telegram", "222", job_id="b", error="down")
    clients = {_labels(a)["client"] for a in collector.wait_for(2)}
    assert clients == {"whatsapp:111", "telegram:222"}


def test_unconfigured_paging_is_a_no_op(collector):
    """The shipped default pages nothing: an install with no board must not spawn threads."""
    from gateway import alert_webhook

    alert_webhook.configure(None)
    assert alert_webhook.transport_down("whatsapp", code="x", message="down", retryable=True) is False
    assert collector.alerts == []


def test_a_dead_webhook_never_raises_into_the_caller():
    """The pager is a side channel: a refused connection costs a log line, not the gateway."""
    from gateway import alert_webhook

    alert_webhook.reset_for_tests()
    alert_webhook.configure("http://127.0.0.1:1/")  # nothing listens here
    try:
        assert alert_webhook.transport_down("whatsapp", code="x", message="down", retryable=True) is True
    finally:
        alert_webhook.reset_for_tests()


def _adopted_whatsapp_adapter():
    """The adapter shape the fatal path needs, with no bridge and no transport."""
    from gateway.config import Platform
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter._runtime_status_platform_key = None
    adapter._running = True
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    return adapter


def test_an_adapter_going_fatal_pages_and_its_recovery_closes(collector):
    """The adapter-facing helpers: a fatal adapter pages, a recovered one closes the card."""
    from gateway import alert_webhook

    adapter = _adopted_whatsapp_adapter()
    adapter._set_fatal_error(
        "whatsapp_bridge_exited", "WhatsApp bridge stopped answering", retryable=True)

    assert alert_webhook.page_platform_fatal(adapter) is True
    fired = collector.wait_for(1)[0]
    assert _labels(fired) == {
        "alertname": "TransportDown", "client": "whatsapp", "severity": "high", "tier": "1"}
    assert "stopped answering" in fired["alerts"][0]["annotations"]["description"]

    assert alert_webhook.page_platform_recovered(adapter) is True
    assert collector.wait_for(2)[1]["status"] == "resolved"


@pytest.mark.asyncio
async def test_a_relay_disabled_verdict_does_not_page(collector):
    """A platform the operator switched off is not an outage."""
    from gateway import alert_webhook

    adapter = _adopted_whatsapp_adapter()
    adapter._set_fatal_error("relay_disabled", "relay is disabled in config", retryable=False)
    assert alert_webhook.page_platform_fatal(adapter) is False
    assert collector.alerts == []


def test_a_failed_delivery_target_pages_and_a_delivered_one_closes(collector):
    """The cron seam: the per-target failure that only logged a line before (PRA-6623)."""
    from cron.scheduler_delivery import _TargetDelivery, _deliver_standalone

    target = _TargetDelivery(
        job={"id": "daily-brief"}, platform=None, platform_name="whatsapp", chat_id="15551234567",
        thread_id=None, transport=None, pconfig=None, runtime_adapter=None, target_adapters=None,
        config=None, loop=None, notify_delivery=False, origin={}, origin_target=False,
        origin_user_id=None, is_dm_target=True, mirror_text="", mirror_this_target=False,
        in_channel_surface=False, inchannel_continuable=False, opened_thread_id=None,
    )
    errors: list = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("cron.scheduler_delivery._standalone_send",
                   lambda *a, **k: ({"error": "Cannot connect to host 127.0.0.1:49641"}, None))
        _deliver_standalone(target, "output", [], [], errors)
    assert errors, "the failure still reaches the job's own delivery_errors"
    alert = collector.wait_for(1)[0]
    assert _labels(alert)["client"] == "whatsapp:15551234567"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("cron.scheduler_delivery._standalone_send", lambda *a, **k: ({"success": True}, None))
        mp.setattr("cron.scheduler_delivery._maybe_mirror_cron_delivery",
                   lambda *a, **k: None)
        _deliver_standalone(target, "output", [], [], [])
    assert collector.wait_for(2)[1]["status"] == "resolved"

