"""Regression for PRA-6622: an adopted WhatsApp bridge could die without the adapter noticing.

At boot the gateway adopted an already-running bridge (``_attach_to_bridge(None)``), so there was no
Popen to poll: ``_check_managed_bridge_exit`` returned ``None`` forever while the poll loop spun on the
dead port (2,571 "Poll error" lines over 6.5 h) with no fatal error, no reconnect and no board signal.
A paging channel was fully down while every status surface said "connected".
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform


def _make_adapter(session_path: Path):
    """An adapter for an ADOPTED bridge: ``_bridge_process`` is ``None``, nothing polls a process."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._bridge_port = 19876
    adapter._bridge_script = "/tmp/test-bridge.js"
    adapter._session_path = session_path
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._bridge_process = None
    adapter._bridge_unreachable_streak = 0
    adapter._running = True
    adapter._shutting_down = False
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._http_session = MagicMock()
    return adapter


def _dead(adapter, streak: int = 6) -> None:
    adapter._bridge_unreachable_streak = streak


@pytest.mark.asyncio
async def test_adopted_bridge_that_stops_answering_is_a_fatal_exit(tmp_path):
    """No pidfile (the out-of-process `hermes whatsapp` case) — the failure streak is the proof."""
    from plugins.platforms.whatsapp.adapter import _ADOPTED_BRIDGE_DEAD_STREAK

    adapter = _make_adapter(tmp_path)
    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)
    _dead(adapter, _ADOPTED_BRIDGE_DEAD_STREAK)

    assert await adapter._check_managed_bridge_exit() is not None
    assert adapter.fatal_error_code == "whatsapp_bridge_exited"
    assert adapter.fatal_error_retryable is True, "a dead bridge must reach the reconnect watcher"
    fatal_handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_adopted_bridge_streak_below_threshold_is_not_an_exit(tmp_path):
    """A busy bridge that misses a request or two must not be declared dead."""
    from plugins.platforms.whatsapp.adapter import _ADOPTED_BRIDGE_DEAD_STREAK

    adapter = _make_adapter(tmp_path)
    _dead(adapter, _ADOPTED_BRIDGE_DEAD_STREAK - 1)

    assert await adapter._check_managed_bridge_exit() is None
    assert adapter.fatal_error_code is None


@pytest.mark.asyncio
async def test_adopted_bridge_with_a_live_pidfile_process_is_unreachable_not_exited(tmp_path):
    """A wedged bridge whose process is still alive is not an exit: killing it would be the worse bug."""
    adapter = _make_adapter(tmp_path)
    _dead(adapter)

    from plugins.platforms.whatsapp import adapter as adapter_module
    with patch.object(adapter_module, "_read_bridge_pidfile", return_value=(4321, 99, 19876)), \
         patch.object(adapter_module, "_bridge_pid_is_ours", return_value=True):
        assert await adapter._check_managed_bridge_exit() is None
    assert adapter.fatal_error_code is None


@pytest.mark.asyncio
async def test_logged_out_session_is_named_instead_of_a_generic_exit(tmp_path):
    """bridge.js exits(1) on DisconnectReason.loggedOut; the exit code alone cannot say so."""
    log = tmp_path / "bridge.log"
    log.write_text("2026-10-20 12:08:01,610 ✅ WhatsApp connected!\n", encoding="utf-8")
    adapter = _make_adapter(tmp_path / "session")
    adapter._bridge_log = log
    adapter._bridge_log_offset = log.stat().st_size  # attached at boot, before the logout
    with log.open("a", encoding="utf-8") as fh:
        fh.write("❌ Logged out. Delete session and restart to re-authenticate.\n")
    _dead(adapter)

    assert await adapter._check_managed_bridge_exit() is not None
    assert adapter.fatal_error_code == "whatsapp_logged_out"
    assert "hermes whatsapp" in adapter.fatal_error_message
    assert adapter.fatal_error_retryable is True, "re-pairing must recover without a gateway restart"


@pytest.mark.asyncio
async def test_old_logged_out_lines_from_before_attach_do_not_mislabel_a_crash(tmp_path):
    """bridge.log is appended across restarts: only bytes written after attach explain this exit."""
    log = tmp_path / "bridge.log"
    log.write_text("❌ Logged out. Delete session and restart to re-authenticate.\n", encoding="utf-8")
    adapter = _make_adapter(tmp_path / "session")
    adapter._bridge_log = log
    adapter._bridge_log_offset = log.stat().st_size
    with log.open("a", encoding="utf-8") as fh:
        fh.write("Error: Cannot find package '@whiskeysockets/baileys'\n")
    _dead(adapter)

    assert await adapter._check_managed_bridge_exit() is not None
    assert adapter.fatal_error_code == "whatsapp_bridge_exited"


@pytest.mark.asyncio
async def test_bridge_up_but_not_connected_is_reported_not_healthy(tmp_path):
    """The card's other half: /health answers 200 with status "disconnected" forever (logged out / QR)."""
    from plugins.platforms.whatsapp.adapter import _BRIDGE_HEALTH_PROBE_SECS, _BRIDGE_NOT_CONNECTED_SECS

    adapter = _make_adapter(tmp_path)
    adapter._probe_bridge_health = AsyncMock(return_value=(True, {"status": "disconnected"}))

    # First sample opens the window, a sample inside it is still consistent with a 428 reconnect, and only
    # a sample past the sustained threshold is an incident.
    adapter._last_bridge_health_probe = 0.0
    assert await adapter._check_bridge_channel_health() is None
    adapter._bridge_not_connected_since = (adapter._bridge_not_connected_since or 0.0) - (_BRIDGE_NOT_CONNECTED_SECS + 1)
    adapter._last_bridge_health_probe = 0.0
    assert await adapter._check_bridge_channel_health() is not None
    assert adapter.fatal_error_code == "whatsapp_bridge_exited"
    assert "disconnected" in adapter.fatal_error_message
    assert _BRIDGE_HEALTH_PROBE_SECS < _BRIDGE_NOT_CONNECTED_SECS


@pytest.mark.asyncio
async def test_a_connected_health_probe_clears_the_not_connected_window(tmp_path):
    """A 428 reconnect that recovers must not be escalated: the window resets on the next connected sample."""
    from plugins.platforms.whatsapp.adapter import _BRIDGE_NOT_CONNECTED_SECS

    adapter = _make_adapter(tmp_path)
    adapter._bridge_not_connected_since = 1.0  # an old, long window
    adapter._probe_bridge_health = AsyncMock(return_value=(True, {"status": "connected"}))
    adapter._last_bridge_health_probe = 0.0

    assert await adapter._check_bridge_channel_health() is None
    assert adapter._bridge_not_connected_since is None
    assert adapter.fatal_error_code is None
    assert _BRIDGE_NOT_CONNECTED_SECS > 5


@pytest.mark.asyncio
async def test_poll_loop_stops_polling_a_dead_adopted_bridge(tmp_path):
    """The leaked loop is the bug: it must end instead of logging "Poll error" forever."""
    from plugins.platforms.whatsapp.adapter import _ADOPTED_BRIDGE_DEAD_STREAK

    adapter = _make_adapter(tmp_path)
    _dead(adapter, _ADOPTED_BRIDGE_DEAD_STREAK - 1)  # one more failed request makes it dead

    class _DeadRequest:
        async def __aenter__(self):
            raise ConnectionError("Cannot connect to host 127.0.0.1:49641")

        async def __aexit__(self, *exc):
            return False

    with patch.object(adapter, "_bridge_req", lambda *a, **k: _DeadRequest()), \
         patch("builtins.print"):
        await asyncio.wait_for(adapter._poll_messages(), timeout=10)

    assert adapter.fatal_error_code == "whatsapp_bridge_exited"
