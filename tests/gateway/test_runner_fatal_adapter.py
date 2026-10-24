import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class _FatalAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="token"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._set_fatal_error(
            "telegram_token_lock",
            "Another local Hermes gateway is already using this Telegram bot token.",
            retryable=False,
        )
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _RuntimeRetryableAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="token"), Platform.WHATSAPP)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _ReplacementDeliveryAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="token", typing_indicator=False),
            Platform.DISCORD,
        )
        self.sent: list[str] = []
        self.connected = True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self.connected = False

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if not self.connected:
            return SendResult(success=False, error="Not connected")
        self.sent.append(content)
        return SendResult(success=True, message_id=f"m-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_runner_pages_the_board_when_a_transport_goes_fatal(monkeypatch, tmp_path):
    """Regression for PRA-6623: a runtime fatal must reach the board, not just the log.

    The 2026-10-21 outage logged 2,571 poll errors and opened no card, because nothing on the fatal
    path escalated. The runner's fatal handler is that path for every adapter.
    """
    paged = []
    monkeypatch.setattr("gateway.run_adapters.page_platform_fatal", lambda adapter: paged.append(adapter) or True)

    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error(
        "whatsapp_bridge_exited", "WhatsApp bridge process exited unexpectedly (code 1).",
        retryable=True,
    )
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    await runner._handle_adapter_fatal_error(adapter)

    assert paged == [adapter], "the fatal handler is where a dead transport becomes a board card"


@pytest.mark.asyncio
async def test_runner_queues_retryable_runtime_fatal_for_reconnection(monkeypatch, tmp_path):
    """Retryable runtime fatal errors queue the platform for reconnection
    AND keep the gateway alive — the background reconnect watcher recovers
    the platform when the underlying issue clears.  (Previously this
    exited-with-failure to trigger a systemd restart; that converted
    transient failures into infinite restart loops.)
    """
    config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error(
        "whatsapp_bridge_exited",
        "WhatsApp bridge process exited unexpectedly (code 1).",
        retryable=True,
    )

    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    await runner._handle_adapter_fatal_error(adapter)

    # Gateway stays alive — watcher will retry in background
    runner.stop.assert_not_awaited()
    assert runner._exit_with_failure is False
    assert Platform.WHATSAPP in runner._failed_platforms
    assert runner._failed_platforms[Platform.WHATSAPP]["attempts"] == 0


@pytest.mark.asyncio
async def test_retryable_fatal_queues_reconnect_after_cancellation_swallowing_disconnect(
    monkeypatch, tmp_path
):
    """A wedged old adapter cannot block runner-owned reconnect recovery."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error("transport_stale", "transport stale", retryable=True)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters

    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def swallow_cancellation():
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        finished.set()

    monkeypatch.setattr(adapter, "disconnect", swallow_cancellation)
    operation = asyncio.create_task(runner._handle_adapter_fatal_error(adapter))
    await started.wait()
    done, _pending = await asyncio.wait({operation}, timeout=0.2)
    try:
        assert operation in done
        assert runner.adapters == {}
        assert Platform.WHATSAPP in runner._failed_platforms
        assert runner._failed_platforms[Platform.WHATSAPP]["attempts"] == 0
    finally:
        release.set()
        await asyncio.wait({operation}, timeout=0.2)
        await asyncio.wait_for(finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_retryable_fatal_queues_before_disconnect_returns(monkeypatch, tmp_path):
    """#80598: reconnect queue must populate before disconnect finishes.

    After a long network outage, Telegram disconnect can wedge on a half-dead
    socket. If the fatal handler only queues after disconnect returns, the
    watcher never learns about the failure and the gateway stays permanently
    deaf. Queue first; teardown is best-effort after.
    """
    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error(
        "telegram_network_error",
        "Telegram polling could not reconnect after 10 network error retries.",
        retryable=True,
    )
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    disconnect_entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_disconnect():
        # Disconnect has started — the platform must already be queued.
        assert Platform.WHATSAPP in runner._failed_platforms
        disconnect_entered.set()
        await release.wait()

    monkeypatch.setattr(adapter, "disconnect", blocking_disconnect)
    operation = asyncio.create_task(runner._handle_adapter_fatal_error(adapter))
    await asyncio.wait_for(disconnect_entered.wait(), timeout=0.5)
    try:
        assert runner.adapters == {}
        assert Platform.WHATSAPP in runner._failed_platforms
        assert runner._failed_platforms[Platform.WHATSAPP]["attempts"] == 0
        runner.stop.assert_not_awaited()
    finally:
        release.set()
        await asyncio.wait_for(operation, timeout=0.5)


@pytest.mark.asyncio
async def test_fatal_handler_outer_timeout_still_queues_platform(monkeypatch, tmp_path):
    """#80598: outer deadline must queue even if the impl task never returns."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.05")
    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error("transport_stale", "transport stale", retryable=True)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    started = asyncio.Event()
    release = asyncio.Event()

    async def wedged_impl(_adapter):
        # Simulate a hang before/without reaching the reconnect queue.
        runner.adapters.pop(Platform.WHATSAPP, None)
        runner.delivery_router.adapters = runner.adapters
        started.set()
        await release.wait()

    monkeypatch.setattr(runner, "_handle_adapter_fatal_error_impl", wedged_impl)
    operation = asyncio.create_task(runner._handle_adapter_fatal_error(adapter))
    await started.wait()
    # Outer budget is disconnect_timeout + min(2, max(0.05, timeout)) ≈ 0.1s.
    done, _pending = await asyncio.wait({operation}, timeout=1.0)
    try:
        assert operation in done
        assert Platform.WHATSAPP in runner._failed_platforms
        runner.stop.assert_not_awaited()
    finally:
        release.set()
        await asyncio.wait({operation}, timeout=0.2)




@pytest.mark.asyncio
async def test_stay_alive_policy_keeps_gateway_up_on_stranded_retryable_fatal(tmp_path):
    """on_all_adapters_down='stay_alive' (the desktop launcher's mode): a retryable fatal
    on the LAST adapter must not produce the exit-with-failure verdict — no service manager
    will revive the process, so exiting only severs the UI's websockets and drops in-flight
    assistant messages (#118080). The platform has no config entry here, so the reconnection
    queue cannot take it and the stranded check is reached with empty _failed_platforms."""
    config = GatewayConfig(sessions_dir=tmp_path / "sessions", on_all_adapters_down="stay_alive")
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error("email_imap_fetch_failed", "IMAP read timeout", retryable=True)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    await runner._handle_adapter_fatal_error(adapter)

    runner.stop.assert_not_awaited()
    assert runner._exit_with_failure is False


@pytest.mark.asyncio
async def test_default_exit_policy_keeps_service_restart_verdict_on_stranded_retryable_fatal(tmp_path):
    """Without the stay_alive policy (service-managed deployments), the stranded retryable
    fatal still exits with the failure verdict so the service manager restarts the gateway
    — the pre-#118080 contract is unchanged."""
    config = GatewayConfig(sessions_dir=tmp_path / "sessions")
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error("email_imap_fetch_failed", "IMAP read timeout", retryable=True)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    await runner._handle_adapter_fatal_error(adapter)

    runner.stop.assert_awaited()
    assert runner._exit_with_failure is True


# ── SOU-4217: on_all_adapters_down=stay_alive must bind the NON-retryable path too ──────────────
# The policy is the operator's "a messaging transport never takes the process down". It was honoured
# only for retryable failures, so one logged-out optional platform (whatsapp as the sole enabled
# platform) exited 78 at startup and put the whole multiplexed gateway into a 361 s launchd restart
# loop for 3.96 h — every `hermes` run refused `adapter_failed / gateway draining: not_ready` while
# cron and local agent ingress needed no messaging platform at all.

_LOGGED_OUT_WHATSAPP = "WhatsApp session is logged out (scan the QR code again)"
_LOGGED_OUT_WHATSAPP_ERROR = f"whatsapp: {_LOGGED_OUT_WHATSAPP}"


def _sole_whatsapp_config(tmp_path, policy: str) -> GatewayConfig:
    return GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
        on_all_adapters_down=policy,
    )


@pytest.mark.asyncio
async def test_startup_stay_alive_parks_a_sole_non_retryable_platform_instead_of_exiting_78(
    tmp_path, monkeypatch,
):
    """SOU-4217: a sole non-retryable platform failure must not exit 78 under stay_alive."""
    writes: list[dict] = []
    monkeypatch.setattr("gateway.run._write_runtime_status_quiet", lambda **kw: writes.append(kw))
    config = _sole_whatsapp_config(tmp_path, "stay_alive")
    runner = GatewayRunner(config)
    runner._request_clean_exit = MagicMock()

    retryable: list[str] = []
    nonretryable: list[str] = []
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error("whatsapp_logged_out", _LOGGED_OUT_WHATSAPP, retryable=False)
    connected = await runner._start_aggregate_connect_results(
        [(Platform.WHATSAPP, adapter, config.platforms[Platform.WHATSAPP], "failed", None)],
        retryable, nonretryable,
    )
    assert connected == 0
    assert retryable == []
    assert nonretryable == [_LOGGED_OUT_WHATSAPP_ERROR]

    must_exit = runner._start_handle_no_connections(connected, 1, retryable, nonretryable)

    assert must_exit is False, "one optional transport must not take the whole gateway down"
    runner._request_clean_exit.assert_not_called()
    assert runner._exit_code is None
    # The platform itself is parked fatal and visible, not silently healthy.
    assert any(
        w.get("platform") == "whatsapp" and w.get("platform_state") == "fatal" for w in writes
    ), writes
    # The run reports degraded, so no dispatcher reads it as a green gateway.
    assert runner._startup_parked_platforms is True
    assert runner._serving_state() == "degraded"
    assert {"gateway_state": "degraded", "exit_reason": None} in writes


def test_startup_exit_policy_still_exits_78_on_a_sole_non_retryable_platform(tmp_path, monkeypatch):
    """The default policy keeps the pre-SOU-4217 contract: exit 78 so the supervisor restarts."""
    writes: list[dict] = []
    monkeypatch.setattr("gateway.run._write_runtime_status_quiet", lambda **kw: writes.append(kw))
    config = _sole_whatsapp_config(tmp_path, "exit")
    runner = GatewayRunner(config)

    must_exit = runner._start_handle_no_connections(0, 1, [], [_LOGGED_OUT_WHATSAPP_ERROR])

    assert must_exit is True
    assert runner._exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    assert writes == [{"gateway_state": "startup_failed", "exit_reason": _LOGGED_OUT_WHATSAPP_ERROR}]
    assert runner._startup_parked_platforms is False


@pytest.mark.asyncio
async def test_stay_alive_policy_parks_a_stranded_nonretryable_fatal_instead_of_stopping(
    tmp_path, monkeypatch,
):
    """SOU-4217 runtime half: a non-retryable fatal on the last adapter stays up when the policy
    says stay_alive. It has no reconnect path, so the run is stamped degraded instead."""
    writes: list[dict] = []
    monkeypatch.setattr("gateway.run._write_runtime_status_quiet", lambda **kw: writes.append(kw))
    config = GatewayConfig(sessions_dir=tmp_path / "sessions", on_all_adapters_down="stay_alive")
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error("whatsapp_logged_out", _LOGGED_OUT_WHATSAPP, retryable=False)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    await runner._handle_adapter_fatal_error(adapter)

    runner.stop.assert_not_awaited()
    assert runner._exit_with_failure is False
    assert runner._startup_parked_platforms is True
    assert {"gateway_state": "degraded", "exit_reason": None} in writes
