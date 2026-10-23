"""An oversized frame must not become loop occupancy on the control socket (PRA-6844).

``_safe_send_many`` bounds its wait with ``_WS_SEND_DEADLINE_S``, but websockets' frame writer does
``write_frame_sync`` (permessage-deflate zlib, then ``transport.write`` -> ``sock.send``) *inside*
``send_text`` before it ever awaits ``drain``. Nothing can preempt that synchronous half, so a large
frame written to a peer whose receive window is full costs the served event loop whatever its size
and compressibility make it — the deadline only governs the wait afterwards.

This drives the real writer (a real uvicorn listener, the real ``WSTransport``) against a peer that
completes the handshake and then never reads, and measures the loop the way an operator sees it: how
late a scheduled 5 ms callback actually fires. Loop lateness — not elapsed send time — is the
invariant: the frame is allowed to take as long as it takes to leave, but the loop must keep running
while it does.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket

from starlette.websockets import WebSocket

from tests.waiting import async_wait_until
from tui_gateway import ws_egress, ws as ws_module
from tui_gateway.ws import WSTransport

# The loop must keep scheduling through a 24 MiB frame to a backed-up peer. Before this change the
# same frame was zlib-compressed on the loop (permessage-deflate) — hundreds of milliseconds of
# un-preemptible occupancy; the bound is far above the copy cost it replaces and far below that.
_LOOP_LATENESS_BUDGET_S = 0.25
_TICK_S = 0.005
_PAYLOAD_CHARS = 24 * 1024 * 1024
_SEND_DEADLINE_S = 1.0

_UPGRADE = (
    b"GET /api/ws HTTP/1.1\r\n"
    b"Host: 127.0.0.1\r\n"
    b"Upgrade: websocket\r\n"
    b"Connection: Upgrade\r\n"
    b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
    b"Sec-WebSocket-Version: 13\r\n"
    # Desktop/CLI/dashboard clients all offer this; the server's answer decides the loop cost.
    b"Sec-WebSocket-Extensions: permessage-deflate\r\n"
    b"\r\n"
)


class _Recorder:
    """Records how late a callback scheduled every ``_TICK_S`` actually fires."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._expected = loop.time()
        self._handle: asyncio.TimerHandle | None = None
        self.lateness: list[float] = []

    def _tick(self) -> None:
        now = self._loop.time()
        self.lateness.append(max(0.0, now - self._expected))
        self._expected = now + _TICK_S
        self._handle = self._loop.call_later(_TICK_S, self._tick)

    def start(self) -> None:
        self._expected = self._loop.time() + _TICK_S
        self._handle = self._loop.call_later(_TICK_S, self._tick)

    def stop(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None


async def _start_listener(app):
    """Bind and start the real uvicorn listener the gateway uses, on an ephemeral loopback port."""
    from hermes_cli.web_server_lifecycle import build_listener_server

    config, server = build_listener_server(app, "127.0.0.1", 0, auth_required=False, lifespan="off")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.setblocking(False)
    port = listener.getsockname()[1]
    if not config.loaded:
        config.load()
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[listener])
    assert server.started
    return server, listener, port


def test_oversized_frame_to_a_backed_up_peer_keeps_the_loop_running(monkeypatch):
    monkeypatch.setattr(ws_module, "_WS_SEND_DEADLINE_S", _SEND_DEADLINE_S)
    ws_egress.reset()
    # One frame, pre-serialized: this test is about the write, not json.dumps. Hex of random bytes
    # so the payload cannot be compressed away — 24 MiB stays 24 MiB on the socket either way, which
    # is what makes the two paths comparable: same bytes written, different loop cost.
    frame = json.dumps({"method": "event", "params": {
        "type": "tool.complete", "text": os.urandom(_PAYLOAD_CHARS // 2).hex()}})

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        send_started, send_done, sending = asyncio.Event(), asyncio.Event(), {}
        recorder = _Recorder(loop)

        async def app(scope, receive, send):
            if scope["type"] != "websocket":
                return
            ws = WebSocket(scope, receive, send)
            await ws.accept()
            transport = WSTransport(ws, loop, peer="pra-6844")
            sending["transport"] = transport
            send_started.set()
            try:
                await transport._safe_send_many([frame])
            finally:
                send_done.set()

        server, listener, port = await _start_listener(app)
        try:
            # Handshake, then never read: the receive window fills and the writer parks in drain().
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_UPGRADE)
            await writer.drain()
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
            assert head.startswith(b"HTTP/1.1 101"), head[:200]
            # The offer above is what a Desktop/CLI client sends; a loopback control socket must
            # answer without the extension, or this frame's zlib runs on the loop (PRA-6844).
            assert b"sec-websocket-extensions" not in head.lower(), head[:300]

            await asyncio.wait_for(send_started.wait(), timeout=10)
            recorder.start()
            await asyncio.wait_for(send_done.wait(), timeout=10)
            recorder.stop()

            transport = sending["transport"]
            # The deadline governs the wait AFTER the synchronous half, so a peer that never drains
            # is closed (1011) instead of leaving every later frame parked behind the writer lock —
            # and the write returned inside the deadline instead of waiting on the deflated socket.
            await async_wait_until(lambda: transport.closed, timeout=_SEND_DEADLINE_S + 5,
                                   what="the stalled send to hit its deadline and latch the transport")

            writer.close()
        finally:
            server.should_exit = True
            await server.shutdown(sockets=[listener])
            listener.close()

        assert recorder.lateness, "the loop probe never ran"
        worst = max(recorder.lateness)
        assert worst < _LOOP_LATENESS_BUDGET_S, (
            f"the loop went {worst * 1000:.0f} ms without scheduling while one {len(frame)}-char frame "
            f"was written to a peer whose window was full (budget {_LOOP_LATENESS_BUDGET_S * 1000:.0f} ms): "
            "the frame's synchronous compression/copy is loop occupancy no send deadline can preempt")

        stats = ws_egress.snapshot()
        assert stats["oversized_frames"] >= 1, stats
        assert stats["max_frame_bytes"] >= len(frame), stats
        assert stats["cost_seconds"] >= _SEND_DEADLINE_S, (
            "the parked frame must be accounted with the time it actually held the writer, "
            f"got {stats['cost_seconds']}s")

    asyncio.run(_run())


def test_egress_stats_record_the_frame_mix():
    """The mix has to be observable after a restart: frame sizes, per-send cost, oversized share."""
    ws_egress.reset()
    assert ws_egress.record_send(2048, 0.0001) is False
    assert ws_egress.record_send(ws_egress.LARGE_FRAME_BYTES + 1, 0.004) is True

    stats = ws_egress.snapshot()
    assert stats["frames"] == 2
    assert stats["bytes"] == 2048 + ws_egress.LARGE_FRAME_BYTES + 1
    assert stats["oversized_frames"] == 1
    assert stats["max_frame_bytes"] == ws_egress.LARGE_FRAME_BYTES + 1
    assert sum(bucket["frames"] for bucket in stats["size_buckets"]) == 2
    assert sum(bucket["frames"] for bucket in stats["cost_buckets"]) == 2
    assert stats["size_buckets"][-1]["le_bytes"] is None  # open-ended tail bucket
