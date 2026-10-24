"""An oversized frame must not become loop occupancy on the control socket (PRA-6844).

``_safe_send_many`` bounds its wait with ``_WS_SEND_DEADLINE_S``, but the synchronous half of a send
(the permessage-deflate zlib walk, then the socket write) happens before it ever awaits: nothing can
preempt it, so a large frame written to a peer whose receive window is full costs the served event
loop whatever its size and compressibility make it. This drives the real writer (a real uvicorn
listener, the real ``WSTransport``) against a peer that completes the handshake and then never reads,
and measures the loop the way an operator sees it: how late a scheduled 5 ms callback actually fires.

Loop lateness — not elapsed send time — is the invariant: the frame is allowed to take as long as it
takes to leave, but the loop must keep running while it does. That holds on every uvicorn websocket
implementation, so it is asserted unconditionally.

Backpressure is NOT implementation-independent, so the two arms differ on purpose:

* ``uvicorn``'s legacy ``websockets`` implementation (deprecated, and what the serving gateway this
  card sampled runs) parks in ``drain()`` once the window fills, so the deadline fires and the
  transport latches closed — the 1011 path that makes a dead peer reconnectable. Pinned explicitly.
* uvicorn >= 0.54 resolves ``ws="auto"`` to ``websockets_sansio_impl``, whose ``asgi_send`` does
  ``conn.send_text`` + ``transport.write`` and never awaits drain for the frame it wrote: no park, no
  deadline, no latch — the frame is buffered in-process instead, and the transport stays open with
  the deadline unarmed until some *later* send happens to find the socket paused (measured on 0.54.0:
  one frame to a non-reading peer returns in 5 ms with ``closed=False`` and 25 MiB resident, PRA-6873).
  The writer now bounds the *backlog* instead of the send: over budget it lets the peer take what it
  already has for one deadline and then latches with the same 1011. So both arms assert the latch and
  that a peer which is not taking frames stops being fed them, and the reading-peer arm pins the other
  side — a legitimately huge frame the peer does read is never a verdict.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket

from starlette.websockets import WebSocket

from tui_gateway import ws_egress, ws as ws_module
from tui_gateway.ws import WSTransport

# The loop must keep scheduling through a 24 MiB frame to a backed-up peer. Before this change the
# same frame was zlib-compressed on the loop (permessage-deflate) — hundreds of milliseconds of
# un-preemptible occupancy; the bound is far above the copy cost it replaces and far below that.
_LOOP_LATENESS_BUDGET_S = 0.25
_TICK_S = 0.005
_PAYLOAD_CHARS = 24 * 1024 * 1024
_SEND_DEADLINE_S = 1.0
# The implementation the sampled serving pid runs, and the only one that applies send backpressure.
_LEGACY_WS_IMPL = "websockets"
# The writer's per-peer backlog ceiling, and a deadline long enough that a peer which IS reading
# drains a 24 MiB frame inside it even on a busy runner.
_BUDGET_BYTES = 8 * 1024 * 1024
_READING_PEER_DEADLINE_S = 10.0

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
        self.one_tick = asyncio.Event()

    def _tick(self) -> None:
        now = self._loop.time()
        self.lateness.append(max(0.0, now - self._expected))
        self._expected = now + _TICK_S
        self._handle = self._loop.call_later(_TICK_S, self._tick)
        self.one_tick.set()

    def start(self) -> None:
        self._expected = self._loop.time() + _TICK_S
        self._handle = self._loop.call_later(_TICK_S, self._tick)

    def stop(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None


async def _start_listener(app, ws_impl: str | None = None):
    """Bind and start the real uvicorn listener the gateway uses, on an ephemeral loopback port.

    *ws_impl* pins uvicorn's websocket implementation before the config loads; ``None`` keeps the
    builder's own choice (``auto``, i.e. whatever uvicorn ships) — which is what production gets.
    Returns the protocol class actually resolved, so a test never has to guess which one it measured.
    """
    from hermes_cli.web_server_lifecycle import build_listener_server

    config, server = build_listener_server(app, "127.0.0.1", 0, auth_required=False, lifespan="off")
    if ws_impl is not None:
        config.ws = ws_impl
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.setblocking(False)
    port = listener.getsockname()[1]
    if not config.loaded:
        config.load()
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[listener])
    assert server.started
    return server, listener, port, config.ws_protocol_class


def _drive_oversized_frames(ws_impl: str | None, *, frames: int = 1, peer_reads: bool = False) -> dict:
    """Send oversized frames to a peer; report what the loop and the writer did.

    *peer_reads* False is the stalled-client case (completes the handshake, then never reads a byte).
    True is the healthy case with a large frame: the peer keeps reading, so nothing may be closed.
    """
    # Frames are pre-serialized: this test is about the write, not json.dumps. Hex of random bytes
    # so the payload cannot be compressed away — 24 MiB stays 24 MiB on the socket either way, which
    # is what makes the two paths comparable: same bytes written, different loop cost.
    frame = json.dumps({"method": "event", "params": {
        "type": "tool.complete", "text": os.urandom(_PAYLOAD_CHARS // 2).hex()}})
    ws_egress.reset()

    async def _run() -> dict:
        loop = asyncio.get_running_loop()
        send_started, send_done, sending = asyncio.Event(), asyncio.Event(), {}
        recorder = _Recorder(loop)
        outcome: dict = {}

        async def app(scope, receive, send):
            if scope["type"] != "websocket":
                return
            ws = WebSocket(scope, receive, send)
            await ws.accept()
            transport = WSTransport(ws, loop, peer="pra-6873")
            sending["transport"] = transport
            send_started.set()
            started = loop.time()
            try:
                await transport._safe_send_many([frame] * frames)
            finally:
                outcome["send_seconds"] = loop.time() - started
                send_done.set()

        async def _keep_reading(peer) -> None:
            while await peer.read(1 << 16):
                pass

        server, listener, port, protocol_class = await _start_listener(app, ws_impl)
        reader_task = None
        try:
            # Handshake, then either keep reading or never read again: in the latter the receive
            # window fills and the backlog the writer has to bound starts growing.
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(_UPGRADE)
            await writer.drain()
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
            assert head.startswith(b"HTTP/1.1 101"), head[:200]
            # The offer above is what a Desktop/CLI client sends; a loopback control socket must
            # answer without the extension, or this frame's zlib runs on the loop (PRA-6844).
            assert b"sec-websocket-extensions" not in head.lower(), head[:300]
            if peer_reads:
                reader_task = asyncio.create_task(_keep_reading(reader))

            await asyncio.wait_for(send_started.wait(), timeout=10)
            recorder.start()
            await asyncio.wait_for(send_done.wait(), timeout=10)
            # A send that never parks can finish inside one tick; take a sample after it as well so
            # the probe is never empty (on that path the honest answer is "no occupancy").
            await asyncio.wait_for(recorder.one_tick.wait(), timeout=10)
            recorder.stop()
            writer.close()
        finally:
            if reader_task is not None:
                reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reader_task
            server.should_exit = True
            await server.shutdown(sockets=[listener])
            listener.close()

        assert recorder.lateness, "the loop probe never ran"
        outcome["worst_lateness"] = max(recorder.lateness)
        outcome["frame_chars"] = len(frame)
        outcome["closed"] = sending["transport"].closed
        outcome["stats"] = ws_egress.snapshot()
        outcome["protocol"] = getattr(protocol_class, "__name__", str(protocol_class))
        return outcome

    return asyncio.run(_run())


def _assert_loop_kept_running(outcome: dict) -> None:
    """The invariant, implementation-independent: the loop keeps scheduling while the frame leaves."""
    worst = outcome["worst_lateness"]
    assert worst < _LOOP_LATENESS_BUDGET_S, (
        f"the loop went {worst * 1000:.0f} ms without scheduling while one "
        f"{outcome['frame_chars']}-char frame was written to a peer whose window was full "
        f"(budget {_LOOP_LATENESS_BUDGET_S * 1000:.0f} ms, protocol {outcome['protocol']}): "
        "the frame's synchronous compression/copy is loop occupancy no send deadline can preempt")
    stats = outcome["stats"]
    assert stats["oversized_frames"] >= 1, stats
    assert stats["max_frame_bytes"] >= outcome["frame_chars"], stats


def test_oversized_frame_to_a_backed_up_peer_keeps_the_loop_running(monkeypatch):
    """Legacy implementation (the one the sampled serving pid runs): the deadline fires and latches."""
    monkeypatch.setattr(ws_module, "_WS_SEND_DEADLINE_S", _SEND_DEADLINE_S)
    outcome = _drive_oversized_frames(_LEGACY_WS_IMPL)

    _assert_loop_kept_running(outcome)
    # Same backlog accounting on the implementation that has real flow control: the parked frame is
    # resident, so the peak is checkable from the serving process here too.
    assert outcome["stats"]["max_buffered_bytes"] > 0, outcome["stats"]
    # The deadline governs the wait AFTER the synchronous half, so a peer that never drains is
    # closed (1011) instead of leaving every later frame parked behind the writer lock.
    assert outcome["send_seconds"] >= _SEND_DEADLINE_S, outcome
    assert outcome["closed"] is True, (
        "a stalled peer must latch the transport so handle_ws's read loop ends and the client "
        f"reconnects, protocol={outcome['protocol']}")
    assert outcome["stats"]["cost_seconds"] >= _SEND_DEADLINE_S, (
        "the parked frame must be accounted with the time it actually held the writer, "
        f"got {outcome['stats']['cost_seconds']}s")


def test_oversized_frame_keeps_the_loop_running_on_the_default_protocol(monkeypatch):
    """Whatever uvicorn ships by default must give the frame a verdict, not just buffer it.

    One frame is the case that had none: uvicorn 0.54's sans-I/O ``send`` writes and returns in
    milliseconds, so a peer that never reads leaves 24 MiB sitting in the process's write buffer with
    the transport open and the deadline unarmed — the socket only latches when some *later* frame
    happens to be sent, and with per-event batches that may never come. Measured on 0.54.0: base
    returns in 5 ms with ``closed=False``; with the budget the same send waits out the deadline and
    latches 1011, leaving exactly one frame resident (the peer took none of it).
    """
    monkeypatch.setattr(ws_module, "_WS_SEND_DEADLINE_S", _SEND_DEADLINE_S)
    outcome = _drive_oversized_frames(None)

    _assert_loop_kept_running(outcome)
    assert outcome["closed"] is True, (
        "a peer that never drains must latch so handle_ws's read loop ends and the client "
        f"reconnects, protocol={outcome['protocol']} "
        f"buffered={outcome['stats'].get('max_buffered_bytes')}")
    assert outcome["send_seconds"] >= _SEND_DEADLINE_S, outcome
    stats = outcome["stats"]
    # One frame is resident — the one the peer took none of. The ceiling is the budget plus whatever
    # that single frame is; it never becomes N frames of growth, because the next send never happens.
    assert stats["frames"] == 1, stats
    assert stats["over_budget_frames"] == 1, stats
    assert _BUDGET_BYTES < stats["max_buffered_bytes"] <= _BUDGET_BYTES + outcome["frame_chars"], stats


def test_a_second_frame_is_never_written_on_top_of_an_undrained_backlog(monkeypatch):
    """The batch stops at the frame the peer could not take, instead of feeding the buffer further."""
    monkeypatch.setattr(ws_module, "_WS_SEND_DEADLINE_S", _SEND_DEADLINE_S)
    outcome = _drive_oversized_frames(None, frames=3)

    assert outcome["closed"] is True, outcome
    assert outcome["stats"]["frames"] == 1, outcome["stats"]
    assert outcome["stats"]["max_buffered_bytes"] <= _BUDGET_BYTES + outcome["frame_chars"], outcome["stats"]


def test_a_peer_that_reads_a_huge_frame_is_never_closed(monkeypatch):
    """The budget bounds a backlog, it is not a per-frame verdict: a reading peer still gets 24 MiB.

    The Desktop ``file.attach`` producer puts hundreds of MiB in one frame, so a budget that closed
    every over-budget send on sight would break the one producer whose frames are that big.
    """
    monkeypatch.setattr(ws_module, "_WS_SEND_DEADLINE_S", _READING_PEER_DEADLINE_S)
    outcome = _drive_oversized_frames(None, peer_reads=True)

    _assert_loop_kept_running(outcome)
    assert outcome["closed"] is False, (
        "a peer that drains its frames must not be latched, protocol="
        f"{outcome['protocol']} buffered={outcome['stats'].get('max_buffered_bytes')}")


def test_egress_stats_record_the_frame_mix():
    """The mix has to be observable after a restart: frame sizes, per-send cost, oversized share."""
    ws_egress.reset()
    assert ws_egress.record_send(2048, 0.0001) is False
    assert ws_egress.record_send(ws_egress.LARGE_FRAME_BYTES + 1, 0.004) is True
    # The backlog half of the same accounting (PRA-6873): the peak the transport was holding, and
    # whether that send found it already past the budget.
    assert ws_egress.record_send(4096, 0.0002, buffered_bytes=5_000_000, budget_bytes=8_388_608) is False
    assert ws_egress.record_send(4096, 0.0002, buffered_bytes=9_000_000, budget_bytes=8_388_608) is False

    stats = ws_egress.snapshot()
    assert stats["frames"] == 4
    assert stats["bytes"] == 2048 + ws_egress.LARGE_FRAME_BYTES + 1 + 4096 * 2
    assert stats["oversized_frames"] == 1
    assert stats["max_frame_bytes"] == ws_egress.LARGE_FRAME_BYTES + 1
    assert stats["max_buffered_bytes"] == 9_000_000
    assert stats["over_budget_frames"] == 1
    assert sum(bucket["frames"] for bucket in stats["size_buckets"]) == 4
    assert sum(bucket["frames"] for bucket in stats["cost_buckets"]) == 4
    assert stats["size_buckets"][-1]["le_bytes"] is None  # open-ended tail bucket
