"""The control socket must not pay per-frame zlib to talk to its own kernel (PRA-6844).

``permessage-deflate`` is compressed inside websockets' ``write_frame_sync`` on the served event
loop, and that call cannot be preempted by the caller's send deadline: every frame on ``/api/ws``
becomes loop occupancy before ``sock.send`` runs. A 1200 s sample of the serving gateway loop put
184/2059 loop-thread shots inside ``permessage_deflate.encode``. On a loopback bind the CPU buys
nothing — the peer is the same kernel — so the extension must not be negotiated there, while a
tunnel bind keeps the bandwidth trade configurable.

The listener assertion is a real handshake against the real control socket, not a Config snapshot:
it is the negotiated extension that decides the loop cost.
"""

from __future__ import annotations

import asyncio
import types

# Imported at collection time on purpose: the first import of the tui_gateway stack runs PM's
# bootstrap environment activation, which reads this checkout's real manifest — fine outside the
# per-test home-I/O guard, refused inside it.
import tui_gateway.ws
from hermes_cli.web_server_boundary import ws_per_message_deflate_for

_OFFER = "Sec-WebSocket-Extensions: permessage-deflate\r\n"


def _upgrade_request(*, offer_extensions: bool, subprotocols: tuple[str, ...] = ()) -> bytes:
    headers = [
        "GET /api/ws HTTP/1.1",
        "Host: 127.0.0.1",
        "Upgrade: websocket",
        "Connection: Upgrade",
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==",
        "Sec-WebSocket-Version: 13",
    ]
    if subprotocols:
        headers.append("Sec-WebSocket-Protocol: " + ", ".join(subprotocols))
    if offer_extensions:
        headers.append(_OFFER.rstrip("\r\n"))
    return ("\r\n".join(headers) + "\r\n\r\n").encode()


async def _handshake(port: int, *, offer_extensions: bool,
                     subprotocols: tuple[str, ...] = ()) -> tuple[int, dict[str, str]]:
    """Drive one raw upgrade against *port*; return the status line's code and the response headers."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(_upgrade_request(offer_extensions=offer_extensions, subprotocols=subprotocols))
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
    finally:
        writer.close()
    lines = head.decode("latin-1").split("\r\n")
    code = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        if name:
            headers[name.strip().lower()] = value.strip()
    return code, headers


async def _with_control_listener(home: str, scenario) -> None:
    """Run *scenario* against a live gateway session listener, then stop it."""
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from gateway.runtime_bootstrap import TicketStore

    accepted: list[dict] = []

    async def fake_handle_ws(ws, **kwargs):
        accepted.append(kwargs)
        await ws.accept()
        await ws.close()

    original = tui_gateway.ws.handle_ws
    tui_gateway.ws.handle_ws = fake_handle_ws
    tickets = TicketStore("owner", frozenset({home}))
    authority = types.SimpleNamespace(db=types.SimpleNamespace(db_path=home + "/state.db"), profile_id=home)
    runner = types.SimpleNamespace(
        _draining=False, session_runtime_descriptor={"state": "ready"},
        session_authority=authority, session_ticket_store=tickets)
    handle = await start_gateway_api(runner)
    try:
        port = int(handle.api_origin.rsplit(":", 1)[1])
        # The control socket only upgrades a caller carrying a valid local ticket, so every
        # handshake below offers one (the native subprotocol pair the desktop/CLI clients send).
        def _ticketed() -> tuple[str, ...]:
            return ("hermes-gateway-v1", "hermes-gateway-ticket." + tickets.mint(
                profile_id=home, subject="human", purpose="interactive"))
        await scenario(port, _ticketed, accepted)
    finally:
        await stop_gateway_api(handle)
        tui_gateway.ws.handle_ws = original


def test_control_socket_negotiates_no_permessage_deflate_on_loopback(tmp_path):
    """A client that OFFERS the extension must be answered without it on the loopback control socket."""
    home = tmp_path / ".hermes"
    home.mkdir()

    async def scenario(port: int, ticketed, accepted: list[dict]) -> None:
        code, headers = await _handshake(port, offer_extensions=True, subprotocols=ticketed())
        assert code == 101, f"the control socket must complete the upgrade, got {code}"
        # The upgrade must land on the gateway's own session path (handle_ws), which is reached only
        # by redeeming the ticket: an unticketed upgrade is answered by the dashboard gate with 403
        # and never gets here. Asserting the accept keeps this test honest about WHICH surface
        # negotiated the extension, instead of accepting a 101 from anywhere.
        assert accepted and accepted[-1]["auth_identity"]["native_bootstrap"] is True, (
            f"upgrade did not reach the gateway session path: accepts={accepted!r}")
        assert "sec-websocket-extensions" not in headers, (
            "the loopback control socket negotiated "
            f"{headers.get('sec-websocket-extensions')!r}: per-frame zlib then runs on the served "
            "event loop, where no send deadline can preempt it (PRA-6844)")

        # The same socket still upgrades when the client offers nothing: the extension is the only
        # thing removed, not the handshake that carries it.
        code_plain, headers_plain = await _handshake(port, offer_extensions=False, subprotocols=ticketed())
        assert code_plain == 101 and "sec-websocket-extensions" not in headers_plain

    asyncio.run(_with_control_listener(str(home), scenario))


def test_tunnel_bind_keeps_deflate_and_config_can_turn_it_off():
    """Only loopback is unconditional: a non-loopback bind trades CPU for wire bytes, configurable."""
    assert ws_per_message_deflate_for("127.0.0.1") is False
    assert ws_per_message_deflate_for("::1") is False
    assert ws_per_message_deflate_for("localhost") is False
    # Non-loopback default keeps the bandwidth behaviour that shipped before this change.
    assert ws_per_message_deflate_for("0.0.0.0") is True
    assert ws_per_message_deflate_for("0.0.0.0", {}) is True
    # ... and an operator whose tunnel is not bandwidth-bound can spend the CPU the other way.
    assert ws_per_message_deflate_for("0.0.0.0", {"ws_per_message_deflate": False}) is False
    assert ws_per_message_deflate_for("0.0.0.0", {"ws_per_message_deflate": "off"}) is False
    assert ws_per_message_deflate_for("0.0.0.0", {"ws_per_message_deflate": "true"}) is True
    # Loopback ignores the knob: there is no wire to save.
    assert ws_per_message_deflate_for("127.0.0.1", {"ws_per_message_deflate": True}) is False
