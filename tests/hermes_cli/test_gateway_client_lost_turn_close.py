"""A lost turn names the close that ended it (PRX-6150).

A managed restart defers ``stop()`` for up to 1800 s to protect in-flight turns, and the restart
window cuts session sockets with close **1013**. On 2026-10-23 eight runs died inside one such
deferral (18:54:37-18:59:16 PDT) carrying the same "turn outcome is unknown" sentence and nothing
else: the copy named neither a code nor a reason, so a QA window could not tell the owner's own cut
from a transport that died on its own — the card's first question. The reader now carries the close
the transport reported into that copy.
"""
import asyncio

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.frames import Close

from hermes_cli.gateway_client import GatewayClient, close_detail


def _closed(code, reason=""):
    return ConnectionClosed(Close(code, reason), None)


def test_close_detail_names_a_websocket_close_and_nothing_else():
    assert close_detail(_closed(1013, "restart wait")) == " (close 1013: restart wait)"
    assert close_detail(_closed(1013)) == " (close 1013)"
    # A failure that carries no close says nothing extra: no invented code in a run's log.
    assert close_detail(OSError("Connection reset by peer")) == ""
    assert close_detail(TimeoutError()) == ""
    assert close_detail(None) == ""


class _ClosingSocket:
    """The owner hung up with *close* before the reader saw a single frame."""

    def __init__(self, close):
        self._close = close

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self._close


@pytest.mark.asyncio
async def test_the_reader_carries_the_close_into_the_lost_turn_copy():
    client = GatewayClient(_ClosingSocket(_closed(1013, "restart wait")))
    pending = asyncio.get_running_loop().create_future()
    client.pending[1] = pending

    await client._read()

    (error,) = [pending.exception()]
    assert "turn outcome is unknown" in str(error)
    assert str(error).endswith(" (close 1013: restart wait)"), str(error)
    # The same copy is what the turn loop reads off the queue: one sentence, one cause.
    assert str(client.events.get_nowait()) == str(error)


@pytest.mark.asyncio
async def test_a_transport_failure_without_a_close_adds_no_code():
    client = GatewayClient(_ClosingSocket(OSError("Connection reset by peer")))
    pending = asyncio.get_running_loop().create_future()
    client.pending[1] = pending

    await client._read()

    (error,) = [pending.exception()]
    assert "turn outcome is unknown" in str(error)
    assert "close " not in str(error), str(error)
