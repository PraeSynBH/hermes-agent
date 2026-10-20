"""The control socket answers on its own executor and reports its queue wait (PRA-6513).

``identify`` IS liveness: a consumer treats a connectable socket that answers it as a live gateway,
and falls back to the state-file/scan layer when it does not. So the answer must never queue behind
the loop's DEFAULT executor, which carries every ``asyncio.to_thread`` in the process — turn tools,
and the periodic profile reconcile sweep (PRA-6507). A drained control pool refuses immediately
(``control_busy``) instead of queueing past the client's deadline, and every answer carries
``queue_wait_ms``/``handler_ms`` so "the loop is starved" is a measurement, not a guess.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest

from gateway.control_socket import (
    CONTROL_PROTOCOL_VERSION,
    GatewayControlServer,
    identify_gateway,
    query_gateway_control,
    resolve_client_socket_path,
)
from tests.waiting import async_wait_until

pytestmark = pytest.mark.platforms("posix")  # unix-socket transport; the pipe half is the wine2e lane

# CPython's default-executor size: ``min(32, cpu+4)``. That many blocked workers saturate it.
_DEFAULT_POOL_SLOTS = min(32, (os.cpu_count() or 1) + 4)
# Wall-clock bound for one request (the runner's >=2s rule). The client's own deadline is 2s, so a
# request that cannot answer inside this never answered on the wire either.
_DEADLINE = 5.0


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    d = tmp_path / "home" / ".hermes"
    d.mkdir(parents=True)
    return d


async def _ask(home: Path, request: dict) -> dict:
    """One request line on the real socket, read back as the raw envelope (timings included)."""
    reader, writer = await asyncio.open_unix_connection(str(resolve_client_socket_path(home)))
    try:
        writer.write(json.dumps(request).encode("utf-8") + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=_DEADLINE)
    finally:
        writer.close()
    return json.loads(line.decode("utf-8"))


def test_identify_answers_while_the_default_executor_is_saturated(home: Path):
    """A control answer must not wait for a free worker in the pool the rest of the process shares.

    Red on main: the request is dispatched through ``run_in_executor(None, ...)``, so it queues
    behind every blocked default-pool worker and the client times out.
    """
    saturated: list[bool] = []

    def block() -> None:
        saturated.append(True)
        release.wait(timeout=60)

    async def scenario() -> tuple[dict, float]:
        server = GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 4242}})
        assert await server.start()
        tasks = [asyncio.ensure_future(asyncio.to_thread(block)) for _ in range(_DEFAULT_POOL_SLOTS)]
        try:
            await async_wait_until(lambda: len(saturated) >= _DEFAULT_POOL_SLOTS,
                                   what="the default executor to be saturated")
            began = time.monotonic()
            answer = await _ask(home, {"verb": "identify", "id": 7, "protocol": 1})
            return answer, (time.monotonic() - began) * 1000
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await server.stop()

    release = threading.Event()
    answer, elapsed_ms = asyncio.run(scenario())
    assert answer["ok"] is True, answer
    assert answer["result"] == {"pid": 4242}
    assert answer["id"] == 7
    assert answer["protocol"] == CONTROL_PROTOCOL_VERSION
    # The breakdown: this request answered in single-digit milliseconds while 32 default-pool
    # workers were blocked, so its wait was the handler's, not the shared pool's.
    assert 0 <= answer["queue_wait_ms"] < elapsed_ms + 1
    assert 0 <= answer["handler_ms"] < elapsed_ms + 1
    assert answer["queue_wait_ms"] + answer["handler_ms"] <= elapsed_ms + 250


def test_drained_control_pool_refuses_with_control_busy(home: Path):
    """Both control workers busy -> a fast structured refusal, never a silent queue past the deadline.

    Red on main: there is no control pool and no admission, so the third request is simply served
    (``ok: true``) — nothing tells a caller the control path is saturated.
    """
    entered: list[bool] = []
    held = threading.Event()

    def hold() -> dict:
        entered.append(True)
        held.wait(timeout=60)
        return {"held": True}

    async def scenario() -> dict:
        server = GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 4242},
                                                           "hold": hold})
        assert await server.start()
        holders = [asyncio.ensure_future(_ask(home, {"verb": "hold", "id": n, "protocol": 1}))
                   for n in (1, 2)]
        try:
            await async_wait_until(lambda: len(entered) >= 2, what="both control workers to be busy")
            return await _ask(home, {"verb": "identify", "id": 9, "protocol": 1})
        finally:
            held.set()
            await asyncio.gather(*holders, return_exceptions=True)
            await server.stop()

    answer = asyncio.run(scenario())
    assert answer["ok"] is False, answer
    assert answer["error"] == "control_busy"
    assert answer["id"] == 9
    assert answer["protocol"] == CONTROL_PROTOCOL_VERSION


def test_existing_consumers_still_get_the_bare_result(home: Path):
    """The wire protocol is unchanged: timings ride the envelope, ``result`` is exactly what it was."""

    async def scenario() -> dict:
        server = GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 7}})
        assert await server.start()
        try:
            return await _ask(home, {"verb": "identify", "id": 1, "protocol": 1})
        finally:
            await server.stop()

    envelope = asyncio.run(scenario())
    assert envelope["result"] == {"pid": 7}
    assert {"queue_wait_ms", "handler_ms"} <= set(envelope)

    async def query() -> tuple[dict, dict]:
        server = GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 7}})
        assert await server.start()
        try:
            return (await asyncio.to_thread(query_gateway_control, home, "identify"),
                    await asyncio.to_thread(identify_gateway, home))
        finally:
            await server.stop()

    for answer in asyncio.run(query()):
        assert answer == {"pid": 7}
