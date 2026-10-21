"""The control socket's peer gate costs the loop a cache read, never a metadata walk (PRX-5821).

``_posix_peer_subject`` used to re-derive the home's trust verdict on every connect — ``lstat`` plus
``home.absolute() != home.resolve()``, a ``realpath()`` that stats every path component — and it ran
on the event loop, because the value was evaluated as an argument to ``await self._answer_off_loop``.
On a writeback-saturated disk one connect parked the loop long enough to swallow a whole shutdown
drain: the 07:06:09 PT SIGTERM never finished, the watchdog fired at 58.1 s and force-exited the
process, costing 28 runs.

The verdict is a constant of the server (the home is fixed for the life of the listener), so it is
resolved once inside ``start()``, off the loop. These tests pin both halves: a blocked ``realpath``
must not delay a connection served during it, and the gate must still answer ``uid:<uid>`` for a
same-uid peer without touching the filesystem at all.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from gateway.control_socket import GatewayControlServer, resolve_client_socket_path

pytestmark = pytest.mark.platforms("posix")  # unix-socket transport + getpeereid/SO_PEERCRED

# Stonewall for the injected realpath: a blocked call cannot be mistaken for a fast one by load, and
# a slow runner only makes the failing (loop-blocked) case fail harder.
_BLOCK_S = 5.0
# A local unix-socket round trip plus a JSON decode answers in milliseconds; a whole second is
# generous even on a loaded runner, and still an order of magnitude under ``_BLOCK_S``.
_ANSWER_DEADLINE = 1.0


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    d = tmp_path / "home" / ".hermes"
    d.mkdir(parents=True)
    return d


def _stonewalled_resolve(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace ``Path.resolve`` with a call that never returns in time, recording every path tried."""
    seen: list[Path] = []
    real = Path.resolve
    gate = threading.Event()  # never set: teardown does not need the call to finish

    def resolve(self, *args, **kwargs):
        seen.append(self)
        gate.wait(timeout=_BLOCK_S)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    return seen


def _peer_writer():
    ours, _theirs = socket.socketpair(socket.AF_UNIX)

    class Writer:
        def get_extra_info(self, _key):
            return ours

    return Writer()


async def _identify(home: Path) -> tuple[dict, float]:
    """One ``identify`` over the real socket: the answer envelope and how long it took."""
    started = time.monotonic()
    reader, writer = await asyncio.open_unix_connection(str(resolve_client_socket_path(home)))
    try:
        writer.write(json.dumps({"protocol": 1, "verb": "identify", "id": 7}).encode("utf-8") + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=_BLOCK_S)
    finally:
        writer.close()
    return json.loads(line.decode("utf-8")), time.monotonic() - started


def test_a_connection_served_during_a_blocked_realpath_does_not_stall_the_loop(
        home: Path, monkeypatch: pytest.MonkeyPatch):
    """Red on bb06d905: the identify answer only arrives once the injected ``realpath`` gives up.

    The peer value is prepared as an ``await`` argument before the hand-off, so a home it must
    resolve holds every other callback on the loop for as long as that walk takes.
    """

    async def scenario() -> tuple[dict, float, bool, list[Path]]:
        server = GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 4242}})
        assert await server.start()
        try:
            # Injected only after start(): the verdict is a startup constant, so nothing a later
            # connection does should touch the home's metadata at all.
            seen = _stonewalled_resolve(monkeypatch)
            answered = asyncio.Event()

            async def noop() -> bool:
                # A callback that needs nothing but the loop: it must keep running while the
                # control request is outstanding.
                while not answered.is_set():
                    await asyncio.sleep(0)
                return True

            noop_task = asyncio.ensure_future(noop())
            try:
                answer, elapsed = await _identify(home)
            finally:
                answered.set()
            return answer, elapsed, await noop_task, seen
        finally:
            monkeypatch.undo()
            await server.stop()

    answer, elapsed, noop_completed, seen = asyncio.run(scenario())
    assert answer.get("ok") is True, answer
    assert seen == [], f"a connection walked the home metadata (resolve on {seen})"
    assert noop_completed is True
    assert elapsed < _ANSWER_DEADLINE, (
        f"identification took {elapsed:.2f}s while a slow realpath was waiting — "
        "the peer subject is still being derived on the loop")


def test_start_resolves_the_verdict_off_the_loop(home: Path):
    """The one walk that remains happens in ``start()``, on a worker, before the socket can answer."""
    threads: list[int] = []

    async def scenario() -> bool | None:
        server = GatewayControlServer(home)
        real = server._compute_home_trust
        server._compute_home_trust = lambda: threads.append(threading.get_ident()) or real()  # type: ignore[method-assign]
        assert await server.start()
        try:
            assert threads, "start() did not resolve the peer gate before serving"
            return threads[0] != threading.get_ident()
        finally:
            await server.stop()

    assert asyncio.run(scenario()) is True


def test_the_peer_gate_answers_from_the_cached_verdict(home: Path, monkeypatch: pytest.MonkeyPatch):
    """A same-uid peer still gets ``uid:<uid>`` while every filesystem call is poisoned.

    The gate's answer is the ticket-minting trust boundary: caching the verdict may make it cheaper,
    never weaker.
    """
    server = GatewayControlServer(home)
    writer = _peer_writer()
    assert server._posix_peer_subject(writer) == f"uid:{os.getuid()}"

    def poisoned(*_args, **_kwargs):
        raise AssertionError("the connection path walked the home again")

    monkeypatch.setattr(Path, "resolve", poisoned)
    monkeypatch.setattr(Path, "lstat", poisoned)
    assert server._posix_peer_subject(writer) == f"uid:{os.getuid()}"


def test_an_untrusted_home_still_refuses_the_label(home: Path):
    """The cached value is a verdict, not an allowance: a world-writable home labels nobody."""
    home.chmod(0o702)
    writer = _peer_writer()
    assert GatewayControlServer(home)._posix_peer_subject(writer) is None
