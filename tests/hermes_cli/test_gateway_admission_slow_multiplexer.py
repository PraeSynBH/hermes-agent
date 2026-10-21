"""A multiplexer that is up but slow to answer is not an absent multiplexer (PRX-5774).

``_served_by_multiplexer`` asked the root for ``identify`` with one ``PROBE_TIMEOUT`` (2.0 s) and
folded *every* failure — including a ``TimeoutError`` from a socket that had just ACCEPTED the
connection — into ``None``, i.e. "this host has no multiplexer here". The caller then read the
profile's own ``gateway.lock``, which the multiplexer holds while serving it, as a reservation and
reported ``starting``: a boot that could never finish. Four runs on 2026-10-20T23:03-00:21Z died
that way (``phase=runtime-boot, elapsed=120.0s ... owner was still booting at the budget``) on pid
88382, which was serving those same profiles throughout and whose ``gateway.log`` carries no boot
banner after 18:08:03Z. Measured at load1 81 — below the load on all four — five root identifies
took 0.42/0.40/0.81/1.90/0.20 s: the top sample already sits at 95% of the probe budget, so every
probe at load1 140+ times out.

The contracts pinned here:

* a root that accepts the connection and stays silent is a ``control_timeout`` naming the socket and
  how long ``identify`` went unanswered — never ``None``, and so never ``starting``;
* a root with nothing listening (``FileNotFoundError``/``ConnectionRefusedError``) is still "no
  multiplexer", so a host without one keeps the profile's own start path.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest


def _multiplexed_profile(tmp_path: Path) -> tuple[Path, Path]:
    """A default-home root with one named profile under it, as the host lays them out."""
    root = tmp_path / ".hermes"
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    home = root / "profiles" / "cto"
    home.mkdir(parents=True, mode=0o700)
    (home / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    return root, home


def _hold_lock(lock_path: Path) -> subprocess.Popen:
    """Hold ``gateway.lock`` from another process, the way a serving multiplexer does.

    The ownership probe is a real ``flock``; taking it in a child keeps the holder a different
    process, which is the only shape the probe answers for.
    """
    script = (
        "import fcntl, sys\n"
        "handle = open(sys.argv[1], 'a+')\n"
        "fcntl.flock(handle, fcntl.LOCK_EX)\n"
        "print('held', flush=True)\n"
        "sys.stdin.read()\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", script, str(lock_path)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None and proc.stdout.readline().strip() == "held"
    return proc


def _release(proc: subprocess.Popen) -> None:
    if proc.stdin is not None:
        proc.stdin.close()
    proc.wait(timeout=30)


@pytest.mark.platforms("posix")
def test_a_silent_multiplexer_is_a_control_timeout_not_an_absent_owner(tmp_path):
    """A root that took the connection and never answered keeps its own verdict, named."""
    from gateway.control_socket import GatewayControlServer
    from hermes_cli import gateway_runtime as runtime

    root, home = _multiplexed_profile(tmp_path)
    stall = threading.Event()  # set at teardown so the stalled handler threads exit

    def identify():
        stall.wait(30)  # outlives the caller's probe
        return {"runtime_protocol": 1, "state": "ready", "served_profiles": []}

    async def probe():
        server = GatewayControlServer(root, verb_handlers={"identify": identify})
        assert await server.start()
        try:
            return await asyncio.to_thread(runtime._served_by_multiplexer, home, timeout=0.5)
        finally:
            stall.set()
            await server.stop()

    result = asyncio.run(probe())
    assert result is not None, "a root that accepted the connection is not 'no multiplexer here'"
    assert (result.state, result.reason_code) == ("inaccessible", "control_timeout")
    assert "did not answer identify" in (result.detail or "")


@pytest.mark.platforms("posix")
def test_a_borrowed_lock_beside_a_silent_multiplexer_is_not_a_boot(tmp_path):
    """The whole 2026-10-20 shape: held lock + slow root must not be reported as ``starting``."""
    from gateway.control_socket import GatewayControlServer
    from hermes_cli import gateway_runtime as runtime

    root, home = _multiplexed_profile(tmp_path)
    holder = _hold_lock(home / "gateway.lock")
    stall = threading.Event()

    def identify():
        stall.wait(30)
        return {"runtime_protocol": 1, "state": "ready", "served_profiles": []}

    async def probe():
        server = GatewayControlServer(root, verb_handlers={"identify": identify})
        assert await server.start()
        try:
            return await asyncio.to_thread(runtime.discover_gateway_endpoint, home, timeout=0.5)
        finally:
            stall.set()
            await server.stop()

    try:
        result = asyncio.run(probe())
    finally:
        _release(holder)
    assert (result.state, result.reason_code) == ("inaccessible", "control_timeout")
    assert "did not answer identify" in (result.detail or "")


@pytest.mark.platforms("posix")
def test_a_host_with_no_multiplexer_keeps_its_own_start_path(tmp_path):
    """Nothing listening on the root is still "no multiplexer": return ``None``, do not invent a stall."""
    from hermes_cli import gateway_runtime as runtime

    _root, home = _multiplexed_profile(tmp_path)
    assert runtime._served_by_multiplexer(home, timeout=0.2) is None
