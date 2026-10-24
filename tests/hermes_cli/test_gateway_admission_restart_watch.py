"""A run dispatched into a RESTART re-observes the owner instead of dying at its budget (SOU-3960).

2026-10-24 05:44:36Z: six heartbeats were released 3 s after a replacing gateway process launched,
46 s before its control socket listened and 2m09s before it published ``running``. Every one minted
a ticket, opened the socket, then sat in ``session.create`` until its 120 s admission budget expired
at 05:46:48-51Z — 3 s after the owner could have served them. ``admit_owner`` re-observes an owner
that is BOOTING (``phase=runtime-boot``) and a restart's after-turn wait (``phase=runtime-drain``);
the in-band verbs had one RPC, one deadline and no way to tell a restart from a stalled loop.

The contracts pinned here:

* an owner that answers ``starting`` for longer than the budget, then serves, ADMITS the run rather
  than killing it at the deadline (the fix's whole point, red before it);
* a WEDGED owner — alive, publishing ``running``, answering nothing — still fails at the budget and
  earns no grace at all;
* an owner that never finishes coming up still fails, bounded by budget + ``RESTART_WAIT_GRACE_S``;
* the death text names the restart, which is what the detector's ``restart_window`` class reads;
* the readiness reader credits exactly the two names a restart is published under (``starting`` while
  the incoming owner boots, ``restarting`` for the outgoing owner's claim across the exit->boot gap,
  SOU-3966) and never invents a restart from a stale record, a dead pid or an absent owner.

Durations are SCALED (a 130 s outage is not a test): the admit test keeps the incident's ratio — the
owner serves 12.5% past the budget (130 s vs 120 s) — and the others use sub-second budgets.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

#: Well above every OS's pid range: the record reads as "owner is gone", which must never credit.
DEAD_PID = 999999


def _write_state(home: Path, state: str, *, pid: int | None = None) -> None:
    """Publish the owner's readiness exactly where the gateway publishes it."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "gateway_state.json").write_text(
        json.dumps({"pid": os.getpid() if pid is None else pid, "gateway_state": state}),
        encoding="utf-8")


def _owner(gateway_client, *, serving):
    """An owner that accepts the upgrade and answers an RPC only once it is serving.

    The 05:44:36Z shape: the socket exists and swallows the in-band verb while the process is still
    coming up, so nothing about the socket says "not yet"."""

    async def handler(ws):
        async def answer(frame):
            await serving.wait()
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": frame["id"], "result": {"ok": True}}))

        tasks = []
        try:
            async for raw in ws:
                tasks.append(asyncio.create_task(answer(json.loads(raw))))
        finally:
            for task in tasks:
                task.cancel()

    from websockets.asyncio.server import serve
    return serve(handler, "127.0.0.1", 0, subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL])


def _attach(gateway_client, port):
    endpoint = SimpleNamespace(api_origin=f"http://127.0.0.1:{port}", profile_id="p",
                               instance_id="i", code_sha=None, control_home=None)
    return gateway_client.connect_gateway(endpoint=endpoint)


def _scaled(monkeypatch, gateway_client, home, *, budget, grace):
    """Point the client at one temp home and at the scaled clock the test drives."""
    monkeypatch.setattr(gateway_client, "admission_budget_seconds", lambda: budget)
    monkeypatch.setattr(gateway_client, "_session_ticket", lambda *a, **k: "ticket")
    # raising=False: on the pre-fix tree none of these exist, and the run has to fail on the
    # BEHAVIOUR (a death at the budget) rather than on a missing name.
    monkeypatch.setattr(gateway_client, "RESTART_POLL_INTERVAL_S", 0.02, raising=False)
    monkeypatch.setattr(gateway_client, "RESTART_WAIT_GRACE_S", grace, raising=False)
    monkeypatch.setattr(gateway_client, "restart_watch_homes", lambda endpoint, own: [home],
                        raising=False)


def test_a_restart_past_the_budget_admits(tmp_path, monkeypatch):
    """The incident in miniature: the owner serves 12.5% past the budget, and the run still admits."""
    from hermes_cli import gateway_client

    home = tmp_path / "owner"
    _write_state(home, "starting")
    _scaled(monkeypatch, gateway_client, home, budget=1.2, grace=5.0)
    admitted = []

    async def run():
        serving = asyncio.Event()
        async with _owner(gateway_client, serving=serving) as server:
            port = server.sockets[0].getsockname()[1]

            def serve_now():
                _write_state(home, "running")
                serving.set()

            # 130 s against a 120 s budget: the owner is not serving for 8.3% of the wait.
            asyncio.get_running_loop().call_later(1.35, serve_now)
            async with _attach(gateway_client, port) as client:
                await client.rpc("session.create",
                                 _timeout=gateway_client.admission_timeout(client, 120.0))
                admitted.append(time.monotonic())

    asyncio.run(run())
    assert admitted, ("a run dispatched into a restart must admit once the owner serves; on the "
                      "pre-fix tree this is a GatewayAdmissionTimeout at the budget")


def test_a_wedged_owner_still_fails_at_the_budget(tmp_path, monkeypatch):
    """The complement: an owner that SAYS it is serving and answers nothing earns no grace."""
    from hermes_cli import gateway_client

    home = tmp_path / "owner"
    _write_state(home, "running")
    _scaled(monkeypatch, gateway_client, home, budget=0.4, grace=5.0)

    async def run():
        serving = asyncio.Event()  # never set: the owner publishes `running` and goes silent
        async with _owner(gateway_client, serving=serving) as server:
            port = server.sockets[0].getsockname()[1]
            async with _attach(gateway_client, port) as client:
                await client.rpc("session.create",
                                 _timeout=gateway_client.admission_timeout(client, 120.0))

    with pytest.raises(gateway_client.GatewayAdmissionTimeout) as caught:
        asyncio.run(run())
    error = caught.value
    assert error.phase == "session" and error.budget == 0.4
    # At the budget, not budget + 5 s: the grace is only ever earned by an owner coming up.
    assert error.elapsed < 2.0, error.elapsed
    assert "restarting" not in str(error)
    assert "no local fallback" in str(error)


def test_a_restart_that_never_finishes_fails_after_the_grace(tmp_path, monkeypatch):
    """Bounded, not unbounded: a boot that never completes is a loss, and it names the restart."""
    from hermes_cli import gateway_client

    home = tmp_path / "owner"
    _write_state(home, "starting")
    _scaled(monkeypatch, gateway_client, home, budget=0.3, grace=0.5)

    async def run():
        serving = asyncio.Event()
        async with _owner(gateway_client, serving=serving) as server:
            port = server.sockets[0].getsockname()[1]
            async with _attach(gateway_client, port) as client:
                await client.rpc("session.create",
                                 _timeout=gateway_client.admission_timeout(client, 120.0))

    with pytest.raises(gateway_client.GatewayAdmissionTimeout) as caught:
        asyncio.run(run())
    line = str(caught.value)
    assert line.startswith("gateway_admission_timeout: ")
    # Both halves of the class: the phase that held the deadline, and the restart that ate the grace.
    assert "phase=session" in line and "no local fallback" in line
    assert "still restarting" in line and "gateway_state=starting" in line
    assert "of restart grace" in line
    # budget 0.3 + grace 0.5 = 0.8; the ceiling is real, and it is not the 120 s cap either.
    assert 0.7 <= caught.value.elapsed < 3.0, caught.value.elapsed


def test_the_reader_only_credits_a_live_owner_that_says_it_is_starting(tmp_path):
    """No invented restarts: a stale record, an absent owner and a serving owner all credit nothing.

    A stale ``gateway_state.json`` from a crashed life is the dangerous case — credit it and every
    dispatch on a host whose gateway is gone is held for the whole grace.
    """
    from hermes_cli import gateway_client

    home = tmp_path / "home"
    home.mkdir()
    assert gateway_client.owner_serving_state([home]) is None  # no record, no owner

    _write_state(home, "starting")
    assert gateway_client.owner_serving_state([home]) == "starting"

    # The outgoing owner's own claim across the exit -> boot gap (SOU-3966): the other half of the
    # window, and the name that must be credited the moment a tree publishes it.
    _write_state(home, "restarting")
    assert gateway_client.owner_serving_state([home]) == "restarting"

    _write_state(home, "running")  # a live owner that says it is serving
    assert gateway_client.owner_serving_state([home]) is None

    _write_state(home, "draining")  # going away, no replacement promised: admit_owner's wait
    assert gateway_client.owner_serving_state([home]) is None

    _write_state(home, "starting", pid=DEAD_PID)
    assert gateway_client.owner_serving_state([home]) is None, "a dead owner's record must not credit"


def test_the_watch_reads_the_home_the_owner_publishes_in():
    """A multiplexer publishes in the home it was LAUNCHED on, not the profile it is serving.

    Reading only the client's own profile home is how the fix would miss every named-profile run —
    the home the ``starting`` state is written to has to be in the set even when it is not the
    client's own, and the endpoint's control home has to win when there is one.
    """
    from hermes_cli import gateway_client

    homes = gateway_client.restart_watch_homes(SimpleNamespace(control_home="/ctl/home"), "/profile/home")
    assert homes[0] == Path("/ctl/home")
    assert Path("/profile/home") in homes
    assert len(homes) == len({str(home) for home in homes}), "the same home must not be sampled twice"


def test_the_grace_is_above_the_measured_restart():
    """The ceiling has to cover the outage it was designed against, but stay a ceiling.

    2026-10-24 measured 131 s of ``starting`` for the replacing owner and 259 s door-to-door from
    the old owner's ``stop()``; the host's own admission budget (300 s) is what the grace is added
    to, not a replacement for it.
    """
    from hermes_cli import gateway_client

    assert gateway_client.RESTART_WAIT_GRACE_S >= 259.0
    assert gateway_client.RESTART_WAIT_GRACE_S <= 600.0, "an unbounded wait trades a dead run for a held one"
    assert gateway_client.RESTART_POLL_INTERVAL_S == gateway_client.BOOT_POLL_INTERVAL_S
