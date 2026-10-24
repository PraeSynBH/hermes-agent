"""The classic CLI's admission verbs share the run's admission budget (PRX-6206).

2026-10-24 11:48:09Z: three heartbeats started 96-101 s after a replacing owner's control socket
listened and died at 11:49:50-52Z (102.5-103.9 s of run) with

    Hermes couldn't start the model connection: Gateway did not answer session.create.

``session.create`` was never on the admission path that SOU-3960/PRX-6199 fixed. The classic CLI
opens its session through ``GatewaySessionStore`` -> ``GatewayConnection.call``, and that path
capped every verb at ``DEFAULT_RPC_TIMEOUT`` (30 s) with no admission deadline behind it: the
owner was still booting (its loop stalling admission in ~10 s slices), ``runtime.describe`` burned
one 30 s cap and ``session.create`` the next, and the run exited 1 — 3 of the 20 runs dispatched
into that window, while the siblings that crossed the same threshold later survived.

The contracts pinned here:

* an admission verb issued while the owner publishes ``starting`` rides the owner's own word and
  the run's admission budget instead of dying at the flat transport cap (the fix's whole point,
  red before it);
* a WEDGED owner — alive, publishing ``running``, answering nothing — is still a bounded loss, and
  it is classified as admission starvation (``GatewayAdmissionTimeout``) rather than a mystery
  "did not answer session.create";
* a verb issued AFTER this viewer has a session keeps the flat cap: ``session.resume`` mid-run
  (``/reload``, ``/rollback``, a switch) must not inherit a deadline that expired with admission.

Durations are SCALED: the transport cap is 0.2 s and the admission budget 1.2 s, i.e. the incident's
shape (a run killed at a cap that is ~4% of the budget its owner was still working inside).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _stamp(epoch: float) -> str:
    from datetime import UTC, datetime

    moment = datetime.fromtimestamp(epoch, UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _write_state(home: Path, state: str) -> None:
    """Publish the owner's readiness where the gateway publishes it (``gateway_state.json``)."""
    home.mkdir(parents=True, exist_ok=True)
    record = {"pid": os.getpid(), "gateway_state": state, "updated_at": _stamp(time.time())}
    (home / "gateway_state.json").write_text(json.dumps(record), encoding="utf-8")


def _owner(gateway_client, *, serving):
    """An owner that accepts the upgrade and answers an in-band verb only once it is serving."""

    async def handler(ws):
        async def answer(frame):
            await serving.wait()
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": frame["id"],
                                      "result": {"stored_session_id": "s-1", "info": {}}}))

        tasks = []
        try:
            async for raw in ws:
                tasks.append(asyncio.create_task(answer(json.loads(raw))))
        finally:
            for task in tasks:
                task.cancel()

    from websockets.asyncio.server import serve

    return serve(handler, "127.0.0.1", 0, subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL])


def _scaled(monkeypatch, gateway_client, home, *, budget, grace=5.0):
    """Point the client at one temp home, a scaled clock and a scaled transport cap."""
    from hermes_cli import gateway_connection

    monkeypatch.setattr(gateway_client, "admission_budget_seconds", lambda: budget)
    monkeypatch.setattr(gateway_client, "_session_ticket", lambda *a, **k: "ticket")
    monkeypatch.setattr(gateway_client, "RESTART_POLL_INTERVAL_S", 0.02)
    monkeypatch.setattr(gateway_client, "RESTART_WAIT_GRACE_S", grace)
    monkeypatch.setattr(gateway_client, "restart_watch_homes", lambda endpoint, own: [home])
    # The flat transport cap that killed the three runs: 30 s against a 120 s admission budget.
    monkeypatch.setattr(gateway_connection, "DEFAULT_RPC_TIMEOUT", 0.5)


def _connection(gateway_client, port):
    from hermes_cli.gateway_connection import GatewayConnection

    endpoint = SimpleNamespace(api_origin=f"http://127.0.0.1:{port}", profile_id="p",
                               instance_id="i", code_sha=None, control_home=None)
    return GatewayConnection(connect=lambda **kwargs: gateway_client.connect_gateway(endpoint=endpoint))


@pytest.mark.parametrize("verb", ["session.create", "session.resume"])
def test_an_admission_verb_rides_the_budget_while_the_owner_boots(tmp_path, monkeypatch, verb):
    """The incident in miniature: the owner serves past the transport cap, inside the budget."""
    from hermes_cli import gateway_client
    from hermes_cli import gateway_connection

    home = tmp_path / "owner"
    _write_state(home, "starting")
    _scaled(monkeypatch, gateway_client, home, budget=8.0)
    admitted = []

    async def run():
        serving = asyncio.Event()
        async with _owner(gateway_client, serving=serving) as server:
            port = server.sockets[0].getsockname()[1]

            def serve_now():
                _write_state(home, "running")
                serving.set()

            # 2 s: several transport caps past T=0, well inside the budget.
            asyncio.get_running_loop().call_later(2.0, serve_now)
            conn = _connection(gateway_client, port)
            try:
                # The classic CLI calls from its own thread; here that must be a thread too, or this
                # loop (which is the fake owner's) stops answering while it waits.
                admitted.append(await asyncio.to_thread(conn.call, verb, request_id="r-1", source="cli"))
            finally:
                conn.close()

    asyncio.run(run())
    assert admitted and admitted[0]["stored_session_id"] == "s-1", (
        "an admission verb issued while the owner publishes `starting` must wait the owner out; "
        "on the pre-fix tree this is `Gateway did not answer session.create` at "
        f"{gateway_connection.DEFAULT_RPC_TIMEOUT!r}s")


def test_a_wedged_owner_is_a_classified_loss_at_the_budget(tmp_path, monkeypatch):
    """The complement: an owner that SAYS it is serving and answers nothing earns no grace."""
    from hermes_cli import gateway_client

    home = tmp_path / "owner"
    _write_state(home, "running")  # a live owner that says it is serving
    _scaled(monkeypatch, gateway_client, home, budget=1.0, grace=5.0)

    async def run():
        serving = asyncio.Event()  # never set: silent while publishing `running`
        async with _owner(gateway_client, serving=serving) as server:
            port = server.sockets[0].getsockname()[1]
            conn = _connection(gateway_client, port)
            try:
                await asyncio.to_thread(conn.call, "session.create", request_id="r-1", source="cli")
            finally:
                conn.close()

    with pytest.raises(gateway_client.GatewayAdmissionTimeout) as caught:
        asyncio.run(run())
    line = str(caught.value)
    assert caught.value.phase == "session" and caught.value.budget == 1.0
    assert "did not answer" not in line, line  # the flat transport cap's unclassifiable copy
    assert "no local fallback" in line
    assert caught.value.elapsed < 3.0, caught.value.elapsed


def test_a_mid_run_verb_keeps_the_flat_cap():
    """``session.resume`` after admission is an ordinary request, not a second admission.

    Its admission deadline expired with the session it admitted, so a wait derived from that
    deadline would fail at once — a run 200 s into its turn would lose ``/reload``/``/rollback``
    to a clock that stopped.
    """
    import asyncio

    from hermes_cli import gateway_connection

    class _Client:
        def __init__(self):
            self.events = asyncio.Queue()
            self.budgets = []

        async def rpc(self, method, *, _timeout=None, **params):
            self.budgets.append(_timeout)
            return {"stored_session_id": "s-1"}

    from contextlib import asynccontextmanager

    client = _Client()
    client.admission_deadline = time.monotonic() - 60.0  # admission ended a minute ago

    @asynccontextmanager
    async def dial(**kwargs):
        yield client

    conn = gateway_connection.GatewayConnection(connect=dial)
    try:
        conn.start(5)
        conn._admitted = True  # this viewer already owns a session
        assert conn.call("session.resume", session_id="s-1") == {"stored_session_id": "s-1"}
    finally:
        conn.close()
    assert client.budgets == [gateway_connection.DEFAULT_RPC_TIMEOUT], client.budgets


def test_the_outer_guard_clears_the_client_s_widest_admission_wait():
    """The connection's own net must never fire before the budget the client may re-arm on.

    ``GatewayClient`` re-arms an admission wait with the seconds the owner spent ``starting``,
    bounded by ``RESTART_WAIT_GRACE_S``; a guard of ``budget + 5`` would kill a run mid-restart at
    the budget and make the grace unobservable from this path.
    """
    from hermes_cli import gateway_client, gateway_connection

    for budget in (0.3, 120.0):
        guard = gateway_connection.answer_guard(budget, admission=True)
        assert guard >= budget + gateway_client.RESTART_WAIT_GRACE_S, (budget, guard)
        flat_guard = gateway_connection.answer_guard(budget, admission=False)
        assert flat_guard < budget + gateway_client.RESTART_WAIT_GRACE_S, (budget, flat_guard)
