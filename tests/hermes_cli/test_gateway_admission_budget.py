"""One admission budget, one classifiable failure (SOU-2824).

Before this, admission was five independent caps (ensure 60 s, ticket 30 s, ws-open 30 s,
``runtime.describe`` 30 s, ``session.create`` 120 s). A starved owner does not fail any single one
of them until they have all run out in turn: on 2026-10-19 35 fleet admissions survived 181 s and
then died as a generic "no local fallback", so the retry chain could not tell starvation from a
broken install and re-paid it — 3 x 181 s — per run. The contracts pinned here:

* every phase waits at most what the ONE budget has left (``admission_timeout``);
* an expired budget fails as ``GatewayAdmissionTimeout`` with a stable ``reason`` code, the phase
  that held it and the owner's build — not the unclassified connection error;
* a refused/absent owner stays a plain connection error, so a retry still rides out a boot.
"""
from __future__ import annotations

import asyncio

import pytest


class _Client:
    """A connection that carries only the budget ``connect_gateway`` gives it."""

    def __init__(self, admission_deadline=None):
        self.admission_deadline = admission_deadline


def test_a_phase_waits_at_most_what_the_budget_has_left(monkeypatch):
    from hermes_cli import gateway_client

    now = [1000.0]
    monkeypatch.setattr(gateway_client.time, "monotonic", lambda: now[0])
    budget = _Client(admission_deadline=now[0] + 40.0)

    assert gateway_client.admission_timeout(budget, 120.0) == 40.0
    # A phase that finishes early hands its remainder to the next one instead of queueing a cap.
    now[0] += 10.0
    assert gateway_client.admission_timeout(budget, 120.0) == 30.0
    # A cap below the remainder still wins: the budget is a ceiling, never an extension.
    assert gateway_client.admission_timeout(budget, 5.0) == 5.0
    # Spent: the next phase must not wait at all (an RPC at 0 fails immediately).
    now[0] += 30.0
    assert gateway_client.admission_timeout(budget, 120.0) == 0.0
    # A peer with no budget (test double, embedded server) is unchanged.
    assert gateway_client.admission_timeout(_Client(), 120.0) == 120.0


def test_a_starved_owner_fails_as_a_classifiable_admission_timeout(monkeypatch):
    """The 2026-10-19 shape: ticket minted, socket open, then the owner stops answering.

    The failure must name the class (a retry chain keys on it), the phase that held the deadline
    and the build of the owner that was holding it — the field that turned this investigation into
    an ``lsof`` dig.
    """
    from types import SimpleNamespace
    from websockets.asyncio.server import serve
    from hermes_cli import gateway_client

    endpoint = SimpleNamespace(api_origin="http://127.0.0.1:1", profile_id="p", instance_id="i",
                               code_sha="4d96afbff73f0aa5e4237957855e8800cb519033")
    monkeypatch.setattr(gateway_client, "_session_ticket", lambda *a, **k: "ticket")
    monkeypatch.setattr(gateway_client, "GATEWAY_ADMISSION_BUDGET_S", 0.6)

    async def silent_owner(ws):
        async for _raw in ws:  # accept the upgrade, read the RPC, never answer it
            pass

    async def attach():
        async with gateway_client.connect_gateway(endpoint=endpoint) as client:
            # What every caller does first: its admission verb, bounded by the shared budget.
            await client.rpc("runtime.describe",
                             _timeout=gateway_client.admission_timeout(client, 120.0))

    async def run():
        # Echo the owner's protocol: the real owner selects it from the ticket subprotocols.
        async with serve(silent_owner, "127.0.0.1", 0,
                         subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL]) as server:
            endpoint.api_origin = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            with pytest.raises(gateway_client.GatewayAdmissionTimeout) as caught:
                await asyncio.wait_for(attach(), 10)
            return caught.value

    error = asyncio.run(run())
    assert error.reason == "gateway_admission_timeout"
    assert str(error).startswith("gateway_admission_timeout: ")
    assert "phase=session" in str(error) and "no local fallback" in str(error)
    assert error.phase == "session" and error.code_sha == "4d96afbff73f0aa5e4237957855e8800cb519033"
    assert error.code_sha[:12] in str(error)
    # The budget, not the 120 s cap it replaced, is what the run paid.
    assert error.elapsed < 5 and error.budget == 0.6
    assert error.budget < gateway_client.GATEWAY_STARTUP_RPC_DEADLINE_S


def test_a_refused_owner_is_not_an_admission_timeout(monkeypatch):
    """An absent owner (closed port) must stay a plain connection error: a retry rides out a boot.

    Classifying it as starvation would tell the retry chain to give up on the one failure it can
    actually recover from.
    """
    import socket
    from types import SimpleNamespace
    from hermes_cli import gateway_client

    monkeypatch.setattr(gateway_client, "_session_ticket", lambda *a, **k: "ticket")
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    # Closed again before the dial: nothing answers, so the kernel refuses it outright.
    endpoint = SimpleNamespace(api_origin=f"http://127.0.0.1:{port}", profile_id="p",
                               instance_id="i", code_sha=None)

    async def attach():
        async with gateway_client.connect_gateway(endpoint=endpoint):
            raise AssertionError("a refused owner must not admit")

    with pytest.raises(gateway_client.GatewayClientError) as caught:
        asyncio.run(attach())
    assert not isinstance(caught.value, gateway_client.GatewayAdmissionTimeout)
    assert "no local fallback" in str(caught.value)


def test_the_owner_build_is_named_when_the_budget_expires():
    """The expiry must carry the owner's commit — the queryable identity a dead run lacked."""
    from hermes_cli.gateway_client import GatewayAdmissionTimeout

    error = GatewayAdmissionTimeout("boom", phase="session", elapsed=181.2,
                                    code_sha="4d96afbff73f0aa5e4237957855e8800cb519033")
    assert "owner=4d96afbff73f" in str(error) and "budget=120s" in str(error)
    assert (error.phase, error.reason) == ("session", "gateway_admission_timeout")
