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


class _SilentSocket:
    """A peer that accepts every frame and answers none."""

    async def send(self, _payload):
        return None


def _open_client(gateway_client, *, budget=120.0, remaining=0.0, code_sha=None):
    """A client mid-admission: its budget started ``budget`` seconds ago and has ``remaining`` left."""
    import time as clock

    return gateway_client.GatewayClient(
        _SilentSocket(), admission_deadline=clock.monotonic() + remaining,
        admission_started=clock.monotonic() - (budget - remaining),
        admission_budget=budget, code_sha=code_sha)


def test_a_mid_run_rpc_timeout_is_not_an_admission_timeout():
    """Regression (SOU-2824 review): only a wait the admission budget bounded is starvation.

    Admission verbs are ordinary RPCs issued after ``connect_gateway`` handed the socket over, so
    a run 200 s into its turn still sits inside that context. When such a run's own 30 s verb
    (``prompt.receipt`` after a long turn) times out, the old classification called it
    ``gateway_admission_timeout`` — ``elapsed`` larger than ``budget`` was the tell — and the
    adapter is being told (SOU-2840) to treat that code as a non-retryable verdict. It must stay
    the retryable connection error instead.
    """
    from hermes_cli import gateway_client

    client = _open_client(gateway_client, remaining=-1.0)  # budget long spent, as mid-run

    async def call():
        with pytest.raises(TimeoutError) as caught:
            await client.rpc("prompt.receipt", _timeout=0.01)
        return caught.value

    error = asyncio.run(call())
    assert not isinstance(error, gateway_client.GatewayClientError)


def test_an_admission_bounded_rpc_is_still_classified():
    """The complement: the same client, the same exhausted budget, an admission-bounded wait."""
    from hermes_cli import gateway_client

    client = _open_client(gateway_client, remaining=-1.0, code_sha="4d96afbff73f0aa5")

    async def call():
        with pytest.raises(gateway_client.GatewayAdmissionTimeout) as caught:
            await client.rpc("session.create",
                             _timeout=gateway_client.admission_timeout(client, 120.0))
        return caught.value

    error = asyncio.run(call())
    assert error.reason == "gateway_admission_timeout"
    assert error.budget == 120.0 and error.code_sha == "4d96afbff73f0aa5"
    # The budget, not the 120 s cap or the caller's own verb cap, is what the run paid.
    assert error.elapsed >= 120.0


def test_the_mid_run_shape_the_adapter_predicate_actually_sees(monkeypatch):
    """Full stack: a mid-run verb at its own cap must not carry the ``gateway_admission_timeout:``
    token, and an admission-bounded verb must.

    The two unit tests above pin ``rpc`` and the context's re-raise separately, so neither of them
    observes what the adapter's stderr predicate sees. This is that layer: same silent owner, same
    context, a healthy budget (the 200-s-into-a-turn case), and the caller's default cap expiring.
    """
    from types import SimpleNamespace
    from websockets.asyncio.server import serve
    from hermes_cli import gateway_client

    monkeypatch.setattr(gateway_client, "_session_ticket", lambda *a, **k: "ticket")
    monkeypatch.setattr(gateway_client, "DEFAULT_RPC_TIMEOUT", 0.05)

    async def silent_owner(ws):
        async for _raw in ws:
            pass

    async def attach(port):
        endpoint = SimpleNamespace(api_origin=f"http://127.0.0.1:{port}", profile_id="p",
                                   instance_id="i", code_sha=None)
        async with gateway_client.connect_gateway(endpoint=endpoint) as client:
            await client.rpc("prompt.receipt")

    async def run():
        async with serve(silent_owner, "127.0.0.1", 0,
                         subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL]) as server:
            await attach(server.sockets[0].getsockname()[1])

    with pytest.raises(gateway_client.GatewayClientError) as caught:
        asyncio.run(run())
    assert not isinstance(caught.value, gateway_client.GatewayAdmissionTimeout)
    assert not str(caught.value).startswith("gateway_admission_timeout:")
    # Still a connection failure the retry chain can act on, not a bare TimeoutError traceback.
    assert "no local fallback" in str(caught.value)


def test_a_ticket_mint_timeout_is_admission_starvation(monkeypatch):
    """The mint runs before the socket exists; its timeout used to escape as a bare TimeoutError."""
    from types import SimpleNamespace
    from hermes_cli import gateway_client

    def expired(*_args, **_kwargs):
        raise TimeoutError("mint expired")

    monkeypatch.setattr(gateway_client, "_session_ticket", expired)
    endpoint = SimpleNamespace(api_origin="http://127.0.0.1:1", profile_id="p", instance_id="i",
                               code_sha="4d96afbff73f0aa5")

    async def attach():
        async with gateway_client.connect_gateway(endpoint=endpoint):
            raise AssertionError("an expired mint must not admit")

    with pytest.raises(gateway_client.GatewayAdmissionTimeout) as caught:
        asyncio.run(attach())
    assert caught.value.phase == "session-ticket"
    assert "phase=session-ticket" in str(caught.value)


def test_a_mid_run_body_timeout_is_not_an_admission_timeout(monkeypatch):
    """The handover boundary: after ``yield``, the caller's own TimeoutError keeps its old class."""
    from types import SimpleNamespace
    from websockets.asyncio.server import serve
    from hermes_cli import gateway_client

    monkeypatch.setattr(gateway_client, "_session_ticket", lambda *a, **k: "ticket")

    async def silent_owner(ws):
        async for _raw in ws:
            pass

    async def attach(port):
        endpoint = SimpleNamespace(api_origin=f"http://127.0.0.1:{port}", profile_id="p",
                                   instance_id="i", code_sha=None)
        async with gateway_client.connect_gateway(endpoint=endpoint):
            raise TimeoutError("a mid-run verb, not admission")

    async def run():
        async with serve(silent_owner, "127.0.0.1", 0,
                         subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL]) as server:
            await attach(server.sockets[0].getsockname()[1])

    with pytest.raises(gateway_client.GatewayClientError) as caught:
        asyncio.run(run())
    assert not isinstance(caught.value, gateway_client.GatewayAdmissionTimeout)
    assert "phase=session" in str(caught.value)


def test_the_budget_comes_from_config_yaml_with_a_floor():
    """``gateway.admission_budget_seconds`` raises the budget, never lowers it (PRX-5598).

    The value is host-dependent — a loop that stalls 10-25 s under load 38 (2026-10-19) is survivable
    only by waiting longer — so it has to be tunable from config.yaml; and the floor keeps a typo or
    a 5 s "fix" from killing runs sooner than the class above was designed for.
    """
    from hermes_cli import gateway_client
    from hermes_constants import get_hermes_home

    config = get_hermes_home() / "config.yaml"
    config.write_text("gateway:\n  admission_budget_seconds: 300\n", encoding="utf-8")
    assert gateway_client.admission_budget_seconds() == 300.0
    config.write_text("gateway:\n  admission_budget_seconds: 30\n", encoding="utf-8")
    assert gateway_client.admission_budget_seconds() == gateway_client.GATEWAY_ADMISSION_BUDGET_S
    config.write_text("gateway:\n  admission_budget_seconds: nonsense\n", encoding="utf-8")
    assert gateway_client.admission_budget_seconds() == gateway_client.GATEWAY_ADMISSION_BUDGET_S
    config.unlink()
    assert gateway_client.admission_budget_seconds() == gateway_client.GATEWAY_ADMISSION_BUDGET_S


@pytest.mark.platforms("posix")
def test_a_death_at_the_budget_reports_the_owner_loop_and_the_load():
    """The failure line carries what nobody could see on 2026-10-19: the stall and the load.

    The owner's loopback witness is the loop-tick socket it already serves; sampling it here is a
    millisecond round trip on a healthy loop, and a no-answer (or an old build with no witness) must
    degrade to "no number", never to a healthy-looking one.
    """
    import os
    import shutil
    import socket
    import tempfile
    import threading
    from pathlib import Path

    from hermes_cli import gateway_client

    # AF_UNIX paths cap out near 104 bytes, which a pytest tmp_path is not: same reason the owner
    # keeps a TCP fallback for its witness.
    home = Path(tempfile.mkdtemp(prefix="adm-", dir="/tmp"))
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        (home / "state").mkdir()
        server.bind(str(home / "state" / f"gateway.loop-tick.{os.getpid()}.sock"))
        server.listen(1)

        def answer():
            conn, _address = server.accept()
            with conn:
                conn.recv(1)
                conn.sendall(b"1")

        thread = threading.Thread(target=answer, daemon=True)
        thread.start()
        try:
            evidence = gateway_client.admission_stall_evidence([home])
        finally:
            thread.join(10)
    finally:
        server.close()
        shutil.rmtree(home, ignore_errors=True)

    assert evidence is not None and "owner_loop=" in evidence, evidence
    assert "load1=" in evidence, evidence
    error = gateway_client.GatewayAdmissionTimeout("boom", phase="session", elapsed=120.0,
                                                   evidence=evidence)
    line = str(error)
    assert "budget=120s" in line and "owner_loop=" in line and "load1=" in line

    # No witness anywhere: the loop number is absent rather than fabricated.
    empty = Path(tempfile.mkdtemp(prefix="adm-", dir="/tmp"))
    try:
        assert "owner_loop=" not in (gateway_client.admission_stall_evidence([empty]) or "")
    finally:
        shutil.rmtree(empty, ignore_errors=True)
