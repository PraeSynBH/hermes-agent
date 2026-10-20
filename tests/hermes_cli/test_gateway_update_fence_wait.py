"""The admission path waits out an update fence, and names who held it (PRX-5598).

``_fenced_verdict`` used to answer ``draining``/``update_paused`` the instant an updater marker was
live, and ``connect_gateway`` turned that into a dead run within a second: on 2026-10-19 an update
owned the install 18:54-19:03Z and six runs starting in that window died on it (CTO x3, Release
Engineer, CSO), before the survivors moved on to the admission-budget failures. The contracts
pinned here:

* a live fence is a WAIT: admission polls it within its one budget and admits as soon as it clears;
* a fence that outlives the budget fails with the fence's owner (pid + update run id) in the text;
* the fence's owner is read from the marker's ``run:<id>`` line, which the CLI updater now writes.
"""
from __future__ import annotations

import asyncio

import pytest


def _discovery(gateway_runtime, state, *, reason_code=None, detail=None, endpoint=None):
    return gateway_runtime.GatewayDiscovery(state, reason_code=reason_code, detail=detail,
                                            endpoint=endpoint)


def _owner_server(gateway_client):
    """A websocket owner that answers the upgrade and echoes the protocol (nothing else)."""
    from websockets.asyncio.server import serve

    async def handler(ws):
        async for _raw in ws:
            pass

    return serve(handler, "127.0.0.1", 0, subprotocols=[gateway_client.GATEWAY_WS_PROTOCOL])


def test_admission_waits_out_a_live_update_fence_and_then_admits(monkeypatch):
    """The 2026-10-19 shape: the fence is live for a while and the run admits when it clears.

    Before this, the second ``ensure`` call never happened — the first ``update_paused`` ended the
    run before any session started.
    """
    from types import SimpleNamespace
    from hermes_cli import gateway_client, gateway_runtime

    monkeypatch.setattr(gateway_client, "UPDATE_FENCE_POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(gateway_client, "_session_ticket", lambda *a, **k: "ticket")
    calls, admitted = [], []

    async def run():
        async with _owner_server(gateway_client) as server:
            origin = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            endpoint = SimpleNamespace(api_origin=origin, profile_id="p", instance_id="i",
                                       code_sha="4d96afbff73f0aa5", control_home=None)

            def ensure(home, *, idle_exit=False, timeout=None):
                calls.append(timeout)
                if len(calls) < 3:
                    return _discovery(gateway_runtime, "draining", reason_code="update_paused",
                                      detail=f"update fence live: pid=46862 update_run=r{len(calls)}")
                return _discovery(gateway_runtime, "ready", endpoint=endpoint)

            monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", ensure)
            async with gateway_client.connect_gateway() as client:
                admitted.append(client)

    asyncio.run(run())
    assert admitted, "a fence that clears inside the budget must admit the run"
    assert len(calls) == 3, "the fence is polled, not turned into a verdict at t=0"


def test_a_fence_that_outlives_the_budget_names_its_owner(monkeypatch):
    """A fence longer than the budget is still a refusal — but it says who held it, and for how long.

    Today's line was ``Gateway draining: update_paused``: no pid, no update, nothing to correlate
    with the updater's own receipts.
    """
    from hermes_cli import gateway_client, gateway_runtime

    monkeypatch.setattr(gateway_client, "admission_budget_seconds", lambda: 0.35)
    monkeypatch.setattr(gateway_client, "UPDATE_FENCE_POLL_INTERVAL_S", 0.0)

    def ensure(home, *, idle_exit=False, timeout=None):
        return _discovery(gateway_runtime, "draining", reason_code="update_paused",
                          detail="update fence live: pid=46862 update_run=6cba9ead578e4be0 "
                                 "age=42s marker=/tmp/x/.hermes-update-in-progress")

    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", ensure)

    async def run():
        async with gateway_client.connect_gateway():
            raise AssertionError("a fence that outlives the budget must not admit")

    with pytest.raises(gateway_client.GatewayUnavailableError) as caught:
        asyncio.run(run())
    line = str(caught.value)
    assert line.startswith("Gateway draining: update_paused")
    assert "update_run=6cba9ead578e4be0" in line and "pid=46862" in line
    assert "admission budget 0.35s" in line
    # Not the starvation class: an update holds the install, the owner is not starving.
    assert not isinstance(caught.value, gateway_client.GatewayAdmissionTimeout)


def test_the_fence_owner_is_read_from_the_marker_it_holds(monkeypatch, tmp_path):
    """``hermes update`` writes its correlation id into the marker; the fence quotes it back."""
    from hermes_cli import gateway_runtime, update_lock, update_receipt

    monkeypatch.setattr(update_receipt, "current_correlation_id", lambda: "6cba9ead578e4be0")
    lock = update_lock.UpdateLock(path=tmp_path / update_lock.MARKER_NAME, install_root=None)
    try:
        assert lock.acquire(), "a free install must be claimable"
        fence = gateway_runtime._update_fence([tmp_path])
        assert fence is not None
        assert "pid=" in fence and "update_run=6cba9ead578e4be0" in fence
        assert update_lock.MARKER_NAME in fence
        assert gateway_runtime._update_fenced([tmp_path]) is True
    finally:
        lock.release()
    assert gateway_runtime._update_fence([tmp_path]) is None


def test_no_marker_is_no_fence(tmp_path):
    from hermes_cli import gateway_runtime

    assert gateway_runtime._update_fence([tmp_path]) is None
    assert gateway_runtime._update_fenced([tmp_path]) is False
