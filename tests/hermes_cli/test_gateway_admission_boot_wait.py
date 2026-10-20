"""A cold boot is a WAIT inside the admission budget, not a verdict (PRX-5625).

``admit_owner`` returned the ensure call's verdict verbatim, so a boot in progress
(``state=starting``) became a dead run the moment the per-call cap expired. On this host a cold boot
costs 45.7-62.1 s (``state/gateway.lifecycle.json``: ``create_time`` -> ``start_time``), i.e. longer
than the 60 s ``DEFAULT_ENSURE_TIMEOUT`` that bounded it, so every dispatch inside a boot window was
a guaranteed loss: 21 runs in 24 h in this company, 151 host-wide, durations 69-85 s. The contracts
pinned here:

* a boot that outlives the per-call cap but completes inside the budget ADMITS the run;
* a boot that outlives the whole budget fails as ``gateway_admission_timeout``, and its text names
  the boot (owner still ``starting`` at the budget) rather than the bare old string;
* the boot deadline is no longer shorter than the boot it bounds.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

#: The measured cold-boot cost this host's deadline has to sit above
#: (``~/.hermes/state/gateway.lifecycle.json``: 21:36:08.81Z -> 21:37:10.89Z).
MEASURED_COLD_BOOT_S = 62.09


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


class _Clock:
    """A monotonic clock an ``ensure`` call can advance, so a 62 s boot costs no wall time."""

    def __init__(self, start=1000.0):
        self.now = start

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _Boot:
    """An ``ensure`` that reports a boot in progress for ``boot_s`` of the fake clock.

    Each call spends ``per_call_s`` of clock — the shape of a real ensure hitting its own cap with
    the owner still starting — and the boot is ready on the first call that reaches ``boot_s``.
    """

    def __init__(self, gateway_runtime, clock, *, boot_s, per_call_s, origin=""):
        self.runtime, self.clock = gateway_runtime, clock
        self.boot_s, self.per_call_s, self.origin = boot_s, per_call_s, origin
        self.calls, self.caps = [], []

    def __call__(self, home, *, idle_exit=False, timeout=None):
        self.calls.append(timeout)
        self.caps.append(timeout)
        self.clock.advance(self.per_call_s)
        if self.clock.now - 1000.0 >= self.boot_s:
            endpoint = SimpleNamespace(api_origin=self.origin, profile_id="p",
                                       instance_id="i", code_sha="0b10b9f8155a", control_home=None)
            return _discovery(self.runtime, "ready", endpoint=endpoint)
        return _discovery(self.runtime, "starting", reason_code="deadline")


def test_a_boot_longer_than_the_per_call_cap_still_admits(monkeypatch):
    """The 62.09 s boot measured on this host, against a client that re-observes instead of dying.

    Each ``ensure`` call returns ``starting: deadline`` at a 30 s cap (a cap BELOW the boot, as the
    60 s constant was), and the boot completes at 62.09 s: the run must be admitted.
    """
    from hermes_cli import gateway_client, gateway_runtime

    clock = _Clock()
    monkeypatch.setattr(gateway_client, "time", SimpleNamespace(monotonic=clock.monotonic,
                                                                time=time.time, sleep=time.sleep))
    # raising=False: on the pre-fix tree this constant does not exist, and the test must fail on
    # the behaviour (a boot turned into a verdict), not on a missing name.
    monkeypatch.setattr(gateway_client, "BOOT_POLL_INTERVAL_S", 0.0, raising=False)
    monkeypatch.setattr(gateway_client, "_session_ticket", lambda *a, **k: "ticket")
    monkeypatch.setattr(gateway_client, "admission_budget_seconds", lambda: 120.0)
    admitted = []

    async def run():
        async with _owner_server(gateway_client) as server:
            origin = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            ensure = _Boot(gateway_runtime, clock, boot_s=MEASURED_COLD_BOOT_S,
                           per_call_s=30.0, origin=origin)
            monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", ensure)
            async with gateway_client.connect_gateway() as client:
                admitted.append(client)
            return ensure

    ensure = asyncio.run(run())
    assert admitted, "a boot that finishes inside the budget must admit the run"
    assert len(ensure.calls) >= 3, "each expired ensure is re-observed, not turned into a verdict"
    assert all(cap is not None and cap > 0 for cap in ensure.caps)


def test_a_boot_that_outlives_the_budget_is_named_as_a_boot(monkeypatch):
    """A boot longer than the whole budget is still a loss — but of the classifiable kind.

    The old line was ``Gateway starting: deadline``: it named neither the phase nor the fact that a
    healthy owner was coming up, and every retry chain read it as a broken install.
    """
    from hermes_cli import gateway_client, gateway_runtime

    clock = _Clock()
    monkeypatch.setattr(gateway_client, "time", SimpleNamespace(monotonic=clock.monotonic,
                                                                time=time.time, sleep=time.sleep))
    # raising=False: on the pre-fix tree this constant does not exist, and the test must fail on
    # the behaviour (a boot turned into a verdict), not on a missing name.
    monkeypatch.setattr(gateway_client, "BOOT_POLL_INTERVAL_S", 0.0, raising=False)
    monkeypatch.setattr(gateway_client, "admission_budget_seconds", lambda: 120.0)

    def ensure(home, *, idle_exit=False, timeout=None):
        clock.advance(60.0)
        return _discovery(gateway_runtime, "starting", reason_code="deadline")

    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", ensure)

    async def run():
        async with gateway_client.connect_gateway():
            raise AssertionError("a boot that outlives the budget must not admit")

    with pytest.raises(gateway_client.GatewayAdmissionTimeout) as caught:
        asyncio.run(run())
    line = str(caught.value)
    assert line.startswith("gateway_admission_timeout")
    assert "phase=runtime-boot" in line and "still booting" in line
    assert "starting: deadline" in line, "the boot state itself stays in the text"
    assert caught.value.reason == "gateway_admission_timeout"


def test_a_boot_verdict_carries_the_boot_it_waited_on(monkeypatch):
    """A boot that outlives the budget says what the boot cost, from ``gateway.lifecycle.json``.

    Otherwise "still booting" is unfalsifiable for the reader: on this host a real boot is 26.9-62.1 s
    and the deadline is 120 s, so a boot verdict needs its own ``create_time -> start_time`` (AC2 of
    PRX-5725).
    """
    import json
    from pathlib import Path

    from hermes_cli import gateway_client, gateway_runtime
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    (home / "state").mkdir(parents=True, exist_ok=True)
    (home / "state" / "gateway.lifecycle.json").write_text(json.dumps({
        "phase": "running", "create_time": 1792519657.06,
        "started_at": "2026-10-20T18:08:03.918183+00:00"}), encoding="utf-8")
    clock = _Clock()
    monkeypatch.setattr(gateway_client, "time", SimpleNamespace(monotonic=clock.monotonic,
                                                                time=time.time, sleep=time.sleep))
    monkeypatch.setattr(gateway_client, "BOOT_POLL_INTERVAL_S", 0.0, raising=False)
    monkeypatch.setattr(gateway_client, "admission_budget_seconds", lambda: 120.0)

    def ensure(home, *, idle_exit=False, timeout=None):
        clock.advance(60.0)
        return _discovery(gateway_runtime, "starting", reason_code="deadline")

    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", ensure)

    async def run():
        async with gateway_client.connect_gateway():
            raise AssertionError("a boot that outlives the budget must not admit")

    with pytest.raises(gateway_client.GatewayAdmissionTimeout) as caught:
        asyncio.run(run())
    line = str(caught.value)
    assert "phase=runtime-boot" in line and "still booting" in line
    assert "create_time 2026-10-20T18:07:37+00:00" in line, line
    assert "start_time 2026-10-20T18:08:03.918183+00:00" in line and "= 26.9s" in line, line


def test_the_boot_deadline_sits_above_the_measured_boot():
    """The constant that bounds a boot must be longer than the boot it bounds.

    60 s was not: the measured cold boot on this host is 62.09 s, so the cap could only ever report
    a healthy boot as a failure. Any future lowering of this constant below a real boot re-opens
    PRX-5625.
    """
    from hermes_cli import gateway_runtime

    assert gateway_runtime.DEFAULT_ENSURE_TIMEOUT > MEASURED_COLD_BOOT_S
    # ...and it is still a cap, not a hang: it spends from the one admission budget.
    from hermes_cli import gateway_client

    assert gateway_runtime.DEFAULT_ENSURE_TIMEOUT <= gateway_client.GATEWAY_ADMISSION_BUDGET_S
