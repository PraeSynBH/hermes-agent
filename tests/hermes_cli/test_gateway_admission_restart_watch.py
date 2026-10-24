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
  SOU-3966) and never invents a restart from a stale record or an absent owner;
* the claim is read while its author is DEAD — the exit->boot gap is the one stretch with no live
  process, no lock and no ``starting``, so keying the claim on liveness made the window it was
  written for unreadable, and a run admitted into it died at its budget (PRX-6199: the six
  ``phase=session, elapsed=120.0s`` rows released on the 05:42:26Z hold's release instant);
* ...and it is bounded by its own age (one admission budget), so a gateway that was SIGKILLed
  mid-claim costs a dispatch one budget rather than holding it for the whole grace.

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


def _write_state(home: Path, state: str, *, pid: int | None = None, age_s: float = 0.0,
                 restart_requested: bool | None = None) -> None:
    """Publish the owner's readiness exactly where the gateway publishes it.

    ``age_s`` backdates ``updated_at`` — the field the reader freshness-checks a claim with, and
    the only thing that separates a replacement that is coming from a gateway that is gone.
    """
    home.mkdir(parents=True, exist_ok=True)
    record = {"pid": os.getpid() if pid is None else pid, "gateway_state": state,
              "updated_at": _stamp(time.time() - age_s)}
    if restart_requested is not None:
        record["restart_requested"] = restart_requested
    (home / "gateway_state.json").write_text(json.dumps(record), encoding="utf-8")


def _stamp(epoch: float) -> str:
    """``_utc_now_iso``'s format (``2026-10-24T05:44:33.957Z``): what the gateway writes."""
    from datetime import UTC, datetime
    moment = datetime.fromtimestamp(epoch, UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


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


def test_a_run_admitted_into_the_exit_boot_gap_admits(tmp_path, monkeypatch):
    """The 05:44:36Z incident exactly: no live process on the home, and the claim still there.

    ``stop()`` publishes ``gateway_state: restarting`` before the drain and the exit phase re-stamps
    it just before the process goes away, so the only stretch that has NO live owner — the one that
    held six heartbeats for 131-145 s — is also the one where the record's pid is already dead. The
    reader threw that claim away (a dead pid was treated as "no owner"), so the six runs earned no
    credit at all: ``phase=session, elapsed=120.0s, TimeoutError:`` in their own rows, and the
    replacement published ``running`` 3 s after their budget expired.
    """
    from hermes_cli import gateway_client

    home = tmp_path / "owner"
    # The outgoing owner is gone; its claim is what is left on the home.
    _write_state(home, "restarting", pid=DEAD_PID, restart_requested=True)
    _scaled(monkeypatch, gateway_client, home, budget=0.5, grace=5.0)
    admitted = []

    async def run():
        serving = asyncio.Event()
        async with _owner(gateway_client, serving=serving) as server:
            port = server.sockets[0].getsockname()[1]

            def boot():
                # The replacement's FIRST publication — past this run's whole budget, which is the
                # only way the dead gap alone decides the run: one sample of `starting` before the
                # deadline is enough credit to re-arm, so the gap has to outlast it.
                _write_state(home, "starting")

            def serve_now():
                _write_state(home, "running")
                serving.set()

            loop = asyncio.get_running_loop()
            loop.call_later(gateway_client.admission_budget_seconds() * 1.8, boot)
            loop.call_later(1.6, serve_now)
            async with _attach(gateway_client, port) as client:
                await client.rpc("session.create",
                                 _timeout=gateway_client.admission_timeout(client, 120.0))
                admitted.append(time.monotonic())

    asyncio.run(run())
    assert admitted, ("a run admitted while the home holds only the outgoing owner's claim must "
                      "credit that gap and admit when the replacement serves; on the pre-fix tree "
                      "it dies at its budget")


def test_the_claim_is_credited_only_while_it_is_fresh(tmp_path):
    """A SIGKILLed gateway leaves the same file, so the claim is a WINDOW, never a promise.

    Credit it unconditionally and every dispatch on a host whose gateway is gone is held for the
    whole grace — the failure the liveness check exists to prevent.
    """
    from hermes_cli import gateway_client

    home = tmp_path / "home"
    home.mkdir()
    fresh = gateway_client.RESTART_CLAIM_FRESH_S

    _write_state(home, "restarting", pid=DEAD_PID, restart_requested=True)
    assert gateway_client.owner_serving_state([home]) == "restarting"

    # Same file, written before the window: a restart that is not happening any more.
    _write_state(home, "restarting", pid=DEAD_PID, restart_requested=True, age_s=fresh + 5.0)
    assert gateway_client.owner_serving_state([home]) is None

    # A `restarting` state with no claim behind it is not a restart anyone promised.
    _write_state(home, "restarting", pid=DEAD_PID, restart_requested=False)
    assert gateway_client.owner_serving_state([home]) is None

    # An undated claim cannot be aged, so it cannot be bounded, so it credits nothing.
    (home / "gateway_state.json").write_text(
        json.dumps({"pid": DEAD_PID, "gateway_state": "restarting", "restart_requested": True}),
        encoding="utf-8")
    assert gateway_client.owner_serving_state([home]) is None


def test_the_reader_reads_the_claim_the_outgoing_owner_actually_writes(tmp_path, monkeypatch):
    """Writer/reader contract, not a fixture: the fields are the ones ``stop()`` really persists.

    The claim is only worth reading if the reader understands the file the owner leaves behind — a
    ``restart_requested`` written as anything but ``true``, or a stamp the reader cannot parse, would
    silently credit nothing and this class would stay exactly as it was. So drive the REAL writer
    (``gateway.status.write_runtime_status``, the one ``_claim_restart_window`` calls), then make the
    record's pid dead, which is what the process exit does to it.
    """
    from gateway import status as gateway_status
    from hermes_cli import gateway_client

    home = tmp_path / "home"
    home.mkdir()
    path = home / "gateway_state.json"
    monkeypatch.setattr(gateway_status, "_get_runtime_status_path", lambda: path)
    monkeypatch.setattr(gateway_status, "_runtime_status_state_path", None)
    monkeypatch.setattr(gateway_status, "_runtime_status_state", None)

    assert gateway_status.write_runtime_status(
        gateway_state="restarting", restart_requested=True, wait_timeout=5.0)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["pid"] = DEAD_PID  # the owner is gone; the claim is what is left
    path.write_text(json.dumps(record), encoding="utf-8")

    assert gateway_client.owner_serving_state([home]) == "restarting"
    assert gateway_client.record_age_s(record) is not None


def test_the_claim_window_is_the_admission_budget():
    """The bound is the number the window is already measured in, not a new knob.

    A claim older than one admission budget cannot be covered by any run's remaining budget, so
    crediting it past that point only holds a run for a restart that is not coming — and a gateway
    that died mid-claim must cost a dispatch no more than an absent owner already does.
    """
    from hermes_cli import gateway_client

    assert gateway_client.RESTART_CLAIM_FRESH_S == gateway_client.GATEWAY_ADMISSION_BUDGET_S


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
