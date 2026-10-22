"""Runtime discovery must distinguish absence from an unsafe/live owner."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest


@asynccontextmanager
async def control_peer(home: Path, payload: dict):
    from gateway.control_socket import GatewayControlServer

    home.mkdir(mode=0o700)
    server = GatewayControlServer(home, verb_handlers={"identify": lambda: payload})
    assert await server.start()
    pointer = home / "gateway.sock.path"
    if pointer.exists():
        pointer.chmod(0o600)
    try:
        yield server
    finally:
        await server.stop()


@pytest.mark.platforms("linux")
def test_live_discovery_does_not_treat_an_unusable_owner_as_absent(tmp_path):
    from hermes_cli.gateway_runtime import discover_gateway_endpoint

    async def probe():
        home = tmp_path / "profile"
        identity = str(home.resolve())
        payload = {
            "protocol": 1,
            "runtime_protocol": 1,
            "instance_id": "fixture-instance",
            "authority_epoch": 1,
            "hermes_home": identity,
            "state": "ready",
            "api_origin": "http://127.0.0.1:41817",
            "supervisor": "none",
            "served_profiles": [{"profile_id": identity, "home": identity}],
            "capabilities": ["session-authority-v1"],
        }
        absent = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=1)
        assert absent.state == "absent"
        async with control_peer(home, payload):
            ready = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=1)
            assert ready.state == "ready"
            assert ready.endpoint.profile_id == identity
            assert ready.endpoint.instance_id == "fixture-instance"
            assert ready.endpoint.api_origin == "http://127.0.0.1:41817"
            for key, value, expected in [
                ("runtime_protocol", 99, "incompatible"),
                ("runtime_protocol", True, "incompatible"),
                ("served_profiles", [], "inaccessible"),
                ("state", "draining", "draining"),
                ("api_origin", "http://secret:credential@127.0.0.1:41817", "inaccessible"),
                ("api_origin", "http://example.com:41817", "inaccessible"),
                ("authority_epoch", True, "inaccessible"),
                ("capabilities", [], "incompatible"),
            ]:
                old = payload[key]
                payload[key] = value
                observed = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=1)
                assert observed.state == expected, (key, observed)
                assert "credential" not in repr(observed)
                payload[key] = old
        # A forged pointer is not permission to connect somewhere else or spawn.
        (home / "gateway.sock.path").write_text(str(tmp_path / "foreign.sock"))
        (home / "gateway.sock.path").chmod(0o666)
        refused = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=1)
        assert refused.state == "inaccessible"

    asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_fifo_control_pointer_is_rejected_without_a_writer(tmp_path):
    import os
    import subprocess
    import sys

    home = tmp_path / 'fifo-profile'
    home.mkdir(mode=0o700)
    os.mkfifo(home / 'gateway.sock.path', 0o600)
    code = (
        'import sys; from pathlib import Path; '
        'from hermes_cli.gateway_runtime import discover_gateway_endpoint; '
        'r = discover_gateway_endpoint(Path(sys.argv[1]), timeout=0.1); '
        'print(r.state, r.reason_code)'
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', code, str(home)],
            cwd=Path(__file__).resolve().parents[2], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=3, check=False
        )
    except subprocess.TimeoutExpired:
        pytest.fail('discovery blocked on a FIFO instead of rejecting its file type')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'inaccessible unsafe_control_pointer'


def test_start_decisions_do_not_turn_uncertainty_into_another_owner():
    from gateway.runtime_contract import RuntimeObservation, next_start_action

    cases = [
        (RuntimeObservation("ready"), "attach"),
        (RuntimeObservation("starting"), "wait-owner"),
        (RuntimeObservation("draining"), "wait-owner"),
        (RuntimeObservation("incompatible"), "reject-version"),
        (RuntimeObservation("inaccessible"), "reject-access"),
        (RuntimeObservation("conflict"), "reject-conflict"),
        (RuntimeObservation("absent"), "spawn-unmanaged"),
        (RuntimeObservation("absent", installed_service=True), "start-service"),
        (RuntimeObservation("absent", service_may_start=True), "wait-service"),
        (RuntimeObservation("ready", update_paused=True), "wait-update"),
    ]
    for observation, expected in cases:
        assert next_start_action(observation) == expected
    with pytest.raises(ValueError, match="unknown runtime state"):
        next_start_action(RuntimeObservation("invalid"))


def test_a_capacity_park_is_a_provisional_wait_carrying_the_cost(tmp_path):
    """A profile over the ceiling is a WAIT, not a verdict — and its cost survives the read intact.

    ``ensure_gateway_runtime`` reports the deferral as ``profile_serving`` (retryable, PRX-5963): the
    rank is rebuilt on every reconcile tick and a waiting client pins a slot (PRX-5738), so reading
    it as terminal killed runs for a profile the host served seconds later. The reason published
    under ``parked_profiles`` still has to arrive intact, cost clause included (SOU-3075 §5), so the
    client can report the trade if the wait runs out. A park that is NOT a ceiling deferral (unusable
    store, home owned elsewhere) keeps the terminal verdict.
    """
    from hermes_cli.gateway_runtime import _parked_by_multiplexer

    reason = ("capacity: 15 profiles per gateway (gateway.multiplex_max_profiles) — this profile is "
              "deferred; cost: 12 Paperclip agents, 3 enabled cron jobs, 0 adapter credentials")
    discovery = _parked_by_multiplexer({"parked_profiles": {"devops": reason}},
                                      tmp_path / "devops")

    assert discovery.state == "inaccessible"
    assert discovery.reason_code == "profile_serving"
    assert discovery.retry_after_s is not None
    assert "cost: 12 Paperclip agents, 3 enabled cron jobs, 0 adapter credentials" in discovery.detail

    settled = _parked_by_multiplexer({"parked_profiles": {"devops": "session store unusable: locked"}},
                                     tmp_path / "devops")
    assert (settled.state, settled.reason_code) == ("inaccessible", "profile_parked")


def test_a_profile_absent_from_parked_profiles_keeps_the_ordinary_verdict(tmp_path):
    from hermes_cli.gateway_runtime import _parked_by_multiplexer

    assert _parked_by_multiplexer({"parked_profiles": {"other": "capacity: 1 profiles"}},
                                 tmp_path / "devops") is None
