"""The restart window is a named state, not an inference a reader has to make (SOU-3966).

The 2026-10-24T05:44:36Z restart left 4m19s of unservable time after the cron hold released. In that
window ``gateway_state.json`` carried whatever the previous life last wrote, and the incoming boot
re-stamps that inherited record (``run_startup`` writes ``gateway_state: starting`` with
``reload_existing=True``), so ``served_profiles``, ``platforms``, ``serving_decision`` and a fresh
``updated_at`` all survive into a process that is not serving yet. "Fresh + served" is therefore not
evidence of serving, and the control plane's dispatch gate had only the state name to go on.

These tests pin the producer half of the contract:

* ``stop()`` claims ``gateway_state: restarting`` ON DISK before the drain begins;
* the claim is the outgoing life's terminal state — the exit→boot gap is the widest part of the
  window, so it must not be overwritten with ``stopped`` (nobody asked for a stop);
* the replacement's own ``starting``/``running`` publication is what clears it;
* ``restarting`` is neither a serving state nor a drain target, so publishing it needs no
  coordinated consumer change.
"""

import pytest

import gateway.run as gateway_run
import gateway.run_runtime as gateway_run_runtime
from gateway import status as gateway_status
from gateway.restart import RESTARTING_GATEWAY_STATE
from tests.gateway.restart_test_helpers import make_restart_runner


def _use_real_status_writes(runner) -> None:
    """Undo the helper's ``_update_runtime_status`` mock so the terminal write is observable on disk."""
    runner._update_runtime_status = gateway_run.GatewayRunner._update_runtime_status.__get__(
        runner, gateway_run.GatewayRunner
    )


def _on_disk_state():
    return (gateway_status.read_runtime_status() or {}).get("gateway_state")


@pytest.fixture(autouse=True)
def _isolated_runtime(monkeypatch, tmp_path):
    """Keep the stop path's markers in the test's home, and let the status writer finish after."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    yield
    gateway_status.flush_runtime_status(timeout=2.0)


def _drain_spy(seen: dict):
    """``drain_gateway_runtime`` that reads the file at the instant the drain starts."""
    real_drain = gateway_run_runtime.drain_gateway_runtime

    async def spy(runner):
        seen["state"] = _on_disk_state()
        await real_drain(runner)

    return spy


@pytest.mark.asyncio
async def test_the_restart_claim_is_on_disk_before_the_drain_starts(monkeypatch):
    """The acceptance criterion: whoever samples at the drain boundary never sees a serving state."""
    runner, _adapter = make_restart_runner()
    _use_real_status_writes(runner)
    gateway_status.write_runtime_status(gateway_state="running", restart_requested=False)
    assert _on_disk_state() == "running"

    seen = {}
    monkeypatch.setattr(gateway_run_runtime, "drain_gateway_runtime", _drain_spy(seen))
    await runner.stop(restart=True)

    assert seen["state"] == RESTARTING_GATEWAY_STATE, (
        f"the drain began with gateway_state.json reading {seen['state']!r}; the restart window must "
        "be named before anything else can read it"
    )
    assert _on_disk_state() == RESTARTING_GATEWAY_STATE


@pytest.mark.asyncio
async def test_the_claim_survives_the_exit_and_the_replacement_publishes_it_away():
    """A restart's terminal state is ``restarting``, not ``stopped`` — and boot clears it."""
    runner, _adapter = make_restart_runner()
    _use_real_status_writes(runner)
    gateway_status.write_runtime_status(gateway_state="running", restart_requested=False)

    await runner.stop(restart=True)
    assert _on_disk_state() == RESTARTING_GATEWAY_STATE, (
        "a restart's exit→boot gap must stay claimed; writing 'stopped' would make an unattended "
        "restart look like an operator stop"
    )

    # What the replacement does: run_startup stamps "starting" over the inherited record, then the
    # serving path publishes "running". The claim is cleared by the second one, and only by it.
    gateway_status.write_runtime_status(gateway_state="starting", reload_existing=True)
    assert _on_disk_state() == "starting"
    gateway_status.write_runtime_status(gateway_state="running")
    assert _on_disk_state() == "running"


@pytest.mark.asyncio
async def test_a_plain_stop_never_claims_the_restart_window(monkeypatch):
    """A stop is a stop: no restarting claim, and the terminal state stays ``stopped``."""
    runner, _adapter = make_restart_runner()
    _use_real_status_writes(runner)
    gateway_status.write_runtime_status(gateway_state="running", restart_requested=False)

    seen = {}
    monkeypatch.setattr(gateway_run_runtime, "drain_gateway_runtime", _drain_spy(seen))
    await runner.stop()

    assert seen["state"] != RESTARTING_GATEWAY_STATE
    assert _on_disk_state() == "stopped"


def test_a_restarting_record_is_neither_serving_nor_drainable():
    """The whole point of the name: one predicate instead of three freshness heuristics."""
    assert not gateway_status.derive_gateway_drainable(
        gateway_running=True, gateway_state=RESTARTING_GATEWAY_STATE
    )
    assert not gateway_status.derive_gateway_busy(
        gateway_running=True, gateway_state=RESTARTING_GATEWAY_STATE, active_agents=3
    )
    # A live pid is not evidence either — the outgoing owner is alive while it drains.
    assert RESTARTING_GATEWAY_STATE not in gateway_status._DRAINABLE_GATEWAY_STATES


def test_a_legacy_container_file_stranded_at_restarting_still_autostarts(tmp_path):
    """A pre-``desired_state`` container hard-killed mid-restart must not come back down forever."""
    from hermes_cli import container_boot

    (tmp_path / "gateway_state.json").write_text(
        '{"gateway_state": "restarting"}', encoding="utf-8"
    )
    assert container_boot._read_desired_state(tmp_path) == "running"
