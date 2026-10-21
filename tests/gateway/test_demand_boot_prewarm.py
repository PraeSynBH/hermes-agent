"""SOU-3125: the demand snapshot is pre-warmed beside boot, never read on the boot/loop thread.

``gateway_multiplex_demand.snapshot()`` used to read on demand, so the first ``profiles_to_serve``
of a boot paid 15–22 s of one-time plugin discovery — inside the restart SOU-3075 lands, and on the
event loop for every route that resolved through the ceiling. Now only ``refresh`` reads, and the
boot path starts that read as a task before the runner exists so the cost overlaps startup. These
pin the task, its thread, and the cold decision that is published when nothing has read yet.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_multiplex_capacity as capacity
from hermes_cli import gateway_multiplex_demand as demand


@pytest.fixture(autouse=True)
def _reset_memos():
    capacity.reset_memo()
    demand.reset()
    yield
    capacity.reset_memo()
    demand.reset()


def _capped_home(home: Path, cap: int = 4) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"gateway:\n  multiplex_max_profiles: {cap}\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_the_boot_prewarm_reads_in_a_worker_thread(monkeypatch):
    from hermes_constants import get_hermes_home
    from gateway.run_bootstrap import _start_gateway_early_demand_prewarm

    _capped_home(get_hermes_home())
    threads: list[threading.Thread] = []
    real_refresh = demand.refresh

    def recording_refresh():
        threads.append(threading.current_thread())
        return real_refresh()

    monkeypatch.setattr(demand, "refresh", recording_refresh)
    config = SimpleNamespace()

    _start_gateway_early_demand_prewarm(config)

    task = config._demand_prewarm_task
    assert task is not None, "a host with a ceiling must pre-warm demand"
    # Scheduled, not run: the calling (boot / event-loop) thread never pays the read itself.
    assert threads == []
    assert demand.snapshot().manifest_state == demand.MANIFEST_COLD

    await asyncio.wait_for(task, 30)

    assert threads and threads[0] is not threading.main_thread()
    assert demand.snapshot().manifest_state != demand.MANIFEST_COLD


@pytest.mark.asyncio
async def test_a_host_without_a_ceiling_starts_no_read(monkeypatch):
    from hermes_constants import get_hermes_home
    from gateway.run_bootstrap import _start_gateway_early_demand_prewarm

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("gateway: {}\n", encoding="utf-8")
    started: list[object] = []
    monkeypatch.setattr(demand, "refresh", lambda: started.append(True))
    config = SimpleNamespace()

    _start_gateway_early_demand_prewarm(config)

    assert getattr(config, "_demand_prewarm_task", None) is None
    assert started == []


def test_a_decision_built_on_nothing_publishes_demand_cold():
    """A cold rank must never read as 'looked, nobody has work' (SOU-3116's bug class)."""
    from gateway.run_runtime import serving_decision_record

    record = serving_decision_record(SimpleNamespace(_served_profile_homes={"default": Path("/h/default")}))

    assert record["demand_source"] == {"path": None, "state": "cold", "age_seconds": None}
    assert record["ranked"][0]["profile"] == "default"
    assert record["ranked"][0]["demand"]["agents"] is None
