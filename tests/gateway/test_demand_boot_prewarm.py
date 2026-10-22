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
# Imported at collection: the FIRST import of ``gateway.run`` runs the checkout recovery probe
# (``hermes_bootstrap._settle_interrupted_update``), which stats ``<gitdir>`` — under the real home
# in a worktree checkout, where the per-test home-I/O guard refuses it (PRX-5963's boot seam is the
# first thing in this file to need it).
from gateway.run import _multiplex_profile_homes


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


@pytest.mark.asyncio
async def test_the_boot_served_set_is_ranked_after_the_read_lands(tmp_path, monkeypatch):
    """PRX-5963: boot freezes the served set from ``profiles_to_serve``, so the rank must be warm.

    Cold, the ceiling falls back to priority-then-alphabetical. On 2026-10-22 that kept an idle
    ``cenergi-web`` and parked ``devops`` (Paperclip agents), and the first warm reconcile tick
    reversed it 2 m 47 s after boot — after a 113 s churn tick that served the demand-bearing tail
    serially — so the served set was 15 of 16 for minutes and three devops runs died in it. Boot has
    to take the set the demand rank makes, not the one a cold snapshot guesses.
    """
    import json
    from datetime import UTC, datetime

    from gateway.run_bootstrap import _ranked_served_set, _start_gateway_early_demand_prewarm
    from hermes_cli.profiles import profiles_to_serve

    root = tmp_path / ".hermes"
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text("gateway:\n  multiplex_max_profiles: 2\n", encoding="utf-8")
    for name in ("alpha", "devops"):
        profile = root / "profiles" / name
        profile.mkdir(parents=True, mode=0o700)
        (profile / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    (root / "gateway_demand.json").write_text(json.dumps({
        "schema": 1, "written_at": datetime.now(UTC).isoformat(), "ttl_seconds": 900,
        "profiles": {"devops": {"agents": 2, "blocked_admission_15m": 0}},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")

    def served():
        return [name for name, _home in profiles_to_serve(True)]

    assert served() == ["default", "alpha"], "cold: priority, then alphabetical — the defect"
    config = SimpleNamespace()
    _start_gateway_early_demand_prewarm(config)
    # What boot reserves and freezes into ``_runtime_profile_homes``.
    assert [name for name, _home in await _ranked_served_set(config)] == ["default", "devops"], \
        "boot must freeze the demand rank, not the cold one"

    # A set already frozen is read back untouched — the reconcile owns it from here.
    config._runtime_profile_homes = (("default", root), ("alpha", root / "profiles" / "alpha"))
    assert await _ranked_served_set(config) == _multiplex_profile_homes(config)
    assert [name for name, _home in await _ranked_served_set(config)] == ["default", "alpha"]


@pytest.mark.asyncio
async def test_a_failed_read_leaves_the_documented_cold_rank(monkeypatch):
    """No task (no ceiling) and a read that dies are both non-fatal: boot proceeds, cold."""
    from gateway.run_bootstrap import _await_early_demand_prewarm

    await _await_early_demand_prewarm(SimpleNamespace())  # no pre-warm task at all

    async def dies():
        raise RuntimeError("manifest read exploded")

    task = asyncio.ensure_future(dies())
    await _await_early_demand_prewarm(SimpleNamespace(_demand_prewarm_task=task))
    assert task.done()


def test_a_decision_built_on_nothing_publishes_demand_cold():
    """A cold rank must never read as 'looked, nobody has work' (SOU-3116's bug class)."""
    from gateway.run_runtime import serving_decision_record

    record = serving_decision_record(SimpleNamespace(_served_profile_homes={"default": Path("/h/default")}))

    assert record["demand_source"] == {"path": None, "state": "cold", "age_seconds": None}
    assert record["ranked"][0]["profile"] == "default"
    assert record["ranked"][0]["demand"]["agents"] is None
