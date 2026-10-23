"""PRA-6796: a pending WAKE outranks assigned work, and recency, in the ceiling's rank.

The 2026-10-22 outage: ``release`` held two Paperclip agents and a blocked admission, and it sat
outside a 16-slot ceiling behind profiles whose only claim was an assigned agent or an enabled cron
job. Both were class 0, the tie broke on the profile NAME, ``release`` was alphabetically last, and
every wake for it was refused — 15 in a row. Assigned work says a profile *holds cards*; a pending
wake says a run was dispatched and cannot be admitted. These pin the second one as its own class,
above the first, and pin the sentence a deferral publishes so the refusal is not silent.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, UTC
from pathlib import Path

import pytest

from hermes_cli import gateway_multiplex_capacity as capacity
from hermes_cli import gateway_multiplex_demand as demand
from hermes_cli.profiles import create_profile, get_profile_dir


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    capacity.reset_memo()
    demand.reset()
    yield home
    capacity.reset_memo()
    demand.reset()


def _profile(name: str) -> Path:
    create_profile(name, no_alias=True)
    return get_profile_dir(name)


def _write_jobs(home: Path, enabled: int) -> None:
    (home / "cron").mkdir(exist_ok=True)
    jobs = [{"id": f"on-{i}", "enabled": True} for i in range(enabled)]
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")


def _touch_store(home: Path, *, age_seconds: float) -> None:
    store = home / "state.db"
    store.write_bytes(b"")
    when = time.time() - age_seconds
    os.utime(store, (when, when))


def _write_manifest(default_home: Path, profiles: dict) -> Path:
    path = default_home / "gateway_demand.json"
    written = datetime.now(UTC) - timedelta(seconds=5)
    path.write_text(json.dumps({
        "schema": 1,
        "written_at": written.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "test",
        "ttl_seconds": 900,
        "profiles": profiles,
    }), encoding="utf-8")
    return path


def _served(names_and_homes, snapshot, cap: int):
    """``(kept names, deferred map)`` for one roster at *cap*, through the real decision."""
    return capacity.apply_ceiling(names_and_homes, cap, (), demand=snapshot)


class TestWakeClass:
    def test_a_blocked_admission_is_served_ahead_of_assigned_agents(self, profile_env):
        """The outage case: one blocked wake, one profile that merely holds more agents."""
        _profile("release")
        idle = _profile("agent")
        # `agent` holds MORE assigned agents and an enabled cron job — and still loses the slot.
        _write_jobs(idle, enabled=3)
        _write_manifest(profile_env, {
            "release": {"agents": 2, "blocked_admission_15m": 1},
            "agent": {"agents": 11, "blocked_admission_15m": 0},
        })
        snapshot = demand.refresh()

        kept, deferred = _served([("release", Path("/r")), ("agent", Path("/a"))], snapshot, 1)

        assert [name for name, _home in kept] == ["release"]
        assert "agent" in deferred

    def test_a_queued_run_is_a_pending_wake(self, profile_env):
        """``busy`` in the manifest is a run already dispatched: the strongest disk signal there is."""
        _profile("product")
        idle = _profile("ceo")
        _write_jobs(idle, enabled=7)
        _write_manifest(profile_env, {
            "product": {"agents": 1, "busy": 1},
            "ceo": {"agents": 14, "busy": 0},
        })
        snapshot = demand.refresh()

        kept, deferred = _served([("product", Path("/p")), ("ceo", Path("/c"))], snapshot, 1)

        assert [name for name, _home in kept] == ["product"]
        assert "ceo" in deferred

    def test_a_pending_wake_outranks_a_profile_a_human_is_using(self, profile_env):
        _profile("release")
        in_use = _profile("designer")
        _touch_store(in_use, age_seconds=60)  # used a minute ago: class "recency"
        _write_manifest(profile_env, {"release": {"agents": 0, "blocked_admission_15m": 2}})
        snapshot = demand.refresh()

        kept, _deferred = _served([("release", Path("/r")), ("designer", in_use)], snapshot, 1)

        assert [name for name, _home in kept] == ["release"]

    def test_the_pending_run_count_is_not_a_weight(self, profile_env):
        """One queued run ties with five: a label, never an ordering (SOU-3075's rule)."""
        _profile("one")
        _profile("five")
        _write_manifest(profile_env, {"one": {"agents": 1, "busy": 1},
                                      "five": {"agents": 1, "busy": 5}})
        snapshot = demand.refresh()

        assert snapshot.rank_class("one") == snapshot.rank_class("five") == 0

    def test_a_manifest_without_the_busy_field_is_not_a_wake(self, profile_env):
        """A collector that predates ``busy`` still reads: absent is no queue, never malformed."""
        _profile("release")
        _write_manifest(profile_env, {"release": {"agents": 2, "blocked_admission_15m": 0}})

        snapshot = demand.refresh()

        assert snapshot.manifest_state == demand.MANIFEST_OK
        assert snapshot.for_profile("release").pending_runs == 0
        assert snapshot.for_profile("release").has_pending_wake() is False
        assert snapshot.rank_class("release") == 1  # assigned work


class TestQueuedRunIsAWake:
    """PRX-6000: a DISPATCHED-but-unstarted run is a waiting client, not assignment.

    ``release`` was evicted 98 s after the 2026-10-22 boot while it held two queued runs. The
    collector reported only ``running`` runs, so its facts read as "2 agents, nothing waiting"
    (class 1) and a rival pinned by a live sighting (class 0) took the slot. The rank could not see
    the client that was actually waiting for it.
    """

    def test_a_queued_run_ranks_with_a_blocked_admission(self, profile_env):
        _profile("release")
        _write_manifest(profile_env, {"release": {"agents": 2, "queued": 2}})

        facts = demand.refresh().for_profile("release")

        assert facts.queued_runs == 2
        assert facts.has_pending_wake() is True
        assert facts.rank_class() == 0
        assert facts.work_signals() == ("wake", "agents")
        assert "2 Paperclip runs queued for this profile" in demand.wake_signal(facts)

    def test_the_incident_tie_goes_to_the_priority_list(self, profile_env):
        """The 2026-10-22 13:28Z decision, replayed: both sides have a waiting client.

        ``cmo`` was pinned by a live pending-admission sighting (class 0) and took the slot from
        ``release``, whose two queued runs were invisible (class 1). With the queued count in the
        rank both are class 0, so the operator's priority list decides — and it puts ``release``
        ahead of ``cmo``. This is also the acceptance shape: a boot that served ``release`` cannot
        lose it to a sighting pin as long as release holds a waiting run.
        """
        _profile("release")
        _profile("cmo")
        _write_manifest(profile_env, {
            "release": {"agents": 2, "queued": 2},
            "cmo": {"agents": 3},
        })
        snapshot = demand.refresh()
        demand.note_wanted("cmo")  # a client is blocked on cmo right now (class 0)

        assert snapshot.rank_class("release") == snapshot.rank_class("cmo") == 0
        kept, deferred = capacity.apply_ceiling(
            [("release", Path("/r")), ("cmo", Path("/c"))], 1, ("release", "cmo"), demand=snapshot)

        assert [name for name, _home in kept] == ["release"]
        assert "cmo" in deferred

    def test_a_negative_queued_count_is_malformed(self, profile_env):
        """A count is never a reason to rank a profile on a guess: refuse the read instead."""
        _profile("release")
        _write_manifest(profile_env, {"release": {"agents": 2, "queued": -1}})

        snapshot = demand.refresh()

        assert snapshot.manifest_state == demand.MANIFEST_MALFORMED
        assert snapshot.for_profile("release").agents is None  # unknown, never a fabricated zero

    def test_a_manifest_without_the_queued_field_still_reads(self, profile_env):
        """Older collectors keep working: no ``queued`` is no dispatched queue, not malformed."""
        _profile("release")
        _write_manifest(profile_env, {"release": {"agents": 2, "busy": 1}})

        facts = demand.refresh().for_profile("release")

        assert facts.queued_runs == 0
        assert facts.rank_class() == 0  # class 0 via ``busy``, unchanged from PRA-6796
        assert demand.refresh().manifest_state == demand.MANIFEST_OK


class TestPublishedWake:
    def test_the_record_carries_the_pending_run_count_and_signal(self, profile_env):
        _profile("release")
        _write_manifest(profile_env, {"release": {"agents": 2, "busy": 1}})

        record = demand.refresh().for_profile("release").as_record()

        assert record["pending_runs"] == 1
        assert record["work"] == ["wake", "agents"]

    def test_a_blocked_wake_is_named_in_the_reason(self, profile_env):
        """The reason is what the refusal books onto the owning card: it must not read as idle."""
        lost = capacity.cost_clause(
            demand.DemandSnapshot(facts={"release": demand.DemandFacts(
                agents=2, blocked_admission_15m=3)}),
            "release")

        assert "a Paperclip wake is blocked on this profile (3 refused admissions in 15 min)" in lost

    def test_a_queued_wake_is_named_in_the_reason(self, profile_env):
        lost = capacity.cost_clause(
            demand.DemandSnapshot(facts={"product": demand.DemandFacts(agents=1, pending_runs=2)}),
            "product")

        assert "2 Paperclip runs queued for this profile" in lost
