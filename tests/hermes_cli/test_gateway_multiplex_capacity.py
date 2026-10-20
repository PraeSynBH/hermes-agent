"""``gateway.multiplex_max_profiles`` — the hard ceiling on profiles per gateway process.

PRA-6524: the per-profile thread baseline (~7 threads per served profile, 286 threads / 43
profiles on a 12-core node) scales with the roster, not with the per-call fan-out, so a host with
more profiles than a node can serve has one option that is actually enforced: stop serving the
excess from that process. These tests pin the decision (which profiles, what a deferred one looks
like to a client) and the boundary that the installed roster is never capped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import gateway_multiplex_capacity as capacity
from hermes_cli.profiles import create_profile, get_profile_dir, profiles_to_serve

CAPACITY = "capacity: 2 profiles per gateway (gateway.multiplex_max_profiles)"


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    capacity.reset_memo()
    yield home
    capacity.reset_memo()


def _serve(*names: str) -> list[tuple[str, Path]]:
    return [("default", Path("/h/default"))] + [(n, Path(f"/h/{n}")) for n in names]


class TestApplyCeiling:
    def test_unset_ceiling_keeps_everyone(self):
        kept, deferred = capacity.apply_ceiling(_serve("a", "b", "c"), None)
        assert [n for n, _ in kept] == ["default", "a", "b", "c"]
        assert deferred == {}

    def test_roster_within_ceiling_is_untouched(self):
        kept, deferred = capacity.apply_ceiling(_serve("a", "b"), 3)
        assert len(kept) == 3
        assert deferred == {}

    def test_ceiling_truncates_and_names_the_reason(self):
        kept, deferred = capacity.apply_ceiling(_serve("a", "b", "c"), 2)
        assert [n for n, _ in kept] == ["default", "a"]
        assert deferred == {"b": CAPACITY, "c": CAPACITY}

    def test_default_is_always_served(self):
        """A ceiling of 1 means "the default profile only", never "no gateway"."""
        kept, deferred = capacity.apply_ceiling(_serve("a", "b"), 1)
        assert [n for n, _ in kept] == ["default"]
        assert set(deferred) == {"a", "b"}

    def test_priority_wins_over_alphabetical_slots(self):
        kept, _deferred = capacity.apply_ceiling(_serve("a", "b", "z"), 2, priority=("z",))
        assert [n for n, _ in kept] == ["default", "z"]

    def test_unknown_priority_names_are_ignored(self):
        """A stale `multiplex_serve_priority` entry must not cost a slot."""
        kept, _deferred = capacity.apply_ceiling(_serve("a", "b"), 2, priority=("gone",))
        assert [n for n, _ in kept] == ["default", "a"]

    def test_decision_is_stable_across_calls(self):
        first, _ = capacity.apply_ceiling(_serve("a", "b", "c", "d"), 3, priority=("c",))
        again, _ = capacity.apply_ceiling(_serve("a", "b", "c", "d"), 3, priority=("c",))
        assert [n for n, _ in first] == [n for n, _ in again] == ["default", "c", "a"]


class TestConfigRead:
    def test_ceiling_and_priority_come_from_the_default_config(self, profile_env):
        (profile_env / "config.yaml").write_text(
            "gateway:\n  multiplex_max_profiles: 4\n  multiplex_serve_priority: [cfo, cto]\n",
            encoding="utf-8")
        assert capacity.profile_ceiling() == 4
        assert capacity.serve_priority() == ("cfo", "cto")

    def test_unset_or_junk_ceiling_means_no_ceiling(self, profile_env):
        assert capacity.profile_ceiling() is None
        (profile_env / "config.yaml").write_text(
            "gateway:\n  multiplex_max_profiles: 0\n", encoding="utf-8")
        assert capacity.profile_ceiling() is None
        (profile_env / "config.yaml").write_text(
            "gateway:\n  multiplex_max_profiles: many\n", encoding="utf-8")
        assert capacity.profile_ceiling() is None

    def test_edit_is_observed_by_the_hot_reader(self, profile_env):
        cfg = profile_env / "config.yaml"
        cfg.write_text("gateway:\n  multiplex_max_profiles: 2\n", encoding="utf-8")
        assert capacity.profile_ceiling() == 2
        cfg.write_text("gateway:\n  multiplex_max_profiles: 12\n", encoding="utf-8")
        assert capacity.profile_ceiling() == 12


class TestServingChokepoint:
    def test_ceiling_applies_to_serving_not_to_the_installed_roster(self, profile_env):
        for name in ("advisor", "cfo", "cto"):
            create_profile(name, no_alias=True)
        (profile_env / "config.yaml").write_text(
            "gateway:\n  multiplex_max_profiles: 2\n  multiplex_serve_priority: [cto]\n",
            encoding="utf-8")

        served = dict(profiles_to_serve(multiplex=True))
        assert set(served) == {"default", "cto"}

        installed = dict(profiles_to_serve(multiplex=True, include_parked=True))
        assert set(installed) == {"default", "advisor", "cfo", "cto"}

    def test_deferred_profiles_are_published_for_the_status_readers(self, profile_env):
        for name in ("advisor", "cfo"):
            create_profile(name, no_alias=True)
        (profile_env / "config.yaml").write_text(
            "gateway:\n  multiplex_max_profiles: 2\n", encoding="utf-8")
        profiles_to_serve(multiplex=True)
        assert set(capacity.deferred_by_capacity()) == {"cfo"}
        assert all(reason.startswith("capacity:") for reason in capacity.deferred_by_capacity().values())

    def test_raising_the_ceiling_serves_the_rest(self, profile_env):
        for name in ("advisor", "cfo"):
            create_profile(name, no_alias=True)
        cfg = profile_env / "config.yaml"
        cfg.write_text("gateway:\n  multiplex_max_profiles: 2\n", encoding="utf-8")
        assert len(profiles_to_serve(multiplex=True)) == 2
        cfg.write_text("gateway:\n  multiplex_max_profiles: 9\n", encoding="utf-8")
        assert len(profiles_to_serve(multiplex=True)) == 3
        assert capacity.deferred_by_capacity() == {}


class TestParkedStatus:
    def test_capacity_deferred_profiles_are_parked_and_unparked(self, profile_env, monkeypatch):
        """A client of a deferred profile gets a terminal verdict (parked), and a raised ceiling
        takes it back out — nobody else's park reason is touched."""
        from gateway import run_runtime

        published: list[dict] = []
        monkeypatch.setattr(run_runtime, "_record_parked_profiles", published.append)

        class Runner:
            session_runtime_descriptor = {"parked_profiles": {"advisor": "session store unusable: disk"}}

        runner = Runner()
        monkeypatch.setattr(capacity, "deferred_by_capacity", lambda: {"cfo": CAPACITY})
        assert run_runtime.sync_capacity_parked(runner) is True
        assert runner.session_runtime_descriptor["parked_profiles"] == {
            "advisor": "session store unusable: disk", "cfo": CAPACITY}
        assert published[-1]["cfo"] == CAPACITY

        monkeypatch.setattr(capacity, "deferred_by_capacity", dict)
        assert run_runtime.sync_capacity_parked(runner) is True
        assert runner.session_runtime_descriptor["parked_profiles"] == {
            "advisor": "session store unusable: disk"}

    def test_nothing_changed_publishes_nothing(self, profile_env, monkeypatch):
        from gateway import run_runtime

        published: list[dict] = []
        monkeypatch.setattr(run_runtime, "_record_parked_profiles", published.append)
        monkeypatch.setattr(capacity, "deferred_by_capacity", lambda: {"cfo": CAPACITY})

        class Runner:
            session_runtime_descriptor = {"parked_profiles": {"cfo": CAPACITY}}

        assert run_runtime.sync_capacity_parked(Runner()) is False
        assert published == []


class _Demand:
    """Minimal duck-typed demand view: what ``apply_ceiling`` needs, nothing more."""

    def __init__(self, names, facts=None, source=None):
        from hermes_cli.gateway_multiplex_demand import DemandFacts
        self.names = set(names)
        self.facts = facts or {}
        self.source = source or {"path": "/h/gateway_demand.json", "state": "ok", "age_seconds": 3}
        self._empty = DemandFacts(agents=0)

    def has(self, name):
        return name in self.names

    def for_profile(self, name):
        return self.facts.get(name, self._empty)

    def demand_source(self):
        return dict(self.source)


def _facts(agents=None, cron_jobs=0, credentials=(), unreadable=()):
    from hermes_cli.gateway_multiplex_demand import DemandFacts
    return DemandFacts(agents=agents, cron_jobs=cron_jobs, credentials=tuple(credentials),
                       unreadable=tuple(unreadable))


class TestDemandRanking:
    """SOU-3075: demand decides, priority is the tiebreak, and the legacy order is untouched."""

    def test_demand_none_reproduces_the_legacy_order(self):
        """The regression guard: every pre-SOU-3075 caller and test keeps its exact order."""
        kept, deferred = capacity.apply_ceiling(_serve("b", "a", "c"), 3, priority=("c",))
        assert [n for n, _ in kept] == ["default", "c", "b"]
        assert deferred == {"a": "capacity: 3 profiles per gateway (gateway.multiplex_max_profiles)"}
        # ... and the legacy reason text is byte-identical (no cost clause without a demand view).
        assert capacity.apply_ceiling(_serve("a", "b"), 2)[1] == {"b": CAPACITY}

    def test_demand_beats_priority(self):
        demand = _Demand({"a"}, facts={"a": _facts(agents=3)})
        kept, deferred = capacity.apply_ceiling(_serve("a", "z"), 2, priority=("z",), demand=demand)
        assert [n for n, _ in kept] == ["default", "a"]
        assert set(deferred) == {"z"}

    def test_priority_orders_within_the_demand_bearing_class(self):
        demand = _Demand({"a", "b", "c"})
        kept, _deferred = capacity.apply_ceiling(_serve("a", "b", "c"), 3, priority=("c", "b"),
                                                 demand=demand)
        assert [n for n, _ in kept] == ["default", "c", "b"]

    def test_a_no_demand_profile_never_outranks_a_demand_bearing_one(self):
        demand = _Demand({"zulu"}, facts={"zulu": _facts(cron_jobs=1)})
        kept, deferred = capacity.apply_ceiling(_serve("alpha", "zulu"), 2, demand=demand)
        assert [n for n, _ in kept] == ["default", "zulu"]
        assert set(deferred) == {"alpha"}

    def test_the_decision_is_deterministic_under_a_shuffled_roster(self):
        demand = _Demand({"alpha", "charlie", "echo"})
        rosters = [["alpha", "charlie", "echo"], ["echo", "alpha", "charlie"],
                   ["charlie", "echo", "alpha"]]
        decisions = {tuple(n for n, _ in capacity.apply_ceiling(_serve(*roster), 3, demand=demand)[0])
                     for roster in rosters}
        assert decisions == {("default", "alpha", "charlie")}

    def test_cap_one_serves_only_the_default_profile(self):
        demand = _Demand({"a"})
        kept, deferred = capacity.apply_ceiling(_serve("a", "b"), 1, demand=demand)
        assert [n for n, _ in kept] == ["default"]
        assert set(deferred) == {"a", "b"}

    def test_a_ceiling_at_or_above_the_roster_defers_nobody(self):
        demand = _Demand({"a"})
        kept, deferred = capacity.apply_ceiling(_serve("a", "b"), 3, demand=demand)
        assert [n for n, _ in kept] == ["default", "a", "b"]
        assert deferred == {}

    def test_a_roster_without_default_is_still_capped(self):
        demand = _Demand({"b"})
        kept, deferred = capacity.apply_ceiling([("a", Path("/h/a")), ("b", Path("/h/b"))], 1,
                                                demand=demand)
        assert [n for n, _ in kept] == ["b"]
        assert set(deferred) == {"a"}


class TestCostClause:
    def test_the_deferral_reason_states_the_trade(self):
        demand = _Demand({"general"}, facts={"general": _facts(agents=12, cron_jobs=0)})
        _kept, deferred = capacity.apply_ceiling(_serve("general", "quiet"), 2,
                                                 demand=demand)
        reason = deferred["quiet"]
        assert reason.startswith("capacity: 2 profiles per gateway (gateway.multiplex_max_profiles)")
        assert "; cost: 0 Paperclip agents, 0 enabled cron jobs, 0 adapter credentials" in reason
        assert len(reason) <= 500

    def test_an_unknown_agent_count_is_never_rendered_as_zero(self):
        """Plan §3.3: with no manifest read, the cost says `unknown`, never `0 agents`."""
        demand = _Demand(set(), facts={"quiet": _facts(agents=None, cron_jobs=3)},
                         source={"path": None, "state": "absent", "age_seconds": None})
        _kept, deferred = capacity.apply_ceiling(_serve("quiet"), 1, demand=demand)
        assert "Paperclip agent demand unknown, 3 enabled cron jobs" in deferred["quiet"]
        assert "0 Paperclip agents" not in deferred["quiet"]

    def test_an_unreadable_source_is_reported_in_the_reason(self):
        demand = _Demand({"broken"}, facts={"broken": _facts(unreadable=("cron",))})
        _kept, deferred = capacity.apply_ceiling(_serve("broken"), 1, demand=demand)
        assert "demand unreadable: cron" in deferred["broken"]

    def test_the_last_decision_is_published_for_the_runtime(self):
        demand = _Demand({"a"}, facts={"a": _facts(agents=1)})
        capacity.apply_ceiling(_serve("a", "b"), 2, demand=demand)
        decision = capacity.last_decision()
        assert decision["ceiling"] == 2
        assert decision["ranked"] == ["default", "a", "b"]
        assert set(decision["deferred"]) == {"b"}
        assert decision["demand_source"]["state"] == "ok"
        # A copy: the published record must not alias the module's cache.
        decision["ranked"].append("mutated")
        assert capacity.last_decision()["ranked"] == ["default", "a", "b"]
