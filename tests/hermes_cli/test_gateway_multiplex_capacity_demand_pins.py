"""PRX-5738: a profile somebody is waiting on is never the one the ceiling defers.

SOU-3075 taught the ceiling to rank by demand read from disk — enabled cron jobs, adapter
credentials, the host's demand manifest. Those all say a profile *could* hold work. The 2026-10-20
outage is the gap they leave: the busiest profile sat deferred while its runs failed admission, and
nothing on disk said "a client is blocked on this profile RIGHT NOW".

These pin the live signal (``note_wanted``), its TTL, the INFO line that reports the trade, and the
one thing that must not regress: with no work anywhere the pre-demand order stands.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from hermes_cli import gateway_multiplex_capacity as capacity
from hermes_cli import gateway_multiplex_demand as demand
from hermes_cli.profiles import create_profile, profiles_to_serve
from hermes_cli.gateway_multiplex_demand import DemandFacts

WANTED = "a client is waiting (pending admission)"


class _LiveView:
    """A demand view backed by the live in-process sightings, on an injectable clock."""

    def __init__(self, clock=None):
        self.clock = clock

    def has(self, name):
        return self.wanted(name) or self.for_profile(name).has_demand()

    def wanted(self, name):
        return demand.wanted(name, clock=self.clock)

    def for_profile(self, name):
        return DemandFacts(agents=0)

    def demand_source(self):
        return {"path": None, "state": "absent", "age_seconds": None}


def _serve(*names: str) -> list[tuple[str, Path]]:
    return [("default", Path("/h/default"))] + [(n, Path(f"/h/{n}")) for n in names]


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


@pytest.fixture(autouse=True)
def _clean_state():
    capacity.reset_memo()
    demand.reset()
    yield
    capacity.reset_memo()
    demand.reset()


class TestPendingAdmissionRanking:
    def test_an_idle_profile_pays_for_the_one_a_client_is_waiting_on(self):
        """Cap 3, four profiles, the alphabetically-LAST has a waiting client: an idle one goes."""
        demand.note_wanted("zulu")
        view = demand.snapshot()
        # Nothing on disk says zulu holds work — the waiting client IS its work.
        assert view.for_profile("zulu").has_demand() is False
        assert view.has("zulu") is True

        kept, deferred = capacity.apply_ceiling(_serve("alpha", "mike", "zulu"), 3, demand=view)

        assert [name for name, _home in kept] == ["default", "zulu", "alpha"]
        assert set(deferred) == {"mike"}

    def test_no_work_anywhere_keeps_todays_priority_and_alphabetical_order(self):
        """The regression guard: an empty predicate view must not move a single slot."""
        view = demand.snapshot()
        with_demand, deferred_with = capacity.apply_ceiling(
            _serve("a", "b", "z"), 2, priority=("z",), demand=view)
        legacy, deferred_legacy = capacity.apply_ceiling(_serve("a", "b", "z"), 2, priority=("z",))

        assert [name for name, _ in with_demand] == [name for name, _ in legacy] == ["default", "z"]
        assert set(deferred_with) == set(deferred_legacy) == {"a", "b"}

    def test_priority_still_orders_within_the_demand_class(self):
        demand.note_wanted("b")
        view = _LiveView()
        kept, _deferred = capacity.apply_ceiling(_serve("a", "b", "c"), 3, priority=("c",),
                                                 demand=view)
        # b is the only demand-bearing one; c is pinned by the priority list, so it follows b.
        assert [name for name, _ in kept] == ["default", "b", "c"]


class TestSightingExpiry:
    def test_an_expired_sighting_frees_the_slot_again(self):
        """A client that asked once and walked away must not hold a slot forever."""
        demand.note_wanted("zulu", ttl=30.0, clock=lambda: 1000.0)

        fresh, _ = capacity.apply_ceiling(_serve("alpha", "mike", "zulu"), 3,
                                          demand=_LiveView(clock=lambda: 1010.0))
        assert [name for name, _ in fresh] == ["default", "zulu", "alpha"]

        stale, deferred = capacity.apply_ceiling(_serve("alpha", "mike", "zulu"), 3,
                                                 demand=_LiveView(clock=lambda: 1031.0))
        assert [name for name, _ in stale] == ["default", "alpha", "mike"]
        assert set(deferred) == {"zulu"}
        assert demand.wanted("zulu", clock=lambda: 1031.0) is False

    def test_a_refreshed_sighting_stays_pinned(self):
        demand.note_wanted("zulu", ttl=30.0, clock=lambda: 1000.0)
        demand.note_wanted("zulu", ttl=30.0, clock=lambda: 1020.0)  # another client asked again
        assert demand.wanted("zulu", clock=lambda: 1045.0) is True
        assert demand.wanted("zulu", clock=lambda: 1051.0) is False


class TestPublishedDecision:
    def test_the_pin_and_its_cost_are_logged_at_info(self, caplog):
        demand.note_wanted("zulu")
        _kept, deferred = capacity.apply_ceiling(_serve("alpha", "mike", "zulu"), 3,
                                                demand=demand.snapshot())
        with caplog.at_level(logging.INFO, logger=capacity.__name__):
            capacity.remember_deferred(deferred)

        pins = [record.getMessage() for record in caplog.records
                if "Pinned by demand" in record.getMessage()]
        assert pins, caplog.text
        assert "'zulu' (a client is waiting (pending admission)) kept a slot" in pins[-1]
        assert "'mike' deferred instead" in pins[-1]

    def test_no_pin_line_when_demand_changed_nothing(self, caplog):
        """A roster whose rank is unchanged must not grow a line per tick."""
        view = demand.snapshot()
        _kept, deferred = capacity.apply_ceiling(_serve("alpha", "mike"), 2, demand=view)
        with caplog.at_level(logging.INFO, logger=capacity.__name__):
            capacity.remember_deferred(deferred)
        assert "Pinned by demand" not in caplog.text

    def test_a_deferred_profile_with_a_waiting_client_says_so(self):
        """The published reason carries the cost, so an operator sees the client, not just a cap."""
        demand.note_wanted("zulu")
        _kept, deferred = capacity.apply_ceiling(_serve("alpha", "zulu"), 1,
                                                demand=demand.snapshot())
        assert deferred["zulu"].startswith("capacity: 1 profiles per gateway")
        assert WANTED in deferred["zulu"]

    def test_the_decision_names_the_pins_for_the_publisher(self):
        demand.note_wanted("zulu")
        capacity.apply_ceiling(_serve("alpha", "mike", "zulu"), 3, demand=demand.snapshot())
        decision = capacity.last_decision() or {}
        assert decision.get("pinned") == {"zulu": WANTED}
        assert decision.get("deferred_for_demand") == ["mike"]
        # A copy: the published record must not alias the module's cache.
        decision["pinned"]["mutated"] = "yes"
        assert (capacity.last_decision() or {}).get("pinned") == {"zulu": WANTED}


class TestServingChokepoint:
    def test_the_waiting_profile_is_served_and_every_deferral_is_still_published(self, profile_env):
        """PRX-5734's contract survives: every deferred profile is named, with a capacity reason."""
        for name in ("advisor", "cfo", "mike"):
            create_profile(name, no_alias=True)
        (profile_env / "config.yaml").write_text(
            "gateway:\n  multiplex_max_profiles: 2\n", encoding="utf-8")
        demand.note_wanted("mike")

        served = dict(profiles_to_serve(multiplex=True))

        assert set(served) == {"default", "mike"}
        deferred = capacity.deferred_by_capacity()
        assert set(deferred) == {"advisor", "cfo"}
        assert all(reason.startswith("capacity:") for reason in deferred.values())
        assert (capacity.last_decision() or {}).get("pinned") == {"mike": WANTED}
