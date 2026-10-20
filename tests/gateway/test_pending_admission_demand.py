"""PRX-5738: the gateway records a pending admission where it already refuses the route.

The demand sources SOU-3075 added are read from disk (cron store, ``.env``, the host manifest). The
sighting that actually distinguishes this outage is taken at the refusal seam, on the path PRX-5725
made fail fast: a client asked for a profile this gateway does not serve, and instead of only
answering "not served" the gateway now remembers the name for one TTL window so the next reconcile
tick can rank it in.

These pin the two seams (the ``/p/<profile>/`` scope refusal and the routed-message refusal) against
the real functions, without importing ``gateway.run`` — both run to completion before any delivery.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway import parked_route_refusal, run_runtime, session_authorities
from hermes_cli import gateway_multiplex_demand as demand

PARK = "capacity: 3 profiles per gateway (gateway.multiplex_max_profiles)"


@pytest.fixture(autouse=True)
def _clean_state():
    demand.reset()
    yield
    demand.reset()


def test_a_refused_profile_scope_records_the_sighting(monkeypatch):
    """An API request under a parked profile's ``/p/<profile>/`` prefix is also a demand sighting."""
    monkeypatch.setattr(session_authorities, "served_profile_name", lambda home: "beta")
    runner = SimpleNamespace(session_runtime_descriptor={"parked_profiles": {"beta": PARK}})

    with pytest.raises(parked_route_refusal.ParkedProfileRefusal):
        run_runtime.refuse_parked_scope(runner)

    assert demand.wanted("beta") is True
    # ... and only for the profile that was asked for.
    assert demand.wanted("alpha") is False


def test_a_refused_route_records_the_sighting():
    """A routed message refused for a parked profile is a client waiting on it, not just a log line."""
    adapter = SimpleNamespace(name="relay", _background_tasks=set())
    source = SimpleNamespace(platform=SimpleNamespace(value="slack"), chat_id="C", message_id="1")

    with parked_route_refusal.retryable_ingress() as scope:
        parked_route_refusal.refuse_parked_route(adapter, source, "beta")

    assert scope.refused is True  # the transport answers 503 and redelivers: nothing was admitted
    assert demand.wanted("beta") is True


def test_a_served_profile_is_not_recorded(monkeypatch):
    """A scope whose profile is served never reaches the refusal, so nothing is pinned."""
    monkeypatch.setattr(session_authorities, "served_profile_name", lambda home: "alpha")
    runner = SimpleNamespace(session_runtime_descriptor={"parked_profiles": {"beta": PARK}})

    run_runtime.refuse_parked_scope(runner)

    assert demand.wanted("alpha") is False
    assert demand.wanted("beta") is False
