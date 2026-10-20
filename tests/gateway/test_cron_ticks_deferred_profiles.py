"""PRX-5717: ``gateway.multiplex_max_profiles`` caps what a gateway SERVES, never what it TICKS.

The ceiling exists to stop the per-profile thread baseline scaling with the roster, so a profile
past it gets no adapters and no session authority. Its ``cron/`` store is a different thing: one
host gateway process ticks every profile's store, and deriving the tick set from the capped
reservation left the deferred profiles' jobs in a store no ticker visited — 17 enabled jobs across
five profiles silently stopped firing on a 12-slot host.

These pin the contract end to end: the tick set keeps the deferred profile, and the real multiplex
ticker visits its store every cycle.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.run import _cron_profile_gate, _cron_tick_profile_homes
from hermes_cli import gateway_multiplex_capacity as capacity
from hermes_cli.profiles import create_profile, profiles_to_serve, profiles_to_tick

# ``gateway.run`` is imported at collection time on purpose: its ``hermes_bootstrap`` import settles a
# killed ``hermes update`` against the checkout's git dir, and on a checkout that lives inside the real
# hermes home (this host's ``~/.hermes/hermes-agent``) the autouse home-I/O guard refuses that read.
# Collection runs before the guard installs; inside a test it would fail for reasons unrelated to us.


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    capacity.reset_memo()
    yield home
    capacity.reset_memo()


def _capped_host(profile_env, *names: str, cap: int = 2) -> SimpleNamespace:
    """A host whose config defers every named profile but the first past *cap*, as boot sees it."""
    for name in names:
        create_profile(name, no_alias=True)
    (profile_env / "config.yaml").write_text(
        f"gateway:\n  multiplex_max_profiles: {cap}\n", encoding="utf-8")
    served = profiles_to_serve(multiplex=True)
    # The reservation boot freezes is exactly the capped serving enumeration.
    return SimpleNamespace(multiplex_profiles=True, _runtime_profile_homes=tuple(served))


def test_the_tick_set_keeps_profiles_the_ceiling_defers(profile_env):
    config = _capped_host(profile_env, "advisor", "cfo", "cto")

    served = {name for name, _home in config._runtime_profile_homes}
    assert served == {"default", "advisor"}, "the fixture must actually defer two profiles"
    assert {name for name, _home in profiles_to_tick(multiplex=True)} == {
        "default", "advisor", "cfo", "cto"}
    assert {name for name, _home in _cron_tick_profile_homes(config)} == {
        "default", "advisor", "cfo", "cto"}


def test_the_tick_set_still_honours_a_park_or_standalone_opt_out(profile_env):
    """Only the CAPACITY decision is dropped from the tick set: an operator's park marker and a
    standalone profile's own gateway still keep this ticker off that store."""
    config = _capped_host(profile_env, "advisor", "cfo", "cto", "solo")
    profiles = {name: home for name, home in profiles_to_serve(
        multiplex=True, include_parked=True, include_standalone=True)}
    (profiles["cfo"] / "gateway.parked").write_text("", encoding="utf-8")
    (profiles["solo"] / "config.yaml").write_text("gateway:\n  standalone: true\n", encoding="utf-8")

    ticked = {name for name, _home in _cron_tick_profile_homes(config)}
    assert "cfo" not in ticked and "solo" not in ticked
    assert {"default", "advisor", "cto"} <= ticked


class _OneCycle:
    """Stop event that ends the ticker loop after its first completed cycle."""

    def __init__(self):
        self._waits = 0

    def is_set(self):
        return self._waits >= 1

    def wait(self, timeout=None):
        self._waits += 1
        return self.is_set()


def test_the_multiplex_ticker_visits_the_deferred_profiles_store(profile_env, monkeypatch):
    """The real ticker loop, driven one cycle with the gateway's own tick enumeration: the deferred
    profile's home is one of the stores it ticks. Before PRX-5717 it was not, every cycle."""
    import cron.scheduler_provider as provider_module
    from hermes_constants import get_hermes_home

    config = _capped_host(profile_env, "advisor", "cfo")
    homes = dict(profiles_to_tick(multiplex=True))
    ticked: list[Path] = []

    def fake_tick(*_args, **_kwargs):
        # ``_profile_cron_scope`` binds the ticked home for the whole tick.
        ticked.append(Path(get_hermes_home()).resolve())
        return 0

    monkeypatch.setattr("cron.scheduler.tick", fake_tick)
    provider = provider_module.InProcessCronScheduler()
    monkeypatch.setattr(provider, "recover_interrupted", lambda: 0)
    provider.start(_OneCycle(), interval=60, profile_homes=lambda: _cron_tick_profile_homes(config),
                   profile_gate=_cron_profile_gate)

    assert homes["cfo"].resolve() in ticked, (
        f"the ceiling must not stop the ticker visiting a deferred store: {ticked}")
    for name in ("default", "advisor"):
        assert homes[name].resolve() in ticked, f"'{name}' stopped being ticked: {ticked}"
