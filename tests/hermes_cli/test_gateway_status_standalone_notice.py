"""Regression for PRX-5944: the status notice reports config opt-outs, not capacity deferrals.

``print_standalone_by_config`` computed ``roster - served`` from
``profiles_to_serve(multiplex=True)``. Once that call began honouring
``gateway.multiplex_max_profiles`` — and ranking the ceiling edge by live demand — the
difference became "every profile this gateway deferred": on the live host the line printed
all 27 capacity-deferred profiles as "standalone by config" and named ``devops``, which the
same ``hermes gateway status`` reported as served one block above. The line has to read the
``gateway.standalone`` flag instead, which is the fact it claims to report.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _make_profile(home: Path, name: str, *, standalone: bool = False) -> None:
    """A live named profile: the config.yaml is the identity marker as well as the config."""
    directory = home / "profiles" / name
    directory.mkdir(parents=True)
    body = "gateway:\n  standalone: true\n" if standalone else "{}\n"
    (directory / "config.yaml").write_text(body, encoding="utf-8")


def _notice() -> str:
    from hermes_cli.gateway_status_notes import print_standalone_by_config

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_standalone_by_config()
    return buf.getvalue()


def test_notice_names_the_profile_that_opted_out_and_not_the_deferred_ones(profile_env):
    (profile_env / "config.yaml").write_text(
        "gateway:\n  multiplex_max_profiles: 2\n", encoding="utf-8"
    )
    _make_profile(profile_env, "opted-out", standalone=True)
    for name in ("alpha", "beta", "gamma"):
        _make_profile(profile_env, name)

    from hermes_cli.profiles import profiles_to_serve

    served = {name for name, _home in profiles_to_serve(True)}
    # The premise: this host really does defer profiles (default + 2 of alpha/beta/gamma).
    assert len(served) == 2
    assert "opted-out" not in served

    out = _notice()
    assert "opted-out" in out
    for deferred in sorted({"alpha", "beta", "gamma"} - served):
        assert deferred not in out


def test_capacity_deferral_alone_prints_no_standalone_line(profile_env):
    """The regression: on a capped host where nobody opted out, the line must not appear."""
    (profile_env / "config.yaml").write_text(
        "gateway:\n  multiplex_max_profiles: 2\n", encoding="utf-8"
    )
    for name in ("alpha", "beta", "gamma"):
        _make_profile(profile_env, name)

    from hermes_cli.profiles import profiles_to_serve

    assert len(profiles_to_serve(True)) == 2  # profiles ARE deferred by the ceiling here
    assert _notice() == ""


def test_uncapped_host_with_no_standalone_profile_prints_nothing(profile_env):
    for name in ("alpha", "beta"):
        _make_profile(profile_env, name)
    assert _notice() == ""
