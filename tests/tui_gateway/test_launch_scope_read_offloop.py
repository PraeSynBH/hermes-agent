"""Entering the launch-profile scope from a gateway ticker must not read files on the loop.

``async_launch_profile_scope_if_multiplexed`` is the "no routed profile here" seam for bodies with
no turn on the stack — the 15 s loop-wakeup watcher, the notification watchers. It used to claim
"no I/O of its own", but entering the scope reads the launch profile's ``.env`` (every entry, no
cache skip) and ``config.yaml`` plus the managed-dir/secret-source stats behind both. PRX-5524's
blocked-loop dump sampled the loop inside exactly that read:
``run_goals.py::_loop_wakeup_watcher`` -> ``async_launch_profile_scope_if_multiplexed`` ->
``load_yaml_file_readonly``.

Contract: the maps are built on a worker thread; the ContextVar binding stays on the loop and the
body still sees the launch profile's own credential, so the hop cannot split the scope.
"""
from __future__ import annotations

import threading

import pytest

from agent.secret_scope import get_secret, is_multiplex_active, set_multiplex_active
from tui_gateway import launch_profile_policy as lpp


@pytest.fixture
def launch_home(tmp_path, monkeypatch):
    home = tmp_path / "launch_home"
    home.mkdir()
    (home / ".env").write_text("LAUNCH_OFFLOOP_PROBE=from-launch-home\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "terminal:\n  backend: local\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(lpp, "_snapshot", None)
    set_multiplex_active(True)
    try:
        yield home
    finally:
        set_multiplex_active(False)


@pytest.mark.asyncio
async def test_the_launch_scope_is_built_on_a_worker_thread(launch_home, monkeypatch):
    from agent import secret_scope

    seen = []
    real_build = secret_scope.build_profile_secret_scope

    def recording_build(home):
        seen.append(threading.get_ident())
        return real_build(home)

    monkeypatch.setattr(secret_scope, "build_profile_secret_scope", recording_build)
    loop_thread = threading.get_ident()

    async with lpp.async_launch_profile_scope_if_multiplexed():
        assert get_secret("LAUNCH_OFFLOOP_PROBE") == "from-launch-home"
        assert threading.get_ident() == loop_thread

    assert len(seen) == 1
    assert seen[0] != loop_thread


@pytest.mark.asyncio
async def test_the_body_stays_inside_the_launched_scope_after_the_hop(launch_home):
    from hermes_constants import get_hermes_home_override

    async with lpp.async_launch_profile_scope_if_multiplexed():
        assert get_hermes_home_override() == str(launch_home)
        assert get_secret("LAUNCH_OFFLOOP_PROBE") == "from-launch-home"

    assert get_hermes_home_override() is None


def test_a_prepared_scope_binds_without_reading_anything(launch_home, monkeypatch):
    """The sync context manager honours the worker's maps: binding is ContextVar-only."""
    from agent import secret_scope
    from tools import terminal_scope

    def refuse(home, **kwargs):  # a second read here would mean the hop was wasted
        raise AssertionError("the bound scope re-read the profile")

    monkeypatch.setattr(secret_scope, "build_profile_secret_scope", refuse)
    monkeypatch.setattr(terminal_scope, "build_profile_terminal_scope", refuse)
    prepared = ({"LAUNCH_OFFLOOP_PROBE": "prebuilt"}, {"TERMINAL_ENV": "local"}, None)

    with lpp.launch_profile_runtime_scope(launch_home, prepared):
        assert get_secret("LAUNCH_OFFLOOP_PROBE") == "prebuilt"


def test_a_failed_terminal_build_still_installs_the_refusal_scope(launch_home):
    """A build that failed off-loop installs the same refusal a failed inline build did."""
    from hermes_constants import get_hermes_home_override
    from tools.terminal_scope import TerminalPolicyRefusal, get_terminal_scope

    prepared = ({"LAUNCH_OFFLOOP_PROBE": "prebuilt"}, None, "cannot parse config.yaml")
    with lpp.launch_profile_runtime_scope(launch_home, prepared):
        assert get_hermes_home_override() == str(launch_home)
        assert get_secret("LAUNCH_OFFLOOP_PROBE") == "prebuilt"
        assert isinstance(get_terminal_scope(), TerminalPolicyRefusal)

    assert is_multiplex_active()
