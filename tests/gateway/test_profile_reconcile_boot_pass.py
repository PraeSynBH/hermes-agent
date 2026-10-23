"""PRX-5989: the profile-reconcile watcher's first pass is a boot cost, not a tick.

The first pass fans out over every served home (each ``config.yaml`` load + shell-hook re-sync)
and, when multiplexing, runs the first full profile diff. Left inside the 30 s tick it measured
p95 49-113 s on a 17-profile boot while the passes right after were 40-2,800 ms, so the work moved
to startup (``prime_profile_reconcile``) and the watcher's first tick is the warm one.
"""
import asyncio

import pytest

from agent import shell_hooks
from gateway import loop_tick_timing
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.run_startup import GatewayStartupMixin
from tests.waiting import async_wait_until

HOOK_CFG = "hooks:\n  pre_tool_call:\n    - command: {command}\n      fail_closed: true\n"


def _hooks_in(home) -> list:
    """Shell-hook callbacks registered for *home* (one gateway process, one plugin manager)."""
    from hermes_cli.plugins import get_plugin_manager
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(home))
    try:
        return [cb for cb in get_plugin_manager()._hooks.get("pre_tool_call", [])
                if getattr(cb, "shell_hook_key", None)]
    finally:
        reset_hermes_home_override(token)


def _approve(home, command: str, approve: bool) -> None:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(home))
    try:
        shell_hooks._record_approval("pre_tool_call", command) if approve else shell_hooks.revoke(command)
    finally:
        reset_hermes_home_override(token)


def _runner(monkeypatch, tmp_path, *, multiplex: bool, homes: dict):
    """A GatewayRunner stand-in with the reconcile bookkeeping the watcher and the boot pass read."""
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex)
    runner._running = True
    runner._served_profile_homes = dict(homes)
    runner._hook_consent_signatures = None
    runner._served_profile_signatures = {}
    return runner


def _two_homes(tmp_path, monkeypatch):
    """Launch home + one served profile, each with a consented ``hooks:`` command."""
    launch = tmp_path / ".hermes"
    profile = launch / "profiles" / "guarded"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    command = str(tmp_path / "guard.sh")
    for home in (launch, profile):
        (home / "config.yaml").write_text(HOOK_CFG.format(command=command), encoding="utf-8")
        _approve(home, command, True)
    return launch, profile, command


@pytest.mark.asyncio
async def test_boot_pass_syncs_every_home_once_and_the_tick_never_repeats_it(tmp_path, monkeypatch):
    """The fan-out happens at startup; the first watcher tick finds nothing changed."""
    shell_hooks.reset_for_tests()
    launch, profile, command = _two_homes(tmp_path, monkeypatch)
    runner = _runner(monkeypatch, tmp_path, multiplex=True, homes={"guarded": profile})

    diffs = []

    async def _reconcile(**kwargs):
        diffs.append(kwargs.get("reason"))
        return {"added": [], "removed": [], "rescanned": []}

    runner.reconcile_served_profiles = _reconcile

    try:
        assert _hooks_in(launch) == [] and _hooks_in(profile) == []

        await runner.prime_profile_reconcile()

        assert diffs == ["boot"], "the boot pass owns the first full profile diff"
        assert len(_hooks_in(launch)) == 1 and len(_hooks_in(profile)) == 1, (
            "the first pass must still sync every home: consent may have landed after startup "
            "registered")

        # From here the tick must carry no config load and no hook sync: those signatures were
        # recorded by the boot pass. A second pass over unchanged files proves it.
        syncs = []
        real_sync = shell_hooks.sync_from_config

        def _counting_sync(load):
            syncs.append(1)
            real_sync(load)

        monkeypatch.setattr(shell_hooks, "sync_from_config", _counting_sync)
        watcher = asyncio.create_task(runner._profile_reconcile_watcher(interval=0.01))
        try:
            await async_wait_until(lambda: len(diffs) >= 3, timeout=10, what="warm watcher ticks")
            assert syncs == [], "the first pass moved to boot; the warm tick re-synced a home"

            # ...and the tick is still live: a consent REVOKE seen by a later pass unregisters.
            _approve(profile, command, False)
            await async_wait_until(lambda: _hooks_in(profile) == [], timeout=10,
                                   what="hook unregistered after a later pass")
            assert syncs, "a changed allowlist must still reach the hook registry"
        finally:
            runner._running = False
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
    finally:
        shell_hooks.reset_for_tests()


@pytest.mark.asyncio
async def test_the_boot_pass_is_counted_as_boot_cost_not_as_a_watcher_tick(tmp_path, monkeypatch):
    """A regression in the first pass must stay visible as boot cost, never hide in ``boot_max``."""
    shell_hooks.reset_for_tests()
    _launch, profile, _command = _two_homes(tmp_path, monkeypatch)
    runner = _runner(monkeypatch, tmp_path, multiplex=True, homes={"guarded": profile})

    async def _reconcile(**_kwargs):
        return {"added": [], "removed": [], "rescanned": []}

    runner.reconcile_served_profiles = _reconcile
    loop_tick_timing.reset()
    try:
        await runner.prime_profile_reconcile()
        snapshot = loop_tick_timing.snapshot()
        assert snapshot["profile_reconcile_boot"]["count"] == 1
        assert "profile_reconcile" not in snapshot, "the boot pass is not a watcher tick"
    finally:
        loop_tick_timing.reset()
        shell_hooks.reset_for_tests()


@pytest.mark.asyncio
async def test_a_failing_boot_pass_never_aborts_the_boot(monkeypatch):
    """Both halves are best-effort: a warm that raises leaves the old tick to do the work."""
    primed = []

    class _Runner:
        async def prime_profile_reconcile(self):
            raise RuntimeError("store down")

        async def _prime_loop_wakeup_stores(self):
            primed.append(1)

    await GatewayStartupMixin._start_prime_watcher_first_pass(_Runner())

    assert primed == [1], "the second half runs even when the first raises"
