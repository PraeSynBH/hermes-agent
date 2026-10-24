"""Checkpoint store pruning rides the gateway housekeeping tick, not the constructor.

Its ``git gc`` repacks the whole store (tens of seconds on a GB store); run at construction it
delayed the control socket, adapters and the code_sha stamp, so the first restart of the day (the
``hermes update`` one) looked hung and failed fleet verification.

The tick has a per-tick ceiling (PRX-6104) and carries chores that no longer fit, so "the prune runs
on the first tick" is only true when the earlier arms leave budget. The arms that can spend it are
stubbed here — they have their own tests — so this file pins the prune's own contract: a tick runs
it. The carry/rotation behaviour is pinned in ``test_housekeeping_tick_budget.py``.
"""

import gateway.run as gateway_run
from gateway import run_profile_reconcile


class _OneTickStopEvent:
    def __init__(self):
        self.waited = False

    def is_set(self):
        return self.waited

    def wait(self, timeout=None):
        self.waited = True
        return True


def test_gateway_housekeeping_runs_the_checkpoint_prune(monkeypatch):
    import tools.checkpoint_maintenance as cm

    calls = []
    monkeypatch.setattr(cm, "auto_prune_from_config", lambda: calls.append(True) or {"skipped": False})
    # Arms whose cost is their own concern: leaving them live makes the outcome depend on how busy
    # this host is (on a 17-profile host they alone pass the 2 s ceiling and carry the prune).
    monkeypatch.setattr(gateway_run, "_housekeeping_memory_trim", lambda: None)
    monkeypatch.setattr(gateway_run, "_housekeeping_plugin_update_check", lambda: None)
    monkeypatch.setattr(
        run_profile_reconcile, "_mcp_config_reconciler", lambda runner=None: lambda: None)

    gateway_run._start_gateway_housekeeping(_OneTickStopEvent(), interval=0)

    assert calls == [True]
