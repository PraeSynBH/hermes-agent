"""The gateway replaces ITSELF when the tree's wire contract moves past its boot snapshot.

SOU-3316 named the skew (``contract_version_mismatch``) and SOU-3318 made it refusable and visible,
but a named skew is still an outage: on the SOU-3328 recurrence the refusal ran for ten minutes while
40 runs across 13 agents and 6 companies died at 0 tokens, because the only actors that could end it
were an hourly monitor and a human running ``restart-gateway-quiet.sh``.

These are the behaviour contracts that close the class, one per failure mode the issue names. The
gateway is the only actor that knows its own boot snapshot and it dies with the process, so each case
is driven tick by tick here against an injected clock — no sleeping, no real gateway:

1. a settled skew -> exactly ONE planned restart, and the published state names both revisions;
2. a one-tick transient (the tree changes and reverts) -> zero restarts;
3. equal digests -> zero restarts AND zero state writes (the watch must be free on a steady tree);
4. the same pair again inside the window -> suppressed, ``contract_stale`` still published, WARNING;
5. no contract document / a non-git tree -> silent (unprovable is not a skew);
6. a good boot whose file becomes unreadable -> silent;
7. turns in flight -> the restart proceeds and the interrupted counts are recorded per profile;
8. the loop guard survives a process restart (a real second process reads the record and stands down).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import contract_watch
from gateway.contract_watch import ContractWatcher, replaced_recently, state_path, watch_seconds

OLD = "old0ld0ld0ld0"
NEW = "new9new9new9"

# The default cadence and guard window, as configured: 10 s ticks, a 20 s settle window.
WATCH, GUARD = 10.0, 600.0


class Rig:
    """A watcher with a pinned clock and observable side effects."""

    def __init__(self, home, boot=OLD, tree=NEW, *, drained=False, restart_ok=True,
                 interrupted=None, publishes=None):
        self.home = home
        self.boot, self.tree = boot, tree
        self.now = 0.0
        self.restarts = 0
        self.restart_ok = restart_ok
        self.drained = drained
        self.interrupted = interrupted or {}
        self.published = [] if publishes is None else publishes
        self.watcher = ContractWatcher(
            home=home, seconds=WATCH, min_seconds_between_restarts=GUARD,
            boot=lambda: self.boot, tree=lambda: self.tree, restart=self._restart,
            publish=self.published.append, interrupted=lambda: self.interrupted,
            drain_active=lambda: self.drained, clock=lambda: self.now)

    def _restart(self) -> bool:
        self.restarts += 1
        return self.restart_ok

    def tick(self, at: float):
        self.now = at
        return self.watcher.tick()

    def settle(self, start: float = 0.0):
        """Three ticks over the settle window: the shape a real skew takes on a steady tree."""
        self.tick(start)
        self.tick(start + WATCH)
        return self.tick(start + 2 * WATCH)


# -- 1. a settled skew replaces the process exactly once -------------------------------------------


def test_a_settled_skew_takes_exactly_one_planned_restart_naming_both_revisions(tmp_path):
    rig = Rig(tmp_path)
    assert rig.tick(0).outcome == "skew_unsettled"
    assert rig.tick(WATCH).outcome == "skew_unsettled", "one digest, once, is not a settled tree"
    assert rig.restarts == 0
    assert rig.tick(2 * WATCH).outcome == "restarting"

    assert rig.restarts == 1
    published = rig.published[-1]
    assert (published["boot"], published["tree"]) == (OLD, NEW)
    assert published["status"] == "requested" and published["pid"] == os.getpid()
    assert published["decided_at"]

    record = json.loads(state_path(tmp_path).read_text())
    assert (record["boot"], record["tree"]) == (OLD, NEW)
    assert record["attempted_at_epoch"] == 2 * WATCH


def test_the_unsettled_window_is_at_least_twenty_seconds(tmp_path):
    """The SOU-3328 merge wrote the tree 45 s before its commit: a single tick is not proof."""
    rig = Rig(tmp_path)
    rig.tick(0)
    rig.tick(WATCH)
    # 15 s after the first sighting: still inside the settle window even though the pair repeated.
    assert rig.tick(WATCH + 5).outcome == "skew_unsettled"
    assert rig.restarts == 0


def test_a_shim_that_keeps_seeing_the_pair_never_stacks_a_second_restart(tmp_path):
    rig = Rig(tmp_path)
    rig.settle()
    assert rig.tick(3 * WATCH).outcome == "restart_in_progress"
    assert rig.restarts == 1


# -- 2. a transient is not a deploy ----------------------------------------------------------------


def test_a_one_tick_transient_takes_zero_restarts(tmp_path):
    rig = Rig(tmp_path)
    assert rig.tick(0).outcome == "skew_unsettled"
    rig.tree = OLD  # the write reverted: a torn checkout, not a merge
    assert rig.tick(WATCH).outcome == "no_skew"
    rig.tree = NEW  # and now it comes back — a fresh sighting, so the window starts again
    assert rig.tick(2 * WATCH).outcome == "skew_unsettled"
    assert rig.restarts == 0
    assert not state_path(tmp_path).exists()


# -- 3. a steady, in-sync tree costs nothing -------------------------------------------------------


def test_equal_digests_write_nothing_at_all(tmp_path):
    rig = Rig(tmp_path, boot=NEW, tree=NEW)
    for at in (0.0, WATCH, 2 * WATCH, 3 * WATCH):
        assert rig.tick(at).outcome == "no_skew"
    assert rig.restarts == 0
    assert rig.published == [], "the watch must not churn gateway_state.json on a healthy tree"
    assert not state_path(tmp_path).exists(), "nor the guard record"


# -- 4. the loop guard: one attempt per window, not a restart loop ---------------------------------


def test_the_same_pair_inside_the_window_is_suppressed_and_stays_visible(tmp_path, caplog):
    Rig(tmp_path).settle()  # the replacement this install already paid for

    # The next process boots, still serving the old contract (a bad checkout / failed swap).
    caplog.set_level(logging.WARNING, logger="gateway.contract_watch")
    after = Rig(tmp_path)
    after.settle(start=30.0)

    assert after.restarts == 0, "a persistent skew plus a failing restart must not become a loop"
    assert after.published, "suppressed or not, contract_stale stays published for S10 and the detector"
    assert after.published[-1]["status"] == "suppressed"
    assert after.published[-1]["suppressed_by"]["attempted_at_epoch"] == 2 * WATCH
    assert any("PERSISTS" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_a_pair_outside_the_window_is_acted_on_again(tmp_path):
    """The guard is a window, not a permanent veto: a genuinely new deploy must still be taken."""
    Rig(tmp_path).settle()
    later = Rig(tmp_path)
    later.settle(start=GUARD + 1)

    assert later.restarts == 1


def test_a_rollback_to_a_different_pair_is_a_new_decision(tmp_path):
    Rig(tmp_path).settle()
    rolled_back = Rig(tmp_path, boot=NEW, tree=OLD)
    rolled_back.settle(start=30.0)
    assert rolled_back.restarts == 1, "the pair changed; the guard must not suppress it"


def test_replaced_recently_ignores_records_it_cannot_read():
    assert replaced_recently(None, OLD, NEW, 100.0, GUARD) is False
    assert replaced_recently({"boot": OLD, "tree": NEW}, OLD, NEW, 100.0, GUARD) is False
    assert replaced_recently({"boot": OLD, "tree": NEW, "attempted_at_epoch": "soon"},
                             OLD, NEW, 100.0, GUARD) is False


# -- 5./6. never act on unprovable input -----------------------------------------------------------


@pytest.mark.parametrize("boot,tree", [(OLD, None), (None, NEW), (None, None)])
def test_no_contract_document_or_a_non_git_tree_is_silent(tmp_path, boot, tree):
    rig = Rig(tmp_path, boot=boot, tree=tree)
    for at in (0.0, WATCH, 3 * WATCH):
        assert rig.tick(at).outcome == "unprovable"
    assert rig.restarts == 0 and rig.published == []
    assert not state_path(tmp_path).exists()


def test_an_unreadable_file_after_a_good_boot_is_not_a_skew(tmp_path):
    """An I/O blip during the read is not proof that the wire contract moved."""
    rig = Rig(tmp_path, boot=OLD, tree=None)
    rig.settle()
    assert rig.restarts == 0 and rig.published == []


# -- 7. turns in flight: restart anyway, count what it interrupted ---------------------------------


def test_in_flight_turns_do_not_hold_the_restart_and_are_counted_per_profile(tmp_path):
    rig = Rig(tmp_path, interrupted={"default": 3, "sou": 1})
    rig.settle()

    assert rig.restarts == 1, "a proven skew dooms every in-flight turn at its next model call"
    assert rig.published[-1]["interrupted"] == {"default": 3, "sou": 1}
    assert json.loads(state_path(tmp_path).read_text())["interrupted"] == {"default": 3, "sou": 1}


def test_without_a_supervisor_the_published_claim_names_who_respawns_it(tmp_path):
    """An unsupervised launch must not read as "replaced": it exits 75 and the boot guard comes back."""
    rig = Rig(tmp_path)
    rig.watcher._supervised = False
    rig.settle()

    assert rig.restarts == 1, "the planned stop is still the route: it exits 75 for the boot guard"
    assert "ensure-gateway-boot" in rig.published[-1]["respawn"]
    assert rig.published[-1]["supervised"] is False


def test_an_in_progress_drain_wins_over_a_second_restart(tmp_path):
    rig = Rig(tmp_path, drained=True)
    rig.settle()
    assert rig.restarts == 0
    assert rig.published == []


# -- 8. the guard is durable: a real second process stands down ------------------------------------


def test_the_loop_guard_survives_a_process_restart(tmp_path):
    """Write the marker, then boot a NEW process: the same pair must be suppressed there too."""
    repo_root = Path(contract_watch.__file__).resolve().parents[1]
    script = """
import json, sys
from gateway.contract_watch import ContractWatcher, state_path
home = sys.argv[1]
calls = []
clock = iter([0.0, 10.0, 20.0, 30.0])
w = ContractWatcher(home=home, seconds=10.0, min_seconds_between_restarts=600.0,
                    boot=lambda: 'old0ld0ld0ld0', tree=lambda: 'new9new9new9',
                    restart=lambda: calls.append('restart') or True,
                    publish=calls.append, interrupted=dict, drain_active=lambda: False,
                    clock=lambda: next(clock))
for _ in range(3):
    w.tick()
print(json.dumps({'restarts': calls.count('restart'),
                  'record': json.loads(state_path(home).read_text())}))
"""
    env = {**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": str(repo_root)}
    first = subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=repo_root, env=env, check=False,
                           capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["restarts"] == 1
    record_after_first = json.loads(first.stdout)["record"]
    assert (record_after_first["boot"], record_after_first["tree"]) == (OLD, NEW)

    # A brand-new process, reading only the file, must not replace itself again for this pair.
    second = subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=repo_root,
                            env=env, check=False, capture_output=True, text=True, timeout=120)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["restarts"] == 0, "the guard must be durable across the restart"


# -- the publish seam really lands in gateway_state.json --------------------------------------------


def test_contract_stale_is_published_into_gateway_state(tmp_path, monkeypatch):
    """``gateway_state.json`` is what S10 and the run-failure detector read: the claim must land there."""
    from gateway.status import read_runtime_status, write_runtime_status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    clock = iter([0.0, WATCH, 2 * WATCH])
    watcher = ContractWatcher(home=tmp_path, seconds=WATCH, boot=lambda: OLD, tree=lambda: NEW,
                              restart=lambda: True, interrupted=dict, clock=lambda: next(clock))
    assert watcher.tick().outcome == "skew_unsettled"
    assert watcher.tick().outcome == "skew_unsettled"
    assert watcher.tick().outcome == "restarting"

    stale = read_runtime_status()["contract_stale"]
    assert (stale["boot"], stale["tree"]) == (OLD, NEW)

    # An in-sync process withdraws the previous life's claim exactly once (never per tick).
    steady = ContractWatcher(home=tmp_path, seconds=WATCH, boot=lambda: NEW, tree=lambda: NEW,
                             publish=lambda payload: write_runtime_status(contract_stale=payload))
    assert steady.tick().outcome == "no_skew"
    assert read_runtime_status().get("contract_stale") is None


# -- config: config.yaml, not a HERMES_* env var ---------------------------------------------------


def test_the_cadence_and_window_come_from_config_yaml(tmp_path, monkeypatch):
    from hermes_cli.config import load_config_readonly

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "gateway:\n  contract_watch_seconds: 3\n  contract_watch_min_seconds_between_restarts: 45\n")
    config = load_config_readonly()
    assert watch_seconds(config) == 3
    assert contract_watch.min_seconds_between_restarts(config) == 45


def test_unreadable_or_disabled_values_fall_back_instead_of_disabling_the_guard(tmp_path):
    assert watch_seconds({"gateway": {}}) == contract_watch.DEFAULT_WATCH_SECONDS
    assert watch_seconds({"gateway": {"contract_watch_seconds": "often"}}) == 10.0
    assert watch_seconds({"gateway": {"contract_watch_seconds": 0}}) == 0.0, "0 disables the watch"
    assert contract_watch.min_seconds_between_restarts({"gateway": {"contract_watch_min_seconds_between_restarts": 0}}) == 600.0


def test_a_disabled_watch_starts_no_ticker():
    assert contract_watch.watcher_for_runner(
        SimpleNamespace(), SimpleNamespace(), config={"gateway": {"contract_watch_seconds": 0}}) is None


# -- the live wiring: the watcher reaches the runner through the gateway loop -----------------------


def test_the_live_watcher_asks_the_runner_for_a_restart_through_the_loop(tmp_path, monkeypatch):
    """The one seam the unit cases cannot prove: a housekeeping-thread tick landing on the runner."""
    import asyncio
    import threading

    from gateway import contract_revision

    # A long-lived process: its boot snapshot is the OLD contract, the tree now declares the NEW one.
    monkeypatch.setattr(contract_revision, "_boot_revision", OLD, raising=False)
    monkeypatch.setattr(contract_revision, "_boot_recorded", True, raising=False)
    monkeypatch.setattr(contract_revision, "revision", lambda root=None: NEW)

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True, name="test-gateway-loop")
    thread.start()
    asked: dict = {}

    class Runner:
        _draining = False
        _restart_task_started = False

        def request_restart(self, **kwargs):
            asked.update(kwargs)
            return True

        def served_profile_names(self):
            return ["default"]

        def _describe_active_work(self):
            return [{"kind": "chat", "session": "agent:main:s1"},
                    {"kind": "chat", "session": "agent:main:s2"},
                    {"kind": "managed"}]

    try:
        watcher = contract_watch.watcher_for_runner(
            Runner(), loop, home=tmp_path, config={"gateway": {"contract_watch_seconds": 0.001}})
        assert watcher is not None
        from tests.waiting import wait_until

        outcome = wait_until(lambda: watcher.tick().outcome == "restarting" and "restarting",
                             timeout=5, what="the watcher to replace the process")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert outcome == "restarting"
    assert asked == {"detached": False, "via_service": True, "interrupt_in_flight": True}
    record = json.loads(state_path(tmp_path).read_text())
    assert record["interrupted"] == {"default": 2, "unattributed": 1}, "per profile, so the next boot settles them"


# -- the housekeeping seam: a ticker runs at its own cadence inside the wait ------------------------


class _StopEvent:
    """Never set: the wait must return on its own deadline."""

    def wait(self, timeout=None):
        return False


class _Ticker:
    seconds = 0.01
    label = "ticker"

    def __init__(self):
        self.ticks = 0

    def tick(self):
        self.ticks += 1


def test_wait_for_next_tick_runs_tickers_faster_than_the_housekeeping_interval():
    from gateway.run_delivery_queue_watch import wait_for_next_tick

    ticker = _Ticker()
    assert wait_for_next_tick(_StopEvent(), 0.05, None, lambda label, fn: fn(), [ticker]) is None
    assert ticker.ticks >= 2, "a 10 s watch inside a 60 s tick only works if the wait runs it"
