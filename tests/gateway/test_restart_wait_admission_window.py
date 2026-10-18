"""A restart's after-turn wait admits new work until its force-drain window opens.

The wait exists so in-flight turns are not amputated (#77184), but it can hold the process for
``restart_after_turn_timeout`` (1800 s by default) while the owner is healthy and serving. Refusing
every new admission for that whole window is a fleet-wide blackout: measured 2026-10-18 05:30-06:00Z,
one restart request at 43 active units produced 267 ``runtime_draining`` refusals in 30 min — 95 % of
all run failures in the window — and the wait then timed out and amputated a turn anyway.

So the wait admits until ``RESTART_WAIT_ADMISSION_CLOSE_S`` before its force-drain deadline, and every
other drain (retirement, update pause, the closing window, ``stop()``) refuses exactly as before.
"""
from types import SimpleNamespace
from typing import cast

import pytest

from gateway import drain_control
from gateway.session_authority import SessionAuthority
from hermes_state_runtime import RuntimeStoreError


def _runner(*, draining=True, running=True, admit_for=None):
    runner = SimpleNamespace(_draining=draining, _running=running,
                             session_runtime_descriptor={"state": "draining"})
    drain_control.open_restart_wait(runner, admit_for=admit_for)
    return runner


def _require(runner, *, queue_only=False):
    """The gate every admission path re-checks, on a stand-in authority (it reads only
    ``retiring`` and ``runner``)."""
    authority = cast(SessionAuthority, SimpleNamespace(retiring=False, runner=runner))
    SessionAuthority._require_admission_open(authority, queue_only=queue_only)


def test_the_wait_admits_new_work_while_it_is_far_from_its_force_drain_deadline():
    runner = _runner(admit_for=1800 - drain_control.RESTART_WAIT_ADMISSION_CLOSE_S)
    assert drain_control.restart_wait_open(runner)
    assert drain_control.restart_wait_admits(runner)
    # The gate every admission path re-checks: a claim is accepted, not refused.
    _require(runner)
    _require(runner, queue_only=True)


def test_the_last_minute_before_the_force_drain_refuses_new_work_again():
    # 30 s left on the force-drain deadline, inside the 60 s closing window: the wait computes a
    # non-positive admit span, so nothing new may claim.
    runner = _runner(admit_for=30.0 - drain_control.RESTART_WAIT_ADMISSION_CLOSE_S)
    assert drain_control.restart_wait_open(runner), 'still the wait, so parked prompts stay reachable'
    assert not drain_control.restart_wait_admits(runner)
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        _require(runner)
    # Queued input for the next owner is still committed (#139041).
    _require(runner, queue_only=True)


def test_closing_the_window_refuses_immediately_even_with_time_left():
    """``_await_active_work_before_restart`` closes the window when the wait ends early (all work
    drained): admission must not stay open into ``stop()``."""
    runner = _runner(admit_for=1800.0)
    assert drain_control.restart_wait_admits(runner)
    drain_control.close_restart_wait_admission(runner)
    assert not drain_control.restart_wait_admits(runner)
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        _require(runner)


@pytest.mark.parametrize('draining,running', [(False, True), (True, False)])
def test_only_a_restart_wait_admits(draining, running):
    """A ready owner obviously admits; a stopping one never does, whatever window is stamped."""
    runner = _runner(draining=draining, running=running, admit_for=1800.0)
    assert not drain_control.restart_wait_admits(runner)
    if draining:
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            _require(runner)


def test_another_drain_names_its_own_reason_and_never_admits(monkeypatch):
    """An update pause / operator drain already published its own ``drain_reason``: the restart wait
    does not overwrite it, and no admission window is opened for it."""
    runner = SimpleNamespace(_draining=True, _running=True,
                             session_runtime_descriptor={"state": "draining",
                                                         "drain_reason": "update_pause"})
    drain_control.open_restart_wait(runner, admit_for=1800.0)
    assert runner.session_runtime_descriptor["drain_reason"] == "update_pause"
    assert not drain_control.restart_wait_admits(runner)


def test_a_non_positive_window_admits_nothing():
    runner = _runner(admit_for=0.0)
    assert not drain_control.restart_wait_admits(runner)


def test_the_window_is_read_off_the_event_loop_without_a_loop(monkeypatch):
    """The gate is re-checked inside an off-loop write transaction: the deadline must be process
    monotonic, readable from a worker thread with no running loop."""
    monkeypatch.setattr(drain_control.time, 'monotonic', lambda: 1000.0)
    runner = _runner(admit_for=600.0)
    assert drain_control.restart_wait_admits(runner)
    monkeypatch.setattr(drain_control.time, 'monotonic', lambda: 1600.0)
    assert not drain_control.restart_wait_admits(runner)
