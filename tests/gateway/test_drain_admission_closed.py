"""Admission closes at T=0 of a drain, and a restart's after-turn wait never reopens it.

The wait exists so in-flight turns are not amputated (#77184), and it can hold the process for
``restart_after_turn_timeout`` (1800 s by default). A previous change kept ADMITTING new claims for
that window and closed only 60 s before the force-drain deadline. Measured 2026-10-21, that window is
where work goes to die: two restarts (21:37 and 23:05 PDT) started 101 runs inside their waits and
killed every one of them (``error='runtime_draining'``, mean lifetime 7.2 s and 4.6 s), while the log
line for the same 30 s tick said "admission open" — the deadline arithmetic, not the gate every
admission actually consults.

So the wait KEEPS what was in flight at T=0 and admits nothing new: a claim is refused for the whole
drain with the retryable ``runtime_draining`` verdict clients already wait out, queued input for the
next owner still commits (#139041), and the reason stays published so an interactive client can still
attach to answer a turn parked on approval/clarify.
"""
import asyncio
import time
from types import SimpleNamespace
from typing import cast

import pytest

from gateway import drain_control
from gateway.run_shutdown import GatewayShutdownMixin
from gateway.session_authority import SessionAuthority
from hermes_state_runtime import RuntimeStoreError
from tests.waiting import async_wait_until


def _restart_wait_runner():
    """A grounded owner mid-restart-wait: draining, still running, its wait published."""
    runner = SimpleNamespace(_draining=True, _running=True,
                             session_runtime_descriptor={"state": "draining"})
    drain_control.open_restart_wait(runner)
    return runner


def _require(runner, *, queue_only=False):
    """The gate every admission path re-checks, on a stand-in authority (it reads only
    ``retiring`` and ``runner``)."""
    authority = cast(SessionAuthority, SimpleNamespace(retiring=False, runner=runner))
    SessionAuthority._require_admission_open(authority, queue_only=queue_only)


def test_the_wait_refuses_a_new_claim_at_t_zero():
    """The moment the restart is requested — before any drain of in-flight work — a claim is out."""
    runner = _restart_wait_runner()
    assert drain_control.restart_wait_open(runner), 'the wait is published for parked prompts'
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        _require(runner)


def test_nothing_reopens_the_claim_path_for_the_rest_of_the_wait():
    """A leftover admission deadline of the old design must not buy a claim: the gate reads state,
    not time. 1700 s of force-drain headroom changes nothing."""
    runner = _restart_wait_runner()
    runner.session_runtime_descriptor["drain_reason"] = drain_control.RESTART_WAIT_REASON
    runner._restart_wait_admit_until = time.monotonic() + 1700.0
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        _require(runner)


def test_queued_input_for_the_next_owner_still_commits():
    """#139041: a queue-only admission WRITES a durable row and never claims it, so the next owner
    runs it exactly once — that is the "runs queued at drain start stay queued" path."""
    _require(_restart_wait_runner(), queue_only=True)


def test_a_plain_drain_accepts_no_admission_at_all():
    """Every drain other than the restart wait (retirement, update pause, a server drain marker)
    refuses a queue-only commit too: there is no next owner promised for it."""
    runner = SimpleNamespace(_draining=True, _running=True,
                             session_runtime_descriptor={"state": "draining",
                                                         "drain_reason": "update_pause"})
    drain_control.open_restart_wait(runner)
    assert runner.session_runtime_descriptor["drain_reason"] == "update_pause"
    for queue_only in (False, True):
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            _require(runner, queue_only=queue_only)


def test_a_ready_owner_admits():
    runner = SimpleNamespace(_draining=False, _running=True,
                             session_runtime_descriptor={"state": "ready"})
    drain_control.open_restart_wait(runner)
    _require(runner)
    _require(runner, queue_only=True)


def test_the_after_turn_wait_never_reopens_admission_while_it_polls():
    """The seam that used to reopen it: ``_await_active_work_before_restart`` publishes the wait and
    polls until the in-flight work finishes. A claim attempted during that poll is refused, and the
    wait still drains to completion (PRX-5771's graceful drain is untouched)."""
    remaining = {"count": 2}

    def awaitable():
        return remaining["count"]

    def active():
        return remaining["count"]

    runner = SimpleNamespace(
        _draining=True, _running=True, session_runtime_descriptor={"state": "draining"},
        # The production default: a wait that would leave the old design ~1740 s of admitting.
        _restart_after_turn_timeout=1800.0, retiring=False,
        _active_work_count=active, _awaitable_work_count=awaitable,
        _wedged_agent_count=lambda: 0, _restart_safe_cron_count=lambda: 0,
        _describe_active_work=list, _scale_to_zero_status=lambda *a, **k: None,
    )

    async def scenario():
        wait = asyncio.create_task(
            GatewayShutdownMixin._await_active_work_before_restart(
                cast(GatewayShutdownMixin, runner)))
        await async_wait_until(lambda: drain_control.restart_wait_open(runner),
                               what='the restart wait is published')
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            _require(runner)
        _require(runner, queue_only=True)
        remaining["count"] = 0  # the in-flight turn finished
        assert await asyncio.wait_for(wait, timeout=5) is True, 'the drain still completes'

    asyncio.run(scenario())
