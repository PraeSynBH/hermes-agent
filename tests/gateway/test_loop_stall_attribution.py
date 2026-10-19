"""A stalled gateway loop must name the frame that is holding it.

PRA-6517: ``/usr/bin/sample`` proved the gateway's loop thread spent its time in ``open``/
``os_stat``/``_Py_read`` inside one coroutine, but a C-level sample cannot name that coroutine,
so the defect could not be pointed at a line. The out-of-loop liveness watchdog already probes
the loop; it now reads the loop thread's Python frames on the first unanswered probe. These
tests pin that behaviour: a loop blocked in a real blocking syscall is reported with the exact
``file:line function`` of the coroutine that made it, a healthy loop reports nothing, and a
recovered stall reports how long it lasted.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading

import pytest

from gateway.shutdown_watchdog import start_loop_liveness_watchdog
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("linux", "macos")


class _LoopThread:
    """A real event loop on its own OS thread, as the gateway runs it."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.ident: int | None = None
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="loop-under-test", daemon=True)

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.ident = threading.get_ident()
        self._ready.set()
        self.loop.run_forever()

    def __enter__(self):
        self.thread.start()
        assert self._ready.wait(timeout=10), "the test loop never started"
        return self

    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        self.loop.close()


def _stalled_messages(records) -> list[str]:
    return [r.getMessage() for r in records if "event loop stalled" in r.getMessage()]


def test_stalled_loop_reports_the_frame_that_holds_it(caplog):
    """Red on main: a missed probe was counted silently, so the stall named no code."""
    unblock = threading.Event()

    async def _turn_that_blocks_the_loop():
        # A coroutine doing synchronous blocking work: exactly the sampled shape (the loop
        # thread parked in a syscall inside one task step).
        unblock.wait(10)

    with _LoopThread() as loop_thread:
        with caplog.at_level(logging.WARNING, logger="gateway.shutdown_watchdog"):
            handle = start_loop_liveness_watchdog(
                loop_thread.loop, probe_interval=0.05, probe_timeout=0.3, max_strikes=1000,
                loop_thread_ident=loop_thread.ident,
            )
            assert handle is not None
            try:
                asyncio.run_coroutine_threadsafe(_turn_that_blocks_the_loop(), loop_thread.loop)
                stalled = wait_until(lambda: _stalled_messages(caplog.records),
                                     what="the watchdog reporting the stalled loop")
                reported = stalled[0]
                assert "test_loop_stall_attribution.py" in reported, reported
                assert "_turn_that_blocks_the_loop" in reported, reported
                assert "->" in reported, reported
            finally:
                unblock.set()
                handle.stop()


def test_healthy_loop_reports_no_stall(caplog):
    """The dump is free when the loop is fine: no warning, no thread read."""
    with _LoopThread() as loop_thread:
        with caplog.at_level(logging.WARNING, logger="gateway.shutdown_watchdog"):
            handle = start_loop_liveness_watchdog(
                loop_thread.loop, probe_interval=0.02, probe_timeout=0.3, max_strikes=1000,
                loop_thread_ident=loop_thread.ident,
            )
            assert handle is not None
            try:
                wait_until(lambda: not loop_thread.loop.is_closed() and loop_thread.ident,
                           what="the probe to run")
                assert _stalled_messages(caplog.records) == []
            finally:
                handle.stop()


def test_recovered_stall_reports_its_duration(caplog):
    """A stall that ends must yield a duration — the number this card is measured in."""
    unblock = threading.Event()

    async def _stall_then_return():
        unblock.wait(10)
        return "done"

    with _LoopThread() as loop_thread:
        with caplog.at_level(logging.WARNING, logger="gateway.shutdown_watchdog"):
            handle = start_loop_liveness_watchdog(
                loop_thread.loop, probe_interval=0.05, probe_timeout=0.3, max_strikes=1000,
                loop_thread_ident=loop_thread.ident,
            )
            assert handle is not None
            try:
                future = asyncio.run_coroutine_threadsafe(_stall_then_return(), loop_thread.loop)
                wait_until(lambda: _stalled_messages(caplog.records), what="the stall to be reported")
                unblock.set()
                assert future.result(timeout=10) == "done"
                recovered = wait_until(
                    lambda: [m for m in _stalled_messages(caplog.records) if "and recovered" in m],
                    what="the watchdog reporting the recovered stall")
                assert "stalled" in recovered[0] and "s and recovered" in recovered[0]
            finally:
                unblock.set()
                handle.stop()
