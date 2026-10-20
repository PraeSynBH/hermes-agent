"""End-to-end proof for PRA-6536: an expiring deadline on a blocked loop leaves a dump, not a corpse.

pid 67785 (the shared gateway, 240 threads) expired a tool deadline on a loop blocked for 181s;
the diagnostic that ran next was ``faulthandler.dump_traceback(all_threads=True)``, which
segfaulted inside ``_Py_DumpTracebackThreads`` and took six in-flight runs with it.

``tests/test_thread_dump.py`` pins the helper's contracts. This file pins the call site the
crash actually happened on: a real ``run_bounded_async`` deadline expiring while the loop thread
is provably blocked, with the thread count at the production plateau (PRA-6502: 200-250 live),
asserting the process survives, the dump names the blocked frame, and the C-level all-thread
walk is never reached.
"""

from __future__ import annotations

import asyncio
import faulthandler
import re
import threading

import pytest

import hermes_thread_dump as td
from agent import deadline as dl


@pytest.fixture(autouse=True)
def _fresh_dump_window():
    td.reset_dump_rate_limit()
    yield
    td.reset_dump_rate_limit()


@pytest.fixture
def plateau_threads():
    """200-250 live threads — the profile the gateway carried when it died."""
    stop = threading.Event()
    threads = [
        threading.Thread(target=stop.wait, name=f"plateau-{i}", daemon=True) for i in range(230)
    ]
    for thread in threads:
        thread.start()
    try:
        assert threading.active_count() >= 200
        yield
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5.0)


@pytest.fixture
def c_all_thread_walks(monkeypatch):
    """Every attempt at faulthandler's C-level all-thread walk — which must stay at zero."""
    walks = []
    monkeypatch.setattr(
        faulthandler, "dump_traceback", lambda *args, **kwargs: walks.append(kwargs)
    )
    return walks


def test_expired_deadline_on_a_blocked_loop_does_not_kill_the_process(
    plateau_threads, c_all_thread_walks, monkeypatch, capsys, caplog
):
    """The incident, replayed: deadline fires, loop is blocked, the process lives to log it."""
    # Shrink the grace so the blocked-loop diagnostic fires while the loop is still stuck.
    monkeypatch.setattr(dl, "_LOOP_BLOCKED_DUMP_GRACE_S", 0.2)

    async def scenario():
        blocker = threading.Event()

        async def blocking_op():
            blocker.wait(0.6)  # a synchronous call on the loop thread: no asyncio timeout fires
            await asyncio.Event().wait()  # never completes, so the deadline is the only outcome

        return await dl.run_bounded_async(blocking_op(), 0.05, label="segv-repro")

    result = asyncio.run(scenario())

    assert result.timed_out is True  # the deadline really did expire, as in the incident
    assert c_all_thread_walks == []  # and the C-level all-thread walk was never reached

    err = capsys.readouterr().err
    assert "thread dump: deadline expired with the loop blocked" in err
    assert "segv-repro" in err
    assert "MainThread" in err  # the blocked loop thread is named
    assert any(
        record.getMessage().startswith("[deadline] 'segv-repro' deadline")
        for record in caplog.records
    )


def test_the_plateau_dump_stays_capped(plateau_threads, capsys):
    """230 live threads must not become a 230-thread write on an already wedged process."""
    dl._dump_blocked_loop_diagnostics("cap-check", 1.0)

    text = capsys.readouterr().err

    dumped = [line for line in text.splitlines() if re.match(r"--- thread \d+ ", line)]
    assert len(dumped) <= td.DEFAULT_MAX_THREADS
    assert "threads omitted" in text
    assert len(text) <= td.DEFAULT_MAX_CHARS + 300
