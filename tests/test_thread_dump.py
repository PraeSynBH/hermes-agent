"""The blocked-loop thread dump must be bounded, rate-limited, and unable to kill the process.

Regression for PRA-6536. ``agent/deadline.py`` (and both watchdogs) called
``faulthandler.dump_traceback(all_threads=True)`` when the event loop looked blocked. That walk
runs in C (``_Py_DumpTracebackThreads``) over every thread state; on 2026-10-19 at ~240 live
threads it faulted and the gateway died with exit 139 (SIGSEGV), taking six in-flight runs with
it. The diagnostic had become the largest measured cause of fleet run loss.

The contracts pinned here: the dump comes from ``sys._current_frames()`` (never the C walk), it
is capped by thread count and characters, it fires at most once per process per interval, and a
frames read that fails costs a log line, not an exception.
"""

import io
import sys
import threading

import pytest

import hermes_thread_dump as td


@pytest.fixture(autouse=True)
def _fresh_rate_limit():
    td.reset_dump_rate_limit()
    yield
    td.reset_dump_rate_limit()


@pytest.fixture()
def busy_threads():
    """~200 live threads blocked on an event: the gateway's production plateau (PRA-6507)."""
    stop = threading.Event()
    threads = [
        threading.Thread(target=stop.wait, name=f"plateau-{i}", daemon=True)
        for i in range(200)
    ]
    for t in threads:
        t.start()
    try:
        yield threads
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)


def test_a_dump_never_uses_the_c_all_thread_walk(monkeypatch):
    """The crash is in faulthandler's C walk, so the helper must not reach it at all."""
    import faulthandler

    def explode(*_args, **_kwargs):
        raise AssertionError("the C all-thread dump was used")

    monkeypatch.setattr(faulthandler, "dump_traceback", explode)
    out = io.StringIO()

    text = td.dump_thread_stacks("unit test", file=out)

    assert text and "thread dump: unit test" in out.getvalue()
    assert "MainThread" in text


def test_the_dump_is_capped_in_threads_and_characters(busy_threads):
    text = td.render_thread_stacks(max_threads=5, max_chars=8_000)

    dumped = [line for line in text.splitlines() if line.startswith("--- thread ")]
    assert 1 <= len(dumped) <= 5, f"capped fan-out violated: {len(dumped)} threads dumped"
    assert "threads omitted" in text, "a capped dump must say how much it left out"
    assert len(text) <= 8_000 + 200, f"character cap violated: {len(text)}"


def test_the_dump_is_rate_limited_per_process():
    first = td.dump_thread_stacks("first", file=io.StringIO(), min_interval_s=3_600)
    second = td.dump_thread_stacks("second", file=io.StringIO(), min_interval_s=3_600)

    assert first, "the first dump must be produced"
    assert second == "", "a second dump inside the floor must be suppressed, not repeated"


def test_unreadable_frames_cost_a_log_line_not_an_exception(monkeypatch):
    def unreadable():
        raise RuntimeError("thread states are being torn down")

    monkeypatch.setattr(sys, "_current_frames", unreadable)

    assert td.dump_thread_stacks("broken", file=io.StringIO()) == ""


def test_the_deadline_path_survives_the_production_thread_plateau(busy_threads, monkeypatch):
    """AC#1's invariant form: the real call site, 200 live threads, no process loss."""
    import faulthandler

    import agent.deadline as deadline

    def explode(*_args, **_kwargs):
        raise AssertionError("the C all-thread dump was used")

    monkeypatch.setattr(faulthandler, "dump_traceback", explode)
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured)

    deadline._dump_blocked_loop_diagnostics("terminal.wait", 182.0)

    out = captured.getvalue()
    assert "thread dump: deadline expired with the loop blocked" in out
    assert "MainThread" in out
    assert "cap " in out or "threads omitted" in out, "the plateau dump must be capped, not whole"
