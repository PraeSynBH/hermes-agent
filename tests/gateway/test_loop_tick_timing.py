"""PRX-5780: the per-tick counters that make a ticker's cost a trend instead of one dump frame.

A blocked-loop dump samples ONE frame of one tick, so "did the fix move the work" was unanswerable
after PRX-5524 — the dumps only say which site was running when the probe expired. These counters
record every periodic ticker's wall time and log one summary per interval, which is what an 8 h
window can be compared across. They are process-local and advisory; the dump and the loopback p95
stay the authority on stalls.
"""

from __future__ import annotations

import logging

import pytest

from gateway import loop_tick_timing as timing


@pytest.fixture(autouse=True)
def _fresh_counters():
    timing.reset()
    yield
    timing.reset()


def test_counters_report_count_mean_p50_and_max():
    for seconds in (0.01, 0.02, 0.03):
        timing.record('ticker', seconds)
    view = timing.snapshot()['ticker']
    assert view['count'] == 3
    assert view['last_s'] == pytest.approx(0.03)
    assert view['mean_s'] == pytest.approx(0.02)
    assert view['p50_s'] == pytest.approx(0.02)
    assert view['max_s'] == pytest.approx(0.03)


def test_window_keeps_the_recent_p50_as_load_changes():
    """A 30 s ticker's p50 must follow the last 64 ticks, not every tick since boot."""
    for _ in range(timing.WINDOW):
        timing.record('ticker', 5.0)
    for _ in range(timing.WINDOW):
        timing.record('ticker', 0.001)
    view = timing.snapshot()['ticker']
    assert view['count'] == 2 * timing.WINDOW
    assert view['p50_s'] < 0.01, 'the window must have rolled past the expensive ticks'
    assert view['max_s'] == pytest.approx(5.0)


def test_timed_records_a_tick_that_raises():
    """A ticker whose work fails still cost the loop that time: record it, re-raise to the caller."""
    with pytest.raises(RuntimeError):
        with timing.timed('ticker'):
            raise RuntimeError('watcher body failed')
    assert timing.snapshot()['ticker']['count'] == 1


def test_summary_logs_once_per_interval(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger='gateway.loop_tick_timing')
    timing.record('profile_reconcile', 0.25, now=100.0)
    assert timing.log_summary(now=100.0 + 1.0) is None, 'not due yet'
    line = timing.log_summary(now=100.0 + timing.SUMMARY_INTERVAL_S + 1.0)
    assert line is not None and 'profile_reconcile n=1' in line
    assert 'ms' in line
    assert timing.log_summary(now=100.0 + timing.SUMMARY_INTERVAL_S + 2.0) is None


def test_snapshot_is_a_copy_and_reset_clears():
    timing.record('ticker', 0.5)
    view = timing.snapshot()
    view['ticker']['count'] = 99
    assert timing.snapshot()['ticker']['count'] == 1
    timing.reset()
    assert timing.snapshot() == {}
