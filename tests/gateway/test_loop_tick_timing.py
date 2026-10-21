"""PRX-5780/PRX-5790: the per-tick counters that make a watcher's cost a trend instead of one dump frame.

A blocked-loop dump samples ONE frame of one tick, so "did the fix move the work" was unanswerable
after PRX-5524 — the dumps only say which site was running when the probe expired. These counters
record every periodic watcher's wall time and log one line per watcher per minute with count /
p50 / p95 / max, which is what a boot can be compared across (cold start: does the first two
minutes sit above the rest at unchanged load?). They are process-local and advisory; the dump and
the loopback p95 stay the authority on stalls.
"""

from __future__ import annotations

import asyncio
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
        timing.record('watcher', seconds)
    view = timing.snapshot()['watcher']
    assert view['count'] == 3
    assert view['window_count'] == 3
    assert view['last_s'] == pytest.approx(0.03)
    assert view['mean_s'] == pytest.approx(0.02)
    assert view['p50_s'] == pytest.approx(0.02)
    assert view['max_s'] == pytest.approx(0.03)


def test_p95_reports_the_tail_the_mean_hides():
    """A watchdog-scale tick in a quiet window is the thing a sweep is looking for: p95 must surface
    it while p50 stays at the steady state."""
    for _ in range(60):
        timing.record('watcher', 0.002, now=-1_000.0)
    for _ in range(4):
        timing.record('watcher', 4.0, now=-1_000.0)
    view = timing.snapshot()['watcher']
    assert view['window_count'] == 64
    assert view['p50_s'] == pytest.approx(0.002)
    assert view['p95_s'] == pytest.approx(4.0)
    assert view['max_window_s'] == pytest.approx(4.0)
    assert view['max_s'] == pytest.approx(4.0)


def test_window_keeps_the_recent_percentiles_as_load_changes():
    """A 30 s watcher's p50/p95 must follow the last 64 ticks, but the boot maximum must survive the
    roll — a cold-start spike has to stay readable after it leaves the ring."""
    for _ in range(timing.WINDOW):
        timing.record('watcher', 5.0, now=-1_000.0)
    for _ in range(timing.WINDOW):
        timing.record('watcher', 0.001)
    view = timing.snapshot()['watcher']
    assert view['count'] == 2 * timing.WINDOW
    assert view['window_count'] == timing.WINDOW
    assert view['p50_s'] < 0.01, 'the window must have rolled past the expensive ticks'
    assert view['p95_s'] < 0.01
    assert view['max_window_s'] < 0.01
    assert view['max_s'] == pytest.approx(5.0), 'boot_max is what answers cold start'


def test_timed_records_a_tick_that_raises():
    """A watcher whose work fails still cost the loop that time: record it, re-raise to the caller."""
    with pytest.raises(RuntimeError):
        with timing.timed('watcher'):
            raise RuntimeError('watcher body failed')
    assert timing.snapshot()['watcher']['count'] == 1


def test_summary_logs_one_line_per_watcher_per_interval(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger='gateway.loop_tick_timing')
    timing.record('profile_reconcile', 0.25, now=100.0)
    timing.record('hosted_room_worker', 0.002, now=100.0)
    assert timing.log_summary(now=100.0 + 1.0) is None, 'not due yet'
    lines = timing.log_summary(now=100.0 + timing.SUMMARY_INTERVAL_S + 1.0)
    assert lines is not None and len(lines) == 2
    joined = '\n'.join(lines)
    assert 'profile_reconcile n=1' in joined
    assert 'hosted_room_worker n=1' in joined
    for token in ('p50=', 'p95=', 'max=', 'boot_max='):
        assert token in lines[0]
    assert timing.log_summary(now=100.0 + timing.SUMMARY_INTERVAL_S + 2.0) is None


def test_slow_tick_warns_once_per_interval_and_is_counted(caplog):
    """A starved loop would otherwise emit a warning per tick of the 1 s watcher; the burst is
    counted and reported in the next summary instead."""
    caplog.set_level(logging.INFO, logger='gateway.loop_tick_timing')
    timing.record('hosted_room_worker', timing.SLOW_TICK_S + 0.5, now=100.0)
    timing.record('hosted_room_worker', timing.SLOW_TICK_S + 0.5, now=101.0)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    view = timing.snapshot()['hosted_room_worker']
    assert view['slow_since_summary'] == 2
    assert view['slow_total'] == 2
    lines = timing.log_summary(now=100.0 + timing.SUMMARY_INTERVAL_S + 1.0)
    assert lines and 'slow=2/2' in lines[0]
    assert timing.snapshot()['hosted_room_worker']['slow_since_summary'] == 0


def test_quiet_timed_skips_the_per_tick_debug_line(caplog):
    """The 1 s watcher reports through the summary only: per-tick DEBUG would be ~86k lines a day."""
    caplog.set_level(logging.DEBUG, logger='gateway.loop_tick_timing')
    with timing.timed('hosted_room_worker', per_tick_debug=False):
        pass
    with timing.timed('loop_wakeup_scan'):
        pass
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(debug) == 1 and debug[0].startswith('tick loop_wakeup_scan ')


def test_housekeeping_tick_is_timed_as_one_unit():
    """The 60 s housekeeping tick is the fifth site: it is a thread, so a chore waiting on the loop
    (channel directory, low-disk prompt) is what inflates it — and that is the number that tells a
    loop-starved tick apart from a slow disk."""
    from gateway.run_housekeeping_chores import _run_due_chores

    ran = []
    chores = [(1, 'every tick', lambda: ran.append('always')),
              (5, 'every fifth', lambda: ran.append('fifth'))]
    _run_due_chores(chores, 1)
    assert ran == ['always']
    assert timing.snapshot()['housekeeping_tick']['count'] == 1
    _run_due_chores(chores, 5)
    assert ran == ['always', 'always', 'fifth']
    assert timing.snapshot()['housekeeping_tick']['count'] == 2


def test_control_repaint_is_timed_where_it_lands_on_the_loop(monkeypatch):
    """``publish_control_if_changed`` runs on the agent thread but its publish is scheduled onto the
    loop; the cost that matters is the scheduled coroutine's, not the caller's."""
    from gateway import local_viewer_events, session_control_actions

    seen = []

    async def _fake_offloop(authority, session_id):
        seen.append(session_id)

    monkeypatch.setattr(session_control_actions, 'publish_if_changed_offloop', _fake_offloop)
    scheduled = []

    class _Turn:
        _approval_owner = (object(), 'session-1', 0)

        def _schedule(self, coro, _label):
            scheduled.append(coro)

    local_viewer_events.publish_control_if_changed(_Turn())
    assert len(scheduled) == 1, 'the repaint is scheduled, not awaited inline'
    asyncio.run(scheduled[0])
    assert seen == ['session-1']
    assert timing.snapshot()['control_repaint']['count'] == 1


def test_snapshot_is_a_copy_and_reset_clears():
    timing.record('watcher', 0.5)
    view = timing.snapshot()
    view['watcher']['count'] = 99
    assert timing.snapshot()['watcher']['count'] == 1
    timing.reset()
    assert timing.snapshot() == {}
