"""PRX-6104: the housekeeping tick is bounded, and it names the chore that spent it.

Pre-fix the tick was one ``timed('housekeeping_tick')`` wrapped around every due chore: p50 4.8 s of
every 60 s with a 17.8 s steady-state tail (PRX-6095, PRX-6112), and no per-chore figure — so "which
arm" stayed a guess. The arm that made the loop's latency the tick's cost was the channel-directory
refresh: it blocked ``fut.result(timeout=30)`` on the gateway loop.

What is pinned here is the behaviour, not the wording: a tick stops starting chores once it has spent
its ceiling and carries them, a carried chore runs first next tick (no starvation), the liveness arms
are exempt, the ceiling is published in the tick-cost line, and an arm waiting on a loop is bounded by
what is left of the tick instead of by a timeout of its own. The chores' cost comes from a fake clock
(the tick's own clock is ``run_housekeeping_chores.time``), so none of this depends on real time.
"""

from __future__ import annotations

import logging
import time

import pytest

from gateway import loop_tick_timing as timing
from gateway import run_housekeeping_chores as chores_mod


class _Clock:
    """Stand-in for ``run_housekeeping_chores.time``: a chore's cost is what the test advances."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def _fresh_state():
    timing.reset()
    chores_mod._channel_directory_inflight = None
    yield
    timing.reset()
    chores_mod._channel_directory_inflight = None


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(chores_mod, 'time', fake)
    return fake


def _chore(clock, seconds: float, marks: list, label: str):
    def _run() -> None:
        marks.append(label)
        clock.advance(seconds)
    return _run


def test_the_tick_split_names_the_arm_that_spent_it(clock, caplog):
    """The over-budget line is the whole product of ask 1: the dearest arm by name, plus the total."""
    caplog.set_level(logging.WARNING, logger='gateway.run')
    marks: list[str] = []
    chores = [(1, 'Channel directory refresh', _chore(clock, 0.2, marks, 'directory')),
              (1, 'Paste sweep', _chore(clock, 0.0, marks, 'paste')),
              (1, 'Runtime heartbeat', _chore(clock, 0.0, marks, 'heartbeat'))]

    chores_mod._run_due_chores(chores, 1, budget_s=0.1)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, 'an over-budget tick has to say so'
    line = warnings[0]
    assert line.startswith('housekeeping_tick over its 100ms budget: ')
    assert 'channel-directory-refresh=200ms' in line
    assert 'ms total (' in line
    # The arm that spent the tick reads before the cheap ones, and the chore it displaced is counted
    # rather than silently dropped.
    assert line.index('channel-directory-refresh=') < line.index('runtime-heartbeat=')
    assert 'carried=1' in line


def test_chores_past_the_ceiling_are_carried_to_the_next_tick(clock):
    """The ceiling is only a ceiling if the work it displaces is remembered, not dropped."""
    marks: list[str] = []
    chores = [(1, 'Busy', _chore(clock, 0.05, marks, 'busy')),
              (1, 'Cheap a', _chore(clock, 0.0, marks, 'a')),
              (1, 'Cheap b', _chore(clock, 0.0, marks, 'b'))]
    carried: dict[int, int] = {}

    chores_mod._run_due_chores(chores, 1, carried, budget_s=0.01)

    assert marks == ['busy']
    assert carried == {1: 1, 2: 1}

    # Next tick the displaced chores are due again anyway; the most-starved run first and their
    # defer counts clear.
    marks.clear()
    chores_mod._run_due_chores(chores, 2, carried, budget_s=1.0)
    assert marks == ['a', 'b', 'busy']
    assert carried == {}


def test_the_most_starved_chore_runs_first(clock):
    """Rotation is what keeps the ceiling from starving an arm forever: wait longer, run sooner."""
    marks: list[str] = []
    chores = [(1, 'a', _chore(clock, 0.0, marks, 'a')),
              (1, 'b', _chore(clock, 0.0, marks, 'b')),
              (1, 'c', _chore(clock, 0.0, marks, 'c'))]

    chores_mod._run_due_chores(chores, 1, {2: 2, 1: 1}, budget_s=5.0)

    assert marks == ['c', 'b', 'a']


def test_liveness_chores_run_even_when_the_ceiling_is_spent(clock):
    """A carried heartbeat would age gateway_state.json past the status warning (2x interval) while
    the tick is behaving exactly as designed, so the liveness arms are exempt from the ceiling."""
    marks: list[str] = []
    chores = [(1, 'Runtime heartbeat', _chore(clock, 0.0, marks, 'heartbeat')),
              (1, 'Busy', _chore(clock, 0.05, marks, 'busy'))]

    # Busy is carried in first, so the ceiling is already spent when the heartbeat comes up.
    chores_mod._run_due_chores(chores, 1, {1: 1}, budget_s=0.01)

    assert marks == ['busy', 'heartbeat']


def test_the_tick_publishes_its_budget_against_the_ceiling():
    """Ask 3: ``budget_used=<spent>/<ceiling> over=<window>/<total>`` beside the other watchers."""
    chores_mod._run_due_chores([(1, 'Cheap', lambda: None)], 1)

    view = timing.snapshot()['housekeeping_tick']
    assert view['budget_s'] == pytest.approx(chores_mod.HOUSEKEEPING_TICK_BUDGET_S)
    lines = timing.log_summary(now=time.monotonic() + timing.SUMMARY_INTERVAL_S + 1.0)
    assert lines and 'budget_used=' in lines[0] and ' over=' in lines[0]


def _stalled_future(captured: dict):
    class _Stalled:
        def add_done_callback(self, cb):
            captured.setdefault('callbacks', []).append(cb)

        def done(self):
            return False

        def result(self, timeout=None):
            captured['timeout'] = timeout
            raise TimeoutError

    return _Stalled()


def test_the_channel_directory_wait_is_clamped_to_the_tick_budget(monkeypatch):
    """The pre-fix arm waited 30 s on a loop it did not own: 15.9 s of a 16.4 s tick on this host."""
    from gateway import channel_directory

    captured: dict = {}
    stalled = _stalled_future(captured)

    def _fake_schedule(coro, loop, **_kwargs):
        coro.close()  # never awaited: the fake loop has not got to it
        return stalled

    async def _build(adapters):
        return {}

    monkeypatch.setattr(channel_directory, 'build_channel_directory', _build)
    monkeypatch.setattr(chores_mod, 'safe_schedule_threadsafe', _fake_schedule)
    chores = [(1, 'Channel directory refresh',
               lambda: chores_mod._housekeeping_channel_directory({'slack': object()}, object()))]

    chores_mod._run_due_chores(chores, 1, budget_s=0.5)

    assert 0 < captured['timeout'] <= 0.5, captured
    assert captured['timeout'] <= chores_mod._CHANNEL_DIRECTORY_WAIT_S


def test_a_refresh_still_in_flight_is_not_scheduled_twice(monkeypatch):
    """A build the tick stopped waiting for is still running on the loop; scheduling a second
    fan-out behind it only doubles the queue it is stuck in."""
    from gateway import channel_directory

    scheduled: list[object] = []

    async def _build(adapters):
        return {}

    def _fake_schedule(coro, loop, **_kwargs):
        coro.close()
        scheduled.append(coro)
        return _stalled_future({})

    monkeypatch.setattr(channel_directory, 'build_channel_directory', _build)
    monkeypatch.setattr(chores_mod, 'safe_schedule_threadsafe', _fake_schedule)

    chores_mod._housekeeping_channel_directory({'slack': object()}, object())
    chores_mod._housekeeping_channel_directory({'slack': object()}, object())

    assert len(scheduled) == 1
    assert chores_mod._channel_directory_inflight is not None
