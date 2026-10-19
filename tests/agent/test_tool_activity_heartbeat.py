"""Tests for the in-flight tool activity heartbeat (#84491).

The gateway's turn-inactivity watchdog
(``gateway/run.py::_watch_gateway_turn_inactivity``) abandons a turn once
``seconds_since_activity`` exceeds the inactivity timeout (default 30 min).
Activity was only stamped when a tool *started* and when it *completed*, so
a tool call that ran silently for 30+ minutes looked idle to the watchdog
and the turn was hard-abandoned mid-execution (processes reaped). The
the heartbeat in ``_run_agent_tool_execution_middleware`` stamps activity
periodically while a tool call is in flight.
"""

import json
import threading
import time
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate_hermes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)


def _make_agent(monkeypatch):
    """Minimal AIAgent-like stub, mirroring test_start_order_gate.py."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "")
    import run_agent as _ra

    class _Stub:
        _interrupt_requested = False
        _interrupt_message = None
        log_prefix = ""
        quiet_mode = True
        verbose_logging = False
        log_prefix_chars = 200
        _checkpoint_mgr = MagicMock(enabled=False)
        tool_progress_callback = None
        tool_start_callback = None
        tool_complete_callback = None
        tool_progress_mode = "off"
        _todo_store = MagicMock()
        _session_db = None
        valid_tool_names = set()
        _turns_since_memory = 0
        _iters_since_skill = 0
        _current_tool = None
        _last_activity = 0.0
        session_id = ""
        _current_turn_id = ""
        _current_api_request_id = ""

        def __init__(self):
            self._tool_worker_threads: set = set()
            self._tool_worker_threads_lock = threading.Lock()
            self._active_children_lock = threading.Lock()

        def _touch_activity(self, desc):
            self._last_activity = time.time()

        def _vprint(self, msg, force=False):
            pass

        def _safe_print(self, msg):
            pass

        def _should_emit_quiet_tool_messages(self):
            return False

        def _should_start_quiet_spinner(self):
            return False

        def _has_stream_consumers(self):
            return False

        def _tool_result_content_for_active_model(self, name, result):
            return result

        def _record_file_mutation_result(self, *a, **kw):
            pass

        def _apply_pending_steer_to_tool_results(self, *a, **kw):
            pass

    stub = _Stub()
    stub._subdirectory_hints = MagicMock()
    stub._subdirectory_hints.check_tool_call = lambda *a, **kw: None
    stub._flush_messages_to_session_db = lambda *a, **kw: None
    stub._append_guardrail_observation = lambda name, result, *a, **kw: result
    stub.interrupt = _ra.AIAgent.interrupt.__get__(stub)
    stub.clear_interrupt = _ra.AIAgent.clear_interrupt.__get__(stub)
    stub._guardrail_block_result = lambda d: json.dumps({"error": "blocked"})
    return stub


def _slow_execute(delay: float = 0.25):
    def _execute(next_args):
        time.sleep(delay)
        return json.dumps({"ok": True})

    return _execute


def test_heartbeat_touches_periodically_and_stops(monkeypatch):
    """A registered call is stamped on cadence, and registering stops when the call returns."""
    import agent.tool_executor as te

    monkeypatch.setattr(te, "_TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S", 0.05)

    touches: list = []
    repeated = threading.Event()
    release = threading.Event()

    class _Agent:
        def _touch_activity(self, desc):
            touches.append(desc)
            if len(touches) >= 2:
                repeated.set()

    thread = threading.Thread(
        target=te._run_with_activity_heartbeat,
        args=(_Agent(), "terminal", lambda: release.wait(10)),
        daemon=True,
    )
    thread.start()
    try:
        assert repeated.wait(5), "heartbeat did not repeat"
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive(), "tool call did not return"
    assert len(touches) >= 2, f"expected periodic touches, got {len(touches)}"
    assert all(touch == "tool running: terminal" for touch in touches)


def test_slow_tool_call_refreshes_activity_during_execution(monkeypatch):
    """A tool call running longer than one interval gets activity stamps.

    Before the fix, only the start stamp ("executing tool: X") and the
    completion stamp existed; a silent 30+ minute call left the clock
    frozen and the gateway watchdog abandoned the turn.
    """
    import agent.tool_executor as te

    monkeypatch.setattr(te, "_TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S", 0.05)

    agent = _make_agent(monkeypatch)
    agent._tool_guardrails = MagicMock(
        before_call=lambda name, args: MagicMock(allows_execution=True)
    )
    touches: list = []
    agent._touch_activity = lambda desc: touches.append(time.time())

    result = te._run_agent_tool_execution_middleware(
        agent,
        function_name="terminal",
        function_args={"command": "true"},
        effective_task_id="task",
        tool_call_id="tc1",
        execute=_slow_execute(delay=0.25),
        display_index=1,
    )

    assert json.loads(result.result) == {"ok": True}

    # Start stamp + at least one heartbeat mid-call (0.25s run, 0.05s cadence).
    assert len(touches) >= 3, f"expected mid-call heartbeats, got {len(touches)}"
    spread = touches[-1] - touches[0]
    assert spread >= 0.15, f"touches not spread across the call: {spread:.3f}s"


def test_fast_tool_call_does_not_leave_stray_heartbeat(monkeypatch):
    """A quick tool exits the heartbeat thread; no touches after return."""
    import agent.tool_executor as te

    monkeypatch.setattr(te, "_TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S", 0.05)

    agent = _make_agent(monkeypatch)
    agent._tool_guardrails = MagicMock(
        before_call=lambda name, args: MagicMock(allows_execution=True)
    )
    touches: list = []
    agent._touch_activity = lambda desc: touches.append(time.time())

    te._run_agent_tool_execution_middleware(
        agent,
        function_name="terminal",
        function_args={"command": "true"},
        effective_task_id="task",
        tool_call_id="tc1",
        execute=_slow_execute(delay=0.02),
        display_index=1,
    )

    n = len(touches)
    time.sleep(0.12)  # several heartbeat intervals
    assert len(touches) == n, "heartbeat thread kept running after tool returned"


def test_heartbeat_stops_when_execute_raises(monkeypatch):
    """If the tool call raises, the heartbeat thread still stops (no leak)."""

    import agent.tool_executor as te

    monkeypatch.setattr(te, "_TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S", 0.05)

    agent = _make_agent(monkeypatch)
    agent._tool_guardrails = MagicMock(
        before_call=lambda name, args: MagicMock(allows_execution=True)
    )
    touches: list = []
    agent._touch_activity = lambda desc: touches.append(time.time())

    def _boom(next_args):
        raise RuntimeError("tool exploded")

    with pytest.raises(RuntimeError):
        te._run_agent_tool_execution_middleware(
            agent,
            function_name="terminal",
            function_args={"command": "true"},
            effective_task_id="task",
            tool_call_id="tc1",
            execute=_boom,
            display_index=1,
        )

    n = len(touches)
    time.sleep(0.12)  # several heartbeat intervals
    assert len(touches) == n, "heartbeat thread kept running after execute() raised"


def test_concurrent_tool_call_heartbeat(monkeypatch):
    """Concurrent execution also stamps activity via the shared chokepoint."""
    import agent.tool_executor as te

    monkeypatch.setattr(te, "_TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S", 0.05)

    agent = _make_agent(monkeypatch)
    agent._tool_guardrails = MagicMock(
        before_call=lambda name, args: MagicMock(allows_execution=True)
    )
    touches: list = []
    agent._touch_activity = lambda desc: touches.append(time.time())

    agent._execute_tool_calls_concurrent = (
        __import__("run_agent").AIAgent._execute_tool_calls_concurrent.__get__(agent)
    )

    class _FakeToolCall:
        def __init__(self, name, call_id):
            self.function = MagicMock(name=name, arguments="{}")
            self.function.name = name
            self.id = call_id

    class _FakeAssistantMsg:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls

    def _invoke(name, *a, **kw):
        time.sleep(0.25)
        return json.dumps({"ok": name})

    agent._invoke_tool = MagicMock(side_effect=_invoke)

    msg = _FakeAssistantMsg([_FakeToolCall("tool_a", "tc_a")])
    messages: list = []
    agent._execute_tool_calls_concurrent(msg, messages, "task")

    assert len(touches) >= 3, f"expected mid-call heartbeats, got {len(touches)}"


def test_heartbeat_skips_and_drops_an_abandoned_worker():
    """The heartbeat must not speak for a worker the executor abandoned (#111922).

    ``_interrupt_worker_tids`` raises the interrupt bit on a wedged worker; that worker never
    reaches its ``finally``, so it cannot unregister. The scheduler must both stop stamping it
    and drop the registration, or the abandoned tool keeps the inactivity watchdog pinned (and
    its agent reference) for the rest of the run.
    """
    import agent.tool_executor as te
    from tools.interrupt import set_interrupt

    scheduler = te._heartbeat_scheduler
    live_touches: list = []
    fake_abandoned_tid = 10**9 + 111922  # not a live thread; only the bit matters

    class _LiveAgent:
        def _touch_activity(self, desc):
            live_touches.append(desc)

    class _AbandonedAgent:
        def _touch_activity(self, desc):  # pragma: no cover - must never fire
            raise AssertionError("stamped an abandoned worker")

    set_interrupt(True, fake_abandoned_tid)
    live_token = scheduler.register(_LiveAgent(), "tool running: terminal", threading.get_ident())
    dead_token = scheduler.register(_AbandonedAgent(), "tool running: terminal", fake_abandoned_tid)
    try:
        abandoned = scheduler._stamp(list(scheduler._waiters.items()))
        assert abandoned == [dead_token], "abandoned waiter was not identified"
        assert live_touches == ["tool running: terminal"]
    finally:
        scheduler.unregister(live_token)
        scheduler.unregister(dead_token)
        set_interrupt(False, fake_abandoned_tid)


def test_sequential_calls_never_accumulate_heartbeat_threads(monkeypatch):
    """N sequential tool calls share one heartbeat thread, even when it wedges.

    The old shape started a daemon thread *per call* and ``join(timeout=2.0)``d it, so an
    activity callback blocked past two seconds left the thread behind. Measured on the live
    gateway: 50 ``tool-activity-hb-*`` threads in one pid, all still runnable Python competing
    for the GIL. Sequential calls must never add a thread each.
    """
    import agent.tool_executor as te

    monkeypatch.setattr(te, "_TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S", 0.02)

    stamped = threading.Event()
    wedge = threading.Event()
    unwedged = threading.Event()

    class _Agent:
        def _touch_activity(self, desc):
            stamped.set()
            wedge.wait(10)  # longer than the old per-call join(2.0): the old thread is orphaned
            unwedged.set()

    def _call():
        # Wait for the heartbeat to have stamped (and wedged) so each call exercises the leak.
        assert stamped.wait(5), "shared heartbeat never stamped"
        return "ok"

    try:
        for _ in range(3):
            assert te._run_with_activity_heartbeat(_Agent(), "terminal", _call) == "ok"
        heartbeat_threads = [
            t for t in threading.enumerate() if t.name.startswith("tool-activity")
        ]
        assert len(heartbeat_threads) <= 1, (
            f"sequential tool calls accumulated {len(heartbeat_threads)} heartbeat threads")
    finally:
        wedge.set()
        assert unwedged.wait(5), "heartbeat thread stayed wedged after release"


def test_sequential_timeout_stops_abandoned_workers_heartbeat(monkeypatch):
    """After the sequential deadline abandons a non-cooperative tool, no more
    ``tool running:`` stamps arrive — the timed-out tool no longer fakes liveness (#111922)."""
    import agent.tool_executor as te

    monkeypatch.setattr(te, "_TOOL_ACTIVITY_HEARTBEAT_INTERVAL_S", 0.05)
    monkeypatch.setattr(te, "_SEQUENTIAL_INTERRUPT_POLL_SECONDS", 0.05)
    monkeypatch.setattr(te, "_resolve_sequential_tool_timeout", lambda: 0.3)
    monkeypatch.setattr(te, "_emit_terminal_post_tool_call", lambda agent, **kw: None)
    # The middleware's execution seam: ``_run_with_activity_heartbeat`` around ``execute`` on the worker.
    monkeypatch.setattr(
        te, "_run_agent_tool_execution_middleware",
        lambda agent, execute, function_name="terminal", **kw: te._ManagedToolResult(
            result=te._run_with_activity_heartbeat(agent, function_name, lambda: execute({})),
            args={}, middleware_trace=[], blocked=False, dispatched=True,
        ),
    )

    class _Agent:
        _interrupt_requested = False
        _tool_interrupt_reason = None

        def __init__(self):
            self._tool_worker_threads = set()
            self._tool_worker_threads_lock = threading.Lock()
            self.stamps: list = []

        def _touch_activity(self, desc):
            self.stamps.append((time.monotonic(), desc))

        def interrupt(self, *a, **k):
            pass

    agent = _Agent()
    release = threading.Event()
    try:
        managed = te._run_sequential_tool_execution_middleware(
            agent, function_name="terminal", function_args={}, effective_task_id="t",
            tool_call_id="tc1", execute=lambda args: release.wait() or "late",
        )
        abandoned_at = time.monotonic()
        assert isinstance(managed.result, te._ToolTimeoutResult)
        time.sleep(0.3)  # several heartbeat intervals past abandonment
        late = [d for ts, d in agent.stamps if ts > abandoned_at + 0.06 and d.startswith("tool running:")]
        assert not late, f"abandoned tool kept stamping activity: {late[:3]}"
    finally:
        release.set()
