"""The after-each-tool control repaint must not read SQLite on the gateway loop.

``publish_control_if_changed`` runs once per tool call of every live turn, and its snapshot read is
SQLite work (the compression tip plus three ``get_meta`` reads). Three of the fourteen blocked-loop
dumps in PRX-5524 sampled the loop inside exactly that stack
(``local_viewer_events.republish`` -> ``session_control_actions.publish`` ->
``session_ancillary.control_snapshot`` -> ``hermes_state.get_meta``), which is the shape that
produces *continuous* starvation rather than a burst.

Contract: the read moves to a worker thread; the revision gate and the publish stay on the loop, so
viewers still receive updates in the same order and an unchanged revision still publishes nothing.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from gateway import session_control_actions as actions


class _EventStream:
    def __init__(self):
        self.published = []
        self.thread = None

    def publish(self, session_id, payload, event_type=None):
        self.thread = threading.get_ident()
        self.published.append((session_id, event_type, payload))


class _Live:
    def __init__(self):
        self.control_revision = 0
        self.event_stream = _EventStream()


class _Authority:
    profile_id = "default"

    def __init__(self, live):
        self.sessions = {"s1": live} if live is not None else {}
        self.db = object()


class _SnapshotLog:
    """What the patched read saw, and the revision it should return next."""

    def __init__(self):
        self.calls = []
        self.revision = 7


@pytest.fixture
def snapshot_calls(monkeypatch):
    """Patch the read where production reads it (``session_ancillary.control_snapshot``)."""
    import gateway.session_ancillary as ancillary

    log = _SnapshotLog()

    def _control_snapshot(authority, ref):
        log.calls.append((threading.get_ident(), ref.session_id))
        return {"goal": None, "loop": None, "heartbeat": None,
                "revision": log.revision, "updated_at": None}

    monkeypatch.setattr(ancillary, "control_snapshot", _control_snapshot)
    return log


def test_after_tool_repaint_reads_off_loop_and_publishes_on_the_loop(snapshot_calls):
    live = _Live()
    authority = _Authority(live)
    loop_thread = threading.get_ident()

    from gateway.local_viewer_events import publish_control_if_changed

    scheduled = []

    class _Turn:
        _approval_owner = (authority, "s1", 1)

        @staticmethod
        def _schedule(coro, label):
            scheduled.append((coro, label))

    publish_control_if_changed(_Turn())
    assert [label for _c, label in scheduled] == ["session.control.update after a tool"]
    asyncio.run(scheduled[0][0])

    ((read_thread, session_id),) = snapshot_calls.calls
    assert session_id == "s1"
    assert read_thread != loop_thread, (
        "the control snapshot was read on the event loop; that SQLite read is the PRX-5524 stall")
    assert live.event_stream.thread == loop_thread, (
        "the publish must stay on the loop so it keeps its order against the slash-command publishes")
    assert [event for _s, event, _p in live.event_stream.published] == ["session.control.update"]
    assert live.control_revision == 7


def test_unchanged_revision_publishes_nothing_and_a_new_one_publishes(snapshot_calls):
    live = _Live()
    authority = _Authority(live)

    asyncio.run(actions.publish_if_changed_offloop(authority, "s1"))
    assert len(live.event_stream.published) == 1

    # Same revision: the gate must swallow it, or every tool of every turn repaints every viewer.
    asyncio.run(actions.publish_if_changed_offloop(authority, "s1"))
    assert len(live.event_stream.published) == 1

    snapshot_calls.revision = 9
    asyncio.run(actions.publish_if_changed_offloop(authority, "s1"))
    assert len(live.event_stream.published) == 2
    assert live.control_revision == 9


def test_sync_publish_keeps_its_own_gates(snapshot_calls):
    """The inline flavour still serves a session with no automation state, or no updates at all."""
    live = _Live()
    authority = _Authority(live)

    snapshot_calls.revision = 0  # a plain turn: no goal/loop/heartbeat state
    actions.publish(authority, "s1", only_if_present=True)
    assert live.event_stream.published == []

    snapshot_calls.revision = 3
    actions.publish(authority, "s1", only_if_present=True)
    actions.publish(authority, "s1", only_if_present=True, only_if_changed=True)
    assert len(live.event_stream.published) == 1
