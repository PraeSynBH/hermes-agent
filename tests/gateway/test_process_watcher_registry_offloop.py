"""The background-process watcher must not read the process registry on the gateway loop.

``_run_process_watcher`` polls ``process_registry.get`` every tick. In steady state that is a dict
lookup, but a session no longer in either store — the tick that closes a finished watcher — falls
through to ``load_completed_results``: a scan of the retained-results directory and a JSON read per
file. One of the fourteen blocked-loop dumps in PRX-5524 sampled the loop inside exactly that stack
(``run_notifications._run_process_watcher`` -> ``tools.process_registry.get``).

Contract: the lookup runs on a worker thread. The watcher's own judgement (surfacing, delivery,
injection) stays on the loop, and the loop keeps serving other profiles while the read waits.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading

import pytest

from gateway.run_notifications import GatewayNotificationsMixin


class _WatcherHost(GatewayNotificationsMixin):
    """Only the collaborators ``_run_process_watcher`` touches before its first registry read."""

    def _completion_event_scope(self, _watcher):
        return contextlib.nullcontext()

    def _load_background_notifications_mode(self):
        return "off"


def _watcher(interval=0):
    return {"session_id": "proc_deadbeef", "check_interval": interval, "platform": "telegram",
            "chat_id": "123"}


@pytest.fixture
def registry_calls(monkeypatch):
    """Patch the registry where production imports it (``tools.process_registry.process_registry``)."""
    import tools.process_registry as registry_module

    calls = []

    class _Registry:
        def get(self, session_id):
            # None = a finished process: the watcher closes on this tick.
            calls.append((threading.get_ident(), session_id))

    monkeypatch.setattr(registry_module, "process_registry", _Registry())
    return calls


def test_process_watcher_reads_the_registry_off_loop(registry_calls):
    loop_thread = threading.get_ident()
    asyncio.run(_WatcherHost()._run_process_watcher(_watcher()))

    ((read_thread, session_id),) = registry_calls
    assert session_id == "proc_deadbeef"
    assert read_thread != loop_thread, (
        "process_registry.get ran on the event loop; its retained-results fallback is the PRX-5524 stall")


def test_process_watcher_still_closes_when_the_session_is_gone(registry_calls):
    """The off-loop hop changes where the read runs, not what the watcher does with it."""
    asyncio.run(asyncio.wait_for(_WatcherHost()._run_process_watcher(_watcher()), timeout=5))

    assert len(registry_calls) == 1, "a watcher whose process is gone must stop after one lookup"
