"""The background-process watcher must not pay a thread hop per poll hop.

``_run_process_watcher`` polls the process registry every tick. It used to do that with
``asyncio.to_thread(process_registry.get, session_id)`` on every hop, which costs the loop a pool
submit, a new OS thread when the pool is saturated, and the GIL hand-back — for a lookup that is a
dict read in steady state. PRX-5524's dumps found the loop inside ``process_registry.get`` because
that method's *miss* path scans the retained results (a directory scan plus a JSON read per file);
PRA-6825 then found the loop sitting in ``ThreadPoolExecutor.submit`` for 19 consecutive one-second
samples while the registry was warm.

Contract, two halves:
  * warm — the read is ``process_registry.peek`` (memory only) on the loop: no submit, no thread;
  * miss  — a session no longer in either store still falls to ``get`` on a worker thread, because
    only that path touches the filesystem.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading

import pytest

from gateway.run_notifications import GatewayNotificationsMixin


class _WatcherHost(GatewayNotificationsMixin):
    """Only the collaborators ``_run_process_watcher`` touches before its first registry read."""

    def _completion_event_scope(self, _watcher):
        return contextlib.nullcontext()

    def _load_background_notifications_mode(self):
        return "off"  # silent: the watcher keeps polling until the process exits


def _watcher(interval=0):
    return {"session_id": "proc_deadbeef", "check_interval": interval, "platform": "telegram",
            "chat_id": "123"}


class _Session:
    """The attributes the poll loop reads before it decides to keep polling."""

    def __init__(self):
        self.exited = False
        self.output_buffer = ""
        self.owner_task_id = ""
        self.task_id = ""


class _WarmRegistry:
    """Live in memory for ``hops`` reads, then reports the process exited on the next one."""

    def __init__(self, hops):
        self.session = _Session()
        self.hops = hops
        self.peeks: list[tuple[int, str]] = []
        self.gets: list[tuple[int, str]] = []

    def peek(self, session_id):
        self.peeks.append((threading.get_ident(), session_id))
        if len(self.peeks) > self.hops:
            self.session.exited = True
        return self.session

    def get(self, session_id):  # the disk fallback: must not be reached while warm
        self.gets.append((threading.get_ident(), session_id))


class _ColdRegistry:
    """Nothing in memory and nothing retained: the miss path."""

    def __init__(self):
        self.peeks: list[tuple[int, str]] = []
        self.gets: list[tuple[int, str]] = []

    def peek(self, session_id):
        self.peeks.append((threading.get_ident(), session_id))

    def get(self, session_id):
        self.gets.append((threading.get_ident(), session_id))


@pytest.fixture
def use_registry(monkeypatch):
    """Install a fake where production imports the registry."""
    import tools.process_registry as registry_module

    def install(registry):
        monkeypatch.setattr(registry_module, "process_registry", registry)
        return registry

    return install


@pytest.fixture
def pool_submits(monkeypatch):
    """Every ``ThreadPoolExecutor.submit`` — the hop ``to_thread`` pays for."""
    submits = []
    real = concurrent.futures.ThreadPoolExecutor.submit

    def counting(self, fn, /, *args, **kwargs):
        submits.append(fn)
        return real(self, fn, *args, **kwargs)

    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "submit", counting)
    return submits


def test_warm_poll_hops_do_not_hop_to_a_thread(use_registry, pool_submits):
    loop_thread = threading.get_ident()
    registry = use_registry(_WarmRegistry(hops=5))

    asyncio.run(asyncio.wait_for(_WatcherHost()._run_process_watcher(_watcher()), timeout=10))

    assert len(registry.peeks) == 6, "the watcher stopped polling before it saw the exit"
    assert registry.gets == [], "a warm registry read fell through to the disk-backed get"
    assert pool_submits == [], (
        "the warm read still paid a ThreadPoolExecutor.submit per poll hop — the cost this card "
        "removes")
    assert {ident for ident, _ in registry.peeks} == {loop_thread}, (
        "the in-memory read left the event loop")


def test_a_missed_session_is_read_off_loop(use_registry, pool_submits):
    """PRX-5524's stall: ``get``'s retained-results scan must stay off the loop."""
    loop_thread = threading.get_ident()
    registry = use_registry(_ColdRegistry())

    asyncio.run(asyncio.wait_for(_WatcherHost()._run_process_watcher(_watcher()), timeout=10))

    ((read_thread, session_id),) = registry.gets
    assert session_id == "proc_deadbeef"
    assert read_thread != loop_thread, (
        "process_registry.get ran on the event loop; its retained-results fallback is the "
        "PRX-5524 stall")
    assert len(pool_submits) == 1, "exactly one hop: the miss, not the poll"


def test_process_watcher_still_closes_when_the_session_is_gone(use_registry):
    """The hoist changes where the read runs, not what the watcher does with it."""
    registry = use_registry(_ColdRegistry())

    asyncio.run(asyncio.wait_for(_WatcherHost()._run_process_watcher(_watcher()), timeout=10))

    assert len(registry.gets) == 1, "a watcher whose process is gone must stop after one lookup"
