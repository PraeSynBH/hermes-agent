"""Regression: #71671 — gateway must survive faulthandler.enable() with sys.stderr=None.

Plus PRA-6536: the SIGUSR2 stack-dump hook must not walk *every* thread's C state from signal
context. That walk (``_Py_DumpTracebackThreads``) took the shared gateway down on 2026-10-19 —
pid 67785, 240 threads, EXC_BAD_ACCESS inside the walk, exit 139, six in-flight runs lost — and
the crash stack names this registration, not a caller: ``faulthandler_user`` ->
``faulthandler_dump_traceback`` -> ``_Py_DumpTracebackThreads``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


_SIGUSR2_CHILD = """
import os, signal, sys, threading, time
from types import SimpleNamespace
sys.path.insert(0, sys.argv[2])
from gateway.run_startup import GatewayStartupMixin
runner = object.__new__(GatewayStartupMixin)
runner.config = SimpleNamespace(log_dir=sys.argv[1])
runner._start_install_faulthandler()
# A second live thread, so an all-thread dump would have something else to walk.
stop = threading.Event()
threading.Thread(target=stop.wait, name="bystander", daemon=True).start()
print("ready", flush=True)
time.sleep(30)
"""


@pytest.mark.platforms("posix")
def test_sigusr2_stack_dump_leaves_the_gateway_running(tmp_path):
    """#110437 — SIGUSR2 must dump stacks to the log and return; chaining to the
    signal's default disposition (terminate) killed the process being inspected.

    PRA-6536 — and the dump must cover only the receiving thread: an all-thread walk from a
    signal handler is the crash class that killed pid 67785.
    """
    import os
    import signal
    import subprocess
    import time

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    child = subprocess.Popen(
        [sys.executable, "-c", _SIGUSR2_CHILD, str(tmp_path), repo_root],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        child.send_signal(signal.SIGUSR2)
        time.sleep(0.5)
        assert child.poll() is None, f"gateway died on SIGUSR2 (rc={child.returncode})"
    finally:
        child.kill()
        child.wait()
    dump = (tmp_path / "gateway_faulthandler.log").read_text(encoding="utf-8")
    assert "Stack (most recent call first):" in dump
    # "Thread 0x…" blocks are faulthandler's per-other-thread walk — the C path that faulted.
    assert "Thread 0x" not in dump, f"SIGUSR2 walked every thread state:\n{dump[:2000]}"


@pytest.mark.platforms("posix")
def test_sigusr2_hook_is_registered_for_the_current_thread_only(tmp_path, monkeypatch):
    """The registration itself is the contract: all_threads=False, chain=False."""
    import faulthandler

    from gateway import run_startup

    registered = []
    monkeypatch.setattr(
        faulthandler, "register", lambda *args, **kwargs: registered.append((args, kwargs))
    )

    runner = object.__new__(run_startup.GatewayStartupMixin)
    runner.config = SimpleNamespace(log_dir=str(tmp_path))
    # Keep the real log opener out of it: it late-imports gateway.run, whose import touches the
    # live checkout (tests/home_io_guard flags that), and this test only cares about the kwargs.
    monkeypatch.setattr(runner, "_open_faulthandler_log", lambda: tmp_path / "fh.log")
    runner._start_install_faulthandler()

    assert registered, "the SIGUSR2 stack-dump hook must still be installed"
    _, kwargs = registered[0]
    assert kwargs["all_threads"] is False
    assert kwargs["chain"] is False
