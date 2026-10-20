"""Unified deadline layer — one bounded-execution primitive, one timeout resolver (#85125).

* :func:`resolve_timeout` — ``timeouts:`` in config.yaml > legacy env var > default.
* :func:`clamp_timeout` — huge timeouts overflow ``time_t`` in ``Lock.acquire`` /
  ``Thread.join`` on macOS (#83220), so every timeout is capped.
* :func:`run_bounded_async` / :func:`run_bounded_sync` — wall-clock deadlines driven by
  a daemon ``threading.Timer`` / pooled worker, so a blocked event loop cannot disable them.
* :func:`kill_process_tree` — portable whole-tree termination.

Invariants: operation exceptions propagate unchanged (only the *timeout* outcome is reified
as :class:`BoundedResult`); a timeout here is OUR deadline, not the provider's (classify
:class:`DeadlineExpired` distinctly from transport timeouts); ``None`` / non-positive means unbounded.
"""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_SAFE_TIMEOUT_S",
    "MAX_ABANDONED_WORKERS",
    "BoundedResult",
    "BoundedWorkersExhausted",
    "DeadlineExpired",
    "clamp_timeout",
    "kill_process_tree",
    "resolve_timeout",
    "run_bounded_async",
    "run_bounded_sync",
    "sync_worker_census",
]

# Upper bound for any timeout handed to platform wait primitives.
#
# CPython converts ``threading.Lock.acquire(timeout=...)`` /
# ``Thread.join(timeout=...)`` deadlines to an absolute timestamp; very large
# relative timeouts overflow ``time_t`` on macOS and raise
# ``OverflowError: timestamp out of range for platform time_t`` (#83220).
# On Windows the binding constraint is narrower: the underlying
# ``WaitForSingleObject`` takes a DWORD *milliseconds* argument, so any
# wait above ~49.7 days (0xFFFFFFFF ms) raises OverflowError rather than
# waiting (verified live: 4294967.0s ok, 4294967.3s overflows). One year
# is the semantic "unbounded" on platforms that can express it; Windows
# takes the DWORD-ms cap, which is still semantically "unbounded" for
# every wait in this codebase.
_WINDOWS_MAX_WAIT_S = (0xFFFFFFFF - 1000) / 1000.0  # DWORD ms, minus a 1s rounding guard
MAX_SAFE_TIMEOUT_S = (
    _WINDOWS_MAX_WAIT_S if sys.platform == "win32" else 31_536_000.0
)  # 365 days off-Windows

# Grace after a deadline fires before concluding the loop thread is blocked and dumping stacks.
_LOOP_BLOCKED_DUMP_GRACE_S = 5.0

# ``Event.wait`` is a C-level block: KeyboardInterrupt / SetAsyncExc only land when the
# thread returns to Python, so the sync wait is sliced to observe /stop or SIGINT promptly.
# Slice the wait so a /stop or SIGINT during a bounded sync call is observed within this window rather than
# at the full deadline (#94285, tools/test_local_interrupt_cleanup).
_BOUNDED_SYNC_WAIT_SLICE_S = 0.2


class DeadlineExpired(TimeoutError):
    """A deadline enforced by this layer expired (Hermes's own bound, not the provider's)."""

    def __init__(self, label: str, timeout_s: float):
        super().__init__(f"deadline expired after {timeout_s:.1f}s: {label}")
        self.label = label
        self.timeout_s = timeout_s


class SuspectableBackend(Protocol):
    """A stateful backend (MCP connection, browser session, LSP client) ``run_bounded_*`` flags via
    ``mark_suspect`` on timeout so the owner can health-check/recycle it before reuse. ``mark_suspect``
    MUST be cheap, non-blocking, and must not acquire locks the guarded operation may hold — it runs
    inline on the event loop / caller's thread while the wedged worker is still alive."""

    def mark_suspect(self, reason: str) -> None: ...

    def ensure_healthy(self) -> bool: ...


def _mark_backend_suspect(backend: object | None, label: str, timeout_s: float) -> None:
    """Best-effort ``mark_suspect``; never raises, non-adopting backends tolerated."""
    if backend is None:
        return
    try:
        mark = getattr(backend, "mark_suspect", None)
        if callable(mark):
            mark(f"{label} timed out after {timeout_s:.1f}s")
    except Exception:
        logger.debug("deadline mark_suspect failed", exc_info=True)


@dataclass(frozen=True, kw_only=True)
class BoundedResult:
    """Outcome of a bounded operation; operation exceptions are never captured here."""

    timed_out: bool
    value: Any
    elapsed_s: float
    timeout_s: Optional[float]
    label: str


def _result(start: float, timeout_s: Optional[float], label: str, *, value: Any = None, timed_out: bool = False) -> BoundedResult:
    return BoundedResult(
        timed_out=timed_out, value=value, elapsed_s=time.monotonic() - start, timeout_s=timeout_s, label=label
    )


def clamp_timeout(timeout: Optional[float]) -> Optional[float]:
    """Normalize a timeout: None/non-positive/non-numeric/NaN -> None (unbounded), else capped."""
    if timeout is None:
        return None
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        logger.warning("clamp_timeout: non-numeric timeout %r; treating as unbounded", timeout)
        return None
    if value != value:  # NaN
        logger.warning("clamp_timeout: NaN timeout; treating as unbounded")
        return None
    return None if value <= 0 else min(value, MAX_SAFE_TIMEOUT_S)


# --- Timeout resolution: config ``timeouts:`` > legacy env var > default ------


def _timeouts_section() -> dict:
    """Read the ``timeouts:`` root section from config.yaml (read-only, fail-open)."""
    try:
        from hermes_cli.config import load_config_readonly
        section = load_config_readonly().get("timeouts")
        return section if isinstance(section, dict) else {}
    except Exception:
        logger.debug("timeouts: config read failed; using defaults", exc_info=True)
        return {}


def _lookup_dotted(section: dict, key: str) -> Any:
    """Walk ``a.b.c`` through nested dicts; return None when absent."""
    node: Any = section
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def resolve_timeout(key: str, *, default: Optional[float], env_var: Optional[str] = None) -> Optional[float]:
    """Resolve a timeout (seconds): dotted ``timeouts.<key>`` > ``env_var`` > ``default``; the winner
    goes through :func:`clamp_timeout`, invalid config/env values fall through with a warning."""
    raw = _lookup_dotted(_timeouts_section(), key)
    if raw is not None:
        # Explicit float() so invalid config values FALL THROUGH to env/default instead of
        # resolving as unbounded. bool rejected (YAML `true` would become a 1s deadline);
        # NaN rejected for the same fall-through reason.
        if not isinstance(raw, bool):
            try:
                value = float(raw)
                if value == value:  # not NaN
                    return clamp_timeout(value)
            except (TypeError, ValueError):
                pass
        logger.warning("timeouts.%s: invalid value %r in config.yaml; ignoring", key, raw)

    if env_var:
        env_raw = os.getenv(env_var, "").strip()
        if env_raw:
            try:
                return clamp_timeout(float(env_raw))
            except ValueError:
                logger.warning("invalid %s=%r; ignoring", env_var, env_raw)

    return clamp_timeout(default)


# --- Bounded execution — async flavor ------------------------------------------
# The deadline is a daemon threading.Timer so a blocked event loop cannot disable it; a
# second timer dumps all thread stacks when the loop provably failed to process the expiry.


# --------------------------------------------------------------------------- Bounded execution — async
# flavor. Generalizes plugins/platforms/telegram/adapter.py:_await_with_thread_deadline (the #63309 fix):
# the deadline is driven by a daemon threading.Timer so a blocked event loop cannot disable it, and a second
# timer dumps all thread stacks when the loop provably failed to process the expiry — the one piece of
# information loop-blocked hangs otherwise never surface.
# ---------------------------------------------------------------------------
def _consume_abandoned(task: asyncio.Future[Any]) -> None:
    """Observe an abandoned task's outcome so it never logs 'never retrieved'."""
    try:
        if not task.cancelled():
            task.exception()
    except Exception:
        pass


def _abandon(task: asyncio.Future[Any]) -> None:
    """Cancel ``task`` and never await it; its outcome is consumed so it stays unobserved-safe."""
    task.cancel()
    task.add_done_callback(_consume_abandoned)


async def _run_abandon_cleanup(on_abandon: Callable[[], Awaitable[Any]]) -> None:
    """Run abandonment cleanup fire-and-forget (its failures swallowed)."""
    try:
        await on_abandon()
    except Exception:
        logger.debug("deadline abandon-cleanup failed", exc_info=True)


def _dump_blocked_loop_diagnostics(label: str, timeout_s: float) -> None:
    logger.warning(
        "[deadline] %r deadline (%.0fs) expired but the event loop has not processed the expiry "
        "after a further %.0fs — the loop thread appears BLOCKED in a synchronous call, which is "
        "why no asyncio timeout can fire. Dumping bounded thread stacks to stderr to identify the "
        "blocking frame.",
        label, timeout_s, _LOOP_BLOCKED_DUMP_GRACE_S,
    )
    # Bounded, rate-limited, pure-Python: the C-level all-thread walk that used to run here
    # faulted in _Py_DumpTracebackThreads and killed the gateway (PRA-6536).
    from hermes_thread_dump import dump_thread_stacks

    dump_thread_stacks(f"deadline expired with the loop blocked: {label!r}")


async def run_bounded_async(
    awaitable: Awaitable[Any],
    timeout: Optional[float],
    *,
    label: str = "operation",
    on_abandon: Optional[Callable[[], Awaitable[Any]]] = None,
    dump_on_blocked_loop: bool = True,
    backend: object | None = None,
) -> BoundedResult:
    """Await ``awaitable`` under a wall-clock deadline independent of loop timers.

    Operation exceptions (incl. ``CancelledError`` from a caller cancelling *us*) propagate
    unchanged. On timeout the task is cancelled and **abandoned** (never awaited —
    cancellation-shielded scopes are exactly the paths that wedge); ``on_abandon`` runs detached."""
    timeout_s = clamp_timeout(timeout)
    start = time.monotonic()
    if timeout_s is None:
        return _result(start, None, label, value=await awaitable)

    task = asyncio.ensure_future(awaitable)
    loop = asyncio.get_running_loop()
    deadline: asyncio.Future[None] = loop.create_future()
    loop_processed_expiry = threading.Event()

    def _mark_expired() -> None:
        loop_processed_expiry.set()
        if not deadline.done():
            deadline.set_result(None)

    def _watchdog_check() -> None:
        if not loop_processed_expiry.is_set():
            _dump_blocked_loop_diagnostics(label, timeout_s)

    timers = [threading.Timer(timeout_s, lambda: loop.call_soon_threadsafe(_mark_expired))]
    if dump_on_blocked_loop:
        timers.append(threading.Timer(timeout_s + _LOOP_BLOCKED_DUMP_GRACE_S, _watchdog_check))
    for t in timers:
        t.daemon = True
        t.start()
    try:
        try:
            done, _ = await asyncio.wait({task, deadline}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            _abandon(task)  # the CALLER cancelled us; `task` must not run unobserved
            raise
        if task in done:
            if not deadline.done():
                deadline.cancel()
            return _result(start, timeout_s, label, value=await task)

        _abandon(task)
        if on_abandon is not None:
            asyncio.ensure_future(_run_abandon_cleanup(on_abandon)).add_done_callback(_consume_abandoned)
        # Deliberately INLINE on the loop: the mark must happen-before this result returns
        # AND before the ensure_future'd on_abandon cleanup starts (next tick).
        _mark_backend_suspect(backend, label, timeout_s)
        logger.warning("[deadline] %r timed out after %.1fs; task abandoned", label, timeout_s)
        return _result(start, timeout_s, label, timed_out=True)
    finally:
        for t in timers:
            t.cancel()
        # cancel() cannot stop a Timer whose callback is already running; setting the
        # event closes that race so a completed await is never misreported as blocked.
        loop_processed_expiry.set()


# --- Bounded execution — sync flavor -------------------------------------------
#
# A thread cannot be killed: a ``fn`` that never returns keeps its worker alive forever.
# One thread per call made every expiry a permanent leak — the shared gateway logged 4,723
# ``worker abandoned`` lines (554 inside one 5 h pid, plus 90 live ``deadline-*`` threads)
# and each leaked worker kept taking the GIL, starving the event loop. Workers are pooled
# and reused instead, and the *abandoned* population is capped: at the cap a new timed call
# fails closed with a retryable :class:`BoundedWorkersExhausted` rather than leaking one more
# thread. A wedged worker that finally returns drops the abandoned count and rejoins the pool.

# Concurrent expired calls allowed to still be running. Above this the host is already
# wedged; refusing the new call (which the caller may retry) beats another stripped thread.
MAX_ABANDONED_WORKERS = 8
# An idle pool thread retires after this long, so a burst does not pin a pool full of them.
_WORKER_IDLE_EXIT_S = 15.0


class BoundedWorkersExhausted(TimeoutError):
    """A bounded sync call was refused: ``MAX_ABANDONED_WORKERS`` abandoned workers are still
    wedged. Retryable — capacity returns when one of them finishes."""

    def __init__(self, label: str, abandoned: int):
        super().__init__(
            f"bounded worker pool exhausted: {abandoned} abandoned worker(s) still running; "
            f"{label!r} was not started"
        )
        self.label = label
        self.abandoned = abandoned


@dataclass
class _SyncJob:
    """One ``run_bounded_sync`` unit: the callable, its context, its outcome box."""

    fn: Callable[[], Any]
    ctx: contextvars.Context
    box: dict[str, Any] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)
    finished: bool = False
    abandoned: bool = False


class _DeadlineWorker(threading.Thread):
    """Pool thread: takes one job at a time, waits up to ``_WORKER_IDLE_EXIT_S`` when idle."""

    def __init__(self, pool: "_SyncWorkerPool"):
        super().__init__(name="deadline-worker", daemon=True)
        self._pool = pool
        self._cond = threading.Condition(pool._lock)
        self._job: Optional[_SyncJob] = None

    def _next_job(self) -> Optional[_SyncJob]:
        """Block under the pool lock until a job arrives or the idle TTL expires."""
        with self._pool._lock:
            while self._job is None:
                if not self._cond.wait(_WORKER_IDLE_EXIT_S):
                    # A submit that won the lock just as the wait expired still owns this worker.
                    if self._job is not None:
                        break
                    self._pool._retire(self)
                    return None
            job, self._job = self._job, None
            return job

    def run(self) -> None:
        while True:
            job = self._next_job()
            if job is None:
                return
            try:
                job.box["value"] = job.ctx.run(job.fn)
            except BaseException as exc:  # re-raised in caller; must not vanish
                job.box["exc"] = exc
            finally:
                self._pool._finish(self, job)


class _SyncWorkerPool:
    """Reusable bounded-sync workers with a cap on abandoned (still-running) jobs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._idle: list[_DeadlineWorker] = []
        self._threads = 0
        self._abandoned = 0
        self._pid = os.getpid()

    def _bind_pid(self) -> None:
        """A forked child does not own the parent's worker threads: start from empty. (Lock held.)"""
        pid = os.getpid()
        if pid != self._pid:
            self._idle = []
            self._threads = 0
            self._abandoned = 0
            self._pid = pid

    def submit(self, job: _SyncJob, *, label: str) -> None:
        with self._lock:
            self._bind_pid()
            if self._abandoned >= MAX_ABANDONED_WORKERS:
                raise BoundedWorkersExhausted(label, self._abandoned)
            worker = self._idle.pop() if self._idle else None
            if worker is None:
                worker = _DeadlineWorker(self)
                self._threads += 1
                worker.start()
            worker._job = job
            worker._cond.notify()

    def claim_expiry(self, job: _SyncJob) -> bool:
        """True when the caller owns the timeout outcome (and has counted the abandonment)."""
        with self._lock:
            if job.finished:
                return False
            job.abandoned = True
            self._abandoned += 1
            return True

    def _finish(self, worker: _DeadlineWorker, job: _SyncJob) -> None:
        with self._lock:
            job.finished = True
            if job.abandoned:
                if self._abandoned > 0:
                    self._abandoned -= 1
                job.abandoned = False
            job.done.set()
            self._idle.append(worker)

    def _retire(self, worker: _DeadlineWorker) -> None:
        """Idle TTL expired: drop the thread from the pool's accounting. (Pool lock held.)"""
        with self._lock:
            if worker in self._idle:
                self._idle.remove(worker)
            if self._threads > 0:
                self._threads -= 1

    def census(self) -> dict[str, int]:
        with self._lock:
            self._bind_pid()
            return {"threads": self._threads, "idle": len(self._idle), "abandoned": self._abandoned,
                    "cap": MAX_ABANDONED_WORKERS}


_SYNC_POOL = _SyncWorkerPool()


def sync_worker_census() -> dict[str, int]:
    """Live pool counters (``threads``/``idle``/``abandoned``/``cap``) for diagnostics and tests."""
    return _SYNC_POOL.census()


def run_bounded_sync(
    fn: Callable[[], Any],
    timeout: Optional[float],
    *,
    label: str = "operation",
    on_timeout: Optional[Callable[[], None]] = None,
    backend: object | None = None,
) -> BoundedResult:
    """Run ``fn`` on a pooled daemon worker under a wall-clock deadline; exceptions re-raise in
    the caller. On expiry the call is **abandoned** — its worker keeps running but the abandoned
    population is capped at :data:`MAX_ABANDONED_WORKERS`; at the cap this raises
    :class:`BoundedWorkersExhausted` (retryable) instead of leaking a thread. ``on_timeout`` runs
    best-effort in the caller's thread. The worker runs under ``contextvars.copy_context()`` so
    secret scope / session id survive.

    See #94285.
    """
    timeout_s = clamp_timeout(timeout)
    start = time.monotonic()
    if timeout_s is None:
        return _result(start, None, label, value=fn())

    job = _SyncJob(fn=fn, ctx=contextvars.copy_context())
    _SYNC_POOL.submit(job, label=label)
    deadline = start + timeout_s
    while not job.done.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        job.done.wait(min(_BOUNDED_SYNC_WAIT_SLICE_S, remaining))

    if _SYNC_POOL.claim_expiry(job):
        logger.warning("[deadline] %r timed out after %.1fs; worker abandoned", label, timeout_s)
        # Mark suspect BEFORE owner cleanup so a recycle in on_timeout never
        # inherits a stale flag on the healed replacement.
        _mark_backend_suspect(backend, label, timeout_s)
        if on_timeout is not None:
            try:
                on_timeout()
            except Exception:
                logger.debug("deadline on_timeout callback failed", exc_info=True)
        return _result(start, timeout_s, label, timed_out=True)

    if "exc" in job.box:
        raise job.box["exc"]
    return _result(start, timeout_s, label, value=job.box.get("value"))


# --- Whole-tree process termination --------------------------------------------


@contextmanager
def _process_tree_snapshot(pid: int, *, hard_kill: bool):
    """Stop each hard-kill target before discovering its children: a running
    parent can fork after psutil builds its PID map and escape the final signal.
    Resume anything we stopped if signalling fails. Graceful signals never stop
    their recipients, since their handlers must remain able to run.
    """
    descendants = []
    stopped = []
    try:
        try:
            import psutil
            root = psutil.Process(pid)
            descendants = root.children(recursive=True)
            if hard_kill:
                pending = [root]
                seen = {os.getpid()}
                known = {process.pid: process for process in descendants}
                stop_deadline = time.monotonic() + 1.0
                while pending:
                    for process in pending:
                        if process.pid in seen:
                            continue
                        seen.add(process.pid)
                        if time.monotonic() >= stop_deadline:
                            raise TimeoutError("process tree did not stop before snapshot deadline")
                        try:
                            status = process.status()
                            if status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                                continue
                            if status != psutil.STATUS_STOPPED:
                                process.suspend()
                                stopped.append(process)
                            while process.status() != psutil.STATUS_STOPPED:
                                if time.monotonic() >= stop_deadline:
                                    raise TimeoutError("process did not stop before snapshot deadline")
                                time.sleep(0.001)
                        except psutil.NoSuchProcess:
                            continue
                    # Rescan after stopping the discovered generation. A child
                    # may have forked while that generation was being stopped.
                    for process in root.children(recursive=True):
                        known.setdefault(process.pid, process)
                    descendants = list(known.values())
                    pending = [process for process in descendants if process.pid not in seen]
        except Exception:
            # Preserve the existing best-effort group fallback when discovery or
            # stopping is unavailable; never strand a successfully stopped target.
            logger.debug("kill_process_tree: snapshot incomplete for pid %s", pid, exc_info=True)
        yield descendants
    finally:
        for process in stopped:
            try:
                process.resume()
            except Exception:
                logger.debug("kill_process_tree: target already gone or resume refused", exc_info=True)


def kill_process_tree(pid: int, *, sig: Optional[int] = None) -> bool:
    """Terminate ``pid`` and all its descendants, portably; True when anything was signalled.

    Windows: ``taskkill /F /T`` (``sig`` ignored). POSIX: snapshot descendants via
    psutil; for SIGKILL, stop and rescan the live tree so a concurrent fork cannot
    escape a stale snapshot. Signal identity-checked descendants before their
    parent, then its group when ``pid`` leads one. Stopping is best-effort with a
    bounded wait; unavailable psutil still leaves process-group cleanup. Other
    signals do not suspend recipients. ``sig`` defaults to ``SIGKILL``."""
    if sys.platform == "win32":
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags
            creationflags = windows_hide_flags()
        except Exception:
            creationflags = 0
        try:
            proc = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdin=subprocess.DEVNULL, capture_output=True, timeout=15, check=False, creationflags=creationflags,
            )
            # taskkill exits non-zero for not-found / access-denied (False = nothing terminated).
            return proc.returncode == 0
        except Exception:
            logger.debug("kill_process_tree: taskkill failed for pid %s", pid, exc_info=True)
            return False

    import signal as _signal
    if sig is None:
        sig = _signal.SIGKILL

    with _process_tree_snapshot(int(pid), hard_kill=sig == _signal.SIGKILL) as descendants:
        signalled = False
        # Signal descendants while their ownership ancestry is still observable.
        # Frozen hard-kill targets cannot fork during this bottom-up teardown.
        for child in reversed(descendants):
            try:
                if child.is_running():
                    child.send_signal(sig)
                    signalled = True
            except Exception:
                continue
        try:
            # getpgid→killpg has an inherent TOCTOU shared by every killpg site; the psutil
            # sweep below is identity-aware (PID + create time) and does not.
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError, OSError):
            pgid = None
        try:
            if pgid is not None and pgid == pid:
                # pid leads its own group (the == check avoids signalling the caller's group).
                os.killpg(pgid, sig)  # windows-footgun: ok — POSIX-only branch (win32 returns above)
            else:
                os.kill(pid, sig)
            signalled = True
        except ProcessLookupError:
            pass
        except (PermissionError, OSError):
            logger.debug("kill_process_tree: signal failed for pid %s", pid, exc_info=True)

        return signalled
