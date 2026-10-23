"""Idle gates for the gateway's per-profile pollers (heartbeat restore, handoff watcher, loop wakeup).

Entering ``_profile_runtime_scope`` costs a config.yaml load, a ``.env`` parse, secret hydration and a
terminal-policy build; on a multiplex gateway the pollers paid that per profile per tick with nothing
to do. Each gate answers "does this profile's store hold work?" from the goals-cached SessionDB with
ONLY the HERMES_HOME contextvar installed. Every gate fails OPEN: an unavailable store, a failing
read or a corrupt row is "cannot prove emptiness", never "idle".

Two rules keep the gate sweeps from turning into thread churn (PRA-6660):

* **One work item per tick, not one per profile.** A poller asks ``off_loop_busy_homes`` for the whole
  set of profiles at once; a dozens-profile gateway used to submit one executor hop per profile per
  tick, and on the shared gateway those hops were ~9 of the ~21 thread spawns per second measured in
  the turn executor (one fresh OS thread each).
* **The poll lane, never the turn pool and never the housekeeping pool.** ``offload_for_poll``
  prefers ``GatewayRunner._run_poll_in_executor``: a fixed pool that reuses its workers. A probe must
  never mint threads on the pool whose size is the gateway's turn concurrency (PRA-6660), and it must
  not queue behind best-effort housekeeping either — a poll hop behind four in-flight housekeeping
  items waited 1970 ms p50 to start on this host against 10 ms on a lane of its own, which is queue
  time reported as poll cost (PRA-6791).
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("gateway.run")


def _profile_session_db_probe(profile_home: Path) -> Optional[Any]:
    """The goals-cached SessionDB for *profile_home*; None when unavailable."""
    from hermes_cli.goals import _get_session_db
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(profile_home))
    try:
        return _get_session_db()
    except Exception:
        logger.debug("session-db probe failed for %s", profile_home, exc_info=True)
        return None
    finally:
        reset_hermes_home_override(token)


def _gate(profile_home: Path, store_has_work: Callable[[Any], bool]) -> bool:
    db = _profile_session_db_probe(profile_home)
    if db is None:
        return True
    try:
        return store_has_work(db)
    except Exception:
        logger.debug("idle probe failed for %s; keeping the full sweep", profile_home, exc_info=True)
        return True


def profile_has_active_heartbeat(profile_home: Path) -> bool:
    from hermes_cli.heartbeat import store_has_active_heartbeat

    return _gate(profile_home, store_has_active_heartbeat)


def active_loop_in_store(db: Any) -> bool:
    """The loop wakeup watcher's gate body, exposed so a whole scan's sweep can share it."""
    from hermes_cli.loops import store_has_active_loop

    return store_has_active_loop(db)


def profile_has_active_loop(profile_home: Path) -> bool:
    return _gate(profile_home, active_loop_in_store)


def pending_handoff_in_store(db: Any) -> bool:
    """The handoff watcher's gate body, exposed so a whole tick's sweep can share it."""
    return db.has_pending_handoffs()


def profile_has_pending_handoff(profile_home: Path) -> bool:
    return _gate(profile_home, pending_handoff_in_store)


def busy_homes(homes, store_has_work: Callable[[Any], bool]) -> set:
    """Which of *homes* hold work; the whole sweep as ONE synchronous work item."""
    return {home for home in homes if _gate(home, store_has_work)}


# How the sweep is spread: only the stores that would pay a COLD open.
# Measured on this host against the live 17-profile set (2026-10-23T02:0x-02:2xZ, load1 ~110):
#   warm, serial inside the hop      p50 1.9-58 ms, per-home read p50 1.4 ms   <- steady state
#   warm, 4 workers inside the hop   p50 24.8-95 ms, p95 67-142 ms             <- the hop IS the cost
#   cold, serial                     one home up to 5.5 s; 18 homes ~18 s      <- a boot-minute stall
#   cold, 4 workers                  whole fan-out 0.78 s                      <- where concurrency pays
# So the split is by coldness, not by profile: an already-open store is read on the calling thread
# (its read is cheaper than a thread hop), a not-yet-open one goes to a bounded pool. Steady state
# therefore takes no hop beyond the single one the tick already made, and a boot or a brand-new
# profile still opens concurrently instead of serialising seconds inside a tick.
_LOOP_SWEEP_FAN_OUT = 4
_LOOP_SWEEP_POOL = None
_LOOP_SWEEP_POOL_LOCK = threading.Lock()


def _loop_sweep_pool():
    """The sweep's bounded worker set, created once per process — never per tick.

    Per-tick ``ThreadPoolExecutor()`` blocks would spawn and retire up to four OS threads every
    tick: the churn PRA-6660 measured on the turn pool, just moved. It is only reached when a home
    is cold, so a steady-state gateway creates it once (or never, when the boot warm covered the
    served set).
    """
    global _LOOP_SWEEP_POOL
    if _LOOP_SWEEP_POOL is None:
        with _LOOP_SWEEP_POOL_LOCK:
            if _LOOP_SWEEP_POOL is None:
                _LOOP_SWEEP_POOL = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_LOOP_SWEEP_FAN_OUT, thread_name_prefix="hermes-loop-sweep")
    return _LOOP_SWEEP_POOL


def _active_loops_in_home(profile_home) -> list:
    """One home's ACTIVE loops off the store the idle gate would open; ``[]`` when unreadable."""
    from hermes_cli.loops import list_active_loops_from

    if profile_home is None:  # the launch home: no override, the process home IS the scope
        from hermes_cli.goals import _get_session_db

        return list_active_loops_from(_get_session_db())
    return list_active_loops_from(_profile_session_db_probe(profile_home))


def _home_store_open(profile_home) -> bool:
    """Would reading *profile_home* now be a warm read (already-open handle)?"""
    from hermes_cli.goals import session_db_is_open

    return session_db_is_open(None if profile_home is None else str(profile_home))


def _read_homes(jobs) -> list:
    """``[(name, home, hits)]`` for the jobs that hold an ACTIVE loop; a bad home is a debug line."""
    out = []
    for name, home in jobs:
        try:
            hits = _active_loops_in_home(home)
        except Exception:
            logger.debug("loop sweep read failed for %s; next tick retries", home, exc_info=True)
            continue
        if hits:
            out.append((name, home, hits))
    return out


def loop_sweep(scopes) -> list:
    """Every scanned home's active loops as ONE synchronous work item (PRA-6806).

    ``scopes`` is ``[(profile_name, profile_home_or_None), ...]`` — the watcher's own resolution,
    root poll first. Returns ``[(profile_name, profile_home, [(session_id, LoopState), ...]), ...]``
    for the homes that hold at least one ACTIVE loop; a home that holds none costs one read and no
    entry, and an unreadable home costs one debug line (the next tick retries it, as before).

    Why one item: the watcher used to submit two hops per profile per tick (warm + read) and await
    them in order, so on a 17-profile host one tick queued 34 hops behind itself on the 2-worker
    poll lane — the queue moved off the housekeeping pool instead of going away (PRA-6794: p50
    236-717 ms against ~2 ms of warm work per home). One hop is one lane slot, whatever N is, and
    the only concurrency left is the cold opens that would otherwise serialise seconds (see above).
    """
    jobs = list(scopes)
    warm = [job for job in jobs if _home_store_open(job[1])]
    cold = [job for job in jobs if not _home_store_open(job[1])]
    if not cold:
        return _read_homes(warm)
    if len(cold) == 1:  # one cold home does not need the pool (nothing to overlap with)
        return _read_homes(warm + cold)
    pool = _loop_sweep_pool()
    pending = {pool.submit(_active_loops_in_home, home): (name, home) for name, home in cold}
    cold_hits = {}
    for future, (name, home) in pending.items():
        try:
            cold_hits[home] = future.result()
        except Exception:
            logger.debug("loop sweep read failed for %s; next tick retries", home, exc_info=True)
    return _read_homes(warm) + [(name, home, cold_hits[home]) for name, home in cold
                                if cold_hits.get(home)]


def offload_for_poll(runner: object) -> Optional[Callable]:
    """The runner hop for per-tick polling: the POLL lane, then housekeeping, then the turn pool.

    ``_run_poll_in_executor`` is a fixed set of workers of its own, so a poller contributes no thread
    spawns and cannot queue behind a session finalize (PRA-6791). ``_run_housekeeping_in_executor``
    is the older bounded pool and stays as the fallback for runners that predate the poll lane; the
    turn helper is last, for test doubles that stub only it.
    """
    for attr in ("_run_poll_in_executor", "_run_housekeeping_in_executor",
                 "_run_in_executor_with_context"):
        offload = getattr(runner, attr, None)
        if callable(offload):
            return offload
    return None


async def off_loop_gate(runner: object, probe: Callable[[], bool]) -> bool:
    """Run a sync gate through the runner's poll hop. Runners without one (bare test stand-ins
    for the handoff watcher) keep the historical always-enter behaviour."""
    offload = offload_for_poll(runner)
    if offload is None:
        return True
    return bool(await offload(probe))


async def off_loop_busy_homes(runner: object, homes, store_has_work: Callable[[Any], bool]) -> set:
    """``busy_homes`` for a whole poller tick in ONE hop, and on the bounded pool.

    Per-profile gates cost one executor submission each; with N served profiles a 2 s handoff tick
    submitted N of them beside the scope walk (PRA-6660: ~9 thread spawns/s on the shared gateway,
    16 profiles). A tick now costs one hop no matter what N is. Fail-open is unchanged: a home whose
    probe cannot answer is IN the result, so its scope is entered as before.
    """
    homes = list(homes)
    if not homes:
        return set()
    offload = offload_for_poll(runner)
    if offload is None:
        return set(homes)
    return set(await offload(busy_homes, homes, store_has_work))


async def poll_plan(runner: object, store_has_work: Callable[[Any], bool]) -> tuple:
    """A poller tick's whole prerequisite: the scopes to walk, and which of them hold work.

    Both halves are offloaded (a multiplex scope walk is a filesystem walk) and both land on the
    bounded pool, so a tick is a fixed cost regardless of how many profiles the gateway serves.
    """
    from gateway.run import _resolve_handoff_watch_scopes

    scopes = await _resolve_handoff_watch_scopes(runner)
    busy = await off_loop_busy_homes(
        runner, [home for _name, home in scopes if home is not None], store_has_work)
    return scopes, busy
