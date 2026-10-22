"""The gateway REPLACES ITSELF when the tree's wire contract moves past its boot snapshot.

SOU-3316 named the skew; SOU-3318 made it refusable and visible. Neither shortened the outage: on
2026-10-2x a second contract merge left a long-lived gateway serving the old params model while
``check-gateway-serving.sh`` S10 FAILed as a finding, ``restart-gateway-quiet.sh`` waited for someone
to run it, and 40 runs across 13 agents and 6 companies died at 0 tokens for ten minutes.

Naming a skew shortens the postmortem. Only the serving process can end it, and it is the one actor
that knows its own boot snapshot (:func:`gateway.contract_revision.boot_revision`) and that dies with
the gateway — so there is no stale watchdog left behind to maintain. This module is that actor: an
in-process ticker (the gateway housekeeping thread) that compares boot versus tree, debounces, and
takes the planned-restart route the process already owns.

Five decisions a future reader will otherwise reverse, each with its reason:

* **Debounce is a SETTLED window, not a second glance.** A merge writes the working tree before the
  commit object — SOU-3328's first refusal preceded its own merge commit by 45 s — so one digest
  once is not a tree worth dying for. The pair must repeat on consecutive ticks AND stay unchanged
  for ``2 * contract_watch_seconds`` (>= 20 s by default).
* **Restart-now, even with turns in flight.** ``restart-gateway-quiet.sh`` refuses a restart while
  more than two agents are active; that is right for a routine bounce and wrong here. A proven skew
  dooms every in-flight turn at its NEXT model call, so waiting converts a 20 s replacement into the
  outage it is meant to end. Interrupted turns are counted per profile into the durable record so
  the next boot can settle them.
* **The loop guard is written BEFORE the restart is requested.** A restart that wedges must still be
  capped; a guard written after the attempt would be lost exactly when it is needed, and a persistent
  skew would become a restart loop.
* **Unprovable input never acts.** No contract document (an install without ``apps/``), a non-git
  tree, or an unreadable file where boot had a digest is an I/O fact, not a skew.
* **An in-progress drain wins.** ``hermes update`` already owns a drain; stacking a second restart on
  it is how two restarts become one interrupted teardown.

The digest is NOT defined here and must not be: ``contract_revision.revision()`` is exactly
"sha256(document)[:12]", because the deploy check (``check-gateway-serving.sh`` S10) recomputes it in
shell and compares — widening the hash to cover the rendered ``gateway-contract.generated.ts`` too
would make every S10 comparison FAIL forever. The JSON is the source of truth the TS is rendered
from; a hand-edited generated TS is a generator drift, not a contract move.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger("gateway.contract_watch")

#: Durable loop-guard record, under ``<HERMES_HOME>/state/``. Names the last pair this install
#: decided to replace itself for, so a restart loop survives the restart that would hide it.
STATE_FILENAME = "contract-self-replace.json"

WATCH_CONFIG_KEY = "contract_watch_seconds"
MIN_RESTART_CONFIG_KEY = "contract_watch_min_seconds_between_restarts"

DEFAULT_WATCH_SECONDS = 10.0
DEFAULT_MIN_SECONDS_BETWEEN_RESTARTS = 600.0

#: Housekeeping chore label (the tick-cost log attributes a slow tick to this name).
LABEL = "Wire-contract self-replace watch"

#: A pair seen once and gone by the next tick was a torn write, not a deploy.
UNSETTLED_TICKS = 2
#: The settle window in ticks: with the 10 s default a pair must survive 20 s of ticks before the
#: process is replaced (documented in the module docstring; asserted by the debounce tests).
SETTLE_WINDOWS = 2


@dataclass(frozen=True)
class Verdict:
    """What one tick decided, and why. Returned (never raised) so a tick can be asserted on."""

    outcome: str
    boot: Optional[str] = None
    tree: Optional[str] = None
    detail: str = ""


def state_path(home: Optional[Path] = None) -> Path:
    """``<HERMES_HOME>/state/contract-self-replace.json``."""
    return Path(home if home is not None else get_hermes_home()) / "state" / STATE_FILENAME


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _gateway_section(config: Any) -> Mapping[str, Any]:
    if config is None:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    section = config.get("gateway") if isinstance(config, Mapping) else getattr(config, "gateway", None)
    return section if isinstance(section, Mapping) else {}


def watch_seconds(config: Any = None) -> float:
    """``gateway.contract_watch_seconds``; ``<=0`` disables the watch entirely."""
    raw = _gateway_section(config).get(WATCH_CONFIG_KEY, DEFAULT_WATCH_SECONDS)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_WATCH_SECONDS
    return value if value > 0 else 0.0


def min_seconds_between_restarts(config: Any = None) -> float:
    """``gateway.contract_watch_min_seconds_between_restarts``; unreadable/``<=0`` -> default.

    A zero window would make the loop guard a no-op, which is the failure it exists to prevent, so
    a nonsensical value falls back rather than disables.
    """
    raw = _gateway_section(config).get(MIN_RESTART_CONFIG_KEY, DEFAULT_MIN_SECONDS_BETWEEN_RESTARTS)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_MIN_SECONDS_BETWEEN_RESTARTS
    return value if value > 0 else DEFAULT_MIN_SECONDS_BETWEEN_RESTARTS


def _boot_revision() -> Optional[str]:
    from gateway.contract_revision import boot_revision

    return boot_revision()


def _tree_revision() -> Optional[str]:
    from gateway.contract_revision import revision

    return revision()


def _read_state(home: Optional[Path]) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(state_path(home).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_state(home: Optional[Path], payload: dict[str, Any]) -> None:
    """Best-effort: a guard that cannot be persisted must not block the replacement it protects."""
    try:
        from hermes_constants import mkdir_under_hermes_home
        from utils import atomic_json_write

        path = state_path(home)
        mkdir_under_hermes_home(path.parent)
        atomic_json_write(path, payload)
    except Exception:
        logger.debug("Could not persist the contract self-replace guard record", exc_info=True)


def replaced_recently(record: Optional[dict[str, Any]], boot: str, tree: str, now: float,
                      window_seconds: float) -> bool:
    """True when THIS pair was already replaced inside ``window_seconds`` — restart once, not in a loop.

    Keyed on the pair, so a rollback (tree moves back, a different pair) is a new decision while a
    restart that failed to change what the process serves is not.
    """
    if not record or record.get("boot") != boot or record.get("tree") != tree:
        return False
    attempted = record.get("attempted_at_epoch")
    if not isinstance(attempted, (int, float)):
        return False
    return (now - float(attempted)) < window_seconds


class ContractWatcher:
    """Tick: compare the booted contract with the tree's, debounce, then replace the process.

    Every collaborator is injected so the behaviour is testable without a gateway: ``boot``/``tree``
    are the two revisions, ``restart`` requests the planned restart (returns whether it was
    accepted), ``publish`` writes ``contract_stale`` into ``gateway_state.json``, ``interrupted``
    returns interrupted-turn counts per profile, ``drain_active`` reports an in-progress drain.
    """

    def __init__(
        self,
        *,
        home: Optional[Path] = None,
        seconds: float = DEFAULT_WATCH_SECONDS,
        min_seconds_between_restarts: float = DEFAULT_MIN_SECONDS_BETWEEN_RESTARTS,
        boot: Callable[[], Optional[str]] = _boot_revision,
        tree: Callable[[], Optional[str]] = _tree_revision,
        restart: Optional[Callable[[], bool]] = None,
        publish: Optional[Callable[[Optional[dict[str, Any]]], None]] = None,
        interrupted: Optional[Callable[[], dict[str, int]]] = None,
        drain_active: Optional[Callable[[], bool]] = None,
        supervised: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.seconds = max(0.001, float(seconds))
        self.label = LABEL
        self._home = home
        self._min_gap = float(min_seconds_between_restarts)
        self._boot = boot
        self._tree = tree
        self._restart = restart
        self._publish = publish or self._publish_status
        self._interrupted = interrupted
        self._drain_active = drain_active or (lambda: False)
        self._supervised = supervised
        self._clock = clock
        self._pending: Optional[tuple[str, str, float]] = None
        self._acted: Optional[tuple[str, str]] = None
        self._sightings = 0
        self._said_unprovable = False
        self._said_suppressed = False
        self._published: Optional[dict[str, Any]] = None
        self._tick_failed = False

    # -- the tick ---------------------------------------------------------------------------

    def tick(self) -> Verdict:
        """One housekeeping tick. Never raises: a watcher fault must not kill the chore loop."""
        try:
            return self._tick()
        except Exception as exc:  # pragma: no cover - defensive boundary, asserted by no test
            # The chore loop's own boundary would swallow this; a replacement that stops happening
            # silently is the failure this module exists to prevent, so the FIRST fault is a warning.
            if not self._tick_failed:
                self._tick_failed = True
                logger.warning("Wire-contract watch tick failed (further failures at debug): %s",
                               exc, exc_info=True)
            else:
                logger.debug("Wire-contract watch tick failed: %s", exc)
            return Verdict("error", detail=str(exc))

    def _tick(self) -> Verdict:
        boot, tree = self._boot(), self._tree()
        if not boot or not tree:
            self._pending = None
            self._forget_sequence()
            if not self._said_unprovable:
                self._said_unprovable = True
                logger.info(
                    "Wire-contract watch idle: boot=%s tree=%s — no digest on one side (no contract "
                    "document, non-git tree, or an unreadable file). Not acting: unprovable is not a skew.",
                    boot or "unknown", tree or "unknown")
            return Verdict("unprovable", boot, tree)
        self._said_unprovable = False

        if boot == tree:
            self._pending = None
            self._forget_sequence()
            self._clear_published()
            return Verdict("no_skew", boot, tree)

        now = self._clock()
        if self._pending is None or self._pending[:2] != (boot, tree):
            # First sighting: a torn write, a half-checked-out merge and a real deploy look identical
            # here. Hold; only a pair that survives the settle window is acted on.
            self._pending = (boot, tree, now)
            self._sightings = 1
            self._acted = None
            self._said_suppressed = False
            return Verdict("skew_unsettled", boot, tree, "first sighting")
        if self._acted == (boot, tree):
            # Already asked for the replacement this pair needs; a slow drain is not a second skew.
            return Verdict("restart_in_progress", boot, tree)

        first_seen = self._pending[2]
        self._sightings += 1
        settle_seconds = SETTLE_WINDOWS * self.seconds
        if self._sightings < UNSETTLED_TICKS or (now - first_seen) < settle_seconds:
            return Verdict("skew_unsettled", boot, tree,
                           f"{self._sightings} sighting(s), {now - first_seen:.0f}s of "
                           f"{settle_seconds:.0f}s settle window")
        return self._act(boot, tree, now)

    def _act(self, boot: str, tree: str, now: float) -> Verdict:
        if self._drain_active():
            # ``hermes update`` (or an operator drain) already owns a restart. Its drain wins: a second
            # one here would amputate the same turns twice and race the supervisor's respawn.
            return Verdict("drain_in_progress", boot, tree)
        record = _read_state(self._home) or {}
        if replaced_recently(record, boot, tree, now, self._min_gap):
            payload = self._payload(boot, tree, status="suppressed", interrupted={})
            payload["suppressed_by"] = {k: record.get(k) for k in ("attempted_at", "attempted_at_epoch", "pid")}
            self._publish_once(payload)
            if not self._said_suppressed:
                self._said_suppressed = True
                logger.warning(
                    "Wire-contract skew PERSISTS (serving contract-%s, tree contract-%s) but this pair was "
                    "already replaced at %s (pid %s) — not restarting again inside %ds. The skew stays "
                    "published as contract_stale for the deploy check and the run-failure detector; a bad "
                    "checkout needs a human, not a restart loop.",
                    boot, tree, record.get("attempted_at"), record.get("pid"), int(self._min_gap))
            return Verdict("suppressed", boot, tree)

        interrupted = self._interrupted_counts()
        payload = self._payload(boot, tree, status="requested", interrupted=interrupted)
        if self._restart is None:
            payload["status"] = "unenforceable"
            payload["unenforceable"] = (
                "this process has no planned-restart path (not a supervised gateway); the skew is "
                "published but nothing here can replace it — ai.hermes.ensure-gateway-boot must")
            self._publish_once(payload)
            logger.warning("Wire-contract skew cannot be enforced by this process: %s", payload["unenforceable"])
            return Verdict("unenforceable", boot, tree, payload["unenforceable"])

        # Durable BEFORE the attempt: a restart that wedges must still be capped at one per window.
        _write_state(self._home, {
            "boot": boot, "tree": tree, "attempted_at": _now_iso(), "attempted_at_epoch": now,
            "pid": os.getpid(), "interrupted": interrupted, "supervised": self._supervised,
        })
        self._publish(payload)
        self._published = payload
        self._acted = (boot, tree)
        accepted = bool(self._restart())
        logger.warning(
            "Wire-contract skew: the tree declares contract-%s but this process serves contract-%s. Taking "
            "the planned restart now (%s) — in-flight turns are counted, not waited for, because each one "
            "is already doomed at its next model call. Interrupted per profile: %s",
            tree, boot, "accepted" if accepted else "already restarting", interrupted)
        return Verdict("restarting" if accepted else "restart_refused", boot, tree)

    # -- publishing -------------------------------------------------------------------------

    def _payload(self, boot: str, tree: str, *, status: str, interrupted: dict[str, int]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "boot": boot, "tree": tree, "decided_at": _now_iso(), "pid": os.getpid(),
            "status": status, "supervised": self._supervised,
            "interrupted": dict(interrupted or {}),
            "restart_after_seconds": int(self._min_gap),
        }
        if not self._supervised:
            # Say who brings it back: the planned stop exits 75 either way, but without a supervisor
            # only the boot guard does, and a reader must not mistake "exited" for "replaced".
            payload["respawn"] = (
                "no supervisor: the planned stop exits 75; ai.hermes.ensure-gateway-boot must respawn")
        return payload

    def _publish_once(self, payload: dict[str, Any]) -> None:
        if self._published == payload:
            return
        self._publish(payload)
        self._published = payload

    def _clear_published(self) -> None:
        """Withdraw a stale claim once the process and the tree agree again (one write, never per tick).

        ``gateway_state.json`` is merged, not rewritten, so the claim from the previous life survives a
        restart: a clean boot owes exactly one clearing write, and a steady in-sync tree owes none.
        """
        if self._published is None:
            from gateway.status import read_runtime_status

            if not (read_runtime_status() or {}).get("contract_stale"):
                return
        self._publish(None)
        self._published = None

    @staticmethod
    def _publish_status(payload: Optional[dict[str, Any]]) -> None:
        from gateway.status import write_runtime_status

        write_runtime_status(contract_stale=payload, wait_timeout=2.0)

    # -- collaborators with a default ---------------------------------------------------------

    def _interrupted_counts(self) -> dict[str, int]:
        if self._interrupted is None:
            return {}
        try:
            return dict(self._interrupted() or {})
        except Exception:
            logger.debug("Could not count interrupted turns for the contract restart", exc_info=True)
            return {}

    def _forget_sequence(self) -> None:
        self._sightings = 0
        self._said_suppressed = False


def _on_loop(loop: Any, fn: Callable[[], Any], timeout: float = 5.0) -> Any:
    """Run *fn* on the gateway loop and return its value (``None`` on timeout).

    The housekeeping thread must never read live runner state directly: agent tables are mutated on
    the loop, and a read that raises there reads as "nothing in flight".
    """
    box: list[Any] = []
    done = threading.Event()

    def _run() -> None:
        try:
            box.append(fn())
        except Exception:
            logger.debug("Contract-watch loop callback failed", exc_info=True)
        finally:
            done.set()

    loop.call_soon_threadsafe(_run)
    done.wait(timeout=timeout)
    return box[0] if box else None


def _profile_key(unit: dict[str, Any], namespaces: Mapping[str, str]) -> str:
    session = str(unit.get("session") or "")
    matches = [(len(ns), profile) for ns, profile in namespaces.items() if session.startswith(ns + ":")]
    return max(matches)[1] if matches else "unattributed"


def counts_by_profile(runner: Any) -> dict[str, int]:
    """In-flight work units per served profile, for the durable record. Best-effort."""
    from gateway.session import _session_key_namespace

    namespaces = {_session_key_namespace(name): name for name in (runner.served_profile_names() or [])}
    counts: dict[str, int] = {}
    for unit in runner._describe_active_work():
        key = _profile_key(unit, namespaces)
        counts[key] = counts.get(key, 0) + 1
    return counts


def drain_active(runner: Any) -> bool:
    """True when a drain/restart is already under way in this process or an external marker."""
    if getattr(runner, "_draining", False) or getattr(runner, "_restart_task_started", False):
        return True
    from gateway.drain_control import drain_requested

    return drain_requested()


def watcher_for_runner(runner: Any, loop: Any, *, config: Any = None,
                       home: Optional[Path] = None) -> Optional[ContractWatcher]:
    """Build the live watcher for the gateway housekeeping thread, or ``None`` when disabled."""
    seconds = watch_seconds(config)
    if seconds <= 0:
        logger.info("%s disabled (%s <= 0)", LABEL, WATCH_CONFIG_KEY)
        return None
    from gateway.restart import is_supervised_gateway_launch

    def _restart() -> bool:
        accepted = _on_loop(loop, lambda: runner.request_restart(
            detached=False, via_service=True, interrupt_in_flight=True))
        return bool(accepted)

    def _interrupted() -> dict[str, int]:
        return _on_loop(loop, lambda: counts_by_profile(runner)) or {}

    return ContractWatcher(
        home=home, seconds=seconds, min_seconds_between_restarts=min_seconds_between_restarts(config),
        restart=_restart, interrupted=_interrupted, drain_active=lambda: drain_active(runner),
        supervised=is_supervised_gateway_launch())
