"""External drain control: the reversible ``.drain_request.json`` state machine.

A dashboard / ``hermes update`` writer cannot reach into a running gateway, so it writes (or
removes) ``{HERMES_HOME}/.drain_request.json`` and this mixin reacts: an ACTIVE marker refuses NEW
turns without exiting the process, and removing it re-accepts them. Split out of
``run_shutdown.py`` (its growth ratchet) — the marker contract itself lives in
``gateway/drain_control.py``; boot adoption lives here because it is the same state machine
(PRX-5819).
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("gateway.run")


class GatewayDrainControlMixin:
    """Enter/exit the external drain, reconcile it while running, adopt it at boot."""

    async def _adopt_pending_drain(self) -> None:
        """Honour a drain marker that predates this process, before admission opens (PRX-5819).

        The marker is a durable file and the gateway process is not, so every drain that spans a
        boot (an ``hermes update`` restart, a resumed acceptance run) starts with the marker already
        on disk. The watcher reconciles it — but only after the startup sequence has published
        ``running``, and measured 2026-10-21 that gap was 35 s: 24 runs spawned in the 2 s after
        boot, 21 of them refused with "draining for a maintenance action" and recorded as
        ``adapter_failed``, indistinguishable from a real adapter fault.

        Off-thread because the read is file I/O that can stall under host pressure, exactly like the
        watcher's tick. Staleness is the marker contract's business: a dead writer, an expired
        lease, another instantiation's epoch or an untrusted principal reads as absent, so a real
        boot is never parked by a leftover file.
        """
        from gateway.drain_control import drain_requested

        if await asyncio.to_thread(drain_requested):
            logger.info("Drain marker present at boot — honouring it before admission opens.")
            self._enter_external_drain()

    def _enter_external_drain(self) -> None:
        """Begin external drain: refuse NEW turns (in-flight ones are NOT interrupted). Idempotent."""
        if self._external_drain_active:
            return
        self._external_drain_active = True
        logger.info(
            "External drain ENGAGED (.drain_request.json present) — refusing "
            "new turns; %d in-flight turn(s) will finish. Process stays up.", self._active_work_count(),
        )
        # Persist "draining" so /api/status tracks it; active_agents is read-merged, only state changes.
        self._update_runtime_status("draining")

    def _exit_external_drain(self) -> None:
        """Cancel external drain: re-accept new turns. Idempotent; never resurrects a stopping gateway."""
        if not self._external_drain_active:
            return
        self._external_drain_active = False
        if self._draining or not self._running:
            logger.info(
                "External drain marker cleared during shutdown — not reverting "
                "to running (shutdown takes precedence)."
            )
            return
        logger.info(
            "External drain RELEASED (.drain_request.json removed) — "
            "re-accepting new turns; gateway_state -> %s.", self._serving_state(),
        )
        self._update_runtime_status(self._serving_state())

    async def _drain_control_watcher(self, interval: float = 1.0) -> None:
        """Poll ``.drain_request.json`` at 1s: present -> enter drain, absent -> exit; a stale epoch = absent."""
        from gateway.drain_control import drain_requested

        while self._running:
            try:
                # Off-thread: a synchronous marker read at 1s cadence can stall 30s+ under host I/O
                # pressure and take every platform heartbeat down.
                if await asyncio.to_thread(drain_requested):
                    self._enter_external_drain()
                    # API and cron work live outside messaging's _running_agents map; refresh the
                    # aggregate while an external caller polls this reversible drain state.
                    self._persist_active_agents()
                else:
                    self._exit_external_drain()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Drain-control watcher tick error: %s", exc, exc_info=True)
            await asyncio.sleep(interval)
