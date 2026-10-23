"""Hot-serve for ``gateway.multiplex_profiles``: keep the served-profile set in step with ``profiles/``
while the multiplexer runs, instead of snapshotting it once at boot.

Three things were start-time snapshots: the secondary adapter set (``_start_secondary_profile_adapters``),
the ``served_profiles`` record in ``gateway_state.json`` (``_record_served_profiles``) and the cron
ticker's ``profile_homes`` list. Everything else (``/p/<profile>/`` prefixes, profile-route eligibility,
handoff/kanban watchers, shared ingress) already reads ``profiles_to_serve()`` / ``_profile_adapters``
live, so reconciling those three is enough for a profile created after boot to be served.

``reconcile_served_profiles`` runs on the loop under one lock, triggered by the ``rescan-profiles`` control
verb (``hermes_cli/profiles.py`` create/delete fire it through the control socket) and by the supervised
``_profile_reconcile_watcher`` every ``_PROFILE_RESCAN_INTERVAL_SECS`` as the safety net. A served profile whose ``config.yaml``/``.env``
changed since its adapters were last built is re-scanned too: creators make the profile first and add the
bot token afterwards, and without this an adapter-less profile would stay adapter-less forever.

The watcher's FIRST pass is not a tick: ``prime_profile_reconcile`` runs the hook-consent sync and
the first full diff during startup, where that cold fan-out over every served home is free
(PRX-5989). Left in the tick it cost p95 49-113 s on a 17-profile boot and 40-2,800 ms on the very
next passes; the tick still owns every pass after the first, unchanged.

The same watcher tick (on every gateway, multiplexed or not) re-syncs a home's shell hooks when its
``shell-hooks-allowlist.json`` (or ``config.yaml``) changed: a non-TTY gateway skips unconsented hooks, so consent recorded
afterwards must register them without a restart (#132993). That re-runs hook registration only, never
an adapter rebuild.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Optional

from gateway.run_shutdown import _log_suppressed
from utils import file_signature

logger = logging.getLogger(__name__)

_PROFILE_RESCAN_INTERVAL_SECS = 30.0
_PROFILE_SIGNATURE_FILES = ("config.yaml", ".env")
# Bound for one own-gateway liveness probe (``live_gateway_pid_for_home``): the control-socket
# read it can end in has no timeout of its own — on Windows the named pipe stalls there until
# its peer answers — so the await needs one. A healthy identify answers in milliseconds.
_OWN_GATEWAY_PROBE_TIMEOUT_SECS = 5.0


def _load_profile_serving_config(profile_home: Path) -> tuple[dict[str, Any], Any, Optional[str]]:
    """``(runtime config, gateway config, open-policy violation)`` for a profile about to be served.

    Runs in a worker thread (``asyncio.to_thread``) because it reads that profile's YAML and walks
    the plugin tree and every installed entry point — three times over, since two of these reads
    discover plugins as a side effect. Inline on the loop that walk is what the 2026-10-19
    starvation dump caught: ``task_step`` -> ``gen_iternext`` -> ``_io_open`` inside
    ``importlib.metadata.entry_points`` (PRX-5524). ``asyncio.to_thread`` copies the context, so the
    profile scope entered here belongs to the worker: every read below sees THIS profile, wherever
    the loop's own ambient scope points. Only gateway-state mutations stay on the loop.
    """
    from gateway.run import (
        _load_gateway_config, _own_policy_open_startup_violation, _profile_runtime_scope,
    )
    from gateway.config import load_gateway_config
    from hermes_cli.plugins import discover_plugins
    with _profile_runtime_scope(profile_home, hydrate_secrets=False):
        runtime_cfg = _load_gateway_config()
        discover_plugins()  # idempotent; registers this profile's plugins process-wide
        cfg = load_gateway_config()
        # Reads the same config + plugin entries, and discovers plugins; keep it in this scope.
        violation = _own_policy_open_startup_violation(cfg)
    return runtime_cfg, cfg, violation


def _stat_signature(path: Path) -> Optional[tuple]:
    try:
        return file_signature(os.stat(path))
    except OSError:
        return None


def profile_serve_signature(home: Path) -> tuple:
    """Cheap change detector for a served profile's credentials/config: file signature per file."""
    sig = []
    for name in _PROFILE_SIGNATURE_FILES:
        try:
            st = os.stat(Path(home) / name)
            sig.append(file_signature(st))
        except OSError:
            sig.append(None)
    return tuple(sig)


def _serve_signatures(homes: dict[str, Path]) -> dict[str, tuple]:
    """Off-loop half of the changed-profile diff: one batched stat pass per reconcile cycle.

    ``profile_serve_signature`` is two ``os.stat`` calls per served profile, run every 30 s over
    every served home. Inline on the loop that is a whole stat sweep competing with turn execution
    for the GIL; the sibling hook-consent sync next door already hops off-loop for the same reason.
    """
    return {name: profile_serve_signature(home) for name, home in homes.items()}


def _added_profile_blocker(launch: Path) -> Optional[str]:
    """Why a single-profile boot must not start multiplexing now that a second profile exists: the
    ``gateway.auto_multiplex_migration: false`` opt-out on the default profile, else boot's preflight."""
    from hermes_cli.gateway_migrate_guards import auto_migration_opted_out
    from hermes_cli.gateway_multiplex_mode import AUTO_MIGRATION_OPTED_OUT_REASON, implicit_multiplex_blocker
    if auto_migration_opted_out(launch):
        return AUTO_MIGRATION_OPTED_OUT_REASON
    return implicit_multiplex_blocker()


class GatewayProfileReconcileMixin:
    """Runtime reconciliation of the multiplexed served-profile set (hot add / unroute / credential-add)."""

    _served_profile_homes: Optional[dict[str, Path]] = None
    _served_profile_signatures: Optional[dict[str, tuple]] = None
    _profile_reconcile_lock: Optional[asyncio.Lock] = None
    _profile_own_gateway_warned: Optional[set[str]] = None
    _profile_probe_timeout_warned: Optional[set[str]] = None
    _hook_consent_signatures: Optional[dict[str, Optional[tuple]]] = None
    _single_profile_blocker_logged: Optional[str] = None

    # ── state helpers ─────────────────────────────────────────────────────────────────────────────

    def _reconcile_lock(self) -> asyncio.Lock:
        if self._profile_reconcile_lock is None:
            self._profile_reconcile_lock = asyncio.Lock()
        return self._profile_reconcile_lock

    def served_profile_names(self) -> list:
        """Profiles this multiplexer currently serves (active first), from the live bookkeeping."""
        homes = self._served_profile_homes or {}
        active = getattr(self, "_primary_profile_name", None) or "default"
        return ([active] if active in homes or not homes else []) + sorted(n for n in homes if n != active)

    def _note_served_profiles(self, profile_homes) -> None:
        """Called by ``_record_served_profiles``: remember the served set and each home's signature."""
        homes = {str(name): Path(home) for name, home in profile_homes}
        self._served_profile_homes = homes
        sigs = self._served_profile_signatures if isinstance(self._served_profile_signatures, dict) else {}
        self._served_profile_signatures = {name: sigs.get(name) or profile_serve_signature(home)
                                           for name, home in homes.items()}

    # ── watcher ───────────────────────────────────────────────────────────────────────────────────

    async def _profile_reconcile_watcher(self, interval: float = _PROFILE_RESCAN_INTERVAL_SECS) -> None:
        """Supervised safety net, every ``interval`` seconds: re-sync shell hooks whose consent changed
        (every gateway), then rescan ``profiles/`` when multiplexing (creators signal the control
        socket for an immediate rescan); a gateway standalone only because the host had one profile
        starts multiplexing here once a second one exists."""
        while self._running:
            await asyncio.sleep(interval)
            if not self._running:
                return
            try:
                # Wall time of the tick, summarised (~10 min) beside the dumps: this ticker is the
                # entry point of five of PRX-5524's blocked-loop dumps, so its trend is the evidence.
                from gateway.loop_tick_timing import timed
                with timed('profile_reconcile'):
                    await self._sync_hooks_on_consent_change()
                    # A standalone gateway never runs a profile reconcile from the watcher, unless it is
                    # the single-profile boot that starts multiplexing in place for a newly created
                    # profile.
                    if self._multiplex_on() or await self._multiplex_for_added_profile():
                        await self.reconcile_served_profiles(reason="watcher")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Served-profile reconcile failed; retrying next cycle", exc_info=True)

    async def prime_profile_reconcile(self, *, reason: str = "boot") -> dict[str, Any]:
        """The watcher's first pass, run during startup instead of inside its first tick.

        On a multiplexed boot the first pass is a cold fan-out over every served home:
        ``_sync_hooks_on_consent_change`` loads each home's ``config.yaml`` and re-registers its
        shell hooks — measured on this 17-profile host at 427 ms of config loads plus 506 ms of
        hook registration, with every ``sync_from_config`` loading the config a SECOND time, and
        each of those loads much dearer while the rest of the boot competes for the interpreter —
        and ``reconcile_served_profiles`` then walks ``profiles/`` and settles the served set.
        The demand snapshot it refreshes is already warm (``run_bootstrap._start_gateway_early_demand_prewarm``,
        SOU-3125); this pass is the remaining cold item. Inside the 30 s watcher tick that burst was
        measured at p95 49-113 s while the very next ticks were 40-2,800 ms (PRX-5989): the work
        is off-loop already, so what the tick paid was thread-pool hops queueing behind the rest
        of the boot plus the GIL contention the burst caused its neighbours
        (``loop_wakeup_scan`` and ``session_housekeeping`` spiked in the same minute). Startup is
        where that latency is free — adapters are connected and ``_running`` is set — and the
        watcher's own first tick is then a warm no-op.

        Recorded under its own counter (``profile_reconcile_boot``), never the watcher's: this is
        boot cost, and a regression here must stay visible AS boot cost instead of hiding in a
        tick's ``boot_max``.
        """
        from gateway.loop_tick_timing import timed
        start = time.monotonic()
        with timed('profile_reconcile_boot'):
            await self._sync_hooks_on_consent_change()
            applied = (await self.reconcile_served_profiles(reason=reason)
                       if self._multiplex_on() else {})
        logger.info(
            "Boot first pass: %d home(s) hook-synced, %d profile change(s) applied (%.0fms)",
            len(self._hook_consent_signatures or {}),
            sum(len(applied.get(key) or ()) for key in ("added", "removed", "rescanned")),
            (time.monotonic() - start) * 1000.0,
        )
        return applied

    async def _sync_hooks_on_consent_change(self) -> None:
        """Re-sync the shell hooks of each home (launch + served) whose consent allowlist or config
        changed since the last tick, under that home's scope: hooks consented since start register, revoked ones stop
        firing. A home seen for the first time syncs once (consent may have landed after startup
        registered). Hook registration only: adapters and sessions are never touched."""
        from hermes_constants import get_process_hermes_home
        homes = {Path(h) for h in (get_process_hermes_home(), *(self._served_profile_homes or {}).values())}
        self._hook_consent_signatures = await asyncio.to_thread(
            self._resync_changed_hook_consent, homes, self._hook_consent_signatures or {})

    def _resync_changed_hook_consent(self, homes, seen: dict) -> dict:
        """Blocking half of ``_sync_hooks_on_consent_change``; returns the signatures it applied. The
        key covers ``config.yaml`` too: a sync that read an empty/unreadable config applied nothing,
        so its recovery must re-run it. A failed sync is left unrecorded and retried next tick."""
        from agent.shell_hooks import ALLOWLIST_FILENAME, sync_from_config
        from gateway.run import _load_gateway_config, _profile_runtime_scope
        from gateway.run_turn import GatewayTurnMixin
        current = {}
        for home in (h.resolve() for h in homes):
            sig = tuple(_stat_signature(home / name) for name in (ALLOWLIST_FILENAME, "config.yaml"))
            if seen.get(str(home), ()) == sig:
                current[str(home)] = sig
                continue
            load = partial(_load_gateway_config, home / "config.yaml")
            scope = (_profile_runtime_scope(home) if self._multiplex_on()
                     else GatewayTurnMixin._standalone_launch_scope())
            try:
                with scope:
                    sync_from_config(load)
            except Exception:  # health: allow BLE001 -- one home's failure never blocks the others
                logger.warning("shell-hook re-sync failed for %s; retrying next cycle", home, exc_info=True)
                continue
            current[str(home)] = sig
        return current

    # ── reconcile ─────────────────────────────────────────────────────────────────────────────────

    async def reconcile_served_profiles(self, *, reason: str = "request") -> dict[str, Any]:
        """Diff ``profiles/`` (what exists now) against the served set (the process reservation):
        reserve + build the runtime and start adapters for new profiles, tear down, unroute and
        unreserve deleted ones, (re)build adapters for served profiles whose config/.env changed.
        Other profiles' adapters are never touched. A new profile whose home another gateway owns or
        whose store is unusable is parked (logged, not served) and left for an explicit rescan.
        Returns ``{"added", "removed", "rescanned", "parked", "served_profiles"}``."""
        from gateway.run import MultiplexConfigError
        from hermes_cli.profiles import profiles_to_serve
        result: dict[str, Any] = {"added": [], "removed": [], "rescanned": [], "parked": [], "reason": reason}
        if not self._multiplex_on() and not await self._multiplex_for_added_profile():
            return {**result, "multiplex": False, "served_profiles": self.served_profile_names()}
        if not self._running or self._served_profile_homes is None:
            # Startup enumerates profiles/ itself; a rescan before it finishes has nothing to diff against.
            return {**result, "pending": True, "served_profiles": self.served_profile_names()}
        async with self._reconcile_lock():
            active = getattr(self, "_primary_profile_name", None) or "default"
            # Demand feeds the ranking the enumeration below applies, and reads every named
            # profile's cron store + .env: off the loop, once per tick, exactly like the served-set
            # signature sweep further down (SOU-3075 §3.4).
            await self._refresh_demand_offloop()
            live = {str(name): Path(home) for name, home in
                    await asyncio.to_thread(profiles_to_serve, True)}
            listed = set(live)  # a probe skip below is not a deletion: it never clears a parked verdict
            known = dict(self._served_profile_homes or {})
            from gateway.status import live_gateway_pid_for_home

            blocked = set()
            timed_out = set()
            warned = self._profile_own_gateway_warned or set()
            timeout_warned = self._profile_probe_timeout_warned or set()
            for name in list(live):
                if name == active or name in known:
                    continue
                # The probe can end in a control-socket read that has no timeout of its own
                # (a Windows named pipe stalls there until its peer answers). Inline on the
                # loop thread it parked shutdown_watchdog liveness probes and the multiplexer
                # was hard-killed with exit 75 (#132547). Probe off the loop, bounded: a
                # stalled probe only wedges one bounded housekeeping worker for this cycle
                # (wait_for cancels the still-queued await), never the event loop.
                try:
                    pid = await asyncio.wait_for(
                        self._run_housekeeping_in_executor(live_gateway_pid_for_home, live[name]),
                        timeout=_OWN_GATEWAY_PROBE_TIMEOUT_SECS)
                except TimeoutError:
                    # Unprovable is not "own gateway running": skip it this cycle without
                    # warning about (or remembering) a gateway that may not exist. Warn once
                    # per stall; a peer that stays wedged repeats at DEBUG every cycle.
                    timed_out.add(name)
                    log = logger.debug if name in timeout_warned else logger.warning
                    log("[MULTIPLEX] Own-gateway probe for profile '%s' timed out; "
                        "not serving it this cycle", name)
                    del live[name]
                    continue
                if pid is not None:
                    blocked.add(name)
                    if name not in warned:
                        logger.warning("[MULTIPLEX] Profile '%s' still runs its own gateway; "
                                       "stop it before the host can serve this profile", name)
                    del live[name]
            self._profile_own_gateway_warned = blocked
            self._profile_probe_timeout_warned = timed_out
            sigs = self._served_profile_signatures or {}
            from gateway.run_runtime import parked_profile_map, sync_capacity_parked, unpark_profile
            from hermes_cli.gateway_multiplex_capacity import CAPACITY_REASON_PREFIX
            parks = dict(parked_profile_map(self))
            # A ceiling deferral is THIS tick's config talking, not a failed attempt: it must not
            # block a re-serve once the ceiling (or the demand rank) makes room, so it is excluded
            # from the ``parked`` filter below and re-derived by the fold at the end of the tick.
            # Every other park (unusable store, home owned elsewhere, config refusal) is a verdict
            # a rescan must retry — unparked only once the profile directory is gone. Without this
            # a demand flip could never hot-serve a profile the ceiling had parked (SOU-3075 A2).
            capacity_parked = {n for n, why in parks.items()
                               if str(why).startswith(CAPACITY_REASON_PREFIX)}
            for name in [n for n in parks if n not in listed and n not in capacity_parked]:
                unpark_profile(self, name)
            parked = set(self._parked_profile_names()) - capacity_parked
            # The watcher never re-parks the same profile every cycle; a creator's explicit signal does.
            retry_parked = reason == "control-socket"
            added = [n for n in live if n not in known and n != active
                     and (retry_parked or n not in parked)]
            removed = [n for n in known if n not in live and n != active]
            # A capacity removal is not a deletion: a profile the ceiling just deferred keeps its
            # slot for this cycle while it holds live work (a turn in flight, a /bg agent, an
            # admitted-but-unstarted task), and is published as ``holding`` instead. Evicting it
            # here would let a cron edit kill the profile a user is talking to (SOU-3075 §6).
            from hermes_cli import gateway_multiplex_capacity as capacity
            held, added = self._hold_busy_capacity_removals(
                removed, added, live, capacity.deferred_by_capacity())
            if held:
                removed = [n for n in removed if n not in held]
                live.update({n: known[n] for n in held})
            # Signature compare off the loop too: it stats config.yaml + .env for every served
            # profile, and this runs on the 30 s watcher while turns are executing.
            candidates = [n for n in live if n in known and n != active and n not in added]
            current_sigs = await asyncio.to_thread(
                _serve_signatures, {n: live[n] for n in candidates})
            changed = [n for n in candidates if current_sigs[n] != sigs.get(n)]
            current = {n: h for n, h in live.items() if n in known or n == active}
            applied = await self._apply_profile_changes(
                current, added, removed, changed, reason=reason, live=live, result=result,
                held=held)
            # Publish this tick's verdict even when it applied no diff: ``parked_profiles`` and
            # ``serving_decision`` are the records a client fails fast on and `hermes gateway
            # status` prints, and a quiet tick used to leave an older (or empty) one in place while
            # profiles were in fact unserved (PRX-5734). A decision can also move without moving the
            # served set at all — cost text, manifest health, a hold — which is why SOU-3075 §0
            # found the publication missing entirely. Both calls are dict compares when nothing moved.
            with _log_suppressed(logging.DEBUG, "serving decision publication failed", exc_info=True):
                from gateway.run_runtime import publish_serving_decision
                sync_capacity_parked(self)
                publish_serving_decision(self)
            return applied

    async def _refresh_demand_offloop(self) -> None:
        """Refresh the demand snapshot for this tick, off the event loop (§3.4).

        A no-op (never a failure) when the host sets no ceiling: nothing is ranked, so nothing is
        read. The read is authoritative rather than TTL-gated so a 30 s tick always sees a cron or
        manifest change within the acceptance window; the snapshot's own 5 s TTL then makes the
        per-route reads inside the tick zero-I/O.
        """
        from hermes_cli import gateway_multiplex_capacity as capacity
        if capacity.profile_ceiling() is None:
            return
        from hermes_cli import gateway_multiplex_demand as demand
        await asyncio.to_thread(demand.refresh)

    def _hold_busy_capacity_removals(self, removed, added, live, deferred_by_capacity):
        """Hold capacity deferrals that hold live work, and park the newcomers they displace.

        Returns ``(held, added)``: *held* stays served this cycle (re-added to *live* by the caller);
        *added* no longer contains the profiles whose slots the holds took — the ceiling promoted
        them into the held slots, so they stay parked with a reason naming the holder. Only a
        CEILING deferral is held: a deletion stays authoritative (its directory is gone, and
        re-adding it would resurrect a profile the operator removed). The set is recomputed every
        tick, so it clears the tick the holder idles or the ceiling makes room.
        """
        from gateway.run_runtime import profile_is_busy, record_capacity_hold
        held = [n for n in removed if n in deferred_by_capacity and profile_is_busy(self, n)]
        displaced: list[str] = []
        if held:
            displaced = [n for n in added if n not in held][:len(held)]
            for name in displaced:
                live.pop(name, None)
            added = [n for n in added if n not in displaced]
        record_capacity_hold(self, held, displaced)
        return held, added

    async def _apply_profile_changes(self, current, added, removed, changed, *, reason,
                                     live=None, result=None, held=()):
        """Apply a selected diff under the reconcile lock, shared by the watcher and the
        serve/unserve control verbs. New profiles first get a runtime (reservation + store); one
        whose home another gateway owns or whose store is unusable is parked, not served."""
        from gateway.run import MultiplexConfigError
        from gateway.run_runtime import park_profile, unpark_profile
        from hermes_cli.profiles import profiles_to_serve
        active = getattr(self, "_primary_profile_name", None) or "default"
        result = dict(result or {"added": [], "removed": [], "rescanned": [], "parked": [], "reason": reason})
        result.setdefault("parked", [])
        live = live if live is not None else dict(current)
        known = dict(self._served_profile_homes or {})
        sigs = self._served_profile_signatures or {}
        if not (added or removed or changed):
            return {**result, "served_profiles": self.served_profile_names()}
        from hermes_cli.gateway_multiplex_capacity import MAX_PROFILES_KEY, deferred_by_capacity
        deferred = deferred_by_capacity()
        for name in removed:
            # An unserving that a client will feel must say WHY, in the one place an operator looks
            # besides the state file: a ceiling deferral is a config choice with a remedy, and it
            # used to be indistinguishable from a deleted profile (PRX-5734).
            why = (f"gateway.{MAX_PROFILES_KEY} defers it ({deferred[name]}) — raise the ceiling or "
                   "move it up gateway.multiplex_serve_priority to keep it served") if name in deferred \
                else "its profile directory is gone"
            log = logger.warning if name in deferred else logger.info
            log("[MULTIPLEX] No longer serving profile '%s' — %s", name, why)
            await self._unserve_profile(name, known[name], reason=why)
            result["removed"].append(name)
        for name in list(added):
            reason_ = await self._serve_profile_runtime(name, live[name])
            if reason_ is not None:
                added.remove(name)
                park_profile(self, name, reason_)
                result["parked"].append(name)
                continue
            unpark_profile(self, name)
            current[name] = live[name]
        claimed = self._live_resource_claims(active)
        transient_failed = set()
        config_parked = set()
        for name in added + changed:
            # Only acknowledge the configuration observed before connecting;
            # a setup save during an awaited handshake needs another scan.
            scan_signature = profile_serve_signature(current[name])
            try:
                connected = await self._start_one_profile_adapters(name, current[name], claimed)
            except MultiplexConfigError as exc:
                # Runtime config refusal is a real park, not an adapter-less served profile:
                # withdraw the authority/tickets, release the reservation and publish the
                # terminal parked verdict.  A control-socket rescan can explicitly retry it.
                logger.error("[MULTIPLEX] Profile '%s' not served: %s", name, exc)
                home = current.pop(name)
                await self._unserve_profile(name, home)
                park_profile(self, name, str(exc))
                result["parked"].append(name)
                config_parked.add(name)
                continue
            except Exception:
                logger.error("[MULTIPLEX] Failed to start adapters for profile '%s'", name, exc_info=True)
                connected = 0
                # A transient failure is not the deliberate park above: leave the signature
                # unacknowledged so the next reconcile retries the connect.
                transient_failed.add(name)
            else:
                sigs[name] = scan_signature
            if name in added:
                logger.info("[MULTIPLEX] Now serving profile '%s' (%s adapter(s) connected; %s)", name, connected, reason)
                result["added"].append(name)
            else:
                logger.info("[MULTIPLEX] Re-scanned profile '%s' after config/.env change (%s adapter(s) connected)", name, connected)
                result["rescanned"].append(name)
        if config_parked:
            added = [name for name in added if name not in config_parked]
        self._served_profile_signatures = sigs
        # A profile deleted while an adapter above was still connecting must not be recorded back
        # (the deleter's signal timed out against this lock and rmtree already ran).
        # Held profiles are absent from the ceiling's enumeration by definition (that is why they
        # were candidates for eviction); without the union this prune would unserve the very holder
        # the previous step kept. A genuinely deleted profile is never in *held*.
        live_now = ({str(name) for name, _home in await asyncio.to_thread(profiles_to_serve, True)}
                    | {str(name) for name in held})
        for name in [n for n in current if n not in live_now and n != active]:
            await self._unserve_profile(name, current.pop(name))
            result["removed"].append(name)
            added = [n for n in added if n != name]
        self._record_served_profiles(active, list(current.items()))
        # ``_note_served_profiles`` fills a missing signature with the current one; that refill
        # would park a transiently-failed profile exactly like the config-error case above.
        for name in transient_failed:
            if isinstance(self._served_profile_signatures, dict):
                self._served_profile_signatures.pop(name, None)
            # A cached config with no live adapters is owed a home-channel notice nothing can
            # deliver, and the planned-restart marker then never clears.
            configs = getattr(self, "_profile_configs", None)
            if isinstance(configs, dict):
                configs.pop(name, None)
        if added:
            from gateway.run_runtime import recover_served_native_rows
            await recover_served_native_rows(self)
            await self._after_profiles_added([(n, current[n]) for n in added])
        result["served_profiles"] = self.served_profile_names()
        return result

    def _parked_profile_names(self) -> list:
        """Profiles that exist but could not be served (unusable store, home owned elsewhere); boot's
        ``initialize_gateway_runtime`` parks into the same published ``name -> reason`` map."""
        from gateway.run_runtime import parked_profile_map
        return list(parked_profile_map(self))

    async def _multiplex_for_added_profile(self) -> bool:
        """Start multiplexing in place when this gateway booted standalone ONLY because the host had
        one profile and a second now exists (#141623). Boot settles the mode once, so without this a
        profile created afterwards (Desktop's New Agent, ``hermes profile create``) was never served:
        every ticket, session and delete for it was refused until a manual restart.

        No restart and no drain: the launch env is frozen and unscoped secret reads fail closed
        (``activate_multi_profile_hosting``, the hosted-room precedent), the launch home becomes the
        first entry of the reservation, and the caller's ordinary hot-serve reconcile serves the new
        profile. Turns already running keep their bound scope; the launch profile's later work binds
        its own scope through ``_standalone_launch_scope`` and the re-wired primary handlers. Any
        other standalone verdict, or a preflight blocker now, leaves the gateway as it is; nothing is
        latched, so the next rescan or watcher tick retries once the blocker clears. The switch is an
        automatic migration, so ``gateway.auto_multiplex_migration: false`` holds it back the same way
        (re-read on every rescan: setting it back to true converges without a restart). True when armed."""
        from hermes_cli.gateway_multiplex_mode import (
            AUTO_MIGRATION_OPTED_OUT_REASON, SINGLE_PROFILE_REASON, MultiplexDecision, log_multiplex_decision,
            record_multiplex_decision, recorded_standalone_reason)
        from hermes_cli.profiles import profiles_to_serve
        from hermes_constants import get_default_hermes_root, get_process_hermes_home
        # Serialized with reconcile: a second rescan (watcher + creator) resuming mid-activation
        # reset the reservation the first one's reconcile had already grown.
        async with self._reconcile_lock():
            if self._multiplex_on():
                return True  # a concurrent rescan activated while this one waited
            if not self._running or getattr(self, "_draining", False):
                return False  # shutdown began: never arm fail-closed secrets under drain-time work
            if recorded_standalone_reason() not in (SINGLE_PROFILE_REASON, AUTO_MIGRATION_OPTED_OUT_REASON):
                return False
            launch = get_process_hermes_home().resolve()
            if launch != get_default_hermes_root().resolve():
                return False  # only the default home runs the multiplexer (``_launch_home_may_multiplex``)
            if len(profiles_to_serve(multiplex=True, include_parked=True)) < 2:
                return False
            blocker = await self._run_housekeeping_in_executor(_added_profile_blocker, launch)
            if blocker is not None:
                if blocker != self._single_profile_blocker_logged:
                    logger.warning("[MULTIPLEX] A new profile exists but this gateway stays standalone: %s", blocker)
                    self._single_profile_blocker_logged = blocker
                if blocker == AUTO_MIGRATION_OPTED_OUT_REASON:
                    # Published so `gateway ensure` refuses the new profile naming the opt-out (Desktop shows
                    # that detail) instead of "only one profile exists".
                    record_multiplex_decision(MultiplexDecision(False, "guard", blocker))
                return False
            if not self._running or getattr(self, "_draining", False):
                return False  # shutdown began during the preflight await
            from hermes_constants import hermes_home_key
            from tools.mcp_tool_scope import rekey_unscoped_connections
            from tui_gateway.launch_profile_policy import activate_multi_profile_hosting
            activate_multi_profile_hosting()
            rekey_unscoped_connections(hermes_home_key(launch))
            self.config.multiplex_profiles = True
            self.config._runtime_profile_homes = (("default", launch),)
            registry = getattr(self, "session_authorities", None)
            if registry is not None:
                registry.multiplexed = True
            for adapter in list(self.adapters.values()):
                adapter.set_message_handler(self._primary_message_handler())
                adapter.set_busy_session_handler(self._primary_busy_session_handler())
                adapter.set_platform_event_handler(self._primary_platform_event_handler())
            self._record_served_profiles("default", [("default", launch)])
            log_multiplex_decision(MultiplexDecision(True, "default", "a second profile was created"))
            logger.info("[MULTIPLEX] A second profile exists; serving every profile from this process (no restart)")
            return True

    async def _serve_profile_runtime(self, name: str, home: Path) -> Optional[str]:
        """Grow the reservation by *home* and build its session authority (boot's per-secondary
        steps). Returns the park reason — reservation released — when another gateway owns the home
        (a stray per-profile daemon) or its store cannot be opened; None when served. An
        adapters-only runner (no authority registry) grows the reservation alone."""
        from gateway.run_runtime import release_profile_home, reserve_profile_home, serve_profile_runtime
        from gateway.runtime_ownership import OwnershipConflict
        try:
            reserve_profile_home(self, name, home)
        except (OwnershipConflict, OSError) as exc:
            logger.error("[MULTIPLEX] Profile '%s' not served: %s", name, exc)
            return f"home owned by another gateway: {exc}"
        if getattr(self, "session_authorities", None) is None:
            return None
        try:
            await serve_profile_runtime(self, name, home)
        except Exception as exc:
            if self.session_authorities.for_home(home) is not None:
                # Retirement of the failed attempt missed its deadline: its writer may still use
                # this home, so the reservation and authority stay; the profile is parked (no
                # adapters) and the batch goes on for every other profile.
                logger.error("[MULTIPLEX] Profile '%s' not served (%s); a turn outlived its Stop, "
                             "ownership retained", name, exc, exc_info=True)
                return f"stopping after a failed serve; ownership retained: {exc}"
            logger.error("[MULTIPLEX] Profile '%s' not served: its session store is unusable (%s): %s",
                         name, home, exc, exc_info=True)
            release_profile_home(self, home)
            return f"session store unusable: {exc}"
        return None

    def _live_resource_claims(self, active: str) -> dict[tuple, str]:
        """Startup's ``claimed`` map rebuilt from what is live now: primary claims plus every connected
        secondary's credential/listener, so a hot-added profile reusing a token is parked, never a
        second poller."""
        claimed = self._primary_resource_claims(active)
        for profile_name, adapters in (getattr(self, "_profile_adapters", None) or {}).items():
            for platform, adapter in list(adapters.items()):
                for claim in (self._adapter_credential_claim(platform, adapter),
                              self._adapter_listener_claim(platform, adapter)):
                    if claim is not None:
                        claimed[claim] = profile_name
        return claimed

    async def _after_profiles_added(self, profile_homes) -> None:
        """Per-profile startup side effects for hot-added profiles: log routing + scoped MCP discovery."""
        from gateway.run import _enable_multiplex_log_routing, _profile_runtime_scope
        from contextvars import copy_context
        with _log_suppressed(logging.DEBUG, "log routing refresh failed", exc_info=True):
            _enable_multiplex_log_routing(self.config)
        from tools.mcp_oauth import suppress_interactive_oauth
        loop = asyncio.get_running_loop()
        for profile_name, profile_home in profile_homes:
            try:
                from tools.mcp_tool_discovery import discover_mcp_tools
                with _profile_runtime_scope(Path(profile_home)), suppress_interactive_oauth():
                    await loop.run_in_executor(None, copy_context().run, discover_mcp_tools)
            except Exception:
                logger.warning("MCP tool discovery failed for profile '%s'", profile_name, exc_info=True)

    async def _unserve_profile(self, name: str, home: Path, *, reason: str = "") -> None:
        """Stop and unroute one profile: cancel its reconnects, tear down its adapters, drop its
        bookkeeping and release this process's handles into its home so the deleter's rmtree succeeds.
        The releasing process is not always the deleter (#130244): under multiplexing THIS gateway
        process routes the profile's logs and owns its scoped MCP servers, so both must be released
        here too or the deleter's rmtree still fails on ``logs/.__agent.lock`` / ``mcp-stderr.log``.

        The whole teardown runs inside the removed profile's own runtime scope: adapter disconnect
        hooks, the agent-cache eviction (provider/memory shutdown) and the state/memory handle
        release all read config and credentials at call time, and this coroutine runs on the
        reconcile task with no profile bound — unscoped they resolved against the LAUNCH home, so a
        teardown hook needing this profile's credential failed closed (or, worse, borrowed the launch
        profile's). Secrets are not re-hydrated: teardown must not block the loop on a source fetch.
        """
        from gateway.run import _profile_runtime_scope, _write_runtime_status_quiet
        from gateway.run_runtime import release_profile_home, unserve_profile_runtime
        # Retire before teardown; a False verdict (writer outlived its Stop) is honoured below.
        retired = getattr(self, 'session_authorities', None) is None or await unserve_profile_runtime(self, home)
        pending = (getattr(self, "_profile_failed_platforms", None) or {}).pop(name, None) or {}
        tasks = [t for t in pending.values() if isinstance(t, asyncio.Task) and not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=self._adapter_disconnect_timeout_secs())
        with _profile_runtime_scope(Path(home), hydrate_secrets=False):
            adapters = (getattr(self, "_profile_adapters", None) or {}).pop(name, None) or {}
            for platform, adapter in list(adapters.items()):
                await self._bounded_adapter_teardown(adapter, platform, profile=name)
            # Its ``<name>:<platform>`` runtime entries describe a profile that no longer exists.
            _write_runtime_status_quiet(drop_profile_platforms=name)
            for attr in ("pairing_stores", "_busy_text_modes_by_profile", "_busy_input_modes_by_profile",
                         "_busy_text_timing_by_profile", "_human_delay_by_profile", "_profile_configs"):
                store = getattr(self, attr, None)
                if isinstance(store, dict):
                    store.pop(name, None)
            if isinstance(self._served_profile_homes, dict):
                self._served_profile_homes.pop(name, None)
            if isinstance(self._served_profile_signatures, dict):
                self._served_profile_signatures.pop(name, None)
            from gateway.session import _session_key_namespace
            prefix = _session_key_namespace(name) + ":"
            cache = getattr(self, "_agent_cache", None)
            for key in [k for k in list(cache or {}) if str(k).startswith(prefix)]:
                with _log_suppressed(logging.DEBUG, "agent eviction failed for %s", key, exc_info=True):
                    self._evict_cached_agent(key)
            if not retired:
                # A writer outlived its Stop and may still write this home: the profile leaves the
                # served set (ingress is gone above) but keeps its reservation and store handles until
                # this process exits. Raising here aborted the reconcile for every other profile.
                logger.error("[MULTIPLEX] Profile '%s' unrouted, but a turn outlived its Stop; ownership retained", name)
                return
            # The reservation (gateway.lock) is released LAST: a deleter waiting on it may rmtree the
            # home as soon as it goes, so every handle this process holds there must close first.
            try:
                with _log_suppressed(logging.DEBUG, "profile handle release failed", exc_info=True):
                    from hermes_state_registry import close_all_under
                    close_all_under(home)
                with _log_suppressed(logging.DEBUG, "memory-store release failed", exc_info=True):
                    from plugins.memory import import_provider_module  # catalog plugin since it left core
                    import_provider_module("holographic", "store").MemoryStore.release_all_under(home)
                # The gateway process is a second holder the CLI-side delete cannot reach: its routed
                # log files (logs/.__agent.lock, logs/.__errors.lock) and its scoped MCP servers'
                # mcp-stderr.log handle live HERE, so unserve must release them in this process too
                # (#130244) — the same two calls ``delete_profile`` makes for the deleting process.
                # Both calls BLOCK (the log release stops/joins the QueueListener, which can be inside
                # a ConcurrentRotatingFileHandler emit waiting on another process's rotation lock; the
                # MCP shutdown waits on future.result for the MCP loop), so they run off the loop in a
                # worker thread — otherwise every other served profile's events stall behind them.
                # ``asyncio.to_thread`` copies the context, so the surrounding
                # ``_profile_runtime_scope`` home/secret override stays visible in the worker.
                with _log_suppressed(logging.DEBUG, "scoped MCP shutdown failed", exc_info=True):
                    from hermes_constants import hermes_home_key
                    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
                    await asyncio.to_thread(shutdown_mcp_servers, scope=hermes_home_key(home))
                with _log_suppressed(logging.DEBUG, "profile log handler release failed", exc_info=True):
                    from hermes_logging import release_profile_log_handlers
                    await asyncio.to_thread(release_profile_log_handlers, home)
            finally:
                release_profile_home(self, home)
            logger.info("[MULTIPLEX] Profile '%s' unserved — %d adapter(s) stopped and unrouted%s",
                        name, len(adapters), f" ({reason})" if reason else "")


def _profile_lifecycle_verb(runner, *, serve: bool):
    """Build on the owning loop; socket handlers themselves run on executor threads."""
    loop = asyncio.get_running_loop()

    async def apply(name):
        from hermes_cli.profiles import profiles_to_serve, profile_is_parked
        if not runner._multiplex_on() or not runner._running or runner._served_profile_homes is None:
            return {"error": "host multiplexer is not ready"}
        async with runner._reconcile_lock():
            active = getattr(runner, "_primary_profile_name", None) or "default"
            if not isinstance(name, str) or not name or name == active:
                return {"error": "a non-launch profile name is required"}
            known = dict(runner._served_profile_homes)
            if not serve:
                if name not in known:
                    return {"error": f"profile '{name}' is not served"}
                await runner._unserve_profile(name, known.pop(name), reason="stopped on request")
                runner._record_served_profiles(active, list(known.items()))
                return {"unserved": name, "served_profiles": runner.served_profile_names()}
            installed = dict(profiles_to_serve(True, include_parked=True))
            if name not in installed:
                return {"error": f"unknown profile '{name}'"}
            if name != "default" and profile_is_parked(installed[name]):
                return {"error": f"profile '{name}' is parked (gateway.parked)"}
            if name in known:
                return {"error": f"profile '{name}' is already served"}
            known[name] = installed[name]
            result = await runner._apply_profile_changes(known, [name], [], [], reason="control-socket")
            if name not in result["served_profiles"]:
                return {"error": f"profile '{name}' was removed or parked during startup"}
            return {"served": name, "served_profiles": result["served_profiles"]}

    def handler(params):
        future = asyncio.run_coroutine_threadsafe(apply(params.get("name")), loop)
        try:
            return future.result(timeout=5.0)
        except TimeoutError:
            # Keep running: the client must not interpret an incomplete teardown as stopped.
            return {"pending": True, "served_profiles": runner.served_profile_names()}
        except Exception as exc:
            logger.warning("Profile lifecycle request failed", exc_info=True)
            return {"error": f"{type(exc).__name__}: {exc}"}

    return handler


def unserve_profile_verb(runner):
    return _profile_lifecycle_verb(runner, serve=False)


def serve_profile_verb(runner):
    return _profile_lifecycle_verb(runner, serve=True)


def want_profile_verb(runner):
    """Control verb: a client waiting on this profile's admission files LIVE demand for it.

    ``hermes_cli.gateway_multiplex_demand.note_wanted`` is the rank's only live reading (PRX-5738)
    and it lives in this process, so a run dispatched to a ceiling-deferred profile — the
    ``profile_serving`` verdict — has to file the sighting here. The next reconcile tick then ranks
    the profile IN, evicting a profile that holds no work, instead of the run waiting out a slot
    nothing was going to free (PRX-5963).

    Pure sighting: it never touches the served set, so a client cannot push this host past
    ``gateway.multiplex_max_profiles`` — the ceiling still makes the trade, and it still decides.
    """

    def handler(params):
        name = str((params or {}).get("name") or "").strip()
        if not name:
            return {"error": "name required"}
        from hermes_cli import gateway_multiplex_demand as demand
        demand.note_wanted(name)
        served: list[str] = []
        with _log_suppressed(logging.DEBUG, "served_profile_names failed", exc_info=True):
            served = [str(n) for n in runner.served_profile_names()]
        return {"wanted": name, "served": name in served,
                "ttl_seconds": demand.WANTED_TTL_SECONDS}

    return handler


def _mcp_config_reconciler(runner=None):
    """Housekeeping chore keeping live MCP servers in step with ``mcp_servers`` on disk, every tick
    after the first (startup discovery owns that one). Reconciling on DRIFT rather than only on a
    config EDIT is what brings back a server whose FIRST connect failed (#112445): it never reached
    ``_servers``, so the parked self-probe — a property of a task that connected once — cannot revive
    it, and its config never changes. The reconcile is a cached config read plus set compares when
    nothing moved; a server dropped from config is torn down (a parked one otherwise self-probes
    every ``_PARKED_RETRY_INTERVAL`` for the life of the process) and a missing one is reconnected
    only once its per-server connect cooldown (30s→600s backoff) has lapsed, so a chronically failing
    server is retried on that schedule, not every tick. Interactive OAuth is suppressed — this runs
    on a housekeeping thread nobody is watching."""
    primed: set = set()

    def _reconcile_current(label: str) -> None:
        from tools.mcp_oauth import suppress_interactive_oauth
        from tools.mcp_tool_discovery import reconcile_mcp_servers_with_config
        if label not in primed:
            primed.add(label)
            return  # first tick: startup discovery already reflects this config (or is still running)
        with suppress_interactive_oauth():
            result = reconcile_mcp_servers_with_config()
        if result["removed"] or result["added"]:
            logger.info("MCP servers reconciled with config (%s): removed=%s added=%s",
                        label, result["removed"], result["added"])

    return lambda: _for_each_served_profile(runner, _reconcile_current)


def _for_each_served_profile(runner, body) -> None:
    """Run ``body(profile_label)`` once per served profile, inside that profile's runtime scope.

    Housekeeping runs on a bare thread with no turn on the stack, so nothing binds a profile for it:
    ``get_hermes_home()`` and ``get_secret()`` see the LAUNCH profile's values, and under
    ``gateway.multiplex_profiles`` a fail-closed credential read logs ``no profile secret scope on a
    multiplexed call`` on every tick (the skills-sync pulls resolved Nous credentials this way, four
    WARNINGs per hourly tick per chore). A single-profile gateway runs ``body`` once, unscoped:
    there the process env IS the profile's own value — unless a hosted room already flipped the
    process-wide guard (#112878), in which case the launch profile's OWN scope is bound, as
    ``run_turn.py::_standalone_launch_scope`` does for turns."""
    from gateway.run import _multiplex_profile_homes, _profile_runtime_scope
    config = getattr(runner, "config", None)
    if not getattr(config, "multiplex_profiles", False):
        from gateway.run_turn import GatewayTurnMixin
        with GatewayTurnMixin._standalone_launch_scope():
            body("default")
        return
    for profile_name, profile_home in _multiplex_profile_homes(config):
        # One boundary per profile: callers (``_housekeeping_chore``) catch only at the tick level,
        # so one profile's unreadable store or broken .env abandoned every profile after it, on
        # every tick. The launch store failing is reachable (``_init_session_db`` tolerates it and
        # keeps running), and serve defers every served profile's sweep to this loop.
        try:
            with _profile_runtime_scope(Path(profile_home)):
                body(str(profile_name))
        except Exception as exc:
            logger.debug("Housekeeping for profile %s skipped: %s", profile_name, exc)


def profile_scoped_chore(runner, chore):
    """Wrap a zero-arg housekeeping chore that reads the profile's home, config or credentials so it
    runs once per served profile under that profile's scope (see ``_for_each_served_profile``)."""
    return lambda: _for_each_served_profile(runner, lambda _label: chore())


def migrate_profile_identity_verb(runner):
    """Build the ``migrate-profile-identity`` control-verb handler for ``hermes profile rename``
    (#111926). The live multiplexer owns the routing index in memory and writes it back
    periodically, so a CLI-side rewrite of ``agent:<old>:*`` would be clobbered on the next save;
    the CLI therefore asks this process to rekey both durable stores AND ``SessionStore._entries``.
    Runs on the control-socket executor thread; ``rekey_profile_routing`` takes the store lock."""

    def _handler(params: dict) -> dict:
        old, new = str(params.get("old") or "").strip(), str(params.get("new") or "").strip()
        if not old or not new or old == new:
            return {"ok": False, "error": "old/new required and must differ"}
        store = getattr(runner, "session_store", None)
        if store is None:
            return {"ok": False, "error": "live gateway has no session store"}
        acquired = []
        try:
            from hermes_state_registry import acquire, release_or_close
            db_counts: dict[str, dict[str, int]] = {}
            routing_db = getattr(store, "_routing_db", None)
            if routing_db is not None and hasattr(routing_db, "rekey_profile_state"):
                db_counts["routing"] = routing_db.rekey_profile_state(old, new)
            routing_home = getattr(store, "_routing_home", None)
            profile_path = Path(routing_home) / "profiles" / new / "state.db" if routing_home else None
            if profile_path is not None and profile_path.exists():
                profile_db = acquire(profile_path)
                acquired.append(profile_db)
                db_counts["profile"] = profile_db.rekey_profile_state(old, new)
            rekeyed = store.rekey_profile_routing(old, new)
            return {"ok": True, "rekeyed": rekeyed, "db": db_counts}
        except Exception as exc:
            logger.warning("Profile identity migration failed for %r->%r: %s", old, new, exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            for db in acquired:
                try:
                    release_or_close(db)
                except Exception:
                    logger.debug("Failed to release renamed profile state DB", exc_info=True)

    return _handler


def purge_profile_identity_verb(runner):
    """Build the ``purge-profile-identity`` control-verb handler for ``hermes profile delete``
    (#111926, delete side). The live multiplexer owns the routing index in memory and writes it back
    periodically, so a CLI-side DELETE of ``agent:<name>:*`` rows would be undone by its next save;
    the CLI therefore asks this process to drop the durable rows AND ``SessionStore._entries``.

    Deliberately NOT part of ``_unserve_profile()``: that path also unserves names that are still
    alive elsewhere in the identity story — a rename's old name leaves the served set exactly like a
    delete does (its directory is gone either way) — and purging there would race the rekey it is
    supposed to leave intact. Only the delete path invokes this verb. Runs on the control-socket
    executor thread; ``purge_profile_routing`` takes the store lock."""

    def _handler(params: dict) -> dict:
        name = str(params.get("name") or "").strip()
        if not name:
            return {"ok": False, "error": "name required"}
        store = getattr(runner, "session_store", None)
        if store is None:
            return {"ok": False, "error": "live gateway has no session store"}
        try:
            db_counts: dict[str, dict[str, int]] = {}
            routing_db = getattr(store, "_routing_db", None)
            if routing_db is not None and hasattr(routing_db, "purge_profile_state"):
                db_counts["routing"] = routing_db.purge_profile_state(name)
            dropped = store.purge_profile_routing(name)
            return {"ok": True, "dropped": dropped, "db": db_counts}
        except Exception as exc:
            logger.warning("Profile identity purge failed for %r: %s", name, exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return _handler
