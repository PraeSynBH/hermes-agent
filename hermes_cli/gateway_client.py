"""Transport-only RPC client. Closing a viewer never stops its authority."""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager, suppress
import json
import math
import os
from pathlib import Path
import time

# The owner's protocol name; session_contract is dependency-free (no config/plugins load).
from gateway.session_contract import CANONICAL_GATEWAY_PROTOCOL as GATEWAY_WS_PROTOCOL
from hermes_cli.gateway_restart_window import RESTART_WINDOW_REASON

GATEWAY_WS_TICKET_PREFIX = "hermes-gateway-ticket."
# Per-request answer budget. Owner verbs are admissions/reads that answer promptly; the one that
# legitimately runs long passes its own budget (``rpc(..., _timeout=...)``).
DEFAULT_RPC_TIMEOUT = 30
# Transport keepalive for the run socket. Explicit, NOT websockets' defaults, because the default
# 20 s pong deadline is inside this host's measured stall distribution: with the gateway under load
# 127 (1-min) a control-socket ``identify`` round trip ran p50 5.7 s / p90 19.4 s / max 24.5 s, and
# 11 attached runs dropped with "Gateway disconnected; turn outcome is unknown" in the same minute
# (2026-10-18T03:45:10Z, gateway pid 38118 alive throughout, no restart). A ping deadline that
# fires on a *slow* owner tears down a turn the owner is still serving.
#
# 60 s was still below this owner's worst *recoverable* stall. `ws_drop_causes.py --hours 4` over
# 15:22-19:22Z found 73 socket deaths, 99 % of them in three bursts, and the faulthandler dump in
# each burst's tail shows a loop starved by one GIL (100 live threads, 4 of them inside macOS
# peer-cert evaluation, 36 inside filesystem scans) rather than a dead process: at 18:12:14Z
# 11 runs died together, the loop then recovered and the process kept serving. At 18:17:16Z 29 runs
# died at the 60 s deadline *while the owner was still running its turns*; the loop-liveness
# watchdog only confirmed the wedge 27 s later. Raising the deadline 60 -> 300 s deliberately puts
# it ABOVE the leash that decides a stall is fatal: `gateway.loop_watchdog_max_strikes: 6`
# (~240 s of no dispatch before the out-of-loop watchdog hard-exits, see shutdown_watchdog.py), so
# the ping deadline can no longer be the thing that kills a run during a stall the owner survives.
# A genuinely dead owner still fails fast: the socket gets a close frame or FIN, which is the case
# the ping deadline exists for.
GATEWAY_WS_PING_INTERVAL_S = 20.0
GATEWAY_WS_PING_TIMEOUT_S = 300.0
# Admission deadline for the session-ticket mint (``_session_ticket``). The owner answers this
# control-socket verb from its event loop, so the budget is really "how long may the loop stall
# before this run dies at admission". 5 s was measured against a healthy host; on this one the
# verb's own latency under live load (43 active agents, gateway pid 38118) is p50 4.0 s, and
# 5 of 12 mints expired at exactly 5.001 s while the owner went on to answer the same verb in
# 7.6 s / 12.8 s (probe_ticket_latency.py, 2026-10-18T05:27Z). That converts a slow owner into
# a dead run: 43 of the 75 fleet deaths in the 30 min to 05:13Z were this class. 30 s matches
# DEFAULT_RPC_TIMEOUT and keeps margin over the widest observed answer (12.8 s) without
# inventing a new number; a truly dead owner still fails, just 25 s later.
GATEWAY_SESSION_TICKET_DEADLINE_S = 30.0
# The ONE admission budget: ensure -> ticket mint -> ws-open -> ``runtime.describe`` ->
# ``session.create``/``session.resume``, as one deadline. The per-phase numbers above are caps
# INSIDE it, never a sum: they used to be independent, so a starved owner could burn every cap in
# turn. Measured under a wedged owner on 2026-10-19 (fleet, 35 admissions in one minute): the run
# survived 181 s of admission and then died with the generic "no local fallback" — 30 s describe +
# 120 s create, plus ~31 s of ticket/ws — because no phase was accountable to the others. One
# number also makes the failure classifiable: at 181 s a retry chain re-pays it 3 times (9 min of a
# run's life) for an owner that is stalled, not slow.
#
# 120 s is not the admission latency budget (that is ~5.4 s p95 / 7.4 s max, measured over 41
# ``ensure_session`` samples at 30 concurrent runs after the 08:23Z restart); it is the ceiling on
# how long a run may be held before it is told, in its own error, that the OWNER was starving.
#
# The DEFAULT, not the value: ``gateway.admission_budget_seconds`` in config.yaml raises it (the
# floor is this number — a run may be held longer, never killed sooner than the class was designed
# for), because the stall distribution is a property of the HOST: on 2026-10-19 the owner's loop
# stalled 10-25 s under load 38 and past 40 s at load 100, and every run admitted inside those
# windows died here with no knob to widen it (`hermes gateway` has no hand on a run's client).
GATEWAY_ADMISSION_BUDGET_S = 120.0
# A live update fence is polled this often while an admission waits it out, so a 9-minute install
# update (2026-10-19: 18:54-19:03Z) costs a wake-up rather than a run.
UPDATE_FENCE_POLL_INTERVAL_S = 2.0
# A boot in progress is re-observed this often when an ensure call expires while the owner is
# still `starting` (a cold boot longer than the per-call cap), so the rest of the admission budget
# is spent watching an owner that IS coming up instead of reporting it as a failure.
BOOT_POLL_INTERVAL_S = 1.0
# A restart's after-turn wait (`draining`/`restart_wait`) is re-observed this often by a client that
# has new work to place. That owner answers identify in milliseconds and its drain is bounded by its
# own cap, so a drain shorter than a second costs a wake-up rather than a run (PRX-5791).
DRAIN_POLL_INTERVAL_S = 1.0
# The bound on sampling the owner's loop at the moment this run dies at admission: the owner is
# already stalling, so the probe is allowed less than the slack a healthy loop answers in
# milliseconds, and a no-answer is reported as a floor (``owner_loop>2s``), never as healthy.
ADMISSION_STALL_PROBE_TIMEOUT_S = 2.0

# The run's own first in-band RPCs (``runtime.describe`` -> ``session.create``/``session.resume``)
# are NOT prompt admissions: ``session.create`` binds the profile, loads the agent config and opens
# the state DB on the owner's event loop, in the same process that is running every other agent
# turn. Measured on this host 2026-10-18T09:23-09:25Z (``probe_session_create.py``, gateway pid
# 30963, 1-min load 87-100, ~300 threads in that process): ``runtime.describe`` p50 1.5 s / max
# 3.1 s, ``session.create`` p50 21.8 s / max 24.8 s. Against ``DEFAULT_RPC_TIMEOUT`` = 30 s that
# is zero headroom, and it is exactly what the fleet dies of: 83 % of connect deaths in the 2 h to
# 09:09Z carried ``phase=session`` + ``TimeoutError`` (``phase_census.py``), i.e. ticket minted,
# socket open, then the in-band RPC outlived its budget and the run exited before any output.
# 120 s keeps ~5x margin over the widest observed answer while still bounding a genuinely hung
# owner (a dead owner closes the socket instead of going silent). Failure mode if this is wrong:
# a run that used to die at 30 s waits longer before dying.
GATEWAY_STARTUP_RPC_DEADLINE_S = 120.0
# Opening-handshake budget for the run socket (``phase=ws-open``). The owner answers the upgrade
# from its event loop, in the process running every other agent turn, so this budget is really
# "how long may the loop stall before this run dies at admission" — the same question
# GATEWAY_SESSION_TICKET_DEADLINE_S answers for the mint. 10 s was measured against a healthy
# host; on this one the loop is starved continuously. Measured 2026-10-19T02:05-02:25Z (gateway
# pid 20061, 412 threads, 70 live agents, 1-min load ~120): a trivial ``GET /healthz`` on the
# owner's own loopback ran p50 13.0 s / max 16.3 s and two of eight exceeded 20 s, while TCP
# connect stayed at 0.3 ms — the kernel accepts, the loop is the queue. A parallel probe of the
# admission path reproduced the reported death exactly: ``designer`` ws-open 7.04 s (opened),
# ``praxis`` ws-open 10.02 s (``TimeoutError: timed out during opening handshake``). 10 s sits
# below the loop's median, which is why PRX-5511 looked like a dead Design Agent when the owner
# was alive and serving throughout. Measured alone, the upgrade is normally instant (n=8, p50
# 0.08 s / p90 0.29 s) and bimodal: the tail is a stall spike landing inside the handshake
# (7.5-10.0 s seen, 10.02 s observed failing). This host's documented stall tail is p90 19.4 s /
# max 24.5 s on the owner's ``identify`` verb (see GATEWAY_WS_PING_TIMEOUT_S). 30 s sits just
# above that tail and matches GATEWAY_SESSION_TICKET_DEADLINE_S, which answers the identical
# question ("how long may the loop stall") for the mint. A genuinely dead owner is not slow, it
# is refused or FIN'd, which fails this immediately rather than at the deadline.
GATEWAY_WS_OPEN_TIMEOUT_S = 30.0
# A typed ``--api-key`` on a resume that the owner does not bind (``rebind_launch_key``): every CLI says this.
RESUME_KEY_MISMATCH = ("--api-key on resume must be the key this session was launched with "
                       "(a session launched without --api-key keeps its profile credentials).")
# Canonical ``/compress`` answers only after the summary call(s) and the commit: a long session's
# summary routinely takes more than 30 s (``auxiliary.compression.timeout`` defaults to 120 s per
# request, ``compression.context_total_ceiling_seconds`` to 600 s). Same budget as Desktop's
# ``SESSION_COMPRESS_TIMEOUT_MS`` so every surface gives up at the same point.
COMPRESS_RPC_TIMEOUT = 660
# A full event queue parks the socket reader; a consumer that takes no event for this long has
# stalled and is failed. Same bound as the owner's own send deadline (``tui_gateway/ws.py``).
EVENT_STALL_TIMEOUT = 30
# Events read past a full queue while an RPC answer is awaited (the owner sends a re-attached
# session's frames ahead of its ``session.resume`` answer, and a recovery RPC is awaited from the
# queue's only consumer). Bounded: past either cap the reader waits for the consumer again.
EVENT_OVERFLOW_FRAMES = 16384
EVENT_OVERFLOW_BYTES = 8 * 1024 * 1024


def gateway_ws_target(endpoint, ticket):
    """``(url, subprotocols)`` for dialing the owner's canonical WebSocket with a session ticket.

    The owner accepts exactly one ticket subprotocol beside ``GATEWAY_WS_PROTOCOL`` and echoes the
    latter; callers check ``ws.subprotocol == GATEWAY_WS_PROTOCOL`` after the upgrade."""
    url = endpoint.api_origin.replace("https:", "wss:").replace("http:", "ws:") + "/api/ws"
    return url, [GATEWAY_WS_PROTOCOL, GATEWAY_WS_TICKET_PREFIX + ticket]


class GatewayClientError(ValueError):
    pass


class GatewayRPCError(GatewayClientError):
    """A received owner refusal, distinct from an ambiguous transport failure.

    ``str(exc)`` is the bounded reason code; ``code`` is the frame's numeric JSON-RPC code (``None``
    when absent or not an int) and ``data`` its bounded, scalar-only ``error.data`` mapping."""

    def __init__(self, reason, code=None, data=None):
        super().__init__(reason)
        self.code = code
        self.data = data or {}


class GatewayAdmissionTimeout(GatewayClientError):
    """The one admission budget (``GATEWAY_ADMISSION_BUDGET_S``) expired before this run was admitted.

    A distinct class, not a string a consumer has to pattern-match on: a retry chain must be able to
    tell "the owner is starving" from "this install is broken" (the class of every other
    ``no local fallback`` failure). ``reason`` is the stable code such a chain keys on, ``phase``
    names the stage that was left holding the deadline, and ``code_sha`` the build of the owner that
    was holding it — the field whose absence turned the 2026-10-19 investigation into an ``lsof``
    dig. ``evidence`` is what the host was doing at that instant (the owner's loopback latency, the
    1-minute load): the run's log line is the only place a death at admission is ever read from, so
    the stall has to be in it rather than 20 minutes of hand-probing later. ``str(exc)`` keeps the
    original wording so existing readers/log scrapers still match.
    """

    reason = "gateway_admission_timeout"

    def __init__(self, detail, *, phase, elapsed, budget=GATEWAY_ADMISSION_BUDGET_S, code_sha=None,
                 evidence=None):
        self.phase, self.elapsed, self.budget = phase, elapsed, budget
        self.code_sha = code_sha if isinstance(code_sha, str) and code_sha else None
        self.evidence = evidence or None
        owner = f", owner={self.code_sha[:12]}" if self.code_sha else ""
        host = f", {self.evidence}" if self.evidence else ""
        super().__init__(f"{self.reason}: {detail} (budget={budget:g}s{owner}{host})")


def admission_budget_seconds() -> float:
    """``gateway.admission_budget_seconds`` from this home's config.yaml, floored at the default.

    Read through the effective-user-config loader — the gateway's own reader, which does not merge
    ``DEFAULT_CONFIG`` — so an absent key stays absent and this function owns the default. The floor
    is ``GATEWAY_ADMISSION_BUDGET_S``: the knob exists so a host whose loop stalls for longer than
    the class was designed against can still admit runs, never so a typo can kill them sooner.
    Never raises: a config read must not be what ends a run at admission.
    """
    try:
        from hermes_cli.config_effective import load_user_config_effective

        gateway = load_user_config_effective().get("gateway")
        value = gateway.get("admission_budget_seconds") if isinstance(gateway, dict) else None
        budget = float(value)
    except (ImportError, TypeError, ValueError, OSError):
        return GATEWAY_ADMISSION_BUDGET_S
    if not math.isfinite(budget):
        return GATEWAY_ADMISSION_BUDGET_S
    return max(GATEWAY_ADMISSION_BUDGET_S, budget)


def admission_stall_evidence(homes) -> str | None:
    """``owner_loop=25.5s, load1=100.21`` — what the owner and the host were doing at this death.

    Sampled once, bounded by ``ADMISSION_STALL_PROBE_TIMEOUT_S``, and only on the path where the run
    has already lost its budget: the owner's loopback witness answers in milliseconds when the loop
    is healthy, so a probe that does not answer at all is reported as a floor, never as healthy.
    Either half may be missing (no witness for this home, no ``getloadavg`` on Windows).
    """
    from gateway.loop_latency import describe_load, describe_probe, load_average_1m, probe_loop

    loop = describe_probe(probe_loop(homes, timeout=ADMISSION_STALL_PROBE_TIMEOUT_S),
                          timeout=ADMISSION_STALL_PROBE_TIMEOUT_S)
    parts = [part for part in (loop, describe_load(load_average_1m())) if part]
    return ", ".join(parts) or None



class _AdmissionBudget(float):
    """A wait bounded by the run's ONE admission budget — and knows it.

    The marker is what makes the classification exact. Admission verbs are ordinary RPCs issued
    after ``connect_gateway`` has already handed the connection to the caller, so "did this fail
    before the handover?" cannot tell an admission stall from a mid-run one: a ``prompt.receipt``
    read 200 s into a healthy session also times out, and it must stay a retryable connection
    error. Only a wait that was itself derived from the admission budget is starvation.
    """


def admission_timeout(client, cap: float) -> float:
    """``cap`` seconds, or what is left of this run's ONE admission budget when that is less.

    The admission verbs (``runtime.describe`` -> ``session.create``/``session.resume``) share one
    deadline, so a phase that ran long shortens the next instead of queueing a fresh cap behind it.
    A spent budget returns 0: the RPC times out at once and ``GatewayClient.rpc`` turns that
    ``TimeoutError`` into a ``GatewayAdmissionTimeout`` naming the owner build (the value carries
    ``_AdmissionBudget`` so only admission-bounded waits are classified that way).

    The deadline rides on the connection ``connect_gateway`` yields; a peer without one (a test
    double, an embedded server) has no budget to share, and there ``cap`` is the whole story.
    """
    deadline = getattr(client, "admission_deadline", None)
    return cap if deadline is None else _AdmissionBudget(max(0.0, min(cap, deadline - time.monotonic())))


def _bounded_reason(value):
    return value if isinstance(value, str) and value.replace("_", "").isalnum() and len(value) <= 80 else None


def _bounded_error_data(data):
    """At most 16 short keys whose values are bool/int/None, short strings or short lists of them;
    anything else (nested objects, raw remote diagnostics) is dropped, never forwarded."""
    def scalar(value):
        return value is None or isinstance(value, (bool, int)) or (isinstance(value, str) and len(value) <= 200)
    if not isinstance(data, dict):
        return {}
    out = {}
    for key, value in data.items():
        if len(out) >= 16 or not isinstance(key, str) or len(key) > 64:
            continue
        if scalar(value):
            out[key] = value
        elif isinstance(value, list) and len(value) <= 16 and all(scalar(item) for item in value):
            out[key] = list(value)
    return out


def rpc_error(error):
    """The ``GatewayRPCError`` for one received ``error`` member: only bounded reason codes, never
    raw remote diagnostics. A free-text message (``unknown method: x``) falls back to the bounded
    ``data.reason`` the owner sends beside it, then to ``request_failed``."""
    error = error if isinstance(error, dict) else {}
    data = _bounded_error_data(error.get("data"))
    code = error.get("code")
    reason = _bounded_reason(error.get("message", "request_failed")) or _bounded_reason(data.get("reason"))
    return GatewayRPCError(reason or "request_failed",
                           code if isinstance(code, int) and not isinstance(code, bool) else None, data)


class GatewayUnavailableError(GatewayClientError):
    """No ready owner to dial: ``state`` is ``ensure_gateway_runtime``'s verdict (``draining`` while an
    update holds the install, ``starting`` past the deadline, ...). Nothing was submitted."""

    def __init__(self, message, state):
        super().__init__(message)
        self.state = state
        #: Seconds until the host is worth asking again. None = no owner-side schedule to honour.
        self.retry_after_s: float | None = None

    @property
    def retryable(self) -> bool:
        """Whether re-dispatching this run can succeed without anything on this host changing."""
        return self.retry_after_s is not None


class GatewayRestartingError(GatewayUnavailableError):
    """A declared restart window owns the gateway: a bounded, RETRYABLE verdict, never a boot.

    Raised the moment the window is read, so a run dispatched into one fails in seconds and can be
    re-queued for ``retry_after_s`` instead of spending its whole admission budget on a daemon the
    owning job has deliberately stopped (PRA-6692: 6 runs, 120 s each, inside a 187 s window).
    """

    def __init__(self, message: str, retry_after_s: float):
        super().__init__(message, "restarting")
        self.retry_after_s = retry_after_s


class EventQueue(asyncio.Queue):
    """The client's bounded event queue plus an ordered overflow behind it: each ``get`` refills the
    queue from the overflow, so items arrive in read order and ``qsize``/``full`` see only the queue."""

    def __init__(self, maxsize):
        super().__init__(maxsize)
        self.overflow = deque()  # (size, item)
        self.overflow_bytes = 0
        self.moved = asyncio.Event()  # an item left the overflow (or an RPC started waiting)

    def _get(self):
        item = super()._get()
        if self.overflow:
            size, held = self.overflow.popleft()
            self.overflow_bytes -= size
            super()._put(held)
        self.moved.set()
        return item

    def hold(self, item, size=0):
        """Queue *item* behind everything already read, never dropping it."""
        if not self.overflow:
            try:
                self.put_nowait(item)
                return
            except asyncio.QueueFull:
                pass
        self.overflow.append((size, item))
        self.overflow_bytes += size


def put_behind(events, item):
    """Queue a local item (ACP's post-ack recovery) behind every frame read so far; never dropped."""
    if isinstance(events, EventQueue):
        events.hold(item)
    else:
        events.put_nowait(item)


class GatewayClient:
    def __init__(self, websocket, *, admission_deadline=None, admission_started=None,
                 admission_budget=None, code_sha=None):
        self.websocket = websocket
        self.events = EventQueue(maxsize=4096)
        self.pending = {}
        self.sequence = 0
        self.reader = None
        # ``time.monotonic()`` at which the ONE admission budget expires (``connect_gateway`` sets
        # it); absent on a hand-built peer, which ``admission_timeout`` treats as "no budget".
        self.admission_deadline = admission_deadline
        # Enough of the admission's own frame to file an overrun as ``GatewayAdmissionTimeout``:
        # when it started, how long it was, and the build of the owner that held it.
        self.admission_started = admission_started
        self.admission_budget = admission_budget
        self.code_sha = code_sha

    def _read_on(self):
        """Keep reading past a full queue: nothing is held, or an awaited answer may sit behind the
        held frames (its waiter may be the queue's only consumer) and the overflow has room."""
        events = self.events
        return not events.overflow or (
            len(events.overflow) < EVENT_OVERFLOW_FRAMES and events.overflow_bytes < EVENT_OVERFLOW_BYTES
            and any(not future.done() for future in self.pending.values()))

    async def _deliver(self, frame, size):
        self.events.hold(frame, size)
        if self._read_on():
            return
        # Frames the socket already buffered (a CLI resumed from Ctrl-Z) are read without yielding,
        # so the consumer has not had a turn yet: wait for it (or for an RPC that needs the reader).
        # Not wait_for: on 3.11 it swallows a close that lands as the wait completes.
        async with asyncio.timeout(EVENT_STALL_TIMEOUT):
            while not self._read_on():
                self.events.moved.clear()
                await self.events.moved.wait()

    async def __aenter__(self):
        self.reader = asyncio.create_task(self._read())
        return self

    async def __aexit__(self, *exc):
        self.reader.cancel()
        with suppress(asyncio.CancelledError):
            await self.reader

    async def _read(self):
        from websockets.exceptions import ConnectionClosed
        try:
            async for raw in self.websocket:
                frame = json.loads(raw)
                future = self.pending.get(frame.get("id"))
                if future is not None and not future.done():
                    if "error" in frame:
                        future.set_exception(rpc_error(frame["error"]))
                    else:
                        future.set_result(frame.get("result", {}))
                elif "method" in frame:
                    await self._deliver(frame, len(raw))
        except (ConnectionClosed, OSError, ValueError, TimeoutError):
            pass
        finally:
            error = GatewayClientError("Gateway disconnected; turn outcome is unknown. "
                "Resume the printed session ID to check whether it completed or was interrupted; do not resend the work.")
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(error)
            # A stalled consumer is failed rather than silently losing terminal events: behind
            # whatever is held, or in place of the oldest queued frame when nothing is.
            if self.events.full() and not self.events.overflow:
                self.events.get_nowait()
            self.events.hold(error)

    async def rpc(self, method, *, _timeout=None, **params):
        """Send one request and await its answer; ``_timeout`` (seconds, never sent on the wire)
        overrides ``DEFAULT_RPC_TIMEOUT`` for a verb that legitimately takes longer (``math.inf``: no
        deadline; the answer, a hang-up or the caller's cancel ends the wait)."""
        self.sequence += 1
        rid = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[rid] = future
        self.events.moved.set()  # a reader waiting on the consumer reads on toward this answer
        try:
            await self.websocket.send(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}))
            budget = DEFAULT_RPC_TIMEOUT if _timeout is None else _timeout
            return await asyncio.wait_for(future, None if budget == float("inf") else budget)
        except TimeoutError as exc:
            # Only a wait this run bounded with its admission budget is starvation. A mid-run RPC
            # that times out against its own (default) cap — ``prompt.receipt`` 200 s into a healthy
            # session, say — must stay a retryable connection error; filing it under the give-up
            # code would tell the retry chain to abandon a run that only needs to be retried.
            if not isinstance(_timeout, _AdmissionBudget):
                raise
            started = self.admission_started if self.admission_started is not None else time.monotonic()
            raise GatewayAdmissionTimeout(
                f"Gateway connection failed; no local fallback (phase=session, "
                f"elapsed={time.monotonic() - started:.1f}s, TimeoutError: {exc})",
                phase="session", elapsed=time.monotonic() - started,
                budget=self.admission_budget if self.admission_budget is not None else GATEWAY_ADMISSION_BUDGET_S,
                code_sha=self.code_sha) from exc
        finally:
            self.pending.pop(rid, None)


def _session_ticket(home: Path, endpoint, *, purpose="interactive", scope=None, subject=None,
                    timeout=GATEWAY_SESSION_TICKET_DEADLINE_S) -> str:
    from hermes_cli.gateway_runtime import control_home_for
    from hermes_cli.gateway_runtime_discovery import connect_private, _identify_response
    # A served secondary's ticket is minted by the multiplexer's socket, bound to the secondary.
    home = control_home_for(home, endpoint)
    # ``subject`` names the principal a same-user caller acts for (the standalone dashboard's
    # authenticated user) instead of the socket UID; the owner's peer gate stays the boundary.
    request = json.dumps({"protocol": 1, "id": 1, "verb": "session-ticket", "params": {
        "profile_id": endpoint.profile_id, "instance_id": endpoint.instance_id, "purpose": purpose,
        **({"scope": scope} if scope else {}), **({"subject": subject} if subject else {}),
    }}).encode() + b"\n"
    if os.name == "nt":
        from gateway.runtime_bootstrap_windows import query_runtime_control
        data = query_runtime_control(home, request, int(timeout))
    else:
        deadline = time.monotonic() + timeout
        with connect_private(home, timeout) as peer:
            peer.sendall(request)
            data = bytearray()
            while b"\n" not in data:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    raise GatewayClientError("Gateway bootstrap timed out")
                peer.settimeout(budget)
                chunk = peer.recv(4096)
                if not chunk or len(data) + len(chunk) > 65536:
                    raise GatewayClientError("Invalid gateway bootstrap response")
                data.extend(chunk)
    grant = _identify_response(bytes(data))
    if any(grant.get(k) != v for k, v in {
        "profile_id": endpoint.profile_id, "instance_id": endpoint.instance_id, "runtime_protocol": 1,
    }.items()) or not isinstance(grant.get("ticket"), str) or not grant["ticket"]:
        raise GatewayClientError("Gateway bootstrap identity changed; retry launch")
    return grant["ticket"]


@asynccontextmanager
async def connect_gateway(endpoint=None, subject=None, *, new_work=False):
    """Attach to this home's owner. ``endpoint``: an already-discovered ready owner to dial as-is
    (maintenance that must never start a gateway); default ensures one, starting it if needed.
    ``subject``: mint the ticket for this principal (the caller's authenticated dashboard user)
    rather than the socket identity; only a peer the owner already trusts may name one.
    ``new_work``: this client has a turn to place, so it must not attach to an owner that is only
    held reachable for its in-flight work (a restart's after-turn wait) — see ``admit_owner``."""
    from websockets.asyncio.client import connect
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import (DEFAULT_ENSURE_TIMEOUT, attachable, boot_evidence,
                                            ensure_gateway_runtime, gateway_profile_home)
    from urllib.parse import urlsplit

    remote = "" if endpoint is not None else os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip()
    protocols = None
    # Admission is the stage a run dies on that leaves no other trace: the run log records one
    # final string, so a death cannot say whether the ticket mint or the WS open burned the
    # budget. Name the phase and the elapsed time in the error itself (2026-10-18T07:00Z: 8
    # fleet deaths at load 129, 30.8-79.2 s, all reported as this one opaque string).
    #
    # Every phase below spends from ONE budget: ``GATEWAY_ADMISSION_BUDGET_S``, or the larger
    # ``gateway.admission_budget_seconds`` from config.yaml (``admission_budget_seconds()``). Each
    # waits at most its own cap AND at most what the budget has left, so four starved phases fail
    # at the budget instead of at their sum. Before that, ensure + ticket + ws-open + describe +
    # create could each run their cap out in turn and the run died at ~181 s with nothing in the
    # error to classify.
    started = time.monotonic()
    budget = admission_budget_seconds()
    deadline = started + budget
    owner_sha = None
    home = None
    handed_over = False

    def left(cap):
        return max(0.0, min(cap, deadline - time.monotonic()))

    def probe_homes():
        """Homes whose loop-tick witness is this owner's: the endpoint's control home first (a named
        profile is served by the multiplexer's loop), then the profile home, then this process's."""
        control = getattr(endpoint, "control_home", None) if endpoint is not None else None
        return list(dict.fromkeys(Path(h) for h in (control, home, get_hermes_home()) if h))

    def spent(detail):
        """The budget is gone: one class, one stable code, the owner's build and the host's state."""
        return GatewayAdmissionTimeout(detail, phase=phase, elapsed=time.monotonic() - started,
                                       budget=budget, code_sha=owner_sha,
                                       evidence=admission_stall_evidence(probe_homes()))

    async def mint(home, target, **kwargs):
        """The ticket mint, filed under admission when it runs out (it used to escape as a bare
        ``TimeoutError`` before the try below, so the class token never reached the dead run)."""
        from hermes_cli.gateway_runtime_discovery import DiscoveryError

        try:
            return await asyncio.to_thread(_session_ticket, home, target, **kwargs,
                                           timeout=left(GATEWAY_SESSION_TICKET_DEADLINE_S))
        # ``DiscoveryError``: the owner answered the ticket verb with something that is not a grant
        # (``_identify_response``). It used to escape this call as a bare traceback — 11 fleet runs on
        # 2026-10-20 died with ``Traceback ... DiscoveryError: invalid_control_response`` and no
        # phase, so no retry chain could classify them (PRX-5725).
        except (OSError, TimeoutError, DiscoveryError) as exc:
            detail = (f"Gateway connection failed; no local fallback (phase=session-ticket, "
                      f"elapsed={time.monotonic() - started:.1f}s, {type(exc).__name__}: {exc})")
            if isinstance(exc, TimeoutError) or left(1.0) <= 0:
                raise spent(detail) from exc
            raise GatewayClientError(detail) from exc

    async def admit_owner(home):
        """Ensure an owner, waiting out what legitimately takes time instead of failing at it.

        Three states are WAITS inside this run's one admission budget, not verdicts:

        * ``update_paused`` — an update owns the install for minutes (2026-10-19: 18:54-19:03Z) and
          every run started in that window died in under a second on
          ``Gateway draining: update_paused`` — six of them, and the survivors went on to die at the
          budget once the fence cleared. Poll it and admit as soon as it clears; name the fence's
          owner (pid + update run) if the fence outlives the budget.
        * ``starting`` — a cold boot is 45.7-62.1 s on this host (``state/gateway.lifecycle.json``),
          and the ensure call returns ``starting: deadline`` the moment its own cap expires. That
          is a healthy daemon coming up, and it was being reported as ``adapter_failed`` on dispatch
          (PRX-5625: 21 runs in 24 h in this company, 151 host-wide). Keep observing it for as long
          as the budget lasts; a boot that outlives the budget is named as a boot, not as a
          mystery string.
        * a restart's after-turn wait (``draining``/``restart_wait``) for a client that has NEW WORK
          (``new_work``) — that owner stays attachable so a turn parked on approval/clarify can be
          answered (#139041), but its claim path is closed from T=0 of the drain (PRX-5787). A
          spawn that attaches to it boots and dies at its first ``session.create``/submit with
          ``runtime_draining``: 101 failed heartbeat runs across the 21:38 and 23:05-23:25 drains,
          every one a run spent on a wait. Wait for the owner that replaces it instead — the drain
          is bounded by the owner's own cap and the boot after it is the ``starting`` wait above
          (PRX-5791). An interactive client keeps attaching: answering the parked prompt is its job.
        """
        nonlocal phase
        fence = None
        boot = None
        drain = None
        while True:
            # The per-attempt cap is the boot deadline (now above the measured boot cost); the
            # whole loop is bounded by the one admission budget, so a lower cap can only cost
            # re-observation, never a run.
            cap = left(DEFAULT_ENSURE_TIMEOUT)
            if cap <= 0:
                if fence is None and boot is None and drain is None:
                    raise spent("Gateway connection failed; no local fallback "
                                f"(phase={phase}, elapsed={time.monotonic() - started:.1f}s, "
                                "admission budget spent before an owner could be ensured)")
                break
            result = await asyncio.to_thread(ensure_gateway_runtime, home, idle_exit=True, timeout=cap)
            if result.reason_code == "update_paused":
                fence = getattr(result, "detail", None) or "update fence live"
                if left(1.0) <= 0:
                    break
                phase = "update-fence"
                await asyncio.sleep(min(UPDATE_FENCE_POLL_INTERVAL_S, left(UPDATE_FENCE_POLL_INTERVAL_S)))
                continue
            if result.reason_code == RESTART_WINDOW_REASON:
                # A declared restart window: the owning job stopped the gateway and will start it
                # itself, on a schedule this client can read. This is the one case in admission that
                # must NOT wait — the outage is longer than the budget by design (187 s vs 120 s), so
                # waiting can only convert a retryable window into a runtime-boot failure. Fail in
                # seconds, name the lease, let the dispatcher re-queue (PRA-6692).
                left_s = result.retry_after_s if result.retry_after_s is not None else 0.0
                raise GatewayRestartingError(
                    "Gateway restarting: a declared restart window owns the gateway, so this run was "
                    f"never admitted (elapsed={time.monotonic() - started:.1f}s, "
                    f"retryAfterS={int(round(left_s))}, retryable=true): "
                    f"{result.detail or 'restart window open'}", float(left_s))
            if new_work and result.state == "draining" and attachable(result):
                drain = f"{result.state}: {result.reason_code or 'draining'}"
                if left(1.0) <= 0:
                    break
                phase = "runtime-drain"
                await asyncio.sleep(min(DRAIN_POLL_INTERVAL_S, left(DRAIN_POLL_INTERVAL_S)))
                continue
            if result.state == "starting":
                boot = f"{result.state}: {result.reason_code or 'not_ready'}"
                if left(1.0) <= 0:
                    break
                phase = "runtime-boot"
                await asyncio.sleep(min(BOOT_POLL_INTERVAL_S, left(BOOT_POLL_INTERVAL_S)))
                continue
            return result
        # The budget is gone while the owner was still coming up. That is the admission class (the
        # owner never accepted this run), and the text says which wait ate the budget.
        if boot is not None:
            # Only a real boot reaches here now: a silent owner is its own verdict (control_timeout)
            # and no longer borrows this name. Say how long this host's boot actually took.
            evidence = f"; {boot_evidence(home)}" if home is not None and boot_evidence(home) else ""
            raise spent(f"Gateway connection failed; no local fallback (phase=runtime-boot, "
                        f"elapsed={time.monotonic() - started:.1f}s, owner was still booting at "
                        f"the budget: {boot}, boot deadline {DEFAULT_ENSURE_TIMEOUT:g}s{evidence})")
        if drain is not None:
            # The drain outlived the budget: the owner is going away, not serving this run, and
            # nothing was ever submitted (so a retry chain may legitimately re-dispatch it).
            raise spent(f"Gateway connection failed; no local fallback (phase=runtime-drain, "
                        f"elapsed={time.monotonic() - started:.1f}s, the owner was still in its "
                        f"restart wait at the budget: {drain}; nothing was submitted)")
        # The fence outlived the budget. This is not owner starvation — an install update holds the
        # install — so the failure keeps the class a retry chain already handles (the owner is
        # draining, not starving) and spends its words on WHO held it and how long we waited.
        raise GatewayUnavailableError(
            f"Gateway draining: update_paused; admission budget {budget:g}s spent "
            f"({time.monotonic() - started:.1f}s) waiting for the update fence to clear, so this run "
            f"was never admitted: {fence}", "draining")

    phase = "session-ticket"
    if left(1.0) <= 0:
        raise spent("Gateway connection failed; no local fallback (admission budget spent before the ticket mint)")
    if endpoint is not None:
        owner_sha = getattr(endpoint, "code_sha", None)
        ticket = await mint(gateway_profile_home(get_hermes_home()), endpoint, subject=subject)
        url, protocols = gateway_ws_target(endpoint, ticket)
    elif remote:
        parsed = urlsplit(remote)
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
            raise GatewayClientError("Invalid explicit gateway WebSocket URL; no local fallback")
        url = remote
    else:
        # A per-task overlay home is served by the profile its symlinked lock names (#138949).
        home = gateway_profile_home(get_hermes_home())
        # A client's own start: the daemon it may launch ends itself once idle (no client, adapter,
        # cron/kanban work or admission), and the next client starts a fresh one.
        phase = "runtime"
        result = await admit_owner(home)
        if not attachable(result):
            # No attachable owner is NOT admission starvation: it is either a boot in progress (which
            # a retry legitimately waits out) or a broken install (whose report the caller must read).
            # Only an owner that ACCEPTED this client and then starved it is the admission class.
            detail = getattr(result, "detail", None)
            if result.reason_code == "runtime_exited" and detail:
                # A multi-line startup report (redacted traceback tail + where the full log is).
                raise GatewayUnavailableError(f"Gateway could not start: {detail}", result.state)
            detail = f" ({detail})" if detail else ""
            raise GatewayUnavailableError(f"Gateway {result.state}: {result.reason_code or 'not_ready'}{detail}",
                                          result.state)
        endpoint = result.endpoint
        owner_sha = getattr(endpoint, "code_sha", None)
        ticket = await mint(home, endpoint)
        url, protocols = gateway_ws_target(endpoint, ticket)
    phase = "ws-open"
    if left(1.0) <= 0:
        raise spent("Gateway connection failed; no local fallback (admission budget spent at the ticket)")
    try:
        # The gateway is a loopback (or explicitly named) peer, never something to route through the
        # user's HTTP(S) proxy; websockets>=14 reads HTTP_PROXY/HTTPS_PROXY by default and a proxy that
        # cannot reach 127.0.0.1 turns every launch into a 10 s open timeout.
        async with connect(url, subprotocols=protocols, open_timeout=left(GATEWAY_WS_OPEN_TIMEOUT_S),
                           max_size=8 * 1024 * 1024,
                           proxy=None,
                           # Explicit keepalive: the default 20 s pong deadline is inside this
                           # host's measured stall distribution (see GATEWAY_WS_PING_TIMEOUT_S).
                           ping_interval=GATEWAY_WS_PING_INTERVAL_S,
                           ping_timeout=GATEWAY_WS_PING_TIMEOUT_S) as ws:
            if protocols and ws.subprotocol != GATEWAY_WS_PROTOCOL:
                raise GatewayClientError("Gateway protocol mismatch; update/restart required")
            phase = "session"
            async with GatewayClient(ws, admission_deadline=deadline, admission_started=started,
                                     admission_budget=budget, code_sha=owner_sha) as client:
                # From here on, errors from the caller's body are NOT admission errors unless the
                # wait that produced them was admission-bounded (``GatewayClient.rpc`` raises
                # ``GatewayAdmissionTimeout`` for those itself). A run 200 s into its turn is
                # still inside this context; its body's failures must stay retryable.
                handed_over = True
                yield client
    except GatewayClientError:
        raise
    except (OSError, TimeoutError) as exc:
        detail = (f"Gateway connection failed; no local fallback (phase={phase}, "
                  f"elapsed={time.monotonic() - started:.1f}s, {type(exc).__name__}: {exc})")
        # A deadline that expired, or a phase that timed out against one, is starvation: the owner
        # took the socket (or left it open) and stopped answering, so a retry re-pays the whole
        # budget for the same result. A refusal — dial error, FIN, closed port — is NOT that: it is
        # an absent/broken owner, which a retry legitimately rides out. Only the former gets the
        # classifiable admission class, and only before the connection changed hands: after
        # ``yield`` the same exception may belong to a 30 s mid-turn verb on a healthy run.
        if not handed_over and (isinstance(exc, TimeoutError) or left(1.0) <= 0):
            raise spent(detail) from exc
        raise GatewayClientError(detail) from exc
