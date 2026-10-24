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
from hermes_cli.turn_exit import TURN_NOT_ADMITTED_EXIT_CODE

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
# An owner that is REPLACING itself accepts this run's socket before it can serve it, and nothing
# re-observed that window from the in-band verbs. 2026-10-24: the new process started at
# 05:44:33.955Z, its control socket listened at 05:45:22.916Z and it published ``running`` at
# 05:46:45.358Z; six runs dispatched at 05:44:36Z minted a ticket inside that window, sat in
# ``session.create`` and died at their 120 s budget at 05:46:48-51Z — seconds before the owner could
# have served them (SOU-3958/3960). ``admit_owner`` re-observes a BOOTING owner (``runtime-boot``)
# and a restart's after-turn wait (``runtime-drain``) only; the in-band verbs had one RPC and one
# deadline. Sample the owner's own readiness at the boot cadence while an admission verb is pending,
# and hand the seconds it spends coming up back to the deadline. Same cadence and same cost as the
# boot poll: one small read of the home's published state.
RESTART_POLL_INTERVAL_S = BOOT_POLL_INTERVAL_S
# The ceiling on that gift. Measured restart (2026-10-24): 131 s of ``starting`` for the replacing
# owner and 259 s door-to-door from the old owner's ``stop()``; 300 s admits a run dispatched
# anywhere inside the whole observed window. It is a CEILING, not a switch: an owner that never
# serves still fails, `budget + this` later, and one that is merely WEDGED (alive, publishing
# ``running``, answering nothing) earns not a single second of it.
RESTART_WAIT_GRACE_S = 300.0
# The owner states in which a LIVE process on that home has not started serving this run yet and an
# owner is coming up: ``starting`` (``run_startup``'s boot stamp, until every platform connects) and
# ``restarting`` (the OUTGOING owner's own claim at ``stop()``, kept on disk through the exit->boot
# gap — SOU-3966 adds the name; a tree that does not publish it simply never sees it). Both are
# "wait", and neither is ``draining``: an owner that is going away with no replacement is
# ``admit_owner``'s wait, not a restart this client may spend its grace on.
OWNER_COMING_UP_STATES = frozenset({"starting", "restarting"})
# How long the OUTGOING owner's claim is allowed to stand after that owner is gone
# (``restart_requested`` + ``gateway_state: restarting``, written by ``stop()`` before the drain and
# re-stamped by the exit phase — SOU-3966). The claim is the ONLY readiness there is in the
# exit -> boot stretch: the home holds no live pid, no lock and no ``starting`` until the replacement
# boots. Measured 2026-10-24: old owner stopped 05:42:26.427Z, replacement launched 05:44:33.957Z,
# so the gap is one supervisor respawn wide (~63 s) and a run dispatched into it spent its whole 120 s
# budget waiting for a listener that did not exist yet (six runs, ``phase=session, elapsed=120.0s``).
#
# A dead pid alone proves nothing: a SIGKILLed gateway leaves the same file, so the claim is honoured
# only while it is FRESH, and the bound is the admission budget — the window this class is already
# measured in. A claim older than one budget cannot be covered by any run's remaining budget, so
# crediting it past that point would only hold a dispatch for a restart that is not coming. Worst case
# is therefore exact rather than open-ended: a gateway that dies mid-claim holds a run one budget
# longer than an absent owner would, and never past budget + ``RESTART_WAIT_GRACE_S``.
RESTART_CLAIM_FRESH_S = GATEWAY_ADMISSION_BUDGET_S
# A profile the multiplexer's ceiling has deferred (`profile_serving`) is re-observed this often.
# Cheap — one control-socket ``identify`` — and well inside the reconcile interval the rank moves on.
PROFILE_SERVE_POLL_INTERVAL_S = 5.0
# How long admission waits for that rank to move. A ceiling deferral is PROVISIONAL: the rank is
# recomputed every reconcile tick (30 s) from live demand, and this client files itself as the
# strongest demand signal there is the moment it sees the deferral (PRX-5738). Two ticks is when the
# deferral is settled FOR THIS RUN: past it the run fails with the ceiling's own reason (cost,
# remedy) instead of spending its whole 120 s admission budget on a slot nobody freed. Reading the
# deferral as terminal at T=0 is what killed three devops runs in the 2-3 min boot reconcile that was
# already re-ranking them in (PRX-5963).
PROFILE_SERVE_WAIT_S = 75.0
# A settled ceiling deferral is a CAPACITY verdict, not a failed run (PRX-5972). The run is told when
# to ask again — the next heartbeat — and nothing was submitted, so re-dispatch is safe. Without this
# the client exited 1 and the Paperclip adapter booked `adapter_failed` onto the agent's card, so a
# card read as a broken agent for a ceiling decision (PRX-5970 was filed over exactly that).
PROFILE_DEFERRED_REASON = "profile_deferred_capacity"
PROFILE_DEFERRAL_RETRY_S = 300.0
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
# The copy a turn's transport loss carries. A lost turn could never say WHICH close ended it, and at a
# managed restart that is the whole question: the restart's deferral protects in-flight turns, so a
# close during one means either "the owner cut this socket" or "the transport died on its own"
# (PRX-6150: eight mid-turn losses inside one deferral all read the same sentence, and neither the run
# log nor the gateway log could tell the two apart). Name the close code and reason when the transport
# reported one; a non-websocket failure (OSError, a parse error) names nothing extra.
LOST_TURN = ("Gateway disconnected; turn outcome is unknown. Resume the printed session ID to check "
             "whether it completed or was interrupted; do not resend the work.")


def close_detail(exc) -> str:
    """``" (close 1013: restart wait)"`` for a websocket close, ``""`` for anything else."""
    from websockets.exceptions import ConnectionClosed

    if not isinstance(exc, ConnectionClosed):
        return ""
    # The frame, not the deprecated ``ConnectionClosed.code``/``.reason`` properties: the one the
    # peer sent when there is one, else our own half of the handshake.
    frame = getattr(exc, "rcvd", None) or getattr(exc, "sent", None)
    code = getattr(frame, "code", None)
    if code is None:
        return ""
    reason = getattr(frame, "reason", None) or ""
    return f" (close {code}{f': {reason}' if reason else ''})"



def gateway_ws_target(endpoint, ticket):
    """``(url, subprotocols)`` for dialing the owner's canonical WebSocket with a session ticket.

    The owner accepts exactly one ticket subprotocol beside ``GATEWAY_WS_PROTOCOL`` and echoes the
    latter; callers check ``ws.subprotocol == GATEWAY_WS_PROTOCOL`` after the upgrade."""
    url = endpoint.api_origin.replace("https:", "wss:").replace("http:", "ws:") + "/api/ws"
    return url, [GATEWAY_WS_PROTOCOL, GATEWAY_WS_TICKET_PREFIX + ticket]


def contract_submit_params() -> dict:
    """The params a client attaches to ``prompt.submit`` to say which wire contract it was built
    against. Empty when this install has no contract document to hash, which keeps the request
    byte-identical to what an older client sends (SOU-3318)."""
    from gateway.contract_revision import declaration

    return declaration()


def declaration_refused(error) -> bool:
    """True when an owner refused a submit ONLY because of the optional contract declaration.

    ``contract_revision`` exists to make a refusal diagnosable; it must never be the reason a
    request fails. A client spawned from a tree newer than the running gateway declares a param
    that process's param model does not define yet, and PRA-6722 made that refusal exact:
    ``invalid_params`` naming that one key in ``data.fields``.

    Anything else propagates: a refusal naming another key too (dropping the declaration would
    leave a genuinely malformed request looking like an old owner), a bare refusal with no key
    named (nothing proves the declaration was the cause), and ``contract_version_mismatch`` (the
    skew the fence exists to report).
    """
    if not isinstance(error, GatewayRPCError) or str(error) != "invalid_params":
        return False
    from gateway.contract_revision import REVISION_PARAM

    data = error.data if isinstance(error.data, dict) else {}
    named = [str(name).strip() for name in (data.get("fields") or ()) if str(name).strip()]
    if not named:
        field = str(data.get("field") or "").strip()
        named = [field] if field else []
    return bool(named) and set(named) == {REVISION_PARAM}


def declaration_retry_params(error, params: dict) -> dict | None:
    """The params to resend when *error* was the declaration's refusal alone, else ``None``.

    The caller resends exactly once, under the SAME admission identity: the param gate refuses
    before anything is admitted, and a resend carrying an identical ``submission_id``/``input_id``
    cannot double-admit even if that ever stops being true.
    """
    from gateway.contract_revision import REVISION_PARAM

    if REVISION_PARAM not in params or not declaration_refused(error):
        return None
    return {key: value for key, value in params.items() if key != REVISION_PARAM}


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


class GatewayContractSkewError(GatewayRPCError):
    """The owner serves a different wire contract revision than this client was built against.

    SOU-3316/SOU-3318: distinguishable from ``invalid_params`` (a malformed request this client can
    fix) and from a transport failure (retry elsewhere). Neither side is wrong alone — the serving
    PROCESS is older than the tree — so the only correct verdict is to fail this run fast, name
    both revisions, and let the deploy replace the gateway. Never retried: a retry against the same
    process answers identically, which is what turned a merge into a 3h24m fleet outage.
    """

    reason = "contract_version_mismatch"

    def __init__(self, code=None, data=None):
        data = data if isinstance(data, dict) else {}
        self.gateway_revision = str(data.get("gateway_revision") or "").strip() or None
        self.client_revision = str(data.get("client_revision") or "").strip() or None
        super().__init__(self.reason, code, data)

    @property
    def retryable(self) -> bool:
        return False


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


class GatewayTurnNotAdmitted(GatewayClientError):
    """The socket died before the owner answered ``prompt.submit``: this run holds no admission id.

    A distinct CLASS because the difference is a contract, not wording: with no admission id there
    is no receipt to read, so nothing is in doubt and there is no turn to resume — the mid-turn copy
    ("turn outcome is unknown … resume the printed session ID") sent an operator looking for work
    that never started, and the run class counter booked a no-op with the runs that lost a whole turn
    (PRX-5964). Retry-safe: ``exit_code`` is EX_TEMPFAIL, and ``reason`` is the stable code a retry
    chain keys on. (A submit whose answer was lost after the owner had already admitted it would be
    admitted once — the retried run then sees that row as a pending admission on the session, so the
    retry is not a silent duplicate.)
    """

    reason = "gateway_turn_not_admitted"
    exit_code = TURN_NOT_ADMITTED_EXIT_CODE

    def __init__(self):
        super().__init__(
            f"{self.reason}: the connection dropped before the turn was admitted, so this run has no "
            "admission id — there is no turn outcome to recover and nothing to resume. Retry the run "
            "if its work is still wanted.")


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


def restart_watch_homes(endpoint, home) -> list[Path]:
    """Homes whose owner publishes readiness for THIS run, most specific first.

    The multiplexer is ONE process: it publishes ``gateway_state.json`` in the home it was LAUNCHED
    on (the default profile / install root), not in each profile it serves, so the client's own
    home alone cannot see a restart on a named profile. ``window_fence_homes`` is the same set the
    restart fence reads, for the same reason (this profile, the launch profile, the install root);
    a standalone owner's ticket names its control home, which comes first.
    """
    from hermes_cli.gateway_restart_window import window_fence_homes

    control = getattr(endpoint, "control_home", None) if endpoint is not None else None
    return list(dict.fromkeys(Path(h) for h in (control, home, *window_fence_homes()) if h))


def record_age_s(record) -> float | None:
    """Seconds since the record's own ``updated_at``, or ``None`` when it cannot be aged.

    The status writer re-stamps ``updated_at`` on every write, so this is the age of the last thing
    the process that wrote it had to say — the only clock a claim left behind by a DEAD owner can
    be bounded with.
    """
    from datetime import UTC, datetime

    stamp = record.get("updated_at") if isinstance(record, dict) else None
    if not isinstance(stamp, str) or not stamp.strip():
        return None
    try:
        # ``status._utc_now_iso`` is ``datetime.now(UTC).isoformat()`` ("+00:00"); an older writer's
        # "Z" parses too, and a zone-less stamp is read as UTC rather than discarded.
        moment = datetime.fromisoformat(stamp.strip())
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)  # an older writer without a zone: read it as UTC
    return max(0.0, (datetime.now(UTC) - moment).total_seconds())


def fresh_restart_claim(record) -> bool:
    """Is this record the outgoing owner's claim, still inside the window it promises?

    Three things together, because any one alone is a lie: ``gateway_state: restarting`` is what
    ``stop()`` publishes, ``restart_requested: true`` says a replacement was asked for (a
    ``restarting`` written by anything else is not a promise), and a FRESH ``updated_at`` keeps the
    claim from outliving the restart it describes (``RESTART_CLAIM_FRESH_S``).
    """
    from gateway.restart import RESTARTING_GATEWAY_STATE

    if not isinstance(record, dict) or record.get("gateway_state") != RESTARTING_GATEWAY_STATE:
        return False
    if record.get("restart_requested") is not True:
        return False
    age = record_age_s(record)
    return age is not None and age <= RESTART_CLAIM_FRESH_S


def owner_serving_state(homes) -> str | None:
    """The owner's own word while an owner on one of *homes* is still coming up, else ``None``.

    The owner publishes its own readiness into ``<home>/gateway_state.json``: ``starting`` from the
    moment it claims the home until every enabled platform is connected (``run_startup`` stamps
    ``running``/``degraded`` only after that), and — once the outgoing life claims it at ``stop()``
    — ``restarting`` across the exit -> boot gap (``OWNER_COMING_UP_STATES``). That is the one
    signal that separates a restart window from a stalled loop without asking the owner a question
    it cannot answer yet: on this host the control socket listens 83 s before the platforms are
    ready, so ``identify`` answers "alive" for the whole window in which ``session.create`` cannot
    complete.

    The claim is read while its writer is DEAD, which is the whole point of it: the stretch between
    the outgoing owner's exit and the replacement's first write has no live process, no lock and no
    ``starting``, so a liveness check on the record made the one stretch the claim was written for
    unreadable. It is gated by ``fresh_restart_claim`` instead — the claim's own age, bounded by one
    admission budget — so a record left behind by a gateway that is never coming back credits
    nothing (that host already costs a run exactly one budget).

    A home with no live record and no fresh claim falls back to the lock reservation
    (``missing_owner_state``): a claimed-but-unpublished owner is the launch -> socket half of the
    same window. Nothing is ever inferred from an absent owner — a stale ``starting``/``running``
    record from a previous life must not credit a run, or a host whose gateway is gone would hold
    every dispatch for the whole grace.
    """
    from gateway.status import read_runtime_status, runtime_status_pid_is_live
    from hermes_cli.gateway_runtime_discovery import missing_owner_state

    for candidate in homes:
        home = Path(candidate)
        try:
            record = read_runtime_status(home / "gateway_state.json")
            if isinstance(record, dict) and runtime_status_pid_is_live(record):
                state = record.get("gateway_state")
                if state in OWNER_COMING_UP_STATES:
                    return str(state)
                continue  # a live owner that says it is serving: no credit from this home
            if fresh_restart_claim(record):
                return "restarting"  # the writer is gone; its claim is the readiness
            if missing_owner_state(home) == "starting":
                return "starting"
        except (OSError, RuntimeError, ValueError):
            continue  # a probe must never be the reason a run dies at admission
    return None


class _RestartWatch:
    """The owner's readiness, sampled while ONE admission verb is pending.

    Started and stopped by ``GatewayClient._await_admission`` rather than run as a background
    ticker: the client's context outlives admission by a whole agent turn, and a per-second probe
    for that long is a cost this fix does not need. ``starting_s`` is how much of the wait the owner
    spent saying it was not serving yet (its word is kept in ``seen`` for the death text); the
    waiter hands exactly that back to the deadline.
    """

    def __init__(self, homes, *, interval=None):
        self._homes = list(homes or ())
        self._interval = RESTART_POLL_INTERVAL_S if interval is None else interval
        self.starting_s = 0.0
        self.seen = None  # the owner's own word at the last sample, for the death text
        self._task = None

    async def _poll(self):
        sampled_at = time.monotonic()
        while True:
            try:
                state = await asyncio.to_thread(owner_serving_state, self._homes)
            except (OSError, RuntimeError, ValueError):
                state = None
            now = time.monotonic()
            if state:  # `starting` or `restarting`: the owner is up, and not serving yet
                self.starting_s += now - sampled_at
                self.seen = state
            sampled_at = now
            await asyncio.sleep(self._interval)

    def start(self):
        if self._homes:
            self._task = asyncio.create_task(self._poll())
        return self

    async def stop(self):
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None


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
    bounded_code = code if isinstance(code, int) and not isinstance(code, bool) else None
    if reason == GatewayContractSkewError.reason:
        # A distinct verdict the caller must not retry (SOU-3318), never a generic RPC refusal.
        return GatewayContractSkewError(bounded_code, data)
    return GatewayRPCError(reason or "request_failed", bounded_code, data)


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


class GatewayProfileDeferredError(GatewayUnavailableError):
    """A ceiling deferral that outlived its window: a CAPACITY booking, not an unusable profile.

    ``profile_not_served`` was the wrong word for it in two ways (PRX-5972): it is the verdict for a
    home that can NEVER be served (no identity marker, no owner of its own), and a retry chain /
    adapter reading it books a failed run. The ceiling deferral means the opposite — the profile is
    perfectly servable and simply did not get a slot among the profiles already served — so it
    carries its own reason code and a retry schedule, and the ceiling's own numbers (served of
    installed) instead of "not served". Re-dispatch is safe because nothing was submitted.

    ``exit_code`` is EX_TEMPFAIL, the same contract ``GatewayTurnNotAdmitted`` carries: the reason
    code alone only reached a reader that parsed the message, while the LOSER here — the process
    exit status — still said 1, i.e. "this run failed", which is what a dispatcher and the adapter
    book (PRX-5964 gave the never-admitted class that exit contract; a deferral needs it for the
    same reason, and the issue's own repro is "a failed run with exit 1"). ``GatewayRestartingError``
    deliberately keeps 1: a declared window is an operator action, not this dispatcher's to retry.
    """

    reason_code = PROFILE_DEFERRED_REASON
    exit_code = TURN_NOT_ADMITTED_EXIT_CODE

    def __init__(self, message: str, retry_after_s: float, *, profile_name: str | None = None):
        super().__init__(message, "deferred")
        self.retry_after_s = retry_after_s
        self.profile_name = profile_name



def _deferral_already_outlived_a_sighting(detail: str | None, profile_name: str | None) -> bool:
    """Is this park the SAME deferral a waiting client already filed itself into?

    ``gateway_multiplex_capacity.cost_clause`` appends the live wanted-sighting to a deferral's reason,
    so a reason that already carries it was published by a reconcile tick that read a waiting client's
    sighting — the strongest demand signal the rank has — and still left the profile outside the
    ceiling. A second dispatch cannot do better by waiting again; it just pays the window over
    (PRX-5972). Detection is the canonical signal text, never a substring of our own words.
    """
    from hermes_cli.gateway_multiplex_demand import wanted_signal
    return wanted_signal(profile_name) in (detail or "")


def _deferral_numbers(detail: str | None) -> tuple[int | None, int | None, int | None]:
    """``(ceiling, served, installed)`` for a deferral verdict, each None when it cannot be read.

    The park reason already spells the ceiling (``capacity: 17 profiles per gateway …``); the served
    set and the roster are two cheap reads. Every read is best-effort: a message builder must never be
    the reason a run dies, so a failure degrades to the words the multiplexer already gave us.
    """
    ceiling = None
    marker = "capacity: "
    text = detail or ""
    if marker in text:
        head = text.split(marker, 1)[1]
        digits = head.split(" ", 1)[0]
        if digits.isdigit():
            ceiling = int(digits)
    served = installed = None
    with suppress(Exception):  # a number in a message must never be the reason a run dies
        from hermes_cli.gateway_multiplex_served import recorded_served_profiles
        recorded = recorded_served_profiles()
        served = len(recorded) if recorded is not None else None
    with suppress(Exception):
        from hermes_cli.profiles_serve import profiles_to_serve
        # INSTALLED, not "would serve": the serving enumeration applies the ceiling, so counting it
        # here would print "17 served of 17 installed" and — worse — run a serve ranking (and its
        # deferral WARNING) from inside a message builder.
        installed = len(profiles_to_serve(multiplex=True, include_standalone=True,
                                         include_parked=True))
    return ceiling, served, installed


def _deferral_slot_phrase(ceiling, served, installed) -> str:
    """``no slot among 17 served of 43 installed (ceiling 17)`` — the ceiling's own arithmetic."""
    if served is None and installed is None:
        return "no free slot at gateway.multiplex_max_profiles"
    slot = f"no slot among {served if served is not None else '?'} served"
    if installed is not None:
        slot += f" of {installed} installed"
    if ceiling is not None:
        slot += f" (ceiling {ceiling})"
    return slot


def _deferral_message(profile_name: str | None, detail: str | None, *, waited_s: float | None,
                      reconciled: bool) -> str:
    """The deferral verdict: a machine token, a retry schedule, the ceiling's numbers, the park's cost."""
    reason = (detail or "the multiplexer's ceiling defers this profile").strip()
    if len(reason) > 400:
        reason = reason[:397] + "..."
    why = ("a waiting client was already deferred by the reconcile after its own sighting, so no "
           "window of this dispatch could have opened a slot" if reconciled else
           f"waited {waited_s:g}s through ceiling re-ranks after filing this run as demand")
    server = _deferral_slot_phrase(*_deferral_numbers(detail))
    return (f"Gateway deferred: {PROFILE_DEFERRED_REASON} "
            f"(retryAfterS={round(PROFILE_DEFERRAL_RETRY_S)}, retryable=true, profile={profile_name}): "
            f"{server}; nothing was submitted, so re-dispatch can be admitted when the rank moves — "
            f"{why}. Park reason: {reason}")


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
                 admission_budget=None, code_sha=None, restart_homes=None):
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
        # Homes whose owner publishes readiness while an admission verb waits
        # (``restart_watch_homes``); empty on a peer with no home (a test double, an embedded
        # server), where admission waits are simply bounded as before.
        self.restart_homes = list(restart_homes or ())
        # Seconds of ``starting`` observed and seconds actually handed back to the deadline: the
        # distinction is what the death text reports (an owner that never served at all vs one
        # whose whole grace went into coming up).
        self.restart_seen_s = 0.0
        self.restart_credit_s = 0.0

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
        close = None
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
        except (ConnectionClosed, OSError, ValueError, TimeoutError) as exc:
            close = exc
        finally:
            # The close the transport reported rides the copy: a lost turn's own log has to name
            # whether the owner cut this socket (1013 at a restart's deferral) or it died on its own.
            error = GatewayClientError(LOST_TURN + close_detail(close))
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
            if isinstance(budget, _AdmissionBudget):
                return await self._await_admission(future, budget)
            return await asyncio.wait_for(future, None if budget == float("inf") else budget)
        except TimeoutError as exc:
            # Only a wait this run bounded with its admission budget is starvation. A mid-run RPC
            # that times out against its own (default) cap — ``prompt.receipt`` 200 s into a healthy
            # session, say — must stay a retryable connection error; filing it under the give-up
            # code would tell the retry chain to abandon a run that only needs to be retried.
            if not isinstance(_timeout, _AdmissionBudget):
                raise
            raise self._admission_death() from exc
        finally:
            self.pending.pop(rid, None)

    async def _await_admission(self, future, budget):
        """Await one admission-bounded answer, giving back the seconds the owner was still coming up.

        A single ``wait_for`` over the whole remaining budget is what made the 2026-10-24 restart
        six dead runs: the owner was booting for 131 s of that wait and the only thing the client
        could do at the deadline was die. Instead the wait is RE-ARMED — the same request, never
        resent — with the deadline wound forward by the time the owner itself reported ``starting``,
        bounded by ``RESTART_WAIT_GRACE_S`` in total. Every case that is not a restarting owner is
        unchanged: a wedged owner (alive, publishing ``running``, answering nothing) earns no credit
        and fails at the budget, and so does an owner that is simply gone — unless it left a fresh
        restart claim behind, which is the exit -> boot half of the same window (``fresh_restart_claim``).
        """
        remaining = None if budget == float("inf") else float(budget)
        while True:
            watch = _RestartWatch(self.restart_homes).start()
            try:
                done, _pending = await asyncio.wait({future}, timeout=remaining)
            finally:
                await watch.stop()
            if done:
                return future.result()
            credit = self._take_restart_credit(watch.starting_s)
            if credit <= 0.0:
                raise self._admission_death(watch.seen)
            # The owner said it was not serving yet for `credit` seconds; this request gets exactly
            # those seconds back. ``admission_deadline`` also governs the verbs that follow
            # (``runtime.describe`` -> ``session.create``/``session.resume``), so they inherit it.
            if self.admission_deadline is not None:
                self.admission_deadline += credit
            remaining = credit

    def _take_restart_credit(self, witnessed):
        """Seconds of the last wait to hand back to the deadline: never more than the grace has left.

        ``witnessed`` is what the owner reported, ``returned`` what the budget can still afford. An
        alternating owner (``starting`` one sample, serving the next) is charged only for its
        starting seconds, and the grace still bounds the total — so the loop terminates on any host.
        """
        if witnessed <= 0.0:
            return 0.0
        self.restart_seen_s += witnessed
        credit = min(witnessed, max(0.0, RESTART_WAIT_GRACE_S - self.restart_credit_s))
        self.restart_credit_s += credit
        return credit

    def _admission_death(self, last_state=None):
        """The ``GatewayAdmissionTimeout`` for this run, naming a restart when that is what held it.

        The run log line is the only place a death at admission is read from, and the detector
        classifies this signature by what it says (``restart_window`` keys on the restart, not on
        the phase), so the restart has to be in the words the run dies with.
        """
        started = self.admission_started if self.admission_started is not None else time.monotonic()
        elapsed = time.monotonic() - started
        held_by = ""
        if self.restart_seen_s > 0.0:
            held_by = (f", the owner was still restarting (gateway_state={last_state or 'starting'}) "
                       f"after {self.restart_credit_s:g}s of restart grace")
        return GatewayAdmissionTimeout(
            f"Gateway connection failed; no local fallback (phase=session, elapsed={elapsed:.1f}s, "
            f"TimeoutError: the in-band admission verb expired{held_by})",
            phase="session", elapsed=elapsed,
            budget=self.admission_budget if self.admission_budget is not None else GATEWAY_ADMISSION_BUDGET_S,
            code_sha=self.code_sha)


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
    from hermes_constants import get_hermes_home, profile_name_for_home
    from hermes_cli.gateway_runtime import (DEFAULT_ENSURE_TIMEOUT, PROFILE_SERVING_REASON, attachable,
                                            boot_evidence, ensure_gateway_runtime, gateway_profile_home)
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

        Four states are WAITS inside this run's one admission budget, not verdicts:

        * ``profile_serving`` — the multiplexer's ceiling has this profile deferred, which is
          PROVISIONAL (the rank is rebuilt every reconcile tick) and revises as soon as a client
          files itself as waiting. Three devops runs read it as permanent and died in the 2-3 min
          boot reconcile that was already re-ranking them in (PRX-5963). Bounded by
          ``PROFILE_SERVE_WAIT_S``, so a profile that holds no slot at all still fails with the
          ceiling's own reason instead of eating the whole budget.

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
        serving = None
        serving_name = None
        serving_until = 0.0
        while True:
            # The per-attempt cap is the boot deadline (now above the measured boot cost); the
            # whole loop is bounded by the one admission budget, so a lower cap can only cost
            # re-observation, never a run.
            cap = left(DEFAULT_ENSURE_TIMEOUT)
            if cap <= 0:
                if fence is None and boot is None and drain is None and serving is None:
                    raise spent("Gateway connection failed; no local fallback "
                                f"(phase={phase}, elapsed={time.monotonic() - started:.1f}s, "
                                "admission budget spent before an owner could be ensured)")
                break
            result = await asyncio.to_thread(ensure_gateway_runtime, home, idle_exit=True, timeout=cap,
                                             new_work=new_work)
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
                    f"retryAfterS={round(left_s)}, retryable=true): "
                    f"{result.detail or 'restart window open'}", float(left_s))
            if result.reason_code == PROFILE_SERVING_REASON:
                # The multiplexer's ceiling has this profile deferred. That is PROVISIONAL — the rank
                # is rebuilt every reconcile tick — so wait it out inside this run's budget and file
                # the sighting that makes the next tick rank the profile in (PRX-5738), instead of
                # reporting a run-killer for a host that serves the profile seconds later (PRX-5963:
                # devops, 3 runs, inside the 2-3 min boot reconcile).
                if serving is None:
                    serving = getattr(result, "detail", None) or "the multiplexer's ceiling defers it"
                    # The park is keyed by profile id, and a home's basename is only that for a named
                    # profile (the root home is `.hermes`, i.e. `default`, an install detail).
                    serving_name = profile_name_for_home(home) or home.name
                    if _deferral_already_outlived_a_sighting(serving, serving_name):
                        # The publish already carries a waiting client's own sighting and STILL
                        # defers this profile: a reconcile tick ran with the strongest demand signal
                        # there is and the rank did not move, so another window cannot open one.
                        # Spend seconds, not the whole window, on a slot nobody is going to free
                        # (PRX-5972: 26 profiles parked, 16 agents behind them).
                        raise GatewayProfileDeferredError(
                            _deferral_message(serving_name, serving, waited_s=None, reconciled=True),
                            PROFILE_DEFERRAL_RETRY_S, profile_name=serving_name)
                    serving_until = time.monotonic() + PROFILE_SERVE_WAIT_S
                    boot = drain = None  # the owner is up; this wait is about a slot, not a boot
                    from hermes_cli.gateway_multiplex_served import note_profile_wanted
                    await asyncio.to_thread(note_profile_wanted, serving_name)
                if left(1.0) <= 0 or time.monotonic() >= serving_until:
                    break
                phase = "profile-serve"
                delay = result.retry_after_s or PROFILE_SERVE_POLL_INTERVAL_S
                await asyncio.sleep(max(0.0, min(delay, left(delay), serving_until - time.monotonic())))
                continue
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
        if serving is not None:
            # The ceiling never made room inside the wait window: for THIS run the deferral is
            # settled. Its own reason (cost, remedy) is the answer — not a boot, not starvation — and
            # it is a CAPACITY booking, not an unusable profile: nothing was submitted, so the
            # dispatcher may re-admit on the next heartbeat instead of reading a broken agent
            # (PRX-5972: `profile_not_served` here booked `adapter_failed` onto a card for a ceiling
            # decision).
            raise GatewayProfileDeferredError(
                _deferral_message(serving_name, serving, waited_s=PROFILE_SERVE_WAIT_S, reconciled=False),
                PROFILE_DEFERRAL_RETRY_S, profile_name=serving_name)
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
            # The in-band verbs are the phase that had no re-observation at all: a restarted owner
            # accepts the socket before it can serve it, and one RPC expiring at the deadline was
            # the verdict (SOU-3960). Give the client the homes whose owner publishes readiness so
            # a wait against an owner that is still `starting` is wound, not lost.
            async with GatewayClient(ws, admission_deadline=deadline, admission_started=started,
                                     admission_budget=budget, code_sha=owner_sha,
                                     restart_homes=restart_watch_homes(endpoint, home)) as client:
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
