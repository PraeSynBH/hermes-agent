"""Frame-size and send-cost accounting for the control WebSocket's egress (PRA-6844).

Every frame on ``/api/ws`` is written by websockets' frame writer ON the served event loop: the
legacy protocol's ``write_frame`` is ``write_frame_sync`` (permessage-deflate zlib compression, then
``transport.write`` -> ``selector_events.write`` -> ``sock.send``) followed by an awaitable
``drain``. The synchronous half is not preemptible, so the caller's send deadline
(``ws._WS_SEND_DEADLINE_S``) cannot shorten any of it: the cost a frame imposes on the loop is
whatever its size and compressibility make it. A 1200 s sample of the gateway loop put 184/2059
loop-thread shots inside ``permessage_deflate.encode`` and 137/2059 inside
``selector_events.write`` — the largest executing cluster on that loop was this socket's egress.

So the mix is what has to be observable: which frame sizes dominate, and what a send costs at each
size. ``record_send`` is called once per frame from the writer; ``snapshot`` is the read side (an
operator probe, or the rate-limited oversized-frame log line). Nothing here is on a request path.
"""

from __future__ import annotations

import logging
import threading
import time

_log = logging.getLogger(__name__)

# A frame at or above this is "oversized": large enough that its synchronous compression + socket
# copy is visible in a loop-latency budget on its own, and the size above which the write path has
# no way to yield (fragmentation would split one JSON-RPC line across frames, which every client
# parses whole — apps/shared/src/json-rpc-channel.ts::handleFrame JSON.parses each frame).
LARGE_FRAME_BYTES = 256 * 1024

# Upper bounds of the size histogram buckets, in bytes; the last bucket is open-ended.
SIZE_BUCKETS = (1024, 16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024)
# Upper bounds of the per-send cost histogram buckets, in seconds; the last bucket is open-ended.
COST_BUCKETS = (0.0005, 0.002, 0.008, 0.032, 0.128)

# One oversized-frame log line per window keeps the mix visible in gateway.log without a log flood.
_OVERSIZED_LOG_INTERVAL_S = 30.0

_lock = threading.Lock()
_sizes = [0] * (len(SIZE_BUCKETS) + 1)
_costs = [0] * (len(COST_BUCKETS) + 1)
_totals = {"frames": 0, "bytes": 0, "oversized_frames": 0, "oversized_bytes": 0,
           "max_frame_bytes": 0, "cost_seconds": 0.0, "max_cost_seconds": 0.0}
_last_oversized_log_at = 0.0


def _bucket(value: float, bounds) -> int:
    for index, upper in enumerate(bounds):
        if value <= upper:
            return index
    return len(bounds)


def record_send(nbytes: int, elapsed_s: float) -> bool:
    """Account one frame the writer put on the control socket; True when it was oversized."""
    global _last_oversized_log_at
    oversized = nbytes >= LARGE_FRAME_BYTES
    size_bucket, cost_bucket = _bucket(nbytes, SIZE_BUCKETS), _bucket(elapsed_s, COST_BUCKETS)
    with _lock:
        _sizes[size_bucket] += 1
        _costs[cost_bucket] += 1
        _totals["frames"] += 1
        _totals["bytes"] += nbytes
        _totals["cost_seconds"] += elapsed_s
        _totals["max_frame_bytes"] = max(_totals["max_frame_bytes"], nbytes)
        _totals["max_cost_seconds"] = max(_totals["max_cost_seconds"], elapsed_s)
        if oversized:
            _totals["oversized_frames"] += 1
            _totals["oversized_bytes"] += nbytes
        now = time.monotonic()
        log_it = oversized and now - _last_oversized_log_at >= _OVERSIZED_LOG_INTERVAL_S
        if log_it:
            _last_oversized_log_at = now
    if log_it:
        _log.warning(
            "ws egress oversized frame bytes=%d send_cost_ms=%.1f oversized=%d/%d frames "
            "oversized_bytes=%d (PRA-6844)",
            nbytes, elapsed_s * 1000.0, _totals["oversized_frames"], _totals["frames"],
            _totals["oversized_bytes"])
    return oversized


def snapshot() -> dict:
    """Current counters as plain data: totals, the size histogram and the per-send cost histogram."""
    with _lock:
        totals = dict(_totals)
        sizes, costs = list(_sizes), list(_costs)
    return {
        **totals,
        "size_buckets": [
            {"le_bytes": upper, "frames": sizes[index]}
            for index, upper in enumerate(SIZE_BUCKETS)
        ] + [{"le_bytes": None, "frames": sizes[-1]}],
        "cost_buckets": [
            {"le_seconds": upper, "frames": costs[index]}
            for index, upper in enumerate(COST_BUCKETS)
        ] + [{"le_seconds": None, "frames": costs[-1]}],
    }


def reset() -> None:
    """Clear every counter (tests)."""
    global _last_oversized_log_at
    with _lock:
        for index in range(len(_sizes)):
            _sizes[index] = 0
        for index in range(len(_costs)):
            _costs[index] = 0
        for key, value in _totals.items():
            _totals[key] = 0 if isinstance(value, int) else 0.0
        _last_oversized_log_at = 0.0
