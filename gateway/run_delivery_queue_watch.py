"""Wake the restart-safe cron delivery drain when a worker writes the queue (#117307).

A restart-safe worker runs its job outside the gateway, queues the final send in
``cron/deliveries.db`` and polls that row once a second; the gateway drained the
queue only on its housekeeping tick, so a turn that ended in 8 s reached the user
a minute later. Between ticks the housekeeping thread now stats the queue file
(and its WAL) and drains as soon as either changed. The tick's own drain stays
as the fallback for anything the stat cannot see.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

# How often the housekeeping thread looks at the queue file between its ticks.
DELIVERY_QUEUE_WATCH_SECONDS = 1.0
DRAIN_LABEL = "Cron durable delivery queue drain"


class DeliveryQueueWatch:
    """Stamp-compare every served profile's queue file; ``drain`` when a stamp moved.

    The watch never resets itself after draining: the drain's own status writes
    move the stamp once more and cost one empty pass, after which the stamp is
    stable, while a worker that enqueued during the drain still moves it again.
    """

    def __init__(self, homes: Callable[[], Iterable[Optional[Path]]], drain: Callable[[], None]) -> None:
        self._homes = homes
        self.drain = drain
        self._served: list = list(homes())
        self._seen = self._stamp()

    def refresh_homes(self) -> None:
        """Re-resolve the served profiles once per tick (a directory read under multiplex),
        not on every poll; a profile that appears mid-interval is drained by the tick itself."""
        self._served = list(self._homes())

    def _stamp(self) -> tuple:
        from cron.delivery_queue import queue_path

        stamps = []
        for home in self._served:
            path = queue_path(home)
            for candidate in (path, path.with_name(path.name + "-wal")):
                try:
                    info = os.stat(candidate)
                except OSError:
                    continue
                stamps.append((str(candidate), info.st_mtime_ns, info.st_size))
        return tuple(stamps)

    def changed(self) -> bool:
        current = self._stamp()
        if current == self._seen:
            return False
        self._seen = current
        return True


def wait_for_next_tick(stop_event, interval: float, watch: Optional[DeliveryQueueWatch], chore,
                        tickers: Iterable = ()) -> None:
    """Sleep one housekeeping interval, running sub-tick work as it falls due.

    Two kinds of sub-tick work ride this wait: the worker-queue drain (a stamp check, as soon as it
    changes) and cadence *tickers* — objects with ``seconds``, ``label`` and ``tick()`` whose period
    is shorter than the housekeeping interval (the wire-contract watch: 10 s inside a 60 s tick).
    Both run on the housekeeping thread, so each is wrapped by the caller's single failure boundary.
    """
    tickers = list(tickers)
    if watch is None and not tickers:
        stop_event.wait(timeout=interval)
        return
    if watch is not None:
        watch.refresh_homes()
    poll = min([DELIVERY_QUEUE_WATCH_SECONDS] + [t.seconds for t in tickers])
    due = {id(t): 0.0 for t in tickers}  # first tick immediately: the boot snapshot is already known
    deadline = time.monotonic() + interval
    while True:
        now = time.monotonic()
        remaining = max(0.0, deadline - now)
        if stop_event.wait(timeout=min(remaining, poll)):
            return
        now = time.monotonic()
        if watch is not None and watch.changed():
            chore(DRAIN_LABEL, watch.drain)
        for ticker in tickers:
            if now >= due[id(ticker)]:
                due[id(ticker)] = now + ticker.seconds
                chore(ticker.label, ticker.tick)
        if now >= deadline:
            return
