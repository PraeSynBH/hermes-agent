"""A background-process completion must not read the session store from the event loop (PRA-6517).

``_completion_event_scope`` resolves the event's source before it binds a profile scope, and that
resolution reads the session store's index under its lock. Every process watcher tick runs it on the
shared gateway loop, so under a loaded host the loop sat in ``session_persistence._ensure_loaded``
waiting for that lock — the frame the out-of-loop liveness watchdog named twice on 2026-10-19
(01:28:43Z, 02:43:19Z) while serial control probes were 3 s p50 / 30 s p95.
"""
import threading
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner


@pytest.mark.asyncio
@pytest.mark.allow_real_home_io  # importing gateway.run activates the checkout's install root; nothing here writes outside tmp_path
async def test_the_completion_scope_resolves_its_source_off_the_loop():
    seen = []

    def build(evt):
        seen.append(threading.get_ident())

    fake = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False))
    fake._build_process_event_source = build
    loop_thread = threading.get_ident()

    # The old sync method resolved the source as it was called, i.e. right here, on the loop.
    scope = GatewayRunner._completion_event_scope(fake, {"type": "completion"})
    if not hasattr(scope, "__aenter__"):
        scope = await scope
    async with scope:
        pass

    assert len(seen) == 1, f'the source was resolved {len(seen)} times, expected once'
    assert seen[0] != loop_thread, 'the session-store read for a completion ran on the event loop'
