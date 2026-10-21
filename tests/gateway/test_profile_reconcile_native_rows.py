"""Regression for PRA-6631: the native-row reconcile pass must not read the store on the loop.

The pass runs from ``_profile_reconcile_watcher`` every 30 s. It used to resolve
``authority.logical_owner`` — two or more synchronous SQLite reads — for EVERY entry in the routing
index, inline on the event loop: the ``loop stalled >10s`` hours the blocked-loop dump named. Two
invariants here, and both fail on that shape:

* no store read is executed on the loop thread, and
* a control request the loop owes an answer to (a stand-in for ``identify`` on the control socket)
  is served while the pass's reads are still in flight, inside a stated bound.

Real ``SessionDB``, real ``SessionAuthority``, real routing index; the reads are only *held* so the
bound is the regression's own deadlock, not a quiet-runner assumption.
"""
import asyncio
import threading

import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input

# How long the loop gets to answer a control request while the pass reads. Only a read executed on
# the loop thread can miss it: everything the pass does off-loop leaves the loop free.
LOOP_SERVICE_BOUND_S = 5.0
ENTRIES = 200


def _signal(monkeypatch):
    from plugins.platforms.signal.adapter import SignalAdapter
    monkeypatch.setenv('SIGNAL_ALLOWED_USERS', '+155****0001')
    adapter = SignalAdapter(PlatformConfig(enabled=True, extra={'account': '+155****9999'}))
    return adapter, adapter.build_source(chat_id='+155****0001', chat_type='dm', user_id='+155****0001')


@pytest.mark.asyncio
async def test_native_row_reconcile_reads_off_the_loop_in_a_constant_number_of_statements(
        monkeypatch):
    adapter, first = _signal(monkeypatch)
    runner = GatewayRunner(GatewayConfig())
    adapter.gateway_runner = runner
    runner.adapters = {adapter.platform: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    try:
        # A routing index the size of a long-lived multiplexed gateway's. The first entry is moved
        # onto its compression continuation by the store's own route advance, so only resolving the
        # lineage can name the owner its queued turn belongs to.
        store = runner.session_store
        entry = store.get_or_create_session(first)
        more = [store.get_or_create_session(
            adapter.build_source(chat_id=f'+155****{i:04d}', chat_type='dm', user_id=f'+155****{i:04d}'))
            for i in range(1, ENTRIES)]
        root = entry.session_id
        authority.db.end_session(root, 'compression')
        tip = f'{root}-tip'
        authority.db.create_session(tip, source='signal', parent_session_id=root)
        advanced = store.advance_compression_session(entry.session_key, root, tip)
        assert advanced is not None and advanced.session_id == tip
        admit_session_input(authority.db, epoch=authority.epoch, principal_id='user', session_id=root,
                            request_id='queued-native', payload={'text': 'queued before the restart'})

        captured = {}

        async def capture(bindings):
            captured['bindings'] = list(bindings)
            return {}
        monkeypatch.setattr(authority, 'recover_native_sessions', capture)

        from gateway.run_runtime import recover_profile_native_sessions
        loop_thread = threading.get_ident()
        read_threads, in_flight, control_served = [], threading.Event(), threading.Event()
        real_read = SessionDB._read_retrying_ioerr

        def recording_read(self, fn):
            read_threads.append(threading.get_ident())
            if len(read_threads) == 1:
                in_flight.set()
                # Hold the FIRST read until the loop has answered the control: a read on the loop
                # thread can never get there, so it waits out the bound and the assertions say why.
                control_served.wait(LOOP_SERVICE_BOUND_S)
            return real_read(self, fn)
        monkeypatch.setattr(SessionDB, '_read_retrying_ioerr', recording_read)

        async def control_identify():
            await asyncio.to_thread(in_flight.wait, LOOP_SERVICE_BOUND_S)
            control_served.set()
        control = asyncio.ensure_future(control_identify())

        await asyncio.wait_for(recover_profile_native_sessions(runner, authority), LOOP_SERVICE_BOUND_S * 3)
        await asyncio.wait_for(control, LOOP_SERVICE_BOUND_S)

        assert control_served.is_set(), 'the loop owed a control answer and could not serve it'
        assert loop_thread not in read_threads, (
            f'a store read ran on the loop thread ({len(read_threads)} reads for {ENTRIES} entries)')
        # The queued-target read plus ONE batched lineage read for the whole index: O(1), never one
        # per entry — which is where the 30 s watch's O(sessions) synchronous queries came from.
        assert len(read_threads) == 2, read_threads
        # The continuation's route is the root's work, and the owner is the root.
        assert [owner for owner, _, _ in captured['bindings']] == [root]
        assert [origin.chat_id for _, origin, _ in captured['bindings']] == [first.chat_id]
        assert more  # the rest of the index is live, and cost the pass nothing
    finally:
        await asyncio.to_thread(authority.db.close)
