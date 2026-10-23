"""Local creation is durable before publication and restart never replays a claim."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_local_create_receipt_survives_cold_authority_atomically(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway import run
    from hermes_state_runtime import RuntimeStoreError

    config = {'platform_toolsets': {'cli': []}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda: config)
    def runner():
        store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        return SimpleNamespace(session_store=store, _session_db=store._db,
                               adapters={}, _draining=False)
    first = runner()
    authority = await initialize_session_authority(first, profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    params = {'request_id': 'retry', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []}
    ref = create_local_session(authority, actor, params)
    policy = first.adapters[Platform.LOCAL].policies[authority.sessions[ref.session_id].source.chat_id]
    second = runner()
    cold = await initialize_session_authority(second, profile_id='fixture', instance_id='second')
    assert ref.session_id in cold.sessions, 'durable creation was lost with process-local registry'
    config['platform_toolsets']['cli'] = ['terminal']
    assert create_local_session(cold, actor, params) == ref
    assert second.adapters[Platform.LOCAL].policies[cold.sessions[ref.session_id].source.chat_id] == policy
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        create_local_session(cold, actor, {**params, 'source': 'tui'})
    foreign = Principal('other', 'fixture', actor.capabilities, 'other-socket')
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await cold.resolve(foreign, ref)

    db = cold.db
    write = db._execute_write
    def abort(callback, **kwargs):
        def fail(conn):
            callback(conn)
            raise RuntimeError('injected transaction abort')
        return write(fail, **kwargs)
    before = db.session_count()
    monkeypatch.setattr(db, '_execute_write', abort)
    with pytest.raises(RuntimeError, match='injected transaction abort'):
        create_local_session(cold, actor, {**params, 'request_id': 'aborted'})
    monkeypatch.setattr(db, '_execute_write', write)
    assert db.session_count() == before
    assert set(cold.sessions) == {ref.session_id}
    with db._read_ctx() as conn:
        policies = conn.execute("SELECT value FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%'").fetchall()
    assert len(policies) == 1
    assert json.loads(policies[0][0])['principal_id'] == actor.subject
    from hermes_state_runtime import admit_session_input, list_session_admissions
    admit_session_input(db, epoch=cold.epoch, principal_id=actor.subject,
                        session_id=ref.session_id, request_id='preclaim', payload={'text': 'never run'})
    db._write_sql('UPDATE state_meta SET value=? WHERE key=?',
                  ('{}', 'gateway.local_policy.v1:' + ref.session_id))
    await cold._drain(ref)
    assert list_session_admissions(db, session_id=ref.session_id, pending_only=False)[0]['status'] == 'queued'


@pytest.mark.platforms("linux")
def test_real_daemon_recovers_only_authorized_never_started_local_work(tmp_path):
    from tests.gateway.fixtures.local_recovery_probe import probe
    print(json.dumps(probe(tmp_path)))


@pytest.mark.allow_real_home_io  # importing gateway.config activates the checkout's install root; this test writes only tmp_path
def test_messaging_session_history_follows_the_compression_tip(tmp_path, monkeypatch):
    """The authority keeps a messaging session's admission identity at the compression root,
    but attach/resume history must come from the current physical transcript."""
    from gateway.config import Platform
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway.session_local_recovery import local_history
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('root', source='telegram', session_key='agent:telegram:dm:1')
        db.append_message('root', 'user', 'BEFORE_COMPRESSION')
        assert db.try_acquire_compression_lock('root', 'holder')
        db.publish_compression_child(parent_session_id='root', child_session_id='child', source='telegram',
                                     messages=[{'role': 'user', 'content': 'SUMMARY'}], compression_lock_holder='holder')
        db.append_message('child', 'user', 'AFTER_COMPRESSION')
        authority = SessionAuthority(SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False)),
                                     profile_id='owned', instance_id='current', db=db,
                                     epoch=begin_runtime_epoch(db, instance_id='current'))
        source = SimpleNamespace(platform=Platform.TELEGRAM, user_id='human')
        authority.sessions['root'] = LiveSession(source, 'agent:telegram:dm:1')
        history = local_history(authority, SessionRef('owned', 'root'))
        contents = [m['content'] for m in history]
        assert 'AFTER_COMPRESSION' in contents
        # The root's turn is still shown, marked archived: display history, never a live row.
        assert [m['content'] for m in history if m.get('archived')] == ['BEFORE_COMPRESSION']
        # Snapshot rows carry their durable row id (a session.mutate rewind's target_message_id),
        # under the public TranscriptMessage name, never the internal ``_row_id``.
        assert [m['row_id'] for m in history if not m.get('archived')] == [m['id'] for m in db.get_messages('child')]
        assert not any('_row_id' in m for m in history)


@pytest.mark.allow_real_home_io  # importing gateway.config activates the checkout's install root; this test writes only tmp_path
def test_a_prepared_history_does_not_resolve_the_transcript_again(tmp_path, monkeypatch):
    """``attach`` reads the resume history inline, under the session's event-stream lock, on the loop.

    Resolving *which* transcript that history is (a receipt lineage read plus the local store's JSON
    load) is filesystem work that ``prepare_history`` already hops off the loop; a second resolve
    inside the lock put ``restore_local_session`` on the loop — the frame the out-of-loop liveness
    watchdog named at 17:16:13Z, while 100 serial control probes took 1-6 s.
    """
    from gateway.config import Platform
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway import session_local_recovery as recovery
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('root', source='telegram', session_key='agent:telegram:dm:1')
        db.append_message('root', 'user', 'Q1')
        authority = SessionAuthority(
            SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False)),
            profile_id='owned', instance_id='current', db=db,
            epoch=begin_runtime_epoch(db, instance_id='current'))
        authority.sessions['root'] = LiveSession(
            SimpleNamespace(platform=Platform.TELEGRAM, user_id='human'), 'agent:telegram:dm:1')
        ref = SessionRef('owned', 'root')
        resolved = []
        real = recovery.transcript_target

        def spy(auth, r):
            resolved.append(r.session_id)
            return real(auth, r)

        monkeypatch.setattr(recovery, 'transcript_target', spy)
        prepared = (authority.physical_target(ref), recovery.display_rows(db, authority.physical_target(ref)))
        before = len(resolved)
        rows = recovery.local_history(authority, ref, prepared)
        assert [r['content'] for r in rows] == ['Q1']
        assert len(resolved) == before, (
            'local_history resolved the transcript a second time, on the caller\'s thread')


@pytest.mark.asyncio
async def test_branch_and_local_entries_keep_the_routing_prune_sweep_running(tmp_path, monkeypatch):
    """Every producer of canonical routing entries (local create, API bind, ``session.mutate``
    branch) stamps the store clock: one aware-UTC entry makes ``prune_old_entries`` raise against
    its naive cutoff, and the watcher swallows that, so no stale route is ever pruned again."""
    from datetime import timedelta
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_api import bind_api_session
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_mutations import mutate_session
    from gateway import run

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
    ref = create_local_session(authority, actor, {'request_id': 'r', 'source': 'gui', 'cwd': str(tmp_path),
                                                  'model': 'frozen', 'toolsets': []})
    bind_api_session(authority, 'api-owner')
    branched = await mutate_session(authority, actor, ref, dict(session_id=ref.session_id, request_id='branch',
        expected_revision=0, expected_generation=0, operation='branch', payload={}))
    telegram = store.get_or_create_session(
        SessionSource(platform=Platform.TELEGRAM, chat_id='7', user_id='7', chat_type='dm'))
    # Cold store: only the durable index feeds the sweep.
    cold = SessionStore(tmp_path / 'sessions', GatewayConfig())
    with cold._lock:
        cold._ensure_loaded_locked()
        assert any(e.session_id == branched['branched_session_id'] for e in cold._entries.values())
        cold._entries[telegram.session_key].updated_at -= timedelta(days=90)
    assert cold.prune_old_entries(30) == 1
    assert cold.lookup_by_session_key(telegram.session_key) is None


@pytest.mark.asyncio
async def test_boot_drops_only_the_receipts_whose_session_row_is_gone(tmp_path, monkeypatch):
    """PRX-6097: a receipt that names no session row can never be restored, yet every boot refused
    the same ids again (34 of them in one live profile, unchanged across a code update). Boot now
    reconciles exactly those — receipt, routing row and api-settings/viewer-close keys — and still
    restores a receipt that names a live row: the class must not be silenced by dropping receipts
    wholesale.
    """
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway import run
    from gateway import session_local_recovery as recovery

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    params = {'request_id': 'stale', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []}
    stale = create_local_session(authority, actor, params)
    live = create_local_session(authority, actor, {**params, 'request_id': 'live'})
    # The residue a delete that bypassed the fence leaves: the row (and its messages) gone, the
    # receipt and routing entry that name it still there.
    assert authority.db.discard_unadmitted_session(stale.session_id) is True
    authority.sessions.pop(stale.session_id, None)

    await recovery.recover_local_sessions_offloop(authority, schedule=False)

    with authority.db._read_ctx() as conn:
        receipts = {k for (k,) in conn.execute(
            "SELECT key FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%'")}
        routed = [k for (k,) in conn.execute('SELECT session_key FROM gateway_routing WHERE entry_json LIKE ?',
                                             ('%' + stale.session_id + '%',))]
    assert 'gateway.local_policy.v1:' + stale.session_id not in receipts
    assert routed == [], 'the routing row naming the gone session outlived its receipt'
    assert 'gateway.local_policy.v1:' + live.session_id in receipts
    assert recovery.restore_local_session(authority, live.session_id) == live


@pytest.mark.asyncio
async def test_boot_drops_only_the_receipts_whose_policy_cannot_be_restored(tmp_path, monkeypatch):
    """PRX-6122: a receipt whose stored policy ``restore_policy`` refuses is as unrestorable as one
    whose row is gone. Boot drops exactly those receipts and keeps their live sessions rows; a
    restorable receipt still restores.
    """
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway import run
    from gateway import session_local_recovery as recovery
    from hermes_state_local import POLICY_PREFIX

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    params = {'request_id': 'good', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []}
    good = create_local_session(authority, actor, params)
    bad = create_local_session(authority, actor, {**params, 'request_id': 'bad'})

    # Simulate the writer class PRX-6122 found: an empty model frozen in the stored policy.
    with authority.db._read_ctx() as conn:
        raw = conn.execute('SELECT value FROM state_meta WHERE key=?',
                           (POLICY_PREFIX + bad.session_id,)).fetchone()[0]
    receipt = json.loads(raw)
    receipt['policy']['model'] = ''
    receipt['policy']['source'] = 'oneshot'
    receipt['policy']['platform'] = 'cli'
    authority.db._write_sql('UPDATE state_meta SET value=? WHERE key=?',
                            (json.dumps(receipt), POLICY_PREFIX + bad.session_id))

    await recovery.recover_local_sessions_offloop(authority, schedule=False)

    with authority.db._read_ctx() as conn:
        receipts = {k for (k,) in conn.execute(
            "SELECT key FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%'")}
        row = conn.execute('SELECT id FROM sessions WHERE id=?', (bad.session_id,)).fetchone()
    assert POLICY_PREFIX + bad.session_id not in receipts
    assert row is not None, 'the sessions row is not the receipt and must stay'
    assert POLICY_PREFIX + good.session_id in receipts
    assert recovery.restore_local_session(authority, good.session_id) == good


@pytest.mark.asyncio
async def test_create_without_a_model_is_refused_instead_of_minting_an_unrestorable_receipt(tmp_path, monkeypatch):
    """PRX-6122: a profile with no model and no explicit ``-m`` used to freeze ``model=''`` in the
    receipt, which ``restore_policy`` refuses forever. The create must be refused, not minted."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway import run
    from hermes_state_runtime import RuntimeStoreError

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda *a, **k: '')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        create_local_session(authority, actor, {'request_id': 'no-model', 'source': 'oneshot',
                                                'cwd': str(tmp_path), 'toolsets': []})
    with authority.db._read_ctx() as conn:
        receipts = conn.execute(
            "SELECT count(*) FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%'").fetchone()[0]
    assert receipts == 0, 'the refused create minted a receipt'


@pytest.mark.asyncio
async def test_delete_retires_owner_receipts_and_never_resurrects_the_id(tmp_path, monkeypatch):
    """The deletion fence takes the local policy receipt (a cron policy's ``request_json`` holds
    the job and raw prompt) and the API binding + declared conversation key with the transcript;
    a later create/cron/API retry of the same deterministic id is refused instead of recreating it."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_api import bind_api_session
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_mutations import mutate_session
    from gateway import run
    from hermes_state_runtime import RuntimeStoreError

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
    params = {'request_id': 'r', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []}
    local = create_local_session(authority, actor, params)
    api = bind_api_session(authority, 'api-owner', declared_key='conversation')
    for ref in (local, api):
        await mutate_session(authority, actor, ref, dict(session_id=ref.session_id, request_id='delete',
            expected_revision=0, expected_generation=0, operation='delete', payload={}))
    with authority.db._read_ctx() as conn:
        left = [k for (k,) in conn.execute("SELECT key FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%' "
                                           "OR key LIKE 'gateway.api.%'")]
    assert left == []
    with pytest.raises(RuntimeStoreError, match='not_found'):
        create_local_session(authority, actor, params)
    with pytest.raises(RuntimeStoreError, match='not_found'):
        bind_api_session(authority, 'api-owner')
    assert authority.db.get_session(local.session_id) is None
    assert authority.db.get_session('api-owner') is None
