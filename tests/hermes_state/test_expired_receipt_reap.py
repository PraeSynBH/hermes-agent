"""Regression for PRX-6167: a settled admission result must not outlive its retention.

The defect was measured, not theorised: every profile store sat at zero admission tombstones
against hundreds of result rows, growing ~0.5-1.7 rows/min fleet-wide, and the parent's reaper
counted exactly those rows as orphans. Re-reading the writers shows the two do share a store —
``settle_session_input`` resolves the admission row in the SAME connection it writes the result
into (``_admission``) — so a cross-store split is not what produced them: the fleet census behind
this fix found 0 rows whose admission is missing and 365/365 resolving to a terminal ledger row in
their own store. What produced them is that the only delete path walks ADMISSION tombstones,
tombstones are written only by session RETIREMENT, and ``sessions.auto_prune`` is off: a settled
result of a session nobody retires was reachable by no reclaim path at all, and the whole
maintenance arm (which gates ``prune_retired_tombstones``) returned before it ran.

So the fix is the retention rule, in two stages, on the result row's own clock: the transcript
copy goes once the turn is spent, the receipt (and its terminal ledger row) at the exact-retry
horizon. These tests drive the real writer, the real reap and two real stores.
"""
import json
import time

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt
from hermes_state_runtime import RuntimeStoreError
from gateway.session_results import admission_result, finish_result, retain_result
from hermes_state_terminal import ADMISSION_PREFIX, RESULT_PREFIX

REPLY = 'the answer, once'


def _store(path, *sessions):
    db = SessionDB(path)
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    for sid in sessions:
        db.create_session(sid, source='api_server')
    return db, epoch


def _settle(db, epoch, sid, n, *, reply=REPLY, writer=retain_result):
    """One real turn cycle: admit, claim, settle — the writer whose row the reap has to reach."""
    rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id=sid,
                           request_id=f'r{n}', payload={'text': f'question {n}'})
    row = rt.claim_session_input(db, epoch=epoch, session_id=sid)
    result = {'result': {'final_response': reply, 'messages': [
        {'role': 'user', 'content': f'question {n}'},
        {'role': 'assistant', 'content': reply}], 'completed': True}, 'usage': {'input_tokens': 3}}
    if writer is finish_result:
        writer(db, epoch=epoch, row=row, response=reply, outcome='completed', result=result)
    else:
        writer(db, epoch=epoch, row=row, result=result)
    return row['admission_id']


def _result_row(db, admission_id):
    with db._read_ctx() as conn:
        return conn.execute('SELECT value FROM state_meta WHERE key=?',
                            (RESULT_PREFIX + admission_id,)).fetchone()


def _ledger_row(db, admission_id):
    return rt.get_session_admission(db, admission_id=admission_id)


def _prune(db, **kwargs):
    from hermes_state_mutation_retirement import prune_retired_tombstones
    return db._execute_write(lambda conn: prune_retired_tombstones(conn, **kwargs))


def test_a_live_sessions_settled_result_is_reaped_past_the_horizon_and_kept_within_it(tmp_path):
    db, epoch = _store(tmp_path / 'state.db', 'live')
    try:
        admission_id = _settle(db, epoch, 'live', 1, writer=finish_result)
        assert admission_result(db, admission_id)['result']['final_response'] == REPLY
        # Inside the horizon nothing is touched — the row is an exact-retry receipt, not garbage.
        assert _prune(db) == 0
        assert _result_row(db, admission_id) is not None
        assert _ledger_row(db, admission_id)['status'] == 'terminal'
        # Past it the PAIR goes: a result-less terminal row would read as an ambiguous/unknown
        # execution to every receipt reader, while forgetting both is what a retired session's
        # receipt already does past this same horizon.
        assert _prune(db, older_than=-1) == 1
        assert _result_row(db, admission_id) is None
        assert _ledger_row(db, admission_id) is None
        # Only the receipts go: the session and its transcript are history, not a receipt.
        assert db.get_session('live') is not None
    finally:
        db.close()


def test_the_transcript_copy_goes_long_before_the_receipt_does(tmp_path):
    """Where the bytes are: a settled result's copy of the turn. Measured 200x on this host's real
    stores (65.4 MB of result rows -> 0.32 MB), and the reply every reader resolves stays exact."""
    db, epoch = _store(tmp_path / 'state.db', 'live')
    try:
        admission_id = _settle(db, epoch, 'live', 1, writer=finish_result)
        full = len(_result_row(db, admission_id)[0])
        assert json.loads(_result_row(db, admission_id)[0])['result']['messages']
        # Spent, not expired: the transcript copy goes through the same compactor retirement uses.
        assert _prune(db, compact_older_than=-1) == 0
        compacted = len(_result_row(db, admission_id)[0])
        assert compacted < full
        kept = admission_result(db, admission_id)
        assert kept['result']['final_response'] == REPLY
        assert kept['result']['messages'] == []
        # The receipt itself still resolves for the whole exact-retry horizon.
        assert _ledger_row(db, admission_id)['status'] == 'terminal'
    finally:
        db.close()


def test_an_orphan_result_is_reaped_even_though_no_admission_row_is_left(tmp_path):
    """The shape the issue named: a result whose admission row is absent from THIS store. Every
    reader resolves the admission first, so such a row is unreadable by construction — and the
    tombstone branch can never see it (there is no tombstone). It ages on its own stamp."""
    db, epoch = _store(tmp_path / 'state.db', 'live')
    try:
        admission_id = _settle(db, epoch, 'live', 1)
        db._execute_write(lambda conn: conn.execute(
            'DELETE FROM session_admissions WHERE admission_id=?', (admission_id,)))
        assert _ledger_row(db, admission_id) is None
        with pytest.raises(RuntimeStoreError):
            admission_result(db, admission_id)  # unreadable, exactly as reported
        # Bounded grace: the first pass only stamps the row it has no settlement time for...
        assert _prune(db) == 0
        assert _result_row(db, admission_id) is not None
        assert json.loads(_result_row(db, admission_id)[0])['retired_at'] <= time.time()
        # ...and the next one, past the horizon, reclaims it.
        assert _prune(db, older_than=-1) == 1
        assert _result_row(db, admission_id) is None
    finally:
        db.close()


def test_the_reap_is_per_store_and_never_store_blind(tmp_path):
    """Two real state.db files: a result in one, an admission tombstone for the same id in the
    other. Neither pass may reach across stores — the rejected fix was a store-blind prefix
    DELETE — and the tombstone's own store still reaps its pair on its own clock."""
    (tmp_path / 'session').mkdir()
    (tmp_path / 'launch').mkdir()
    admitting, epoch = _store(tmp_path / 'session' / 'state.db', 'live')
    launch, _ = _store(tmp_path / 'launch' / 'state.db', 'live')
    try:
        admission_id = _settle(admitting, epoch, 'live', 1)
        assert _result_row(admitting, admission_id) is not None
        # A retired-session tombstone for that same admission, in the LAUNCH store, aged out.
        launch._execute_write(lambda conn: conn.execute(
            'INSERT INTO state_meta(key,value) VALUES(?,?)',
            (ADMISSION_PREFIX + admission_id, json.dumps({
                'admission_id': admission_id, 'principal_id': 'api', 'target_session_id': 'live',
                'request_id': 'r1', 'retired_at': 0}))))
        assert _prune(launch, older_than=-1) == 0
        assert _result_row(launch, admission_id) is None
        # The session store's own result survived a pass in the other store...
        assert _result_row(admitting, admission_id) is not None
        assert admission_result(admitting, admission_id)['result']['final_response'] == REPLY
        # ...and its own pass reclaims it.
        assert _prune(admitting, older_than=-1) == 1
        assert _result_row(admitting, admission_id) is None
    finally:
        admitting.close()
        launch.close()


def test_the_housekeeping_entry_point_runs_once_per_interval_and_carries_the_reap(tmp_path):
    from hermes_state_mutation_retirement import RECEIPT_REAP_MARKER, prune_expired_receipts
    db, epoch = _store(tmp_path / 'state.db', 'live')
    try:
        admission_id = _settle(db, epoch, 'live', 1)
        # First pass stamps (no settlement stamp existed), so nothing is old enough yet.
        assert prune_expired_receipts(db, min_interval_hours=24) == 0
        assert _result_row(db, admission_id) is not None
        # Inside the interval the pass does not even run again.
        assert prune_expired_receipts(db, min_interval_hours=24) == 0
        with db._read_ctx() as conn:
            marker = conn.execute('SELECT value FROM state_meta WHERE key=?',
                                  (RECEIPT_REAP_MARKER,)).fetchone()
        assert json.loads(marker[0])['at'] > 0
        # Interval elapsed and horizon passed: the row is gone, and the ledger row with it.
        assert prune_expired_receipts(db, min_interval_hours=0, older_than=-1) == 1
        assert _result_row(db, admission_id) is None
        assert _ledger_row(db, admission_id) is None
    finally:
        db.close()


def test_an_unsettled_turn_never_loses_its_row_to_the_reap(tmp_path):
    """The guard that keeps this from being a store-blind prefix DELETE of live work: the result
    key namespace is only written by settlement, but a queued/started claim must survive any pass."""
    db, epoch = _store(tmp_path / 'state.db', 'live')
    try:
        rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id='live',
                               request_id='r1', payload={'text': 'question'})
        claimed = rt.claim_session_input(db, epoch=epoch, session_id='live')
        # A result row that a crashed writer left behind for a claim that is still STARTED.
        db._execute_write(lambda conn: conn.execute(
            'INSERT INTO state_meta(key,value) VALUES(?,?)',
            (RESULT_PREFIX + claimed['admission_id'], json.dumps({'retired_at': 0, 'result': {}}))))
        assert _prune(db, older_than=-1) == 0
        assert _result_row(db, claimed['admission_id']) is not None
        assert _ledger_row(db, claimed['admission_id'])['status'] == 'started'
    finally:
        db.close()
