"""PRX-5850: a lost finite one-shot turn has no waiter, so it settles instead of blocking forever.

``hermes chat --resume <id> -q`` / ``hermes -z`` submits ``{'finite': True, 'one_shot': True, 'text':
...}`` and the process exits when the turn settles. After an owner restart fences that admission ``unknown`` there
is no live process to consult, yet the row paused the session's FIFO forever: every later wake
refused with "Unknown execution blocks this session; nothing was submitted" and only a manual
``hermes sessions discard`` cleared it (58 failed runs / 11 agents / 20 sessions in 12 h, PRX-5850).
Startup recovery does fence it and then reports the *route* ``not_found`` (the row carries no
``native_text_v1`` envelope), so nothing used to settle it.

Real ``SessionStore`` SQLite, two real owner epochs; no store monkeypatching.
"""
import queue
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_authority import LiveSession, initialize_session_authority
from gateway.session_contract import SessionRef, Submission
from gateway.session_controls import AuthorityConnection
from hermes_state_runtime import claim_session_input, list_session_admissions
from tests.waiting import async_wait_until


class Peer:
    def __init__(self):
        self.frames = queue.Queue()

    def write(self, frame):
        self.frames.put(frame)
        return True


def _runner(tmp_path, monkeypatch, executed):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()['result'] = {'final_response': 'ACK_' + event.text, 'completed': True, 'messages': []}
        return 'ACK_' + event.text

    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    return store, runner


async def _lost_one_shot_then_restart(tmp_path, monkeypatch, executed):
    """Owner 1 admits a finite one-shot head, claims it, dies mid-turn. Owner 2 starts."""
    store, runner = _runner(tmp_path, monkeypatch, executed)
    db = store._db
    # The admission is what a one-shot submits (``{'finite', 'one_shot', 'text'}``, stamped by
    # ``session_finite.admit_finite``); the *route* is a plain chat so the fixture needs no local
    # receipt to drain (the verdict and the settle read the row).
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='gw5850', user_id='uid:501')
    db.create_session('s', source='telegram')
    first = await initialize_session_authority(runner, profile_id='p', instance_id='owner-1')
    first.sessions['s'] = LiveSession(source, 's')
    ref = SessionRef('p', 's')
    actor = AuthorityConnection(first, Peer(), {'user_id': 'uid:501'}, operator=True).actor
    first._schedule = lambda ref: None
    # The payload a one-shot submits: exactly ``{'finite', 'text'}``, no native envelope.
    head = await first.submit(actor, Submission('head', ref, {'text': 'WAKE', 'finite': True, 'one_shot': True}, 'queue'))
    claimed = claim_session_input(db, epoch=first.epoch, session_id='s')
    assert claimed['admission_id'] == head.admission_id
    assert claimed['payload']['one_shot'] is True and claimed['payload']['finite'] is True
    second = await initialize_session_authority(runner, profile_id='p', instance_id='owner-2')
    second.sessions['s'] = LiveSession(source, 's')
    return store, second, head.admission_id, actor


@pytest.mark.asyncio
async def test_a_lost_one_shot_settles_at_boot_and_the_session_serves_the_next_one(tmp_path, monkeypatch):
    """The blocked class: the lost one-shot settles ``interrupted``, and the *next* one-shot run on
    the same persisted session is served instead of refusing with "Unknown execution blocks this
    session"."""
    executed = []
    store, authority, head_id, _actor = await _lost_one_shot_then_restart(tmp_path, monkeypatch, executed)
    db = authority.db
    try:
        rows = {r['admission_id']: (r['status'], r['outcome'])
                for r in list_session_admissions(db, session_id='s', pending_only=False)}
        assert rows[head_id] == ('terminal', 'interrupted'), rows
        # Nothing is replayed: the lost input stays in the transcript, closing with its notice.
        notices = [m['content'] for m in db.get_messages('s') if m.get('display_kind') == 'failed_turn']
        assert notices and 'one-shot' in notices[0].lower(), notices
        # The next run on the same session (what the adapter does on every wake) is admitted.
        actor = AuthorityConnection(authority, Peer(), {'user_id': 'uid:501'}, operator=True).actor
        receipt = await authority.submit(actor, Submission('next', SessionRef('p', 's'),
                                                          {'text': 'NEXT', 'finite': True, 'one_shot': True}, 'queue'))
        assert receipt.admission_id != head_id
        live = authority.sessions['s']
        authority._schedule(SessionRef('p', 's'))
        await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain finished')
        assert executed == ['NEXT']
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_native_recovery_not_found_after_a_recorded_execution_is_not_a_permanent_block(tmp_path, monkeypatch):
    """Behaviour contract for the ``not_found`` verdict (untested before PRX-5850): startup recovery
    finds no native route for the recorded execution's session — the row carries no
    ``native_text_v1`` envelope — and says so. That verdict alone must not leave the session
    blocked; the recorded one-shot has settled and the session runs again."""
    executed = []
    store, authority, head_id, _actor = await _lost_one_shot_then_restart(tmp_path, monkeypatch, executed)
    try:
        source = authority.sessions['s'].source
        outcome = await authority.recover_native_sessions([('s', source, None)])
        assert outcome == {'s': 'not_found'}, outcome
        rows = {r['admission_id']: (r['status'], r['outcome'])
                for r in list_session_admissions(authority.db, session_id='s', pending_only=False)}
        assert rows[head_id] == ('terminal', 'interrupted'), rows
        assert authority.db.get_session('s') is not None  # the session itself is intact
    finally:
        store.close_all_db_handles()


@pytest.mark.parametrize('payload, expected', [
    ({'text': 'typed into a chat'}, False),                        # a person's turn still pauses
    ({'text': 'a one-shot wake', 'finite': True, 'one_shot': True}, True),
    ({'text': 'a card attempt', 'finite': True}, False),           # finite alone: Kanban keeps pausing
    ({'text': 'a pre-PRX-5850 one-shot', 'finite': True}, False),  # an unmarked legacy row keeps pausing
    ({'text': 'x', 'local_automation_v1': {'identity': 'p'}}, True),
])
def test_ownerless_predicate_keeps_a_persons_turn(payload, expected):
    """The settle predicate must cover the one-shot payload and nothing wider: a person's turn
    (no ``finite``) keeps pausing the FIFO for Discard."""
    from gateway.session_unattended_unknown import settles_without_a_waiter
    assert settles_without_a_waiter({'status': 'unknown', 'principal_id': 'uid:501', 'payload': payload}) is expected
