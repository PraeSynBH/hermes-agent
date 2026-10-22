"""A gateway refusal names the check that fired and the keys that tripped it (PRA-6722).

PRA-6720's 47-run / 3h10m outage was silent because three closed key sets raised the bare reason
code: the run log read ``Error: invalid_params``, the gateway log wrote nothing, and the operator
had to re-derive the cause from reflog, process start times and 500 run rows. The key sets stay
closed (the refusal was correct); these are the messages' behaviour contracts.
"""
import logging

import pytest

from gateway.session_authority import SessionAuthority
from gateway.session_contract import SessionRef, Submission
from gateway.session_controls import AuthorityConnection
from gateway.session_operator import check_local_input
from hermes_state_runtime import RuntimeStoreError

SID = 'local-' + '6a63e8f4adf7140d2354fa04345f76af3b2a940336e5534749ad1156b3817b60'
REQUEST_ID = '0' * 32


def _ref(session_id=SID):
    return SessionRef('test', session_id)


def _connection():
    # submission() reads only ``self.subscriptions`` before the key-set check: no authority, no db.
    conn = object.__new__(AuthorityConnection)
    conn.subscriptions = {SID: object()}
    return conn


def test_submission_names_the_unexpected_param_and_the_key():
    with pytest.raises(RuntimeStoreError) as raised:
        _connection().submission(_ref(), {'session_id': SID, 'text': 'hi', 'input_id': REQUEST_ID,
                                          'one_shit': True})
    assert raised.value.reason == 'invalid_params'
    assert raised.value.field == 'submission_params'
    assert 'one_shit' in raised.value.detail


@pytest.mark.parametrize('request_key', ['submission_id', 'input_id'])
def test_submission_names_a_missing_request_id(request_key):
    with pytest.raises(RuntimeStoreError) as raised:
        _connection().submission(_ref(), {'session_id': SID, 'text': 'hi', request_key: ''})
    assert raised.value.reason == 'invalid_params'
    assert raised.value.field == 'submission_id'
    assert request_key in raised.value.detail


def _bare_authority():
    # _submit refuses on the payload before it touches the session, db or mutation lock.
    authority = object.__new__(SessionAuthority)
    authority.retiring = False
    authority.runner = type('Runner', (), {'_draining': False})()
    return authority


@pytest.mark.asyncio
async def test_authority_submit_names_the_unexpected_payload_key():
    request = Submission('req-1', _ref(), {'text': 'hi', 'one_shit': True}, 'queue')
    with pytest.raises(RuntimeStoreError) as raised:
        await _bare_authority()._submit(object(), request)
    assert raised.value.reason == 'invalid_params'
    assert raised.value.field == 'payload'
    assert 'one_shit' in raised.value.detail


@pytest.mark.asyncio
async def test_authority_submit_names_a_payload_that_is_not_text():
    request = Submission('req-1', _ref(), {'text': ['hi']}, 'queue')
    with pytest.raises(RuntimeStoreError) as raised:
        await _bare_authority()._submit(object(), request)
    assert raised.value.reason == 'invalid_params'
    assert raised.value.field == 'text'


def test_local_input_refusal_names_the_unexpected_key():
    authority = type('Authority', (), {
        'profile_id': 'test',
        'sessions': {'s': type('Live', (), {'source': type('Source', (), {'user_id': 'u'})()})()},
    })()
    row = {'request_id': 'viewer-1', 'principal_id': 'u', 'payload': {'text': 'hi', 'one_shit': True}}
    with pytest.raises(RuntimeStoreError) as raised:
        check_local_input(authority, _ref('s'), row)
    # The reason is unchanged (the row is not admitted); only the message gained content.
    assert raised.value.reason == 'permission_denied'
    assert raised.value.field == 'payload'
    assert 'one_shit' in raised.value.detail


@pytest.mark.asyncio
async def test_canonical_gate_refusal_surfaces_the_keys_and_logs_them(caplog):
    """The gate that refuses before any handler runs named the keys on the wire (``data.fields``)
    but wrote no log line: a run that never reached a handler left the gateway log empty too."""
    authority = type('Authority', (), {'profile_id': 'test', 'instance_id': 'inst', 'events': {},
                                       'retiring': False, 'sessions': {}})()
    conn = AuthorityConnection(authority, object(), {'user_id': 'human'})
    with caplog.at_level(logging.WARNING, logger='gateway.session_controls'):
        reply = await conn.dispatch({'id': 7, 'method': 'prompt.submit', 'params': {
            'session_id': SID, 'text': 'hi', 'input_id': REQUEST_ID, 'one_shit': True}})
    assert reply['error']['data']['fields'] == ['one_shit']
    assert any('one_shit' in record.getMessage() and 'invalid_params' in record.getMessage()
               for record in caplog.records), [record.getMessage() for record in caplog.records]
