"""The optional contract declaration must never be able to fail the request it decorates.

SOU-3318 shipped two halves in one commit: the owner learned to *accept* ``contract_revision`` and
the client learned to *send* it. Between that merge and the gateway restart, every agent process
spawned from the new tree declared a param the serving process did not define — **40 runs across 13
agents and 6 companies refused at 0 tokens** (00:56:38Z–01:06:45Z), the SOU-3316 class caused by the
fence's own metadata.

The rule these tests hold: a refusal is settled locally, exactly once, ONLY when the owner names the
declaration and nothing else. A skew, another unknown key, or a refusal that names no key at all
propagates untouched — dropping the declaration there would either hide the skew the fence exists to
report or turn a genuinely malformed request into a retry loop.
"""

from __future__ import annotations

import pytest

from gateway import contract_revision
from gateway.session_controls import _canonical_param_refusal, _refusal_data
from hermes_cli.gateway_agent import GatewayAgent
from hermes_cli.gateway_chat_view import GatewayChatView
from hermes_cli.gateway_client import (
    GatewayContractSkewError, GatewayRPCError, declaration_refused, declaration_retry_params, rpc_error,
)

DECLARED = 'ffffffffffff'
OTHER = 'eeeeeeeeeeee'


def _frame(reason, **data):
    """A JSON-RPC error member exactly as ``AuthorityConnection.dispatch`` builds one."""
    return {'code': 4001, 'message': reason, 'data': {'reason': reason, **data}}


def _gateway_frame(problems, declared=None):
    """The frame the owner's canonical param gate emits, built from the owner's own raiser."""
    params = {'session_id': 's', 'text': 'hi'}
    if declared is not None:
        params[contract_revision.REVISION_PARAM] = declared
    exc = _canonical_param_refusal(params, problems)
    return {'code': 4001, 'message': exc.reason, 'data': {**_refusal_data(exc), 'fields': problems}}


# -- the predicate -------------------------------------------------------------------------------


def test_the_owners_own_declaration_refusal_is_recognized():
    """Cross-boundary: the frame the gateway actually emits is the one the client settles."""
    error = rpc_error(_gateway_frame(['contract_revision'], declared=contract_revision.revision()))

    assert error.data['fields'] == ['contract_revision']
    assert declaration_refused(error) is True
    assert declaration_retry_params(error, {'session_id': 's', 'contract_revision': DECLARED, 'text': 'hi'}) \
        == {'session_id': 's', 'text': 'hi'}


def test_a_refusal_naming_another_key_is_not_the_declarations():
    assert declaration_refused(rpc_error(_frame('invalid_params', fields=['one_shot']))) is False
    assert declaration_refused(
        rpc_error(_frame('invalid_params', fields=['contract_revision', 'one_shot']))) is False


def test_a_bare_refusal_with_no_key_named_is_never_settled():
    """No key named = nothing proves the declaration was the cause: propagate, as before."""
    assert declaration_refused(rpc_error(_frame('invalid_params'))) is False


def test_a_skew_is_never_settled_by_dropping_the_declaration():
    """The fence's own verdict must reach the caller: dropping the param would hide the skew."""
    skew = rpc_error(_gateway_frame(['contract_revision'], declared=OTHER))
    assert isinstance(skew, GatewayContractSkewError)
    assert declaration_refused(skew) is False
    assert declaration_retry_params(skew, {'session_id': 's', 'contract_revision': OTHER}) is None


def test_the_field_form_is_recognized_too():
    """PRA-6722's ``submission()`` refusal names the key in ``field`` only."""
    assert declaration_refused(rpc_error(_frame('invalid_params', field='contract_revision'))) is True


# -- the one-shot path (the 40-run incident) -----------------------------------------------------


class _RefusingPeer:
    """A peer whose first ``prompt.submit`` is refused on the declaration alone, then accepts."""

    def __init__(self):
        self.calls = []

    async def rpc(self, method, **params):
        self.calls.append((method, params))
        if len(self.calls) == 1:
            raise rpc_error(_frame('invalid_params', fields=['contract_revision']))
        return {'admission_id': 'mine'}


@pytest.mark.asyncio
async def test_a_one_shot_resends_once_without_the_declaration_and_keeps_its_identity():
    peer = _RefusingPeer()
    view = GatewayChatView(peer, {'stored_session_id': 's'}, quiet=True)
    view.finite = True

    receipt = await view.submit('WAKE')

    assert receipt == {'admission_id': 'mine'}
    assert len(peer.calls) == 2, 'exactly one resend, never a loop'
    first, second = peer.calls[0][1], peer.calls[1][1]
    assert 'contract_revision' in first
    assert 'contract_revision' not in second
    assert second['input_id'] == first['input_id'], 'the resend keeps the admission identity'
    assert {k: v for k, v in second.items() if k != 'input_id'} == \
           {k: v for k, v in first.items() if k not in ('input_id', 'contract_revision')}


# -- the agent path ------------------------------------------------------------------------------


class _FakeConnection:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def call(self, method, **params):
        self.calls.append((method, params))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _agent(connection):
    agent = object.__new__(GatewayAgent)
    agent.connection = connection
    return agent


def _params():
    return {'session_id': 's', 'submission_id': 'sub-1', 'text': 'hi', 'contract_revision': DECLARED}


def test_the_agent_resends_once_with_the_same_submission_id():
    connection = _FakeConnection([rpc_error(_frame('invalid_params', fields=['contract_revision'])),
                                  {'admission_id': 'a'}])

    receipt = _agent(connection)._submit_turn(_params(), 's')

    assert receipt == {'admission_id': 'a'}
    assert len(connection.calls) == 2
    resent = connection.calls[1][1]
    assert 'contract_revision' not in resent
    assert resent['submission_id'] == 'sub-1'
    assert resent['session_id'] == 's'


def test_another_unknown_key_propagates_without_a_resend():
    connection = _FakeConnection([rpc_error(_frame('invalid_params', fields=['one_shot']))])
    with pytest.raises(GatewayRPCError):
        _agent(connection)._submit_turn(_params(), 's')
    assert len(connection.calls) == 1


def test_a_second_refusal_propagates_instead_of_looping():
    connection = _FakeConnection([rpc_error(_frame('invalid_params', fields=['contract_revision'])),
                                  rpc_error(_frame('invalid_params', fields=['contract_revision']))])
    with pytest.raises(GatewayRPCError):
        _agent(connection)._submit_turn(_params(), 's')
    assert len(connection.calls) == 2


def test_a_skew_fails_the_turn_as_a_skew():
    skew = rpc_error(_gateway_frame(['contract_revision'], declared=OTHER))
    connection = _FakeConnection([skew])
    params = {**_params(), 'contract_revision': OTHER}

    with pytest.raises(GatewayContractSkewError):
        _agent(connection)._submit_turn(params, 's')
    assert len(connection.calls) == 1, 'a skew is a verdict, not a retryable noise'
