"""The client's half of the contract-skew fence (SOU-3318).

A ``contract_version_mismatch`` frame must become a distinct, non-retryable verdict that names both
revisions — never a generic RPC refusal a caller might retry, and never the bare code the SOU-3316
run log was stuck with.
"""

from __future__ import annotations

from hermes_cli.cli_chat_error_copy import contract_skew_text, refusal_text
from hermes_cli.gateway_client import GatewayContractSkewError, GatewayRPCError, rpc_error

CLIENT_REV = 'aaaaaaaaaaaa'
GATEWAY_REV = 'bbbbbbbbbbbb'


def _frame(reason='contract_version_mismatch'):
    return {
        'code': 4001,
        'message': reason,
        'data': {'reason': reason, 'field': 'one_shot',
                 'detail': "unexpected keys: ['one_shot']",
                 'client_revision': CLIENT_REV, 'gateway_revision': GATEWAY_REV},
    }


def test_a_skew_frame_becomes_the_distinct_non_retryable_verdict():
    error = rpc_error(_frame())

    assert isinstance(error, GatewayContractSkewError)
    assert str(error) == 'contract_version_mismatch'
    assert error.retryable is False
    assert (error.client_revision, error.gateway_revision) == (CLIENT_REV, GATEWAY_REV)


def test_an_ordinary_refusal_is_unchanged():
    error = rpc_error({'code': 4001, 'message': 'invalid_params',
                       'data': {'reason': 'invalid_params', 'field': 'submission_params',
                                'detail': "unexpected keys: ['bogus']"}})

    assert type(error) is GatewayRPCError
    assert str(error) == 'invalid_params'


def test_the_copy_names_both_revisions_and_the_fix():
    text = refusal_text(rpc_error(_frame()))

    assert 'contract_version_mismatch' in text
    assert CLIENT_REV in text and GATEWAY_REV in text
    assert 'one_shot' in text
    assert 'Restart the gateway' in text and 'do not retry' in text.lower()


def test_the_copy_survives_a_frame_without_revisions():
    """A skew named with no revisions (an owner that could not hash its contract) still says restart."""
    text = contract_skew_text(GatewayContractSkewError(4001, {'reason': 'contract_version_mismatch'}))

    assert 'unknown' in text and 'Restart the gateway' in text
