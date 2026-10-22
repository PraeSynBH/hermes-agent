"""The one-shot CLI prints the refusal's check, not just its reason code (PRA-6722).

PRA-6720: 47 one-shot runs died in ~0.5 s and every one of them logged, in full, ``Error:
invalid_params``. The owner had the content (``field``/``detail``, or the canonical gate's
``data.fields``); this is the client half — the display reads it.
"""
import argparse
import asyncio

from hermes_cli.cli_chat_error_copy import refusal_text
from hermes_cli.gateway_client import GatewayRPCError, rpc_error


def test_refusal_text_renders_the_owners_field_and_detail():
    error = GatewayRPCError('invalid_params', 4001, {
        'reason': 'invalid_params', 'field': 'submission_params',
        'detail': "unexpected keys: ['one_shot']"})
    assert refusal_text(error) == "invalid_params (submission_params: unexpected keys: ['one_shot'])"


def test_refusal_text_renders_the_canonical_gates_fields():
    # The gate refuses before any handler runs and sends key paths, not a field/detail pair.
    error = rpc_error({'code': 4001, 'message': 'invalid_params',
                       'data': {'reason': 'invalid_params', 'fields': ['one_shot']}})
    assert refusal_text(error) == "invalid_params (params: unknown or missing keys: one_shot)"


def test_a_refusal_without_content_reads_exactly_as_before():
    assert refusal_text(GatewayRPCError('permission_denied', 4001, {'reason': 'permission_denied'})) == \
        'permission_denied'


def test_a_one_shot_prints_the_named_refusal(monkeypatch, capsys):
    from hermes_cli import gateway_chat

    def _refused(coro):
        coro.close()
        raise GatewayRPCError('invalid_params', 4001, {
            'reason': 'invalid_params', 'field': 'submission_params',
            'detail': "unexpected keys: ['one_shot']"})

    monkeypatch.setattr(gateway_chat, '_register_terminal_process', lambda: None)
    monkeypatch.setattr('hermes_cli.gateway_chat_startup.ensure_launch_provider', lambda args: True)
    monkeypatch.setattr(asyncio, 'run', _refused)
    code = gateway_chat.launch_from_args(argparse.Namespace(query='hi', output_format='text'))
    assert code == 1
    assert ("Error: invalid_params (submission_params: unexpected keys: ['one_shot'])"
            in capsys.readouterr().err)
