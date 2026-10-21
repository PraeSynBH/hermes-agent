"""PRX-5850: a finite one-shot submission declares that it has no waiter.

The gateway can only settle a lost turn on its own when its submitter says nobody is waiting on it
(``session_finite.admit_finite``): ``unattended`` for ``-z``, and now ``one_shot`` for the finite
``chat -q`` path too. Without the declaration a turn fenced ``unknown`` across an owner restart
paused its session forever, and every later wake of a persisted automation session refused with
"Unknown execution blocks this session" instead of running.
"""
import asyncio

import pytest

from hermes_cli.gateway_chat_view import GatewayChatView


class Peer:
    def __init__(self):
        self.events = asyncio.Queue()
        self.calls = []

    async def rpc(self, method, **params):
        self.calls.append((method, params))
        return {'admission_id': 'mine'}


@pytest.mark.asyncio
@pytest.mark.parametrize('unattended, expected', [
    (False, {'finite': True, 'one_shot': True}),                       # `chat -q` / `-Q`: one-shot only
    (True, {'finite': True, 'one_shot': True, 'unattended': True}),    # `-z`: one-shot + auto-approve
])
async def test_a_finite_one_shot_submits_the_no_waiter_declaration(unattended, expected):
    peer = Peer()
    view = GatewayChatView(peer, {'stored_session_id': 's'}, quiet=True)
    view.finite, view.unattended = True, unattended
    await view.submit('WAKE')
    method, params = peer.calls[0]
    assert method == 'prompt.submit'
    assert {k: params[k] for k in ('finite', 'one_shot', 'unattended') if k in params} == expected
