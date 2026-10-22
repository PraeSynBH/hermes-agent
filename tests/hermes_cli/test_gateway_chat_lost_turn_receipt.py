"""PRX-5855: a dropped socket mid-turn is answered from the owner's receipt, not booked as a lost run.

The `hermes_local` Paperclip adapter records a run as ``adapter_failed`` with "Gateway disconnected;
turn outcome is unknown" whenever this CLI exits on the transport error — even though the owner kept
executing and the admission settled with its result. These tests pin the contract on the one-shot
view: ask the owner for ``prompt.receipt`` (include_result) before failing, poll while it is
unsettled, and report the owner's committed answer and exit code when it settled.
"""
import asyncio

import pytest


class _LostPeer:
    """The first socket: the turn is admitted, then the transport drops under it."""

    def __init__(self, admission="mine"):
        self.events, self.admission = asyncio.Queue(), admission
        self.events.put_nowait(self._lost())

    @staticmethod
    def _lost():
        from hermes_cli.gateway_client import GatewayClientError
        from hermes_cli.gateway_connection import DISCONNECTED
        return GatewayClientError(DISCONNECTED)

    async def rpc(self, method, **params):
        assert method == "prompt.submit", method
        return {"admission_id": self.admission}


class _ReceiptPeer:
    """The re-dialled socket: answers ``prompt.receipt`` with each scripted receipt in turn."""

    def __init__(self, receipts):
        self.receipts, self.reads = list(receipts), []

    async def rpc(self, method, **params):
        assert method == "prompt.receipt", method
        assert params == {"session_id": "stored", "admission_id": "mine", "include_result": True}
        self.reads.append(dict(params))
        return self.receipts[min(len(self.reads), len(self.receipts)) - 1]


def _connect_to(monkeypatch, peer):
    """Point the view's re-dial at *peer* (a fresh ``connect_gateway`` context manager)."""
    import contextlib

    @contextlib.asynccontextmanager
    async def connect(*args, **kwargs):
        yield peer

    monkeypatch.setattr("hermes_cli.gateway_client.connect_gateway", connect)
    monkeypatch.setattr("hermes_cli.gateway_chat_view.LOST_TURN_POLL_INTERVAL_S", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_json", [False, True])
async def test_a_lost_socket_reports_the_owners_settled_turn_as_succeeded(monkeypatch, capsys, stream_json):
    """The adapter-side contract: exit 0 with the owner's ``final_response``, run recorded succeeded.

    ``--format stream-json`` (what the Paperclip ``hermes_local`` adapter consumes) must close with a
    ``result`` record carrying ``exit_code`` 0 — that record, not the socket, is the run's outcome.
    """
    import json
    from hermes_cli.gateway_chat_view import GatewayChatView
    from hermes_cli.stream_json import StreamJsonEmitter

    peer = _ReceiptPeer([
        {"status": "started"},  # still running when the socket died: keep waiting
        {"status": "terminal", "outcome": "completed",
         "result": {"completed": True, "final_response": "the work finished"}},
    ])
    _connect_to(monkeypatch, peer)

    emitter = StreamJsonEmitter(model="m", session_id="stored") if stream_json else None
    view = GatewayChatView(_LostPeer(), {"stored_session_id": "stored"}, quiet=True, emitter=emitter)
    view.finite = True
    assert await asyncio.wait_for(view.run("query", oneshot=True), 5) == 0
    out = capsys.readouterr().out
    assert "the work finished" in out and "outcome is unknown" not in out
    if stream_json:
        result = [json.loads(line) for line in out.splitlines() if line.strip()][-1]
        assert result["type"] == "result" and result["exit_code"] == 0, result
        assert result["text"] == "the work finished", result
    assert len(peer.reads) == 2  # polled once while started, then read the settlement


@pytest.mark.asyncio
async def test_a_turn_the_owner_lost_fails_at_once_instead_of_polling(monkeypatch):
    """A rediscovered ``unknown`` must never consume the run: one read, then the run fails."""
    from hermes_cli.gateway_chat_view import GatewayChatView
    from hermes_cli.gateway_client import GatewayClientError

    peer = _ReceiptPeer([{"status": "unknown"}])
    _connect_to(monkeypatch, peer)
    monkeypatch.setattr("hermes_cli.gateway_chat_view.LOST_TURN_WAIT_BUDGET_S", 3_600)

    view = GatewayChatView(_LostPeer(), {"stored_session_id": "stored"}, quiet=True)
    view.finite = True
    with pytest.raises(GatewayClientError, match="outcome is unknown"):
        await asyncio.wait_for(view.run("query", oneshot=True), 5)
    assert len(peer.reads) == 1


@pytest.mark.asyncio
async def test_an_unsettled_receipt_is_bounded_by_the_wait_budget(monkeypatch):
    """A receipt that never settles costs the budget, then the run fails — never the whole heartbeat."""
    from hermes_cli.gateway_chat_view import GatewayChatView
    from hermes_cli.gateway_client import GatewayClientError

    peer = _ReceiptPeer([{"status": "started"}])
    _connect_to(monkeypatch, peer)
    monkeypatch.setattr("hermes_cli.gateway_chat_view.LOST_TURN_WAIT_BUDGET_S", 0)

    view = GatewayChatView(_LostPeer(), {"stored_session_id": "stored"}, quiet=True)
    view.finite = True
    with pytest.raises(GatewayClientError, match="outcome is unknown"):
        await asyncio.wait_for(view.run("query", oneshot=True), 5)


class _NoAdmissionPeer:
    """The socket died while ``prompt.submit`` was in flight: no admission id ever came back."""

    def __init__(self):
        self.events = asyncio.Queue()

    async def rpc(self, method, **params):
        assert method == "prompt.submit", method
        raise _LostPeer._lost()


@pytest.mark.asyncio
async def test_a_disconnect_before_admission_is_not_a_lost_turn(monkeypatch, capsys):
    """PRX-5964: with no admission id there is nothing a receipt could be read for.

    The run's own copy and its retryable exit code say "never admitted"; the mid-turn "outcome is
    unknown" must not appear, and no receipt is read, so the class counter stops booking a no-op
    with the runs that lost a whole turn.
    """
    from hermes_cli.gateway_chat_view import GatewayChatView
    from hermes_cli.gateway_client import GatewayTurnNotAdmitted
    from hermes_cli.turn_exit import TURN_NOT_ADMITTED_EXIT_CODE

    receipt_peer = _ReceiptPeer([])  # any read here is the bug this test pins
    _connect_to(monkeypatch, receipt_peer)

    view = GatewayChatView(_NoAdmissionPeer(), {"stored_session_id": "stored"}, quiet=True)
    with pytest.raises(GatewayTurnNotAdmitted) as caught:
        await asyncio.wait_for(view.run("query", oneshot=True), 5)
    text = str(caught.value)
    assert "outcome is unknown" not in text and "Resume the printed session ID" not in text
    assert "before the turn was admitted" in text and "Retry the run" in text
    assert caught.value.exit_code == TURN_NOT_ADMITTED_EXIT_CODE
    assert receipt_peer.reads == []
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_a_refused_submit_keeps_its_own_verdict(monkeypatch):
    """Only a transport loss is a never-admitted disconnect; an owner's refusal is not rewritten."""
    from hermes_cli.gateway_chat_view import GatewayChatView
    from hermes_cli.gateway_client import GatewayRPCError

    class Refusing:
        events = asyncio.Queue()

        async def rpc(self, method, **params):
            raise GatewayRPCError("queue_full")

    view = GatewayChatView(Refusing(), {"stored_session_id": "stored"}, quiet=True)
    with pytest.raises(GatewayRPCError, match="queue_full"):
        await asyncio.wait_for(view.run("query", oneshot=True), 5)


def test_launch_reports_the_never_admitted_exit_code_and_copy(monkeypatch, capsys):
    """The exit contract the adapter reads: EX_TEMPFAIL, and a copy with no "outcome is unknown"."""
    from hermes_cli import gateway_chat, gateway_chat_startup
    from hermes_cli.gateway_client import GatewayTurnNotAdmitted
    from hermes_cli.turn_exit import TURN_NOT_ADMITTED_EXIT_CODE

    async def run(args, emitter=None):
        raise GatewayTurnNotAdmitted()

    monkeypatch.setattr(gateway_chat_startup, "ensure_launch_provider", lambda args: True)
    monkeypatch.setattr(gateway_chat, "run_gateway_chat", run)
    assert gateway_chat.launch_from_kwargs({"q": "literal"}) == TURN_NOT_ADMITTED_EXIT_CODE
    err = capsys.readouterr().err
    assert "gateway_turn_not_admitted" in err and "turn outcome is unknown" not in err


@pytest.mark.asyncio
async def test_an_interactive_viewer_never_reads_a_receipt_for_a_lost_socket(monkeypatch):
    """Only a finite run has a turn whose outcome the caller needs; interactivity re-attaches."""
    from hermes_cli.gateway_chat_view import GatewayChatView

    peer = _ReceiptPeer([{"status": "terminal", "outcome": "completed",
                          "result": {"completed": True, "final_response": "x"}}])
    _connect_to(monkeypatch, peer)

    view = GatewayChatView(_LostPeer(), {"stored_session_id": "stored"}, quiet=True)
    assert await view._recover_lost_turn("mine", _LostPeer._lost()) is False
    assert peer.reads == []
