"""A gateway and a client that disagree about the wire contract must say so, once, in seconds.

SOU-3316: ``merge(PRX-5850)`` added ``one_shot`` to the submit allow-list while the serving gateway
had booted 4h14m earlier with the old set. Every ``hermes -z`` child from the new tree was refused
with a bare ``invalid_params`` — indistinguishable from a malformed request — for 3h24m and 71 runs.

These are the behaviour contracts that close it, one per direction:

* new client vs old gateway → ``contract_version_mismatch`` naming field and BOTH revisions, refused
  before any admission (so no model call can happen);
* old client (declares nothing) vs new gateway → unchanged: accepted, and an unknown key stays a
  closed-set ``invalid_params``;
* deploy shipping a contract change without restarting the gateway → the drain-marker path refuses to
  declare the deploy complete.
"""

from __future__ import annotations

import json

import pytest

from gateway import contract_revision
from gateway.drain_control import clear_drain_request, drain_request_path, write_drain_request
from gateway.lifecycle_ledger import contract_revision_verdict, get_lifecycle_sentinel_path
from gateway.session_contract import SessionRef
from gateway.session_controls import (
    AuthorityConnection, _canonical_param_refusal, _submit_param_refusal,
)
from hermes_state_runtime import RuntimeStoreError

REF = SessionRef('default', 'sess-1')

OLD_CONTRACT = 'old0ld0ld0ld0'
NEW_CONTRACT = 'new9new9new9'


@pytest.fixture
def serving(monkeypatch):
    """Pin what the SERVING process booted with (the boot snapshot), as a long-lived gateway holds it."""
    def _pin(revision):
        monkeypatch.setattr(contract_revision, '_boot_revision', revision, raising=False)
        monkeypatch.setattr(contract_revision, '_boot_recorded', True, raising=False)
    _pin(OLD_CONTRACT)
    return _pin


@pytest.fixture
def premerge_contract(monkeypatch):
    """The serving process's allow-list from BEFORE the merge that added ``one_shot``.

    Exactly the SOU-3316 process: it imported its set 4h14m before the merge, so ``one_shot`` is
    not in it, and the client from the new tree sends it.
    """
    from gateway import session_controls

    monkeypatch.setattr(session_controls, 'SUBMIT_PARAMS',
                        frozenset(session_controls.SUBMIT_PARAMS - {'one_shot'}))


def _connection():
    """A submit-capable connection without an authority: every case here refuses before admission."""
    connection = object.__new__(AuthorityConnection)
    connection.subscriptions = {REF.session_id: 'sub-1'}
    return connection


def _submit_params(**extra):
    return {'session_id': REF.session_id, 'submission_id': 'sub-1', 'text': 'hi', **extra}


# -- direction 1: new client vs old gateway -------------------------------------------------------


def test_new_client_against_old_gateway_is_told_it_is_skewed_not_malformed(serving, premerge_contract):
    """The SOU-3316 refusal, exactly: the field, the client's revision and the gateway's."""
    with pytest.raises(RuntimeStoreError) as caught:
        _connection().submission(REF, _submit_params(one_shot=True, finite=True,
                                                     contract_revision=NEW_CONTRACT))

    refusal = caught.value
    assert refusal.reason == 'contract_version_mismatch'
    assert refusal.field == 'one_shot'
    assert refusal.client_revision == NEW_CONTRACT
    assert refusal.gateway_revision == OLD_CONTRACT
    assert 'one_shot' in (refusal.detail or '') and NEW_CONTRACT in refusal.detail and OLD_CONTRACT in refusal.detail


def test_the_skew_is_refused_before_anything_is_admitted(serving, premerge_contract):
    """No admission, no receipt: the run costs one round trip, never a model call (zero tokens)."""
    connection = _connection()
    with pytest.raises(RuntimeStoreError):
        connection.submission(REF, _submit_params(one_shot=True, finite=True,
                                                 contract_revision=NEW_CONTRACT))
    # submission() is the only producer of an admission on this path, and it refused on the param
    # set — before ``admit_finite`` (which is the first code to touch the authority) ever ran.
    assert not hasattr(connection, 'authority'), 'the refusal must not need (or touch) the authority'


def test_the_canonical_dispatcher_also_names_the_skew(serving):
    """The declared-param gate runs BEFORE the handler, so it carries the same discriminator."""
    refusal = _canonical_param_refusal(
        _submit_params(one_shot=True, contract_revision=NEW_CONTRACT), ['one_shot'])
    assert refusal.reason == 'contract_version_mismatch'
    assert (refusal.client_revision, refusal.gateway_revision) == (NEW_CONTRACT, OLD_CONTRACT)


# -- direction 2: old client vs new gateway --------------------------------------------------------


def test_a_client_that_declares_nothing_is_accepted_unchanged(serving):
    """The declaration is optional: the old wire, byte for byte, still admits."""
    submission = _connection().submission(REF, _submit_params())
    assert submission.request_id == 'sub-1'
    assert submission.payload['text'] == 'hi'


def test_a_declaration_matching_the_gateway_does_not_by_itself_change_anything(serving):
    """Declaring the same revision buys nothing: a genuinely unknown key stays ``invalid_params``."""
    refusal = _submit_param_refusal(_submit_params(contract_revision=OLD_CONTRACT), {'bogus'})
    assert refusal.reason == 'invalid_params'
    assert refusal.field == 'submission_params'
    assert 'bogus' in (refusal.detail or '')


# -- direction 3: undeclared client + unknown key (PRA-6722 must not regress) ----------------------


def test_undeclared_client_with_an_unknown_key_keeps_the_named_closed_set_refusal(serving):
    refusal = _submit_param_refusal(_submit_params(one_shot=True), {'one_shot'})
    assert refusal.reason == 'invalid_params'
    assert 'one_shot' in (refusal.detail or '')
    assert refusal.client_revision is None and refusal.gateway_revision is None


def test_a_malformed_declaration_is_a_client_bug_not_a_skew(serving):
    with pytest.raises(RuntimeStoreError) as caught:
        _connection().submission(REF, _submit_params(contract_revision=7))
    assert caught.value.reason == 'invalid_params'
    assert caught.value.field == 'contract_revision'


def test_an_unreadable_contract_document_degrades_to_todays_behaviour(monkeypatch):
    """No contract document (an install without ``apps/``): nothing is declared, nothing is named."""
    monkeypatch.setattr(contract_revision, '_boot_recorded', False, raising=False)
    monkeypatch.setattr(contract_revision, '_boot_revision', None, raising=False)
    monkeypatch.setattr(contract_revision, 'revision', lambda root=None: None)
    assert contract_revision.declaration() == {}
    assert _submit_param_refusal(_submit_params(one_shot=True), {'one_shot'}).reason == 'invalid_params'


# -- direction 4: the deploy check ---------------------------------------------------------------


def _claim_running(home, revision):
    path = get_lifecycle_sentinel_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'phase': 'running', 'pid': 4242, 'contract_revision': revision}))


def test_a_contract_change_without_a_restart_fails_the_deploy_check(tmp_path, monkeypatch):
    """The SOU-3316 deploy, graded: tree moved, process still serving the old contract."""
    _claim_running(tmp_path, OLD_CONTRACT)
    monkeypatch.setattr(contract_revision, 'revision', lambda root=None: NEW_CONTRACT)

    verdict, detail = contract_revision_verdict(tmp_path)
    assert verdict == 'stale'
    assert OLD_CONTRACT in detail and NEW_CONTRACT in detail


def test_the_drain_marker_survives_a_skewed_deploy_and_the_fleet_stays_quiesced(tmp_path, monkeypatch):
    """Clearing the drain is the deploy's completion claim; a skew refuses to re-open the fleet."""
    _claim_running(tmp_path, OLD_CONTRACT)
    monkeypatch.setattr(contract_revision, 'revision', lambda root=None: NEW_CONTRACT)
    write_drain_request(principal='dashboard', home=tmp_path)

    assert clear_drain_request(home=tmp_path) is False
    assert drain_request_path(tmp_path).exists(), 'the marker is the fleet staying quiesced'

    # The restart lands: the new process stamps the tree's revision, and the same call now clears.
    _claim_running(tmp_path, NEW_CONTRACT)
    assert clear_drain_request(home=tmp_path) is True
    assert not drain_request_path(tmp_path).exists()


def test_an_in_sync_deploy_is_never_blocked(tmp_path, monkeypatch):
    _claim_running(tmp_path, NEW_CONTRACT)
    monkeypatch.setattr(contract_revision, 'revision', lambda root=None: NEW_CONTRACT)
    write_drain_request(principal='dashboard', home=tmp_path)

    assert contract_revision_verdict(tmp_path)[0] == 'in_sync'
    assert clear_drain_request(home=tmp_path) is True


def test_an_unknown_serving_revision_never_invents_a_block(tmp_path, monkeypatch):
    """No sentinel / no stamp: the check reports unknown and the drain still clears."""
    monkeypatch.setattr(contract_revision, 'revision', lambda root=None: NEW_CONTRACT)
    write_drain_request(principal='dashboard', home=tmp_path)

    assert contract_revision_verdict(tmp_path)[0] == 'unknown'
    assert clear_drain_request(home=tmp_path) is True
