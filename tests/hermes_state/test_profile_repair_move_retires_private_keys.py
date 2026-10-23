"""A profile-repair move takes everything the moved row's id owns with it (PRX-6097).

``delete_moved_session`` removes the source row after the target store confirmed the copy. The
private keys written beside a row — the local conversation's creation receipt, its routing entry,
the api-settings and viewer-close keys — are not cascaded by that DELETE. Left behind, the receipt
names a row nothing recreates in this store: every gateway boot refused it again, forever (34 such
ids in one live profile, across a code update).
"""

from __future__ import annotations

import json
import time


def _private_keys(db, session_id: str) -> None:
    # A creation receipt names its current physical target in ``entry``; that is the id its
    # retirement matches on.
    db.set_meta('gateway.local_policy.v1:' + session_id,
                json.dumps({'session_id': session_id, 'entry': {'session_id': session_id}}))
    db.set_meta('gateway.api.settings.v1.' + session_id, '{}')
    db.set_meta('gateway.viewer_close.v1:' + session_id, '{}')
    db._write_sql(
        "INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,?)",
        ('agent:cli:local:' + session_id, json.dumps({'session_id': session_id}), time.time()))


def _leftovers(db, session_id: str) -> list[str]:
    """Everything still holding *session_id* except the retirement fence itself, which is the
    marker that says the id left this store (``gateway.retired_session.v1.<id>``)."""
    with db._read_ctx() as conn:
        keys = [k for (k,) in conn.execute('SELECT key FROM state_meta WHERE key LIKE ?',
                                           ('%' + session_id + '%',))
                if not k.startswith('gateway.retired_session.v1.')]
        keys += [k for (k,) in conn.execute('SELECT session_key FROM gateway_routing WHERE entry_json LIKE ?',
                                            ('%' + session_id + '%',))]
    return keys


def test_a_moved_session_leaves_no_key_naming_its_id(tmp_path):
    from hermes_state import SessionDB

    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('moved', source='tui', session_key='agent:cli:local:moved')
        db.append_message('moved', 'user', 'hi')
        _private_keys(db, 'moved')
        db.create_session('kept', source='tui', session_key='agent:cli:local:kept')
        _private_keys(db, 'kept')

        assert db.delete_moved_session('moved') is True

        assert db.get_session('moved') is None
        assert _leftovers(db, 'moved') == [], 'the move left keys naming the row it removed'
        # The move is one row's, and its fence is a sweep's: the neighbour keeps its keys and the
        # id may be created here again once the fence ages out.
        assert _leftovers(db, 'kept') != []
