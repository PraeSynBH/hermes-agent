"""An ``unknown`` turn nobody is waiting on settles as interrupted (#134889 GW-10, #139532, PRX-5850).

A turn lost across an owner restart is fenced ``unknown`` and pauses its session's FIFO, because
only the person who sent it knows whether to resend it. Two kinds of row have no such person:

* an automation delivery: a cron run (``CRON_PRINCIPAL_ID``), a process completion or heartbeat
  (``automation:<route>``, ``local_automation_v1``, a native envelope's ``automation`` block), a
  webhook delivery (``webhook_delivery``) or a Kanban card attempt (``kanban_card_v1``, set by the
  dispatcher);
* a one-shot turn the submitter declared waiter-less (``one_shot: true``): ``hermes chat
  --resume <id> -q`` / ``hermes -z`` stamps it (``session_finite.admit_finite``) because its process
  exits when the turn settles, so once the restarted owner has fenced the admission there is no live
  process to consult — the caller re-runs its command instead. Pausing on one is how a Paperclip
  agent's persisted session died for good: every later wake refused with "Unknown execution blocks
  this session" and only a human ``hermes sessions discard`` cleared it (58 failed runs, 11 agents,
  20 sessions in 12 h, PRX-5850). ``finite`` alone is NOT the signal — a Kanban card attempt is
  finite too and must keep pausing until its dispatcher marks it (``kanban_card_v1``).

Pausing on either stalls every message queued behind it until an operator notices (a Bot Chat sat
ten hours behind one). Such a row settles through the same fenced resolution a person's Discard
uses (``resolve_unknown_session_input``): terminal ``interrupted``, never requeued or replayed,
with a Hermes transcript notice committed in the same transaction. No second recovery store.
"""
from __future__ import annotations

import logging

from hermes_state_keys import CRON_PRINCIPAL_ID

logger = logging.getLogger(__name__)


def is_unattended(row) -> bool:
    """True for an admission no person submitted (so nobody can be asked to resend it)."""
    principal = row.get('principal_id') or ''
    if principal == CRON_PRINCIPAL_ID or principal.startswith('automation:'):
        return True
    payload = row.get('payload') or {}
    native = payload.get('native_text_v1') or {}
    # A webhook delivery: the provider already got its 202 and nobody can be asked to resend it.
    # A Kanban card attempt (#134889 KB-8): the dispatcher's marker, committed only by
    # ``SessionAuthority.submit(_kanban_card=...)``; a card row main minted without it still pauses.
    return ('local_automation_v1' in payload or 'kanban_card_v1' in payload
            or 'automation' in native or 'webhook_delivery' in native)


def settles_without_a_waiter(row) -> bool:
    """True for an ``unknown`` row no live caller is waiting on: an automation delivery, or a
    one-shot that declared itself waiter-less (``one_shot: true``, stamped by ``admit_finite`` for
    ``hermes chat -q`` / ``hermes -z`` so its ``{finite, text}`` payload carries the declaration; a
    bare ``finite`` row is a Kanban attempt or a pre-PRX-5850 row and keeps today's pause)."""
    return is_unattended(row) or (row.get('payload') or {}).get('one_shot') is True


def _ownerless_unknown(rows):
    return [row for row in rows if row['status'] == 'unknown' and settles_without_a_waiter(row)]


def _notice_for(row):
    from agent.turn_failure_copy import AUTOMATION_INTERRUPTED_NOTICE, ONE_SHOT_INTERRUPTED_NOTICE
    return AUTOMATION_INTERRUPTED_NOTICE if is_unattended(row) else ONE_SHOT_INTERRUPTED_NOTICE


def settle_recovered_unattended(db, epoch) -> int:
    """Boot recovery, right after ``recover_session_inputs`` fenced lost turns ``unknown`` and before
    any FIFO is scheduled: settle every row no one is waiting on. Synchronous; runs off the owner
    loop. Returns the count, and logs it once when any settled, so the class is visible in
    ``gateway.log`` instead of only as a stream of CLI refusals."""
    from gateway.session_ingress_media import release_admission_media
    from gateway.session_results import close_discarded_turn
    from hermes_state_runtime import RuntimeStoreError, _row, resolve_unknown_session_input
    with db._read_ctx() as conn:
        rows = [_row(r) for r in conn.execute(
            "SELECT * FROM session_admissions WHERE status='unknown' ORDER BY seq").fetchall()]
    settled = 0
    for row in _ownerless_unknown(rows):
        notice = _notice_for(row)
        try:
            resolve_unknown_session_input(
                db, epoch=epoch, admission_id=row['admission_id'], generation=row['generation'],
                _terminal_write=lambda conn, lost: close_discarded_turn(db, conn, lost, notice))
        except RuntimeStoreError as exc:
            logger.warning('Lost admission %s left unknown: %s', row['admission_id'], exc.reason)
            continue
        release_admission_media(db, row['admission_id'])
        settled += 1
        logger.info('Lost admission %s on session %s settled interrupted (no waiter, not replayed)',
                    row['admission_id'], row['target_session_id'])
    if settled:
        logger.info('Settled %d lost turn(s) no one is waiting on at boot (not replayed)', settled)
    return settled


async def settle_unattended_unknown(authority, ref) -> bool:
    """The drain found an ``unknown`` head (fenced in this owner, e.g. a lost managed worker):
    settle the rows no one is waiting on through the owner's resolution. True when any settled, so
    the drain re-reads its FIFO instead of pausing; a person's ``unknown`` row still pauses it."""
    import asyncio

    from hermes_state_runtime import RuntimeStoreError, list_session_admissions
    rows = await asyncio.to_thread(list_session_admissions, authority.db, session_id=ref.session_id)
    settled = False
    for row in _ownerless_unknown(rows):
        try:
            await authority.resolve_unknown_as_owner(ref, row['admission_id'], row['generation'],
                                                     notice=_notice_for(row))
        except RuntimeStoreError as exc:
            logger.warning('Lost admission %s left unknown: %s', row['admission_id'], exc.reason)
            continue
        settled = True
    return settled
