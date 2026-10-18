"""Parametrized regression matrix over the launchd guard's spelling corpus (SOU-1934).

The corpus (``guard_spelling_corpus.ROWS``) is the data; this file is only the pytest shape of it,
so advancing the guard's coverage is an ordinary data edit and never a new test.

Every row is fed through all three surfaces the production gate uses — the registration pass, the
removal pass, and the whole ``scan_gateway_lifecycle`` walk. A row whose spelling is only reachable
through the walk (a referenced script that hides the program word in a variable) says so in
``must_not`` and is therefore a distinct row: it fails if the walk stops reading scripts, and it
fails if the job-set passes start "catching" a command that never mentions launchctl.

Why a corpus: the guard has been patched six times for launchd bypasses (#62891, #77083, #80269,
#113667, SOU-1850, SOU-1900) and every patch started from one hand-written example. SOU-1900 turned
the passes into a resolver, so the next change is more likely to break a neighbouring spelling.
"""

from __future__ import annotations

import pytest

from cron import lifecycle_guard
from tests.cron import guard_spelling_corpus

ALLOWED = guard_spelling_corpus.ALLOWED
ROWS = guard_spelling_corpus.ROWS


@pytest.mark.parametrize("row", ROWS, ids=[row.id for row in ROWS])
def test_spelling_row_matches_the_fence(row, tmp_path):
    result = guard_spelling_corpus.evaluate(row, lifecycle_guard, tmp_path)

    assert result.passed, f"{row.id} ({row.reason}): {result.why}"


def test_corpus_covers_both_verdicts_and_a_walk_only_row():
    """A corpus that lost its fence rows, or its walk-only rows, no longer tests what it claims."""
    blocked, allowed, residual = guard_spelling_corpus.counts(ROWS)

    assert blocked > 0
    assert allowed > 0
    assert residual > 0
    assert any(row.walk_only for row in ROWS)
    assert any(row.residual for row in ROWS)


def test_residual_rows_name_the_boundary_they_document():
    """`residual` is a claim about SOU-1932, not a place to park an expected failure."""
    for row in ROWS:
        if row.residual:
            assert row.expected == ALLOWED
            assert row.reason.startswith("residual — see SOU-1932"), row.id


def test_row_ids_are_unique():
    ids = [row.id for row in ROWS]

    assert len(ids) == len(set(ids))
