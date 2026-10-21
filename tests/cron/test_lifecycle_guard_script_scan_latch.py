"""A repeated guard scan of the same script costs stats, not a re-tokenization (PRX-5758).

The referenced-script walk is pure work over file bytes: its verdict for a script is a property of
that script and of everything the script references, never of the call. It used to be re-derived on
every terminal call — a 221-line, 9.7 KB routine script (``disk-reclaimer.sh``) measured 1.1-2.5 s
across 7 recursion levels and ~3000 tokenizations — and a bounded guard worker could not finish that
inside the fixed 45 s ceiling once the gateway was in its high-load GIL convoy, so the command was
refused with a misleading "process-identity probe wedged" error instead of run.

These tests pin the behaviour contract, not a latency snapshot: N calls do one walk, and any file
the answer depended on changing forces the next call to walk again.
"""

from __future__ import annotations

import shlex

import pytest

from cron import lifecycle_guard

guard = lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script

LIFECYCLE = "hermes gateway restart"


def _clear_memos():
    """Both memos live for the process lifetime, so each test starts from a cold gateway."""
    cache = getattr(lifecycle_guard, "_script_verdict_cache", None)
    if cache is not None:
        cache.clear()
    tokens = getattr(lifecycle_guard, "_cached_shlex_tokens", None)
    if tokens is not None:
        tokens.cache_clear()


@pytest.fixture(autouse=True)
def clear_latches():
    _clear_memos()
    yield
    _clear_memos()


@pytest.fixture
def script_reads(monkeypatch):
    """Every local referenced-script read the walk performs, in order."""
    reads = []
    read = lifecycle_guard._read_referenced_script

    def counting(path, *, max_bytes=None):
        reads.append(path)
        return read(path, max_bytes=max_bytes)

    monkeypatch.setattr(lifecycle_guard, "_read_referenced_script", counting)
    return reads


def _write(path, *lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _bash(path):
    return f"bash {shlex.quote(str(path))}"


def test_repeat_scans_of_one_script_read_it_once(tmp_path, script_reads):
    """N terminal calls must not become N tokenizations of the same 9.7 KB script."""
    script = _write(tmp_path / "routine.sh", "#!/bin/bash", *(f"echo filler-{i}" for i in range(40)))

    verdicts = [guard(_bash(script)) for _ in range(5)]

    assert verdicts == [False] * 5, "the latch must not change the verdict"
    assert len(script_reads) == 1, f"the script was read {len(script_reads)} times, expected 1"


def test_the_latch_is_keyed_on_the_script_not_on_the_command(tmp_path, script_reads):
    """Different argv for the same script is the same question about the same bytes."""
    script = _write(tmp_path / "routine.sh", "#!/bin/bash", "echo ok")

    assert guard(f"{_bash(script)} --dry-run") is False
    assert guard(f"{_bash(script)} --help") is False

    assert len(script_reads) == 1


def test_an_edited_script_is_walked_again(tmp_path, script_reads):
    script = _write(tmp_path / "routine.sh", "#!/bin/bash", "echo ok")
    assert guard(_bash(script)) is False

    _write(script, "#!/bin/bash", LIFECYCLE)

    assert guard(_bash(script)) is True, "a rewritten script must be scanned, not answered from the latch"
    assert len(script_reads) == 2


def test_a_changed_nested_script_invalidates_the_script_that_sources_it(tmp_path, script_reads):
    """The verdict covers the whole subtree, so its dependencies must all still hold."""
    child = _write(tmp_path / "child.sh", "#!/bin/bash", "echo clean")
    parent = _write(tmp_path / "parent.sh", "#!/bin/bash", f"source {shlex.quote(str(child))}", "echo done")

    assert guard(_bash(parent)) is False

    _write(child, "#!/bin/bash", LIFECYCLE)

    assert guard(_bash(parent)) is True, (
        "the parent's latched verdict depended on the child's bytes, and the child now blocks"
    )


def test_a_grandchild_appearing_invites_a_fresh_walk(tmp_path, script_reads):
    """A verdict that found nothing at a referenced path assumes it stays that way."""
    child = tmp_path / "child.sh"
    parent = _write(tmp_path / "parent.sh", "#!/bin/bash", f"bash {shlex.quote(str(child))}")

    assert guard(_bash(parent)) is False, "a referenced script that does not exist yet is nothing to scan"

    _write(child, "#!/bin/bash", LIFECYCLE)

    assert guard(_bash(parent)) is True, (
        "the parent was latched clean while the child was missing; its appearance must force a walk"
    )


def test_a_changed_grandchild_invalidates_the_whole_chain(tmp_path, script_reads):
    """A latched subtree hands its witnesses up, not just the one stamp of the script it entered."""
    grandchild = _write(tmp_path / "grandchild.sh", "#!/bin/bash", "echo clean")
    child = _write(tmp_path / "child.sh", "#!/bin/bash", f"bash {shlex.quote(str(grandchild))}")
    parent = _write(tmp_path / "parent.sh", "#!/bin/bash", f"bash {shlex.quote(str(child))}")

    assert guard(_bash(parent)) is False

    _write(grandchild, "#!/bin/bash", LIFECYCLE)

    assert guard(_bash(parent)) is True


def test_the_current_profile_is_part_of_the_verdict(tmp_path, script_reads, monkeypatch):
    """`-p <profile>` only blocks when <profile> runs the guard, and one gateway serves many."""
    script = _write(tmp_path / "routine.sh", "#!/bin/bash", "hermes -p alpha gateway restart")

    monkeypatch.setattr(lifecycle_guard, "_current_profile_name", lambda: "beta")
    assert guard(_bash(script)) is False, "a sibling profile's lifecycle command is a fleet operation"

    monkeypatch.setattr(lifecycle_guard, "_current_profile_name", lambda: "alpha")
    assert guard(_bash(script)) is True, (
        "the same script now targets the profile running the guard; a latched verdict must not survive"
    )


def test_a_blocked_script_is_never_latched(tmp_path, script_reads):
    """Only clean verdicts are remembered: a refusal or a block is re-derived and reported."""
    script = _write(tmp_path / "blocked.sh", "#!/bin/bash", LIFECYCLE)

    assert [guard(_bash(script)) for _ in range(3)] == [True] * 3
    assert len(script_reads) == 3, "a blocking verdict must be recomputed every time"


def test_a_latched_script_spends_no_walk_budget(tmp_path, monkeypatch):
    """A quota bounds the scripts a walk must READ; one it can answer from the latch costs nothing."""
    first = _write(tmp_path / "first.sh", "#!/bin/bash", "echo ok")
    second = _write(tmp_path / "second.sh", "#!/bin/bash", "echo ok")
    pair = f"{_bash(first)} && {_bash(second)}"

    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_PATHS", 1)
    assert guard(pair) is True, "two reads under a one-path quota must fail closed"

    assert guard(_bash(first)) is False
    assert guard(pair) is False, "the latched half costs no path, so the one remaining read fits"


def test_one_text_is_tokenized_once_per_distinct_line(tmp_path, monkeypatch):
    """The walk tokenizes the same text for several passes; each line is only read once."""
    lexers = []
    real_lexer = lifecycle_guard.shlex.shlex

    def counting_lexer(*args, **kwargs):
        lexers.append(args)
        return real_lexer(*args, **kwargs)

    monkeypatch.setattr(lifecycle_guard.shlex, "shlex", counting_lexer)
    command = "\n".join(f"echo line-{i} | cat" for i in range(20))

    assert guard(command) is False
    first_pass = len(lexers)
    assert first_pass > 0

    assert guard(command) is False
    assert len(lexers) == first_pass, (
        f"{len(lexers) - first_pass} lines were tokenized again on a repeat scan"
    )
