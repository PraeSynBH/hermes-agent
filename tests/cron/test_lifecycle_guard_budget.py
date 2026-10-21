"""Whole-walk work budget for the gateway lifecycle guard (#78398).

The per-file byte cap and recursion depth bound one read, not the walk. These
tests pin the shared budget that bounds the whole referenced-script walk and
is charged *before* any text reaches ``shlex``.

Budget constants are monkeypatched to tiny values so the tests are fast and
deterministic; ``_LifecycleScanBudget`` reads them at construction time.
"""

from __future__ import annotations

import pytest
import shlex
from pathlib import Path

from cron import lifecycle_guard

guard = lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script


def _explode(*_args, **_kwargs):
    raise AssertionError("over-budget text reached shlex")


# --- root command (depth 0) -----------------------------------------------


def test_root_byte_limit_allows_exact_and_rejects_plus_one(monkeypatch):
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_BYTES", 8)
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_LINE_BYTES", 8)

    assert guard("x" * 8) is False

    monkeypatch.setattr(lifecycle_guard.shlex, "shlex", _explode)
    assert guard("x" * 9) is True


def test_root_line_limit_allows_exact_and_rejects_plus_one(monkeypatch):
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_LINES", 2)

    assert guard("one\ntwo") is False
    assert guard("one\ntwo\nthree") is True


def test_single_giant_line_rejected_before_shlex(monkeypatch):
    """One enormous token is the quadratic shlex case."""
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_LINE_BYTES", 8)
    monkeypatch.setattr(lifecycle_guard.shlex, "shlex", _explode)

    assert guard("xxxxxxxxx\necho ok") is True


def test_root_budget_counts_utf8_bytes(monkeypatch):
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_BYTES", 4)
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_LINE_BYTES", 4)

    assert guard("éé") is False
    assert guard("ééé") is True




def test_lifecycle_scan_root_within_budget_is_not_a_verdict(monkeypatch):
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_BYTES", 8)
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_LINE_BYTES", 8)

    assert lifecycle_guard.lifecycle_scan_root_within_budget("x" * 8) is True
    assert lifecycle_guard.lifecycle_scan_root_within_budget("x" * 9) is False


# --- referenced-script walk ------------------------------------------------


def test_unique_path_budget_bounds_reads_and_fails_closed(monkeypatch, tmp_path):
    """A walk that has to READ more paths than the cap fails closed.

    Both commands name FRESH scripts: a script the guard already walked clean is answered from its
    latch without a read, so it spends no path (that is the point of the latch — the budget bounds
    work, and there is none left to do). The cap is asserted on the scripts that are actually read.
    """
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_PATHS", 2)
    for i in range(5):
        (tmp_path / f"s{i}.sh").write_text("echo ok\n", encoding="utf-8")

    def command(*indexes):
        return " && ".join(f"bash {shlex.quote(str(tmp_path / f's{i}.sh'))}" for i in indexes)

    assert guard(command(0, 1)) is False
    assert guard(command(2, 3, 4)) is True


def test_repeated_path_does_not_spend_unique_path_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_PATHS", 1)
    script = tmp_path / "s.sh"
    script.write_text("echo ok\n", encoding="utf-8")

    assert guard(f"bash {shlex.quote(str(script))} && bash {shlex.quote(str(script))} && sh {shlex.quote(str(script))}") is False


def test_remote_read_budget_charged_before_remote_read(monkeypatch):
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_REMOTE_READS", 1)
    reads: list[str] = []

    def remote(path: str):
        reads.append(path)
        return "echo ok\n"

    assert (
        guard(
            "bash /remote/a.sh && bash /remote/b.sh",
            read_remote_script=remote,
        )
        is True
    )
    assert [Path(p) for p in reads] == [Path("/remote/a.sh").resolve()]


def test_cumulative_text_budget_bounds_recursive_scan(monkeypatch, tmp_path):
    """Two scripts individually under the per-file cap exceed the walk cap.

    Relative references keep the root command short so the budget arithmetic
    is about the scripts, not the tmp_path length. The second command names two
    scripts the first never walked: a latched script is not tokenized again and
    so spends no text budget, and this asserts the cap on the text that IS scanned."""
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_BYTES", 48)
    for letter in "abcd":
        (tmp_path / f"{letter}.sh").write_text("echo " + letter * 10 + "\n", encoding="utf-8")  # 16
    cwd = str(tmp_path)

    # 9 (root) + 16 fits in 48; 19 (root) + 16 + 16 does not → fail closed.
    assert guard("bash a.sh", cwd=cwd) is False
    assert guard("bash c.sh;bash d.sh", cwd=cwd) is True




def test_remote_script_sanitizer_honours_remaining_budget():
    text, unsafe = lifecycle_guard._sanitize_remote_script_text(
        "echo ok\n", max_bytes=4
    )
    assert (text, unsafe) == (None, True)
    text, unsafe = lifecycle_guard._sanitize_remote_script_text(
        "echo ok\n", max_bytes=8
    )
    assert (text, unsafe) == ("echo ok\n", False)


def test_line_budget_fails_closed_before_tokenizing_every_line(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_LINES", 4)
    script = tmp_path / "many.sh"
    script.write_text("echo ok\n" * 10, encoding="utf-8")

    lexers = 0
    real_shlex = lifecycle_guard.shlex.shlex

    def counting(*args, **kwargs):
        nonlocal lexers
        lexers += 1
        return real_shlex(*args, **kwargs)

    monkeypatch.setattr(lifecycle_guard.shlex, "shlex", counting)
    root = f"bash {shlex.quote(str(script))}"
    assert guard(root) is True
    # Only the one-line root was tokenized (a handful of lexers across the
    # direct scans); the 10-line script never was.
    assert 0 < lexers < 10


# --- scheduler entry point --------------------------------------------------


def test_check_gateway_lifecycle_shell_script_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_BYTES", 8)
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_LINE_BYTES", 8)
    script = tmp_path / "long-line.sh"

    script.write_text("x" * 7, encoding="utf-8")
    lifecycle_guard.check_gateway_lifecycle("", str(script))

    script.write_text("x" * 9, encoding="utf-8")
    with pytest.raises(lifecycle_guard.GatewayLifecycleBlocked):
        lifecycle_guard.check_gateway_lifecycle("", str(script))


def test_check_gateway_lifecycle_python_path_charges_masker(monkeypatch, tmp_path):
    """The .py branch's data-exemption masker tokenizes too, so it is budgeted
    and fails closed before shlex on an over-budget line."""
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_LINE_BYTES", 16)

    small = tmp_path / "small.py"
    small.write_text("x = 1\n", encoding="utf-8")
    lifecycle_guard.check_gateway_lifecycle("run report", str(small))

    monkeypatch.setattr(lifecycle_guard.shlex, "shlex", _explode)
    long_line = tmp_path / "long.py"
    long_line.write_text("x = 1\n" + "y" * 40 + "\n", encoding="utf-8")
    with pytest.raises(lifecycle_guard.GatewayLifecycleBlocked):
        lifecycle_guard.check_gateway_lifecycle("run report", str(long_line))


# --- no regression on realistic benign graphs ------------------------------


def test_default_budget_admits_a_wide_benign_wrapper_graph(tmp_path):
    """Issue #78398's shape: one wrapper invoking 200 small legitimate scripts
    must still be allowed under the DEFAULT limits (an earlier fail-closed
    attempt with a 64-path cap blocked exactly this)."""
    children = []
    for i in range(200):
        child = tmp_path / f"c{i}.sh"
        child.write_text("echo step && ls -la /tmp\n" * 20, encoding="utf-8")
        children.append(child)
    hub = tmp_path / "hub.sh"
    hub.write_text("".join(f"bash {shlex.quote(str(c))}\n" for c in children), encoding="utf-8")

    assert guard(f"bash {shlex.quote(str(hub))}") is False

    # ...and a lifecycle command hidden behind the 200 benign scripts is still
    # found: the budget bounds work, it does not stop the walk early.
    evil = tmp_path / "evil.sh"
    evil.write_text("hermes gateway restart\n", encoding="utf-8")
    hub.write_text(hub.read_text() + f"bash {shlex.quote(str(evil))}\n", encoding="utf-8")
    assert guard(f"bash {shlex.quote(str(hub))}") is True
