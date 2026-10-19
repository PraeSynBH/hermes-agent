"""Invariants for PRA-6515 — a broken serving checkout must be named, never reported as a timeout.

On 2026-10-19 an unresolved merge-conflict marker in ``hermes_cli/gateway_chat_view.py`` of the
shared checkout every profile imports made all 21 sampled agent runs fail at import in ~4 s. The
adapter recorded each as ``adapter_failed`` ("Hermes exited with code 1") — the same code the
gateway/handshake timeouts carry — so 3 h 22 m of fleet outage was triaged as infrastructure
saturation, and a gateway restart could not touch it because the syntax error was on disk.

These are behaviour contracts, not source checks: a marker-contaminated entry module produces
``checkout_broken`` (its own exit code, its own stderr line) and a pull that stops in a rebase never
hands back a successful update.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import _checkout_preflight as preflight
from hermes_cli import update_cmd


def _git(*args, cwd):
    return subprocess.run(
        ["git", "--literal-pathspecs", *args], cwd=cwd, check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def _repo(path):
    _git("init", "-b", "main", cwd=path)
    _git("config", "user.email", "test@example.invalid", cwd=path)
    _git("config", "user.name", "Test", cwd=path)
    return path


def _launch_module(root, rel):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_a_marker_contaminated_entry_module_is_named_checkout_broken(tmp_path, capsys):
    """The incident's exact shape: a conflicted launch module refuses the run under its own code."""
    for rel in preflight.LAUNCH_MODULES:
        _launch_module(tmp_path, rel).write_text("VALUE = 1\n", encoding="utf-8")
    contaminated = _launch_module(tmp_path, "hermes_cli/gateway_chat_view.py")
    contaminated.write_text(
        "from __future__ import annotations\n"
        "\n"
        "<<<<<<< HEAD\n"
        "GATEWAY_STARTUP_RPC_DEADLINE_S = 120\n"
        "=======\n"
        ">>>>>>> 75cb61d54f (gateway chat: the turn's admission gets the startup deadline, not 30 s)\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as stop:
        preflight.refuse_broken_launch("chat", tmp_path)

    assert stop.value.code == preflight.EXIT_CODE != 1
    stderr = capsys.readouterr().err
    assert f"error: {preflight.CHECKOUT_BROKEN}:" in stderr
    assert "hermes_cli/gateway_chat_view.py:3:" in stderr
    assert "75cb61d54f" in stderr, "the first line of the real error must reach the run's log"


def test_a_conflicted_index_is_checkout_broken_and_a_resolved_tree_launches(tmp_path):
    """A stopped merge is caught from the index, and repairing the tree clears the verdict."""
    repo = _repo(tmp_path)
    source = repo / "app.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git("add", "app.py", cwd=repo)
    _git("commit", "-m", "base", cwd=repo)

    _git("checkout", "-b", "side", cwd=repo)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    _git("commit", "-am", "side", cwd=repo)

    _git("checkout", "main", cwd=repo)
    source.write_text("VALUE = 3\n", encoding="utf-8")
    _git("commit", "-am", "main", cwd=repo)
    subprocess.run(["git", "merge", "side"], cwd=repo, capture_output=True, check=False)

    problem = preflight.checkout_problem(repo)
    assert problem is not None
    assert problem.kind == "unmerged_paths"
    assert problem.path == "app.py"

    _git("merge", "--abort", cwd=repo)
    assert preflight.checkout_problem(repo) is None


def test_a_pull_that_stops_in_a_rebase_does_not_exit_success(tmp_path, capsys):
    """`.git/rebase-merge` at exit is the degradation: name the file and the commit, exit non-zero."""
    repo = _repo(tmp_path)
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git("add", "app.py", cwd=repo)
    _git("commit", "-m", "base", cwd=repo)
    (repo / ".git" / "rebase-merge").mkdir()

    with pytest.raises(SystemExit) as stop:
        update_cmd._refuse_stopped_git_operation(["git"], repo)

    assert stop.value.code not in (0, None)
    out = capsys.readouterr().out
    assert "Git rebase" in out
    assert "degraded" in out
    assert _git("rev-parse", "HEAD", cwd=repo) in out, "the commit a human must look at is named"


def test_an_updater_holding_the_checkout_is_not_refused(tmp_path):
    """A move in flight shows a half-written tree; that window belongs to the updater, not the launch."""
    from hermes_cli import update_lock

    repo = _repo(tmp_path)
    _launch_module(repo, "hermes_cli/gateway_chat_view.py").write_text("<<<<<<< HEAD\n", encoding="utf-8")
    assert update_lock._acquire_checkout(repo) is None, "an updater owns the checkout"
    assert update_lock.checkout_lock_held(repo), "the lock this contract is about"
    try:
        preflight.refuse_broken_launch("chat", repo)  # an updater owns the tree: no refusal
    finally:
        update_lock._release_checkout()


def test_a_healthy_tree_is_not_refused(tmp_path, capsys):
    """No git operation, no conflict: the guard is silent and the launch proceeds."""
    repo = _repo(tmp_path)
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git("add", "app.py", cwd=repo)
    _git("commit", "-m", "base", cwd=repo)

    update_cmd._refuse_stopped_git_operation(["git"], repo)
    preflight.refuse_broken_launch("chat", repo)

    captured = capsys.readouterr()
    assert preflight.CHECKOUT_BROKEN not in captured.err + captured.out
