"""Name a broken serving checkout at launch instead of dying inside it (PRA-6515).

On 2026-10-19 an unresolved merge-conflict marker in ``hermes_cli/gateway_chat_view.py`` — left in
the working tree of the shared checkout every profile imports by a stopped ``git pull --rebase`` —
made every agent run die at import in about four seconds with ``SyntaxError: invalid decimal
literal``. The agent adapter recorded all 21 failures as ``adapter_failed`` ("Hermes exited with
code 1"), indistinguishable from the gateway-handshake timeouts, so a 3 h 22 m fleet outage was
triaged as infrastructure saturation and a gateway restart could not touch it.

This module asks one question before a launch imports anything heavy: can the tree it is about to
run from be imported at all? A tree that cannot is reported as :data:`CHECKOUT_BROKEN` — a name of
its own, carrying the first line of the real error — and the process exits :data:`EXIT_CODE`, so a
run that never started can never be read as an infrastructure failure.

Stdlib-only and importing nothing from the checkout at module level (like ``_early_recovery``): a
launch has to be able to name a broken tree from inside that tree.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

#: The error code every surface reports for a checkout that cannot be imported. Not
#: ``adapter_failed``: that name belongs to the gateway/handshake timeouts (PRA-6380 / PRA-6502).
CHECKOUT_BROKEN = "checkout_broken"

#: ``EX_CONFIG`` from sysexits: the launch cannot succeed until a human fixes the tree. Never 1,
#: which every adapter in the fleet already reads as "the run failed".
EXIT_CODE = 78

#: Verdict cache beside the git dir's other state; keyed by HEAD, the index and these files.
CACHE_NAME = "hermes-checkout-preflight"

#: What a launch imports before any model handshake, so what a broken one kills every run for.
#: ``gateway_chat_view.py`` is on the list because it is exactly the file PRA-6515's stopped rebase
#: left conflicted.
LAUNCH_MODULES = ("hermes_cli/main.py", "hermes_cli/gateway_chat_view.py", "run_agent.py")

#: Commands that must keep working while the tree is broken: they are how a human repairs it.
_REPAIR_COMMANDS = frozenset({"update", "pm", "doctor", "uninstall"})

#: What git leaves at the head of every conflicted hunk.
_CONFLICT_STARTS = ("<<<<<<< ", ">>>>>>> ", "||||||| ")


class Problem(NamedTuple):
    """Why a checkout cannot be launched from: what a human opens, and what the log must say."""

    kind: str  # "unmerged_paths" | "conflict_marker" | "syntax_error"
    path: str  # the file to open, "" when git names none
    line: int | None
    detail: str  # one line of the real error — the run's cause, never a timeout's


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else Path(__file__).resolve().parent.parent


def _git_dir(root: Path) -> Path:
    """``root``'s git dir: ``.git`` itself, or where a linked worktree's ``.git`` file points."""
    dot_git = root / ".git"
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8-sig", errors="replace").strip()
        if text.startswith("gitdir:"):
            return (root / text[len("gitdir:"):].strip()).resolve()
    return dot_git


def _head(git_dir: Path) -> str:
    """HEAD's commit read from the git dir itself — no git spawned, and "" when it names none."""
    try:
        text = (git_dir / "HEAD").read_text(encoding="utf-8-sig", errors="replace").strip()
    except OSError:
        return ""
    if not text.startswith("ref:"):
        return text
    try:
        return (git_dir / text[4:].strip()).read_text(encoding="utf-8-sig", errors="replace").strip()
    except OSError:
        return ""  # packed or unborn ref: the fingerprint names no HEAD and the checks simply rerun


def _stamp(path: Path) -> str:
    try:
        st = path.stat()
    except OSError:
        return "-"
    return f"{st.st_mtime_ns}:{st.st_size}"


def _fingerprint(root: Path, git_dir: Path) -> str:
    """Everything a cached verdict depends on: HEAD, the index, and the files it names.

    A conflicted index has a new ``mtime``, and a marker written into a file changes its stamp, so
    a verdict cached here is never older than the tree it describes.
    """
    stamps = [f"head={_head(git_dir)}", f"index={_stamp(git_dir / 'index')}"]
    stamps += [f"{rel}={_stamp(root / rel)}" for rel in LAUNCH_MODULES]
    return " ".join(stamps)


def _cached_clean(git_dir: Path, fingerprint: str) -> bool:
    try:
        return (git_dir / CACHE_NAME).read_text(encoding="utf-8-sig", errors="replace").strip() == fingerprint
    except OSError:
        return False


def _remember_clean(git_dir: Path, fingerprint: str) -> None:
    try:
        from hermes_cli._early_recovery import write_durable_text

        write_durable_text(git_dir / CACHE_NAME, fingerprint + "\n")
    except (OSError, ImportError):
        pass  # a cache nobody can write only costs the next launch its checks


def _unmerged(root: Path) -> Problem | None:
    """A stopped merge/rebase, from the index: ``git ls-files -u`` names every conflicted path.

    Unanswerable is not broken: no git, no index, a failing git all fall through to the file
    checks below rather than refusing a launch on a guess.
    """
    git = shutil.which("git") or "git"
    try:
        listed = subprocess.run(
            [git, "--literal-pathspecs", "-C", str(root), "ls-files", "-u", "-z"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=20, stdin=subprocess.DEVNULL, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if listed.returncode != 0:
        return None
    paths = sorted({entry.partition("\t")[2] for entry in listed.stdout.split("\0") if entry})
    if not paths:
        return None
    return Problem("unmerged_paths", paths[0], None,
                   f"{len(paths)} path(s) have unresolved conflicts in the index "
                   f"(first: {paths[0]}) — a merge or rebase is stopped")


def _launch_module_problem(root: Path) -> Problem | None:
    """Conflict markers first (the incident's exact shape), then a real parse of each entry module."""
    for rel in LAUNCH_MODULES:
        try:
            source = (root / rel).read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue  # a refactor may legitimately drop one of them; the rest still answer
        lines = source.splitlines()
        first = next((n for n, line in enumerate(lines, 1) if line.startswith(_CONFLICT_STARTS)), None)
        if first is not None:
            # The closing marker names the incoming commit, which is the line a reader needs;
            # the opening marker's "HEAD" names nothing.
            closing = next((line for line in lines if line.startswith(">>>>>>> ")), lines[first - 1])
            return Problem("conflict_marker", rel, first,
                           f"unresolved merge conflict in the working tree: {closing.strip()}")
        try:
            ast.parse(source, filename=rel)
        except SyntaxError as exc:
            return Problem("syntax_error", rel, exc.lineno, exc.msg or "syntax error")
    return None


def checkout_problem(root: Path | None = None) -> Problem | None:
    """Why the checkout at ``root`` cannot be launched from, or ``None`` when it can.

    Cached per HEAD (plus the index and the entry modules' stamps): a healthy launch pays two
    ``stat`` calls and one small read, and a repair that changes any of them reruns the checks.
    """
    root = _root(root)
    git_dir = _git_dir(root)
    tracked = git_dir.is_dir()
    fingerprint = _fingerprint(root, git_dir) if tracked else ""
    if tracked and _cached_clean(git_dir, fingerprint):
        return None
    problem = _unmerged(root) if tracked else None
    if problem is None:
        problem = _launch_module_problem(root)
    if problem is None and tracked:
        _remember_clean(git_dir, fingerprint)
    return problem


def describe(problem: Problem) -> str:
    """``file:line: detail`` — the one line an adapter shows as the run's cause."""
    if problem.line:
        return f"{problem.path}:{problem.line}: {problem.detail}"
    return f"{problem.path or 'the checkout'}: {problem.detail}"


def _command(argv: list[str] | None = None) -> str | None:
    """The invocation's first positional (``None`` for a bare ``hermes``, which is a chat)."""
    try:
        from hermes_cli._parser import command_argv

        args = command_argv(sys.argv[1:] if argv is None else argv)
    except Exception:  # health: allow BLE001 -- a preflight may never fail the launch it guards
        return None
    return args[0] if args else None


def _update_owns_the_tree(root: Path) -> bool:
    """An updater holds the checkout's kernel lock: its move is in flight or being settled.

    That window legitimately shows a half-written tree, and the updater's own guards plus the
    interrupted-pull repair own it — refusing a launch for it would name the wrong cause.
    """
    try:
        from hermes_cli.update_lock import checkout_lock_held

        return checkout_lock_held(root)
    except Exception:  # health: allow BLE001 -- no lock module, no update: the checks below decide
        return False


def refuse_broken_launch(command: str | None = None, root: Path | None = None) -> None:
    """Report a checkout no launch can import from and exit :data:`EXIT_CODE`; else return.

    ``command`` is the invocation's first positional, derived from ``sys.argv`` when omitted: the
    repair commands stay runnable, because nothing on this host recovers until one of them — or a
    human with git — runs.
    """
    if command is None:
        command = _command()
    if command in _REPAIR_COMMANDS:
        return
    if _update_owns_the_tree(_root(root)):
        return
    problem = checkout_problem(root)
    if problem is None:
        return
    checkout = _root(root)
    head = _head(_git_dir(checkout))[:10] or "?"
    print(f"hermes: error: {CHECKOUT_BROKEN}: {describe(problem)}", file=sys.stderr)
    print(f"hermes: {checkout} cannot be imported, so no run was started (HEAD {head}). Fix the tree "
          f"and launch again: `git -C {checkout} status`, then `git -C {checkout} rebase --abort` "
          f"(or `hermes update`).", file=sys.stderr)
    raise SystemExit(EXIT_CODE)
