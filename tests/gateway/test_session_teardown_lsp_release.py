"""Session teardown must free the language servers that served the session.

A language-server client is keyed by ``(server, workspace root)``, is spawned by the long-lived
gateway, and is otherwise only freed by the idle timeout or by a root disappearing from disk. A run
that ends on a *persistent* worktree therefore left a multi-GiB tsserver tree in the gateway's process
tree until something else noticed — which nothing did. ``release_session_workspaces`` is the session
teardown seam, so the release belongs there.
"""
from __future__ import annotations

import contextlib

import pytest

from gateway import session_policy
from tools import terminal_tool


class _NowThread:
    """Replacement for ``threading.Thread`` that runs the release inline, so the test can assert it."""

    def __init__(self, target, daemon=False, name=None):
        self._target = target

    def start(self) -> None:
        self._target()


@pytest.fixture(autouse=True)
def _clean_session_table():
    with terminal_tool._session_cwd_lock:
        saved = dict(terminal_tool._session_cwd)
        terminal_tool._session_cwd.clear()
    saved_overrides = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    yield
    with terminal_tool._session_cwd_lock:
        terminal_tool._session_cwd.clear()
        terminal_tool._session_cwd.update(saved)
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(saved_overrides)


def _patch_scope(monkeypatch):
    import gateway.session_authorities as authorities
    monkeypatch.setattr(authorities, "owner_scope", lambda authority: contextlib.nullcontext())


def test_teardown_releases_the_closed_sessions_workspace(monkeypatch):
    _patch_scope(monkeypatch)
    terminal_tool.record_session_cwd("s1", "/w/one")
    terminal_tool.record_session_cwd("s2", "/w/two")
    released: list[str] = []
    monkeypatch.setattr(session_policy, "_release_lsp_workspaces", released.extend)

    session_policy.release_session_workspaces(object(), ["s1"])

    assert released == ["/w/one"]
    assert terminal_tool.get_session_cwd("s2") == "/w/two"


def test_teardown_reads_the_cwd_before_dropping_the_binding(monkeypatch):
    """``unbind_turn_workspace`` clears the cwd record — the release must capture it first."""
    _patch_scope(monkeypatch)
    # A real bound workspace: the override is what ``unbind_turn_workspace`` drops, and dropping it
    # also clears the session's cwd record — so the release must read the cwd first.
    terminal_tool.register_task_env_overrides(
        "s1", {"cwd": "/w/one", "cwd_source": "session",
               "workspace_key": terminal_tool.workspace_key_for("/w/one")})
    assert terminal_tool.get_session_cwd("s1") == "/w/one"
    seen: list[str] = []

    def _release(roots):
        seen.extend(roots)
        assert terminal_tool.get_session_cwd("s1") is None  # already unbound

    monkeypatch.setattr(session_policy, "_release_lsp_workspaces", _release)

    session_policy.release_session_workspaces(object(), ["s1"])

    assert seen == ["/w/one"]


def test_release_skips_a_workspace_another_live_session_still_uses(monkeypatch):
    monkeypatch.setattr(session_policy.threading, "Thread", _NowThread)
    import agent.lsp as lsp_pkg

    released: list[str] = []
    monkeypatch.setattr(lsp_pkg, "release_workspace", released.append)
    terminal_tool.record_session_cwd("other-run", "/w/shared")

    session_policy._release_lsp_workspaces(["/w/shared", "/w/alone", "/w/alone/"])

    assert released == ["/w/alone"], "a live session's checkout keeps its server; duplicates collapse"


def test_release_is_a_noop_without_roots(monkeypatch):
    monkeypatch.setattr(session_policy.threading, "Thread", _NowThread)
    import agent.lsp as lsp_pkg

    def _boom(_root):
        raise AssertionError("no release without a workspace")

    monkeypatch.setattr(lsp_pkg, "release_workspace", _boom)

    session_policy._release_lsp_workspaces([])


def test_release_survives_a_failing_service(monkeypatch):
    """Teardown is best-effort: one bad root must not stop the rest, nor escape to the caller."""
    monkeypatch.setattr(session_policy.threading, "Thread", _NowThread)
    import agent.lsp as lsp_pkg

    calls: list[str] = []

    def _release(root):
        calls.append(root)
        raise RuntimeError("service wedged")

    monkeypatch.setattr(lsp_pkg, "release_workspace", _release)

    session_policy._release_lsp_workspaces(["/w/a", "/w/b"])

    assert calls == ["/w/a", "/w/b"]
