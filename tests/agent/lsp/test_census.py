"""The LSP census: what a ps sweep cannot see, published where an outside instrument can read it.

PRA-6854's closing measurement asks for "zero LSP roots whose owning run is terminal". A ps census
knows pid, RSS and cwd; a worktree path carries no run id and the newest worktrees are hash-named, so
nothing in the process table answers "whose work is this tree?". The service knows the requesting
session, so it writes it down — and a census must never fail a sweep.
"""
from __future__ import annotations

import asyncio
import json

import agent.lsp.census as census_mod
from agent.lsp.census import write_census
from agent.lsp.manager import LSPService


class _FakeClient:
    def __init__(self, pid: int, root: str, server_id: str = "typescript") -> None:
        self._pid = pid
        self.workspace_root = root
        self.server_id = server_id
        self.stopped = False

    @property
    def pid(self):
        return self._pid

    async def shutdown(self) -> None:
        self.stopped = True


def test_write_census_replaces_the_file_atomically(tmp_path):
    target = tmp_path / "state" / "lsp_census.json"
    assert write_census([{"server": "typescript", "root": "/w", "owner": "s1"}], path=target) == target

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["clients"] == [{"server": "typescript", "root": "/w", "owner": "s1"}]
    assert payload["pid"] > 0

    write_census([], path=target)
    assert json.loads(target.read_text(encoding="utf-8"))["clients"] == []


def test_write_census_never_raises_on_an_unwritable_home(tmp_path):
    """A census is an instrument: a read-only or deleted home must not fail a sweep."""
    blocked = tmp_path / "file-not-a-dir"
    blocked.write_text("x", encoding="utf-8")

    assert write_census([{"server": "typescript"}], path=blocked / "state" / "lsp_census.json") is None


def _service() -> LSPService:
    return LSPService(enabled=False, wait_mode="document", wait_timeout=0.1,
                      install_strategy="manual", idle_timeout=0.0,
                      max_total_rss_mb=0.0, max_tree_rss_mb=0.0)


def test_census_carries_the_owner_session_and_the_idle_age(monkeypatch):
    service = _service()
    client = _FakeClient(71, "/w/owned")
    key = (client.server_id, client.workspace_root)
    service._clients[key] = client
    service._owners[key] = "agent:main:pra-6854"
    service._last_used[key] = 0.0  # long idle

    published: list[list[dict]] = []
    monkeypatch.setattr(census_mod, "write_census", lambda entries, path=None: published.append(entries))

    asyncio.run(service._write_census())

    assert len(published) == 1
    entry = published[0][0]
    assert entry["owner"] == "agent:main:pra-6854"
    assert entry["root"] == "/w/owned"
    assert entry["idle_seconds"] > 0


def test_census_is_emptied_when_no_client_is_left(monkeypatch):
    service = _service()
    published: list[list[dict]] = []
    monkeypatch.setattr(census_mod, "write_census", lambda entries, path=None: published.append(entries))

    asyncio.run(service._write_census())

    assert published == [[]], "a stale census file would outlive the servers it described"


class _QuietClient:
    """Enough of an ``LSPClient`` for one real edit check to end with "no verdict in budget"."""

    seeds_first_push = False

    def __init__(self, pid: int, root: str, server_id: str = "fake-ls") -> None:
        self.pid = pid
        self.workspace_root = root
        self.server_id = server_id
        self.workspace_folders = [root]

    async def open_file(self, path, *, language_id=None, text=None) -> int:
        return 1

    async def save_file(self, path) -> None:
        return None

    async def wait_for_diagnostics(self, path, version, *, mode, timeout) -> bool:
        return False

    def document_text(self, path):
        return None

    def diagnostics_for(self, path, *, fresh_only=False):
        return []

    async def shutdown(self) -> None:
        return None


def _enabled_service() -> LSPService:
    """A real, running service: the edit check runs on its loop, as it does in the gateway."""
    return LSPService(enabled=True, wait_mode="document", wait_timeout=0.1,
                      install_strategy="manual", idle_timeout=0.0,
                      max_total_rss_mb=0.0, max_tree_rss_mb=0.0)


def _wire_quiet_server(monkeypatch, service: LSPService, root: str, client: _QuietClient) -> None:
    from agent.lsp.servers import ServerDef

    async def _spawn(_srv, _root, _trusted):
        return client

    monkeypatch.setattr(service, "enabled_for", lambda fp: True)
    monkeypatch.setattr(service, "_server_for", lambda fp: ServerDef(
        server_id="fake-ls", extensions=(".fake",), resolve_root=lambda fp, ws: ws,
        build_spawn=lambda root, ctx: None, seed_first_push=False))
    monkeypatch.setattr("agent.lsp.manager.resolve_workspace_for_file", lambda fp: (root, True))
    monkeypatch.setattr(service, "_trusted", lambda r: True)
    monkeypatch.setattr(service, "_spawn_client", _spawn)


def _published_census(monkeypatch) -> list[list[dict]]:
    published: list[list[dict]] = []
    monkeypatch.setattr(census_mod, "write_census", lambda entries, path=None: published.append(entries))
    return published


def test_an_edit_check_from_the_tool_path_publishes_its_session_as_the_owner(monkeypatch):
    """PRA-6980: the check dropped the owner on the way to ``_get_or_spawn``, so every client the
    write path served was published with ``owner: ""`` — an unmeasured row, not a clean one."""
    service = _enabled_service()
    try:
        _wire_quiet_server(monkeypatch, service, "/w/tool", _QuietClient(91, "/w/tool"))
        published = _published_census(monkeypatch)

        future = service.start_edit_check("/w/tool/x.fake", pre_text="a = 1\n", post_text="a = ]\n",
                                          owner="local-abc123")
        assert future is not None
        future.result(timeout=10.0)
        asyncio.run(service._write_census())
    finally:
        service.shutdown()

    assert [entry["owner"] for entry in published[0]] == ["local-abc123"]


def test_an_edit_that_bound_no_owner_is_attributed_to_the_ambient_session(monkeypatch):
    """A programmatic writer (``execute_code``'s ``hermes_tools.write_file``, a skill script) never
    crosses ``_owned()``; the service boundary takes the session the caller runs for instead."""
    monkeypatch.setenv("HERMES_SESSION_ID", "local-ambient999")
    service = _enabled_service()
    try:
        _wire_quiet_server(monkeypatch, service, "/w/amb", _QuietClient(92, "/w/amb"))
        published = _published_census(monkeypatch)

        future = service.start_edit_check("/w/amb/x.fake", pre_text=None, post_text="b = 2\n")
        assert future is not None
        future.result(timeout=10.0)
        asyncio.run(service._write_census())
    finally:
        service.shutdown()

    assert [entry["owner"] for entry in published[0]] == ["local-ambient999"]


def test_owner_is_recorded_for_the_session_that_uses_a_root(monkeypatch):
    from agent.lsp.servers import ServerDef

    service = _service()
    fake_client = _FakeClient(72, "/w/new")

    async def _spawn(_srv, _root, _trusted):
        return fake_client

    async def _attach(_srv, client, _root):
        return client

    monkeypatch.setattr(service, "_server_for", lambda fp: ServerDef(
        server_id="fake-ls", extensions=(".fake",), resolve_root=lambda fp, ws: ws,
        build_spawn=lambda root, ctx: None, seed_first_push=False))
    monkeypatch.setattr("agent.lsp.manager.resolve_workspace_for_file", lambda fp: ("/w/new", True))
    monkeypatch.setattr(service, "_trusted", lambda root: True)
    monkeypatch.setattr(service, "_spawn_client", _spawn)
    monkeypatch.setattr(service, "_attach_root", _attach)

    asyncio.run(service._get_or_spawn("/w/new/x.fake", owner="agent:main:pra-6854"))

    assert service._owners[("fake-ls", "/w/new")] == "agent:main:pra-6854"
