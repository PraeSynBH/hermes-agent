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
