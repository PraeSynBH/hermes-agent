"""``lsp.max_total_rss_mb``: the population bound for language servers.

Idle/deleted-root reaping cannot bound this fleet: each agent run edits its own worktree, so its
tsserver tree (~1 GiB) is *live*, and 20-30 concurrent runs leave 20-30 live trees in the gateway's
process tree.  These tests pin the budget's behaviour: it charges a client its whole process tree (not
just the ~60 MiB root), evicts least-recently-used first, never evicts the only client, and does
nothing at all when it is off.
"""
from __future__ import annotations

import asyncio
import subprocess

import pytest

from agent.lsp import manager as manager_mod
from agent.lsp.manager import DEFAULT_MAX_TOTAL_RSS_MB, LSPService, _process_tree_rss_kb


class _FakeClient:
    """Stand-in for ``LSPClient``: a pid, a shutdown flag, no pipes."""

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


def _service(max_total_rss_mb: float) -> LSPService:
    """Disabled service (no loop thread, no spawns) — the budget pass is exercised directly."""
    return LSPService(enabled=False, wait_mode="document", wait_timeout=0.1,
                      install_strategy="manual", idle_timeout=0.0,
                      max_total_rss_mb=max_total_rss_mb)


def _attach(service: LSPService, clients: list[_FakeClient], order: list[float]) -> None:
    """Populate the service's client table with ``clients`` used at ``order`` (ascending = LRU)."""
    for client, used in zip(clients, order):
        key = (client.server_id, client.workspace_root)
        service._clients[key] = client
        service._last_used[key] = used


def _rss(monkeypatch, per_pid: dict[int, int]) -> None:
    monkeypatch.setattr(manager_mod, "_process_tree_rss_kb", lambda pids: {p: per_pid[p] for p in pids if p in per_pid})


def test_default_budget_is_a_real_cap():
    """The shipped default must actually bound the host — 2 GiB, not "unlimited by default"."""
    assert DEFAULT_MAX_TOTAL_RSS_MB == 2048.0


def test_no_eviction_under_budget(monkeypatch):
    service = _service(2048.0)
    clients = [_FakeClient(11, "/r/a"), _FakeClient(12, "/r/b")]
    _attach(service, clients, [1.0, 2.0])
    _rss(monkeypatch, {11: 600 * 1024, 12: 600 * 1024})

    asyncio.run(service._enforce_rss_budget())

    assert not any(c.stopped for c in clients)
    assert len(service._clients) == 2


def test_evicts_least_recently_used_until_under_budget(monkeypatch):
    service = _service(2048.0)
    old, mid, new = (_FakeClient(21, "/r/old"), _FakeClient(22, "/r/mid"), _FakeClient(23, "/r/new"))
    _attach(service, [old, mid, new], [100.0, 200.0, 300.0])
    # 3 x 1.2 GiB = 3.6 GiB: one eviction still leaves 2.4 GiB, so the two oldest must go.
    _rss(monkeypatch, {21: 1229 * 1024, 22: 1229 * 1024, 23: 1229 * 1024})

    asyncio.run(service._enforce_rss_budget())

    assert old.stopped and mid.stopped
    assert not new.stopped, "the most recently used client is the one a live edit needs"
    assert list(service._clients.values()) == [new]


def test_stops_evicting_once_the_budget_is_met(monkeypatch):
    """One 1.6 GiB tree plus two 0.3 GiB trees fits after a single eviction — not two."""
    service = _service(2048.0)
    old, mid, new = (_FakeClient(31, "/r/old"), _FakeClient(32, "/r/mid"), _FakeClient(33, "/r/new"))
    _attach(service, [old, mid, new], [100.0, 200.0, 300.0])
    _rss(monkeypatch, {31: 300 * 1024, 32: 300 * 1024, 33: 1600 * 1024})

    asyncio.run(service._enforce_rss_budget())

    assert old.stopped
    assert not mid.stopped and not new.stopped


def test_single_client_is_never_evicted(monkeypatch):
    """A lone over-budget client would be the one serving the current edit; leave it alone."""
    service = _service(100.0)
    only = _FakeClient(41, "/r/only")
    _attach(service, [only], [1.0])
    _rss(monkeypatch, {41: 4096 * 1024})

    asyncio.run(service._enforce_rss_budget())

    assert not only.stopped


def test_zero_disables_the_budget(monkeypatch):
    service = _service(0.0)
    clients = [_FakeClient(51, "/r/a"), _FakeClient(52, "/r/b")]
    _attach(service, clients, [1.0, 2.0])

    def _boom(_pids):
        raise AssertionError("no RSS sweep when the budget is off")

    monkeypatch.setattr(manager_mod, "_process_tree_rss_kb", _boom)

    asyncio.run(service._enforce_rss_budget())

    assert not any(c.stopped for c in clients)


def test_clients_still_spawning_are_skipped(monkeypatch):
    """A client whose spawn is in flight has no pid yet and must not be evicted out from under it."""
    service = _service(2048.0)
    a, b = _FakeClient(61, "/r/a"), _FakeClient(62, "/r/b")
    _attach(service, [a, b], [1.0, 2.0])
    service._spawning[("typescript", "/r/b")] = object()  # type: ignore[assignment]
    _rss(monkeypatch, {61: 4096 * 1024, 62: 4096 * 1024})

    asyncio.run(service._enforce_rss_budget())

    assert not a.stopped and not b.stopped
    assert len(service._clients) == 2


@pytest.mark.platforms("posix")
def test_rss_sweep_charges_the_whole_process_tree(monkeypatch):
    """The root is ~60 MiB; its tsserver/typingsInstaller children are >1 GiB — the tree is the cost."""
    ps_out = (
        "  100   1   60000\n"   # root: language server wrapper
        "  101 100 1048576\n"   # tsserver.js
        "  102 101   51200\n"   # typingsInstaller.js (grandchild)
        "  200   1   70000\n"   # a second root
        "  300   1 200000\n"    # unrelated process
    )
    fake = subprocess.CompletedProcess(args=["ps"], returncode=0, stdout=ps_out, stderr="")
    monkeypatch.setattr(manager_mod.subprocess, "run", lambda *a, **k: fake)

    totals = _process_tree_rss_kb([100, 200])

    assert totals[100] == 60000 + 1048576 + 51200
    assert totals[200] == 70000


def test_rss_sweep_reports_nothing_for_a_pid_that_already_exited(monkeypatch):
    fake = subprocess.CompletedProcess(args=["ps"], returncode=0, stdout="  1   0  100\n", stderr="")
    monkeypatch.setattr(manager_mod.subprocess, "run", lambda *a, **k: fake)
    monkeypatch.setattr(manager_mod.os, "name", "posix")

    assert _process_tree_rss_kb([999]) == {}


@pytest.mark.parametrize("value, expected", [("1024", 1024.0), (None, DEFAULT_MAX_TOTAL_RSS_MB)])
def test_create_from_config_reads_the_budget(monkeypatch, value, expected):
    import hermes_cli.config as config_mod

    lsp_cfg = {"enabled": True}
    if value is not None:
        lsp_cfg["max_total_rss_mb"] = value
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {"lsp": lsp_cfg})

    service = LSPService.create_from_config()

    assert service is not None
    assert service._max_total_rss_mb == expected
    service.shutdown()


def test_spawn_charges_the_budget_before_starting_a_new_tree(monkeypatch):
    """The sweep alone leaves peaks at ~2x: tsserver reaches ~1 GiB between two 60s sweeps."""
    from agent.lsp.servers import ServerDef

    service = _service(2048.0)
    fake_client = _FakeClient(71, "/w/new")
    order: list[str] = []
    srv = ServerDef(server_id="fake-ls", extensions=(".fake",), resolve_root=lambda fp, ws: ws,
                    build_spawn=lambda root, ctx: None, seed_first_push=False)
    monkeypatch.setattr(service, "_server_for", lambda fp: srv)
    monkeypatch.setattr(manager_mod, "resolve_workspace_for_file", lambda fp: ("/w/new", True))
    monkeypatch.setattr(service, "_trusted", lambda root: True)

    async def _enforce():
        order.append("budget")

    async def _spawn(_srv, _root, _trusted):
        order.append("spawn")
        return fake_client

    async def _attach(_srv, client, _root):
        order.append("attach")
        return client

    monkeypatch.setattr(service, "_enforce_rss_budget", _enforce)
    monkeypatch.setattr(service, "_spawn_client", _spawn)
    monkeypatch.setattr(service, "_attach_root", _attach)

    assert asyncio.run(service._get_or_spawn("/w/new/x.fake")) is fake_client
    assert order == ["budget", "spawn", "attach"]
