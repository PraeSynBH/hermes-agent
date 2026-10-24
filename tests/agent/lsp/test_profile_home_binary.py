"""The LSP binary a profile spawns comes out of THAT profile's home (PRA-6980).

One gateway process serves every profile from a single environment, and every profile stages its
own servers under ``<home>/lsp/bin``.  Two process-wide shortcuts broke that: the install cache was
keyed by package alone (so the first profile to need ``bash-language-server`` answered every other
profile's lookup with its own path — a cto session ran the agent profile's install), and the binary
search went to PATH before the served profile's own staging dir.
"""
from __future__ import annotations

import os

import agent.lsp.install as install_mod
from agent.lsp.install import try_install
from agent.lsp.servers import ServerContext, _find_binary


def _executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_the_install_cache_answers_per_profile_not_per_package(tmp_path, monkeypatch):
    installed: list[str] = []

    def fake_install(pkg: str) -> str:
        path = _executable(tmp_path / f"home{len(installed)}" / "lsp" / "bin" / "fake-language-server")
        installed.append(str(path))
        return str(path)

    monkeypatch.setattr(install_mod, "_install_results", {})
    monkeypatch.setattr(install_mod, "_install_locks", {})
    monkeypatch.setattr(install_mod, "INSTALL_RECIPES", {
        "fake-lsp": {"strategy": "npm", "pkg": "fake-lsp", "bin": "fake-language-server"},
    })
    monkeypatch.setattr(install_mod, "_do_install", fake_install)

    first_home, second_home = tmp_path / "profile-a", tmp_path / "profile-b"
    first_home.mkdir()
    second_home.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(first_home))
    assert try_install("fake-lsp") == installed[0]

    monkeypatch.setenv("HERMES_HOME", str(second_home))
    assert try_install("fake-lsp") == installed[1], "the second profile was handed the first one's binary"


def test_the_spawned_binary_is_this_profiles_staged_copy_not_a_path_hit(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "cto"
    own = _executable(home / "lsp" / "bin" / "bash-language-server")
    other_profile_bin = _executable(tmp_path / "profiles" / "agent" / "lsp" / "bin" / "bash-language-server")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PATH", str(other_profile_bin.parent) + os.pathsep + os.environ.get("PATH", ""))

    found = _find_binary(ServerContext(workspace_root=str(tmp_path), install_strategy="manual"),
                         "bash-language-server", ("bash-language-server",), None)

    assert found == str(own), "a session spawned another profile's server"
