"""``get_active_profile_name`` answers from the home path, so it must not re-walk it every call.

The profile reconciler publishes runtime status once per gateway-loop tick, and every publish asks
for the profile name (``agent.monitoring.gateway_health._safe_profile``). Answering it resolved
three paths — the home, the default root, the profiles root — and ``Path.resolve`` is a ``realpath``
walk: in the serving pid's ~240-thread process, 91 of 268 sampled slow-tick loop frames sat inside
exactly this chain, 17 of them one second apart (PRA-6825/PRA-6826).

Contract: the answer is remembered per home path, so a repeat call resolves nothing; a different
home (the next profile this process serves) is still answered correctly.
"""

from __future__ import annotations

import pathlib

import pytest

from hermes_cli import profiles


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    """An isolated ``~/.hermes`` with one named profile, and HERMES_HOME pointing at it."""
    home = tmp_path / ".hermes"
    (home / "profiles" / "coder").mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "coder"))
    return home


def _count_resolves(monkeypatch):
    """Every ``Path.resolve`` reached from here on, in order."""
    calls: list[pathlib.Path] = []
    real = pathlib.Path.resolve

    def counting(self, *args, **kwargs):
        calls.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "resolve", counting)
    return calls


def test_repeat_call_resolves_no_paths(profile_home, monkeypatch):
    assert profiles.get_active_profile_name() == "coder"  # first call resolves, and remembers
    resolved = _count_resolves(monkeypatch)

    assert profiles.get_active_profile_name() == "coder"
    assert resolved == [], (
        "get_active_profile_name re-walked the home path on the gateway loop; the name is a pure "
        "function of that path and must be answered from it, not from the filesystem")


def test_a_different_home_is_still_answered_correctly(profile_home, monkeypatch):
    """The cache is keyed by home path, not frozen: a second profile gets its own answer."""
    assert profiles.get_active_profile_name() == "coder"
    (profile_home / "profiles" / "reviewer").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home / "profiles" / "reviewer"))

    assert profiles.get_active_profile_name() == "reviewer"

    monkeypatch.setenv("HERMES_HOME", str(profile_home / "elsewhere"))
    assert profiles.get_active_profile_name() == "custom"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    assert profiles.get_active_profile_name() == "default"


def test_a_symlinked_home_still_answers_by_its_target(profile_home, monkeypatch):
    """Caching the answer must not cost the resolution it exists for: a symlinked home is its target."""
    link = profile_home / "profiles" / "linked"
    link.symlink_to(profile_home / "profiles" / "coder", target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(link))

    assert profiles.get_active_profile_name() == "coder"
    assert profiles.get_active_profile_name() == "coder"
