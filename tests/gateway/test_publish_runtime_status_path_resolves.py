"""Publishing runtime status is a per-tick loop operation; it must not resolve paths to do it.

``reconcile_served_profiles`` publishes the serving decision inline on the gateway loop, and the
publish chain used to walk the filesystem four times every time: three inside
``get_active_profile_name`` (home, default root, profiles root) and one canonicalizing HERMES_HOME
for the pid record. ``realpath`` is pure Python under the GIL, so at the serving pid's ~240 threads
those walks are seconds each — 91 of 268 sampled slow-tick loop frames were in the first chain
(PRA-6825/PRA-6826).

Contract: a warm publish resolves nothing, and still publishes the same record (the pid record's
home, the emitted profile name).
"""

from __future__ import annotations

import os
import pathlib

import pytest

from gateway import status


class _StubWriter:
    """Records the payload the way the background writer would persist it; never touches disk."""

    def __init__(self):
        self.payloads = []

    def submit(self, path, payload):
        self.payloads.append(payload)
        return len(self.payloads)

    def wait(self, generation, timeout=None):
        return True

    def flush(self, timeout=None):
        return True


@pytest.fixture
def writer(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    stub = _StubWriter()
    monkeypatch.setattr(status, "_runtime_status_writer", stub)
    return stub


def _count_resolves(monkeypatch):
    calls: list[pathlib.Path] = []
    real = pathlib.Path.resolve

    def counting(self, *args, **kwargs):
        calls.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "resolve", counting)
    return calls


def test_warm_publish_resolves_no_paths(writer, monkeypatch):
    status.publish_runtime_status(active_agents=1)  # first publish resolves, and remembers
    resolved = _count_resolves(monkeypatch)

    for agents in range(20):
        status.publish_runtime_status(active_agents=agents)

    assert resolved == [], (
        "publish_runtime_status walked the filesystem again; this runs on the gateway loop once "
        "per reconciler tick")


def test_a_warm_publish_still_publishes_the_same_record(writer, tmp_path):
    """The hoist is a cache, not a removal: the record keeps its identity fields."""
    status.publish_runtime_status(active_agents=3)
    status.publish_runtime_status(active_agents=4)

    payload = writer.payloads[-1]
    assert payload["active_agents"] == 4
    assert payload["pid"] == os.getpid()
    assert payload["start_time"] == status._get_process_start_time(os.getpid())
    assert status._build_pid_record()["hermes_home"] == str((tmp_path / ".hermes").resolve())
