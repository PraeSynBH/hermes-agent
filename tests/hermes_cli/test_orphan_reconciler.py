"""The startup reconciler reaps ledger-PROVEN orphans across REAPABLE_PURPOSES — and only those.

A dead recorded spawner is necessary but never sufficient for a long-lived entry: a gateway a
client auto-started at ``ppid=1`` with a dead spawner is healthy by design, and a ``serve``
backend another client still owns is not residue either. The reconciler therefore needs a second
liveness fact per purpose, and these tests drive the real terminate path (actual ``/bin/sleep``
processes against a real ledger file) for each arm of that proof.

Regression for PRA-6599 / PRA-6594.
"""
import json
import os
import subprocess

import pytest


def _entry(pid, home, *, registered_at=1.0, spawner_pid, purpose, install="install-test"):
    return {"pid": pid, "create_time": None, "purpose": purpose, "install": install,
            "spawner_pid": spawner_pid, "spawner_create": None, "registered_at": registered_at,
            "argv": "dummy", "host": "", "port": None, "profile": "", "hermes_home": home,
            "isolated": False}


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    from hermes_cli import process_identity as pi

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pi, "install_id", lambda *a, **k: "install-test")
    children = []

    def write(*entries):
        for entry in entries:
            entry["create_time"] = pi._process_create_time(entry["pid"])
        assert pi._ledger_path().parent == tmp_path, pi._ledger_path()
        (tmp_path / pi.LEDGER_FILENAME).write_text(json.dumps(list(entries)))

    def sleeper():
        proc = subprocess.Popen(["/bin/sleep", "120"])
        children.append(proc)
        return proc

    def dead_spawner():
        """The pid of a process that has already exited (a client that auto-started a daemon)."""
        proc = subprocess.Popen(["/bin/sleep", "0"])
        proc.wait(timeout=15)
        return proc.pid

    try:
        yield write, sleeper, dead_spawner
    finally:
        for proc in children:
            proc.kill()
            proc.wait(timeout=10)


@pytest.mark.platforms("posix")
def test_an_unclaimed_serve_backend_from_a_dead_session_is_reaped(ledger):
    """``serve``/``dashboard``: dead spawner AND the home claims no live owner any more."""
    from hermes_cli.process_identity import reconcile_orphans

    write, sleeper, dead_spawner = ledger
    backend = sleeper()
    write(_entry(backend.pid, "/h/one", spawner_pid=dead_spawner(), purpose="serve"))
    reaped = reconcile_orphans(reparented_fn=lambda pid: True)
    assert [record["pid"] for record in reaped] == [backend.pid]
    backend.wait(timeout=15)  # a real SIGTERM reached it
    assert backend.poll() is not None
    # The record says what was killed, so the next census can be read against the log.
    assert reaped[0]["purpose"] == "serve" and reaped[0]["hermes_home"] == "/h/one"


@pytest.mark.platforms("posix")
def test_a_dashboard_backend_with_a_live_owner_is_never_reaped(ledger):
    """A lock-claiming client (or a live parent) still owns the backend: unclaimed is the proof."""
    from hermes_cli.process_identity import reconcile_orphans

    write, sleeper, dead_spawner = ledger
    backend = sleeper()
    write(_entry(backend.pid, "/h/one", spawner_pid=dead_spawner(), purpose="dashboard"))
    assert reconcile_orphans(reparented_fn=lambda pid: False) == []
    assert backend.poll() is None


@pytest.mark.platforms("posix")
def test_a_backend_with_a_live_or_unrecorded_spawner_is_never_reaped(ledger):
    from hermes_cli.process_identity import reconcile_orphans

    write, sleeper, _dead_spawner = ledger
    live_spawned, unrecorded = sleeper(), sleeper()
    write(_entry(live_spawned.pid, "/h/one", spawner_pid=os.getpid(), purpose="serve"),
          _entry(unrecorded.pid, "/h/two", spawner_pid=None, purpose="serve"))
    assert reconcile_orphans(reparented_fn=lambda pid: True) == []
    assert live_spawned.poll() is None and unrecorded.poll() is None


@pytest.mark.platforms("posix")
def test_interactive_and_self_entries_are_never_reaped(ledger):
    """``REAPABLE_PURPOSES`` is the fence: an idle shell is not residue, and neither is this process."""
    from hermes_cli.process_identity import reconcile_orphans

    write, sleeper, dead_spawner = ledger
    shell = sleeper()
    write(_entry(shell.pid, "/h/one", spawner_pid=dead_spawner(), purpose="cli"),
          _entry(os.getpid(), "/h/one", spawner_pid=dead_spawner(), purpose="serve"))
    assert reconcile_orphans(reparented_fn=lambda pid: True) == []
    assert shell.poll() is None


@pytest.mark.platforms("posix")
def test_the_reconciler_covers_gateways_and_mcp_helpers_too(ledger):
    """One startup pass, every purpose: a superseded gateway and a helper's corpse go together."""
    from hermes_cli.process_identity import reconcile_orphans

    write, sleeper, dead_spawner = ledger
    old_gateway, new_gateway, helper = sleeper(), sleeper(), sleeper()
    write(_entry(old_gateway.pid, "/h/g", registered_at=1.0, spawner_pid=dead_spawner(), purpose="gateway"),
          _entry(new_gateway.pid, "/h/g", registered_at=9.0, spawner_pid=dead_spawner(), purpose="gateway"),
          _entry(helper.pid, "/h/g", spawner_pid=dead_spawner(), purpose="mcp-helper"))
    reaped = reconcile_orphans(reparented_fn=lambda pid: True)
    assert sorted(record["pid"] for record in reaped) == sorted([old_gateway.pid, helper.pid])
    assert {record["purpose"] for record in reaped} == {"gateway", "mcp-helper"}
    old_gateway.wait(timeout=15)
    helper.wait(timeout=15)
    assert old_gateway.poll() is not None and helper.poll() is not None
    assert new_gateway.poll() is None  # the live successor owns the home


@pytest.mark.platforms("posix")
def test_a_corrupt_ledger_reaps_nothing_and_never_raises(ledger):
    from hermes_cli import process_identity as pi

    _write, _sleeper, _dead_spawner = ledger
    pi._ledger_path().write_text("{not json")
    assert pi.reconcile_orphans(reparented_fn=lambda pid: True) == []
