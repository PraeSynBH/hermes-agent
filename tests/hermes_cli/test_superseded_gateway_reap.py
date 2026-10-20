"""Only a gateway a newer live gateway replaced is residue; a lone detached daemon is not.

PRA-6594's reaper has to tell apart two shapes that look identical in ``ps``: an auto-started
daemon whose client exited (``ppid=1``, dead spawner — *healthy* by design, since
``spawn_unmanaged_gateway`` detaches and the messaging gateway is meant to survive the app) and a
gateway the runtime already replaced (two live gateways on one ``HERMES_HOME``). The machine spawn
ledger is what separates them, so these tests drive ``reap_superseded_gateways`` through real
processes against a real ledger file.
"""
import json
import os
import subprocess

import pytest


def _entry(pid, home, *, registered_at, spawner_pid, purpose="gateway"):
    return {"pid": pid, "create_time": None, "purpose": purpose, "install": "install-test",
            "spawner_pid": spawner_pid, "spawner_create": None, "registered_at": registered_at,
            "argv": "dummy", "host": "", "port": None, "profile": "", "hermes_home": home,
            "isolated": False}


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Write ledger entries for live dummy processes into the isolated home's ledger file."""
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
        """The pid of a process that has already exited — a client that auto-started a daemon."""
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
def test_a_lone_detached_daemon_is_never_reaped(ledger):
    from hermes_cli.process_identity import reap_superseded_gateways

    write, sleeper, dead_spawner = ledger
    daemon = sleeper()  # no recorded spawner at all: the auto-started daemon's normal shape
    write(_entry(daemon.pid, "/h/one", registered_at=1.0, spawner_pid=dead_spawner()))
    assert reap_superseded_gateways() == []
    assert daemon.poll() is None


@pytest.mark.platforms("posix")
def test_the_older_of_two_live_gateways_on_one_home_is_reaped(ledger):
    from hermes_cli.process_identity import reap_superseded_gateways

    write, sleeper, dead_spawner = ledger
    old, new = sleeper(), sleeper()
    write(_entry(old.pid, "/h/one", registered_at=1.0, spawner_pid=dead_spawner()),
          _entry(new.pid, "/h/one", registered_at=9.0, spawner_pid=dead_spawner()))
    # A real kill, not a spy: proves the terminate path reaches the process.
    assert reap_superseded_gateways() == [old.pid]
    old.wait(timeout=15)
    assert old.poll() is not None
    assert new.poll() is None  # the surviving gateway is left alone


@pytest.mark.platforms("posix")
def test_a_supervised_gateway_is_never_reaped_even_when_older(ledger):
    from hermes_cli.process_identity import reap_superseded_gateways

    write, sleeper, dead_spawner = ledger
    supervised, daemon = sleeper(), sleeper()
    # The older gateway's launchd/systemd/Desktop wrapper is still alive, so its spawner is not
    # dead: it owns the home and stays, whatever a later daemon claims.
    write(_entry(supervised.pid, "/h/one", registered_at=1.0, spawner_pid=os.getpid()),
          _entry(daemon.pid, "/h/one", registered_at=9.0, spawner_pid=dead_spawner()))
    assert reap_superseded_gateways() == []
    assert supervised.poll() is None
    assert daemon.poll() is None


@pytest.mark.platforms("posix")
def test_each_home_is_judged_on_its_own(ledger):
    from hermes_cli.process_identity import reap_superseded_gateways

    write, sleeper, dead_spawner = ledger
    old_a, new_a, lone_b = sleeper(), sleeper(), sleeper()
    write(_entry(old_a.pid, "/h/a", registered_at=1.0, spawner_pid=dead_spawner()),
          _entry(new_a.pid, "/h/a", registered_at=9.0, spawner_pid=dead_spawner()),
          _entry(lone_b.pid, "/h/b", registered_at=1.0, spawner_pid=dead_spawner()))
    assert reap_superseded_gateways() == [old_a.pid]
    assert lone_b.poll() is None


@pytest.mark.platforms("posix")
def test_duplicate_non_gateway_entries_are_a_different_class(ledger):
    from hermes_cli.process_identity import reap_superseded_gateways

    write, sleeper, dead_spawner = ledger
    first, second = sleeper(), sleeper()
    # Two interactive `cli` entries on one home is not this rung's business: only `gateway` is
    # judged by the "a newer one replaced it" rule.
    write(_entry(first.pid, "/h/c", registered_at=1.0, spawner_pid=dead_spawner(), purpose="cli"),
          _entry(second.pid, "/h/c", registered_at=2.0, spawner_pid=dead_spawner(), purpose="cli"))
    assert reap_superseded_gateways() == []
    assert first.poll() is None
