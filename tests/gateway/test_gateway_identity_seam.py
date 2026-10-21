"""The gateway's spawn-ledger identity is taken at the seam EVERY launch reaches.

PRA-6652 regression: the identity registration and the orphan sweep used to live inside
``gateway.run.main()``. The installed service (launchd/systemd/Desktop) runs
``hermes gateway run`` → ``hermes_cli.gateway.run_gateway()`` → ``gateway.run.start_gateway()``,
which never calls ``main()`` — so on every supervised launch the gateway wrote no ledger entry and
``reconcile_orphans()`` never ran. Measured on this host: a gateway up 2h13m on the deployed code
whose ledger held 27 ``cli`` + 1 ``mcp-helper`` + 1 ``browser-harness-daemon`` entries and no
``gateway``.
"""
import asyncio
import json

import pytest


# The facade wiring itself (``gateway.run.start_gateway`` → ``run_identity.prepare_process``) is not
# unit-tested: importing ``gateway.run`` probes the install store for ``manifest.json``, which the
# suite's real-home guard rejects, and the same import trips ``tests/test_scratch_dir.py`` on an
# unmodified tree. The behaviour below is the contract; the wiring is one call at the shared seam and
# is verified in the field by a ``gateway`` entry appearing in the ledger after a supervised boot.


@pytest.mark.platforms("posix")
def test_identity_is_taken_once_per_process_even_with_both_seams(monkeypatch):
    from gateway import run_identity
    from hermes_cli import process_identity

    monkeypatch.setattr(run_identity, "_prepared", False)
    registrations = []
    monkeypatch.setattr(process_identity, "register_self", lambda purpose, **kw: registrations.append(purpose))
    monkeypatch.setattr(process_identity, "reconcile_orphans", lambda **kw: [])
    monkeypatch.setattr(process_identity, "attach_self_to_kill_on_close_job", lambda: None)

    run_identity.prepare_process()
    run_identity.prepare_process()  # a second seam (main() → start_gateway) must not redo it
    assert registrations == ["gateway"]


@pytest.mark.platforms("posix")
def test_prepare_process_writes_a_real_gateway_entry(tmp_path, monkeypatch):
    """End to end against a real ledger file: no mocks of the write path."""
    from gateway import run_identity
    from hermes_cli import process_identity

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(run_identity, "_prepared", False)
    monkeypatch.setattr(process_identity, "reconcile_orphans", lambda **kw: [])

    run_identity.prepare_process()

    entries = json.loads((tmp_path / process_identity.LEDGER_FILENAME).read_text())
    gateway_entries = [entry for entry in entries if entry.get("purpose") == "gateway"]
    assert [entry["pid"] for entry in gateway_entries] == [__import__("os").getpid()]


@pytest.mark.platforms("posix")
def test_a_failing_identity_step_never_fails_the_boot(monkeypatch):
    from gateway import run_identity
    from hermes_cli import process_identity

    monkeypatch.setattr(run_identity, "_prepared", False)

    def boom(*a, **k):
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr(process_identity, "register_self", boom)
    swept = []
    monkeypatch.setattr(process_identity, "reconcile_orphans", lambda **kw: swept.append(True) or [])
    run_identity.prepare_process()  # must not raise
    assert swept == [True]  # the sweep still runs when identity bookkeeping fails
