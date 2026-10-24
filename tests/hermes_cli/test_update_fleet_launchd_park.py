"""SOU-4222 — the post-update check that names a LaunchAgent unable to park a fatal verdict.

``hermes update`` warns about a systemd unit missing ``RestartPreventExitStatus=78``
(:func:`hermes_cli.update_cmd_fleet._repair_unit_without_fatal_exit_park`). launchd's equivalent is
the plist pair this module judges: ``KeepAlive.SuccessfulExit=false`` plus the
``hermes_cli.stderr_timestamp`` wrapper that turns exit 78 into the clean exit that pair acts on.
These cases pin the verdict itself, including the shapes that must NOT raise an alarm (an
unparseable plist, a job that never relaunches at all).
"""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from hermes_cli.update_cmd_fleet import (
    _launchd_plist_parks_on_fatal_verdict,
    _repair_launchd_plist_without_fatal_exit_park,
)

_WRAPPER_ARGV = [
    "/usr/bin/osascript",
    "-l",
    "JavaScript",
    "-e",
    (
        'ObjC.import("stdlib"); const status=$.system("exec /x/hermes --run-module '
        "hermes_cli.stderr_timestamp --error-log /x/gateway.error.log -- /x/hermes gateway run"
        '"); const signal=status & 127;'
    ),
]


def _plist(**overrides) -> str:
    spec = {
        "Label": "ai.hermes.gateway",
        "ProgramArguments": list(_WRAPPER_ARGV),
        "KeepAlive": {"SuccessfulExit": False},
    }
    spec.update(overrides)
    return plistlib.dumps(spec).decode("utf-8")


def test_parks_when_both_halves_are_present():
    assert _launchd_plist_parks_on_fatal_verdict(_plist()) is True


def test_does_not_park_without_the_stderr_wrapper():
    """A 78 relayed unchanged is a failure to ``SuccessfulExit=false``, so launchd relaunches."""
    bare = ["/usr/bin/osascript", "-l", "JavaScript", "-e", "$.exit(0)"]
    assert _launchd_plist_parks_on_fatal_verdict(_plist(ProgramArguments=bare)) is False


def test_does_not_park_when_keepalive_ignores_the_exit_status():
    assert _launchd_plist_parks_on_fatal_verdict(_plist(KeepAlive=True)) is False
    assert _launchd_plist_parks_on_fatal_verdict(_plist(KeepAlive={"SuccessfulExit": True})) is False


def test_a_job_that_never_relaunches_already_parks():
    """No KeepAlive at all: a clean exit is never respawned, which is the contract's outcome."""
    spec = plistlib.loads(_plist().encode())
    spec.pop("KeepAlive")
    assert _launchd_plist_parks_on_fatal_verdict(plistlib.dumps(spec).decode()) is True


@pytest.mark.parametrize("text", ["", "not a plist", "<?xml version='1.0'?><plist><array/></plist>"])
def test_unjudgeable_plists_are_not_reported_as_broken(text):
    assert _launchd_plist_parks_on_fatal_verdict(text) is None


def test_missing_program_arguments_is_not_a_verdict():
    spec = plistlib.loads(_plist().encode())
    spec.pop("ProgramArguments")
    assert _launchd_plist_parks_on_fatal_verdict(plistlib.dumps(spec).decode()) is None


def test_repair_check_names_the_installed_plist_and_its_fix(tmp_path, monkeypatch, capsys):
    """The warning must name the job and carry a command that actually refreshes the plist."""
    from hermes_cli import gateway

    plist_path = tmp_path / "ai.hermes.gateway.plist"
    plist_path.write_text(_plist(ProgramArguments=["/x/hermes", "gateway", "run"]), encoding="utf-8")
    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: plist_path)

    _repair_launchd_plist_without_fatal_exit_park("ai.hermes.gateway")

    out = capsys.readouterr().out
    assert "ai.hermes.gateway cannot park a fatal verdict (exit 78)" in out
    assert "hermes gateway install --force" in out


def test_repair_check_is_silent_for_a_current_plist(tmp_path, monkeypatch, capsys):
    from hermes_cli import gateway

    plist_path = tmp_path / "ai.hermes.gateway.plist"
    plist_path.write_text(_plist(), encoding="utf-8")
    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: plist_path)

    _repair_launchd_plist_without_fatal_exit_park("ai.hermes.gateway")

    assert capsys.readouterr().out == ""


def test_repair_check_ignores_a_missing_plist(tmp_path, monkeypatch, capsys):
    from hermes_cli import gateway

    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: Path(tmp_path) / "absent.plist")

    _repair_launchd_plist_without_fatal_exit_park("ai.hermes.gateway")

    assert capsys.readouterr().out == ""
