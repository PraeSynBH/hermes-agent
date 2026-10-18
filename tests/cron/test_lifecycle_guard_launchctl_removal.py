"""The in-gateway launchd policy is symmetric: a session that cannot REGISTER a job cannot REMOVE one.

Regression for SOU-1850. Observed live on 2026-10-17: an agent session ran the old detector arm
script (bootout-if-loaded, then bootstrap). The guard refused the bootstrap and let the bootout
through, so two human-armed host detectors came off the host 59s after they were armed and the
session could not put them back — a silent blind spot that only the detectors' own ticker fallback
absorbed. ``launchctl bootstrap`` was label-independent; ``launchctl bootout`` was not.

The contract pinned here: every registration verb (submit/bootstrap/load) and every removal verb
(bootout/unload/disable/remove) is refused whatever the label, while reads and transient verbs stay
allowed and the label-scoped gateway-lifecycle pass stays label-scoped.
"""

from __future__ import annotations

import pytest

from cron.lifecycle_guard import (
    contains_gateway_lifecycle_command,
    contains_gateway_lifecycle_command_or_referenced_script as guard,
    contains_launchctl_registration_command,
    contains_launchctl_removal_command,
)

# Labels of the two host detectors that were disarmed, plus a neutral one an attacker would pick.
LABELS = ("com.praesyn.canary-deadman", "com.praesyn.routine-trigger-audit", "com.example.watchdog")


@pytest.mark.parametrize("verb", ["bootout", "unload", "disable", "remove"])
@pytest.mark.parametrize("label", LABELS)
def test_removal_verbs_are_refused_whatever_the_label(verb, label):
    command = f"launchctl {verb} gui/501/{label}"
    assert contains_launchctl_removal_command(command) is True
    assert guard(command) is True


@pytest.mark.parametrize("prefix", ["sudo", "env", "nohup", "setsid", "timeout 60"])
def test_wrapped_removal_is_refused(prefix):
    assert guard(f"{prefix} launchctl bootout gui/501/com.example.watchdog") is True


def test_absolute_path_executable_is_refused():
    assert guard("/bin/launchctl bootout gui/501/com.praesyn.canary-deadman") is True


def test_removal_inside_a_referenced_script_is_refused(tmp_path):
    """The arm script's shape: the agent runs a file, not the command line."""
    arm = tmp_path / "arm-host-detectors.sh"
    arm.write_text(
        "#!/usr/bin/env bash\n"
        "launchctl bootout gui/$(id -u)/com.praesyn.canary-deadman 2>/dev/null\n"
        "launchctl bootstrap gui/$(id -u) ./canary-deadman.plist\n"
    )
    assert guard(f"bash {arm}", cwd=str(tmp_path)) is True


def test_removal_inside_a_shell_c_payload_is_refused():
    assert guard("sh -c 'launchctl bootout gui/501/com.example.watchdog'") is True


def test_removal_through_a_python_argv_list_is_refused():
    """``execute_code`` is a second surface for the same write (#68289): argv lists split the words."""
    code = 'import subprocess; subprocess.run(["launchctl", "bootout", "gui/501/com.example.watchdog"])'
    assert contains_launchctl_removal_command(code) is True
    assert guard(code) is True


@pytest.mark.parametrize("verb", ["submit", "bootstrap", "load"])
def test_registration_stays_label_independent(verb):
    """``load`` is the legacy spelling of ``bootstrap`` — same registration, same refusal."""
    command = f"launchctl {verb} gui/501/com.example.watchdog"
    assert contains_launchctl_registration_command(command) is True
    assert guard(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "launchctl print gui/501/com.example.watchdog",
        "launchctl list",
        "launchctl limits",
        # Transient verbs: they do not add or remove a job from the host's job set.
        "launchctl kickstart -k gui/501/com.example.watchdog",
        "launchctl stop com.example.watchdog",
        # A mention is not a write.
        'grep launchctl bootout /var/log/system.log',
        "echo 'launchctl bootout com.example.watchdog'",
        "cat ~/notes/launchctl-bootout.md",
    ],
)
def test_reads_and_mentions_stay_allowed(command):
    assert contains_launchctl_removal_command(command) is False
    assert guard(command) is False


def test_label_scoped_gateway_pass_is_unchanged():
    """The new label-independent pass must not widen the label-scoped one: that pass keeps naming
    a gateway lifecycle command, and its unrelated-label arm is pinned by
    tests/tools/test_approval_lifecycle_guard_parity.py."""
    assert contains_gateway_lifecycle_command("launchctl bootout gui/501/ai.hermes.gateway") is True
    assert contains_gateway_lifecycle_command("launchctl bootout gui/501/com.example.watchdog") is False
