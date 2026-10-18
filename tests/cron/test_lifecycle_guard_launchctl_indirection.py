"""Indirection-aware launchd job-set detection in the gateway lifecycle guard (SOU-1900).

The guard's launchctl passes were content scans for a literal `launchctl <verb>` token, so a
variable reached the same launchd call invisibly: `~/.paperclip/bin/arm-host-detectors.sh` calls
`"$LAUNCHCTL" bootstrap …` / `bootout …` (`ARM_LAUNCHCTL`, default `launchctl`) and, measured from
an agent shell inside the gateway, booted a probe job into `gui/501` with no refusal.

These tests pin both halves of the fix: the spellings a static scan CAN resolve are refused, and the
shapes that are not a job-set write stay allowed (the guard is fail-closed on what it cannot read,
never "block anything with a dollar sign").
"""

from __future__ import annotations

import pytest

from cron import lifecycle_guard

registration = lifecycle_guard.contains_launchctl_registration_command
removal = lifecycle_guard.contains_launchctl_removal_command


# --- the reported bypass, and the spellings around it ---------------------


@pytest.mark.parametrize(
    "command",
    [
        # The measured shape: the program word is a launchctl-named variable.
        '"$LAUNCHCTL" bootstrap gui/501 ~/Library/LaunchAgents/a.plist',
        "${ARM_LAUNCHCTL} bootstrap gui/501 ~/Library/LaunchAgents/a.plist",
        # `${VAR:-default}` states the program in its own spelling.
        "${ARM_LAUNCHCTL:-launchctl} bootstrap gui/501 ~/Library/LaunchAgents/a.plist",
        "${ANY_PROGRAM:=/bin/launchctl} bootstrap gui/501 ~/Library/LaunchAgents/a.plist",
        # The assignment names the program in an earlier segment of the same command.
        'LAUNCHCTL=launchctl; "$LAUNCHCTL" bootstrap gui/501 ~/Library/LaunchAgents/a.plist',
        'export L=/bin/launchctl; "$L" submit -l com.example.helper -- /bin/sh',
        'env L=/usr/bin/launchctl "$L" bootstrap gui/501 ~/Library/LaunchAgents/a.plist',
        # Wrappers still peel down to the command word.
        'sudo "$LAUNCHCTL" bootstrap gui/501 ~/Library/LaunchAgents/a.plist',
        # A neutral variable name cannot hide a launchd argv: a domain target or a job plist.
        '"$TOOL" bootstrap gui/501 ~/Library/LaunchAgents/a.plist',
        '"$TOOL" bootstrap user/501/com.example.agent',
        '"$TOOL" submit -l com.example.helper -- /bin/sh',
    ],
)
def test_variable_indirection_is_refused_like_the_literal(command):
    assert registration(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "launchctl bootstrap gui/501 ~/Library/LaunchAgents/a.plist",
        "launchctl submit -l com.example.helper -- /bin/sh",
    ],
)
def test_literal_spellings_still_refused(command):
    """The indirection pass must not have replaced the literal one."""
    assert registration(command) is True


# --- stays allowed: no job-set write --------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # A read, not a write.
        "launchctl list | head",
        'launchctl print "gui/501/com.example.agent"',
        # A mention handed to a data sink is documentation, not execution.
        'grep -n "launchctl bootstrap" ~/.paperclip/bin/arm-host-detectors.sh',
        'echo "launchctl bootstrap gui/501 x.plist"',
        # An unrelated program reached through an unrelated variable, even with a launchd-ish word.
        '"$NPM" install',
        '"$BROWSER" load https://example.com',
        '"$PYTHON" -m pytest tests/cron',
        # A variable in argv (not the command word) with no launchd shape.
        '"$MKDIR" -p "$RUNDIR"',
        # launchctl in argv position only: the command word is a sink for the text.
        'printf "%s\\n" launchctl bootstrap',
    ],
)
def test_non_job_set_writes_stay_allowed(command):
    assert registration(command) is False


# --- the removal direction inherits the fix (SOU-1850's verb set) ---------
#
# SOU-1850 exists because `launchctl bootout` reached launchd while `bootstrap` was refused. If
# the indirection resolution lived only in the registration pass, `"$LAUNCHCTL" bootout …` would
# re-open the exact hole through the verb that disarms a host detector.


@pytest.mark.parametrize(
    "command",
    [
        # The shipped SOU-1842 shape, which removed two human-armed detectors.
        '"$LAUNCHCTL" bootout "gui/501/com.example.detector"',
        "${ARM_LAUNCHCTL:-launchctl} bootout gui/501/com.example.detector",
        'LAUNCHCTL=launchctl; "$LAUNCHCTL" unload ~/Library/LaunchAgents/a.plist',
        'sudo "$LAUNCHCTL" disable gui/501/com.example.detector',
        'env L=/bin/launchctl "$L" remove com.example.detector',
        # A neutral variable name still cannot hide a launchd domain target or a job plist.
        '"$TOOL" bootout gui/501/com.example.detector',
        '"$TOOL" unload ~/Library/LaunchAgents/a.plist',
        # Literal spellings stay refused on this path too.
        "launchctl bootout gui/501/com.example.detector",
        "launchctl remove com.example.detector",
    ],
)
def test_variable_indirection_is_refused_for_removal_verbs(command):
    assert removal(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "launchctl list | head",
        'launchctl print "gui/501/com.example.agent"',
        'grep -n "launchctl bootout" ~/.paperclip/bin/arm-host-detectors.sh',
        # `load` and `remove` are ordinary words for other programs: no launchd shape, no refusal.
        '"$BROWSER" load https://example.com',
        '"$GIT" rm --cached notes.plist.md',
        '"$PYTHON" -c "import shutil; shutil.rmtree(dir)"',
    ],
)
def test_removal_pass_stays_allowed_without_a_launchd_shape(command):
    assert removal(command) is False


# --- the whole walk: a script that hides launchctl in a variable ----------


def test_executed_script_hiding_launchctl_in_a_variable_is_unsafe(tmp_path):
    """`bash wrapper.sh` inherits the verdict of the script's own text (the shipped shape)."""
    script = tmp_path / "arm.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'LAUNCHCTL="${ARM_LAUNCHCTL:-launchctl}"\n'
        '"$LAUNCHCTL" bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/a.plist\n',
        encoding="utf-8",
    )

    unsafe, refusal = lifecycle_guard.scan_gateway_lifecycle(f"bash {script}", cwd=str(tmp_path))

    assert unsafe is True
    assert refusal is None


def test_executed_script_hiding_launchctl_behind_a_neutral_name_is_unsafe(tmp_path):
    """A neutral variable name does not help: the argv names a launchd domain target and a plist."""
    script = tmp_path / "probe.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'PROG=/bin/launchctl\n'
        '"$PROG" bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/a.plist\n',
        encoding="utf-8",
    )

    unsafe, _ = lifecycle_guard.scan_gateway_lifecycle(f"bash {script}", cwd=str(tmp_path))

    assert unsafe is True


def test_read_only_script_without_a_job_set_write_stays_safe(tmp_path):
    script = tmp_path / "status.sh"
    script.write_text(
        "#!/usr/bin/env bash\n" 'LAUNCHCTL="${ARM_LAUNCHCTL:-launchctl}"\n' '"$LAUNCHCTL" list\n',
        encoding="utf-8",
    )

    unsafe, _ = lifecycle_guard.scan_gateway_lifecycle(f"bash {script}", cwd=str(tmp_path))

    assert unsafe is False


# --- unit level: the resolvers not covered by the command-level cases -----


@pytest.mark.parametrize(
    ("word", "assigned", "expected"),
    [
        ("launchctl", frozenset(), True),
        ("/bin/launchctl", frozenset(), True),
        ("$LAUNCHCTL", frozenset(), True),
        ("${MY_LCTL}", frozenset(), True),
        ("${X:-launchctl}", frozenset(), True),
        ("${X}", frozenset({"X"}), True),
        ("${X}", frozenset(), False),  # an opaque name with no stated value
        ('"$X"', frozenset(), False),
    ],
)
def test_command_word_resolution(word, assigned, expected):
    assert lifecycle_guard._command_word_reaches_launchctl(word, assigned) is expected


def test_defined_variable_with_a_launchctl_value_is_collected():
    found = lifecycle_guard._assigned_launchctl_variables(
        "A=launchctl B=/usr/bin/launchctl C=git; echo done"
    )

    assert found == frozenset({"A", "B"})


def test_defined_variable_with_a_defaulted_launchctl_value_is_collected():
    """`PROG="${ARM_PROG:-launchctl}"` states the program in the default (found by SOU-1934)."""
    found = lifecycle_guard._assigned_launchctl_variables(
        'PROG="${ARM_PROG:-launchctl}" TAIL=${X} OTHER=git'
    )

    assert found == frozenset({"PROG"})
    assert lifecycle_guard.contains_launchctl_removal_command(
        'PROG="${ARM_PROG:-launchctl}"; "$PROG" bootout "gui/$uid/$label"'
    ) is True
