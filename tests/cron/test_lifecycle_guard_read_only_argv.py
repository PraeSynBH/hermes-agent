"""A script may declare its pure-query argv, and the guard judges the invocation (PRX-5769).

``restart-gateway.sh`` necessarily contains the restart commands of its restarting modes, so the
body-only walk refused the file's own documented read-only probes — ``--schedulers`` (prints which
schedulers the host accepts) and ``--list-stores`` were treated exactly like a restart. The
declaration keeps the guard on the argv: anything it did not declare is still scanned as code.
"""

import os

import pytest

from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script as guard

SCRIPT_BODY = """#!/bin/sh
{declaration}
case "${{1:-}}" in
  --schedulers) printf 'at: present\\n' ;;
  --list-stores) printf 'store: main\\n' ;;
  --schedule) shift; queue "$@" ;;
  *) hermes gateway restart ;;
esac
"""


def _script(tmp_path, declaration="# lifecycle-guard: read-only-argv --schedulers --list-stores"):
    script = tmp_path / "restart-gateway.sh"
    script.write_text(SCRIPT_BODY.format(declaration=declaration), encoding="utf-8")
    os.chmod(script, 0o755)
    return script


@pytest.mark.parametrize("flag", ["--schedulers", "--list-stores"])
@pytest.mark.parametrize(
    "invocation", ["bash {script} {flag}", "sh {script} {flag}", "{script} {flag}", "sudo bash {script} {flag}"]
)
def test_declared_read_only_flag_is_allowed(tmp_path, flag, invocation):
    script = _script(tmp_path)
    command = invocation.format(script=script, flag=flag)
    assert guard(command, cwd=str(tmp_path)) is False


@pytest.mark.parametrize(
    "args, why",
    [
        ("", "no argv: the script's default mode restarts"),
        ("--schedule 1", "schedules a restart — a positional value, not a declared flag"),
        ("--schedulers --schedulers", "still declared, still read-only"),
        ("--forcible", "not declared"),
        ("--schedulers extra", "a bare positional argument is not part of the declared contract"),
        ("--schedulers; hermes gateway restart", "the command line itself is still scanned"),
    ],
)
def test_undeclared_or_ambiguous_invocations_still_block(tmp_path, args, why):
    script = _script(tmp_path)
    if args == "--schedulers --schedulers":
        assert guard(f"bash {script} {args}", cwd=str(tmp_path)) is False, why
        return
    assert guard(f"bash {script} {args}", cwd=str(tmp_path)) is True, why


def test_script_without_a_declaration_is_scanned_as_before(tmp_path):
    script = _script(tmp_path, declaration="# a script that declares nothing")
    assert guard(f"bash {script} --schedulers", cwd=str(tmp_path)) is True


def test_declaration_does_not_exempt_a_different_script_in_the_same_command(tmp_path):
    """Only the declaring file is exempt for that argv; a sibling reference is still scanned."""
    declared = _script(tmp_path)
    other = tmp_path / "other.sh"
    other.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
    assert guard(f"bash {declared} --schedulers && bash {other}", cwd=str(tmp_path)) is True


def test_argv_exemption_does_not_leak_through_a_caller_script(tmp_path):
    """A wrapper script that calls the declaring script read-only is itself still scanned: its own
    bytes say nothing about argv, so only the exact declaring invocation is exempt."""
    declared = _script(tmp_path)
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(
        f"#!/bin/sh\nbash {declared} --schedulers\nhermes gateway restart\n", encoding="utf-8"
    )
    assert guard(f"bash {wrapper}", cwd=str(tmp_path)) is True


def test_read_only_invocation_inside_a_wrapper_is_allowed_when_the_wrapper_is_clean(tmp_path):
    declared = _script(tmp_path)
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(f"#!/bin/sh\nbash {declared} --schedulers\n", encoding="utf-8")
    assert guard(f"bash {wrapper}", cwd=str(tmp_path)) is False
