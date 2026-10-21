"""A quoted argument of a non-executing program is data, not a command (PRX-5773).

The guard decides "does this text run a gateway lifecycle command?". A commit message, a log
search pattern, a JSON field and an issue body all quote one constantly while writing README
notes and incident write-ups up; none of them executes anything. `_mask_option_values` masks the
VALUE of such an option and lets the rest of the line stand, which is the same shape PRX-5770
established for an inline HTTP request body.

The exemption stays fail-closed. A value the shell expands (`$(…)`, backticks) disables masking
for the whole text, a line that pipes into a shell is never masked, and a real lifecycle command
after the data — or a launchd job-set write anywhere — is still a verdict.
"""

import pytest

from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script as guard


@pytest.mark.parametrize(
    "command",
    [
        # The exact case that opened the card: a commit subject quoting the restart command.
        'git commit -m "note: a human runs hermes gateway restart after merging this"',
        'git commit -m"note: a human runs hermes gateway restart" --no-verify',
        'git commit --message="docs: stop using launchctl kickstart ai.hermes.gateway"',
        'git -C ~/repo commit -F notes.txt -m "hermes gateway restart"',
        # Log search patterns are regular expressions handed to git, never executed.
        'git log -S "hermes gateway restart" --oneline',
        'git log -S"hermes gateway restart" -1',
        'git log -G "launchctl bootstrap ai.hermes.gateway"',
        "git log --grep 'systemctl restart hermes-gateway'",
        'git grep -e "hermes gateway stop" HEAD',
        # gh and jq are HTTP/JSON tools: the value is payload text (jq takes name AND value).
        'gh pr comment 42 --body "this needed hermes gateway restart to take effect"',
        'gh issue comment 42 --title "hermes gateway stop" --body "-"',
        'jq --arg note "hermes gateway restart" . payload.json',
        'jq --argjson n \'{"t":"hermes gateway restart"}\' . payload.json',
        # A wrapper in front of the program does not change who receives the data.
        'sudo -u ben git commit -m "hermes gateway restart"',
        'env A=1 gh pr create --body "hermes gateway restart"',
    ],
    ids=[
        "commit-message",
        "commit-message-attached",
        "commit-message-long",
        "commit-message-file",
        "log-search",
        "log-search-attached",
        "log-search-regexp",
        "log-grep",
        "git-grep",
        "gh-pr-body",
        "gh-issue-title",
        "jq-arg",
        "jq-argjson",
        "sudo-wrapper",
        "env-wrapper",
    ],
)
def test_argument_data_is_not_a_command(command):
    assert guard(command) is False


@pytest.mark.parametrize(
    "command",
    [
        # The value is built by the shell before git sees it: still code, still refused.
        'git commit -m "$(hermes gateway restart)"',
        "git commit -m `hermes gateway restart`",
        # A real command after the data is judged normally.
        'git commit -m "note"; hermes gateway restart',
        'git log -S "x" && hermes gateway restart',
        # The line feeds a shell, so nothing on it is masked.
        'git log -S "hermes gateway restart" | bash',
        # Masking a value never masks the launchd half of the guard: a job-set write is a verdict
        # whatever carries the prose.
        "git commit -m \"hermes gateway restart\"; launchctl bootout gui/501/ai.hermes.gateway",
        "git commit -m \"launchctl bootstrap ai.hermes.gateway\" && launchctl bootstrap ai.hermes.gateway",
    ],
    ids=[
        "substitution",
        "backtick",
        "command-after-data",
        "command-after-search",
        "piped-to-shell",
        "launchd-after-data",
        "launchd-beside-data",
    ],
)
def test_data_that_executes_is_still_blocked(command):
    assert guard(command) is True


@pytest.mark.parametrize(
    "command",
    [
        # The refusals this exemption must not erode (#62891, PRX-5770, SOU-1850).
        "hermes gateway restart",
        "launchctl bootout ai.hermes.gateway",
        "launchctl bootstrap gui/501/com.example.anything",
    ],
    ids=["self-restart", "bootout", "bootstrap-any-label"],
)
def test_in_gateway_refusals_survive_the_exemption(command):
    assert guard(command) is True


def test_a_referenced_script_is_still_read_under_argument_data(tmp_path):
    """A masked value must not hide the OTHER half of the guard: the script walk."""
    script = tmp_path / "restart-gateway.sh"
    script.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
    assert guard(
        'git commit -m "hermes gateway restart"; bash restart-gateway.sh', cwd=str(tmp_path)
    ) is True
