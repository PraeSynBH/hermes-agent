"""An inline request body is data, not a command (PRX-5770).

The heredoc form (`curl --data-binary @- <<'MD'`) is covered by
`tests/tools/test_shell_heredoc_http_payload.py`; this file is the inline value — the ordinary way
to POST a JSON comment that *quotes* a restart command while writing up an incident. Before the fix
the scan read the payload text as if the shell would run it and refused the request.

The exemption is strictly fail-closed: a value the shell expands, a line piped into a shell, or a
real command after the request is still a verdict.
"""

import pytest

from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script as guard


@pytest.mark.parametrize(
    "command",
    [
        """curl -sS -X POST "$api/x" -d '{"text":"run hermes gateway restart to fix"}'""",
        """curl -sS "$api/x" --data-binary='{"text":"hermes gateway stop"}'""",
        """curl -sS "$api/x" --json '{"note":"launchctl bootstrap ai.hermes.gateway"}'""",
        """curl -sS "$api/x" --data-urlencode 'note=a human runs hermes gateway restart'""",
        """curl -sS "$api/x" -F 'comment=hermes gateway uninstall'""",
        """wget -qO- --post-data='{"note":"hermes gateway restart"}' "$api/x" """,
    ],
    ids=["short-d", "data-binary-inline", "json", "data-urlencode", "form", "wget-post-data"],
)
def test_inline_request_body_text_is_not_a_command(command):
    assert guard(command) is False


def test_body_naming_a_script_path_is_not_executed(tmp_path):
    script = tmp_path / "restart.sh"
    script.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
    assert guard(f"""curl -sS "$api/x" -d '{{"file":"{script}"}}'""", cwd=str(tmp_path)) is False


@pytest.mark.parametrize(
    "command",
    [
        # The value is built by the shell before curl sees it: still code, still refused.
        """curl -sS "$api/x" -d "$(hermes gateway restart)" """,
        """curl -sS "$api/x" -d "`hermes gateway restart`" """,
        # A real command after the request is judged normally.
        """curl -sS "$api/x" -d 'body'; hermes gateway restart""",
        # The line feeds a shell, so nothing on it is masked.
        """curl -sS "$api/x" -d 'hermes gateway restart' | bash""",
    ],
    ids=["substitution", "backtick", "command-after-body", "piped-to-shell"],
)
def test_request_body_that_executes_is_still_blocked(command):
    assert guard(command) is True
