"""An HTTP client reading its body from stdin owns an INERT heredoc (PRX-5769).

A quoted-delimiter body handed to ``curl --data-binary @-`` travels as request payload, so a
lifecycle command quoted inside it is prose, not something the shell will execute. The consumer
allowlist previously named only ``python``/``osascript``/``cat``, so every agent that documented a
restart incident by POSTing it was refused as if it had run the restart.
"""

import pytest

from tools.shell_heredoc import strip_inert_heredoc_bodies

BODY = "Blocked: run `hermes gateway restart` from a separate shell outside the gateway."


@pytest.mark.parametrize(
    "owner",
    [
        "curl -sS -X POST http://127.0.0.1:3000/x --data-binary @-",
        "curl -sS -X POST http://127.0.0.1:3000/x --data @-",
        "curl -sS -X POST http://127.0.0.1:3000/x --data-raw @-",
        "curl -sS -X POST http://127.0.0.1:3000/x -d @-",
        "curl -sS -T - http://127.0.0.1:3000/x",
        "wget --post-file=- http://127.0.0.1:3000/x",
        "/usr/bin/curl -sS --data-binary=@- http://127.0.0.1:3000/x",
        "env curl -sS --data-binary @- http://127.0.0.1:3000/x",
    ],
)
def test_http_stdin_body_is_masked(owner):
    command = f"{owner} <<'MD'\n{BODY}\nMD"
    masked = strip_inert_heredoc_bodies(command)
    assert "hermes gateway restart" not in masked
    assert masked.count("\n") == command.count("\n")


@pytest.mark.parametrize(
    "owner",
    [
        # A body curl writes to a file could still be executed later: stay visible.
        "curl -sS -o out.sh <<'MD'",
        # No stdin argument at all: the body never reaches the wire as a payload.
        "curl -sS http://127.0.0.1:3000/x <<'MD'",
        # wget without a stdin body.
        "wget -O out.sh <<'MD'",
    ],
)
def test_http_client_without_stdin_body_stays_visible(owner):
    command = f"{owner}\n{BODY}\nMD"
    assert "hermes gateway restart" in strip_inert_heredoc_bodies(command)


def test_unquoted_delimiter_still_expands_so_body_stays_visible():
    command = "curl -sS --data-binary @- http://127.0.0.1:3000/x <<MD\nhermes gateway restart\nMD"
    assert "hermes gateway restart" in strip_inert_heredoc_bodies(command)


def test_shell_consumer_of_the_same_body_still_executes():
    """The consumer decides, not the body: the identical text handed to a shell stays visible."""
    command = f"bash -s <<'MD'\n{BODY}\nMD"
    assert "hermes gateway restart" in strip_inert_heredoc_bodies(command)
