"""MCP stdio children must not print dotenvx's banner on their JSON-RPC channel (PRX-5605).

A Node MCP server started through dotenvx logs ``◇ injected env (N) from .env`` on stdout by
default. For a stdio server stdout *is* the JSON-RPC framing, so every banner is an unframeable
message — the incident log shows the SDK dropping the session every ~5 s with
``Failed to parse JSONRPC message ... input_value='◇ injected env (0) fro...'``.
"""


def test_stdio_child_env_quiets_dotenvx():
    from tools.mcp_tool_config import _build_safe_env

    env = _build_safe_env(None)

    assert env["DOTENV_CONFIG_QUIET"] == "true"
    assert env["DOTENV_QUIET"] == "true"


def test_a_server_config_can_still_opt_back_in():
    """The default never overrides an explicit ``env`` entry on the server config."""
    from tools.mcp_tool_config import _build_safe_env

    env = _build_safe_env({"DOTENV_CONFIG_QUIET": "false"})

    assert env["DOTENV_CONFIG_QUIET"] == "false"
    assert env["DOTENV_QUIET"] == "true"  # the untouched half of the default still applies
