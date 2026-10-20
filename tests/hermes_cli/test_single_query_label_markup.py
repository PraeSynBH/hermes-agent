"""The one-shot prompt label is the user's own text, never rich markup.

``_run_single_query_mode`` printed the whole prompt through
``console.print(f"[bold blue]{label}[/] {query}")``, so a stray tag in it — a model that
writes ``[/ACTION]`` while imitating an action-block convention, most visibly on a wake /
continuation turn whose prompt carries the previous answer — made ``render_str`` raise
``rich.errors.MarkupError`` before the turn even started. The process died with exit 1 and
no answer: on the board it was indistinguishable from a framework crash (PRX-5748).
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from rich.console import Console

import cli


@pytest.fixture(autouse=True)
def _no_inherited_kanban_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)


def _run_with_prompt(monkeypatch, query: str) -> str:
    """Drive the non-quiet one-shot tail with ``query`` as the prompt; return what it printed."""
    out = io.StringIO()
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda c: None)
    stub = SimpleNamespace(
        _single_query_mode=False,
        _claim_active_session=lambda *a, **k: True,
        console=Console(file=out, width=200, force_terminal=False),
        _show_security_advisories=lambda: None,
        chat=lambda *a, **k: "response",
        _print_exit_summary=lambda **k: None,
        _last_turn_result={"final_response": "done", "completed": True},
    )
    with pytest.raises(SystemExit) as exc:
        cli._run_single_query_mode(stub, query, None, False, True)
    assert exc.value.code == 0
    return out.getvalue()


def test_a_stray_closing_tag_in_the_prompt_does_not_kill_the_run(monkeypatch):
    """``[/ACTION]`` used to raise MarkupError out of the label print, before any turn ran."""
    prompt = "Continue the work.\n\n" * 200 + "[/ACTION]"
    printed = _run_with_prompt(monkeypatch, prompt)
    assert "[/ACTION]" in printed


def test_a_valid_tag_in_the_prompt_is_printed_as_text_not_swallowed(monkeypatch):
    """Escaped, not merely tolerated: the markup is literal, so no style eats the words."""
    printed = _run_with_prompt(monkeypatch, "[bold]ship it[/bold]")
    assert "[bold]ship it[/bold]" in printed
