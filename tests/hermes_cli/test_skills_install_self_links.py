"""The skill install must never carry a self-referential link into the skills tree (PRX-5605).

A bundle whose temp dance leaves ``<skill>/.sou2124-tmp -> <skill>`` makes every followlinks scan
of the skills tree re-enter that directory forever: the incident host logged 8,229
``Failed to parse skill file`` warnings in one second with the gateway's main thread CPU-bound.
The scanner prunes the loop now; the install is where the shape is attributable, so it is removed
there and named.
"""

import io
import os

from rich.console import Console


def _console() -> tuple[Console, io.StringIO]:
    out = io.StringIO()
    return Console(file=out, force_terminal=False, width=200), out


def test_a_self_link_left_by_the_install_is_removed_and_named(tmp_path):
    from hermes_cli.skills_hub import _assert_no_self_links

    skill = tmp_path / "alpha"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: alpha\ndescription: d\n---\n", encoding="utf-8")
    (skill / "scripts").mkdir()
    os.symlink(skill, skill / ".sou2124-tmp")          # the incident shape
    os.symlink(skill, skill / "scripts" / ".nested")   # an ancestor link, same loop

    console, out = _console()
    _assert_no_self_links(skill, console)

    assert not (skill / ".sou2124-tmp").is_symlink()
    assert not (skill / "scripts" / ".nested").is_symlink()
    assert (skill / "SKILL.md").is_file()              # the skill itself is untouched
    assert (skill / "scripts").is_dir()
    message = out.getvalue()
    assert ".sou2124-tmp" in message and ".nested" in message


def test_a_clean_install_reports_nothing(tmp_path):
    from hermes_cli.skills_hub import _assert_no_self_links

    skill = tmp_path / "alpha"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: alpha\n---\n", encoding="utf-8")
    elsewhere = tmp_path / "shared-scripts"
    elsewhere.mkdir()
    os.symlink(elsewhere, skill / "scripts-link")  # a link out of the skill is legitimate

    console, out = _console()
    _assert_no_self_links(skill, console)

    assert (skill / "scripts-link").is_symlink()
    assert out.getvalue() == ""
