"""Doctor's skills-tree cycle check (_check_skills_tree_cycles, PRX-5605).

A skill dir holding a symlink to itself made every followlinks scan of the skills tree
recurse forever (8,229 parse warnings in one second on the incident host). The scanner now
prunes the cycle; doctor names the offending entry once and removes it under --fix.
"""

import inspect
import os

from hermes_cli import doctor_state
from hermes_cli.doctor_report import Finding


def _run_check(should_fix: bool) -> Finding:
    """The check body with a fresh Finding (the decorator's one-arg wrapper is what doctor calls)."""
    f = Finding()
    inspect.unwrap(doctor_state._check_skills_tree_cycles)(should_fix, f)
    return f


def _plant_cycle(home):
    skills = home / "skills"
    skill = skills / "alpha"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: alpha\n---\n", encoding="utf-8")
    os.symlink(skill, skill / ".sou2124-tmp")
    return skill


def test_cycle_is_reported_without_fix(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr("hermes_cli.doctor.HERMES_HOME", home)
    skill = _plant_cycle(home)

    finding = _run_check(False)

    out = capsys.readouterr().out
    assert "link" in out and "back into" in out
    assert finding.issues, "a planted cycle must be recorded as an issue"
    assert (skill / ".sou2124-tmp").is_symlink()  # report-don't-delete


def test_fix_removes_the_cycle(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr("hermes_cli.doctor.HERMES_HOME", home)
    skill = _plant_cycle(home)

    finding = _run_check(True)

    assert finding.fixed == 1
    assert not (skill / ".sou2124-tmp").is_symlink()
    assert (skill / "SKILL.md").is_file()  # the skill itself is untouched
    assert "Removed 1 skill dir symlink cycle(s)" in capsys.readouterr().out


def test_clean_tree_reports_nothing(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    (home / "skills" / "alpha").mkdir(parents=True)
    (home / "skills" / "alpha" / "SKILL.md").write_text("---\nname: alpha\n---\n", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.doctor.HERMES_HOME", home)

    finding = _run_check(False)

    assert not finding.issues
    assert "No skill dir entry links back into its own dir" in capsys.readouterr().out
