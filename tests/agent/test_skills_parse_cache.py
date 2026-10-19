"""A one-skill change must not re-read the whole skills tree (PRA-6517).

The shared gateway assembles a system prompt (and the TUI's context gauge) against the same skills
tree on every turn and every profile switch. The disk snapshot is keyed on the WHOLE manifest, so any
single skill edit — an install, the curator, another agent's symlinked skill dir — invalidated it and
the build re-opened and re-parsed every SKILL.md in the tree. On this host that was 495 YAML parses
(4-8 s, tens of seconds while a co-tenant disk scan contends), run inside whatever caller assembled
the prompt, including coroutines on the shared loop.

The invariant below is the loop budget in deterministic form: the number of SKILL.md parses a build
performs is the amount of blocking filesystem work it hands its caller.
"""

import pytest

import agent.prompt_builder as pb


_SKILL_BODY = "".join(f"  key_{index}: value_{index}\n" for index in range(40))


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch, tmp_path):
    """Fresh home, fresh skills prompt cache, fresh parse cache, snapshot dropped."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_parse_cache()
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    yield
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    _clear_parse_cache()


def _clear_parse_cache():
    # The cache is what this fix adds; keep the fixture usable against a tree without it so the
    # pre-fix run fails on the behaviour assertions, not on setup.
    cache = getattr(pb, "_SKILL_PARSE_CACHE", None)
    if cache is not None:
        cache.clear()


def _write_skill(root, category, name, description):
    path = root / "skills" / category / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n{_SKILL_BODY}---\n\n# {name}\n",
        encoding="utf-8",
    )


def _counting_parse(monkeypatch):
    calls = []
    real = pb.parse_frontmatter

    def counting(raw):
        calls.append(raw)
        return real(raw)

    monkeypatch.setattr(pb, "parse_frontmatter", counting)
    return calls


def test_only_the_changed_skill_is_reparsed(monkeypatch, tmp_path):
    for index in range(12):
        _write_skill(tmp_path, "tools", f"skill-{index}", f"Skill {index}")
    parsed = _counting_parse(monkeypatch)

    first = pb.build_skills_system_prompt()
    assert "Skill 3" in first
    assert len(parsed) >= 12, "first build must parse the tree"

    parsed.clear()
    _write_skill(tmp_path, "tools", "skill-3", "Skill three, rewritten")

    second = pb.build_skills_system_prompt()
    assert "Skill three, rewritten" in second, "an edited skill must not serve a stale description"
    assert "Skill 3" not in second
    assert len(parsed) == 1, f"a one-file change re-parsed {len(parsed)} skill files"


def test_unchanged_skills_are_not_reparsed_for_a_new_cache_key(monkeypatch, tmp_path):
    """A second build for a different platform/tool set must not read the tree again."""
    for index in range(8):
        _write_skill(tmp_path, "tools", f"skill-{index}", f"Skill {index}")
    pb.build_skills_system_prompt()
    parsed = _counting_parse(monkeypatch)

    pb.build_skills_system_prompt(
        available_tools={"skills_list", "skill_view"},
        available_toolsets={"skills"},
    )
    assert parsed == [], "a different cache key must not re-read unchanged SKILL.md files"


def test_removed_skills_disappear_without_a_full_reparse(monkeypatch, tmp_path):
    """Deletion is a manifest change too: the removed entry must not resurrect from the parse cache."""
    for index in range(6):
        _write_skill(tmp_path, "tools", f"skill-{index}", f"Skill {index}")
    assert "skill-5" in pb.build_skills_system_prompt()

    import shutil
    shutil.rmtree(tmp_path / "skills" / "tools" / "skill-5")
    parsed = _counting_parse(monkeypatch)
    result = pb.build_skills_system_prompt()
    assert "skill-5" not in result
    assert parsed == [], "a deletion needs no SKILL.md parse at all"
