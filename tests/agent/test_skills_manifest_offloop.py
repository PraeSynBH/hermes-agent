"""A prompt build on the gateway's event loop must not walk the skills tree (PRA-6517).

The shared gateway assembles system prompts on its one event loop, and the manifest walk under that
build is the load-sensitive half: 0.1-0.6 s warm on an idle host, 226-290 s at load 115-130, because
it is the one part of a build that the parse cache cannot remove — the walk IS the staleness check.

Two contracts here:

* on a loop thread a build never walks: it serves the last known manifest and refreshes it off-loop;
* the walk itself cannot re-enter a directory, or one back-link makes the manifest grow without bound
  (194,279 entries in 156 s was measured on this host from exactly that).
"""

import asyncio
import threading
from pathlib import Path

import pytest

import agent.prompt_builder as pb


_SKILL_BODY = "".join(f"  key_{index}: value_{index}\n" for index in range(10))


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    _clear_state()
    yield
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    _clear_state()


def _clear_state():
    # Keep the fixture usable against a tree without the caches this fix adds, so the pre-fix run
    # fails on the behaviour assertions, not on setup.
    manifest_lock = getattr(pb, "_MANIFEST_LOCK", None)
    if manifest_lock is not None:
        with manifest_lock:
            pb._MANIFEST_CACHE.clear()
            pb._MANIFEST_REFRESHING.clear()
    snapshot_lock = getattr(pb, "_SKILLS_SNAPSHOT_MEMO_LOCK", None)
    if snapshot_lock is not None:
        with snapshot_lock:
            pb._SKILLS_SNAPSHOT_MEMO.clear()


def _write_skill(root, category, name, description):
    path = root / "skills" / category / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n{_SKILL_BODY}---\n\n# {name}\n",
        encoding="utf-8",
    )


def _build_on_the_loop():
    """Run one prompt build inside a real event loop, as the shared gateway does."""

    async def build():
        return pb.build_skills_system_prompt()

    return asyncio.run(build())


def _expire_manifest(skills_dir):
    cache = getattr(pb, "_MANIFEST_CACHE", None)
    if not cache:  # a tree without the cache walks on every build anyway
        return
    key = str(skills_dir)
    with pb._MANIFEST_LOCK:
        timestamp, manifest = cache[key]
        cache[key] = (timestamp - pb._MANIFEST_TTL - 1.0, manifest)


def test_a_build_on_the_event_loop_does_not_walk_the_tree(monkeypatch, tmp_path):
    _write_skill(tmp_path, "tools", "skill-0", "Skill zero")
    assert "Skill zero" in pb.build_skills_system_prompt()  # off-loop build: exact, caches the manifest

    walks = []
    monkeypatch.setattr(pb, "_build_skills_manifest", lambda d: walks.append(d) or {})

    prompt = _build_on_the_loop()

    assert "Skill zero" in prompt, "an on-loop build must still serve the known index"
    assert walks == [], f"the on-loop build walked the tree {len(walks)} time(s)"


def test_a_stale_manifest_on_the_loop_refreshes_without_blocking(monkeypatch, tmp_path):
    _write_skill(tmp_path, "tools", "skill-0", "Skill zero")
    pb.build_skills_system_prompt()
    _expire_manifest(pb.get_skills_dir())

    walked = threading.Event()
    release = threading.Event()
    real = pb._build_skills_manifest

    def blocking_walk(skills_dir):
        walked.set()
        release.wait(10)
        return real(skills_dir)

    monkeypatch.setattr(pb, "_build_skills_manifest", blocking_walk)

    prompt = {}
    done = threading.Event()

    def build():
        prompt["value"] = _build_on_the_loop()
        done.set()

    worker = threading.Thread(target=build, daemon=True)
    worker.start()
    try:
        assert done.wait(5), "the on-loop build blocked on the skills walk"
        assert walked.wait(5), "a stale manifest must start an off-loop refresh"
        assert "Skill zero" in prompt["value"], "the build went ahead without the served manifest"
    finally:
        release.set()
        worker.join(10)


def test_a_repeat_build_reads_no_snapshot_file(monkeypatch, tmp_path):
    _write_skill(tmp_path, "tools", "skill-0", "Skill zero")
    pb.build_skills_system_prompt()  # writes the snapshot
    pb.build_skills_system_prompt()  # reads it once and memoises it

    snapshot_path = str(pb._skills_prompt_snapshot_path())
    reads = []
    real_read_text = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        if str(self) == snapshot_path:
            reads.append(str(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    pb.build_skills_system_prompt()

    assert reads == [], "a cache hit still read the snapshot off disk"


def test_a_symlink_back_into_the_tree_is_walked_once(tmp_path):
    for index in range(3):
        _write_skill(tmp_path, "tools", f"skill-{index}", f"Skill {index}")
    skills = tmp_path / "skills"
    (skills / "tools" / "skill-0" / "loop").symlink_to(skills, target_is_directory=True)

    manifest = pb._build_skills_manifest(skills)

    assert set(manifest) == {f"tools/skill-{index}/SKILL.md" for index in range(3)}, (
        "the walk re-entered a directory it had already expanded"
    )
