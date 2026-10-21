"""What the FIRST scan of a referenced script may cost, and what it may not look at (PRX-5784).

The walk latches a clean verdict per script (PRX-5758), so a repeat scan is a few stats — but the
memo used to die with the process, and a gateway restart is exactly when the first scan lands. Two
behaviour contracts, not latency snapshots:

* the memo survives the process, under the same witness contract (an edited script, or one whose
  dependencies changed, is walked again rather than answered from the file);
* a referenced body whose shebang names a non-shell interpreter is scanned for a LITERAL command and
  not descended into. Shell-tokenizing `#!/usr/bin/env node` → `lib/cli.js` → … → npm's
  `package.json` walked a dependency graph the guard can never veto on, and it is what made the cold
  scan 40-119 ms instead of a few stats.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from cron import lifecycle_guard

guard = lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script
LIFECYCLE = "hermes gateway restart"
REPO_ROOT = Path(lifecycle_guard.__file__).resolve().parents[1]


def _clear_memos():
    cache = getattr(lifecycle_guard, "_script_verdict_cache", None)
    if cache is not None:
        cache.clear()
    tokens = getattr(lifecycle_guard, "_cached_shlex_tokens", None)
    if tokens is not None:
        tokens.cache_clear()
    lifecycle_guard._latch_state_loaded = False


@pytest.fixture(autouse=True)
def clear_latches():
    _clear_memos()
    yield
    _clear_memos()


@pytest.fixture
def script_reads(monkeypatch):
    reads = []
    read = lifecycle_guard._read_referenced_script

    def counting(path, *, max_bytes=None):
        reads.append(str(path))
        return read(path, max_bytes=max_bytes)

    monkeypatch.setattr(lifecycle_guard, "_read_referenced_script", counting)
    return reads


@pytest.fixture
def latch_state(tmp_path, monkeypatch):
    """A private memo file, so a case can seed it (the profile home is the real seam elsewhere)."""
    state = tmp_path / "state" / "lifecycle-guard-latch.json"
    state.parent.mkdir(parents=True)
    monkeypatch.setattr(lifecycle_guard, "_latch_state_path", lambda: state)
    return state


def _write(path, *lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _invoked(path):
    return f"{shlex.quote(str(path))} --dry-run"


def _names(reads):
    """Basenames of the files a walk read (macOS resolves the temp dir through /private)."""
    return [Path(read).resolve().name for read in reads]


def _fresh_process(command, home, reads_file):
    """Answer *command* in a NEW interpreter with its own reporting hook (a real gateway restart)."""
    probe = f"""
import json, sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from cron import lifecycle_guard as guard_module

reads = []
read = guard_module._read_referenced_script


def counting(path, *, max_bytes=None):
    reads.append(str(path))
    return read(path, max_bytes=max_bytes)


guard_module._read_referenced_script = counting
verdict = guard_module.contains_gateway_lifecycle_command_or_referenced_script(sys.argv[1])
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps({{"verdict": verdict, "reads": reads}}))
"""
    env = {**os.environ, "HERMES_HOME": str(home)}
    result = subprocess.run(
        [sys.executable, "-c", probe, command, str(reads_file)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(reads_file.read_text(encoding="utf-8"))


@pytest.mark.platforms("posix")
def test_a_clean_verdict_survives_the_process_and_follows_the_bytes(tmp_path):
    """The gateway restarted: the script it already proved clean is not walked again — until the
    script changes, which is a miss like any other (the file is a cache, never an authority)."""
    script = _write(tmp_path / "routine.sh", "#!/bin/bash", "echo ok")
    home = tmp_path / "home"
    command = _invoked(script)

    first = _fresh_process(command, home, tmp_path / "first.json")
    assert first["verdict"] is False
    assert _names(first["reads"]) == ["routine.sh"], "the first process has to walk it once"

    second = _fresh_process(command, home, tmp_path / "second.json")
    assert second["verdict"] is False
    assert second["reads"] == [], f"the new process re-walked {second['reads']}"

    _write(script, "#!/bin/bash", LIFECYCLE)

    third = _fresh_process(command, home, tmp_path / "third.json")
    assert third["verdict"] is True, "a rewritten script must be re-scanned, not answered from disk"
    assert _names(third["reads"]) == ["routine.sh"]


def test_a_non_shell_body_is_read_but_never_descended_into(tmp_path, script_reads):
    """`#!/usr/bin/env node` is not shell source: the shell script it names is not a reference."""
    child = _write(tmp_path / "restart.sh", "#!/bin/bash", LIFECYCLE)
    shim = _write(
        tmp_path / "cli.js", "#!/usr/bin/env node",
        f"require('child_process').execSync({shlex.quote(f'bash {child}')})",
    )

    assert guard(_invoked(shim)) is False
    assert _names(script_reads) == ["cli.js"], f"the node body dragged the walk into {script_reads}"


def test_a_literal_command_inside_a_non_shell_body_still_blocks(tmp_path):
    """Not descending is not the same as not reading: the body's own text is still judged."""
    script = _write(tmp_path / "kill.js", "#!/usr/bin/env node", f"execSync('{LIFECYCLE}')")

    assert guard(_invoked(script)) is True


def test_a_python_body_hands_a_shell_script_to_the_same_rule(tmp_path, script_reads):
    """The interpreter family is decided by the shebang, not by the suffix or by node specifically."""
    child = _write(tmp_path / "restart.sh", "#!/bin/bash", LIFECYCLE)
    script = _write(tmp_path / "helper.py", "#!/usr/bin/env python3", f"os.system({shlex.quote(f'bash {child}')})")

    assert guard(_invoked(script)) is False
    assert _names(script_reads) == ["helper.py"]

    blocked = _write(tmp_path / "blocked.py", "#!/usr/bin/env python3", f"os.system('{LIFECYCLE}')")
    assert guard(_invoked(blocked)) is True


@pytest.mark.parametrize("shebang", ["#!/bin/bash", "#!/bin/sh", "#!/usr/bin/env bash", "#!/usr/bin/env -S bash -e"])
def test_a_shell_shebang_keeps_the_recursive_walk(tmp_path, script_reads, shebang):
    """Only a NON-shell interpreter is terminal; the nested-script walk keeps its full reach."""
    child = _write(tmp_path / "child.sh", "#!/bin/bash", LIFECYCLE)
    parent = _write(tmp_path / "parent.sh", shebang, f"bash {shlex.quote(str(child))}")

    assert guard(_invoked(parent)) is True
    assert "child.sh" in _names(script_reads)


def test_a_body_without_a_shebang_is_still_shell_source(tmp_path, script_reads):
    """An include with no shebang keeps the walk (it may be a shell fragment, not a JS file)."""
    child = _write(tmp_path / "child.sh", "#!/bin/bash", LIFECYCLE)
    parent = _write(tmp_path / "fragment", f"bash {shlex.quote(str(child))}")

    assert guard(_invoked(parent)) is True
    assert "child.sh" in _names(script_reads)


def test_a_corrupt_memo_is_an_empty_memo(tmp_path, latch_state):
    """A half-written or hand-edited file can drop entries; it can never answer a scan."""
    script = _write(tmp_path / "routine.sh", "#!/bin/bash", "echo ok")
    latch_state.write_text("{not json", encoding="utf-8")

    assert guard(_invoked(script)) is False

    # The file is rewritten from the clean walk that just happened.
    assert json.loads(latch_state.read_text(encoding="utf-8"))["entries"]


def test_a_memo_entry_with_a_bogus_witness_shape_is_ignored(tmp_path, latch_state):
    """A witness list is trusted only in its exact shape; anything else is a miss, not an allow."""
    script = _write(tmp_path / "routine.sh", "#!/bin/bash", "echo ok")
    metadata = script.stat()
    latch_state.write_text(json.dumps({"schemaVersion": 1, "entries": [
        {"path": str(script), "mtimeNs": metadata.st_mtime_ns, "size": metadata.st_size,
         "profile": lifecycle_guard._current_profile_name(), "witnesses": [[str(script), "not-a-stamp"]]},
    ]}), encoding="utf-8")

    reads = []
    real = lifecycle_guard._read_referenced_script

    def counting(path, *, max_bytes=None):
        reads.append(str(path))
        return real(path, max_bytes=max_bytes)

    lifecycle_guard._read_referenced_script = counting
    try:
        assert guard(_invoked(script)) is False
    finally:
        lifecycle_guard._read_referenced_script = real

    assert _names(reads) == ["routine.sh"], "a malformed entry must not stand in for a walk"
