"""A concurrent registration from another process must not erase the ledger's entries.

Regression for PRA-6594: the ledger is how every reaper proves a process is an orphan
(``spawner_is_dead`` needs the entry; ``None`` means "never touch"). Its only write path,
``_append_entry``, is a read-prune-write of the whole JSON array serialized by ``_LEDGER_LOCK``
— a ``threading.Lock``, i.e. one interpreter. Two processes registering at the same instant (a
gateway booting while agent shells start) each write a full array that lacks the other's entry,
and the loser is simply gone: measured on this host, the ledger held 20 entries and not one
``gateway`` while six gateways were running.

The writers here are separate interpreters registering concurrently against one ledger, each
re-registering repeatedly so the windows overlap, and each staying alive until the parent has read
the roster (a dead pid is legitimately pruned on the next write). The ledger is pre-seeded with
live entries because ``_append_entry`` probes every entry's liveness before writing — a busy host
ledger is exactly what widens the read-prune-write window that loses the update (measured: 5/5
runs failed on the base, 0/5 with the cross-process lock).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Reads ONE line from stdin to start (the parent's barrier), then registers repeatedly; reports
# "done" on stdout and blocks on a second readline so its entry cannot be pruned as a dead pid.
WRITER = '''
import os, sys
repo, home, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.environ["HERMES_HOME"] = home
sys.path.insert(0, repo)
sys.stdin.readline()
from hermes_cli.process_identity import register_self
for _ in range(rounds):
    register_self("gateway")
sys.stdout.write("done\\n"); sys.stdout.flush()
sys.stdin.readline()
'''

# Appends enter/leave pairs around the real ledger lock; two interpreters looping it must never
# interleave (one pair is one critical section).
PAIR = '''
import os, sys
repo, home, marker, rounds = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.environ["HERMES_HOME"] = home
sys.path.insert(0, repo)
sys.stdin.readline()
from hermes_cli.process_identity import _ledger_file_lock, _ledger_path
with open(marker, "a", buffering=1) as out:
    for _ in range(rounds):
        with _ledger_file_lock(_ledger_path()):
            out.write("enter %d\\n" % os.getpid())
            out.write("leave %d\\n" % os.getpid())
sys.stdout.write("done\\n"); sys.stdout.flush()
sys.stdin.readline()
'''

SEED_ENTRIES = 100
WRITERS = 10
ROUNDS = 40


def _spawn(script, home, *extra, count):
    return [
        subprocess.Popen([sys.executable, "-c", script, str(REPO), str(home), *extra],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        for _ in range(count)
    ]


def _release_and_collect(children):
    """Release every child at once, collect one line from each, then let them exit."""
    for child in children:
        assert child.stdin is not None
        child.stdin.write("\n")
        child.stdin.flush()
    lines = []
    try:
        for child in children:
            assert child.stdout is not None
            lines.append(child.stdout.readline())
    finally:
        for child in children:
            try:
                assert child.stdin is not None
                child.stdin.close()
                child.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                child.kill()
    return lines


@pytest.mark.platforms("posix")
def test_concurrent_registrations_from_several_processes_all_survive(tmp_path):
    from hermes_cli.process_identity import LEDGER_FILENAME, _process_create_time, install_id

    home = tmp_path / "home"
    home.mkdir()
    seeders = [subprocess.Popen(["/bin/sleep", "120"]) for _ in range(SEED_ENTRIES)]
    try:
        (home / LEDGER_FILENAME).write_text(json.dumps([
            {"pid": proc.pid, "create_time": _process_create_time(proc.pid), "purpose": "cli",
             "install": install_id(), "spawner_pid": None, "spawner_create": None,
             "registered_at": 0.0, "argv": "seed", "host": "", "port": None,
             "profile": "", "hermes_home": "", "isolated": False}
            for proc in seeders
        ]))
        writers = _spawn(WRITER, home, str(ROUNDS), count=WRITERS)
        lines = _release_and_collect(writers)
    finally:
        for proc in seeders:
            proc.kill()
    assert lines == ["done\n"] * WRITERS, lines
    entries = json.loads((home / LEDGER_FILENAME).read_text())
    registered = {entry["pid"] for entry in entries if entry.get("purpose") == "gateway"}
    assert registered == {child.pid for child in writers}


@pytest.mark.platforms("posix")
def test_the_ledger_lock_is_exclusive_across_processes(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    marker = tmp_path / "pairs.txt"
    children = _spawn(PAIR, home, str(marker), "60", count=2)
    lines = _release_and_collect(children)
    assert lines == ["done\n", "done\n"], lines
    pairs = [line.split()[0] for line in marker.read_text().splitlines()]
    assert len(pairs) == 240
    # Strictly alternating enter/leave: no second holder can be inside the critical section.
    assert pairs == ["enter" if index % 2 == 0 else "leave" for index in range(len(pairs))]
