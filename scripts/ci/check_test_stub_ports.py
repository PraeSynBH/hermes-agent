#!/usr/bin/env python3
"""Fail when a test-side stub server binds a LITERAL TCP port instead of an ephemeral one.

A test stub that binds ``:8765`` is a helper with no owner the moment its runner dies: the next
run either collides with the corpse (``EADDRINUSE``, an intermittently red suite) or silently
drives the LEAKED server from the previous run instead of the one it started. ``:0`` makes the
port the kernel's business and the leftover harmless — that is the half of
``tests/AGENTS.md`` § "Stub servers and other helpers" a machine can check.

The other half is not lintable and is not negotiable by a comment: a stub must stay inside its
test file's process group, because ``scripts/run_tests_parallel.py`` gives every test file its own
session (``start_new_session=True``) and SIGKILLs that whole group on timeout — a stub that
``setsid``s away from it survives the runner, and ``atexit`` never runs when the group is killed.
So the runner owns teardown; the stub must not escape it.

Flags, per ``tests/**/*.py``:

  socketserver binds   ``HTTPServer(("127.0.0.1", 8765))``, ``ThreadingHTTPServer``, ``TCPServer``,
                       ``UDPServer``, ``FakeServer.address_family``-style tuples — any maxport that
                       is not ``0``
  uvicorn / hypercorn  ``uvicorn.run(app, port=8765)``, ``Config(..., port=8765)``
  raw binds            ``sock.bind(("127.0.0.1", 8765))``

Opt out of one line with ``# stub-port: ok — <why>`` on that line: a test whose whole point is
"the client refuses a port already in use" has to name one, and a port read from a *variable*
that the OS just handed out does not match at all.

Run: python scripts/ci/check_test_stub_ports.py [tests_root]
Exit 1 on any violation, 0 when clean.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

OPT_OUT = "stub-port: ok"

# A port literal is the tuple's second element: `("127.0.0.1", 8765)`. `0` (and `"0"`) asks the
# kernel for an ephemeral port — never flagged. A variable/attribute (`port`, `self.port`) does
# not match: only a number written into the test is a collision waiting for the next run.
_PORT = r"[1-9][0-9]{0,4}"
_PATTERNS = (
    re.compile(rf"\b(?:HTTPServer|ThreadingHTTPServer|TCPServer|UDPServer|ThreadingUDPServer)\s*\(\s*\([^)]*,\s*{_PORT}\b"),
    re.compile(rf"\bsocket\.socket\b.*\bbind\s*\(\s*\([^)]*,\s*{_PORT}\b"),
    re.compile(rf"\.bind\s*\(\s*\([^)]*,\s*{_PORT}\s*\)"),
    re.compile(rf"\buvicorn\.run\s*\([^)]*\bport\s*=\s*{_PORT}\b"),
    re.compile(rf"\buvicorn\.Config\s*\([^)]*\bport\s*=\s*{_PORT}\b"),
)


def _code(line: str) -> str:
    """*line* with its trailing comment stripped (a ``#`` in a string only ever hides a hit)."""
    return line.split("#", 1)[0]


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _scan(path: Path) -> list[tuple[int, str]]:
    """``(lineno, line)`` for every literal port in *path*; opt-out lines are skipped."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if OPT_OUT in line:
            continue
        code = _code(line)
        if any(pattern.search(code) for pattern in _PATTERNS):
            hits.append((lineno, line.rstrip()))
    return hits


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[2] / "tests"
    if not root.is_dir():
        print(f"check_test_stub_ports: {root} is not a directory", file=sys.stderr)
        return 2
    violations = [(path, hits) for path in _files(root) if (hits := _scan(path))]
    for path, hits in violations:
        for lineno, line in hits:
            print(f"{path}:{lineno}: stub server binds a literal port — use :0 (ephemeral):\n    {line.strip()}")
    if violations:
        print(f"\n{sum(len(h) for _, h in violations)} literal stub port(s) in {len(violations)} file(s). "
              "A fixed port collides with a stub leaked by an earlier run; bind :0 and read the "
              "assigned port back (tests/AGENTS.md § Stub servers and other helpers).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
