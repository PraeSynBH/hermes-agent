"""``scripts/ci/check_test_stub_ports.py``: a stub that binds a literal port is flagged, an
ephemeral one is not, and the opt-out is the only way to keep a deliberate fixed port.

The regression this guards (PRA-6601) is not the lint itself but the class behind it: a stub server
left over from a killed runner is only harmless if the next run's stub asks the kernel for a fresh
port, so the fixed-port habit has to fail CI instead of surfacing as an intermittent
``EADDRINUSE``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ci"))

import check_test_stub_ports as check


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "test_stub.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_literal_port_in_a_socketserver_bind_is_flagged(tmp_path):
    path = _write(tmp_path, 'server = ThreadingHTTPServer(("127.0.0.1", 8765), _Handler)\n')  # stub-port: ok — the sample under test
    hits = check._scan(path)
    assert [lineno for lineno, _line in hits] == [1]


def test_an_ephemeral_port_is_the_expected_shape(tmp_path):
    """``:0`` is the rule — the kernel hands out the port and a leftover cannot collide."""
    path = _write(tmp_path, 'server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)\n'
                             'other = HTTPServer(("", 0), _Handler)\n')
    assert check._scan(path) == []


def test_a_port_the_os_just_handed_back_is_not_a_literal(tmp_path):
    path = _write(tmp_path, "sock.bind((host, port))\nuvicorn.run(app, port=server_port)\n")
    assert check._scan(path) == []


def test_a_literal_port_in_a_uvicorn_run_or_raw_bind_is_flagged(tmp_path):
    uvicorn_sample = "uvicorn.run(app, port=9101)\n"  # stub-port: ok — the sample under test
    bind_sample = 'sock.bind(("127.0.0.1", 9102))\n'  # stub-port: ok — the sample under test
    path = _write(tmp_path, uvicorn_sample + bind_sample)
    assert len(check._scan(path)) == 2


def test_the_opt_out_keeps_a_deliberate_fixed_port(tmp_path):
    """A test whose subject IS port reuse has to name a port; the marker says why."""
    path = _write(tmp_path, 'server = HTTPServer(("127.0.0.1", 8765), _H)  # stub-port: ok — asserts EADDRINUSE\n')
    assert check._scan(path) == []


def test_the_cli_exits_nonzero_on_a_violation_and_zero_when_clean(tmp_path):
    _write(tmp_path, 'server = HTTPServer(("127.0.0.1", 8765), _H)\n')  # stub-port: ok — the sample under test
    assert check.main(["check", str(tmp_path)]) == 1
    _write(tmp_path, 'server = HTTPServer(("127.0.0.1", 0), _H)\n')
    assert check.main(["check", str(tmp_path)]) == 0
