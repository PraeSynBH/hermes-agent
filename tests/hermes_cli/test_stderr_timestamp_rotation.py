"""The supervised gateway's error log must stay bounded (PRA-6517).

``hermes --run-module hermes_cli.stderr_timestamp --error-log ...`` owns gateway.error.log for the
life of the supervised process, and nothing else ever reopened or truncated it: on this host it
reached 421 MB, where a single ``grep`` pass took 2-4 minutes and every diagnostic read of it was
itself loop-visible I/O.
"""

from hermes_cli.stderr_timestamp import _BoundedLogWriter, _error_log_max_bytes


def test_rotation_caps_the_file_and_keeps_the_newest_line(tmp_path):
    log = tmp_path / "gateway.error.log"
    cap = 4096
    writer = _BoundedLogWriter(log, max_bytes=cap)
    try:
        for index in range(400):
            writer.write_line(f"line-{index:04d} " + "x" * 40 + "\n")
    finally:
        writer.close()

    assert log.stat().st_size <= cap
    assert log.with_name(log.name + ".1").exists(), "the overflow must be kept, not dropped"
    assert "line-0399" in log.read_text(encoding="utf-8")


def test_rotation_keeps_a_bounded_number_of_backups(tmp_path):
    log = tmp_path / "gateway.error.log"
    writer = _BoundedLogWriter(log, max_bytes=1024)
    try:
        for index in range(600):
            writer.write_line(f"line-{index:04d} " + "y" * 40 + "\n")
    finally:
        writer.close()

    backups = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(log.name + "."))
    assert backups == [log.name + ".1", log.name + ".2"], backups


def test_a_cap_of_zero_disables_rotation(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_ERROR_LOG_MAX_BYTES", raising=False)
    assert _error_log_max_bytes(0) == 0
    log = tmp_path / "gateway.error.log"
    writer = _BoundedLogWriter(log, max_bytes=0)
    try:
        for index in range(200):
            writer.write_line(f"line-{index:04d} " + "z" * 40 + "\n")
    finally:
        writer.close()
    assert log.stat().st_size > 4096
    assert not (tmp_path / "gateway.error.log.1").exists()


def test_cap_is_read_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_ERROR_LOG_MAX_BYTES", "12345")
    assert _error_log_max_bytes() == 12345
