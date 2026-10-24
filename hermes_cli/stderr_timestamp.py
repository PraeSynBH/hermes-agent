"""Run a child process while prefixing each stdout and stderr line with a timestamp."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Sequence, TextIO

EXTERNAL_SUPERVISOR_FLAG = "--external-supervisor"
_LAUNCHD_LABEL_ENV = "HERMES_LAUNCHD_LABEL"
# gateway.restart.GATEWAY_FATAL_CONFIG_EXIT_CODE. This wrapper is a launcher boot
# file: it runs from a source slice and stays stdlib-only.
_GATEWAY_FATAL_CONFIG_EXIT_CODE = 78

_TIMESTAMP_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}(?:\s|$)")
# A grandchild that inherited stdout can hold the pipe open after the child exits; the
# wrapper must still exit with the child's status so launchd KeepAlive sees it.
_STDOUT_DRAIN_TIMEOUT_S = 5.0


def timestamp() -> str:
    """Local time in logging.Formatter's default ``%(asctime)s`` shape, which ``hermes logs --since`` parses."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:23]


def stamp_line(line: str) -> str:
    """*line* with a leading :func:`timestamp` (kept as-is if it already has one) and one ``\\n``."""
    rendered = line.rstrip("\r\n")
    prefix = "" if _TIMESTAMP_PREFIX.match(rendered) else f"{timestamp()} "
    return f"{prefix}{rendered}\n"


def _write_timestamped_line(log_file: TextIO, line: str) -> None:
    log_file.write(stamp_line(line))
    log_file.flush()


def _open_log(log_path: Path) -> TextIO:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path.open("a", encoding="utf-8", buffering=1)


_ERROR_LOG_ROTATE_ENV = "HERMES_ERROR_LOG_MAX_BYTES"
# 64 MiB: a week of gateway noise, and small enough that `grep`/`hermes logs` stay interactive. The
# file reached 421 MB on this host, where one diagnostic pass took 2-4 minutes and every scan of it was
# loop-visible I/O (PRA-6517).
_DEFAULT_ERROR_LOG_MAX_BYTES = 64 * 1024 * 1024
_ERROR_LOG_BACKUPS = 2


def _error_log_max_bytes(override: int | None = None) -> int:
    """Cap from the argument, else ``HERMES_ERROR_LOG_MAX_BYTES``, else 64 MiB; 0 disables rotation."""
    if override is not None:
        return max(0, int(override))
    raw = os.environ.get(_ERROR_LOG_ROTATE_ENV, "").strip()
    try:
        return max(0, int(raw)) if raw else _DEFAULT_ERROR_LOG_MAX_BYTES
    except ValueError:
        return _DEFAULT_ERROR_LOG_MAX_BYTES


class _BoundedLogWriter:
    """Append lines to *log_path*, rotating to ``<name>.1`` / ``<name>.2`` at a byte cap.

    The wrapper owns this file for the life of the supervised gateway, so the cap has to be enforced
    here: nothing else ever reopens or truncates it (that is why it grew to 421 MB).
    """

    def __init__(self, log_path: Path, max_bytes: int | None = None) -> None:
        self.log_path = Path(log_path)
        self.max_bytes = _error_log_max_bytes(max_bytes)
        self._file: TextIO | None = None
        self._open()

    def _open(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("a", encoding="utf-8", buffering=1)

    def _size(self) -> int:
        try:
            return self.log_path.stat().st_size
        except OSError:
            return 0

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def rotate(self) -> None:
        """Shift live → .1 → .2, drop the oldest, and reopen a fresh live file."""
        self.close()
        try:
            oldest = self.log_path.with_name(f"{self.log_path.name}.{_ERROR_LOG_BACKUPS}")
            oldest.unlink(missing_ok=True)
            for index in range(_ERROR_LOG_BACKUPS - 1, 0, -1):
                src = self.log_path.with_name(f"{self.log_path.name}.{index}")
                if src.exists():
                    src.replace(self.log_path.with_name(f"{self.log_path.name}.{index + 1}"))
            if self.log_path.exists():
                self.log_path.replace(self.log_path.with_name(f"{self.log_path.name}.1"))
        except OSError:
            # A rotation that cannot rename must not lose the stream: reopen and keep appending.
            pass
        self._open()
        if self._file is not None:
            self._file.write(f"{timestamp()} [gateway-log] rotated at {self.max_bytes} bytes\n")

    def write_line(self, line: str) -> None:
        stamped = stamp_line(line)
        if self.max_bytes and self._size() + len(stamped.encode("utf-8")) > self.max_bytes:
            self.rotate()
        if self._file is not None:
            self._file.write(stamped)
            self._file.flush()


def _copy_stderr_with_timestamps(stderr: BinaryIO, log_path: Path) -> None:
    with _BoundedLogWriter(log_path) as writer:
        for raw_line in iter(stderr.readline, b""):
            writer.write_line(raw_line.decode("utf-8", errors="replace"))


def _copy_stdout_with_timestamps(stdout: BinaryIO) -> None:
    # fd 1 is the caller's target (launchd appends it to gateway.log, the logging handler's
    # file), so lines keep their destination and gain the stamp `hermes logs --since` needs.
    with open(1, "w", encoding="utf-8", buffering=1, closefd=False) as out:
        for raw_line in iter(stdout.readline, b""):
            _write_timestamped_line(out, raw_line.decode("utf-8", errors="replace"))


def _install_signal_forwarders(proc: subprocess.Popen[bytes]) -> dict[int, object]:
    def _forward(signum: int, _frame: object) -> None:
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    previous: dict[int, object] = {}
    # SIGUSR1 is the gateway's drain-aware restart request. launchd owns THIS wrapper's PID,
    # so `hermes update` signals us, not the gateway; an unforwarded SIGUSR1 kills the wrapper
    # (Python's default action), launchd tears the group down with SIGTERM and applies its
    # ~60 s crash back-off per sibling profile (#101426). SIGUSR2 is the gateway's
    # faulthandler stack-dump request (gateway/run_startup.py); unforwarded it terminates
    # the wrapper the same way instead of dumping stacks.
    forwarded = (
        signal.SIGTERM,
        signal.SIGINT,
        getattr(signal, "SIGHUP", None),
        getattr(signal, "SIGUSR1", None),
        getattr(signal, "SIGUSR2", None),
    )
    for signum in forwarded:
        if signum is not None:
            try:
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, _forward)
            except (OSError, RuntimeError, ValueError):
                previous.pop(signum, None)
    return previous


def _gateway_argv_matcher():
    """The canonical argv matcher, or ``None`` when this tree cannot be imported.

    ``None`` is a real state, not an error to swallow: the wrapper runs from a source slice that a
    half-applied ``hermes update`` can leave unimportable, and the caller's next decision (park or
    relay a fatal exit) must not silently flip to "relay" just because the matcher went missing.
    """
    try:
        from gateway.status import looks_like_gateway_command_line
    except Exception:  # health: allow BLE001 -- any import failure means "unresolvable" to the caller
        return None
    return looks_like_gateway_command_line


def _is_hermes_gateway_run_argv(command: Sequence[str]) -> bool:
    """True for Hermes ``gateway run`` argv this wrapper is allowed to upgrade.

    The wrapper is generic. Only historical/current Hermes gateway shapes get ``--external-
    supervisor``; an arbitrary launchd child must not be marked as gateway-supervised (#87005).
    """
    matcher = _gateway_argv_matcher()
    return bool(matcher and matcher(" ".join(str(part) for part in command)))


def _child_launchd_label_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Env vars that carry this wrapper's launchd identity to the grandchild.

    launchd stamps ``XPC_SERVICE_NAME=<job label>`` only on this wrapper (its direct child; an
    interactive shell has none, the grandchild sees ``XPC_SERVICE_NAME=0``). Re-exporting the
    label lets the gateway resolve its job without it (the stop-drain cap reading the live
    ``ExitTimeOut``, the exit-75 restart route, the control-socket supervisor declaration — all
    via ``gateway.restart.launchd_job_label``). Only ``ai.hermes.*`` labels are exported;
    app-coalition labels are meaningless as a job identity.
    """
    env = os.environ if environ is None else environ
    for variable in ("XPC_SERVICE_NAME", _LAUNCHD_LABEL_ENV):
        label = str(env.get(variable, "") or "").strip()
        if label.startswith("ai.hermes"):
            return {_LAUNCHD_LABEL_ENV: label}
    return {}


def _prepare_child_command(command: Sequence[str], environ: Mapping[str, str] | None = None) -> list[str]:
    """Return the argv to exec, upgrading stale launchd-wrapped gateway commands.

    launchd stamps ``XPC_SERVICE_NAME=<job label>`` only on this wrapper (its direct child; an
    interactive shell has none, the grandchild sees ``XPC_SERVICE_NAME=0``). Newly generated
    plists put ``--external-supervisor`` on the inner ``gateway run`` so ``hermes update`` can see
    the flag on the live process argv.
    """
    argv = [str(part) for part in command]
    env = os.environ if environ is None else environ
    xpc_service = str(env.get("XPC_SERVICE_NAME", "")).strip()
    if EXTERNAL_SUPERVISOR_FLAG not in argv and xpc_service and xpc_service != "0" and _is_hermes_gateway_run_argv(argv):
        argv.append(EXTERNAL_SUPERVISOR_FLAG)
    return argv


def _child_returncode_for_supervisor(command: Sequence[str], returncode: int) -> int:
    """Exit status the launchd wrapper reports for *returncode* from *command*.

    Signal deaths stay 128+N. Gateway EX_CONFIG (78) becomes 0 so
    ``KeepAlive.SuccessfulExit=false`` parks the job instead of crash-looping;
    a non-gateway child that happens to exit 78 is left alone.

    A 78 is also parked when the argv matcher cannot be resolved at all: EX_CONFIG is a
    PERMANENT failure by definition, and every other supervisor Hermes supports decides it
    argv-blind (systemd's ``RestartPreventExitStatus=78``, the s6 finish script's 78→125). Relaying
    an unclassifiable 78 instead makes launchd relaunch a permanent refusal, which is the exact
    loop this mapping exists to stop — so the unresolvable case fails toward parking, not toward
    the loop.
    """
    if returncode < 0:
        return 128 + abs(returncode)
    if returncode == _GATEWAY_FATAL_CONFIG_EXIT_CODE:
        if _gateway_argv_matcher() is None or _is_hermes_gateway_run_argv(command):
            return 0
    return returncode


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command, timestamp each stderr line into a log file and each stdout line to stdout."
    )
    parser.add_argument("--error-log", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    log_path: Path = args.error_log

    try:
        proc = subprocess.Popen(
            _prepare_child_command(args.command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, **_child_launchd_label_env()},
        )
    except OSError as exc:
        with _open_log(log_path) as log_file:
            _write_timestamped_line(log_file, f"failed to start stderr-timestamped command: {exc}")
        return 127

    assert proc.stdout is not None and proc.stderr is not None
    stdout_pump = threading.Thread(target=_copy_stdout_with_timestamps, args=(proc.stdout,), daemon=True)
    stdout_pump.start()
    previous_handlers = _install_signal_forwarders(proc)
    try:
        _copy_stderr_with_timestamps(proc.stderr, log_path)
        # Keep forwarding until the child has actually exited: a signal that lands between
        # its stderr EOF and wait() would otherwise kill the wrapper with the default action.
        returncode = proc.wait()
        stdout_pump.join(_STDOUT_DRAIN_TIMEOUT_S)
    finally:
        proc.stderr.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return _child_returncode_for_supervisor(args.command, returncode)


if __name__ == "__main__":
    sys.exit(main())
