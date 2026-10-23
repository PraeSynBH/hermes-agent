"""Process identity: spawn tags, the machine-wide spawn ledger, and the Windows job-object self-attach.

Three layers make every long-lived Hermes process positively identifiable, so reapers (``hermes
update``, Desktop startup sweeps) never guess lineage from PPID archaeology or cmdline matching:
1. spawn tags (``HERMES_SPAWN`` env stamped by the spawner); 2. a ``(pid, create_time)`` ledger;
3. Windows job-object self-attach with ``KILL_ON_JOB_CLOSE`` so the whole child tree dies with the
root — no launcher→worker chains left holding ``.pyd`` locks.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import signal
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

from utils import atomic_json_write

logger = logging.getLogger(__name__)

SPAWN_ENV_VAR = "HERMES_SPAWN"
_TAG_VERSION = "v1"
LEDGER_FILENAME = "spawn-ledger.json"

#: Purposes of processes that exist ONLY to serve the spawner that created them: a stdio MCP
#: helper, a browser-harness daemon a browser_exec call caused to start, a private cua-driver
#: serve daemon. None has an independent reason to outlive its spawner, so a provably dead
#: spawner is the WHOLE reap proof (no second liveness fact, unlike serve/dashboard/gateway).
#: Every spawn site that starts one registers it through :func:`register_child` with one of these
#: purpose strings; the fence below is what makes the startup reconciler cover it (PRA-6594/6601).
HELPER_PURPOSES = frozenset({"mcp-helper", "browser-harness-daemon", "cua-driver-daemon"})

#: Purposes a reaper may treat as "safe to kill when the owner is gone".
#: Interactive processes (chat, REPLs) are deliberately NOT in this set.
REAPABLE_PURPOSES = frozenset({"serve", "dashboard", "gateway"}) | HELPER_PURPOSES

#: Host-role owners a fresh backend may reap on a HELD_BY_OTHER conflict (#121964): the
#: serve/dashboard/gateway subset. The helper rung stays owned by the proven-helper sweep.
_BACKEND_OWNER_PURPOSES = REAPABLE_PURPOSES - HELPER_PURPOSES

_IS_WINDOWS = platform.system() == "Windows"

# Module-global job handle: must live exactly as long as this process so the
# kernel closes it (and kills the job) when we die. Never close it manually.
_JOB_HANDLE = None
_LEDGER_LOCK = threading.Lock()


def install_id(project_root: Optional[Path] = None) -> str:
    """Stable 12-hex identifier for THIS install (derived from its path)."""
    if project_root is None:
        try:
            from hermes_constants import PROJECT_ROOT as _root

            project_root = Path(_root)
        except Exception:
            project_root = Path(__file__).resolve().parent.parent
    try:
        canonical = str(Path(project_root).resolve()).lower()
    except OSError:
        canonical = str(project_root).lower()
    return hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest()[:12]


def process_birth(proc) -> float:
    """A managed worker's identity birth for the ``psutil.Process`` *proc*, the same in every process.

    On macOS psutil shifts ``create_time()`` by ``abs(INIT_BOOT_TIME - boot_time())``
    (``psutil._psosx.adjust_proc_create_time``), with a boot-time reference frozen when psutil
    was imported in the CALLING process. After a clock step (NTP, sleep/wake) a long-lived
    gateway owner and a freshly spawned worker read different births for the same process, and
    the exact identity match refused every managed worker (#140580). The kernel's own start
    (``monotonic=True``: the unadjusted ``kinfo_proc`` epoch start) is what every process sees.
    Other platforms keep ``create_time()``: Linux and Windows apply no per-process reference.
    Raises ``psutil.Error`` like ``create_time()``.
    """
    import psutil
    if getattr(psutil, "MACOS", False):
        platform_proc = getattr(proc, "_proc", None)
        if platform_proc is not None:
            try:
                return float(platform_proc.create_time(monotonic=True))
            except TypeError:  # a psutil without the monotonic read: fall back to the adjusted one
                pass
    return float(proc.create_time())


def _process_create_time(pid: Optional[int] = None) -> Optional[float]:
    """:func:`process_birth` of ``pid`` (default: this process); ``None`` when psutil can't say."""
    try:
        import psutil

        return process_birth(psutil.Process(os.getpid() if pid is None else pid))
    except Exception:
        return None


# Layer 1 — spawn tags


@dataclass(frozen=True)
class SpawnTag:
    install: str
    purpose: str
    spawner_pid: int
    spawner_create: Optional[float]


def build_spawn_tag(purpose: str, *, project_root: Optional[Path] = None) -> str:
    """Value for the child's ``HERMES_SPAWN`` env var, stamped by the spawner."""
    create = _process_create_time()
    create_part = f"{create:.3f}" if create is not None else "-"
    return ":".join((_TAG_VERSION, install_id(project_root), purpose, str(os.getpid()), create_part))


def spawn_env(purpose: str, *, project_root: Optional[Path] = None) -> dict[str, str]:
    """Env fragment a spawner merges into a child's environment."""
    return {SPAWN_ENV_VAR: build_spawn_tag(purpose, project_root=project_root)}


def parse_spawn_tag(raw: object) -> Optional[SpawnTag]:
    """Parse a ``HERMES_SPAWN`` value; ``None`` for anything malformed."""
    parts = raw.split(":") if isinstance(raw, str) else []
    if len(parts) != 5 or parts[0] != _TAG_VERSION:
        return None
    _, install, purpose, pid_s, create_s = parts
    if not install or not purpose:
        return None
    try:
        pid = int(pid_s)
        create = None if create_s == "-" else float(create_s)
    except ValueError:
        return None
    return SpawnTag(install, purpose, pid, create) if pid > 0 else None


# Layer 2 — spawn ledger


@dataclass
class LedgerEntry:
    pid: int
    create_time: Optional[float]
    purpose: str
    install: str
    spawner_pid: Optional[int]
    spawner_create: Optional[float]
    registered_at: float
    argv: str
    # Structured launch identity a relauncher needs after an update, without parsing argv. Empty
    # for purposes that don't supply it; readers must use .get() — older ledger files predate these.
    host: str = ""
    port: Optional[int] = None
    profile: str = ""
    hermes_home: str = ""
    # `serve --isolated`: opted out of the host singleton (Desktop's SSH backend for another
    # machine). Attach-first readers must never adopt it; argv is truncated, so this is canonical.
    isolated: bool = False
    # Session ownership: the session ID this child was spawned for (if any) and its process group.
    # The session teardown kills the whole group, so a restart wrapper's successor goes with it.
    session_id: Optional[str] = None
    pgid: Optional[int] = None


def _ledger_path() -> Path:
    """Machine-root ledger path (shared by every profile of this install)."""
    try:
        from hermes_constants import get_default_hermes_root

        return Path(get_default_hermes_root()) / LEDGER_FILENAME
    except Exception:
        from hermes_cli.config import get_hermes_home

        return Path(get_hermes_home()) / LEDGER_FILENAME


def _lock_handle(handle) -> None:
    """Take an exclusive OS lock on the sidecar handle: ``flock``, or a byte lock on Windows."""
    if os.name == "nt":
        import msvcrt
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:  # msvcrt can only lock a byte that exists
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    import fcntl
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_handle(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


@contextlib.contextmanager
def _ledger_file_lock(path: Path):
    """Cross-PROCESS exclusion for the ledger read-prune-write, held on a sidecar ``flock``.

    ``_LEDGER_LOCK`` is a ``threading.Lock`` — it serializes one interpreter. Two processes
    registering at the same moment (a gateway booting while agent shells start) each read the
    roster, append their own entry and write the whole array back, so whichever writes second
    erases the other's entry. The victim is invisible afterwards: measured on this host
    2026-10-20, the ledger held 20 entries and NOT ONE ``gateway`` while six gateways were
    running, and every ledger-driven reaper needs the entry to prove an orphan — without it
    ``spawner_is_dead`` answers ``None``, which no reaper may act on (PRA-6594).

    Best-effort on purpose: a lock that cannot be taken (an exotic filesystem, a Windows
    ``msvcrt`` retry timeout) must never make registration fail, so the write proceeds
    unserialized exactly as it did before. Callers hold ``_LEDGER_LOCK`` first — thread lock
    outside, file lock inside, so a process never queues its own threads on the same flock.
    """
    lock_path = path.with_name(path.name + ".lock")
    handle = None
    try:
        from hermes_constants import mkdir_under_hermes_home

        mkdir_under_hermes_home(lock_path.parent)
        handle = lock_path.open("a+b")
        _lock_handle(handle)
    except OSError:
        if handle is not None:
            handle.close()
        handle = None
        logger.debug("spawn ledger lock unavailable; registering unserialized", exc_info=True)
    try:
        yield
    finally:
        if handle is not None:
            _unlock_handle(handle)
            handle.close()


def _read_ledger(path: Path) -> Optional[list[dict]]:
    """Entries list, ``[]`` for empty/missing, ``None`` for CORRUPT (never silently an empty roster).

    Mirrors the #89298 contract: corrupt is a distinct state that must never be silently treated as an empty
    roster.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError):
        return None
    if not text.strip():
        return []
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return [e for e in parsed if isinstance(e, dict)] if isinstance(parsed, list) else None


def _read_ledger_or_quarantine(path: Path) -> Optional[list[dict]]:
    """Ledger entries; ``None`` after parking a corrupt file. Caller holds ``_LEDGER_LOCK``."""
    entries = _read_ledger(path)
    if entries is None:
        parked = path.with_suffix(path.suffix + ".corrupt")
        try:
            os.replace(path, parked)
            logger.warning("spawn ledger was unreadable; moved to %s", parked)
        except OSError:
            pass
    return entries


def _same_incarnation(proc, create_time: Optional[float]) -> bool:
    """Does the live ``proc`` match a recorded ``create_time`` (2 s tolerance; ``None`` matches)?"""
    return create_time is None or abs(process_birth(proc) - float(create_time)) < 2.0


def _pid_alive_matches(pid: int, create_time: Optional[float], *, strict: bool = False) -> Optional[bool]:
    """True/False when provable; ``None`` when psutil can't say."""
    try:
        import psutil
    except Exception:
        return None
    try:
        proc = psutil.Process(int(pid))
        if strict:
            return (create_time is not None and process_birth(proc) == create_time
                    and proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE)
        # A zombie keeps its create_time until its parent reaps it, but it is already dead: the
        # dashboard stop check (``gateway.status._pid_exists``) books it stopped, so the ledger must
        # not report it as a live pre-update survivor. Windows has no zombies (and status() is slow there).
        return _same_incarnation(proc, create_time) and (
            os.name == "nt" or proc.status() != getattr(psutil, "STATUS_ZOMBIE", "zombie"))
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return None


def register_self(purpose: str, *, project_root: Optional[Path] = None, detail: Optional[dict] = None,
                  session_id: Optional[str] = None, pgid: Optional[int] = None) -> bool:
    """Record this process in the machine spawn ledger. Best-effort.

    Called at the top of every long-lived entry point; dead ``(pid, create_time)`` entries are
    pruned on every write. ``detail`` may carry ``host``/``port``/``profile`` so the update
    pipeline can relaunch a manually-started serve with its real bind address, and ``isolated``
    so attach-first discovery skips a backend that opted out of the host singleton.
    ``session_id``/``pgid`` record the owning session and process group so a session teardown can
    killpg the whole tree when the turn is killed.
    """
    from hermes_constants import hermes_home_key

    tag = parse_spawn_tag(os.environ.get(SPAWN_ENV_VAR))
    spawner_pid, spawner_create = (tag.spawner_pid, tag.spawner_create) if tag else _desktop_spawner_identity()
    if pgid is None and not _IS_WINDOWS:
        try:
            pgid = os.getpgid(os.getpid())
        except OSError:
            pgid = None
    entry = _new_entry(os.getpid(), _process_create_time(), purpose, project_root, spawner_pid, spawner_create,
                       session_id=session_id, pgid=pgid)
    entry.hermes_home = hermes_home_key()
    if detail:
        try:
            entry.host = str(detail.get("host") or "")
            entry.port = int(detail["port"]) if detail.get("port") is not None else None
            entry.profile = str(detail.get("profile") or "")
            entry.isolated = bool(detail.get("isolated"))
        except (TypeError, ValueError):
            pass
    try:
        import sys as _sys

        # 10 tokens: enough for `hermes serve --host X --port N --profile P` while bounding
        # pathological argv. Structured detail is canonical; argv is the human-readable fallback.
        entry.argv = " ".join(_sys.argv[:10])
    except Exception:
        pass
    return _append_entry(entry)


def is_desktop_owned_backend(argv: Optional[Sequence[str]] = None) -> bool:
    """Whether this process is the backend Desktop spawned and owns.

    ``HERMES_DESKTOP=1`` is inherited by every shell and agent child the app launches, so the
    flag alone is not ownership proof (same class as #116107). Desktop hands its backend a
    per-spawn credential the terminal pane never receives (and the terminal tool's env policy
    strips from agent children): the local pool spawn mints ``HERMES_DASHBOARD_SESSION_TOKEN``,
    the SSH spawn passes a 0600 token FILE on argv and deliberately sets no token env var.
    ``argv`` defaults to this process's own.
    """
    if os.environ.get("HERMES_DESKTOP") != "1":
        return False
    if os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN"):
        return True
    from hermes_cli._startup_fast import is_desktop_ssh_backend_argv

    return is_desktop_ssh_backend_argv(list(sys.argv[1:] if argv is None else argv))


# Set while this process works for the Desktop app on its own machine: one request from, or one turn of
# a chat created by, a client ``local_desktop_client`` accepted. The gateway serves the Desktop, the CLI,
# the TUI and bots from one process, so this is a property of the activity, never of the process.
_SERVING_LOCAL_DESKTOP: ContextVar[bool] = ContextVar("hermes_serving_local_desktop", default=False)


@contextmanager
def serving_local_desktop(active: bool) -> Iterator[None]:
    """Mark the enclosed activity (and threads it hands its context to) as work for the local Desktop."""
    token = _SERVING_LOCAL_DESKTOP.set(bool(active) or _SERVING_LOCAL_DESKTOP.get())
    try:
        yield
    finally:
        _SERVING_LOCAL_DESKTOP.reset(token)


def serves_local_desktop_now() -> bool:
    return _SERVING_LOCAL_DESKTOP.get()


def local_desktop_client(identity: Optional[dict], declared: object) -> bool:
    """A connected client is the Desktop app on this machine when it says so (it knows its connection
    mode) AND it redeemed a native ticket, which only this machine's private control socket mints, for
    this OS user. Neither alone: the CLI and TUI hold native tickets too, and the claim is the client's
    word. A remote Desktop reaches a backend through SSH, URL or Cloud credentials, never a native ticket."""
    return (declared is True and isinstance(identity, dict) and identity.get("provider") == "local"
            and identity.get("native_bootstrap") is True)


def _desktop_spawner_identity() -> tuple[Optional[int], Optional[float]]:
    """Spawner ``(pid, create_time)`` from the Electron app's HERMES_PARENT_PID (+ optional
    ``winms:<ms>`` start marker) parent-death watchdog vars, so ledger lineage works with every
    Desktop version without a TS change. ``(None, None)`` when absent/malformed."""
    try:
        spawner_pid = int(os.environ.get("HERMES_PARENT_PID", ""))
    except (TypeError, ValueError):
        spawner_pid = 0
    if spawner_pid <= 0:
        return None, None
    marker = os.environ.get("HERMES_PARENT_START_MARKER", "")
    if not marker.startswith("winms:"):
        return spawner_pid, None
    try:
        return spawner_pid, float(marker.split(":", 1)[1]) / 1000.0
    except (ValueError, IndexError):
        return spawner_pid, None


def _new_entry(
    pid: int, create_time: Optional[float], purpose: str, project_root: Optional[Path],
    spawner_pid: Optional[int], spawner_create: Optional[float],
    session_id: Optional[str] = None, pgid: Optional[int] = None,
) -> LedgerEntry:
    return LedgerEntry(
        pid, create_time, purpose, install_id(project_root), spawner_pid, spawner_create, time.time(), argv="",
        session_id=session_id, pgid=pgid,
    )


def _append_entry(entry: LedgerEntry) -> bool:
    """Prune dead entries and append ``entry`` — the ONLY ledger write path.

    Serialized under ``_LEDGER_LOCK`` (threads) AND ``_ledger_file_lock`` (processes) with an
    atomic tmp+replace; no writer touches the file outside this function. The flock is what keeps
    a concurrent registration from another process from erasing this one.

    See #91660.
    """
    path = _ledger_path()
    with _LEDGER_LOCK, _ledger_file_lock(path):
        entries = _read_ledger_or_quarantine(path) or []
        # Drop malformed entries, our own stale entry, and provably dead pids.
        pruned = [
            e for e in entries
            if isinstance(e.get("pid"), int)
            and e["pid"] != entry.pid
            and _pid_alive_matches(e["pid"], e.get("create_time")) is not False
        ]
        pruned.append(asdict(entry))
        try:
            from hermes_constants import mkdir_under_hermes_home
            mkdir_under_hermes_home(path.parent)
            # argv may carry surrogate-escaped bytes (non-UTF-8 paths); ensure_ascii keeps the
            # utf-8 text handle from raising UnicodeEncodeError (a ValueError, not an OSError).
            atomic_json_write(path, pruned, mode=0o600, ensure_ascii=True)
            return True
        except OSError:
            logger.debug("spawn ledger write failed", exc_info=True)
            return False


def register_child(pid: int, purpose: str, *, project_root: Optional[Path] = None,
                   session_id: Optional[str] = None, pgid: Optional[int] = None) -> bool:
    """Record a CHILD process this process just spawned. Best-effort.

    Mirror of :func:`register_self` for children that cannot register themselves (stdio MCP
    helpers: arbitrary ``npx``/binary servers never import Hermes code). Records the child's
    ``(pid, create_time)`` with THIS process as spawner, so a helper whose spawner is provably gone
    is a reapable orphan and one whose spawner is alive is never reaped. ``session_id``/``pgid``
    record the owning session and the group to kill on session teardown; ``pgid`` defaults to the
    child's own group when it leads one (``start_new_session``), matching the reap contract.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    child_create = _process_create_time(pid) if pid > 0 else None
    if child_create is None:
        return False
    if pgid is None and not _IS_WINDOWS:
        try:
            pgid = os.getpgid(pid)
        except OSError:
            pgid = None
    entry = _new_entry(pid, child_create, purpose, project_root, os.getpid(), _process_create_time(),
                       session_id=session_id, pgid=pgid)
    try:
        import psutil

        entry.argv = " ".join(psutil.Process(pid).cmdline()[:10])
    except Exception:
        pass
    return _append_entry(entry)


def ledger_entries(
    *, project_root: Optional[Path] = None, all_installs: bool = False, verified_only: bool = False,
) -> list[dict]:
    """Ledger entries for this install, or all installs when explicitly requested.

    ``verified_only`` requires an exact live PID/create-time pair. The default preserves
    unknown processes for reapers, which must not mistake missing proof for a dead process.

    Entries whose ``(pid, create_time)`` no longer matches a live process are excluded (PID reuse reads as
    dead, thanks to the create-time pair). A corrupt ledger is quarantined and read as empty — identical
    philosophy to the backend-ownership fix (#89298): never let corruption erase or fake a roster; never let
    it block the caller either.
    """
    want_install = install_id(project_root)
    with _LEDGER_LOCK:
        entries = _read_ledger_or_quarantine(_ledger_path())
    if entries is None:
        return []
    return [
        e for e in entries
        if (all_installs or e.get("install") == want_install)
        and isinstance(e.get("pid"), int)
        and (_pid_alive_matches(e["pid"], e.get("create_time"), strict=True) is True
             if verified_only else _pid_alive_matches(e["pid"], e.get("create_time")) is not False)
    ]


def ledger_entries_for_sessions(session_ids: Sequence[str], *,
                               project_root: Optional[Path] = None) -> list[dict]:
    """Live ledger entries whose recorded ``session_id`` is in *session_ids*.

    The session teardown calls this to find the process groups a turn spawned. An empty
    ``session_ids`` returns nothing: never a sweep of "every entry without a session".
    """
    if not session_ids:
        return []
    wanted = set(session_ids)
    try:
        entries = ledger_entries(project_root=project_root)
    except Exception:
        logger.debug("could not read the spawn ledger for session reap", exc_info=True)
        return []
    return [e for e in entries if e.get("session_id") in wanted]


def spawner_is_dead(entry: dict) -> Optional[bool]:
    """Is the recorded spawner of this entry provably gone? ``None`` when unrecorded/unprovable."""
    spawner_pid = entry.get("spawner_pid")
    if not isinstance(spawner_pid, int) or spawner_pid <= 0:
        return None
    alive = _pid_alive_matches(spawner_pid, entry.get("spawner_create"))
    return None if alive is None else not alive


def _terminate_then_kill(pid: int, create_time: Optional[float]) -> bool:
    """SIGTERM, 2 s grace, then SIGKILL. False when the incarnation moved on or is already gone."""
    try:
        import psutil

        proc = psutil.Process(pid)
        if not _same_incarnation(proc, create_time):
            return False  # PID reused since registration — never signal a stranger
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except psutil.TimeoutExpired:
            proc.kill()
        return True
    except Exception:
        logger.debug("orphan terminate failed for pid %s", pid, exc_info=True)
        return False


def _home_of(entry: dict) -> str:
    """The home an entry serves, or ``"-"`` — the greppable field of every reconcile log line."""
    return str(entry.get("hermes_home") or "") or "-"


def _reap_reconciled(entry: dict, kill_fn) -> Optional[dict]:
    """Reap one PROVEN orphan and log it; ``None`` when the incarnation moved on or the kill failed.

    One log line per reap, ``/reconciled orphaned/``-greppable, carrying pid, purpose and home so
    the next host census can be read against it.
    """
    pid = entry.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        if kill_fn is not None:
            kill_fn(pid)
        elif not _terminate_then_kill(pid, entry.get("create_time")):
            return None
    except Exception:
        logger.debug("orphan reap failed for %s", entry, exc_info=True)
        return None
    record = {"pid": pid, "purpose": str(entry.get("purpose") or ""), "hermes_home": _home_of(entry)}
    logger.info(
        "reconciled orphaned %s pid=%s home=%s spawner=%s",
        record["purpose"], pid, record["hermes_home"], entry.get("spawner_pid"),
    )
    return record


def _reparented_orphan(pid: int) -> bool:
    """Null-spawner owner with no live supervisor: reparented to init, old, lock-unclaimed.

    Mirrors ``_reap_orphaned_desktop_local_serves``: operator-managed remote backends legitimately
    sit at ppid 1 (systemd, an exited sshd), so ppid alone is never proof — the SSH-lock claim
    and the 180 s lock-write grace exclude live-supervised backends. Windows: the ppid probe is
    unavailable there, so this rung never fires (desktop tree-kill reaps instead).
    """
    try:
        from hermes_cli.dashboard_procs import (
            _REAP_MIN_AGE_SECONDS, _lock_owned_serve_pids, _process_ppid)
        if _process_ppid(pid) not in (0, 1):
            return False
        if pid in (_lock_owned_serve_pids() or ()):
            return False
        import time as _time

        import psutil
        return max(0.0, _time.time() - psutil.Process(pid).create_time()) >= _REAP_MIN_AGE_SECONDS
    except Exception:
        return False  # unprovable → never touch


def reap_orphaned_backend_owner(
    pid: int, create_time: Optional[float], *, kill_fn=None) -> Optional[int]:
    """Kill the conflicting host owner when the ledger PROVES it is a dead session's orphan.

    HELD_BY_OTHER rung for #121964: the owner PID is alive (re-parented to init), so
    ``record_is_stale()`` never fires and a fresh backend would loop observe-only against the
    same orphan forever. Reaped only with a live ``serve``/``dashboard``/``gateway`` entry for
    THIS install whose spawner is provably dead — or whose spawner was never recorded and which
    is reparented to init, old, and claimed by no Desktop SSH lock. Live-spawned, lock-claimed,
    young, interactive, self, or unprovable owners are never touched. Returns the reaped pid,
    else ``None``; never raises.

    # ponytail: a dead session's surviving backend.lock.json still shields its orphan (the lock
    # has no session liveness); clearing stale locks is a Desktop-side lifecycle job, not this rung.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0 or pid == os.getpid():
        return None
    try:
        entry = next(
            (e for e in ledger_entries()
             if e.get("pid") == pid
             and e.get("purpose") in _BACKEND_OWNER_PURPOSES
             and (e.get("create_time") is None or create_time is None
                  or abs(float(e["create_time"]) - float(create_time)) < 2.0)),
            None)
    except Exception:
        return None
    if entry is None:
        return None
    if spawner_is_dead(entry) is not True and not (
            entry.get("spawner_pid") is None and _reparented_orphan(pid)):
        return None  # live or unprovable spawner → never touch
    try:
        if kill_fn is not None:
            kill_fn(pid)
        elif not _terminate_then_kill(pid, entry.get("create_time")):
            return None
    except Exception:
        logger.debug("backend owner reap failed for %s", entry, exc_info=True)
        return None
    logger.info("reaped orphaned %s backend owner pid %s", entry.get("purpose"), pid)
    return pid


def _live_gateways_by_home() -> dict[str, list[dict]]:
    """Ledger entries that PROVE a live gateway, grouped by the home each one serves."""
    by_home: dict[str, list[dict]] = {}
    for entry in ledger_entries(verified_only=True):
        home = str(entry.get("hermes_home") or "")
        if entry.get("purpose") == "gateway" and home:
            by_home.setdefault(home, []).append(entry)
    return by_home


def _superseded_gateway_entries() -> list[dict]:
    """Ledger entries that are residue because a NEWER live gateway replaced them, oldest first.

    A detached gateway's dead spawner proves nothing on its own: ``spawn_unmanaged_gateway``
    detaches by design (the client exits), and the messaging gateway is meant to survive the app
    (gateway/AGENTS.md). ``ppid=1`` plus a dead spawner is the *healthy* shape of an auto-started
    daemon. The orphan signature is duplication — two live gateways claiming one home, which the
    runtime's own exclusive ownership fence exists to prevent — so only then is the older one
    residue.

    Never returned: this process, a home with a single live gateway, an unprovable entry
    (``ledger_entries(verified_only=True)`` needs a live ``(pid, create_time)`` pair) and any
    entry whose spawner is alive — a launchd/systemd/Desktop-supervised gateway always has one,
    so the supervised gateway is never a candidate however old it is.
    """
    candidates: list[dict] = []
    own = os.getpid()
    for entries in _live_gateways_by_home().values():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda e: (float(e.get("registered_at") or 0.0), int(e.get("pid") or 0)))
        for older in entries[:-1]:
            pid = older.get("pid")
            if not isinstance(pid, int) or pid <= 0 or pid == own:
                continue
            if spawner_is_dead(older) is not True:
                continue  # supervised, or unprovable → never touch
            candidates.append(older)
    return candidates


def reap_superseded_gateways(*, kill_fn=None) -> list[int]:
    """Kill the gateways :func:`_superseded_gateway_entries` proves a newer live one replaced."""
    return [
        record["pid"]
        for record in (_reap_reconciled(entry, kill_fn) for entry in _superseded_gateway_entries())
        if record is not None
    ]


def reap_orphaned_mcp_helpers(*, project_root: Optional[Path] = None, kill_fn=None) -> list[int]:
    """Kill ledger-registered stdio MCP helpers whose spawner is provably dead.

    Ledger-driven startup-sweep rung (not cmdline-heuristic): a helper is reaped ONLY when it has a
    live ``mcp-helper`` entry for THIS install AND ``spawner_is_dead`` is ``True`` — never
    ``None``/unprovable, never a live spawner. The startup reconciler runs the same sweep over
    every :data:`HELPER_PURPOSES` member.
    """
    return [
        record["pid"]
        for record in _reap_proven_helpers(project_root=project_root, kill_fn=kill_fn,
                                           purposes=frozenset({"mcp-helper"}))
    ]


def _reap_proven_helpers(*, project_root: Optional[Path] = None, kill_fn=None,
                         purposes: frozenset = HELPER_PURPOSES) -> list[dict]:
    """Reap the proven-orphan helpers, one ``{pid, purpose, hermes_home}`` record per kill."""
    reaped: list[dict] = []
    for entry in _helper_orphans(project_root=project_root, purposes=purposes):
        try:
            if kill_fn is not None:
                kill_fn(entry["pid"])
            elif not _reap_helper_process(entry):
                continue  # PID reused since registration
            logger.info(
                "reconciled orphaned %s pid=%s home=%s spawner=%s",
                entry.get("purpose"), entry["pid"], _home_of(entry), entry.get("spawner_pid"),
            )
            reaped.append({"pid": entry["pid"], "purpose": entry.get("purpose"), "hermes_home": _home_of(entry)})
        except Exception:
            logger.debug("helper orphan reap failed for %s", entry, exc_info=True)
    return reaped


def _helper_orphans(*, project_root: Optional[Path] = None,
                    purposes: frozenset = HELPER_PURPOSES) -> list[dict]:
    """Live ``purposes`` entries for THIS install whose recorded spawner is provably dead."""
    try:
        entries = ledger_entries(project_root=project_root)
    except Exception:
        logger.debug("could not read the spawn ledger for helper orphans", exc_info=True)
        return []
    own_pid = os.getpid()
    return [
        entry for entry in entries
        if entry.get("purpose") in purposes
        and isinstance(entry.get("pid"), int) and entry["pid"] > 0 and entry["pid"] != own_pid
        and spawner_is_dead(entry) is True
    ]


def _reap_helper_process(entry: dict) -> bool:
    """Terminate one proven-orphan helper; ``False`` when the PID's incarnation moved on.

    POSIX signals the helper's process GROUP when the helper leads one (every helper here is
    started with ``start_new_session``, so its pgid is its own pid): a helper's children — a harness
    daemon's browser, a stub's server — must go with it, and a helper that is really a
    ``while true; do …; done`` restart wrapper only bounces on a single SIGTERM (PRA-6601). The
    leader check is the safety fence: if the pgid is anyone else's, only this pid is signalled.
    """
    import psutil

    proc = psutil.Process(entry["pid"])
    if not _same_incarnation(proc, entry.get("create_time")):
        return False
    # Windows: descendants (npx.cmd → node.exe) have no pgid to group-kill and reparent with
    # ParentId=null when the direct child exits first (#61059), so reap the whole tree.
    if _IS_WINDOWS:
        _kill_process_tree_windows(proc)
        return True
    try:
        own_group = os.getpgid(entry["pid"]) == entry["pid"]
    except OSError:
        own_group = False

    def _signal(number: int) -> None:
        if not own_group:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(entry["pid"], number)  # windows-footgun: ok — POSIX branch only

    if own_group:
        _signal(signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except psutil.TimeoutExpired:
        pass  # a wrapper that bounces on SIGTERM, or a leader that ignores it: escalate below
    # Escalate to the GROUP even after the leader is gone: a restart wrapper's successor, or a
    # daemon's browser, is still in the group and would otherwise be an orphan of an orphan.
    _signal(signal.SIGKILL)
    if not own_group and proc.is_running():
        proc.kill()
    return True


#: Second liveness fact each purpose needs before a dead spawner means "orphan" — see
#: :func:`reconcile_orphans`. ``gateway`` is absent on purpose: it is judged by supersession.
_UNCLAIMED_HOME_PURPOSES = frozenset({"serve", "dashboard"})


def reconcile_orphans(*, kill_fn=None, reparented_fn=None) -> list[dict]:
    """Startup sweep: reap every ledger entry across ``REAPABLE_PURPOSES`` that is PROVEN residue.

    A dead recorded spawner is necessary but NOT sufficient for a long-lived entry, and getting
    that wrong kills healthy processes: ``spawn_unmanaged_gateway`` detaches by design (a gateway
    serving a live TUI at ``ppid=1`` with a dead spawner is *healthy*), and the messaging gateway
    is meant to survive its app (gateway/AGENTS.md). So each purpose carries its own second fact:

    * ``gateway`` — a NEWER live gateway already claims the same home (the runtime's exclusive
      ownership fence exists precisely to prevent that pair): :func:`_superseded_gateway_entries`;
    * ``serve`` / ``dashboard`` — the home claims no live backend for it: reparented to init, no
      valid Desktop SSH ``backend.lock.json``, and past the lock-write grace
      (:func:`_reparented_orphan`, the rung #121964 already uses for backend owners);
    * every :data:`HELPER_PURPOSES` member (``mcp-helper``, ``browser-harness-daemon``,
      ``cua-driver-daemon``) — strict ``spawner_is_dead``: a helper process exists only to serve
      the spawner that started it, so it has no independent reason to outlive it.

    Never this process, never a live spawner, never an unprovable entry (``None``), never an
    interactive purpose. Returns one ``{pid, purpose, hermes_home}`` record per reap; never raises.
    """
    reaped: list[dict] = []
    reparented_fn = _reparented_orphan if reparented_fn is None else reparented_fn
    try:
        entries = ledger_entries()
    except Exception:
        logger.debug("could not read the spawn ledger; reconciling nothing", exc_info=True)
        entries = []
    own = os.getpid()
    for entry in entries:
        purpose = entry.get("purpose")
        pid = entry.get("pid")
        if purpose not in _UNCLAIMED_HOME_PURPOSES or not isinstance(pid, int) or pid <= 0 or pid == own:
            continue
        if spawner_is_dead(entry) is not True:
            continue  # live or unprovable spawner → never touch
        try:
            if not reparented_fn(pid):
                continue  # a live owner still claims this backend
        except Exception:
            logger.debug("orphan proof failed for pid %s; leaving it alone", pid, exc_info=True)
            continue  # unprovable → never touch
        record = _reap_reconciled(entry, kill_fn)
        if record is not None:
            reaped.append(record)
    for entry in _superseded_gateway_entries():
        record = _reap_reconciled(entry, kill_fn)
        if record is not None:
            reaped.append(record)
    reaped.extend(_reap_proven_helpers(kill_fn=kill_fn))
    return reaped


# Layer 3 — Windows job-object self-attach


def _kill_process_tree_windows(proc) -> None:
    """Terminate *proc* and every still-alive descendant (npx.cmd → node.exe), Windows-only
    (#61059): without a pgid there is no group-kill, and grandchildren reparent to nothing
    (ParentId=null) once the direct child exits, so they must be reached through the tree."""
    import psutil

    try:
        descendants = proc.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        descendants = []
    for child in descendants:
        try:
            child.terminate()
        except Exception:
            pass
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        _, alive = psutil.wait_procs(descendants + [proc], timeout=2.0)
    except Exception:
        return
    for survivor in alive:
        try:
            survivor.kill()
        except Exception:
            pass


def attach_self_to_kill_on_close_job() -> bool:
    """Place this process in a job that dies (whole tree) when we die. Windows-only, idempotent.

    ``BREAKAWAY_OK`` keeps children spawned with ``CREATE_BREAKAWAY_FROM_JOB`` (gateway relaunch
    during update, detached watchers) escaping exactly as before.
    """
    global _JOB_HANDLE
    if not _IS_WINDOWS or _JOB_HANDLE is not None:
        return _JOB_HANDLE is not None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x0800
        JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x1000
        JobObjectExtendedLimitInformation = 9

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER), ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(wintypes.ULONG)), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                *((n, ctypes.c_size_t) for n in (
                    "ProcessMemoryLimit", "JobMemoryLimit", "PeakProcessMemoryUsed", "PeakJobMemoryUsed")),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return False
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
        )
        ok = kernel32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info))
        if not ok or not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            kernel32.CloseHandle(job)
            return False
        _JOB_HANDLE = job  # keep alive for the life of the process — never close
        logger.debug("attached to kill-on-close job object")
        return True
    except Exception:
        logger.debug("job object self-attach failed", exc_info=True)
        return False
