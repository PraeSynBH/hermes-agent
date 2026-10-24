"""PRA-6850: the demand sweep must cost one stat per unchanged profile, not a parse.

Every 30 s the reconcile tick re-reads each named profile's cron store and `.env` to rank the
ceiling. On a gateway carrying ~950 threads each of those reads is a GIL round trip, so the sweep —
not the arithmetic — was measured as the tick's floor (`profile_reconcile_served` p50 1.3 s over 42
profiles). The memo keyed on `(mtime_ns, size)` is what makes the sweep a stat pass; these pin both
halves of that contract: an unchanged file is NOT re-read, and a rewritten one IS (a cron job
enabled since the last tick has to reach the ranking).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import gateway_multiplex_demand as demand


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "worker").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    demand.reset()
    yield home
    demand.reset()


def _write_jobs(home: Path, enabled: int) -> Path:
    path = home / "cron" / "jobs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    jobs = [{"id": f"on-{i}", "enabled": True} for i in range(enabled)]
    path.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    return path


def _write_env(home: Path, keys: int = 2) -> Path:
    path = home / ".env"
    path.write_text("".join(f"TELEGRAM_BOT_TOKEN_{i}=tok-{i}\n" for i in range(keys)),
                    encoding="utf-8")
    return path


def _bump(path: Path, *, pad: str = "") -> None:
    """Rewrite *path* so its signature changes without waiting for a clock tick."""
    path.write_text(path.read_text(encoding="utf-8") + pad, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def test_an_unchanged_cron_store_is_not_re_read(profile_env, monkeypatch):
    home = profile_env / "profiles" / "worker"
    _write_jobs(home, 3)

    assert demand._cron_jobs(home) == (3, ())

    reads = []
    real_read = demand._read_cron_jobs
    monkeypatch.setattr(demand, "_read_cron_jobs", lambda path: reads.append(path) or real_read(path))
    # The tick's steady state: the same file, stat'd again, parsed never.
    assert demand._cron_jobs(home) == (3, ())
    assert demand._cron_jobs(home) == (3, ())
    assert reads == [], "an unchanged cron store was parsed again on the tick"


def test_a_rewritten_cron_store_reaches_the_ranking(profile_env):
    home = profile_env / "profiles" / "worker"
    path = _write_jobs(home, 1)
    assert demand._cron_jobs(home) == (1, ())

    # A job enabled since the last tick must not stay invisible behind the memo.
    _write_jobs(home, 4)
    _bump(path)
    assert demand._cron_jobs(home) == (4, ())


def test_an_unchanged_env_is_not_re_read(profile_env, monkeypatch):
    home = profile_env / "profiles" / "worker"
    _write_env(home)
    keys = {"TELEGRAM_BOT_TOKEN_0": "telegram", "TELEGRAM_BOT_TOKEN_1": "telegram"}

    assert demand._credentials(home, keys) == (("telegram",), ())

    reads = []
    real_read = demand._read_credentials
    monkeypatch.setattr(demand, "_read_credentials",
                        lambda path, table: reads.append(path) or real_read(path, table))
    assert demand._credentials(home, keys) == (("telegram",), ())
    assert reads == [], "an unchanged .env was parsed again on the tick"


def test_a_rewritten_env_is_re_read(profile_env):
    home = profile_env / "profiles" / "worker"
    path = _write_env(home)
    keys = {"TELEGRAM_BOT_TOKEN_0": "telegram"}
    assert demand._credentials(home, keys) == (("telegram",), ())

    path.write_text("DISCORD_BOT_TOKEN=other\n", encoding="utf-8")
    _bump(path)
    assert demand._credentials(home, keys) == ((), ()), "a credential removed since the last tick"


def test_a_corrupt_store_still_counts_as_unreadable_demand(profile_env):
    """The memo must not turn a corrupt file into `0 jobs` (the fail-open this sweep forbids)."""
    home = profile_env / "profiles" / "worker"
    path = home / "cron" / "jobs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    assert demand._cron_jobs(home) == (0, ("cron",))
    assert demand._cron_jobs(home) == (0, ("cron",)), "the unreadable verdict is memoised, not lost"


def test_an_absent_store_is_empty_not_unreadable(profile_env):
    home = profile_env / "profiles" / "worker"
    assert demand._cron_jobs(home) == (0, ())
    assert demand._credentials(home, {"K": "telegram"}) == ((), ())
