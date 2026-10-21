"""``gateway.multiplex_demand`` — ranking input for the multiplex profile ceiling (SOU-3075).

The ceiling used to be ranked by a hand-maintained priority list, so a profile that acquired its
first agent or cron job stayed deferred behind one that held nothing. These pin the three demand
sources, the manifest's staleness policy, and the two properties the hot path depends on: a snapshot
never re-reads once it exists (the serving path calls it per inbound route on the event loop), and a
source that cannot be read counts as demand (never as "no work").
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path

import pytest

from hermes_cli import gateway_multiplex_capacity as capacity
from hermes_cli import gateway_multiplex_demand as demand
from hermes_cli.profiles import create_profile, get_profile_dir


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    capacity.reset_memo()
    demand.reset()
    yield home
    capacity.reset_memo()
    demand.reset()


def _profile(name: str) -> Path:
    create_profile(name, no_alias=True)
    return get_profile_dir(name)


def _write_jobs(home: Path, enabled: int, disabled: int = 0, *, raw: str | None = None) -> None:
    (home / "cron").mkdir(exist_ok=True)
    if raw is not None:
        (home / "cron" / "jobs.json").write_text(raw, encoding="utf-8")
        return
    jobs = [{"id": f"on-{i}", "enabled": True} for i in range(enabled)]
    jobs += [{"id": f"off-{i}", "enabled": False} for i in range(disabled)]
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")


def _write_manifest(home: Path, profiles: dict, *, schema: int = 1, age_seconds: float = 0,
                    ttl_seconds: int = 900, raw: str | None = None) -> Path:
    path = home / "gateway_demand.json"
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return path
    written = datetime.now(UTC) - timedelta(seconds=age_seconds)
    path.write_text(json.dumps({
        "schema": schema,
        "written_at": written.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "test",
        "ttl_seconds": ttl_seconds,
        "profiles": profiles,
    }), encoding="utf-8")
    return path


class TestCronSource:
    def test_enabled_jobs_are_demand_and_disabled_ones_are_not(self, profile_env):
        busy = _profile("busy")
        quiet = _profile("quiet")
        _write_jobs(busy, enabled=2, disabled=3)
        _write_jobs(quiet, enabled=0, disabled=4)

        snapshot = demand.refresh()

        assert snapshot.for_profile("busy").cron_jobs == 2
        assert snapshot.for_profile("quiet").cron_jobs == 0
        assert snapshot.has("busy") is True
        assert snapshot.has("quiet") is False

    def test_a_profile_with_no_cron_store_holds_no_cron_demand(self, profile_env):
        _profile("fresh")
        snapshot = demand.refresh()
        assert snapshot.for_profile("fresh").cron_jobs == 0
        assert snapshot.for_profile("fresh").unreadable == ()

    def test_a_corrupt_cron_store_is_demand_bearing_not_zero(self, profile_env):
        """An unreadable source fails toward SERVING the profile (plan F7)."""
        broken = _profile("broken")
        _write_jobs(broken, enabled=0, raw='{"jobs": [')

        snapshot = demand.refresh()

        facts = snapshot.for_profile("broken")
        assert facts.cron_jobs == 0
        assert facts.unreadable == ("cron",)
        assert snapshot.has("broken") is True


class TestCredentialSource:
    def test_env_credentials_use_the_shared_key_table(self, profile_env):
        channelled = _profile("channelled")
        (channelled / ".env").write_text("DISCORD_BOT_TOKEN=discord-token\n", encoding="utf-8")
        _profile("bare")

        snapshot = demand.refresh()

        assert "discord" in snapshot.for_profile("channelled").credentials
        assert snapshot.has("channelled") is True
        assert snapshot.for_profile("bare").credentials == ()
        assert snapshot.has("bare") is False

    def test_an_unreadable_env_is_demand_bearing(self, profile_env):
        locked = _profile("locked")
        env_path = locked / ".env"
        env_path.write_text("DISCORD_BOT_TOKEN=discord-token\n", encoding="utf-8")
        os.chmod(env_path, 0o000)
        try:
            if os.access(env_path, os.R_OK):
                pytest.skip("file permissions do not bind for this user (root)")
            snapshot = demand.refresh()
        finally:
            os.chmod(env_path, 0o600)

        facts = snapshot.for_profile("locked")
        assert facts.unreadable == ("env",)
        assert snapshot.has("locked") is True


class TestManifest:
    def test_a_fresh_manifest_supplies_the_agent_counts(self, profile_env):
        _profile("devops")
        _write_manifest(profile_env, {"devops": {"agents": 2, "blocked_admission_15m": 6}})

        snapshot = demand.refresh()

        assert snapshot.manifest_state == demand.MANIFEST_OK
        assert snapshot.for_profile("devops").agents == 2
        assert snapshot.for_profile("devops").blocked_admission_15m == 6
        assert snapshot.demand_source()["state"] == "ok"

    def test_an_absent_manifest_leaves_the_agent_count_unknown(self, profile_env):
        quiet = _profile("quiet")
        _write_jobs(quiet, enabled=1)

        snapshot = demand.refresh()

        assert snapshot.manifest_state == demand.MANIFEST_ABSENT
        # Unknown is not zero: the cost clause must not claim a deferred profile holds nothing.
        assert snapshot.for_profile("quiet").agents is None
        assert snapshot.demand_source()["state"] == "absent"
        assert snapshot.has("quiet") is True  # cron still counts

    def test_a_stale_manifest_keeps_ranking_with_the_last_known_values(self, profile_env):
        _profile("general")
        _write_manifest(profile_env, {"general": {"agents": 12}}, age_seconds=1200, ttl_seconds=900)

        snapshot = demand.refresh()

        assert snapshot.manifest_state == demand.MANIFEST_STALE
        assert snapshot.for_profile("general").agents == 12
        assert snapshot.has("general") is True
        assert snapshot.demand_source()["age_seconds"] >= 1200

    @pytest.mark.parametrize("raw,schema", [
        ('{"schema": 1, "profiles": {', 1),
        ('{"schema": 2, "written_at": "2026-10-20T20:00:00Z", "profiles": {}}', 2),
        ('{"schema": 1, "written_at": "not-a-date", "profiles": {}}', 1),
        ('{"schema": 1, "written_at": "2026-10-20T20:00:00Z", "profiles": {"x": 3}}', 1),
    ])
    def test_a_malformed_manifest_degrades_to_cron_and_credentials(self, profile_env, raw, schema):
        broken = _profile("broken")
        _write_jobs(broken, enabled=1)
        _write_manifest(profile_env, {}, schema=schema, raw=raw)

        snapshot = demand.refresh()

        assert snapshot.manifest_state == demand.MANIFEST_MALFORMED
        assert snapshot.manifest_error
        assert snapshot.for_profile("broken").agents is None
        assert snapshot.has("broken") is True

    def test_a_manifest_naming_an_unknown_profile_is_reported_not_ranked(self, profile_env):
        _profile("real")
        _write_manifest(profile_env, {"real": {"agents": 1}, "generall": {"agents": 4}})

        snapshot = demand.refresh()

        assert snapshot.unknown_profiles == ("generall",)
        assert snapshot.has("generall") is False

    def test_the_configured_manifest_path_is_honoured(self, profile_env, tmp_path):
        _profile("devops")
        elsewhere = tmp_path / "elsewhere.json"
        _write_manifest(profile_env, {"devops": {"agents": 5}})
        elsewhere.write_text((profile_env / "gateway_demand.json").read_text(encoding="utf-8"),
                             encoding="utf-8")
        (profile_env / "config.yaml").write_text(
            f"gateway:\n  multiplex_demand_manifest: {elsewhere}\n", encoding="utf-8")
        capacity.reset_memo()

        snapshot = demand.refresh()

        assert snapshot.manifest_path == elsewhere
        assert snapshot.for_profile("devops").agents == 5


class TestColdSnapshot:
    """SOU-3125: no snapshot means a COLD view, never a read on the caller's thread.

    The first read of a process pays one-time plugin discovery (``credential_env_keys``: 15–22 s on
    a loaded host), and the two callers that reached it that way were cold boot and the event loop
    (``profiles_to_serve`` per inbound route). So ``snapshot()`` performs no I/O at all; the cold
    view is a claim about what this process has *not* looked at, and it says so.
    """

    def test_a_cold_snapshot_performs_no_io_and_says_so(self, profile_env, monkeypatch):
        busy = _profile("busy")
        _write_jobs(busy, enabled=2)
        _write_manifest(profile_env, {"busy": {"agents": 3}})

        def reading_refused(*_args, **_kwargs):
            raise AssertionError("snapshot() must not read")

        monkeypatch.setattr(demand, "refresh", reading_refused)
        monkeypatch.setattr(demand, "manifest_path", reading_refused)
        monkeypatch.setattr(demand, "_cron_jobs", reading_refused)

        view = demand.snapshot()

        assert view.manifest_state == demand.MANIFEST_COLD
        assert view.facts == {}
        assert view.captured_at is None
        assert view.has("busy") is False
        assert view.for_profile("busy").agents is None
        assert view.demand_source() == {"path": None, "state": "cold", "age_seconds": None}

    def test_refresh_populates_the_view_the_serving_path_then_reads(self, profile_env):
        busy = _profile("busy")
        _write_jobs(busy, enabled=1)
        cold = demand.snapshot()
        assert cold.manifest_state == demand.MANIFEST_COLD

        refreshed = demand.refresh()

        assert refreshed.has("busy") is True
        assert refreshed.manifest_state != demand.MANIFEST_COLD
        assert refreshed.captured_at is not None
        assert demand.snapshot() is refreshed
        assert demand.snapshot() is not cold

    def test_cold_ranking_is_exactly_the_no_demand_ordering(self, profile_env):
        """The boot-time regression: a cold view must reproduce ``demand=None``, not invent."""
        serve = [("default", Path("/h/default"))] + [(n, Path(f"/h/{n}")) for n in ("zulu", "alpha", "mike")]

        cold_kept, cold_deferred = capacity.apply_ceiling(serve, 2, ("zulu",), demand=demand.snapshot())
        none_kept, none_deferred = capacity.apply_ceiling(serve, 2, ("zulu",), demand=None)

        assert [n for n, _ in cold_kept] == [n for n, _ in none_kept] == ["default", "zulu"]
        assert set(cold_deferred) == set(none_deferred) == {"alpha", "mike"}


class TestSnapshotCost:
    def test_snapshot_never_reads_again_however_long_it_has_been_captured(self, profile_env, monkeypatch):
        """The serving path calls snapshot() per inbound route on the event loop: never a read."""
        busy = _profile("busy")
        _write_jobs(busy, enabled=1)
        reads: list[str] = []

        for name in ("_read_manifest", "_cron_jobs", "_credentials"):
            original = getattr(demand, name)

            def counting(*args, _name=name, _original=original, **kwargs):
                reads.append(_name)
                return _original(*args, **kwargs)

            monkeypatch.setattr(demand, name, counting)

        first = demand.refresh()
        performed = len(reads)
        assert performed > 0
        with monkeypatch.context() as clock:
            # Any clock-based window has long lapsed; the read path still touches nothing.
            clock.setattr(demand.time, "monotonic", lambda: 10 ** 9)
            clock.setattr(demand.time, "time", lambda: 10 ** 9)
            assert demand.snapshot() is first
        assert len(reads) == performed
        # A change lands on the tick's refresh, and only there.
        _write_jobs(busy, enabled=0)
        assert demand.snapshot().has("busy") is True
        assert demand.refresh().has("busy") is False

    def test_refresh_is_the_authoritative_read(self, profile_env):
        busy = _profile("busy")
        _write_jobs(busy, enabled=1)
        assert demand.snapshot().has("busy") is False  # cold: nothing has read yet

        assert demand.refresh().has("busy") is True

        _write_jobs(busy, enabled=0)
        demand.reset()

        assert demand.snapshot().has("busy") is False
        assert demand.refresh().has("busy") is False
