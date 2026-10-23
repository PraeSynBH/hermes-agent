"""A profile holding a QUEUED run keeps its ceiling slot against a sighting pin (PRX-6000).

The 2026-10-22 incident, replayed at the seam the reconcile uses: the host booted with ``release``
served at a 17-profile ceiling and, 98 s later, the first rescan deferred it for ``cmo``:

```
20:27:17Z gateway boot (17 served, release in)
20:28:55Z [MULTIPLEX] No longer serving profile 'release' — gateway.multiplex_max_profiles defers it
20:47:42Z [MULTIPLEX] Now serving profile 'release' (0 adapter(s) connected; watcher)
```

``release`` held two runs in Paperclip's queue through that window (``heartbeat_runs.status='queued'``,
created 20:00 and 20:04, cancelled an hour later as never-started), and ``cmo`` was pinned by a live
pending-admission sighting — class 0. The rank could not see release's queue at all because the
demand collector reported only ``running`` runs as ``busy``, so release ranked as merely assigned
(class 1) and lost the tie. Ten Release Engineer runs expired and none succeeded.

Both sides are "a client is waiting" now that the manifest's ``queued`` count is read
(``hermes_cli.gateway_multiplex_demand``), so the operator's ``gateway.multiplex_serve_priority``
decides — which is what keeps the deploy path served. This is the acceptance test the issue asks
for: boot with the profile served, then assert it is NEVER absent from ``served_profiles`` across
the following five minutes of rescans, rather than a point-in-time reading after the fact.
"""
import json
from datetime import datetime, timedelta, UTC

import pytest
from unittest.mock import patch

from tests.gateway.test_multiplex_hot_serve import (
    _boot,
    _capped_runner,
    _cron_job,
    _mkprofile,
    _served_record,
    _state,
)


def _manifest(home, profiles):
    """Write the host demand manifest the collector would write for *profiles*."""
    written = datetime.now(UTC) - timedelta(seconds=5)
    (home / "gateway_demand.json").write_text(json.dumps({
        "schema": 1,
        "written_at": written.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "test",
        "ttl_seconds": 900,
        "profiles": profiles,
    }), encoding="utf-8")


async def _boot_with_queued_runs(tmp_path, monkeypatch, cap=2):
    """Boot at *cap* with ``release`` holding two QUEUED runs and ``cmo`` merely assigned."""
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        runner, home = _capped_runner(tmp_path, monkeypatch, cap=cap)
        (home / "config.yaml").write_text(
            f"gateway:\n  multiplex_max_profiles: {cap}\n"
            # The host's own tiebreak (PRX-5970): release ahead of cmo.
            "  multiplex_serve_priority: [release, cmo]\n", encoding="utf-8")
        _mkprofile(home, "release", "DISCORD_BOT_TOKEN=release-token\n")
        _mkprofile(home, "cmo", "DISCORD_BOT_TOKEN=cmo-token\n")
        _manifest(home, {"release": {"agents": 2, "queued": 2}, "cmo": {"agents": 3}})
        _boot(runner, home)
        await runner._start_secondary_profile_adapters()
    assert _served_record(home) == ["default", "release"]
    return runner, home


@pytest.mark.asyncio
async def test_the_boot_served_profile_is_never_absent_across_five_minutes(tmp_path, monkeypatch):
    """The issue's acceptance test, in CI: 10 rescans × 30 s, the served profile never drops."""
    from hermes_cli import gateway_multiplex_demand as demand

    runner, home = await _boot_with_queued_runs(tmp_path, monkeypatch)
    # A client is now blocked on cmo (PRX-5738's live sighting, class 0) — the exact 20:28Z shape.
    demand.note_wanted("cmo")

    seen = []
    for _tick in range(10):
        await runner.reconcile_served_profiles(reason="watcher")
        seen.append(list(_served_record(home)))

    assert all("release" in served for served in seen), seen
    assert _served_record(home) == ["default", "release"]
    parked = _state(home)["parked_profiles"]
    assert parked["cmo"].startswith("capacity: ")
    # The reason names the waiting work on cmo — the sighting is not silent either.
    assert "a client is waiting (pending admission)" in parked["cmo"]


@pytest.mark.asyncio
async def test_the_slot_is_the_rank_s_when_the_queue_drains(tmp_path, monkeypatch):
    """Not a permanent pin: once release has nothing waiting, the sighting's profile takes the slot.

    A queued run is a claim while it exists, not a lease on the ceiling — the same rule the
    live sighting has (PRX-5738's TTL), applied to the dispatcher's queue.
    """
    from hermes_cli import gateway_multiplex_demand as demand

    runner, home = await _boot_with_queued_runs(tmp_path, monkeypatch)
    demand.note_wanted("cmo")
    _manifest(home, {"release": {"agents": 2, "queued": 0}, "cmo": {"agents": 3}})

    result = await runner.reconcile_served_profiles(reason="watcher")

    assert result["removed"] == ["release"], result
    assert result["added"] == ["cmo"], result
    assert _served_record(home) == ["default", "cmo"]
    assert _state(home)["parked_profiles"]["release"].startswith("capacity: ")
