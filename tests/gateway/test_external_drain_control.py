"""Tests for the external drain-control marker contract + gateway state machine.

Task 2.2/2.3. Two layers:
  * drain_control.py — the presence-based marker contract (write/clear/read,
    HERMES_HOME-scoped, never-raises).
  * GatewayRunner enter/exit/watcher + the new-turn accept gate — the
    reversible state machine driven by the marker.

Mocked tests are necessary-not-sufficient here (the HARD live-validation gate,
Q-B, exercises a real `hermes gateway run`); these lock the unit contract.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import UTC
from pathlib import Path

import pytest

import gateway.drain_control as dc
from gateway.run import GatewayRunner
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source
from tests.waiting import wait_until


# ---------------------------------------------------------------------------
# Marker contract (drain_control.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


class TestMarkerContract:
    def test_absent_by_default(self, home):
        assert dc.drain_requested() is False
        assert dc.read_drain_request() is None

    def test_write_then_present(self, home):
        payload = dc.write_drain_request(principal="drain-control")
        assert dc.drain_requested() is True
        assert payload["action"] == "drain"
        assert payload["principal"] == "drain-control"
        body = dc.read_drain_request()
        assert body is not None and body["principal"] == "drain-control"


class TestSuppressNotification:
    """The generic suppress_notification flag on the drain marker.

    Gates ONLY the gateway's home-channel shutdown broadcast (NAS auto-update
    sets it true). Default-false so legacy/operator drains behave as before.
    The reader reuses the NS-570 epoch-staleness check so an orphaned marker
    can never silence a fresh gateway.
    """

    def test_default_false(self, home):
        payload = dc.write_drain_request(principal="drain-control")
        assert payload["suppress_notification"] is False
        assert dc.drain_notification_suppressed() is False

    def test_flag_round_trips_true(self, home):
        payload = dc.write_drain_request(principal="drain-control", suppress_notification=True)
        assert payload["suppress_notification"] is True
        body = dc.read_drain_request()
        assert body is not None and body["suppress_notification"] is True
        assert dc.drain_notification_suppressed() is True


# ---------------------------------------------------------------------------
# Instantiation-epoch staleness (NS-570: orphaned marker on durable volume)
# ---------------------------------------------------------------------------


class TestInstantiationEpoch:
    def test_write_stamps_current_epoch(self, home):
        payload = dc.write_drain_request(principal="drain-control")
        assert payload["epoch"] == dc.current_instantiation_epoch()
        body = dc.read_drain_request()
        assert body is not None and body["epoch"] == dc.current_instantiation_epoch()


    def test_marker_from_prior_instantiation_reads_as_absent(self, home, monkeypatch):
        # THE NS-570 REGRESSION. A begin-drain marker written by a PREVIOUS
        # container/VM instantiation survives on the durable HERMES_HOME volume
        # across a machine restart. The freshly-restarted gateway (new epoch)
        # must treat it as absent, NOT re-engage drain.
        monkeypatch.setattr(dc, "current_instantiation_epoch", lambda: "epoch-OLD")
        dc.write_drain_request(principal="drain-control")  # stamps "epoch-OLD"
        assert dc.drain_requested() is True  # same epoch → active

        # Simulate the restart: a brand-new instantiation epoch.
        monkeypatch.setattr(dc, "current_instantiation_epoch", lambda: "epoch-NEW")
        # The marker file is still physically present on the volume…
        assert dc.drain_request_path().exists() is True
        # …but it is ignored because its epoch belongs to a prior instantiation.
        assert dc.drain_requested() is False


    @pytest.mark.platforms("linux")
    def test_current_epoch_empty_when_proc_unreadable(self, monkeypatch):
        # When neither /proc identity source is readable, the epoch is "" so
        # the staleness check is disabled rather than crashing.
        from pathlib import Path as _P

        orig_read_text = _P.read_text

        def _boom(self, *a, **k):
            if str(self).startswith("/proc/"):
                raise OSError("no /proc")
            return orig_read_text(self, *a, **k)

        dc.current_instantiation_epoch.cache_clear()
        monkeypatch.setattr(_P, "read_text", _boom)
        try:
            assert dc.current_instantiation_epoch() == ""
        finally:
            dc.current_instantiation_epoch.cache_clear()


@pytest.mark.platforms("macos")
class TestDarwinInstantiationEpoch:
    """SOU-3249: off Linux the epoch was ``""``, so no marker could ever read as stale — the
    restart-surviving guard was disabled on the only host acceptance harnesses run on."""

    def _epoch_with_boot_time(self, monkeypatch, boot_time):
        import psutil

        monkeypatch.setattr(psutil, "boot_time", lambda: boot_time)
        dc.current_instantiation_epoch.cache_clear()
        return dc.current_instantiation_epoch()

    def test_epoch_is_a_boot_stamp(self, monkeypatch):
        try:
            assert self._epoch_with_boot_time(monkeypatch, 1_700_000_000.9) == "darwin-boot:1700000000"
        finally:
            dc.current_instantiation_epoch.cache_clear()

    def test_a_reboot_makes_a_surviving_marker_stale(self, home, monkeypatch):
        try:
            self._epoch_with_boot_time(monkeypatch, 1_700_000_000.9)
            dc.write_drain_request(principal="drain-control")
            assert dc.drain_requested() is True  # same boot → honoured
            self._epoch_with_boot_time(monkeypatch, 1_700_009_999.0)  # the machine rebooted
        finally:
            dc.current_instantiation_epoch.cache_clear()
        # The marker survived on the durable volume; the new boot must ignore it.
        assert dc.drain_request_path().exists() is True
        assert dc.drain_requested() is False

    def test_real_host_epoch_is_not_empty(self):
        # The lane that runs acceptance harnesses must be able to prove its own boot identity.
        dc.current_instantiation_epoch.cache_clear()
        try:
            assert dc.current_instantiation_epoch().startswith("darwin-boot:")
        finally:
            dc.current_instantiation_epoch.cache_clear()


class TestMarkerMaxAge:

    def test_max_age_stays_under_the_canary_threshold(self):
        # 45 min of fleet silence pages CanaryDeadmanDown, so a leaked marker's own bound must be
        # well inside it (SOU-3249: 3600 s made one leaked harness marker a guaranteed page).
        assert dc.DRAIN_REQUEST_MAX_AGE_SECONDS <= 15 * 60

    def test_expired_marker_reads_as_absent(self, home):
        # THE #85433 REGRESSION. A drain-gated action completes WITHOUT a
        # machine restart, so the epoch still matches — but the writer never
        # cancelled the drain. The orphan must not wedge the gateway forever.
        from datetime import datetime, timedelta, timezone

        dc.write_drain_request(principal="drain-control", suppress_notification=True)
        body = dc.read_drain_request()
        assert body is not None
        body["requested_at"] = (
            datetime.now(UTC)
            - timedelta(seconds=dc.DRAIN_REQUEST_MAX_AGE_SECONDS + 60)
        ).isoformat()
        dc.drain_request_path().write_text(json.dumps(body), encoding="utf-8")

        # The marker file is still physically present, with the CURRENT epoch…
        assert dc.drain_request_path().exists() is True
        # …but it is ignored because it outlived any legitimate drain.
        assert dc.drain_requested() is False
        # The suppression flag of an expired orphan is likewise ignored.
        assert dc.drain_notification_suppressed() is False

    def test_marker_without_timestamp_still_honoured(self, home):
        # Leniency contract: no requested_at (legacy/corrupt body) must fail
        # toward quiescing, exactly like the epoch check.
        payload = {"action": "drain", "epoch": dc.current_instantiation_epoch()}
        dc.drain_request_path().write_text(json.dumps(payload), encoding="utf-8")
        assert dc.drain_requested() is True

    def test_unparseable_timestamp_still_honoured(self, home):
        payload = {
            "action": "drain",
            "epoch": dc.current_instantiation_epoch(),
            "requested_at": "not-a-timestamp",
        }
        dc.drain_request_path().write_text(json.dumps(payload), encoding="utf-8")
        assert dc.drain_requested() is True

    def test_naive_timestamp_treated_as_utc(self, home):
        # A writer that stamped a tz-naive ISO string must still expire.
        from datetime import datetime, timedelta, timezone

        stale_naive = (
            datetime.now(UTC)
            - timedelta(seconds=dc.DRAIN_REQUEST_MAX_AGE_SECONDS + 60)
        ).replace(tzinfo=None)
        payload = {
            "action": "drain",
            "epoch": dc.current_instantiation_epoch(),
            "requested_at": stale_naive.isoformat(),
        }
        dc.drain_request_path().write_text(json.dumps(payload), encoding="utf-8")
        assert dc.drain_requested() is False

    def test_expiry_warning_logged_once_per_marker(self, home, caplog):
        # The watcher polls every 1s; an expired orphan must warn ONCE, not
        # once per tick (~86k/day). A refreshed marker that expires again
        # warns again (new requested_at).
        import logging
        from datetime import datetime, timedelta, timezone

        def _write_expired(offset_seconds):
            dc.write_drain_request(principal="drain-control")
            body = dc.read_drain_request()
            assert body is not None
            body["requested_at"] = (
                datetime.now(UTC)
                - timedelta(seconds=dc.DRAIN_REQUEST_MAX_AGE_SECONDS + offset_seconds)
            ).isoformat()
            dc.drain_request_path().write_text(json.dumps(body), encoding="utf-8")

        _write_expired(60)
        with caplog.at_level(logging.WARNING, logger="gateway.drain_control"):
            assert dc.drain_requested() is False
            assert dc.drain_requested() is False  # second poll tick
            assert dc.drain_requested() is False  # third poll tick
        expired_logs = [r for r in caplog.records if "expired drain marker" in r.message]
        assert len(expired_logs) == 1

        caplog.clear()
        _write_expired(120)  # a DIFFERENT requested_at that is also expired
        with caplog.at_level(logging.WARNING, logger="gateway.drain_control"):
            assert dc.drain_requested() is False
        expired_logs = [r for r in caplog.records if "expired drain marker" in r.message]
        assert len(expired_logs) == 1

    def test_rewrite_refreshes_the_clock(self, home):
        # The sanctioned keep-alive: re-writing the marker bumps requested_at,
        # so a deliberately long drain stays honoured.
        from datetime import datetime, timedelta, timezone

        dc.write_drain_request(principal="drain-control")
        body = dc.read_drain_request()
        assert body is not None
        body["requested_at"] = (
            datetime.now(UTC)
            - timedelta(seconds=dc.DRAIN_REQUEST_MAX_AGE_SECONDS + 60)
        ).isoformat()
        dc.drain_request_path().write_text(json.dumps(body), encoding="utf-8")
        assert dc.drain_requested() is False  # expired…
        dc.write_drain_request(principal="drain-control")  # …keep-alive re-write
        assert dc.drain_requested() is True


# ---------------------------------------------------------------------------
# Principal trust boundary + writer liveness (SOU-3249)
# ---------------------------------------------------------------------------


def _marker_body(age_seconds: float = 0.0, **overrides):
    """A hand-written marker: the shape an out-of-tree harness produces."""
    from datetime import datetime, timedelta

    body = {
        "action": "drain",
        "requested_at": (datetime.now(UTC) - timedelta(seconds=age_seconds)).isoformat(),
        "principal": "drain-control",
        "epoch": dc.current_instantiation_epoch(),
        "suppress_notification": False,
    }
    body.update(overrides)
    return body


def _write_marker(body):
    dc.drain_request_path().write_text(json.dumps(body), encoding="utf-8")


class TestPrincipalTrustBoundary:
    """A drain refuses admission fleet-wide, so only the authenticated seams may ask for one."""

    def test_acceptance_harness_marker_is_not_honoured(self, home):
        # THE SOU-3249 PAYLOAD. The PRX-5802 harness hand-wrote exactly this
        # (paperclip/prx-5802-drain-gate/measure4.py:135-147) at 2026-10-21T12:17:08Z; the live
        # gateway refused admission for 60 min and paged CanaryDeadmanDown.
        _write_marker(_marker_body(principal="prx-5802 acceptance v5", epoch="", suppress_notification=True))
        assert dc.drain_requested() is False
        assert dc.drain_notification_suppressed() is False

    def test_unknown_principal_written_through_the_api_is_inert(self, home):
        payload = dc.write_drain_request(principal="prx-5802 acceptance v5", suppress_notification=True)
        assert payload["test_only"] is True
        assert payload["suppress_notification"] is False  # a harness may never silence the signal
        assert dc.drain_requested() is False

    def test_test_only_flag_is_refused_even_for_a_trusted_principal(self, home):
        payload = dc.write_drain_request(principal="drain-control", test_only=True)
        assert payload["test_only"] is True
        assert payload["suppress_notification"] is False
        assert dc.drain_requested() is False

    def test_corrupt_marker_still_drains(self, home):
        # Fail-safe toward quiescing is unchanged: a half-written file carries no principal to
        # refuse, and un-quiescing a fleet mid-update is the worse failure.
        dc.drain_request_path().write_text("{not json", encoding="utf-8")
        assert dc.drain_requested() is True


# A writer that outlives its own marker unless the reader notices it is gone: it writes, reports
# the path, and then blocks on stdin (which is what a killed harness looks like).
_CHILD_WRITER = """
import sys
sys.path.insert(0, sys.argv[1])
import gateway.drain_control as dc
dc.write_drain_request(principal="drain-control")
print(dc.drain_request_path(), flush=True)
sys.stdin.read()
"""


class TestWriterLiveness:

    def test_marker_from_a_sigkilled_writer_is_not_honoured(self, home):
        # THE SOU-3249 REGRESSION. SIGKILL runs no finally and no atexit, so the marker outlives
        # its writer; the live gateway must stop honouring it on the next watcher tick instead of
        # refusing admission until the max-age.
        proc = subprocess.Popen(
            [sys.executable, "-c", _CHILD_WRITER, str(Path(dc.__file__).resolve().parents[1])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={**os.environ, "HERMES_HOME": str(home)},
        )
        assert proc.stdin is not None and proc.stdout is not None
        try:
            assert proc.stdout.readline().strip() == str(dc.drain_request_path())
            assert dc.drain_requested() is True  # a live writer still holds the drain
        finally:
            proc.kill()
            proc.wait(timeout=60)
            for stream in (proc.stdin, proc.stdout):
                if stream is not None:
                    stream.close()
        assert wait_until(lambda: dc.drain_requested() is False,
                          what="the dead writer's marker to be ignored")

    def test_a_recycled_pid_is_not_the_writer(self, home):
        import psutil
        from hermes_cli.process_identity import process_birth

        # Same pid as this (live) process, born an hour earlier: a different incarnation, so the
        # marker is an orphan and not a live drain. Bare PID existence proves nothing.
        _write_marker(_marker_body(writer_pid=os.getpid(),
                                   writer_birth=process_birth(psutil.Process()) - 3600))
        assert dc.drain_requested() is False


class TestPerMarkerLease:
    """PRX-5819: a writer that knows its own lifetime bounds its own blast radius.

    Measured 2026-10-21: an acceptance harness died with the marker engaged and the host stayed
    drained for the whole 900 s constant — 17 refused spawns, every ``agent``-profile cron silent
    for 64 min. A writer emitting ``max_age_seconds`` is now harmless when it dies.
    """

    def test_a_short_lease_expires_long_before_the_constant(self, home):
        # THE PRX-5819 REGRESSION. 61 s old is far inside DRAIN_REQUEST_MAX_AGE_SECONDS (900), so
        # only the marker's own lease can make it stale.
        _write_marker(_marker_body(age_seconds=61, max_age_seconds=60))
        assert dc.drain_requested() is False

    def test_a_short_lease_is_honoured_inside_its_window(self, home):
        _write_marker(_marker_body(age_seconds=30, max_age_seconds=60))
        assert dc.drain_requested() is True

    def test_a_lease_longer_than_the_constant_is_clamped_down(self, home):
        # A writer may ask to be forgotten sooner, never to outlive the constant's blast radius.
        _write_marker(_marker_body(age_seconds=dc.DRAIN_REQUEST_MAX_AGE_SECONDS + 60,
                                   max_age_seconds=86_400))
        assert dc.drain_requested() is False

    def test_an_unusable_lease_falls_back_to_the_constant(self, home):
        # Absent/!numeric/bool/non-positive must never read as "expire now" (that would un-quiesce a
        # fleet mid-update) nor as "no bound".
        for bad in (None, "60", True, 0, -5):
            _write_marker(_marker_body(age_seconds=120, max_age_seconds=bad))
            assert dc.drain_requested() is True, bad
            _write_marker(_marker_body(age_seconds=dc.DRAIN_REQUEST_MAX_AGE_SECONDS + 60,
                                       max_age_seconds=bad))
            assert dc.drain_requested() is False, bad

    def test_lease_arithmetic_rejects_non_finite_values(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            assert dc._clamp_lease(bad) == dc.DRAIN_REQUEST_MAX_AGE_SECONDS

    def test_write_clamps_or_omits_the_lease(self, home):
        assert dc.DRAIN_LEASE_FIELD not in dc.write_drain_request(principal="drain-control")
        body = dc.read_drain_request()
        assert body is not None and dc.DRAIN_LEASE_FIELD not in body
        payload = dc.write_drain_request(principal="drain-control", max_age_seconds=60)
        assert payload[dc.DRAIN_LEASE_FIELD] == 60
        # A marker that keeps its lifetime alive by re-writing keeps its lease.
        body = dc.read_drain_request()
        assert body is not None and body[dc.DRAIN_LEASE_FIELD] == 60
        payload = dc.write_drain_request(principal="drain-control", max_age_seconds=86_400)
        assert payload[dc.DRAIN_LEASE_FIELD] == dc.DRAIN_REQUEST_MAX_AGE_SECONDS

    def test_a_marker_without_a_lease_behaves_exactly_as_before(self, home):
        _write_marker(_marker_body(age_seconds=120))
        assert dc.drain_requested() is True  # honoured well past a short lease…
        _write_marker(_marker_body(age_seconds=dc.DRAIN_REQUEST_MAX_AGE_SECONDS + 60))
        assert dc.drain_requested() is False  # …and expired at the constant, as today


# ---------------------------------------------------------------------------
# Boot adoption (PRX-5819: the marker outlives the process it quiesced)
# ---------------------------------------------------------------------------


def _boot_runner():
    """A runner wired for ``start()``: the real adoption, serving state and status publish, with
    everything after the marker check stubbed (the startup sequence needs a whole install)."""
    from gateway.run_startup import GatewayStartupMixin

    runner, _ = _drain_runner()
    runner.start = GatewayStartupMixin.start.__get__(runner, GatewayRunner)
    runner._adopt_pending_drain = GatewayRunner._adopt_pending_drain.__get__(runner, GatewayRunner)
    runner._serving_state = GatewayStartupMixin._serving_state.__get__(runner, GatewayRunner)
    runner._startup_parked_platforms = False
    published: list[str] = []

    async def _start_impl():
        # What the real sequence does after `self._running = True`: publish the serving state. If
        # admission were already open here, `_external_drain_active` would be False.
        published.append(runner._serving_state())
        runner._update_runtime_status(published[-1])
        return True

    async def _flush():
        return None

    runner._start_impl = _start_impl
    runner._start_flush_runtime_status = _flush
    return runner, published


class TestBootAdoption:

    @pytest.mark.asyncio
    async def test_a_marker_present_at_boot_is_honoured_before_the_first_publish(self, home, caplog):
        # THE PRX-5819 REGRESSION. For 35 s after a boot the state file truthfully said `running`
        # with a live writer pid while a durable marker wanted a drain, so a reader that respects
        # the file spawned 21 runs this gateway then refused as `adapter_failed`.
        runner, published = _boot_runner()
        dc.write_drain_request(principal="drain-control")
        with caplog.at_level(logging.INFO, logger="gateway.run"):
            assert await runner.start() is True
        assert runner._external_drain_active is True
        assert published == ["draining"]  # never `running`
        assert any("External drain ENGAGED" in r.getMessage() for r in caplog.records), \
            "the engagement must be logged before the first published state"

    @pytest.mark.asyncio
    async def test_a_plain_boot_still_publishes_running(self, home):
        runner, published = _boot_runner()
        assert await runner.start() is True
        assert runner._external_drain_active is False
        assert published == ["running"]

    @pytest.mark.asyncio
    async def test_a_marker_this_gateway_must_ignore_does_not_park_the_boot(self, home):
        # An untrusted principal's marker is inert (SOU-3249); adopting it would quiesce a boot that
        # no authenticated seam asked to quiesce.
        runner, published = _boot_runner()
        _write_marker(_marker_body(principal="prx-5819 acceptance"))
        assert await runner.start() is True
        assert runner._external_drain_active is False
        assert published == ["running"]


# ---------------------------------------------------------------------------
# Gateway state machine (enter / exit / idempotency)
# ---------------------------------------------------------------------------


def _drain_runner():
    runner, adapter = make_restart_runner()
    runner._external_drain_active = False
    # Bind the real methods under test.
    runner._enter_external_drain = GatewayRunner._enter_external_drain.__get__(
        runner, GatewayRunner
    )
    runner._exit_external_drain = GatewayRunner._exit_external_drain.__get__(
        runner, GatewayRunner
    )
    return runner, adapter


class TestDrainStateMachine:




    def test_exit_during_shutdown_does_not_revert_to_running(self):
        runner, _ = _drain_runner()
        runner._enter_external_drain()
        runner._update_runtime_status.reset_mock()
        # A shutdown drain is now in progress — exit must NOT resurrect running.
        runner._draining = True
        runner._exit_external_drain()
        assert runner._external_drain_active is False
        runner._update_runtime_status.assert_not_called()


# ---------------------------------------------------------------------------
# Watcher reconciliation
# ---------------------------------------------------------------------------


class TestDrainWatcher:

    @pytest.mark.asyncio
    async def test_watcher_enters_then_exits_with_marker(self, home):
        runner, _ = _drain_runner()
        runner._drain_control_watcher = GatewayRunner._drain_control_watcher.__get__(
            runner, GatewayRunner
        )
        # Drive a few ticks manually rather than spinning the loop.
        dc.write_drain_request()
        task = asyncio.create_task(runner._drain_control_watcher(interval=0.02))
        await asyncio.sleep(0.06)
        assert runner._external_drain_active is True
        dc.clear_drain_request()
        await asyncio.sleep(0.06)
        assert runner._external_drain_active is False
        runner._running = False
        await asyncio.sleep(0.04)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# New-turn accept gate
# ---------------------------------------------------------------------------


class TestNewTurnGate:
    @pytest.mark.asyncio
    async def test_new_turn_refused_during_external_drain(self):
        runner, _ = _drain_runner()
        runner._external_drain_active = True
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=make_restart_source(),
            message_id="m1",
        )
        result = await runner._handle_message(event)
        assert result is not None
        assert "draining" in result.lower()

