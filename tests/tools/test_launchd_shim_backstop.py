"""Regression for SOU-1932 — the action-side launchd backstop.

``cron/lifecycle_guard.py`` refuses launchd job-set writes by reading the command text, so
it can never see a value built at runtime (``$(printf launchctl)``, ``${L}ctl``, a name read
from a file, an env-provided command word). These pin the layer that runs after a shell
exists: a ``launchctl`` shim put first on the PATH of children of a SUPERVISED gateway, which
decides on the verb and walks its own ancestry.

Both directions matter. Refusing under the gateway is the guard; refusing (or shimming) when
the gateway is NOT an ancestor would break the release path, which arms host-detector jobs
from a login shell. The tests therefore drive the real shim as a real process — an argv-only
assertion would pass on a shim that refuses nothing.
"""

import os
import subprocess

import pytest

from tools.environments import local as local_mod
from tools.environments.local import (
    LAUNCHD_SHIM_DIR,
    LAUNCHD_SHIM_GATEWAY_ENV,
    _finalize_child_env,
    _make_run_env,
    build_subprocess_env,
)

# launchd and /bin/launchctl exist only on macOS; the shim is injected only there.
pytestmark = pytest.mark.platforms("macos")

SHIM = LAUNCHD_SHIM_DIR / "launchctl"
REFUSED = 77
# A job-set verb aimed at a domain that cannot exist: the real launchctl fails on the
# domain (never on the verb), which is how a pass-through is told from a refusal.
PROBE_DOMAIN = "gui/999999"
PROBE_PLIST = "/nonexistent/sou-1932-probe.plist"


def _shim_env(tmp_path, *, gateway, log=None, path=None):
    env = {
        "PATH": os.environ.get("PATH", "/bin:/usr/bin"),
        "HOME": str(tmp_path),
        "HERMES_LAUNCHD_SHIM_LOG": str(log or tmp_path / "refusals.log"),
    }
    if gateway is not None:
        env[LAUNCHD_SHIM_GATEWAY_ENV] = str(gateway)
    if path is not None:
        env["PATH"] = path
    return env


def _run_shim(tmp_path, *argv, gateway, log=None, path=None):
    return subprocess.run(
        [str(SHIM), *argv], env=_shim_env(tmp_path, gateway=gateway, log=log, path=path),
        capture_output=True, text=True, timeout=60, check=False,
    )


def _log_lines(tmp_path, log=None):
    log_path = log or tmp_path / "refusals.log"
    if not log_path.exists():
        return []
    return [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def unrelated_pid():
    """A live pid that is NOT in this test process's ancestry."""
    helper = subprocess.Popen(["/bin/sleep", "30"])
    try:
        yield helper.pid
    finally:
        helper.terminate()
        helper.wait(timeout=10)


def test_job_set_verb_refused_when_the_gateway_is_an_ancestor(tmp_path):
    # This test process stands in for the gateway: the shim's ancestry walk must find it.
    result = _run_shim(tmp_path, "bootstrap", PROBE_DOMAIN, PROBE_PLIST, gateway=os.getpid())

    assert result.returncode == REFUSED, (result.stdout, result.stderr)
    assert "launchd-shim: refused `launchctl bootstrap`" in result.stderr
    assert "no launchd job-set write from inside the supervised Hermes gateway" in result.stderr
    # The refusal names the outside-the-gateway remedy, not just the block.
    assert "login shell outside the gateway" in result.stderr

    lines = _log_lines(tmp_path)
    assert len(lines) == 1, lines
    assert "verb=bootstrap" in lines[0]
    assert f"argv=bootstrap {PROBE_DOMAIN} {PROBE_PLIST}" in lines[0]
    assert f" {os.getpid()} " in f" {lines[0].split(' ancestry=')[1].split(' argv=')[0]} "


def test_verb_hidden_behind_runtime_substitution_is_refused(tmp_path):
    """The exact class the static scan cannot close: the command word is built at runtime."""
    log = tmp_path / "refusals.log"
    command = f'L=$(printf launchctl); "$L" bootstrap {PROBE_DOMAIN} {PROBE_PLIST}'
    result = subprocess.run(
        ["sh", "-c", command],
        env=_shim_env(tmp_path, gateway=os.getpid(), log=log,
                      path=f"{LAUNCHD_SHIM_DIR}:{os.environ.get('PATH', '/bin:/usr/bin')}"),
        capture_output=True, text=True, timeout=60, check=False,
    )

    assert result.returncode == REFUSED, (result.stdout, result.stderr)
    assert "launchd-shim: refused `launchctl bootstrap`" in result.stderr
    assert "verb=bootstrap" in _log_lines(tmp_path, log)[0]


def test_read_verbs_pass_through_under_the_gateway(tmp_path):
    """`launchctl list` from an agent shell must keep working: the shim decides on the verb."""
    result = _run_shim(tmp_path, "list", gateway=os.getpid())

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("PID\tStatus\tLabel"), result.stdout[:200]
    assert _log_lines(tmp_path) == []


def test_job_set_verb_passes_through_when_the_gateway_is_not_an_ancestor(tmp_path, unrelated_pid):
    """The release path: arming jobs from outside the gateway stays possible."""
    result = _run_shim(tmp_path, "bootstrap", PROBE_DOMAIN, PROBE_PLIST, gateway=unrelated_pid)

    assert result.returncode != REFUSED, result.stderr
    assert "launchd-shim" not in result.stderr
    # Proof it reached the real launchctl: launchd's own complaint about the domain.
    assert "domain" in result.stderr.lower(), result.stderr
    assert _log_lines(tmp_path) == []


def test_refuses_when_the_gateway_marker_is_missing(tmp_path):
    """Fail closed: the shim is only ever on PATH as a gateway child's entry."""
    result = _run_shim(tmp_path, "bootstrap", PROBE_DOMAIN, PROBE_PLIST, gateway=None)

    assert result.returncode == REFUSED, result.stderr
    assert "fail closed" in result.stderr
    assert "ancestry=unknown" in _log_lines(tmp_path)[0]


def test_refuses_when_the_ancestry_walk_fails(tmp_path):
    """A walk that cannot reach pid 1 proves nothing, so it refuses instead of guessing."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    broken_ps = fake_bin / "ps"
    broken_ps.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    broken_ps.chmod(0o755)

    result = _run_shim(
        tmp_path, "bootstrap", PROBE_DOMAIN, PROBE_PLIST, gateway=os.getpid(), path=str(fake_bin),
    )

    assert result.returncode == REFUSED, result.stderr
    assert "could not walk this process's ancestry" in result.stderr


def test_every_policy_verb_is_refused(tmp_path):
    """Contract with cron/lifecycle_guard.py: a verb added there cannot slip past the shim."""
    from cron.lifecycle_guard import _LAUNCHCTL_REGISTRATION_VERBS, _LAUNCHCTL_REMOVAL_VERBS

    verbs = sorted(_LAUNCHCTL_REGISTRATION_VERBS | _LAUNCHCTL_REMOVAL_VERBS | {"kickstart"})
    for verb in verbs:
        result = _run_shim(tmp_path, verb, f"{PROBE_DOMAIN}/ai.hermes.gateway", gateway=os.getpid())
        assert result.returncode == REFUSED, (verb, result.stdout, result.stderr)
    assert [line.split("verb=")[1].split()[0] for line in _log_lines(tmp_path)] == verbs


def test_leading_flags_do_not_hide_the_verb(tmp_path):
    result = _run_shim(tmp_path, "-w", "bootout", PROBE_DOMAIN, gateway=os.getpid())

    assert result.returncode == REFUSED, result.stderr
    assert "verb=bootout" in _log_lines(tmp_path)[0]


def test_shim_is_shipped_executable():
    assert SHIM.is_file()
    assert os.access(SHIM, os.X_OK), "git must carry the executable bit or PATH lookup skips it"


# --- the injection side: build_subprocess_env's shared guard ------------------------------

@pytest.fixture
def supervised(monkeypatch):
    from tools import process_registry

    # A test process spawned by a real, already-patched gateway inherits the marker; drop it so
    # the supervised probe is the thing under test. Reset the one-way latch too.
    monkeypatch.delenv(LAUNCHD_SHIM_GATEWAY_ENV, raising=False)
    monkeypatch.setattr(local_mod, "_LAUNCHD_SHIM_CONFIRMED", False)
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)


def _path_entries(env):
    return env["PATH"].split(os.pathsep)


def test_terminal_spawn_env_gets_the_shim(supervised):
    """`_make_run_env` is how the terminal tool spawns — the path the guard exists for."""
    env = _make_run_env({})

    assert _path_entries(env)[0] == str(LAUNCHD_SHIM_DIR)
    assert env[LAUNCHD_SHIM_GATEWAY_ENV] == str(os.getpid())


@pytest.mark.parametrize("scrub_secrets", [True, False])
def test_build_subprocess_env_gets_the_shim(supervised, scrub_secrets):
    env = build_subprocess_env(scrub_secrets=scrub_secrets, inherit_profile_home=False)

    assert _path_entries(env)[0] == str(LAUNCHD_SHIM_DIR)
    assert env[LAUNCHD_SHIM_GATEWAY_ENV] == str(os.getpid())


def test_inherited_marker_keeps_the_shim_without_reprobing(monkeypatch):
    """A descendant of an already-shimmed gateway keeps the rail and skips the probe.

    The probe reads the live gateway's pid file and start-time, so re-running it in every
    descendant would put that I/O on every spawn in the tree.
    """
    from tools import process_registry

    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
    monkeypatch.setenv(LAUNCHD_SHIM_GATEWAY_ENV, "424242")
    env = _make_run_env({})

    assert _path_entries(env)[0] == str(LAUNCHD_SHIM_DIR)
    assert env[LAUNCHD_SHIM_GATEWAY_ENV] == "424242"


def test_no_shim_when_this_process_is_not_the_supervised_gateway(monkeypatch):
    from tools import process_registry

    monkeypatch.delenv(LAUNCHD_SHIM_GATEWAY_ENV, raising=False)
    monkeypatch.setattr(local_mod, "_LAUNCHD_SHIM_CONFIRMED", False)
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
    env = _make_run_env({})

    assert str(LAUNCHD_SHIM_DIR) not in _path_entries(env)
    assert LAUNCHD_SHIM_GATEWAY_ENV not in env


def test_a_positive_verdict_is_latched_and_a_negative_never_is(monkeypatch):
    """The probe costs ~10 ms (pid file + start time), so a gateway must not pay it per spawn.

    A negative is never cached: a gateway that spawns before its pid file exists has to be
    picked up by a later spawn.
    """
    from tools import process_registry

    monkeypatch.delenv(LAUNCHD_SHIM_GATEWAY_ENV, raising=False)
    calls = []
    verdict = {"supervised": False}
    monkeypatch.setattr(local_mod, "_LAUNCHD_SHIM_CONFIRMED", False)
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process",
                        lambda: calls.append(1) or verdict["supervised"])

    assert LAUNCHD_SHIM_GATEWAY_ENV not in _make_run_env({})  # negative: not shimmed
    assert LAUNCHD_SHIM_GATEWAY_ENV not in _make_run_env({})  # negative: asked again
    assert len(calls) == 2

    verdict["supervised"] = True
    for _ in range(3):  # positive: shimmed, and the verdict is not re-proven
        env = _make_run_env({})
        assert env[LAUNCHD_SHIM_GATEWAY_ENV] == str(os.getpid())
    assert len(calls) == 3


def test_base_without_path_still_gets_a_usable_path(supervised, monkeypatch):
    """A base without PATH still gets a usable PATH: the shim is prepended, never the whole."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = _finalize_child_env({})

    entries = _path_entries(env)
    assert entries[0] == str(LAUNCHD_SHIM_DIR)
    assert entries[1:] == ["/usr/bin", "/bin"]


def test_prepend_is_idempotent(supervised):
    env = _finalize_child_env(_finalize_child_env({"PATH": "/usr/bin:/bin"}))

    assert _path_entries(env).count(str(LAUNCHD_SHIM_DIR)) == 1


def test_injected_child_is_refused_end_to_end(supervised, tmp_path, monkeypatch):
    """Real factory → real child → real refusal, with no gateway needed to prove wiring."""
    log = tmp_path / "refusals.log"
    monkeypatch.setenv("HERMES_LAUNCHD_SHIM_LOG", str(log))
    env = build_subprocess_env(scrub_secrets=True, inherit_profile_home=False)
    command = f'L=$(printf launchctl); "$L" bootstrap {PROBE_DOMAIN} {PROBE_PLIST}'

    result = subprocess.run(["sh", "-c", command], capture_output=True, text=True,
                            timeout=60, env=env, check=False)

    assert result.returncode == REFUSED, (result.stdout, result.stderr)
    assert "launchd-shim: refused" in result.stderr
    assert "verb=bootstrap" in log.read_text(encoding="utf-8")
