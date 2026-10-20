"""An auto-started daemon never believes it belongs to the host's launchd job.

Regression for PRA-6594: a client running under ``ai.hermes.gateway`` inherits launchd's
``XPC_SERVICE_NAME`` (stamped on the job process) and the stderr wrapper's re-export
``HERMES_LAUNCHD_LABEL`` — so every agent shell, terminal-tool shell and Paperclip run shell
carries them. The detached daemon those clients auto-start is NOT part of that job, but it used
to inherit both markers, answer ``launchd`` from ``_detect_supervisor()``, and therefore log
"Auto-started gateway idle exit not armed: supervised by launchd" — never arming ``--idle-exit``
and outliving its client forever. Twelve-plus of those accumulated in one day.
"""
import json
import os
from pathlib import Path
import sys
import time

import pytest

# What the host launchd job exports into every descendant session.
LAUNCHD_MARKERS = {
    'XPC_SERVICE_NAME': 'ai.hermes.gateway',
    'HERMES_LAUNCHD_LABEL': 'ai.hermes.gateway',
    'LAUNCHD_SOCKET': '/tmp/launchd-1f2e.sock',
}


@pytest.mark.platforms("macos")
@pytest.mark.spawns_gateway_lookalike  # stub interpreter records env then exits; reaped below
def test_daemon_spawned_from_a_launchd_session_is_not_launchd_owned(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime_start as start
    from gateway.control_socket import _detect_supervisor

    home = tmp_path / 'home'
    home.mkdir(mode=0o700)
    witness = home / 'env.json'
    executable = tmp_path / 'owned-interpreter'
    executable.write_text(
        '#!' + sys.executable + '\nimport json, os\n'
        + f'open({str(witness)!r}, "w").write(json.dumps('
        + '{k: os.environ.get(k) for k in ' + repr(list(LAUNCHD_MARKERS)) + '}))\n',
        encoding='utf-8')
    executable.chmod(0o700)
    monkeypatch.setattr(sys, 'executable', str(executable))
    for key, value in LAUNCHD_MARKERS.items():
        monkeypatch.setenv(key, value)
    # The client itself IS launchd-owned — that is the whole trap.
    assert _detect_supervisor() == 'launchd'

    child = start.spawn_unmanaged_gateway(home, deadline=time.monotonic() + 5)
    try:
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)

    assert json.loads(Path(witness).read_text()) == dict.fromkeys(LAUNCHD_MARKERS)
    # …and with the inherited job identity stripped, the daemon's own answer is "manual", which is
    # what lets `gateway run --idle-exit` arm (arm_unmanaged_idle_exit refuses any other answer).
    for key in LAUNCHD_MARKERS:
        monkeypatch.delenv(key, raising=False)
    assert _detect_supervisor() == 'manual'
    assert os.environ.get('XPC_SERVICE_NAME') is None
