"""``hermes gateway status`` names the profiles the multiplexer does not serve (PRX-5734).

The ceiling's verdict used to live in one aggregate log line and, between reconcile ticks, not even
in ``gateway_state.json``: an operator looking at the host saw a healthy gateway while 28 profiles
could not be admitted. These pin the operator surface — a live record's ``parked_profiles`` becomes
a status block with its reasons and remedy, and a dead or empty record prints nothing.
"""
from hermes_cli import gateway_multiplex_served as served

CAPACITY = "capacity: 15 profiles per gateway (gateway.multiplex_max_profiles)"


def _live_record(monkeypatch, parked, *, gateway_live=True, record_live=True):
    monkeypatch.setattr(served, "live_default_gateway_pid", lambda: 4242 if gateway_live else None)
    monkeypatch.setattr("gateway.status.read_runtime_status",
                        lambda *_a, **_k: {"gateway_state": "running", "pid": 4242,
                                           "parked_profiles": parked})
    monkeypatch.setattr("gateway.status.runtime_status_pid_is_live", lambda *_a, **_k: record_live)


def test_lines_name_every_unserved_profile_with_its_reason():
    lines = served.unserved_profile_lines({
        "advisor": CAPACITY,
        "cmo": CAPACITY,
        "broken": "session store unusable: disk I/O error",
    })
    text = "\n".join(lines)

    assert "advisor" in text and "cmo" in text and "broken" in text
    assert f"2 unserved — {CAPACITY}" in text
    assert "session store unusable" in text
    # A ceiling deferral carries its remedy: the ceiling itself, and the priority order.
    assert "gateway.multiplex_max_profiles" in text
    assert "gateway.multiplex_serve_priority" in text


def test_no_lines_when_everything_is_served():
    assert served.unserved_profile_lines({}) == []


def test_record_reader_returns_the_live_verdict(monkeypatch):
    _live_record(monkeypatch, {"advisor": CAPACITY})

    assert served.recorded_unserved_profiles() == {"advisor": CAPACITY}


def test_record_reader_ignores_a_dead_gateway(monkeypatch):
    """A stopped gateway's leftover parks are history, not a verdict about the host."""
    _live_record(monkeypatch, {"advisor": CAPACITY}, gateway_live=False)
    assert served.recorded_unserved_profiles() == {}

    _live_record(monkeypatch, {"advisor": CAPACITY}, record_live=False)
    assert served.recorded_unserved_profiles() == {}


def test_status_printer_stays_quiet_on_a_clean_host(monkeypatch, capsys):
    _live_record(monkeypatch, {})
    monkeypatch.setattr("hermes_cli.gateway_multiplex_mode.recorded_standalone_warning_lines",
                        lambda: [])

    served.print_served_set_notices()

    assert capsys.readouterr().out == ""


def test_status_printer_shows_the_unserved_block(monkeypatch, capsys):
    _live_record(monkeypatch, {"advisor": CAPACITY})
    monkeypatch.setattr("hermes_cli.gateway_multiplex_mode.recorded_standalone_warning_lines",
                        lambda: [])

    served.print_served_set_notices()

    out = capsys.readouterr().out
    assert "does NOT serve" in out and "advisor" in out
