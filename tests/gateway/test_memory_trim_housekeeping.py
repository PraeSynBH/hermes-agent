"""Memory-trim coverage for the long-lived messaging gateway housekeeper."""

import gateway.run as gateway_run


class _Ticks:
    """Run *n* housekeeping ticks without a sleep or background thread."""

    def __init__(self, n: int = 1):
        self.n = n
        self.count = 0

    def is_set(self):
        return self.count >= self.n

    def wait(self, timeout=None):
        self.count += 1
        return self.count >= self.n


def test_gateway_housekeeping_calls_periodic_memory_trim(monkeypatch):
    from hermes_cli import mem_trim

    calls = []
    monkeypatch.setattr(
        mem_trim,
        "trim_memory",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    # A full gc.collect() costs seconds and cannot release pages on macOS (malloc_trim is
    # glibc-only), so the arm is polled every 5th tick rather than every tick (PRX-6104).
    # The contract this pins is unchanged: the housekeeping thread does call the trim.
    monkeypatch.setattr(gateway_run, "_housekeeping_plugin_update_check", lambda: None)
    from gateway import run_profile_reconcile
    monkeypatch.setattr(
        run_profile_reconcile, "_mcp_config_reconciler", lambda runner=None: lambda: None)

    gateway_run._start_gateway_housekeeping(_Ticks(5), interval=0)

    assert calls == [{"reason": "messaging gateway housekeeping"}]
