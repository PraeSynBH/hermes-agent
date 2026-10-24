"""The ``stay_alive`` answer to losing messaging adapters (gateway.on_all_adapters_down).

The policy is the operator's answer to "may one messaging transport take the process down with
it". ``exit`` (default) leaves that to a supervising service manager; ``stay_alive`` keeps the
process up because cron, the control socket and agent ingress need no platform at all.

Both fatal paths read it: the startup gate (``run_startup.py::_start_handle_no_connections``) and
the runtime last-adapter verdict (``run_adapters.py::_handle_adapter_fatal_error_impl``). The loss
this exists for is a sole enabled platform failing NON-retryably (SOU-4201: a logged-out WhatsApp
link exited 78 forty times in 3.96 h, and every run's first model call was refused
``adapter_failed / gateway draining: not_ready`` meanwhile).

A retryable loss stays with the reconnect watcher. A non-retryable one has no recovery path, so it
is parked ``fatal`` (its platform entry already says so) and the run is reported ``degraded``:
stamping a green ``gateway_state`` there would hand a dispatcher a healthy gateway with no
transport left.
"""

from __future__ import annotations

import logging

from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger("gateway.run")


def park_and_report_degraded(runner) -> None:
    """Park the fatally failed platform(s) and publish ``degraded`` — the one way this stays honest.

    Both a boot that connected no platform and a mid-run loss of the last adapter end here, so the
    run's own status can never say ``running`` while a configured transport is unserved.
    """
    from gateway.run import _write_runtime_status_quiet
    runner._startup_parked_platforms = True
    _write_runtime_status_quiet(gateway_state="degraded", exit_reason=None)


def stay_alive_after_losing_the_last_adapters(runner, adapter: BasePlatformAdapter) -> None:
    """Keep the process serving after its last messaging adapter died, degraded if unrecoverable."""
    if adapter.fatal_error_retryable:
        logger.warning(
            "No connected messaging platforms remain; on_all_adapters_down=stay_alive — "
            "gateway staying alive, reconnect watcher owns recovery of %s.",
            adapter.platform.value,
        )
        runner._ensure_reconnect_watcher_running()
        return
    logger.error(
        "No connected messaging platforms remain; %s failed non-retryably (%s) and "
        "on_all_adapters_down=stay_alive — parking it and staying alive: cron / api_server / "
        "dashboard keep serving. Fix the reported error, then `hermes gateway restart`.",
        adapter.platform.value, adapter.fatal_error_code or "unknown",
    )
    park_and_report_degraded(runner)
