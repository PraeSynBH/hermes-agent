"""Spawn-ledger identity and the orphan reconcile, at the one seam EVERY gateway launch reaches.

``gateway/run.py::main()`` is not the only way a gateway starts: the installed service, launchd and
systemd all run ``hermes gateway run`` → ``hermes_cli.gateway.run_gateway()`` → ``start_gateway()``,
which never calls ``main()``. Until PRA-6652 the ledger registration and the reconciler lived inside
``main()``, so on every supervised launch the gateway was invisible to the spawn ledger and
``reconcile_orphans()`` never ran — the exact residue class PRA-6594 exists to remove (measured: a
gateway up 2h13m on that code, ``spawn-ledger.json`` holding 27 cli + 1 mcp-helper + 1
browser-harness-daemon entries and no ``gateway``).

Both steps are best-effort: identity bookkeeping must never fail a gateway boot.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Set by :func:`prepare_process`; one identity pass per gateway process, whichever seam got here first.
_prepared = False


def register_identity() -> None:
    """Record this gateway in the machine spawn ledger (Windows also takes the job object)."""
    from hermes_cli.process_identity import attach_self_to_kill_on_close_job, register_self

    register_self("gateway")
    attach_self_to_kill_on_close_job()


def reconcile_orphans() -> None:
    """Reap the ledger-proven residue of dead sessions. Proof-only; see ``reconcile_orphans``.

    Runs after :func:`register_identity` so this process is the newest entry for its home and can
    therefore never be the "superseded" side of its own sweep.
    """
    from hermes_cli.process_identity import reconcile_orphans as sweep

    reaped = sweep()
    for record in reaped:
        logger.info("reaped orphaned %s pid %s (%s)", record.get("purpose"), record.get("pid"),
                    record.get("hermes_home"))


def prepare_process() -> None:
    """Identity then reconcile, each best-effort, once per gateway boot."""
    global _prepared
    if _prepared:  # both entry points funnel here; the sweep is per-process, not per-call
        return
    _prepared = True
    for step in (register_identity, reconcile_orphans):
        try:
            step()
        except Exception:
            logger.debug("gateway identity step %s failed", step.__name__, exc_info=True)
