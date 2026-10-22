"""``last_fire_error``: the durable answer on the JOB to "why has this not run?".

A tick can hold a due slot for a reason that is not a job failure — no ready gateway for a headless
tick, or the gateway draining in-flight work. Both leave the slot intact (the job fires at the next
allowed tick, nothing is booked as a run), and both must leave the reason where ``hermes cron list``
reads it. A log line alone is not enough: measured 2026-10-21 (SOU-3252), a drain that skipped 68 min
of a ``*/10`` pager left nothing on disk but a DEBUG line, so every downstream liveness check could
only report the symptom (``last_run_at`` aging) and the reaper's own healing pass looked like the
cause.
"""
from __future__ import annotations

from typing import Iterable


def stamp_fire_error(job_ids: Iterable[str], detail: str, *, drop_marks: bool = False) -> None:
    """Record *detail* as ``last_fire_error`` on every job in *job_ids*.

    ``drop_marks`` also clears the ``pending_slot`` / ``run_claim`` that a MUTATING due scan just
    wrote: the slot was never handed over, so nothing must "restore" or "reclaim" it. Pass it only
    when the caller read the due set through ``get_due_jobs()``; a read-only scan wrote no marks and
    has nothing to drop.

    ``last_fire_error`` is display/liveness state only — scheduling never reads it, and a successful
    fire clears it (``_record_run_outcome``) — so a held slot cannot wedge the job.
    """
    ids = {str(job_id) for job_id in job_ids}
    if not ids:
        return
    from cron.jobs import _hermes_now, _jobs_lock, load_jobs, save_jobs

    with _jobs_lock():
        jobs = load_jobs()
        changed = False
        for job in jobs:
            if str(job.get("id")) not in ids:
                continue
            job["last_fire_error"] = {"at": _hermes_now().isoformat(), "detail": detail[:500]}
            if drop_marks:
                job.pop("pending_slot", None)
                if job.get("run_claim") is not None:
                    job["run_claim"] = None
            changed = True
        if changed:
            save_jobs(jobs)
