"""Default ``worktrees`` config section (stale-tree sweep + admission bound).

Split out of ``config_defaults`` as its own unit (#PRX-6144): the section carries an operator
facing comment per key, and the admission/ratchet knobs kept pushing the parent module past its
line ratchet. This is a pure-data leaf — `hermes_cli.config_defaults.DEFAULT_CONFIG` imports the
dict, so the keys and their comments still live in one place.
"""

WORKTREES_DEFAULTS: dict = {
    # Stale git worktree lifecycle (`hermes -w`, `/worktree new`, subagent isolation, Desktop). The
    # sweep runs at `hermes -w` startup and every 6h from the cron ticker over every repo Hermes made
    # a worktree in. Trees idle past their tier (hermes-*/subagent-* 24h, named 72h) that are clean
    # and merged are removed; dirty or unpushed ones are archived to refs/hermes/archive/<tree>
    # first (`hermes worktree restore <name>`). Live trees and Kanban t_<hex> trees are never touched.
    # false = never remove a dirty/unpushed tree (the pre-archive behaviour, exactly).
    "archive_dirty": True,
    # Untracked files larger than this (MB) are not archived, so a tree holding one is kept.
    "archive_max_file_mb": 50,
    # The sweep deletes archive refs older than this many days. 0 keeps them forever.
    "archive_retention_days": 30,
    # Optional size budget (GB) for idle trees: past it the oldest idle trees are archived and
    # removed first. Live, young and Kanban trees are never counted or touched. 0 = off.
    "max_total_gb": 0,
    "max_linked": 0,               # admission cap: linked trees per repo, 0 = no cap (worktree_pool)
    # Per-repo admission caps, keyed by repo basename (`voyonder`) or absolute path. An
    # unnamed repo falls back to `max_linked`; 0 keeps its unbounded meaning. One number
    # cannot say both "this repo runs ~18 at a time" and "that repo needs ~340 trees".
    "max_linked_by_repo": {},
    # The ratchet: after every sweep, lower a repo's cap toward `population +
    # ratchet_headroom` (never up). A cap set once re-freezes the cliff at today's number;
    # the ratchet makes the bound follow the population down as the reaper drains it. A
    # changed configured number starts a fresh ratchet at that value. Off by default: it
    # narrows a number the operator chose, so enabling it is an explicit fleet decision.
    "ratchet": False,
    "ratchet_headroom": 20,        # trees kept above the measured population by the ratchet
    "reclaim_before_admit": True,  # at the cap, one archive-first reclaim pass before refusing
}
