"""Spelling corpus for the gateway lifecycle guard's launchd job-set passes (SOU-1934).

The guard has been patched six times for launchd bypasses (#62891, #77083, #80269, #113667,
SOU-1850, SOU-1900), and every patch started from one hand-written example. Since SOU-1900 the
guard is a *resolver* (variable indirection, assignment tracking, argv shape), so the next change
to it is more likely to break a neighbouring spelling than the spelling that motivated it. This
module judges the guard by a corpus instead of by examples.

Every row is one spelling, with the verdict it must get and the reason it exists. A row runs
through all three surfaces the production gate uses:

* ``contains_launchctl_registration_command`` — the launchd job-set registration pass
* ``contains_launchctl_removal_command``      — the launchd job-set removal pass
* ``scan_gateway_lifecycle``                  — the whole walk (direct scans + referenced scripts)

``must`` names the surfaces a blocked row MUST trip, ``must_not`` the ones it must not: a row whose
verb is only reachable through the whole walk (a script that hides the program word in a variable)
is a distinct row, and so is a row caught only by the gateway-label regex (``kill"start"``) rather
than by the job-set passes. Adding a spelling is a data edit, nothing else.

Run it two ways::

    python3 tests/cron/guard_spelling_corpus.py            # this checkout's guard
    python3 tests/cron/guard_spelling_corpus.py --live     # the LIVE checkout's guard + disagreements

``--live`` imports ``~/.hermes/hermes-agent/cron/lifecycle_guard.py`` and prints the deployed
state as a red/green table, so "we deployed the fix" is one command to check. It then prints every
row where the live checkout and this checkout disagree — that disagreement is the SOU-1931
verification signal, not a corpus bug.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BLOCKED = "blocked"
ALLOWED = "allowed"
SURFACES = ("registration", "removal", "walk")
LIVE_GUARD_PATH = Path.home() / ".hermes" / "hermes-agent"

# The three surfaces, short enough for a table column.
_HEADERS = ("REG", "REM", "WALK")
_RESIDUAL_PREFIX = "residual — see SOU-1932"


@dataclass(frozen=True)
class Row:
    """One spelling. ``command`` is shell source, or shell source with ``{script}`` filled in."""

    id: str
    command: str
    expected: str
    reason: str
    # Surfaces a blocked row must trip. Allowed rows must trip none of them.
    must: tuple[str, ...] = ()
    # Surfaces a blocked row must NOT trip (its distinctness: walk-only rows).
    must_not: tuple[str, ...] = ()
    # Script body written to a temp file and substituted for ``{script}`` in ``command``.
    script: Optional[str] = None
    residual: bool = False

    @property
    def walk_only(self) -> bool:
        return self.expected == BLOCKED and "walk" in self.must and not (
            set(self.must) - {"walk"}
        ) and bool(self.must_not)


ROWS: tuple[Row, ...] = (
    # --- literal job-set writes ------------------------------------------
    Row(
        "literal-bootstrap",
        "launchctl bootstrap gui/501 ~/Library/LaunchAgents/a.plist",
        BLOCKED,
        "#62891: bootstrap registers a persistent keepalive job",
        must=("registration", "walk"),
    ),
    Row(
        "literal-submit",
        "launchctl submit -l com.example.helper -- /bin/sh",
        BLOCKED,
        "#62891: submit launderers a restart into a keepalive job",
        must=("registration", "walk"),
    ),
    Row(
        "literal-load-legacy",
        "launchctl load -w ~/Library/LaunchAgents/a.plist",
        BLOCKED,
        "load is the legacy spelling of bootstrap — same registration",
        must=("registration", "walk"),
    ),
    Row(
        "literal-bootout",
        "launchctl bootout gui/501/com.example.agent",
        BLOCKED,
        "SOU-1850: removing a human-armed job disarms it silently",
        must=("removal", "walk"),
    ),
    Row(
        "literal-disable",
        "launchctl disable gui/501/com.example.agent",
        BLOCKED,
        "disable is what makes an unload survive a reboot",
        must=("removal", "walk"),
    ),
    # --- quote / escape splice -------------------------------------------
    Row(
        "splice-quote-kickstart",
        'launchctl kick"start" gui/501/ai.hermes.gateway',
        BLOCKED,
        '#80269: the shell resolves kick"start" into kickstart before running it',
        must=("walk",),
    ),
    Row(
        "splice-backslash-kickstart",
        "launchctl kick\\start gui/501/ai.hermes.gateway",
        BLOCKED,
        "#80269: the shell resolves kick\\start into kickstart before running it",
        must=("walk",),
    ),
    # --- label built in an earlier segment --------------------------------
    Row(
        "loop-built-label-bootout",
        'label=${item%%:*}; launchctl bootout "gui/$uid/$label"',
        BLOCKED,
        "#77083: verb and label in different ;-segments; removal is label-independent",
        must=("removal", "walk"),
    ),
    # --- python argv list -------------------------------------------------
    Row(
        "argv-list-bootout",
        'subprocess.run(["launchctl", "bootout", "gui/501/com.example.agent"])',
        BLOCKED,
        "#68289: brackets/commas hide the words from the token walk",
        must=("removal", "walk"),
    ),
    # --- wrappers ---------------------------------------------------------
    Row(
        "wrapper-sudo",
        "sudo launchctl bootstrap gui/501 ~/Library/LaunchAgents/a.plist",
        BLOCKED,
        "#62891: a wrapper must be peeled to the command it runs",
        must=("registration", "walk"),
    ),
    Row(
        "wrapper-env-i",
        "env -i /bin/launchctl bootstrap gui/501 ~/Library/LaunchAgents/a.plist",
        BLOCKED,
        "env -i clears PATH but still runs its operand",
        must=("registration", "walk"),
    ),
    Row(
        "wrapper-command",
        "command launchctl bootout gui/501/com.example.agent",
        BLOCKED,
        "the `command` builtin is transparent",
        must=("removal", "walk"),
    ),
    # --- variable indirection (SOU-1900) ----------------------------------
    Row(
        "indirection-launchctl-variable",
        '"$LAUNCHCTL" bootstrap gui/501 ~/Library/LaunchAgents/a.plist',
        BLOCKED,
        "SOU-1900: the measured shape — arm-host-detectors.sh reaches launchctl through a variable",
        must=("registration", "walk"),
    ),
    Row(
        "indirection-default-stated",
        "${ARM_LAUNCHCTL:-launchctl} bootstrap gui/501 ~/Library/LaunchAgents/a.plist",
        BLOCKED,
        "SOU-1900: ${VAR:-launchctl} states the program in its own default",
        must=("registration", "walk"),
    ),
    Row(
        "indirection-assigned-earlier",
        'LAUNCHCTL=launchctl; "$LAUNCHCTL" bootstrap gui/501 ~/Library/LaunchAgents/a.plist',
        BLOCKED,
        "SOU-1900: the assignment states the program in an earlier ;-segment",
        must=("registration", "walk"),
    ),
    Row(
        "indirection-neutral-name-argv",
        '"$TOOL" bootstrap gui/501 a.plist',
        BLOCKED,
        "SOU-1900: neutral variable name, but argv can only be a launchd write",
        must=("registration", "walk"),
    ),
    Row(
        "indirection-removal-variable",
        '"$LAUNCHCTL" bootout "gui/$uid/$label"',
        BLOCKED,
        "SOU-1850 x SOU-1900: one resolver answers both directions",
        must=("removal", "walk"),
    ),
    Row(
        "indirection-defaulted-assignment",
        'PROG="${ARM_PROG:-launchctl}"; "$PROG" bootout "gui/$uid/$label"',
        BLOCKED,
        "corpus-found (SOU-1934): a neutral name whose assignment states launchctl as its default",
        must=("removal", "walk"),
    ),
    Row(
        "indirection-assigned-alias",
        'LCTLX=launchctl; "$LCTLX" bootstrap gui/501 a.plist',
        BLOCKED,
        "the issue listed this as residual; assignment tracking closes it (verified by the corpus)",
        must=("registration", "walk"),
    ),
    # --- script walk: the verb is hidden in a file the command executes ---
    Row(
        "script-walk-registration",
        "bash {script}",
        BLOCKED,
        "SOU-1900: `bash <file>` inherits the script's verdict — registration through a variable",
        must=("walk",),
        must_not=("registration", "removal"),
        script='LAUNCHCTL="${ARM_LAUNCHCTL:-launchctl}"\n"$LAUNCHCTL" bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/a.plist\n',
    ),
    Row(
        "script-walk-removal",
        "bash {script}",
        BLOCKED,
        "SOU-1850: the same walk, removal direction — bootout through a variable",
        must=("walk",),
        must_not=("registration", "removal"),
        script='PROG="${ARM_PROG:-launchctl}"\n"$PROG" bootout "gui/$(id -u)/com.example.agent"\n',
    ),
    Row(
        "script-walk-readonly",
        "bash {script}",
        ALLOWED,
        "a read-only script using $LAUNCHCTL list is not a job-set write",
        script='LAUNCHCTL="${ARM_LAUNCHCTL:-launchctl}"\n"$LAUNCHCTL" list | head -5\n',
    ),
    # --- MUST stay allowed: the false-positive fence ----------------------
    Row("allow-list", "launchctl list", ALLOWED, "a read, not a write"),
    Row("allow-print", "launchctl print gui/501/com.example.agent", ALLOWED, "a read, not a write"),
    Row(
        "allow-grep",
        'grep "launchctl bootstrap" ~/.paperclip/bin/arm-host-detectors.sh',
        ALLOWED,
        "a data sink's argument is documentation, not execution",
    ),
    Row("allow-echo", 'echo "launchctl bootout x"', ALLOWED, "quoted text is not a command"),
    Row("allow-npm", '"$NPM" install', ALLOWED, "an unrelated program through an unrelated variable"),
    Row(
        "allow-browser",
        '"$BROWSER" load https://example.com',
        ALLOWED,
        "`load` is unrelated here: the argv is not launchd's",
    ),
    Row("allow-brew", '"$BREW" remove somepkg', ALLOWED, "`remove` is unrelated here: the argv is not launchd's"),
    Row(
        "allow-printf",
        'printf "%s" launchctl bootstrap',
        ALLOWED,
        "launchctl sits in printf's argv, never in command position",
    ),
    # --- known-unreachable: the documented boundary, not a bug ------------
    Row(
        "residual-command-substitution",
        "$(printf launchctl) bootstrap gui/501 a.plist",
        ALLOWED,
        f"{_RESIDUAL_PREFIX}: a value built at runtime is not in the text; the action-side shim has to catch it",
        residual=True,
    ),
    Row(
        "residual-split-name",
        'L=launchc; "${L}ctl" bootstrap gui/501 a.plist',
        ALLOWED,
        f"{_RESIDUAL_PREFIX}: the program word is assembled across an expansion boundary",
        residual=True,
    ),
)


@dataclass(frozen=True)
class RowResult:
    row: Row
    observed: dict[str, bool]
    refusal: Optional[str] = None

    @property
    def passed(self) -> bool:
        if self.row.expected == BLOCKED:
            return all(self.observed[s] for s in self.row.must) and not any(
                self.observed[s] for s in self.row.must_not
            )
        return not any(self.observed[s] for s in SURFACES)

    @property
    def why(self) -> str:
        """One-line failure explanation: what was expected vs what the guard did."""
        seen = " ".join(
            f"{short}={mark(self.observed[s])}" for s, short in zip(SURFACES, _HEADERS)
        )
        if self.passed:
            return seen
        if self.row.expected == BLOCKED:
            wanted = "+".join(self.row.must) or "none"
            return f"expected {wanted}; got {seen}"
        return f"expected no surface to fire; got {seen}"


def evaluate(row: Row, guard, workdir: Path) -> RowResult:
    """Run one row through all three surfaces of *guard*."""
    workdir.mkdir(parents=True, exist_ok=True)
    command = row.command
    if row.script is not None:
        script_path = workdir / f"{row.id}.sh"
        script_path.write_text("#!/usr/bin/env bash\n" + row.script, encoding="utf-8")
        command = command.format(script=script_path)
    unsafe, refusal = guard.scan_gateway_lifecycle(command, cwd=str(workdir))
    observed = {
        "registration": bool(guard.contains_launchctl_registration_command(command)),
        "removal": bool(guard.contains_launchctl_removal_command(command)),
        "walk": bool(unsafe),
    }
    return RowResult(row=row, observed=observed, refusal=refusal)


def run(guard, rows: tuple[Row, ...] = ROWS, workdir: Optional[Path] = None) -> list[RowResult]:
    """Evaluate every row. Rows sharing ``{script}`` get their own temp dir, so ids must be unique."""
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="guard-spelling-corpus-") as scratch:
            return [evaluate(row, guard, Path(scratch)) for row in rows]
    return [evaluate(row, guard, workdir) for row in rows]


def counts(rows: tuple[Row, ...] = ROWS) -> tuple[int, int, int]:
    """``(blocked, allowed, residual)`` — residual rows are counted inside *allowed* too."""
    blocked = sum(1 for row in rows if row.expected == BLOCKED)
    residual = sum(1 for row in rows if row.residual)
    return blocked, len(rows) - blocked, residual


def load_guard(root: Path):
    """Import ``<root>/cron/lifecycle_guard.py`` by path, without disturbing this checkout's import."""
    path = root / "cron" / "lifecycle_guard.py"
    spec = importlib.util.spec_from_file_location("_corpus_lifecycle_guard", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mark(value: bool) -> str:
    return "✓" if value else "·"


_COLOR = {"green": "\033[32m", "red": "\033[31m", "dim": "\033[2m", "reset": "\033[0m"}


def _paint(text: str, color: str, enabled: bool) -> str:
    return f"{_COLOR[color]}{text}{_COLOR['reset']}" if enabled else text


def format_table(results: list[RowResult], title: str, paint) -> str:
    width = max(len(r.row.id) for r in results) + 1
    lines = [title, f"{'ROW':<{width}} {'EXPECTED':<8} {'REG':>3} {'REM':>4} {'WALK':>4}  RESULT"]
    for result in results:
        row = result.row
        colour = "green" if result.passed else "red"
        verdict = _paint("PASS" if result.passed else "FAIL", colour, paint)
        detail = "" if result.passed else f"  {_paint(result.why, 'red', paint)}"
        suffix = "  (residual)" if row.residual else ""
        lines.append(
            f"{row.id:<{width}} {row.expected:<8} "
            f"{mark(result.observed['registration']):>3} {mark(result.observed['removal']):>4} "
            f"{mark(result.observed['walk']):>4}  {verdict}{suffix}{detail}"
        )
    return "\n".join(lines)


def summarize(label: str, results: list[RowResult], paint) -> str:
    failed = [r for r in results if not r.passed]
    blocked, allowed, residual = counts(tuple(r.row for r in results))
    head = "GREEN" if not failed else "RED"
    colour = "green" if not failed else "red"
    body = (
        f"{label}: {_paint(head, colour, paint)} — {len(results)} rows "
        f"({blocked} blocked / {allowed} allowed, {residual} residual)"
    )
    if failed:
        body += f", {len(failed)} failing: " + ", ".join(r.row.id for r in failed)
    return body


def disagreements(
    branch: list[RowResult], live: list[RowResult]
) -> list[str]:
    """Rows whose verdict differs between the two checkouts — the SOU-1931 signal."""
    out = []
    for this, deployed in zip(branch, live):
        if this.observed == deployed.observed:
            continue
        before = " ".join(mark(this.observed[s]) for s in SURFACES)
        after = " ".join(mark(deployed.observed[s]) for s in SURFACES)
        out.append(
            f"{this.row.id}: branch {before} / live {after}"
            f" — expected {this.row.expected} ({this.row.reason})"
        )
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the launchd guard spelling corpus (see module docstring).")
    parser.add_argument(
        "--live",
        action="store_true",
        help="also run the LIVE checkout's guard (~/.hermes/hermes-agent) and report disagreements",
    )
    parser.add_argument(
        "--live-root",
        type=Path,
        default=LIVE_GUARD_PATH,
        help=f"checkout to read the live guard from (default {LIVE_GUARD_PATH})",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="ANSI colour for the table (auto = only on a terminal)",
    )
    args = parser.parse_args(argv)
    paint = args.color == "always" or (args.color == "auto" and sys.stdout.isatty())

    # The checkout that CONTAINS this file, not whatever the venv has installed: `--live` then
    # compares two real checkouts instead of comparing one against itself.
    checkout_root = Path(__file__).resolve().parents[2]
    if str(checkout_root) not in sys.path:
        sys.path.insert(0, str(checkout_root))
    if str(args.live_root) not in sys.path:
        sys.path.append(str(args.live_root))
    from cron import lifecycle_guard as local_guard

    branch = run(local_guard)
    print(format_table(branch, "guard spelling corpus — this checkout", paint))
    print(summarize("THIS CHECKOUT", branch, paint))
    failed = any(not result.passed for result in branch)

    if args.live:
        live_guard = load_guard(args.live_root)
        live = run(live_guard)
        print()
        print(format_table(live, f"guard spelling corpus — LIVE checkout  {args.live_root}", paint))
        print(summarize("LIVE", live, paint))
        diff = disagreements(branch, live)
        print()
        if diff:
            print(f"DISAGREEMENTS (this checkout vs live): {len(diff)} row(s) — SOU-1931 signal")
            for line in diff:
                print(f"  {_paint(line, 'red', paint)}")
        else:
            print(_paint("no disagreement: the live checkout and this checkout agree on every row", "green", paint))
        # The deployed state is what a "we deployed" claim has to satisfy.
        failed = failed or any(not result.passed for result in live)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
