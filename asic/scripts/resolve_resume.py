#!/usr/bin/env python3
"""Resolve the LibreLane resume step (-F <id>) for an existing run dir.

Usage (inside the iic-osic-tools container, where librelane is importable):
    python3 /foss/designs/OpenEye/asic/scripts/resolve_resume.py <run_dir>

<run_dir> is the flow's run directory, e.g.
    /foss/designs/OpenEye/asic/runs/openeye_p2_15/design/runs/p2_15
The design-level config is read from <run_dir>/../../openeye.json (the
config the NEXT launch will use) and overrides the flow's defaults when
evaluating step gating.

Prints the step id to resume at (or nothing) and exits 0:
  - no step dirs at all (fresh run)             -> nothing (full run)
  - first step (in flow order) whose latest dir
    is incomplete (no state_out.json)          -> that step (crashed)
  - otherwise the first step after the frontier
    of completed steps that is NOT gated by the
    current config and has no later step with a
    dir (a later dir means the previous launch
    skipped it via -F -> it must not re-run)   -> that step
  - all steps complete / gated / -F-skipped     -> nothing (flow done)

Rules learned the hard way (see QWEN.md Pitfalls):
  - Step dirs are named "<ordinal>-<slugify(step.id)>" where step.id is
    LIBRELANE'S UNIQFIED id (OpenROAD.STAMidPNR-1, -2, -3, ...). Print
    the same uniquified id: -F <non-uniquified id> would resolve to the
    FIRST occurrence, not the one that is actually next.
  - Gated-off steps never create a dir. The old resolver rule "no dir ->
    resume here" fired on them (it returned OpenROAD.RepairDesignPostGRT
    against p2_15, a step that is off by config). Gating is evaluated
    the same way as librelane/flows/sequential.py does it:
    Chip.gating_config_vars (wildcard keys) expanded against the step
    ids, a step runs only if ALL its gating vars are truthy.
  - A dir is complete iff it holds state_out.json, or holds ONLY
    config.json + state_in.json (output-less steps like
    Checker.PowerGridViolations write no state_out at all).
  - Several dirs per step (crash re-runs, forked runs): the LATEST
    (highest ordinal) decides.

Exit codes: 0 = resolved (possibly empty), 2 = setup error (missing
config / gating var without known default).
"""
import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <run_dir>", file=sys.stderr)
        return 2

    run_dir = os.path.abspath(sys.argv[1])
    if not os.path.isdir(run_dir):
        return 0  # fresh run -> no resume info needed

    try:
        from librelane.common.misc import Filter, slugify
        from librelane.flows.chip import Chip
    except ImportError as e:
        print(f"resolve_resume: cannot import librelane: {e}", file=sys.stderr)
        return 2

    # Design config of the NEXT launch (run dir = <design>/runs/<tag>)
    design_dir = os.path.dirname(os.path.dirname(run_dir))
    cfg_path = os.path.join(design_dir, "openeye.json")
    raw = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf8") as f:
            raw = json.load(f)

    # Flow-level defaults for the gating vars
    defaults = {v.name: v.default for v in Chip.config_vars}

    # Expand the wildcard gating keys exactly like sequential.py does
    step_ids = [cls.id for cls in Chip.Steps]
    gating_for = {}
    for key, vars in Chip.gating_config_vars.items():
        for sid in Filter([key]).filter(step_ids):
            gating_for.setdefault(sid, []).extend(vars)

    missing = sorted(
        {var for vars in gating_for.values() for var in vars
         if var not in raw and var not in defaults}
    )
    if missing:
        print(
            f"resolve_resume: gating vars without config value or flow "
            f"default: {missing}", file=sys.stderr,
        )
        return 2

    def gated(sid: str) -> bool:
        return any(
            not (raw[var] if var in raw else defaults[var])
            for var in gating_for.get(sid, [])
        )

    entries = sorted(
        (e for e in os.listdir(run_dir)
         if "-" in e and e.split("-", 1)[0].isdigit()),
        key=lambda e: int(e.split("-", 1)[0]),
    )
    if not entries:
        return 0  # fresh run: full flow, no -F needed

    # slug -> [(ordinal, dirname)] (several dirs per step happen after
    # crash re-runs; forked runs keep the old run's dirs too). Keep the
    # ORIGINAL dirname: ordinals are zero-padded in the dir names
    # ("01-verilator-lint") and must not be re-formatted.
    by_slug = {}
    for e in entries:
        ordinal, slug = e.split("-", 1)
        by_slug.setdefault(slug, []).append((int(ordinal), e))

    def complete(path: str) -> bool:
        if os.path.isfile(os.path.join(path, "state_out.json")):
            return True
        return set(os.listdir(path)) <= {"config.json", "state_in.json"}

    for i, cls in enumerate(Chip.Steps):
        slug = slugify(cls.id)
        if slug in by_slug:
            latest = os.path.join(run_dir, max(by_slug[slug])[1])
            if not complete(latest):
                print(cls.id)  # crashed (or still running) -> resume here
                return 0
            continue
        if gated(cls.id):
            continue  # never creates a dir -> not a resume point
        later_has_dir = any(by_slug.get(slugify(s.id)) for s in Chip.Steps[i + 1:])
        if not later_has_dir:
            print(cls.id)  # first not-yet-run step after the frontier
            return 0
        # no dir, not gated, but a LATER step has a dir: the previous
        # launch skipped this step with -F -> it must not re-run

    return 0  # every step complete, gated, or -F-skipped: flow is done


if __name__ == "__main__":
    sys.exit(main())
