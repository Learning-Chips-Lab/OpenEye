#!/usr/bin/env python3
"""Synthetic tests for resolve_resume.py (run inside the container).

Builds throwaway run dirs under /tmp/resolve_resume_test (container /tmp,
NOT visible from the host) using the real Chip flow definition, then
checks the resolver's output against the expected resume step.

Scenarios:
  S1 fresh run (no step dirs)                    -> ""
  S2 crashed step mid-flow                        -> that step
  S3 frontier at RepairAntennas (42) complete,
     RUN_POST_GRT_RESIZER_TIMING=true in config   -> ResizerTimingPostGRT
  S4 same frontier, knob unset (default False)    -> STAMidPNR-3
  S5 every ungated step complete                   -> "" (flow done)
  S6 fork-like: step re-run complete (latest),
     a later step crashed                         -> the later step
  S7 fork-like: latest re-run crashed              -> that step

Usage:  python3 /foss/designs/OpenEye/asic/scripts/test_resolve_resume.py
Exits 0 iff all scenarios pass.
"""
import json
import os
import shutil
import subprocess
import sys

from librelane.common.misc import Filter, slugify
from librelane.flows.chip import Chip

BASE = "/tmp/resolve_resume_test"
RESOLVER = "/foss/designs/OpenEye/asic/scripts/resolve_resume.py"

defaults = {v.name: v.default for v in Chip.config_vars}

gating_for = {}
for key, vars in Chip.gating_config_vars.items():
    for sid in Filter([key]).filter([c.id for c in Chip.Steps]):
        gating_for.setdefault(sid, []).extend(vars)


def is_gated(sid: str, raw: dict) -> bool:
    return any(
        not (raw[var] if var in raw else defaults[var])
        for var in gating_for.get(sid, [])
    )


def make_run(tag: str, plan: list, config: dict) -> str:
    """plan: list of (step_index or None, state) for the flow steps in
    order; state is 'complete', 'crashed', or 'missing'. Dir ordinals
    count only the steps that get a dir (like the real flow, which does
    not increment the ordinal for skipped/gated steps)."""
    root = os.path.join(BASE, "design", "runs", tag)
    shutil.rmtree(os.path.join(BASE, "design"), ignore_errors=True)
    os.makedirs(root)
    with open(os.path.join(BASE, "design", "openeye.json"), "w") as f:
        json.dump(config, f)
    ordinal = 1
    for idx, state in plan:
        if state == "missing" or idx is None:
            continue
        slug = slugify(Chip.Steps[idx].id)
        d = os.path.join(root, f"{ordinal:02d}-{slug}")
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"meta": {}}, f)
        with open(os.path.join(d, "state_in.json"), "w") as f:
            json.dump({"views": {}}, f)
        if state == "complete":
            with open(os.path.join(d, "state_out.json"), "w") as f:
                json.dump({"views": {}}, f)
        else:  # crashed: start artifacts + a log, no state_out
            with open(os.path.join(d, "some_step.log"), "w") as f:
                f.write("partial output\n")
        ordinal += 1
    return root


def resolve(run_dir: str) -> str:
    p = subprocess.run(
        [sys.executable, RESOLVER, run_dir],
        capture_output=True, text=True,
    )
    assert p.returncode == 0, f"resolver failed: {p.stderr}"
    return p.stdout.strip()


def full_plan(states: dict, default: str = "missing") -> list:
    return [(i, states.get(Chip.Steps[i].id, default)) for i in range(len(Chip.Steps))]


def main() -> int:
    n_steps = len(Chip.Steps)
    idx = {cls.id: i for i, cls in enumerate(Chip.Steps)}
    ra = idx["OpenROAD.RepairAntennas"]

    # cases: (name, builder -> run_dir, expected output)
    cases = []

    # S1: fresh run
    cases.append((
        "S1 fresh run -> empty",
        lambda: make_run("s1", [(None, "missing")] * n_steps, {}),
        "",
    ))

    # S2: steps 0..5 complete, 6 crashed, rest missing
    def plan2() -> list:
        plan = full_plan({})
        for i in range(6):
            plan[i] = (i, "complete")
        plan[6] = (6, "crashed")
        return plan

    cases.append((
        "S2 crashed step -> that step",
        lambda: make_run("s2", plan2(), {}),
        Chip.Steps[6].id,
    ))

    # S3/S4: everything up to and incl. RepairAntennas complete, rest
    # missing (gated steps get no dir anyway)
    def frontier_plan() -> list:
        plan = []
        for i in range(n_steps):
            if i <= ra:
                plan.append((i, "complete"))
            else:
                plan.append((i, "missing"))
        return plan

    cases.append((
        "S3 frontier 42 complete + RUN_POST_GRT_RESIZER_TIMING=true",
        lambda: make_run("s3", frontier_plan(), {"RUN_POST_GRT_RESIZER_TIMING": True}),
        "OpenROAD.ResizerTimingPostGRT",
    ))
    cases.append((
        "S4 frontier 42 complete + knob unset (default False)",
        lambda: make_run("s4", frontier_plan(), {}),
        "OpenROAD.STAMidPNR-3",
    ))

    # S5: every ungated step complete
    cases.append((
        "S5 all ungated complete -> empty (flow done)",
        lambda: make_run(
            "s5",
            [(i, "missing" if is_gated(Chip.Steps[i].id, {}) else "complete")
             for i in range(n_steps)],
            {},
        ),
        "",
    ))

    # S6 (fork-like): step 6 crashed (dir 7), re-run complete (dir 8 =
    # latest), steps 7-8 complete (dirs 9-10), step 9 crashed (dir 11)
    # -> the duplicate dirs must resolve to the LATEST; resume at 9.
    cases.append((
        "S6 fork-like: latest dir per step decides -> resume at crashed 9",
        lambda: make_run_s6(complete_re=True),
        Chip.Steps[9].id,
    ))

    # S7 (same layout, step 6 re-run is the latest and crashed)
    cases.append((
        "S7 fork-like: latest re-run crashed -> resume at step 6",
        lambda: make_run_s6(complete_re=False),
        Chip.Steps[6].id,
    ))

    fails = 0
    for name, build, expected in cases:
        run_dir = build()
        got = resolve(run_dir)
        ok = got == expected
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got!r} expected={expected!r}")

    shutil.rmtree(BASE, ignore_errors=True)
    print(f"{len(cases) - fails}/{len(cases)} scenarios passed")
    return 1 if fails else 0


def make_run_s6(complete_re: bool) -> str:
    """Fork-like layout: steps 0-5 complete (dirs 1-6); step 6 first
    attempt dir 7 crashed; step 6 re-run dir 8 (complete iff
    complete_re, else crashed); steps 7-8 complete (dirs 9-10); step 9
    crashed (dir 11)."""
    root = os.path.join(BASE, "design", "runs", "s6")
    shutil.rmtree(os.path.join(BASE, "design"), ignore_errors=True)
    os.makedirs(root)
    with open(os.path.join(BASE, "design", "openeye.json"), "w") as f:
        json.dump({}, f)

    def add_dir(ordinal: int, step_idx: int, complete: bool) -> None:
        slug = slugify(Chip.Steps[step_idx].id)
        d = os.path.join(root, f"{ordinal:02d}-{slug}")
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"meta": {}}, f)
        with open(os.path.join(d, "state_in.json"), "w") as f:
            json.dump({"views": {}}, f)
        if complete:
            with open(os.path.join(d, "state_out.json"), "w") as f:
                json.dump({"views": {}}, f)
        else:
            with open(os.path.join(d, "some_step.log"), "w") as f:
                f.write("partial output\n")

    for i in range(6):
        add_dir(i + 1, i, True)
    add_dir(7, 6, False)          # step 6 first attempt: crashed
    add_dir(8, 6, complete_re)    # step 6 re-run (latest dir for step 6)
    add_dir(9, 7, True)
    add_dir(10, 8, True)
    add_dir(11, 9, False)         # step 9 crashed
    return root


if __name__ == "__main__":
    sys.exit(main())
