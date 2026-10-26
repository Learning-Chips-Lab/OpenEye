# VPI exposure decision probes

Why `run_verilator.sh` needs BOTH the `/*verilator public_on*/` port marks
on the wrapper AND `--flatten` — root-caused on 2026-10-25 after the
p2_10 gate-sim build completed codegen+link but the cocotb run fataled
("Can not find root handle") or, with marks only, read all outputs as
0x0.

## The Verilator 5.052 mechanics (all verified in generated code)

- Verilator's `$root` class ALWAYS holds a mirror copy of the top
  module's ports. The top module's own scope (a VPI scope view named
  after the top module) gets a var table bound to the MIRROR variables
  (`top__DOT__<port>`), while the C++ API and the `$root` scope view see
  the plain top-level port variables.
- VPI var tables are generated only when at least one public variable
  exists. No marks anywhere → no table → cocotb init fatal
  "Can not find root handle" (p2_10 build 5).
- WITHOUT `--flatten`: the marked wrapper survives auto-inlining as its
  own C++ class. The `$root`→wrapper port sync is an ico_sequent copy
  guarded by a REAL clock-edge trigger → cocotb's writes to the wrapper
  scope (the table variables) are clobbered at the next edge → outputs
  read 0x0. The `$root` scope ("TOP", SCOPE_OTHER) is NOT addressable as
  cocotb root handle (`get_root_handle('$root')`/`('TOP')` both fail), so
  driving the live side is impossible.
- WITH `--flatten`: everything is inlined into ONE `$root` class (no
  second class, no per-instance objects). The mirror variables are
  re-synced only ONCE, at the first iteration (ico_sequent guard =
  `1ULL & __VicoTriggered[<firstIteration word/bit>]`); after that they
  are the variables the inlined logic (combinational AND sequential)
  reads, and cocotb drives exactly those. No clobber — including with
  the timing scheduler active (VM_TIMING=1).
- Scope table with `--flatten`: every inlined cell gets a candidate
  scope, materialized only for cells whose module is `modPublic` (their
  vars keep per-instance hierarchical names; non-public cells merge
  vars into top-level `__Vcellout__` members). Verilator 5.052
  auto-marks a module public iff it is parsed AFTER the `--top-module`
  definition — hence `run_verilator.sh` lists all cell files before the
  wrapper (see `p18/` below). Correct order: 2 scopes, ~9-20 var
  entries on the real netlist; reversed order: 3.19M scopes / ~360 GB
  (build 6, killed).
- Degenerate-probe gotcha: a fully constant-foldable delay-free design
  without FFs has no triggers left at all (the ico trigger word holds
  only the firstIteration bit) → VPI writes never wake evaluation → a
  toy probe reads 0x0 even though the structure is fine. The P15b/P15c
  probes therefore carry an FF.

## Probes in this directory

| probe | structure | VM_TIMING | result |
|---|---|---|---|
| P15b (`run_p15b.sh`, `gen_p15b.py`, `p15b_tb.py`) | wrapper (9 marked ports) + 1000-cell flat netlist + FF, no delays | 0 | PASS: comb write propagates without a clock edge (0x123 read back after 1 ns), FF path via posedge/NBA |
| P15c (`run_p15c.sh`, `gen_p15c.py`) | P15b + one `initial #0` (= the real build's scheduling mode) | 1 | PASS: same test, identical single-class structure, mirror sync firstIteration-only |

`p15d/` (subdirectory): functional verdict of the specify-strip
transform on real PDK cells — raw variant FF-FAIL (undriven
`delayed_*` folds to 0), stripped+aliased variant full PASS; both with
the fixed file order → 2 scopes. `p15b_verdict.txt` / `p15c_verdict.txt`:
structural reports (class count, scope table, var-table bindings,
ico_sequent guard) + cocotb verdicts.

Run (container `iic-osic-tools_asic`): `bash run_p15c.sh` (≈2 min).

## `p18/` — the parse-order auto-publicity series (build 6 root cause)

`p18/` contains the full probe series that root-caused the 3.19M-scope
codegen explosion (auto-publicity of modules parsed after the top) and
validated the file-order fix — including the 2-level case (non-public
parent + non-public child → 0 per-instance scopes). See
`p18/README.md` for the rule, the 14-run evidence table, and
`p18/results.txt` for recorded outputs.

The earlier, larger probe matrix (P15b/c raw workspaces, unmarked
builds, H1 with-delay calibration, 10k-instance variants, P15d/P16/P17)
lives outside the repo in `/home/utz/projekte/vpi_micro/` on est-srv02.
