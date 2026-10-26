# P18 series — Verilator 5.052 parse-order auto-publicity (root cause of the build 6 codegen explosion)

The p2_10 gate-sim build 6 (2026-10-25, `--flatten --vpi`, wrapper listed
FIRST) generated **3,190,296 per-instance VPI scopes** (~204 MB ctor chunks
× 1597, ~60 GB/h, killed after 357 GB). This series of probes found the
trigger and validated the fix.

## The rule (empirical, 14 consistent runs; source setter not located)

**A module is auto-marked `modPublic` iff its definition is parsed AFTER
the `--top-module` definition** — command-line file order first, then
in-file order. In any output mode (`-cc`, `--lint-only`, ...).

Why publicity matters (verified in generated code, 5.052):
- Every inlined cell gets an `AstCellInlineScope` candidate when `--vpi`
  is on (`V3Scope.cpp:198`) and is registered as a VPI scope candidate
  unconditionally (`V3EmitCSyms.cpp`, `visit(AstCellInlineScope)`).
- The scope is **materialized** into `VpiScopeTable` only if some
  variable's name path walks through it (`varsExpand`/
  `varHierarchyScopes`, `V3EmitCSyms.cpp` ~614).
- Public inlined cells keep per-instance hierarchical var names
  (`top.inst.var` → `topT2__DOT__u1_0__DOT__q` in the `$root` class) →
  the path exists → the scope materializes, one 6-entry VPI var table
  per instance.
- Non-public inlined cells have their vars merged into top-level
  `__Vcellout__*` members (no hierarchical path) → the candidate scope
  is erased. Scope table stays at TOP + top module only.
- `modPublic` sources found in the tree: `$root` wrap, `VAR_PUBLIC`,
  `PUBLIC_MODULE`/`PUBLIC_TASK` pragmas, SystemC, and child→parent
  inheritance (`V3LinkResolve.cpp:488`, LinkBotup — a non-public parent
  becomes public if it instantiates a public child). None of them reads
  parse order; the auto-publicity setter site was not located (the `[P]`
  flag first appears in the `009_param` tree dumps).

## Evidence

| probe | layout | result |
|---|---|---|
| P15d | real PDK stdcells + primitives; top LAST in one file, cells FIRST | `dfrbpq` (instantiates `ihp_dff_r`): 1000 per-instance scopes — public via **child→parent inheritance** (the primitives file parsed after the top made `ihp_dff_r` public); `and2`/`IOPadIn` leaves (no public child): 0 scopes |
| P16 | 6 leaf-cell styles, all non-public | all erased |
| P17 | hand-made `dfrbpq`, non-public alone, but public child `ihp_dff_r` | 20 scopes/instance (botup); other leaves erased |
| P18 | single file, 4 FF-leaf variants, cells BEFORE top | all non-public, all erased (2 scopes total) |
| P18c T1 | `ihp_dff_r` in SAME file as top, before it | non-public |
| P18c T2 | `a1` in its OWN file listed after the top file | **public** |
| P18c T3 | `ihp_dff_r` in its own file listed after the top file | **public** |
| P18e | T2 layout + `--vpi --flatten` | public LEAF retained: 2 per-instance scopes, 6 var entries each |
| P18g | `--lint-only` instead of `-cc` | publicity unchanged → mode-independent |
| P18h h1 | cell file FIRST, top file second | non-public |
| P18h h2 | top file first, cell file second | **public** |
| P18i i1 | ONE file, top defined FIRST, cell after | **public** |
| P18i i2 | ONE file, cell defined FIRST, top after | non-public |
| P18m A/D vs B/C | leaf + 2-level (parent→child) fixtures, both orders, `--vpi --flatten` | A: 2 per-instance scopes; **B: 0**; D: 12 (4 parent + 8 child); **C: 0** |

P18m case C (2-level, parent AND child parsed before the top) closes the
last open question: a non-public parent instantiating a non-public child
stays fully erased — no per-instance scopes at either level.

## The fix

List every module definition BEFORE the `--top-module` definition on the
Verilator command line: netlist → primitives → stdcell (stripped) →
RM×7 → io → **wrapper last**. `run_verilator.sh` does this; the
`FILE ORDER IS FUNCTIONAL` comment marks the block. Expected result on
the real netlist: 2 scopes, ~9-20 var entries.

## Run

Container `iic-osic-tools_asic`, Verilator 5.052 on PATH
(`/foss/tools/verilator`): `bash run_p18m.sh` (all four cases, ~1 min).
`dump_inline_why.py <dot> [module]` prints the inlining-decision nodes
from `--dumpi-graph 6` output.

Reading `run_p18c.sh` output: the awk/grep pipeline drops the `$root`
name line (`__024root @ ...` starts with `_`, not lowercase) but keeps
its attributes, so attribute blocks are off by one node relative to the
name lines — a `NoInlineSoft: Public module` line belongs to the name
line that follows it, not the one before. (Cross-check against the raw
dot with `dump_inline_why.py` when in doubt.)

Footgun hit in this series: a Verilator invocation with NO output-format
flag fatals immediately; behind `|| true` that reads as "module not
public" (missing dot file) — always check the dot/log exists before
counting.
