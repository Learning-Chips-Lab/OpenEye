# yosys for-loop unrolling tests (IHP SG13G2, yosys 0.70)

Minimal experiments that settle how yosys 0.70 `proc` treats constant-bounded
`for` loops inside **clocked** (`always @(posedge …)`) blocks. Result
(2026-10-23, container `iic-osic-tools_asic`):

**The loops ARE unrolled** — for all of these:

- single / nested loops, unconditional or inside `if` branches,
- loop bodies with or without `begin/end`, with or without `if` in the body,
- several sequential loops sharing the same counter variable in one block,
- counters shared across several always blocks (tb2, tb7, tb9, tb10),
- parameter/constant bounds (tb5/tb6).

What `proc` leaves behind:

1. The **loop counter variable** (module-scope `integer` → 32-bit signed FF)
   persists as a *zombie FF*: its next-state logic is a self-referential mux
   (e.g. `i <= rst_n ? 4 : i`) and it has **zero external fanout** — the loop
   body was emitted with constant indices.
2. If several always blocks drive the same counter variable, `write_verilog`
   emits the merged FF as **several `always` blocks** → a fresh
   `read_verilog + check` reports "multiple conflicting drivers" (32 bits per
   variable). This is exactly the p2_8/p2_9 failure (128 / 64 check errors).
3. The flow's own `opt` passes (`opt_dff` / `opt_clean`, see
   `librelane/scripts/pyosys/synthesize.py`) **remove all zombie counters**,
   so the final netlist is clean and functionally identical to RTL
   simulation — provided the loop bodies themselves are unrolled (they are).

Files:

- `forloop_test.v` … `forloop_test5.v` — minimal modules `tb1` … `tb14`.
  Quick check for one top (container):

  ```
  yosys -p "read_verilog forloop_test.v; hierarchy -top tb1 -check; \
          proc; check; write_verilog -noattr /tmp/tb1.v"
  ```

  - unrolled = body logic uses constant indices (no `i` in the final
    assigns);
  - the zombie counter = `always @(posedge clk) i <= …` (present without
    `opt`, gone after `opt; opt_clean`);
  - multi-driver warning = counter shared by 2+ always blocks (tb2, tb7,
    tb8, tb9, tb10).

- `loopchk_after_proc.ys` — full real design: read all RTL, `hierarchy -top
  OpenEye_Asic -check; proc; check; write_verilog` (→
  `asic/runs/loopchk_after_proc.v`). Confirms: all 15 loop-counter FFs
  (`a, b, word, cc, cc_fc, cc_psum, cr, cr_psum, g, g_psum, pec, per, r, w,
  pipe_idx`) are dead (self-referential only); `a`/`b` are the only ones
  shared by 4 always blocks → 64 conflicting-driver warnings.

- `flowchk_real_design.ys` — same RTL plus the flow's opt sequence
  (`opt_expr; opt_merge; opt_dff; opt_clean; opt_muxtree; opt_reduce`,
  twice) between the two `check` passes. Confirms: 64 warnings after `proc`
  → **0 errors after opt**, and no loop-counter FF left in the netlist
  (→ `asic/runs/flowchk_after_proc_opt.v`).

Both `.ys` scripts use container paths (`/foss/designs/…`); run them inside
`iic-osic-tools_asic` with `yosys -p "$(cat …)"` (`-c` would parse them as
TCL).
