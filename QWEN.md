# QWEN.md — OpenEye ASIC (IHP SG13G2) Handoff

Instructions for continuing the OpenEye ASIC backend work. Read `asic/README.md`
for the phase-by-phase history and this file for the **current state and the
open problem**.

## Objective

Implement the OpenEye DNN accelerator as an ASIC on IHP SG13G2: full RTL→GDS
backend via LibreLane 3.x in the iic-osic-tools container, using the real
`RM_IHPSG13_1P` SRAM macros for the top-level staging buffers.

## SESSION INSTRUCTIONS (user prompt: "Lies QWEN.md und setze es um.")

These instructions are the user's standing orders for every session that
starts with that prompt. They come from the user (2026-10-30 / 11-01).

1. Read this whole file. The authoritative plan is **TODO PLAN v2**
   directly below. Plan v1 and everything after it is history/reference.
2. **p2_15 is stopped and must not be resumed** (cause in v2 section 0).
3. Claude changed code on 2026-11-01 (commits "ASIC v2: floorplan v2,
   latch-RF scratchpads, iact_sc SRAM" and its follow-ups). Read the diffs
   (`git log -5`, `git show <commit>`) and v2 section 1 before working:
   what Claude verified and what not is listed there.
4. **Autonomy mode:** work phases F → G → P → H in this order on your own.
   As soon as all checks of a phase pass, start the next phase without
   asking — including fresh runs (p2_16, p2_16b). Phase I: document
   only, do not start.
5. Per task: (a) implement; (b) run the listed checks and compare the
   numbers with the expected values in the plan (cell counts, 41 macros,
   GRT overflow, WNS); (c) tick it off (`[x]`) in this file and record
   results with numbers, timestamp and log path; (d) one small commit per
   task. At the end of each phase: `git push utz12 dev_asic_sg13g2`.
6. Monitoring: check running flows every 30 min (flow.log, monitor,
   watchdog, `ps`, `free -g`, `df -h`) and note a short status line in
   "Current state". Check the G1 gates actively (GlobalRouting: no
   GRT-0115/0116, overflow ≈ 0, post-GRT timing at most 1 ns worse than
   pre-GRT), ideally automated in `monitor_flow.sh`.
7. Obey the stop criteria at the end of plan v2 strictly: stop, analyze,
   write cause + proposal into "Current state", commit, push, wait for
   the user. Never `--overwrite`, never restart blindly, never delete a
   finished run dir.

## TODO PLAN v2 (2026-11-01, Claude) — START HERE

Plan v1 below (phases A–E) is DONE or SUPERSEDED by this plan. Everything
after plan v1 is history. **Autonomy mode continues** (see SESSION
INSTRUCTIONS above): work the phases in order and move on automatically
when all checks of a phase pass; stop only on the stop criteria at the
end of this plan.

### 0. Starting point (what happened since your last session)

**p2_15 was stopped by Claude on 2026-11-01 ~10:05** (user decision):
host watchdog 410876 first, then the monitors (`monitor_p2_15.sh` and a
stale `monitor_p2_14.sh` 196642), then librelane 52926 and openroad 52943
(step 43 `ResizerTimingPostGRT`, writing SPEF). Run dir
`asic/runs/openeye_p2_15/` is kept as evidence; dir 43 has no
`state_out.json`. **Do not resume p2_15** — its routing is unusable.

**Root cause of the post-GRT collapse (replaces your hypothesis P1
"estimate_parasitics artifact" — the numbers are real routes):**
- Every RM_IHPSG13_1P macro has ALL signal pins incl. `A_CLK` on Metal2
  at its BOTTOM edge (LEF y=0; e.g. 2048x64: 352 pins, x 3.3..780.9) and
  OBS on Metal1..Metal4 over the whole macro (checked in the PDK LEFs).
- Floorplan v1 placed 33 macros R0 in three stacked rows at the core
  bottom, so every pin edge faced a narrow channel: row 1 (wght, 1,600
  pins) 3.7 µm to the core edge, row 2 (psum, 2,416 pins) 18.3 µm to row
  1, row 3 (BUFFER_A, 1,400 pins) 21.7 µm to row 2. ~5,400 pins had to
  escape sideways through 15 µm gaps.
- GRT (p2_15 step 43 log): Metal4 17 % usage but 124,188 overflow,
  Metal2 51 % / 20,087 → local pin-access congestion → multi-mm
  unbuffered Metal2 detours (RC of tens of ns). Worst setup endpoints:
  `PSUM…u_mac0/A_DIN[2]` (−193.9 ns), `wght_buffer…u_mac2/A_ADDR[4]`;
  worst hold `wght_buffer…u_mac0/A_ADDR[3]` (−30.9 ns = A_CLK routed
  through the same channel, very late). RSZ-2021 "no solution meets
  design rules", 1000+ EST-0026 missing routes. DRT follows the GRT
  guides → continuing would not have produced usable numbers.
- The long runtimes came from this congestion too (150+ extra FastRoute
  iterations); with 48 threads step 43 still ran ~35 h at only ~3 CPUs
  busy — GRT (FastRoute) and the resizer barely parallelize.

**Flop breakdown of the p2_15 netlist** (506,219 `sg13g2_dfrbpq_1`,
grouped by hierarchy; 464,212 of them `$_DFFE_PP_` = storage):

| Storage | bits | note |
|---|---|---|
| `pe.weight_data_SPad` (48 PE × 256 × 24) | 294,912 | 58 % of all flops |
| `iact_stream_constructor` RAM (4 × 1024 × 24) | 65,600+ | fell back to flops (see below) |
| `pe.psum_SPad` (48 × 2 × 32 × 20, SPad_DP) | 61,440 | SERIAL=1 → SPad_DP, SPad_DP_RW unused |
| `pe.weight_addr_SPad` (48 × 64 × 8) | 24,576 | |
| `pe.iact_data_SPad` (48 × 16 × 12) | 9,216 | |
| `pe.iact_addr_SPad` (48 × 16 × 4) | 3,072 | |

The iact_stream_constructor RAM was meant to be SRAM (target.sh comment
"1024x72") but its width is `WORD_BITWIDTH = TRANS_BITWIDTH_IACT ×
NUM_GLB_IACT = 24 × 1 = 24` → no `RAM_SP.v` config matched → `g_generic`.

### 1. Changes Claude made (2026-11-01) — read them before you start

Commit "ASIC v2: floorplan v2, latch-RF scratchpads, iact_sc SRAM"
(see `git log`):

1. **Latch register file for the PE scratchpads** —
   `asic/rtl/sg13g2/sg13g2_latch_rf.v` (new): `oe_latch_rf_1w1r`,
   `oe_icg`, `oe_dlatch`. Selected by `OPENEYE_LATCH_RF` in
   `hdl/SPad_SP.v` and `hdl/SPad_DP.v` (else unchanged RAM_SP/RAM_DP
   path). One `sg13g2_dlhq_1` (30.84 µm²) per bit + one `sg13g2_lgcp_1`
   (27.22 µm²) per word + one shared write-data register; replaces
   dfrbpq_1 + enable mux (~67 µm²/bit). Write: ICG latches
   `we & (waddr==w)` in the low phase of the request cycle, `wdata_q`
   captured at the next posedge, word latch transparent in that high
   phase → cycle-equivalent to the flop RAM (read at the same edge sees
   the old value, next cycle the new one; no hold race at the latch
   closing edge). With `OPENEYE_SG13G2_CELLS` the PDK cells are
   instantiated explicitly (no inferred latches → no Verilator LATCH
   lint error, CTS sees real ICGs); without it behavioral models (RTL sim:
   one behavioral latch per WORD, `g_word[w].word_q` — the first
   per-bit-module version made iverilog elaborate >12 min on the full
   chip, killed; log `asic/runs/regress_macros_latchrf_perbit_killed.log`).
   Instance name `ram` is kept, so ASIC hierarchical paths are
   `…SPad.ram.g_word[w].g_bit[b].u_lat.u_lat`. NOTE: `test/cocotb_PE_cluster/
   PE_cluster_tb.py` and `src/open_eye/rtl_test_utils.py:566` peek into
   `….ram.impl.mem` — they only work WITHOUT `OPENEYE_LATCH_RF` (the
   ASIC regressions do not use them).
2. **iact_stream_constructor RAM → SRAM**: config `(24, 10, WM=2)` =
   `RM_IHPSG13_1P_1024x16` (d[23:8]) + `RM_IHPSG13_1P_1024x8` (d[7:0])
   in `asic/rtl/sg13g2/sg13g2_ram_sp.v` (`g_cfg1024x24`), enabled in
   `hdl/RAM_SP.v`; 1024x16 blackbox in `sg13g2_sram_blackbox.v`; model
   added to `regress_macros.sh`, `asic/sim/gatesim/Makefile`,
   `run_verilator.sh`; `MACRO_TYPES` in `gen_config.py`. **41 macros now.**
3. **Floorplan v2** — `asic/scripts/macro_placement.py` rewritten (header
   documents the rules, asserts them): two macro rows only, one macro
   deep, along the BOTTOM (orientation **MX**, pins on top) and TOP
   (orientation **R0**, pins at the bottom) core edges; pin edges face the
   >8 mm open core interior; ≥30 µm between macros (actual 58–83 µm);
   10 µm back-side margin. Grouped per cluster:
   bottom `[psum X0Y0 g0 g1][iact X0Y0][wght ×5][iact X1Y0][psum X1Y0 g0 g1]`,
   top `[psum X0Y1 g0 g1][iact X0Y1][BUFFER_A ×4][iact X1Y1][psum X1Y1 g0 g1]`.
   Orientation semantics verified with OpenDB (`dbInst.setOrient` +
   `setLocation` = lower-left of the oriented bbox): MX at y=335 → A_CLK
   at y=961.44–961.70 (top edge), R0 at y=9545 → A_CLK at y=9545 (bottom
   edge). The 8 new iact instance names
   (`u_openeye.IACT_CONVERTER_X[x].IACT_CONVERTER_Y[y].iact_stream_
   constructor.BUFFER[0].iact_buffer_SP.g_specific.impl.g_cfg1024x24.
   u_mac0/1`) are derived from the p2_15 netlist naming but NOT yet seen in
   a real synthesized netlist → check in F2.
4. **`gen_config.py`**: `VERILOG_DEFINES += OPENEYE_LATCH_RF,
   OPENEYE_SG13G2_CELLS`; `sg13g2_latch_rf.v` in `VERILOG_FILES`;
   `FP_MACRO_HORIZONTAL/VERTICAL_HALO: 20` (std-cell keep-out, room for
   pin access; `GRT_MACRO_EXTENSION` deliberately left 0 — it enlarges
   the routing blockage). Your A1/A5/A7 knobs are unchanged.
   `PL_TIMING_DRIVEN` stays off for this run (runtime; see I2).
5. **Tests** — `asic/sim/tb_latch_rf_equiv.v` (new): latch RF vs
   RAM_SP_generic (256×24, 64×8, 16×12) and RAM_DP_generic (32×20) with
   random stimulus incl. write+read (write wins) and DP concurrent R/W.
   `asic/sim/tb_ram_equiv.v`: cfg4 (24×1024, WM=2) added.

**Verified by Claude (2026-11-01, container):**
- `tb_latch_rf_equiv` behavioral models: **PASS** (200,000 cycles,
  699,358 defined comparisons).
- `tb_latch_rf_equiv` with the REAL PDK cells (`-DOPENEYE_SG13G2_CELLS`,
  `pdk_specify_strip.py` copy of `sg13g2_stdcell.v` + `sg13g2_udp.v` —
  iverilog cannot compile the original specify blocks): **PASS**, same
  numbers.
- `tb_ram_equiv` cfg0–cfg4 with PDK functional SRAM models: **PASS**.
- `regress_macros.sh` (full FPGA cocotb test, SRAM macros + latch RF):
  **PASS** (2026-11-01 10:52, `OpenEye_FPGA_tb.start_test_fpga`, 10210 ns,
  238 s wall; log `asic/runs/regress_macros_latchrf.log`).
- `macro_placement.py`: 41 instances, all assertions pass;
  `gen_config.py` writes 7 macro types / 41 placed instances.

**NOT yet verified (your job, F1–F3):** `regress.sh` (generic path),
Verilator lint + yosys synthesis of the new RTL in the flow, the macro
instance names, the gate-level sim of the new netlist.

### Phase F — pre-flow checks (blocking)

- [x] **F0** Read the diff of Claude's commit; `git log -3`. Start the
  monitors only for runs that exist.
  **Done 2026-11-01 10:55:** diffs of `1880b6b` (floorplan v2 + latch RF
  + iact_sc SRAM) and `4caea86` (word-level behavioral sim model) read in
  full (gen_config.py, macro_placement.py, sg13g2_latch_rf.v,
  sg13g2_ram_sp.v cfg4, sg13g2_sram_blackbox.v 1024x16, SPad_SP/SPad_DP
  hooks, RAM_SP.v cfg condition); `git log -3` = a7b5421/4caea86/1880b6b.
  No flow running (p2_15 stopped, zombie procs only); the STALE p2_15
  monitor (PID 411032) was killed — p2_15 stays evidence, no monitor.
- [x] **F1** `regress.sh` (generic RAMs, no ASIC defines) must still
  PASS. `regress_macros.sh` must PASS (Claude's result above; rerun
  only if the RTL changed since).
  **Done 2026-11-01 10:57:** `regress.sh` PASS — TESTS=1 PASS=1 FAIL=0,
  10210 ns, 4.50 s sim (log `asic/runs/regress_generic_latchrf.log`).
  `regress_macros.sh`: Claude's 10:52 PASS valid (log
  `asic/runs/regress_macros_latchrf.log`; last RTL change 10:43:56
  < log 10:52:24) — not re-run.
- [x] **F2 — synthesis gate of p2_16.** Add a stop-at option to the
  launch chain first: `run_openeye.sh <run> [resume-step] [stop-step]`
  (→ `librelane … -F <resume> -T <stop>`), and `watchdog.sh` passes it
  through. With a stop step, success = the stop step's latest dir has
  `state_out.json` (there is no `final/`), the watchdog then exits 0
  without restart. Test with `DRY_RUN=1`. Then launch the fresh run
  **p2_16** with stop step `OpenROAD.STAPrePNR` (flow step 12).
  Checks (all must hold, record the numbers):
  - steps 1–4 lint clean (no LATCH / unknown-module errors;
    `sg13g2_lgcp_1`/`sg13g2_dlhq_1` must resolve via LibreLane's
    `CELL_VERILOG_MODELS` blackboxes),
  - step 7 unmapped cells 0, step 8 synth checks 0 errors,
  - `06-yosys-synthesis/reports/stat.rpt` (explicit cells whose output is
    unused may be removed by opt_clean, so allow a few % below):
    `sg13g2_dlhq_1` ≈ **393,216**
    (294,912 + 61,440 + 24,576 + 9,216 + 3,072); `sg13g2_lgcp_1` =
    **19,968** (48 × (256 + 64 + 16 + 16) + 48 × 2 × 32);
    `sg13g2_dfrbpq_1` ≪ 506,219 (expect ~15–30k); RM macros = **41**
    (4 × 1024x16 and 4 × 1024x8 new); chip area ≈ 30–36 mm² (was
    55.3). Big deviations → stop and analyze (e.g. yosys removed or
    duplicated cells).
  - step 11 `CheckMacroInstances` PASS = all 41 configured instance names
    exist. If it fails on the iact names: take the real names from
    `grep RM_IHPSG13_1P_1024x16 06-yosys-synthesis/*.nl.v` and fix the
    `I` template in `macro_placement.py` (commit), then resume `-F
    OpenROAD.CheckMacroInstances`.
  - step 12 `STAPrePNR`: no errors; note worst setup/hold per corner and
    check that latch paths are timed (look for the `dlhq` / ICG
    clock-gating checks in the reports; unconstrained endpoints must not
    jump).
   **Done 2026-11-01 12:25:** all steps 1–12 PASS, run stopped at
   `OpenROAD.STAPrePNR` (see Current state, 12:25 line). stat.rpt:
   dlhq 393,216 / lgcp 19,968 exact, dfrbpq **51,883** — the 15–30 k band
   was based on a wrong assumption (p2_15 "other" = 43,585, not 14,699);
   pre-abc census (now `asic/scripts/dff_census.py`) accounts the whole
   −487,392 DFF delta to the latch-RF structural change, "other" logic
   bit-identical → benign, not a stop criterion. RM 41, 32.78 mm²,
   CheckMacroInstances 41/41, STAPrePNR hold WNS identical to p2_15
   (−0.514/−0.966/−0.665), setup WNS a pre-PNR baseline artifact,
   violators 49–59 k vs 400 k+. Launch deviation: steps 1–12 ran on the
   base config (48 threads, 26-thread overlay not yet applicable) —
   artifact-neutral.
- [ ] **F3 — gate-level sim of the new netlist, in parallel to the flow.**
  As soon as `06-yosys-synthesis/OpenEye_Asic.nl.v` exists, start
  `asic/sim/gatesim/run_verilator.sh` on it (detached, logs in
  `asic/runs/gatesim_p2_16.log`). `start_test_fpga` must PASS. It now
  covers the real `sg13g2_lgcp_1`/`sg13g2_dlhq_1` cells + the new SRAM
  config end-to-end. The flow does not wait for it, but a FAIL is a stop
  criterion (kill the flow, analyze).
  **Done 2026-11-01 15:02: PASS.** `start_test_fpga` PASS,
  TESTS=1 PASS=1 FAIL=0 (10210 ns, 4.72 s sim; build 11:41 → 15:02,
  ~3.4 h vs p2_10 ref 8.4 h for the bigger netlist; log `asic/runs/
  gatesim_p2_16.log`, results.xml timestamp 14:02:27 UTC). The new
  netlist (latch RF + 41 RM configs) is functionally correct at gate
  level — one stop criterion cleared.

### Phase G — p2_16 physical run with gates

- [ ] **G1** Resume p2_16 with `-F OpenROAD.Floorplan` and stop step
  `OpenROAD.ResizerTimingPostGRT` (flow step 44). Gates, checked by you
  (extend `monitor_flow.sh` so it flags them automatically):
  - step 18: "Successfully placed 41 instances"; readback with
    `asic/scripts/macro_positions.py` on the step-18 DEF: 21 macros MX
    at y=335, 20 R0 at the top, coordinates = `macro_placement.py`
    output.
  - step 22 GeneratePDN: no PDN-0235 (PSM-0069 on pad vdd is the known
    PDK gap).
  - step 27 GlobalPlacement: final overflow ≤ 0.1; note the routability
    inflation messages.
  - step 34 CTS: number of clock sinks ≪ p2_15 (ICGs + ~20k flops +
    41 macros instead of 506k flops); skew per corner.
  - step 37 (STAMidPNR after ResizerTimingPostCTS): nom_typ setup WNS
    ≥ 0, hold WNS ≥ −0.5 ns.
  - **step 38 GlobalRouting — the key gate**: no `GRT-0115/0116`, total
    overflow 0 (tolerate ≤ 0.01 % of demand), few or no
    "extra iteration" lines, no `EST-0026` storm. Runtime should be
    hours, not 14 h.
  - step 44 ResizerTimingPostGRT: setup WNS (nom) within 1 ns of the
    pre-GRT value, hold WNS ≥ −0.5 ns, no RSZ-2021 storm.
  Any gate failing → stop criterion (analyze; most likely levers:
  GPL density/padding, halos, GRT layer adjustments).
  **STOPPED 2026-11-01 18:1x — G1-27 gate FAIL (overflow 28.54 vs ≤ 0.1).**
  Gates so far: 18 placed=41 OK, 22 PDN OK, **27 overflow 28.5388 FAIL**
  (inflation 0.00 %, single iteration; p2_15 ref 1039.35 across 4
  iterations; p2_16b identical 28.5388 → reproducible geometry
  property). **Decision (a), 18:2x: accepted with deviation, both
  runs resumed** (p2_16 at step 31, p2_16b at step 30; see Current
  state, 18:2x line). Analysis + proposals in the 18:1x line.
- [ ] **G2** All G1 gates green → resume `-F OpenROAD.STAMidPNR`? NO —
  `STAMidPNR` occurs 4× in the flow; resume with the watchdog's resolver
  (no stop step), which picks the first incomplete step after 44. Run to
  the end (DRT → fill → RCX → STAPostPNR → GDS → DRC → LVS → gates).
  DRT: log the violation count per iteration; it must converge to 0.

### Phase P — use the idle cores (start right after F2 passed)

One run uses the machine poorly (GRT/resizer ≈ 1–3 cores, synthesis 1
core, only DRT/GPL/CTS scale). Use the spare cores for a SECOND run in
parallel that answers the next question with the same wall time:

- [ ] **P1** Add a config-overlay option to `run_openeye.sh`
  (`CONFIG_OVERLAY=<json>` env → merged into the run's
  `design/openeye.json` after the copy, keys override; record the
  overlay file in the run dir). Test with `DRY_RUN=1`.
- [ ] **P2** Launch **p2_16b** = p2_16 at **66 MHz** (`CLOCK_PERIOD: 15`,
  `IO_DELAY_CONSTRAINT` so that period × % ≥ 6.4 ns → 43) only AFTER p2_16
  has passed step 12 (the fresh launch re-stages `hdl/` via `gen.sh`;
  identical content, but don't race step 6). Threads: p2_16 and p2_16b
  each `OPENROAD_THREADS: 26`, `KLAYOUT_DRC_THREADS: 26` (overlay for b;
  for p2_16 the next resume picks up the new config). RAM: watch
  `free -g`, each openroad peaked ~55 GB on the old netlist. Same gates
  as G1/G2. If p2_16 fails a gate, kill p2_16b too (same floorplan).
- (Optional, only if both run cleanly through GRT:) a third variant
  with `PL_TIMING_DRIVEN: true` at 15 ns.

### Phase H — evaluation and report

- [ ] **H1** For p2_16 (and p2_16b): table per corner of setup/hold
  WNS/TNS (STAPostPNR), area (stdcell / macro / utilization), instance
  count, clock skew, power report if present, runtime per step, DRC/LVS
  result classified against the 3 documented PDK gaps.
- [ ] **H2** Gate-sim of the FINAL netlist (`final/nl/…`) with
  `run_verilator.sh` → PASS.
- [ ] **H3** Summary in "Current state", commit, `git push utz12
  dev_asic_sg13g2`. Leave the IO-delay question (6.4 ns external
  clock-to-pad assumption) in "Blocked / open threads" for the user.

### Phase I — next levers (document only, do not start without the user)

- **I1 weight_data_SPad as SRAM**: 256×24 per PE = `RM_IHPSG13_1P_256x16`
  (236.8 × 118.78 µm) + `256x8` (236.8 × 74.1 µm) ≈ 0.046 mm²/PE vs.
  ≈ 0.19 mm² latches + the same 256:1 read mux — ~4–6× smaller, but 96
  more macros inside the PE fabric (floorplan per PE, pin edges again).
- **I2** `PL_TIMING_DRIVEN`, GPL cell padding, for 66–100 MHz.
- **I3** Die shrink: with ~35 mm² stdcell the 10.5 mm die is half
  empty; floorplan v2 needs ~8.9 mm core width per macro row, so a
  smaller die needs macros on the left/right edges too (R90/R270, pins
  then on the side edges — check access).
- **I4** 100 MHz: re-pipelining of the cross-cluster paths (p2_12
  analysis).
- **I5** SPad_DP_RW latch version (only needed for SERIAL=0), clock
  gating of the remaining flops, sg13g2_io liberty for pad timing, SRAM
  fast-corner liberty m55C vs. stdcell m40C.

### Stop criteria (stop, analyze, write cause + proposal to "Current state", commit, push, wait for the user)

- A check of phase F/G fails and cannot be fixed in 2 attempts.
- A regression, `tb_*` unit test or gate-sim FAILS.
- A new error class in the flow, or the same step fails twice.
- A G1 gate fails (esp. GRT overflow / post-GRT timing).
- DRC/LVS errors outside the 3 documented PDK gaps.
- Free disk < 200 GB or OOM.
- Anything that would delete or overwrite a finished run dir.
Never restart blindly; every restart needs a documented cause and fix.

## TODO PLAN v1 (2026-10-30, review by Claude) — DONE / SUPERSEDED by v2

Everything below this section up to "Key commands" is **history**; the
"Next steps (as of 2026-10-27)" section is superseded by this plan.
Work the phases in order. Ground rules for every task:

- One task = one small commit (config/script change + matching QWEN.md
  note). Never `--overwrite`. Never kill p2_15 without the user's OK.
- "Verify" lines are mandatory — a task is only done when its check passed.
  Record results (numbers, timestamps) in "Current state".
- Ask the user before any action that discards compute time (killing a
  running step, starting a new run name).

### Phase A — prepare fixes while p2_15 keeps running (no restart yet)

- [x] **A1 (CRITICAL) — OpenROAD runs single-threaded in EVERY step.**
  Evidence: `ps` shows `openroad -exit -threads None …` at 99.7 % CPU on
  a 56-core host; `warning.log` has 19× `[ORD-0032] Invalid thread number
  specification: None.`; flow.log `OpenROAD will use None threads`
  (same in p2_14). Root cause: LibreLane 3.1 bug in
  `librelane/steps/openroad.py` `get_command()`:
  `threads = str(self.config["OPENROAD_THREADS"]) or str(_get_process_limit())`
  — `str(None)` is the truthy string `"None"`, the fallback never fires.
  This is why GRT took 14:42 h, resizer 10:40 h, CTS 4:13 h; DetailedRouting
  (heavily parallel) would take days single-threaded.
  Fix: add `"OPENROAD_THREADS": 48` to the config dict in
  `asic/scripts/gen_config.py` (with a comment citing the bug; 48 leaves
  headroom on 56 cores, RAM is fine: openroad ~23 GB, host 251 GB).
  Verify: regenerated `openeye.json` contains it; after relaunch (Phase B)
  flow.log says `OpenROAD will use 48 threads`, `ps` shows `-threads 48`
  and >100 % CPU, no new ORD-0032 in `warning.log`.
  **Done 2026-10-30 18:35:** `"OPENROAD_THREADS": 48` in `gen_config.py`
  (comment cites the `str(None)` bug). Regenerated
  `asic/rtl/sg13g2/openeye.json` — diff vs. committed version = exactly
  that one knob (1 line, no other change). The running flow's copy
  `asic/runs/openeye_p2_15/design/openeye.json` untouched (mtime
  29.10. 19:13, no `OPENROAD_THREADS`). The live part of Verify
  (`-threads 48`, >100 % CPU, no new ORD-0032) is checked in Phase B
  (B5) after the relaunch.

- [x] **A2 (CRITICAL) — watchdog resume resolver picks SKIPPED steps.**
  Run read-only against p2_15 (2026-10-30 18:10) it returned
  `OpenROAD.RepairDesignPostGRT`: that step is gated off by config and
  never creates a step dir, so the resolver's "no dir → never ran →
  resume here" rule fires on it. A restart now would re-run
  CheckAntennas + DiodesOnPorts + RepairAntennas (7 h+). Same trap for
  every later skipped step (HeuristicDiodeInsertion, IR drop, EQY, …).
  Fix in `asic/scripts/watchdog.sh` `resolve_resume_step()`: resume at
  the step **after the last complete step in flow order**, i.e. walk the
  flow, remember the last step with a complete latest dir; a step with an
  incomplete latest dir is the resume point; steps without any dir are
  ignored (skipped) unless no later step has a dir. Factor the Python
  into its own file (e.g. `asic/scripts/resolve_resume.py <run_dir>`) so
  it can be tested standalone.
  Verify: against p2_15 it must print `OpenROAD.RepairAntennas` while
  step 42 is running (dir without `state_out.json`), and the step after
  it once `42-…/state_out.json` exists. Also test against a p2_14 copy /
  synthetic dirs: fresh run → empty output; crashed step → that step.
  **Done 2026-10-30 18:41:** logic factored into
  `asic/scripts/resolve_resume.py <run_dir>` (standalone, container-side),
  `watchdog.sh resolve_resume_step()` now just calls it. Rules: crashed/
  running step (latest dir w/o `state_out.json`) → that step; otherwise
  first step after the frontier of completed steps that is NOT gated by
  the design config (gating evaluated like `sequential.py`:
  `Chip.gating_config_vars` expanded, defaults from the flow, overrides
  from `<design>/openeye.json`); a no-dir step followed by a step WITH a
  dir = -F-skipped by the previous launch → must not re-run; fresh run →
  nothing. Prints the UNIQFIED step id (`OpenROAD.STAMidPNR-3`, ...),
  because `-F <plain id>` would resolve to the first occurrence.
  Verified (2026-10-30 18:41): `bash -n`, `py_compile`; live p2_15 →
  `OpenROAD.RepairAntennas` (step 42 running);
  `asic/scripts/test_resolve_resume.py` (synthetic runs in container
  /tmp from the real Chip step list) → **7/7 PASS**: fresh → empty;
  crashed step → that step; frontier 42 complete +
  `RUN_POST_GRT_RESIZER_TIMING=true` → `OpenROAD.ResizerTimingPostGRT`;
  same frontier + knob unset → `OpenROAD.STAMidPNR-3`; all ungated
  complete → empty (flow done); fork-like duplicate dirs (latest wins,
  both variants). The "42 complete" case is re-checked live in Phase B
  (B3). NOTE: the B3 expectation (ResizerTimingPostGRT) requires the A7
  knob — with p2_15's current design config the resolver would print
  `OpenROAD.STAMidPNR-3` (resizer gated off by default).

- [x] **A3 — `run_openeye.sh`: don't lose logs, don't change RTL on resume.**
  (a) `tee` overwrites `flow.log` on every (re)launch — a watchdog crash
  restart destroys the crash evidence. Use `tee -a` and write a
  `=== launch <date> resume=<step> ===` separator line first.
  (b) On resume (2nd arg set) skip `gen.sh` and `gen_top.py` (RTL must not
  change mid-run); only `gen_config.py` runs (config knobs may change).
  Optionally store `git rev-parse HEAD` + `git status --porcelain hdl asic/rtl`
  in the run dir at fresh start and warn on resume if it differs.
  (c) Success check: instead of `grep "ERROR|Traceback"` on a 500 MB log,
  use librelane's exit code (`set -o pipefail` + `${PIPESTATUS[0]}`) and
  the presence of `runs/<tag>/final/`.
  Verify: `bash -n`; dry-run the staging part with a dummy run name.
  **Done 2026-10-30 18:44:** all three implemented (a: host-side
  `>> flow.log` append + separator written before the launch; b: resume
  path runs only `gen_config.py`, git state recorded at fresh start in
  `design/launch_git_state.txt` and diffed on resume with `[WARN]`;
  c: success = librelane exit 0 AND `design/runs/<tag>/final/` present —
  librelane saves the final snapshot only after ALL steps completed; the
  "finished without ERROR lines" marker is kept verbatim for the
  watchdog's grep). Added `DRY_RUN=1` mode (stops after staging) for
  testing. Verified (18:43–18:50): `bash -n`; DRY_RUN fresh with dummy
  name `p2_drytest` (separator in flow.log, git state file, gen.sh ran —
  hdl mtimes updated); DRY_RUN resume with `-F OpenROAD.Floorplan`
  (hdl/ + OpenEye_Asic.v mtimes UNCHANGED, only openeye.json rewritten,
  2nd separator appended); git-state warning fires on a temporary
  `hdl/PE.v` modification and is clean after revert. Dummy run dir
  deleted.

- [x] **A4 — monitor false STALL.** `monitor_p2_15.log` reports
  "STALL (no activity 165 min)" while openroad is at 100 % CPU. Add a
  liveness check on the CPU time (`ps -o times=` of the newest
  openroad/yosys/klayout/magic process in the container); STALL only if
  log, step files AND cpu time are all flat.
  Verify: restart the monitor for p2_15, no STALL while RepairAntennas runs.
  **Done 2026-10-30 18:47:** tracked generic script
  `asic/scripts/monitor_flow.sh <runname>` (per-run instances in
  asic/runs/ are gitignored; the new one writes the same
  `asic/runs/monitor_<run>.log`). Liveness = newest non-zombie tool
  process (host ps sees the container processes; they share the PID
  namespace): a new process or increasing cumulative CPU seconds
  (`times`) resets the CPU-flat timer. STALL now requires log flat AND
  no step files newer than STALL_S AND CPU time flat — all at once.
  Snapshot line gained `cpu=<pid>:<seconds>s`. Old monitor PID 330001
  killed, new one (PID 397962) launched 18:45; first snapshot
  18:45:02 shows `cpu=370311:28691s`, no STALL (the 18:25/18:30
  false STALLs were the buffered-log artifact). No-STALL while
  RepairAntennas runs continues to be watched in the 30-min status
  notes.

- [x] **A5 — make setup a real signoff gate.** `Checker.SetupViolations`
  is a no-op (`SETUP_VIOLATION_CORNERS` unset). Add
  `"SETUP_VIOLATION_CORNERS": ["*"]` to `gen_config.py` (50 MHz setup is
  met with +4.25 ns post-resizer, so this should pass; if it fails, that
  is a real finding).
  **Done 2026-10-30 18:52:** knob in `gen_config.py` (comment explains
  the default `[""]` no-op); regenerated `openeye.json` — diff = exactly
  `"SETUP_VIOLATION_CORNERS": ["*"]`. Variable name verified in
  librelane source: `TimingViolations.get_corner_variable()` derives
  `SETUP_VIOLATION_CORNERS` from `violation_type="setup"`, default `[""]`
  (matches no corner); `HoldViolations` has `corner_override=["*"]`
  (already a hard gate). The gate itself runs at the end of p2_15
  (step 79 `Checker.SetupViolations`, ungated).

- [x] **A6 — comment fix in `gen_config.py`:** "70 % PE-pad latches" is
  wrong — the netlist has no latches; area is 506,219 `sg13g2_dfrbpq_1`
  (24.8 of 55.3 mm², ~45 %, async-reset DFFs).
  **Done 2026-10-30 18:58:** the wrong "70 % PE-pad latches" comment was
  already removed by `2711d09` (README fact-check, replaced with the
  measured p2_14 floorplan metrics). What was missing is now in the
  `DIE_AREA` comment: the netlist has NO latches; of the 55.3 mm²
  synthesized area, 24.8 mm² (~45 %) are the 506,219 async-reset DFFs
  (`sg13g2_dfrbpq_1`). Comment-only change — regenerated
  `openeye.json` is byte-identical (verified by diff).

- [x] **A7 (new, found while implementing A2, 2026-10-30) — the
  post-GRT resizer is OFF by default, contradicting the plan's own
  assumptions.** `RUN_POST_GRT_RESIZER_TIMING` defaults to False
  (librelane `classic.py`; p2_15 `resolved.json` confirms: False). So
  `OpenROAD.ResizerTimingPostGRT` (flow step 44) never runs: the four
  `GRT_RESIZER_*_REPAIR_TNS_PCT` caps in `gen_config.py` are inert, and
  the p2_14 analysis line "the post-GRT hold resizer (capped) takes the
  41" (remaining hold violators, WNS −0.44 pre-GRT, all on SRAM-macro
  inputs) never happens — the final hold gate could then trip the stop
  criterion (final Hold-WNS < −0.5 ns). Fix:
  `"RUN_POST_GRT_RESIZER_TIMING": True` in `gen_config.py` (runtime is
  bounded by the 1 % TNS caps; at post-GRT the hold TNS to repair is
  small, ~−4.65 ns). Verify: knob in regenerated `openeye.json`; A2
  resolver then prints `OpenROAD.ResizerTimingPostGRT` for the 42
  frontier (test S3).
  **Done 2026-10-30 19:05:** knob added to `gen_config.py` (comment
  cites the default-False in librelane classic.py, the p2_15
  resolved.json evidence, the 41-hold-violator motivation, and the
  TNS-cap runtime bound); regenerated `openeye.json` — diff = exactly
  `"RUN_POST_GRT_RESIZER_TIMING": true`. Resolver end-to-end with the
  REAL regenerated openeye.json as design config (synthetic frontier
  run in container /tmp, 41 complete dirs): prints
  `OpenROAD.ResizerTimingPostGRT` (test S3 of test_resolve_resume.py
  covers the minimal-config variant). Takes effect at the Phase B
  relaunch; next step dir after 42 will be
  `43-openroad-resizertimingpostgrt`.

### Phase B — apply A1–A7 to p2_15 (Autonomie-Modus: user OK granted 2026-10-30)

- [x] **B1** Let step 42 `RepairAntennas` finish (7 h+ invested, do not
  kill it). Watch for `42-openroad-repairantennas/state_out.json`.
  **Done 2026-10-30 22:45:** state_out.json appeared 22:45 (watcher
  fired). Step 42 took 12:00 total (10:45 → 22:45): 11 h
  diode-insertion repair (12,970 → 420 antenna violations, 3,600+
  diodes/jumpers) + 45 min final `check_antennas -verbose` substep.
  Final cell count 3,125,970. Odb 2.5 GB + DEF written.
- [x] **B2** Then immediately: (1) stop the host watchdog for p2_15 FIRST
  (otherwise it may auto-restart), (2) stop the monitor, (3) kill
  librelane + its openroad child in the container. Check the next step
  dir (43-…) has no `state_out.json` (it was interrupted) — that's fine.
  **Done 2026-10-30 22:50:** killed in the planned order — watchdog
  329446 (+wrapper 329444) FIRST, monitor 397962, librelane 329603,
  openroad 410431. By 22:48 the old-config flow had already started
  `OpenROAD.STAMidPNR-3` (dir 43, 0-byte log, no state_out, ~3 min of
  STA) — **deleted dir 43** (incomplete, wrong-config start; it re-runs
  AFTER the resizer in the new launch). Frontier back to 42 (complete).
  NOTE: the old launcher (run_openeye.sh from 29.10) exited with
  `Zeile 67: GIT_STATE ist nicht gesetzt` — bash had the OLD file's
  read offset and continued reading the NEW (A3-replaced) file after
  the 27.5-h docker exec returned; cosmetic only, nothing executed
  beyond the error (line 67 = `if [ -f "${GIT_STATE}" ]`, before its
  definition in the re-read fragment). PITFALL for the record: never
  replace run_openeye.sh while a launch is in flight.
- [x] **B3** Regenerate config (`python3 asic/scripts/gen_config.py`),
  copy to `asic/runs/openeye_p2_15/design/openeye.json`, check the diff is
  exactly the intended knobs. Run the fixed resolver (A2): expected is
  the step right after RepairAntennas (`OpenROAD.ResizerTimingPostGRT`).
  **Done 2026-10-30 22:51:** diff old running config vs. new = exactly
  `OPENROAD_THREADS: 48`, `RUN_POST_GRT_RESIZER_TIMING: true`,
  `SETUP_VIOLATION_CORNERS: ["*"]` (nothing else). Copied to
  `asic/runs/openeye_p2_15/design/openeye.json`. Resolver (container,
  real run dir + new config) → **`OpenROAD.ResizerTimingPostGRT`** —
  live confirmation of A2/A7 (synthetic test S3 had predicted it).
- [x] **B4** Relaunch the watchdog with the SAME run name p2_15 (this is a
  resume, not a new run): `sg docker -c "cd ~/projekte/OpenEye && setsid
  nohup asic/scripts/watchdog.sh p2_15 >> asic/runs/watchdog_p2_15.log 2>&1 < /dev/null &"`.
  Restart the fixed monitor.
  **Done 2026-10-30 22:52:** watchdog 410875 (log line "resuming at
  step: OpenROAD.ResizerTimingPostGRT"), monitor 411032. flow.log
  separator `=== launch 2026-10-30T22:53:11 resume=OpenROAD.ResizerTimingPostGRT ===`;
  run_openeye.sh resume path (RTL untouched, only gen_config re-run);
  dir `43-openroad-resizertimingpostgrt` created (ordinal 43 reused
  from the deleted STAMidPNR-3 dir, as expected).
- [x] **B5** Verify A1 (threads) within the first minutes. Then monitor
  DetailedRouting: per-iteration DRC violation count must decrease;
  report runtime and final violation count. If DRT stalls on congestion
  hotspots → analyze (do not blindly re-run); levers are in Phase D.
  **Threads part done 2026-10-30 23:02:** flow.log 22:53:12 "OpenROAD
  will use 48 threads"; ps: openroad 410982 `-threads 48`;
  /proc/410982/status: **Threads: 49** (48 workers + main — direct
  multithreading evidence; >100 % CPU shows in the parallel repair
  phase, library load is single-threaded I/O); warning.log flat at
  2612 lines / 21× ORD-0032, all pre-relaunch. DRT monitoring continues
  when step 45 starts.

### Phase C — evaluate the 50 MHz baseline (after p2_15 ends)

- [ ] **C1** Collect per-corner setup/hold WNS/TNS (STAPostPNR), power,
  utilization, wire length, DRC/LVS results. Classify every DRC/LVS error:
  documented PDK gap (bondpad LVS, PSM-0069, sealring) vs. NEW.
- [ ] **C2** Post-route gate-level sim: rerun `asic/sim/gatesim/run_verilator.sh`
  on the FINAL netlist (`final/nl/OpenEye_Asic.nl.v`) — it differs a lot
  from p2_10 (846k repair buffers, hold buffers, 5k diodes). Must PASS
  `start_test_fpga`.
- [ ] **C3** Ask the user to confirm the IO assumption (6.4 ns pad delay,
  external driver ≤ 6 ns clock-to-pad at 50 MHz) against the board/tester.
- [ ] **C4** Write a short results summary in "Current state" + push to
  `utz12`.

### Phase D — next physical iteration p2_16 (proposal; user decides scope)

Metrics at step 37 of p2_15 that motivate this: 3.12 M instances of which
**846k timing-repair buffers (27 %)** from RepairDesignPostGPL (4:35 h),
clock skew ~2.1 ns, 501k hold violators pre-resizer, GRT local overflow
in PE-fabric/SRAM regions, worst 10 ns path = cross-chip wire.

- [ ] **D1 Macro placement near consumers** (`macro_placement.py`): today
  all 33 SRAMs sit in three rows at the core bottom, so every cluster
  routes across the chip. Place each cluster's psum buffers
  (`PSUM_RAM_X[x].PSUM_RAM_Y[y]`, 6 macros each) next to that cluster,
  wght/iact buffers central. Needs a look at where GPL puts the clusters
  (p2_15 placement DEF / ODB) to decide the floorplan.

  **D1 data (verified 2026-10-31 from the p2_15 post-DPL DEF, step 33,
  read-only; units nm, die 10.5×10.5 mm):**
  - All 33 macros in THREE left-aligned rows at the core bottom,
    orientation N: WGHT `RM_2048x64`×4 + `2048x32`×1 at y=0.325 mm,
    x=0.325–3.525 mm; PSUM `4096x16`×16 + `4096x8`×8 at y=0.970 mm,
    x=0.325–8.995 mm; IACT `1024x64`×4 at y=1.610 mm, x=0.325–2.725 mm.
  - PSUM row is in cluster-grid order (RTL `OpenEye_FPGA.v:2989` —
    X=CLUSTER_COLUMNS=2, Y=CLUSTER_ROWS=2, 2 GLB×3 macros per group):
    X0Y0 @0.3–2.3 mm, X0Y1 @2.6–4.5, X1Y0 @4.8–6.8, X1Y1 @7.0–9.0.
  - Placement map (506,010 DFFs + 1.22 M other stdcells, 0.5/0.25 mm
    bin histogram): the PE fabric is ONE continuous dense block
    x≈0.5–10.5 mm, y≈2.0–10.25 mm (uniform density, no resolvable
    2×2 cluster seams at 0.25 mm — the fabric is FLATTENED in the
    ODB; only the macro-wrapper hierarchies survive, so exact
    cluster→region mapping needs netlist-level tracing, Phase D).
  - Hold-violating DFFs (e.g. `_2911245_` @0.48/2.86 mm) sit directly
    above the macro band → consumers are at the fabric bottom edge.
  - D1 floorplan implications: (a) split the single PSUM row so each
    6-macro group sits under its half of the fabric (X index = column),
    ideally one row for Y[0] (fabric bottom) and one for Y[1] (fabric
    top / inside the block as a channel); (b) move WGHT+IACT (global
    buffers) from the bottom-left corner toward center x≈5 mm; (c) the
    fabric itself could then shrink its bottom edge from y≈2.0 to
    y≈1.9, i.e. the macro band stays below the consumers.
- [ ] **D2** `PL_TIMING_DRIVEN: true`; `GPL_CELL_PADDING`/`DPL_CELL_PADDING`
  1–2 sites or a lower `PL_TARGET_DENSITY_PCT` (now 0.65 auto) against the
  local overflow.
- [ ] **D3** Larger macro halos (`FP_MACRO_HORIZONTAL/VERTICAL_HALO`, now
  10 µm) + `GRT_MACRO_EXTENSION` for the SRAM pin hotspots.
- [ ] **D4** CTS / skew analysis: find where the 2.1 ns skew comes from
  (suspect: SRAM clock pins far from the cluster sinks, IO capture regs);
  tune `CTS_SINK_CLUSTERING_SIZE/MAX_DIAMETER`, buffer distance.
- [ ] **D5** Fast corner: stdcell `fast_1p32V_m40C` is paired with SRAM
  liberty `fast_1p32V_m55C` (`CORNERS` in `gen_config.py`) — document or
  align. Optional: add the sg13g2_io liberty for pad timing.
- Goal for D: same 50 MHz closure in far less runtime, then try 66 MHz.

### Phase E — architecture/RTL (only after discussion with the user)

- [ ] **E1** PE scratchpads in DFFs are ~45 % of area. Evaluate the small
  PDK macros (`RM_IHPSG13_1P_64x16/64x64`, `2P_64x22/64x32`, `*_256x*`)
  or latch-based register files for the iact/wght/psum SPads.
- [ ] **E2** Datapath registers without functional need for reset →
  reset-less DFFs (smaller cells, lighter reset tree).
- [ ] **E3** Clock gating (none today; 506k FFs).
- [ ] **E4** Re-pipelining of cross-cluster fabric paths for 100 MHz
  (already identified in the p2_12 analysis).

## Machine / environment

Work happens on **est-srv02** (`est-srv02.med.fh-koeln.de`, AlmaLinux 10,
user `utz`, ~256 GB RAM — the old WSL machine only had 24 GB and could not run
synthesis + gate simulation in parallel).

- Repo: `~/projekte/OpenEye` (branch `dev_asic_sg13g2`, origin = GitHub;
  all local work is 6 commits ahead of `origin/main`, push still blocked).
- iic-osic-tools source: `~/projekte/iic-osic-tools`
- Docker: docker-ce installed (user is in group `docker`).
- All heavy tools run inside the Docker container
  **`iic-osic-tools_asic`** (headless worker):
  - image: `hpretl/iic-osic-tools:latest` (PDK at `/foss/pdks/ihp-sg13g2`,
    tools at `/foss/tools`: yosys 0.70, libreLane 3.1, iverilog,
    verilator 5.052, cocotb 2.1)
  - repo mounted at `/foss/designs/OpenEye`
    (`-v $HOME/projekte:/foss/designs:rw`)
  - env: source `asic/scripts/env.sh` inside the container
    (`PDK_ROOT=/foss/pdks`, `PDK=ihp-sg13g2`, `STD_CELL_LIBRARY=sg13g2_stdcell`,
    `PAD_CELL_LIBRARY=sg13g2_io`)

### Container setup on est-srv02 (DONE 2026-10-23, verified)

1. Pull the image: `docker pull hpretl/iic-osic-tools:latest` (image was
   already loaded on est-srv02).
2. Create the headless worker (est-srv02 uid is 1002; SELinux is enforcing →
   `--security-opt label=disable`). **The image entrypoint is
   `/dockerstartup/scripts/ui_startup.sh` and requires `-s` as its FIRST
   argument** — a bare `sleep infinity` command makes it abort with
   "Unexpected option". Working invocation (same pattern as
   `asic/scripts/container.sh`):
   ```
   docker run -d --name iic-osic-tools_asic \
     --user 1002:1002 --security-opt label=disable \
     -v $HOME/projekte:/foss/designs:rw \
     hpretl/iic-osic-tools:latest -s /bin/bash -c "sleep infinity"
   ```
3. **Install the Python packages that the base image is missing** (they were
   installed in the old machine's container layer, not in the image). Split
   into two pip calls — a single call with `--index-url` in the middle would
   point the *whole* install at the PyTorch index (tensorflow et al. are not
   there):
   ```
   docker exec iic-osic-tools_asic bash -lc \
     "pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu"
   docker exec iic-osic-tools_asic bash -lc \
     "pip install tensorflow==2.21.0 onnx==1.23.2 onnxruntime==1.31.0 cocotb-test==0.3.0"
   ```
   (base image already has: cocotb 2.1.0, cocotbext-ams, numpy 2.5.4,
   torch-geometric 2.8.1)
4. Sanity check inside the container:
   ```
   docker exec iic-osic-tools_asic bash -lc "cd /foss/designs/OpenEye && PYTHONPATH=src asic/scripts/gen.sh"
   docker exec iic-osic-tools_asic bash -lc "cd /foss/designs/OpenEye && asic/scripts/regress.sh"
   docker exec iic-osic-tools_asic bash -lc "cd /foss/designs/OpenEye && asic/scripts/regress_macros.sh"
   ```
   **All three pass on est-srv02** (logs: `asic/runs/regress_generic.log`,
   `asic/runs/regress_macros.log`). Both regressions must stay green before
   any flow run.

est-srv02 host gotchas (learned 2026-10-23):
- `utz` IS in the `docker` group (`/etc/group`), but existing shell sessions
  don't see it until re-login → run docker via `sg docker -c "docker ..."`
  until the session is renewed.
- `run_openeye.sh` / `watchdog.sh` must be launched **on the host** (they
  call `docker exec`; there is no docker inside the container — the
  `docker exec -d … watchdog.sh` pattern only works with docker-in-docker).
  Host launch: `sg docker -c "cd ~/projekte/OpenEye && setsid nohup asic/scripts/watchdog.sh p2_10 > asic/runs/watchdog_p2_10.log 2>&1 < /dev/null &"`
- `asic/sim/gatesim/run_verilator.sh <netlist>` — Verilator gate-sim build
  (cocotb VPI, `--timing` is mandatory for the Python-driven clock).
  **Flag pitfalls (all hit in practice, see script header):**
  - **VPI exposure needs BOTH `/*verilator public_on*/` marks AND
    `--flatten`** (build 5 hit the first half; the second half was
    root-caused 2026-10-25, probes in `asic/sim/gatesim/vpi_probes/`).
    Verilator 5.052 generates a VPI var table only when at least one
    public variable exists — without any, the trivial wrapper is inlined
    into `$root` and cocotb fataled with "Can not find root handle"
    (build 5). The marks on the wrapper's 9 ports
    (`OpenEye_Asic_gatesim_top.v`) are therefore REQUIRED. The marks
    alone are NOT sufficient: without `--flatten` the wrapper survives as
    its own class while `$root` holds a mirror copy of the 9 ports that
    is re-synced onto the wrapper on EVERY clock edge (ico_sequent),
    clobbering cocotb's writes (symptom: outputs read 0x0). With
    `--flatten` everything is one `$root` class and the top module's
    scope view binds its var table directly to `$root` storage, so the
    cocotb root handle (`COCOTB_TOPLEVEL=OpenEye_Asic_gatesim_top`)
    exposes the LIVE port variables — the mirror copies (`*_DOT__*`) are
    synced only once at the first iteration and then form the driven
    path. Verified end-to-end at probe scale in the exact real-build
    corner (flatten + marks + VM_TIMING=1 + Python-driven clock, P15c):
    combinational write propagation without a clock edge AND the FF path
    via posedge/NBA.
    - **File order is functional (build 6 root cause, probes P18c–P18m):**
      with `--flatten`, every inlined cell gets a candidate VPI scope; it
      materializes only if the cell's module is `modPublic`, because only
      public inlined cells keep per-instance hierarchical var names
      (`top.inst.var`) — non-public cells merge their vars into top-level
      `__Vcellout__` members. Verilator 5.052 auto-marks a module public
      iff its definition is parsed **AFTER** the `--top-module` definition
      (command-line file order, then in-file order; 14 probe runs, source
      setter not located in the 5.052 tree). `run_verilator.sh` therefore
      lists ALL cell files (netlist, primitives, stdcell, RM, io) BEFORE
      the wrapper. Reversed order = every one of the ~2.1M instances
      public → 3.19M per-instance scopes / ~360 GB codegen (build 6,
      killed). Correct order → 2 scopes (TOP + wrapper), ~9-20 var-table
      entries (the wrapper's public_on ports + its local nets), verified
      on the full 84-module stdcell file (P18m + sanity compile).
  - NO `--public`/`--public-flat-rw`: exposes every internal signal of
    every module and exploded codegen to **653 GB of C++ / 700 GB written**
    for this flat netlist (build of 2026-10-23/24, killed at the PCH
    compile stage).
  - Build with `-cc --exe` + explicit `make -f Vtop.mk`, NOT `--binary`:
    cocotb's `verilator.cpp` provides `main`; `--binary` generates a
    second one → `duplicate symbol: main` at link.
  - cocotb 2.1 runtime env: `COCOTB_TEST_MODULES`/`COCOTB_TOPLEVEL`/
    `COCOTB_TEST_FILTER`, `GPI_USERS=<libpython>;<pygpi entry point>`
    (both from `python3 -m cocotb_tools.config`), `PYGPI_PYTHON_BIN`,
    `COCOTB_TRUST_INERTIAL_WRITES=1`. The old `MODULE`/`TOPLEVEL`/
    `TESTCASE` names are NOT honored by the cocotb 2.1 Verilator runtime.
  - `make` parallelism low (`-j 4`, `MAKE_JOBS`): the generated Vtop
    translation units are huge; high `-j` risks OOM.
   - The script's codegen-size guard must use GNU `du -sm`, not BSD
     `du -sg`: the container has GNU coreutils, and with `set -e` the
     failed `du -sg` command substitution ABORTS THE WHOLE SCRIPT. This
     actually happened on 2026-10-25: codegen finished fine (2516 MB in
     2222 files), but `make` never ran; recovered with
     `asic/runs/finish_gatesim_make.sh` (make + cocotb run continuation).
  `sg13g2_udp.v` is NOT compiled (Verilator has no UDP support): the PDK
  standard-cell models instantiate 17 `ihp_*` UDP primitives internally,
  replaced by the behavioral stand-ins in
  `asic/sim/gatesim/sg13g2_primitives_vlt.v` (same port lists, semantics
  transcribed from the UDP tables). `sg13g2_IOPadAnalog` has a circular
  `assign` that Verilator rejects, but it is unreachable from
  `OpenEye_Asic_gatesim_top` with `--top-module`, so the full
  `sg13g2_io.v` file is fine. The PDK stdcell models instantiate those
  primitives **without instance names** (legal per LRM, iverilog accepts
  it, but Verilator only allows unnamed instances for gate-level
  primitives): `run_verilator.sh` therefore compiles a generated copy of
  `sg13g2_stdcell.v` in which the 24 `ihp_* (` lines are rewritten to
  `ihp_* u_prim_N (` (the iverilog Makefile path still uses the PDK
  original and is unaffected). The same generated copy additionally has
  its 84 specify blocks stripped and 57 `assign delayed_X = X;` aliases
  added by `asic/scripts/pdk_specify_strip.py`: Verilator is 2-state and
  ignores specify contents, so the `delayed_*` wires driven only by the
  timing blocks of the 16 FF/latch cells would fold to 0 and break every
  sequential cell. All PDK specify delays are 0.0, hence the aliasing is
  exact. The io models are compiled as-is (their specify blocks carry no
  `delayed_*` data; Verilator just ignores them).

## Current state (as of 2026-10-30 19:10)

### Session 2026-11-01 (plan v2, Autonomie-Modus F→G→P→H)

**10:55 — F0/F1 done.** Diffs of Claude's 3 commits read (F0, see plan);
stale p2_15 monitor (411032) killed, no flow running, p2_15 run dir kept
as evidence (stopped at step 43, no `state_out.json` there — as
documented). F1: `regress.sh` PASS (10:57, TESTS=1 PASS=1, log
`asic/runs/regress_generic_latchrf.log`); `regress_macros.sh` covered by
Claude's 10:52 PASS (RTL unchanged since). Machine: 1.3 T disk free,
243 G RAM available, container `iic-osic-tools_asic` up 6 days.

**11:0x — F2 (part 1) done: stop-step launch chain.**
`run_openeye.sh <run> [resume-step] [stop-step]` → `librelane … -F … -T …`
(third arg; `librelane --help` confirms `-T, --to`); success for a
stopped run = librelane exit 0 AND the stop step's latest dir (slug via
librelane's `slugify`, highest ordinal) carries `state_out.json` (no
`final/` check); success marker line kept verbatim (suffix in parens) so
the watchdog exits 0 without restart. `watchdog.sh` passes the
`STOP_STEP` env through as third positional (always passes all three
args — positional shift on fresh+stop would otherwise corrupt the args).
Tests (stub `docker`, synthetic run dirs, all PASS, artifacts deleted):
resolver on stopped frontier (step 12 complete) → `OpenROAD.Floorplan`;
resume+stop with state_out → marker/exit 0; without → `[ERROR]`/exit 1;
watchdog `STOP_STEP=OpenROAD.STAPrePNR` → "flow completed", exit 0, no
restart; DRY_RUN fresh+stop prints `-T 'OpenROAD.STAPrePNR'`; DRY_RUN
plain unchanged. Regenerated tracked files (OpenEye_Asic.v, pad_order.
json, openeye.json) byte-identical (git status clean).

**11:05 — p2_16 launched** (fresh run, stop step `OpenROAD.STAPrePNR`,
watchdog with `STOP_STEP` env, monitor with G1 gates). F3 watcher armed
on step-6 `state_out.json` → detached `run_verilator.sh` on
`06-yosys-synthesis/OpenEye_Asic.nl.v`, log `asic/runs/gatesim_p2_16.log`.

**11:30 status (30-min cadence):** steps 1–5 DONE — Verilator.Lint clean
(0× LATCH with `--Werror-LATCH`, 0× errors; `sg13g2_lgcp_1`/
`sg13g2_dlhq_1` resolve via LibreLane auto-blackbox in `tmp/*.bb.v`,
84 stdcell modules incl. both latch cells); defines
`OPENEYE_LATCH_RF`+`OPENEYE_SG13G2_CELLS` confirmed in the lint command
line. Step 6 Yosys.Synthesis running (flow.log ~5 MB). Monitor 544861
(gates enabled), watchdog 0 restarts, machine idle otherwise.

**Monitor G1 gates implemented + validated** (commit 49440b9):
`monitor_flow.sh` evaluates each completed key step once (18 placed=41,
22 no PDN-0235, 27 overflow ≤ 0.1, 34 sinks report, 37 nom setup ≥ 0 /
hold ≥ −0.5, 38 no GRT-0115/0116 + total overflow ≤ 0.01 % of demand +
extra-iteration/EST-0026 report, 44 setup within 1 ns of pre-GRT /
hold ≥ −0.5 / RSZ-2021 report); GATE line printed only on change.
Read-only dry run against p2_15 reproduces all documented numbers
(placed=33, overflow=1039.35, sinks=506219, setup=0.0, hold=−0.440,
GRT-0115=1, extra_iter=99).

**P1 done** (commit 0901113): `CONFIG_OVERLAY=<json>` deep-merge into
the staged `design/openeye.json` (every launch, copy recorded as
`design/config_overlay.json`); tested fresh/nested/resume/missing-file
with DRY_RUN + stub docker. Overlay files committed:
`openeye_overlay_26thr.json` (G1 resume of p2_16),
`openeye_overlay_66mhz_26thr.json` (p2_16b: 15 ns, IO 43, 26 threads).

**Pre-flight verification (11:4x): iact instance names de-risked.** All
41 placement names verified 1:1 against the p2_15 netlist hierarchy
(`BUFFER_A[i].iact_layer_buffer.g_specific.impl.g_cfg1024x64.u_mac`
×4, `wght_buffer…g_cfg2048x288.u_mac0..4`, `PSUM_RAM_X[x].PSUM_RAM_Y[y].
PSUM_RAM_GLB[g].psum_buffer…g_cfg4096x40.u_mac0..2` ×24, iact path
`IACT_CONVERTER_X[x].IACT_CONVERTER_Y[y].iact_stream_constructor.BUFFER[0].
iact_buffer_SP` ×4 — only the `g_cfg1024x24.u_mac0/1` suffix is new and
matches the wrapper structure). DEF names are ODB-escaped (`\[0\]`),
handled.

**macro_positions.py --check** (commit 41bfc3f): G1 step-18 readback
compares DEF macros against the placement JSON (position 0.5 µm tol,
orientation, cell, both-direction completeness, MX/R0 counts). Validated
on p2_15's GeneratePDN DEF: round-trip ALL MATCH (33, all N); 3 tamper
types detected, exit 1. G1 command: `python3 asic/scripts/
macro_positions.py <18-dir>/OpenEye_Asic.def --check <placement.json>`
(placement.json from `compute_placement()` — the script's `__main__`
appends a `#` comment line that would break json.load).

**F3 readiness:** run_verilator.sh carries the new RM_1024x16 model
(line 147, PDK file exists); PDK stdcell models include sg13g2_dlhq_1/
lgcp_1. Build will take hours (p2_10 ref: 8.4 h for the 55 mm² netlist;
p2_16 netlist is smaller) — the flow does not wait for it, a FAIL is a
stop criterion.

**12:00 status (30-min cadence):** step 6 Yosys.Synthesis running
(yosys 544104, 163 % CPU, CPU time 9:56 at 11:5x, pass ~133 OPT_EXPR;
p2_11 ref 58:41 → ETA ~12:10–12:30). Steps 1–5 clean (see 11:30 line).
Watchdog 0 restarts, monitor (gates) silent so far, disk 1.3 T, RAM
243 G free. F3 watcher armed on step-6 state_out.json.

**12:25 — F2 DONE: p2_16 steps 1–12 all PASS** (run stopped at
`OpenROAD.STAPrePNR` per contract: `state_out.json` in the 12 dir,
watchdog exited 0 at 11:49, flow.log "Flow complete").

- Step 6 Yosys.Synthesis finished 11:41:28 (00:38:07; p2_11 ref 58:41).
- `06-yosys-synthesis/reports/stat.rpt`: `sg13g2_dlhq_1` = **393,216**
  (expected exactly), `sg13g2_lgcp_1` = **19,968** (expected exactly),
  `sg13g2_dfrbpq_1` = **51,883**, RM macros = **41** (1024x16×4,
  1024x64×4, 1024x8×4, 2048x32×1, 2048x64×4, 4096x16×16, 4096x8×8),
  chip area **32,783,956 µm² = 32.78 mm²** (was 55.29). PDK cell
  identities: dlhq_1 = D-latch ("IQ"), lgcp_1 = ICG ("CLK*int_GATE"),
  dfrbpq_1 = enable-DFF (48.99 µm²).
- Latch-RF geometry resolved exactly by census — 5 param sets
  (Words×DataWidth)×count → latch / FF / ICG bits: (8,24)×48 → 294,912 /
  2,304 / 12,288; (5,20)×96 → 61,440 / 3,840 / 3,072; (6,8)×48 → 24,576 /
  768 / 3,072; (4,12)×48 → 9,216 / 1,152 / 768; (4,4)×48 → 3,072 / 384 /
  768. Total = **393,216 latch + 8,448 FF + 19,968 ICG** (ICG total
  matches the plan formula 48×(256+64+16+16)+48×2×32).
- Steps 7–11: unmapped 0; synth checks 0 errors, 0 inferred latches
  (instance_count 1,161,732, area 32,783,956 µm², lint_warning 637);
  CheckMacroInstances **41/41**.
- Step 12 STAPrePNR done 11:48:51, no errors (error.log 0, warning.log
  3). Pre-PNR absolute setup WNS is a flow baseline artifact (no
  placement → routing estimates); the meaningful gates are hold WNS and
  violator counts:

  | corner | p2_16 setup / hold WNS (ns) | p2_15 setup / hold WNS (ns) | setup violators p2_16 / p2_15 |
  |---|---|---|---|
  | fast | −458.56 / **−0.514** | −455.91 / **−0.514** | 49,096 / 402,000 |
  | slow | −1034.48 / **−0.966** | −1028.64 / **−0.966** | 58,789 / 430,000 |
  | typ | −706.62 / **−0.665** | −702.60 / **−0.665** | 56,005 / 403,000 |

  Hold WNS identical on all three corners (same clock/IO constraints)
  → parity with p2_15 confirmed, gate met. Power 0.282 W vs 1.519 W.

- **dfrbpq census — deviation RESOLVED, benign (NOT a stop criterion).**
  Method: census of the pre-abc hierarchical dump (`asic/runs/
  diag_pre_abc_p2_{16,15}.v`, scripts `diag_ys_p2_1{6,5}.ys`, parser now
  `asic/scripts/dff_census.py` — commit a205b6b). Identical pipeline for
  both (read → hierarchy → proc → memory → memory_map; no
  opt/techmap/abc). Totals: **p2_16 54,049 FF / 393,216 latch / 19,968
  ICG / 41 RM** (latch/ICG/RM exact vs stat.rpt → parser validated);
  **p2_15 541,441 FF / 0 / 0 / 33 RM** (RTL of f0e14fd, no
  `OPENEYE_LATCH_RF`). Δ = −487,392 = −397,440 (SPad RAM FFs → latch
  RF: 296,064+63,360+24,960+9,792+3,264) − 98,496 (iact_sc RAM FFs → 8
  RM macros) + 8,448 (RF wdata_q/memout) + 96 (g_cfg1024x24 wrapper) —
  **100 % accounted**. The "other" logic (**43,585 FF**) is
  bit-identical per module in both runs: PE 15,888, delay_cluster 7,744,
  data_pipeline_wght 6,048, OpenEye_FPGA 4,637, data_pipeline_iact
  4,224, psum_pipeline 1,711, dma_storage 896, iact_stream_constructor
  754×2, OpenEye_Parallel 593, router_configurator 145,
  quantization_unit 112, adder_tree 48, RST_SYNC 12,
  ring_buffer_pipelined 11, PE_cluster 8 (ring RAM_DP_generic 1,056 and
  the sg13g2_ram_sp wrapper FFs 320+288+256(+96 new) counted in their
  own buckets). No duplicated/missing logic. **The plan's 15–30 k band
  was based on a wrong assumption** (p2_15 "other" taken as 14,699;
  actually 43,585) — 51,883 is the correct post-opt number. Post-opt:
  51,883 (p2_16, −2,166 removed by opt) vs 506,219 (p2_15, −35,222);
  context-independent FF types are identical across runs (DFF_PN0 16,808,
  DFFE_PN1P 483, DFFE_PN 32, DFF_PN1 52); the remaining enabled variants
  are context-dependent ABC mappings; p2_15's final netlist carries 0
  lgcp (checked in its stat.rpt), p2_16's 19,968 all from the latch RF.
- **Launch deviation (recorded):** steps 1–12 ran with the base config
  `OPENROAD_THREADS=48`, not the planned 26 — the effective launch
  (10:59:53, HEAD 00a6698) predates P1 (0901113); a second launch
  attempt ~11:05:30 died in staging with a bash syntax error in an
  uncommitted P1-WIP overlay (`run_openeye.sh: Zeile 121 … »)«`), whose
  "failed with step/checker errors" line in `watchdog_p2_16.log` is a
  false alarm — the flow was never affected (watchdog of the first
  launch kept running it). Both committed scripts pass `bash -n`; no
  artifact impact (netlist/STA are thread-independent). The 26-thread
  overlay applies from the G1 resume on.
- F3 gate-sim auto-launched 11:41:28 by the step-6 watcher (Verilator
  build, −j 32; running ~50 min as of 12:29, log `asic/runs/
  gatesim_p2_16.log`).

**12:35 — Phase F complete; G1 + P2 launched.**
- F2 doc + tick + census tool committed (a205b6b `asic/scripts/
  dff_census.py`, 555ba3c QWEN.md) and **pushed to utz12**
  (476c187..555ba3c) — phase-F push duty fulfilled.
- **G1 resume** of p2_16 (12:3x): resolver auto-picked
  `OpenROAD.Floorplan`, `STOP_STEP=OpenROAD.ResizerTimingPostGRT`,
  26-thread overlay applied (staged config verified: 26/26; copy
  recorded as `design/config_overlay.json`). openroad 64907 runs with
  `-threads 26`, step 13 Floorplan in flight. The launcher's "RTL state
  differs" warning is expected: `git diff 00a6698..555ba3c -- hdl
  asic/rtl asic/sim` = only the two new overlay JSONs, no RTL.
- **P2** launched: p2_16b fresh run at 66 MHz (staged config verified:
  period 15 / IO_DELAY 43 / 26 threads), no stop step (runs to the end),
  own watchdog + monitor (G1 gates belong to p2_16's monitor 544861,
  still running).
- F3 gate-sim still building (Verilator ~55 min at 12:35, −j 32).
  Machine: 231 G RAM free (headroom for two GRT peaks, old openroad
  ~55 G each), 1.3 T disk. Temp census tree `asic/runs/diag_rtl_p2_15/`
  removed (dumps + ys scripts kept, gitignored).

**18:1x — F3 PASS; G1-27 GATE-FAIL → STOP per plan (both runs killed).**

- **F3 gate-sim: PASS** (stop criterion cleared, finished 15:02 —
  before the G1-27 gate failed): `start_test_fpga` PASS,
  TESTS=1 PASS=1 FAIL=0 (10210 ns, 4.72 s sim; build 11:41 → 15:02,
  ~3.4 h vs p2_10 ref 8.4 h for the bigger netlist; log `asic/runs/
  gatesim_p2_16.log`, results.xml timestamp 14:02:27 UTC). Latch RF +
  all 41 RM configs functionally correct at gate level.
- **G1 gates (p2_16, monitor 544861):** 14:14 `G1-18:placed=41 OK`;
  16:04 `G1-22:PDN OK` (no PDN-0235); **17:29 `G1-27:overflow=28.5388
  GATE-FAIL(exp <=0.1)`**. Flow had continued (steps 28–30 done, 31
  in flight) — the monitor flags, the flow does not self-stop.
- **G1-27 analysis** (step-27 logs, both runs):
  - p2_16: `global_placement -density 0.421845 -routability_driven`,
    **single iteration**, `[GPL-0041] Total routing overflow: 28.5388`,
    `[GPL-1012] Total routability artificial inflation: 0.00
    (+0.00%)`.
  - p2_15 (the run the v2 floorplan was built to fix): overflow rose
    785.09 → 834.10 → 924.27 → **1039.35** over multiple iterations →
    downstream GRT-0115, 99 extra iterations, 14 h GRT.
  - **p2_16b (66 MHz, same geometry, different timing): identical
    28.5388 / 0.00 %** — the overflow is a reproducible property of the
    floorplan v2 geometry, not a run artifact.
  - Interpretation: `Total routing overflow` = Σ max(0,
    wireDensity−1) in tracks over the whole routing grid. 28.54 is
    36× below p2_15's 1039.35, with zero artificial inflation and no
    extra iterations — i.e. OpenROAD itself did not consider
    congestion a global problem; realistically a handful of slightly
    over-capacity blocks (macro/IO-ring neighborhood) that GRT's
    normal rip-up&reroute typically absorbs. But that is a
    prediction, not a gate pass — the gate says ≤ 0.1 and 28.54 > 0.1.
- **Stop action taken 18:1x (plan: "Any gate failing → stop
  criterion"; P2 rule: "if p2_16 fails a G1 gate, kill p2_16b too"):**
  killed watchdogs 560669 (p2_16) + 561027 (p2_16b), container
  librelane 64890/64985 + openroad 76249/76279, monitors 544861/
  561028. Verified: host + container clean. Run dirs kept untouched
  (p2_16 completed through step 30, step 31 in flight at kill →
  resume would pick step 31; p2_16b through step 29, step 30 in
  flight → resume would pick step 30). RAM 230 G free, disk 1.2 T
  free at stop.
- **Proposal (decision needed):**
  - **(a) Accept deviation, resume p2_16 at step 31 (my
    recommendation).** The gate's purpose was to catch p2_15's GRT
    pathology (1039 overflow → GRT-0115 → 99 extra iterations → 14
    h). 28.54 / 0.00 % inflation / single iteration / p2_16b
    corroboration is a different regime; the real congestion
    checkpoint — G1 step 38 (no GRT-0115/0116, overflow ≤ 0.01 % of
    demand) — is stricter and still ahead. Document G1-27 as "failed
    numerically, accepted with deviation". Cost ≈ 0 (Floorplan→GPL
    not re-run; next is step 31 → CTS).
  - **(b) Tighten, re-run Floorplan→GPL:** raise target density
    (0.4218 → ~0.45) and/or macro halos/padding or GRT layer
    adjustments, re-run steps 13→27 (~5 h, today 12:32→17:29), accept
    only if new overflow ≤ 0.1. Cost ~5–6 h plus the risk that a
    higher density worsens CTS/DRT timing.
  - **(c) Amend the gate formally:** keep ≤ 0.1 as warning, define
    "pass" as ≤ 30 with 0.00 % inflation (i.e. codify option a with a
    tighter monitor rule), then resume.
- **Time estimates** (from 18:1x, scenario a/c). p2_15 per-step
  runtimes as upper-bound reference (its 506 k-FF design is far
  heavier than p2_16's 52 k-FF/393 k-latch design): 31
  RepairDesignPostGPL 4h35, 33 DetailedPlacement 43m, 34 CTS 4h13
  (506 k sinks → p2_16 ~20 k sinks: est. 0.5–2 h), 35 17m, 36
  ResizerTimingPostCTS 10h40 (est. 1–4 h for p2_16), 37 18m, **38/39
  GRT 14h42 (pathological, 99 extra iterations; p2_16 target "hours
  not 14 h": est. 2–6 h)**, 40 CheckAntennas 49m, 42 RepairAntennas
  11h59 (est. 1–5 h). → **G1 stop step 44: est. ~10–22 h → 11/2 noon
  to 11/3 morning.** G2 (run-to-end: DRT → fill → RCX → STAPostPNR →
  GDS → DRC → LVS) adds ~1–2 days (DRT convergence several hours to a
  day at 50 MHz; 66 MHz harder) → **final metrics + gates 11/3–11/4.**
  p2_16b (66 MHz) in parallel from step 30; same geometry, its GRT
  outcome is the open question (tighter period). Scenario b adds
  ~5–6 h before any G1 progress.

**18:2x — decision (a): deviation accepted, both runs resumed.**
User chose option (a). G1-27 is documented as "failed numerically,
accepted with deviation" (28.5388 vs ≤ 0.1; 0.00 % inflation, single
iteration, 36× better than p2_15, p2_16b-corroborated geometry
property). The monitor (new instance, same rules) will re-print the
`G1-27 … GATE-FAIL` line on its first step-27 evaluation — that line
is the accepted deviation, not a new failure. Resumes:
- p2_16: watchdog → resolver picked `OpenROAD.RepairDesignPostGPL`
  (step 31), `STOP_STEP=OpenROAD.ResizerTimingPostGRT` unchanged,
  26-thread overlay re-applied (verified: openroad 76400
  `-threads 26`).
- p2_16b: resolver picked `OpenROAD.STAMidPNR` (step 30), no stop
  step (runs to the end), 66 MHz overlay re-applied (openroad 76398
  `-threads 26`).
- Monitors: 600572 (p2_16, G1 gates), 600558 (p2_16b).

**06:00 (11/2) status (cadence resumed):** both runs alive, in CTS.
- p2_16: resumed 18:21 → 31 (orphan, killed attempt) / 32
  RepairDesignPostGPL 18:21→19:34 (1 h 13) / 33 instant / 34
  DetailedPlacement →20:06 (32 min) / **35 CTS running since 20:06**
  (9.5 h; openroad 76576, ~100 % CPU, log 46 MB and growing ~10 MB/min
  in the "Balancing latency for clock clk_pad" phase, 865k+
  "matched with" sink lines).
- p2_16b: 30 (orphan) / 31 STAMidPNR →18:38 / 32 RepairDesignPostGPL
  →19:53 (1 h 15) / 33 / 34 DetailedPlacement →20:25 / **35 CTS
  running since 20:25** (9.7 h; log 243 MB, +4.6 MB/min, already in
  delaybuf matching on clk_pad — further along than p2_16).
- CTS pace context: p2_15's CTS (506k DFF sinks, 4 h 13 min total)
  contains **10.3M "matched with" lines** — the spam is normal flow
  logging, not pathology; both current runs are actively iterating.
  Note: p2_16's CTS sink count is ~465k, NOT ≪ p2_15 as the G1 plan
  expected — the 393,216 latch GATE pins + 19,968 ICGs + 51,883 DFFs
  dominate (same misestimation class as the dfrbpq band).
- Gates: no new GATE lines since 18:21 (18 OK, 22 OK, 27 accepted
  deviation). Next gate events: CTS sinks (p2_16 step 35 done),
  STAMidPNR-2, GRT, step 44. RAM: 190 G avail (two CTS + caches),
  disk 1.2 T free. Watchdog restarts: 0.
- ETA unchanged-ish: CTS likely done 08:00–10:00; then
  ResizerTimingPostCTS (p2_15 ref 10 h 40, p2_16 est. 1–5 h), GRT
  (est. 2–6 h), antennas, → **G1 stop step 44: 11/2 evening → 11/3
  morning.**

**10:2x (11/2) status: CTS done (both), now in ResizerTimingPostCTS.**
- **CTS finished: p2_16 07:05 (11 h), p2_16b 06:39 (10.3 h)** — slower
  than p2_15's 4 h 13 min despite fewer sinks (latency balancing on
  the clk_pad tree over the PE array; the sink-match log spam is the
  same as p2_15's 10.3M lines).
- **CTS sink counts (p2_16, corrected monitor rule):** max per-clock
  = **51,883** (clk_pad — exactly the DFF count), plus 20,009
  (= 19,968 ICG + 41 macros) and 3×24 (RM A_CLK trees). **Total ~72k
  ≪ p2_15's 506,219 → G1-34 gate intent satisfied.** The 393,216
  latch GATE pins are NOT CTS sinks in this flow (corrects my 06:00
  ~465k fear). Monitor bug fixed: rule took the LAST per-clock line
  (24 — the small macro trees process after clk_pad in p2_16) instead
  of the max; `monitor_flow.sh` now reports the max, monitors
  restarted (790792 p2_16, 790793 p2_16b; stateless, re-derives from
  step dirs), 10:25 GATE line confirms `G1-34:sinks=51883`.
- **Step 36 STAMidPNR-1:** p2_16 20 min (p2_15 ref 17 min).
- **Step 37 ResizerTimingPostCTS running:** p2_16 since 07:25:41,
  p2_16b since 06:59:32; both actively repairing (logs ~1 MB, growing):
  current setup WNS **−2.888 ns (p2_16)** / **−2.892 ns (p2_16b)**;
  violation scale ~8.7k (p2_16) vs ~50k (p2_16b) table rows — the 66
  MHz run works ~6× harder, as expected. p2_15 ref for this step:
  10 h 40 min; p2_16 est. 1–5 h.
- Machine: 195 G RAM avail, 1.2 T disk, 0 watchdog restarts. No new
  gate events beyond the accepted G1-27.

**08:5x (11/3) status:**
- **p2_16 G1-Run beendet 18:39:11 (11/2)** (watchdog exit 0 am Stop-
  Step 43 ResizerTimingPostGRT, 5 h 10 m): 218 Setup- + 190 Hold-
  Buffer eingesetzt; 351,257 Setup- + 22,217 Hold-Endpoints
  gefunden; RSZ-0062 "Unable to repair all setup violations",
  RSZ-0064 "Unable to repair all hold checks within margin".
- **G1-38 GRT PASS (p2_16):** overflow 0/15,057,112 (0.00 %), keine
  GRT-0115/0116, extra_iter=1, **41 min 46 s** (p2_15-Ref: 14 h 42 m,
  99 Extra-Iterationen, overflow 1039.35) — die akzeptierte G1-27-
  Abweichung (28.54 GPL-Overflow) war harmlos. p2_16b GRT auch OK:
  0/15,058,918, extra_iter=1, 41 min 55 s.
- **G1-37 hold FAIL −2.0365 (pre-GRT) — Ursache eingegrenzt:**
  p2_16 STAMidPNR-2 violator_list: **583 hold reg-reg, 0 setup**;
  alle FF-Q → RM-Makro-Eingänge (wght_buffer 212, BUFFER_A[0..3] 292,
  IACT_CONVERTER_X 32, …), Range −2.00…−1.00 ns. p2_16b hat pre-GRT
  den **bitidentischen** hold WNS (−2.036539439175264) bei anderer
  Periode (15 vs 20 ns) → periode-unabhängig, strukturell:
  Makro-A_CLK-Bäume vs clk_pad (Capture-Clock ~2 ns früh an den
  Makro-Inputs). p2_16b post-GRT-Resizer: hold weiter
  verschlechtert (−3.2791), weil Setup-Buffer dort Delay added.
- **p2_16b 66-MHz-Verdict (STAMidPNR-3, post-GRT-Resizer):**
  setup WNS **−7.8079** (TNS −237.83), hold **−3.2791** (nom_typ)
  → fmax ≈ 44 MHz; 66 MHz ohne Designänderung nicht erreichbar.
  Hold-Verletzer: 543 Zeilen, systematisch reg → Makro
  A_DIN/A_ADDR (−3.28…−2.85). p2_16b DRT läuft seit 21:41 11/2
  (11 h 15 m, 26 Threads ~100 %, no convergence message yet;
  log 42 min flat = bekanntes Buffering-Pattern, CPU aktiv).
- **G1-44 in flight:** p2_16 Post-GRT-Resizer-STA gestartet (~08:50,
  STOP_STEP=STAMidPNR-3, ~20 min). Monitor-Regel gefixt: G1-44 liest
  jetzt wns.max.rpt/wns.min.rpt des neuesten openroad-stamidpnr-3-
  Verzeichnisses (Resizer-Logs enthalten KEIN WNS, nur RSZ-Nachrichten);
  Monitore neu gestartet (1080677 p2_16, 1080678 p2_16b).
  **G1-44 FAIL ist die Favoritin** (setup: GRT legte Delay auf
  0.0-Slack-Pfade + Resizer gab auf; hold: strukturelle −2.0 bleibt)
  → dann Stop-Kriterium laut Plan: Analyse + Vorschlag
  (Hold-Fix per A_CLK/clk_pad-Latenz-Matching statt 22k Data-Buffer),
  Entscheidung Nutzer.
- Machine: 193 G RAM avail, 1.2 T disk, 0 watchdog restarts.
  p2_16b weiter/stoppen: Entscheidung des Nutzers pending.

**Status:** **G1-44 pending (08:5x 11/3).** p2_16: G1-Run done
(Stop 43, 18:39 11/2), GRT PASS (0 overflow, 41 min 46 s), STA
(STAMidPNR-3) läuft → G1-44-Verdict ~09:10; FAIL die Favoritin
(hold −2.0 strukturell, reg→Makro-A_DIN). p2_16b: 66 MHz
beantwortet (fmax ≈ 44 MHz: −7.8079/−3.2791), DRT 11 h —
weiter/stoppen pending Nutzer. F3 gate-sim PASS (15:02 11/1).
Nächster Meilenstein: G1-44-Verdict → G2 (p2_16 run-to-end) erst
nach Entscheidung.

### p2_15 status + Phase A progress (2026-10-30, Autonomie-Modus)

**19:10 status:** **Phase A complete (A1–A7), pushed to utz12 (f840180).**
Step 42 `OpenROAD.RepairAntennas` still running (since 10:45, openroad
PID 370311, CPU time ~28.7k s ≈ 99.7 % of elapsed — working, log
buffered). Last console output (15:11): internal GRT re-route in the
repair iteration (NDR off on net781646) → 6152 antenna violations →
9200 jumpers inserted. Watchdog 329446 (0 restarts), new monitor
397962 (no STALL). Config diff for Phase B verified: new openeye.json
vs. the running copy = exactly `OPENROAD_THREADS`,
`RUN_POST_GRT_RESIZER_TIMING`, `SETUP_VIOLATION_CORNERS` (the three
intended knobs). B2 kill chain mapped: host watchdog 329446 →
run_openeye.sh 329490 → docker exec 329555; container bash 329587 →
librelane 329603 → openroad 370311. Background watcher armed on
`42-openroad-repairantennas/state_out.json` → Phase B starts there.

Status lines (30-min cadence):
- **19:59:** step 42 still running (openroad 370311, etimes 33,265 s,
  CPU 33,162 s = 99.7 %). New console output since 15:11: repair
  iteration converging — "Found 1053 antenna violations" → "Inserted
  2035 diodes" (was 12,970 at start, 2,714 at 18:09, 6,152 after the
  internal GRT re-route). Monitor: alive, no STALL. Watchdog: 0
  restarts. Disk 1.3 T free. No `state_out.json` yet.
- **20:25:** step 42 still running (CPU 34,674 s, +1,512 s in 26 min ≈
  97 % — working, no new console output since 19:59; post-diode phase,
  likely internal GRT/STA). No dir 43 yet, monitor clean, watchdog 0
  restarts.
- **20:50:** repair converging: another internal GRT re-route
  (920 jumpers / 551 nets) → "Found 420 antenna violations" → "Inserted
  590 diodes". Openroad 370311: etimes 36,292 s, CPU 36,180 s ≈ 99.5 %.
  flow.log 508.1 MB. No state_out.json, no dir 43 yet.
- **21:19:** final phase of RepairAntennas = verbose net matching
  (".* matched with _NNNNNN_", 7.8 M lines so far, net index ~3.23 M and
  rising at a stable ~2,200 nets/s). I/O-bound: openroad mostly blocked
  in pipe_write, ~2–7 % CPU, flow.log grows ~300 KB/s (633 MB and
  counting — disk fine, 1.3 T free). Healthy, not a stall: net index,
  CPU time and log all advancing. Watcher on state_out.json armed.
- **21:52:** flood identified — it is NOT a stall and NOT the repair
  loop. Step-42 log shows `repair_antennas` is DONE (final iteration:
  "Found 420 antenna violations" → "Inserted 590 diodes"); the
  3.27-M-instance flood is `write_views` → `set_global_connections`
  (io.tcl:488) printing one line per instance per `.*` entry of
  PDN_MACRO_CONNECTIONS (2 entries: vdd/vss + iovdd/iovss). GRT (step
  39) printed the same 6.2 M lines in its write_views — known behavior,
  ~6.5 M lines per openroad step. Step-42 log: 4.94 M of ~6.53 M lines
  (≈76 %), growing 69–300 KB/s. Remaining: flood + area report +
  write_db (2.5 GB odb) + DEF → state_out.json. ETA ~30–60 min. No
  action; watcher will fire Phase B.
- **22:20:** step 42 structure clarified: it is a two-substep step —
  `1-openroad-diodeinsertion` (the 11-h repair, done ~22:00, odb+DEF
  written) and `2-openroad-checkantennas` (running, openroad 408120,
  `antenna_check.tcl` = `check_antennas -verbose`, report-only, cannot
  fail the step). Reference runtime: step 40 (same check) took 00:49:21
  → ETA ~22:50, then 42's state_out.json. **Phase B kill window:**
  state_out.json appears, and within seconds–minutes the CURRENT
  (old-config) flow starts `OpenROAD.STAMidPNR-3` and creates dir 43.
  Playbook: (1) `kill 329446` (watchdog) FIRST — with the OLD design
  config its resolver would pick STAMidPNR-3 (resizer gated off) and
  auto-restart the flow; (2) `kill 397962` (monitor); (3) `kill 329603`
  (librelane) + the current openroad (pgrep in container); (4) if dir
  43 exists it is an INCOMPLETE STAMidPNR-3 started under the old
  config minutes ago → **delete it** (its re-run will happen after the
  resizer in the new launch; deleting an incomplete dir is not
  discarding a finished run); (5) B3: gen_config.py + copy to
  design/openeye.json, diff vs. running copy = exactly the 3 knobs,
  resolver → expect `OpenROAD.ResizerTimingPostGRT`; (6) B4 relaunch
  watchdog + monitor; (7) B5 verify threads.

### Phase B EXECUTED (2026-10-30 22:45–22:55) — p2_15 relaunched with A1–A7

- **22:45** step 42 `state_out.json` → watcher fired. Kill sequence in
  60 s: watchdog 329446 first, monitor 397962, librelane 329603,
  openroad 410431 (already STAMidPNR-3 — the old flow had moved on at
  22:45). Dir 43 (incomplete STAMidPNR-3, ~3 min of STA, wrong config)
  **deleted**; frontier back to 42 (complete).
- **22:51** B3: diff old running config vs. new = exactly the 3 knobs
  (threads 48, RUN_POST_GRT_RESIZER_TIMING, SETUP_VIOLATION_CORNERS);
  copied to `asic/runs/openeye_p2_15/design/openeye.json`; resolver
  (real run dir, container) → **`OpenROAD.ResizerTimingPostGRT`**
  (live confirmation of A2+A7).
- **22:53** relaunch (watchdog 410875, monitor 411032): flow.log
  separator `resume=OpenROAD.ResizerTimingPostGRT`; run_openeye.sh
  resume path (RTL untouched, only gen_config re-run); dir 43 =
  resizer (ordinal reused from the deleted STAMidPNR-3 dir).
- **22:54+** B5: flow.log `OpenROAD will use 48 threads` (22:53:12);
  ps: openroad 410982 with `-threads 48`; **/proc/410982/status:
  Threads: 49** (48 workers + main, verified 23:02 — the >100 % CPU
  shows in the parallel repair phase; library load is single-threaded
  I/O at ~99.6 %). warning.log flat: 2612 lines / 21× ORD-0032, all
  pre-relaunch. **Phase B complete.**
- **23:27** resizer (43, openroad 410982, 34 min) is in the
  `estimate_parasitics -global_routing` phase of rsz_timing_postgrt.tcl
  — an internal FastRoute run (GRT-0102 extra iterations 8/50,
  TopMetal2 horizontal 40.3 %). Multithreading confirmed in work:
  ps %cpu 527 %, monitor CPU time 10,459 s in 2,027 s elapsed (~5.1×),
  49 threads. Remaining: per-corner repair_timing (1 % TNS caps — fast)
  + write_views (flood + odb + DEF). warning.log still flat (21× ORD-0032).
- **23:54** resizer internal GRT at extra iteration 12/50 (~6.5 min/iter,
  main routing phase was silent/fast). p2_15's real GRT (step 39) needed
  ALL 50 extra iterations (residual overflow never reached 0), so expect
  the same: ~4.5 h more GRT + short capped repair_timing + write_views
  → step 43 done ~05:00–06:00. CPU 5.3× (19,082 s / 3,603 s), flow.log
  nearly flat (one line per iteration — expected, not a stall; the
  monitor's CPU-time liveness covers this).
- **00:34 (31.10.)** extra iter 18/50 (~6.8 min/iter, steady), CPU
  4.6× (27,923 s / 6,055 s), no restarts, disk 1.3 T. Internal GRT ETA
  ~04:20, step 43 done ~05:30–06:30.
- **01:14** extra iter 23/50 (~8 min/iter), CPU 4.7× (36,760 s), no
  restarts, no errors.
- **01:55** extra iter 27/50; per-iter time creeping up (6.5 → 8 →
  ~10 min/iter, remaining overflow is harder), CPU 3.9× (43,056 s /
  10,916 s). Internal GRT ETA ~05:30–05:45, step 43 done ~07:00.
- **02:55** extra iter 32/50 (~12 min/iter), CPU 3.7× (54,456 s /
  14,537 s), no restarts, no errors. Internal GRT ETA ~06:30,
  step 43 done ~07:30–08:00.
- **04:25** extra iter 39/50 (~13 min/iter), CPU 3.4× (67,656 s /
  19,956 s), no restarts, no errors. Internal GRT ETA ~07:00.
- **05:56** extra iter 44/50 (~18 min/iter, slowing on the last
  overflow), CPU 3.2× (80,914 s / 25,379 s), no restarts, no errors.
  Internal GRT ETA ~08:00, step 43 done ~09:30–10:00.
- **07:26** extra iter 48/50 — internal GRT finishes in ~40 min, then
  capped repair_timing (fast) + write_views (flood + odb + DEF).
  CPU 3.1× (94,108 s / 30,802 s), no restarts, no errors. Step 43
  done ~09:00–09:45, then STAMidPNR-3 (44).
- **08:33 (31.10.)** 2nd GRT run investigated — **it is expected,
  stock librelane 3.1 behavior**, and there will be a THIRD one.
  Read the full `rsz_timing_postgrt.tcl`: (1) line 29 sources
  `grt.tcl` → `global_route` = run #1 (22:54–07:49, 49 extra iters,
  ~9 h; NOTE: the 23:27 line above mislabeled run #1 as the
  `estimate_parasitics` phase — it was this `grt.tcl` call); (2) line
  31 `estimate_parasitics -global_routing` internally runs FastRoute
  AGAIN = run #2 (GRT-0102 counter restarts at 1/50, started ~07:49,
  at 08:33 at 9/50, ~4–5 min/iter — faster because it reuses run #1's
  routing results already in the DB; total iter lines after the
  22:53 separator = 58 = 49 + 9); (3) after the two capped
  `repair_timing` calls (setup, then hold — `GRT_RESIZER_FIX_HOLD_FIRST`
  default 0) the script sources `dpl.tcl` (remove_fillers +
  detailed_placement + optimize_mirroring on 3.1 M cells) and line 66
  sources `grt.tcl` a second time, because `GRT_RESIZER_RUN_GRT`
  defaults to **True** (`steps/openroad.py:2900`; the knob is NOT in
  `openeye.json`) → **run #3** after the re-DPL. No intervention:
  this is the flow's designed sequence (the caps bound only the
  resizer repair time, not the GRT runs), no errors, 0 restarts.
  CPU 3.15× (108,327 s), disk 1.3 T free.
- **08:33 ETA revision (replaces the ~09:00–09:45 estimate):**
  GRT#2 → ~10:00–12:00; repair_timing setup+hold (1 % TNS caps) ~1–3 h;
  re-DPL ~1–4 h; GRT#3 (main phase fast, extra iters 2–8 h);
  write_views (flood + 2.5 GB odb + DEF) ~1 h → **step 43 done
  ~15:00–21:00 (central ~18:00)**. Then 44 STAMidPNR-3 (~30–60 min,
  48 thr) + 45 DRT (~6–12 h) → **p2_15 done tonight ~23:00 → early
  01.11** (previously: "18:00–24:00 on 31.10" — roughly still holds,
  but now leaning to the early-01.11 side).
- **09:35** GRT#2 at extra iter 19/50 (10 iters in 62 min ≈ 6.2
  min/iter, steady — no slowdown yet; GRT#1 crept 4→20 min/iter on
  its tail). GRT#2 ETA ~13:00–15:00. CPU 3.2× (124,841 s), 0 restarts,
  disk 1.3 T, no dir 44 yet. Step-43 ETA unchanged (~15:00–21:00).
- **10:35** GRT#2 at extra iter 25/50 — 6 iters in 60 min = 10
  min/iter, the GRT#1-style tail slowdown starting (GRT#1: 4→20
  min/iter). Remaining 25 iters ≈ 4–8 h → **GRT#2 done ~14:45–19:00**.
  CPU 3.2× (136,117 s), 0 restarts, disk 1.3 T, no dir 44 yet.
  **Step-43 full breakdown (replaces the 08:33 estimate):** GRT#2
  4–8 h + repair_timing setup (clean at 20 ns, <1 h) + hold (1 % TNS
  cap; p2_14's capped postCTS resizer ref ~8 h worst case, post-GRT
  hold set is small — 48 violators / TNS −4.65 — expect 1–3 h) +
  re-DPL (1–3 h @ 48 thr) + GRT#3 (main ~40 min + extra iters,
  3–8 h) + write_views (~1 h) → **step 43 done 01.11 ~00:00–12:00
  (central ~03:00)**, then 44 STAMidPNR-3 (30–60 min) + 45 DRT
  (6–12 h) → **p2_15 done during the day 01.11** (previously:
  "tonight 23:00 → early 01.11" — that is stale, the 3rd GRT inside
  step 43 plus the GRT#2 tail pushed everything by ~1 day).
  **CALIBRATED 10:45 (runtime.txt refs):** p2_14/p2_15's step 36
  (SAME capped problem class: 48 hold violators, TNS −4.65, 1 % cap)
  actually took **10:40:28** — but it ran SINGLE-threaded (pre-A1);
  whether 26Q4's `repair_timing` benefits from 48 threads is unverified
  locally, so budget hold phase **3–12 h** (single-threaded ref
  10:40 is the upper bound). DPL ref: step 33 took only 00:43:33 →
  re-DPL ≤1 h. GRT ref: step 39 took 14:42:19 single-threaded; GRT#1
  with 48 threads took ~9 h → GRT#3 ≈ 9–12 h. Revised sum from 10:35:
  GRT#2 4–8 h + setup <1 h + hold 3–12 h + re-DPL ≤1 h + GRT#3 9–12 h
  + write_views ~1 h ≈ **17–35 h (central ~24 h)** → **step 43 done
  01.11 noon → 02.11 noon (central ~01.11 evening)**; then
  STAMidPNR-3 (~1 h) + DRT (6–12 h @ 48 thr, no local ref) →
  **p2_15 done late 01.11 → 02.11 (central ~02.11 morning)**.
- **09:43 (01.11.)** — MAJOR FINDING + flow re-mapped (full log
  archaeology; session had been suspended overnight, flow ran
  unattended healthy: openroad 410982 etimes 34.8 h, CPU 3.0×, 0
  restarts, disk 1.3 T). Command echoes in flow.log decode the real
  step-43 structure — FOUR internal FastRoute runs, not three:
  1. `global_route` (grt.tcl #1, echo rel 327): 10/30 22:54 → 10/31
     07:49, 49 extra iters (residual overflow, NDR off net781646).
  2. `estimate_parasitics -global_routing` (no echo): 07:49:23 →
     ~15:00–17:30, **50/50** extra iters + via filling (22,339,297
     vias) + final congestion report + extraction.
  3. `repair_timing -setup` (echo rel 493): started right after,
     1,500+ iters, ended 19:30:42 RSZ-0062 "Unable to repair all
     setup". **Setup at GRT (nom corner): WNS −193.865 → −73.773
     (FROZEN from ~iter 850), TNS −654,027 → −381,642, 84,625 →
     84,435 violators** (1 % cap allowed only 846 endpoints; only
     ~190 actually repaired). Worst endpt iter 0: PSUM_RAM
     u_mac0/A_DIN[2]; final worst: _2932751_/D ("no solution meets
     design rules" on _1829321_/Y).
  4. `repair_timing -hold` (echo rel 2009): 10,575 endpoints,
     10,570 iters, ended 21:35:21 RSZ-0064 "Unable to repair all
     hold checks" (6 hold buffers). **Hold at GRT: WNS −30.911
     FROZEN from early iters to final, TNS −52,369 → −36,955.**
  5. `detailed_placement -max_displacement 500 100` (echo rel 4141,
     ~5 min — near-zero incremental displacement).
  6. `global_route` (grt.tcl #2, echo rel 4155): 21:40 → 04:35:42,
     49 extra iters (~8.5 min/iter).
  7. **CURRENT**: `write_views` → `write_spef` (SAVE_SPEF set) runs
     its OWN internal FastRoute (NDR 04:35:42) — at 09:43 extra iter
     43/50, ~7.2 min/iter → done ~10:30–11:00, then via filling +
     REAL SPEF extraction + set_global_connections flood + ODB/DEF →
     **step 43 done ~12:00–13:00**, then step 44 STAMidPNR-3
     (multicorner STA, ~30–60 min @ 48 thr) → ~13:00–14:00.
  - Side observation: an EST-0026 "Missing route to pin" burst
    (×1000, message limit reached, 21:40:26) during GRT#C — some
    pins have no GRT route (residual overflow) → estimated with
    defaults; contributes to the pathological parasitics below.
  **FINDING — post-GRT timing is catastrophically worse than pre-GRT
  (this is NOT normal congestion degradation; stop criteria
  triggered):**
  - Pre-GRT (step 37, nom): setup WNS **+4.25** / TNS 0 / 0
    violators; hold WNS **−0.440** / TNS −4.65 / 48 violators.
  - Post-GRT estimate (step 43 resizer, nom): setup WNS **−73.77
    (frozen, unrepairable)** / TNS −381,642 / **84,435 violators**;
    hold WNS **−30.91 (frozen)** / TNS −36,955 / ~10,5xx violators.
  - Sanity check: a wire on a 10.5 mm die adds at most ~0.1–0.3 ns
    per mm; even a fully detoured worst path cannot physically
    explain −194 ns (pre-repair) / −74 ns (post-repair) at a 20 ns
    period. The frozen WNS (resizer exhausted: "no solution meets
    design rules") + EST-0026 missing-route burst + 49–50 extra
    GRT iterations with residual overflow point to **pathological
    `estimate_parasitics` results on the congested GRT solution**
    (cf. the tcl comment citing OpenROAD issue #5590 — exactly this
    code path). The true post-DRT signoff SPEF may be far better OR
    the worst paths may be real but still far from −74 ns.
  - **STOP CRITERIA TRIGGERED (two of them):** "post-route timing
    worse than pre-route by >1 ns WNS" (198 ns pre-repair / 78 ns
    post-repair on setup) AND "final Hold-WNS < −0.5 ns" (−30.91).
    Per the plan: analyze, record cause + proposal, **wait for the
    user** — no blind DRT re-run. DRT (12–18 h tail incl. fill/
    MCSTA/DRC/LVS) cannot fix a frozen −74 ns setup WNS (it follows
    the GRT guides), so continuing would only buy physical
    artifacts with guaranteed SetupViolations-gate failure (A5 made
    it a real gate) + HoldViolations failure.
  - **DECISION (autonomy mode, per stop criteria):** let step 43
    finish (SPEF + ODB/DEF — ~2 h) and step 44 STAMidPNR-3 finish
    (definitive per-corner STA, ~1 h), THEN KILL THE FLOW before
    step 45 DRT (kill chain: watchdog → monitor → librelane →
    openroad; dirs 43+44 complete, no restart). Then: (1) analyze
    step-44 reports per corner (worst paths, which nets, macro
    inputs vs fabric); (2) record finding + cause + proposals in
    "Current state"; (3) commit, push, WAIT for user.
  - PROPOSALS for the user (after the step-44 data):
    - **P1 — investigate the estimate before any further routing**:
      extract the GRT route lengths of the worst nets (ODB/guide
      files in dir 43), check whether −73.8 ns is a real detour
      mass or an estimation artifact. If artifact → the signoff
      SPEF after a DRT would tell the truth; a short DRT-only
      continuation (or a p2_16 re-run) becomes the measurement.
    - **P2 — continue p2_15 to GDS anyway** (~12–18 h): delivers the
      full physical baseline (GDS + DRC + LVS, 3 known PDK gaps)
      with a documented timing FAIL at the signoff gates; timing
      closure deferred to Phase D. Cost: 12–18 h that won't change
      the setup result.
    - **P3 — stop now (after step 44) and go straight to Phase D
      p2_16**: macro placement D1 (data ready: 3 bottom rows →
      split per cluster), D2 density/overflow, D3 halos; target the
      GRT congestion that produced this. p2_15 stands as proof the
      flow runs end-to-end through GRT with 48 threads.
- Remaining flow (new config): 44 STAMidPNR-3 → **STOP HERE
  (decision above)** → [45 DRT → fill → SPEF → MCSTA → DRC → LVS →
  signoff gates] only if the user chooses P2.

- **18:35 — A1 done** (threads 48, see plan item): knob in `gen_config.py`
  + regenerated `openeye.json`, diff = exactly 1 line.
- **18:41 — A2 done** (resume resolver, see plan item): new
  `asic/scripts/resolve_resume.py` + `test_resolve_resume.py`, watchdog
  rewired. Live p2_15 → `OpenROAD.RepairAntennas` (correct while step 42
  runs); 7/7 synthetic scenarios pass. New task **A7** added to the plan
  (post-GRT resizer is off by default — `RUN_POST_GRT_RESIZER_TIMING`
  False in p2_15 resolved.json; plan B3/ETA assumed it runs).
- **18:44 — A3 done** (run_openeye.sh, see plan item): append-mode
  flow.log + launch separator, RTL untouched on resume, success check
  via exit code + `final/`, DRY_RUN mode. Verified with dummy run
  `p2_drytest` (fresh/resume/warning paths), dummy dir deleted.
- **18:47 — A4 done** (monitor, see plan item): new tracked
  `asic/scripts/monitor_flow.sh <runname>` with CPU-time liveness
  (STALL only when log + step files + CPU time are ALL flat). Old
  monitor (330001) replaced by PID 397962; first snapshot 18:45:02:
  `cpu=370311:28691s`, no STALL.
- **18:52 — A5 done** (setup signoff gate, see plan item):
  `SETUP_VIOLATION_CORNERS: ["*"]` in gen_config.py + openeye.json
  (diff = exactly that knob). Effective from the Phase B relaunch; the
  gate runs at p2_15 end (step 79).
- **18:58 — A6 done** (DFF comment, see plan item): the wrong
  "70 % PE-pad latches" line was already gone (`2711d09`); the
  `DIE_AREA` comment now carries the verified DFF facts (no latches;
  506,219 `sg13g2_dfrbpq_1` = 24.8/55.3 mm² ≈ 45 %). Comment-only:
  regenerated openeye.json byte-identical.
- **19:05 — A7 done** (post-GRT resizer, see plan item):
  `RUN_POST_GRT_RESIZER_TIMING: true` in gen_config.py + openeye.json
  (diff = exactly that knob); resolver end-to-end with the real
  regenerated config → `OpenROAD.ResizerTimingPostGRT` for the 42
  frontier. **Phase A complete (A1–A7)** — Phase B triggers when step
  42 `state_out.json` exists (watching, ~30 min cadence).

**p2_15 (fork of p2_14, 50 MHz / 20 ns) relaunched ~19:30 — resuming at
`OpenROAD.GlobalRouting` (step 38) with `-F`.** Steps 1–37 reused from
p2_14 (run dir copied, state paths rewritten, mtimes restored).

### p2_14 result (launched 10-27 19:53, died at GRT 10-29 17:02)

- Steps 1–37 all clean; 20 ns confirmed in SDC (`Period: 20.000000`).
  CTS 4:13 (p2_12: 4:07), post-CTS resizer 10:40 (capped at 1 % TNS).
- **Timing (nom_typ 1.20 V 25 °C) — 50 MHz setup is MET:**

  | | Setup WNS | Setup TNS | Hold WNS | Hold TNS | Hold violators |
  |---|---|---|---|---|---|
  | pre-resizer (35) | 0.0 | 0.0 | −1.01 | −687 | — |
  | post-resizer (37) | 0.0 | 0.0 | −0.44 | −4.65 | 41 |

  Pre-resizer best setup slack +0.42 ns (prediction: ~+0.4). Post-resizer
  +4.25 ns. The 41 remaining hold violators are all on SRAM-macro inputs
  (wght_buffer A_DIN, BUFFER_A, PSUM); the fatal `_29536xx_` DMA
  input-capture net of the 10 ns runs is GONE — the 6.4 ns input delay
  works as designed. The post-GRT hold resizer (capped) takes the 41.
- **GRT failure (step 38), 13.8 h, 150 iterations:** `[GRT-0116] Global
  routing finished with congestion`. Final congestion report: GLOBAL
  usage only 30.3 % (24.2M/80M resources) — not a density problem;
  LOCAL overflow in the dense PE-fabric/SRAM regions (max 143 units in
  one M3-H cell, 108 M2-V; 216,531 total overflow = 0.9 % of demand).
  Metal1 is the PDN layer (routing M2..TM2), no spare layer; GRT
  already disabled the NDR on `net781646`. Evidence: p2_14 step dir
  `38-openroad-globalrouting` + flow.log (kept intact).
- **Fix: `GRT_ALLOW_CONGESTION: true`** (commit `82d393b`,
  `gen_config.py`). GRT then completes with the residual overflow
  (OpenROAD `-allow_congestion`); the detailed router (zroute) detours
  around it — at 30 % global usage / 6 layers the residual is inside
  what DRT can absorb. If DRT chokes on a hotspot, the next levers are
  `GRT_LAYER_ADJUSTMENTS` (steer demand between layers) or a placement
  change (expensive).

### p2_15 relaunch details (2026-10-29 evening)

- First launch at 19:03 went WITHOUT `-F` (wrong assumption, see
  Pitfalls) → re-ran from step 1 in dirs 39–44; killed at synthesis,
  junk dirs deleted, that flow.log archived as
  `flow.log.launch1_killed_1903`.
- `run_openeye.sh` now takes `<runname> <resume-step-id>` (→ `-F`),
  `watchdog.sh` resolves the resume step automatically before every
  launch (commit `234c2de`).
- **ETA (updated 10-30 12:20):** GRT **done 10-30 ~09:56 (14:42, with
  `-allow_congestion`)** → CheckAntennas 0:49 done → RepairAntennas
  running (12,970 antenna violations, 21,723 jumpers inserted) → DRT
  (10–20 h wildcard) → post-GRT resizer + STA → **STAPostPNR = 50 MHz
  signoff numbers ~10-31 midday/evening** → GDS ~10-31 evening →
  DRC/LVS (3 documented PDK-gap failures expected, see Blocked).

### History (as of 2026-10-27, evening)

#### Done and verified

- **Phase 0**: LibreLane Chip-flow smoke test (counter) passed; 3 PDK gaps
  documented (see `asic/README.md`).
- **Phase 1** (committed, branch `dev_asic_sg13g2`):
  - ASIC target: 2×2 clusters (48 PE), 10 ns, sparse, 2 MAC/PE, DMA 64 bit.
    Config in `asic/target.sh`, generation in `asic/scripts/gen.sh`
    (stages generated `.vh` files into `hdl/` — gitignored).
  - `hdl/iact_stream_constructor.v` NUM_GLB_IACT=3 path is **broken in this
    codebase snapshot** — target uses `NUM_GLB_IACT=1` (verified). Follow-up.
  - `asic/rtl/sg13g2/sg13g2_ram_sp.v`: SRAM macro wrapper (4 configs),
    unit-tested vs. generic model (`asic/sim/tb_ram_equiv.v`).
  - `hdl/RAM_SP.v` hook via `OPENEYE_RAM_SPECIFIC` define.
  - Chip top `asic/rtl/sg13g2/OpenEye_Asic.v` (generated by
    `asic/scripts/gen_top.py`, 143 pads) + `pad_order.json`.
  - **RTL regressions pass** (run in the container):
    ```
    docker exec iic-osic-tools_asic bash -lc "cd /foss/designs/OpenEye && asic/scripts/regress.sh"
    docker exec iic-osic-tools_asic bash -lc "cd /foss/designs/OpenEye && asic/scripts/regress_macros.sh"
    ```
    (macro variant uses the PDK functional SRAM models with `-DFUNCTIONAL`)
  - Verilator lint clean; yosys parse/elaborate clean.

### In progress — the open problem (TOP PRIORITY)

Flow run `p2_9` (config: `asic/rtl/sg13g2/openeye.json`, die 10.5 mm) got
through yosys synthesis (55.3 mm² netlist, 2.2 M cells) but failed at
step 8 `Checker.YosysSynthChecks` with **64 check errors**.

**Root cause (re-verified 2026-10-23 on est-srv02 — supersedes the earlier
hypothesis below it): the loops ARE unrolled. The 64 errors are a checker
artifact on dead loop-counter FFs, and the netlist is expected to be
functionally identical to the RTL.**

yosys 0.70 `proc` unrolls all 244 constant-bounded `for` loops in clocked
blocks (verified with minimal tests tb1–tb14 and on the real design): the
loop bodies come out with constant indices. What remains after `proc`:

- The loop-counter variables (module-scope `integer`) stay as FFs with
  self-referential next-state logic ("zombie counters") and **zero
  external fanout** — verified for all 15 of them in the post-`proc`
  netlist (`asic/runs/loopchk_after_proc.v`): `a, b, word, cc, cc_fc,
  cc_psum, cr, cr_psum, g, g_psum, pec, per, r, w, pipe_idx`.
- In `hdl/OpenEye_FPGA.v`, `a` and `b` are shared by **4 always blocks**
  (~lines 850/1037/1135/1547, 9/10/15/59 loops each). `write_verilog`
  emits such a merged FF as several `always` blocks driving one register
  → exactly **64 "multiple conflicting drivers" warnings** (32 bits × 2
  vars) in `pre_synth_chk.rpt`. The synthesis step counts them via
  `_parse_yosys_check`, and step 8 `Checker.YosysSynthChecks`
  (`ERROR_ON_SYNTH_CHECKS` defaults to true) aborts the flow — even though
  the netlist itself is fine.
- The flow's own `opt` passes remove every zombie counter: running the
  flow's pass sequence (`proc; check; opt_expr; opt_merge; opt_dff;
  opt_clean; opt_muxtree; opt_reduce; …`) on the real design leaves
  **zero** loop-counter FFs and **zero** check errors in the netlist
  (`asic/runs/flowchk.log`, `asic/runs/flowchk_after_proc_opt.v`).

**Resolved (2026-10-25/26), all gates green:**

1. **Gate-sim of the p2_10 netlist: PASS** (Verilator, build 7,
   2026-10-25 10:09→18:34, log `asic/runs/gatesim_p2_10c.log`):
   `OpenEye_FPGA_tb.start_test_fpga passed`, TESTS=1 PASS=1 FAIL=0 at
   10210 ns, 4.6 s sim time. Structural check of the generated code
   confirms the file-order fix holds at netlist scale (VpiScopeTable:
   exactly 2 rows — TOP + wrapper; `Vtop__Syms__Slow.cpp` 5 KB).
   A parallel Icarus run of the same netlist+test hung in the compute
   phase (14 h frozen at 4190 ns, 99.6% CPU, killed 2026-10-26):
   iverilog is not a viable full-netlist path for this design.
2. **Rename applied** (2026-10-26 05:14):
   `python3 asic/scripts/rename_loops_fpga.py --apply` → 210 lines in
   `hdl/OpenEye_FPGA.v` (`a`→`a1..a4`, `b`→`b1..b4`, 6 always blocks,
   one declaration line). Each FF now single-driven. **Both RTL
   regressions PASS post-rename** (05:13, `asic/runs/regress_after_rename.log`).
3. **Post-rename yosys recheck: PASS** (2026-10-26 05:23,
   `loopchk_after_proc.ys`, `proc; check` on the full RTL): **0**
   "multiple conflicting drivers" (pre-rename: 64), no ERROR lines,
   `write_verilog` clean (log `asic/runs/loopchk_after_rename3.log`,
   output `asic/runs/loopchk_after_proc.v`).
   Run with `yosys -p "$(cat …)"` — NOT `-c` (see pitfall below).
4. Alternative (documented workaround, not preferred):
   `ERROR_ON_SYNTH_CHECKS=false` in `openeye.json`.

~~Key facts (all experimentally verified):~~ *(superseded — kept for the
record)*

- The 64 errors = loop variables `a` and `b` (32-bit integers) in
  `hdl/OpenEye_FPGA.v`, shared by 4 always blocks → "multiple conflicting
  drivers" warnings (see `asic/runs/openeye_p2_9/design/runs/p2_9/06-yosys-synthesis/reports/pre_synth_chk.rpt`).
  An earlier run (`p2_8`) had 128 errors from the same class in
  `iact_stream_constructor.v` — that one is fixed (commit `c6125c5`).
- Scope: **244 `for` loops inside clocked blocks across 9 files**
  (OpenEye_FPGA.v: 94, psum_pipeline.v: 58, PE.v: 49,
  router_configurator.v: 24, PE_cluster.v: 12, iact_stream_constructor.v: 6,
  af_cluster.v: 5, PE_simple.v: 3, GLB_cluster.v: 2, adder_tree.v: 1,
  varlenFIFO.v: 2). All bounds are parameter expressions (compile-time
  constants); there are no `while` loops and no signal-dependent bounds.
- yosys unrolls loops **correctly** when they are in purely combinational
  processes (`always @(*)`) — verified with a minimal test.
- The slang frontend (`read_slang`) unrolls clocked-block loops correctly,
  BUT: (a) blackbox output ports end up undriven ("used but has no driver"),
  (b) it flattens the whole hierarchy and **drops blackbox cells** (RM macro
  instances vanish from the netlist; pads become uniquified stub modules).
  A two-stage `read_slang → write_verilog → read_verilog` workaround is a
  **dead end** because of (b).
- Block-scope `integer` declarations inside for loops are rejected by
  yosys's read_verilog ("Left hand side of 1st expression of procedural
  for-loop is not a register!").

### p2_11 result + fix, p2_12 running (2026-10-26)

`p2_11` (launched 05:47, ran ~5.4 h):

- **Step 8 `Checker.YosysSynthChecks` PASSED** (`error__count: 0`, was 64 in
  p2_9/p2_10 — the rename fixed it). Netlist 2,196,211 inst, 55.29 mm²,
  synthesis 58:41 min.
- Flow progressed through floorplan/padring/setpowerconnections/cutrows/
  tapendcap/addpdnobstructions (steps 9–21) and **died at step 22
  `OpenROAD.GeneratePDN`:
  `[PDN-0235] Design has unplaced macros`**. All 33 RM SRAM instances were
  unplaced: the flow's `Odb.ManualMacroPlacement` (step 18) is a SKIP step
  when `MACROS.<cell>.instances` is absent (it was — `openeye.json` never
  carried instance data), and OpenROAD never places macros on its own.
- **Fix (commit `3d6c242`)**:
  - `asic/scripts/macro_placement.py` — deterministic R0 placement of the
    33 instances in 3 bottom rows of the core (y=325: 5× wght
    2048x64/2048x32; y=970: 24× psum 4096x16/4096x8; y=1610: 4× BUFFER_A
    1024x64), snapped to the 5 µm grid, asserted in-core/no-overlap.
    Names verified 1:1 against the p2_11 padring DEF (ground truth:
    `asic/runs/macro_instances_p2_11.txt`).
  - `gen_config.py` now calls `merge_into_macros()` (openeye.json is
    regenerated every run, so the fix lives in the generator).
  - **End-to-end verified**: `placers.py manual-macro-placement` on the
    p2_11 padring ODB → "Successfully placed 33 instances", readback:
    33/33 `FIRM` at exactly the designed coordinates.
- **Second root cause found while verifying** (would have killed p2_12 at
  step 15/16): the container's default `openroad`
  (`/foss/tools/bin/openroad`, 26Q4-407) **does not accept `-python`** —
  LibreLane 3.1's Odbpy steps all use it and print usage + exit 1.
  p2_11 had only worked because a transient `$DESIGNS/.designinit` PATH
  override (sourced by the container login profile) pointed at the
  `-python`-capable build; the file vanished mid-run (06:52, mtime evidence)
  after the librelane process had already captured its 05:47 environment.
  `env.sh` now pins `_LLN_OVERRIDE_OPENROAD=/foss/tools/openroad-librelane/
  bin/openroad` (26Q1-1024, LibreLane's documented override) for ALL
  openroad steps. The `sta` binary stays at the 26Q4 build (pure TCL, fine).

`p2_12` launched **12:09:58** via the watchdog (host side), monitor running
(`asic/runs/monitor_p2_12.sh`, 5-min snapshots → `monitor_p2_12.log`).
Expected: steps 1–21 replay (~5 h), then step 22 `GeneratePDN` must pass
(macros now FIRM-placed). Beyond that: global/detailed placement → CTS →
routing → fill → multi-corner STA (10 ns) → DRC/LVS, where the 3 documented
PDK-gap failures are EXPECTED (see "Blocked / open threads").

### p2_12 timing situation (analyzed 2026-10-27 ~18:45) — DESIGN LEVEL

Steps 1–35 done (synthesis → floorplan → PDN (33 macros placed) → GPL →
DetPlace → CTS 03:36→07:44 → STAMidPNR-1 07:44→08:01). **Step 36
`ResizerTimingPostCTS` in progress since 08:01** (classic.py index 37/80).
Design: 2,196,211 instances, die 10.5×10.5 mm, core ~9.86 mm, util 56.9%.

The resizer runs two phases (one `repair_timing` each):

- **Setup phase (08:02→10:41, done)**: 270,709 violating endpoints at start.
  WNS **-19.87 → -7.66**, TNS **-746,260 → -601,575** (sum over 3 corners;
  nom_typ alone: WNS -9.58, TNS -88,584 per step-35 report). Only **324
  endpoints fixed** (270,385 still violating) — 200 buffers, 730 upsizes.
  Gave up: `[RSZ-0062] Unable to repair all setup violations`.
- **Hold phase (10:41→, running)**: 501,294 endpoints with hold violations
  at start. WHS **-4.63 → -4.06 (flat)**, THS **-27,166 → ~-350** (converging,
  ~10 ns/min at 18:37, decelerating). **509k hold buffers inserted,
  area +14.6%**. Ends when OpenROAD stops making progress.

Why setup cannot be closed (evidence, step-35 reports):
- Worst setup path (nom_typ, -9.58): FF `_2929373_` (PE cluster
  gen_x[0].gen_y[0]) → **~15.6 ns of buffer/inverter chain on a
  cross-chip net** (caps 0.19–0.79 pF) → ~30-stage a21o/a21oi/o21ai
  carry chain (~2.8 ns) → FF `_2926539_`; data arrival 23.8 ns vs 10 ns
  period. **Wire-dominated** — resizer upsizing/cloning can't help.
- The violations are broad (270k endpoints), not one path: the PE fabric
  data paths are systematically too long for 10 ns at this placement.
- Worst HOLD endpoint: `data_dma_i_pad[N] → _29536(10+N)_/D` (-2.85
  nom_typ): clock reaches these input-capture FFs **4.952 ns late** (deep
  clock-tree leaf `clkbuf_leaf_38075`), while external input delay is only
  2.0 ns → data arrival 2.327 vs required 5.173. Not fixable by data-side
  buffers; it is a clock-latency-vs-input-delay mismatch (fix later: input
  delay consistent with clock latency, or CTS skew target at IO regs).

**Consequence:** the 10 ns target is NOT reachable by P&R on this netlist.
The flow still completes physically (GDS/DRC/LVS baseline). Endgame
(expected): steps 38–76 produce everything; step 77 `Checker.HoldViolations`
(corner list `['*']` — HARD gate) will most likely FAIL on the remaining
hold WNS → flow stops controllably there (watchdog will NOT restart —
correct). Step 76 `Checker.SetupViolations` is a NO-OP by default
(`SETUP_VIOLATION_CORNERS` unset → `['']` matches no corners). Skipped steps
per config: 41/43/45 (GRT repairs, RUN_*=False), 59 (IR drop), 75 (EQY).

Timing fix options (after the run, needs a decision):
1. RTL re-pipelining of the cross-cluster fabric data paths (proper fix).
2. Clock relaxation (e.g. 15–20 ns) if the product allows.
3. Placement/floorplan rework (grouping the clusters) — largest effort.

### p2_12 resizer decision + data-loss incident (2026-10-27 evening)

User decision (evening 2026-10-27): cap resizer TNS repair at 1 % and
re-run step 36. While executing it, the p2_12 run was destroyed:

1. openroad (step 36, PID 20591) killed 19:13; LibreLane exited cleanly
   19:14:11 (resizer in its HOLD phase, iter 491,980, THS -219.9, WHS
   -4.064). Watchdog was disabled before the kill (no auto-restart).
2. The resume attempt used `librelane openeye.json --overwrite --run-tag
   p2_12 -F OpenROAD.ResizerTimingPostCTS` — **WRONG**: `--overwrite`
   means "Overwrite run, if exists" and LibreLane implements it as
   `shutil.rmtree(run_dir)` (`librelane/flows/flow.py`, `flow.start`).
   The entire run dir (steps 1–35, ~30 h of synthesis/floorplan/PDN/
   placement/CTS outputs) was deleted; the resume then failed with
   `ResizerTimingPostCTS: missing required input 'odb'` and started a
   NEW run numbering (step 36 became dir `01-…`).
3. No reliable recovery: the ext4 blocks may still be on disk (would
   need `debugfs` with root on /dev/sdd1), but LibreLane's run state
   (state_out.json, per-step outputs, view lineage) was wiped with them,
   so the flow cannot be re-attached. Even a perfect file-level recovery
   would not give a resumable run.

**Lessons (also in Pitfalls):** resume with the plain command
`librelane openeye.json --run-tag <tag> [-F <StepId>]` — NO
`--overwrite`. `run_openeye.sh` has been fixed accordingly (its old
`--overwrite` would also have wiped the run on EVERY watchdog
crash-restart). Corrected phase picture: the 11 h+ tail was the HOLD
phase (not setup, which gives up on its own after ~2.7 h, RSZ-0062),
so ALL FOUR resizer TNS knobs are capped at 1 % now (commit `92675e6`):
`PL_RESIZER_{SETUP,HOLD}_REPAIR_TNS_PCT` (postCTS) +
`GRT_RESIZER_{SETUP,HOLD}_REPAIR_TNS_PCT` (postGRT). The HoldViolations
gate fails on WHS with or without caps — the caps only bound resizer run
time (~2.7 h setup + ~8 h hold at step 36).

**p2_13** relaunched 2026-10-27 ~19:25 via the watchdog (host side),
fresh run name (naming rule: never reuse a run name), config otherwise
identical (macros placed, openroad pin, 4 resizer caps). Monitor:
`asic/runs/monitor_p2_13.sh` → `monitor_p2_13.log` (p2_12 monitor
stopped). p2_12 evidence retained: `asic/runs/openeye_p2_12/flow.log`
(3 GB — full resizer iteration table via tee; the archived step log
`…log.uncapped_20261027_1914` was inside the wiped dir and is lost).

ETA p2_13 (from ~19:25 on 10-27): steps 1–35 ~20 h (→ 10-28 ~15:30),
step 36 resizer ~11 h capped (→ 10-29 ~03:00), GRT/DRT/RCX/STA/fill/
GDS/DRC/LVS ~6–10 h, DRT the longest pole (→ 10-29 afternoon).
Expected stop: step 77 HoldViolations (WHS ~-4.06, controllable) — or
earlier on a NEW error class (then analyze, don't blindly re-run).

**50 MHz decision (user, ~19:45) — p2_13 killed, p2_14 launched.**
After the frequency analysis (50 MHz realistic on the current
RTL/floorplan, 66 MHz borderline, 100 MHz needs PE-fabric
re-pipelining — table in the chat of 2026-10-27), the user chose to
build the first closed physical baseline at 50 MHz as soon as possible.
p2_13 was killed ~19:50 (synthesis stage, ~25 min lost, partial run dir
removed) and **p2_14** relaunched ~19:53 with (commit `cbf35ff`):
- `CLOCK_PERIOD: 20` (50 MHz)
- `IO_DELAY_CONSTRAINT: 32` — pad in/out delay = period × 32/100 =
  **6.4 ns** (default 20 would give 4.0 ns, still below the worst hold
  requirement 5.173 ns at the DMA input-capture regs; 6.4 ns covers it
  with ~1.5 ns margin). Assumption: external synchronous driver at
  50 MHz with ≤ 6 ns clock-to-pad — **verify against the board spec**.
- unchanged otherwise: 33 macros placed, openroad pin, 4 resizer caps.

Note: the internal hold problem (deep clock-tree skew, 501k endpoints
pre-resizer) is period-independent — the hold phase does the same work
at 20 ns, bounded by the 1 % cap. The SETUP phase, however, is nearly
clean at 20 ns (pre-resizer WNS ≈ +0.4 nom / +3.5 slow) and finishes in
minutes.

ETA p2_14 (from ~19:55 on 10-27): steps 1–35 ~20 h (→ 10-28 ~16:00),
step 36 resizer ~8 h (setup minutes + hold capped, → 10-29 ~00:00),
GRT + antenna + postGRT resizer (capped) ~8 h (→ 10-29 ~08:00),
**DetailedRouting 10–20 h = longest pole** → **50 MHz post-route STA
numbers ~10-29 evening**, then fill/GDS/DRC/LVS → **GDS ~10-30
morning**. Expected stop: none (hold gate 77 should pass with the 6.4 ns
input delay + capped resizers); if the caps leave a few −0.x ns hold
endpoints, step 77 fails controllably with all physical artifacts
already existing — then decide: tighten the hold cap (0.1 %) for a
finishing pass or accept.

### Gate-level simulation harness (built, p2_10 netlist validated: PASS)

`asic/sim/gatesim/` contains:
- `run_verilator.sh` — Verilator gate-sim build + run (preferred path;
  flag pitfalls documented above).
- `OpenEye_Asic_gatesim_top.v` — thin wrapper mapping the pad-level ports of
  the synthesized top back to the `OpenEye_FPGA` signal names, so the
  existing cocotb testbench (`test/cocotb_fpga/OpenEye_FPGA_tb.py`,
  `start_test_fpga`) runs unchanged. Carries the 9 `/*verilator
  public_on*/` port marks (REQUIRED, see the VPI pitfall entry above).
- `sg13g2_primitives_vlt.v` — behavioral stand-ins for the 17 `ihp_*`
  UDP primitives (Verilator has no UDP support; semantics transcribed
  from the UDP tables).
- `Makefile` — cocotb Makefile for iverilog (slow fallback): netlist +
  `sg13g2_stdcell.v` (functional models; the UDP file is not needed
  functionally) + 6 RM FUNCTIONAL models + `sg13g2_io.v`. Exports the
  ASIC-target env vars. NOTE: `NETLIST` defaults to the p2_9 netlist —
  pass the p2_10 path explicitly. Usage (container):
  ```
  cd /foss/designs/OpenEye/asic/sim/gatesim && make run NETLIST=<...>/OpenEye_Asic.nl.v
  ```
- `gatesim_probe_tb.v` — standalone liveness probe (reset + ready handshake).
- `forloop_tests/` — minimal yosys tests + real-design evidence settling
  the for-loop question (see above).
- `vpi_probes/` — the Verilator VPI exposure decision probes (P15b/P15c)
  with scope-table/codegen excerpts; evidence for the
  marks+`--flatten` requirement.

Caveats learned (final):
- iverilog on the full netlist: elaboration peaks at **~17.6 GB RAM**,
  and the cocotb run **hung in the compute phase** — log flat at
  4190 ns ("Stream is sent") for 14 h at 99.6% CPU (killed 2026-10-26,
  log `asic/runs/gatesim_iverilog_p2_10.log`). Flat log tail + growing
  vvp TIME column = stuck, not slow. Verilator runs the same 10210 ns
  in 4.6 s — it is the only viable full-netlist path.
- Verilator 5.052: build 7 economics: codegen+link ~8.4 h total
  (cvt effectively single-threaded: `Walltime 31744 s (cvt=31016 s)`
  on 32 threads), sim 4.6 s. The verified flag set (incl. why NOT
  `--binary`, why NOT `--public*`, why file order matters, why the
  specify-strip is needed) is documented above at the
  `run_verilator.sh` entry.

## Next steps (as of 2026-10-27 evening) — SUPERSEDED by the TODO PLAN at the top

p2_14 is running (launched ~19:53 via the watchdog, host side; config:
**50 MHz** — `CLOCK_PERIOD: 20`, `IO_DELAY_CONSTRAINT: 32` (6.4 ns pad
delay), 4 resizer TNS caps at 1 %; see "p2_12 resizer decision +
data-loss incident"). The job now is: **self-monitor p2_14** (user
requirement 2026-10-26) and analyze — not blindly re-run — on any failure.

### 1. Self-monitoring (USER REQUIREMENT — do it proactively, ~every 30 min)

- `tail asic/runs/openeye_p2_14/flow.log` — step progress visible?
- `tail -20 asic/runs/monitor_p2_14.log` — background monitor (5-min
  snapshots; flags STALL / RESTART / WD-EXITED / DEAD / DONE).
- `tail asic/runs/watchdog_p2_14.log` — restart count (budget 3).
- `docker exec iic-osic-tools_asic bash -lc 'ps aux --sort=-pcpu | head -8'`
  — the current step process alive and burning CPU?

Decision rules:
- **Stall** (flow.log flat > 1 h on a long step, step process dead or
  0% CPU) or **dead beyond the watchdog budget** → re-launch with a
  **NEW run name** (p2_15):
  `setsid nohup asic/scripts/watchdog.sh p2_15 > asic/runs/watchdog_p2_15.log 2>&1 < /dev/null &`
  (never reuse the run name; the old dir stays as evidence).
- **Design-level checker failure** (flow stops controllably, no crash) →
  do NOT blindly re-run: analyze `asic/runs/openeye_p2_14/flow.log` + the
  failing step's dir (log, reports, COMMANDS file) first.
- p2_14 milestones (steps 1–35 replayed from p2_12 evidence): 18 macro
  placement ("Successfully placed 33 instances"), 22 GeneratePDN (1 h 08,
  0× PDN-0235, PSM-0069 expected), 34 CTS (~4 h 07), 35 STAMidPNR-1
  (~18 min — at 20 ns expect WNS setup ≈ +0.4 nom_typ, hold ≈ -2.85
  pre-resizer, internal), 36 resizer (~8 h: setup minutes, hold capped),
  39 GlobalRouting (hours), 47 DetailedRouting (longest pole, 10–20 h),
  then fill/RCX/STAPostPNR (**the 50 MHz numbers**, ~10-29 evening) →
  GDS (~10-30 morning) → DRC/LVS. Expected flow stop: none, unless the
  hold caps leave −0.x ns tails (then step 77 fails controllably — see
  ETA note in the incident section).

### 2. After the flow finishes (or dies)

- If DRC/LVS fails ONLY with the documented PDK gaps (bondpad LVS, PSM-0069
  pad vdd, sealring offset) → that is the expected outcome; collect
  metrics (utilization, timing at 10 ns, WNS/TNS per corner) and report.
- Any NEW error class → analyze, fix in `gen_config.py`/env (config is
  regenerated every run), relaunch with the next run name.
- Scratch cleanup (host `/home/utz/projekte/`): done 2026-10-26, incl.
  `vpi_micro/` (3.9 GB, deleted per user decision — the verified VPI probe
  evidence lives in `asic/sim/gatesim/vpi_probes/`, committed).
- Update this file's "Current state" + "Git state" with the outcome.

### 3. Re-launch chain (reference)

`watchdog.sh p2_NN` (HOST, detached) → `run_openeye.sh p2_NN [StepId]` →
staging (container `gen.sh` + host `gen_top.py` + `gen_config.py`) →
`docker exec iic-osic-tools_asic bash -lc "cd …/openeye_p2_NN/design &&
source asic/scripts/env.sh && librelane openeye.json --run-tag p2_NN
[-F StepId]"`
— **no `--overwrite`** (it wipes the run dir) and **resume only with
`-F`** (a plain `--run-tag` re-runs everything from step 1; see
Pitfalls). The watchdog resolves the resume step automatically since
2026-10-29. All pre-flow gates are green (RTL regressions post-rename,
p2_10 netlist gate-sim PASS, loopchk 0 conflicts, synthesis checker 0
errors in p2_11).

## Key commands

| Task | Command |
|---|---|
| RTL regression (generic RAM) | `docker exec iic-osic-tools_asic bash -lc "cd /foss/designs/OpenEye && asic/scripts/regress.sh"` |
| RTL regression (PDK SRAM macros) | `… asic/scripts/regress_macros.sh` |
| Generate ASIC RTL (target 2×2) | `… bash -lc "cd /foss/designs/OpenEye && asic/scripts/gen.sh"` |
| Run flow (new run name each time) | HOST side: `setsid nohup asic/scripts/watchdog.sh p2_11 > asic/runs/watchdog_p2_11.log 2>&1 < /dev/null &` (watchdog→run_openeye.sh does `docker exec` itself; do NOT wrap in docker exec) |
| Flow progress | `tail asic/runs/openeye_p2_10/flow.log`; step dirs under `asic/runs/openeye_p2_10/design/runs/p2_10/` |
| yosys parse/elaborate check | pattern in `asic/README.md` (`read_verilog -sv -noautowire …; hierarchy -top OpenEye_Asic -check`) — add `proc; check` to catch driver conflicts |

Important: cocotb target values must be **make command-line variables** (see
`asic/scripts/regress.sh` header) — the test Makefile's `export` lines
override inherited environment.

## Pitfalls (learned the hard way)

- yosys loop semantics (re-verified 2026-10-23): constant-bounded `for`
  loops in clocked blocks are **unrolled** by `proc`; the counter variable
  remains as a dead FF that the flow's `opt` passes remove. Loops are NOT
  a synthesis hazard by themselves (minimal tests:
  `asic/sim/gatesim/forloop_tests/`).
- Two `always` blocks must never drive the same reg (even "only in reset"
  or "only as loop counter") — yosys merges them into one FF and
  `write_verilog` emits several `always` blocks for it → "multiple
  conflicting drivers" check errors that fail step 8. This is the actual
  cause of the p2_8/p2_9/p2_10 failures (commit `c6125c5` + the
  `a`/`b` rename in this commit fix all known cases).
- **yosys `-c` is broken in this container's yosys 0.70 build** (2026-10-26):
  `yosys -c script.ys` dies with `ERROR: TCL interpreter returned an
  error: invalid command name "read_verilog"` (both the
  `/foss/tools/bin/yosys` wrapper and the bare `/foss/tools/yosys/bin/yosys`;
  `-p` works fine, the LibreLane flow is unaffected — it uses pyosys `-y`).
  Always run command scripts as `yosys -p "$(cat script.ys)"` (container
  paths). The wrapper additionally force-injects `-m ghdl -m slang` on
  every non-`-h` call.
- iverilog vvp can hang at ~100% CPU in event storms without sim time
  advancing (full p2_10 netlist: 14 h frozen at 4190 ns, killed
  2026-10-26). Flat log tail + growing vvp TIME column = stuck, not slow;
  kill it and use Verilator (4.6 s for the same 10210 ns).
- `pgrep -f "pattern"` self-matches its own wrapper shell (e.g. the
  `bash -c` that runs the pgrep) — use bracket patterns (`[y]osys -c`)
  and verify hits with `ps -p <pid>`.
- LibreLane 3.1: `meta` in the JSON has no `name` field; config paths must be
  **container paths** (`/foss/...`); `DIE_AREA=[0,0,10500,10500]`;
  SRAM liberty files cannot be read by yosys in non-`-lib` mode → the blackboxes
  in `asic/rtl/sg13g2/sg13g2_sram_blackbox.v` are mandatory (EXTRA_VERILOG_MODELS).
- **LibreLane 3.1 never places macros by itself**: `Odb.ManualMacroPlacement`
  only acts on `MACROS.<cell>.instances` (unescaped ODB names +
  `{"location": [x, y], "orientation": "R0"}`; unknown instance keys are
  rejected, missing names hard-fail the step). Without it, the flow dies at
  `OpenROAD.GeneratePDN` with `[PDN-0235] Design has unplaced macros`
  (p2_11). The instances come from `asic/scripts/macro_placement.py` via
  `gen_config.py`.
- **Two OpenROAD builds in the container**: `/foss/tools/bin/openroad`
  (26Q4-407, NO `-python` flag) and `/foss/tools/bin/openroad-librelane`
  (26Q1-1024, has `-python`, required by all Odbpy steps). `env.sh` pins
  `_LLN_OVERRIDE_OPENROAD` to the librelane build. Do NOT rely on
  `$DESIGNS/.designinit` (a login-profile hook that vanished mid-p2_11 and
  silently changed which binary the flow used — the librelane process keeps
  its launch-time env, so steps can run on a different build than a fresh
  shell sees). Symptom of the mismatch: openroad prints its usage and exits
  1 even though the argv is identical to a known-good step.
- **`librelane --overwrite` WIPEs the run** (2026-10-27, cost: 30 h of
  p2_12): the flag means "Overwrite run, if exists" → LibreLane does
  `shutil.rmtree(run_dir)` (`librelane/flows/flow.py`, `flow.start`) and
  starts a fresh run — step dirs are renumbered from `01-`. NEVER use it
  for a resume. `run_openeye.sh` no longer passes it.
- **Resume REQUIRES `-F <StepId>`** (verified in librelane source +
  empirically 2026-10-29, p2_15 fork): a plain
  `librelane openeye.json --run-tag <tag>` on an existing run does NOT
  skip finished steps — it re-runs the WHOLE flow from step 1 in NEW
  numbered directories (39, 40, ...), using only the latest
  `state_out.json` as initial state (the flow.py loop only skips via the
  `-F`/`-T`/`--skip`/gating flags; `Step.start` has no
  "already finished" short-circuit). The p2_15 fork launch at 19:03
  without `-F` re-ran synthesis and was killed. Use
  `run_openeye.sh <tag> <StepId>` — the watchdog now auto-resolves the
  resume step (first flow step without a complete step dir; a dir is
  complete if it has `state_out.json`, or holds ONLY `config.json` +
  `state_in.json` like the output-less `Checker.PowerGridViolations`).
- **Forking a run dir (run-tag copy) has two traps**: (1) every
  `state_*.json` embeds ABSOLUTE paths to the old run dir → rewrite
  them; (2) `sed -i` (or any rewriting) resets mtimes to "now" → the
  latest-`state_out.json` selection (by mtime) then picks the wrong
  state → restore original mtimes with `touch -r <old> <new>` per file.
  After both, verify: latest `state_out.json` = last finished step, and
  all paths referenced in it exist in the new dir.
- `/tmp/opencode/` (host) is NOT visible inside the container — write
  container outputs to container paths (`/tmp/...` inside the container).
- Make variable name collisions: never name a Makefile variable `RM` (clobbers
  the built-in remove command) or reuse env-set names like `PDK`/`PDK_ROOT`.
- WSL/old machine: only one yosys synthesis at a time on <32 GB RAM.
- Push to GitHub is blocked (see below) — all Phase-0/1/2 work is in local
  commits on `dev_asic_sg13g2`.

## Blocked / open threads

- **Push: RESOLVED 2026-10-30.** No write access to upstream
  `Learning-Chips-Lab/OpenEye`; the work lives in the fork
  `github.com/utz12/OpenEye`, branch `dev_asic_sg13g2` (pushed, HEAD
  9654ac4). SSH: default key `~/.ssh/id_ed25519` on est-srv02 is
  registered on GitHub (comment `est-srv02-github`), remote `utz12`
  configured in the repo. Future pushes: `git push utz12 dev_asic_sg13g2`.
  A PR into upstream can be created from the fork if ever needed
  (https://github.com/utz12/OpenEye/pull/new/dev_asic_sg13g2).
- PDK gaps (documented in `asic/README.md`): bondpad↔pad-port LVS mismatch,
  pad `vdd` pins unconnected (PSM-0069), sealring offset (bondpads outside
  sealring). Plus SRAM `vddarray`/`vssarray` have no PDK power net.
- `NUM_GLB_IACT=3` datapath broken (target uses 1).
- Pad timing: pads are blackboxes in the flow; sg13g2_io liberty files exist
  and could be added for pad-timing accuracy (low priority).

## Git state

Branch `dev_asic_sg13g2`, commits (newest first) as of 2026-10-29 evening:
```
234c2de Scripts: correct LibreLane resume semantics (verified in source 2026-10-29)
82d393b Config: GRT_ALLOW_CONGESTION=true after p2_14 GRT-0116 failure
4ceb2ea QWEN.md: correct p2_13/p2_14 timestamps and ETAs
9eb8e6d QWEN.md: 50 MHz decision, p2_14 as current run, updated ETAs
cbf35ff Config: 50 MHz variant — CLOCK_PERIOD 20 ns, IO_DELAY_CONSTRAINT 32
f0e14fd QWEN.md: p2_12 --overwrite data-loss incident + p2_13 plan; drop --overwrite from run_openeye.sh
92675e6 gen_config: cap resizer HOLD TNS repair at 1% as well (4 knobs)
f55a817 gen_config: cap resizer TNS repair at 1% (p2_12 resizer incident)
ca4e74f QWEN.md: p2_12 timing analysis (2026-10-27) — 10 ns not closable by P&R
3d6c242 Fix PDN-0235: place the 33 SRAM macros; pin -python-capable openroad
```
15+ commits ahead of `origin/main`; **push blocked** (see above).
`asic/runs/` (flow outputs, logs, netlists) is gitignored — all evidence
(gatesim logs, loopchk outputs, regression logs, the p2_10 netlist)
exists only on est-srv02.
Generated files (`hdl/*.vh`, `hdl/dma_storage.v`) are gitignored and must
be regenerated with `asic/scripts/gen.sh` after cloning.
