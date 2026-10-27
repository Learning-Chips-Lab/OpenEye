# QWEN.md — OpenEye ASIC (IHP SG13G2) Handoff

Instructions for continuing the OpenEye ASIC backend work. Read `asic/README.md`
for the phase-by-phase history and this file for the **current state and the
open problem**.

## Objective

Implement the OpenEye DNN accelerator as an ASIC on IHP SG13G2: full RTL→GDS
backend via LibreLane 3.x in the iic-osic-tools container, using the real
`RM_IHPSG13_1P` SRAM macros for the top-level staging buffers.

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

## Current state (as of 2026-10-26)

### Done and verified

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
Until then: let p2_12 finish for the full physical baseline.

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

## Next steps (as of 2026-10-26 ~12:10)

p2_12 is running (launched 12:09:58 via the watchdog, host side). The job
now is: **self-monitor p2_12** (user requirement 2026-10-26) and analyze —
not blindly re-run — on any failure.

### 1. Self-monitoring (USER REQUIREMENT — do it proactively, ~every 30 min)

- `tail asic/runs/openeye_p2_12/flow.log` — step progress visible?
- `tail -20 asic/runs/monitor_p2_12.log` — background monitor (5-min
  snapshots; flags STALL / RESTART / WD-EXITED / DEAD / DONE).
- `tail asic/runs/watchdog_p2_12.log` — restart count (budget 3).
- `docker exec iic-osic-tools_asic bash -lc 'ps aux --sort=-pcpu | head -8'`
  — the current step process alive and burning CPU?

Decision rules:
- **Stall** (flow.log flat > 1 h on a long step, step process dead or
  0% CPU) or **dead beyond the watchdog budget** → re-launch with a
  **NEW run name** (p2_13):
  `setsid nohup asic/scripts/watchdog.sh p2_13 > asic/runs/watchdog_p2_13.log 2>&1 < /dev/null &`
  (never reuse the run name; the old dir stays as evidence).
- **Design-level checker failure** (flow stops controllably, no crash) →
  do NOT blindly re-run: analyze `asic/runs/openeye_p2_12/flow.log` + the
  failing step's dir (log, reports, COMMANDS file) first.
- Milestones 15/18 (macro antenna / placement: "Successfully placed 33
  instances") and 22 (GeneratePDN) all PASSED in p2_12 (10-26 17:56).
- Current milestone (10-27): step 36 resizer hold phase converging
  (THS ~-350 and falling, WHS flat -4.06 — see "p2_12 timing situation").
  Next: step 38 STAMidPNR (~17 min), 39 GlobalRouting (hours), 47
  DetailedRouting (longest pole, ~10–20 h), then fill/RCX/STA/GDS/XOR/
  DRC/LVS. Expected stop point: step 77 HoldViolations (controllable,
  all physical artifacts exist by then).

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

`watchdog.sh p2_NN` (HOST, detached) → `run_openeye.sh p2_NN` →
staging (container `gen.sh` + host `gen_top.py` + `gen_config.py`) →
`docker exec iic-osic-tools_asic bash -lc "cd …/openeye_p2_NN/design &&
source asic/scripts/env.sh && librelane openeye.json --overwrite
--run-tag p2_NN"`. All pre-flow gates are green (RTL regressions post-
rename, p2_10 netlist gate-sim PASS, loopchk 0 conflicts, synthesis
checker 0 errors in p2_11).

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
- `/tmp/opencode/` (host) is NOT visible inside the container — write
  container outputs to container paths (`/tmp/...` inside the container).
- Make variable name collisions: never name a Makefile variable `RM` (clobbers
  the built-in remove command) or reuse env-set names like `PDK`/`PDK_ROOT`.
- WSL/old machine: only one yosys synthesis at a time on <32 GB RAM.
- Push to GitHub is blocked (see below) — all Phase-0/1/2 work is in local
  commits on `dev_asic_sg13g2`.

## Blocked / open threads

- **Push**: `utz12` has no write access to `Learning-Chips-Lab/OpenEye`, no
  fork, no PAT. Either add `utz12` as collaborator, create a fork, or use a
  PAT. Until then, transfer the branch via git bundle:
  `git bundle create openeye_dev_asic_sg13g2.bundle dev_asic_sg13g2`
  — a bundle has been created in the repo root on the old machine (untracked);
  copy it over and `git clone openeye_dev_asic_sg13g2.bundle` on the new one.
- PDK gaps (documented in `asic/README.md`): bondpad↔pad-port LVS mismatch,
  pad `vdd` pins unconnected (PSM-0069), sealring offset (bondpads outside
  sealring). Plus SRAM `vddarray`/`vssarray` have no PDK power net.
- `NUM_GLB_IACT=3` datapath broken (target uses 1).
- Pad timing: pads are blackboxes in the flow; sg13g2_io liberty files exist
  and could be added for pad-timing accuracy (low priority).

## Git state

Branch `dev_asic_sg13g2`, commits (newest first) as of 2026-10-26 ~12:15:
```
3d6c242 Fix PDN-0235: place the 33 SRAM macros; pin -python-capable openroad
665c5c9 QWEN.md: make the re-entry plan self-contained for the post-reboot session
f7a4c05 Split shared loop counters in OpenEye_FPGA; add Verilator gate-sim harness
8562ade QWEN.md: document est-srv02 setup and missing container packages
0b4a157 Add gate-level simulation harness and handoff notes (QWEN.md)
c6125c5 Fix multi-driven rd_cycle_loop_cnt_* in iact_stream_constructor (yosys check)
d65c16b Make the ASIC target Yosys-synthesisable; add Phase-2 flow scaffolding
56b65ec Add ASIC target RTL for IHP SG13G2 (Phase 1: 2x2 target, SRAM macros, chip top)
3f9f18d Add missing counter.json smoke config (root .gitignore excludes *.json)
ed66784 Add ASIC backend setup for IHP SG13G2 (Phase 0: LibreLane Chip-flow smoke test)
```
10 commits ahead of `origin/main`; **push blocked** (see above).
`asic/runs/` (flow outputs, logs, netlists) is gitignored — all evidence
(gatesim logs, loopchk outputs, regression logs, the p2_10 netlist)
exists only on est-srv02.
Generated files (`hdl/*.vh`, `hdl/dma_storage.v`) are gitignored and must
be regenerated with `asic/scripts/gen.sh` after cloning.
