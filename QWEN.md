# QWEN.md — OpenEye ASIC (IHP SG13G2) Handoff

Instructions for continuing the OpenEye ASIC backend work. Read `asic/README.md`
for the phase-by-phase history and this file for the **current state and the
open problem**.

## Objective

Implement the OpenEye DNN accelerator as an ASIC on IHP SG13G2: full RTL→GDS
backend via LibreLane 3.x in the iic-osic-tools container, using the real
`RM_IHPSG13_1P` SRAM macros for the top-level staging buffers.

## TODO PLAN (2026-10-30, review by Claude) — START HERE

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
- Remaining flow (new config): 43 ResizerTimingPostGRT (hold/setup
  repair, 1 % TNS caps) → 44 STAMidPNR-3 → 45 DRT (first 48-thread
  routing step) → fill → SPEF → MCSTA → DRC → LVS → signoff gates
  (setup is now a real gate via A5; hold was already gated).

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
