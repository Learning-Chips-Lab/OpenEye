#!/bin/bash
# Gate-level simulation of the synthesized OpenEye_Asic netlist with Verilator
# (fast path; the iverilog `make run` in this directory is the slow fallback).
#
# Builds a cocotb-VPI Verilator binary from
#   wrapper (OpenEye_Asic_gatesim_top.v) + netlist + PDK functional models
# and runs the same functional test as the RTL regression
# (MODULE=OpenEye_FPGA_tb, TESTCASE=start_test_fpga).
#
# Notes:
# - Verilator has no UDP support: the PDK's sg13g2_udp.v (17 UDP primitives
#   used inside the standard-cell models) is NOT compiled; instead
#   sg13g2_primitives_vlt.v provides behavioral stand-ins with identical
#   port lists (semantics transcribed from the UDP tables).
# - The PDK's sg13g2_stdcell.v instantiates those primitives WITHOUT an
#   instance name (legal per LRM for gate-style instantiations, accepted by
#   iverilog, but Verilator only allows it for gate-level primitives).
#   The script therefore compiles a generated copy in which every
#   `\tihp_xxx (` line is rewritten to `\tihp_xxx u_prim_N (` (24 lines,
#   verified against the PDK file; the iverilog path uses the PDK original
#   and is unaffected).
# - Specify blocks are stripped from the stdcell models by
#   asic/scripts/pdk_specify_strip.py (same generated copy): Verilator
#   ignores specify contents (%Warning-SPECIFYIGN) and is 2-state, so the
#   57 `delayed_*` wires driven only by the timing blocks of the 16
#   FF/latch cells would fold to 0 and break every sequential cell. All
#   PDK specify delays are 0.0, so the script's exact transform
#   `assign delayed_X = X;` (plus leaving the unused `reg notifier` alone)
#   preserves the functional behavior. The io models need no strip
#   (their specify blocks carry no delayed_* data).
# - VPI exposure: the wrapper's 9 ports carry /*verilator public_on*/
#   marks (in OpenEye_Asic_gatesim_top.v) — REQUIRED. Without any public
#   variable Verilator 5.052 generates NO VPI var table at all and cocotb
#   init fatals with "Can not find root handle" (happened in build p2_10,
#   2026-10-25).
# - --flatten is REQUIRED (probe-verified, vpi_micro P15b, 2026-10-25):
#   it inlines every module into the single $root class, so no second
#   class mirrors the top's 9 ports. Verilator's $root class always holds
#   a mirror copy of the top module's ports, and when the marked wrapper
#   survives as its own class (WITHOUT --flatten), a ico_sequent copy
#   $root->wrapper runs on EVERY clock edge and clobbers cocotb's writes
#   to the wrapper scope (symptom: outputs read 0x0). The $root scope is
#   NOT addressable as cocotb root handle, so driving it is impossible —
#   the only working structure is a single flattened class in which the
#   top module's scope view binds its var table directly to $root
#   storage. COCOTB_TOPLEVEL stays OpenEye_Asic_gatesim_top; the root
#   handle then exposes the LIVE port variables. Combinational VPI writes
#   propagate without a clock edge; FF paths work via posedge+NBA
#   (P15b: both verified end-to-end at 1000-cell scale).
# - Scope-table size with --flatten: every inlined cell gets a candidate
#   VPI scope; it materializes only if the cell's module is modPublic,
#   because only public inlined cells keep per-instance hierarchical
#   variable names (top.inst.var) — non-public cells merge their vars into
#   top-level __Vcellout__ members with no scope path. Verilator 5.052
#   auto-marks a module public iff its definition is parsed AFTER the
#   --top-module definition (command-line file order, then in-file order;
#   empirically verified in 14 probe runs, vpi_micro P18c/P18h/P18i/P18m,
#   2026-10-25; the source setter was not located in the 5.052 tree —
#   see the P18 notes in asic/sim/gatesim/vpi_probes/). Therefore the cell
#   files are listed BEFORE the wrapper below. Reversed order made every
#   one of the ~2.1M netlist instances a public scope: 3.19M per-instance
#   scopes / ~360 GB of generated C++ (build 6, killed 2026-10-25).
#   Expected with the correct order: 2 scopes (TOP + wrapper) and ~9-20
#   var-table entries (the wrapper's public_on ports only).
# - Do NOT use --public / --public-flat-rw instead of the marks: they
#   expose every internal signal of every module and blew codegen up to
#   653 GB of C++ for this ~3M-signal flat netlist (build p2_10, killed
#   2026-10-24).
# - --timing is mandatory: cocotb drives the clock from Python (the VM
#   timing queue is required; VM_TIMING=1 comes from the flag, the design
#   is essentially delay-free — see the RM SRAM note above).
# - Build with -cc --exe (NOT --binary): cocotb's verilator.cpp provides
#   main(); --binary generates a second main and the link fails with
#   "duplicate symbol: main".
# - cocotb 2.1 environment: COCOTB_TEST_MODULES/COCOTB_TOPLEVEL/
#   COCOTB_TEST_FILTER + GPI_USERS (libpython + pygpi entry point) +
#   PYGPI_PYTHON_BIN + COCOTB_TRUST_INERTIAL_WRITES=1 (the old
#   MODULE/TOPLEVEL/TESTCASE names are not honored by the cocotb 2.1
#   Verilator runtime).
#
# Run inside the iic-osic-tools container:
#   cd /foss/designs/OpenEye/asic/sim/gatesim
#   asic/sim/gatesim/run_verilator.sh <path-to-OpenEye_Asic.nl.v>
set -euo pipefail

NETLIST="${1:?usage: run_verilator.sh <path-to-OpenEye_Asic.nl.v>}"
# resolve to an absolute path before changing directory
NETLIST="$(cd "$(dirname "${NETLIST}")" && pwd)/$(basename "${NETLIST}")"
[ -f "${NETLIST}" ] || { echo "netlist not found: ${NETLIST}" >&2; exit 1; }
cd "$(dirname "${BASH_SOURCE[0]}")"

# repo root (this script lives in <root>/asic/sim/gatesim)
REPO="$(cd ../../.. && pwd)"
PDK=/foss/pdks/ihp-sg13g2
SRAM_DIR="${PDK}/libs.ref/sg13g2_sram/verilog"
JOBS="${JOBS:-32}"
# keep make parallelism low: the generated Vtop translation units are huge
# (hundreds of MB to GB); -j 4 x ~40 GB RAM stays safe on a 243 GB host
MAKE_JOBS="${MAKE_JOBS:-4}"
SIM_BUILD="${SIM_BUILD:-sim_build_vlt}"

# cocotb build bits
COCOTB_LIB_DIR="$(python3 -m cocotb_tools.config --lib-dir)"
COCOTB_CPP="$(python3 -m cocotb_tools.config --share)/lib/verilator/verilator.cpp"
COCOTB_PYTHON_BIN="$(python3 -m cocotb_tools.config --python-bin)"
COCOTB_LIBPYTHON="$(python3 -m cocotb_tools.config --libpython)"
COCOTB_PYGPI_ENTRY="$(python3 -m cocotb_tools.config --pygpi-entry-point)"

echo "[INFO] netlist:  ${NETLIST}"
echo "[INFO] sim build: ${SIM_BUILD} (verilator -j ${JOBS}, make -j ${MAKE_JOBS})"
rm -rf "${SIM_BUILD}"
mkdir -p "${SIM_BUILD}"

# --- 1. generate C++ and Makefile -------------------------------------------
# PDK stdcell models need two transforms for Verilator (see header notes):
#   (a) unnamed primitive instances -> named copy (perl)
#   (b) specify blocks stripped + delayed_* aliases (pdk_specify_strip.py)
STDC_NAMED="${SIM_BUILD}/sg13g2_stdcell_vlt_named.v"
STDC_VLT="${SIM_BUILD}/sg13g2_stdcell_vlt.v"
perl -pe 'BEGIN { $n = 0 } $n++ if s/^([ \t]*ihp_)([a-z0-9_]+)([ \t]*)\(/"$1$2 u_prim_$n$3 ( "/e' \
  "${PDK}/libs.ref/sg13g2_stdcell/verilog/sg13g2_stdcell.v" > "${STDC_NAMED}"
[ "$(grep -cE '^[[:space:]]*ihp_[a-z0-9_]+[[:space:]]*\(' "${STDC_NAMED}")" -eq 0 ] || \
  { echo "generated stdcell file still has unnamed instances" >&2; exit 1; }
python3 "${REPO}/asic/scripts/pdk_specify_strip.py" "${STDC_NAMED}" "${STDC_VLT}"

# FILE ORDER IS FUNCTIONAL (see header, scope-table note): every module
# definition parsed BEFORE the --top-module definition stays non-public ->
# its inlined instances get no per-instance VPI scope. The wrapper (top)
# is therefore the LAST Verilog file; moving it earlier re-triggers the
# 3.19M-scope codegen explosion (build 6).
# (No '#' comment lines inside the backslash continuation below!)
verilator -cc --exe -j "${JOBS}" \
  --top-module OpenEye_Asic_gatesim_top \
  --timescale 1ns/10ps \
  --timing \
  --flatten \
  -Wno-fatal \
  -DFUNCTIONAL \
  -Mdir "${SIM_BUILD}" \
  --vpi --prefix Vtop -o Vtop \
  -LDFLAGS "-Wl,-rpath,${COCOTB_LIB_DIR} -L${COCOTB_LIB_DIR} -lcocotbvpi_verilator" \
  "${NETLIST}" \
  "${PWD}/sg13g2_primitives_vlt.v" \
  "${STDC_VLT}" \
  "${SRAM_DIR}/RM_IHPSG13_1P_core_behavioral_bm_bist.v" \
  "${SRAM_DIR}/RM_IHPSG13_1P_1024x64_c2_bm_bist.v" \
  "${SRAM_DIR}/RM_IHPSG13_1P_1024x16_c2_bm_bist.v" \
  "${SRAM_DIR}/RM_IHPSG13_1P_1024x8_c2_bm_bist.v" \
  "${SRAM_DIR}/RM_IHPSG13_1P_2048x64_c2_bm_bist.v" \
  "${SRAM_DIR}/RM_IHPSG13_1P_2048x32_c2_bm_bist.v" \
  "${SRAM_DIR}/RM_IHPSG13_1P_4096x16_c3_bm_bist.v" \
  "${SRAM_DIR}/RM_IHPSG13_1P_4096x8_c3_bm_bist.v" \
  "${PDK}/libs.ref/sg13g2_io/verilog/sg13g2_io.v" \
  "${PWD}/OpenEye_Asic_gatesim_top.v" \
  "${COCOTB_CPP}"

# sanity: codegen must not be exploding (the --public-flat-rw blow-up wrote
# 700 GB); warn loudly above 80 GB.
# NOTE: GNU du has no -g (BSD-only) — with set -e a failed du -sg aborts the
# whole script (this actually happened: codegen had finished, but make never
# started; 2026-10-25). Use -sm and guard against empty output.
GEN_MB="$(du -sm "${SIM_BUILD}" 2>/dev/null | cut -f1)"
if [ -n "${GEN_MB}" ]; then
  echo "[INFO] generated size: ${GEN_MB} MB"
  if [ "${GEN_MB}" -gt 80000 ]; then
    echo "WARNING: generated C++ is ${GEN_MB} MB — something is off, check flags" >&2
  fi
else
  echo "WARNING: could not measure generated size (du failed)" >&2
fi

# --- 2. compile and link -----------------------------------------------------
make -C "${SIM_BUILD}" -f Vtop.mk -j "${MAKE_JOBS}"

# --- 3. run the functional test (same environment as the iverilog Makefile) --
cd "${SIM_BUILD}"

# Functional test (same as test/cocotb_fpga defaults)
export LAYER=Convolution_Single
export NUM_FILTERS=16
export KERNEL_SIZE_X=3
export KERNEL_SIZE_Y=3
export INPUT_SIZE_X=16
export INPUT_SIZE_Y=1
export STRIDE=1
export INPUT_CHANNELS=4
export OUTPUT_SIZE=64
export USE_RANDOM_VALUES=1
export ONLY_FILES=0

# ASIC target configuration (must match the parameters baked into the netlist)
export CLUSTER_COLUMNS=2
export CLUSTER_ROWS=2
export NUM_GLB_IACT=1
export NUM_GLB_PSUM=4
export NUM_GLB_WGHT=3
export IACT_RAM_CELLS=4
export BUFFER_WIDTH=10
export BUFFER_WIDTH_WGHT=11
export BUFFER_WIDTH_PSUM=12
export BRANCHES=2
export QUANT_AMOUNT=32
export TRANS_WORDS=8
export STRIDE_ENABLED=0
export DMA_BITWIDTH=64
export PARALLEL_MACS=2
export SPARSITY_EN=1
export DATA_PSUM_BITWIDTH=20
export IACT_SC_ADDRWIDTH=10
export TRANS_BITWIDTH_IACT=24
export TRANS_BITWIDTH_WGHT=24
export TRANS_BITWIDTH_PSUM=20
export USE_SPARSE_IACTS=1
export USE_SPARSE_WEIGHTS=1

export CLOCK_LEN=10
export CLOCK_UNIT=ns
export CLOCK_DELAY_INPUT=100
export CLOCK_DELAY_UNIT_INPUT=ps
export CLOCK_DELAY_OUTPUT=100
export CLOCK_DELAY_UNIT_OUTPUT=ps
export LOGGER_LEVEL=20

# cocotb 2.1 runtime environment
export TOPLEVEL_LANG=verilog
export COCOTB_TOPLEVEL=OpenEye_Asic_gatesim_top
export COCOTB_TEST_MODULES=OpenEye_FPGA_tb
export COCOTB_TEST_FILTER=start_test_fpga
export COCOTB_TRUST_INERTIAL_WRITES=1
export PYGPI_PYTHON_BIN="${COCOTB_PYTHON_BIN}"
export GPI_USERS="${COCOTB_LIBPYTHON};${COCOTB_PYGPI_ENTRY}"
export PYTHONPATH="${REPO}/src:${REPO}/test/cocotb_fpga${PYTHONPATH:+:${PYTHONPATH}}"

./Vtop
