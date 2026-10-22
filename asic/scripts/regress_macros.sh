#!/bin/bash
# ASIC-target regression WITH sg13g2 SRAM macros (OPENEYE_RAM_SPECIFIC).
#
# Runs the same single-layer cocotb test as regress.sh, but compiles the
# PDK functional SRAM macro models + asic/rtl/sg13g2/sg13g2_ram_sp.v so the
# wrapper (tiling, mask expansion, read latency) is verified end-to-end.
#
# It drives cocotb's Makefile.sim directly instead of test/cocotb_fpga/Makefile
# (whose `VERILOG_SOURCES +=` lines would conflict with command-line values).
# The test environment (LAYER, test vectors, ...) is imported from the
# export lines of that Makefile.
#
# Run inside the iic-osic-tools container:
#   docker exec iic-osic-tools_asic bash -lc "cd /foss/designs/OpenEye && asic/scripts/regress_macros.sh"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO="$(pwd)"

PDK_SRAM_VERILOG=/foss/pdks/ihp-sg13g2/libs.ref/sg13g2_sram/verilog
MACRO_MODELS=(
  RM_IHPSG13_1P_core_behavioral_bm_bist.v
  RM_IHPSG13_1P_1024x64_c2_bm_bist.v
  RM_IHPSG13_1P_1024x8_c2_bm_bist.v
  RM_IHPSG13_1P_2048x64_c2_bm_bist.v
  RM_IHPSG13_1P_2048x32_c2_bm_bist.v
  RM_IHPSG13_1P_4096x16_c3_bm_bist.v
  RM_IHPSG13_1P_4096x8_c3_bm_bist.v
)

# --- test environment from the FPGA test Makefile (export lines only) -------
# normalize: "export X = 1" -> "export X=1" (shell export needs no spaces),
# drop bare "export" directive lines
awk '/^export /{
       line=$0; sub(/^export[ \t]+/,"",line);
       if (line ~ /=/) { gsub(/[ \t]*=[ \t]*/,"=",line); print "export " line }
     }' "${REPO}/test/cocotb_fpga/Makefile" > /tmp/asic_testenv.sh
# shellcheck disable=SC1091
source /tmp/asic_testenv.sh
unset VH_PATH HDL_PATH 2>/dev/null || true

# --- ASIC target overrides (must beat the Makefile defaults) ----------------
# shellcheck source=asic/target.sh
source asic/target.sh
if [ "${SPARSITY_EN}" = "1" ]; then
  export TRANS_BITWIDTH_IACT=24 TRANS_BITWIDTH_WGHT=24
else
  export TRANS_BITWIDTH_IACT=12 TRANS_BITWIDTH_WGHT=12
fi
export TRANS_BITWIDTH_PSUM="${DATA_PSUM_BITWIDTH}"
export USE_SPARSE_IACTS="${SPARSITY_EN}" USE_SPARSE_WEIGHTS="${SPARSITY_EN}"
export TOPLEVEL=OpenEye_FPGA
export MODULE=OpenEye_FPGA_tb
export TESTCASE=start_test_fpga
export SIM=icarus
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

# --- regenerate (generator + vh files for the target) -----------------------
python -m open_eye.generator "${REPO}/test/cocotb_fpga" "${REPO}/hdl/include" "${REPO}/hdl" "${REPO}/asic/rtl/generated"
export VH_PATH="${REPO}/test/cocotb_fpga"
python -m open_eye.vh_file_creator

# --- source list: accelerator + wrapper + macro models ----------------------
VSRC=()
for f in OpenEye_FPGA psum_pipeline dma_storage varlenFIFO OpenEye_Parallel \
         OpenEye_Cluster GLB_cluster router_iact router_wght router_psum \
         PE_cluster PE dsp_unit adder adder_tree delay_cluster data_pipeline \
         data_pipeline_iact data_pipeline_wght multiplier mux2 demux2 mux_iact \
         SPad_DP SPad_DP_RW SPad_SP quantization_unit RST_SYNC RAM_DP RAM_DP_RW \
         RAM_SP RAM_DP_generic RAM_DP_RW_generic ring_buffer_pipelined \
         RAM_SP_generic iact_stream_constructor router_configurator; do
  VSRC+=("${REPO}/hdl/${f}.v")
done
VSRC+=("${REPO}/asic/rtl/sg13g2/sg13g2_ram_sp.v")
for m in "${MACRO_MODELS[@]}"; do
  VSRC+=("${PDK_SRAM_VERILOG}/${m}")
done

CA="-DUSE_INTERNAL_PARAMS_PE -DUSE_INTERNAL_PARAMS_PE_cluster \
    -DOPENEYE_RAM_SPECIFIC -DFUNCTIONAL \
    -I${REPO}/hdl/include -I${REPO}/test/cocotb_fpga -Wall"

cd "${REPO}/test/cocotb_fpga"
rm -rf sim_build
make -f "$(cocotb-config --makefiles)/Makefile.sim" \
  VERILOG_SOURCES="${VSRC[*]}" \
  COMPILE_ARGS="${CA}"
echo "[OK] ASIC-target regression with sg13g2 SRAM macros passed"
