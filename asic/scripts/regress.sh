#!/bin/bash
# ASIC-target cocotb regression (single-layer conv test through OpenEye_FPGA).
#
# IMPORTANT: all target values must be passed as MAKE command-line variables.
# Passing them as environment (e.g. via `source target.sh`) does NOT work:
# test/cocotb_fpga/Makefile contains `export VAR=default` lines, and makefile
# exports override inherited environment for recipe subshells, producing a
# Frankenstein configuration (observed: X on data_dma_o).
#
# Run inside the iic-osic-tools container:
#   docker exec iic-osic-tools_asic bash -lc "cd /foss/designs/OpenEye && asic/scripts/regress.sh"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO="$(pwd)"

# shellcheck source=asic/target.sh
source asic/target.sh

# Make command-line variable string from target.sh (only vars make needs;
# TRANS_BITWIDTH_* derived from SPARSITY_EN/PARALLEL_MACS/DATA_PSUM_BITWIDTH).
if [ "${SPARSITY_EN}" = "1" ]; then
  TBI=24 TWB=24   # 2x12 iact, 12x2 wght (PARALLEL_MACS=2)
else
  TBI=12 TWB=12
fi
TPB="${DATA_PSUM_BITWIDTH}"  # SERIAL=1 top: no PARALLEL_MACS multiplier

VARS="
  CLUSTER_COLUMNS=${CLUSTER_COLUMNS} CLUSTER_ROWS=${CLUSTER_ROWS}
  NUM_GLB_IACT=${NUM_GLB_IACT} NUM_GLB_PSUM=${NUM_GLB_PSUM} NUM_GLB_WGHT=${NUM_GLB_WGHT}
  IACT_RAM_CELLS=${IACT_RAM_CELLS}
  BUFFER_WIDTH=${BUFFER_WIDTH} BUFFER_WIDTH_WGHT=${BUFFER_WIDTH_WGHT} BUFFER_WIDTH_PSUM=${BUFFER_WIDTH_PSUM}
  BRANCHES=${BRANCHES} QUANT_AMOUNT=${QUANT_AMOUNT} TRANS_WORDS=${TRANS_WORDS}
  STRIDE_ENABLED=${STRIDE_ENABLED} DMA_BITWIDTH=${DMA_BITWIDTH}
  PARALLEL_MACS=${PARALLEL_MACS} SPARSITY_EN=${SPARSITY_EN} DATA_PSUM_BITWIDTH=${DATA_PSUM_BITWIDTH}
  IACT_SC_ADDRWIDTH=${IACT_SC_ADDRWIDTH}
  TRANS_BITWIDTH_IACT=${TBI} TRANS_BITWIDTH_WGHT=${TWB} TRANS_BITWIDTH_PSUM=${TPB}
  USE_SPARSE_IACTS=${SPARSITY_EN} USE_SPARSE_WEIGHTS=${SPARSITY_EN}
"

cd "${REPO}/test/cocotb_fpga"
rm -rf sim_build
# shellcheck disable=SC2086
make run ${VARS} PYTHONPATH="${REPO}/src"
echo "[OK] ASIC-target regression passed"
