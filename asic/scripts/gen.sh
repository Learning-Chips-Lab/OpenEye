#!/bin/bash
# Generate the ASIC-target RTL: dma_storage.v, regmap_params.vh,
# parameters_FPGA.vh / parameters_PE_cluster.vh / parameters_PE.vh.
#
# Must run inside the iic-osic-tools container (python deps) with the repo
# at /foss/designs/OpenEye:
#   docker exec iic-osic-tools_asic bash -lc "cd /foss/designs/OpenEye && asic/scripts/gen.sh"
#
# Output: asic/rtl/generated/ (gitignored).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO="$(pwd)"

# shellcheck source=asic/target.sh
source asic/target.sh

OUT="${REPO}/asic/rtl/generated"
mkdir -p "${OUT}"

python -m open_eye.generator \
    "${REPO}/test/cocotb_fpga" \
    "${OUT}" \
    "${OUT}" \
    "${OUT}"

export VH_PATH="${OUT}"
export HDL_PATH="${REPO}/hdl"
export TOPLEVEL=OpenEye_FPGA
python -m open_eye.vh_file_creator

echo "[OK] Generated files:"
ls -la "${OUT}"
