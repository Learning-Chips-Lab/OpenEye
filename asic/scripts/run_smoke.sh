#!/bin/bash
# Run the counter smoke test through the full LibreLane flow on ihp-sg13g2
# (synthesis -> P&R -> fill -> multi-corner STA -> DRC -> LVS -> GDS).
#
# Validates the toolchain/PDK before running the OpenEye design.
#
# Usage (from the host, container must be started via scripts/container.sh):
#   asic/scripts/run_smoke.sh [runname]
#
# Output lands in asic/runs/smoke_sg13g2_<runname>/ (gitignored).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASIC_DIR="$(dirname "${SCRIPT_DIR}")"
CONTAINER_NAME="${CONTAINER_NAME:-iic-osic-tools_asic}"

RUN_NAME="${1:-$(date +%Y%m%d_%H%M)}"
WORKDIR_HOST="${ASIC_DIR}/runs/smoke_sg13g2_${RUN_NAME}"
WORKDIR_CTR="/foss/designs/OpenEye/asic/runs/smoke_sg13g2_${RUN_NAME}"

mkdir -p "${WORKDIR_HOST}/design"
cp -r "${ASIC_DIR}/smoke/counter/." "${WORKDIR_HOST}/design/"

echo "[INFO] Smoke run '${RUN_NAME}': ${WORKDIR_CTR}"
echo "[INFO] Log: ${WORKDIR_HOST}/smoke.log"

docker exec "${CONTAINER_NAME}" bash -lc "
  cd '${WORKDIR_CTR}/design' || exit 1
  source '${WORKDIR_CTR%/runs/smoke_sg13g2_${RUN_NAME}}/scripts/env.sh'
  librelane counter.json --overwrite --run-tag '${RUN_NAME}' 2>&1 | tee '${WORKDIR_CTR}/smoke.log'
"

if grep -q "ERROR" "${WORKDIR_HOST}/smoke.log"; then
  echo "[ERROR] Smoke run '${RUN_NAME}' reported errors. Check ${WORKDIR_HOST}/smoke.log" >&2
  exit 1
fi
echo "[INFO] Smoke run '${RUN_NAME}' finished without ERROR lines."
