#!/bin/bash
# Run the OpenEye ASIC flow (LibreLane Chip) on ihp-sg13g2.
#
# Stages the ASIC-target generated RTL into hdl/ (include resolution),
# then runs the full Chip flow: synthesis -> P&R (SRAM macros + padring +
# bondpads) -> fill -> multi-corner STA -> DRC -> LVS -> GDS.
#
# Usage (from the host, container must be started via scripts/container.sh):
#   asic/scripts/run_openeye.sh [runname]
#
# Output lands in asic/runs/openeye_<runname>/ (gitignored).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASIC_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_DIR="$(dirname "${ASIC_DIR}")"
CONTAINER_NAME="${CONTAINER_NAME:-iic-osic-tools_asic}"

RUN_NAME="${1:-$(date +%Y%m%d_%H%M)}"
WORKDIR_HOST="${ASIC_DIR}/runs/openeye_${RUN_NAME}"
WORKDIR_CTR="/foss/designs/OpenEye/asic/runs/openeye_${RUN_NAME}"

# Regenerate the ASIC-target RTL and stage it (gen.sh needs the container
# python deps; gen_top/gen_config are stdlib-only and run on the host).
docker exec "${CONTAINER_NAME}" bash -lc \
  "cd /foss/designs/OpenEye && PYTHONPATH=src bash asic/scripts/gen.sh"
(cd "${REPO_DIR}" && python3 asic/scripts/gen_top.py)
(cd "${REPO_DIR}" && python3 asic/scripts/gen_config.py)

mkdir -p "${WORKDIR_HOST}/design"
cp "${ASIC_DIR}/rtl/sg13g2/openeye.json" "${WORKDIR_HOST}/design/"

echo "[INFO] OpenEye ASIC run '${RUN_NAME}': ${WORKDIR_CTR}"
echo "[INFO] Log: ${WORKDIR_HOST}/flow.log"

# NO --overwrite: with it, LibreLane does shutil.rmtree() on the existing
# run dir (p2_12 incident 2026-10-27: a capped-resizer resume wiped 30 h of
# step outputs). Without it, an existing run resumes in place from the
# latest state_out.json / finished steps -- which is also exactly what a
# watchdog crash-restart must do.
docker exec "${CONTAINER_NAME}" bash -lc "
  cd '${WORKDIR_CTR}/design' || exit 1
  source '${WORKDIR_CTR%/runs/openeye_${RUN_NAME}}/scripts/env.sh'
  librelane openeye.json --run-tag '${RUN_NAME}' 2>&1 | tee '${WORKDIR_CTR}/flow.log'
"

if grep -qE "ERROR|Traceback" "${WORKDIR_HOST}/flow.log"; then
  echo "[ERROR] Run '${RUN_NAME}' reported errors. Check ${WORKDIR_HOST}/flow.log" >&2
  exit 1
fi
echo "[INFO] Run '${RUN_NAME}' finished without ERROR lines."
