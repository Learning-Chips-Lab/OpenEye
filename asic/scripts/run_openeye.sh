#!/bin/bash
# Run the OpenEye ASIC flow (LibreLane Chip) on ihp-sg13g2.
#
# Stages the ASIC-target generated RTL into hdl/ (include resolution),
# then runs the full Chip flow: synthesis -> P&R (SRAM macros + padring +
# bondpads) -> fill -> multi-corner STA -> DRC -> LVS -> GDS.
#
# Usage (from the host, container must be started via scripts/container.sh):
#   asic/scripts/run_openeye.sh [runname] [resume-step-id]
#
# Output lands in asic/runs/openeye_<runname>/ (gitignored).
#
# Resume semantics (verified in librelane source 2026-10-29, p2_15):
#   --run-tag alone does NOT skip finished steps -- it re-runs the whole flow
#   from step 1 in NEW numbered directories (39, 40, ...), using only the
#   latest state_out.json as initial state. To resume at a specific step,
#   pass its id as second argument -> librelane -F <id>:
#     - all flow steps before <id> are skipped (no re-run, no new dirs),
#     - initial state = latest state_out.json in the run dir (by mtime),
#     - <id> and later steps run in the next free directory numbers.
#   Caveat for forked runs (run dir copied from another tag): rewrite the
#   absolute paths inside all state_*.json AND restore the original file
#   mtimes afterwards (any tool that rewrites them, e.g. sed -i, resets
#   mtime to "now" and breaks the latest-state selection).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASIC_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_DIR="$(dirname "${ASIC_DIR}")"
CONTAINER_NAME="${CONTAINER_NAME:-iic-osic-tools_asic}"

RUN_NAME="${1:-$(date +%Y%m%d_%H%M)}"
RESUME_STEP="${2:-}"
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
# step outputs).
# A restart without -F re-runs ALL steps from step 1 in new directories
# (verified in librelane source 2026-10-29) -- so every restart (including
# watchdog crash-restarts) MUST pass the resume step, which watchdog.sh
# derives from the run dir (first flow step without a complete step dir).
docker exec "${CONTAINER_NAME}" bash -lc "
  cd '${WORKDIR_CTR}/design' || exit 1
  source '${WORKDIR_CTR%/runs/openeye_${RUN_NAME}}/scripts/env.sh'
  librelane openeye.json --run-tag '${RUN_NAME}' ${RESUME_STEP:+-F '${RESUME_STEP}'} 2>&1 | tee '${WORKDIR_CTR}/flow.log'
"

if grep -qE "ERROR|Traceback" "${WORKDIR_HOST}/flow.log"; then
  echo "[ERROR] Run '${RUN_NAME}' reported errors. Check ${WORKDIR_HOST}/flow.log" >&2
  exit 1
fi
echo "[INFO] Run '${RUN_NAME}' finished without ERROR lines."
