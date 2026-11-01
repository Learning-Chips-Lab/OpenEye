#!/bin/bash
# Run the OpenEye ASIC flow (LibreLane Chip) on ihp-sg13g2.
#
# Stages the ASIC-target generated RTL into hdl/ (include resolution),
# then runs the full Chip flow: synthesis -> P&R (SRAM macros + padring +
# bondpads) -> fill -> multi-corner STA -> DRC -> LVS -> GDS.
#
# Usage (from the host, container must be started via scripts/container.sh):
#   asic/scripts/run_openeye.sh [runname] [resume-step-id] [stop-step-id]
#
# Stop-step semantics (2026-11-01, plan v2 F2): the third argument is passed
# to librelane as -T <stop-step-id>: the flow runs at most up to and
# INCLUDING that step, then exits 0. Success for a stopped run is NOT the
# final/ snapshot (librelane writes it only after ALL steps completed) but
# the stop step's latest run dir carrying state_out.json. The success marker
# line is still printed, so watchdog.sh exits 0 without restarting.
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
#   On resume the RTL is NOT regenerated (gen.sh / gen_top.py are skipped)
#   -- the netlist must not change mid-run; only gen_config.py runs (config
#   knobs may change between relaunches of the same run). The git state
#   recorded at the fresh start is compared on resume and a [WARN] is
#   printed if hdl/ or asic/rtl/ diverge.
#   Caveat for forked runs (run dir copied from another tag): rewrite the
#   absolute paths inside all state_*.json AND restore the original file
#   mtimes afterwards (any tool that rewrites them, e.g. sed -i, resets
#   mtime to "now" and breaks the latest-state selection).
#
# Logging: flow.log is APPENDED (a watchdog crash-restart must not destroy
# the crash evidence) and each launch writes a
#   === launch <date> resume=<step|<full run>> ===
# separator first.
#
# Success = librelane exit code 0 AND runs/<tag>/final/ present (librelane
# saves the final state snapshot only after ALL steps completed; a
# checker failure raises before that). No more grepping of a 500 MB log.
# The success marker line is kept verbatim for watchdog.sh's grep.
#
# DRY_RUN=1 stops after staging (before the flow launch) -- for testing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASIC_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_DIR="$(dirname "${ASIC_DIR}")"
CONTAINER_NAME="${CONTAINER_NAME:-iic-osic-tools_asic}"

RUN_NAME="${1:-$(date +%Y%m%d_%H%M)}"
RESUME_STEP="${2:-}"
STOP_STEP="${3:-}"
WORKDIR_HOST="${ASIC_DIR}/runs/openeye_${RUN_NAME}"
WORKDIR_CTR="/foss/designs/OpenEye/asic/runs/openeye_${RUN_NAME}"
FLOW_LOG="${WORKDIR_HOST}/flow.log"
GIT_STATE="${WORKDIR_HOST}/design/launch_git_state.txt"

git_state_now() {
    {
        echo "head: $(git -C "${REPO_DIR}" rev-parse HEAD)"
        echo "status: $(git -C "${REPO_DIR}" status --porcelain hdl asic/rtl | tr '\n' '|')"
    }
}

# Regenerate the ASIC-target RTL and stage it (gen.sh needs the container
# python deps; gen_top/gen_config are stdlib-only and run on the host).
mkdir -p "${WORKDIR_HOST}/design"
if [ -n "${RESUME_STEP}" ]; then
    # RESUME: skip gen.sh + gen_top.py (RTL must not change mid-run);
    # only the config is regenerated (knobs may change for the relaunch).
    (cd "${REPO_DIR}" && python3 asic/scripts/gen_config.py)
    if [ -f "${GIT_STATE}" ]; then
        if ! diff -q <(cat "${GIT_STATE}") <(git_state_now) >/dev/null 2>&1; then
            echo "[WARN] Run '${RUN_NAME}' resumes, but the RTL state differs"
            echo "[WARN] from the fresh start (recorded in ${GIT_STATE}):"
            diff <(cat "${GIT_STATE}") <(git_state_now) || true
        fi
    fi
else
    docker exec "${CONTAINER_NAME}" bash -lc \
      "cd /foss/designs/OpenEye && PYTHONPATH=src bash asic/scripts/gen.sh"
    (cd "${REPO_DIR}" && python3 asic/scripts/gen_top.py)
    (cd "${REPO_DIR}" && python3 asic/scripts/gen_config.py)
    git_state_now > "${GIT_STATE}"
fi

cp "${ASIC_DIR}/rtl/sg13g2/openeye.json" "${WORKDIR_HOST}/design/"

echo "[INFO] OpenEye ASIC run '${RUN_NAME}': ${WORKDIR_CTR}"
echo "[INFO] Log: ${FLOW_LOG} (appended, separator per launch)"

# Separator first: crash evidence from the previous launch must survive a
# watchdog restart in the log.
echo "=== launch $(date -Is) resume=${RESUME_STEP:-<full run>} ===" >> "${FLOW_LOG}"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY_RUN] staging done; would launch in ${CONTAINER_NAME}:"
    echo "[DRY_RUN]   librelane openeye.json --run-tag '${RUN_NAME}' ${RESUME_STEP:+"-F '${RESUME_STEP}'"} ${STOP_STEP:+"-T '${STOP_STEP}'"}"
    exit 0
fi

# NO --overwrite: with it, LibreLane does shutil.rmtree() on the existing
# run dir (p2_12 incident 2026-10-27: a capped-resizer resume wiped 30 h of
# step outputs).
# A restart without -F re-runs ALL steps from step 1 in new directories
# (verified in librelane source 2026-10-29) -- so every restart (including
# watchdog crash-restarts) MUST pass the resume step, which watchdog.sh
# derives from the run dir via asic/scripts/resolve_resume.py.
rc=0
docker exec "${CONTAINER_NAME}" bash -lc "
  cd '${WORKDIR_CTR}/design' || exit 1
  source '${WORKDIR_CTR%/runs/openeye_${RUN_NAME}}/scripts/env.sh'
  librelane openeye.json --run-tag '${RUN_NAME}' ${RESUME_STEP:+-F '${RESUME_STEP}'} ${STOP_STEP:+-T '${STOP_STEP}'}
" >> "${FLOW_LOG}" 2>&1 || rc=$?

RUNS_DIR="${WORKDIR_HOST}/design/runs/${RUN_NAME}"

if [ -n "${STOP_STEP}" ]; then
    # Stopped run: final/ is NOT written (librelane saves it only after ALL
    # steps completed) -> success = exit 0 AND the stop step's latest dir
    # carries state_out.json (several dirs per step happen after crash
    # re-runs; the highest ordinal decides, as in resolve_resume.py).
    slug="$(docker exec "${CONTAINER_NAME}" python3 -c \
        "from librelane.common.misc import slugify; print(slugify('${STOP_STEP}'))")"
    # "|| true" inside: without a matching dir, ls's failure would kill
    # the script (set -e + pipefail) before the error branch below.
    stop_dir="$( { ls -d "${RUNS_DIR}"/[0-9]*-"${slug}" 2>/dev/null || true; } | sort | tail -1 )"
    stop_ok=$([ -n "${stop_dir}" ] && [ -f "${stop_dir}/state_out.json" ] && echo yes || echo no)
    if [ "${rc}" -ne 0 ] || [ "${stop_ok}" != "yes" ]; then
        echo "[ERROR] Run '${RUN_NAME}' failed: librelane exit ${rc}; stop-step '${STOP_STEP}' dir: ${stop_dir:-<none>}; state_out.json: ${stop_ok}. Check ${FLOW_LOG}" >&2
        exit 1
    fi
    echo "[INFO] Run '${RUN_NAME}' finished without ERROR lines (stopped at ${STOP_STEP})."
    exit 0
fi

# Full run: success = librelane exit code 0 AND the final/ state snapshot
# present (sequential.py saves it only after every step completed; checker
# failures raise before it).
FINAL_DIR="${RUNS_DIR}/final"
if [ "${rc}" -ne 0 ] || [ ! -d "${FINAL_DIR}" ]; then
    echo "[ERROR] Run '${RUN_NAME}' failed: librelane exit ${rc}; final/ present: $([ -d "${FINAL_DIR}" ] && echo yes || echo no). Check ${FLOW_LOG}" >&2
    exit 1
fi
echo "[INFO] Run '${RUN_NAME}' finished without ERROR lines."
