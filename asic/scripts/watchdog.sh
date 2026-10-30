#!/bin/bash
# Watchdog for the OpenEye ASIC flow: restarts the run automatically if it
# crashes with an unexpected error (OOM/SIGKILL/traceback), up to
# MAX_RESTARTS times. Design-level checker failures after all steps ran
# are NOT auto-restarted (they need a look, not a rerun).
#
# IMPORTANT (verified in librelane source 2026-10-29): a restart WITHOUT
# -F re-runs the whole flow from step 1 in new numbered directories. So
# every launch resolves the resume step first (asic/scripts/
# resolve_resume.py): a crashed/running step -> that step; otherwise the
# first step after the frontier of completed steps that is not gated by
# the design config; fresh run -> nothing (full run).
#
# Usage (detached):
#   setsid nohup asic/scripts/watchdog.sh p2_7 > asic/runs/watchdog_p2_7.log 2>&1 < /dev/null &
set -uo pipefail

RUN_NAME="${1:-p2_7}"
MAX_RESTARTS=3
CONTAINER_NAME="${CONTAINER_NAME:-iic-osic-tools_asic}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
LAUNCHER_LOG="asic/runs/openeye_${RUN_NAME}_launcher.log"
FLOW_LOG="asic/runs/openeye_${RUN_NAME}/flow.log"
RUN_DIR_CTR="/foss/designs/OpenEye/asic/runs/openeye_${RUN_NAME}/design/runs/${RUN_NAME}"

# Resolve the step to resume at. Runs inside the container (needs the
# librelane package for the Chip flow definition). The logic lives in
# asic/scripts/resolve_resume.py (standalone-testable, see its header):
#   - crashed/running step (latest dir without state_out.json) -> that step
#   - otherwise the first step after the frontier of completed steps that
#     is not gated by the design config (gated steps never create a dir;
#     the old "no dir -> resume here" rule fired on them and would have
#     re-run 7 h+ of antenna work after a p2_15 crash)
#   - fresh run -> nothing (full run)
# Prints the uniquified step id (e.g. "OpenROAD.GlobalRouting",
# "OpenROAD.STAMidPNR-3") or nothing.
resolve_resume_step() {
    docker exec "${CONTAINER_NAME}" python3 \
        "/foss/designs/OpenEye/asic/scripts/resolve_resume.py" \
        "${RUN_DIR_CTR}" 2>/dev/null | grep -vE '^\[INFO' | tail -1
}

for ((n = 0; n <= MAX_RESTARTS; n++)); do
    if [ "$n" -gt 0 ]; then
        echo "[watchdog] restart ${n}/${MAX_RESTARTS} at $(date -Is)"
        sleep 60
    fi
    resume_step="$(resolve_resume_step)"
    if [ -n "$resume_step" ]; then
        echo "[watchdog] resuming at step: ${resume_step}"
    fi
    rm -f "${LAUNCHER_LOG}"
    asic/scripts/run_openeye.sh "${RUN_NAME}" ${resume_step:+"${resume_step}"} >> "${LAUNCHER_LOG}" 2>&1

    if grep -q "finished without ERROR lines" "${LAUNCHER_LOG}" 2>/dev/null; then
        echo "[watchdog] flow ${RUN_NAME} completed $(date -Is)"
        exit 0
    fi
    # unexpected crash (e.g. OOM/SIGKILL) vs. normal step failure:
    if grep -q "unexpected" "${FLOW_LOG}" 2>/dev/null; then
        echo "[watchdog] unexpected error in flow ${RUN_NAME} (see ${FLOW_LOG})"
        continue
    fi
    echo "[watchdog] flow ${RUN_NAME} failed with step/checker errors -- not auto-restarting"
    exit 1
done
echo "[watchdog] max restarts reached for ${RUN_NAME}"
exit 1
