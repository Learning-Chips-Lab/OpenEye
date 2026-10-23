#!/bin/bash
# Watchdog for the OpenEye ASIC flow: restarts the run automatically if it
# crashes with an unexpected error (OOM/SIGKILL/traceback), up to
# MAX_RESTARTS times. Design-level checker failures after all steps ran
# are NOT auto-restarted (they need a look, not a rerun).
#
# Usage (detached):
#   setsid nohup asic/scripts/watchdog.sh p2_7 > asic/runs/watchdog_p2_7.log 2>&1 < /dev/null &
set -uo pipefail

RUN_NAME="${1:-p2_7}"
MAX_RESTARTS=3
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
LAUNCHER_LOG="asic/runs/openeye_${RUN_NAME}_launcher.log"
FLOW_LOG="asic/runs/openeye_${RUN_NAME}/flow.log"

for ((n = 0; n <= MAX_RESTARTS; n++)); do
    if [ "$n" -gt 0 ]; then
        echo "[watchdog] restart ${n}/${MAX_RESTARTS} at $(date -Is)"
        sleep 60
    fi
    rm -f "${LAUNCHER_LOG}"
    asic/scripts/run_openeye.sh "${RUN_NAME}" >> "${LAUNCHER_LOG}" 2>&1

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
