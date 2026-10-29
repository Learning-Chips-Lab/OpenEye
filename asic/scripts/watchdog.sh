#!/bin/bash
# Watchdog for the OpenEye ASIC flow: restarts the run automatically if it
# crashes with an unexpected error (OOM/SIGKILL/traceback), up to
# MAX_RESTARTS times. Design-level checker failures after all steps ran
# are NOT auto-restarted (they need a look, not a rerun).
#
# IMPORTANT (verified in librelane source 2026-10-29): a restart WITHOUT
# -F re-runs the whole flow from step 1 in new numbered directories. So
# every launch resolves the resume step first: the first flow step (in
# flow order) that has no complete step dir (no state_out.json) in the
# run dir. For a fresh run that is flow step 1 (no behavior change);
# for a crashed run it is the crashed step -> -F <id> skips all finished
# steps and reuses the latest state_out.json.
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

# Resolve the step to resume at: first flow step without a complete step
# dir. Runs inside the container (needs the librelane package). Prints the
# step id (e.g. "OpenROAD.GlobalRouting") or nothing on failure.
resolve_resume_step() {
    docker exec "${CONTAINER_NAME}" python3 -c "
import os, sys
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')
from librelane.flows.chip import Chip
from librelane.common.misc import slugify
run = '${RUN_DIR_CTR}'
if not os.path.isdir(run):
    sys.exit(0)  # fresh run -> no resume info needed, -F <step 1> == full run
existing = os.listdir(run)
dirs = [d for d in existing
        if '-' in d and d.split('-', 1)[0].isdigit()]
for step_cls in Chip.Steps:
    sid = step_cls.get_implementation_id()
    matching = [d for d in dirs if d.split('-', 1)[1] == slugify(sid)]
    if not matching:
        print(sid)  # step has never run -> resume here
        break
    # step complete iff its LATEST dir (highest ordinal) has a state_out
    # -- or holds ONLY config.json + state_in.json (output-less steps like
    # Checker.PowerGridViolations write no state_out at all, and a dir with
    # log/COMMANDS artifacts but no state_out means the step crashed)
    latest = max(matching, key=lambda d: int(d.split('-', 1)[0]))
    latest_path = os.path.join(run, latest)
    complete = os.path.isfile(os.path.join(latest_path, 'state_out.json')) or (
        set(os.listdir(latest_path)) <= {'config.json', 'state_in.json'}
    )
    if not complete:
        print(sid)  # last execution of this step did not finish -> resume here
        break
" 2>/dev/null | grep -vE '^\[INFO' | tail -1
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
