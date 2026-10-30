#!/bin/bash
# Progress monitor for an OpenEye ASIC flow run (host side, tracked copy
# of the per-run asic/runs/monitor_<run>.sh instances).
#
# Usage (detached, from the host):
#   setsid nohup bash asic/scripts/monitor_flow.sh p2_15 >> asic/runs/monitor_p2_15.log 2>&1 < /dev/null &
#
# Appends a snapshot to asic/runs/monitor_<run>.log every MONITOR_INTERVAL
# seconds and flags:
#   STALL   - flow.log AND step files AND tool CPU time all flat > STALL_S.
#             The CPU-time liveness check kills the false STALLs caused by
#             buffered output: a 100 %-CPU single-threaded openroad that
#             only flushes its log occasionally looked "stalled" on log
#             size alone (p2_15 RepairAntennas, 2026-10-30).
#   RESTART - watchdog restart lines increased (pattern "restart N/3")
#   WD-EXITED - watchdog exited (completed or not auto-restarting)
#   DEAD    - librelane process gone while flow incomplete
#   DONE    - flow reported finished (monitor exits)
#
# The container shares the host PID namespace here, so the current
# step's tool (openroad/yosys/klayout/magic) is visible in host ps.
set -u
cd /home/utz/projekte/OpenEye

RUN="${1:?usage: monitor_flow.sh <runname>}"
FLOW="asic/runs/openeye_${RUN}/flow.log"
STEPS="asic/runs/openeye_${RUN}/design/runs/${RUN}"
WATCHDOG="asic/runs/watchdog_${RUN}.log"
OUT="asic/runs/monitor_${RUN}.log"
MONITOR_INTERVAL=300
STALL_S=3600

# (pid cpu_seconds) of the NEWEST non-zombie tool process, "none 0" if
# there is none. etimes (elapsed seconds) picks the current step's tool;
# a new process also means the flow advanced.
tool_cpu() {
    ps -eo pid=,stat=,etimes=,times=,comm= 2>/dev/null | awk '
        $2 !~ /^Z/ && ($5 == "openroad" || $5 == "yosys" || $5 == "klayout" || $5 == "magic") {
            print $1, $3, $4
        }' | sort -k2,2n | head -1 | awk '{print $1, $3}'
}

snap_steps() { ls "$STEPS" 2>/dev/null | grep -E '^[0-9]{2}-' | tr '\n' ' '; }

now=$(date +%s)
last_size=0; last_change=$now; last_stepset=""; last_wd=0
last_cpu_pid="none"; last_cpu_time=0; last_cpu_change=$now

while true; do
  now=$(date +%s)
  size=$(stat -c%s "$FLOW" 2>/dev/null || echo 0)
  stepset=$(snap_steps)
  wd=$(grep -c "restart [0-9]/" "$WATCHDOG" 2>/dev/null || true)
  [ -n "$wd" ] || wd=0
  # actual cmdline: python3 /usr/local/bin/librelane --manual-pdk openeye.json ...
  # run_openeye.sh stays alive for the whole flow (parent of docker exec)
  # -> covers the gen.sh staging phase before librelane starts (p2_15 gave
  # a false "DEAD" flag during gen.sh)
  alive=$( { pgrep -f "librelane.*openeye\.json" >/dev/null 2>&1 || pgrep -f "run_openeye\.sh ${RUN}" >/dev/null 2>&1; } && echo yes || echo no)
  flags=""

  if [ "$size" -ne "$last_size" ]; then
    last_size=$size
    last_change=$now
  fi
  if [ -n "$stepset" ] && [ "$stepset" != "$last_stepset" ]; then
    if [ -n "$last_stepset" ]; then flags="$flags NEW-STEP"
    fi
    last_stepset="$stepset"
  fi
  if [ "$wd" -gt "${last_wd:-0}" ]; then flags="$flags RESTART(count=$wd)"; fi
  last_wd=$wd
  if [ -s "$WATCHDOG" ] && grep -qE "completed|not auto-restarting" "$WATCHDOG" 2>/dev/null; then
    flags="$flags WD-EXITED"
  fi

  # CPU liveness: a new tool process or increasing cumulative CPU time
  # means the flow is working (even with a flat, buffered log).
  read -r cpu_pid cpu_time < <(tool_cpu)
  if [ "${cpu_pid:-none}" != "${last_cpu_pid:-none}" ]; then
      last_cpu_change=$now
      last_cpu_pid="${cpu_pid:-none}"
      last_cpu_time="${cpu_time:-0}"
  elif [ "${cpu_time:-0}" -gt "${last_cpu_time:-0}" ]; then
      last_cpu_change=$now
      last_cpu_time="${cpu_time:-0}"
  fi

  # STALL only when log AND step files AND cpu time are all flat.
  log_flat=0; [ $((now - last_change)) -gt $STALL_S ] && log_flat=1
  files_flat=0
  if [ "$log_flat" -eq 1 ]; then
      fresh=$(find "$STEPS" -type f -mmin $((STALL_S / 60)) 2>/dev/null | head -1)
      [ -z "$fresh" ] && files_flat=1
  fi
  cpu_flat=0; [ $((now - last_cpu_change)) -gt $STALL_S ] && cpu_flat=1
  if [ "$log_flat" -eq 1 ] && [ "$files_flat" -eq 1 ] && [ "$cpu_flat" -eq 1 ] && [ "$alive" = "yes" ]; then
    flags="$flags STALL(no activity $(( (now-last_change)/60 )) min)"
  fi

  if [ "$alive" = "no" ]; then
    if grep -q "finished without ERROR" "asic/runs/openeye_${RUN}_launcher.log" 2>/dev/null; then
      flags="$flags DONE"
      echo "$(date -Is) FLOW COMPLETED - monitor exiting" >> "$OUT"
      break
    else
      flags="$flags DEAD(librelane gone)"
    fi
  fi
  echo "$(date -Is) steps:[$stepset] flow=${size}B alive=$alive cpu=${last_cpu_pid}:${last_cpu_time}s wd_restarts=$wd$flags" >> "$OUT"
  sleep "$MONITOR_INTERVAL"
done
