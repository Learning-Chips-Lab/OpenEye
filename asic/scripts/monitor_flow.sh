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
# G1 gates (plan v2, 2026-11-01): when a key step's latest dir is complete
# (state_out.json), the gate is evaluated once and a GATE line is appended
# to the monitor log whenever the set of gate results changes (state in
# asic/runs/monitor_<run>_gates.txt, so the flags are not re-printed
# every 5 min). Gates (slug -> check, thresholds from plan v2 G1):
#   odb-manualmacroplacement    "Successfully placed N instances" = 41
#   openroad-generatepdn        no PDN-0235 (PSM-0069 pad vdd is the known
#                               PDK gap, allowed)
#   openroad-globalplacement    last "Total routing overflow" <= 0.1
#   openroad-cts                "Total number of sinks" (report; WARN if
#                               > 100k -- p2_15 was 506,219)
#   openroad-stamidpnr-2        nom_typ setup WNS >= 0, hold WNS >= -0.5
#   openroad-globalrouting      no GRT-0116, no GRT-0115, final report
#                               total overflow <= 0.01 % of demand,
#                               report GRT-0102 extra-iteration count,
#                               WARN on > 50 EST-0026
#   openroad-resizertimingpostgrt  setup WNS (from post-GRT STA
#                               stamidpnr-3 rpt files; the resizer
#                               log prints no WNS) within 1 ns of
#                               pre-GRT (stamidpnr-2), hold >= -0.5,
#                               WARN on > 100 RSZ-2021
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
GATE_STATE="asic/runs/monitor_${RUN}_gates.txt"
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

# G1 gates (see header). Prints one compact summary string; empty while no
# gated step is complete yet. Only completed steps (latest dir has
# state_out.json) are evaluated, so a running step never produces a
# half-baked gate result.
check_gates() {
    local out="" d log n
    latest_dir() { ls -d "$STEPS"/[0-9]*-"$1" 2>/dev/null | sort | tail -1; }
    # metric value for a corner (first match, nom_typ)
    wns() { grep -oE "Writing metric timing__${1}__wns__corner:nom_typ[^:]*: [-0-9.e]+" \
              "$2" 2>/dev/null | head -1 | awk '{print $NF}'; }

    # 18: macro placement count (expect 41)
    d=$(latest_dir odb-manualmacroplacement)
    if [ -n "$d" ] && [ -f "$d/state_out.json" ]; then
        n=$(grep -hoE "Successfully placed [0-9]+ instances" "$d"/*.log 2>/dev/null | head -1 | grep -oE "[0-9]+")
        if [ "$n" = "41" ]; then out="${out} G1-18:placed=41 OK"
        else out="${out} G1-18:placed=${n:-NO-MESSAGE} GATE-FAIL(exp 41)"; fi
    fi

    # 22: GeneratePDN (PDN-0235 = unplaced macros -> fail)
    d=$(latest_dir openroad-generatepdn)
    if [ -n "$d" ] && [ -f "$d/state_out.json" ]; then
        if grep -q "PDN-0235" "$d"/*.log 2>/dev/null; then
            out="${out} G1-22:PDN-0235 GATE-FAIL"
        else
            out="${out} G1-22:PDN OK"
        fi
    fi

    # 27: GlobalPlacement final overflow <= 0.1
    d=$(latest_dir openroad-globalplacement)
    if [ -n "$d" ] && [ -f "$d/state_out.json" ]; then
        n=$(grep -hoE "Total routing overflow: [0-9.]+" "$d"/*.log 2>/dev/null | tail -1 | awk '{print $NF}')
        if [ -z "$n" ]; then
            out="${out} G1-27:overflow=NO-MESSAGE"
        elif awk -v v="$n" 'BEGIN{exit !(v > 0.1)}'; then
            out="${out} G1-27:overflow=${n} GATE-FAIL(exp <=0.1)"
        else
            out="${out} G1-27:overflow=${n} OK"
        fi
    fi

    # 34: CTS clock sinks (report; p2_15 was 506219)
    d=$(latest_dir openroad-cts)
    if [ -n "$d" ] && [ -f "$d/state_out.json" ]; then
        # MAX occurrence: CTS prints one per clock, in processing order
        # (p2_15: last = total 506219; p2_16: the 51883-sink clk_pad tree
        # is processed BEFORE the small macro A_CLK trees, so the last
        # line is 24 — the max is the main-clock sink count)
        n=$(grep -hoE "Total number of sinks: [0-9]+" "$d"/*.log 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
        if [ -n "$n" ] && [ "$n" -gt 100000 ]; then
            out="${out} G1-34:sinks=${n} GATE-WARN(exp << 506219)"
        else
            out="${out} G1-34:sinks=${n:-NO-MESSAGE}"
        fi
    fi

    # 37: STAMidPNR-2 (postCTS resizer) nom_typ setup/hold WNS
    d=$(latest_dir openroad-stamidpnr-2)
    if [ -n "$d" ] && [ -f "$d/state_out.json" ]; then
        # dir name = "<2-digit ordinal>-<slug>"; the slug starts at char 4
        log="$d"/$(basename "$d" | cut -c4-).log
        s=$(wns setup "$log"); h=$(wns hold "$log")
        echo "pregrt_setup=${s:-?}" > "$GATE_STATE"
        if [ -n "$s" ] && awk -v v="$s" 'BEGIN{exit !(v < 0)}'; then
            out="${out} G1-37:setup=${s} GATE-FAIL(exp >=0)"
        else
            out="${out} G1-37:setup=${s:-?}"
        fi
        if [ -n "$h" ] && awk -v v="$h" 'BEGIN{exit !(v < -0.5)}'; then
            out="${out} G1-37:hold=${h} GATE-FAIL(exp >=-0.5)"
        else
            out="${out} G1-37:hold=${h:-?}"
        fi
    fi

    # 38: GlobalRouting — the key gate
    d=$(latest_dir openroad-globalrouting)
    if [ -n "$d" ] && [ -f "$d/state_out.json" ]; then
        log="$d"/$(basename "$d" | cut -c4-).log
        local e115 e116 xiter est total_line dem ovf
        e115=$(grep -c "GRT-0115" "$log" 2>/dev/null); e116=$(grep -c "GRT-0116" "$log" 2>/dev/null)
        xiter=$(grep -c "Start extra iteration" "$log" 2>/dev/null)
        est=$(grep -c "EST-0026" "$log" 2>/dev/null)
        total_line=$(grep -A12 "Final congestion report" "$log" 2>/dev/null | grep -E "^Total " | tail -1)
        dem=$(echo "$total_line" | awk '{print $3}')
        ovf=$(echo "$total_line" | awk '{print $NF}')
        if [ "${e116:-0}" -gt 0 ]; then out="${out} G1-38:GRT-0116=${e116} GATE-FAIL"
        elif [ "${e115:-0}" -gt 0 ]; then out="${out} G1-38:GRT-0115=${e115} GATE-FAIL"
        elif [ -n "$ovf" ] && [ -n "$dem" ] && awk -v o="$ovf" -v d="$dem" 'BEGIN{exit !(o > d*0.0001)}'; then
            out="${out} G1-38:overflow=${ovf}/${dem} GATE-FAIL(exp <=0.01%)"
        else
            out="${out} G1-38:GRT OK overflow=${ovf:-?}/${dem:-?}"
        fi
        out="${out} extra_iter=${xiter:-?}"
        [ "${est:-0}" -gt 50 ] && out="${out} EST-0026=${est} GATE-WARN"
    fi

    # 44: ResizerTimingPostGRT vs. pre-GRT (step 37) setup WNS
    d=$(latest_dir openroad-resizertimingpostgrt)
    if [ -n "$d" ] && [ -f "$d/state_out.json" ]; then
        log="$d"/$(basename "$d" | cut -c4-).log
        # The resizer log carries no final WNS (RSZ messages only) -> the
        # timing truth is the post-GRT STA: wns.max.rpt (setup) /
        # wns.min.rpt (hold) of the latest completed openroad-stamidpnr-3
        # ("Writing metric timing__<k>__wns__corner:nom_typ...: <v>",
        # first corner line; mid-PNR STAs here report nom_typ).
        local d3 s="" h=""
        d3=$(latest_dir openroad-stamidpnr-3)
        if [ -n "$d3" ] && [ -f "$d3/state_out.json" ]; then
            s=$(wns setup "$d3/wns.max.rpt"); h=$(wns hold "$d3/wns.min.rpt")
        fi
        local rsz2021 pregrt
        rsz2021=$(grep -c "RSZ-2021" "$log" 2>/dev/null)
        pregrt=$(grep "^pregrt_setup=" "$GATE_STATE" 2>/dev/null | cut -d= -f2)
        local smsg="setup=${s:-?}"
        if [ -n "$s" ] && [ -n "$pregrt" ] && [ "$pregrt" != "?" ] && \
           awk -v v="$s" -v p="$pregrt" 'BEGIN{exit !(v < p - 1.0)}'; then
            smsg="${smsg}(pre ${pregrt}) GATE-FAIL(exp >= pre-1ns)"
        elif [ -n "$s" ] && [ -n "$pregrt" ] && [ "$pregrt" != "?" ]; then
            smsg="${smsg}(pre ${pregrt})"
        fi
        if [ -n "$h" ] && awk -v v="$h" 'BEGIN{exit !(v < -0.5)}'; then
            out="${out} G1-44:${smsg} hold=${h} GATE-FAIL(exp >=-0.5)"
        else
            out="${out} G1-44:${smsg} hold=${h:-?}"
        fi
        [ "${rsz2021:-0}" -gt 100 ] && out="${out} RSZ-2021=${rsz2021} GATE-WARN"
    fi

    printf '%s' "$out"
}

now=$(date +%s)
last_size=0; last_change=$now; last_stepset=""; last_wd=0
last_cpu_pid="none"; last_cpu_time=0; last_cpu_change=$now
last_gate=""   # gate summary string; re-printed only when it changes

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

  # G1 gates: evaluate completed key steps once each; print a GATE line
  # when the summary changes (grows by one gate per finished step).
  gates="$(check_gates)"
  if [ -n "$gates" ] && [ "$gates" != "$last_gate" ]; then
    echo "$(date -Is) GATE:${gates}" >> "$OUT"
    last_gate="$gates"
  fi

  sleep "$MONITOR_INTERVAL"
done
