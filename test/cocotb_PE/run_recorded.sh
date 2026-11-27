#!/usr/bin/env bash
set -e

if (( $# > 1 )); then
    echo "Usage: $0 [seed|case]" >&2
    exit 2
fi
sim=${SIM:-icarus}
case "$sim" in
    icarus) version_cmd=(iverilog -V) ;;
    verilator) version_cmd=(verilator --version) ;;
    *) echo "Unknown simulator: $sim (use icarus or verilator)" >&2; exit 2 ;;
esac
case_name=${1-42}
testbench=PE
case $case_name in
    pe_seed42) seed=42;;
    pe_seed43) seed=43;;
    pe_simple_seed42) seed=42; testbench=PE_simple;;
    ''|*[!0-9]*) echo "Unknown case or invalid seed: $case_name" >&2; exit 2;;
    *) seed=$case_name; case_name=pe_seed$seed;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $testbench == PE_simple ]]; then cd ../cocotb_PE_simple; fi
mkdir -p results
# Headers and CSV files remain shared; run sequentially with other tests here.
exec 9>results/.run.lock
flock -n 9 || { echo "Another recorded $testbench run is active" >&2; exit 1; }
run_dir=$(mktemp -d "$PWD/results/${sim}_seed${seed}.XXXXXX")
command=(make run "SIM=$sim" "SEED=$seed"
    "SIM_BUILD=$run_dir/sim_build"
    "COCOTB_RESULTS_FILE=$run_dir/results.xml"
    "COCOTB_PLUSARGS=+FST_PATH=$run_dir/$testbench.fst")
{
    printf 'case=%s\nseed=%s\nworking_directory=%s\ncommand=cd %q && ' "$case_name" "$seed" "$PWD" "$PWD"
    printf '%q ' "${command[@]}"
    printf '\n'
    printf 'simulator=%s\nsource_revision=%s\n' "$sim" "$(git rev-parse HEAD)"
    printf 'tool_version:\n'
    if ! "${version_cmd[@]}" 2>&1; then
        printf 'Version query failed; see output.log for the simulation result.\n'
    fi
    printf 'working_tree:\n'
    git status --short
} > "$run_dir/metadata.txt"
git diff --binary HEAD > "$run_dir/source.patch"
printf 'Results: %s\n' "$run_dir"

set +e
"${command[@]}" 2>&1 | tee "$run_dir/output.log"
statuses=("${PIPESTATUS[@]}")
set -e
status=${statuses[0]}
if (( status == 0 )); then status=${statuses[1]}; fi
if [[ $testbench == PE && -f "$run_dir/results.xml" ]]; then
    cp -- parameters_PE.vh parameters_PE_cluster.vh "$run_dir/" || status=$?
fi
printf 'exit_status=%s\n' "$status" >> "$run_dir/metadata.txt"
exit "$status"
