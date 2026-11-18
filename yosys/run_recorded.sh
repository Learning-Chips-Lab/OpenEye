#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 pe_dense" >&2
    exit 2
fi
profile=$1
case "$profile" in
    pe_dense) synth_script=yosys/pe_dense.tcl ;;
    *) echo "Unknown synthesis profile: $profile" >&2; exit 2 ;;
esac

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_root"
repo_commit=$(git rev-parse HEAD)

mkdir -p yosys/results
run_dir=$(mktemp -d "$repo_root/yosys/results/${profile}.XXXXXX")
printf 'Results: %s\n' "$run_dir"

cp -- "$synth_script" "$run_dir/pe_dense.tcl"
cp -- yosys/run_recorded.sh "$run_dir/run_recorded.sh"
git diff --binary HEAD > "$run_dir/source.patch"
command=(yosys -Q -T -c "$run_dir/pe_dense.tcl")

{
    printf 'profile=%s\nsource_revision=%s\n' "$profile" "$repo_commit"
    printf 'configuration=pe_dense.tcl\n'
    printf 'command=SYNTH_RUN_DIR=%q ' "$run_dir"
    printf '%q ' "${command[@]}"
    printf '\ntool_version='
    if ! yosys -V 2>&1; then
        printf 'Version query failed; see output.log.\n'
    fi
    printf 'working_tree:\n'
    git status --short
} > "$run_dir/metadata.txt"

set +e
SYNTH_RUN_DIR="$run_dir" "${command[@]}" 2>&1 | tee "$run_dir/output.log"
statuses=("${PIPESTATUS[@]}")
set -e

status=${statuses[0]}
if (( status == 0 )); then status=${statuses[1]}; fi
validation_status=not_run

if (( status == 0 )); then
    validation_status=0
    for filename in resources.json netlist.json; do
        if ! python3 -m json.tool "$run_dir/$filename" > /dev/null 2>> "$run_dir/output.log"; then
            printf 'Missing or invalid JSON: %s\n' "$filename" >&2
            validation_status=1
        fi
    done
    status=$validation_status
fi

printf 'yosys_exit_status=%s\nlog_exit_status=%s\nvalidation_exit_status=%s\nexit_status=%s\n' \
    "${statuses[0]}" "${statuses[1]}" "$validation_status" "$status" >> "$run_dir/metadata.txt"

printf '\nExit status: %s\nSaved results: %s\n' "$status" "$run_dir"
exit "$status"