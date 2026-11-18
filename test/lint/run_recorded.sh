#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 {pe_dense|parallel}" >&2
    exit 2
fi
profile=$1
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$repo_root"

lint_cmd=(verilator --lint-only -Wall -Ihdl/)
case "$profile" in
    pe_dense)
        lint_cmd+=(--top-module PE -GPARALLEL_MACS=1 -GSPARSITY_EN=0 "-GSERIAL=1'b1")
        hdl_file=hdl/PE.v
        ;;
    parallel)
        lint_cmd+=(--top-module OpenEye_Parallel +define+USE_INTERNAL_PARAMS_PE_cluster)
        hdl_file=hdl/OpenEye_Parallel.v
        ;;
    *) echo "Unknown lint profile: $profile" >&2; exit 2 ;;
esac

repo_commit=$(git rev-parse HEAD)
mkdir -p test/lint/results
run_dir=$(mktemp -d "$repo_root/test/lint/results/${profile}.XXXXXX")
printf 'Results: %s\n' "$run_dir"

# Exclude SERIAL=0 adder-tree waivers from the SERIAL=1 profile.
waiver_file=hdl/lint_waivers.vlt
if [[ "$profile" == "pe_dense" ]]; then
    waiver_file="$run_dir/lint_waivers.vlt"
    sed '/^\/\/ Adder 3 exists only for SERIAL=1,/d; /^lint_off .*adder_tree_/d' \
        hdl/lint_waivers.vlt > "$waiver_file"

    cat >> "$waiver_file" <<'VLT'

// Dense mode keeps the 8-bit payload and discards the 4-bit position tag.
lint_off -rule WIDTHTRUNC -file "*data_pipeline_iact.v" -match "*ASSIGNW expects 8 bits*REPLICATE generates 12 bits*"
VLT
fi
lint_cmd+=(+define+USE_INTERNAL_PARAMS_PE "$waiver_file" "$hdl_file")

{
    printf 'profile=%s\nsource_revision=%s\n' "$profile" "$repo_commit"
    printf 'command='
    printf '%q ' "${lint_cmd[@]}"
    printf '\n'
    printf 'tool_version='
    if ! verilator --version 2>&1; then
        printf 'Version query failed. See output.log for the lint result.\n'
    fi
    printf 'working_tree:\n'
    git status --short
} > "$run_dir/metadata.txt"
git diff --binary HEAD > "$run_dir/source.patch"

set +e
"${lint_cmd[@]}" 2>&1 | tee "$run_dir/output.log"
statuses=("${PIPESTATUS[@]}")
set -e
status=${statuses[0]}
if (( status == 0 )); then status=${statuses[1]}; fi
printf 'verilator_exit_status=%s\nlog_exit_status=%s\nexit_status=%s\n' \
    "${statuses[0]}" "${statuses[1]}" "$status" >> "$run_dir/metadata.txt"
printf '\nExit status: %s\nSaved results: %s\n' "$status" "$run_dir"
exit "$status"