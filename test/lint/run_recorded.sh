#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 {pe_dense|parallel|pe_cluster_dense|openeye_cluster|openeye_fpga}" >&2
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
    pe_cluster_dense)
        lint_cmd+=(--top-module PE_cluster -GSPARSITY_EN=0 -GPARALLEL_MACS=1 -GSERIAL=1 -GPE_ROWS=3 -GPE_COLUMNS=4 +define+USE_INTERNAL_PARAMS_PE_cluster)
        hdl_file=hdl/PE_cluster.v
        ;;
    openeye_cluster)
        lint_cmd+=(--top-module OpenEye_Cluster +define+USE_INTERNAL_PARAMS_PE_cluster)
        hdl_file=hdl/OpenEye_Cluster.v
        ;;
    openeye_fpga)
        lint_cmd+=(--top-module OpenEye_FPGA +define+USE_INTERNAL_PARAMS +define+USE_INTERNAL_PARAMS_PE_cluster)
        hdl_file=hdl/OpenEye_FPGA.v
        ;;
    *) echo "Unknown lint profile: $profile" >&2; exit 2 ;;
esac

repo_commit=$(git rev-parse HEAD)
mkdir -p test/lint/results
run_dir=$(mktemp -d "$repo_root/test/lint/results/${profile}.XXXXXX")
printf 'Results: %s\n' "$run_dir"

if [[ "$profile" == "openeye_fpga" ]]; then
    lint_cmd+=(-I"$run_dir/generated" "$run_dir/generated/dma_storage.v")
fi

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

if [[ "$profile" == "openeye_fpga" ]]; then
    # Match the FPGA's internal defaults and keep all generator output local.
    if (
        cd -- "$run_dir" || exit
        PYTHONPATH="$repo_root/src" \
        CLUSTER_ROWS=2 CLUSTER_COLUMNS=2 BRANCHES=1 DMA_BITWIDTH=64 \
        BUFFER_WIDTH_IACT=10 BUFFER_WIDTH_WGHT=12 CHANNEL_CYCLE_WIDTH=12 \
        LIMIT_SIZE_IACT=18 LIMIT_SIZE_WGHT=12 LIMIT_SIZE_PSUM=12 \
        python3 -B -m open_eye.generator "$repo_root/test/cocotb_fpga" \
            "$run_dir/generated" "$run_dir/generated" "$run_dir/generated"
    ) > "$run_dir/generate.log" 2>&1; then
        printf 'generation_exit_status=0\n' >> "$run_dir/metadata.txt"
    else
        status=$?
        printf 'generation_exit_status=%s\nexit_status=%s\n' "$status" "$status" >> "$run_dir/metadata.txt"
        cat "$run_dir/generate.log" >&2
        exit "$status"
    fi
fi

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