# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
Full-system OpenEye_FPGA simulation of a GEMM layer under both dataflows.

A GEMM layer (C = A x B + bias, realized as a single Dense layer) is run
through the complete DMA flow with one and two MAC lanes under both dataflows:

  - DATAFLOW=row_stationary:    the classic dense/FC mapping; iact routing
                                is derived from the iact_choose pattern of
                                the iact_stream_constructor.
  - DATAFLOW=output_stationary: the same layer with gemm_mode=1; the
                                gemm_mode DMA register switches every PE
                                cluster to the output-stationary binding
                                (iact GLB bank j -> PE row j) and each PE
                                keeps its output tile stationary in the
                                local psum SPad.

Every run checks random operands against the exact integer Dense reference
(test_utils_main.collect_results), so a pass demonstrates the selected
routing mode end to end:
DMA words -> dma_storage.gemm_mode -> OpenEye_Parallel -> OpenEye_Cluster
-> PE_cluster row/bank binding -> PE MAC/accumulate -> psum readout.

Usage:
    pytest test_gemm_layer.py -v
    pytest "test_gemm_layer.py::test_gemm_layer[output_stationary-...]" -v
"""

import os

import pytest
import cocotb_test.simulator

import open_eye.test_utils_main as ptu
import open_eye.vh_file_creator as vh_file_creator
import open_eye.generator as generator
from open_eye import hdl_dir, test_dir

clk_cycle          = 20
clk_cycle_unit     = "ns"
clk_delay_in       = 100
clk_delay_unit_in  = "ps"
clk_delay_out      = 100
clk_delay_unit_out = "ps"


@pytest.mark.parametrize("PARALLEL_MACS", [1, 2])
@pytest.mark.parametrize("INPUT_SIZE",   [32])   # K (inner dimension)
@pytest.mark.parametrize("OUTPUT_SIZE",  [32])   # N (output columns)
@pytest.mark.parametrize("CLUSTER_ROWS", [2])
@pytest.mark.parametrize("NUM_GLB_IACT", [3])
@pytest.mark.parametrize("NUM_GLB_PSUM", [4])
@pytest.mark.parametrize("NUM_GLB_WGHT", [3])
@pytest.mark.parametrize("DATAFLOW", ["row_stationary", "output_stationary"])
def test_gemm_layer(
    INPUT_SIZE, OUTPUT_SIZE,
    CLUSTER_ROWS, NUM_GLB_IACT, NUM_GLB_PSUM, NUM_GLB_WGHT,
    DATAFLOW,
    request,
    PARALLEL_MACS,
    CLUSTER_COLUMNS=2,
):
    # OpenEyeParameters and the parameters.vh generator read these at import
    # of the accelerator dimensions, so they must be set before
    # create_vh_file_from_envvars runs.
    os.environ["PARALLEL_MACS"] = str(PARALLEL_MACS)
    os.environ["CLUSTER_ROWS"] = str(CLUSTER_ROWS)
    os.environ["NUM_GLB_IACT"] = str(NUM_GLB_IACT)
    os.environ["NUM_GLB_PSUM"] = str(NUM_GLB_PSUM)
    os.environ["NUM_GLB_WGHT"] = str(NUM_GLB_WGHT)
    os.environ["DATAFLOW"]     = DATAFLOW
    # Fixed platform configuration (matches the cocotb_fpga Makefile exports;
    # see test_conv3x3_sparse.py for the QUANT_AMOUNT rationale).
    os.environ["BRANCHES"]        = "1"
    os.environ["BUFFER_WIDTH"]    = "12"
    os.environ["QUANT_AMOUNT"]    = "1024"
    os.environ["RAM_CELLS"]       = "32"
    os.environ["CLUSTER_COLUMNS"] = str(CLUSTER_COLUMNS)

    toplevel = "OpenEye_FPGA"
    module   = "OpenEye_FPGA_tb"

    nodeid = request.node.nodeid.replace("::", "_").replace("/", "_") \
                                .replace("[", "_").replace("]", "_")
    target_dir = os.path.join(test_dir, ".temp", nodeid)
    os.makedirs(target_dir, exist_ok=True)

    # Regenerate dma_storage.v + regmap_params.vh from this directory's
    # regmap.yaml, and parameters.vh from the env vars, all into sim_build.
    regmap_dir = os.path.join(test_dir, "cocotb_fpga")
    vh_file_creator.create_vh_file_from_envvars(
        target_dir, hdl_dir + "/", toplevel=toplevel)
    generator.create_regmap_params_vh_file(regmap_dir, target_dir, target_dir)

    # Compile the freshly generated register decoder so it matches the
    # generated regmap_params.vh (see test_conv3x3_sparse.py).
    verilog_sources = [
        src for src in ptu.get_verilog_sources(hdl_dir)
        if os.path.basename(src) != "dma_storage.v"
    ]
    verilog_sources.append(os.path.join(target_dir, "dma_storage.v"))

    cocotb_test.simulator.run(
        python_search=[test_dir],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=target_dir,
        testcase="start_test_fpga",
        defines={"NO_TRACE": "TRUE", "USE_INTERNAL_PARAMS_PE": "TRUE"},
        force_compile=True,
        waves=os.environ.get("OPENEYE_FPGA_WAVES") == "1",
        simulator="icarus",
        extra_env={
            "CLOCK_LEN":               str(clk_cycle),
            "CLOCK_UNIT":              clk_cycle_unit,
            "CLOCK_DELAY_INPUT":       str(clk_delay_in),
            "CLOCK_DELAY_UNIT_INPUT":  clk_delay_unit_in,
            "CLOCK_DELAY_OUTPUT":      str(clk_delay_out),
            "CLOCK_DELAY_UNIT_OUTPUT": clk_delay_unit_out,
            "LAYER":             "GEMM",
            "INPUT_SIZE_X":      str(INPUT_SIZE),
            "INPUT_SIZE_Y":      "1",
            "OUTPUT_SIZE":       str(OUTPUT_SIZE),
            "NUM_FILTERS":       "1",
            "STRIDE":            "1",
            "KERNEL_SIZE_X":     "1",
            "KERNEL_SIZE_Y":     "1",
            "INPUT_CHANNELS":    "1",
            "USE_RANDOM_VALUES": "1",
            "USE_SPARSE_IACTS":  "0",
            "USE_SPARSE_WGHTS":  "0",
            "USE_SPARSE_WEIGHTS": "0",
            "DATAFLOW":     DATAFLOW,
            "CLUSTER_ROWS": str(CLUSTER_ROWS),
            "NUM_GLB_IACT": str(NUM_GLB_IACT),
            "NUM_GLB_PSUM": str(NUM_GLB_PSUM),
            "NUM_GLB_WGHT": str(NUM_GLB_WGHT),
            "LOGGER_LEVEL": "10",
            "COCOTB_LOG_FILE_PATH": os.path.join(target_dir, "cocotb_sim.log"),
            "COCOTB_TRACE": "1",
        },
    )


@pytest.mark.parametrize("columns", [1, 2])
@pytest.mark.parametrize("parallel_macs", [1, 2])
def test_dense_ten_outputs(columns, parallel_macs, request, monkeypatch):
    """Pin the one-column capture bug and retain two-column coverage."""
    monkeypatch.setenv("OPENEYE_PROBE_FSM", "1")
    monkeypatch.setenv("OPENEYE_FAIL_ON_STALL", "5000")
    monkeypatch.setenv("OPENEYE_CHECK_FC_WRITES", "1")
    test_gemm_layer(
        INPUT_SIZE=32, OUTPUT_SIZE=10,
        CLUSTER_ROWS=2, NUM_GLB_IACT=3, NUM_GLB_PSUM=4, NUM_GLB_WGHT=3,
        DATAFLOW="row_stationary", PARALLEL_MACS=parallel_macs,
        CLUSTER_COLUMNS=columns, request=request,
    )


@pytest.mark.parametrize("input_size", [31, 63])
def test_gemm_split_k_padding(input_size, request):
    """Odd K, zero-padded tails and input sizes spanning multiple buffer words."""
    test_gemm_layer(
        INPUT_SIZE=input_size, OUTPUT_SIZE=8,
        CLUSTER_ROWS=2, NUM_GLB_IACT=3, NUM_GLB_PSUM=4, NUM_GLB_WGHT=3,
        DATAFLOW="row_stationary", PARALLEL_MACS=2, request=request,
    )


@pytest.mark.parametrize("shape", [(4, 4), (8, 8), (4, 8), (8, 4), (16, 16)],
                         ids=lambda shape: f"K{shape[0]}-N{shape[1]}")
@pytest.mark.parametrize("parallel_macs", [1, 2])
@pytest.mark.parametrize("dataflow", ["row_stationary", "output_stationary"])
def test_gemm_split_k_shapes(shape, parallel_macs, dataflow, request):
    """Square/rectangular GEMMs, including K slices containing only padding."""
    test_gemm_layer(
        INPUT_SIZE=shape[0], OUTPUT_SIZE=shape[1],
        CLUSTER_ROWS=2, NUM_GLB_IACT=3, NUM_GLB_PSUM=4, NUM_GLB_WGHT=3,
        DATAFLOW=dataflow, PARALLEL_MACS=parallel_macs, request=request,
    )


if __name__ == "__main__":
    class _Node:
        nodeid = "gemm_layer_standalone"
    class _Request:
        node = _Node()
    test_gemm_layer(
        INPUT_SIZE=32, OUTPUT_SIZE=32,
        CLUSTER_ROWS=2, NUM_GLB_IACT=3, NUM_GLB_PSUM=4, NUM_GLB_WGHT=3,
        DATAFLOW="output_stationary",
        PARALLEL_MACS=2,
        request=_Request(),
    )
