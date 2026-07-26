# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
Full-system OpenEye_FPGA simulation of a single 3x3 convolution layer.

Configuration requested for this scenario:
  - 3x3 kernel, 8 input channels, random values (random Keras Conv2D init)
  - 50% sparsity on iacts and weights (checkerboard pattern zeroing every
    element with (c+x+y) % 2 == 0 — see DRAM.write_initial_data_to_dram)
  - Reduced accelerator size:
        CLUSTER_ROWS = 2   (cluster array 2x2, CLUSTER_COLUMNS fixed at 2)
        NUM_GLB_IACT = 3
        NUM_GLB_PSUM = 4   (== PE_COLUMNS)
        NUM_GLB_WGHT = 3   (== PE_ROWS)

The simulation drives the DMA stream interface of OpenEye_FPGA exactly as
the FPGA sees it (config words, iact/wght/bias/quant streams) and compares
the returned PSUM stream against the TensorFlow reference computed by the
test infrastructure (OpenEye_FPGA_tb / rtl_test_utils.compare_stream_Conv).

Usage:
    pytest test_conv3x3_sparse.py -v
"""

import os

import pytest
import cocotb_test.simulator

import open_eye.test_utils_main as ptu
import open_eye.vh_file_creator as vh_file_creator
import open_eye.generator as generator
from open_eye import hdl_dir, test_dir, open_eye_dir

clk_cycle          = 20
clk_cycle_unit     = "ns"
clk_delay_in       = 100
clk_delay_unit_in  = "ps"
clk_delay_out      = 100
clk_delay_unit_out = "ps"


@pytest.mark.parametrize("NUM_FILTERS",    [8, 16])
@pytest.mark.parametrize("STRIDE",         [1])
@pytest.mark.parametrize("KERNEL_SIZE_X",  [3])
@pytest.mark.parametrize("KERNEL_SIZE_Y",  [3])
@pytest.mark.parametrize("INPUT_SIZE_X",   [32, 128])
@pytest.mark.parametrize("INPUT_SIZE_Y",   [1, 4])
@pytest.mark.parametrize("INPUT_CHANNELS", [4, 8, 16])
# 50% sparsity: DRAM zeroes elements on the (c+x+y) % 2 == 0 checkerboard.
# SPARSE=0 is the dense control case with otherwise identical configuration.
@pytest.mark.parametrize("SPARSE", [1, 0])
# Reduced accelerator size (PE_COLUMNS = NUM_GLB_PSUM, PE_ROWS = NUM_GLB_WGHT)
@pytest.mark.parametrize("CLUSTER_ROWS", [2, 4])
@pytest.mark.parametrize("NUM_GLB_IACT", [3])
@pytest.mark.parametrize("NUM_GLB_PSUM", [4])
@pytest.mark.parametrize("NUM_GLB_WGHT", [3])
def test_conv3x3_8ch_sparse50(
    NUM_FILTERS, STRIDE, KERNEL_SIZE_X, KERNEL_SIZE_Y,
    INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_CHANNELS,
    SPARSE,
    CLUSTER_ROWS, NUM_GLB_IACT, NUM_GLB_PSUM, NUM_GLB_WGHT,
    request,
):
    USE_SPARSE_IACTS = SPARSE
    USE_SPARSE_WGHTS = SPARSE
    # OpenEyeParameters and the parameters.vh generator read these at import
    # of the accelerator dimensions, so they must be set before
    # create_vh_file_from_envvars runs.
    os.environ["CLUSTER_ROWS"] = str(CLUSTER_ROWS)
    os.environ["NUM_GLB_IACT"] = str(NUM_GLB_IACT)
    os.environ["NUM_GLB_PSUM"] = str(NUM_GLB_PSUM)
    os.environ["NUM_GLB_WGHT"] = str(NUM_GLB_WGHT)
    # Fixed platform configuration for this branch (matches the values the
    # cocotb_fpga Makefile exports). regmap.yaml width expressions and
    # parameters.vh generation read these from the environment. QUANT_AMOUNT
    # must be 1024: conv_mapper.write_quantize sends 512 DMA words and the
    # GET_QUANTIZE FSM expects QUANT_AMOUNT/2 of them.
    os.environ["BRANCHES"]        = "1"
    os.environ["BUFFER_WIDTH"]    = "12"
    os.environ["QUANT_AMOUNT"]    = "1024"
    os.environ["RAM_CELLS"]       = "32"
    os.environ["CLUSTER_COLUMNS"] = "2"

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

    # test/cocotb_fpga/regmap.yaml carries registers that the checked-in
    # hdl/dma_storage.v (generated from hdl/config/regmap.yaml) does not
    # know yet. Compile the freshly generated decoder from sim_build instead
    # of the stale HDL copy so it matches the generated regmap_params.vh.
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
        waves=True,
        simulator="icarus",
        extra_env={
            "CLOCK_LEN":               str(clk_cycle),
            "CLOCK_UNIT":              clk_cycle_unit,
            "CLOCK_DELAY_INPUT":       str(clk_delay_in),
            "CLOCK_DELAY_UNIT_INPUT":  clk_delay_unit_in,
            "CLOCK_DELAY_OUTPUT":      str(clk_delay_out),
            "CLOCK_DELAY_UNIT_OUTPUT": clk_delay_unit_out,
            "LAYER":             "Convolution",
            "NUM_FILTERS":       str(NUM_FILTERS),
            "STRIDE":            str(STRIDE),
            "KERNEL_SIZE_X":     str(KERNEL_SIZE_X),
            "KERNEL_SIZE_Y":     str(KERNEL_SIZE_Y),
            "INPUT_SIZE_X":      str(INPUT_SIZE_X),
            "INPUT_SIZE_Y":      str(INPUT_SIZE_Y),
            "INPUT_CHANNELS":    str(INPUT_CHANNELS),
            "USE_RANDOM_VALUES": "1",
            "USE_SPARSE_IACTS":  str(USE_SPARSE_IACTS),
            # OpenEye_FPGA_tb reads USE_SPARSE_WEIGHTS; other runners set
            # USE_SPARSE_WGHTS. Set both spellings so the flag arrives.
            "USE_SPARSE_WGHTS":   str(USE_SPARSE_WGHTS),
            "USE_SPARSE_WEIGHTS": str(USE_SPARSE_WGHTS),
            "CLUSTER_ROWS": str(CLUSTER_ROWS),
            "NUM_GLB_IACT": str(NUM_GLB_IACT),
            "NUM_GLB_PSUM": str(NUM_GLB_PSUM),
            "NUM_GLB_WGHT": str(NUM_GLB_WGHT),
            "LOGGER_LEVEL": "10",
            "COCOTB_LOG_FILE_PATH": os.path.join(target_dir, "cocotb_sim.log"),
            "COCOTB_TRACE": "1",
        },
    )


if __name__ == "__main__":
    class _Node:
        nodeid = "conv3x3_8ch_sparse50_standalone"
    class _Request:
        node = _Node()
    test_conv3x3_8ch_sparse50(
        NUM_FILTERS=8, STRIDE=1, KERNEL_SIZE_X=3, KERNEL_SIZE_Y=3,
        INPUT_SIZE_X=32, INPUT_SIZE_Y=1, INPUT_CHANNELS=8,
        SPARSE=1,
        CLUSTER_ROWS=2, NUM_GLB_IACT=3, NUM_GLB_PSUM=4, NUM_GLB_WGHT=3,
        request=_Request(),
    )
