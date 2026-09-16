# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Smallest convolution the FPGA top level can run, with constant operands.

Why this exists
---------------
Every other FPGA conv test uses random weights and activations, so when the
result is wrong the only thing the failure tells you is "wrong". With every
activation and every weight forced to 1, each output becomes a *count*:

    out[f][y][x] = (terms that actually accumulated) + bias[f]

and for an interior pixel the correct count is KERNEL_X * KERNEL_Y * CHANNELS.
A DUT value short by a whole factor therefore names how many products reached
the MACs, which is the question random data cannot answer. That is exactly how
the gemm shortfall was pinned down to 9 of 32 products (see
OPENEYE_CONST_IACTS / OPENEYE_CONST_WGHTS in OpenEye_FPGA_tb.py); this test
makes the same probe available for the convolution path.

It is also deliberately tiny - an 8x8 input with 1 channel and 4 filters - so
it runs in well under a minute and can be iterated on while debugging, unlike
the MNIST net that test_single_layers.py builds by default.

LAYER=Convolution builds two stacked conv layers (see data_create.create_layer),
which means this test also exercises the interlayer psum->iact write-back
without a pooling layer in between. That matters: in the MNIST net the failing
interlayer check sits after a Pooling layer, so conv and pooling cannot be told
apart there.

Usage:
    pytest test_conv_const.py -v
    # or, for the arithmetic to be interpretable, run a single config:
    pytest "test_conv_const.py::test_conv_const_operands[2-3-4-3-1-1-8-3-3-1-4]" -v
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


@pytest.mark.parametrize("NUM_FILTERS",    [4])
@pytest.mark.parametrize("STRIDE",         [1])
@pytest.mark.parametrize("KERNEL_SIZE_X",  [3])
@pytest.mark.parametrize("KERNEL_SIZE_Y",  [3])
@pytest.mark.parametrize("INPUT_SIZE_X",   [8])
@pytest.mark.parametrize("INPUT_SIZE_Y",   [1])
@pytest.mark.parametrize("INPUT_CHANNELS", [4])
# Both operands constant: see the module docstring. Keep them at 1 so the
# output reads directly as a product count; a second value guards against a
# fault that happens to be invisible at 1 (for example a dropped multiply).
# 8 is included because 1 and 2 both quantize to the same interlayer byte:
# if the reference and the DUT are BOTH constant across operand values, the
# check cannot discriminate, and only a value far enough apart shows whether
# either side actually tracks the input.
@pytest.mark.parametrize("CONST_VALUE", [1, 2, 8])
# CLUSTER_ROWS=1 keeps the mapping to a single cluster row, so a failure here
# cannot be blamed on the multi-row psum accumulation chain. CLUSTER_ROWS=2
# adds exactly that chain and nothing else, so the pair isolates it.
@pytest.mark.parametrize("CLUSTER_ROWS", [1, 2])
@pytest.mark.parametrize("NUM_GLB_IACT", [1])
@pytest.mark.parametrize("NUM_GLB_PSUM", [4])
@pytest.mark.parametrize("NUM_GLB_WGHT", [3])
def test_conv_const_operands(
    NUM_FILTERS, STRIDE, KERNEL_SIZE_X, KERNEL_SIZE_Y,
    INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_CHANNELS,
    CONST_VALUE,
    CLUSTER_ROWS, NUM_GLB_IACT, NUM_GLB_PSUM, NUM_GLB_WGHT,
    request,
):
    # OpenEyeParameters and parameters.vh read these from the environment at
    # generation time, so they have to be set before create_vh_file_from_envvars.
    os.environ["CLUSTER_ROWS"] = str(CLUSTER_ROWS)
    os.environ["NUM_GLB_IACT"] = str(NUM_GLB_IACT)
    os.environ["NUM_GLB_PSUM"] = str(NUM_GLB_PSUM)
    os.environ["NUM_GLB_WGHT"] = str(NUM_GLB_WGHT)
    # Fixed platform configuration, matching the other cocotb_fpga runners.
    # QUANT_AMOUNT must stay 1024: conv_mapper.write_quantize sends 512 DMA
    # words and GET_QUANTIZE expects QUANT_AMOUNT/2 of them.
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

    regmap_dir = os.path.join(test_dir, "cocotb_fpga")
    vh_file_creator.create_vh_file_from_envvars(
        target_dir, hdl_dir + "/", toplevel=toplevel)
    generator.create_regmap_params_vh_file(regmap_dir, target_dir, target_dir)

    # The checked-in hdl/dma_storage.v predates registers that this directory's
    # regmap.yaml defines, so compile the decoder just generated into sim_build.
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
            # Dense (non-sparse) data: zero-skipping would make the product
            # count depend on the sparsity pattern and defeat the whole point.
            "USE_RANDOM_VALUES": "1",
            "USE_SPARSE_IACTS":  "0",
            "USE_SPARSE_WGHTS":  "0",
            "USE_SPARSE_WEIGHTS": "0",
            # The two hooks that make the arithmetic interpretable. The
            # reference is recomputed from the same DRAM, so it stays exact.
            "OPENEYE_CONST_IACTS": str(CONST_VALUE),
            "OPENEYE_CONST_WGHTS": str(CONST_VALUE),
            # Fail a hang in minutes instead of running to the 15 ms sim
            # timeout; the stall report names the FSM states and per-PE counts.
            "OPENEYE_PROBE_FSM":      "1",
            "OPENEYE_FAIL_ON_STALL":  "100000",
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
        nodeid = "conv_const_standalone"

    class _Request:
        node = _Node()

    test_conv_const_operands(
        NUM_FILTERS=4, STRIDE=1, KERNEL_SIZE_X=3, KERNEL_SIZE_Y=3,
        INPUT_SIZE_X=8, INPUT_SIZE_Y=1, INPUT_CHANNELS=4,
        CONST_VALUE=1,
        CLUSTER_ROWS=1, NUM_GLB_IACT=1, NUM_GLB_PSUM=4, NUM_GLB_WGHT=3,
        request=_Request(),
    )
