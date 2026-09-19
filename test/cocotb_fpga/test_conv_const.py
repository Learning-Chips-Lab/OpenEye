# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Smallest convolutions the FPGA top level can run, with constant operands.

Why this exists
---------------
Every other FPGA conv test uses random weights and activations, so when the
result is wrong the only thing the failure tells you is "wrong". With every
activation and every weight forced to one constant c, each output becomes a
*count*:

    out[f][y][x] = (terms that actually accumulated) * c**2 + bias[f]

and for an interior pixel the correct count is KERNEL_X * KERNEL_Y * CHANNELS.
A DUT value short by a whole factor therefore names how many products reached
the MACs, which is the question random data cannot answer. That is exactly how
the gemm shortfall was pinned down to 9 of 32 products (see
OPENEYE_CONST_IACTS / OPENEYE_CONST_WGHTS in OpenEye_FPGA_tb.py).

The two tests
-------------
test_conv_const_single_layer   LAYER=Convolution_Single, one conv layer. Checks the
                               compute and read-out path end to end with no
                               interlayer step involved.

test_conv_const_two_layers     LAYER=Convolution_Stack, two stacked conv
                               layers. The first layer's psums are quantised
                               and written back into the iact buffer as the
                               second layer's input, so this is the only
                               focused test of the interlayer write-back. It
                               has no pooling layer in between: in the MNIST
                               net the failing interlayer check sits after a
                               Pooling layer, so conv and pooling cannot be
                               told apart there.

The explicit Convolution_Single mode and layer-count assertion keep the
control independent of the legacy Convolution model, which has two layers.
The ramp tests additionally distinguish spatial positions and output channels.

The September 19 debugging milestone targets CLUSTER_ROWS=2, NUM_GLB_IACT=1,
INPUT_CHANNELS=4. The one-channel geometry and CLUSTER_ROWS=1 remain separate
regressions; see doc/test_status_handover.md for measured results.

Both tests are tiny - an 8-wide input and 4 filters - so a case runs in about
a minute and they can be iterated on while debugging, unlike the MNIST net
that test_single_layers.py builds by default.

Usage:
    pytest test_conv_const.py -v
    pytest test_conv_const.py -v -k two_layers
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

# Fixed geometry shared by both tests: small enough to run in about a minute.
NUM_FILTERS   = 4
STRIDE        = 1
KERNEL_SIZE_X = 3
KERNEL_SIZE_Y = 3
INPUT_SIZE_X  = 8
INPUT_SIZE_Y  = 1
NUM_GLB_PSUM  = 4
NUM_GLB_WGHT  = 3


def _run_conv_const(layer_mode, const_value, cluster_rows, num_glb_iact,
                    input_channels, request, ramp_iacts=False,
                    psum_width=20, trans_words=8, bias_step=0):
    """Build and simulate one constant-operand conv configuration."""
    # OpenEyeParameters and parameters.vh read these from the environment at
    # generation time, so they have to be set before create_vh_file_from_envvars.
    os.environ["CLUSTER_ROWS"] = str(cluster_rows)
    os.environ["NUM_GLB_IACT"] = str(num_glb_iact)
    os.environ["NUM_GLB_PSUM"] = str(NUM_GLB_PSUM)
    os.environ["NUM_GLB_WGHT"] = str(NUM_GLB_WGHT)
    os.environ["DATA_PSUM_BITWIDTH"] = str(psum_width)
    os.environ["TRANS_WORDS"] = str(trans_words)
    # Fixed platform configuration, matching the other cocotb_fpga runners.
    # Use the standard quantization table: 1024 entries of 40 bits produce
    # 640 DMA words, matching GET_QUANTIZE's derived transfer count.
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
            "LAYER":             layer_mode,
            "OPENEYE_EXPECT_LAYERS": "1" if layer_mode == "Convolution_Single" else "2",
            "NUM_FILTERS":       str(NUM_FILTERS),
            "STRIDE":            str(STRIDE),
            "KERNEL_SIZE_X":     str(KERNEL_SIZE_X),
            "KERNEL_SIZE_Y":     str(KERNEL_SIZE_Y),
            "INPUT_SIZE_X":      str(INPUT_SIZE_X),
            "INPUT_SIZE_Y":      str(INPUT_SIZE_Y),
            "INPUT_CHANNELS":    str(input_channels),
            # Dense (non-sparse) data: zero-skipping would make the product
            # count depend on the sparsity pattern and defeat the whole point.
            "USE_RANDOM_VALUES": "1",
            "USE_SPARSE_IACTS":  "0",
            "USE_SPARSE_WGHTS":  "0",
            "USE_SPARSE_WEIGHTS": "0",
            # The two hooks that make the arithmetic interpretable. The
            # reference is recomputed from the same DRAM, so it stays exact.
            "OPENEYE_CONST_IACTS": str(const_value),
            "OPENEYE_CONST_WGHTS": str(const_value),
            "OPENEYE_RAMP_IACTS": "1" if ramp_iacts else "",
            "OPENEYE_BIAS_STEP": str(bias_step) if bias_step else "",
            # Fail a hang in minutes instead of running to the 15 ms sim
            # timeout; the stall report names the FSM states and per-PE counts.
            "OPENEYE_PROBE_FSM":      "1",
            "OPENEYE_FAIL_ON_STALL":  "100000",
            "CLUSTER_ROWS": str(cluster_rows),
            "NUM_GLB_IACT": str(num_glb_iact),
            "NUM_GLB_PSUM": str(NUM_GLB_PSUM),
            "NUM_GLB_WGHT": str(NUM_GLB_WGHT),
            "LOGGER_LEVEL": "10",
            "COCOTB_LOG_FILE_PATH": os.path.join(target_dir, "cocotb_sim.log"),
            "COCOTB_TRACE": "1",
        },
    )


# c = 1 reads directly as a product count; c = 2 guards against a fault that is
# invisible when every operand is 1 (a dropped multiply, say).
@pytest.mark.parametrize("CONST_VALUE", [1, 2, 8])
# CLUSTER_ROWS=1 keeps the mapping to one cluster row; CLUSTER_ROWS=2 adds the
# multi-row psum accumulation chain and nothing else, so the pair isolates it.
@pytest.mark.parametrize("CLUSTER_ROWS", [1, 2])
def test_conv_const_single_layer(CONST_VALUE, CLUSTER_ROWS, request):
    """One conv layer: compute and read-out, no interlayer write-back."""
    _run_conv_const("Convolution_Single", CONST_VALUE, CLUSTER_ROWS,
                    num_glb_iact=1, input_channels=4, request=request)


# 1 and 2 quantise to the same interlayer byte, so a write-back that ignores
# its input looks identical to a reference that happens to be constant. Only a
# value far apart (8) shows whether either side tracks the operands, which is
# how the data-independent write-back was proven. c = 2 adds nothing here.
@pytest.mark.parametrize("CONST_VALUE", [1, 8])
@pytest.mark.parametrize("CLUSTER_ROWS", [1, 2])
# Two geometries. (iact=3, ch=1) is the configuration that first reproduced the
# data-independent write-back; (iact=1, ch=4) matches the single-layer test, so
# a difference between the tests is not a difference in geometry.
@pytest.mark.parametrize("NUM_GLB_IACT,INPUT_CHANNELS", [(3, 1), (1, 4)],
                         ids=["iact3_ch1", "iact1_ch4"])
def test_conv_const_two_layers(CONST_VALUE, CLUSTER_ROWS, NUM_GLB_IACT,
                               INPUT_CHANNELS, request):
    """Two stacked conv layers: exercises the interlayer psum->iact write-back.

    The testbench checks the iact buffer the second layer reads before that
    layer runs (rtl_test_utils.compare_iact_storage), so a write-back failure
    is reported there, ahead of the final output comparison.
    """
    _run_conv_const("Convolution_Stack", CONST_VALUE, CLUSTER_ROWS,
                    num_glb_iact=NUM_GLB_IACT, input_channels=INPUT_CHANNELS,
                    request=request)


@pytest.mark.parametrize("psum_width", [20, 32])
@pytest.mark.parametrize("trans_words", [4, 8])
def test_conv_ramp_writeback(psum_width, trans_words, request):
    """Nonuniform pixels expose ordering errors hidden by constant operands.

    Check the intermediate activation RAM and the second layer's DMA output,
    with both a partial and a full DMA payload and extra quantizer lanes.
    """
    _run_conv_const("Convolution_Stack", 8, 2, num_glb_iact=1,
                    input_channels=4, request=request, ramp_iacts=True,
                    psum_width=psum_width, trans_words=trans_words)


def test_conv_channel_order(request):
    """Distinguish spatial positions and output channels across write-back."""
    _run_conv_const("Convolution_Stack", 8, 2, num_glb_iact=1,
                    input_channels=4, request=request, ramp_iacts=True,
                    bias_step=128)


if __name__ == "__main__":
    class _Node:
        nodeid = "conv_const_standalone"

    class _Request:
        node = _Node()

    test_conv_const_two_layers(
        CONST_VALUE=8, CLUSTER_ROWS=2, NUM_GLB_IACT=3, INPUT_CHANNELS=1,
        request=_Request(),
    )
