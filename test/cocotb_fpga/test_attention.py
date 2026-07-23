# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
Tests for the Phase-0 host-orchestrated attention flow.

Two levels:

1. test_attention_scheduler_software — pure Python, no simulator. Executes
   the full schedule with the bit-exact host model and checks the
   dequantized result against the float attention reference. Fast; covers
   the scheduler logic, scale bookkeeping and multi-head decomposition.

2. test_attention_fpga — full OpenEye_FPGA simulation. Every GEMV pass of
   the schedule runs through the DMA flow and is checked bit-exactly; the
   final result must equal the host integer model exactly. Slow (tens of
   passes, several minutes); run explicitly or in nightly CI:

       pytest test_attention.py::test_attention_fpga -v

Usage:
    pytest test_attention.py -v
    pytest test_attention.py -k software      # quick, no simulator
"""

import os

import numpy as np
import pytest
import cocotb_test.simulator

import open_eye.test_utils_main as ptu
import open_eye.vh_file_creator as vh_file_creator
import open_eye.generator as generator
from open_eye import hdl_dir, test_dir
from open_eye.attention_scheduler import AttentionScheduler

clk_cycle          = 20
clk_cycle_unit     = "ns"
clk_delay_in       = 100
clk_delay_unit_in  = "ps"
clk_delay_out      = 100
clk_delay_unit_out = "ps"


# ---------------------------------------------------------------------------
# 1. Software-only scheduler test (no simulator)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("SEQ_LEN,D_MODEL,NUM_HEADS", [
    (4, 8, 1),
    (4, 8, 2),
    (8, 16, 4),
    (16, 32, 4),
])
@pytest.mark.parametrize("SEED", [0, 1])
def test_attention_scheduler_software(SEQ_LEN, D_MODEL, NUM_HEADS, SEED):
    rng = np.random.default_rng(SEED)
    x   = rng.integers(-64, 64, size=(SEQ_LEN, D_MODEL))
    w_q = rng.integers(-64, 64, size=(D_MODEL, D_MODEL))
    w_k = rng.integers(-64, 64, size=(D_MODEL, D_MODEL))
    w_v = rng.integers(-64, 64, size=(D_MODEL, D_MODEL))
    w_o = rng.integers(-64, 64, size=(D_MODEL, D_MODEL))
    s = 1.0 / 64.0

    sched = AttentionScheduler(x, w_q, w_k, w_v, w_o, num_heads=NUM_HEADS,
                               s_x=s, s_w=s)
    out_int, out_scale = sched.run_on_host()

    assert out_int.shape == (SEQ_LEN, D_MODEL)
    assert sched.total_passes() == (3 + 2 * NUM_HEADS + 1) * SEQ_LEN

    # All on-chip operands must be INT8 (hardware constraint). Zeros are
    # allowed in every operand since the raw weight-stream fix (raw_wght).
    for stage_name, tensor in (("q8", sched.q8), ("k8", sched.k8),
                               ("v8", sched.v8), ("p8", sched.p8),
                               ("o8", sched.o8)):
        assert np.max(np.abs(tensor)) <= 127, f"{stage_name} exceeds INT8"

    # Dequantized result must track the float attention reference
    ref = sched.float_reference()
    hw = out_int.astype(np.float64) * out_scale
    err = np.linalg.norm(hw - ref) / max(np.linalg.norm(ref), 1e-12)
    assert err < 0.25, f"relative error {err} too large"

    # Determinism: a second host run reproduces the result bit-exactly
    sched2 = AttentionScheduler(x, w_q, w_k, w_v, w_o, num_heads=NUM_HEADS,
                                s_x=s, s_w=s)
    out2, scale2 = sched2.run_on_host()
    assert np.array_equal(out_int, out2) and out_scale == scale2


# ---------------------------------------------------------------------------
# 2. Full OpenEye_FPGA simulation (slow; every pass over the DMA flow)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("SEQ_LEN",   [4])
@pytest.mark.parametrize("D_MODEL",   [8])
@pytest.mark.parametrize("NUM_HEADS", [1, 2])
@pytest.mark.parametrize("CLUSTER_ROWS", [2])
@pytest.mark.parametrize("DATAFLOW", ["output_stationary"])
def test_attention_fpga(SEQ_LEN, D_MODEL, NUM_HEADS, CLUSTER_ROWS, DATAFLOW,
                        request):
    NUM_GLB_IACT, NUM_GLB_PSUM, NUM_GLB_WGHT = 3, 4, 3
    # Platform configuration; must be set before parameters.vh generation
    # (see test_conv3x3_sparse.py for the QUANT_AMOUNT rationale).
    os.environ["CLUSTER_ROWS"] = str(CLUSTER_ROWS)
    os.environ["NUM_GLB_IACT"] = str(NUM_GLB_IACT)
    os.environ["NUM_GLB_PSUM"] = str(NUM_GLB_PSUM)
    os.environ["NUM_GLB_WGHT"] = str(NUM_GLB_WGHT)
    os.environ["DATAFLOW"]     = DATAFLOW
    os.environ["BRANCHES"]        = "1"
    os.environ["BUFFER_WIDTH"]    = "12"
    os.environ["QUANT_AMOUNT"]    = "1024"
    os.environ["RAM_CELLS"]       = "32"
    os.environ["CLUSTER_COLUMNS"] = "2"

    toplevel = "OpenEye_FPGA"
    module   = "OpenEye_FPGA_attention_tb"

    nodeid = request.node.nodeid.replace("::", "_").replace("/", "_") \
                                .replace("[", "_").replace("]", "_")
    target_dir = os.path.join(test_dir, ".temp", nodeid)
    os.makedirs(target_dir, exist_ok=True)

    regmap_dir = os.path.join(test_dir, "cocotb_fpga")
    vh_file_creator.create_vh_file_from_envvars(
        target_dir, hdl_dir + "/", toplevel=toplevel)
    generator.create_regmap_params_vh_file(regmap_dir, target_dir, target_dir)

    # Compile the freshly generated register decoder (see
    # test_conv3x3_sparse.py for why hdl/dma_storage.v is excluded).
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
        testcase="start_test_attention",
        defines={"NO_TRACE": "TRUE"},
        force_compile=True,
        simulator="icarus",
        extra_env={
            "CLOCK_LEN":               str(clk_cycle),
            "CLOCK_UNIT":              clk_cycle_unit,
            "CLOCK_DELAY_INPUT":       str(clk_delay_in),
            "CLOCK_DELAY_UNIT_INPUT":  clk_delay_unit_in,
            "CLOCK_DELAY_OUTPUT":      str(clk_delay_out),
            "CLOCK_DELAY_UNIT_OUTPUT": clk_delay_unit_out,
            "SEQ_LEN":   str(SEQ_LEN),
            "D_MODEL":   str(D_MODEL),
            "NUM_HEADS": str(NUM_HEADS),
            "SEED":      "0",
            "DATAFLOW":     DATAFLOW,
            "CLUSTER_ROWS": str(CLUSTER_ROWS),
            "NUM_GLB_IACT": str(NUM_GLB_IACT),
            "NUM_GLB_PSUM": str(NUM_GLB_PSUM),
            "NUM_GLB_WGHT": str(NUM_GLB_WGHT),
            "LOGGER_LEVEL": "20",
            "COCOTB_LOG_FILE_PATH": os.path.join(target_dir, "cocotb_sim.log"),
        },
    )


if __name__ == "__main__":
    class _Node:
        nodeid = "attention_standalone"
    class _Request:
        node = _Node()
    test_attention_fpga(SEQ_LEN=4, D_MODEL=8, NUM_HEADS=1, CLUSTER_ROWS=2,
                        DATAFLOW="output_stationary", request=_Request())
