# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
End-to-end single-/multi-head attention on OpenEye_FPGA (Phase 0).

Runs a complete attention operation through the accelerator using the
host-orchestrated schedule from open_eye.attention_scheduler:

    stage 1: Q/K/V projections   (3 * seq_len GEMV passes on-chip)
    stage 2: S = Q K^T            (heads * seq_len passes on-chip,
                                   softmax + INT8 quantization on host)
    stage 3: O = P V              (heads * seq_len passes on-chip,
                                   requantization on host)
    stage 4: out = O W_O          (seq_len passes on-chip)

Every pass is executed through the full DMA flow (config words, iact,
weight, bias, quantize, offset streams) and its captured output is checked
bit-exactly against the integer reference. Dynamic operands (K, P, V)
produced by earlier passes are re-injected as the weight stream of later
passes — the host round-trip that Phase 1 (psum-to-weight feedback) will
remove. Finally the full hardware result is compared bit-exactly against
the host-executed integer schedule and, dequantized, against the float
attention reference.

Environment variables: SEQ_LEN, D_MODEL, NUM_HEADS, SEED plus the usual
platform configuration (see test_attention.py).
"""

import logging
import os
import types

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import with_timeout, SimTimeoutError

import open_eye.test_utils_main as tum
import open_eye.rtl_test_utils as rtl_test_utils
import open_eye.timing_parameters as tp
import open_eye.open_eye_parameters as oep
import open_eye.layer_parameters as lp
import open_eye.layer_execution_state as les_mod
import open_eye.DRAM as DRAM
from open_eye.attention_scheduler import AttentionScheduler

logger = logging.getLogger("cocotb")


def _dense_layer_shim(k, n):
    """Minimal stand-in for a Keras Dense layer, as far as LayerParameters
    needs it: name (dispatch), input/output shapes and kernel shape."""
    return types.SimpleNamespace(
        name="dense_attention_pass",
        input=types.SimpleNamespace(shape=(1, k)),
        output=types.SimpleNamespace(shape=(1, n)),
        kernel=types.SimpleNamespace(shape=(k, n)),
    )


# The dense datapath has a minimum problem size (8x8 GEMVs fail, 32x32 is
# the proven envelope from test_gemm_layer.py). Passes are zero-padded up
# to that size; padded inputs contribute 0 to every accumulation and padded
# output rows are discarded, so the real [N, K] result region stays
# bit-exact. (Zero padding in the weight matrix is fine since the raw
# weight-stream fix — see data_pipeline_wght.raw_mode_i.)
PAD_K = 32
PAD_N = 32


async def _run_gemm_pass(dut, ptp, params, spec, log_level):
    """Execute one GemmPass on the DUT through the full DMA flow.

    Returns the captured output vector (raw integer psums, length N).
    """
    k_pad = max(spec.k, PAD_K)
    n_pad = max(spec.n, PAD_N)
    weight = np.zeros((n_pad, k_pad), dtype=np.int64)
    weight[:spec.n, :spec.k] = spec.weight
    iact = np.zeros(k_pad, dtype=np.int64)
    iact[:spec.k] = spec.iact

    layer = _dense_layer_shim(k_pad, n_pad)
    layer_params = lp.LayerParameters([0], layer, params, 0, 1)

    dram = DRAM.DRAMContents([layer], [layer_params])
    dram.weights[0] = [[int(v) for v in row] for row in weight]
    dram.fmap[0] = [int(v) for v in iact]
    dram.bias[0] = [0 for _ in range(n_pad)]

    reference = tum.collect_results(0, layer_params, dram, params.SERIAL)

    dram_layer_content = [dram.fmap[0], dram.weights[0], dram.bias[0]]
    stream = tum.write_stream(params, layer_params, dram_layer_content, 0, 0)

    layer_es = les_mod.LayerExecutionState()
    await cocotb.start_soon(rtl_test_utils.send_stream(
        ptp, dut, stream[0], params, layer_params, 0))
    await cocotb.start_soon(rtl_test_utils.await_enable_signal(ptp, dut))
    await cocotb.start_soon(rtl_test_utils.compare_stream_Dense(
        ptp, dut, 0, 0, layer_params, params, layer_es, dram, log_level))

    # Hardware must match the DRAM-based reference over the full padded
    # output, and the scheduler's integer model over the real N outputs.
    assert tum.compare_dram_with_ref(layer_params, reference, dram.fmap[1]), \
        f"pass {spec.name}: DUT output differs from DRAM reference"
    captured = np.array([int(dram.fmap[1][f]) for f in range(spec.n)],
                        dtype=np.int64)
    expected = spec.expected_int()
    assert np.array_equal(captured, expected), \
        f"pass {spec.name}: DUT {captured} != expected {expected}"
    return captured


async def _run_attention(dut):
    seq_len = int(os.environ.get("SEQ_LEN", "4"))
    d_model = int(os.environ.get("D_MODEL", "8"))
    num_heads = int(os.environ.get("NUM_HEADS", "1"))
    seed = int(os.environ.get("SEED", "0"))
    try:
        log_level = int(os.getenv("LOGGER_LEVEL"))
    except (TypeError, ValueError):
        log_level = logging.INFO

    clk_cycle = int(os.environ["CLOCK_LEN"])
    clk_cycle_unit = os.environ["CLOCK_UNIT"]
    ptp = tp.PortTimingParameters()
    ptp.initiate_params(
        clk_cycle, clk_cycle_unit,
        int(os.environ["CLOCK_DELAY_INPUT"]), os.environ["CLOCK_DELAY_UNIT_INPUT"],
        int(os.environ["CLOCK_DELAY_OUTPUT"]), os.environ["CLOCK_DELAY_UNIT_OUTPUT"])

    params = oep.get_oep(1)  # serial / DMA mode

    # Moderate operand magnitudes leave psum headroom (|S| <= d_k * 63^2).
    rng = np.random.default_rng(seed)
    x = rng.integers(-64, 64, size=(seq_len, d_model))
    w_q = rng.integers(-64, 64, size=(d_model, d_model))
    w_k = rng.integers(-64, 64, size=(d_model, d_model))
    w_v = rng.integers(-64, 64, size=(d_model, d_model))
    w_o = rng.integers(-64, 64, size=(d_model, d_model))
    s_x = s_w = 1.0 / 64.0

    scheduler = AttentionScheduler(x, w_q, w_k, w_v, w_o,
                                   num_heads=num_heads, s_x=s_x, s_w=s_w)
    golden = AttentionScheduler(x, w_q, w_k, w_v, w_o,
                                num_heads=num_heads, s_x=s_x, s_w=s_w)
    golden_out, golden_scale = golden.run_on_host()

    clk = Clock(dut.clk_i, ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    cocotb.start_soon(clk.start())
    await cocotb.start_soon(
        rtl_test_utils.reset_all_signals(ptp, dut, params.SERIAL))

    executed = 0
    for stage in scheduler.stages():
        dut._log.info("Attention stage '%s': %d passes",
                      stage.name, len(stage.passes))
        for spec in stage.passes:
            spec.result = await _run_gemm_pass(dut, ptp, params, spec,
                                               log_level)
            executed += 1
            dut._log.info("pass %s ok (%d/%d)", spec.name, executed,
                          scheduler.total_passes())
        stage.finalize()

    out_int, out_scale = scheduler.final_output()

    # 1) Bit-exact: hardware-executed schedule == host-executed schedule.
    assert np.array_equal(out_int, golden_out), \
        "hardware attention differs from host integer reference"
    assert out_scale == golden_scale

    # 2) Numerically sane: dequantized output tracks the float reference.
    float_ref = scheduler.float_reference()
    hw_float = out_int.astype(np.float64) * out_scale
    err = np.linalg.norm(hw_float - float_ref) / \
        max(np.linalg.norm(float_ref), 1e-12)
    dut._log.info("attention relative error vs float reference: %.4f", err)
    assert err < 0.25, f"quantized attention error too large: {err}"

    assert dut.rst_ni.value == 1, "rst_ni is not 1!"


@cocotb.test()
async def start_test_attention(dut):
    """Cocotb entry point: full attention operation, Phase 0 schedule."""
    timeout_time = 200_000_000  # ns of simulated time for all passes
    timeout_unit = "ns"
    try:
        await with_timeout(_run_attention(dut), timeout_time, timeout_unit)
    except SimTimeoutError:
        dut._log.error("Attention test did not finish in time!")
        raise
