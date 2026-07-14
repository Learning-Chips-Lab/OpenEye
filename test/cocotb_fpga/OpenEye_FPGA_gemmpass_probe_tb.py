# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
Probe: which zero-valued operands does the dense DMA datapath tolerate?

Runs a single 32x32 GEMV with zeros injected according to ZERO_MODE:
  none  - all operands nonzero (control, must pass)
  iact  - 8 zero entries in the input vector
  wght  - an 8x8 zero block in the weight matrix

Used to pin down the operand constraints of the Phase-0 attention executor
(the DRAM writer historically replaces dense zeros with +-1, suggesting
the hardware path has never been exercised with zero operands).
"""

import os

import numpy as np
import cocotb
from cocotb.triggers import with_timeout, SimTimeoutError
from cocotb.clock import Clock

import open_eye.rtl_test_utils as rtl_test_utils
import open_eye.timing_parameters as tp
import open_eye.open_eye_parameters as oep

from OpenEye_FPGA_attention_tb import _run_gemm_pass
from open_eye.attention_scheduler import GemmPass


async def _run_probe(dut):
    zero_mode = os.environ.get("ZERO_MODE", "none")
    rng = np.random.default_rng(7)

    def nonzero(shape):
        v = rng.integers(1, 64, size=shape)
        return v * rng.choice([-1, 1], size=shape)

    weight = nonzero((32, 32))
    iact = nonzero(32)
    if zero_mode == "iact":
        iact[8:16] = 0
    elif zero_mode == "wght":
        weight[8:16, 8:16] = 0

    clk_cycle = int(os.environ["CLOCK_LEN"])
    ptp = tp.PortTimingParameters()
    ptp.initiate_params(
        clk_cycle, os.environ["CLOCK_UNIT"],
        int(os.environ["CLOCK_DELAY_INPUT"]), os.environ["CLOCK_DELAY_UNIT_INPUT"],
        int(os.environ["CLOCK_DELAY_OUTPUT"]), os.environ["CLOCK_DELAY_UNIT_OUTPUT"])
    params = oep.get_oep(1)

    cocotb.start_soon(Clock(dut.clk_i, ptp.clk_cycle,
                            unit=ptp.clk_cycle_unit).start())
    await cocotb.start_soon(
        rtl_test_utils.reset_all_signals(ptp, dut, params.SERIAL))

    spec = GemmPass(f"probe_{zero_mode}", weight, iact)
    result = await _run_gemm_pass(dut, ptp, params, spec, 20)
    assert np.array_equal(result, spec.expected_int())


@cocotb.test()
async def start_test_probe(dut):
    try:
        await with_timeout(_run_probe(dut), 1_000_000, "ns")
    except SimTimeoutError:
        dut._log.error("probe timed out (FSM hang)")
        raise
