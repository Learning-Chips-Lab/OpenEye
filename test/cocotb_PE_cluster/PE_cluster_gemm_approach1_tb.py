# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
Approach 1 – GEMM via existing conv dataflow (zero RTL changes).

Maps C = A × B onto PE_cluster by reusing the standard conv dataflow with
gemm_mode_i held low.  The iact GLB-bank routing is driven by send_iact()
(imported from PE_cluster_tb) exactly as in the conv test, and the same
reference model in get_psum() validates the result.  No RTL changes and no
custom iact_choose_i override are needed: the conv routing already realises
the desired row->bank mapping, and overriding iact_choose_i before send_iact
had no effect because send_iact reprograms it while streaming.

All SPad packing, parameter streaming and output collection reuse the
existing PE_cluster_tb helpers exactly.
"""

import math
import os
import sys

# Provide sane defaults so importing PE_cluster_tb (which reads these at
# import time) does not crash when this module is imported by the host
# interpreter (e.g. the __main__ runner) before the simulator sets them.
os.environ.setdefault("CLOCK_LEN", "10")
os.environ.setdefault("CLOCK_UNIT", "ns")
os.environ.setdefault("CLOCK_DELAY_INPUT", "100")
os.environ.setdefault("CLOCK_DELAY_UNIT_INPUT", "ps")
os.environ.setdefault("CLOCK_DELAY_OUTPUT", "100")
os.environ.setdefault("CLOCK_DELAY_UNIT_OUTPUT", "ps")

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, Combine, with_timeout, SimTimeoutError

import open_eye.timing_parameters as timing_parameters
import open_eye.rtl_test_utils as rtl_test_utils
import pe_cluster_test_utils as pctu
from open_eye import hdl_dir, test_dir

# ---------------------------------------------------------------------------
# Timing – read from environment (set by pytest runner or __main__ block)
# ---------------------------------------------------------------------------
clk_cycle          = int(os.environ.get("CLOCK_LEN", "10"))
clk_cycle_unit     = os.environ.get("CLOCK_UNIT", "ns")
clk_delay_in       = int(os.environ.get("CLOCK_DELAY_INPUT", "100"))
clk_delay_unit_in  = os.environ.get("CLOCK_DELAY_UNIT_INPUT", "ps")
clk_delay_out      = int(os.environ.get("CLOCK_DELAY_OUTPUT", "100"))
clk_delay_unit_out = os.environ.get("CLOCK_DELAY_UNIT_OUTPUT", "ps")

# Test dimensions – read from environment
iactsize_x  = 0
iactsize_y  = 0
wghtsize_x  = 0
wghtsize_y  = 0
sparse_iact = 0
sparse_wght = 0
pe_iact_cycles = 0


# ────────────────────────────────────────────────────────────────────────────
# Reuse send_iact / send_wght / send_bias / get_psum / generate_spad from
# PE_cluster_tb verbatim – just import them from there.
# ────────────────────────────────────────────────────────────────────────────
import PE_cluster_tb
from PE_cluster_tb import (
    send_iact,
    send_wght,
    send_bias,
    get_psum,
    generate_spad,
    create_iact_wght_psum_arrays,
)


@cocotb.test()
async def start_test_gemm_approach1(dut):
    """
    Approach 1 GEMM test.

    Uses the existing conv dataflow (iact_choose_i driven by send_iact) with
    gemm_mode_i held low.  No RTL changes are needed.
    """
    timeout_time = 40000
    timeout_unit = "ns"
    try:
        await with_timeout(_run_gemm_approach1(dut), timeout_time, timeout_unit)
    except SimTimeoutError:
        dut._log.error("Approach-1 GEMM test timed out!")
        raise


async def _run_gemm_approach1(dut):
    global iactsize_x, iactsize_y, wghtsize_x, wghtsize_y
    global sparse_iact, sparse_wght, pe_iact_cycles

    iactsize_x  = int(os.environ["IACTSIZE_X"])
    iactsize_y  = int(os.environ["IACTSIZE_Y"])
    wghtsize_x  = int(os.environ["WGHTSIZE_X"])
    sparse_iact = int(os.environ.get("SPARSE_IACT", "0"))
    sparse_wght = int(os.environ.get("SPARSE_WGHT", "0"))
    np.random.seed(int(os.environ.get("SEED", "0")))

    pe_iact_cycles = math.ceil(
        (int(dut.PE_ROWS.value) + int(dut.PE_COLUMNS.value) - 1)
        / int(dut.NUM_GLB_IACT.value)
    )
    wghtsize_y = iactsize_x * iactsize_y

    # Propagate globals into PE_cluster_tb module namespace so its helpers work
    PE_cluster_tb.iactsize_x    = iactsize_x
    PE_cluster_tb.iactsize_y    = iactsize_y
    PE_cluster_tb.wghtsize_x    = wghtsize_x
    PE_cluster_tb.wghtsize_y    = wghtsize_y
    PE_cluster_tb.sparse_iact   = sparse_iact
    PE_cluster_tb.sparse_wght   = sparse_wght
    PE_cluster_tb.pe_iact_cycles = pe_iact_cycles

    ptp = timing_parameters.PortTimingParameters()
    ptp.initiate_params(clk_cycle, clk_cycle_unit,
                        clk_delay_in, clk_delay_unit_in,
                        clk_delay_out, clk_delay_unit_out)

    cocotb.start_soon(Clock(dut.clk_i, clk_cycle, unit=clk_cycle_unit).start())

    (iacts, wghts, psums) = create_iact_wght_psum_arrays(dut)

    await pctu.reset_all_signals(ptp, dut)

    # Approach 1: keep gemm_mode_i low.  iact_choose_i is driven by send_iact
    # (the standard conv routing); no separate override is applied here.
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.gemm_mode_i, 0))

    # Also initialise new systolic ports (tied off; SYSTOLIC_GEMM_EN=0)
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.iact_pass_data_i,   0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.iact_pass_enable_i, 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.iact_pass_ready_i,  0))

    await pctu.send_data_params(ptp, dut, iactsize_x, iactsize_y, wghtsize_x)

    send_iact_thread = cocotb.start_soon(send_iact(ptp, dut, iacts))
    send_wght_thread = cocotb.start_soon(send_wght(ptp, dut, wghts))
    await Combine(send_iact_thread, send_wght_thread)

    await Timer(clk_cycle, unit=clk_cycle_unit)

    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.compute_i,
                                                (2 ** (int(dut.PE_ROWS.value) * int(dut.PE_COLUMNS.value))) - 1))
    await Timer(clk_cycle, unit=clk_cycle_unit)
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.compute_i, 0))
    await Timer(3 * clk_cycle, unit=clk_cycle_unit)

    cocotb.start_soon(rtl_test_utils.set_input(
        ptp, dut.pe_router_psum_ready_i, (2 ** int(dut.PE_COLUMNS.value)) - 1))

    col_mask = (2 ** int(dut.PE_COLUMNS.value)) - 1
    while int(dut.pe_router_psum_ready_o.value) != col_mask:
        await Timer(clk_cycle, unit=clk_cycle_unit)

    cocotb.start_soon(send_bias(ptp, dut, psums))

    while int(dut.pe_router_psum_enable_o.value) != col_mask:
        await Timer(clk_cycle, unit=clk_cycle_unit)

    # Await get_psum so its internal PSUM-vs-reference assertion is actually
    # enforced.  Launching it with start_soon and never awaiting it (as the
    # original conv test does) lets the test pass even when PSUMs are wrong,
    # because cocotb ignores exceptions raised by orphaned coroutines.
    get_psum_thread = cocotb.start_soon(get_psum(ptp, dut, iacts, wghts, psums))

    while int(dut.pe_router_psum_enable_o.value) != 0:
        await Timer(clk_cycle, unit=clk_cycle_unit)

    await get_psum_thread

    assert dut.rst_ni.value == 1, "rst_ni is not 1!"


if __name__ == "__main__":
    import cocotb_test.simulator

    tests_dir  = os.path.abspath(os.path.dirname(__file__))
    target_dir = os.path.join(test_dir, ".temp", "gemm_approach1_standalone")
    os.makedirs(target_dir, exist_ok=True)

    cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=pctu.get_verilog_sources(hdl_dir),
        toplevel="PE_cluster",
        module="PE_cluster_gemm_approach1_tb",
        sim_build=target_dir,
        testcase="start_test_gemm_approach1",
        defines={"NO_TRACE": "TRUE"},
        force_compile=True,
        simulator="icarus",
        extra_env={
            "CLOCK_LEN":              "10",
            "CLOCK_UNIT":             "ns",
            "CLOCK_DELAY_INPUT":      "100",
            "CLOCK_DELAY_UNIT_INPUT": "ps",
            "CLOCK_DELAY_OUTPUT":     "100",
            "CLOCK_DELAY_UNIT_OUTPUT": "ps",
            "IACTSIZE_X":  str(int(sys.argv[1]) if len(sys.argv) > 1 else 4),
            "IACTSIZE_Y":  str(int(sys.argv[2]) if len(sys.argv) > 2 else 1),
            "WGHTSIZE_X":  str(int(sys.argv[3]) if len(sys.argv) > 3 else 6),
            "SPARSE_IACT": "0",
            "SPARSE_WGHT": "0",
            "SEED":        str(int(sys.argv[4]) if len(sys.argv) > 4 else 0),
        },
    )
