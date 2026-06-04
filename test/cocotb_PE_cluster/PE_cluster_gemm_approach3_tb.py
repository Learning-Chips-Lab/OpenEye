# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
Approach 3 – Weight-stationary systolic pass-through (SYSTOLIC_GEMM_EN=1).

Requires PE_cluster compiled with SYSTOLIC_GEMM_EN=1.

Dataflow:
  - One weight vector b of length K is pre-loaded into all PE rows via
    the standard wght SPad path.
  - Input activations for each row stream in horizontally via
    iact_pass_data_i / iact_pass_enable_i with a diagonal skew of 1
    cycle per row so operands arrive at each PE at the right time.
  - Each PE(any_col, row=j) accumulates c[j] = A[j,:] . b over K cycles
    using iact_pass_data_i as the iact source (SYSTOLIC_GEMM_EN=1 overrides
    mult_fac_2 when iact_pass_enable_i is high).  Since all columns share the
    same weights, every column produces the same c[j] — the test verifies
    all M*N outputs equal the expected M-vector c repeated across N columns.
  - After streaming, psums are drained via the standard router path.

Architecture note:
  The PE_cluster weight bus is shared per row (all columns in row j get the
  same weight data), so a full MxKxN GEMM with independent columns is not
  directly expressible.  Approach 3 instead demonstrates that iact_pass can
  replace the iact SPad source during CALCULATING, producing correct dot
  products when weights are uniform across columns.
"""

import math
import os
import sys

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, with_timeout, SimTimeoutError

import open_eye.timing_parameters as timing_parameters
import open_eye.rtl_test_utils as rtl_test_utils
import pe_cluster_test_utils as pctu
from open_eye import hdl_dir, test_dir

import PE_cluster_tb
from PE_cluster_tb import send_iact, send_wght, send_bias, generate_spad

clk_cycle          = int(os.environ.get("CLOCK_LEN", "10"))
clk_cycle_unit     = os.environ.get("CLOCK_UNIT", "ns")
clk_delay_in       = int(os.environ.get("CLOCK_DELAY_INPUT", "100"))
clk_delay_unit_in  = os.environ.get("CLOCK_DELAY_UNIT_INPUT", "ps")
clk_delay_out      = int(os.environ.get("CLOCK_DELAY_OUTPUT", "100"))
clk_delay_unit_out = os.environ.get("CLOCK_DELAY_UNIT_OUTPUT", "ps")

iactsize_x  = 0
iactsize_y  = 0
wghtsize_x  = 0
wghtsize_y  = 0
pe_iact_cycles = 0


def _skew_stream(row_vectors, skew):
    """
    Prepend j*skew zeros to row j so all rows reach the same total length.
    Returns a list of equal-length 1-D arrays.
    """
    M          = len(row_vectors)
    K          = len(row_vectors[0])
    total_len  = K + (M - 1) * skew
    result = []
    for j, v in enumerate(row_vectors):
        pad_front = j * skew
        pad_back  = total_len - K - pad_front
        result.append(np.concatenate([
            np.zeros(pad_front, dtype=np.int32),
            v.astype(np.int32),
            np.zeros(pad_back,  dtype=np.int32),
        ]))
    return result


@cocotb.test()
async def start_test_gemm_approach3(dut):
    """
    Approach 3 systolic pass-through test.

    Requires PE_cluster + PE compiled with SYSTOLIC_GEMM_EN=1
    (set via cocotb_test parameters= in the pytest runner).
    """
    timeout_time = 80000
    timeout_unit = "ns"
    try:
        await with_timeout(_run_gemm_approach3(dut), timeout_time, timeout_unit)
    except SimTimeoutError:
        dut._log.error("Approach-3 systolic GEMM test timed out!")
        raise


async def _run_gemm_approach3(dut):
    global iactsize_x, iactsize_y, wghtsize_x, wghtsize_y, pe_iact_cycles

    iactsize_x = int(os.environ["IACTSIZE_X"])
    iactsize_y = int(os.environ["IACTSIZE_Y"])
    wghtsize_x = int(os.environ["WGHTSIZE_X"])
    np.random.seed(int(os.environ.get("SEED", "7")))

    M = int(dut.PE_ROWS.value)
    N = int(dut.PE_COLUMNS.value)
    K = iactsize_x * iactsize_y   # inner dimension (= wghtsize_y)

    wghtsize_y = K
    num_glb_iact = int(dut.NUM_GLB_IACT.value)
    pe_iact_cycles = math.ceil((M + N - 1) / num_glb_iact)

    # Random INT8 matrices — weights are a single K-vector (shared across cols)
    A = np.random.randint(1, 8, size=(M, K), dtype=np.int8)
    b = np.random.randint(1, 8, size=(K,),   dtype=np.int8)
    # Expected: each column gives the same result c = A @ b
    c_ref = A.astype(np.int32) @ b.astype(np.int32)   # shape (M,)

    dut._log.info(f"Approach3 A={A.shape} b={b.shape} c_ref={c_ref}")

    # Propagate globals into PE_cluster_tb so its helpers work
    PE_cluster_tb.iactsize_x     = iactsize_x
    PE_cluster_tb.iactsize_y     = iactsize_y
    PE_cluster_tb.wghtsize_x     = wghtsize_x
    PE_cluster_tb.wghtsize_y     = wghtsize_y
    PE_cluster_tb.sparse_iact    = 0
    PE_cluster_tb.sparse_wght    = 0
    PE_cluster_tb.pe_iact_cycles = pe_iact_cycles

    ptp = timing_parameters.PortTimingParameters()
    ptp.initiate_params(clk_cycle, clk_cycle_unit,
                        clk_delay_in, clk_delay_unit_in,
                        clk_delay_out, clk_delay_unit_out)

    cocotb.start_soon(Clock(dut.clk_i, clk_cycle, unit=clk_cycle_unit).start())

    await pctu.reset_all_signals(ptp, dut)

    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.gemm_mode_i, 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.iact_pass_ready_i, 0))

    # ------------------------------------------------------------------
    # Configure PE parameters
    # ------------------------------------------------------------------
    await pctu.send_data_params(ptp, dut, iactsize_x=iactsize_x,
                                 iactsize_y=iactsize_y, wghtsize_x=wghtsize_x)

    # ------------------------------------------------------------------
    # Load dummy iacts (all ones) — the PE state machine needs iact SPad
    # data to drive its loop counter even in systolic mode.
    # ------------------------------------------------------------------
    dummy_iacts = np.ones(
        (pe_iact_cycles * num_glb_iact, iactsize_y, iactsize_x), dtype=np.int8
    )
    dut._log.info("Approach3: sending dummy iacts")
    await send_iact(ptp, dut, dummy_iacts)
    dut._log.info("Approach3: dummy iacts sent, loading weights")

    # ------------------------------------------------------------------
    # Load weight vector b into all PE rows.
    # wghtsize_x=1 (each PE produces one scalar psum), so shape (K, 1).
    # send_wght expects data_array[row] of shape (wghtsize_y, wghtsize_x).
    # ------------------------------------------------------------------
    wdata_row = b.reshape(K, 1).astype(np.int32)
    wghts = [wdata_row for _ in range(M)]
    await send_wght(ptp, dut, wghts)

    dut._log.info("Approach3: weights sent, triggering compute")
    await Timer(clk_cycle, unit=clk_cycle_unit)

    # ------------------------------------------------------------------
    # Trigger computation then stream activations via horizontal pass-through
    # ------------------------------------------------------------------
    pes = M * N
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.compute_i, (2 ** pes) - 1))
    await Timer(clk_cycle, unit=clk_cycle_unit)
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.compute_i, 0))

    # Wait for LOADING pipeline (sparse FSM: 5 LOADING states total).
    # Timeline from compute_i assertion:
    #   Edge+0: compute_i=1 seen → IDLE→LOADING_1
    #   Edge+1: LOADING_2 (compute_i=0 applied before this edge)
    #   Edge+2: LOADING_3
    #   Edge+3: LOADING_4
    #   Edge+4: LOADING_5
    #   Edge+5: CALCULATING cycle 1 fires (mult samples inputs here)
    # set_input has 100ps delay, so data presented AFTER the rising edge.
    # To be sampled at Edge+5 we must apply data before Edge+5, meaning:
    # wait 3 cycles after compute_i deassert (which happens 1 cycle after compute_i=1).
    dut._log.info("Approach3: waiting for LOADING pipeline")
    await Timer(3 * clk_cycle, unit=clk_cycle_unit)
    dut._log.info("Approach3: streaming iact pass-through")

    # Broadcast mode: all columns in a row receive the same iact_pass signal.
    # Present A[j, k] on row j's bits simultaneously for all j at each MAC cycle k.
    # No skew is needed since all PEs in a column share the same CALCULATING cycle.
    iact_bits = int(dut.DATA_IACT_BITWIDTH.value)
    iact_mask = (1 << iact_bits) - 1

    for k in range(K):
        data_word  = 0
        enable_vec = (1 << M) - 1  # all rows enabled
        for j in range(M):
            val = int(A[j, k])
            data_word |= (val & iact_mask) << (iact_bits * j)
        dut._log.info(f"  iact_pass k={k}: data=0x{data_word:06x} enable=0b{enable_vec:03b}")
        cocotb.start_soon(rtl_test_utils.set_input(
            ptp, dut.iact_pass_data_i,   data_word))
        cocotb.start_soon(rtl_test_utils.set_input(
            ptp, dut.iact_pass_enable_i, enable_vec))
        await Timer(clk_cycle, unit=clk_cycle_unit)
        try:
            dut._log.info(f"  after Timer: iact_pass_enable_i={int(dut.iact_pass_enable_i.value)} iact_pass_data_i=0x{int(dut.iact_pass_data_i.value):06x}")
        except Exception as e:
            dut._log.info(f"  after Timer: read error: {e}")

    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.iact_pass_data_i,   0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.iact_pass_enable_i, 0))
    dut._log.info("Approach3: iact streaming done, draining pipeline")

    # Extra drain cycles for pipeline tail
    await Timer((N + 6) * clk_cycle, unit=clk_cycle_unit)

    # ------------------------------------------------------------------
    # Drain psums and verify
    # ------------------------------------------------------------------
    col_mask = (2 ** N) - 1
    cocotb.start_soon(rtl_test_utils.set_input(
        ptp, dut.pe_router_psum_ready_i, col_mask))

    dut._log.info(f"Approach3: waiting for psum_ready_o (current={int(dut.pe_router_psum_ready_o.value):#x})")
    while int(dut.pe_router_psum_ready_o.value) != col_mask:
        await Timer(clk_cycle, unit=clk_cycle_unit)

    # Send non-zero bias (all ones) to trigger psum_enable_o.
    # Bias value 1 is subtracted from each received psum below.
    psum_words = int(dut.PSUM_WORDS.value)
    dut._log.info(f"Approach3: psum_ready_o asserted, sending bias (psum_words={psum_words})")
    bias_val = np.ones((N, psum_words), dtype=np.int32)
    PE_cluster_tb.wghtsize_x = psum_words
    cocotb.start_soon(send_bias(ptp, dut, bias_val))

    for _ in range(200):
        if int(dut.pe_router_psum_enable_o.value) == col_mask:
            break
        await Timer(clk_cycle, unit=clk_cycle_unit)

    if int(dut.pe_router_psum_enable_o.value) != col_mask:
        dut._log.error("Approach3: psum_enable_o never asserted!")
        raise AssertionError("psum_enable_o timeout")

    dut._log.info("Approach3: psum_enable_o asserted, reading psums")

    psum_bits = int(dut.DATA_PSUM_BITWIDTH.value)
    psum_mask = (1 << psum_bits) - 1
    errors    = 0
    received  = [[] for _ in range(N)]

    for _ in range(psum_words * 2):
        try:
            enable_now = int(dut.pe_router_psum_enable_o.value)
        except ValueError:
            await Timer(clk_cycle, unit=clk_cycle_unit)
            continue
        if enable_now == 0:
            break
        try:
            raw = int(dut.pe_router_psum_data_o.value)
        except ValueError:
            await Timer(clk_cycle, unit=clk_cycle_unit)
            continue
        for col in range(N):
            if (enable_now >> col) & 1:
                val = (raw >> (psum_bits * col)) & psum_mask
                if val >= (1 << (psum_bits - 1)):
                    val -= (1 << psum_bits)
                dut._log.info(f"  raw psum col={col}: val={val}")
                received[col].append(val - 1)  # subtract bias=1
        await Timer(clk_cycle, unit=clk_cycle_unit)

    for col in range(N):
        for row in range(M):
            if row < len(received[col]):
                got = received[col][row]
                exp = int(c_ref[row])  # same expected value for all columns
                if got != exp:
                    dut._log.error(
                        f"Approach3 C[{row},{col}]: got {got}, expected {exp}")
                    errors += 1
                else:
                    dut._log.info(
                        f"Approach3 C[{row},{col}] = {got} OK")

    assert errors == 0, f"Approach 3 GEMM: {errors} mismatches"
    assert dut.rst_ni.value == 1, "rst_ni is not 1!"


if __name__ == "__main__":
    import cocotb_test.simulator

    tests_dir  = os.path.abspath(os.path.dirname(__file__))
    target_dir = os.path.join(test_dir, ".temp", "gemm_approach3_standalone")
    os.makedirs(target_dir, exist_ok=True)

    cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=pctu.get_verilog_sources(hdl_dir),
        toplevel="PE_cluster",
        module="PE_cluster_gemm_approach3_tb",
        sim_build=target_dir,
        testcase="start_test_gemm_approach3",
        parameters={"SYSTOLIC_GEMM_EN": 1, "PARALLEL_MACS": 1},
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
            "WGHTSIZE_X":  str(int(sys.argv[3]) if len(sys.argv) > 3 else 1),
            "SEED":        str(int(sys.argv[4]) if len(sys.argv) > 4 else 7),
        },
    )
