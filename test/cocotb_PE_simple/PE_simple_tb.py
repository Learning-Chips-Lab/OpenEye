# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
Testbench for PE_simple — the dense, sparsity-free variant of the Processing Element.

PE_simple computes the same dense GEMM as PE.v:

    for m in 0..M0-1:
        psum[m] = bias[m] + sum_i( iact[i] * wght[i*M0 + m] )   for i in 0..N-1

with N = B * C0 contraction terms and M0 output channels.  Weights are stored
row-major (contraction index i outer, output channel m inner).  The bias is
streamed in over psum_data_i during the readout phase, exactly as PE.v folds in
the upstream partial sum.

Test flow:
  1. Reset DUT
  2. Stream 4 config words (enable_stream_i / data_stream_i)
  3. Load iact (N values) and wght (N*M0 values, packed PARALLEL_MACS/word)
  4. Pulse compute_i
  5. After psum_ready_o, assert psum_enable_i and drive bias on psum_data_i;
     collect M0 psum_data_o values
  6. Compare against the software golden model

Every wait is bounded by a cycle count so a broken DUT still produces a
complete waveform instead of hanging.
"""

import math
import os
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------
CLK      = int(os.environ["CLOCK_LEN"])
CLK_UNIT = os.environ["CLOCK_UNIT"]

TIMEOUT_CYCLES = 4000   # hard simulation-time budget (cycles)


async def clk_edge(dut):
    await RisingEdge(dut.clk_i)


async def wait_clk(dut, n=1):
    for _ in range(n):
        await clk_edge(dut)


def _drive(sig, val):
    sig.value = val


def _log(dut, msg):
    dut._log.info(msg)


# ---------------------------------------------------------------------------
# Config streaming (4 words, one per clock, no handshake)
# ---------------------------------------------------------------------------
async def send_config(dut, N, C0, M0, pmacs):
    # One weight word holds PARALLEL_MACS weights.  Total weights = N * M0.
    n_wght        = N * M0
    wght_addr_max = math.ceil(n_wght / pmacs) - 1   # last weight-word index
    iact_addr_max = N - 1                            # last iact index
    stride        = 1

    _log(dut, f"CONFIG: M0={M0} N={N} wght_addr_max={wght_addr_max} iact_addr_max={iact_addr_max}")

    _drive(dut.enable_stream_i, 1)

    # FIRST_PARAMS:  [11:9]=stride, [8:1]=wght_addr_max
    _drive(dut.data_stream_i, (stride << 9) | (wght_addr_max << 1))
    await wait_clk(dut)

    # SECOND_PARAMS: [11:7]=M0, [6:3]=C0
    _drive(dut.data_stream_i, (M0 << 7) | (C0 << 3))
    await wait_clk(dut)

    # THIRD_PARAMS:  [iact_addr_max_bw:1]=iact_addr_max, [0]=data_mode(0)
    _drive(dut.data_stream_i, iact_addr_max << 1)
    await wait_clk(dut)

    # FOURTH_PARAMS: fraction_bits=0, iact_x_line_reps=0
    _drive(dut.data_stream_i, 0)
    await wait_clk(dut)

    _drive(dut.enable_stream_i, 0)
    _drive(dut.data_stream_i, 0)
    await wait_clk(dut)


# ---------------------------------------------------------------------------
# Iact loading — one value per clock, ready observed but not awaited
# ---------------------------------------------------------------------------
async def send_iact(dut, iact_flat):
    iact_bw = int(dut.DATA_IACT_BITWIDTH.value)
    mask    = (1 << iact_bw) - 1

    _drive(dut.iact_select_i, 1)
    _drive(dut.iact_enable_i, 1)

    for i, val in enumerate(iact_flat):
        raw = int(val) & mask
        _drive(dut.iact_data_i, raw)
        await wait_clk(dut)
        ready = bool(int(dut.iact_ready_o.value))
        _log(dut, f"  iact[{i}]={int(val)}  iact_ready_o={int(ready)}")

    _drive(dut.iact_enable_i, 0)
    _drive(dut.iact_select_i, 0)
    _drive(dut.iact_data_i, 0)
    await wait_clk(dut)


# ---------------------------------------------------------------------------
# Weight loading — one packed word per clock, ready observed but not awaited
# ---------------------------------------------------------------------------
async def send_wght(dut, wght_flat, pmacs):
    wght_bw = int(dut.DATA_WGHT_BITWIDTH.value)
    mask    = (1 << wght_bw) - 1

    _drive(dut.wght_enable_i, 1)

    idx   = 0
    total = len(wght_flat)
    word_idx = 0
    while idx < total:
        word = 0
        for lane in range(pmacs):
            if idx + lane < total:
                raw   = int(wght_flat[idx + lane]) & mask
                word |= raw << (lane * wght_bw)
        _drive(dut.wght_data_i, word)
        await wait_clk(dut)
        ready = bool(int(dut.wght_ready_o.value))
        _log(dut, f"  wght_word[{word_idx}]={word:#08x}  wght_ready_o={int(ready)}")
        idx      += pmacs
        word_idx += 1

    _drive(dut.wght_enable_i, 0)
    _drive(dut.wght_data_i, 0)
    await wait_clk(dut)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_pe_simple(dut):
    B    = int(os.environ.get("B",    "4"))
    C0   = int(os.environ.get("C0",   "3"))
    M0   = int(os.environ.get("M0",   "6"))
    seed = int(os.environ.get("SEED", "42"))
    np.random.seed(seed)

    N     = B * C0                      # contraction length
    pmacs = int(dut.PARALLEL_MACS.value)

    wght_cap = int(dut.WGHT_DATA_ADDR.value)
    iact_cap = int(dut.IACT_DATA_ADDR.value)
    psum_cap = int(dut.PSUM_ADDR.value)
    assert N <= iact_cap,    f"N={N} exceeds iact SPad depth {iact_cap}"
    assert N * M0 <= wght_cap, f"N*M0={N*M0} exceeds wght SPad depth {wght_cap}"
    assert M0 <= psum_cap,   f"M0={M0} exceeds psum SPad depth {psum_cap}"

    # iact: N activations; wght: N x M0 matrix (row-major); bias: M0 values
    iacts = np.random.randint(-63, 64, size=N).astype(int)
    iacts[iacts == 0] = 1
    wghts = np.random.randint(-63, 64, size=(N, M0)).astype(int)
    wghts[wghts == 0] = 1
    bias  = np.arange(1, M0 + 1, 1).astype(int)

    # Golden: psum[m] = bias[m] + sum_i iact[i]*wght[i][m]
    golden = [int(bias[m]) + int(np.sum(iacts * wghts[:, m])) for m in range(M0)]

    wght_flat = wghts.reshape(-1)       # row-major i*M0 + m

    _log(dut, f"Params: B={B} C0={C0} M0={M0} N={N} pmacs={pmacs}")
    _log(dut, f"iacts : {iacts.tolist()}")
    _log(dut, f"bias  : {bias.tolist()}")
    _log(dut, f"golden: {golden}")

    # --- Clock ---
    cocotb.start_soon(Clock(dut.clk_i, CLK, unit=CLK_UNIT).start())

    # --- Reset all inputs ---
    for sig in [dut.rst_ni, dut.iact_select_i, dut.iact_data_i, dut.iact_enable_i,
                dut.wght_data_i, dut.wght_enable_i, dut.psum_data_i,
                dut.psum_enable_i, dut.psum_ready_i, dut.compute_i,
                dut.enable_stream_i, dut.data_stream_i,
                dut.iact_pass_data_i, dut.iact_pass_enable_i, dut.iact_pass_ready_i]:
        _drive(sig, 0)

    await wait_clk(dut, 2)
    _drive(dut.rst_ni, 1)
    await wait_clk(dut, 2)

    # --- Config ---
    await send_config(dut, N, C0, M0, pmacs)

    # --- Load iact and wght sequentially ---
    await send_iact(dut, iacts)
    await send_wght(dut, wght_flat, pmacs)

    _log(dut, f"After loading: iact_set={int(dut.iact_set.value)}  wght_set={int(dut.wght_set.value)}")
    assert int(dut.iact_set.value) == 1, "iact_set not asserted after loading"
    assert int(dut.wght_set.value) == 1, "wght_set not asserted after loading"

    # --- Trigger compute ---
    _drive(dut.compute_i, 1)
    await wait_clk(dut)
    _drive(dut.compute_i, 0)

    # --- Wait for psum_ready_o, then assert psum_enable_i + drive bias ---
    psum_ready_seen = False
    for cycle in range(TIMEOUT_CYCLES):
        await wait_clk(dut)
        if dut.psum_ready_o.value:
            psum_ready_seen = True
            break

    if not psum_ready_seen:
        _log(dut, "TIMEOUT: psum_ready_o never asserted.")
        await wait_clk(dut, 10)
        assert False, "psum_ready_o did not assert within timeout"

    _drive(dut.psum_ready_i, 1)
    _drive(dut.psum_enable_i, 1)

    # --- Collect psums; bias[m] is driven on psum_data_i for output index m ---
    psum_bw  = int(dut.DATA_PSUM_BITWIDTH.value)
    sign_bit = 1 << (psum_bw - 1)
    mask     = (1 << psum_bw) - 1

    # Drive the first bias value before the PE leaves WAIT_TO_SEND_PSUM.
    _drive(dut.psum_data_i, int(bias[0]) & mask)

    results = []
    out_idx = 0
    for cycle in range(TIMEOUT_CYCLES):
        await wait_clk(dut)
        if not dut.psum_enable_o.value:
            if results:
                break
            # not started emitting yet; keep the current bias word stable
            continue
        raw    = int(dut.psum_data_o.value) & mask
        signed = raw - (1 << psum_bw) if raw & sign_bit else raw
        results.append(signed)
        _log(dut, f"  psum[{out_idx}]: raw={raw:#07x}  signed={signed}")
        out_idx += 1
        # Present the next bias word for the next output index.
        if out_idx < M0:
            _drive(dut.psum_data_i, int(bias[out_idx]) & mask)

    _drive(dut.psum_enable_i, 0)
    _drive(dut.psum_ready_i, 0)
    _drive(dut.psum_data_i, 0)
    await wait_clk(dut, 5)

    # --- Verify ---
    _log(dut, f"Collected {len(results)} psums: {results}")
    _log(dut, f"Golden    {M0} psums: {golden}")

    assert len(results) == M0, f"Expected {M0} psum outputs, got {len(results)}"
    for i, (got, exp) in enumerate(zip(results, golden)):
        assert got == exp, f"psum[{i}] mismatch: hw={got}, golden={exp}"

    _log(dut, f"All {M0} psums verified correctly.")
