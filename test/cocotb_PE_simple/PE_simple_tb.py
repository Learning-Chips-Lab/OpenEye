# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
Testbench for PE_simple — the dense, sparsity-free variant of the Processing Element.

Test flow:
  1. Reset DUT
  2. Stream 4 config words (enable_stream_i / data_stream_i)
  3. Load iact and wght scratch-pads cycle-by-cycle (no blocking ready waits)
  4. Pulse compute_i
  5. Assert psum_enable_i after a fixed delay; collect psum_data_o
  6. Compare collected psums against the software golden model

The testbench never waits indefinitely on a DUT output — every wait is bounded
by a cycle count so that a broken DUT still produces a complete waveform.
"""

import math
import os
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------
CLK      = int(os.environ["CLOCK_LEN"])
CLK_UNIT = os.environ["CLOCK_UNIT"]

TIMEOUT_CYCLES = 2000   # hard simulation-time budget (cycles)


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
async def send_config(dut, B, C0, M0_use, pmacs):
    n_vals        = B * C0
    wght_addr_max = math.ceil(n_vals / pmacs) - 1
    iact_addr_max = n_vals - 1
    stride        = 1

    _log(dut, f"CONFIG: wght_addr_max={wght_addr_max}  iact_addr_max={iact_addr_max}")

    _drive(dut.enable_stream_i, 1)

    # FIRST_PARAMS:  [11:9]=stride, [8:1]=wght_addr_max
    _drive(dut.data_stream_i, (stride << 9) | (wght_addr_max << 1))
    await wait_clk(dut)

    # SECOND_PARAMS: [11:7]=M0, [6:3]=C0
    _drive(dut.data_stream_i, (M0_use << 7) | (C0 << 3))
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
# Iact loading — one value per clock, ready signal is observed but not awaited
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
        _log(dut, f"  iact[{i}]={val:#04x}  iact_ready_o={int(ready)}")

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
    M0   = int(os.environ.get("M0",  "12"))
    seed = int(os.environ.get("SEED", "42"))
    np.random.seed(seed)

    N     = B * C0
    pmacs = int(dut.PARALLEL_MACS.value)

    iacts = np.random.randint(-63, 64, size=N).astype(int)
    iacts[iacts == 0] = 1

    # One weight per iact slot; N == M0_use so each iact maps to its own psum.
    M0_use    = min(M0, N)
    wghts_use = np.random.randint(-63, 64, size=M0_use).astype(int)
    wghts_use[wghts_use == 0] = 1
    iacts_use = iacts[:M0_use]
    golden    = [int(iacts_use[i]) * int(wghts_use[i]) for i in range(M0_use)]

    _log(dut, f"Params: B={B} C0={C0} M0={M0} N={N} M0_use={M0_use} pmacs={pmacs}")
    _log(dut, f"iacts : {iacts_use.tolist()}")
    _log(dut, f"wghts : {wghts_use.tolist()}")
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
    await send_config(dut, B, C0, M0_use, pmacs)

    # --- Load iact and wght sequentially (avoids two coroutines fighting the clock) ---
    await send_iact(dut, iacts_use)
    await send_wght(dut, wghts_use, pmacs)

    # Log data-loaded flags
    _log(dut, f"After loading: iact_set={int(dut.iact_set.value)}  wght_set={int(dut.wght_set.value)}")

    # --- Trigger compute ---
    _drive(dut.compute_i, 1)
    await wait_clk(dut)
    _drive(dut.compute_i, 0)

    # --- Wait up to TIMEOUT_CYCLES for psum_ready_o, then assert psum_enable_i ---
    psum_ready_seen = False
    for cycle in range(TIMEOUT_CYCLES):
        await wait_clk(dut)
        state = int(dut.current_state_computing.value)
        _log(dut, f"  cycle {cycle:4d}  compute_state={state}  psum_ready_o={int(dut.psum_ready_o.value)}")
        if dut.psum_ready_o.value:
            psum_ready_seen = True
            break

    if not psum_ready_seen:
        _log(dut, "TIMEOUT: psum_ready_o never asserted — dumping waveform and ending.")
        await wait_clk(dut, 10)
        assert False, "psum_ready_o did not assert within timeout"

    _drive(dut.psum_ready_i, 1)
    _drive(dut.psum_enable_i, 1)
    _drive(dut.psum_data_i, 0)

    # --- Collect psums; sample on each rising edge while psum_enable_o is high ---
    psum_bw  = int(dut.DATA_PSUM_BITWIDTH.value)
    sign_bit = 1 << (psum_bw - 1)
    mask     = (1 << psum_bw) - 1

    results = []
    for cycle in range(TIMEOUT_CYCLES):
        await wait_clk(dut)
        if not dut.psum_enable_o.value:
            _log(dut, f"  psum cycle {cycle}: psum_enable_o=0, done collecting")
            break
        raw    = int(dut.psum_data_o.value) & mask
        signed = raw - (1 << psum_bw) if raw & sign_bit else raw
        results.append(signed)
        _log(dut, f"  psum cycle {cycle}: raw={raw:#07x}  signed={signed}")

    _drive(dut.psum_enable_i, 0)
    _drive(dut.psum_ready_i, 0)
    await wait_clk(dut, 5)

    # --- Verify ---
    _log(dut, f"Collected {len(results)} psums: {results}")
    _log(dut, f"Golden    {M0_use} psums: {golden}")

    assert len(results) == M0_use, (
        f"Expected {M0_use} psum outputs, got {len(results)}"
    )
    for i, (got, exp) in enumerate(zip(results, golden)):
        assert got == exp, f"psum[{i}] mismatch: hw={got}, golden={exp}"

    _log(dut, f"All {M0_use} psums verified correctly.")
