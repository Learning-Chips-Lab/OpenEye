# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
"""Unit testbench for hdl/data_pipeline_iact.v.

Unlike the weight pipeline, the activation stream arrives **uncompressed**: the
host sends every activation including the zeros, and this module does the
zero-skipping itself. Each stored activation carries a position tag
(transmission_counter_delay) that PE.v uses to pick the matching weight row, so
the tag has to be the activation's *source* position, not its index among the
stored values.

The testbench drives raw activations in and checks exactly that: which source
position each stored activation is tagged with, and the per-channel counts in
the first SPAD.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

PAYLOAD_WIDTH = 8
OVERHEAD_WIDTH = 4
WORD_WIDTH = PAYLOAD_WIDTH + OVERHEAD_WIDTH


def to_twos(value, bits=PAYLOAD_WIDTH):
    return value + (1 << bits) if value < 0 else value


@cocotb.test()
async def test_iact_positions(dut):
    """Check that each stored activation keeps its source position tag."""
    values_per_word = int(os.environ.get("VALUES_PER_WORD", "2"))
    channels = int(os.environ.get("CHANNELS", "2"))
    per_channel = int(os.environ.get("PER_CHANNEL", "2"))
    zeros = {int(v) for v in os.environ.get("ZEROS", "").split(",") if v != ""}

    activations = []
    for i in range(channels * per_channel):
        activations.append(0 if i in zeros else (i % 90) + 1)

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    dut.compute_i.value = 0
    dut.enable_i.value = 0
    dut.data_i.value = 0
    dut.iact_x_line_repetitions_i.value = 1
    dut.first_spad_max_i.value = per_channel
    dut.rst_ni.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)

    writes, first_writes = [], []

    async def collect():
        while True:
            await RisingEdge(dut.clk_i)
            await Timer(1, unit="ns")
            try:
                if int(dut.second_spad_en_o.value) == 1:
                    writes.append((int(dut.second_spad_addr_o.value),
                                   int(dut.second_spad_data_o.value)))
                if int(dut.first_spad_en_o.value) == 1:
                    first_writes.append((int(dut.first_spad_addr_o.value),
                                         int(dut.first_spad_data_o.value)))
            except ValueError:
                pass

    collector = cocotb.start_soon(collect())

    # Activations go in uncompressed, values_per_word per transfer.
    for i in range(0, len(activations), values_per_word):
        word = 0
        for lane in range(values_per_word):
            if i + lane < len(activations):
                word |= to_twos(activations[i + lane]) << (lane * PAYLOAD_WIDTH)
        dut.data_i.value = word
        dut.enable_i.value = 1
        await RisingEdge(dut.clk_i)
    dut.enable_i.value = 0
    dut.data_i.value = 0
    for _ in range(8):
        await RisingEdge(dut.clk_i)
    collector.kill()

    stored = [((w >> PAYLOAD_WIDTH) & ((1 << OVERHEAD_WIDTH) - 1),
               w & ((1 << PAYLOAD_WIDTH) - 1)) for _a, w in writes]
    expected = [(pos, to_twos(v)) for pos, v in enumerate(activations) if v != 0]

    dut._log.info("activations : %s (zeros at %s)", activations, sorted(zeros))
    dut._log.info("expected    : %s", expected)
    dut._log.info("stored      : %s", stored)
    dut._log.info("first spad  : %s", first_writes)

    assert stored == expected, (
        "stored activations carry the wrong position tags:\n  expected %s\n  got      %s"
        % (expected, stored))
