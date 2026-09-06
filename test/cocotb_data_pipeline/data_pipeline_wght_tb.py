# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
"""Unit testbench for hdl/data_pipeline_wght.v.

The weight pipeline is the component that turns the compressed weight stream
into scratchpad writes, and it owns the filter-position bookkeeping that the
MAC engine relies on. Exercising it on its own - rather than through a whole
PE_cluster - makes it possible to state exactly which filter each weight was
placed under, which is what a cluster-level PSUM mismatch cannot tell you.

The encoding under test
-----------------------
Weights are walked row-major over a [rows][filters_w] matrix, where filters_w
is the number of filters (M0). Zero weights are not transmitted; instead the
next transmitted weight carries an "overhead" tag counting how many positions
were skipped immediately before it. A value with tag t at running position p
therefore belongs to filter p + t.

Each DATA_WIDTH input word carries PARALLEL_MACS sub-words of SECOND_SPAD_DATA
bits, each laid out as {overhead[3:0], payload[7:0]}.

What this testbench checks
--------------------------
For a given weight matrix it computes the expected (filter position, value)
pairs, feeds the corresponding compressed stream in, and compares against the
sequence of second-SPAD writes the module produces. `decode_positions` mirrors
the intended semantics, so a mismatch localises the defect to the RTL's
position accounting rather than to the encoder.
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


def encode_matrix(matrix, parallel_macs):
    """Compress a [rows][filters] weight matrix exactly as the cluster does.

    Mirrors PE_cluster_tb.generate_spad: the skip counter runs on across row
    boundaries (its per-row reset is commented out there), and a row that would
    end on an odd sub-word count stores its final element - even when that
    element is zero - as padding carrying the accumulated skip, then leaves the
    next slot empty so the following row starts on a fresh word.

    That combination is why a trailing zero normally does not leak its skip into
    the next row, and why an entirely zero row (which triggers neither path)
    carries its whole skip forward in the next stored value's tag.

    Returns (words, expected, row_lengths) where expected holds, per row, the
    (filter position, payload) pairs the pipeline should store.
    """
    subwords, expected, row_lengths = [], [], []
    skipped = 0
    count = 0
    for row in matrix:
        start = len(subwords)
        row_expected = []
        for position, value in enumerate(row):
            last = position == len(row) - 1
            store = value != 0 or (last and count % 2 != 0)
            if not store:
                skipped += 1
                continue
            subwords.append((skipped, to_twos(value)))
            if value != 0:
                row_expected.append((position, to_twos(value)))
            skipped = 0
            count += 1
        if count % 2 == 1:
            subwords.append(None)   # slot left empty to realign the next row
            count += 1
        row_lengths.append(len(subwords) - start)
        expected.append(row_expected)

    words = []
    for i in range(0, len(subwords), parallel_macs):
        word = 0
        for lane in range(parallel_macs):
            entry = subwords[i + lane] if i + lane < len(subwords) else None
            if entry is None:
                continue
            overhead, payload = entry
            word |= ((overhead << PAYLOAD_WIDTH) | payload) << (lane * WORD_WIDTH)
        words.append(word)
    return words, expected, row_lengths


async def load_matrix(dut, matrix, parallel_macs, filters_w):
    """Drive one compressed weight matrix in and collect the SPAD writes."""
    words, expected, row_lengths = encode_matrix(matrix, parallel_macs)

    dut.first_spad_max_i.value = filters_w
    dut.raw_mode_i.value = 0
    dut.compute_i.value = 0
    dut.enable_i.value = 0
    dut.data_i.value = 0
    dut.rst_ni.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)

    writes = []
    first_writes = []

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

    async def trace():
        while True:
            await RisingEdge(dut.clk_i)
            await Timer(1, unit="ns")
            def rd(name):
                try:
                    return int(getattr(dut, name).value)
                except Exception:
                    return None
            dut._log.info("t oh_reg=%s oh_new=%s oh_delay=%s addr=%s en=%s data=%s",
                          rd("overhead_reg"), rd("overhead_new_calc_reg"),
                          rd("overhead_delay_reg"), rd("first_spad_addr_o"),
                          rd("first_spad_en_o"), rd("first_spad_data_o"))

    if os.environ.get("TRACE_WGHT"):
        cocotb.start_soon(trace())
    collector = cocotb.start_soon(collect())

    for word in words:
        dut.data_i.value = word
        dut.enable_i.value = 1
        await RisingEdge(dut.clk_i)
    dut.enable_i.value = 0
    dut.data_i.value = 0
    # Let the two-stage pipeline drain before sampling stops.
    for _ in range(6):
        await RisingEdge(dut.clk_i)
    collector.kill()
    return words, expected, row_lengths, writes, first_writes


def decode_positions(writes, parallel_macs, row_lengths):
    """Reconstruct, per weight row, which filter each stored value landed on.

    One SPAD word holds PARALLEL_MACS sub-words of {overhead, payload}. Within a
    row the position starts at 0 and each value advances it by its overhead tag
    plus one, so a value with tag t at running position p belongs to filter
    p + t.

    Rows are segmented using the sub-word counts the encoder produced, mirroring
    what the MAC engine gets from the first (address) SPAD: without that, a
    row-end pad would be mistaken for a weight and shift every later filter.
    Trailing all-zero sub-words are that padding and are dropped - in sparse
    mode a zero weight is never transmitted, so a zero payload cannot be real.

    Note the RTL only rewrites sub-word 0's overhead (premade_spad_2_output
    splices overhead_output into bits [11:8]); higher lanes pass through from
    data_i unchanged.
    """
    flat = []
    for _addr, word in writes:
        for lane in range(parallel_macs):
            sub = (word >> (lane * WORD_WIDTH)) & ((1 << WORD_WIDTH) - 1)
            flat.append(((sub >> PAYLOAD_WIDTH) & ((1 << OVERHEAD_WIDTH) - 1),
                         sub & ((1 << PAYLOAD_WIDTH) - 1)))

    decoded, cursor = [], 0
    for length in row_lengths:
        row = flat[cursor:cursor + length]
        cursor += length
        while row and row[-1] == (0, 0):
            row.pop()  # padding / realignment slots carry no weight
        position, row_decoded = 0, []
        for tag, payload in row:
            position += tag
            row_decoded.append((position, payload))
            position += 1
        decoded.append(row_decoded)
    return decoded


@cocotb.test()
async def test_weight_positions(dut):
    """Report where each weight lands for a configurable zero pattern."""
    parallel_macs = int(os.environ.get("PARALLEL_MACS", "2"))
    filters_w = int(os.environ.get("FILTERS_W", "6"))
    rows = int(os.environ.get("ROWS", "4"))
    zeros = {int(v) for v in os.environ.get("ZEROS", "").split(",") if v != ""}

    matrix = []
    value = 1
    for r in range(rows):
        row = []
        for c in range(filters_w):
            row.append(0 if (r * filters_w + c) in zeros else value)
            value = value + 1 if value < 100 else 1
        matrix.append(row)

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    words, expected, row_lengths, writes, first_writes = await load_matrix(
        dut, matrix, parallel_macs, filters_w)

    dut._log.info("filters_w=%d PARALLEL_MACS=%d zeros=%s",
                  filters_w, parallel_macs, sorted(zeros))
    dut._log.info("input words   : %s", [hex(w) for w in words])
    dut._log.info("expected      : %s", expected)
    dut._log.info("spad writes   : %s", [(a, hex(d)) for a, d in writes])
    decoded = decode_positions(writes, parallel_macs, row_lengths)

    # The first (address) SPAD is how the MAC engine finds row boundaries: entry
    # r should hold the number of SPAD words consumed up to the end of weight
    # row r. If these are wrong the data SPAD being correct does not help, since
    # the engine would segment the stream in the wrong places.
    expected_first = []
    running = 0
    for length in row_lengths:
        running += length // parallel_macs
        expected_first.append(running)
    # The pipeline writes the first SPAD once per input word, re-writing the
    # same address as a row is consumed, so the meaningful value is the last one
    # written to each address.
    final_first = {}
    for addr, value in first_writes:
        final_first[addr] = value
    got_first = [final_first.get(r) for r in range(len(row_lengths))]
    dut._log.info("first spad writes : %s", first_writes)
    dut._log.info("first spad per row: got %s, expected %s", got_first, expected_first)
    dut._log.info("decoded       : %s", decoded)

    mismatches = []
    for r, (exp_row, got_row) in enumerate(zip(expected, decoded)):
        if exp_row != got_row:
            mismatches.append((r, exp_row, got_row))
            dut._log.error("row %d: expected %s", r, exp_row)
            dut._log.error("row %d: got      %s", r, got_row)
    assert not mismatches, "weight pipeline mis-placed values in %d of %d rows" % (
        len(mismatches), len(expected))
    assert got_first == expected_first, (
        "first (address) SPAD row boundaries wrong: got %s, expected %s"
        % (got_first, expected_first))
