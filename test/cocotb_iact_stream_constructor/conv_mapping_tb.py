# This file is part of the OpenEye project.
# SPDX-License-Identifier: SHL-2.1
"""Distinct values expose missing rows, duplicated banks, and byte swaps."""
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, Timer


@cocotb.test()
async def convolution_window(dut):
    ports = int(os.environ["CONV_PORTS"])
    channels = int(os.environ["CONV_CHANNELS"])
    x_start = int(os.environ["CONV_X_START"])
    pairs = (channels + 1) // 2
    groups = 6 // ports
    choose_bits = ports.bit_length()
    for name in ("reset_cycle_i", "params", "enable_config", "enable_store",
                 "enable_converter", "fc_storage_valid_i", "storage_i",
                 "needed_y_cls_i", "needed_iact_channel_cycles_i", "fc_size_i",
                 "x_lines_i", "needed_wght_cycles_i", "y_lines_per_calc",
                 "fully_connected_i", "padding_x", "padding_y"):
        getattr(dut, name).value = 0
    for prefix, count in (("rd_loop_limit_", 5), ("rd_addr_inc_", 5),
                          ("wr_loop_limit_", 3), ("wr_addr_inc_", 3)):
        for i in range(count):
            getattr(dut, prefix + str(i)).value = 0
    config = {
        "params": x_start << 24,
        "iact_channels_per_pe_i": channels, "channel_div_trans": pairs,
        "iact_x_add_up": 8, "needed_iact_router_cycles_i": groups,
        "wght_size_x_i": 3, "wght_size_y_i": 3, "stride_x_i": 1,
        "stride_y_i": 1, "needed_iact_buffer_words_i": 18 * pairs,
        "iact_words_per_compute": groups * 3 * channels + 1,
        "rd_loop_limit_0": groups * pairs - 1, "rd_addr_inc_0": 1,
        "rd_addr_inc_1": 6 * pairs, "wr_loop_limit_0": 18 * pairs - 1,
        "wr_loop_limit_2": 2, "wr_addr_inc_0": 1,
        "wr_addr_inc_1": 36 * pairs, "wr_addr_inc_2": 18 * pairs,
        "iact_ready_i": (1 << ports) - 1,
    }
    for name, value in config.items():
        getattr(dut, name).value = value
    dut.rst_ni.value = 0
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())

    async def tick():
        await FallingEdge(dut.clk_i)
        await Timer(1, unit="ns")

    for _ in range(3):
        await tick()
    dut.rst_ni.value = 1
    for _ in range(3):
        await tick()
    dut.enable_config.value = 1
    await tick()
    dut.enable_config.value = 0
    await tick()

    for load in range(2):
        def value(y, x, c):
            if (y, x, c) == (1, 4, 0):
                return 0
            return (1 + load * 17 + y * 40 + x * channels + c) % 127

        dut.enable_store.value = 1
        await tick()
        dut.enable_store.value = 0
        for _ in range(2):
            await tick()
        for y in range(3):
            for x in range(10):
                for pair in range(pairs):
                    word = value(y, x, 2 * pair)
                    if channels > 1:
                        word |= value(y, x, 2 * pair + 1) << 8
                    dut.storage_i.value = word
                    await tick()
        for _ in range(8):
            await tick()
        dut.iact_ready_i.value = 0
        dut.enable_converter.value = 1
        await tick()
        dut.enable_converter.value = 0
        for _ in range(3):
            await tick()
            assert int(dut.iact_enable_o.value) == 0
        dut.iact_ready_i.value = (1 << ports) - 1
        seen = [[] for _ in range(12)]
        held = [0] * 12
        for _ in range(groups * 3 * channels + 12):
            await tick()
            enable = int(dut.iact_enable_o.value)
            if not enable:
                continue
            data = int(dut.iact_data_o.value)
            choose = int(dut.iact_choose_o.value)
            for pe in range(12):
                bank = (choose >> (pe * choose_bits)) & ((1 << choose_bits) - 1)
                if bank >= ports or not (enable & (1 << bank)):
                    continue
                # The PE latches a pair on its first enabled cycle, then
                # consumes the saved high subword on its next enabled cycle.
                if len(seen[pe]) % 2 == 0:
                    held[pe] = (data >> (bank * 24)) & 0xffffff
                seen[pe].append((held[pe] >> (12 * (len(seen[pe]) % 2))) & 255)
        for pe in range(12):
            x = x_start + pe % 4 + pe // 4
            expected = [value(y, x, c) for y in range(3) for c in range(channels)]
            assert seen[pe] == expected, (load, pe, seen[pe], expected)
