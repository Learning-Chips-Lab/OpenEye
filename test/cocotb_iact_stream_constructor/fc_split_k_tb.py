# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Check the Dense converter interface independently of PE arithmetic."""
import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge


@cocotb.test()
async def split_k_pairs(dut):
    rows, row, channels = (int(os.environ[key]) for key in ('FC_ROWS', 'FC_ROW', 'FC_CHANNELS'))
    # Two K tiles exercise the read cursor; repeated loads exercise resetting it.
    span = rows * 3 * channels
    for name in ('reset_cycle_i', 'params', 'enable_config', 'enable_store',
                 'enable_converter', 'fc_storage_valid_i', 'storage_i',
                 'needed_y_cls_i', 'needed_iact_channel_cycles_i', 'fc_size_i',
                 'iact_x_add_up', 'channel_div_trans', 'x_lines_i',
                 'needed_wght_cycles_i', 'needed_iact_router_cycles_i',
                 'wght_size_x_i', 'wght_size_y_i', 'stride_x_i', 'stride_y_i',
                 'y_lines_per_calc', 'needed_iact_buffer_words_i', 'iact_words_per_compute',
                 'padding_x', 'padding_y'):
        getattr(dut, name).value = 0
    for prefix, count in (('rd_loop_limit_', 5), ('rd_addr_inc_', 5),
                          ('wr_loop_limit_', 3), ('wr_addr_inc_', 3)):
        for i in range(count):
            getattr(dut, prefix + str(i)).value = 0
    dut.fully_connected_i.value = 1
    dut.iact_channels_per_pe_i.value = channels
    dut.iact_ready_i.value = 7
    dut.rst_ni.value = 0
    cocotb.start_soon(Clock(dut.clk_i, 10, unit='ns').start())

    async def tick():
        await FallingEdge(dut.clk_i)
        await Timer(1, unit='ns')

    for _ in range(3):
        await tick()
    dut.rst_ni.value = 1
    dut.enable_config.value = 1
    await tick()
    dut.enable_config.value = 0

    for load in range(2):
        values = [((i + 13 * load) % 101) + 1 if i < 2 * span - 4 else 0
                  for i in range(2 * span)]
        values[1] = 0  # A real zero must retain its position.
        dut.enable_store.value = 1
        await tick()
        dut.enable_store.value = 0
        for pos in range(0, len(values), 2):
            dut.storage_i.value = values[pos] | (values[pos + 1] << 8)
            dut.fc_storage_valid_i.value = 1
            await tick()
            dut.fc_storage_valid_i.value = 0
            if pos % 4 == 0:
                await tick()  # Gaps must not advance the write cursor.
        dut.fc_storage_valid_i.value = 0
        for _ in range(4):
            await tick()
        for tile in range(2):
            dut.iact_ready_i.value = 0
            dut.enable_converter.value = 1
            await tick()
            dut.enable_converter.value = 0
            for _ in range(3):
                await tick()
                assert int(dut.iact_enable_o.value) == 0
            dut.iact_ready_i.value = 7
            seen = [[] for _ in range(3)]
            for _ in range(channels + 12):
                await tick()
                enable = int(dut.iact_enable_o.value)
                if not enable:
                    continue
                data = int(dut.iact_data_o.value)
                choose = int(dut.iact_choose_o.value)
                for bank in range(3):
                    assert (enable >> bank) & 1
                    assert (choose >> (bank * 4 * 2)) & 3 == bank
                    slot = len(seen[bank]) % 2
                    seen[bank].append((data >> (bank * 24 + slot * 12)) & 255)
            for bank in range(3):
                base = tile * span + row * 3 * channels
                expected = [values[base + pair * 6 + bank * 2 + slot]
                            for pair in range(channels // 2) for slot in range(2)]
                assert seen[bank] == expected, (load, tile, row, bank, seen[bank], expected)
