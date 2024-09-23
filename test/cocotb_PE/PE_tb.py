# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import math
import sys
import os
import numpy as np
from numpy import genfromtxt

import cocotb
from cocotb.triggers import Timer, Combine
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

directory = (os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
import test_utils.timing_parameters as timing_parameters
import test_utils.rtl_test_utils as rtl_test_utils

clk_cycle = int(os.environ["CLOCK_LEN"])
clk_cycle_unit = os.environ["CLOCK_UNIT"]

clk_delay_in = int(os.environ["CLOCK_DELAY_INPUT"])
clk_delay_unit_in = os.environ["CLOCK_DELAY_UNIT_INPUT"]

clk_delay_out = int(os.environ["CLOCK_DELAY_OUTPUT"])
clk_delay_unit_out = os.environ["CLOCK_DELAY_UNIT_OUTPUT"]

async def test_hdls(ptp, dut, iacts_array, wghts_array, psum_array, hyperparameter_list):
    cocotb.start_soon(Clock(dut.clk_i, 10, units=clk_cycle_unit).start())
    dut._log.info("Clock is %s " + clk_cycle_unit, clk_cycle)
    await cocotb.start_soon(reset_all_signals(ptp, dut))
    send_iact_thread = cocotb.start_soon(send_iact(ptp, dut, iacts_array, hyperparameter_list))
    send_wght_thread = cocotb.start_soon(send_wght(ptp, dut, wghts_array, hyperparameter_list))

    await Combine(send_iact_thread, send_wght_thread)
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.compute_i, 1))
    #dut.compute_i.value = 1

    await Timer(clk_cycle, units=clk_cycle_unit)
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.compute_i, 0))
    #dut.compute_i.value = 0
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.psum_ready_i, 1))

    await RisingEdge(dut.psum_ready_o)
    await Timer(clk_cycle, units=clk_cycle_unit)

    cocotb.start_soon(send_bias(ptp, dut, psum_array))

    await RisingEdge(dut.psum_enable_o)

    await Timer(clk_cycle, units=clk_cycle_unit)

    cocotb.start_soon(get_psum(dut, iacts_array, wghts_array, psum_array))

    await FallingEdge(dut.psum_enable_o)
    for x in range(100):
        await Timer(clk_cycle, units=clk_cycle_unit)

    assert dut.compute_i.value == 0, "rst_ni is not 0!"

@cocotb.test()
async def start_test_pe(dut):
    iactsize_x = 3   # Dimensions
    iactsize_y = 1   # Channels
    wghtsize_x = 1  # Filters
    wghtsize_y = iactsize_x * iactsize_y
    sparse_iact = 0  # Sparsity of Iacts, 0 No Sparsity, 1 Full Sparse
    sparse_wght = 0  # Sparsity of Wghts, 0 No Sparsity, 1 Full Sparse
    hyperparameter_list = [iactsize_x, iactsize_y, sparse_iact, wghtsize_x, wghtsize_y, sparse_wght]
    ptp = timing_parameters.PortTimingParameters()
    ptp.initiate_params(clk_cycle, clk_cycle_unit, clk_delay_in, clk_delay_unit_in, clk_delay_out, clk_delay_unit_out)
    
    print("clk_cycle: ")
    print(clk_cycle)
    (iacts, wghts, psums) = create_iact_wght_psum_arrays(
        dut, hyperparameter_list
    )

    await cocotb.start_soon(test_hdls(ptp, dut, iacts, wghts, psums, hyperparameter_list))

async def send_iact(ptp, dut, data_array, hyperparameter_list):
    spad_data = generate_spad(
        data_array,
        dut.IACT_ADDR_ADDR.value,
        dut.IACT_DATA_ADDR.value,
        dut.DATA_IACT_BITWIDTH.value,
        True,
        0,
        True,
    )
    for x in range(len(spad_data[1])):
        if((spad_data[1][x] != 0)):
            spad_data[1][x] = spad_data[1][x] + int(math.floor(spad_data[1][x-1] / 256) * 256)
    for x in range(len(spad_data[1])):
        if((spad_data[1][x] != 0)):
            spad_data[1][x] = spad_data[1][x] + x * 256
    dut._log.info("IACT ADDR is %s", spad_data[0])
    dut._log.info("IACT DATA is %s", spad_data[1])
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.iact_enable_i[0], 1))
    await send_to_spad(
        ptp,
        spad_data[0],
        dut.iact_data_i,
        1 + hyperparameter_list[1], #TODO: Make variable
        dut.TRANS_BITWIDTH_IACT.value,
        dut.IACT_ADDR_DATA.value,
        False,
    )
    await send_to_spad(
        ptp,
        spad_data[1],
        dut.iact_data_i,
        hyperparameter_list[1]*hyperparameter_list[0],
        dut.TRANS_BITWIDTH_IACT.value,
        dut.IACT_DATA_DATA.value,
        False,
    )
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.iact_enable_i[0], 0))
    await Timer(clk_cycle, units=clk_cycle_unit)

async def get_psum(dut, iacts_array, wghts_array, psum_array):
    control = np.zeros(64, dtype=int)

    iact = iacts_array
    wght = wghts_array
    bias = psum_array

    if iact.ndim == 1:
        iact = [iact]

    if wght.ndim == 1:
        wght = [wght]

    for psum_x in range(len(bias)):
        control[psum_x] = bias[psum_x]

    current_iact = 0

    for iact_y in range(len(iact)):
        for iact_x in range(len(iact[iact_y])):
            for wght_x in range(len(wght[current_iact])):
                control[wght_x] = (
                    control[wght_x] + wght[current_iact][wght_x] * iact[iact_y][iact_x]
                )
            current_iact = current_iact + 1

    current_control = 0

    while dut.psum_enable_o.value == 1:
        assert dut.adder_1.sum_o.value.integer == control[current_control], (
            "PSUM("
            + str(dut.adder_1.sum_o.value.integer)
            + ") is not equal to control("
            + str(control[current_control])
            + "), "
            + str(current_control + 1)
            + ". PSUM Value"
        )
        current_control = current_control + 1
        assert dut.adder_2.sum_o.value.integer == control[current_control], (
            "PSUM("
            + str(dut.adder_2.sum_o.value.integer)
            + ") is not equal to control("
            + str(control[current_control])
            + "), "
            + str(current_control + 1)
            + ". PSUM Value"
        )

        current_control = current_control + 1
        await Timer(clk_cycle, units=clk_cycle_unit)

async def send_wght(ptp, dut, data_array, hyperparameter_list):
    spad_data = generate_spad(
        data_array,
        dut.WGHT_ADDR_ADDR.value,
        dut.WGHT_DATA_ADDR.value,
        dut.DATA_WGHT_BITWIDTH.value,
        False,
        dut.DATA_WGHT_BITWIDTH.value + dut.DATA_WGHT_IGNORE_ZEROS.value,
        True,
    )
    for x in reversed(range(len(spad_data[0]))):
        if(x == 0):
            spad_data[0][x] = 0
        else:
            spad_data[0][x] = spad_data[0][x - 1]
        
    dut._log.info("WGHT ADDR is %s", spad_data[0])
    dut._log.info("WGHT DATA is %s", spad_data[1])

    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.wght_enable_i, 1))
    await send_to_spad(
        ptp,
        spad_data[0],
        dut.wght_data_i,
        hyperparameter_list[0] * hyperparameter_list[1] + 2, #TODO: Make variable
        dut.TRANS_BITWIDTH_WGHT.value,
        dut.WGHT_ADDR_DATA.value,
        False,
    )
    await send_to_spad(
        ptp,
        spad_data[1],
        dut.wght_data_i,
        dut.WGHT_DATA_ADDR.value,
        dut.TRANS_BITWIDTH_WGHT.value,
        dut.WGHT_DATA_DATA.value,
        False,
    )
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.wght_enable_i, 0))
    await Timer(clk_cycle, units=clk_cycle_unit)

async def send_bias(ptp, dut, data_array):
    spad_data = generate_spad(
        data_array,
        dut.PSUM_ADDR.value,
        dut.PSUM_ADDR.value,
        dut.DATA_PSUM_BITWIDTH.value,
        False,
        dut.DATA_PSUM_BITWIDTH.value,
        False,
    )
    dut._log.info("PSUM is %s", spad_data[1])
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.psum_enable_i, 1))
    await send_to_spad(
        ptp,
        spad_data[1],
        dut.psum_data_i,
        dut.PSUM_ADDR.value,
        dut.TRANS_BITWIDTH_PSUM.value,
        dut.DATA_PSUM_BITWIDTH.value * 2,
        False,
    )
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.psum_enable_i, 0))
    await Timer(clk_cycle, units=clk_cycle_unit)

async def reset_all_signals(ptp, dut):
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.rst_ni), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.data_mode_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.iact_select_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.compute_i), 0))

    for glb_iact in range(dut.NUM_GLB_IACT.value):
        cocotb.start_soon(rtl_test_utils.set_input(ptp,dut.iact_data_i[glb_iact], 0))
        cocotb.start_soon(rtl_test_utils.set_input(ptp,dut.iact_enable_i[glb_iact], 0))

    cocotb.start_soon(rtl_test_utils.set_input(ptp,dut.wght_data_i, 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,dut.wght_enable_i, 0))

    cocotb.start_soon(rtl_test_utils.set_input(ptp,dut.psum_data_i, 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,dut.psum_enable_i, 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,dut.psum_ready_i, 0))

    cocotb.start_soon(rtl_test_utils.set_input(ptp,dut.fraction_bit_i, 0))

    await Timer(clk_cycle, units=clk_cycle_unit)
    cocotb.start_soon(rtl_test_utils.set_input(ptp,dut.rst_ni, 1))
    await Timer(clk_cycle, units=clk_cycle_unit)

async def send_to_spad(ptp, spad, data_signal, addr_bits, trans_bits, data_bits, parallel):
    words_per_transmit = 0
    sending_data = 0
    current_storage_position = 0
    offset = 0

    for cycle in range(addr_bits):
        words_per_transmit = int(math.floor(trans_bits / data_bits))
        for writing_cycle in range(words_per_transmit):
            offset = data_bits * writing_cycle
            if parallel:
                current_storage_position = cycle
            else:
                current_storage_position = int(
                    writing_cycle
                    + math.floor(cycle / words_per_transmit) * words_per_transmit
                )
            try:
                sending_data = sending_data + (
                    int(spad[current_storage_position]) << offset
                )
            except:
                sending_data = sending_data
        cocotb.start_soon(rtl_test_utils.set_input(ptp, (data_signal), sending_data))
        sending_data = 0
        await Timer(clk_cycle, units=clk_cycle_unit)
    cocotb.start_soon(rtl_test_utils.set_input(ptp, data_signal, 0))

def generate_spad(
    array, addr_spad_words, data_spad_words, bitwidth, sisd, offset, ignore_zeros
):
    data = array
    if data.ndim == 1:
        data = [data]
    addr_spad_data = np.zeros(addr_spad_words)
    data_spad_data = np.zeros(data_spad_words)
    current_count = 0
    overhead = 0
    for y in range(len(data)):
        for x in range(len(data[y])):
            if (data[y][x] != 0) | (not ignore_zeros):
                if sisd:
                    data_spad_data[current_count] = data[y][x] + (overhead << bitwidth)
                else:
                    data_spad_data[int(math.floor(current_count / 2))] = data_spad_data[
                        int(math.floor(current_count / 2))
                    ] + (
                        (data[y][x] + (overhead << bitwidth))
                        << (offset * (current_count % 2))
                    )

                current_count = current_count + 1
                overhead = 0
            else:
                overhead = overhead + 1
        if sisd:
            addr_spad_data[y] = current_count
        else:
            addr_spad_data[y] = int(math.ceil(current_count / 2))
            if current_count % 2 == 1:
                current_count = current_count + 1
        overhead = 0
    spad_data = (addr_spad_data, data_spad_data)
    return spad_data

def create_iact_wght_psum_arrays(
    dut, hyperparameter_list
):
    iacts = np.arange(1, hyperparameter_list[1] * hyperparameter_list[0] + 1, 1).reshape(hyperparameter_list[1], hyperparameter_list[0])
    
    indices = np.random.choice(
        np.arange(iacts.size), replace=False, size=int(iacts.size * hyperparameter_list[2])
    )
    
    #indices = [] #Manuell
    for x in range(len(indices)):
        iacts[int(indices[x] / hyperparameter_list[0])][int(indices[x] % hyperparameter_list[0])] = 0

    #iacts[indices] = 0

    wghts = np.arange(1, hyperparameter_list[4] * hyperparameter_list[3] + 1, 1).reshape(hyperparameter_list[4], hyperparameter_list[3])
    indices = np.random.choice(
        np.arange(wghts.size), replace=False, size=int(wghts.size * hyperparameter_list[5])
    )
    for x in range(len(indices)):
        wghts[int(indices[x] / hyperparameter_list[3])][int(indices[x] % hyperparameter_list[3])] = 0

    #wghts[indices] = 0

    psums = np.arange(1, hyperparameter_list[3] + 1, 1)

    return iacts, wghts, psums
