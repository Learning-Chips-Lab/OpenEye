# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import math
import os
import sys
import numpy as np
from numpy import genfromtxt
directory = (os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
import test_utils.open_eye_parameters as oep

import cocotb
from cocotb.triggers import Timer, Combine
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

import cocotb_PE_cluster.pe_cluster_test_utils as pe_cluster_test_utils

clk_cycle = int(os.environ["CLOCK_LEN"])
clk_cycle_unit = os.environ["CLOCK_UNIT"]

clk_delay_in = int(os.environ["CLOCK_DELAY_INPUT"])
clk_delay_unit_in = os.environ["CLOCK_DELAY_UNIT_INPUT"]

clk_delay_out = int(os.environ["CLOCK_DELAY_OUTPUT"])
clk_delay_unit_out = os.environ["CLOCK_DELAY_UNIT_OUTPUT"]

signals_dict = {}

async def test_pe_cluster(dut, params, iacts_array, wghts_array, psum_array):
    """_summary_

    Args:
        dut (_type_): _description_
        iacts_array (_type_): _description_
        wghts_array (_type_): _description_
        psum_array (_type_): _description_
    """
    global signals_dict
    # start the clock
    global clk
    last_time_point = 0
    time = 0
    clk = Clock(dut.clk_i, clk_cycle, units=clk_cycle_unit)
    cocotb.start_soon(clk.start())
    dut._log.info("Clock is %s " + clk_cycle_unit, clk_cycle)

    # reset the DUT
    await cocotb.start_soon(reset_all_signals(dut, params))
    
    last_time_point = cocotb.utils.get_sim_time("ns")
    time = cocotb.utils.get_sim_time("ns")
    print("Reset time: From: " + str(last_time_point - time) + " Till: " + str(last_time_point) + " Duration: " + str(time) + " ns!\n")
    # start the test threads
    send_iact_thread = cocotb.start_soon(send_iact(dut, params, iacts_array))
    send_wght_thread = cocotb.start_soon(send_wght(dut, params, wghts_array))

    await Combine(send_iact_thread, send_wght_thread)
    time = cocotb.utils.get_sim_time("ns") - last_time_point
    last_time_point = cocotb.utils.get_sim_time("ns")
    print("Loading time: From: " + str(last_time_point - time) + " Till: " + str(last_time_point) + " Duration: " + str(time) + " ns!\n")
    await RisingEdge(clk.signal)
    # activate all PEs to start the computation
    # (this is done by setting the compute_i signal to 1 for one clock cycle)
    for pe_x in range(params.PEs_X):
        for pe_y in range(params.PEs_Y):
            cocotb.start_soon(set_flat_input(signals_dict["compute_i"][pe_x][pe_y], 1))

    await RisingEdge(clk.signal)

    # reset compute_i signal to 0 after one clock cycle
    for pe_x in range(params.PEs_X):
        for pe_y in range(params.PEs_Y):
            cocotb.start_soon(set_flat_input(signals_dict["compute_i"][pe_x][pe_y], 0))

    # configure the vertical routing of the PEs to send the psums upwards in the column of the PEs
    # (by default, the accumulate the psums inside the PE)
    for glb_cluster in range(params.NUM_GLB_PSUM):
        cocotb.start_soon(set_flat_input(signals_dict["pe_router_psum_ready_i"][glb_cluster], 1))
    
    # wait until all PEs have finished the computation
    # (i.e. the psums are ready to be read out)
    pes_ready = 0
    while pes_ready != params.NUM_GLB_PSUM:
        pes_ready = 0
        for glb_cluster in range(params.NUM_GLB_PSUM):
            if dut.pe_router_psum_ready_o[glb_cluster].value == 1:
                pes_ready = pes_ready + 1
        await RisingEdge(clk.signal)
    # now send the bias to the PEs
    time = cocotb.utils.get_sim_time("ns") - last_time_point
    last_time_point = cocotb.utils.get_sim_time("ns")
    print("Computing time: From: " + str(last_time_point - time) + " Till: " + str(last_time_point) + " Duration: " + str(time) + " ns!\n")
    cocotb.start_soon(send_bias(dut, params, psum_array))
    
    # wait until the PEs have send out the data
    pes_ready = 0
    while pes_ready != params.NUM_GLB_PSUM:
        await RisingEdge(clk.signal)
        await Timer(clk_delay_out, units=clk_delay_unit_out)
        pes_ready = 0
        for glb_cluster in range(params.NUM_GLB_PSUM):
            if dut.pe_router_psum_enable_o[glb_cluster].value == 1:
                pes_ready = pes_ready + 1
    
    # now we can read out the psums and compare them to the expected values
    cocotb.start_soon(get_psum(dut, params, iacts_array, wghts_array, psum_array))
    
    # check if all enable signals are 0
    pes_ready = 0
    while pes_ready != params.NUM_GLB_PSUM:
        await RisingEdge(clk.signal)
        pes_ready = 0
        for glb_cluster in range(params.NUM_GLB_PSUM):
            if dut.pe_router_psum_enable_o[glb_cluster].value == 0:
                pes_ready = pes_ready + 1
    await RisingEdge(clk.signal)
    
    # finally check if reset is still 1
    assert dut.rst_ni.value == 1, "rst_ni is not 1!"

@cocotb.test()
async def start_test_pe_cluster(dut):
    # Sparse Testcase
    try:
        iactsize_x = int(os.environ["IACTSIZE_X"]) # Dimensions
    except:
        iactsize_x = 3

    try:
        iactsize_y = int(os.environ["IACTSIZE_Y"]) # Channels
    except:
        iactsize_y = 2

    try:
        wghtsize_x = int(os.environ["WGHTSIZE_X"]) # Filters
    except:
        wghtsize_x = 8

    wghtsize_y = iactsize_x * iactsize_y
    sparse_iact = 0 # Sparsity of Iacts, 0 No Sparsity, 1 Full Sparse
    sparse_wght = 0 # Sparsity of Wghts, 0 No Sparsity, 1 Full Sparse
    sel_iacts_zero = np.array([], int) # Added Zeros to iacts
    sel_wghts_zero = np.array([], int) # Added Zeros to wghts

    global pe_iact_cycles
    global signals_dict
    params = oep.create_vh_file()
    pe_iact_cycles = int(math.ceil((params.PEs_X + params.PEs_Y - 1) / params.NUM_GLB_IACT))

    await cocotb.start_soon(create_dict(dut, params))
    (iacts, wghts, psums) = create_iact_wght_psum_arrays(
        iactsize_x,
        iactsize_y,
        wghtsize_x,
        wghtsize_y,
        sparse_iact,
        sparse_wght,
        sel_iacts_zero,
        sel_wghts_zero,
        params,
        pe_iact_cycles * params.NUM_GLB_IACT,
    )
    await cocotb.start_soon(test_pe_cluster(dut, params, iacts, wghts, psums))

async def send_iact(dut, params, data_array):
    global signals_dict
    spad_data = [[0 for x in range(params.NUM_GLB_IACT)] for y in range(pe_iact_cycles)]
    for cycle in range(pe_iact_cycles):
        for glb_cluster in range(params.NUM_GLB_IACT):
            spad_data[cycle][glb_cluster] = generate_spad(
                data_array[glb_cluster + cycle * params.NUM_GLB_IACT],
                params.Iacts_Addr_per_PE,
                params.Iacts_per_PE,
                params.IACT_Bitwidth,
                True,
                0,
                True,
                True
            )

    max_transimission_addr = [[] for x in range(pe_iact_cycles)]
    current_max = 0
    for cycle in range(pe_iact_cycles):
        for glb_cluster in range(params.NUM_GLB_IACT):
            for x in range(len(spad_data[cycle][glb_cluster][0])):
                if (spad_data[cycle][glb_cluster][0][x] != 0):
                    current_max = x + 2
            max_transimission_addr[cycle].append(current_max)
    print("MAX: " + str(max_transimission_addr))
    print("ADDR: " + str(spad_data[cycle][glb_cluster][0]))
    max_transimission_data = [[] for x in range(pe_iact_cycles)]
    current_max = 0
    for cycle in range(pe_iact_cycles):
        for glb_cluster in range(params.NUM_GLB_IACT):
            for x in range(len(spad_data[cycle][glb_cluster][1])):
                if (spad_data[cycle][glb_cluster][1][x] != 0):
                    current_max = x + 2
            max_transimission_data[cycle].append(current_max)
    print("MAX: " + str(max_transimission_data))

    for cycle in range(pe_iact_cycles):
        for glb_cluster in range(params.NUM_GLB_IACT):
            for x in range(len(spad_data[cycle][glb_cluster][1])):
                if((spad_data[cycle][glb_cluster][1][x] != 0)):
                    spad_data[cycle][glb_cluster][1][x] = spad_data[cycle][glb_cluster][1][x] + int(math.floor(spad_data[cycle][glb_cluster][1][x-1] / 256) * 256)
            for x in range(len(spad_data[cycle][glb_cluster][1])):
                if((spad_data[cycle][glb_cluster][1][x] != 0)):
                    spad_data[cycle][glb_cluster][1][x] = spad_data[cycle][glb_cluster][1][x] + x * 256

    thread = []
    for cycle in range(pe_iact_cycles):
        for pe_x in range(params.PEs_X):
            for pe_y in range(params.PEs_Y):
                if ((pe_x + pe_y) >= (cycle * params.NUM_GLB_IACT)) & (
                    (pe_x + pe_y) < ((params.NUM_GLB_IACT * cycle) + params.NUM_GLB_IACT)
                ):
                    cocotb.start_soon(set_flat_input(signals_dict["iact_choose_i"][pe_x][pe_y], pe_x + pe_y - (cycle * params.NUM_GLB_IACT)))
                else:
                    cocotb.start_soon(set_flat_input(signals_dict["iact_choose_i"][pe_x][pe_y], 3))
        
        for glb_cluster in range(params.NUM_GLB_IACT):
            
            thread.append(
                cocotb.start_soon(
                    send_both_spads(
                        spad_data[cycle][glb_cluster][0],
                        signals_dict["pe_iact_data"][glb_cluster],
                        signals_dict["pe_iact_enable"][glb_cluster],
                        max_transimission_addr[cycle][glb_cluster],
                        params.IACT_Trans_Bitwidth,
                        int(math.ceil(math.log2(params.Iacts_per_PE))),
                        False,
                        "IACT_ADDR_" + str(glb_cluster) + "_" + str(cycle),
                        spad_data[cycle][glb_cluster][1],
                        signals_dict["pe_iact_data"][glb_cluster],
                        signals_dict["pe_iact_enable"][glb_cluster],
                        max_transimission_data[cycle][glb_cluster],
                        params.IACT_Trans_Bitwidth,
                        12,
                        False,
                        "IACT_DATA_" + str(glb_cluster) + "_" + str(cycle)
                    )
                )
            )
        for glb_cluster in range(params.NUM_GLB_IACT):
            await thread[glb_cluster]
        thread = []

async def get_psum(dut, params, iacts_array, wghts_array, psum_array):
    control = np.zeros(
        params.NUM_GLB_PSUM * params.Psums_per_PE * 2, dtype=int
    ).reshape(params.NUM_GLB_PSUM, params.Psums_per_PE * 2)

    iact = iacts_array
    wght = wghts_array
    bias = psum_array

    if iact.ndim == 1:
        iact = [iact]

    if wght.ndim == 1:
        wght = [wght]

    for glb_cluster in range(params.NUM_GLB_PSUM):
        for psum_x in range(len(bias[glb_cluster])):
            control[glb_cluster][psum_x] = bias[glb_cluster][psum_x]

    current_iact = 0
    iact_line = 0
    for pe_x in range(params.NUM_GLB_PSUM):
        for pe_y in range(params.NUM_GLB_WGHT):
            for iact_y in range(len(iact[pe_x + pe_y])):
                for iact_x in range(len(iact[pe_x + pe_y][iact_y])):
                    for wght_x in range(len(wght[pe_y][current_iact])):
                        control[pe_x][wght_x] = (
                            control[pe_x][wght_x]
                            + wght[pe_y][current_iact + iact_line][wght_x]
                            * iact[pe_x + pe_y][iact_y][iact_x]
                        )
                        if (pe_x == 1) & (wght_x == 0):
                            print(
                                "Iact is ", iact[pe_x + pe_y][iact_y][iact_x]
                            )
                            print(
                                "Wght is ",
                                wght[pe_y][current_iact + iact_line][wght_x]

                            )
                            print(
                                "Partial is ",
                                wght[pe_y][current_iact + iact_line][wght_x]
                                * iact[pe_x + pe_y][iact_y][iact_x],
                            )
                            print("Control is ", control[pe_x][wght_x])
                    current_iact = current_iact + 1
                iact_line = current_iact + iact_line
                current_iact = 0
            iact_line = 0
    thread = []
    global first_error_found
    first_error_found = 0

    for pe_x in range(params.NUM_GLB_PSUM):
        thread.append(cocotb.start_soon(check_psum(dut, pe_x, control[pe_x])))

    for pe_x in range(params.NUM_GLB_PSUM):
        await thread[pe_x]
    print("First error is: " + str(first_error_found))
    assert first_error_found == 0, "Outcoming Partial Sums are not equal to Calculated data!"

async def check_psum(dut, pe_x, control):
    current_control = 0
    global first_error_found
    while dut.pe_router_psum_enable_o[pe_x].value == 1:
        # Make sure there are no X values for gate level simulation
        assert 'x' not in dut.pe_router_psum_data_o[pe_x].value, "x values in PSUM"
        
        if ((read_flat_output(signals_dict["pe_router_psum_data_o"][pe_x])%(2**20)) == control[current_control]):
            print(
            "PSUM("
            + str(read_flat_output(signals_dict["pe_router_psum_data_o"][pe_x])%(2**20))
            + ") is equal to control("
            + str(control[current_control])
            + "), "
            + str(current_control + 1)
            + ". PSUM Value, "
            + str(pe_x + 1)
            + ". PE_X"
            )
        else:
            print(
            "PSUM("
            + str(read_flat_output(signals_dict["pe_router_psum_data_o"][pe_x])%(2**20))
            + ") is not equal to control("
            + str(control[current_control])
            + "), "
            + str(current_control + 1)
            + ". PSUM Value, "
            + str(pe_x + 1)
            + ". PE_X"
            )
            first_error_found = 1

            
        current_control = current_control + 1
        
        if (int(read_flat_output(signals_dict["pe_router_psum_data_o"][pe_x])/(2**20)) == control[current_control]):
            print(
            "PSUM("
            + str(int(read_flat_output(signals_dict["pe_router_psum_data_o"][pe_x])/(2**20)))
            + ") is equal to control("
            + str(control[current_control])
            + "), "
            + str(current_control + 1)
            + ". PSUM Value, "
            + str(pe_x + 1)
            + ". PE_X"
            )
        else:
            print(
            "PSUM("
            + str(int(read_flat_output(signals_dict["pe_router_psum_data_o"][pe_x])/(2**20)))
            + ") is not equal to control("
            + str(control[current_control])
            + "), "
            + str(current_control + 1)
            + ". PSUM Value, "
            + str(pe_x + 1)
            + ". PE_X"
            )
            first_error_found = 1
        current_control = current_control + 1
        await RisingEdge(clk.signal)
        await Timer(clk_delay_out, units=clk_delay_unit_out)

async def send_wght(dut, params, data_array):
    global signals_dict
    spad_data = []
    for glb_cluster in range(params.NUM_GLB_WGHT):
        spad_data.append(
            generate_spad(
                data_array[glb_cluster],
                params.Wghts_Addr_per_PE,
                params.Wghts_per_PE,
                params.WGHT_Bitwidth,
                False,
                params.WGHT_WOH_Bitwidth,
                True,
                False
            )
        )
    for glb_cluster in range(params.NUM_GLB_WGHT):
        for x in reversed(range(len(spad_data[glb_cluster][0]))):
            if(x == 0):
                spad_data[glb_cluster][0][x] = 0
            else:
                spad_data[glb_cluster][0][x] = spad_data[glb_cluster][0][x - 1]
    
    max_transimission_addr = []
    current_max = params.Wghts_Addr_per_PE
    for glb_cluster in range(params.NUM_GLB_WGHT):
        for x in range(len(spad_data[glb_cluster][0]) - 1):
            if (spad_data[glb_cluster][0][params.Wghts_Addr_per_PE - x - 1] == 0):
                current_max = params.Wghts_Addr_per_PE - x
        max_transimission_addr.append(current_max)

    max_transimission_data = []
    current_max = 0
    for glb_cluster in range(params.NUM_GLB_WGHT):
        for x in range(len(spad_data[glb_cluster][1])):
            if (spad_data[glb_cluster][1][x] != 0):
                current_max = x + 2
        max_transimission_data.append(current_max)
        
    thread = []
    for glb_cluster in range(params.NUM_GLB_WGHT):
        
        thread.append(
            cocotb.start_soon(
                send_both_spads(
                    spad_data[glb_cluster][0],
                    signals_dict["pe_wght_data"][glb_cluster],
                    signals_dict["pe_wght_enable"][glb_cluster],
                    int(max_transimission_addr[glb_cluster]), 
                    params.WGHT_Trans_Bitwidth,
                    int(math.ceil(math.log2(params.Wghts_per_PE/params.PARALLEL_MACS))),
                    False,
                    "WGHT_ADDR_" + str(glb_cluster),
                    spad_data[glb_cluster][1],
                    signals_dict["pe_wght_data"][glb_cluster],
                    signals_dict["pe_wght_enable"][glb_cluster],
                    int(max_transimission_data[glb_cluster]), 
                    params.WGHT_Trans_Bitwidth,
                    24,
                    False,
                    "WGHT_DATA_" + str(glb_cluster)
                )
            )
        )
    for glb_cluster in range(params.NUM_GLB_WGHT):
        await thread[glb_cluster]

async def send_bias(dut, params, data_array):
    global signals_dict
    spad_data = []
    for glb_cluster in range(params.NUM_GLB_PSUM):
        spad_data.append(
            generate_spad(
                data_array[glb_cluster],
                params.Psums_per_PE,
                params.Psums_per_PE,
                params.PSUM_Bitwidth,
                False,
                params.PSUM_Bitwidth,
                False,
                False
            )
        )
    #dut._log.info("PSUM is %s", spad_data[1])
    thread = []
    for glb_cluster in range(params.NUM_GLB_PSUM):
        thread.append(
            cocotb.start_soon(
                send_to_spad(
                    spad_data[glb_cluster][1],
                    signals_dict["pe_router_psum_data_i"][glb_cluster],
                    signals_dict["pe_router_psum_enable_i"][glb_cluster],
                    math.ceil(params.Psums_per_PE/params.PARALLEL_MACS),
                    params.PSUM_Bitwidth * 2,
                    params.PSUM_Trans_Bitwidth,
                    False,
                    "PSUM_" + str(glb_cluster)
                )
            )
        )

    for glb_cluster in range(params.NUM_GLB_PSUM):
        await thread[glb_cluster]
    await RisingEdge(clk.signal)

async def send_both_spads(spad1, data_signal_dict_1, enable_signal_dict_1, addr_bits1, trans_bits1, data_bits1, parallel1,txt_name1,
                          spad2, data_signal_dict_2, enable_signal_dict_2, addr_bits2, trans_bits2, data_bits2, parallel2,txt_name2):
    process = cocotb.start_soon(
        send_to_spad(
            spad1,
            data_signal_dict_1,
            enable_signal_dict_1,
            addr_bits1, 
            trans_bits1,
            data_bits1,
            parallel1,
            txt_name1
        )
    )
    await process
    process = cocotb.start_soon(
        send_to_spad(
            spad2,
            data_signal_dict_2,
            enable_signal_dict_2,
            addr_bits2, 
            trans_bits2,
            data_bits2,
            parallel2,
            txt_name2
        )
    )
    await process
    
async def send_to_spad(spad, data_signal_dict, enable_signal_dict, addr_bits, trans_bits, data_bits, parallel,txt_name):
    txt = pe_cluster_test_utils.open_or_create_file("TXTs/" + str(txt_name) + ".txt")
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
        cocotb.start_soon(set_flat_input(data_signal_dict, sending_data))
        cocotb.start_soon(set_flat_input(enable_signal_dict, 1))
        txt.write(str(bin(sending_data)[2:].zfill(trans_bits)) + "\n")
        sending_data = 0
        await RisingEdge(clk.signal)
    cocotb.start_soon(set_flat_input(data_signal_dict, 0))
    cocotb.start_soon(set_flat_input(enable_signal_dict, 0))
    txt.close()

def generate_spad(
    array, addr_spad_words, data_spad_words, bitwidth, sisd, offset, ignore_zeros,count_around_lines
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
        if count_around_lines == False:
            overhead = 0
            
    spad_data = (addr_spad_data, data_spad_data)
    return spad_data

def create_iact_wght_psum_arrays(
    iactsize_x,
    iactsize_y,
    wghtsize_x,
    wghtsize_y,
    sparse_iact,
    sparse_wght,
    sel_iacts_zero,
    sel_wghts_zero,
    params,
    pe_iact_cycles,
):
    temp = ((np.arange(0, iactsize_y * iactsize_x * pe_iact_cycles, 1)%126)+1)
    indices = np.random.choice(
        np.arange(temp.size), replace=False, size=int(temp.size * sparse_iact)
    )
    indices = np.append(indices, sel_iacts_zero)
    temp[indices] = 0
    iacts = temp.reshape(
        pe_iact_cycles, iactsize_y, iactsize_x
    )

    temp = ((np.arange(0, wghtsize_y * wghtsize_x * params.PEs_Y, 1)%126)+1)
    indices = np.random.choice(
        np.arange(temp.size), replace=False, size=int(temp.size * sparse_wght)
    )
    indices = np.append(indices, sel_wghts_zero)
    temp[indices] = 0
    wghts = temp.reshape(
        params.PEs_Y, wghtsize_y, wghtsize_x
    )

    psums = np.arange(1, wghtsize_x * params.PEs_X + 1, 1).reshape(params.PEs_X, wghtsize_x)

    return iacts, wghts, psums

async def reset_all_signals(dut, params):

    cocotb.start_soon(set_flat_input(signals_dict["rst_ni"], 0))
    cocotb.start_soon(set_flat_input(signals_dict["data_mode_i"], 0))
    cocotb.start_soon(set_flat_input(signals_dict["fraction_bit_i"], 0))
    
    for pe_columns in range(params.PEs_X):
        for pe_rows in range(params.PEs_Y):
            cocotb.start_soon(set_flat_input(signals_dict["iact_choose_i"][pe_columns][pe_rows], 3))
            cocotb.start_soon(set_flat_input(signals_dict["compute_i"][pe_columns][pe_rows], 0))

    for glb_iact in range(params.NUM_GLB_IACT):
        cocotb.start_soon(set_flat_input(signals_dict["pe_iact_enable"][glb_iact], 0))
        cocotb.start_soon(set_flat_input(signals_dict["pe_iact_data"][glb_iact], 0))

    for glb_wght in range(params.NUM_GLB_WGHT):
        cocotb.start_soon(set_flat_input(signals_dict["pe_wght_enable"][glb_wght], 0))
        cocotb.start_soon(set_flat_input(signals_dict["pe_wght_data"][glb_wght], 0))

    for glb_psum in range(params.NUM_GLB_PSUM):
        cocotb.start_soon(set_flat_input(signals_dict["psum_choose_i"][glb_psum], 1))

        cocotb.start_soon(set_flat_input(signals_dict["pe_psum_data_i"][glb_psum], 0))
        cocotb.start_soon(set_flat_input(signals_dict["pe_psum_enable_i"][glb_psum], 0))
        cocotb.start_soon(set_flat_input(signals_dict["pe_psum_ready_i"][glb_psum], 0))

        cocotb.start_soon(set_flat_input(signals_dict["pe_router_psum_data_i"][glb_psum], 0))
        cocotb.start_soon(set_flat_input(signals_dict["pe_router_psum_enable_i"][glb_psum], 0))
        cocotb.start_soon(set_flat_input(signals_dict["pe_router_psum_ready_i"][glb_psum], 0))


    # Fixed 10 ns of reset
    await Timer(clk_cycle, units="ns")
    cocotb.start_soon(set_flat_input(signals_dict["rst_ni"], 1))

    # After deasserting reset, we wait 3 clock cycles
    for _ in range(3):
        await RisingEdge(clk.signal)

async def create_dict(dut, params):
    global signals_dict
    current_signal = {"signal":dut.clk_i, "start_bit":0, "end_bit":0}
    signals_dict["clk_i"] = current_signal
    current_signal = {"signal":dut.rst_ni, "start_bit":0, "end_bit":0}
    signals_dict["rst_ni"] = current_signal
    current_signal = {"signal":dut.data_mode_i, "start_bit":0, "end_bit":0}
    signals_dict["data_mode_i"] = current_signal
    current_signal = {"signal":dut.fraction_bit_i, "start_bit":4, "end_bit":0}
    signals_dict["fraction_bit_i"] = current_signal
    current_signal = []
    for x in range(params.PEs_X):
        current_signal.append([])
        for y in range(params.PEs_Y):
            current_signal[x].append({"signal":dut.compute_i, \
                                      "start_bit":x+y*params.PEs_X, \
                                      "end_bit":x+y*params.PEs_X})
    signals_dict["compute_i"] = current_signal
    current_signal = []
    for x in range(params.PEs_X):
        current_signal.append([])
        for y in range(params.PEs_Y):
            current_signal[x].append({"signal":dut.iact_choose_i, \
                                      "start_bit":2*x+2*y*params.PEs_X, \
                                      "end_bit":1+2*x+2*y*params.PEs_X})
    signals_dict["iact_choose_i"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_IACT):
        current_signal.append({"signal":dut.pe_iact_enable, "start_bit":glb, "end_bit":glb})
    signals_dict["pe_iact_enable"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_IACT):
        current_signal.append({"signal":dut.pe_iact_data, \
                               "start_bit":glb*params.IACT_Trans_Bitwidth, \
                                "end_bit":(glb+1)*params.IACT_Trans_Bitwidth-1})
    signals_dict["pe_iact_data"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_IACT):
        current_signal.append({"signal":dut.pe_iact_ready, "start_bit":glb, "end_bit":glb})
    signals_dict["pe_iact_ready"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_WGHT):
        current_signal.append({"signal":dut.pe_wght_enable, "start_bit":glb, "end_bit":glb})
    signals_dict["pe_wght_enable"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_WGHT):
        current_signal.append({"signal":dut.pe_wght_data, \
                               "start_bit":glb*params.WGHT_Trans_Bitwidth, \
                               "end_bit":(glb+1)*params.WGHT_Trans_Bitwidth-1})
    signals_dict["pe_wght_data"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_WGHT):
        current_signal.append({"signal":dut.pe_wght_ready, "start_bit":glb, "end_bit":glb})
    signals_dict["pe_wght_ready"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_PSUM):
        current_signal.append({"signal":dut.psum_choose_i, "start_bit":glb, "end_bit":glb})
    signals_dict["psum_choose_i"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_PSUM):
        current_signal.append({"signal":dut.pe_psum_enable_i, "start_bit":glb, "end_bit":glb})
    signals_dict["pe_psum_enable_i"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_PSUM):
        current_signal.append({"signal":dut.pe_psum_data_i, \
                                "start_bit":glb*params.PSUM_Trans_Bitwidth, \
                                "end_bit":(glb+1)*params.PSUM_Trans_Bitwidth-1})
    signals_dict["pe_psum_data_i"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_PSUM):
        current_signal.append({"signal":dut.pe_psum_ready_i, "start_bit":glb, "end_bit":glb})
    signals_dict["pe_psum_ready_i"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_PSUM):
        current_signal.append({"signal":dut.pe_psum_enable_o, "start_bit":glb, "end_bit":glb})
    signals_dict["pe_psum_enable_o"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_PSUM):
        current_signal.append({"signal":dut.pe_psum_data_o, \
                                "start_bit":glb*params.PSUM_Trans_Bitwidth, \
                                "end_bit":(glb+1)*params.PSUM_Trans_Bitwidth-1})
    signals_dict["pe_psum_data_o"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_PSUM):
        current_signal.append({"signal":dut.pe_psum_ready_o, "start_bit":glb, "end_bit":glb})
    signals_dict["pe_psum_ready_o"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_PSUM):
        current_signal.append({"signal":dut.pe_router_psum_enable_i, "start_bit":glb, "end_bit":glb})
    signals_dict["pe_router_psum_enable_i"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_PSUM):
        current_signal.append({"signal":dut.pe_router_psum_data_i, \
                                "start_bit":glb*params.PSUM_Trans_Bitwidth, \
                                "end_bit":(glb+1)*params.PSUM_Trans_Bitwidth-1})
    signals_dict["pe_router_psum_data_i"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_PSUM):
        current_signal.append({"signal":dut.pe_router_psum_ready_i, "start_bit":glb, "end_bit":glb})
    signals_dict["pe_router_psum_ready_i"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_PSUM):
        current_signal.append({"signal":dut.pe_router_psum_enable_o, "start_bit":glb, "end_bit":glb})
    signals_dict["pe_router_psum_enable_o"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_PSUM):
        current_signal.append({"signal":dut.pe_router_psum_data_o, \
                                "start_bit":glb*params.PSUM_Trans_Bitwidth, \
                                "end_bit":(glb+1)*params.PSUM_Trans_Bitwidth-1})
    signals_dict["pe_router_psum_data_o"] = current_signal
    current_signal = []
    for glb in range(params.NUM_GLB_PSUM):
        current_signal.append({"signal":dut.pe_router_psum_ready_o, "start_bit":glb, "end_bit":glb})
    signals_dict["pe_router_psum_ready_o"] = current_signal

def read_flat_output(signal_dict):
    signal = signal_dict["signal"]
    start_bit = signal_dict["start_bit"]
    end_bit = signal_dict["end_bit"]
    if(len(signal) != 1):
        value = 0
        value_pos = 1
        for bit in range(start_bit, end_bit + 1):
            value = int(signal[bit].value) * value_pos + value
            value_pos = value_pos * 2
        return value
    else:
        return signal.value

async def set_flat_input(signal_dict, new_value):
    # input delay of 100 ps relative to rising edge
    # this is the value used for the implementation constraints of OpenEye
    signal = signal_dict["signal"]
    start_bit = signal_dict["start_bit"]
    end_bit = signal_dict["end_bit"]
    await Timer(clk_delay_in, units=clk_delay_unit_in)
    if(len(signal) != 1):
        value_pos = 0
        new_value = bin(new_value)[2:]
        new_value = new_value[::-1]
        for bit in range(start_bit, end_bit + 1):
            try:
                signal[bit].value = int(new_value[value_pos])
            except:
                signal[bit].value = 0
            value_pos = value_pos + 1
    else:
        signal.value = new_value
