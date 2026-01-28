# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import logging
logger = logging.getLogger("test_logger")
import math
import sys
import os
import numpy as np
from numpy import genfromtxt

import cocotb
from cocotb.triggers import Timer, Combine
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer, with_timeout, SimTimeoutError

# Add parent directory to path for importing OpenEye modules
import open_eye.timing_parameters as timing_parameters
import open_eye.rtl_test_utils as rtl_test_utils

def get_verilog_sources(hdl_dir):

    verilog_sources = [
        os.path.join(hdl_dir, "PE_cluster.v"),
        os.path.join(hdl_dir, "RST_SYNC.v"),
        os.path.join(hdl_dir, "PE.v"),
        os.path.join(hdl_dir, "adder.v"),
        os.path.join(hdl_dir, "data_pipeline.v"),
        os.path.join(hdl_dir, "multiplier.v"),
        os.path.join(hdl_dir, "mux2.v"),
        os.path.join(hdl_dir, "demux2.v"),
        os.path.join(hdl_dir, "mux_iact.v"),
        os.path.join(hdl_dir, "SPad_DP_RW.v"),
        os.path.join(hdl_dir, "SPad_SP.v"),
        os.path.join(hdl_dir, "SPad_DP.v"),
        os.path.join(hdl_dir, "RAM_DP_RW.v"),
        os.path.join(hdl_dir, "RAM_DP.v"),
        os.path.join(hdl_dir, "RAM_DP_generic.v"),
        os.path.join(hdl_dir, "RAM_SP.v"),
        os.path.join(hdl_dir, "RAM_DP_RW_generic.v"),
        os.path.join(hdl_dir, "RAM_SP_generic.v"),
        os.path.join(hdl_dir, "data_pipeline_iact.v"),
        os.path.join(hdl_dir, "data_pipeline_wght.v")
    ]
    return verilog_sources


async def reset_all_signals(ptp,dut):

    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.clk_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.rst_ni), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.compute_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.iact_choose_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.psum_choose_i), 15))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_iact_enable), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_iact_data), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_wght_enable), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_wght_data), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_psum_data_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_psum_enable_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_psum_ready_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_router_psum_data_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_router_psum_enable_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_router_psum_ready_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.enable_stream_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.data_stream_i), 0))

    # Fixed 3 clock cycles of reset
    await Timer(3*ptp.clk_cycle, units="ns")
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.rst_ni), 1))
    dut.iact_choose_i.value = 1

    # After deasserting reset, we wait 3 clock cycles
    await Timer(3*ptp.clk_cycle, units="ns")

async def send_data_params(ptp, dut, iactsize_x, iactsize_y,wghtsize_x):
    # List all needed parameters
    stride_reg = 1
    wght_addr_max_reg = (iactsize_x * iactsize_y) + 2
    filters_reg_i = wghtsize_x
    channel_reg_i = iactsize_y
    iact_addr_max_i = iactsize_x

    data_reg_i =  0
    # Enable the params reading
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.enable_stream_i), 1))

    data_reg_i =  (wght_addr_max_reg << 4) + (stride_reg << 1)
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.data_stream_i), data_reg_i))
    await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    data_reg_i =  (filters_reg_i << 4) + (channel_reg_i << 0)
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.data_stream_i), data_reg_i))
    await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    data_reg_i =  (iact_addr_max_i << 0)
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.data_stream_i), data_reg_i))
    await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

    # Disable the params reading
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.enable_stream_i), 0))
    await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
   
def send_to_wght_spad(ptp, spad, dut):
    """
    Generic function to transmit SPAD data over a limited-width bus.

    Packs multiple data words into each transmission cycle based on bus width,
    then sends them serially over multiple clock cycles.

    Args:
        ptp: Port timing parameters for signal timing
        spad: Array of data words to transmit
        dut: Device under Test

    Operation:
        - Calculates words_per_transmit = trans_bits / data_bits
        - Packs multiple words into single transmission by bit-shifting
        - Sends one transmission per clock cycle
        - Handles index out of bounds gracefully

    """
    words_per_transmit = 0
    sending_data = 0
    current_storage_position = 0
    offset = 0
    data_array = [[],[]]
    # Send data over multiple clock cycles
    words_per_transmit = int(math.floor(24 / 24))
    for cycle in range(int(dut.WGHT_DATA_WORDS.value)):
        # Calculate how many words fit in one transmission
        # Pack multiple words into this transmission
        for writing_cycle in range(words_per_transmit):
            offset = int(dut.WGHT_DATA_WORDS.value) * writing_cycle  # Bit position for this word
            # Sequential mode: pack words sequentially
            current_storage_position = int(
                writing_cycle
                + math.floor(cycle / words_per_transmit) * words_per_transmit
            )
            try:
                # Add this word to the transmission, shifted to correct position
                sending_data = sending_data + (int(spad[current_storage_position]) << offset)
            except:
                # Handle out of bounds (sparse data)
                sending_data = sending_data
        # Send the packed transmission
        data_array[1].append(sending_data)
        if (sending_data != 0) :
            data_array[0].append(1)
        else: 
            data_array[0].append(0)

        sending_data = 0
    return data_array   
def send_to_iact_spad(ptp, spad, dut, words):
    """
    Generic function to transmit SPAD data over a limited-width bus.

    Packs multiple data words into each transmission cycle based on bus width,
    then sends them serially over multiple clock cycles.

    Args:
        ptp: Port timing parameters for signal timing
        spad: Array of data words to transmit
        dut: Device under Test

    Operation:
        - Calculates words_per_transmit = trans_bits / data_bits
        - Packs multiple words into single transmission by bit-shifting
        - Sends one transmission per clock cycle
        - Handles index out of bounds gracefully

    """
    words_per_transmit = 0
    sending_data = 0
    current_storage_position = 0
    offset = 0
    data_array = [[],[]]
    # Send data over multiple clock cycles
    words_per_transmit = int(math.floor(24 / 12))
    for cycle in range(int(dut.IACT_DATA_WORDS.value)):
        # Calculate how many words fit in one transmission
        # Pack multiple words into this transmission
        for writing_cycle in range(words_per_transmit):
            offset = 12 * writing_cycle  # Bit position for this word
            # Sequential mode: pack words sequentially
            current_storage_position = int(writing_cycle+ math.floor(cycle / words_per_transmit) * words_per_transmit)
            try:
                # Add this word to the transmission, shifted to correct position
                sending_data = sending_data + (int(spad[current_storage_position]) << offset)
            except:
                # Handle out of bounds (sparse data)
                sending_data = sending_data
        # Send the packed transmission
        data_array[1].append(sending_data)
        if (sending_data != 0) :
            data_array[0].append(1)
        else: 
            data_array[0].append(0)
        sending_data = 0
    return data_array

def send_to_psum_spad(ptp, spad, dut):
    """
    Generic function to transmit SPAD data over a limited-width bus.

    Packs multiple data words into each transmission cycle based on bus width,
    then sends them serially over multiple clock cycles.

    Args:
        ptp: Port timing parameters for signal timing
        spad: Array of data words to transmit
        dut: Device under Test

    Operation:
        - Calculates words_per_transmit = trans_bits / data_bits
        - Packs multiple words into single transmission by bit-shifting
        - Sends one transmission per clock cycle
        - Handles index out of bounds gracefully

    """
    words_per_transmit = 0
    sending_data = 0
    current_storage_position = 0
    offset = 0
    data_array = [[],[]]
    # Send data over multiple clock cycles
    words_per_transmit = int(math.floor(20 / 20))
    for cycle in range(int(dut.PSUM_WORDS.value)):
        # Calculate how many words fit in one transmission
        # Pack multiple words into this transmission
        for writing_cycle in range(words_per_transmit):
            offset = int(dut.PSUM_WORDS.value) * writing_cycle  # Bit position for this word
            # Sequential mode: pack words sequentially
            current_storage_position = int(
                writing_cycle
                + math.floor(cycle / words_per_transmit) * words_per_transmit
            )
            try:
                # Add this word to the transmission, shifted to correct position
                sending_data = sending_data + (int(spad[current_storage_position]) << offset)
            except:
                # Handle out of bounds (sparse data)
                sending_data = sending_data
        # Send the packed transmission
        data_array[1].append(sending_data)
        if (sending_data != 0) :
            data_array[0].append(1)
        else: 
            data_array[0].append(0)
        sending_data = 0
    return data_array   