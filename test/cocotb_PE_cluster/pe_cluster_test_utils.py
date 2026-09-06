# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
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
        os.path.join(hdl_dir, "PE_simple.v"),
        os.path.join(hdl_dir, "adder.v"),
        os.path.join(hdl_dir, "adder_tree.v"),
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

def use_pe_simple():
    """True when the cluster is compiled with the dense PE_simple.v variant.

    Selected by the PE_MODULE env var that the GEMM pytest runner exports.
    PE_simple.v has a different config-word bit layout and dense weight packing
    than the sparsity-capable PE.v, so the testbench feed branches on this.
    """
    return os.environ.get("PE_MODULE", "PE") == "PE_simple"


async def send_data_params(ptp, dut, iactsize_x, iactsize_y, wghtsize_x):
    # List all needed parameters
    stride_reg      = 1
    filters_reg_i   = wghtsize_x          # M0 output channels
    channel_reg_i   = iactsize_y          # C0
    iact_addr_max_i = iactsize_x

    # Enable the params reading
    cocotb.start_soon(rtl_test_utils.set_input(ptp, (dut.enable_stream_i), 1))

    # Log DUT compile-time/run-time configuration for easier debugging
    try:
        cocotb.log.info(f"DUT PARALLEL_MACS = {int(dut.PARALLEL_MACS.value)}")
    except Exception:
        cocotb.log.info(f"DUT PARALLEL_MACS not available on DUT instance")
    cocotb.log.info(f"Environment SPARSITY_EN = {os.environ.get('SPARSITY_EN', 'unset')}" )

    if use_pe_simple():
        # PE_simple.v config-word layout (matches hdl/PE_simple.v stream FSM):
        #   FIRST  : [11:9]=stride, [8:1]=wght_addr_max
        #   SECOND : [11:7]=M0,     [6:3]=C0
        #   THIRD  : [iact_bw:1]=iact_addr_max, [0]=data_mode
        # Dense weights occupy N*M0 values, PARALLEL_MACS per loaded word.
        n_terms       = iactsize_x * iactsize_y
        pmacs         = int(dut.PARALLEL_MACS.value)
        wght_words    = int(math.ceil((n_terms * wghtsize_x) / pmacs))
        wght_addr_max = wght_words - 1            # last weight-word index
        iact_addr_max = n_terms - 1               # last iact index

        data_reg_i = (stride_reg << 9) | (wght_addr_max << 1)
        cocotb.start_soon(rtl_test_utils.set_input(ptp, (dut.data_stream_i), data_reg_i))
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        data_reg_i = (filters_reg_i << 7) | (channel_reg_i << 3)
        cocotb.start_soon(rtl_test_utils.set_input(ptp, (dut.data_stream_i), data_reg_i))
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        data_reg_i = (iact_addr_max << 1)
        cocotb.start_soon(rtl_test_utils.set_input(ptp, (dut.data_stream_i), data_reg_i))
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        # FOURTH params (fraction bits / line repetitions): unused for GEMM
        cocotb.start_soon(rtl_test_utils.set_input(ptp, (dut.data_stream_i), 0))
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    else:
        # PE.v (sparse) config-word layout. PE.v shifts each 9-bit
        # data_stream_i word through a 3-cycle register (stream_data);
        # by the time all 3 cycles have arrived, cycle 1's word has
        # shifted up to stream_data[26:18] and cycle 3's word is sitting
        # unread in stream_data[8:0] (nothing consumes it). The bit
        # layout PE.v actually decodes is:
        #   FIRST  : [8]=raw_wght, [7:4]=iact_x_line_repetitions, [3:0]=iact_addr_max
        #   SECOND : [3:0]=C0 (channels)
        #   THIRD  : [5:0]=M0 (filters)
        # M0 moved out of the SECOND word and widened to 6 bits in e8f39ad
        # (PE.v: filters_reg_M0 = stream_data[5:0], was stream_data[17:13]).
        # raw_wght must mirror send_wght()'s packing: sparse mode packs with
        # ignore_zeros=True (zero-skip encoding), which needs raw_mode_i 0,
        # while dense mode sends every weight verbatim, which data_pipeline_wght
        # only stores when raw_mode_i is 1 (zero weights are dropped otherwise).
        raw_wght = 0 if int(dut.SPARSITY_EN.value) == 1 else 1
        data_reg_i = (iact_addr_max_i << 0) | (raw_wght << 8)
        cocotb.start_soon(rtl_test_utils.set_input(ptp, (dut.data_stream_i), data_reg_i))
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        data_reg_i = (channel_reg_i << 0)
        cocotb.start_soon(rtl_test_utils.set_input(ptp, (dut.data_stream_i), data_reg_i))
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        data_reg_i = (filters_reg_i << 0)
        cocotb.start_soon(rtl_test_utils.set_input(ptp, (dut.data_stream_i), data_reg_i))
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

    # Disable the params reading
    cocotb.start_soon(rtl_test_utils.set_input(ptp, (dut.enable_stream_i), 0))
    await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
   
def send_to_wght_spad(ptp, spad, dut, words):
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
    words_per_transmit = 1
    # generate_spad packs one weight per SPAD word in SISD mode and two in
    # packed mode, so that is how many transfers carry real data.
    values_per_word = 1 if int(dut.PARALLEL_MACS.value) == 1 else 2
    dense_mode = int(dut.SPARSITY_EN.value) == 0
    valid_transfers = math.ceil(words / values_per_word)
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
        # Send the packed transmission.
        # As on the iact path: sparse mode drops all-zero transfers because the
        # zeros were compressed away, dense mode must keep sending them.
        data_array[1].append(sending_data)
        if dense_mode:
            data_array[0].append(1 if cycle < valid_transfers else 0)
        elif (sending_data != 0) :
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
    # PE.v (sparse) unpacks two 12-bit iact values per 24-bit transfer; PE_simple
    # consumes one 8-bit value per transfer (low bits of the bus). Match the
    # packing to whichever PE the cluster was compiled with.
    if use_pe_simple():
        words_per_transmit = 1
        word_stride        = int(dut.DATA_IACT_BITWIDTH.value)
    else:
        if (int(dut.SPARSITY_EN.value) == 1): 
            word_stride        = int(dut.DATA_IACT_BITWIDTH.value) + int(dut.DATA_IACT_OVERHEAD.value)
        else:
            word_stride        = int(dut.DATA_IACT_BITWIDTH.value)
        # data_pipeline_iact unpacks DATA_WIDTH/SECOND_SPAD_DATA sub-words per
        # transfer, so the packing has to follow the compiled bus width instead
        # of assuming the 24-bit (2 x 12 bit) sparse case.
        words_per_transmit = int(dut.TRANS_BITWIDTH_IACT.value) // word_stride
    # The position stepping below repeats each group of words_per_transmit SPAD
    # words for that many cycles, so a transfer still carries data up to the
    # cycle whose group holds the last word.
    valid_transfers = math.ceil(words / words_per_transmit) * words_per_transmit
    # Send data over multiple clock cycles
    for cycle in range(int(dut.IACT_DATA_WORDS.value)):
        # Calculate how many words fit in one transmission
        # Pack multiple words into this transmission
        for writing_cycle in range(words_per_transmit):
            offset = word_stride * writing_cycle  # Bit position for this word
            # Sequential mode: pack words sequentially
            current_storage_position = int(writing_cycle+ math.floor(cycle / words_per_transmit) * words_per_transmit)
            try:
                # Add this word to the transmission, shifted to correct position
                sending_data = sending_data + (int(spad[current_storage_position]) << offset)
            except:
                # Handle out of bounds (sparse data)
                sending_data = sending_data
        # Send the packed transmission.
        # Sparse mode compresses zeros away, so an all-zero transfer is padding
        # past the end of the data and stays disabled. Dense mode stores every
        # value including the zeros, so the enable has to follow the word count
        # instead - dropping a zero transfer would starve the PE of a word.
        data_array[1].append(sending_data)
        # Activations are always handed over uncompressed - send_iact builds the
        # SPAD with ignore_zeros=False, and data_pipeline_iact does the
        # zero-skipping itself, tagging each stored value with its source
        # position. A zero activation therefore still has to be transferred: its
        # position is what the tag counts. Gating the enable on a non-zero word
        # dropped that transfer, so every later activation was tagged one
        # position too low and got paired with the wrong weight row.
        data_array[0].append(1 if cycle < valid_transfers else 0)
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