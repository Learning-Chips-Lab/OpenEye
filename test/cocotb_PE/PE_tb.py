# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
Processing Element (PE) Testbench

This module provides a cocotb-based testbench for verifying a Processing Element (PE)
hardware module used in neural network acceleration. The PE performs multiply-accumulate
operations on input activations and weights, producing partial sums.

Key Features:
    - Tests PE with configurable input dimensions and sparsity
    - Supports compressed scratchpad (SPAD) memory format with zero-skipping
    - Validates MAC operations against software golden model
    - Handles parallel data loading and serial transmission to SPADs
"""

import math
import sys
import os
import numpy as np
from numpy import genfromtxt

import cocotb
from cocotb.triggers import Timer, Combine
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

# Add parent directory to path for importing OpenEye modules
directory = (os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
import open_eye.timing_parameters as timing_parameters
import open_eye.rtl_test_utils as rtl_test_utils

# Timing configuration from environment variables
clk_cycle = int(os.environ["CLOCK_LEN"])  # Clock cycle length
clk_cycle_unit = os.environ["CLOCK_UNIT"]  # Clock cycle unit (e.g., "ns", "ps")

clk_delay_in = int(os.environ["CLOCK_DELAY_INPUT"])  # Input delay
clk_delay_unit_in = os.environ["CLOCK_DELAY_UNIT_INPUT"]  # Input delay unit

clk_delay_out = int(os.environ["CLOCK_DELAY_OUTPUT"])  # Output delay
clk_delay_unit_out = os.environ["CLOCK_DELAY_UNIT_OUTPUT"]  # Output delay unit


# Dimensions for PE initiliazed as globals
iactsize_x = 0  # Number of input activation values (spatial dimension)
iactsize_y = 0  # Number of input channels
sparse_iact = 0 # Input activation sparsity: 0 = no sparsity, 1 = fully sparse
wghtsize_x = 0  # Number of output filters
wghtsize_y = 0  # Weights match input dimensions
sparse_wght = 0 # Weight sparsity: 0 = no sparsity, 1 = fully sparse

@cocotb.test()
async def start_test_pe(dut):
    """
    Main cocotb test entry point for PE verification.

    Configures test parameters, generates test data, and launches the main test
    sequence. This is the function that cocotb calls when running the test.

    Args:
        dut: Device Under Test (PE module instance from cocotb)

    Test Configuration:
        - 3 input activation values (dimensions)
        - 1 channel
        - 1 output filter
        - No sparsity (all values are non-zero)
    """
    # Configure test dimensions
    global iactsize_x   # Number of input activation values (spatial dimension)
    global iactsize_y   # Number of input channels
    global sparse_iact  # Input activation sparsity: 0 = no sparsity, 1 = fully sparse
    global wghtsize_x   # Number of output filters
    global wghtsize_y   # Weights match input dimensions
    global sparse_wght  # Weight sparsity: 0 = no sparsity, 1 = fully sparse

    iactsize_x = int(os.environ["IACTSIZE_X"])
    iactsize_y = int(os.environ["IACTSIZE_Y"])
    wghtsize_x = int(os.environ["WGHTSIZE_X"])
    sparse_iact = int(os.environ["SPARSE_IACT"])
    sparse_wght = int(os.environ["SPARSE_WGHT"])
    wghtsize_y = iactsize_x * iactsize_y

    # Initialize timing parameters from environment variables
    ptp = timing_parameters.PortTimingParameters()
    ptp.initiate_params(clk_cycle, clk_cycle_unit, clk_delay_in, clk_delay_unit_in, clk_delay_out, clk_delay_unit_out)


    # Generate test input data (activations, weights, partial sums)
    (iacts, wghts, psums) = create_iact_wght_psum_arrays(dut)

    # Launch main test sequence
    await cocotb.start_soon(test_hdls(ptp, dut, iacts, wghts, psums))

async def test_hdls(ptp, dut, iacts_array, wghts_array, psum_array):
    """
    Main test orchestration function for the PE module.

    This function coordinates the entire test sequence: initializing the clock,
    resetting signals, loading test data into SPADs, triggering computation,
    and validating outputs against a golden model.

    Args:
        ptp: Port timing parameters object for controlling signal timing
        dut: Device Under Test (the PE hardware module)
        iacts_array: Input activation test data (numpy array)
        wghts_array: Weight test data (numpy array)
        psum_array: Partial sum/bias initial values (numpy array)

    Test Flow:
        1. Start clock generation
        2. Reset all signals to known state
        3. Load input activations and weights in parallel
        4. Trigger computation with compute_i pulse
        5. Load bias/partial sum values
        6. Validate computed outputs
        7. Wait for completion
    """
    # Start the clock (10 time units per cycle)
    cocotb.start_soon(Clock(dut.clk_i, 10, unit=clk_cycle_unit).start())
    dut._log.info("Clock is %s " + clk_cycle_unit, clk_cycle)

    # Reset all input signals to initial state
    await cocotb.start_soon(reset_all_signals(ptp, dut))
    # Reset all input signals to initial state
    await cocotb.start_soon(send_data_params(ptp, dut))

    # Load input activations and weights in parallel (independent operations)
    send_iact_thread = cocotb.start_soon(send_iact(ptp, dut, iacts_array))
    send_wght_thread = cocotb.start_soon(send_wght(ptp, dut, wghts_array))

    # Wait for both data loading operations to complete
    await Combine(send_iact_thread, send_wght_thread)

    # Trigger computation: pulse compute_i high for one cycle
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.compute_i, 1))
    await Timer(clk_cycle, unit=clk_cycle_unit)
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.compute_i, 0))

    # Signal that we're ready to accept partial sums
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.psum_ready_i, 1))

    # Wait for PE to signal it's ready for partial sum data
    await RisingEdge(dut.psum_ready_o)
    await Timer(clk_cycle, unit=clk_cycle_unit)

    # Send bias/initial partial sum values
    cocotb.start_soon(send_bias(ptp, dut, psum_array))

    # Wait for PE to start outputting partial sums
    await RisingEdge(dut.psum_enable_o)
    await Timer(clk_cycle, unit=clk_cycle_unit)

    # Start output validation (compares against golden model)
    cocotb.start_soon(get_psum(dut, iacts_array, wghts_array, psum_array))

    # Wait for output to complete
    await FallingEdge(dut.psum_enable_o)

    # Additional settling time
    for _ in range(100):
        await Timer(clk_cycle, unit=clk_cycle_unit)

    # Final sanity check
    assert dut.compute_i.value == 0, "rst_ni is not 0!"

async def send_iact(ptp, dut, data_array):
    """
    Formats and sends input activation data to the IACT scratchpad memory.

    Converts the input activation array into compressed SPAD format and transmits
    it to the PE's input activation memory via the iact_data_i port.

    Args:
        ptp: Port timing parameters for signal timing
        dut: Device Under Test
        data_array: Input activation data (numpy array)

    SPAD Format:
        - Address array: Cumulative count of non-zero elements per row
        - Data array: Non-zero values with overhead encoding for skipped zeros
    """
    # Generate SPAD format: (address_array, data_array)
    # SISD mode (True) = one value per word, ignore_zeros (True) = compress zeros
    spad_data = generate_spad(
        data_array,
        int(dut.IACT_ADDR_ADDR.value),  # Convert LogicArray to int
        int(dut.IACT_DATA_ADDR.value),  # Convert LogicArray to int
        int(dut.DATA_IACT_BITWIDTH.value),  # Convert LogicArray to int
        True,  # SISD mode
        0,
        True,  # Ignore zeros
    )

    # Adjust data array addresses: add offset from previous element's upper bits
    for x in range(len(spad_data)):
        if((spad_data[x] != 0)):
            spad_data[x] = spad_data[x] + int(math.floor(spad_data[x-1] / 256) * 256)

    dut._log.info("IACT DATA is %s", spad_data)

    # Enable input activation interface
    # Note: iact_enable_i is a packed array, set bit 0 by setting the whole signal to 1
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.iact_enable_i, 1))


    # Send data array second
    await send_to_spad(
        ptp,
        spad_data,
        dut.iact_data_i,
        iactsize_y*iactsize_x,  # Total elements
        int(dut.TRANS_BITWIDTH_IACT.value),  # Convert LogicArray to int
        int(dut.IACT_DATA_DATA.value),  # Convert LogicArray to int
        False,  # Sequential mode
    )

    # Disable input activation interface
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.iact_enable_i, 0))
    await Timer(clk_cycle, unit=clk_cycle_unit)

async def get_psum(dut, iacts_array, wghts_array, psum_array):
    """
    Validates PE output partial sums against a software golden model.

    Computes the expected multiply-accumulate results in software and compares
    them against the hardware outputs from the PE's two parallel adders.

    Args:
        dut: Device Under Test
        iacts_array: Input activations used in the test
        wghts_array: Weights used in the test
        psum_array: Initial bias/partial sum values

    Algorithm:
        Golden model computes: result[filter] = bias[filter] + sum(iact[i] * weight[filter][i])
        Then validates each output from psum_out against expected values.

    Raises:
        AssertionError: If any computed partial sum doesn't match the golden model
    """
    # Create golden model array (up to 64 output values)
    control = np.zeros(32, dtype=int)

    iact = iacts_array
    wght = wghts_array
    bias = psum_array

    # Ensure arrays are at least 2D for consistent indexing
    if iact.ndim == 1:
        iact = [iact]

    if wght.ndim == 1:
        wght = [wght]

    # Initialize golden model with bias values
    for psum_x in range(len(bias)):
        control[psum_x] = bias[psum_x]
        
    # Compute expected MAC (Multiply-ACcumulate) results
    # For each input activation value, multiply with corresponding weights and accumulate
    current_iact = 0

    for iact_y in range(len(iact)):  # For each channel
        for iact_x in range(len(iact[iact_y])):  # For each activation in channel
            for wght_x in range(len(wght[current_iact])):  # For each filter/output
                # Accumulate: output[filter] += activation * weight[filter]
                control[wght_x] = (
                    control[wght_x] + wght[current_iact][wght_x] * iact[iact_y][iact_x]
                )
            current_iact = current_iact + 1

    # Validate hardware outputs against golden model
    current_control = 0
    for output_word in range(len(control)):
        if (output_word < math.ceil(wghtsize_x/2)):
            control[output_word] = control[2*output_word] + (control[1+(2*output_word)] << 20)
        else:
            control[output_word] = 0

    # Check outputs while PE is producing results (psum_enable_o is high)
    while dut.psum_enable_o.value == 1:
        # Validate adder_1 output
        assert dut.psum_data_o.value.to_unsigned() == control[current_control], (
            "PSUM("
            + str(dut.psum_data_o.value.to_unsigned())
            + ") is not equal to control("
            + str(control[current_control])
            + "), "
            + str(current_control + 1)
            + ". PSUM Value"
        )
        current_control = current_control + 1
        await Timer(clk_cycle, unit=clk_cycle_unit)

async def send_wght(ptp, dut, data_array):
    """
    Formats and sends weight data to the WGHT scratchpad memory.

    Converts the weight array into compressed SPAD format and transmits it
    to the PE's weight memory via the wght_data_i port.

    Args:
        ptp: Port timing parameters for signal timing
        dut: Device Under Test
        data_array: Weight data (numpy array)

    SPAD Format:
        - Packed mode (2 values per word) with zero-skipping
        - Address array is shifted to align with hardware expectations
    """
    
    # Generate SPAD format with packed mode (False = 2 values per word)
    spad_data = generate_spad(
        data_array,
        int(dut.WGHT_ADDR_ADDR.value),  # Convert LogicArray to int
        int(dut.WGHT_DATA_ADDR.value),  # Convert LogicArray to int
        int(dut.DATA_WGHT_BITWIDTH.value),  # Convert LogicArray to int
        False,  # Packed mode (not SISD)
        int(dut.DATA_WGHT_BITWIDTH.value) + int(dut.DATA_WGHT_IGNORE_ZEROS.value),  # Convert to int
        False  # Ignore zeros
    )

    dut._log.info("WGHT DATA is %s", spad_data)

    # Enable weight interface
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.wght_enable_i, 1))

    # Send data array second
    await send_to_spad(
        ptp,
        spad_data,
        dut.wght_data_i,
        int(dut.WGHT_DATA_ADDR.value),  # Convert LogicArray to int
        int(dut.TRANS_BITWIDTH_WGHT.value),  # Convert LogicArray to int
        int(dut.WGHT_DATA_DATA.value),  # Convert LogicArray to int
        False,  # Sequential mode
    )

    # Disable weight interface
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.wght_enable_i, 0))
    await Timer(clk_cycle, unit=clk_cycle_unit)

async def send_bias(ptp, dut, data_array):
    """
    Sends bias/initial partial sum values to the PE.

    Transmits initial partial sum values that will be added to the MAC results.
    Unlike iact and wght, this does not use zero-skipping compression.

    Args:
        ptp: Port timing parameters for signal timing
        dut: Device Under Test
        data_array: Bias/initial partial sum values (numpy array)

    Note:
        - No address array needed (simpler than iact/wght)
        - No zero compression (all values sent)
        - Packed mode with 2 values per word
    """
    # Generate SPAD data without zero-skipping
    spad_data = generate_spad(
        data_array,
        int(dut.PSUM_ADDR.value),           # Convert LogicArray to int
        int(dut.PSUM_ADDR.value),           # Convert LogicArray to int
        int(dut.DATA_PSUM_BITWIDTH.value),  # Convert LogicArray to int
        False,                              # Packed mode
        int(dut.DATA_PSUM_BITWIDTH.value),  # Convert LogicArray to int
        False                               # Don't ignore zeros
    )
    dut._log.info("PSUM is %s", spad_data)

    # Enable partial sum interface
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.psum_enable_i, 1))

    # Send data directly (only data array, no address array)
    await send_to_spad(
        ptp,
        spad_data,
        dut.psum_data_i,
        int(dut.PSUM_ADDR.value),  # Convert LogicArray to int
        int(dut.TRANS_BITWIDTH_PSUM.value),  # Convert LogicArray to int
        int(dut.DATA_PSUM_BITWIDTH.value) * 2,  # Convert LogicArray to int
        False,  # Sequential mode
    )

    # Disable partial sum interface
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.psum_enable_i, 0))
    await Timer(clk_cycle, unit=clk_cycle_unit)

    

async def send_data_params(ptp, dut):
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
    await Timer(clk_cycle, unit=clk_cycle_unit)
    data_reg_i =  (filters_reg_i << 4) + (channel_reg_i << 0)
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.data_stream_i), data_reg_i))
    await Timer(clk_cycle, unit=clk_cycle_unit)
    data_reg_i =  (iact_addr_max_i << 0)
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.data_stream_i), data_reg_i))
    await Timer(clk_cycle, unit=clk_cycle_unit)

    # Disable the params reading
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.enable_stream_i), 0))
    await Timer(clk_cycle, unit=clk_cycle_unit)

async def reset_all_signals(ptp, dut):
    """
    Initializes all DUT input signals to known reset state.

    Sets all input signals to 0, asserts reset (rst_ni = 0) for one cycle,
    then releases reset. This ensures the PE starts in a clean, predictable state.

    Args:
        ptp: Port timing parameters for signal timing
        dut: Device Under Test

    Reset Sequence:
        1. Set all inputs to 0 (including rst_ni active low reset)
        2. Wait 1 clock cycle
        3. Release reset (rst_ni = 1)
        4. Wait 1 clock cycle for reset to propagate
    """
    # Assert reset and zero all inputs
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.clk_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.rst_ni), 0))  # Active low reset
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.iact_select_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.iact_data_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.iact_enable_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.wght_data_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.wght_enable_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.psum_data_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.psum_enable_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.psum_ready_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.compute_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.enable_stream_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.data_stream_i), 0))

    # Hold reset for one cycle
    await Timer(clk_cycle, unit=clk_cycle_unit)

    # Release reset
    cocotb.start_soon(rtl_test_utils.set_input(ptp,dut.rst_ni, 1))

    # Wait for reset to propagate
    await Timer(clk_cycle, unit=clk_cycle_unit)

async def send_to_spad(ptp, spad, data_signal, addr_bits, trans_bits, data_bits, parallel):
    """
    Generic function to transmit SPAD data over a limited-width bus.

    Packs multiple data words into each transmission cycle based on bus width,
    then sends them serially over multiple clock cycles.

    Args:
        ptp: Port timing parameters for signal timing
        spad: Array of data words to transmit
        data_signal: DUT signal to write data to
        addr_bits: Number of transmission cycles (address space)
        trans_bits: Transmission bus width in bits
        data_bits: Width of each data word in bits
        parallel: Packing mode
            - True: Parallel mode - one word per cycle (position = cycle)
            - False: Sequential mode - pack multiple words per transmission

    Operation:
        - Calculates words_per_transmit = trans_bits / data_bits
        - Packs multiple words into single transmission by bit-shifting
        - Sends one transmission per clock cycle
        - Handles index out of bounds gracefully

    Example:
        If trans_bits=64, data_bits=16, then 4 words are packed per transmission
    """
    words_per_transmit = 0
    sending_data = 0
    current_storage_position = 0
    offset = 0

    # Send data over multiple clock cycles
    for cycle in range(addr_bits):
        # Calculate how many words fit in one transmission
        words_per_transmit = int(math.floor(trans_bits / data_bits))

        # Pack multiple words into this transmission
        for writing_cycle in range(words_per_transmit):
            offset = data_bits * writing_cycle  # Bit position for this word

            if parallel:
                # Parallel mode: one word per cycle
                current_storage_position = cycle
            else:
                # Sequential mode: pack words sequentially
                current_storage_position = int(
                    writing_cycle
                    + math.floor(cycle / words_per_transmit) * words_per_transmit
                )

            try:
                # Add this word to the transmission, shifted to correct position
                sending_data = sending_data + (
                    int(spad[current_storage_position]) << offset
                )
            except:
                # Handle out of bounds (sparse data)
                sending_data = sending_data

        # Send the packed transmission
        cocotb.start_soon(rtl_test_utils.set_input(ptp, (data_signal), sending_data))
        sending_data = 0
        await Timer(clk_cycle, unit=clk_cycle_unit)

    # Clear the signal after transmission complete
    cocotb.start_soon(rtl_test_utils.set_input(ptp, data_signal, 0))

def generate_spad(
    array, addr_spad_words, data_spad_words, bitwidth, sisd, offset, ignore_zeros
):
    """
    Converts 2D data arrays into compressed scratchpad (SPAD) format.

    Creates two arrays: an address array tracking data locations, and a data array
    with optional zero-compression. Supports two packing modes (SISD and packed).

    Args:
        array: Input data as numpy array (1D or 2D)
        addr_spad_words: Size of address array
        data_spad_words: Size of data array
        bitwidth: Bit width of each data element
        sisd: Packing mode
            - True: SISD (Single Instruction Single Data) - one value per word
            - False: Packed mode - two values per word
        offset: Bit offset for packed mode (where to place second value)
        ignore_zeros: Enable zero-compression
            - True: Skip zeros, encode skip count in overhead bits
            - False: Include all values

    Returns:
        Tuple of (addr_spad_data, data_spad_data)
            - addr_spad_data: Cumulative count of non-zero elements per row
            - data_spad_data: Non-zero values with overhead encoding

    Compression Format:
        - Non-zero values stored with overhead count in upper bits
        - overhead = number of consecutive zeros skipped before this value
        - Encoded as: (overhead << bitwidth) | value
        - In packed mode: two values concatenated with bit offset

    Example:
        Input: [[1, 0, 0, 2], [3, 4, 0, 5]] with ignore_zeros=True
        - Data encodes: 1 (0 skipped), 2 (2 skipped), 3 (0 skipped), etc.
        - Addresses track: [2, 5] (cumulative non-zero counts)
    """
    data = array
    # Ensure data is 2D for consistent processing
    if data.ndim == 1:
        data = [data]

    # Initialize SPAD arrays
    data_spad_data = np.zeros(data_spad_words)
    current_count = 0  # Count of non-zero elements processed
    overhead = 0  # Count of consecutive zeros skipped

    # Process each element in the input array
    for y in range(len(data)):  # For each row
        for x in range(len(data[y])):  # For each element in row
            # Include this element if it's non-zero OR we're not ignoring zeros
            if (data[y][x] != 0) | (not ignore_zeros):
                if sisd:
                    # SISD mode: one value per word
                    # Encode: overhead in upper bits, value in lower bits
                    data_spad_data[current_count] = data[y][x] + (overhead << bitwidth)
                else:
                    # Packed mode: two values per word
                    # Pack values at different bit offsets
                    data_spad_data[int(math.floor(current_count / 2))] = data_spad_data[
                        int(math.floor(current_count / 2))
                    ] + (
                        (data[y][x] + (overhead << bitwidth))
                        << (offset * (current_count % 2))
                    )

                current_count = current_count + 1
                overhead = 0  # Reset zero counter after storing a value
            else:
                # This element is zero - increment skip counter
                overhead = overhead + 1

        # Store cumulative count for this row in address array
        if not sisd:
            # If odd number of values, advance to next word
            if current_count % 2 == 1:
                current_count = current_count + 1

        overhead = 0  # Reset overhead counter for next row

    spad_data = data_spad_data
    return spad_data

def create_iact_wght_psum_arrays(dut):
    """
    Generates test input data with configurable dimensions and sparsity.

    Creates sequential test arrays for input activations, weights, and partial sums,
    then applies random sparsity by zeroing out a specified percentage of elements.

    Args:
        dut: Device Under Test (not used, kept for compatibility)

    Returns:
        Tuple of (iacts, wghts, psums):
            - iacts: Input activations array, shape (iactsize_y, iactsize_x)
            - wghts: Weights array, shape (wghtsize_y, wghtsize_x)
            - psums: Partial sums/bias array, shape (wghtsize_x,)

    Data Generation:
        - All arrays use sequential values (1, 2, 3, ...) for predictability
        - Random elements are zeroed based on sparsity parameters
        - This allows testing zero-skipping compression logic

    Example:
        Returns:
            - iacts: [[1, 2, 3]] (1 channel, 3 values)
            - wghts: [[1], [2], [3]] (3 weights, 1 filter)
            - psums: [1] (1 bias value)
    """
    # Generate input activations: sequential values from 1 to (channels * dimensions)
    iacts = np.arange(1,iactsize_y * iactsize_x + 1, 1).reshape(
        iactsize_y, iactsize_x
    )

    # Apply random sparsity to activations
    # Choose random indices to zero out (sparse_iact fraction of total)
    indices = np.random.choice(
        np.arange(iacts.size), replace=False, size=int(iacts.size * sparse_iact)
    )

    # Zero out selected elements (convert flat index to 2D coordinates)
    # indices = [] #Manual override option
    for x in range(len(indices)):
        iacts[int(indices[x] / iactsize_x)][int(indices[x] % iactsize_x)] = 0

    # Generate weights: sequential values from 1 to (filters * weights_per_filter)
    wghts = np.arange(1, wghtsize_y * wghtsize_x + 1, 1).reshape(
        wghtsize_y, wghtsize_x
    )

    # Apply random sparsity to weights
    indices = np.random.choice(
        np.arange(wghts.size), replace=False, size=int(wghts.size * sparse_wght)
    )

    # Zero out selected weight elements
    for x in range(len(indices)):
        wghts[int(indices[x] / wghtsize_x)][int(indices[x] % wghtsize_x)] = 0

    # wghts[indices] = 0 #Alternative: direct indexing (may not work with 2D reshape)

    # Generate partial sums/bias: sequential values from 1 to (number of filters)
    psums = np.arange(1, wghtsize_x + 1, 1)

    return iacts, wghts, psums
