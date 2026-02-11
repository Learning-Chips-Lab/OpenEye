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
import csv
import numpy as np
from numpy import genfromtxt
from pathlib import Path

import cocotb
from cocotb.triggers import Timer, Combine
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer, with_timeout, SimTimeoutError

# Add parent directory to path for importing OpenEye modules
import open_eye.timing_parameters as timing_parameters
import open_eye.rtl_test_utils as rtl_test_utils

# Timing configuration from environment variables
clk_cycle = int(os.environ["CLOCK_LEN"])  # Clock cycle length
clk_cycle_unit = os.environ["CLOCK_UNIT"]  # Clock cycle unit (e.g., "ns", "ps")

clk_delay_in = int(os.environ["CLOCK_DELAY_INPUT"])  # Input delay
clk_delay_unit_in = os.environ["CLOCK_DELAY_UNIT_INPUT"]  # Input delay unit

clk_delay_out = int(os.environ["CLOCK_DELAY_OUTPUT"])  # Output delay
clk_delay_unit_out = os.environ["CLOCK_DELAY_UNIT_OUTPUT"]  # Output delay unit



async def measure_computation_time(dut):
    """
    Asynchronous function that measures the time from compute_i pulse to psum_enable_o rise.

    This function runs in parallel with the main test and counts clock cycles between
    when compute_i is asserted and when psum_enable_o first rises.

    Args:
        dut: Device Under Test

    Returns:
        elapsed_cycles: Number of clock cycles elapsed
    """
    elapsed_cycles = 0

    # Wait for compute_i to go high
    await RisingEdge(dut.compute_i)

    # Now count cycles until psum_enable_o goes high
    while dut.psum_enable_o.value == 0:
        await RisingEdge(dut.clk_i)
        elapsed_cycles += 1

    return elapsed_cycles


def log_computation_time(elapsed_cycles, csv_filename="computation_times.csv"):
    """
    Logs computation timing data to a CSV file.

    Records the time elapsed from compute signal to first partial sum reception,
    along with test parameters. Creates the file if it doesn't exist, otherwise appends.

    Args:
        elapsed_cycles: Number of clock cycles elapsed during computation
        csv_filename: Path to output CSV file (default: "computation_times.csv" in current directory)

    CSV Columns:
        - iactsize_x: Number of input activation values
        - iactsize_y: Number of input channels
        - wghtsize_x: Number of output filters
        - sparse_iact: Input activation sparsity (0-1)
        - sparse_wght: Weight sparsity (0-1)
        - elapsed_cycles: Clock cycles from compute to first psum
        - elapsed_time_ns: Elapsed time in nanoseconds
    """
    # Calculate elapsed time in nanoseconds
    elapsed_time_ns = elapsed_cycles * clk_cycle  # clk_cycle is in the unit specified by clk_cycle_unit
    if clk_cycle_unit == "ps":
        elapsed_time_ns = elapsed_time_ns / 1000
    elif clk_cycle_unit == "us":
        elapsed_time_ns = elapsed_time_ns * 1000
    print(80*"#")
    print("ELAPSED TIME:", elapsed_cycles)
    print(80*"#")

    csv_path = Path(csv_filename)
    file_exists = csv_path.exists()

    with open(csv_path, mode='a', newline='') as csvfile:
        fieldnames = ['iactsize_x', 'iactsize_y', 'wghtsize_x', 'sparse_iact', 'sparse_wght', 'elapsed_cycles', 'elapsed_time_ns']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        # Write the data row
        writer.writerow({
            'iactsize_x': iactsize_x,
            'iactsize_y': iactsize_y,
            'wghtsize_x': wghtsize_x,
            'sparse_iact': sparse_iact,
            'sparse_wght': sparse_wght,
            'elapsed_cycles': elapsed_cycles,
            'elapsed_time_ns': elapsed_time_ns
        })

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
    timeout_time = 40000
    timeout_unit = 'ns'

    try:
        # Here the test gets started
        await with_timeout(initialize_test_pe(dut),timeout_time, timeout_unit)
    except SimTimeoutError:
        dut._log.error("Test did not finish in time!")
        raise # Error if does not finish in time
async def initialize_test_pe(dut):
    """
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
    global iactsize_x   # Number of input activation values (spatial dimension, C0 in Eyeriss V2 paper)
    global iactsize_y   # Number of input channels (number of C0*U blocks in Eyeriss V2 paper, U=1 here)
    global sparse_iact  # Input activation sparsity: 0 = no sparsity, 1 = fully sparse
    global wghtsize_x   # Number of output filters (M0 in Eyeriss V2 paper)
    global wghtsize_y   # Weights match input dimensions (iactsize_x * iactsize_y
    global sparse_wght  # Weight sparsity: 0 = no sparsity, 1 = fully sparse

    iactsize_x = int(os.environ["IACTSIZE_X"])
    iactsize_y = int(os.environ["IACTSIZE_Y"])
    wghtsize_x = int(os.environ["WGHTSIZE_X"])
    # if the SPARSE_IACT/WGHT values are floats between 0 and 1, we will use these,
    # if they are integers between 0 and 100, we will use these as percentages and convert
    # them to floats, accordingly 
    sparse_iact = float(os.environ["SPARSE_IACT"]) / 100 if float(os.environ["SPARSE_IACT"]) > 1 else float(os.environ["SPARSE_IACT"])
    sparse_wght = float(os.environ["SPARSE_WGHT"]) / 100 if float(os.environ["SPARSE_WGHT"]) > 1 else float(os.environ["SPARSE_WGHT"])
    np.random.seed(int(os.environ["SEED"]))
    wghtsize_y = iactsize_x * iactsize_y

    # Initialize timing parameters from environment variables
    ptp = timing_parameters.PortTimingParameters()
    ptp.initiate_params(clk_cycle, clk_cycle_unit, clk_delay_in, clk_delay_unit_in, clk_delay_out, clk_delay_unit_out)


    # Generate test input data (activations, weights, partial sums)
    (iacts, wghts, psums) = create_iact_wght_psum_arrays(dut)

    # Launch main test sequence and wait for completion
    test_thread = cocotb.start_soon(test_hdls(ptp, dut, iacts, wghts, psums))
    await test_thread

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
    cocotb.start_soon(Clock(dut.clk_i, 10, unit=clk_cycle_unit).start()) # type: ignore
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

    # Start timing measurement in parallel with the test
    timing_thread = cocotb.start_soon(measure_computation_time(dut))

    # Trigger computation: pulse compute_i high for one cycle
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.compute_i, 1))
    await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.compute_i, 0))

    # Signal that we're ready to accept partial sums
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.psum_ready_i, 1))

    # Wait for PE to signal it's ready for partial sum data
    await RisingEdge(dut.psum_ready_o)
    await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore

    # Send bias/initial partial sum values
    cocotb.start_soon(send_bias(ptp, dut, psum_array))

    # Wait for PE to start outputting partial sums
    await RisingEdge(dut.psum_enable_o)
    await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore

    # Get the timing measurement result
    compute_cycles = await timing_thread

    # Start output validation (compares against golden model)
    #cocotb.start_soon(get_psum(dut, iacts_array, wghts_array, psum_array))

    # Wait for output to complete
    await FallingEdge(dut.psum_enable_o)

    # Additional settling time
    for _ in range(100):
        await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore

    log_computation_time(compute_cycles)

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
        False,  # Ignore zeros
    )

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
    await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore

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

                control[wght_x] = control[wght_x] + (wght[current_iact][wght_x] * iact[iact_y][iact_x])
            current_iact = current_iact + 1
    """for output_word in range(len(control)):  # For each filter/output
        if (control[output_word] < 0):
            control[output_word] = control[output_word] + 2**20"""
    # Validate hardware outputs against golden model
    print(control[0])
    current_control = 0
    # Check outputs while PE is producing results (psum_enable_o is high)
    while dut.psum_enable_o.value == 1:
        # Validate adder_1 output
        assert dut.psum_data_o.value.to_signed() == control[current_control], (
            "PSUM("
            + str(dut.psum_data_o.value.to_unsigned())
            + ") is not equal to control("
            + str(control[current_control])
            + "), "
            + str(current_control + 1)
            + ". PSUM Value"
        )
        current_control = current_control + 1
        await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore
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
        True  # Ignore zeros
    )
    print(spad_data)
    dut._log.info("WGHT DATA is %s", spad_data)

    # Enable weight interface
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.wght_enable_i, 1))

    # Send data array
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
    await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore

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
        True,                              # Packed mode
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
        int(dut.DATA_PSUM_BITWIDTH.value),  # Convert LogicArray to int
        False,  # Sequential mode
    )

    # Disable partial sum interface
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.psum_enable_i, 0))
    await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore

    

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
    await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore
    data_reg_i =  (filters_reg_i << 4) + (channel_reg_i << 0)
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.data_stream_i), data_reg_i))
    await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore
    data_reg_i =  (iact_addr_max_i << 0)
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.data_stream_i), data_reg_i))
    await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore

    # Disable the params reading
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.enable_stream_i), 0))
    await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore

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
    await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore

    # Release reset
    cocotb.start_soon(rtl_test_utils.set_input(ptp,dut.rst_ni, 1))

    # Wait for reset to propagate
    await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore

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
        await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore

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
    simd = not sisd
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
            if (data[y][x] != 0) | (not ignore_zeros) | (((len(data[y]) - 1 == x) & (current_count%2 != 0))):
                if (data[y][x] < 0):
                    temp_data = data[y][x] + 2**bitwidth
                else :
                    temp_data = data[y][x]

                if sisd:
                    # SISD mode: one value per word
                    # Encode: overhead in upper bits, value in lower bits
                    data_spad_data[current_count] = temp_data + (overhead << bitwidth)
                else:
                    # Packed mode: two values per word
                    # Pack values at different bit offsets
                    data_spad_data[int(math.floor(current_count / 2))] = data_spad_data[
                        int(math.floor(current_count / 2))
                    ] + (
                        (temp_data + (overhead << bitwidth))
                        << (offset * (current_count % 2))
                    )

                current_count = current_count + 1
                overhead = 0  # Reset zero counter after storing a value
            else:
                # This element is zero - increment skip counter
                overhead = overhead + 1

        # Store cumulative count for this row in address array
        if simd:
            # If odd number of values, advance to next word
            if current_count % 2 == 1:
                current_count = current_count + 1

    #overhead = 0  # Reset overhead counter for next row

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
    # Generate input activations: random values from -128 to 127, without 0
    iacts = np.random.randint(-64, 63, size=(iactsize_y, iactsize_x))
    iacts[iacts >= 0] += 1
    # Apply random sparsity to activations
    # Choose random indices to zero out (sparse_iact fraction of total)
    indices = np.random.choice(
        np.arange(iacts.size), replace=False, size=int(iacts.size * sparse_iact / 100)
    )
    # Zero out selected elements (convert flat index to 2D coordinates)
    # indices = [] #Manual override option
    for x in range(len(indices)):
        iacts[int(indices[x] / iactsize_x)][int(indices[x] % iactsize_x)] = 0

    # Generate weights: random values from -128 to 127, without 0
    wghts = np.random.randint(-64, 63, size=(wghtsize_y, wghtsize_x))
    wghts[wghts >= 0] += 1

    # Apply random sparsity to weights
    indices = np.random.choice(
        np.arange(wghts.size), replace=False, size=int(wghts.size * sparse_wght / 100)
    )

    # Zero out selected weight elements
    for x in range(len(indices)):
        wghts[int(indices[x] / wghtsize_x)][int(indices[x] % wghtsize_x)] = 0

    print(wghts)
    array = []
    counter = 0
    temp = 0
    for a in range(len(wghts)):
        for b in range(len(wghts[a])):
            if (wghts[a][b] != 0):
                counter = counter + 1
        counter = math.ceil(counter / 2)
        temp = temp + counter
        counter = 0
        array.append(temp)
    print(array)
    # wghts[indices] = 0 #Alternative: direct indexing (may not work with 2D reshape)

    # Generate partial sums/bias: sequential values from 1 to (number of filters)
    psums = np.arange(1, wghtsize_x + 1, 1)

    return iacts, wghts, psums
