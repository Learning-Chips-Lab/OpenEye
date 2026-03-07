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
        - B: Number of input activation values
        - C0: Number of input channels
        - M0: Number of output filters
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
        fieldnames = ['B', 'C0', 'M0', 'sparse_iact', 'sparse_wght', 'elapsed_cycles', 'elapsed_time_ns']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        # Write the data row
        writer.writerow({
            'B': B,
            'C0': C0,
            'M0': M0,
            'sparse_iact': sparse_iact,
            'sparse_wght': sparse_wght,
            'elapsed_cycles': elapsed_cycles,
            'elapsed_time_ns': elapsed_time_ns
        })

# Dimensions for PE initiliazed as globals
B = 0  # Number of input activation values (spatial dimension)
C0 = 0  # Number of input channels
sparse_iact = 0 # Input activation sparsity: 0 = no sparsity, 1 = fully sparse
M0 = 0  # Number of output filters
C0S = 0  # Weights match input dimensions
sparse_wght = 0 # Weight sparsity: 0 = no sparsity, 1 = fully sparse
sparsity_en = 1 # Sparsity enable: 1 = sparse mode (default), 0 = dense mode

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
    # Configure test dimensions (Eyeriss v2 terminology from paper 1807.07928v2)
    global B   # Input blocks - spatial dimension of sliding window
    global C0   # Input channels per PE
    global sparse_iact  # Input activation sparsity: 0 = no sparsity, 1 = fully sparse
    global M0   # Output channels per PE (M0 in Eyeriss v2)
    global C0S   # Total weights per output channel = S * C0
    global sparse_wght  # Weight sparsity: 0 = no sparsity, 1 = fully sparse
    global sparsity_en  # Sparsity enable: 1 = sparse mode, 0 = dense mode

    B = int(os.environ["B"])
    C0 = int(os.environ["C0"])
    M0 = int(os.environ["M0"])
    # if the SPARSE_IACT/WGHT values are floats between 0 and 1, we will use these,
    # if they are integers between 0 and 100, we will use these as percentages and convert
    # them to floats, accordingly
    sparse_iact = float(os.environ["SPARSE_IACT"]) / 100 if float(os.environ["SPARSE_IACT"]) > 1 else float(os.environ["SPARSE_IACT"])
    sparse_wght = float(os.environ["SPARSE_WGHT"]) / 100 if float(os.environ["SPARSE_WGHT"]) > 1 else float(os.environ["SPARSE_WGHT"])
    sparsity_en = int(os.environ.get("SPARSITY_EN", "1"))  # Default to 1 (sparse mode)
    np.random.seed(int(os.environ["SEED"]))
    C0S = B * C0

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
    cocotb.start_soon(get_psum(dut, iacts_array, wghts_array, psum_array))

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
        sparsity_en=sparsity_en  # Use global sparsity_en setting
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
        C0*B,  # Total elements
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
    them against the hardware outputs from the PE module.

    Architecture (based on Eyeriss v2 paper 1807.07928v2):
        The PE processes a sliding window computation where:
        - Input: C0 channels * B spatial positions => C0 * B activations
        - Weights: M0 output channels * (C0 * B) weights = (M0 * C0S) matrix
        - Output: M0 partial sums (M0 values)

        For each output channel m in [0, M0):
            psum[m] = bias[m] + sum(iact[i] * weight[i][m] for i in range(C0*S))

    Args:
        dut: Device Under Test
        iacts_array: Input activations, shape (C0, S) = (C0, B)
        wghts_array: Weights, shape (C0*S, M0) = (C0S, M0)
        psum_array: Initial bias/partial sum values, shape (M0,) = (M0,)

    Raises:
        AssertionError: If any computed partial sum doesn't match the golden model
    """
    # Create golden model array sized to match the number of output filters
    # Use max possible size to avoid overflow
    max_outputs = M0
    golden_model = np.zeros(max_outputs, dtype=int)

    iact = iacts_array
    wght = wghts_array
    bias = psum_array

    # Ensure arrays are at least 2D for consistent indexing
    if iact.ndim == 1:
        iact = np.array([iact])

    if wght.ndim == 1:
        wght = np.array([wght])

    # Initialize golden model with bias values
    for filter_idx in range(M0):
        golden_model[filter_idx] = bias[filter_idx]

    # Compute expected MAC (Multiply-ACcumulate) results
    # Eyeriss v2 Architecture (see Fig. 15 in paper 1807.07928v2):
    #   - C0 input channels * S spatial positions = C0 * B total activations
    #   - M0 output channels
    #   - Weight matrix shape: (C0*S, M0) = (C0S, M0)
    #   - Each row in wght corresponds to one position in the sliding window (one activation)
    #   - Each column in wght corresponds to one output channel
    #   - Formula: psum[m] = bias[m] + sum(iact[c,s] * wght[c*S+s][m]) for all c in [0,C0), s in [0,B)

    weight_row_idx = 0  # Index into weight matrix rows (ranges from 0 to C0*B-1)

    for channel_idx in range(len(iact)):  # For each input channel (C0)
        for spatial_idx in range(len(iact[channel_idx])):  # For each spatial position (S)
            activation_value = iact[channel_idx][spatial_idx]

            # Multiply this activation with all weights for this position across all output channels
            for output_channel_idx in range(M0):  # For each output channel (M0)
                weight_value = wght[weight_row_idx][output_channel_idx]
                # Accumulate: psum[m] += iact[c,s] * wght[c*S+s][m]
                golden_model[output_channel_idx] += activation_value * weight_value

            weight_row_idx += 1  # Move to next row of weights (next position in sliding window)

    # Log the golden model for debugging
    dut._log.info(f"Golden model computation complete:")
    dut._log.info(f"  Input: C0={C0} channels * S={B} spatial = {C0 * B} activations")
    dut._log.info(f"  Weights: {C0S} * {M0} (C0*S rows, M0 columns)")
    dut._log.info(f"  Output: M0={M0} channels")
    dut._log.info(f"  Expected partial sums: {golden_model[:M0]}")

    # Validate hardware outputs against golden model
    output_idx = 0
    num_verified = 0

    # Check outputs while PE is producing results (psum_enable_o is high)
    all_equal = True
    while dut.psum_enable_o.value == 1:
        hw_output = dut.psum_data_o.value.to_signed()
        expected_output = golden_model[output_idx]

        # Validate hardware output matches golden model
        if hw_output != expected_output:
            dut._log.error(
            f"Output mismatch at index {output_idx}: "
            f"hardware={hw_output}, expected={expected_output}")
            all_equal = False

        output_idx += 1
        num_verified += 1
        await Timer(clk_cycle, unit=clk_cycle_unit) # type: ignore

    assert all_equal, "One or more outputs did not match the golden model!"

    # Verify we got the expected number of outputs
    assert num_verified == M0, (
        f"Expected {M0} outputs but received {num_verified}"
    )

    dut._log.info(f"Successfully verified {num_verified} partial sum outputs")
        
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
    # For dense mode, offset is just bitwidth (no overhead)
    offset_val = int(dut.DATA_WGHT_BITWIDTH.value) if sparsity_en == 0 else (int(dut.DATA_WGHT_BITWIDTH.value) + int(dut.DATA_WGHT_IGNORE_ZEROS.value))
    spad_data = generate_spad(
        data_array,
        int(dut.WGHT_ADDR_ADDR.value),  # Convert LogicArray to int
        int(dut.WGHT_DATA_ADDR.value),  # Convert LogicArray to int
        int(dut.DATA_WGHT_BITWIDTH.value),  # Convert LogicArray to int
        False,  # Packed mode (not SISD)
        offset_val,  # Offset for packing
        True,  # Ignore zeros
        sparsity_en=sparsity_en  # Use global sparsity_en setting
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
    # Psum doesn't use sparsity encoding, so sparsity_en doesn't affect it
    spad_data = generate_spad(
        data_array,
        int(dut.PSUM_ADDR.value),           # Convert LogicArray to int
        int(dut.PSUM_ADDR.value),           # Convert LogicArray to int
        int(dut.DATA_PSUM_BITWIDTH.value),  # Convert LogicArray to int
        True,                              # Packed mode
        int(dut.DATA_PSUM_BITWIDTH.value),  # Convert LogicArray to int
        False,                              # Don't ignore zeros
        sparsity_en=False  # Psum never uses sparsity encoding
    )
    dut._log.info("PSUM is %s", spad_data)

    # Enable partial sum interface
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.psum_enable_i, 1))
    # Send data directly (only data array, no address array)
    await send_to_spad(
        ptp,
        spad_data,
        dut.psum_data_i,
        M0,  # Convert LogicArray to int
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
    wght_addr_max_reg = (B * C0) + 2
    filters_reg_i = M0
    channel_reg_i = C0
    iact_addr_max_i = B

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
        parallel: Packing mode - True: Parallel mode (one word per cycle), False: Sequential mode (pack multiple words per transmission)

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
    array, addr_spad_words, data_spad_words, bitwidth, sisd, offset, ignore_zeros, sparsity_en=True
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
        sisd: Packing mode - True: SISD (Single Instruction Single Data, one value per word), False: Packed mode (two values per word)
        offset: Bit offset for packed mode (where to place second value)
        ignore_zeros: Enable zero-compression - True: Skip zeros and encode skip count in overhead bits, False: Include all values

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
                    if sparsity_en:
                        # Sparse mode: Encode overhead in upper bits, value in lower bits
                        data_spad_data[current_count] = temp_data + (overhead << bitwidth)
                    else:
                        # Dense mode: Just the raw value, no overhead
                        data_spad_data[current_count] = temp_data
                else:
                    # Packed mode: two values per word
                    if sparsity_en:
                        # Sparse mode: Pack with overhead
                        data_spad_data[int(math.floor(current_count / 2))] = data_spad_data[
                            int(math.floor(current_count / 2))
                        ] + (
                            (temp_data + (overhead << bitwidth))
                            << (offset * (current_count % 2))
                        )
                    else:
                        # Dense mode: Pack without overhead
                        data_spad_data[int(math.floor(current_count / 2))] = data_spad_data[
                            int(math.floor(current_count / 2))
                        ] + (
                            temp_data << (bitwidth * (current_count % 2))
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

    Creates random test arrays for input activations, weights, and partial sums,
    following the Eyeriss v2 architecture for PE sliding window computation.

    Args:
        dut: Device Under Test (not used, kept for compatibility)

    Returns:
        Tuple of (iacts, wghts, psums):

            - iacts: Input activations, shape (C0, S) = (C0, B)
            - wghts: Weights, shape (C0*S, M0) = (C0S, M0)
            - psums: Bias/partial sums, shape (M0,) = (M0,)

    Data Generation:
        - Values are random in range [-64, -1] ∪ [1, 63] (excludes 0)
        - Random elements are zeroed based on sparsity parameters
        - This allows testing zero-skipping compression logic

    Example (C0=1, S=3, M0=1):
        Returns:
            - iacts: [[1, 2, 3]] (1 channel, 3 spatial positions)
            - wghts: [[1], [2], [3]] (3 rows for C0*S=3 activations, 1 column for M0=1 output)
            - psums: [1] (1 bias value for M0=1 output channel)
    """
    # Generate input activations: random values from -128 to 127, without 0
    iacts = np.random.randint(-64, 63, size=(C0, B))
    iacts[iacts >= 0] += 1
    # Apply random sparsity to activations
    # Choose random indices to zero out (sparse_iact fraction of total)
    indices = np.random.choice(
        np.arange(iacts.size), replace=False, size=int(iacts.size * sparse_iact / 100)
    )
    # Zero out selected elements (convert flat index to 2D coordinates)
    # indices = [] #Manual override option
    for x in range(len(indices)):
        iacts[int(indices[x] / B)][int(indices[x] % B)] = 0

    # Generate weights: random values from -128 to 127, without 0
    wghts = np.random.randint(-64, 63, size=(C0S, M0))
    wghts[wghts >= 0] += 1

    # Apply random sparsity to weights
    indices = np.random.choice(
        np.arange(wghts.size), replace=False, size=int(wghts.size * sparse_wght / 100)
    )

    # Zero out selected weight elements
    for x in range(len(indices)):
        wghts[int(indices[x] / M0)][int(indices[x] % M0)] = 0

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
    psums = np.arange(1, M0 + 1, 1)

    return iacts, wghts, psums
