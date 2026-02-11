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
from cocotb.triggers import FallingEdge, RisingEdge, Timer, with_timeout, SimTimeoutError

# Add parent directory to path for importing OpenEye modules
import open_eye.timing_parameters as timing_parameters
import open_eye.rtl_test_utils as rtl_test_utils
import pe_cluster_test_utils as pctu

# Timing configuration from environment variables
clk_cycle = int(os.environ["CLOCK_LEN"])  # Clock cycle length
clk_cycle_unit = os.environ["CLOCK_UNIT"]  # Clock cycle unit (e.g., "ns", "ps")

clk_delay_in = int(os.environ["CLOCK_DELAY_INPUT"])  # Input delay
clk_delay_unit_in = os.environ["CLOCK_DELAY_UNIT_INPUT"]  # Input delay unit

clk_delay_out = int(os.environ["CLOCK_DELAY_OUTPUT"])  # Output delay
clk_delay_unit_out = os.environ["CLOCK_DELAY_UNIT_OUTPUT"]  # Output delay unit

signals_dict = {}

async def test_hdls(ptp, dut, iacts_array, wghts_array, psum_array):
    """_summary_

    Args:
        dut (_type_): _description_
        iacts_array (_type_): _description_
        wghts_array (_type_): _description_
        psum_array (_type_): _description_
    """
    # Start the clock (10 time units per cycle)
    cocotb.start_soon(Clock(dut.clk_i, 10, unit=clk_cycle_unit).start())
    dut._log.info("Clock is %s " + clk_cycle_unit, clk_cycle)

    # Reset the DUT
    await cocotb.start_soon(pctu.reset_all_signals(ptp, dut))
    # Send needed parameters to PEs
    await cocotb.start_soon(pctu.send_data_params(ptp, dut, iactsize_x, iactsize_y, wghtsize_x))
    # start the test threads
    send_iact_thread = cocotb.start_soon(send_iact(ptp, dut, iacts_array))
    send_wght_thread = cocotb.start_soon(send_wght(ptp, dut, wghts_array))

    await Combine(send_iact_thread, send_wght_thread)
    await Timer(clk_cycle, unit=clk_cycle_unit)
    # Trigger computation: pulse compute_i high for one cycle
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.compute_i, (2**12)-1))
    await Timer(clk_cycle, unit=clk_cycle_unit)
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.compute_i, 0))
    await Timer(3*clk_cycle, unit=clk_cycle_unit)
    # configure the vertical routing of the PEs to send the psums upwards in the column of the PEs
    # (by default, the accumulate the psums inside the PE)
    cocotb.start_soon(rtl_test_utils.set_input(ptp, dut.pe_router_psum_ready_i, (2**int(dut.PE_COLUMNS.value))-1))
    
    # wait until all PEs have finished the computation
    # (i.e. the psums are ready to be read out)
    while int(dut.pe_router_psum_ready_o.value) != (2**int(dut.PE_COLUMNS.value))-1:
        await Timer(clk_cycle, unit=clk_cycle_unit)
    # now send the bias to the PEs
    cocotb.start_soon(send_bias(ptp, dut, psum_array))
    
    # wait until the PEs have send out the data
    pes_ready = 0
    while int(dut.pe_router_psum_enable_o.value) != (2**int(dut.PE_COLUMNS.value))-1:
        await Timer(clk_cycle, unit=clk_cycle_unit)
    
    # now we can read out the psums and compare them to the expected values
    cocotb.start_soon(get_psum(ptp, dut, iacts_array, wghts_array, psum_array))
    
    # check if all enable signals are 0
    while int(dut.pe_router_psum_enable_o.value) != 0:
        await Timer(clk_cycle, unit=clk_cycle_unit)
    
    # finally check if reset is still 1
    assert dut.rst_ni.value == 1, "rst_ni is not 1!"

@cocotb.test()
async def start_test_pe_cluster(dut):
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
        await with_timeout(initialize_test_pe_cluster(dut),timeout_time, timeout_unit)
    except SimTimeoutError:
        dut._log.error("Test did not finish in time!")
        raise # Error if does not finish in time
async def initialize_test_pe_cluster(dut):
    # Sparse Testcase
    
    # Configure test dimensions
    global iactsize_x   # Number of input activation values (spatial dimension, C0 in Eyeriss V2 paper)
    global iactsize_y   # Number of input channels (number of C0*U blocks in Eyeriss V2 paper, U=1 here)
    global sparse_iact  # Input activation sparsity: 0 = no sparsity, 1 = fully sparse
    global wghtsize_x   # Number of output filters (M0 in Eyeriss V2 paper)
    global wghtsize_y   # Weights match input dimensions (iactsize_x * iactsize_y
    global sparse_wght  # Weight sparsity: 0 = no sparsity, 1 = fully sparse
    global pe_iact_cycles

    iactsize_x = int(os.environ["IACTSIZE_X"])
    iactsize_y = int(os.environ["IACTSIZE_Y"])
    wghtsize_x = int(os.environ["WGHTSIZE_X"])
    sparse_iact = int(os.environ["SPARSE_IACT"])
    sparse_wght = int(os.environ["SPARSE_WGHT"])
    np.random.seed(int(os.environ["SEED"]))
    pe_iact_cycles = math.ceil((int(dut.PE_ROWS.value) + int(dut.PE_COLUMNS.value) - 1)/int(dut.NUM_GLB_IACT.value))
    wghtsize_y = iactsize_x * iactsize_y

    # Initialize timing parameters from environment variables
    ptp = timing_parameters.PortTimingParameters()
    ptp.initiate_params(clk_cycle, clk_cycle_unit, clk_delay_in, clk_delay_unit_in, clk_delay_out, clk_delay_unit_out)


    # Generate test input data (activations, weights, partial sums)
    (iacts, wghts, psums) = create_iact_wght_psum_arrays(dut)

    # Launch main test sequence
    await cocotb.start_soon(test_hdls(ptp, dut, iacts, wghts, psums))

async def send_iact(ptp, dut, data_array):
    spad_data = [[0 for x in range(int(dut.NUM_GLB_IACT.value))] for y in range(pe_iact_cycles)]
    iact_transmission = [[0 for x in range(int(dut.NUM_GLB_IACT.value))] for y in range(pe_iact_cycles)]
    for cycle in range(pe_iact_cycles):
        for glb_cluster in range(int(dut.NUM_GLB_IACT.value)):
            spad_data[cycle][glb_cluster] = generate_spad(
                data_array[glb_cluster + cycle * int(dut.NUM_GLB_IACT.value)],
                int(dut.IACT_ADDR_WORDS.value),  # Convert LogicArray to int
                int(dut.IACT_DATA_WORDS.value),  # Convert LogicArray to int
                int(dut.DATA_IACT_BITWIDTH.value),  # Convert LogicArray to int
                True,  # SISD mode
                0,
                False,  # Ignore zeros
            )
    for cycle in range(pe_iact_cycles):
        for glb_cluster in range(int(dut.NUM_GLB_IACT.value)):
            iact_transmission[cycle][glb_cluster] = pctu.send_to_iact_spad(ptp,spad_data[cycle][glb_cluster],dut,iactsize_x*iactsize_y)

    # Send Iact data
    for transmission in range(16):
        for cycle in range(pe_iact_cycles):
            temp_enable = 0
            temp_data = 0
            if (cycle == 0) :
                temp_choose = (0 << 0) + (1 << 2) + (1 << 8) + (2 << 4) + (2 << 10)+ (2 << 16)
                temp_choose = temp_choose + (3 << 6) + (3 << 12) + (3 << 14) + (3 << 18) + (3 << 20)+ (3 << 22)
            else :
                temp_choose = (0 << 6) + (0 << 12) + (0 << 18) + (1 << 14) + (1 << 20) + (2 << 22)
                temp_choose = temp_choose + (3 << 0) + (3 << 2) + (3 << 4) + (3 << 8) + (3 << 10)+ (3 << 16)
            cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.iact_choose_i), temp_choose))
            for glb_cluster in range(int(dut.NUM_GLB_IACT)):
                temp_enable = temp_enable + (iact_transmission[cycle][glb_cluster][0][transmission] << (1 * glb_cluster))
                temp_data = temp_data + (iact_transmission[cycle][glb_cluster][1][transmission] << (int(dut.TRANS_BITWIDTH_IACT.value) * glb_cluster))
            cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_iact_enable), temp_enable))
            cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_iact_data), temp_data))
            if (temp_enable != 0):
                await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_iact_enable), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_iact_data), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.iact_choose_i), 0))

        

async def get_psum(ptp, dut, iacts_array, wghts_array, psum_array):
    control = np.zeros(
        int(dut.PE_COLUMNS.value) * int(dut.PSUM_WORDS.value) * 2, dtype=int
    ).reshape(int(dut.PE_COLUMNS.value), int(dut.PSUM_WORDS.value) * 2)

    iact = iacts_array
    wght = wghts_array
    bias = psum_array

    if iact.ndim == 1:
        iact = [iact]

    if wght.ndim == 1:
        wght = [wght]

    for glb_cluster in range(int(dut.PE_COLUMNS.value)):
        for psum_x in range(len(bias[glb_cluster])):
            control[glb_cluster][psum_x] = bias[glb_cluster][psum_x]

    current_iact = 0
    iact_line = 0
    for pe_x in range(int(dut.PE_COLUMNS.value)):
        for pe_y in range(int(dut.PE_ROWS.value)):
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
    print(control)
    thread = []
    global first_error_found
    first_error_found = 0

    for pe_x in range(int(dut.PE_COLUMNS.value)):
        thread.append(cocotb.start_soon(check_psum(dut, pe_x, control[pe_x], ptp)))

    for pe_x in range(int(dut.PE_COLUMNS.value)):
        await thread[pe_x]
    print("First error is: " + str(first_error_found))
    assert first_error_found == 0, "Outcoming Partial Sums are not equal to Calculated data!"

async def check_psum(dut, pe_x, control, ptp):
    current_control = 0
    global first_error_found
    while int(int(dut.pe_router_psum_enable_o.value)/(2**pe_x))%2 == 1:
        # Make sure there are no X values for gate level simulation
        assert 'x' not in dut.pe_router_psum_data_o.value, "x values in PSUM"
        
        if ((int(int(dut.pe_router_psum_data_o.value)/(2**(20*pe_x)))%(2**20)) == control[current_control]):
            print(
            "PSUM("
            + str(int(int(dut.pe_router_psum_data_o.value)/(2**(20*pe_x)))%(2**20))
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
            + str(int(int(dut.pe_router_psum_data_o.value)/(2**(20*pe_x)))%(2**20))
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
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

async def send_wght(ptp, dut, data_array):
    global signals_dict
    spad_data = [0 for _ in range(int(dut.PE_ROWS.value))]
    for glb_cluster in range(int(dut.PE_ROWS.value)):
        spad_data[glb_cluster] = generate_spad(
            data_array[glb_cluster],
            int(dut.WGHT_ADDR_WORDS.value),  # Convert LogicArray to int
            int(dut.WGHT_DATA_WORDS.value),  # Convert LogicArray to int
            int(dut.DATA_WGHT_BITWIDTH.value),  # Convert LogicArray to int
            False,  # Packed mode (not SISD)
            int(dut.DATA_WGHT_BITWIDTH.value) + int(dut.DATA_WGHT_IGNORE_ZEROS.value),  # Convert to int
            True  # Ignore zeros
        )
    wght_transmission = []
    for glb_cluster in range(int(dut.PE_ROWS.value)):
        wght_transmission.append(
            pctu.send_to_wght_spad(ptp, spad_data[glb_cluster], dut))

    for transmission in range (int(dut.WGHT_DATA_WORDS.value)):
        temp_enable = 0
        temp_data = 0
        for glb_cluster in range(int(dut.PE_ROWS.value)):
            temp_enable = temp_enable + (wght_transmission[glb_cluster][0][transmission] << (1 * glb_cluster))
            temp_data = temp_data + (wght_transmission[glb_cluster][1][transmission] << (int(dut.TRANS_BITWIDTH_WGHT.value) * glb_cluster))
        cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_wght_enable), temp_enable))
        cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_wght_data), temp_data))
        if (temp_enable != 0):
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

    #Transmit wght data

async def send_bias(ptp, dut, data_array):
    spad_data = []
    print(data_array)
    for glb_cluster in range(int(dut.PE_COLUMNS.value)):
        spad_data.append(
            generate_spad(
                data_array[glb_cluster],
                int(dut.PSUM_WORDS.value),
                int(dut.PSUM_WORDS.value),
                int(dut.DATA_PSUM_BITWIDTH.value),
                True,
                int(dut.DATA_PSUM_BITWIDTH.value),
                False
            )
        )
    psum_transmission = []
    for glb_cluster in range(int(dut.PE_COLUMNS.value)):
        psum_transmission.append(pctu.send_to_psum_spad(ptp, spad_data[glb_cluster], dut))
    
    print(psum_transmission)

    for transmission in range (int(dut.PSUM_WORDS.value)):
        temp_enable = 0
        temp_data = 0
        for glb_cluster in range(int(dut.PE_COLUMNS.value)):
            temp_enable = temp_enable + (psum_transmission[glb_cluster][0][transmission] << (1 * glb_cluster))
            temp_data = temp_data + (psum_transmission[glb_cluster][1][transmission] << (int(dut.TRANS_BITWIDTH_PSUM.value) * glb_cluster))
        cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_router_psum_enable_i), temp_enable))
        cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_router_psum_data_i), temp_data))
        if (temp_enable != 0):
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_router_psum_enable_i), 0))
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.pe_router_psum_data_i), 0))

   
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
    # Generate input activations: random values from -128 to 127, without 0
    shape = (pe_iact_cycles * int(dut.NUM_GLB_IACT.value), iactsize_y, iactsize_x)
    iacts = np.random.randint(-64, 63, size=shape)
    iacts[iacts >= 0] += 1
    # Apply random sparsity to activations
    mask = np.random.rand(*iacts.shape) < (sparse_iact / 100.0)
    iacts[mask] = 0
    print(iacts)
    # Generate weights: random values from -128 to 127, without 0
    shape = (int(dut.PE_ROWS.value), wghtsize_y, wghtsize_x)
    wghts = np.random.randint(-64, 63, size=shape)
    wghts[wghts >= 0] += 1

    # Apply random sparsity to weights
    mask = np.random.rand(*wghts.shape) < (sparse_wght / 100.0)
    wghts[mask] = 0

    psums = np.arange(1, wghtsize_x * int(dut.PE_COLUMNS.value) + 1, 1).reshape(int(dut.PE_COLUMNS.value), wghtsize_x)

    return iacts, wghts, psums


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
    for glb in range(int(dut.PE_ROWS.value)):
        current_signal.append({"signal":dut.pe_wght_enable, "start_bit":glb, "end_bit":glb})
    signals_dict["pe_wght_enable"] = current_signal
    current_signal = []
    for glb in range(int(dut.PE_ROWS.value)):
        current_signal.append({"signal":dut.pe_wght_data, \
                               "start_bit":glb*params.WGHT_Trans_Bitwidth, \
                               "end_bit":(glb+1)*params.WGHT_Trans_Bitwidth-1})
    signals_dict["pe_wght_data"] = current_signal
    current_signal = []
    for glb in range(int(dut.PE_ROWS.value)):
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
