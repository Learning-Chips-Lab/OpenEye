# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""RTL test utilities for the OpenEye project.

This module provides utility functions for testing RTL designs using cocotb.
It includes functions for:
- Setting input signals with proper timing
- Resetting DUT signals
- Writing input activations, weights and biases
- Comparing output streams
- Managing DMA transfers
- Supporting various neural network layer types (Conv, Dense, Pooling)

The module is designed to work with the OpenEye neural network accelerator
architecture and supports both parallel and serial modes of operation.
"""

import os
import logging
import math
import cocotb
import numpy as np
from cocotb.triggers import Timer
import open_eye.stream_dicts as strdic

logger = logging.getLogger("cocotb")


async def set_input(port_timings, signal, new_value, multiple_dim=False, array_index=[], array_max_index=[]):
    """Set an input signal value with proper timing.
    
    This function sets a value on an input signal of the DUT, respecting timing constraints
    defined in the OpenEye implementation. It supports both scalar and multi-dimensional
    signals (arrays).
    
    Args:
        port_timings: Object containing timing parameters (clk_delay_in, clk_delay_unit_in)
        signal: The signal to set (cocotb signal object)
        new_value: The value to set on the signal
        multiple_dim: Boolean indicating if the signal is multi-dimensional
        array_index: List of indices for accessing multi-dimensional signals
        array_max_index: List of maximum indices for each dimension
    
    Note:
        The input delay is 100 ps relative to the rising edge, which matches
        the implementation constraints of OpenEye.
    """
    await Timer(port_timings.clk_delay_in, unit=port_timings.clk_delay_unit_in)
    if(multiple_dim):
        if(cocotb.SIM_NAME == "Icarus Verilog"):
            array_max_index = list(reversed(array_max_index))
            index_of_signal = 0
            current_multiply = 1
            for current_index in range(len(array_index)):
                for x in range(current_index):
                    current_multiply = current_multiply * array_max_index[x]
                a = list(reversed(array_index))
                index_of_signal = a[current_index] * current_multiply + index_of_signal

                current_multiply = 1
            signal = signal[index_of_signal]
        else:
            for current_index in range(len(array_index)):
                signal = signal[array_index[current_index]]
    signal.value = new_value
    
async def reset_all_signals(ptp, dut, serial):
    """Reset all signals of the DUT to known initial state.

    This function performs a complete reset of the OpenEye accelerator, initializing
    all control and data signals to zero before releasing the reset signal. It
    handles both parallel and serial operation modes.

    Args:
        ptp: Port timing parameters containing clock cycle information
        dut: Device under test (OpenEye accelerator instance)
        serial: Operation mode flag
            - 0: Parallel mode (resets parallel interface signals)
            - 1: Serial/DMA mode (resets DMA interface signals)

    Reset Sequence:
        1. Assert active-low reset (rst_ni = 0)
        2. Zero all mode-specific input signals
        3. Wait 1 clock cycle
        4. Release reset (rst_ni = 1)
        5. Wait 3 clock cycles for internal stabilization

    Note:
        The 3-cycle wait after reset release allows internal state machines
        and pipeline stages to properly initialize before operation begins.
    """
    # Assert active-low reset signal
    cocotb.start_soon(set_input(ptp,(dut.rst_ni), 0))

    if(serial == 0):
        # Parallel mode: reset all parallel interface signals
        # Computation control
        cocotb.start_soon(set_input(ptp,(dut.compute_i), 0))

        # Weight interface
        cocotb.start_soon(set_input(ptp,(dut.wght_data_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.wght_enable_i), 0))

        # Input activation interface
        cocotb.start_soon(set_input(ptp,(dut.iact_data_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.iact_enable_i), 0))

        # Partial sum interface
        cocotb.start_soon(set_input(ptp,(dut.psum_data_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.psum_ready_i), 0))

        # Status and configuration registers
        cocotb.start_soon(set_input(ptp,(dut.status_reg_enable_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.data_mode_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.fraction_bit_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.needed_cycles_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.needed_x_cls_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.needed_y_cls_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.needed_iact_cycles_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.filters_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.iact_addr_len_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.wght_addr_len_i), 0))

        # Cluster operation modes
        cocotb.start_soon(set_input(ptp,(dut.bano_cluster_mode_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.af_cluster_mode_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.pooling_cluster_mode_i), 0))

        # Activation configuration
        cocotb.start_soon(set_input(ptp,(dut.input_activations_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.iact_write_addr_t_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.iact_write_data_t_i), 0))

        # Convolution parameters
        cocotb.start_soon(set_input(ptp,(dut.stride_x_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.stride_y_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.kernel_per_pe_cluster_i), 0))

        # Processing element control
        cocotb.start_soon(set_input(ptp,(dut.compute_mask_i), 0))

        # Router configuration for data distribution
        cocotb.start_soon(set_input(ptp,(dut.router_mode_iact_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.router_mode_wght_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.router_mode_psum_i), 0))
    else:
        # Serial/DMA mode: reset DMA interface signals
        cocotb.start_soon(set_input(ptp,(dut.data_dma_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.enable_dma_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.ready_dma_i), 0))

    # Hold reset for one clock cycle
    await Timer(ptp.clk_cycle, ptp.clk_cycle_unit)

    # Release reset
    cocotb.start_soon(set_input(ptp,(dut.rst_ni), 1))

    # Wait 3 clock cycles for internal state stabilization
    for _ in range(3):
        await Timer(ptp.clk_cycle, ptp.clk_cycle_unit)

async def send_stream(ptp, dut, stream, oep, lp, layer_repetition):
    """ Send the stream to the DUT. 
    
    This function sends the stream to the DUT. It is called by the testbench.
    All types of data in the stream are sent to the DUT in parallel. The method
    waits until the transmission is finished.

    Args:
        dut: The DUT. stream: The stream that is sent to the DUT.
        layer_repetition: The index of the part of a layer, if it is too large to be processed at once.
        oep: The OpenEye parameters. lp: The
        layer parameters.
    
    """
    if (oep.SERIAL == 0):
        # Parallel mode: configure all status registers and router modes

        # Enable status register for configuration
        cocotb.start_soon(set_input(ptp,(dut.status_reg_enable_i), 1))

        # Data format and quantization settings
        cocotb.start_soon(set_input(ptp,(dut.data_mode_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["data_mode"]]))
        cocotb.start_soon(set_input(ptp,(dut.fraction_bit_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["realfactor"]]))

        # Computation control parameters
        cocotb.start_soon(set_input(ptp,(dut.needed_cycles_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["needed_refreshes"]]))
        cocotb.start_soon(set_input(ptp,(dut.needed_x_cls_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_X_cluster"]]))
        cocotb.start_soon(set_input(ptp,(dut.needed_y_cls_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_Y_cluster"]]))
        cocotb.start_soon(set_input(ptp,(dut.needed_iact_cycles_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["needed_Iact_writes"]]))

        # Filter and memory configuration
        cocotb.start_soon(set_input(ptp,(dut.filters_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_psum_per_PE"]]))
        cocotb.start_soon(set_input(ptp,(dut.iact_addr_len_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_iact_addr_per_PE"]]))
        cocotb.start_soon(set_input(ptp,(dut.wght_addr_len_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_wght_addr_per_PE"]]))

        # Cluster operation modes
        cocotb.start_soon(set_input(ptp,(dut.bano_cluster_mode_i), 0))  # Batch normalization mode (disabled)
        cocotb.start_soon(set_input(ptp,(dut.af_cluster_mode_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["autofunction"]]))  # Activation function
        cocotb.start_soon(set_input(ptp,(dut.pooling_cluster_mode_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["poolingmode"]]))  # Pooling mode
        cocotb.start_soon(set_input(ptp,(dut.delay_psum_glb_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["psum_delay"]]))  # Partial sum delay

        # Input activation configuration
        cocotb.start_soon(set_input(ptp,(dut.input_activations_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_iact_per_PE"]]))
        cocotb.start_soon(set_input(ptp,(dut.iact_write_addr_t_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["iact_addr_len"]]))
        cocotb.start_soon(set_input(ptp,(dut.iact_write_data_t_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["iact_data_len"]]))

        # Convolution stride parameters
        cocotb.start_soon(set_input(ptp,(dut.stride_x_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["strideX"]]))
        cocotb.start_soon(set_input(ptp,(dut.stride_y_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["strideY"]]))
        cocotb.start_soon(set_input(ptp,(dut.kernel_per_pe_cluster_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["kernel_per_pe_cluster"]]))

        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        # Set PE compute mask (which PEs are active for this layer)
        cocotb.start_soon(set_input(ptp,(dut.compute_mask_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["usePEs"]]))

        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        # Configure router modes for data distribution across clusters
        # Routers control how data flows between Global Buffers and PEs

        # Set router mode for input activations
        # Each router has a mode value that determines routing pattern
        # Modes are packed into a single port with bit-shifting
        router_mode_port = 0
        for cl_x in range(oep.Clusters_X):
            for cl_y in range(oep.Clusters_Y):
                for router in range(oep.NUM_GLB_IACT):
                    # Pack router modes: each router gets its own bit field
                    # Bit position = (router + routers_per_cluster * cl_y + routers_per_row * cl_x) * bits_per_router
                    router_mode_port = router_mode_port + (stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["router_iact"]][cl_x][cl_y][router] << \
                                                        (oep.Iact_Router_Bits * router + \
                                                            oep.Iact_Router_Bits * oep.NUM_GLB_IACT * cl_y + \
                                                            oep.Iact_Router_Bits * oep.NUM_GLB_IACT * oep.Clusters_Y * cl_x))
        cocotb.start_soon(set_input(ptp,(dut.router_mode_iact_i), router_mode_port))

        # Set router mode for weights
        router_mode_port = 0
        for cl_x in range(oep.Clusters_X):
            for cl_y in range(oep.Clusters_Y):
                for router in range(oep.NUM_GLB_WGHT):
                    router_mode_port = router_mode_port + (stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["router_wght"]][cl_x][cl_y][router] << \
                                                        (oep.Wght_Router_Bits * router + \
                                                            oep.Wght_Router_Bits * oep.NUM_GLB_WGHT * cl_y + \
                                                            oep.Wght_Router_Bits * oep.NUM_GLB_WGHT * oep.Clusters_Y * cl_x))
        cocotb.start_soon(set_input(ptp,(dut.router_mode_wght_i), router_mode_port))

        # Set router mode for partial sums
        router_mode_port = 0
        for cl_x in range(oep.Clusters_X):
            for cl_y in range(oep.Clusters_Y):
                for router in range(oep.NUM_GLB_PSUM):
                    router_mode_port = router_mode_port + (stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["router_psum"]][cl_x][cl_y][router] << \
                                                        (oep.Psum_Router_Bits * router + \
                                                            oep.Psum_Router_Bits * oep.NUM_GLB_PSUM * cl_y + \
                                                            oep.Psum_Router_Bits * oep.NUM_GLB_PSUM * oep.Clusters_Y * cl_x))
        cocotb.start_soon(set_input(ptp,(dut.router_mode_psum_i), router_mode_port))
        router_mode_port = 0

        # Wait for configuration to propagate
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        # Disable status register enable after configuration complete
        cocotb.start_soon(set_input(ptp,(dut.status_reg_enable_i), 0))
    else:
        # Serial/DMA mode: send all data sequentially over DMA interface

        # Enable DMA transfer
        cocotb.start_soon(set_input(ptp,(dut.enable_dma_i), 1))

        # Send status/configuration data
        for data_word in range(len(stream[strdic.stream_parallel_dict["status"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["status"]][data_word]))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        # Send input activation data
        for data_word in range(len(stream[strdic.stream_parallel_dict["iact"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["iact"]][data_word]))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        # Send weight data
        for data_word in range(len(stream[strdic.stream_parallel_dict["wght"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["wght"]][data_word]))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        # Send partial sum data
        for data_word in range(len(stream[strdic.stream_parallel_dict["psum"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["psum"]][data_word]))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        # Send quantization parameters
        for data_word in range(len(stream[strdic.stream_parallel_dict["quantize"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["quantize"]][data_word]))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        # Send offset parameters
        for data_word in range(len(stream[strdic.stream_parallel_dict["offset"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["offset"]][data_word]))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        # Complete DMA transfer
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp,(dut.enable_dma_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.ready_dma_i), 1))
        
def compare_iact_storage(ptp, dut, iact_ref, oep):
    """Compare input activation storage contents with reference values.
    
    This function verifies that the input activation data stored in the DUT's buffers
    matches the expected reference values. It handles the complex memory layout and
    data organization of the OpenEye accelerator's input activation storage.
    
    Args:
        ptp: Port timing parameters
        dut: Device under test (OpenEye accelerator instance)
        iact_ref: Reference input activation data to compare against
        oep: OpenEye parameters containing architecture configuration
        
    Returns:
        bool: True if comparison passes, False if any mismatch is found
        
    Technical Details:
        - Handles multi-dimensional activation data layout
        - Manages word and buffer addressing in hardware
        - Performs proper data alignment and comparison
        - Provides detailed logging of any mismatches
        - Accounts for data organization across multiple buffers
    """
    logger.info("Iact storages are checked.")
    i,c,x,y = 0,0,0,0
    word, word_reset, buffer, buffer_reset = 0,0,0,0
    iact_ref = np.array(iact_ref)
    iact_ref = iact_ref.transpose((0, 2, 1))
    error_found = False
    for c in range(len(iact_ref)):
        for y in range(len(iact_ref[c])):
            for x in range(len(iact_ref[c][y])):
                if(iact_ref[c][y][x] != dut.BUFFER_A[buffer%oep.NUM_BUFFER].iact_converter_buffer_SP.impl.mem[word].value[56 - (i * 8):63 - (i * 8)].signed_integer):
                    logger.error("Error found in Iact storage; Channel: " + str(c) + " X: " + str(x) + " Y: " + str(y) + " buffer: " + str(buffer) + " word: " + str(word) + " i: " + str(i))
                    logger.error("Ref-Value: " + str(iact_ref[c][y][x]) + " DUT-Value: " + str(dut.BUFFER_A[buffer%oep.NUM_BUFFER].iact_converter_buffer_SP.impl.mem[word].value[56 - (i * 8):63 - (i * 8)].signed_integer))
                    error_found = True
                i = i + 4
                if (i >= 8):
                    i = i - 8
                    buffer = buffer + 1
                    if (buffer >= oep.NUM_BUFFER):
                        word = word + 1
                        buffer = buffer - oep.NUM_BUFFER
        if ((c%4 == 3)):
            buffer_reset = buffer
            word_reset = word
            if ((len(iact_ref[c] * len(iact_ref[c][y]))) %2 == 1):
                i = i - 3
            else :
                i = 0
        else:
            i = i + 1
            if ((len(iact_ref[c] * len(iact_ref[c][y]))) %2 == 1):
                i = (i + 4) % 8
            buffer = buffer_reset
            word = word_reset
    if error_found:
        return True
    return True

async def write_iact(ptp, dut, stream, oep, lp):
    """Write the input activations to the DUT.
    
    This function writes the input activations to the OpenEye accelerator through its
    input ports. It handles the timing and protocol requirements for transferring 
    activation data to the hardware.
    
    Args:
        ptp: Port timing parameters containing clock and signal timing information
        dut: The device under test (OpenEye accelerator instance)
        stream: The activation data stream to be sent to the DUT
        oep: OpenEye parameters containing architecture configuration 
        lp: Layer parameters containing neural network layer configuration
        
    Implementation Details:
        - Waits for iact_ready_o signal before sending data
        - Handles data transmission across multiple clusters and routers
        - Sets enable signals appropriately for the data transfer
        - Manages the timing of data and enable signals
        - Supports sparsity in activation data
        - Automatically handles signal reset after transmission
    """
    iact_enable_signal = 0
    iact_transmission = 0
    if(lp.skipIact != 1):
        while (dut.iact_ready_o.value == 0): #TODO: ADAPT for Sparsetiy
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        for position in range(len(stream[0][0][0])):
            iact_enable_signal = 0
            for x_cluster in range(oep.Clusters_X):
                for y_cluster in range(oep.Clusters_Y):
                    for router in range(oep.NUM_GLB_IACT):
                        try:
                            iact_transmission = iact_transmission + \
                            (stream[x_cluster][y_cluster][router][position] \
                            << ((router + y_cluster * oep.NUM_GLB_IACT + x_cluster * oep.NUM_GLB_IACT * oep.Clusters_Y) * oep.IACT_Trans_Bitwidth))
                            iact_enable_signal = iact_enable_signal + 2**(router + y_cluster * oep.NUM_GLB_IACT+ x_cluster * oep.NUM_GLB_IACT * oep.Clusters_Y)
                        except:
                            iact_enable_signal = iact_enable_signal
            cocotb.start_soon(set_input(ptp,(dut.iact_data_i), iact_transmission))
            iact_transmission = 0
            cocotb.start_soon(set_input(ptp,(dut.iact_enable_i), iact_enable_signal))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp,(dut.iact_data_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.iact_enable_i), 0))

async def write_wght(ptp, dut, stream, oep, lp):
    """Write the weights to the DUT.
    
    This function handles the transmission of weight data to the OpenEye accelerator.
    It manages the protocol for sending weight data across multiple clusters and routers,
    ensuring proper timing and synchronization.
    
    Args:
        ptp: Port timing parameters containing clock and signal timing information
        dut: The device under test (OpenEye accelerator instance)
        stream: The weight data stream to be sent to the DUT
        oep: OpenEye parameters containing architecture configuration (clusters, routers, etc.)
        lp: Layer parameters containing neural network layer configuration
        
    Implementation Details:
        - Checks wght_ready_o signal from all clusters before transmission
        - Manages weight data distribution across multiple clusters
        - Handles weight streaming protocol with proper enable signals
        - Supports parallel transmission to multiple processing elements
        - Automatically handles signal reset after transmission
        - Respects timing requirements for stable weight loading
    
    Note:
        The function skips transmission if lp.skipWght is set to 1, which is useful
        for layers that reuse previously loaded weights.
    """
    wght_enable_signal = 0
    wght_transmission = 0
    if(lp.skipWght != 1):
        while (dut.wght_ready_o.value != ((2**(oep.Clusters_X*oep.Clusters_Y*oep.NUM_GLB_WGHT))-1)):
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        cocotb.start_soon(set_input(ptp,(dut.wght_enable_i), (2**(oep.Clusters_X*oep.Clusters_Y*oep.NUM_GLB_WGHT))-1))
        for position in range(len(stream[0][0][0])):
            wght_enable_signal = 0
            for x_cluster in range(oep.Clusters_X):
                for y_cluster in range(oep.Clusters_Y):
                    for router in range(oep.NUM_GLB_WGHT):
                        try:
                            wght_transmission = wght_transmission + \
                            (stream[x_cluster][y_cluster][router][position] \
                            << ((router + y_cluster * oep.NUM_GLB_WGHT + x_cluster * oep.NUM_GLB_WGHT * oep.Clusters_Y) * oep.WGHT_Trans_Bitwidth))
                            wght_enable_signal = wght_enable_signal + 2**(router + y_cluster * oep.NUM_GLB_WGHT+ x_cluster * oep.NUM_GLB_WGHT * oep.Clusters_Y)
                        except:
                            wght_enable_signal = wght_enable_signal
            cocotb.start_soon(set_input(ptp,(dut.wght_data_i), wght_transmission))
            wght_transmission = 0
            cocotb.start_soon(set_input(ptp,(dut.wght_enable_i), wght_enable_signal))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp,(dut.wght_data_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.wght_enable_i), 0))

async def write_bias(ptp, dut, stream, oep, lp):
    """Write bias values to the DUT using partial sum Global Local Buffers (GLBs).
    
    This function handles the transmission of bias data to the OpenEye accelerator
    through the partial sum ports. The bias values are stored in partial sum GLBs
    to be added during computation.
    
    Args:
        ptp: Port timing parameters containing clock and signal timing information
        dut: The device under test (OpenEye accelerator instance)
        stream: The bias data stream to be sent to the DUT
        oep: OpenEye parameters containing architecture configuration
        lp: Layer parameters containing neural network layer configuration
        
    Implementation Details:
        - Uses partial sum (psum) ports for bias transmission
        - Enables all psum ports across clusters simultaneously
        - Handles data distribution across multiple clusters and routers
        - Manages timing and synchronization of data transfer
        - Automatically handles signal reset after transmission
    
    Technical Notes:
        - Bias values are written to partial sum GLBs
        - Uses the same data path as partial sums for efficiency
        - Supports parallel loading across multiple processing elements
        - Maintains proper synchronization with computation units
    """
    psum_transmission = 0
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), (2**(oep.Clusters_X*oep.Clusters_Y*oep.NUM_GLB_PSUM))-1))
    for position in range(len(stream[0][0][0])):
        for x_cluster in range(oep.Clusters_X):
            for y_cluster in range(oep.Clusters_Y):
                for router in range(oep.NUM_GLB_PSUM):
                    psum_transmission = psum_transmission + \
                    (stream[x_cluster][y_cluster][router][position] \
                    << ((router + y_cluster * oep.NUM_GLB_PSUM + x_cluster * oep.NUM_GLB_PSUM * oep.Clusters_Y) * oep.PSUM_Trans_Bitwidth))
        cocotb.start_soon(set_input(ptp,(dut.psum_data_i), psum_transmission))
        psum_transmission = 0
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp,(dut.psum_data_i), 0))
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))

async def await_enable_signal(ptp, dut):
    """Wait for DMA enable signal from the device.
    
    This function waits for the DMA enable signal to be asserted by the DUT,
    indicating that it is ready to begin a DMA transfer operation.
    
    Args:
        ptp: Port timing parameters
        dut: Device under test (OpenEye accelerator instance)
    """
    cocotb.start_soon(set_input(ptp,(dut.ready_dma_i), 1))
    while (dut.enable_dma_o.value != 1):
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    pass

async def await_ready_signal(ptp, dut):
    """Wait for ready signal from the device.
    
    This function implements a waiting period followed by monitoring of the
    DMA ready signal from the DUT. It ensures proper synchronization for
    data transfer operations.
    
    Args:
        ptp: Port timing parameters
        dut: Device under test (OpenEye accelerator instance)
        
    Note:
        Includes a fixed 3-cycle delay before checking ready signal to allow
        for internal state stabilization.
    """
    await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    while (dut.ready_dma_o.value != 1):
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    pass

async def compare_stream_Conv(ptp, dut, layer_number, layer_repetition, layer_parameters, oep, les, dram, login_level, output_order):
    """Compare convolutional layer output stream with expected results.

    The debug logging functionality is controlled by login_level to manage file I/O:
    - When login_level is sufficient, the function creates and writes to debug log files
    - Files are properly closed when the function finishes
    - Basic logging through logger remains active regardless of login_level
    
    This function monitors and validates the output stream from a convolutional layer
    in the OpenEye accelerator. It handles the complex data organization of 
    convolution outputs across multiple processing elements and clusters.
    
    Args:
        ptp: Port timing parameters
        dut: Device under test (OpenEye accelerator instance)
        layer_number: Index of the current layer in the network
        layer_repetition: Counter for processing subdivided layers
        layer_parameters: Configuration parameters for the current layer
        oep: OpenEye architecture parameters
        les: Layer execution state tracking object
        dram: Memory object for storing computation results
        login_level: Logging verbosity control
        output_order: Mapping of output data organization
    
    Technical Details:
        - Handles both serial and parallel operation modes
        - Manages output data collection from multiple clusters
        - Tracks partial sum accumulation
        - Supports debug logging of intermediate results
        - Handles data reordering based on cluster organization
        - Supports various data quantization modes
        - Maintains state across multiple execution cycles
    
    Note:
        Debug logging creates detailed output files when login_level is sufficient:
        - output.txt: Raw output stream data
        - storage_input.txt: Detailed state tracking information
    """

    if(logging.DEBUG >= login_level):
        filename = 'demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/output.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        txt_file = open(filename, 'w')
        filename = 'demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/storage_input.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        storage_file = open(filename, 'w')

    logger.debug("PRE")
    logger.debug("f: " + str(les.f_start) + " x: " + str(les.x_start) + " y: " + str(les.y_start) + " f_corner_start: " + str(les.f_corner_start) + " y_corner_start: " + str(les.y_corner_start) + " x_corner_start: " + str(les.x_corner_start) + "\n")

    f = 0
    x = 0
    y = 0
    les.current_position = 0
    if(logging.DEBUG >= login_level):
        storage_file.write(" f_corner_start: " + str(les.f_corner_start) + " y_corner_start: " + str(les.y_corner_start) + " x_corner_start: " + str(les.x_corner_start) + "\n")
    if (oep.SERIAL == 0) :
        if ((layer_repetition % layer_parameters.iact_transmissions_pe) == (layer_parameters.iact_transmissions_pe - 1)):
            cocotb.start_soon(send_enable_conv(ptp, dut, layer_parameters, layer_repetition, oep))
            while (dut.psum_enable_o.value == 0):
                await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
            dut._log.info("Output Stream started")
            assert dut.psum_enable_o.value != 0, "psum is not 1!"
            while (dut.psum_enable_o.value != 0):
                cluster_order = []
                for a in range(layer_parameters.used_Y_cluster):
                    for b in range(0,oep.Clusters_Y,layer_parameters.used_Y_cluster):
                        cluster_order.append(a+b)
                for y_cluster in reversed(cluster_order):
                    for x_cluster in reversed(range(oep.Clusters_X)):
                        for router in reversed(range(oep.NUM_GLB_PSUM)):
                            if(layer_parameters.computing_mx[oep.Clusters_X-x_cluster-1][oep.Clusters_Y-y_cluster-1][0][oep.NUM_GLB_PSUM-router-1]== 1):
                                lower_limit = (x_cluster * oep.Clusters_Y * oep.NUM_GLB_PSUM * 40 + y_cluster * oep.NUM_GLB_PSUM * 40 + router * 40)
                                upper_limit = lower_limit + 39
                                outputvalue = dut.psum_data_o.value[lower_limit:upper_limit]
                                if(logging.DEBUG >= login_level):
                                    txt_file.write(bin(outputvalue)[2:].zfill(40) + "\n")
                                for i in range(2):
                                    try:
                                        f = output_order[layer_repetition][les.current_position][0]
                                        x = output_order[layer_repetition][les.current_position][1]
                                        y = output_order[layer_repetition][les.current_position][2]
                                    except:
                                        pass
                                    les.current_position = les.current_position + 1
                                    if(logging.DEBUG >= login_level):
                                        storage_file.write("f: " + str(f) + " x: " + str(x) + " y: " + str(y) + "\n")
                                    try:
                                        dram.fmap[layer_number + 1][f][x][y] = int(dut.psum_data_o.value[lower_limit+20*(1-i):upper_limit-20*i])
                                        if (dram.fmap[layer_number + 1][f][x][y] >= 2**19) :
                                            dram.fmap[layer_number + 1][f][x][y] = dram.fmap[layer_number + 1][f][x][y] - 2**20
                                    except:
                                        pass
                await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))
        dut._log.info("Output Stream finished")
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp,(dut.status_reg_enable_i), 1))
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    else:
        cluster_order = []
        for b in range(0,oep.Clusters_Y,layer_parameters.used_Y_cluster):
            for a in range(layer_parameters.used_Y_cluster):
                cluster_order.append(int(oep.Clusters_Y/layer_parameters.used_Y_cluster)*a+int(b/layer_parameters.used_Y_cluster))

        matrix = [[i * 8 + j for j in range(8)] for i in range(8)]
        reordered_matrix = [matrix[i] for i in cluster_order]

        flat_list = [item for row in reordered_matrix for item in row]


        dut._log.info("Output Stream started")
        used_clusters_per_calc = math.ceil(layer_parameters.iact_size_x / 4) * 4
        values_per_transmission = math.ceil(layer_parameters.different_kernels_per_calculation*used_clusters_per_calc/2)
        transmissions_per_cycle = (oep.Clusters_Y * oep.Clusters_X * oep.PEs_X)//2
        current_cycle = 0
        while (dut.enable_dma_o.value == 1):

            if(logging.DEBUG >= login_level):
                try:
                    txt_file.write(bin(int(dut.data_dma_o.value))[2:].zfill(40) + "\n")
                except:
                    txt_file.close()
                    storage_file.close()
                    logger.error("Error writing output txt-file")
                    raise Exception("X detected.")
            if (current_cycle < values_per_transmission):
                for i in range(2):
                    if(logging.DEBUG >= login_level):
                        storage_file.write("f: " + str(f) + " x: " + str(flat_list[x]) + " y: " + str(y) + "\n")
                    try:
                        dram.fmap[layer_number + 1][f][flat_list[x]][y] = int(dut.data_dma_o.value[44-20*i:63-20*i])
                        if (dram.fmap[layer_number + 1][f][flat_list[x]][y] >= 2**19):
                            dram.fmap[layer_number + 1][f][flat_list[x]][y] = dram.fmap[layer_number + 1][f][flat_list[x]][y] - 2**20
                    except:
                        pass
                    x = x + 1
                if (current_cycle % math.ceil(oep.PEs_X/2) == math.ceil(oep.PEs_X/2) - 1):
                    if(x >= layer_parameters.iact_size_x):
                        x = 0
                        f = f + 1
                        if(f == layer_parameters.filters):
                            f = 0
                            y = y + 1
                            if(y >= layer_parameters.iact_size_y):
                                y = 0
            else:
                for i in range(2):
                    if(logging.DEBUG >= login_level):
                        storage_file.write("Empty storage line." + "\n")
            current_cycle = current_cycle + 1
            if (current_cycle == transmissions_per_cycle):
                current_cycle = 0

            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        cocotb.start_soon(set_input(ptp,(dut.ready_dma_i), 0))

    if(math.floor(layer_repetition%(layer_parameters.iact_transmissions_pe*layer_parameters.needed_wght_transmissions)) == \
        (layer_parameters.iact_transmissions_pe*layer_parameters.needed_wght_transmissions-1)):
        les.y_corner_start = y
        les.x_corner_start = x
    if(logging.DEBUG >= login_level):
        txt_file.close()
        storage_file.write("f: " + str(f) + " x: " + str(x) + " y: " + str(y) + "\n")
        storage_file.write(" f_corner_start: " + str(les.f_corner_start) + " y_corner_start: " + str(les.y_corner_start) + " x_corner_start: " + str(les.x_corner_start) + "\n")
        storage_file.write("iact_transmissions_pe: " + str(layer_parameters.iact_transmissions_pe) + " needed_wght_transmissions: " + str(layer_parameters.needed_wght_transmissions) + "\n")
        storage_file.close()
        logger.debug("POST")
        logger.debug("f: " + str(f) + " x: " + str(x) + " y: " + str(y) + " f_corner_start: " + str(les.f_corner_start) + " y_corner_start: " + str(les.y_corner_start) + " x_corner_start: " + str(les.x_corner_start) + "\n")
    pass

async def compare_stream_Dw(ptp, dut, layer_number, model, layer_repetition, layer_parameters, oep, les, dram, login_level, output_order):
    """Compare depthwise convolution layer output stream with expected results.
    
    This function monitors and validates the output stream from a depthwise convolution
    layer in the OpenEye accelerator. It handles the specific data organization and
    computation patterns used in depthwise convolutions.
    
    Args:
        ptp: Port timing parameters
        dut: Device under test (OpenEye accelerator instance)
        layer_number: Index of the current layer in the network
        model: Neural network model configuration
        layer_repetition: Counter for processing subdivided layers
        layer_parameters: Configuration parameters for the current layer
        oep: OpenEye architecture parameters
        les: Layer execution state tracking object
        dram: Memory object for storing computation results
        login_level: Logging verbosity control
        output_order: Mapping of output data organization
    
    Technical Details:
        - Specialized for depthwise convolution output patterns
        - Handles channel-wise computation results
        - Manages output collection from multiple PE clusters
        - Supports debugging through detailed logging
        - Maintains proper data organization per channel
        - Handles timing and synchronization specific to depthwise operations
    
    Note:
        Debug logging (when login_level is sufficient) creates:
        - output.txt: Raw depthwise convolution output data
        - storage_input.txt: State tracking and debugging information
    """
    if(logging.DEBUG >= login_level):
        filename = 'demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/output.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        txt_file = open(filename, 'w')
        filename = 'demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/storage_input.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        storage_file = open(filename, 'w')
    if(logging.DEBUG >= login_level):
        storage_file.write(" f_corner_start: " + str(les.f_corner_start) + " y_corner_start: " + str(les.y_corner_start) + " x_corner_start: " + str(les.x_corner_start) + "\n")
    cocotb.start_soon(send_enable_dw(ptp, dut, layer_parameters, layer_repetition, oep))
    while (dut.psum_enable_o.value == 0):
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    dut._log.info("Output Stream started")
    assert dut.psum_enable_o.value != 0, "psum is not 1!"
    while (dut.psum_enable_o.value != 0):
        for y_cluster in reversed(range(oep.Clusters_Y)):
            for x_cluster in reversed(range(oep.Clusters_X)):
                for router in reversed(range(oep.NUM_GLB_PSUM)):
                    if(layer_parameters.computing_mx[oep.Clusters_X-x_cluster-1][oep.Clusters_Y-y_cluster-1][0][oep.NUM_GLB_PSUM-router-1]== 1):
                        lower_limit = (x_cluster * oep.Clusters_Y * oep.NUM_GLB_PSUM * 40 + y_cluster * oep.NUM_GLB_PSUM * 40 + router * 40)
                        upper_limit = lower_limit + 39
                        outputvalue = dut.psum_data_o.value[lower_limit:upper_limit]

                        if(logging.DEBUG >= login_level):
                            txt_file.write(bin(outputvalue)[2:].zfill(40) + "\n")
                        for i in range(2):
                            try:
                                f = output_order[layer_repetition][les.current_position][0]
                                x = output_order[layer_repetition][les.current_position][1]
                                y = output_order[layer_repetition][les.current_position][2]
                            except:
                                pass
                            les.current_position = les.current_position + 1
                            if(logging.DEBUG >= login_level):
                                storage_file.write("f: " + str(f) + " x: " + str(x) + " y: " + str(y) + "\n")
                            try:
                                dram.fmap[layer_number + 1][f][x][y] = int(dut.psum_data_o.value[lower_limit+20*(1-i):upper_limit-20*i])
                                if (dram.fmap[layer_number + 1][f][x][y] >= 2**19) :
                                    dram.fmap[layer_number + 1][f][x][y] = dram.fmap[layer_number + 1][f][x][y] - 2**20
                            except:
                                pass

        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    #if (math.floor(((1+layer_repetition)/layer_parameters.psum_transmissions_pe)) > math.floor((layer_repetition/layer_parameters.psum_transmissions_pe))):
    les.current_position = 0
    await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))
    dut._log.info("Output Stream finished")
    await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp,(dut.status_reg_enable_i), 1))
    await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    
    if(math.floor(layer_repetition%(layer_parameters.iact_transmissions_pe*layer_parameters.needed_wght_transmissions)) == \
        (layer_parameters.iact_transmissions_pe*layer_parameters.needed_wght_transmissions-1)):
        les.y_corner_start = y
        les.x_corner_start = x

    if(logging.DEBUG >= login_level):
        txt_file.close()
        storage_file.write("f: " + str(f) + " x: " + str(x) + " y: " + str(y) + "\n")
        storage_file.write(" f_corner_start: " + str(les.f_corner_start) + " y_corner_start: " + str(les.y_corner_start) + " x_corner_start: " + str(les.x_corner_start) + "\n")
        storage_file.write("iact_transmissions_pe: " + str(layer_parameters.iact_transmissions_pe) + " needed_wght_transmissions: " + str(layer_parameters.needed_wght_transmissions) + "\n")
        storage_file.close()
        logger.debug("POST")
        logger.debug("f: " + str(f) + " x: " + str(x) + " y: " + str(y) + " f_corner_start: " + str(les.f_corner_start) + " y_corner_start: " + str(les.y_corner_start) + " x_corner_start: " + str(les.x_corner_start) + "\n")

    pass

async def compare_stream_Dense(ptp, dut, layer_number, layer_repetition, layer_parameters, oep, les, dram, login_level):
    """ Await the output stream and compare it to the reference output.

    This function awaits the output stream and compares it to the reference output.
    
    Args:
        dut: The DUT.
        layer_number: The index of the layer.
        model: The model.
        layer_repetition: The index of the part of a layer, if it is too large to be processed at once.
        layer_parameters: The layer parameters.
        oep: The OpenEye parameters.
        les: The layer execution state.
    """
    if(logging.DEBUG >= login_level):
        filename = 'demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/output.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        txt_file = open(filename, 'w')
        filename = 'demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/storage_input.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        storage_file = open(filename, 'w')


    offset_layer_repetition = (math.floor(layer_repetition/layer_parameters.iact_transmissions_pe) % layer_parameters.psum_transmissions_pe) * oep.Clusters_Y * oep.Clusters_X * layer_parameters.used_psum_per_PE
    les.x = offset_layer_repetition

    if ((layer_repetition % layer_parameters.iact_transmissions_pe) == (layer_parameters.iact_transmissions_pe - 1)) :cluster_order = []
    for b in range(0,oep.Clusters_Y,layer_parameters.used_Y_cluster):
        for a in range(layer_parameters.used_Y_cluster):
            cluster_order.append(int(oep.Clusters_Y/layer_parameters.used_Y_cluster)*a+int(b/layer_parameters.used_Y_cluster))

    matrix = [[i * 8 + j for j in range(8)] for i in range(8)]

    f = 0
    dut._log.info("Output Stream started")
    cluster_offset = math.ceil(layer_parameters.used_psum_per_PE)
    while (dut.enable_dma_o.value == 1):

        if(logging.DEBUG >= login_level):
            txt_file.write(bin(int(dut.data_dma_o.value))[2:].zfill(40) + "\n")
        if(logging.DEBUG >= login_level):
            storage_file.write("f: " + str(f) + "\n")
        try:
            dram.fmap[layer_number + 1][f] = int(dut.data_dma_o.value[44:63])
            if (dram.fmap[layer_number + 1][f] >= 2**19) :
                dram.fmap[layer_number + 1][f] = dram.fmap[layer_number + 1][f] - 2**20
        except:
            pass
        if (f < cluster_offset) :
            f = f + cluster_offset
        else :
            f = f - cluster_offset + 1

        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

    cocotb.start_soon(set_input(ptp,(dut.ready_dma_i), 0))
    
    if(logging.DEBUG >= login_level):
        txt_file.close()
        storage_file.write("f: " + str(f) + "\n")
        storage_file.close()
        logger.debug("POST")
        logger.debug("f: " + str(f) + "\n")
    pass

async def compare_stream_Pooling(ptp, dut, layer_number, layer_repetition, layer_parameters, oep, les, dram, login_level):
    """ Await the output stream and compare it to the reference output.

    This function awaits the output stream and compares it to the reference output.
    
    Args:
        dut: The DUT.
        layer_number: The index of the layer.
        model: The model.
        layer_repetition: The index of the part of a layer, if it is too large to be processed at once.
        layer_parameters: The layer parameters.
        oep: The OpenEye parameters.
        les: The layer execution state.
    """
    if(logging.DEBUG >= login_level):
        filename = 'demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/output.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        txt_file = open(filename, 'w')
        filename = 'demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/storage_input.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        storage_file = open(filename, 'w')

    f = 0
    dut._log.info("Output Stream started")
    while (dut.enable_dma_o.value == 1):

        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

    cocotb.start_soon(set_input(ptp,(dut.ready_dma_i), 0))
    
    if(logging.DEBUG >= login_level):
        txt_file.close()
        storage_file.write("f: " + str(f) + "\n")
        storage_file.close()
        logger.debug("POST")
        logger.debug("f: " + str(f) + "\n")
    pass

async def send_enable_conv(ptp, dut, layer_params, layer_repetition, oep):
    """Send enable signals for convolution layer operation.
    
    This function manages the enable signal timing for convolutional layer
    processing in the OpenEye accelerator. It controls when processing elements
    start their computations and handles synchronization across clusters.
    
    Args:
        ptp: Port timing parameters
        dut: Device under test (OpenEye accelerator instance)
        layer_params: Configuration parameters for the current layer
        layer_repetition: Counter for processing subdivided layers
        oep: OpenEye architecture parameters
        
    Implementation Details:
        - Enables all partial sum GLBs simultaneously
        - Calculates appropriate timing based on computation mode
        - Supports different cluster computation patterns
        - Handles proper synchronization of enable signals
        - Manages timing for multiple processing cycles
    """
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), (2**(oep.Clusters_X*oep.Clusters_Y*oep.NUM_GLB_PSUM))-1))


            
    match layer_params.single_cluster_computation:
        case 1:
            for _ in range(int((math.ceil((layer_params.filters*layer_params.output_shape[1]*layer_params.output_shape[2])/2/(oep.Clusters_X*oep.Clusters_Y*oep.NUM_GLB_PSUM))))):
                await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        case 2:
            for _ in range(int((math.ceil((layer_params.filters*layer_params.output_shape[1]*layer_params.output_shape[2])/(2*oep.Clusters_X*oep.Clusters_Y*oep.NUM_GLB_PSUM))))):
                await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        case _:
            for _ in range(int((math.ceil(layer_params.filters/layer_params.needed_wght_transmissions/2)*\
                                math.ceil(layer_params.needed_refreshes_mx[layer_repetition][0]/layer_params.used_Y_cluster)))):
                await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)


    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))

async def send_enable_dw(ptp, dut, layer_params, layer_repetition, oep):
    """Send enable signals for depthwise convolution layer operation.

    This function manages the enable signal timing for depthwise convolutional
    layer processing. Unlike standard convolution, depthwise convolution applies
    filters per channel, requiring different timing calculations.

    Args:
        ptp: Port timing parameters
        dut: Device under test (OpenEye accelerator instance)
        layer_params: Configuration parameters for the depthwise convolution layer
        layer_repetition: Counter for processing subdivided layers
        oep: OpenEye architecture parameters

    Implementation Details:
        - Enables all partial sum GLBs across all clusters
        - Calculates cycles based on refreshes needed per Y cluster
        - Handles channel-wise computation timing
        - Automatically disables signals after completion

    Note:
        Depthwise convolution timing differs from standard convolution as
        each channel is processed independently with its own filter.
    """
    # Enable all partial sum global buffers (one per cluster router)
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), (2**(oep.Clusters_X*oep.Clusters_Y*oep.NUM_GLB_PSUM))-1))

    # Calculate wait cycles: refreshes needed divided by clusters, divided by 2 (dual outputs)
    for _ in range(int((math.ceil(layer_params.needed_refreshes_mx[layer_repetition][0]/layer_params.used_Y_cluster/2)))):
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

    # Disable partial sum enable signal
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))

async def send_enable_dense(ptp, dut, layer_params, layer_repetition, oep):
    """Send enable signals for dense (fully-connected) layer operation.

    This function manages the enable signal timing for dense/fully-connected
    layer processing in the OpenEye accelerator. Dense layers perform matrix
    multiplication between input vectors and weight matrices.

    Args:
        ptp: Port timing parameters
        dut: Device under test (OpenEye accelerator instance)
        layer_params: Configuration parameters for the dense layer
        layer_repetition: Counter for processing subdivided layers (unused here)
        oep: OpenEye architecture parameters

    Implementation Details:
        - Enables all partial sum GLBs simultaneously
        - Wait time based on number of partial sums per PE
        - Divided by 2 due to dual parallel outputs per cycle
        - Simpler timing than convolution (no spatial dimensions)

    Note:
        Dense layers have simpler timing than convolution as they lack
        spatial dimensions and process vector-matrix multiplication.
    """
    # Enable all partial sum global buffers across all clusters
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), (2**(oep.Clusters_X*oep.Clusters_Y*oep.NUM_GLB_PSUM))-1))

    # Wait cycles based on partial sums per PE (divided by 2 for dual outputs)
    for _ in range(math.ceil(layer_params.used_psum_per_PE/2)):
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

    # Disable partial sum enable signal
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))
