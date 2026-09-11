# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
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
import open_eye.iact_stream_mapper as iact_stream_mapper
from cocotb.triggers import Timer
import open_eye.stream_dicts as strdic
import random

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
        # iact_addr_len_i was removed from OpenEye_Parallel in fed1325; PE.v has
        # no reader for it any more, so only the weight length is driven.
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
        # stride_x_i/stride_y_i were dropped from OpenEye_Parallel in 4cc9e07;
        # the config stream no longer carries stride.
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

        # Dataflow selection: 0 = row-stationary (default), 1 = output-stationary GEMM.
        # Guarded so testbenches for DUTs without the port keep working.
        if hasattr(dut, "gemm_mode_i"):
            gemm_mode_value = stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["gemm_mode"]]
            if gemm_mode_value == []:
                gemm_mode_value = 0
            cocotb.start_soon(set_input(ptp,(dut.gemm_mode_i), gemm_mode_value))

        # Raw weight stream flag: dense/FC and GEMM layers send uncompressed
        # weights whose all-zero words must be stored, not skipped.
        if hasattr(dut, "raw_wght_i"):
            raw_wght_value = 1 if (getattr(lp, "fully_connected", 0) or
                                   getattr(lp, "gemm_mode", 0)) else 0
            cocotb.start_soon(set_input(ptp,(dut.raw_wght_i), raw_wght_value))
        cocotb.start_soon(set_input(ptp,(dut.fraction_bit_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["realfactor"]]))

        # Computation control parameters
        cocotb.start_soon(set_input(ptp,(dut.needed_cycles_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["needed_refreshes"]]))
        cocotb.start_soon(set_input(ptp,(dut.needed_x_cls_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_X_cluster"]]))
        cocotb.start_soon(set_input(ptp,(dut.needed_y_cls_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_Y_cluster"]]))
        cocotb.start_soon(set_input(ptp,(dut.needed_iact_cycles_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["needed_Iact_writes"]]))

        # Filter and memory configuration
        cocotb.start_soon(set_input(ptp,(dut.filters_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_psum_per_PE"]]))
        # iact_addr_len_i no longer exists on OpenEye_Parallel (removed in fed1325).
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
        # stride_x_i/stride_y_i were dropped from OpenEye_Parallel in 4cc9e07;
        # the config stream no longer carries stride.
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

        # The serial stream is assembled with stream_serial_dict but indexed
        # below with stream_parallel_dict names; log what is actually being
        # sent so a section-count mismatch with the DUT is visible.
        logger.info("DMA stream sections (words each): %s",
                    [len(section) for section in stream])
        for name in ("trans_cycles_iact", "trans_cycles_wght", "trans_cycles_psum"):
            try:
                logger.info("DUT expects %s = %d", name, int(getattr(dut, name).value))
            except Exception:
                pass

        # Enable DMA transfer
        cocotb.start_soon(set_input(ptp,(dut.enable_dma_i), 1))

        # Send status/configuration data
        for data_word in range(len(stream[strdic.stream_parallel_dict["status"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["status"]][data_word]))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        # dma_storage has now decoded the configuration words, so the DUT's
        # own expectations are readable and can be compared with what is about
        # to be sent. A mismatch here stalls the corresponding GET_* state.
        for name in ("trans_cycles_iact", "trans_cycles_wght", "trans_cycles_psum"):
            try:
                logger.info("DUT expects %s = %d", name, int(getattr(dut, name).value))
            except Exception as exc:
                logger.info("DUT %s unreadable (%s)", name, type(exc).__name__)
        try:
            logger.info("DUT compute_mask_reg = %s", hex(int(dut.compute_mask_reg.value)))
        except Exception as exc:
            logger.info("DUT compute_mask_reg unreadable (%s)", type(exc).__name__)
        for name in ("send_data_out", "store_in_psum", "fully_connected_layer"):
            try:
                logger.info("DUT %s = %d", name, int(getattr(dut, name).value))
            except Exception as exc:
                logger.info("DUT %s unreadable (%s)", name, type(exc).__name__)
        logger.info("Host will send: iact %d, wght %d, psum %d, quantize %d words",
                    len(stream[strdic.stream_parallel_dict["iact"]]),
                    len(stream[strdic.stream_parallel_dict["wght"]]),
                    len(stream[strdic.stream_parallel_dict["psum"]]),
                    len(stream[strdic.stream_parallel_dict["quantize"]]))

        # Send input activation data
        for data_word in range(len(stream[strdic.stream_parallel_dict["iact"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["iact"]][data_word]))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        # Send weight data
        for data_word in range(len(stream[strdic.stream_parallel_dict["wght"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["wght"]][data_word]))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        # Send partial sum data
        """for data_word in range(len(stream[strdic.stream_parallel_dict["psum"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["psum"]][data_word]))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)"""
        index = 0
        while index < len(stream[strdic.stream_parallel_dict["psum"]]):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["psum"]][index]))
            if random.random() < 1.0:
                index += 1
                cocotb.start_soon(set_input(ptp,(dut.enable_dma_i), 1))
            else:
                cocotb.start_soon(set_input(ptp,(dut.enable_dma_i), 0))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        # Send quantization parameters
        for data_word in range(len(stream[strdic.stream_parallel_dict["quantize"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["quantize"]][data_word]))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        # Complete DMA transfer
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp,(dut.enable_dma_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.ready_dma_i), 1))
        
# Diagnostic: main-FSM states the DUT has visited, filled by probe_fsm_states
# when OPENEYE_PROBE_FSM is set. Reported by the iact buffer check on failure.
fsm_states_seen = set()
fsm_probe_samples = [0]
fsm_state_trace = []


async def trace_pe_iact(ptp, dut, oep, max_lines=200):
    """Per-cycle iact trace at one PE (opt-in: TRACE_PE_IACT=cx,cy,col,row).

    Logs every cycle any iact lane is valid at the PE: which lane the PE
    selects, lane valid/ready, the lane data, and whether the PE's iact data
    SPAD is written. Shows whether a selected PE sees the words at all and,
    if so, why it does not store them.
    """
    try:
        cx, cy, col, row = [int(v) for v in os.environ.get("TRACE_PE_IACT", "0,0,0,0").split(",")]
        pe = (dut.OpenEye_Parallel.gen_x[cx].gen_y[cy].OpenEye_Cluster
              .pe_cluster.gen_X[col].gen_Y[row].pe)
        sp = pe.iact_data_SPad
    except Exception as exc:
        logger.error("trace_pe_iact: PE not reachable (%s)", type(exc).__name__)
        return

    def txt(sig):
        try:
            return str(sig.value)
        except Exception:
            return "?"

    lines = 0
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        en = txt(pe.iact_enable_i)
        if "1" not in en:
            continue
        logger.info("peiact t=%s st=%s sel=%s en=%s rdy=%s data=%s | spad we=%s addr=%s d=%s",
                    cocotb.utils.get_sim_time("ns"), txt(pe.current_state_computing),
                    txt(pe.iact_select_i), en, txt(pe.iact_ready_o), txt(pe.iact_data_i),
                    txt(sp.we_i), txt(sp.addr_i), txt(sp.data_i))
        lines += 1


async def trace_converter(ptp, dut, oep, max_lines=260):
    """Per-cycle trace of one iact converter (opt-in: TRACE_CONVERTER=cx,cy).

    Follows the iact data from BUFFER_A cell 0 into the converter's window,
    its internal RAM writes, the ENCODE reads and the words it hands to the
    array. Logs only cycles with some activity, from CONVERT_IACT on.
    """
    try:
        cx, cy = [int(v) for v in os.environ.get("TRACE_CONVERTER", "0,0").split(",")]
        cv = dut.IACT_CONVERTER_X[cx].IACT_CONVERTER_Y[cy].iact_stream_constructor
        cell0 = dut.BUFFER_A[0].iact_layer_buffer
    except Exception as exc:
        logger.error("trace_converter: converter not reachable (%s)", type(exc).__name__)
        return

    def txt(sig):
        try:
            return str(sig.value)
        except Exception:
            return "?"

    def hexv(sig):
        t = txt(sig)
        if t and set(t) <= {"0", "1"}:
            return hex(int(t, 2))
        return "X" if ("x" in t.lower() or "z" in t.lower()) else t

    lines = 0
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        try:
            st = int(dut.fsm_current_state.value)
        except ValueError:
            continue
        if st < 8:
            continue
        active = ("1" in txt(cv.enable_store) or "1" in txt(cv.ram_wr_en_q)
                  or "1" in txt(cv.ram_rd_en) or "1" in txt(cv.iact_enable_o)
                  or "1" in txt(cell0.rd_en_i))
        if not active:
            continue
        logger.info("convtrace t=%s st=%d enc=%s | A0 rd=%s addr=%s q=%s | win=%s store=%s | "
                    "wr=%s waddr=%s | rd=%s raddr=%s rdata=%s | en_o=%s data_o=%s",
                    cocotb.utils.get_sim_time("ns"), st, hexv(cv.fsm_enc_current_state),
                    txt(cell0.rd_en_i), hexv(cell0.addr_i), hexv(cell0.data_o),
                    hexv(cv.storage_i), txt(cv.enable_store),
                    txt(cv.ram_wr_en_q), hexv(cv.ram_wr_addr_q),
                    txt(cv.ram_rd_en), hexv(cv.ram_rd_addr), hexv(cv.ram_data_o),
                    txt(cv.iact_enable_o), hexv(cv.iact_data_o))
        lines += 1


def dump_pe_occupancy(dut, oep, depth=32):
    """Log how many of the first `depth` words each PE has written.

    One line per cluster, one cell per PE (column/row): iact data SPAD /
    weight data SPAD / psum lane 0. Cheap enough to call when a run stalls,
    which is exactly when the end-of-run dumps never get a chance to run.
    """
    for cx in range(oep.Clusters_X):
        for cy in range(oep.Clusters_Y):
            cells = []
            for col in range(oep.PEs_X):
                for row in range(oep.PEs_Y):
                    try:
                        pe = (dut.OpenEye_Parallel.gen_x[cx].gen_y[cy].OpenEye_Cluster
                              .pe_cluster.gen_X[col].gen_Y[row].pe)
                    except Exception:
                        cells.append("c%dr%d:?" % (col, row))
                        continue
                    counts = []
                    for mem in (lambda: pe.iact_data_SPad, lambda: pe.weight_data_SPad,
                                lambda: pe.gen_serial_psum_spad.gen_psum_spad[0].psum_SPad):
                        try:
                            m = mem().ram.impl.mem
                            n = 0
                            for a in range(depth):
                                try:
                                    int(m[a].value)
                                    n += 1
                                except ValueError:
                                    pass
                                except IndexError:
                                    break
                            counts.append(str(n))
                        except Exception:
                            counts.append("?")
                    cells.append("c%dr%d:%s" % (col, row, "/".join(counts)))
            logger.error("occupancy cluster(%d,%d) iact/wght/psum0: %s", cx, cy, " ".join(cells))


def _written_words(mem, depth):
    """Count words among the first `depth` addresses that are not X."""
    n = 0
    for a in range(depth):
        try:
            int(mem[a].value)
            n += 1
        except ValueError:
            pass
        except IndexError:
            break
    return n


def dump_iact_path(dut, oep, depth=64):
    """Log written-word counts along the iact path: BUFFER_A, then iact GLBs.

    Together with dump_pe_occupancy this shows where iacts stop: never
    stored in the FPGA iact buffer, stored but never written into a
    cluster's iact GLBs, or in the GLBs but never delivered to the PEs.
    """
    cells = []
    for cell in range(oep.IACT_RAM_CELLS):
        try:
            cells.append(_written_words(dut.BUFFER_A[cell].iact_layer_buffer.impl.mem, depth))
        except Exception:
            cells.append("?")
    logger.error("iact path BUFFER_A written words per cell (first %d addrs): %s", depth, cells)
    if oep.SERIAL:
        return  # no cluster iact GLBs in the SERIAL build (GLB_cluster gen_iact is empty)
    for cx in range(oep.Clusters_X):
        for cy in range(oep.Clusters_Y):
            glbs = []
            for g in range(oep.NUM_GLB_IACT):
                try:
                    glb = (dut.OpenEye_Parallel.gen_x[cx].gen_y[cy].OpenEye_Cluster
                           .glb_cluster.gen_iact[g].iact_glb.impl.mem)
                    glbs.append(_written_words(glb, depth))
                except Exception as exc:
                    glbs.append("?(%s)" % type(exc).__name__)
            logger.error("iact path cluster(%d,%d) iact GLB written words: %s", cx, cy, glbs)


# Filled by monitor_iact_handoff, reported by report_iact_handoff.
iact_handoff_counts = {"enable": {}, "handshake": {}, "choose": {}, "cycles": 0}


async def monitor_iact_handoff(ptp, dut):
    """Count iact valid and handshake cycles per bit at the OpenEye_Parallel input.

    In the SERIAL FPGA build there are no cluster iact GLBs: the converter
    drives iact_data/enable/choose into OpenEye_Parallel directly. Counting
    enable and enable&ready per bit shows whether the converter emits nothing,
    emits on the wrong lanes, or emits without being accepted.
    """
    def read(name):
        try:
            return int(getattr(dut, name).value)
        except Exception:
            return None

    while True:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        iact_handoff_counts["cycles"] += 1
        en = read("iact_enable_i_oep_w")
        if not en:
            continue
        rdy = read("iact_ready_o_oep_w") or 0
        bit = 0
        while en >> bit:
            if (en >> bit) & 1:
                iact_handoff_counts["enable"][bit] = iact_handoff_counts["enable"].get(bit, 0) + 1
                if (rdy >> bit) & 1:
                    iact_handoff_counts["handshake"][bit] = iact_handoff_counts["handshake"].get(bit, 0) + 1
            bit += 1
        ch = read("iact_choose_i_oep_w")
        if ch is not None:
            key = (hex(ch), hex(en))
            if key in iact_handoff_counts["choose"] or len(iact_handoff_counts["choose"]) < 32:
                iact_handoff_counts["choose"][key] = iact_handoff_counts["choose"].get(key, 0) + 1


def report_iact_handoff(oep=None):
    c = iact_handoff_counts
    logger.error("iact handoff over %d cycles: enable cycles per bit %s, handshakes per bit %s",
                 c["cycles"], dict(sorted(c["enable"].items())), dict(sorted(c["handshake"].items())))
    # Which PE each (choose, enable) combination actually selects. Each PE
    # has a $clog2(NUM_GLB_IACT+1)-bit field; NUM_GLB_IACT means "none".
    for (ch, en), n in sorted(c["choose"].items(), key=lambda kv: -kv[1]):
        line = "  %d valid cycles with enable=%s choose=%s" % (n, en, ch)
        if oep is not None:
            w = max(1, (oep.NUM_GLB_IACT).bit_length())
            pes = oep.PEs_X * oep.PEs_Y
            v = int(ch, 16)
            parts = []
            for cl in range(oep.Clusters_X * oep.Clusters_Y):
                sel = []
                for pe in range(pes):
                    f = (v >> ((cl * pes + pe) * w)) & ((1 << w) - 1)
                    if f != oep.NUM_GLB_IACT:
                        sel.append("c%dr%d<-g%d" % (pe % oep.PEs_X, pe // oep.PEs_X, f))
                parts.append("cl%d[%s]" % (cl, ",".join(sel) if sel else "none"))
            line += " -> " + " ".join(parts)
        logger.error(line)


async def probe_fsm_states(ptp, dut, stall_report_after=20000, oep=None):
    """Record every main-FSM state the DUT enters (opt-in diagnostic).

    Sampling every clock costs simulation time, so the testbench only starts
    this when OPENEYE_PROBE_FSM is set. It answers questions the post-hoc
    checks cannot, such as whether RECEIVE_PSUMS_TO_IACT is ever reached.
    """
    previous = None
    stuck_for = 0
    reported = False
    while True:
        fsm_probe_samples[0] += 1
        try:
            state = int(dut.fsm_current_state.value)
        except ValueError:
            state = None
        if state is not None:
            fsm_states_seen.add(state)
            if state != previous:
                if len(fsm_state_trace) < 200:
                    fsm_state_trace.append((str(cocotb.utils.get_sim_time("ns")), state))
                # Log live: a test that stalls may never reach a check that
                # would otherwise report the trace.
                # Also sample the decoded config fields that steer the FSM:
                # they come from dma_storage's shift chain, so a stray write
                # after GET_PARAMETERS would change them mid-layer.
                fields = []
                for name in ("fsm_psum_current_state", "send_data_out", "store_in_psum",
                             "trans_cycles_iact", "trans_cycles_psum"):
                    try:
                        fields.append("%s=%d" % (name, int(getattr(dut, name).value)))
                    except Exception:
                        fields.append("%s=?" % name)
                logger.info("FSM state -> %d at %s ns (%s)", state,
                            cocotb.utils.get_sim_time("ns"), ", ".join(fields))
                stuck_for = 0
                reported = False
            else:
                stuck_for += 1
                # A state that stops advancing is the signature of a section
                # word-count mismatch, so report the counters that gate the
                # exit once rather than leaving a silent run to time out.
                if stuck_for == stall_report_after and not reported:
                    reported = True
                    counters = []
                    for name in ("fsm_cycle", "fsm_psum_cycle", "fsm_psum_current_state",
                                 "trans_cycles_iact", "trans_cycles_wght", "trans_cycles_psum"):
                        try:
                            counters.append("%s=%d" % (name, int(getattr(dut, name).value)))
                        except Exception:
                            pass
                    logger.error("FSM stuck in state %d for %d cycles: %s",
                                 state, stuck_for, ", ".join(counters))
                    try:
                        pp = dut.psum_pipeline_inst
                        for name in ("psum_enable_o", "psum_ready_o_reg", "router_mode_psum"):
                            logger.error("  %s = %s", name, str(getattr(pp, name).value))
                        for name in ("iact_ready_o_oep_w", "iact_enable_i_oep_w"):
                            logger.error("  %s = %s", name, str(getattr(dut, name).value))
                        for name in ("router_mode_iact", "router_mode_wght", "router_mode_psum"):
                            try:
                                logger.error("  DUT %s = %s", name, str(getattr(dut, name).value))
                            except Exception:
                                pass
                    except Exception as exc:
                        logger.error("  psum handshake vectors unreadable (%s)", type(exc).__name__)
                    report_iact_handoff(oep)
                    if oep is not None:
                        dump_iact_path(dut, oep)
                        dump_pe_occupancy(dut, oep)
            previous = state
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)


def _ram_word(mem_entry):
    """Read a whole RAM word as an int, or None if it holds X/Z.

    Words the DUT never wrote read back as X, and LogicArray conversion raises
    on those. Returning None lets the caller report an unwritten word as a
    mismatch instead of aborting the comparison with a ValueError.
    """
    try:
        return int(mem_entry.value)
    except ValueError:
        return None

def _signed(value, bits):
    """Reinterpret an unsigned field of the given width as two's complement."""
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def _to_ram_format(datapoints, params, layer_params):
    if datapoints == None:
        return None
    ram_words = []
    iacts_per_word = 8
    iact_word_width = params.IACT_Bitwidth
    temp_word = 0
    for index, data in enumerate(datapoints):
        position = index % iacts_per_word
        if (data < 0):
            data = data + 256
        temp_word = temp_word + (data << (position * params.IACT_Bitwidth))

        if (position == iacts_per_word - 1):
            ram_words.append(temp_word)
            temp_word = 0
        if (index == layer_params.iact_size_x*layer_params.iact_size_y*layer_params.channels - 1):
            break


    if (position != iacts_per_word - 1):
        ram_words.append(temp_word)
        temp_word = 0

    return ram_words

def _iact_buffer_words(oep, layer_params, iact_ref):
    """Expected BUFFER_A image for `iact_ref`, in cell order.

    Delegates to the stream mappers so the layout has exactly one definition,
    shared with the DMA stream builder. Returns None when the layer type has no
    raw-pixel buffer layout to compare against - Pooling is the case in
    practice: pooling_mapper builds no iact stream, and a pooled layer's
    LayerParameters never run calculate_transmission_cycles, so its layout
    fields are still at their constructor defaults.
    """
    name = str(getattr(layer_params, "layer_name", ""))
    if "Dense" in name:
        mapper = iact_stream_mapper.DenseIactStreamMapper
    elif "Conv" in name or "Depthwise" in name:
        mapper = iact_stream_mapper.IactStreamMapper
    else:
        return None
    return mapper(oep, layer_params, 0, iact_ref, 0).build_iact_buffer_words()


def iact_half_offset(dut, oep):
    """Base RAM address of the buffer half the DUT currently has selected.

    OpenEye_FPGA assembles the address as {choose_iact_buffer, addr_reg}, so the
    half-select occupies the top BRANCHES_CLOG bits and each half owns the lower
    BRANCHES_WIDTH = BUFFER_WIDTH - BRANCHES_CLOG address bits.
    """
    branches = getattr(oep, "BRANCHES", 1)
    if branches <= 1:
        return 0
    try:
        half = int(dut.choose_iact_buffer.value)
    except Exception:
        return 0
    return half << (oep.BUFFER_WIDTH - math.ceil(math.log2(branches)))


def iact_cells_per_group(oep):
    """Number of RAM cells one full buffer write cycle spans.

    Mirrors IACT_ONE_WORD_ALL_RAM in OpenEye_FPGA.v. It is a property of the
    hardware, not of a layer, so it is derived from the OpenEye parameters
    rather than read off LayerParameters (where it is only populated for the
    layer types that run calculate_transmission_cycles).
    """
    return math.ceil((oep.IACT_RAM_CELLS * oep.IACT_RAM_CELLS_WORD_BITWIDTH)
                     / oep.DMA_BITWIDTH)


def compare_iact_storage(ptp, dut, iact_ref, oep, layer_params):
    """Compare the iact double-buffer contents with reference activations.

    The expected buffer image comes from the same mapper that builds the DMA
    iact stream, so the checker cannot drift from the layout the hardware is
    actually fed. Word k of that image belongs in RAM cell
    ``k % iact_cycles_one_word_all_ram`` at address
    ``k // iact_cycles_one_word_all_ram``.

    Args:
        ptp: Port timing parameters (unused; kept for call-site symmetry)
        dut: Device under test (OpenEye_FPGA instance)
        iact_ref: Reference activations for the layer that reads this buffer
        oep: OpenEye parameters
        layer_params: Parameters of the layer that reads this buffer, i.e. the
            next layer - it owns used_channels and iact_cycles_one_word_all_ram

    Returns:
        bool: True when every word matches, False on any mismatch.
    """
    logger.info("Iact storages are checked.")

    try:
        shape = np.array(iact_ref).shape
    except Exception:
        shape = "?"
    logger.info("Iact reference: shape %s, used_channels %s, IACT_RAM_CELLS %s",
                shape, getattr(layer_params, "used_channels", "?"), oep.IACT_RAM_CELLS)

    expected = _to_ram_format(_iact_buffer_words(oep, layer_params, iact_ref),oep,layer_params)

    if expected is None:
        logger.warning("No iact buffer layout known for layer '%s'; skipping the check.",
                       getattr(layer_params, "layer_name", "?"))
        return True

    cells_per_group = iact_cells_per_group(oep)
    try:
        depth = len(dut.BUFFER_A[0].iact_layer_buffer.impl.mem)
    except TypeError:
        depth = 1 << oep.BUFFER_WIDTH
    logger.info("Iact buffer check: %d expected words, %d cells per group, depth %d (layer '%s')",
                len(expected), cells_per_group, depth, getattr(layer_params, "layer_name", "?"))
    if len(expected) > cells_per_group * depth:
        logger.error("Expected iact image (%d words) exceeds the buffer (%d cells x %d addresses); "
                     "check the layout parameters for layer '%s'.",
                     len(expected), cells_per_group, depth,
                     getattr(layer_params, "layer_name", "?"))
        return False
    # Addresses are relative to the half the DUT currently has selected.
    half_offset = iact_half_offset(dut, oep)
    error_found = False
    errors_logged = 0
    for index, ref_word in enumerate(expected):
        cell = index % oep.IACT_RAM_CELLS
        addr = index // oep.IACT_RAM_CELLS
        dut_word = _ram_word(dut.BUFFER_A[cell].iact_layer_buffer.impl.mem[half_offset + addr])
        if dut_word != ref_word:
            error_found = True
            errors_logged += 1
            if errors_logged <= 16:
                logger.error("Error found in Iact storage; word: %d cell: %d addr: %d", index, cell, addr)
                logger.error("Ref-Value: 0x%x DUT-Value: %s", ref_word,
                             "X (never written)" if dut_word is None else hex(dut_word))
    if errors_logged > 16:
        logger.error("... and %d further Iact storage mismatches.", errors_logged - 16)

    if error_found:
        _log_iact_value_comparison(dut, oep, expected, cells_per_group, half_offset)
        _log_iact_buffer_occupancy(dut, oep)

    return not error_found


def _log_iact_value_comparison(dut, oep, expected, cells_per_group, half_offset=0):
    """Say whether the buffer holds the right values in the wrong order.

    If the expected and actual byte multisets agree, the write-back produced
    the correct activations and only the placement differs, which points at
    the layout. If they disagree, the values themselves are wrong and the
    layout is not the place to look.
    """
    def to_bytes(word):
        return [(word >> (8 * i)) & 0xFF for i in range(8)]

    actual_words, missing = [], 0
    for index in range(len(expected)):
        word = _ram_word(dut.BUFFER_A[index % oep.IACT_RAM_CELLS]
                         .iact_layer_buffer.impl.mem[half_offset + index // oep.IACT_RAM_CELLS])
        if word is None:
            missing += 1
        else:
            actual_words.append(word)

    exact = sum(1 for e, a in zip(expected, actual_words) if e == a)
    # Compare like with like: only the prefix of the expected image that the
    # DUT actually wrote. Comparing all 392 expected words against 196 written
    # ones can only ever report a difference, whatever the cause.
    comparable = expected[:len(actual_words)]
    expected_bytes = sorted(b for w in comparable for b in to_bytes(w))
    actual_bytes = sorted(b for w in actual_words for b in to_bytes(w))
    same_multiset = expected_bytes == actual_bytes
    logger.error("Iact value comparison: %d/%d words match exactly, %d of %d expected "
                 "words unwritten; over the %d written words the byte multisets %s",
                 exact, len(expected), missing, len(expected), len(actual_words),
                 "MATCH - right values, wrong placement" if same_multiset
                 else "DIFFER - the values themselves are wrong")


def _log_iact_buffer_occupancy(dut, oep, max_addr=64):
    """Report where the iact buffer actually holds data, to localise a mismatch.

    Says whether the comparison looked in the wrong place (data present, other
    cells/addresses) or the buffer was never filled at all (nothing anywhere).
    """
    for name in ("send_data_out", "fsm_current_state", "choose_iact_buffer",
                 "iact_channels_per_pe_next_layer", "fsm_cycle", "skipIact_reg",
                 "max_pooling", "FSM_CEIL_IACT_RTR_CCLS", "FSM_CEIL_WGHT_RTR_CCLS",
                 "FSM_CEIL_PSUM_RTR_CCLS", "CLUSTERS", "NUM_GLB_IACT",
                 "NUM_GLB_WGHT", "NUM_GLB_PSUM", "fsm_psum_cycle",
                 "trans_cycles_psum", "fsm_psum_current_state", "filters",
                 "needed_cycles", "finished_cycles_psum"):
        try:
            logger.error("Iact buffer context: %s = %s", name,
                         int(getattr(dut, name).value))
        except Exception as exc:
            logger.error("Iact buffer context: %s unavailable (%s)", name, type(exc).__name__)

    if fsm_states_seen:
        logger.error("Iact buffer context: main FSM states visited = %s over %d samples "
                     "(RECEIVE_PSUMS_TO_IACT is 11)", sorted(fsm_states_seen), fsm_probe_samples[0])
        logger.error("Iact buffer context: FSM transitions (ns, state) = %s", fsm_state_trace)

    written = []
    for cell in range(oep.IACT_RAM_CELLS):
        for addr in range(max_addr):
            try:
                word = int(dut.BUFFER_A[cell].iact_layer_buffer.impl.mem[addr].value)
            except (ValueError, IndexError):
                continue
            written.append((cell, addr, word))
    if not written:
        logger.error("Iact buffer occupancy: no written word in any of %d cells over the first %d addresses.",
                     oep.IACT_RAM_CELLS, max_addr)
        return
    cells = sorted({c for c, _, _ in written})
    logger.error("Iact buffer occupancy: %d written words, cells %s, first entries %s",
                 len(written), cells[:8],
                 [(c, a, hex(w)) for c, a, w in written[:4]])


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

def _signal_is(handle, expected):
    """True when a signal reads exactly `expected`; False while it holds X/Z."""
    try:
        return int(handle.value) == expected
    except ValueError:
        return False


async def await_ready_signal(ptp, dut, settle_cycles=5000, max_wait_cycles=2000000,
                             report_every=200000):
    """Wait until the DUT has finished the current layer and wants the next one.

    ready_dma_o means "I can accept DMA input", and it is already high through
    GET_PARAMETERS, GET_ROUTER_CONFIG, GET_IACT, GET_WGHT, GET_BIAS and
    GET_QUANTIZE. Sampling it as a level therefore returned immediately, while
    the layer was still loading, and the caller went on to the next layer (and
    to compare_iact_storage) before any computation or psum write-back had
    happened.

    The port does carry the information as an edge: it is deasserted on the way
    out of GET_QUANTIZE and re-asserted only when the FSM comes back round to
    GET_PARAMETERS for the next layer. So wait for the falling edge, then for
    the rising one.

    Args:
        ptp: Port timing parameters
        dut: Device under test (OpenEye_FPGA instance)
        settle_cycles: How long to wait for the DUT to leave its input phase
            before assuming it already has
        max_wait_cycles: Give up (and say where the FSM is stuck) after this
        report_every: Log a progress line at this interval while waiting
    """
    # Phase 1: the DUT leaves its input phase and starts computing.
    for _ in range(settle_cycles):
        if not _signal_is(dut.ready_dma_o, 1):
            break
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    else:
        # Not a benign case: the DUT should drop ready_dma_o when it stops
        # accepting input and starts computing. If it never does, the wait
        # below is satisfied immediately and this call degenerates into the
        # level sample it replaced, so say so rather than passing quietly.
        logger.warning("ready_dma_o stayed high for %d cycles%s; the completion "
                       "wait below cannot be trusted for this layer.",
                       settle_cycles, _fsm_state_note(dut))

    # Phase 2: it comes back ready for the next layer's configuration.
    for cycle in range(max_wait_cycles):
        if _signal_is(dut.ready_dma_o, 1):
            return
        if cycle and cycle % report_every == 0:
            logger.info("Still waiting for the layer to finish after %d cycles%s",
                        cycle, _fsm_state_note(dut))
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

    raise TimeoutError(
        "ready_dma_o never came back high within {} cycles{} - the accelerator "
        "did not finish the layer.".format(max_wait_cycles, _fsm_state_note(dut)))


def _fsm_state_note(dut):
    """' (main FSM in state N)' when readable, else an empty string."""
    try:
        return " (main FSM in state {})".format(int(dut.fsm_current_state.value))
    except Exception:
        return ""


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
                                lower_limit = (x_cluster * oep.Clusters_Y * oep.NUM_GLB_PSUM * self.PSUM_Trans_Bitwidth + y_cluster * oep.NUM_GLB_PSUM * self.PSUM_Trans_Bitwidth + router * self.PSUM_Trans_Bitwidth)
                                upper_limit = lower_limit + oep.DATA_PSUM_BITWIDTH
                                outputvalue = dut.psum_data_o.value[lower_limit:upper_limit]
                                if(logging.DEBUG >= login_level):
                                    txt_file.write(bin(outputvalue)[2:].zfill(self.PSUM_Trans_Bitwidth) + "\n")
                                for i in range(self.PARALLEL_MACS):
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
                                        dram.fmap[layer_number + 1][f][x][y] = int(dut.psum_data_o.value[lower_limit+oep.DATA_PSUM_BITWIDTH*(1-i):upper_limit-oep.DATA_PSUM_BITWIDTH*i])
                                        if (dram.fmap[layer_number + 1][f][x][y] >= 2**(oep.DATA_PSUM_BITWIDTH-1)) :
                                            dram.fmap[layer_number + 1][f][x][y] = dram.fmap[layer_number + 1][f][x][y] - 2**oep.DATA_PSUM_BITWIDTH
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
        """ #Re-Enable Flatted List for outputs again
        cluster_order = []
        for b in range(0,oep.Clusters_Y,layer_parameters.used_Y_cluster):
            for a in range(layer_parameters.used_Y_cluster):
                cluster_order.append(int(oep.Clusters_Y/layer_parameters.used_Y_cluster)*a+int(b/layer_parameters.used_Y_cluster))#+c*(oep.Clusters*oep.PEs_X)

        matrix = [[[i * 16 + j * 8 + k for k in range(8)] for j in range(8)] for i in range(layer_parameters.iact_x_line_repetitions)]
        reordered_matrix = [matrix[i] for i in cluster_order]
        flat_list = [item for row in reordered_matrix for item in row]
        """
        words = (oep.DMA_BITWIDTH//oep.DATA_PSUM_BITWIDTH)
        dut._log.info("Output Stream started")
        used_clusters_per_calc = math.ceil(layer_parameters.iact_size_x / 4) * 4
        values_per_transmission = math.ceil(layer_parameters.different_kernels_per_calculation*used_clusters_per_calc/2)
        transmissions_per_cycle = (oep.Clusters_Y * oep.Clusters_X * oep.PEs_X)//2
        current_cycle = 0
        read_data = 1
        chance = 100
        while (dut.enable_dma_o.value == 1):
            if (read_data):
                if(logging.DEBUG >= login_level):
                    try:
                        txt_file.write(bin(int(dut.data_dma_o.value))[2:].zfill(oep.DMA_BITWIDTH) + "\n")
                    except:
                        txt_file.close()
                        storage_file.close()
                        logger.error("Error writing output txt-file")
                        raise Exception("X detected.")
                if (current_cycle < values_per_transmission):
                    for i in range(words):
                        if(logging.DEBUG >= login_level):
                            storage_file.write("f: " + str(f) + " x: " + str(x) + " y: " + str(y) + "\n")
                        
                        try:
                            dram.fmap[layer_number + 1][f][x][y] = int(dut.data_dma_o.value[oep.DATA_PSUM_BITWIDTH*(i+1)-1:oep.DATA_PSUM_BITWIDTH*i])
                            if (dram.fmap[layer_number + 1][f][x][y] >= 2**(oep.DATA_PSUM_BITWIDTH-1)):
                                dram.fmap[layer_number + 1][f][x][y] = dram.fmap[layer_number + 1][f][x][y] - 2**oep.DATA_PSUM_BITWIDTH
                        except:
                            pass
                        x = x + 1
                    if (current_cycle % math.ceil(oep.PEs_X/words) == math.ceil(oep.PEs_X/words) - 1):
                        if(x >= layer_parameters.iact_size_x):
                            x = 0
                            f = f + 1
                            if(f == layer_parameters.filters):
                                f = 0
                                y = y + 1
                                if(y >= layer_parameters.iact_size_y):
                                    y = 0
                else:
                    for i in range(words):
                        if(logging.DEBUG >= login_level):
                            storage_file.write("Empty storage line." + "\n")
                current_cycle = current_cycle + 1
                if (current_cycle == transmissions_per_cycle):
                    current_cycle = 0

            if random.randint(1, 100) <= chance:
                read_data = 1
                cocotb.start_soon(set_input(ptp,(dut.ready_dma_i), 1))
            else:
                read_data = 0
                cocotb.start_soon(set_input(ptp,(dut.ready_dma_i), 0))
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
    words = (oep.DMA_BITWIDTH//oep.DATA_PSUM_BITWIDTH)
    while (dut.psum_enable_o.value == 0):
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    dut._log.info("Output Stream started")
    assert dut.psum_enable_o.value != 0, "psum is not 1!"
    while (dut.psum_enable_o.value != 0):
        for y_cluster in reversed(range(oep.Clusters_Y)):
            for x_cluster in reversed(range(oep.Clusters_X)):
                for router in reversed(range(oep.NUM_GLB_PSUM)):
                    if(layer_parameters.computing_mx[oep.Clusters_X-x_cluster-1][oep.Clusters_Y-y_cluster-1][0][oep.NUM_GLB_PSUM-router-1]== 1):
                        lower_limit = (x_cluster * oep.Clusters_Y * oep.NUM_GLB_PSUM * self.PSUM_Trans_Bitwidth + y_cluster * oep.NUM_GLB_PSUM * self.PSUM_Trans_Bitwidth + router * self.PSUM_Trans_Bitwidth)
                        upper_limit = lower_limit - 1
                        for _ in range(words - 1):
                            upper_limit = lower_limit + self.PSUM_Trans_Bitwidth - 1
                        outputvalue = dut.psum_data_o.value[lower_limit:upper_limit]

                        if(logging.DEBUG >= login_level):
                            txt_file.write(bin(outputvalue)[2:].zfill(oep.DMA_BITWIDTH) + "\n")
                        for i in range(words):
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
                                dram.fmap[layer_number + 1][f][x][y] = int(dut.psum_data_o.value[lower_limit+oep.DATA_PSUM_BITWIDTH*(1-i):upper_limit-oep.DATA_PSUM_BITWIDTH*i])
                                if (dram.fmap[layer_number + 1][f][x][y] >= 2**(oep.DATA_PSUM_BITWIDTH-1)) :
                                    dram.fmap[layer_number + 1][f][x][y] = dram.fmap[layer_number + 1][f][x][y] - 2**oep.DATA_PSUM_BITWIDTH
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

# Cycles each psum_enable_o bit was asserted, filled by probe_psum_stream and
# reported by dump_psum_buffers so the totals survive however the run ends.
psum_stream_counts = {}


async def probe_psum_stream(ptp, dut, oep):
    """Count how many psum values each router actually emits.

    psum_enable_o is the valid signal on psum_data_o_w, one bit per
    (cluster, GLB). Counting its assertions says whether the PE array streamed
    the expected number of results, which separates "the PE produced too few"
    from "the GLB buffer captured too few".
    """
    previous = 0
    while True:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        try:
            value = int(dut.psum_enable_o.value)
        except ValueError:
            continue
        # Count cycles held high, not rising edges: a burst of consecutive
        # psums keeps the bit asserted, so edges would report one.
        for bit in range(oep.Clusters * oep.NUM_GLB_PSUM):
            if (value >> bit) & 1:
                psum_stream_counts[bit] = psum_stream_counts.get(bit, 0) + 1
        previous = value


async def trace_psum_capture(ptp, dut, oep, max_lines=400):
    """Per-cycle trace of the psum feed and capture (opt-in: TRACE_PSUM_CAPTURE).

    Logs every clock while the PSUM FSM is in CALCULATE_PSUM (2) or
    PSUM_GET_RESULTS (3) for cluster column 0, row 0, GLB 0: the counter that
    ends each state, the buffer address and read data, the word driven into
    the array (psum_data_i_reg) with its enable, the returning valid and the
    buffer write enable. Fields are sliced from the bus text so X bits in
    unrelated slots do not blank them.
    """
    pp = dut.psum_pipeline_inst
    try:
        aw = int(dut.BUFFER_WIDTH_PSUM.value)
    except Exception:
        aw = 14

    def field(sig, lo, width, signed=None):
        txt = str(sig.value)
        bits = txt[len(txt) - lo - width:len(txt) - lo]
        if any(c not in "01" for c in bits):
            return "X"
        v = int(bits, 2)
        return _signed(v & ((1 << signed) - 1), signed) if signed else v

    lines = 0
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        try:
            state = int(pp.fsm_psum_current_state.value)
        except ValueError:
            continue
        if state not in (2, 3):
            continue
        logger.info("psumtrace t=%s st=%d cyc=%s addr=%s rd=%s data_i=%s en_i=%s en_o=%s wen=%s",
                    cocotb.utils.get_sim_time("ns"), state,
                    field(pp.fsm_psum_cycle, 0, 8),
                    field(pp.psum_buffer_addr, 0, aw),
                    field(pp.psum_buffer_data_r, 0, 32, 20),
                    field(pp.psum_data_i_reg, 0, 32, 20),
                    field(pp.psum_enable_i_reg, 0, 1),
                    field(pp.psum_enable_o, 0, 1),
                    field(pp.psum_buffer_en_w, 0, 1))
        lines += 1


async def trace_pe_psum(ptp, dut, oep, max_lines=500):
    """Per-cycle psum SPAD port trace of one PE (opt-in: TRACE_PE_PSUM).

    Watches PE (cluster 0,0; PE column 0; top PE row) at each lane's psum SPAD
    ports, which is where a lane's accumulation actually lands. Logs only
    cycles with a SPAD write or an incoming psum_enable_i.
    """
    pe = (dut.OpenEye_Parallel.gen_x[0].gen_y[0].OpenEye_Cluster
          .pe_cluster.gen_X[0].gen_Y[oep.PEs_Y - 1].pe)
    spads = [pe.gen_serial_psum_spad.gen_psum_spad[l].psum_SPad
             for l in range(oep.PARALLEL_MACS)]

    def val(sig, signed_bits=None):
        try:
            v = int(sig.value)
        except ValueError:
            return "X"
        return _signed(v & ((1 << signed_bits) - 1), signed_bits) if signed_bits else v

    lines = 0
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        we = [val(sp.we_i) for sp in spads]
        pen = val(pe.psum_enable_i)
        if pen != 1 and not any(w == 1 for w in we):
            continue
        lanes = []
        for l, sp in enumerate(spads):
            lanes.append("L%d we=%s aw=%s d=%s ar=%s" % (
                l, we[l], val(sp.addr_w_i), val(sp.data_i, 20), val(sp.addr_r_i)))
        logger.info("petrace t=%s st=%s pen=%s pin=%s | %s",
                    cocotb.utils.get_sim_time("ns"), val(pe.current_state_computing),
                    pen, val(pe.psum_data_i, 20), " | ".join(lanes))
        lines += 1


async def trace_bias_load(ptp, dut, oep, max_lines=120):
    """Trace psum-buffer writes while the main FSM is in GET_BIAS (state 5).

    Opt-in via TRACE_PSUM_CAPTURE. Shows which buffer address and data each
    bias word lands on, for cluster column 0 and 1 (row 0, GLB pair 0).
    Addresses are read from the bus text so X bits elsewhere do not abort.
    """
    pp = dut.psum_pipeline_inst
    ng = oep.NUM_GLB_PSUM
    cr = oep.Clusters_Y
    try:
        aw = int(dut.BUFFER_WIDTH_PSUM.value)
    except Exception:
        aw = 14
    pairs = (ng + 1) // 2

    def field(sig, lo, width, signed=None):
        txt = str(sig.value)
        bits = txt[len(txt) - lo - width:len(txt) - lo]
        if any(c not in "01" for c in bits):
            return "X"
        v = int(bits, 2)
        return _signed(v & ((1 << signed) - 1), signed) if signed else v

    lines = 0
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        try:
            st = int(dut.fsm_current_state.value)
        except ValueError:
            continue
        if st != 5:
            continue
        logger.info("biastrace t=%s dma_in=%s en_w=%s addr[x0,x1]=%s,%s data_w[x0,x1]=%s,%s",
                    cocotb.utils.get_sim_time("ns"),
                    field(pp.data_dma_i_reg, 0, 32, 20),
                    field(pp.psum_buffer_en_w, 0, oep.Clusters_X * cr * pairs),
                    field(pp.psum_buffer_addr, 0, aw),
                    field(pp.psum_buffer_addr, cr * pairs * aw, aw),
                    field(pp.psum_buffer_data_w, 0, 32, 20),
                    field(pp.psum_buffer_data_w, 32 * cr * ng, 32, 20))
        lines += 1


def dump_pe_psum_spads(dut, oep, pe_col=0, max_addr=20):
    """Print the psum SPADs of one PE inside the FPGA design.

    Cheaper and just as decisive as counting psum_enable_o cycles: if the PE
    holds the full set of results then the loss is in the GLB capture or the
    DMA read-out, whereas a short PE SPAD means the compute never produced
    them. Sampling every clock instead costs hours on a full FPGA run.
    """
    # Sweep the PE rows: which one holds the results is not obvious, and only
    # PE column 0 is enabled for Dense layers (layer_parameters disables
    # x_pe != 0), so the column is fixed but the row has to be searched.
    for cl_x in range(oep.Clusters_X):
     for cl_y in range(oep.Clusters_Y):
      for pe_row in range(oep.PEs_Y):
        try:
            pe = (dut.OpenEye_Parallel.gen_x[cl_x].gen_y[cl_y].OpenEye_Cluster
                  .pe_cluster.gen_X[pe_col].gen_Y[pe_row].pe)
        except Exception as exc:
            logger.info("PE [%d][%d][row %d] not reachable (%s)", cl_x, cl_y,
                        pe_row, type(exc).__name__)
            return
        occ = []
        for label, path in (("iact", "iact_data_SPad"), ("wght", "weight_data_SPad")):
            try:
                m = getattr(pe, path).ram.impl.mem
                written = 0
                for addr in range(64):
                    try:
                        int(m[addr].value)
                        written += 1
                    except ValueError:
                        pass
                    except IndexError:
                        break
                occ.append("%s=%d" % (label, written))
            except Exception as exc:
                occ.append("%s=?(%s)" % (label, type(exc).__name__))
        logger.info("PE[x=%d][y=%d][col=%d][row=%d] written operand words (of first 64): %s",
                    cl_x, cl_y, pe_col, pe_row, " ".join(occ))
        if os.environ.get("DUMP_WGHT_WORDS") and cl_y == 0 and pe_row == oep.PEs_Y - 1:
            # Decode each weight word the way PE.v's sparse unpack does:
            # lane p = bits [12p +: 12] = {overhead[3:0], payload[7:0]}. The
            # overhead feeds the psum address offset (wght_data_spad_oh_acc),
            # so a non-zero value in a raw (uncompressed) stream moves lanes.
            for path in ("weight_data_SPad", "gen_wght_addr_spad.weight_addr_SPad"):
                try:
                    m = pe
                    for part in path.split("."):
                        m = getattr(m, part)
                    m = m.ram.impl.mem
                except Exception as exc:
                    logger.info("PE[x=%d] %s not reachable (%s)", cl_x, path, type(exc).__name__)
                    continue
                dec = []
                for addr in range(28):
                    try:
                        w = int(m[addr].value)
                    except ValueError:
                        dec.append("X")
                        continue
                    except IndexError:
                        break
                    if path == "weight_data_SPad":
                        lanes = []
                        for lane in range(oep.PARALLEL_MACS):
                            f = (w >> (12 * lane)) & 0xFFF
                            pay = f & 0xFF
                            pay = pay - 256 if pay & 0x80 else pay
                            lanes.append("%d:%d" % (f >> 8, pay))
                        dec.append("/".join(lanes))
                    else:
                        dec.append(str(w))
                logger.info("PE[x=%d][row=%d] %s = %s", cl_x, pe_row, path, " ".join(dec))
        for lane in range(oep.PARALLEL_MACS):
            try:
                mem = pe.gen_serial_psum_spad.gen_psum_spad[lane].psum_SPad.ram.impl.mem
            except Exception as exc:
                logger.info("PE [%d] lane %d SPAD not reachable (%s)",
                            cl_x, lane, type(exc).__name__)
                continue
            words = []
            for addr in range(max_addr):
                try:
                    v = int(mem[addr].value)
                    words.append(v - (1 << 20) if v >> 19 else v)
                except ValueError:
                    words.append("X")
                except IndexError:
                    break
            if any(w != "X" for w in words):
                logger.info("PE[x=%d][y=%d][col=%d][row=%d] psum_SPad lane %d = %s",
                            cl_x, cl_y, pe_col, pe_row, lane, words)


def dump_psum_buffers(dut, oep, max_addr=None):
    """Print the FPGA psum buffer RAM contents.

    Read after compute, this separates two very different failures: if the
    buffers hold the expected results then only the DMA read-out packing is
    wrong, whereas missing or X entries mean the compute never produced them.

    Each RAM word is PSUM_BUFFER_WIDTH = DATA_PSUM_BITWIDTH*2 wide and packs
    TWO psums (the even/odd GLB of the pair), which is why psum_buffer_en_w
    carries only (NUM_GLB_PSUM+1)/2 enables. Printing the raw word makes a
    full buffer look half empty, so split it here: a buffer with N written
    addresses holds 2*N results, not N.
    """
    if max_addr is None:
        max_addr = int(os.environ.get("DUMP_PSUM_ADDRS", "8"))
    for cc in range(oep.Clusters_X):
        for cr in range(oep.Clusters_Y):
            for g in range((oep.NUM_GLB_PSUM + 1) // 2):
                try:
                    mem = (dut.PSUM_RAM_X[cc].PSUM_RAM_Y[cr]
                           .PSUM_RAM_GLB[g].psum_buffer.impl.mem)
                except Exception as exc:
                    logger.info("psum buffer [%d][%d][%d] not reachable (%s)",
                                cc, cr, g, type(exc).__name__)
                    return
                bw = oep.DATA_PSUM_BITWIDTH
                lo_half, hi_half = [], []
                for addr in range(max_addr):
                    try:
                        word = int(mem[addr].value)
                    except ValueError:
                        lo_half.append("X")
                        hi_half.append("X")
                        continue
                    except IndexError:
                        break
                    lo_half.append(_signed(word & ((1 << bw) - 1), bw))
                    hi_half.append(_signed((word >> bw) & ((1 << bw) - 1), bw))
                logger.info("psum_buffer[x=%d][y=%d][glb=%d] even=%s", cc, cr, g,
                            lo_half)
                logger.info("psum_buffer[x=%d][y=%d][glb=%d] odd =%s", cc, cr, g,
                            hi_half)


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
    dump_dma_words = [] if os.environ.get("DUMP_PSUM_BUFFERS") else None
    cluster_offset = math.ceil(layer_parameters.used_psum_per_PE)
    while (dut.enable_dma_o.value == 1):

        if(logging.DEBUG >= login_level):
            try:
                txt_file.write(bin(int(dut.data_dma_o.value))[2:].zfill(oep.PSUM_Trans_Bitwidth) + "\n")
            except ValueError:
                # data_dma_o can read X here (not just a transient edge
                # case - some psum_pipeline.v configs leave portions of
                # the output genuinely unwritten). The functional capture
                # below already tolerates this via its own try/except;
                # mirror that here instead of crashing the whole test on
                # the first X sample.
                txt_file.write(str(dut.data_dma_o.value) + "\n")
        if(logging.DEBUG >= login_level):
            storage_file.write("f: " + str(f) + "\n")
        if dump_dma_words is not None:
            # The Dense reader below takes only bits [DATA_PSUM_BITWIDTH-1:0]
            # of each DMA word, while the conv reader takes DMA_BITWIDTH //
            # DATA_PSUM_BITWIDTH psums per word. Capture the raw word so the
            # high half can be inspected: if it is non-zero the Dense reader
            # is silently discarding half of every transfer.
            try:
                dump_dma_words.append(int(dut.data_dma_o.value))
            except ValueError:
                dump_dma_words.append(None)
        try:
            dram.fmap[layer_number + 1][f] = int(dut.data_dma_o.value[oep.DATA_PSUM_BITWIDTH-1:0])
            if (dram.fmap[layer_number + 1][f] >= 2**(oep.DATA_PSUM_BITWIDTH-1)) :
                dram.fmap[layer_number + 1][f] = dram.fmap[layer_number + 1][f] - 2**oep.DATA_PSUM_BITWIDTH
        except:
            pass
        if (f < cluster_offset) :
            f = f + cluster_offset
        else :
            f = f - cluster_offset + 1
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

    if dump_dma_words is not None:
        # Careful with the two meanings of DATA_PSUM_BITWIDTH: the HDL
        # parameter is the 32-bit *lane* the psum is packed into, while
        # oep.DATA_PSUM_BITWIDTH is the 20-bit signed accumulator. So step
        # across the word in lanes but sign-extend at the accumulator width,
        # otherwise every negative psum prints as a large positive.
        acc_bits = oep.DATA_PSUM_BITWIDTH
        lane_bits = 32
        lanes = oep.DMA_BITWIDTH // lane_bits
        logger.info("captured %d DMA words on the Dense output stream",
                    len(dump_dma_words))
        for idx, word in enumerate(dump_dma_words):
            if word is None:
                logger.info("dma word %2d = X", idx)
                continue
            vals = [_signed((word >> (lane_bits * i)) & ((1 << acc_bits) - 1),
                            acc_bits) for i in range(lanes)]
            logger.info("dma word %2d = %s", idx, vals)

    if os.environ.get("DUMP_PSUM_BUFFERS"):
        report_iact_handoff(oep)
        dump_iact_path(dut, oep)
        dump_pe_occupancy(dut, oep)
        dump_pe_psum_spads(dut, oep)
        dump_psum_buffers(dut, oep)
        if psum_stream_counts:
            logger.info("psum_enable_o asserted cycles per bit: %s",
                        {k: v for k, v in sorted(psum_stream_counts.items())})

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
