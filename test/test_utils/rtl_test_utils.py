# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import os
import logging
import math
import cocotb
from cocotb.triggers import Timer
import test_utils.stream_dicts as strdic

logger = logging.getLogger("cocotb")


async def set_input(port_timings, signal, new_value, multiple_dim = False, array_index = [], array_max_index = []):
    # input delay of 100 ps relative to rising edge
    # this is the value used for the implementation constraints of OpenEye
    await Timer(port_timings.clk_delay_in, units=port_timings.clk_delay_unit_in)
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
    """ Reset all signals of the DUT.

    This function resets all signals of the DUT. It is called by the testbench.
    """
    cocotb.start_soon(set_input(ptp,(dut.rst_ni), 0))
    if(serial == 0):
        cocotb.start_soon(set_input(ptp,(dut.compute_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.wght_data_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.wght_enable_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.iact_data_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.iact_enable_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.psum_data_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.psum_ready_i), 0))
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
        cocotb.start_soon(set_input(ptp,(dut.bano_cluster_mode_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.af_cluster_mode_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.pooling_cluster_mode_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.input_activations_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.iact_write_addr_t_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.iact_write_data_t_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.stride_x_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.stride_y_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.compute_mask_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.router_mode_iact_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.router_mode_wght_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.router_mode_psum_i), 0))
    else:
        cocotb.start_soon(set_input(ptp,(dut.data_dma_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.enable_dma_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.ready_dma_i), 0))

    for _ in range(3):
        await Timer(ptp.clk_cycle, ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp,(dut.rst_ni), 1))

    # After deasserting reset, we wait 4 clock cycles
    for _ in range(4):
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
        # Set the input signals
        cocotb.start_soon(set_input(ptp,(dut.status_reg_enable_i), 1))
        cocotb.start_soon(set_input(ptp,(dut.data_mode_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["data_mode"]]))
        cocotb.start_soon(set_input(ptp,(dut.fraction_bit_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["realfactor"]]))
        cocotb.start_soon(set_input(ptp,(dut.needed_cycles_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["needed_refreshes"]]))
        cocotb.start_soon(set_input(ptp,(dut.needed_x_cls_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_X_cluster"]]))
        cocotb.start_soon(set_input(ptp,(dut.needed_y_cls_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_Y_cluster"]]))
        cocotb.start_soon(set_input(ptp,(dut.needed_iact_cycles_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["needed_Iact_writes"]]))
        cocotb.start_soon(set_input(ptp,(dut.filters_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_psum_per_PE"]]))
        cocotb.start_soon(set_input(ptp,(dut.iact_addr_len_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_iact_addr_per_PE"]]))
        cocotb.start_soon(set_input(ptp,(dut.wght_addr_len_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_wght_addr_per_PE"]]))
        cocotb.start_soon(set_input(ptp,(dut.bano_cluster_mode_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.af_cluster_mode_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["autofunction"]]))
        cocotb.start_soon(set_input(ptp,(dut.pooling_cluster_mode_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["poolingmode"]]))
        cocotb.start_soon(set_input(ptp,(dut.delay_psum_glb_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["psum_delay"]]))
        cocotb.start_soon(set_input(ptp,(dut.input_activations_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["used_iact_per_PE"]]))
        cocotb.start_soon(set_input(ptp,(dut.iact_write_addr_t_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["iact_addr_len"]]))
        cocotb.start_soon(set_input(ptp,(dut.iact_write_data_t_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["iact_data_len"]]))
        cocotb.start_soon(set_input(ptp,(dut.stride_x_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["strideX"]]))
        cocotb.start_soon(set_input(ptp,(dut.stride_y_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["strideY"]]))
        cocotb.start_soon(set_input(ptp,(dut.compute_mask_i), stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["usePEs"]]))
        
        # Set the router mode for the input activations
        router_mode_port = 0
        for cl_x in range(oep.Clusters_X):
            for cl_y in range(oep.Clusters_Y):
                for router in range(oep.NUM_GLB_IACT):
                    router_mode_port = router_mode_port + (stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["router_iact"]][cl_x][cl_y][router] << \
                                                        (oep.Iact_Router_Bits * router + \
                                                            oep.Iact_Router_Bits * oep.NUM_GLB_IACT * cl_y + \
                                                            oep.Iact_Router_Bits * oep.NUM_GLB_IACT * oep.Clusters_Y * cl_x))
        cocotb.start_soon(set_input(ptp,(dut.router_mode_iact_i), router_mode_port))
        
        # Set the router mode for the weights
        router_mode_port = 0
        for cl_x in range(oep.Clusters_X):
            for cl_y in range(oep.Clusters_Y):
                for router in range(oep.NUM_GLB_WGHT):
                    router_mode_port = router_mode_port + (stream[strdic.stream_parallel_dict["status"]][strdic.status_dict["router_wght"]][cl_x][cl_y][router] << \
                                                        (oep.Wght_Router_Bits * router + \
                                                            oep.Wght_Router_Bits * oep.NUM_GLB_WGHT * cl_y + \
                                                            oep.Wght_Router_Bits * oep.NUM_GLB_WGHT * oep.Clusters_Y * cl_x))
        cocotb.start_soon(set_input(ptp,(dut.router_mode_wght_i), router_mode_port))
        
        # Set the router mode for the partial sums
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
        # Wait until the status register and the router mode are set
        await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp,(dut.status_reg_enable_i), 0))
    else:
        cocotb.start_soon(set_input(ptp,(dut.enable_dma_i), 1))
        for data_word in range(len(stream[strdic.stream_parallel_dict["status"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["status"]][data_word]))
            await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
        for data_word in range(len(stream[strdic.stream_parallel_dict["iact"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["iact"]][data_word]))
            await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
        for data_word in range(len(stream[strdic.stream_parallel_dict["wght"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["wght"]][data_word]))
            await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
        for data_word in range(len(stream[strdic.stream_parallel_dict["psum"]])):
            cocotb.start_soon(set_input(ptp,(dut.data_dma_i), stream[strdic.stream_parallel_dict["psum"]][data_word]))
            await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
        await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp,(dut.enable_dma_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.ready_dma_i), 1))
        
async def write_iact(ptp, dut, stream, oep, lp):
    """ Write the input activations to the DUT.
    
    This function writes the input activations to the DUT. It is called by the testbench.
    
    Args:
        dut: The DUT. 
        stream: The stream that is sent to the DUT.
        layer_repetition: The index of the part of a layer, if it is too large to be processed at once.
        oep: The OpenEye parameters.
        lp: The layer parameters.
    """
    iact_enable_signal = 0
    iact_transmission = 0
    if(lp.skipIact != 1):
        for position in range(len(stream[0][0][0])):
            iact_enable_signal = 0
            for x_cluster in range(oep.Clusters_X):
                for y_cluster in range(oep.Clusters_Y):
                    for router in range(oep.NUM_GLB_IACT):
                        try:
                            iact_transmission = iact_transmission + \
                            (stream[x_cluster][y_cluster][router][position] \
                            << ((router + y_cluster * oep.NUM_GLB_IACT + x_cluster * oep.NUM_GLB_WGHT * oep.Clusters_Y) * int(oep.DMA_Bits/oep.Clusters_X)))
                            iact_enable_signal = iact_enable_signal + 2**(router + y_cluster * oep.NUM_GLB_IACT+ x_cluster * oep.NUM_GLB_IACT * oep.Clusters_Y)
                        except:
                            iact_enable_signal = iact_enable_signal
            cocotb.start_soon(set_input(ptp,(dut.iact_data_i), iact_transmission))
            iact_transmission = 0
            cocotb.start_soon(set_input(ptp,(dut.iact_enable_i), iact_enable_signal))
            await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp,(dut.iact_data_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.iact_enable_i), 0))

async def write_wght(ptp, dut, stream, oep, lp):
    """ Write the weights to the DUT.
    
    This function writes the weights to the DUT. It is called by the testbench.
    
    Args:
        dut: The DUT.
        stream: The stream that is sent to the DUT.
        layer_repetition: The index of the part of a layer, if it is too large to be processed at once.
        oep: The OpenEye parameters.
        lp: The layer parameters.
    """
    wght_enable_signal = 0
    wght_transmission = 0
    if(lp.skipWght != 1):
        cocotb.start_soon(set_input(ptp,(dut.wght_enable_i), (2**(oep.Clusters_X*oep.Clusters_Y*oep.NUM_GLB_WGHT))-1))
        for position in range(len(stream[0][0][0])):
            wght_enable_signal = 0
            for x_cluster in range(oep.Clusters_X):
                for y_cluster in range(oep.Clusters_Y):
                    for router in range(oep.NUM_GLB_WGHT):
                        try:
                            wght_transmission = wght_transmission + \
                            (stream[x_cluster][y_cluster][router][position] \
                            << ((router + y_cluster * oep.NUM_GLB_WGHT + x_cluster * oep.NUM_GLB_WGHT * oep.Clusters_Y) * int(oep.DMA_Bits/oep.Clusters_X)))
                            wght_enable_signal = wght_enable_signal + 2**(router + y_cluster * oep.NUM_GLB_WGHT+ x_cluster * oep.NUM_GLB_WGHT * oep.Clusters_Y)
                        except:
                            wght_enable_signal = wght_enable_signal
            cocotb.start_soon(set_input(ptp,(dut.wght_data_i), wght_transmission))
            wght_transmission = 0
            cocotb.start_soon(set_input(ptp,(dut.wght_enable_i), wght_enable_signal))
            await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp,(dut.wght_data_i), 0))
        cocotb.start_soon(set_input(ptp,(dut.wght_enable_i), 0))

async def write_bias(ptp, dut, stream, oep, lp):
    """ Write the bias to the DUT.

    This function writes the bias to the DUT. It is called by the testbench.
    The bias is written to the partial sum GLBs. Therefore, the partial sum
    ports are used here.

    Args:
        dut: The DUT.
        stream: The stream that is sent to the DUT.
        layer_repetition: The index of the part of a layer, if it is too large to be processed at once.
        oep: The OpenEye parameters.
        lp: The layer parameters.

    """
    psum_transmission = 0
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), (2**(oep.Clusters_X*oep.Clusters_Y*oep.NUM_GLB_PSUM))-1))
    for position in range(len(stream[0][0][0])):
        for x_cluster in range(oep.Clusters_X):
            for y_cluster in range(oep.Clusters_Y):
                for router in range(oep.NUM_GLB_PSUM):
                    psum_transmission = psum_transmission + \
                    (stream[x_cluster][y_cluster][router][position] \
                    << ((router + y_cluster * oep.NUM_GLB_PSUM + x_cluster * oep.NUM_GLB_PSUM * oep.Clusters_Y) * oep.DMA_Bits))
        cocotb.start_soon(set_input(ptp,(dut.psum_data_i), psum_transmission))
        psum_transmission = 0
        await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp,(dut.psum_data_i), 0))
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))

async def await_ready_signal(ptp, dut, layer_number, model, layer_repetition, layer_parameters, oep, les, dram, login_level, stream):
    if (oep.SERIAL == 0):
        while (dut.psum_ready_o.value != (2**(oep.Clusters_X*oep.Clusters_Y*oep.NUM_GLB_PSUM))-1):
            await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    else:
        cocotb.start_soon(set_input(ptp,(dut.ready_dma_i), 1))
        while (dut.enable_dma_o.value != 1):
            await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    pass

async def compare_stream_Conv(ptp, dut, layer_number, model, layer_repetition, layer_parameters, oep, les, dram, login_level, stream):
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
        dram: The storage used.
        login_level: What kind of logs should be outputed
    """

    if(logging.DEBUG >= login_level):
        filename = 'demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/output.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        txt_file = open(filename, 'w')
        filename = 'demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/storage_input.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        storage_file = open(filename, 'w')
    if(layer_repetition == 0):
        les.y_corner_start = 0
        les.x_corner_start = 0

    les.f_start = int((math.floor(layer_repetition/layer_parameters.iact_transmissions_pe)%layer_parameters.needed_wght_transmissions) * math.ceil(layer_parameters.used_wght_per_PE/layer_parameters.used_iact_per_PE))
    les.f_corner_start = int((math.floor(layer_repetition/layer_parameters.iact_transmissions_pe)%layer_parameters.needed_wght_transmissions) * math.ceil(layer_parameters.used_wght_per_PE/layer_parameters.used_iact_per_PE))
    les.f_corner_end = layer_parameters.filters
    les.f_end = les.f_start + math.ceil(layer_parameters.used_wght_per_PE/layer_parameters.used_iact_per_PE)
    les.y_start = les.y_corner_start
    les.x_start = les.x_corner_start
    les.x_end = int(model.layers[layer_number].output.shape[1])
    les.y_end = int(model.layers[layer_number].output.shape[2])
    les.f_corner_end = int(model.layers[layer_number].output.shape[1])
    les.y_corner_end = int(model.layers[layer_number].output.shape[2])
    logger.debug("PRE")
    logger.debug("f: " + str(les.f_start) + " x: " + str(les.x_start) + " y: " + str(les.y_start) + " f_corner_start: " + str(les.f_corner_start) + " y_corner_start: " + str(les.y_corner_start) + " x_corner_start: " + str(les.x_corner_start) + "\n")

    f = les.f_start
    x = les.x_start
    y = les.y_start
    if(logging.DEBUG >= login_level):
        storage_file.write(" f_corner_start: " + str(les.f_corner_start) + " y_corner_start: " + str(les.y_corner_start) + " x_corner_start: " + str(les.x_corner_start) + "\n")
    if (oep.SERIAL == 0) :
        if ((layer_repetition % layer_parameters.iact_transmissions_pe) == (layer_parameters.iact_transmissions_pe - 1)):
            cocotb.start_soon(send_enable_conv(ptp, dut, layer_parameters, layer_repetition, oep))
            while (dut.psum_enable_o.value == 0):
                await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
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
                                    if(logging.DEBUG >= login_level):
                                        storage_file.write("f: " + str(f) + " x: " + str(x) + " y: " + str(y) + "\n")
                                    try:
                                        dram.fmap[layer_number + 1][f][x][y] = int(dut.psum_data_o.value[lower_limit+20*(1-i):upper_limit-20*i])
                                        if (dram.fmap[layer_number + 1][f][x][y] >= 2**19) :
                                            dram.fmap[layer_number + 1][f][x][y] = dram.fmap[layer_number + 1][f][x][y] - 2**20
                                    except:
                                        pass
                                    f = f + 1
                                if((y_cluster == 0) and (x_cluster == 0) and (router == layer_parameters.add_up)):
                                    if(f >= les.f_end):
                                        les.f_start = les.f_corner_start
                                        f = les.f_start
                                        if(x == les.x_end - 1):
                                            x = 0
                                            if(y >= les.y_end - 1):
                                                y = 0
                                            else:
                                                y = y + 1
                                        else:
                                            x = x + 1
                                        les.y_start = y
                                        les.x_start = x
                                    else:
                                        les.f_start = f
                                        x = les.x_start
                                        y = les.y_start
                                else:
                                    f = f - 2
                                    if(x == les.x_end - 1):
                                        x = 0
                                        y = y + 1
                                    else:
                                        x = x + 1
                await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
        await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))
        dut._log.info("Output Stream finished")
        await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp,(dut.status_reg_enable_i), 1))
        await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    else:
        dut._log.info("Output Stream started")
        while (dut.enable_dma_o.value == 1):

            if ((layer_repetition % layer_parameters.iact_transmissions_pe) == (layer_parameters.iact_transmissions_pe - 1)) :
                if(logging.DEBUG >= login_level):
                    txt_file.write(bin(int(dut.data_dma_o.value))[2:].zfill(40) + "\n")
                for i in range(2):

                    if(logging.DEBUG >= login_level):
                        storage_file.write("f: " + str(f) + " x: " + str(x) + " y: " + str(y) + "\n")
                    try:
                        dram.fmap[layer_number + 1][f][x][y] = int(dut.data_dma_o.value[28-20*i:47-20*i])
                        if (dram.fmap[layer_number + 1][f][x][y] >= 2**19) :
                            dram.fmap[layer_number + 1][f][x][y] = dram.fmap[layer_number + 1][f][x][y] - 2**20
                    except:
                        pass
                    f = f + 1
                if(f >= les.f_end):
                    les.f_start = les.f_corner_start
                    f = les.f_start
                    if(x == les.x_end - 1):
                        x = 0
                        if(y >= les.y_end - 1):
                            y = 0
                        else:
                            y = y + 1
                    else:
                        x = x + 1
                    les.y_start = y
                    les.x_start = x
                else:
                    les.f_start = f
                    x = les.x_start
                    y = les.y_start
            await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)

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

async def compare_stream_Dw(ptp, dut, layer_number, model, layer_repetition, layer_parameters, oep, les, dram, login_level, stream):
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

    les.f_start = int((math.floor(layer_repetition%layer_parameters.iact_transmissions_pe)) * layer_parameters.filters)
    les.f_corner_start = int((math.floor(layer_repetition)%layer_parameters.needed_wght_transmissions) * (layer_parameters.filters/layer_parameters.needed_wght_transmissions))
    les.f_corner_end = layer_parameters.filters
    les.f_end = les.f_start + int(layer_parameters.filters/layer_parameters.needed_wght_transmissions)
    les.y_start = les.y_corner_start
    les.x_start = les.x_corner_start
    les.x_end = int(model.layers[layer_number].output.shape[1])
    les.y_end = int(model.layers[layer_number].output.shape[2])
    les.f_corner_end = int(model.layers[layer_number].output.shape[1])
    les.y_corner_end = int(model.layers[layer_number].output.shape[2])
    logger.debug("PRE")
    logger.debug("f: " + str(les.f_start) + " x: " + str(les.x_start) + " y: " + str(les.y_start) + " f_corner_start: " + str(les.f_corner_start) + " y_corner_start: " + str(les.y_corner_start) + " x_corner_start: " + str(les.x_corner_start) + "\n")

    f = les.f_start
    x = les.x_start
    y = les.y_start
    if(logging.DEBUG >= login_level):
        storage_file.write(" f_corner_start: " + str(les.f_corner_start) + " y_corner_start: " + str(les.y_corner_start) + " x_corner_start: " + str(les.x_corner_start) + "\n")
    cocotb.start_soon(send_enable_conv(ptp, dut, layer_parameters, layer_repetition, oep))
    while (dut.psum_enable_o.value == 0):
        await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
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
                        for i in range(1):

                            if(logging.DEBUG >= login_level):
                                storage_file.write("f: " + str(f) + " x: " + str(x) + " y: " + str(y) + "\n")
                            try:
                                dram.fmap[layer_number + 1][f][x][y] = int(dut.psum_data_o.value[lower_limit+20*(1-i):upper_limit-20*i])
                                if (dram.fmap[layer_number + 1][f][x][y] >= 2**19) :
                                    dram.fmap[layer_number + 1][f][x][y] = dram.fmap[layer_number + 1][f][x][y] - 2**20
                            except:
                                pass

                            x = x + 1
                        if((y_cluster == 0) and (x_cluster == 0) and (router == layer_parameters.add_up)):
                            if(x >= les.x_end):
                                x = 0
                                y = y + 1
                        else:
                            if(x >= les.x_end):
                                x = 0
                                y = y + 1
        await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))
    dut._log.info("Output Stream finished")
    await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp,(dut.status_reg_enable_i), 1))
    await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    
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

async def compare_stream_Dense(ptp, dut, layer_number, model, layer_repetition, layer_parameters, oep, les, dram, login_level, stream):
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

    values_per_trans = 2

    offset_layer_repetition = (math.floor(layer_repetition/layer_parameters.iact_transmissions_pe) % layer_parameters.psum_transmissions_pe) * oep.Clusters_Y * oep.Clusters_X * layer_parameters.used_psum_per_PE
    les.x = offset_layer_repetition
    router = 3
    offset = 0

    if ((layer_repetition % layer_parameters.iact_transmissions_pe) == (layer_parameters.iact_transmissions_pe - 1)) :
        cocotb.start_soon(send_enable_dense(ptp, dut, layer_parameters, layer_repetition, oep))
        while (dut.psum_enable_o.value == 0):
            await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
        dut._log.info("Output Stream started")
        assert dut.psum_enable_o.value != 0, "psum is not 1!"
        while (dut.psum_enable_o.value != 0):
            for y_cluster in reversed(range(oep.Clusters_Y)):
                for x_cluster in reversed(range(oep.Clusters_X)):
                    lower_limit = (x_cluster * oep.Clusters_Y * oep.NUM_GLB_PSUM * 40 + y_cluster * oep.NUM_GLB_PSUM * 40 + router * 40)
                    upper_limit = lower_limit + 39
                    outputvalue = dut.psum_data_o.value[lower_limit:upper_limit]
                    if(logging.DEBUG >= login_level):
                        txt_file.write(bin(outputvalue)[2:].zfill(40) + "\n")
                    x = offset + offset_layer_repetition + \
                    (oep.Clusters_Y-1-y_cluster) * oep.Clusters_X * layer_parameters.used_psum_per_PE+ \
                    (oep.Clusters_X-1-x_cluster) * layer_parameters.used_psum_per_PE
                    for i in range(values_per_trans):

                        if(logging.DEBUG >= login_level):
                            storage_file.write("x: " + str(x) + "\n")
                        try:
                            dram.fmap[layer_number + 1][x] = int(dut.psum_data_o.value[lower_limit+20*(1-i):upper_limit-20*i])
                            if (dram.fmap[layer_number + 1][x] >= 2**19) :
                                dram.fmap[layer_number + 1][x] = dram.fmap[layer_number + 1][x] - 2**20
                        except:
                            #storage_file.close()
                            #assert dut.rst_ni.value == 0, "Output is not in range of memory"
                            logger.debug("Empty File")

                        x = x + 1
                    

            offset = offset + values_per_trans
            await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))

    await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp,(dut.status_reg_enable_i), 1))
    await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    if(logging.DEBUG >= login_level):
        storage_file.close()
        txt_file.close()
        dut._log.info("Output Stream finished")
    
    pass

async def send_enable_conv(ptp, dut, layer_params, layer_repetition, oep):

    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), (2**(oep.Clusters_X*oep.Clusters_Y*oep.NUM_GLB_PSUM))-1))
    for _ in range(int((math.ceil(layer_params.filters/layer_params.needed_wght_transmissions/2)*\
                        math.ceil(layer_params.needed_refreshes_mx[layer_repetition][0]/layer_params.used_Y_cluster)))):
        await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))

async def send_enable_dense(ptp, dut, layer_params, layer_repetition, oep):
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), (2**(oep.Clusters_X*oep.Clusters_Y*oep.NUM_GLB_PSUM))-1))
    for _ in range(math.ceil(layer_params.used_psum_per_PE/2)):
        await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp,(dut.psum_enable_i), 0))
