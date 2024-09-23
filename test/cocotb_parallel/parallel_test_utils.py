# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import sys
import os
directory = (os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
import logging
import math
from test_utils.dense_mapper import DenseMapper
from test_utils.conv_mapper import ConvMapper
from test_utils.dw_mapper import DWMapper
import test_utils.generic_test_utils as gtu
import test_utils.stream_dicts as strdic
import multiprocessing as mp

logger = logging.getLogger("cocotb")

def get_verilog_sources(hdl_dir, serial):

    verilog_sources =[
    os.path.join(hdl_dir, "OpenEye_Parallel.v"),
    os.path.join(hdl_dir, "OpenEye_Cluster.v"),
    os.path.join(hdl_dir, "GLB_cluster.v"),
    os.path.join(hdl_dir, "af_cluster.v"),
    os.path.join(hdl_dir, "bano_cluster.v"),
    os.path.join(hdl_dir, "delay_cluster.v"),
    os.path.join(hdl_dir, "router_iact.v"),
    os.path.join(hdl_dir, "router_wght.v"),
    os.path.join(hdl_dir, "router_psum.v"),
    os.path.join(hdl_dir, "PE_cluster.v"),
    os.path.join(hdl_dir, "PE.v"),
    os.path.join(hdl_dir, "adder.v"),
    os.path.join(hdl_dir, "data_pipeline.v"),
    os.path.join(hdl_dir, "multiplier.v"),
    os.path.join(hdl_dir, "mux2.v"),
    os.path.join(hdl_dir, "demux2.v"),
    os.path.join(hdl_dir, "mux_iact.v"),
    os.path.join(hdl_dir, "SPad_DP_RW.v"),
    os.path.join(hdl_dir, "SPad_SP.v"),
    os.path.join(hdl_dir, "RST_SYNC.v"),
    os.path.join(hdl_dir, "memory/RAM_DP_RW.v"),
    os.path.join(hdl_dir, "memory/RAM_DP.v"),
    os.path.join(hdl_dir, "memory/RAM_SP.v"),
    os.path.join(hdl_dir, "memory/impl/RAM_DP_RW_generic.v"),
    os.path.join(hdl_dir, "memory/impl/RAM_SP_generic.v")
    ]
    if (serial):
        verilog_sources.append(os.path.join(hdl_dir, "OpenEye_Wrapper.v"))
    return verilog_sources

def write_stream_layer_mp(params, layer_params, layer, dram_layer_content, return_dict, layer_repetition):
    if "Depthwise" in str(layer):
        LayerStreamGenerator = DWMapper(params, layer_params, layer_repetition, dram_layer_content)
        LayerStreamGenerator.make_stream()
    elif "Conv" in str(layer):
        LayerStreamGenerator = ConvMapper(params, layer_params, layer_repetition, dram_layer_content)
        LayerStreamGenerator.make_stream()
    elif "Dense" in str(layer):
        LayerStreamGenerator = DenseMapper(params, layer_params, layer_repetition, dram_layer_content)
        LayerStreamGenerator.make_stream()
    return_dict[layer_repetition] = LayerStreamGenerator.get_stream()

def write_stream(params, layer_params, layer, dram_layer_content):
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []

    for layer_repetition in range(layer_params.needed_total_transmissions):
        p = mp.Process(target = write_stream_layer_mp, args = (params, layer_params, layer, dram_layer_content, return_dict, layer_repetition))
        p.start()
        jobs.append(p)

    for proc in range(len(jobs)):
        jobs[proc].join()
    #assert False
    return return_dict

#Reference

def make_ref(params, layer_params, layer, layer_number, dram, calculated_results):
    

    #Write wght File
    write_weight_file(layer, layer_number, dram)
    logger.info("All weight-files written")

    #Write iact File
    write_iact_file(layer, layer_number, dram)
    logger.info("All iact-files written")

    logger.info("All results calculated")

    #Write psum File
    write_psum_file(layer, layer_number, dram, calculated_results)
    logger.info("All psum-files written")
    
    dma_line = 0
    
    if "Depthwise" in str(layer):
        if(params.SERIAL):
            file_dma_ref = [0 for layer_repetition in range(layer_params.needed_total_transmissions)]
            for layer_repetition in range(layer_params.needed_total_transmissions):
                file_dma_ref[layer_repetition] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_ref.txt')
                for refresh in range(math.floor((math.floor((layer_repetition%layer_params.iact_transmissions_pe))/layer_params.needed_total_transmissions) * layer_params.Used_refreshes),
                                    math.floor(((math.floor((layer_repetition%layer_params.iact_transmissions_pe))+1)/layer_params.needed_total_transmissions) * layer_params.Used_refreshes)):
                    for cl_y in range(params.Clusters_Y):
                        for cl_x in range(params.Clusters_X):
                            for router in range(params.Psum_Routers):
                                for psum_pe in range(int((layer.filters*(layer_repetition%layer_params.needed_wght_transmissions)/layer_params.needed_wght_transmissions)/2),\
                                    int((layer.filters*(1+(layer_repetition%layer_params.needed_wght_transmissions))/layer_params.needed_wght_transmissions)/2)):
                                    for counter in range(math.floor(params.DMA_Bits/params.PSUM_Bitwidth)):
                                        x_cor= int(((router + cl_x * params.PEs_X + cl_y * params.Clusters_X * params.PEs_X + refresh * params.Clusters_Y * params.Clusters_X * params.PEs_X ) % layer.output.shape[2]))
                                        y_cor= int(((router + cl_x * params.PEs_X + cl_y * params.Clusters_X * params.PEs_X + refresh * params.Clusters_Y * params.Clusters_X * params.PEs_X ) / layer.output.shape[2]))
                                        if((x_cor < layer.output.shape[1]) & (y_cor < layer.output.shape[2])):
                                            if(calculated_results[2 * psum_pe + counter][x_cor][y_cor] >= 0):
                                                dma_line = dma_line + (calculated_results[2 * psum_pe + counter][x_cor][y_cor] << (params.PSUM_Bitwidth * counter))
                                            else:
                                                dma_line = dma_line

                                    file_dma_ref[layer_repetition].write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
                                    dma_line = 0
                file_dma_ref[layer_repetition].close()
        else:
            cluster_order = []
            for a in range(layer_params.used_Y_cluster):
                for b in range(0,params.Clusters_Y,layer_params.used_Y_cluster):
                    cluster_order.append(a+b)

            manager = mp.Manager()
            jobs = []

            for layer_repetition in range(layer_params.needed_total_transmissions):
                p = mp.Process(target = calculate_dw_output_stream_mp, \
                               args = (layer_repetition, layer_number, params, layer_params, layer, cluster_order, calculated_results))
                p.start()
                jobs.append(p)
            
            for proc in range(len(jobs)):
                jobs[proc].join()
    elif "Conv" in str(layer):
        cluster_order = []
        for a in range(layer_params.used_Y_cluster):
            for b in range(0,params.Clusters_Y,layer_params.used_Y_cluster):
                cluster_order.append(a+b)

        manager = mp.Manager()
        jobs = []

        for layer_repetition in range(layer_params.needed_total_transmissions):
            p = mp.Process(target = calculate_conv_output_stream_mp, \
                            args = (layer_repetition, layer_number, params, layer_params, layer, cluster_order, calculated_results))
            p.start()
            jobs.append(p)
        
        for proc in range(len(jobs)):
            jobs[proc].join()
    elif "Dense" in str(layer):
        if(params.SERIAL):
            assert False, "not realized yet"
        else:
            file_dma_ref = [0 for layer_repetition in range(layer_params.needed_total_transmissions)]
            for layer_repetition in range(layer_params.needed_total_transmissions):
                file_dma_ref[layer_repetition] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_ref.txt')
                for psum_pe in range(math.ceil(layer_params.used_psum_per_PE/2)):
                    for cl_y in range(params.Clusters_Y):
                        for cl_x in range(params.Clusters_X):
                            partial_result_a = gtu.to_twos_complement_string(0,20)
                            partial_result_b = gtu.to_twos_complement_string(0,20)
                            for counter in range(math.floor(params.DMA_Bits/params.PSUM_Bitwidth)):
                                layer_repetition_cycle = math.floor(layer_repetition/layer_params.iact_transmissions_pe)
                                output = \
                                    counter + \
                                    2 * psum_pe +\
                                    cl_x * layer_params.used_psum_per_PE + \
                                    cl_y * layer_params.used_psum_per_PE * params.Clusters_X + \
                                    layer_repetition_cycle * layer_params.used_psum_per_PE * params.Clusters_X * params.Clusters_Y
                                if (counter == 0):
                                    try:
                                        partial_result_b = gtu.to_twos_complement_string(calculated_results[output],20)
                                    except:
                                        partial_result_b = gtu.to_twos_complement_string(0,20)
                                else:
                                    try:
                                        partial_result_a = gtu.to_twos_complement_string(calculated_results[output],20)
                                    except:
                                        partial_result_a = gtu.to_twos_complement_string(0,20)

                            file_dma_ref[layer_repetition].write(partial_result_a)
                            file_dma_ref[layer_repetition].write(partial_result_b)
                            file_dma_ref[layer_repetition].write("\n")
                            dma_line = 0
                file_dma_ref[layer_repetition].close()
    
    logger.info("Reference Output calculated.")

def write_weight_file(layer, layer_number, dram):

    if "Dense" in str(layer):
        wght_ref = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/weight/wght_ref' + '_0.csv')
        for c in range(layer.input.shape[1]):
            for x in range(layer.output.shape[1]):
                wght_ref.write(str(int(dram.weights[layer_number][x][c])).rjust(5) + ";")
            wght_ref.write("\n")
        wght_ref.close()
    elif "Depthwise" in str(layer):
        wght_ref = [0  for c in range(layer.input.shape[3])]
        for c in range(layer.input.shape[3]):
            wght_ref[c] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/weight/wght_ref' + '_' + str(c) + '.csv')
            for x in range(layer.kernel_size[0]):
                for y in range(layer.kernel_size[1]):
                    wght_ref[c].write(str(dram.weights[layer_number][c][y][x]).rjust(5) + ";")
                wght_ref[c].write("\n")
            wght_ref[c].close()
    else:
        wght_ref = [[0 for f in range(layer.filters)] for c in range(layer.input.shape[3])]
        for c in range(layer.input.shape[3]):
            for f in range(layer.filters):
                wght_ref[c][f] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/weight/wght_ref' + '_' + str(c) + '_' + str(f) + '.csv')
                for x in range(layer.kernel_size[0]):
                    for y in range(layer.kernel_size[1]):
                        wght_ref[c][f].write(str(dram.weights[layer_number][c][f][y][x]).rjust(5) + ";")
                    wght_ref[c][f].write("\n")
                wght_ref[c][f].close()
    return wght_ref

def write_iact_file(layer, layer_number, dram):
    if "Dense" in str(layer):
        iact_ref = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/iact/iact_ref' + '_0.csv')
        for c in range(layer.input.shape[1]):
            iact_ref.write(str(int(dram.fmap[layer_number][c])))
            iact_ref.write("\n")
        iact_ref.close()

    else:
        iact_ref = [0 for c in range(layer.input.shape[3])]
        for c in range(layer.input.shape[3]):
            iact_ref[c] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/iact/iact_ref' + '_' +  str(c) + '.csv')
            for y in range(0 - math.floor(layer.kernel_size[1]/2),layer.input.shape[2] + math.ceil(layer.kernel_size[1]/2) - 1):
                for x in range(0 - math.floor(layer.kernel_size[0]/2),layer.input.shape[1] + math.ceil(layer.kernel_size[0]/2) - 1):
                    if(((x >= 0) & (x  < layer.input.shape[2])) & \
                        ((y >= 0) & (y < layer.input.shape[1]))):
                        iact_ref[c].write(str(int(dram.fmap[layer_number][c][x][y])).rjust(5) + ";")
                    else:
                        iact_ref[c].write(str(1).rjust(5) + ";")
                iact_ref[c].write("\n")
            iact_ref[c].close()
        return iact_ref

def write_psum_file(layer, layer_number, dram, calculated_results):
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []
    if "Dense" in str(layer):
        psum_ref = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/psum/psum_ref' + '_0.csv')
        for x in range(layer.output.shape[1]):
            psum_ref.write(str(calculated_results[x]))
            psum_ref.write("\n")
        psum_ref.close()

    elif "Depthwise" in str(layer):
        psum_ref = [0 for f in range(layer.output.shape[3])]
        for c in range(layer.output.shape[3]):
            psum_ref[c] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/psum/psum_ref' + '_' +  str(c) + '.csv')
            for x in range(layer.output.shape[1]):
                for y in range(layer.output.shape[2]):
                    psum_ref[c].write(str(calculated_results[c][y][x]).rjust(8) + ";")
                psum_ref[c].write("\n")
            psum_ref[c].close()

    elif "Conv" in str(layer):
        for f in range(layer.output.shape[3]):
            p = mp.Process(target = write_psum_file_conv_mp, args = (f, layer, calculated_results, return_dict))
            p.start()
            jobs.append(p)
        for proc in range(len(jobs)):
            jobs[proc].join()
        psum_ref = [0 for f in range(layer.output.shape[3])]
        for f in range(layer.output.shape[3]):
            psum_ref[f] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/psum/psum_ref' + '_' +  str(f) + '.csv')
            psum_ref[f].write(return_dict[f])
            psum_ref[f].close()
    return psum_ref

def write_psum_file_conv_mp(f, layer, calculated_results, return_dict):
    psum_ref = ""
    for x in range(layer.output.shape[1]):
        for y in range(layer.output.shape[2]):
            psum_ref = psum_ref + (str(calculated_results[f][y][x]).rjust(8) + ";")
        psum_ref = psum_ref + ("\n")
    return_dict[f] = psum_ref

#Collect and get results
def collect_results(layer, layer_number, layer_params, dram):
    #Calculate Bias
    if "Dense" in str(layer):
        calculated_results = [0 for i in range(layer.output.shape[1])]
        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []
        for x in range(layer.output.shape[1]):
            p = mp.Process(target = calculate_dense_results_mp, args = (x, layer, layer_number, dram, calculated_results[x], return_dict))
            p.start()
            jobs.append(p)
        for proc in range(len(jobs)):
            jobs[proc].join()
        calculated_results = return_dict

    elif "Depthwise" in str(layer):
        calculated_results = [[[0 for i in range(layer.output.shape[1])] for j in range(layer.output.shape[2])]for k in range(layer.output.shape[3])]
        for j in range(layer.output.shape[1]):
            for i in range(layer.output.shape[2]):
                for f in range(layer.output.shape[3]):
                    calculated_results[f][i][j] = int(calculated_results[f][i][j] + int(layer.bias[f]))
        for j in range(layer.output.shape[1]):
            for i in range(layer.output.shape[2]):
                for y in range(0 - math.floor(layer.kernel_size[1]/2),math.ceil(layer.kernel_size[1]/2)):
                    for c in range(layer.input.shape[3]):
                        for x in range(0 - math.floor(layer.kernel_size[0]/2),math.ceil(layer.kernel_size[0]/2)):
                            if((((x + i * layer_params.strideX) >= 0) & ((x + i * layer_params.strideX) < (layer.output.shape[2] * layer_params.strideX))) & \
                            (((y + j * layer_params.strideY) >= 0) & ((y + j * layer_params.strideY) < (layer.output.shape[1] * layer_params.strideY)))):
                                calculated_results[c][i][j] = int(calculated_results[c][i][j] + \
                                                                dram.weights[layer_number][c][x + math.floor(layer.kernel_size[0]/2)][y + math.floor((layer.kernel_size[1]-1)/2)] * \
                                                                dram.fmap[layer_number][c][x + (i * layer_params.strideX)][y + (j * layer_params.strideY)])
                            else:
                                calculated_results[c][i][j] = int(calculated_results[c][i][j] + \
                                                                dram.weights[layer_number][c][x + math.floor(layer.kernel_size[0]/2)][y + math.floor((layer.kernel_size[1]-1)/2)])
    elif "Conv" in str(layer):
        calculated_results = [[[0 for i in range(layer.output.shape[1])] for j in range(layer.output.shape[2])]for k in range(layer.output.shape[3])]
        for j in range(layer.output.shape[1]):
            for i in range(layer.output.shape[2]):
                for f in range(layer.output.shape[3]):
                    calculated_results[f][i][j] = int(calculated_results[f][i][j] + int(layer.bias[f]))

        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []
        max_parallel_jobs = 1
        semaphore = mp.Semaphore(max_parallel_jobs)

        for f in range(layer.output.shape[3]):
            p = mp.Process(target = calculate_conv_results_mp, args = (f, layer, layer_number, layer_params, dram, calculated_results[f], return_dict, semaphore))
            p.start()
            jobs.append(p)
        
        for proc in range(len(jobs)):
            jobs[proc].join()

        calculated_results = return_dict
    return calculated_results

def calculate_dense_results_mp(x, layer, layer_number, dram, calculated_results,return_dict):
    for c in range(layer.input.shape[1]):
        calculated_results = int(calculated_results + dram.weights[layer_number][x][c] * dram.fmap[layer_number][c])
    return_dict[x] = calculated_results

def calculate_conv_output_stream_mp(layer_repetition, layer_number, params, layer_params, layer, cluster_order, calculated_results):
    file_dma_ref = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_ref.txt')
    layer_repetition_cycle = math.floor(layer_repetition/layer_params.iact_transmissions_pe)
    if (params.SERIAL):
        for refresh in range(math.ceil(layer_params.needed_refreshes_mx[layer_repetition][1]/layer_params.used_Y_cluster),
                            math.ceil(layer_params.needed_refreshes_mx[layer_repetition][2]/layer_params.used_Y_cluster)):
            for cl_y in cluster_order:
                for cl_x in range(params.Clusters_X):
                    for router in range(params.Psum_Routers):
                        for psum_pe in range((layer_repetition_cycle%layer_params.needed_wght_transmissions)*math.ceil(layer.filters/layer_params.needed_wght_transmissions/2),\
                            ((layer_repetition_cycle%layer_params.needed_wght_transmissions)+1)*math.ceil(layer.filters/layer_params.needed_wght_transmissions/2)):
                            if(layer_params.computing_mx[cl_x][cl_y][0][router] == 1):
                                partial_result_a = gtu.to_twos_complement_string(0,20)
                                partial_result_b = gtu.to_twos_complement_string(0,20)
                                for counter in range(math.floor(params.DMA_Bits/params.PSUM_Bitwidth)):
                                    x_cor= int(((router + \
                                    cl_x * params.PEs_X + \
                                    math.floor(cl_y/layer_params.used_Y_cluster) * params.Clusters_X * params.PEs_X + \
                                    ((cl_y%layer_params.used_Y_cluster) + refresh*layer_params.used_Y_cluster) * (params.Clusters_Y * params.Clusters_X * params.PEs_X/layer_params.used_Y_cluster)) \
                                    % (layer.output.shape[2] + layer_params.add_up)))

                                    y_cor= int(((router + \
                                    cl_x * params.PEs_X + \
                                    math.floor(cl_y/layer_params.used_Y_cluster) * params.Clusters_X * params.PEs_X + \
                                    ((cl_y%layer_params.used_Y_cluster) + refresh*layer_params.used_Y_cluster) * params.Clusters_Y * params.Clusters_X * params.PEs_X/layer_params.used_Y_cluster) \
                                    / (layer.output.shape[2] + layer_params.add_up)))
                                    filter = 2 * psum_pe + counter
                                    try:
                                        if((x_cor < layer.output.shape[1]) & (y_cor < layer.output.shape[2])):
                                            if (counter == 0):
                                                partial_result_b = gtu.to_twos_complement_string(calculated_results[filter][x_cor][y_cor],20)
                                            else:
                                                partial_result_a = gtu.to_twos_complement_string(calculated_results[filter][x_cor][y_cor],20)
                                    except:
                                        partial_result_b = partial_result_b
                                        partial_result_a = partial_result_a

                                file_dma_ref.write(partial_result_a + partial_result_b + "\n")
    else:
        for refresh in range(math.ceil(layer_params.needed_refreshes_mx[layer_repetition][1]/layer_params.used_Y_cluster),
                            math.ceil(layer_params.needed_refreshes_mx[layer_repetition][2]/layer_params.used_Y_cluster)):
            for psum_pe in range((layer_repetition_cycle%layer_params.needed_wght_transmissions)*math.ceil(layer.filters/layer_params.needed_wght_transmissions/2),\
                ((layer_repetition_cycle%layer_params.needed_wght_transmissions)+1)*math.ceil(layer.filters/layer_params.needed_wght_transmissions/2)):
                for cl_y in cluster_order:
                    for cl_x in range(params.Clusters_X):
                        for router in range(params.Psum_Routers):
                            if(layer_params.computing_mx[cl_x][cl_y][0][router] == 1):
                                partial_result_a = gtu.to_twos_complement_string(0,20)
                                partial_result_b = gtu.to_twos_complement_string(0,20)
                                for counter in range(math.floor(params.DMA_Bits/params.PSUM_Bitwidth)):
                                    x_cor= int(((router + \
                                    cl_x * params.PEs_X + \
                                    math.floor(cl_y/layer_params.used_Y_cluster) * params.Clusters_X * params.PEs_X + \
                                    ((cl_y%layer_params.used_Y_cluster) + refresh*layer_params.used_Y_cluster) * (params.Clusters_Y * params.Clusters_X * params.PEs_X/layer_params.used_Y_cluster)) \
                                    % (layer.output.shape[2] + layer_params.add_up)))

                                    y_cor= int(((router + \
                                    cl_x * params.PEs_X + \
                                    math.floor(cl_y/layer_params.used_Y_cluster) * params.Clusters_X * params.PEs_X + \
                                    ((cl_y%layer_params.used_Y_cluster) + refresh*layer_params.used_Y_cluster) * params.Clusters_Y * params.Clusters_X * params.PEs_X/layer_params.used_Y_cluster) \
                                    / (layer.output.shape[2] + layer_params.add_up)))
                                    filter = 2 * psum_pe + counter
                                    try:
                                        if((x_cor < layer.output.shape[1]) & (y_cor < layer.output.shape[2])):
                                            if (counter == 0):
                                                partial_result_b = gtu.to_twos_complement_string(calculated_results[filter][x_cor][y_cor],20)
                                            else:
                                                partial_result_a = gtu.to_twos_complement_string(calculated_results[filter][x_cor][y_cor],20)
                                    except:
                                        partial_result_b = partial_result_b
                                        partial_result_a = partial_result_a

                                file_dma_ref.write(partial_result_a + partial_result_b + "\n")
    file_dma_ref.close()

    logger.info("Stream " + str(layer_repetition) + " / " + str(layer_params.needed_total_transmissions) + " calculated.")

def calculate_dw_output_stream_mp(layer_repetition, layer_number, params, layer_params, layer, cluster_order, calculated_results):
    file_dma_ref = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_ref.txt')
    layer_repetition_cycle = math.floor(layer_repetition/layer_params.iact_transmissions_pe)
    for refresh in range(math.floor((layer_repetition_cycle/layer_params.needed_total_transmissions) * layer_params.Used_refreshes),
                        math.floor(((layer_repetition_cycle+1)/layer_params.needed_total_transmissions) * layer_params.Used_refreshes)):
        for psum_pe in range((layer_repetition%layer_params.iact_transmissions_pe)*math.ceil(layer_params.filters),\
            ((layer_repetition%layer_params.iact_transmissions_pe)+1)*math.ceil(layer_params.filters)):
            for cl_y in range(params.Clusters_Y):
                for cl_x in range(params.Clusters_X):
                    for router in range(params.Psum_Routers):
                        if(layer_params.computing_mx[cl_x][cl_y][0][router] == 1):
                            partial_result_a = gtu.to_twos_complement_string(0,20)
                            partial_result_b = gtu.to_twos_complement_string(0,20)
                            for counter in range(math.ceil(1)):  #TODO:Flexibel machen
                                x_cor= int(((router + cl_x * params.PEs_X + cl_y * params.Clusters_X * params.PEs_X + refresh * params.Clusters_Y * params.Clusters_X * params.PEs_X ) % (layer.output.shape[2] + layer_params.add_up)))
                                y_cor= int(((router + cl_x * params.PEs_X + cl_y * params.Clusters_X * params.PEs_X + refresh * params.Clusters_Y * params.Clusters_X * params.PEs_X ) / (layer.output.shape[2] + layer_params.add_up)))
                                filter = psum_pe + counter
                                try:
                                    if((x_cor < layer.output.shape[1]) & (y_cor < layer.output.shape[2])):
                                        if (counter == 0):
                                            partial_result_b = gtu.to_twos_complement_string(calculated_results[filter][x_cor][y_cor],20)
                                        else:
                                            partial_result_a = gtu.to_twos_complement_string(calculated_results[filter][x_cor][y_cor],20)
                                except:
                                    partial_result_b = partial_result_b
                                    partial_result_a = partial_result_a

                            file_dma_ref.write(partial_result_a)
                            file_dma_ref.write(partial_result_b)
                            file_dma_ref.write("\n")
    file_dma_ref.close()

    logger.info("Stream " + str(layer_repetition) + " / " + str(layer_params.needed_total_transmissions) + " calculated.")

def calculate_conv_results_mp(f, layer, layer_number, layer_params, dram, calculated_results, return_dict, semaphore):
    with semaphore:
        if(f < layer.kernel.shape[3]):
            for j in range(layer.output.shape[1]):
                for i in range(layer.output.shape[2]):
                    for x in range(0 - math.floor(layer.kernel_size[0]/2),math.ceil(layer.kernel_size[0]/2)):
                        for c in range(layer.input.shape[3]):
                            for y in range(0 - math.floor(layer.kernel_size[1]/2),math.ceil(layer.kernel_size[1]/2)):
                                if((((x + i * layer_params.strideX) >= 0) & ((x + i * layer_params.strideX) < (layer.output.shape[2] * layer_params.strideX))) & \
                                (((y + j * layer_params.strideY) >= 0) & ((y + j * layer_params.strideY) < (layer.output.shape[1] * layer_params.strideY)))):
                                    calculated_results[i][j] = int(calculated_results[i][j] + \
                                                                    dram.weights[layer_number][c][f][x + math.floor(layer.kernel_size[0]/2)][y + math.floor((layer.kernel_size[1]-1)/2)] * \
                                                                    dram.fmap[layer_number][c][x + (i * layer_params.strideX)][y + (j * layer_params.strideY)])

                                else:
                                    calculated_results[i][j] = int(calculated_results[i][j] + \
                                                                    dram.weights[layer_number][c][f][x + math.floor(layer.kernel_size[0]/2)][y + math.floor((layer.kernel_size[1]-1)/2)])
        return_dict[f] = calculated_results

def compare_dram_with_ref(layer, ref_output, dram):
    logger.info("Results are checked.")

    if "Conv" in str(layer):
        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []

        for f in range(len(ref_output)):
            p = mp.Process(target = compare_dram_with_ref_mp, args = (f, ref_output[f], dram[f], return_dict))
            p.start()
            jobs.append(p)
        
        for proc in range(len(jobs)):
            jobs[proc].join()

        for f in range(len(ref_output)):
            if (return_dict[f] == False) :
                return False

    elif "Dense" in str(layer):
        for f in range(len(ref_output)):
            if dram[f] != ref_output[f]:
                logger.error(f'Difference found at f = {f}')
                logger.error(f'ReferenceData: {str(ref_output[f])}')
                logger.error(f'Output Stream: {str(dram[f])}')
                return False

    return True

def compare_dram_with_ref_mp(f, ref_output, dram, return_dict):
    return_dict[f] = True
    for x in range(len(ref_output)):
        for y in range(len(ref_output[x])):
            if dram[x][y] != ref_output[x][y]:
                logger.error(f'Difference found at f = {f}, x = {x}, y= {y}')
                logger.error(f'ReferenceData: {str(ref_output[x][y])}')
                logger.error(f'Output Stream: {str(dram[x][y])}')
                return_dict[f] = False
                return
                 