# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until params.DATA_PSUM_BITWIDTH25), Universität Duisburg-Essen (since params.DATA_PSUM_BITWIDTH25).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Main test utilities for the OpenEye neural network accelerator.

This module provides high-level test utilities and orchestration for testing
the OpenEye neural network accelerator implementation. It includes functionality for:

- Stream generation and mapping for different layer types
- Reference data generation and validation
- Multi-process computation and comparison
- File I/O for test data and results
- DRAM state management and verification

The module supports testing of:
- Convolutional layers (standard and depthwise)
- Dense (fully connected) layers
- Pooling layers
- Batch normalization operations

Key features:
- Parallel processing for computationally intensive operations
- Detailed logging and debugging capabilities
- Comprehensive result validation
- Support for both serial and parallel hardware configurations
"""

import os
import logging
import math
import numpy as np

from open_eye.pooling_mapper import PoolingMapper
from open_eye.dense_mapper import DenseMapper
from open_eye.conv_mapper import ConvMapper
from open_eye.dw_mapper import DWMapper
import open_eye.generic_test_utils as gtu
import open_eye.layer_execution_state as les
import multiprocessing as mp

logger = logging.getLogger("cocotb")

def get_verilog_sources(hdl_dir):
    """Gather all Verilog source files from a directory tree.
    
    This function recursively walks through the HDL directory to find all Verilog
    source files needed for simulation and synthesis.
    
    Args:
        hdl_dir: Root directory containing Verilog source files
        
    Returns:
        list: List of absolute paths to all Verilog source files
    """
    verilog_sources = []
    for root, _, files in os.walk(hdl_dir):
        for f in files:
            full_path = os.path.join(root, f)
            if full_path.endswith('.v') or full_path.endswith('.sv'):
                verilog_sources.append(full_path)

    return verilog_sources

def write_stream_layer_mp(params, layer_params, dram_layer_content, return_dict, layer_repetition, sparse_iacts, sparse_wghts):
    """Generate test data streams for a layer in a multiprocessing context.
    
    This function creates appropriate data streams for testing different types of
    neural network layers. It selects the correct stream generator based on the
    layer type and handles both dense and sparse data formats.
    
    Args:
        params: Global OpenEye configuration parameters
        layer_params: Layer-specific parameters and configuration
        dram_layer_content: Layer data content from DRAM
        return_dict: Multiprocessing dictionary for returning results
        layer_repetition: Current repetition count for layer processing
        sparse_iacts: Sparsity information for input activations
        sparse_wghts: Sparsity information for weights
    
    The function supports multiple layer types:
    - Depthwise convolution layers
    - Standard convolution layers
    - Dense (fully connected) layers
    - Pooling layers
    
    Each layer type uses a specialized mapper class to generate appropriate
    test streams that match the hardware's expected data format and timing.
    """
    if "Depthwise" in str(layer_params.layer_name):
        LayerStreamGenerator = DWMapper(params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts)
        LayerStreamGenerator.make_stream()
    elif "Conv" in str(layer_params.layer_name):
        LayerStreamGenerator = ConvMapper(params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts)
        LayerStreamGenerator.make_stream()
    elif "Dense" in str(layer_params.layer_name):
        LayerStreamGenerator = DenseMapper(params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts)
        LayerStreamGenerator.make_stream()
    elif "Pooling" in str(layer_params.layer_name):
        LayerStreamGenerator = PoolingMapper(params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts)
        LayerStreamGenerator.make_stream()
    return_dict[layer_repetition] = LayerStreamGenerator.get_stream()

def write_stream(params, layer_params, dram_layer_content, sparse_iacts, sparse_wghts):
    """Generate test data streams in parallel for all layer repetitions.
    
    This function orchestrates parallel generation of test data streams for neural
    network layer testing. It uses Python's multiprocessing to parallelize the
    stream generation across multiple processes.
    
    Args:
        params: Global OpenEye configuration parameters
        layer_params: Layer-specific parameters and configuration
        dram_layer_content: Layer data content from DRAM
        sparse_iacts: Sparsity information for input activations
        sparse_wghts: Sparsity information for weights
    
    Returns:
        dict: Dictionary containing generated streams for each layer repetition
    
    Implementation Details:
        - Creates a multiprocessing manager for shared memory
        - Spawns parallel processes for each layer repetition
        - Collects results through a shared dictionary
        - Ensures all processes complete before returning
        - Handles both dense and sparse data formats
    """
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []

    for layer_repetition in range(layer_params.needed_total_transmissions):
        p = mp.Process(target=write_stream_layer_mp, 
                      args=(params, layer_params, dram_layer_content, 
                            return_dict, layer_repetition, 
                            sparse_iacts, sparse_wghts))
        p.start()
        jobs.append(p)

    for proc in range(len(jobs)):
        jobs[proc].join()
        
    return return_dict

# Reference Generation

def make_ref(params, layer_params, layer_number, dram, calculated_results):
    """Generate reference data files for layer verification.
    
    This function creates a complete set of reference files needed to verify
    the correct operation of a neural network layer in hardware. It generates
    files for weights, input activations, and expected outputs.
    
    Args:
        params: Global OpenEye configuration parameters
        layer_params: Layer-specific parameters and configuration
        layer_number: Current layer index in the network
        dram: Memory object containing layer data
        calculated_results: Pre-calculated expected results
        
    The function creates three types of reference files:
    1. Weight files: Layer weight parameters
    2. Input activation files: Layer input data
    3. Partial sum files: Expected output results
    
    All files are organized by layer number and stored in a demo directory
    structure for test verification.
    """
    # Write weight files
    write_weight_file(layer_params, layer_number, dram)
    logger.info("All weight-files written")

    # Write input activation files
    write_iact_file(layer_params, layer_number, dram)
    logger.info("All iact-files written")

    logger.info("All results calculated")

    # Write partial sum (output) files
    write_psum_file(layer_params, layer_number, dram, calculated_results)
    logger.info("All psum-files written")
    
    dma_line = 0
    
    output_order = []
    if "Depthwise" in str(layer_params.layer_name):
        if(params.SERIAL):
            file_dma_ref = [0 for layer_repetition in range(layer_params.needed_total_transmissions)]
            for layer_repetition in range(layer_params.needed_total_transmissions):
                file_dma_ref[layer_repetition] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_ref.txt')
                for refresh in range(math.floor((math.floor((layer_repetition%layer_params.iact_transmissions_pe))/layer_params.needed_total_transmissions) * layer_params.Used_refreshes),
                                    math.floor(((math.floor((layer_repetition%layer_params.iact_transmissions_pe))+1)/layer_params.needed_total_transmissions) * layer_params.Used_refreshes)):
                    for cl_y in range(params.Clusters_Y):
                        for cl_x in range(params.Clusters_X):
                            for router in range(params.Psum_Routers):
                                for psum_pe in range(int((layer_params.filters*(layer_repetition%layer_params.needed_wght_transmissions)/layer_params.needed_wght_transmissions)/2),\
                                    int((layer_params.filters*(1+(layer_repetition%layer_params.needed_wght_transmissions))/layer_params.needed_wght_transmissions)/2)):
                                    for counter in range(params.DMA_Bit_AXI//params.DATA_PSUM_BITWIDTH):
                                        x_cor= int(((router + cl_x * params.PEs_X + cl_y * params.Clusters_X * params.PEs_X + refresh * params.Clusters_Y * params.Clusters_X * params.PEs_X ) % layer_params.output_shape[2]))
                                        y_cor= int(((router + cl_x * params.PEs_X + cl_y * params.Clusters_X * params.PEs_X + refresh * params.Clusters_Y * params.Clusters_X * params.PEs_X ) / layer_params.output_shape[2]))
                                        if((x_cor < layer_params.output_shape[1]) & (y_cor < layer_params.output_shape[2])):
                                            if(calculated_results[2 * psum_pe + counter][x_cor][y_cor] >= 0):
                                                dma_line = dma_line + (calculated_results[2 * psum_pe + counter][x_cor][y_cor] << (params.DATA_PSUM_BITWIDTH * counter))
                                            else:
                                                dma_line = dma_line
                                    file_dma_ref[layer_repetition].write(bin(dma_line)[2:].zfill(params.DMA_Bit_AXI) + "\n")
                                    dma_line = 0
                file_dma_ref[layer_repetition].close()
        else:
            cluster_order = []
            for a in range(layer_params.used_Y_cluster):
                for b in range(0,params.Clusters_Y,layer_params.used_Y_cluster):
                    cluster_order.append(a+b)
            manager = mp.Manager()
            return_dict = manager.dict()
            jobs = []
            for layer_repetition in range(layer_params.needed_total_transmissions):
                p = mp.Process(target = calculate_dw_output_stream_mp, \
                               args = (layer_repetition, layer_number, params, layer_params, cluster_order, calculated_results, return_dict))
                p.start()
                jobs.append(p)
            
            for proc in range(len(jobs)):
                jobs[proc].join()
            for layer_repetition in range(layer_params.needed_total_transmissions):
                output_order.append(return_dict[layer_repetition])
    elif "Conv" in str(layer_params.layer_name):
        cluster_order = []
        for a in range(layer_params.used_Y_cluster):
            for b in range(0,params.Clusters_Y,layer_params.used_Y_cluster):
                cluster_order.append(a+b)

        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []
        for layer_repetition in range(layer_params.needed_total_transmissions):
            p = mp.Process(target = calculate_conv_output_stream_mp, \
                            args = (layer_repetition, layer_number, params, layer_params, cluster_order, calculated_results, return_dict))
            p.start()
            jobs.append(p)
        
        for proc in range(len(jobs)):
            jobs[proc].join()
        for layer_repetition in range(layer_params.needed_total_transmissions):
            output_order.append(return_dict[layer_repetition])
    elif "Dense" in str(layer_params.layer_name):
        layer_repetition = 0
        file_dma_ref = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_ref.txt')
        if(params.SERIAL):
            for refresh in range(math.ceil(len(calculated_results)/2)):
                for x in range(2) :
                    partial_result_a = gtu.to_twos_complement_string(0,params.DATA_PSUM_BITWIDTH)
                    partial_result_b = gtu.to_twos_complement_string(0,params.DATA_PSUM_BITWIDTH)
                    try:
                        partial_result_b = gtu.to_twos_complement_string(calculated_results[refresh + x * layer_params.used_psum_per_PE],params.DATA_PSUM_BITWIDTH)
                    except:
                        partial_result_b = partial_result_b
                    file_dma_ref.write(partial_result_a + partial_result_b + "\n")
            file_dma_ref.close()
        else:
            file_dma_ref = [0 for layer_repetition in range(layer_params.needed_total_transmissions)]
            for layer_repetition in range(layer_params.needed_total_transmissions):
                file_dma_ref[layer_repetition] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_ref.txt')
                for psum_pe in range(math.ceil(layer_params.used_psum_per_PE/2)):
                    for cl_y in range(params.Clusters_Y):
                        for cl_x in range(params.Clusters_X):
                            partial_result_a = gtu.to_twos_complement_string(0,params.DATA_PSUM_BITWIDTH)
                            partial_result_b = gtu.to_twos_complement_string(0,params.DATA_PSUM_BITWIDTH)
                            for counter in range(params.PARALLEL_MACS):
                                layer_repetition_cycle = math.floor(layer_repetition/layer_params.iact_transmissions_pe)
                                output = \
                                    counter + \
                                    2 * psum_pe +\
                                    cl_x * layer_params.used_psum_per_PE + \
                                    cl_y * layer_params.used_psum_per_PE * params.Clusters_X + \
                                    layer_repetition_cycle * layer_params.used_psum_per_PE * params.Clusters_X * params.Clusters_Y
                                if (counter == 0):
                                    try:
                                        partial_result_b = gtu.to_twos_complement_string(calculated_results[output],params.DATA_PSUM_BITWIDTH)
                                    except:
                                        partial_result_b = gtu.to_twos_complement_string(0,params.DATA_PSUM_BITWIDTH)
                                else:
                                    try:
                                        partial_result_a = gtu.to_twos_complement_string(calculated_results[output],params.DATA_PSUM_BITWIDTH)
                                    except:
                                        partial_result_a = gtu.to_twos_complement_string(0,params.DATA_PSUM_BITWIDTH)

                            file_dma_ref[layer_repetition].write(partial_result_a)
                            file_dma_ref[layer_repetition].write(partial_result_b)
                            file_dma_ref[layer_repetition].write("\n")
                            dma_line = 0
                file_dma_ref[layer_repetition].close()
    logger.info("Reference Output calculated.")
    return output_order

def write_weight_file(layer_params, layer_number, dram):
    """Write weight reference data to files for each layer type.
    
    This function handles the writing of weight data to reference files,
    with different formats depending on the layer type. It supports dense,
    convolutional, and depthwise convolutional layers.
    
    Args:
        layer_params: Layer-specific parameters and configuration
        layer_number: Current layer index in the network
        dram: Memory object containing weight data
        
    File Organization:
    - Dense layers: Single file with 2D weight matrix
    - Depthwise Conv: One file per input channel
    - Standard Conv: One file per input channel and filter combination
    
    File Format:
    - CSV format with semicolon separators
    - Right-justified numeric values
    - Each row represents a slice of the weight tensor
    """
    if "Dense" in str(layer_params.layer_name):
        wght_ref = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/weight/wght_ref' + '_0.csv')
        for c in range(layer_params.iact_size_x):
            for x in range(layer_params.filters):
                wght_ref.write(str(int(dram.weights[layer_number][x][c])).rjust(5) + ";")
            wght_ref.write("\n")
        wght_ref.close()
    elif "Depthwise" in str(layer_params.layer_name):
        wght_ref = [0  for c in range(layer_params.input_shape[3])]
        for c in range(layer_params.input_shape[3]):
            wght_ref[c] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/weight/wght_ref' + '_' + str(c) + '.csv')
            for x in range(layer_params.kernel_size[0]):
                for y in range(layer_params.kernel_size[1]):
                    wght_ref[c].write(str(dram.weights[layer_number][c][y][x]).rjust(5) + ";")
                wght_ref[c].write("\n")
            wght_ref[c].close()
    elif "Conv" in str(layer_params.layer_name):
        wght_ref = [[0 for f in range(layer_params.filters)] for c in range(layer_params.input_shape[3])]
        for c in range(layer_params.input_shape[3]):
            for f in range(layer_params.filters):
                wght_ref[c][f] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/weight/wght_ref' + '_' + str(c) + '_' + str(f) + '.csv')
                for y in range(layer_params.kernel_size[1]):
                    for x in range(layer_params.kernel_size[0]):
                        wght_ref[c][f].write(str(dram.weights[layer_number][c][f][y][x]).rjust(5) + ";")
                    wght_ref[c][f].write("\n")
                wght_ref[c][f].close()
    else:
        pass

def write_iact_file(layer_params, layer_number, dram):
    if "Dense" in str(layer_params.layer_name):
        iact_ref = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/iact/iact_ref' + '_0.csv')
        for c in range(layer_params.iact_size_x):
            iact_ref.write(str(int(dram.fmap[layer_number][c])))
            iact_ref.write("\n")
        iact_ref.close()
    elif "Pooling" in str(layer_params.layer_name):
        iact_ref = [0 for c in range(layer_params.input_shape[3])]
        for c in range(layer_params.input_shape[3]):
            iact_ref[c] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/iact/iact_ref' + '_' +  str(c) + '.csv')
            for y in range(layer_params.input_shape[2]):
                for x in range(layer_params.input_shape[1]):
                    iact_ref[c].write(str(int(dram.fmap[layer_number][c][x][y])).rjust(5) + ";")
                iact_ref[c].write("\n")
            iact_ref[c].close()
    else:
        iact_ref = [0 for c in range(layer_params.input_shape[3])]
        for c in range(layer_params.input_shape[3]):
            iact_ref[c] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/iact/iact_ref' + '_' +  str(c) + '.csv')
            for y in range(0 - math.floor(layer_params.kernel_size[1]/2),layer_params.input_shape[2] + math.ceil(layer_params.kernel_size[1]/2) - 1):
                for x in range(0 - math.floor(layer_params.kernel_size[0]/2),layer_params.input_shape[1] + math.ceil(layer_params.kernel_size[0]/2) - 1):
                    if(((x >= 0) & (x  < layer_params.input_shape[1])) & \
                        ((y >= 0) & (y < layer_params.input_shape[2]))):
                        iact_ref[c].write(str(int(dram.fmap[layer_number][c][x][y])).rjust(5) + ";")
                    else:
                        iact_ref[c].write(str(0).rjust(5) + ";")
                iact_ref[c].write("\n")
            iact_ref[c].close()

def write_psum_file(layer_params, layer_number, dram, calculated_results):
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []
    if "Dense" in str(layer_params.layer_name):
        psum_ref = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/psum/psum_ref' + '_0.csv')
        for x in range(layer_params.filters):
            psum_ref.write(str(calculated_results[x]))
            psum_ref.write("\n")
        psum_ref.close()

    elif "Depthwise" in str(layer_params.layer_name):
        psum_ref = [0 for f in range(layer_params.output_shape[3])]
        for c in range(layer_params.output_shape[3]):
            psum_ref[c] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/psum/psum_ref' + '_' +  str(c) + '.csv')
            for x in range(layer_params.output_shape[1]):
                for y in range(layer_params.output_shape[2]):
                    psum_ref[c].write(str(calculated_results[c][y][x]).rjust(8) + ";")
                psum_ref[c].write("\n")
            psum_ref[c].close()
    elif "Conv" in str(layer_params.layer_name):
        for f in range(layer_params.output_shape[3]):
            p = mp.Process(target = write_psum_file_conv_mp, args = (f, layer_params, calculated_results, return_dict))
            p.start()
            jobs.append(p)
        for proc in range(len(jobs)):
            jobs[proc].join()
        psum_ref = [0 for f in range(layer_params.output_shape[3])]
        for f in range(layer_params.output_shape[3]):
            psum_ref[f] = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/psum/psum_ref' + '_' +  str(f) + '.csv')
            psum_ref[f].write(return_dict[f])
            psum_ref[f].close()
    elif "Pooling" in str(layer_params.layer_name):
        psum_ref = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '/psum/psum_ref' + '_0.csv')
        for x in range(layer_params.output_shape[3]):
            psum_ref.write(str(calculated_results[x]))
            psum_ref.write("\n")
        psum_ref.close()

def write_psum_file_conv_mp(f, layer_params, calculated_results, return_dict):
    psum_ref = ""
    for y in range(layer_params.output_shape[2]):
        for x in range(layer_params.output_shape[1]):
            psum_ref = psum_ref + (str(calculated_results[f][x][y]).rjust(8) + ";")
        psum_ref = psum_ref + ("\n")
    return_dict[f] = psum_ref

#Collect and get results

def collect_results(layer_number, layer_params, dram, serial):
    #Calculate Bias
    if "Dense" in str(layer_params.layer_name):
        calculated_results = [0 for i in range(layer_params.filters)]
        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []
        for x in range(layer_params.filters):
            p = mp.Process(target = calculate_dense_results_mp, args = (x, layer_params, layer_number, dram, calculated_results[x], return_dict))
            p.start()
            jobs.append(p)
        for proc in range(len(jobs)):
            jobs[proc].join()
        calculated_results = return_dict

    elif "Depthwise" in str(layer_params.layer_name):
        calculated_results = [[[0 for i in range(layer_params.output_shape[2])] for j in range(layer_params.output_shape[1])]for k in range(layer_params.output_shape[3])]
        for j in range(layer_params.output_shape[1]):
            for i in range(layer_params.output_shape[2]):
                for f in range(layer_params.output_shape[3]):
                    calculated_results[f][i][j] = int(calculated_results[f][i][j]) # + int(layer.bias[f])) Later add back in
        for j in range(layer_params.output_shape[1]):
            for i in range(layer_params.output_shape[2]):
                for y in range(0 - math.floor(layer_params.kernel_size[1]/2),math.ceil(layer_params.kernel_size[1]/2)):
                    for c in range(layer_params.input_shape[3]):
                        for x in range(0 - math.floor(layer_params.kernel_size[0]/2),math.ceil(layer_params.kernel_size[0]/2)):
                            if((((x + i * layer_params.strideX) >= 0) & ((x + i * layer_params.strideX) < (layer_params.output_shape[2] * layer_params.strideX))) & \
                            (((y + j * layer_params.strideY) >= 0) & ((y + j * layer_params.strideY) < (layer_params.output_shape[1] * layer_params.strideY)))):
                                calculated_results[c][i][j] = int(calculated_results[c][i][j] + \
                                                                dram.weights[layer_number][c][x + math.floor(layer_params.kernel_size[0]/2)][y + math.floor((layer_params.kernel_size[1]-1)/2)] * \
                                                                dram.fmap[layer_number][c][x + (i * layer_params.strideX)][y + (j * layer_params.strideY)])
                            else:
                                calculated_results[c][i][j] = int(calculated_results[c][i][j] + \
                                                                dram.weights[layer_number][c][x + math.floor(layer_params.kernel_size[0]/2)][y + math.floor((layer_params.kernel_size[1]-1)/2)])
    elif "Conv" in str(layer_params.layer_name):
        calculated_results = [[[0 for i in range(layer_params.output_shape[2])] for j in range(layer_params.output_shape[1])]for k in range(layer_params.output_shape[3])]
        for j in range(layer_params.output_shape[1]):
            for i in range(layer_params.output_shape[2]):
                for f in range(layer_params.output_shape[3]):
                    calculated_results[f][j][i] = int(calculated_results[f][j][i] + int(dram.bias[layer_number][f]))
        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []
        max_parallel_jobs = 1
        semaphore = mp.Semaphore(max_parallel_jobs)

        for f in range(layer_params.output_shape[3]):
            p = mp.Process(target = calculate_conv_results_mp, args = (f, layer_number, layer_params, serial, dram, calculated_results[f], return_dict, semaphore))
            p.start()
            jobs.append(p)
        
        for proc in range(len(jobs)):
            jobs[proc].join()

        calculated_results = return_dict

    elif "Pooling" in str(layer_params.layer_name):
        calculated_results = [[[0 for i in range(layer_params.output_shape[2])] for j in range(layer_params.output_shape[1])]for k in range(layer_params.output_shape[3])]
        if (layer_params.pooling_mode == 0): #Is Max Pooling
            for f in range(layer_params.output_shape[3]):
                for i in range(layer_params.output_shape[2]):
                    for j in range(layer_params.output_shape[1]):
                        block = [
                            dram.fmap[layer_number][f][2*j][2*i],
                            dram.fmap[layer_number][f][2*j][2*i+1],
                            dram.fmap[layer_number][f][2*j+1][2*i],
                            dram.fmap[layer_number][f][2*j+1][2*i+1]
                        ]
                        calculated_results[f][j][i] = int(max(block))
        else:#Is Average Pooling
            for f in range(layer_params.output_shape[3]):
                for i in range(layer_params.output_shape[2]):
                    for j in range(layer_params.output_shape[1]):
                        temp = 0
                        for x in range(layer_params.input_shape[1]):
                            for y in range(layer_params.input_shape[2]):
                                temp = temp + dram.fmap[layer_number][x][y]
                        temp = temp//(layer_params.inut_shape[1]*layer_params.input_shape[2])
                        calculated_results[f][j][i] = int(temp)

    return calculated_results

def calculate_dense_results_mp(x, layer_params, layer_number, dram, calculated_results,return_dict):
    for c in range(layer_params.iact_size_x):
        calculated_results = int(calculated_results + dram.weights[layer_number][x][c] * dram.fmap[layer_number][c])
    calculated_results = int(calculated_results + dram.bias[layer_number][x])
    return_dict[x] = calculated_results

def refresh_position(x_cor, y_cor, filter, y_line_counter, kernel_counter, layer_params, position) :
    return x_cor, y_cor, filter, y_line_counter, kernel_counter

def calculate_conv_serial(params, layer_params, calculated_results, file_dma_ref):
    filter_cycles = ((layer_params.filters//layer_params.used_psum_per_PE)//layer_params.different_kernels_per_calculation)
    output_number = layer_params.iact_size_y*layer_params.iact_size_x*layer_params.filters
    needed_refreshes = math.ceil(output_number / (layer_params.iact_size_x * layer_params.different_kernels_per_calculation * layer_params.y_lines_per_calculation) / layer_params.used_psum_per_PE)
    elements_per_calculation = layer_params.different_kernels_per_calculation * layer_params.y_lines_per_calculation * (layer_params.iact_size_x + layer_params.add_up)
    layer_es = les.LayerExecutionState()
    les.x_start = 0
    array = []
    for refresh in range(layer_params.needed_refreshes_mx[0][0]//layer_params.diff_iact_layer) :
        les.y_start = ((refresh // filter_cycles) // layer_params.iact_x_line_repetitions) * layer_params.y_lines_per_calculation
        les.f_start = layer_params.used_psum_per_PE * (refresh % filter_cycles) * layer_params.different_kernels_per_calculation

        filter = les.f_start
        position = 0
        for psum_pe in range(layer_params.used_psum_per_PE) :
            kernel_counter = 0
            x_cor = les.x_start
            y_cor = les.y_start
            for cl_y in range(params.Clusters_Y//layer_params.used_Y_cluster) :
                for cl_x in range(params.Clusters_X) :
                    if (x_cor >= layer_params.psum_size_x) :
                        if (kernel_counter < layer_params.different_kernels_per_calculation - 1) :
                            kernel_counter = kernel_counter + 1
                            x_cor = les.x_start
                            filter = filter + 1
                    for router in range(0, params.Psum_Routers, 2) :
                        partial_result_a, partial_result_b = gtu.to_twos_complement_string(0,params.DATA_PSUM_BITWIDTH), gtu.to_twos_complement_string(0,params.DATA_PSUM_BITWIDTH)
                        for counter in range(params.PARALLEL_MACS) :
                                array.append((x_cor,y_cor,filter))
                                if (kernel_counter < layer_params.different_kernels_per_calculation) :
                                    if((x_cor < layer_params.psum_size_x) & (y_cor < layer_params.psum_size_y)) :
                                        if (counter == 0):
                                            partial_result_b = gtu.to_twos_complement_string(calculated_results[filter][x_cor][y_cor],params.DATA_PSUM_BITWIDTH)
                                        else:
                                            partial_result_a = gtu.to_twos_complement_string(calculated_results[filter][x_cor][y_cor],params.DATA_PSUM_BITWIDTH)
                                        x_cor = x_cor + 1
                        file_dma_ref.write(partial_result_a + partial_result_b + "\n")
            filter = filter + 1
        if ((filter >= layer_params.filters)) :
            if (x_cor >= layer_params.psum_size_x) :
                les.x_start = 0
            else :
                les.x_start = x_cor

def calculate_conv_parallel(params, layer_params, calculated_results, cluster_order, layer_repetition, file_dma_ref):
    coordinates = []
    layer_repetition_cycle = math.floor(layer_repetition/layer_params.iact_transmissions_pe)
    refresh_lower = math.ceil(layer_params.needed_refreshes_mx[layer_repetition][1]/layer_params.used_Y_cluster)
    refresh_upper = math.ceil(layer_params.needed_refreshes_mx[layer_repetition][2]/layer_params.used_Y_cluster)
    for refresh in range(refresh_lower,refresh_upper):

        
        match layer_params.single_cluster_computation:
            case 1:
                psum_pe_lower = (layer_repetition_cycle%layer_params.needed_wght_transmissions)*math.ceil(layer_params.filters/layer_params.needed_wght_transmissions/params.Clusters/2)
                psum_pe_upper = ((layer_repetition_cycle%layer_params.needed_wght_transmissions)+1)*math.ceil(layer_params.filters/layer_params.needed_wght_transmissions/params.Clusters/2)
            case 2:
                psum_pe_lower = (layer_repetition_cycle%layer_params.needed_wght_transmissions)*math.ceil(layer_params.filters/layer_params.needed_wght_transmissions/params.Clusters_Y/2)
                psum_pe_upper = ((layer_repetition_cycle%layer_params.needed_wght_transmissions)+1)*math.ceil(layer_params.filters/layer_params.needed_wght_transmissions/params.Clusters_Y/2)
            case _:
                psum_pe_lower = (layer_repetition_cycle%layer_params.needed_wght_transmissions)*math.ceil(layer_params.filters/layer_params.needed_wght_transmissions/2)
                psum_pe_upper = ((layer_repetition_cycle%layer_params.needed_wght_transmissions)+1)*math.ceil(layer_params.filters/layer_params.needed_wght_transmissions/2)

        for psum_pe in range(psum_pe_lower,psum_pe_upper):
            for cl_y in cluster_order:
                for cl_x in range(params.Clusters_X):
                    for router in range(params.Psum_Routers):
                        if(layer_params.computing_mx[cl_x][cl_y][0][router] == 1):
                            partial_result_a, partial_result_b = gtu.to_twos_complement_string(0,params.DATA_PSUM_BITWIDTH), gtu.to_twos_complement_string(0,params.DATA_PSUM_BITWIDTH)
                            for counter in range(params.PARALLEL_MACS):
                                match layer_params.single_cluster_computation:
                                    case 1:
                                        x_cor= int(((router + \
                                        (refresh * params.PEs_X)) \
                                        % (layer_params.output_shape[1] + layer_params.add_up)))

                                        y_cor= int(((router + \
                                        (refresh * params.PEs_X)) \
                                        / (layer_params.output_shape[1] + layer_params.add_up)))
                                        filter = 2 * psum_pe + counter + (2 * psum_pe_upper * (cl_y * params.Clusters_X + cl_x))
                                    case 2:
                                        x_cor= int((((cl_x * params.PEs_X) + router + \
                                        (refresh * (params.PEs_X * params.Clusters_X))) \
                                        % (layer_params.output_shape[1] + layer_params.add_up)))

                                        y_cor= int(((router + \
                                        (refresh * (params.PEs_X * params.Clusters_X))) \
                                        / (layer_params.output_shape[1] + layer_params.add_up)))
                                        filter = 2 * psum_pe + counter + (2 * psum_pe_upper * cl_y)
                                    case _:
                                        x_cor= int(((router + \
                                        cl_x * params.PEs_X + \
                                        math.floor(cl_y/layer_params.used_Y_cluster) * params.Clusters_X * params.PEs_X + \
                                        ((cl_y%layer_params.used_Y_cluster) + refresh*layer_params.used_Y_cluster) * (params.Clusters_Y * params.Clusters_X * params.PEs_X/layer_params.used_Y_cluster)) \
                                        % (layer_params.output_shape[1] + layer_params.add_up)))

                                        y_cor= int(((router + \
                                        cl_x * params.PEs_X + \
                                        math.floor(cl_y/layer_params.used_Y_cluster) * params.Clusters_X * params.PEs_X + \
                                        ((cl_y%layer_params.used_Y_cluster) + refresh*layer_params.used_Y_cluster) * params.Clusters_Y * params.Clusters_X * params.PEs_X/layer_params.used_Y_cluster) \
                                        / (layer_params.output_shape[1] + layer_params.add_up)))

                                        filter = 2 * psum_pe + counter

                                coordinates.append([filter,x_cor,y_cor])
                                try:
                                    if((x_cor < layer_params.output_shape[1]) & (y_cor < layer_params.output_shape[2])):
                                        if (counter == 0):
                                            partial_result_b = gtu.to_twos_complement_string(calculated_results[filter][x_cor][y_cor],params.DATA_PSUM_BITWIDTH)
                                        else:
                                            partial_result_a = gtu.to_twos_complement_string(calculated_results[filter][x_cor][y_cor],params.DATA_PSUM_BITWIDTH)
                                except:
                                    partial_result_b = partial_result_b
                                    partial_result_a = partial_result_a

                            file_dma_ref.write(partial_result_a + partial_result_b + "\n")
    return coordinates

def calculate_conv_output_stream_mp(layer_repetition, layer_number, params, layer_params, cluster_order, calculated_results, return_dict):
    file_dma_ref = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_ref.txt')
    if (params.SERIAL):
        calculate_conv_serial(params, layer_params, calculated_results, file_dma_ref)
        return_dict[layer_repetition] = []
    else:
        return_dict[layer_repetition] = calculate_conv_parallel(params, layer_params, calculated_results, cluster_order, layer_repetition, file_dma_ref)
    file_dma_ref.close()
    logger.info("Stream " + str(layer_repetition) + " / " + str(layer_params.needed_total_transmissions) + " calculated.")

def calculate_dw_output_stream_mp(layer_repetition, layer_number, params, layer_params, cluster_order, calculated_results, return_dict):
    coordinates = []
    filter_number = 0
    file_dma_ref = gtu.open_or_create_file('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_ref.txt')
    max_refresh = math.floor(((layer_repetition+1)/layer_params.needed_total_transmissions) * layer_params.Used_refreshes) - math.floor((layer_repetition/layer_params.needed_total_transmissions) * layer_params.Used_refreshes)
    max_refresh = math.ceil(max_refresh/math.floor(params.PSUM_Trans_Bitwidth/params.DATA_PSUM_BITWIDTH))
    for refresh in range(0,max_refresh):
        for cl_y in range(params.Clusters_Y):
            for cl_x in range(params.Clusters_X):
                for router in range(params.Psum_Routers):
                    if(layer_params.computing_mx[cl_x][cl_y][0][router] == 1):
                        partial_result_a, partial_result_b = gtu.to_twos_complement_string(0,params.DATA_PSUM_BITWIDTH), gtu.to_twos_complement_string(0,params.DATA_PSUM_BITWIDTH)
                        for counter in range(math.floor(params.PSUM_Trans_Bitwidth/params.DATA_PSUM_BITWIDTH)):
                            match layer_params.single_cluster_computation:
                                case 1:
                                    x_cor= int(((router + (2*refresh+counter) * params.PEs_X ) % (layer_params.output_shape[1] + layer_params.add_up)))
                                    y_cor= int(((router + (2*refresh+counter) * params.PEs_X ) / (layer_params.output_shape[1] + layer_params.add_up))) + \
                                        max_refresh * 2 * math.floor(layer_repetition/layer_params.iact_transmissions_pe)

                                    filter_number = cl_x + (layer_repetition * params.Clusters) + (cl_y * params.Clusters_X) 
                                case 2:
                                    x_cor= int(((router + cl_x * params.PEs_X + ((2*refresh+counter) * params.PEs_X * params.Clusters_X)) % (layer_params.output_shape[1] + layer_params.add_up)))
                                    y_cor= int(((router + cl_x * params.PEs_X + ((2*refresh+counter) * params.PEs_X * params.Clusters_X)) / (layer_params.output_shape[1] + layer_params.add_up))) + \
                                        max_refresh * 2 * math.floor(layer_repetition/layer_params.iact_transmissions_pe)

                                    filter_number = (layer_repetition * params.Clusters_Y) + cl_y  
                                case _:
                                    x_cor= int(((router + cl_x * params.PEs_X + cl_y * params.Clusters_X * params.PEs_X + (2*refresh+counter) * params.Clusters_Y * params.Clusters_X * params.PEs_X ) % (layer_params.output_shape[1] + layer_params.add_up)))
                                    y_cor= int(((router + cl_x * params.PEs_X + cl_y * params.Clusters_X * params.PEs_X + (2*refresh+counter) * params.Clusters_Y * params.Clusters_X * params.PEs_X ) / (layer_params.output_shape[1] + layer_params.add_up))) + \
                                        max_refresh * 2 * math.floor(layer_repetition/layer_params.iact_transmissions_pe)
                                    filter_number = math.floor(layer_repetition%layer_params.iact_transmissions_pe)

                            coordinates.append([filter_number,x_cor,y_cor])
                            try:
                                if((x_cor < layer_params.output_shape[1]) & (y_cor < layer_params.output_shape[1])):
                                    if (counter == 0):
                                        partial_result_b = gtu.to_twos_complement_string(calculated_results[filter_number][x_cor][y_cor],params.DATA_PSUM_BITWIDTH)
                                    else:
                                        partial_result_a = gtu.to_twos_complement_string(calculated_results[filter_number][x_cor][y_cor],params.DATA_PSUM_BITWIDTH)
                            except:
                                partial_result_b = partial_result_b
                                partial_result_a = partial_result_a

                        file_dma_ref.write(partial_result_a)
                        file_dma_ref.write(partial_result_b)
                        file_dma_ref.write("\n")
    file_dma_ref.close()

    logger.info("Stream " + str(layer_repetition) + " / " + str(layer_params.needed_total_transmissions) + " calculated.")
    return_dict[layer_repetition] = coordinates

def calculate_conv_results_mp(f, layer_number, layer_params, serial, dram, calculated_results, return_dict, semaphore):
    with semaphore:
        if(f < layer_params.filters):
            for j in range(layer_params.output_shape[1]):
                for i in range(layer_params.output_shape[2]):
                    for x in range(0 - math.floor(layer_params.kernel_size[0]/2),math.ceil(layer_params.kernel_size[0]/2)):
                        for c in range(layer_params.input_shape[3]):
                            for y in range(0 - math.floor(layer_params.kernel_size[1]/2),math.ceil(layer_params.kernel_size[1]/2)):
                                if((((x + j * layer_params.strideX) >= 0) & ((x + j * layer_params.strideX) < (layer_params.input_shape[1]))) & \
                                (((y + i * layer_params.strideY) >= 0) & ((y + i * layer_params.strideY) < (layer_params.input_shape[2])))):
                                    calculated_results[j][i] = int(calculated_results[j][i] + \
                                                                    dram.weights[layer_number][c][f][y + math.floor((layer_params.kernel_size[1]-1)/2)][x + math.floor(layer_params.kernel_size[0]/2)] * \
                                                                    dram.fmap[layer_number][c][x + (j * layer_params.strideX)][y + (i * layer_params.strideY)])
                                else:
                                    if (serial) :
                                        pass
                                    else:
                                        calculated_results[j][i] = int(calculated_results[j][i] + \
                                            dram.weights[layer_number][c][f][x + math.floor(layer_params.kernel_size[0]/2)][y + math.floor((layer_params.kernel_size[1]-1)/2)])
        return_dict[f] = calculated_results

def compare_dram_with_ref(layer_params, ref_output, dram):
    """Compare hardware output in DRAM with reference data.
    
    This function verifies that the hardware's output matches the expected
    reference results, handling different layer types appropriately.
    
    Args:
        layer_params: Layer-specific parameters and configuration
        ref_output: Reference output data to compare against
        dram: Memory object containing hardware output
        
    Returns:
        bool: True if hardware output matches reference, False otherwise
        
    Implementation Details:
        - Handles different layer types (Conv, Dense, Pooling)
        - Uses parallel processing for convolutional layer verification
        - Provides detailed error logging for mismatches
        - Performs element-wise comparison with exact matching
        
    Error Reporting:
    - Logs precise location of mismatches (layer, position)
    - Shows both expected and actual values
    - Maintains processing even after finding errors
    """
    logger.info("Results are checked.")

    if "Conv" in str(layer_params.layer_name):
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

    elif "Dense" in str(layer_params.layer_name):
        for f in range(len(ref_output)):
            if dram[f] != ref_output[f]:
                logger.error(f'Difference found at f = {f}')
                logger.error(f'ReferenceData: {str(ref_output[f])}')
                logger.error(f'Output Stream: {str(dram[f])}')
                return False
            
    elif "Pooling" in str(layer_params.layer_name):
        for f in range(len(ref_output)):
            if dram[f] != ref_output[f]:
                logger.error(f'Difference found at f = {f}')
                logger.error(f'ReferenceData: {str(ref_output[f])}')
                logger.error(f'Output Stream: {str(dram[f])}')
                return False

    return True

def compare_dram_with_ref_mp(f, ref_output, dram, return_dict):
    """Compare a single feature map's DRAM output with reference data.
    
    Helper function for parallel verification of convolutional layer outputs.
    Compares one feature map's worth of data, checking for exact matches
    at each spatial position.
    
    Args:
        f: Feature map index being compared
        ref_output: Reference data for this feature map
        dram: DRAM data for this feature map
        return_dict: Multiprocessing dictionary to store results
        
    Implementation:
        - Compares each spatial position (x,y) within the feature map
        - Sets return_dict[f] = True initially
        - Sets to False and returns early if any mismatch is found
        - Provides detailed error logging of mismatches
    """
    return_dict[f] = True
    for x in range(len(ref_output)):
        for y in range(len(ref_output[x])):
            if dram[x][y] != ref_output[x][y]:
                logger.error(f'Difference found at f = {f}, x = {x}, y= {y}')
                logger.error(f'ReferenceData: {str(ref_output[x][y])}')
                logger.error(f'Output Stream: {str(dram[x][y])}')
                return_dict[f] = False
                return
          
def fill_dram_with_ref(ref_output, dram, current_layer_params, next_layer_params):
    """Fill DRAM with reference output data for testing.
    
    This function copies reference output data into a DRAM object for
    testing and verification purposes. Handles different layer types
    with appropriate data organization.
    
    Args:
        ref_output: Reference data to copy into DRAM
        dram: Target DRAM object to fill with data
        layer_params: Layer parameters specifying the data format
        
    Returns:
        dram: DRAM object filled with reference data
        
    Implementation:
    - For convolutional layers:
        - 3D data organization (features x width x height)
        - Nested iteration for complete data copy
    - For pooling layers:
        - Direct feature map assignment
        - 1D data organization
        
    Error Handling:
    - Supports both serial and parallel architectures
    - Maintains data precision and format
    - Preserves layer-specific data organization
    """
    logger.info("Results are transmitted.")
    if "Conv" in str(current_layer_params.layer_name):
        if any(x in str(next_layer_params.layer_name) for x in ("Conv", "Pooling")):
            for f in range(len(ref_output)):    
                for x in range(len(ref_output[f])):
                    for y in range(len(ref_output[f][x])):
                        dram[f][x][y] = ref_output[f][x][y]
        else:
            if any(x in str(next_layer_params.layer_name) for x in ("Dense")):
                for f in range(len(ref_output)):  
                    for x in range(len(ref_output[f])):
                        for y in range(len(ref_output[f][x])):
                            dram[x+current_layer_params.iact_size_x*y+current_layer_params.iact_size_x*current_layer_params.iact_size_y*f] = ref_output[f][x][y]
    elif "Pooling" in str(current_layer_params.layer_name):
        for f in range(len(ref_output)):    
            for x in range(len(ref_output[f])):
                for y in range(len(ref_output[f][x])):
                    dram[f][x][y] = ref_output[f][x][y]
    elif "Dense" in str(current_layer_params.layer_name):
        for f in range(len(ref_output)):  
            dram[f] = ref_output[f]
    return dram