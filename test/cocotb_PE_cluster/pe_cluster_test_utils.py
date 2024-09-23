# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import os
import math

import logging
logger = logging.getLogger("test_logger")

def make_vh_file(params):
    filename = 'demo/parameters.vh'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    txt_file = open(filename, 'w')
    txt_file.write("parameter PARALLEL_MACS  = "  + str(params.PARALLEL_MACS)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter DATA_IACT_BITWIDTH  = "  + str(params.IACT_Bitwidth)+ ";\n")
    txt_file.write("parameter DATA_PSUM_BITWIDTH  = "  + str(params.PSUM_Bitwidth)+ ";\n")
    txt_file.write("parameter DATA_WGHT_BITWIDTH  = "  + str(params.WGHT_Bitwidth)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter TRANS_BITWIDTH_IACT  = "  + str(params.IACT_Trans_Bitwidth)+ ";\n")
    txt_file.write("parameter TRANS_BITWIDTH_PSUM  = "  + str(params.PSUM_Trans_Bitwidth)+ ";\n")
    txt_file.write("parameter TRANS_BITWIDTH_WGHT = "  + str(params.WGHT_Trans_Bitwidth)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter NUM_GLB_IACT  = "  + str(params.NUM_GLB_IACT)+ ";\n")
    txt_file.write("parameter NUM_GLB_PSUM  = "  + str(params.NUM_GLB_PSUM)+ ";\n")
    txt_file.write("parameter NUM_GLB_WGHT = "  + str(params.NUM_GLB_WGHT)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter PE_ROWS  = "  + str(params.PEs_Y)+ ";\n")
    txt_file.write("parameter PE_COLUMNS  = "  + str(params.PEs_X)+ ";\n")
    txt_file.write("parameter PES = "  + str(params.PEs_Y * params.PEs_X)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter CLUSTER_ROWS  = "  + str(params.Clusters_Y)+ ";\n")
    txt_file.write("parameter CLUSTER_COLUMNS  = "  + str(params.Clusters_X)+ ";\n")
    txt_file.write("parameter CLUSTERS = "  + str(params.Clusters_Y * params.Clusters_X)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter IACT_PER_PE  = "  + str(params.Iacts_per_PE)+ ";\n")
    txt_file.write("parameter PSUM_PER_PE  = "  + str(params.Psums_per_PE)+ ";\n")
    txt_file.write("parameter WGHT_PER_PE = "  + str(params.Wghts_per_PE)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter IACT_ADDR_PER_PE  = "  + str(params.Iacts_Addr_per_PE)+ ";\n")
    txt_file.write("parameter WGHT_ADDR_PER_PE  = "  + str(params.Wghts_Addr_per_PE)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter IACT_MEM_ADDR_WORDS  = "  + str(params.Iact_Mem_Addr_Words)+ ";\n")
    txt_file.write("parameter IACT_MEM_ADDR_BITS  = $clog2(IACT_MEM_ADDR_WORDS);\n")
    txt_file.write("parameter IACT_FSM_CYCL_WORDS = IACT_PER_PE + IACT_ADDR_PER_PE;\n")
    txt_file.write("parameter WGHT_FSM_CYCL_WORDS = WGHT_PER_PE + WGHT_ADDR_PER_PE;\n")
    txt_file.write("parameter PSUM_MEM_ADDR_WORDS  = "  + str(params.Psum_Mem_Addr_Words)+ ";\n")
    txt_file.write("parameter PSUM_MEM_ADDR_BITS  = $clog2(PSUM_MEM_ADDR_WORDS);\n")
    txt_file.write("\n")
    txt_file.write("parameter ROUTER_MODES_IACT  = "  + str(params.Router_Modes_IACT)+ ";\n")
    txt_file.write("parameter ROUTER_MODES_WGHT  = "  + str(params.Router_Modes_WGHT)+ ";\n")
    txt_file.write("parameter ROUTER_MODES_PSUM = "  + str(params.Router_Modes_PSUM)+ ";\n")
    txt_file.write("\n")
    txt_file.write("parameter DMA_BITWIDTH  = "  + str(params.DMA_Bits)+ ";\n")
    txt_file.write("parameter FSM_CYCLE_BITWIDTH  = "  + str(params.FSM_CYCLE_BITWIDTH)+ ";\n")
    txt_file.write("parameter FSM_STATES = "  + str(params.FSM_STATES)+ ";\n")
    txt_file.close()

class OpenEyeParameters():
    """ Parameter class for the OpenEye accelerator.
    
    This class contains all the parameters for the OpenEye accelerator that
    are fixed before the accelerator is synthesized.
    
    """
    def __init__(self):
        self.PARALLEL_MACS = 2

        self.IACT_Bitwidth = 8
        self.WGHT_Bitwidth = 8
        self.IACT_WOH_Bitwidth = self.IACT_Bitwidth + 4
        self.WGHT_WOH_Bitwidth = self.WGHT_Bitwidth + 4
        self.IACT_Addr_Bitwidth = 4
        self.WGHT_Addr_Bitwidth = 7
        self.PSUM_Bitwidth = 20
        self.IACT_Trans_Bitwidth = 24
        self.WGHT_Trans_Bitwidth = 24
        self.PSUM_Trans_Bitwidth = 20 * self.PARALLEL_MACS
        self.PEs_X = 4
        self.PEs_Y = 3
        self.NUM_GLB_IACT = 3
        self.NUM_GLB_PSUM = 4
        self.NUM_GLB_WGHT = 3
        self.Clusters_X = 2
        self.Clusters_Y = 8
        self.PEs = self.PEs_X * self.PEs_Y
        self.Clusters = self.Clusters_X * self.Clusters_Y
        self.PE_Complete = self.PEs * self.Clusters
        self.Iacts_Addr_per_PE = 9
        self.Iacts_per_PE = 16
        self.Wghts_Addr_per_PE = 16
        self.Wghts_per_PE = 192
        self.Psums_per_PE  = 32
        self.Iact_Routers = 3
        self.Wght_Routers = self.PEs_Y
        self.Psum_Routers = self.PEs_X
        self.Data_mode = 1
        self.Autofunction = 1
        self.Poolingmode = 1
        
        self.Iact_Mem_Addr_Words = 1024
        self.Psum_Mem_Addr_Words = 1024

        self.Router_Modes_IACT = 1
        self.Router_Modes_WGHT = 1
        self.Router_Modes_PSUM = 1

        self.Poolingmode = 1
        self.Poolingmode = 1

        self.DMA_Bits = 48
        self.FSM_CYCLE_BITWIDTH = 1024
        self.FSM_STATES = 9
        self.Iact_Router_Bits = 6
        self.Wght_Router_Bits = 1
        self.Psum_Router_Bits = 2

class LayerParameters():
    """
    Parameter class for a single layer.
    
    It contains the parameters that are computed during the generation of the output files
    for subsequent usage and insepction.
    """

    def __init__(self):
        self.used_PEs_X = 1
        self.used_PEs_Y = 0
        self.Used_refreshes = 0
        self.current_input_X = 0
        self.current_input_Y = 0

        self.used_iact_per_PE = []
        self.used_wght_per_PE = []
        self.used_psum_per_PE = []
        self.diff_iact_layer = []
        self.ceil_used_PE_per_clm = 0

        self.needed_Iact_writes = 0

        self.current_highest_number = 0
        self.realfactor = 0

    def print_layer_parameters(self, debug_file):
        logger.info("params.PEs X: " + str(self.used_PEs_X) + "\n")
        logger.info("params.PEs Y: " + str(self.used_PEs_Y) + "\n")
        logger.info("Iact PE: " + str(self.used_iact_per_PE) + "\n")
        logger.info("Wght PE: " + str(self.used_wght_per_PE) + "\n")
        logger.info("Psum PE: " + str(self.used_psum_per_PE) + "\n")
        logger.info("Needed Cycles: " + str(self.Used_refreshes) + "\n")
        logger.info("Factor: " + str(self.current_highest_number) + "\n")
        logger.info("Real Factor: " + str(self.realfactor) + "\n")
        logger.info("End of Layer")

def partition_weights(tensor, partition_size):
    """ Partition the weights of a layer into smaller partitions."""
    pass

def partition_iacts(iacts, partition_size):
    """ Partition the weights of a layer into smaller partitions."""
    pass

def encode_sparse_tensor(m):
    """ Converts a (possibly) sparse tensor or matrix into a matrix for the adress SPads and a matrix for the data SPads.
    
    This uses CSR format to encode the sparse matrix.
    """
    pass

def get_verilog_sources(hdl_dir):

    verilog_sources = [
        os.path.join(hdl_dir, "PE_cluster.v"),
        os.path.join(hdl_dir, "RST_SYNC.v"),
        os.path.join(hdl_dir, "PE.v"),
        os.path.join(hdl_dir, "adder.v"),
        os.path.join(hdl_dir, "data_pipeline.v"),
        os.path.join(hdl_dir, "multiplier.v"),
        os.path.join(hdl_dir, "mux2.v"),
        os.path.join(hdl_dir, "demux2.v"),
        os.path.join(hdl_dir, "mux_iact.v"),
        os.path.join(hdl_dir, "SPad_DP_RW.v"),
        os.path.join(hdl_dir, "SPad_SP.v"),
        os.path.join(hdl_dir, "memory/RAM_DP_RW.v"),
        os.path.join(hdl_dir, "memory/RAM_DP.v"),
        os.path.join(hdl_dir, "memory/RAM_SP.v"),
        os.path.join(hdl_dir, "memory/impl/RAM_DP_RW_generic.v"),
        os.path.join(hdl_dir, "memory/impl/RAM_SP_generic.v"),
    ]
    return verilog_sources

def open_or_create_file(filepath):
    filename = filepath
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    open_file = open(filename, 'w')
    return open_file
    
def write_openeye_configuration(model, params, filename='demo/configuration_of_OpenEye.txt'):
    debug_file = open_or_create_file("TXTs/debug_file.txt")
    lps = []

    for i, layer in enumerate(model.layers):
        lp = LayerParameters()
        file_weights = open_or_create_file('TXTs/weights/l' + str(i) + '_complete.txt')
        file_bias = open_or_create_file('TXTs/bias/l' + str(i) + '_complete.txt')
        current_highest_number = 0
        
        if "Conv2D" in str(layer):
            write_conv2d_layer(params, lp, model.layers[i], file_weights, file_bias)
 
        elif "Flatten" in str(layer):
            debug_file.write("\nFlatten---")
            continue
            
        else:
            continue        
        
        write_router_reference_files(params, lp, debug_file)

        file_weights.close()
        file_bias.close()
        lps.append(lp)

    debug_file.close()
    return lps

def write_conv2d_layer(params, layer_params, layer, file_weights, file_bias):
    """ Write the weights and bias of a Conv2D layer to a file. """
    
    # Get the weights and bias of the layer
    for current_number_x in range(layer.kernel_size[0]):
        for current_number_y in range(layer.kernel_size[1]):
            for current_number_z in range(len(layer.weights[0][current_number_x][current_number_y])):
                for current_number_c in range(len(layer.weights[0][current_number_x][current_number_y][current_number_z])):
                    if(abs(layer.weights[0][current_number_x][current_number_y][current_number_z][current_number_c]) >= layer_params.current_highest_number):
                        layer_params.current_highest_number = abs(layer.weights[0][current_number_x][current_number_y][current_number_z][current_number_c])
                        
    # Write the weights to a file
    for current_number_x in range(layer.kernel_size[0]):
        for current_number_y in range(layer.kernel_size[1]):
            for current_number_z in range(len(layer.weights[0][current_number_x][current_number_y])):
                for current_number_c in range(len(layer.weights[0][current_number_x][current_number_y][current_number_z])):
                    file_weights.write(str(round(float(128 * layer.weights[0][current_number_x][current_number_y][current_number_z][current_number_c] / layer_params.current_highest_number))) + "\n")
                    
                    
    # Write the bias to a file
    for filter_number in range(layer.filters):
        file_bias.write(str(round(float(128 * layer.weights[1][filter_number]/layer_params.current_highest_number))) + "\n")
                    
    # Calculate the number of PEs needed for the layer
    if(layer.padding == "same"):
        calc_X = layer.input.shape[1]
        calc_Y = layer.input.shape[2]
        current_input_X = layer.input.shape[1] + layer.kernel_size[0] - 1
        current_input_Y = layer.input.shape[2] + layer.kernel_size[1] - 1
    else:
        calc_X = layer.input.shape[1] - layer.kernel_size[0] + 1
        calc_Y = layer.input.shape[2] - layer.kernel_size[1] + 1
        current_input_X = layer.input.shape[1]
        current_input_Y = layer.input.shape[2]
    layer_params.complete_computations = calc_X * calc_Y
        
    layer_params.needed_Iact_writes = math.ceil((params.PEs_X + layer.kernel_size[1] - 1) /(params.Iact_Routers))
    
    layer_params.realfactor = math.floor(abs(math.log2(abs(layer_params.current_highest_number))))
    # Calculate the number of refreshes needed for the layer
    if((layer.kernel_size[0] <= params.Iacts_per_PE) & (layer.kernel_size[1] <= params.PEs_Y * params.Clusters_Y)):
        
        if(layer.input.shape[3] * layer.kernel_size[0] <= params.Iacts_per_PE):
            layer_params.used_iact_per_PE = layer.input.shape[3] * layer.kernel_size[0]
            layer_params.diff_iact_layer  = 1
        else:
            logger.debug("IactWert: ")
            possible_iacts_in_PE = math.trunc(params.Iacts_per_PE/layer.kernel_size[0])
            
            layer_params.diff_iact_layer = math.ceil(layer.input.shape[3]/possible_iacts_in_PE)
            layer_params.used_iact_per_PE[0] = layer.kernel_size[0] * math.ceil(layer.input.shape[3]/layer_params.diff_iact_layer)
            
        layer_params.used_wght_per_PE = layer_params.used_iact_per_PE
        if(layer.filters * layer_params.used_iact_per_PE <= params.Wghts_per_PE):
            layer_params.used_wght_per_PE = layer.filters*layer_params.used_iact_per_PE
            layer_params.used_psum_per_PE = layer.filters
        else:
            logger.debug("#Function, caculate if not working---2\n")
            
        #Calculation of seperate PE-Cluster
        layer_params.used_PEs_Y    = layer.kernel_size[1]
        used_PEs_per_clm     = layer_params.used_PEs_Y/params.PEs_Y
        layer_params.ceil_used_PE_per_clm = math.ceil(used_PEs_per_clm)
        efficiency1 = (used_PEs_per_clm / layer_params.ceil_used_PE_per_clm)
        logger.debug("Efficiency1: " + str(100 * efficiency1) + "%")
        
        PE_Blocks_Usage      = params.Clusters_Y / layer_params.ceil_used_PE_per_clm
        trunc_PE_Blocks_Usage = math.trunc(PE_Blocks_Usage)
        efficiency2 = trunc_PE_Blocks_Usage * layer_params.ceil_used_PE_per_clm / params.Clusters_Y
        logger.debug("Efficiency2: " + str(100 * efficiency2) + "%")
        
        PE_clmn_usage = layer_params.complete_computations/(PE_Blocks_Usage * params.Clusters_X * params.PEs_X)
        trunc_PE_clmn_usage = math.trunc(PE_clmn_usage)
        efficiency3 = trunc_PE_clmn_usage/PE_clmn_usage
        logger.debug("Efficiency3: " + str(100 * efficiency3) + "%")
        
        logger.debug("Efficiency Complete: " + str(100 * efficiency1 * efficiency2 * efficiency3) + "%")
        layer_params.Used_refreshes = math.ceil(PE_clmn_usage) * layer_params.diff_iact_layer
        logger.debug("Used complete new descriptions: " + str(layer_params.Used_refreshes))
    else:
        logger.error("Can't fit model, kernel size must be adjusted.")
    return

def write_router_reference_files(params, layer_params, file_dma):
    file_i_router = open_or_create_file('TXTs/router/l' + str(0) + '_Iact' + '.txt')
    file_w_router = open_or_create_file('TXTs/router/l' + str(0) + '_Wght' + '.txt')
    file_p_router = open_or_create_file('TXTs/router/l' + str(0) + '_Psum' + '.txt')
    
    router_cycle = 0
    
    for cl_x in range(params.Clusters_X):
        for cl_y in range(params.Clusters_Y):
            for router in range(params.Iact_Routers):
                if((layer_params.ceil_used_PE_per_clm == 1) & (layer_params.used_PEs_Y == 1)):
                    file_i_router.write("000001\n")
                else:
                    if(layer_params.used_PEs_Y > 1):
                        if((cl_y % layer_params.ceil_used_PE_per_clm) == 0):
                            file_i_router.write("001001\n")
                        else:
                            if((cl_y % layer_params.ceil_used_PE_per_clm) + 1 == layer_params.ceil_used_PE_per_clm):
                                file_i_router.write("100001\n")
                            else:
                                file_i_router.write("101001\n")
                    else:
                        if(cl_y == 0):
                            file_i_router.write("000011\n")
                        else:
                            file_i_router.write("100001\n")
                router_cycle = router_cycle + 1
                if(router_cycle == math.floor(params.DMA_Bits/params.Iact_Router_Bits)):
                    router_cycle = 0
                    
    if(router_cycle != 0):
        router_cycle = 0
                
            
    for cl_x in range(params.Clusters_X):
        for cl_y in range(params.Clusters_Y):   
            for router in range(params.Wght_Routers):
                if(False):
                    file_w_router.write("0\n")
                else:
                    if(cl_x == 0):
                        file_w_router.write("0\n")
                    else:
                        file_w_router.write("1\n")
                router_cycle = router_cycle + 1
                if(router_cycle == math.floor(params.DMA_Bits/params.Wght_Router_Bits)):
                    router_cycle = 0
                        
    if(router_cycle != 0):
        router_cycle = 0
                
                
    for cl_x in range(params.Clusters_X):
        for cl_y in range(params.Clusters_Y):
            for router in range(params.Psum_Routers):
                if((layer_params.ceil_used_PE_per_clm == 1)):
                    file_p_router.write("00\n")
                else:
                    if((cl_y % layer_params.ceil_used_PE_per_clm) == 0):
                        file_p_router.write("01\n")
                    else:
                        if((cl_y % layer_params.ceil_used_PE_per_clm) + 1 == layer_params.ceil_used_PE_per_clm):
                            file_p_router.write("11\n")
                        else:
                            file_p_router.write("10\n")
                router_cycle = router_cycle + 1
                if(router_cycle == math.floor(params.DMA_Bits/params.Psum_Router_Bits)):
                    router_cycle = 0
    if(router_cycle != 0):
        router_cycle = 0
                
    file_i_router.close()
    file_w_router.close()
    file_p_router.close()

def write_input_data_file(Iact_storage, x_dim, y_dim):        
    #Create Inputdata
    for x in range(x_dim):
        for y in range(y_dim):
            Iact_storage.data[x][y] = ((x/64 - y/64))
            
def write_working_parameters(params, layer_params, file_dma):
    dma_line = 0
    dma_line = params.Data_mode + ((layer_params.realfactor) << 1) 
    dma_line = dma_line + (params.Autofunction << 6) + (params.Poolingmode << 7)
    dma_line = dma_line + (layer_params.Used_refreshes << 8) + ((math.ceil(layer_params.used_PEs_X/params.PEs_X)) << 16)
    dma_line = dma_line + ((math.ceil(layer_params.used_PEs_Y/params.PEs_Y)) << 18)
    dma_line = dma_line + (math.ceil(((math.ceil(layer_params.used_PEs_Y/params.PEs_Y))*params.PEs_Y + params.PEs_X - 1)/params.NUM_GLB_IACT) << 22)
    dma_line = dma_line + ((layer_params.used_psum_per_PE)<< 25)
    file_dma.write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
    return dma_line

def write_router_iact(params, layer_params, file_dma):
    dma_line = 0
    dma_storage = []
    router_cycle = 0
    
    for cl_x in range(params.Clusters_X):
        for cl_y in range(params.Clusters_Y):
            for router in range(params.Iact_Routers):
                if((layer_params.ceil_used_PE_per_clm == 1) & (layer_params.used_PEs_Y == 1)):
                    dma_line = dma_line + (1 << (params.Iact_Router_Bits * router_cycle))
                else:
                    if(layer_params.used_PEs_Y > 1):
                        if((cl_y % layer_params.ceil_used_PE_per_clm) == 0):
                            dma_line = dma_line + (9 << (params.Iact_Router_Bits * router_cycle))
                        else:
                            if((cl_y % layer_params.ceil_used_PE_per_clm) + 1 == layer_params.ceil_used_PE_per_clm):
                                dma_line = dma_line + (33 << (params.Iact_Router_Bits * router_cycle))
                            else:
                                dma_line = dma_line + (41 << (params.Iact_Router_Bits * router_cycle))
                    else:
                        if(cl_y == 0):
                            dma_line = dma_line + (3 << (params.Iact_Router_Bits * router_cycle))
                        else:
                            dma_line = dma_line + (33 << (params.Iact_Router_Bits * router_cycle))
                router_cycle = router_cycle + 1
                if(router_cycle == math.floor(params.DMA_Bits/params.Iact_Router_Bits)):
                    router_cycle = 0
                    file_dma.write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
                    dma_storage.append(dma_line)
                    dma_line = 0
                    
                    
    if(router_cycle != 0):
        router_cycle = 0
        file_dma.write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
        dma_storage.append(dma_line)
        dma_line = 0
    return dma_storage

def write_router_wght(params, file_dma):
    dma_line = 0             
    dma_storage = []
    router_cycle = 0        
    for cl_x in range(params.Clusters_X):
        for cl_y in range(params.Clusters_Y):   
            for router in range(params.Wght_Routers):
                if(cl_x == 0):
                    dma_line = dma_line + (0 << (params.Wght_Router_Bits * router_cycle))
                else:
                    dma_line = dma_line + (1 << (params.Wght_Router_Bits * router_cycle))
                router_cycle = router_cycle + 1
                if(router_cycle == math.floor(params.DMA_Bits/params.Wght_Router_Bits)):
                    router_cycle = 0
                    file_dma.write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
                    dma_storage.append(dma_line)
                    dma_line = 0
                        
    if(router_cycle != 0):
        router_cycle = 0
        file_dma.write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
        dma_storage.append(dma_line)
        dma_line = 0

    return dma_storage

def write_router_psum(params, layer_params, file_dma):
    dma_line = 0             
    dma_storage = []
    router_cycle = 0          
    for cl_x in range(params.Clusters_X):
        for cl_y in range(params.Clusters_Y):
            for router in range(params.Psum_Routers):
                if((layer_params.ceil_used_PE_per_clm == 1)):
                    dma_line = dma_line + (0 << (params.Psum_Router_Bits * router_cycle))
                else:
                    if((cl_y % layer_params.ceil_used_PE_per_clm) == 0):
                        dma_line = dma_line + (1 << (params.Psum_Router_Bits * router_cycle))
                    else:
                        if((cl_y % layer_params.ceil_used_PE_per_clm) + 1 == layer_params.ceil_used_PE_per_clm):
                            dma_line = dma_line + (3 << (params.Psum_Router_Bits * router_cycle))
                        else:
                            dma_line = dma_line + (2 << (params.Psum_Router_Bits * router_cycle))
                router_cycle = router_cycle + 1
                if(router_cycle == math.floor(params.DMA_Bits/params.Psum_Router_Bits)):
                    router_cycle = 0
                    file_dma.write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
                    dma_storage.append(dma_line)
                    dma_line = 0
                    
    if(router_cycle != 0):
        router_cycle = 0
        file_dma.write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
        dma_storage.append(dma_line)
        dma_line = 0    

    return dma_storage

def write_iact_data(params, layer_params, layer, Inputdata, file_dma):
    dma_line = 0             
    dma_storage = []
    iact_temp_pos_x = 0
    iact_temp_pos_y = 0
    channel = 0
    overhead_counter = 0
    overhead_counter_before_x_cl = 0
    
    for cl_y in range(params.Clusters_Y):
        for router in range(params.Iact_Routers):
            for cycle in range(layer_params.Used_refreshes):
                for iact_cycle in range(layer_params.needed_Iact_writes):
                    for dma_part_data_num in range(math.ceil(params.Iacts_Addr_per_PE/3)):
                        for cl_x in range(params.Clusters_X):
                            for txt_row in range(params.Iacts_Addr_per_PE):
                                if((cl_y % layer_params.ceil_used_PE_per_clm) == 0):
                                    if(txt_row + 3 * dma_part_data_num < (math.ceil(layer_params.used_iact_per_PE/layer.kernel_size[0]))):
                                        dma_line = dma_line + (int(layer.kernel_size[0]) << (params.IACT_Addr_Bitwidth * txt_row + params.IACT_Addr_Bitwidth * 6 * cl_x))
                                    else:
                                        dma_line = dma_line
                                        
                                else:
                                    dma_line = 0
                            
                        file_dma.write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
                        dma_storage.append(dma_line)
                        dma_line = 0
                        
                    for dma_part_data_num in range(math.ceil(params.Iacts_per_PE/2)):
                        for cl_x in range(params.Clusters_X):
                            for txt_row in range(2):
                                if((cl_y % layer_params.ceil_used_PE_per_clm) == 0) & ((txt_row + dma_part_data_num * 2) < layer_params.used_iact_per_PE):
                                    iact_temp_pos_x = int(math.floor(cl_x * params.PEs_X + cl_y * params.PEs_X * params.Clusters_X + cycle * params.PEs_X * params.Clusters / layer_params.ceil_used_PE_per_clm) %\
                                                        (layer.output.shape[2])+\
                                                        router + iact_cycle * params.Iact_Routers)
                                    iact_temp_pos_y = int(((txt_row + dma_part_data_num * 2) + \
                                                        ((params.Clusters_X * params.PEs_X) * cl_y + cl_x * params.PEs_X + cycle *params.PEs_X * params.Clusters)/layer.output.shape[2]/ layer_params.ceil_used_PE_per_clm))
                                    if iact_temp_pos_x < (layer.input.shape[1] + layer.kernel_size[1] - 1):
                                        if iact_temp_pos_y < (layer.input.shape[1] + layer.kernel_size[1] - 1):
                                            if(((cycle * 8 * 2 * 4) + (cl_y * 2 * 4) + (cl_x * 4)) < (28 * 28)):
                                                if(0 > float(Inputdata[iact_temp_pos_x][iact_temp_pos_y])):
                                                    dma_line = dma_line + ((2**(params.IACT_Bitwidth) - abs(int(math.floor(float(Inputdata[iact_temp_pos_x][iact_temp_pos_y])))) + \
                                                            (2**(params.IACT_Bitwidth)) * overhead_counter) \
                                                            << ((params.IACT_WOH_Bitwidth) * (txt_row + cl_x*2)))
                                                else:
                                                    dma_line = dma_line + (int(math.floor(float(Inputdata[iact_temp_pos_x][iact_temp_pos_y])) + \
                                                            (2**(params.IACT_Bitwidth)) * overhead_counter) \
                                                            << ((params.IACT_WOH_Bitwidth) * (txt_row + cl_x*2)))
                                                overhead_counter = overhead_counter + 1
                                            else:
                                                dma_line = dma_line

                                else:
                                    dma_line = dma_line
                            overhead_counter = overhead_counter_before_x_cl
                        overhead_counter_before_x_cl = overhead_counter_before_x_cl + 2
                        overhead_counter = overhead_counter + 2

                        file_dma.write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
                        dma_storage.append(dma_line)
                        dma_line = 0
                    overhead_counter = 0
                    overhead_counter_before_x_cl = 0
    return dma_storage

def write_wght_data(params, layer_params, layer, file_dma):
    dma_line = 0             
    dma_storage = []
    overhead_counter = 0
    overhead_counter_before_x_cl = 0
    if "Conv2D" in str(layer):
        for cl_y in range(params.Clusters_Y):
            for router in range(params.Wght_Routers):
                filters = 0
                kernel_x = 0
                channel = 0
                
                for dma_part_data_num in range(math.ceil(params.Wghts_Addr_per_PE/math.floor(params.WGHT_Trans_Bitwidth/params.WGHT_Addr_Bitwidth))):
                    for cl_x in range(params.Clusters_X):
                        for txt_row in range(3):
                            if((txt_row + dma_part_data_num * 3)< (math.ceil(layer.kernel_size[0]) + 1)):
                                dma_line = dma_line + (int((txt_row + dma_part_data_num * 3) * math.ceil(layer_params.used_wght_per_PE/layer.kernel_size[0]/2))  << (params.WGHT_Addr_Bitwidth * (txt_row + cl_x * 3)))
                            else:
                                dma_line = dma_line
                    
                    for write_time in range(int(params.WGHT_Trans_Bitwidth/params.WGHT_Addr_Bitwidth)):
                        
                        if(dma_part_data_num != (math.ceil(params.Wghts_Addr_per_PE/(params.WGHT_Trans_Bitwidth/params.WGHT_Addr_Bitwidth)) - 1)) | (write_time == 0):
                            file_dma.write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
                            dma_storage.append(dma_line)
                        dma_line = 0
                            
                for dma_part_data_num in range(int(params.Wghts_per_PE/(params.WGHT_Trans_Bitwidth/params.WGHT_WOH_Bitwidth))):
                    Mtrx_Row = (cl_y % layer_params.ceil_used_PE_per_clm) * params.PEs_Y + router
                    if(Mtrx_Row < layer.kernel_size[1]):
                        if(channel == layer.input.shape[3]):
                            dma_line = 0
                            logger.debug(str(dma_part_data_num) + " CHANNEL\n")
                        else:
                            for txt_row in range(2):
                                if(channel == layer.input.shape[3]):
                                    dma_line = dma_line
                                else:
                                    for cl_x in range(params.Clusters_X):
                                        if(0 > int(round(float(layer.weights[0][Mtrx_Row][kernel_x][channel][filters]) * (2 **(params.WGHT_Bitwidth))))):
                                            dma_line = dma_line + \
                                            (2**(params.WGHT_Bitwidth) - abs(int(round(float((2**(params.WGHT_Bitwidth - 1)) * layer.weights[0][Mtrx_Row][kernel_x][channel][filters])))) \
                                                << ((params.WGHT_WOH_Bitwidth) * (txt_row + cl_x * 2)))
                                        else:
                                            dma_line = dma_line + \
                                            ((int(round(float((2**(params.WGHT_Bitwidth - 1))*layer.weights[0][Mtrx_Row][kernel_x][channel][filters])) \
                                                << ((params.WGHT_WOH_Bitwidth) * (txt_row + cl_x * 2)))))

                                    overhead_counter = overhead_counter + 1
                                    filters = filters + 1
                                    if(filters == layer.filters):
                                        filters = 0
                                        kernel_x = kernel_x + 1
                                        overhead_counter = 0
                                    if(kernel_x == layer.kernel_size[0]):
                                        kernel_x = 0
                                        channel = channel + 1
                    else:
                        dma_line = 0
                    file_dma.write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
                    dma_storage.append(dma_line)
                    dma_line = 0
    return dma_storage

def write_psum_data(params, layer_params, layer, file_dma):
    dma_line = 0             
    dma_storage = []
    for cl_y in range(params.Clusters_Y):
        for router in range(params.Psum_Routers):
            for cycle in range(layer_params.Used_refreshes):
                for dma_part_data_num in range(layer.filters):
                    if "Conv2D" in str(layer):
                        if((cl_y % layer_params.ceil_used_PE_per_clm) == 0):

                            if(dma_part_data_num < layer_params.used_psum_per_PE):
                                for cl_x in range(params.Clusters_X):
                                    if(0 > float(layer.bias[dma_part_data_num])):
                                        dma_line = dma_line + ((2**(params.PSUM_Bitwidth) - abs(int(round(float((2**(params.IACT_Bitwidth + params.WGHT_Bitwidth - 1)) * layer.bias[dma_part_data_num]))))) << (params.PSUM_Bitwidth * cl_x))
                                    else:
                                        dma_line = dma_line + (int(round(float((2**(params.IACT_Bitwidth + params.WGHT_Bitwidth - 1)) * layer.bias[dma_part_data_num]))) << (params.PSUM_Bitwidth * cl_x))
                            else:
                                dma_line = 0

                        file_dma.write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
                        dma_storage.append(dma_line)
                        dma_line = 0

    return dma_storage

def write_dma(params, layer_params, layer, Inputdata):
    file_dma = open_or_create_file('demo/dma_stream.txt')
    dma_storage = []

    dma_storage.append(write_working_parameters(params, layer_params, file_dma))
    
    iact_router_list = write_router_iact(params, layer_params, file_dma)
    for x in range(len(iact_router_list)):
        dma_storage.append(iact_router_list[x])

    wght_router_list = write_router_wght(params, file_dma)
    for x in range(len(wght_router_list)):
        dma_storage.append(wght_router_list[x])

    psum_router_list = write_router_psum(params, layer_params, file_dma)
    for x in range(len(psum_router_list)):
        dma_storage.append(psum_router_list[x])

    iact_data_list = write_iact_data(params, layer_params, layer, Inputdata, file_dma)
    for x in range(len(iact_data_list)):
        dma_storage.append(iact_data_list[x])

    wght_data_list = write_wght_data(params, layer_params, layer, file_dma)
    for x in range(len(wght_data_list)):
        dma_storage.append(wght_data_list[x])

    psum_data_list = write_psum_data(params, layer_params, layer, file_dma)
    for x in range(len(psum_data_list)):
        dma_storage.append(psum_data_list[x])

    file_dma.close()
    return dma_storage
    
def make_ref(params, layer_params, layer, Inputdata):
    file_dma_ref = open_or_create_file('demo/dma_stream_ref.txt')
    iact_ref = open_or_create_file('demo/iact_ref.txt')
    weight_ref = [0 for i in range(layer.filters)]
    for filter_number in range(layer.filters):
        weight_ref[filter_number] = open_or_create_file('demo/weight_ref' + str(filter_number) + '.txt')

    for filter_number in range(layer.kernel.shape[3]):
        for y in range(layer.kernel_size[0]):
            for x in range(layer.kernel_size[1]):
                weight_ref[filter_number].write(str(int(round(float(128*layer.weights[0][x][y][0][filter_number])))) + " ")
            weight_ref[filter_number].write("\n")
        weight_ref[filter_number].close()

    for y in range(30):
        for x in range(30):
            iact_ref.write(str(int(round(float(Inputdata[x][y])))) + " ")
        iact_ref.write("\n")
        
    iact_ref.close()
    dma_line = 0
    calculated_results = [[[0 for i in range(28)] for j in range(28)]for k in range(32)]
    
    for j in range(28):
        for i in range(28):
            for k in range(layer.filters):
                if(k < layer.kernel.shape[3]):
                    for x in range(layer.kernel_size[0]):
                        for y in range(layer.kernel_size[1]):
                            calculated_results[k][i][j] = int(calculated_results[k][i][j] + int(round(float(128*layer.weights[0][x][y][0][k]))) * int(math.floor(float(Inputdata[x + i][y + j]))))

                            if((k == 1) & (j == 0) & (i == 0)):
                                logger.debug("Result currently:" + str(calculated_results[k][i][j]) + "\n")
                    calculated_results[k][i][j] = int(calculated_results[k][i][j] + int(layer.bias[k]))
                    if(calculated_results[k][i][j] <= 0):
                        calculated_results[k][i][j] = int(0)

    for refresh in range(layer_params.Used_refreshes):
        for cl_y in range(params.Clusters_Y):
            for cl_x in range(params.Clusters_X):
                for router in range(params.Psum_Routers):
                    for psum_pe in range(math.ceil(layer.filters/2)):
                        for counter in range(math.ceil(2)):  #Flexibel machen
                            x_cor= int(((router + cl_x * params.PEs_X + cl_y * params.Clusters_X * params.PEs_X + refresh * params.Clusters_Y * params.Clusters_X * params.PEs_X ) % layer.output.shape[2]))
                            y_cor= int(((router + cl_x * params.PEs_X + cl_y * params.Clusters_X * params.PEs_X + refresh * params.Clusters_Y * params.Clusters_X * params.PEs_X ) / layer.output.shape[2]))

                            if((x_cor < 28) & (y_cor < 28)):
                                if(calculated_results[2 * psum_pe + counter][x_cor][y_cor] >= 0):
                                    dma_line = dma_line + (calculated_results[2 * psum_pe + counter][x_cor][y_cor] << (params.PSUM_Bitwidth * counter))
                                else:
                                    dma_line = dma_line

                        file_dma_ref.write(bin(dma_line)[2:].zfill(params.DMA_Bits) + "\n")
                        dma_line = 0
                    
    file_dma_ref.close()

def check_results(file_1,file_2):
    # Open the two files in read-only mode
    with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
        # Read the contents of the two files into two lists
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # Compare the two lists line by line and print any differences
    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        if line1 != line2:
            logger.error(f'Difference found at line {i + 1}:')
            logger.error(f'File 1: {line1.strip()}')
            logger.error(f'File 2: {line2.strip()}')