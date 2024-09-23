# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import sys
import os
directory = (os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
import math
import logging
import test_utils.stream_dicts as strdic
from test_utils.layer_mapper import LayerMapper
from test_utils.iact_stream_mapper import DenseIactStreamMapper
from test_utils.wght_stream_mapper import DenseWghtStreamMapper
from test_utils.psum_stream_mapper import DensePsumStreamMapper

logger = logging.getLogger("cocotb")

class DenseMapper(LayerMapper):
    
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        input_mapper = DenseIactStreamMapper(params, layer_params, layer_repetition, dram_layer_content[0])
        weight_mapper = DenseWghtStreamMapper(params, layer_params, layer_repetition, dram_layer_content[1])
        bias_mapper = DensePsumStreamMapper(params, layer_params, layer_repetition, dram_layer_content[2])
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, input_mapper, weight_mapper, bias_mapper)

    def write_working_parameters(self, params, layer_params, layer_repetition):
        if (params.SERIAL):
            storage = [[] for b in range(len(strdic.stream_serial_dict))]
        else:
            storage = [[] for b in range(len(strdic.status_dict))]
        needed_refreshes = 1
        if ((layer_repetition % layer_params.needed_iact_transmissions) == 0) :
            layer_params.skipPsum = 0
        else :
            layer_params.skipPsum = 1
        
        counter = 0
        computing_pes = 0

        for x in range(params.Clusters_X):
            for y in range(params.Clusters_Y):
                for pe_y in range(params.PEs_Y):
                    for pe_x in range(params.PEs_X):
                        if(layer_params.computing_mx[x][y][pe_y][pe_x]== 1):
                            computing_pes = computing_pes + 2**(counter)
                        counter = counter + 1
        computing_pes = format(computing_pes, "0192b")
        if (params.SERIAL):
            dma_line = 0
            dma_storage = []
            dma_line = params.data_mode + ((layer_params.realfactor) << 1) 
            dma_line = dma_line + (params.autofunction << 6)
            dma_line = dma_line + (params.poolingmode << 7)
            dma_line = dma_line + ((needed_refreshes << 8))
            dma_line = dma_line + (layer_params.used_X_cluster << 16)
            dma_line = dma_line + (layer_params.used_Y_cluster << 18)
            dma_line = dma_line + (layer_params.needed_Iact_writes << 22)
            dma_line = dma_line + (layer_params.used_psum_per_PE << 26)
            dma_line = dma_line + (layer_params.used_iact_addr_per_PE << 32)
            dma_line = dma_line + (layer_params.used_wght_addr_per_PE << 36)
            dma_line = dma_line + (layer_params.used_iact_per_PE << 41)
            dma_storage.append(dma_line)
            dma_line = 0
            dma_line = dma_line + (layer_params.iact_addr_len)
            dma_line = dma_line + (layer_params.iact_data_len << 2)
            dma_line = dma_line + (layer_params.strideX << 6)
            dma_line = dma_line + (layer_params.strideY << 10)
            if ((layer_repetition % layer_params.needed_wght_transmissions) == 0):
                dma_line = dma_line + (0 << 14)
            else:
                dma_line = dma_line + (1 << 14)
            dma_line = dma_line + (layer_params.skipWght << 15)
            dma_line = dma_line + (layer_params.skipPsum << 16)
            dma_storage.append(dma_line)

            for x in reversed(range(4)):
                dma_line = int(computing_pes[x*48:(x+1)*48],2)
                dma_storage.append(dma_line)
            storage = dma_storage
        else:
            storage[strdic.status_dict["data_mode"]] = params.data_mode
            storage[strdic.status_dict["realfactor"]] = layer_params.realfactor
            storage[strdic.status_dict["autofunction"]] = params.autofunction
            storage[strdic.status_dict["poolingmode"]] = params.poolingmode
            storage[strdic.status_dict["psum_delay"]] = layer_params.psum_delay
            storage[strdic.status_dict["needed_refreshes"]] = needed_refreshes
            storage[strdic.status_dict["used_X_cluster"]] = layer_params.used_X_cluster
            storage[strdic.status_dict["used_Y_cluster"]] = layer_params.used_Y_cluster
            storage[strdic.status_dict["needed_Iact_writes"]] = layer_params.needed_Iact_writes
            storage[strdic.status_dict["used_psum_per_PE"]] = layer_params.used_psum_per_PE
            storage[strdic.status_dict["used_iact_addr_per_PE"]] = layer_params.used_iact_addr_per_PE
            storage[strdic.status_dict["used_wght_addr_per_PE"]] = layer_params.used_wght_addr_per_PE
            storage[strdic.status_dict["used_iact_per_PE"]] = layer_params.used_iact_per_PE
            storage[strdic.status_dict["iact_addr_len"]] = layer_params.iact_addr_len
            storage[strdic.status_dict["iact_data_len"]] = layer_params.iact_data_len
            storage[strdic.status_dict["strideX"]] = layer_params.strideX
            storage[strdic.status_dict["strideY"]] = layer_params.strideY
            storage[strdic.status_dict["skipIact"]] = layer_params.skipIact
            storage[strdic.status_dict["skipWght"]] = layer_params.skipWght
            storage[strdic.status_dict["skipPsum"]] = layer_params.skipPsum
            storage[strdic.status_dict["usePEs"]] = int(computing_pes,2)

            storage[strdic.status_dict["router_iact"]] = self.write_router_iact(params, layer_params)
            storage[strdic.status_dict["router_wght"]] = self.write_router_wght(params)
            storage[strdic.status_dict["router_psum"]] = self.write_router_psum(params, layer_params)
            return storage

    def write_router_iact(self, params, layer_params):
        line = 0
        if(params.SERIAL):
            storage = []
        else:
            storage = [[[[] for c in range(params.Iact_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]
        router_cycle = 0
        for cl_y in range(params.Clusters_Y):
            for cl_x in range(params.Clusters_X):
                for router in range(params.Iact_Routers):
                    if(layer_params.used_Y_cluster == 1):
                        if(params.SERIAL):
                            line = line + (1 << (params.Iact_Router_Bits * router_cycle))
                        else:
                            storage[cl_x][cl_y][router] = 1
                    else:
                        if((cl_y % layer_params.used_Y_cluster) == 0):
                            if(params.SERIAL):
                                line = line + (9 << (params.Iact_Router_Bits * router_cycle))
                            else:
                                storage[cl_x][cl_y][router] = 9
                        else:
                            if(((cl_y+ 1) % layer_params.used_Y_cluster) == 0):
                                if(params.SERIAL):
                                    line = line + (17 << (params.Iact_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 17
                            else:
                                if(params.SERIAL):
                                    line = line + (25 << (params.Iact_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 25
                    router_cycle = router_cycle + 1
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_Bits/params.Iact_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0
                        
                        
        if((params.SERIAL) and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)
            line = 0
        return storage

    def write_router_wght(self, params):
        #Nach write_working_parameters_Dense
        line = 0
        if(params.SERIAL):
            storage = []
        else:
            storage = [[[[] for c in range(params.Wght_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]
        router_cycle = 0    
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):   
                for router in range(params.Wght_Routers):
                    if(params.SERIAL):
                        line = line + (0 << (params.Wght_Router_Bits * router_cycle))
                    else:
                        storage[cl_x][cl_y][router] = 0
                    router_cycle = router_cycle + 1
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_Bits/params.Wght_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0
                            
        if((params.SERIAL) and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)
            line = 0
        return storage

    def write_router_psum(self, params, layer_params):
        #Nach write_working_parameters_Dense
        line = 0
        if(params.SERIAL):
            storage = []
        else:
            storage = [[[[] for c in range(params.Psum_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]
        router_cycle = 0          
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Psum_Routers):
                    if(params.SERIAL):
                        if (router == 0): 
                            line = line + (4 << (params.Psum_Router_Bits * router_cycle))
                        else:
                            line = line
                    else:
                        if (router == 0): 
                            storage[cl_x][cl_y][router] = 4
                        else:
                            storage[cl_x][cl_y][router] = 0
                    router_cycle = router_cycle + 1
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_Bits/params.Psum_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0
                        
        if(params.SERIAL and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)
            line = 0
        return storage

    def write_psum_data_glb(self, params, layer_params, layer_repetition, dram, cl_y, router, cycle):
        storage, line = self.initialize_storage(params.SERIAL), 0
        for part_data_num in range(math.ceil(self.layer_params.used_psum_per_PE/2)):
            if(part_data_num < self.layer_params.used_psum_per_PE):
                for cl_x in range(self.params.Clusters_X):
                    if(self.params.SERIAL):
                        line = line + (int(round(float((2**(params.IACT_Bitwidth + params.WGHT_Bitwidth - 1)) * 0))) << (params.PSUM_Bitwidth * cl_x))
                        storage.append(line)
                        line = 0
                    else:
                        storage[cl_x].append(int(round(float((2**(params.IACT_Bitwidth + params.WGHT_Bitwidth - 1)) * 0))))
            else:
                line = 0
        return storage
