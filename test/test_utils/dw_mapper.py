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
from test_utils.iact_stream_mapper import DwIactStreamMapper
from test_utils.wght_stream_mapper import DwWghtStreamMapper
from test_utils.psum_stream_mapper import DwPsumStreamMapper

logger = logging.getLogger("cocotb")

class DWMapper(LayerMapper):
        
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        input_mapper = DwIactStreamMapper(params, layer_params, layer_repetition, dram_layer_content[0])
        weight_mapper = DwWghtStreamMapper(params, layer_params, layer_repetition, dram_layer_content[1])
        bias_mapper = DwPsumStreamMapper(params, layer_params, layer_repetition, dram_layer_content[2])
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, input_mapper, weight_mapper, bias_mapper)
        
    def write_working_parameters(self, params, layer_params, layer_repetition):
        if (params.SERIAL):
            storage = [[] for b in range(len(strdic.stream_serial_dict))]
        else:
            storage = [[] for a in range(len(strdic.status_dict))]
        layer_params.skipPsum = 0
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
            dma_line = dma_line + ((layer_params.needed_refreshes_mx[layer_repetition][0] << 8))
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
            if ((layer_params.needed_refreshes_mx[layer_repetition][0] % layer_params.needed_wght_transmissions) == 0):
                dma_line = dma_line + (0 << 14)
            else:
                dma_line = dma_line + (1 << 14)
            dma_line = dma_line + (layer_params.skipWght << 15)
            dma_line = dma_line + (layer_params.skipPsum << 16)
            dma_storage.append(dma_line)
            for x in reversed(range(4)):
                dma_line = int(computing_pes[x*48:(x+1)*48],2)
                dma_storage.append(dma_line)
            storage[strdic.stream_serial_dict["status"]] = dma_storage
        else:
            storage[strdic.status_dict["data_mode"]] = params.data_mode
            storage[strdic.status_dict["realfactor"]] = layer_params.realfactor
            storage[strdic.status_dict["autofunction"]] = params.autofunction
            storage[strdic.status_dict["poolingmode"]] = params.poolingmode
            storage[strdic.status_dict["psum_delay"]] = layer_params.psum_delay
            storage[strdic.status_dict["needed_refreshes"]] = layer_params.needed_refreshes_mx[layer_repetition][0]
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
                        if(layer_params.used_PEs_Y > 1):
                            if((cl_y % layer_params.ceil_used_PE_per_clm) == 0):
                                if(params.SERIAL):
                                    line = line + (9 << (params.Iact_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 9
                            else:
                                if((cl_y % layer_params.ceil_used_PE_per_clm) + 1 == layer_params.ceil_used_PE_per_clm):
                                    if(params.SERIAL):
                                        line = line + (33 << (params.Iact_Router_Bits * router_cycle))
                                    else:
                                        storage[cl_x][cl_y][router] = 33
                                else:
                                    if(params.SERIAL):
                                        line = line + (41 << (params.Iact_Router_Bits * router_cycle))
                                    else:
                                        storage[cl_x][cl_y][router] = 41
                        else:
                            if(cl_y == 0):
                                if(params.SERIAL):
                                    line = line + (3 << (params.Iact_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 3
                            else:
                                if(params.SERIAL):
                                    line = line + (33 << (params.Iact_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 33
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
        line = 0
        if(params.SERIAL):
            storage = []
        else:
            storage = [[[[] for c in range(params.Wght_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]
        router_cycle = 0    
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):   
                for router in range(params.Wght_Routers):
                    if(cl_x == 0):
                        if(params.SERIAL):
                            line = line + (0 << (params.Wght_Router_Bits * router_cycle))
                        else:
                            storage[cl_x][cl_y][router] = 0
                    else:
                        if(params.SERIAL):
                            line = line + (1 << (params.Wght_Router_Bits * router_cycle))
                        else:
                            storage[cl_x][cl_y][router] = 1
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
                        line = line + (4 << (params.Psum_Router_Bits * router_cycle))
                    else:
                        storage[cl_x][cl_y][router] = 4
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
