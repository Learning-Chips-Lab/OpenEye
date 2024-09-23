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
import generic_test_utils as gtu
import stream_dicts as strdic

logger = logging.getLogger("cocotb")


class IactStreamMapper(object):
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        self.params = params
        self.layer_params = layer_params
        self.layer_repetition = layer_repetition
        self.dram_fmap = dram_layer_content
        if (params.SERIAL):
            self.storage = [[] for _ in range(len(strdic.stream_serial_dict))]
        else:
            self.storage = [[] for _ in range(len(strdic.stream_parallel_dict))]

    def get_iact_stream(self):
        iact_stream = [[[[] for c in range(self.params.Iact_Routers)] for b in range(self.params.Clusters_Y)] for a in range(self.params.Clusters_X)]
        for cl_x in range(self.params.Clusters_X):
            for cl_y in range(self.params.Clusters_Y):
                for router in range(self.params.Iact_Routers):
                    iact_stream[cl_x][cl_y][router] = self.write_iact_data_glb(cl_x, cl_y, router)
        iact_stream = self.create_complete_iact_stream(iact_stream)
        return iact_stream
    
    def write_iact_data_glb(self, cl_x, cl_y, router):
        storage = []
        for cycle in range(self.layer_params.needed_refreshes_mx[self.layer_repetition][1],self.layer_params.needed_refreshes_mx[self.layer_repetition][2]):
            for iact_cycle in range(self.layer_params.needed_Iact_writes):
                if ((cl_y - cycle) % self.layer_params.used_Y_cluster == 0) :
                    data_spad = self.write_iact_data_storage(cl_x, cl_y, router, cycle, iact_cycle)
                    addr_spad = self.write_iact_addr_storage(cl_x, cl_y, router, cycle, iact_cycle)
                    storage.append([addr_spad, data_spad])

        return storage

    def write_iact_data_storage(self, cl_x, cl_y, router, cycle, iact_cycle):

        layer_params = self.layer_params
        params = self.params
        layer_repetition = self.layer_repetition
        dram_fmap = self.dram_fmap

        overhead_counter = 0
        spad_storage = [[0 for _ in range(2)] for _ in range(params.Iacts_per_PE)]
        for words_in_storage in range(math.ceil(params.Iacts_per_PE)):
            if(words_in_storage < layer_params.used_iact_per_PE):
                iact_temp_pos_x = \
                int((math.floor(((cl_x * params.PEs_X) + \
                (math.floor(cl_y / layer_params.used_Y_cluster) * params.PEs_X * params.Clusters_X) + \
                cycle * params.PEs_X * math.floor(params.Clusters/self.layer_params.used_Y_cluster)) * layer_params.strideX) % \
                ((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)) + \
                router + iact_cycle * params.Iact_Routers) - \
                (math.ceil((layer_params.kernel_size[1]-1)/2))                             #Zero Padding
                
                iact_temp_pos_y = \
                int((words_in_storage % layer_params.kernel_size[1]) + \
                (layer_params.strideY * \
                math.floor((((math.floor(cl_y / layer_params.used_Y_cluster) * params.PEs_X * params.Clusters_X) + \
                cycle * params.PEs_X * math.floor(params.Clusters/self.layer_params.used_Y_cluster)) * layer_params.strideX)/((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)))) - \
                (math.ceil((layer_params.kernel_size[1]-1)/2))                             #Zero Padding

                channel = math.floor(words_in_storage/layer_params.kernel_size[0]) + ((layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)) #HIer was aendern
                if(((math.floor(cycle/ self.layer_params.used_Y_cluster) * params.Clusters_Y *  params.Clusters_X * params.PEs_X) + \
                    ((math.floor(cl_y/ self.layer_params.used_Y_cluster) *  params.Clusters_X * params.PEs_X)) + (cl_x * params.PEs_X)) < \
                   (layer_params.output_shape[1] * (layer_params.output_shape[2]+layer_params.add_up))):
                    if(((iact_temp_pos_x >= 0) & (iact_temp_pos_x < (layer_params.output_shape[1] * layer_params.strideX))) & \
                    ((iact_temp_pos_y) >= 0) & (iact_temp_pos_y < (layer_params.output_shape[2] * layer_params.strideY))):
                        spad_storage[words_in_storage][0] = dram_fmap[channel][iact_temp_pos_x][iact_temp_pos_y]

                    else:
                        spad_storage[words_in_storage][0] = 1
                    spad_storage[words_in_storage][1] = overhead_counter
                    overhead_counter = overhead_counter + 1

        return spad_storage
        
    def write_iact_addr_storage(self, cl_x, cl_y, router, cycle, iact_cycle):

        layer_params = self.layer_params
        params = self.params
        line_counter = 0
        spad_storage = [0 for _ in range(self.params.Iacts_Addr_per_PE)]

        
        for words_in_storage in range(params.Iacts_Addr_per_PE):
            if(words_in_storage < (math.ceil(layer_params.used_iact_per_PE/layer_params.kernel_size[0]))):
                spad_storage[words_in_storage] = (layer_params.kernel_size[0] * (words_in_storage + 1))
            line_counter = line_counter + 1
        return spad_storage
        
    def create_complete_iact_stream(self, spad_storage):
        params = self.params

        stream = [[[[] for c in range(params.Iact_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Iact_Routers):

                    current_spad = spad_storage[cl_x][cl_y][router]
                    for cycle in range(len(current_spad)):
                        stream[cl_x][cl_y][router].extend(self.create_pe_addr_iact_stream(current_spad[cycle]))

                        stream[cl_x][cl_y][router].extend(self.create_pe_data_iact_stream(current_spad[cycle]))

        if(params.SERIAL):
            temp_stream = stream
            stream = []
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Iact_Routers):
                    for word in range(len(temp_stream[0][cl_y][router])):
                        stream.append(temp_stream[0][cl_y][router][word] + (temp_stream[1][cl_y][router][word] * (2**24)))

        return stream
    
    def create_pe_addr_iact_stream(self, spad):
        layer_params = self.layer_params
        params = self.params

        addr_per_trans = math.floor(params.IACT_Trans_Bitwidth/params.IACT_Addr_Bitwidth)
        line_counter = 0
        stream = []
        for spad_addr_trans in range(math.ceil(params.Iacts_Addr_per_PE/addr_per_trans)):
            temp_trans = 0
            for addr_in_trans in range(addr_per_trans):
                try:
                    spad_word = addr_in_trans + spad_addr_trans * addr_per_trans
                    temp_trans = temp_trans + \
                        (spad[0][spad_word] \
                        << (params.IACT_Addr_Bitwidth * addr_in_trans)) 
                except:
                    pass
            stream.append(temp_trans)
            line_counter = line_counter + 1
            if (line_counter == math.ceil(layer_params.used_iact_addr_per_PE/addr_per_trans)):
                return stream
        return stream
    
    def create_pe_data_iact_stream(self, spad):
        layer_params = self.layer_params
        params = self.params

        data_per_trans = math.floor(params.IACT_Trans_Bitwidth/params.IACT_WOH_Bitwidth)
        line_counter = 0
        stream = []
        for spad_data_trans in range(math.floor(params.Iacts_per_PE/data_per_trans)):
            temp_trans = 0
            for data_in_trans in range(data_per_trans):
                try:
                    number_of_value = (data_in_trans + spad_data_trans * data_per_trans)
                    value = gtu.to_twos_complement_string(spad[1][number_of_value][1], 4) + \
                        gtu.to_twos_complement_string(spad[1][number_of_value][0], self.params.IACT_Bitwidth)
                    temp_trans = temp_trans + (int(value,2) << (data_in_trans * params.IACT_WOH_Bitwidth))
                except:
                    pass
            stream.append(temp_trans)
            line_counter = line_counter + 1
            if (line_counter == math.ceil(layer_params.used_iact_per_PE/2)):
                break
        return stream
    
class ConvIactStreamMapper(IactStreamMapper):
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def write_iact_data_storage(self, cl_x, cl_y, router, cycle, iact_cycle):

        layer_params = self.layer_params
        params = self.params
        layer_repetition = self.layer_repetition
        dram_fmap = self.dram_fmap

        overhead_counter = 0
        line_counter = 0
        spad_storage = [[0 for _ in range(2)] for _ in range(params.Iacts_per_PE)]
        for words_in_storage in range(math.ceil(params.Iacts_per_PE)):
            if(words_in_storage < layer_params.used_iact_per_PE):

                iact_temp_pos_x = \
                int((math.floor(((cl_x * params.PEs_X) + \
                (math.floor(cl_y / layer_params.used_Y_cluster) * params.PEs_X * params.Clusters_X) + \
                cycle * params.PEs_X * math.floor(params.Clusters/self.layer_params.used_Y_cluster)) * layer_params.strideX) % \
                ((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)) + \
                router + iact_cycle * params.Iact_Routers) - \
                (math.ceil((layer_params.kernel_size[1]-1)/2))                             #Zero Padding
                
                iact_temp_pos_y = \
                int((words_in_storage % layer_params.kernel_size[1]) + \
                (layer_params.strideY * \
                math.floor((((math.floor(cl_y / layer_params.used_Y_cluster) * params.PEs_X * params.Clusters_X) + \
                (cl_x * params.PEs_X) + \
                cycle * params.PEs_X * math.floor(params.Clusters/self.layer_params.used_Y_cluster)) * layer_params.strideX)/((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)))) - \
                (math.ceil((layer_params.kernel_size[1]-1)/2))                             #Zero Padding

                channel = math.floor(words_in_storage/layer_params.kernel_size[0]) + ((layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))
                if(((math.floor(cycle/ self.layer_params.used_Y_cluster) * params.Clusters_Y *  params.Clusters_X * params.PEs_X) + \
                    ((math.floor(cl_y/ self.layer_params.used_Y_cluster) *  params.Clusters_X * params.PEs_X)) + (cl_x * params.PEs_X)) < \
                   (layer_params.output_shape[1] * (layer_params.output_shape[2]+layer_params.add_up))):
                    if(((iact_temp_pos_x >= 0) & (iact_temp_pos_x < (layer_params.output_shape[1] * layer_params.strideX))) & \
                    ((iact_temp_pos_y) >= 0) & (iact_temp_pos_y < (layer_params.output_shape[2] * layer_params.strideY))):
                        spad_storage[words_in_storage][0] = dram_fmap[channel][iact_temp_pos_x][iact_temp_pos_y]

                    else:
                        spad_storage[words_in_storage][0] = 1
                    spad_storage[words_in_storage][1] = overhead_counter
                    overhead_counter = overhead_counter + 1


        return spad_storage
        
    def write_iact_addr_storage(self, cl_x, cl_y, router, cycle, iact_cycle):

        layer_params = self.layer_params
        params = self.params
        line_counter = 0
        spad_storage = [0 for _ in range(self.params.Iacts_Addr_per_PE)]

        
        for words_in_storage in range(params.Iacts_Addr_per_PE):
            if(words_in_storage < (math.ceil(layer_params.used_iact_per_PE/layer_params.kernel_size[0]))):
                spad_storage[words_in_storage] = (layer_params.kernel_size[0] * (words_in_storage + 1))
            line_counter = line_counter + 1
        return spad_storage
     
class DenseIactStreamMapper(IactStreamMapper):
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)
    
    def write_iact_data_glb(self, cl_x, cl_y, router):
        storage = []
        for cycle in range(self.layer_params.needed_refreshes_mx[self.layer_repetition][1],self.layer_params.needed_refreshes_mx[self.layer_repetition][2]):
            for iact_cycle in range(self.layer_params.needed_Iact_writes):
                data_spad = self.write_iact_data_storage(cl_x, cl_y, router, cycle, iact_cycle)
                addr_spad = self.write_iact_addr_storage(cl_x, cl_y, router, cycle, iact_cycle)
                storage.append([addr_spad, data_spad])

        return storage
    
    def write_iact_addr_storage(self, cl_x, cl_y, router, cycle, iact_cycle):

        layer_params = self.layer_params
        params = self.params
        spad_storage = [0 for _ in range(self.params.Iacts_Addr_per_PE)]

        if(cl_y == 0):
            for words_in_storage in range(params.Iacts_Addr_per_PE):
                if(words_in_storage < 1):
                    spad_storage[words_in_storage] = (layer_params.used_iact_per_PE)
        return spad_storage

    def write_iact_data_storage(self, cl_x, cl_y, router, cycle, iact_cycle):

        params = self.params
        layer_params = self.layer_params
        dram_fmap = self.dram_fmap
        layer_repetition = self.layer_repetition

        overhead_counter = 0
        spad_storage = [[0 for _ in range(2)] for _ in range(params.Iacts_per_PE)]
        if(cl_y == 0):
            for words_in_storage in range(math.ceil(params.Iacts_per_PE)):
                if(words_in_storage < layer_params.used_iact_per_PE):

                    iact_temp_pos_x = words_in_storage + \
                    router * layer_params.used_iact_per_PE + \
                    (layer_repetition % layer_params.iact_transmissions_pe) * params.Iact_Routers * layer_params.used_iact_per_PE

                    try:
                        spad_storage[words_in_storage][0]= dram_fmap[iact_temp_pos_x]
                    except:
                        spad_storage[words_in_storage][0]= 0
                    spad_storage[words_in_storage][1]= overhead_counter

                    overhead_counter = overhead_counter + 1

        return spad_storage
    
class DwIactStreamMapper(IactStreamMapper):
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def write_iact_addr_storage(self, cl_x, cl_y, router, cycle, iact_cycle):

        layer_params = self.layer_params
        params = self.params

        spad_storage = [0 for _ in range(self.params.Iacts_Addr_per_PE)]

        
        for words_in_storage in range(params.Iacts_Addr_per_PE):
            if(words_in_storage < (math.ceil(layer_params.used_iact_per_PE/layer_params.kernel_size[0]))):
                spad_storage[words_in_storage] = (layer_params.kernel_size[0] * (words_in_storage + 1))
        return spad_storage
    
    def write_iact_data_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        params = self.params
        layer_params = self.layer_params
        dram_fmap = self.dram_fmap

        iact_temp_pos_x = 0
        iact_temp_pos_y = 0
        overhead_counter = 0
        spad_storage = [[0 for _ in range(2)] for _ in range(params.Iacts_per_PE)]

        for words_in_storage in range(math.ceil(params.Iacts_per_PE)):
            if(words_in_storage < layer_params.used_iact_per_PE):

                iact_temp_pos_x = \
                int((math.floor(((cl_x * params.PEs_X) + \
                ((cl_y / layer_params.ceil_used_PE_per_clm) * params.PEs_X * params.Clusters_X) + \
                ((cl_y % layer_params.ceil_used_PE_per_clm) * params.PEs_X * (params.Clusters / layer_params.ceil_used_PE_per_clm )) + \
                cycle * params.PEs_X * params.Clusters) * \
                layer_params.strideX) % \
                ((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)) + \
                router + iact_cycle * params.Iact_Routers) - \
                (math.ceil((layer_params.kernel_size[1]-1)/2))                             #Zero Padding
                
                iact_temp_pos_y = \
                int(((words_in_storage) % layer_params.kernel_size[1]) + \
                (layer_params.strideY * math.floor( \
                ((((cl_y / layer_params.ceil_used_PE_per_clm) * params.PEs_X * params.Clusters_X) + \
                ((cl_y % layer_params.ceil_used_PE_per_clm) * params.Clusters_X * params.PEs_X) + \
                (cl_x * params.PEs_X) + \
                cycle *params.PEs_X * params.Clusters) * \
                layer_params.strideX)/((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)))) - \
                (math.ceil((layer_params.kernel_size[1]-1)/2))                             #Zero Padding
                
                channel = math.floor(words_in_storage/layer_params.kernel_size[0]) + ((self.layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)) #HIer was aendern

                if(((cycle * 8 * 2 * 4) + (cl_y * 2 * 4) + (cl_x * 4)) < (layer_params.output_shape[1] * (layer_params.output_shape[2]+layer_params.add_up))):
                    if(((iact_temp_pos_x >= 0) & (iact_temp_pos_x < (layer_params.output_shape[1] * layer_params.strideX))) & \
                    ((iact_temp_pos_y) >= 0) & (iact_temp_pos_y < (layer_params.output_shape[2] * layer_params.strideY))):
                        spad_storage[words_in_storage][0]= dram_fmap[channel][iact_temp_pos_x][iact_temp_pos_y]
                    else:
                        spad_storage[words_in_storage][0]= 1
                    spad_storage[words_in_storage][1] = overhead_counter
                    overhead_counter = overhead_counter + 1

        return spad_storage
    