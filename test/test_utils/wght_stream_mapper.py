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


class WghtStreamMapper(object):
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        self.params = params
        self.layer_params = layer_params
        self.layer_repetition = layer_repetition
        self.dram_weights = dram_layer_content
        if (params.SERIAL):
            self.storage = [[] for _ in range(len(strdic.stream_serial_dict))]
        else:
            self.storage = [[] for _ in range(len(strdic.stream_parallel_dict))]

    def get_wght_stream(self):
        storage = [[[[] for c in range(self.params.Wght_Routers)] for b in range(self.params.Clusters_Y)] for a in range(self.params.Clusters_X)]
        for cl_x in range(self.params.Clusters_X):
            for cl_y in range(self.params.Clusters_Y):
                for router in range(self.params.Wght_Routers):
                    if(self.layer_params.computing_mx[cl_x][cl_y][router][0] == 1):
                        spad = self.write_wght_pe(cl_x, cl_y, router)
                        storage[cl_x][cl_y][router] = spad
        wght_stream = self.create_complete_wght_stream(storage)
        return wght_stream
    
    def write_wght_pe(self, cl_x, cl_y, router):
        data_spad = self.write_wght_data_storage(cl_x, cl_y, router)
        addr_spad = self.write_wght_addr_storage(cl_x, cl_y, router)
        return [addr_spad, data_spad]

    def write_wght_data_storage(self, cl_x, cl_y, router):

        layer_repetition = self.layer_repetition
        layer_params = self.layer_params
        params = self.params
        dram = self.dram_weights

        spad_storage = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(int(self.params.Wghts_per_PE/self.params.PARALLEL_MACS))]
        overhead_counter = 0
        kernel_x = 0
        channel = (layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)
        filters_per_calculation = math.ceil(layer_params.used_wght_per_PE/layer_params.used_iact_per_PE)
        start_current_repetition = int((math.floor(layer_repetition/layer_params.iact_transmissions_pe) % layer_params.needed_wght_transmissions) * filters_per_calculation)
        filters = start_current_repetition
                    
        for words_in_storage in range(int(self.params.Wghts_per_PE/self.params.PARALLEL_MACS)):
            kernel_row = (cl_y % layer_params.ceil_used_PE_per_clm) * params.PEs_Y + router
            if(kernel_row < (layer_params.kernel_size[1] * int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))):
                for spad_val_number in range(self.params.PARALLEL_MACS): 
                    if(channel != int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe) + (layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)):
                        spad_storage[words_in_storage][spad_val_number][0] = dram[channel][filters][kernel_row][kernel_x]
                        spad_storage[words_in_storage][spad_val_number][1] = overhead_counter
                        filters = filters + 1
                        if((filters == (start_current_repetition + filters_per_calculation))):
                            filters = start_current_repetition
                            kernel_x = kernel_x + 1
                            overhead_counter = 0
                        if(kernel_x == layer_params.kernel_size[0]):
                            kernel_x = 0
                            channel = channel + 1
            if (words_in_storage == math.ceil(layer_params.used_wght_per_PE/2)):
                break
        return spad_storage
        
    def write_wght_addr_storage(self, cl_x, cl_y, router):

        layer_params = self.layer_params
        params = self.params

        spad_storage = [0 for _ in range(self.params.Wghts_Addr_per_PE)]

        for words_in_storage in range(math.ceil(params.Wghts_Addr_per_PE)):
            if((words_in_storage != (self.layer_params.used_wght_addr_per_PE - 1))):
                spad_storage[words_in_storage] = \
                    int(words_in_storage * math.ceil(layer_params.used_wght_per_PE/layer_params.kernel_size[0]/2/ int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)))
            else:
                break
        return spad_storage
        
    def create_complete_wght_stream(self, spad_storage):
        params = self.params

        stream = [[[[] for c in range(params.Wght_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Wght_Routers):

                    current_spad = spad_storage[cl_x][cl_y][router]
                    stream[cl_x][cl_y][router] = self.create_pe_addr_wght_stream(current_spad)
                    stream[cl_x][cl_y][router].extend(self.create_pe_data_wght_stream(current_spad))
        if(params.SERIAL):
            temp_stream = stream
            stream = []
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Iact_Routers):
                    for word in range(len(temp_stream[0][cl_y][router])):
                        stream.append(temp_stream[0][cl_y][router][word] + (temp_stream[1][cl_y][router][word] * (2**24)))
        return stream
    
    def create_pe_addr_wght_stream(self, spad):
        layer_params = self.layer_params
        params = self.params

        addr_per_trans = math.floor(params.WGHT_Trans_Bitwidth/params.WGHT_Addr_Bitwidth)
        line_counter = 0
        stream = []

        for spad_addr_trans in range(math.ceil(params.Wghts_Addr_per_PE/addr_per_trans)):
            temp_trans = 0
            for addr_in_trans in range(addr_per_trans):
                try:
                    spad_word = addr_in_trans + spad_addr_trans * addr_per_trans
                    temp_trans = temp_trans + \
                        (spad[0][spad_word] \
                        << (params.WGHT_Addr_Bitwidth * addr_in_trans)) 
                except:
                    pass
            for write_time in range(addr_per_trans):
                stream.append(temp_trans)
                line_counter = line_counter + 1
                if (line_counter == layer_params.used_wght_addr_per_PE):
                    break
            if (line_counter == layer_params.used_wght_addr_per_PE):
                break
        return stream
    
    def create_pe_data_wght_stream(self, spad):
        layer_params = self.layer_params
        params = self.params

        data_per_trans = math.floor(params.WGHT_Trans_Bitwidth/params.WGHT_WOH_Bitwidth)
        line_counter = 0
        stream = []
        for spad_data_trans in range(math.floor(params.Wghts_per_PE/data_per_trans)):
            temp_trans = 0
            for data_in_trans in range(data_per_trans):
                try:
                    spad_word = data_in_trans + data_per_trans * spad_data_trans
                    value = gtu.to_twos_complement(spad[1][spad_data_trans][data_in_trans][0], self.params.WGHT_Bitwidth)
                    temp_trans = temp_trans + (value << (data_in_trans * params.WGHT_WOH_Bitwidth))
                except:
                    pass
            stream.append(temp_trans)
            line_counter = line_counter + 1
            if (line_counter == math.ceil(layer_params.used_wght_per_PE/2)):
                break
        return stream
    
class ConvWghtStreamMapper(WghtStreamMapper):
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def write_wght_data_storage(self, cl_x, cl_y, router):

        layer_repetition = self.layer_repetition
        layer_params = self.layer_params
        params = self.params
        dram = self.dram_weights

        spad_storage = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(int(self.params.Wghts_per_PE/self.params.PARALLEL_MACS))]
        overhead_counter = 0
        kernel_x = 0
        channel = (layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)
        filters_per_calculation = math.ceil(layer_params.used_wght_per_PE/layer_params.used_iact_per_PE)
        start_current_repetition = int((math.floor(layer_repetition/layer_params.iact_transmissions_pe) % layer_params.needed_wght_transmissions) * filters_per_calculation)
        filters = start_current_repetition
                    
        for words_in_storage in range(int(self.params.Wghts_per_PE/self.params.PARALLEL_MACS)):
            kernel_row = (cl_y % layer_params.ceil_used_PE_per_clm) * params.PEs_Y + router
            if(kernel_row < (layer_params.kernel_size[1])):
                for spad_val_number in range(self.params.PARALLEL_MACS): 
                    if(channel != int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe) + (layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)):
                        
                        try:
                            spad_storage[words_in_storage][spad_val_number][0] = dram[channel][filters][kernel_row][kernel_x]
                        except:
                            spad_storage[words_in_storage][spad_val_number][0] = 0

                        spad_storage[words_in_storage][spad_val_number][1] = overhead_counter
                        filters = filters + 1
                        if((filters == (start_current_repetition + filters_per_calculation))):
                            filters = start_current_repetition
                            kernel_x = kernel_x + 1
                            overhead_counter = 0
                        if(kernel_x == layer_params.kernel_size[0]):
                            kernel_x = 0
                            channel = channel + 1
            if (words_in_storage == math.ceil(layer_params.used_wght_per_PE/2)):
                break
        return spad_storage
        
    def write_wght_addr_storage(self, cl_x, cl_y, router):

        layer_params = self.layer_params
        params = self.params

        spad_storage = [0 for _ in range(self.params.Wghts_Addr_per_PE)]
        for words_in_storage in range(math.ceil(params.Wghts_Addr_per_PE)):
            if((words_in_storage != (self.layer_params.used_wght_addr_per_PE - 1)) | (self.layer_params.used_wght_addr_per_PE == self.params.Wghts_Addr_per_PE)):
                spad_storage[words_in_storage] = \
                    int(words_in_storage * math.ceil(layer_params.used_wght_per_PE/layer_params.kernel_size[0]/2/ int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)))
            else:
                break
        return spad_storage

class DenseWghtStreamMapper(WghtStreamMapper):
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def write_wght_data_storage(self, cl_x, cl_y, router):

        layer_repetition = self.layer_repetition
        layer_params = self.layer_params
        params = self.params
        dram = self.dram_weights

        spad_storage = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(int(self.params.Wghts_per_PE/self.params.PARALLEL_MACS))]
        overhead_counter = 0
                
        for words_in_storage in range(int(self.params.Wghts_per_PE/self.params.PARALLEL_MACS)):
            for spad_val_number in range(self.params.PARALLEL_MACS):

                position = words_in_storage * 2 + spad_val_number
                filters =  (position%layer_params.used_psum_per_PE) + \
                cl_x * layer_params.used_psum_per_PE + \
                cl_y * params.Clusters_X * layer_params.used_psum_per_PE + \
                (math.floor(layer_repetition/layer_params.iact_transmissions_pe) % layer_params.psum_transmissions_pe) * params.Clusters_X * params.Clusters_Y * layer_params.used_psum_per_PE

                channel = math.floor((layer_repetition%layer_params.iact_transmissions_pe)*params.Wght_Routers*layer_params.used_iact_per_PE) + \
                math.floor(position/layer_params.used_psum_per_PE) + \
                router * layer_params.used_iact_per_PE
                try:
                    spad_storage[words_in_storage][spad_val_number][0] = dram[filters][channel]
                    spad_storage[words_in_storage][spad_val_number][1] = overhead_counter
                except:
                    pass
            if (words_in_storage == math.ceil(layer_params.used_wght_per_PE/2)):
                break
        return spad_storage
        
    def write_wght_addr_storage(self, cl_x, cl_y, router):

        layer_params = self.layer_params
        params = self.params

        spad_storage = [0 for _ in range(self.params.Wghts_Addr_per_PE)]

        for words_in_storage in range(math.ceil(params.Wghts_Addr_per_PE)):
            if(words_in_storage != (self.layer_params.used_wght_addr_per_PE - 1)):
                spad_storage[words_in_storage] = \
                    int(words_in_storage * math.ceil(layer_params.used_psum_per_PE/2))
            else:
                break
        return spad_storage

class DwWghtStreamMapper(WghtStreamMapper):
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def write_wght_data_storage(self, cl_x, cl_y, router):

        layer_repetition = self.layer_repetition
        layer_params = self.layer_params
        params = self.params
        dram = self.dram_weights

        spad_storage = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(int(self.params.Wghts_per_PE/self.params.PARALLEL_MACS))]
        overhead_counter = 0
        kernel_x = 0
        channel = (layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)
                    
        if(layer_params.filters == 1):
            values_per_wght_data = 1
        else:
            values_per_wght_data = 2
        for words_in_storage in range(int(self.params.Wghts_per_PE/self.params.PARALLEL_MACS)):
            Mtrx_Row = (cl_y % layer_params.ceil_used_PE_per_clm) * params.PEs_Y + router
            if(Mtrx_Row < (layer_params.kernel_size[1] * int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))):
                for spad_val_number in range(values_per_wght_data): 
                    if(channel != int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe) + (layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)):
                        spad_storage[words_in_storage][spad_val_number][0] = dram[channel][Mtrx_Row][kernel_x]
                        spad_storage[words_in_storage][spad_val_number][1] = overhead_counter
                        overhead_counter = overhead_counter + 1
                        kernel_x = kernel_x + 1
                        overhead_counter = 0
                        if(kernel_x == layer_params.kernel_size[0]):
                            kernel_x = 0
                            channel = channel + 1
            if (words_in_storage == math.ceil(layer_params.used_wght_per_PE/2)):
                break
        return spad_storage
        
    def write_wght_addr_storage(self, cl_x, cl_y, router):

        layer_params = self.layer_params
        params = self.params

        spad_storage = [0 for _ in range(self.params.Wghts_Addr_per_PE)]

        for words_in_storage in range(math.ceil(params.Wghts_Addr_per_PE)):
            if(words_in_storage != (self.layer_params.used_wght_addr_per_PE - 1)):
                spad_storage[words_in_storage] = \
                    int(words_in_storage * math.ceil(layer_params.used_wght_per_PE/layer_params.kernel_size[0]/2/ int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)))
            else:
                break
        return spad_storage
