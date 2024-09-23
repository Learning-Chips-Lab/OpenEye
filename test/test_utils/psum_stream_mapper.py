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


class PsumStreamMapper(object):
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        self.params = params
        self.layer_params = layer_params
        self.layer_repetition = layer_repetition
        self.dram_bias = dram_layer_content
        if (self.params.SERIAL):
            self.storage = [[] for _ in range(len(strdic.stream_serial_dict))]
        else:
            self.storage = [[] for _ in range(len(strdic.stream_parallel_dict))]

    def get_psum_stream(self):
        psum_stream = [[[[] for c in range(self.params.Psum_Routers)] for b in range(self.params.Clusters_Y)] for a in range(self.params.Clusters_X)]
        for cl_x in range(self.params.Clusters_X):
            for cl_y in range(self.params.Clusters_Y):
                for router in range(self.params.Psum_Routers):
                    psum_stream[cl_x][cl_y][router] = self.write_psum_data_glb(cl_x, cl_y, router)
        psum_stream = self.create_complete_psum_stream(psum_stream)
        return psum_stream

    def write_psum_data_glb(self, cl_x, cl_y, router):
        storage = []
        for cycle in range(self.layer_params.needed_refreshes_mx[self.layer_repetition][1],self.layer_params.needed_refreshes_mx[self.layer_repetition][2]):
            storage.append(self.write_psum_storage(cl_x, cl_y, router, cycle))
        return storage

    def create_complete_psum_stream(self, spad_storage):
        params = self.params
        stream = [[[[] for c in range(self.params.Psum_Routers)] for b in range(self.params.Clusters_Y)] for a in range(self.params.Clusters_X)]
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Psum_Routers):

                    current_spad = spad_storage[cl_x][cl_y][router]
                    for cycle in range(len(current_spad)):
                        stream[cl_x][cl_y][router].extend(spad_storage[cl_x][cl_y][router][cycle])


        if(self.params.SERIAL):
            temp_stream = stream
            stream = []
            for cl_y in range(self.params.Clusters_Y):
                for router in range(self.params.Iact_Routers):
                    for word in range(len(temp_stream[0][cl_y][router])):
                        stream.append(temp_stream[0][cl_y][router][word] + (temp_stream[1][cl_y][router][word] * (2**24)))
        else:
            return stream
        return stream

class ConvPsumStreamMapper(PsumStreamMapper):
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def write_psum_storage(self, cl_x, cl_y, router, cycle):
        storage = []
        for part_data_num in range(int(self.layer_params.filters/self.layer_params.needed_wght_transmissions)):
            if(part_data_num < self.layer_params.used_psum_per_PE):
                storage.append(0)
            else:
                line = 0
        return storage

class DwPsumStreamMapper(PsumStreamMapper):
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def write_psum_storage(self, cl_x, cl_y, router, cycle):
        storage = []
        for part_data_num in range(int(self.layer_params.filters)):
            if((cl_y % self.layer_params.ceil_used_PE_per_clm) == 0):
                if(part_data_num < self.layer_params.used_psum_per_PE):
                    storage.append(0)
                else:
                    line = 0
        return storage

class DensePsumStreamMapper(PsumStreamMapper):
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def write_psum_storage(self, cl_x, cl_y, router, cycle):
        storage = []
        for part_data_num in range(math.ceil(self.layer_params.used_psum_per_PE/2)):
            if(part_data_num < self.layer_params.used_psum_per_PE):
                if(part_data_num < self.layer_params.used_psum_per_PE):
                    storage.append(0)
                else:
                    line = 0
        return storage


    