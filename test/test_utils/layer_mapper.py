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


class LayerMapper(object):
    """
    Abstract parent class for a single layer.
    
    It contains default methods, that are needed for calculating the stream for one layer.

    """
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, inputstream_mapper = None, weightstream_mapper = None, bias_mapper = None):
        self.params = params
        self.layer_params = layer_params
        self.layer_repetition = layer_repetition
        self.dram_fmap = dram_layer_content[0]
        self.dram_weights = dram_layer_content[1]
        self.dram_bias = dram_layer_content[2]
        self.IactStreamCreator = inputstream_mapper
        self.WghtStreamCreator = weightstream_mapper
        self.PsumStreamCreator = bias_mapper
        if (params.SERIAL):
            self.storage = [[] for _ in range(len(strdic.stream_serial_dict))]
        else:
            self.storage = [[] for _ in range(len(strdic.stream_parallel_dict))]

    def make_stream(self):
        self.storage[strdic.stream_parallel_dict["status"]] = self.write_working_parameters(self.params, self.layer_params, self.layer_repetition)
        self.storage[strdic.stream_parallel_dict["iact"]] = self.IactStreamCreator.get_iact_stream()
        self.storage[strdic.stream_parallel_dict["wght"]] = self.WghtStreamCreator.get_wght_stream()
        self.storage[strdic.stream_parallel_dict["psum"]] = self.PsumStreamCreator.get_psum_stream()
        logger.info("Stream finished: " + str(self.layer_repetition + 1) + " of " + str(self.layer_params.needed_total_transmissions))

    def get_stream(self):
        return self.storage
    
    def write_working_parameters(self):
        pass
 
    def get_psum_stream(self):
        if(self.params.SERIAL):
            storage = []
        else:
            storage = [[[[] for c in range(self.params.Psum_Routers)] for b in range(self.params.Clusters_Y)] for a in range(self.params.Clusters_X)]

        for cl_y in range(self.params.Clusters_Y):
            for router in range(self.params.Psum_Routers):
                for cycle in range(self.layer_params.needed_refreshes_mx[self.layer_repetition][1],self.layer_params.needed_refreshes_mx[self.layer_repetition][2]):
                    temp_storage = self.write_psum_data_glb(self.params, self.layer_params, self.layer_repetition, self.dram_weights, cl_y, router, cycle)
                    if(self.params.SERIAL):
                        storage.extend(temp_storage)
                    else:
                        storage[0][cl_y][router].extend(temp_storage[0])
                        storage[1][cl_y][router].extend(temp_storage[1])
        return storage

    def initialize_storage(self, serial):
        if(serial):
            storage = []
        else:
            storage = [[] for a in range(self.params.Clusters_X)]
        return storage
        
    def line_reset(self, serial):
        if(serial):
            return 0
        else:
            return [0,0]
        
    def line_to_storage(self, serial, line, storage):
        if(serial):
            storage.append(line)
        else:
            storage[0].append(line[0])
            storage[1].append(line[1])
        return storage