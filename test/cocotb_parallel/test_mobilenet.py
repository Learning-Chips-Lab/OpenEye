# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import logging
import os
import sys

import pytest
import cocotb_test.simulator
logger = logging.getLogger("cocotb")

directory = (os.path.abspath(os.getcwd()))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
tests_dir = os.path.abspath(os.path.dirname(__file__))
hdl_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir, "hdl")

import parallel_test_utils as ptu


#As ref:
#clk_cycle = 10; clk_cycle_unit = "ns"

clk_cycle = 20
clk_cycle_unit = "ns"

clk_delay_in = 100
clk_delay_unit_in = "ps"

clk_delay_out = 100
clk_delay_unit_out = "ps"

##########################################################################################
layer_type_array = ["Convolution", "Depthwise_Convolution","Convolution", "Depthwise_Convolution",\
                    "Convolution", "Depthwise_Convolution","Convolution", "Depthwise_Convolution",\
                    "Convolution", "Depthwise_Convolution","Convolution", "Depthwise_Convolution",\
                    "Convolution", "Depthwise_Convolution","Convolution", "Depthwise_Convolution",\
                    "Convolution", "Depthwise_Convolution","Convolution", "Depthwise_Convolution",\
                    "Convolution", "Depthwise_Convolution","Convolution", "Depthwise_Convolution",\
                    "Convolution", "Depthwise_Convolution","Convolution", "FC"]

filter_array = [16,16,32,32,\
                64,64,64,64,\
                128,128,128,128,\
                256,256,256,256,\
                256,256,256,256,\
                256,256,256,256,\
                512,512,512,1000
                ]

stride_array = [2,1,1,2,\
                1,1,1,2,\
                1,1,1,2,\
                1,1,1,1,\
                1,1,1,1,\
                1,1,1,1,\
                1,1,1,2,\
                1,2,1,1
                ]

kernel_size_array = [3,3,1,3,\
                     1,3,1,3,\
                     1,3,1,3,\
                     1,3,1,3,\
                     1,3,1,3,\
                     1,3,1,3,\
                     1,3,1,1000
                     ]

input_size_array =  [128,64,64,64,\
                    32,32,32,32,\
                    16,16,16,16,\
                    8,8,8,8,\
                    8,8,8,8,\
                    8,8,8,8,\
                    4,4,4,512,\
                        ]

input_channel_array =   [3,16,16,32,\
                        32,64,64,64,\
                        64,128,128,128,\
                        128,256,256,256,\
                        256,256,256,256,\
                        256,256,256,256,\
                        256,512,512,512,\
                            ]

##########################################################################################

@pytest.mark.parametrize("LAYER_NUMBER",    [(0),(1),(2),(3),\
                                             (4),(5),(6),(7),\
                                             (8),(9),(10),(11),\
                                             (12),(13),(14),(15),\
                                             (16),(17),(18),(19),\
                                             (20),(21),(22),(23),\
                                             (24),(25),(26),(27)])
def test_mobilnet(LAYER_NUMBER):
    dut = 'OpenEye_Parallel'
    module = 'OpenEye_Parallel_tb'
    toplevel = dut

    layer_type = layer_type_array[LAYER_NUMBER]
    num_filters = filter_array[LAYER_NUMBER]
    stride = stride_array[LAYER_NUMBER]
    kernel_size = kernel_size_array[LAYER_NUMBER]
    input_size = input_size_array[LAYER_NUMBER]
    input_channels = input_channel_array[LAYER_NUMBER]

    verilog_sources = ptu.get_verilog_sources(hdl_dir, 0)

    target_dir = os.path.join(tests_dir, 'simulation/layer_' + str(LAYER_NUMBER)) 
    results = cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=target_dir,
        testcase='single_layer_test',
        force_compile=False,
        waves=True,
        simulator="verilator",
        extra_env = {"CLOCK_LEN" : str(clk_cycle)
                    ,"CLOCK_UNIT" : clk_cycle_unit
                    ,"CLOCK_DELAY_INPUT" : str(clk_delay_in)
                    ,"CLOCK_DELAY_UNIT_INPUT" : clk_delay_unit_in
                    ,"CLOCK_DELAY_OUTPUT" : str(clk_delay_out)
                    ,"CLOCK_DELAY_UNIT_OUTPUT" : clk_delay_unit_out
                    ,"LAYER" : layer_type
                    ,"NUM_FILTERS" : str(num_filters)
                    ,"STRIDE" : str(stride)
                    ,"KERNEL_SIZE" : str(kernel_size)
                    ,"INPUT_SIZE" : str(input_size)
                    ,"INPUT_CHANNELS" : str(input_channels)
                    }
    )
  


if __name__ == '__main__':
    test_mobilnet()
