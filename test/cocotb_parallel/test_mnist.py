# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import os

import cocotb
import cocotb_test.simulator
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

tests_dir = os.path.abspath(os.path.dirname(__file__))
hdl_dir = (os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, "hdl")))

import math

import numpy as np
from numpy import genfromtxt

import logging
logger = logging.getLogger("cocotb")

import pytest

import sys
directory = (os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])

import parallel_test_utils as test_utils

#As ref:
#clk_cycle = 10; clk_cycle_unit = "ns"

clk_cycle = 20
clk_cycle_unit = "ns"

clk_delay_in = 100
clk_delay_unit_in = "ps"

clk_delay_out = 100
clk_delay_unit_out = "ps"

##########################################################################################

@pytest.mark.parametrize("DNN", [(1)])#, 11, 32, 33, 63])
def MNIST_test(DNN):
    dut = 'OpenEye_Parallel'
    module = 'MNIST_tb'
    toplevel = dut
    verilog_sources = test_utils.get_verilog_sources(hdl_dir)

    target_dir = os.path.join(tests_dir, '.temp') 

    results = cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=target_dir,
        testcase='MNIST_test',
        force_compile=True,
        #waves=True,
        simulator="icarus",
        extra_env = {"CLOCK_LEN" : str(clk_cycle)
                    ,"CLOCK_UNIT" : clk_cycle_unit
                    ,"CLOCK_DELAY_INPUT" : str(clk_delay_in)
                    ,"CLOCK_DELAY_UNIT_INPUT" : clk_delay_unit_in
                    ,"CLOCK_DELAY_OUTPUT" : str(clk_delay_out)
                    ,"CLOCK_DELAY_UNIT_OUTPUT" : clk_delay_unit_out}
    )
    

if __name__ == '__main__':
    MNIST_test(1)
