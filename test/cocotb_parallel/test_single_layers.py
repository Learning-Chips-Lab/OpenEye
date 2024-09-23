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

@pytest.mark.parametrize("NUM_FILTERS", [(8)])#, 11, 32, 33, 63])
@pytest.mark.parametrize("STRIDE", [(1)])#, (2,2)])
@pytest.mark.parametrize("KERNEL_SIZE", [(3)])
@pytest.mark.parametrize("INPUT_SIZE", [(32)])
@pytest.mark.parametrize("INPUT_CHANNELS", [(4)])#, 4, 8])
def test_single_conv_layer(NUM_FILTERS, STRIDE, KERNEL_SIZE, INPUT_SIZE, INPUT_CHANNELS):
    layer = "Convolution"
    dut = 'OpenEye_Parallel'
    module = 'OpenEye_Parallel_tb'
    toplevel = dut
    verilog_sources = ptu.get_verilog_sources(hdl_dir)

    target_dir = os.path.join(tests_dir, '.temp') 

    results = cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=target_dir,
        testcase='single_layer_test',
        force_compile=True,
        waves=False,
        simulator="verilator",
        extra_env = {"CLOCK_LEN" : str(clk_cycle)
                    ,"CLOCK_UNIT" : clk_cycle_unit
                    ,"CLOCK_DELAY_INPUT" : str(clk_delay_in)
                    ,"CLOCK_DELAY_UNIT_INPUT" : clk_delay_unit_in
                    ,"CLOCK_DELAY_OUTPUT" : str(clk_delay_out)
                    ,"CLOCK_DELAY_UNIT_OUTPUT" : clk_delay_unit_out
                    ,"LAYER" : layer
                    ,"NUM_FILTERS" : str(NUM_FILTERS)
                    ,"STRIDE" : str(STRIDE)
                    ,"KERNEL_SIZE" : str(KERNEL_SIZE)
                    ,"INPUT_SIZE" : str(INPUT_SIZE)
                    ,"INPUT_CHANNELS" : str(INPUT_CHANNELS)}
    )
    
@pytest.mark.parametrize("STRIDE", [(1)])#, (2,2)])
@pytest.mark.parametrize("KERNEL_SIZE", [(7)])
@pytest.mark.parametrize("INPUT_SIZE", [(7)])
@pytest.mark.parametrize("INPUT_CHANNELS", [(10)])#, 4, 8])
def test_single_pool_layer(STRIDE,KERNEL_SIZE,INPUT_SIZE,INPUT_CHANNELS):
    layer = "Pooling"
    dut = 'OpenEye_Parallel'
    module = 'OpenEye_Parallel_tb'
    toplevel = dut
    verilog_sources = ptu.get_verilog_sources(hdl_dir)

    target_dir = os.path.join(tests_dir, '.temp') 

    results = cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=target_dir,
        testcase='single_layer_test',
        force_compile=True,
        #waves=True,
        simulator="icarus",
        extra_env = {"CLOCK_LEN" : str(clk_cycle)
                    ,"CLOCK_UNIT" : clk_cycle_unit
                    ,"CLOCK_DELAY_INPUT" : str(clk_delay_in)
                    ,"CLOCK_DELAY_UNIT_INPUT" : clk_delay_unit_in
                    ,"CLOCK_DELAY_OUTPUT" : str(clk_delay_out)
                    ,"CLOCK_DELAY_UNIT_OUTPUT" : clk_delay_unit_out
                    ,"LAYER" : layer
                    ,"STRIDE" : str(STRIDE)
                    ,"KERNEL_SIZE" : str(KERNEL_SIZE)
                    ,"INPUT_SIZE" : str(INPUT_SIZE)
                    ,"INPUT_CHANNELS" : str(INPUT_CHANNELS)}
    )

@pytest.mark.parametrize("STRIDE", [(1)])#, (2,2)])
@pytest.mark.parametrize("KERNEL_SIZE", [(3)])
@pytest.mark.parametrize("INPUT_SIZE", [(32)])
@pytest.mark.parametrize("INPUT_CHANNELS", [(4),(8)])#, 4, 8])
def test_depthwise_conv_layer(STRIDE,KERNEL_SIZE,INPUT_SIZE,INPUT_CHANNELS):
    NUM_FILTERS = 1
    layer = "Depthwise_Convolution"
    dut = 'OpenEye_Parallel'
    module = 'OpenEye_Parallel_tb'
    toplevel = dut
    verilog_sources = ptu.get_verilog_sources(hdl_dir)

    target_dir = os.path.join(tests_dir, '.temp') 

    results = cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=target_dir,
        testcase='single_layer_test',
        force_compile=True,
        #waves=True,
        simulator="icarus",
        extra_env = {"CLOCK_LEN" : str(clk_cycle)
                    ,"CLOCK_UNIT" : clk_cycle_unit
                    ,"CLOCK_DELAY_INPUT" : str(clk_delay_in)
                    ,"CLOCK_DELAY_UNIT_INPUT" : clk_delay_unit_in
                    ,"CLOCK_DELAY_OUTPUT" : str(clk_delay_out)
                    ,"CLOCK_DELAY_UNIT_OUTPUT" : clk_delay_unit_out
                    ,"LAYER" : layer
                    ,"NUM_FILTERS" : str(NUM_FILTERS)
                    ,"STRIDE" : str(STRIDE)
                    ,"KERNEL_SIZE" : str(KERNEL_SIZE)
                    ,"INPUT_SIZE" : str(INPUT_SIZE)
                    ,"INPUT_CHANNELS" : str(INPUT_CHANNELS)}
    )

# TODO: parameters...
@pytest.mark.parametrize("INPUT_SIZE", [(32)])
@pytest.mark.parametrize("OUTPUT_SIZE", [(32)])
def test_fc_layer(INPUT_SIZE, OUTPUT_SIZE):
    layer = "FC"
    dut = 'OpenEye_Parallel'
    module = 'OpenEye_Parallel_tb'
    toplevel = dut
    verilog_sources = ptu.get_verilog_sources(hdl_dir)

    target_dir = os.path.join(tests_dir, '.temp') 

    results = cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=target_dir,
        testcase='single_layer_test',
        force_compile=True,
        #waves=True,
        simulator="icarus",
        extra_env = {"CLOCK_LEN" : str(clk_cycle)
                    ,"CLOCK_UNIT" : clk_cycle_unit
                    ,"CLOCK_DELAY_INPUT" : str(clk_delay_in)
                    ,"CLOCK_DELAY_UNIT_INPUT" : clk_delay_unit_in
                    ,"CLOCK_DELAY_OUTPUT" : str(clk_delay_out)
                    ,"CLOCK_DELAY_UNIT_OUTPUT" : clk_delay_unit_out
                    ,"LAYER" : layer
                    ,"INPUT_SIZE" : str(INPUT_SIZE)
                    ,"OUTPUT_SIZE" : str(OUTPUT_SIZE)}
    )


if __name__ == '__main__':
    test_single_conv_layer()
