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

import pe_test_utils as ptu


#As ref:
#clk_cycle = 10; clk_cycle_unit = "ns"

clk_cycle = 10
clk_cycle_unit = "ns"

clk_delay_in = 100
clk_delay_unit_in = "ps"

clk_delay_out = 100
clk_delay_unit_out = "ps"

##########################################################################################

@pytest.mark.parametrize("IACTSIZE_X", [(4),(3),(2),(1)])
@pytest.mark.parametrize("IACTSIZE_Y", [(3),(2),(1)])
@pytest.mark.parametrize("WGHTSIZE_X", [(12),(10),(8),(1)])
@pytest.mark.parametrize("SPARSE_IACT", [(0),(0.1),(0.2)])
@pytest.mark.parametrize("SPARSE_WGHT", [(0),(0.1),(0.2),(0.3),(0.4)])
def test_single_pe(IACTSIZE_X, IACTSIZE_Y, WGHTSIZE_X, SPARSE_IACT,SPARSE_WGHT):
    dut = 'PE'
    module = 'PE_tb'
    toplevel = dut
    verilog_sources = ptu.get_verilog_sources(hdl_dir)

    target_dir = os.path.join(tests_dir, '.temp') 

    results = cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=target_dir,
        testcase='start_test_pe',
        force_compile=True,
        waves=True,
        simulator="icarus",
        extra_env = {"CLOCK_LEN" : str(clk_cycle)
                    ,"CLOCK_UNIT" : clk_cycle_unit
                    ,"CLOCK_DELAY_INPUT" : str(clk_delay_in)
                    ,"CLOCK_DELAY_UNIT_INPUT" : clk_delay_unit_in
                    ,"CLOCK_DELAY_OUTPUT" : str(clk_delay_out)
                    ,"CLOCK_DELAY_UNIT_OUTPUT" : clk_delay_unit_out
                    ,"IACTSIZE_X" : str(IACTSIZE_X)
                    ,"IACTSIZE_Y" : str(IACTSIZE_Y)
                    ,"WGHTSIZE_X" : str(WGHTSIZE_X)
                    ,"WGHTSIZE_Y" : str(IACTSIZE_X*IACTSIZE_Y)
                    ,"SPARSE_IACT" : str(SPARSE_IACT)
                    ,"SPARSE_WGHT" : str(SPARSE_WGHT)}
    )

if __name__ == '__main__':
    test_single_pe()
