# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import logging
import os
import sys

import pytest
logger = logging.getLogger("cocotb")

directory = (os.path.abspath(os.getcwd()))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
tests_dir = os.path.abspath(os.path.dirname(__file__))
hdl_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir, "hdl")

import pe_cluster_test_utils as pctu
from cluster_simulator import run_cluster_simulation
from open_eye import hdl_dir, test_dir, open_eye_dir, vh_file_creator


#As ref:
#clk_cycle = 10; clk_cycle_unit = "ns"

clk_cycle = 10
clk_cycle_unit = "ns"

clk_delay_in = 100
clk_delay_unit_in = "ps"

clk_delay_out = 100
clk_delay_unit_out = "ps"

# Dimensions for PE Cluster initiliazed as globals
iactsize_x = 0  # Number of input activation values (spatial dimension)
iactsize_y = 0  # Number of input channels
sparse_iact = 0 # Input activation sparsity: 0 = no sparsity, 1 = fully sparse
wghtsize_x = 0  # Number of output filters
wghtsize_y = 0  # Weights match input dimensions
sparse_wght = 0 # Weight sparsity: 0 = no sparsity, 1 = fully sparse

##########################################################################################

def test_pe_cluster_conv(IACTSIZE_X, IACTSIZE_Y, WGHTSIZE_X, SPARSE_IACT,SPARSE_WGHT,SEED,PARALLEL_MACS,SPARSITY_EN,request):
    dut = 'PE_cluster'
    module = 'PE_cluster_tb'
    toplevel = dut
    verilog_sources = pctu.get_verilog_sources(hdl_dir)
    nodeid = request.node.nodeid.replace("::", "_").replace("/", "_").replace("[","_").replace("]","_")
    target_dir = os.path.join(test_dir, '.temp/' + nodeid)
    os.makedirs(target_dir, exist_ok=True)

    results = run_cluster_simulation(
        python_search=[tests_dir],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=target_dir,
        testcase='start_test_pe',
        defines={"NO_TRACE": "TRUE"},
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
                    ,"SPARSE_WGHT" : str(SPARSE_WGHT)
                    ,"PARALLEL_MACS" : str(PARALLEL_MACS)
                    ,"SPARSITY_EN" : str(SPARSITY_EN)
                    ,"COCOTB_TRACE": "1"
                    ,"SEED" : str(SEED)}
    )

