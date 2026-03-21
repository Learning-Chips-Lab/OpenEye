# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
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
from open_eye import hdl_dir

import pe_test_utils as ptu
from open_eye import hdl_dir, test_dir


#As ref:
#clk_cycle = 10; clk_cycle_unit = "ns"

clk_cycle = 10
clk_cycle_unit = "ns"

clk_delay_in = 100
clk_delay_unit_in = "ps"

clk_delay_out = 100
clk_delay_unit_out = "ps"

##########################################################################################

@pytest.mark.parametrize("B", [(4)])#,(3),(2),(1)]) # Number of blocks processed PE (stride U=1 here)
@pytest.mark.parametrize("C0", [(3)])#,(2),(1)]) # Input channels per PE (C0 in Eyeriss v2)
@pytest.mark.parametrize("M0", [(12)])#,(10),(8),(4)]) # Output channels per PE (M0 in Eyeriss v2)
@pytest.mark.parametrize("SPARSE_IACT", [(0)])#, (10), (20), (30), (40), (50), (60), (70), (80), (90)]) # Input activation sparsity
@pytest.mark.parametrize("SPARSE_WGHT", [(0)]) # Weight sparsity
@pytest.mark.parametrize("SEED", [0]) # Random seed
def test_single_pe(B, C0, M0, SPARSE_IACT, SPARSE_WGHT, SEED, request):
    dut = 'PE_IO_debug' # Name of the top-level module (without .v extension)
    module = 'PE_tb'
    toplevel = dut
    verilog_sources = ptu.get_verilog_sources(hdl_dir)
    nodeid = request.node.nodeid.replace("::", "_").replace("/", "_").replace("[","_").replace("]","_")
    target_dir = os.path.join(test_dir, '.temp/' + nodeid)
    os.makedirs(target_dir, exist_ok=True)

    results = cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=target_dir,
        testcase='start_test_pe',
        defines={"NO_TRACE": "TRUE"},  # Disable PE.v internal dumping, use cocotb's instead
        force_compile=True,
        waves=True,
        simulator="icarus",
        extra_env = {"CLOCK_LEN" : str(clk_cycle),
                    "CLOCK_UNIT" : clk_cycle_unit,
                    "CLOCK_DELAY_INPUT" : str(clk_delay_in),
                    "CLOCK_DELAY_UNIT_INPUT" : clk_delay_unit_in,
                    "CLOCK_DELAY_OUTPUT" : str(clk_delay_out),
                    "CLOCK_DELAY_UNIT_OUTPUT" : clk_delay_unit_out,
                    "B" : str(B),
                    "C0" : str(C0),
                    "M0" : str(M0),
                    "C0S" : str(B*C0),
                    "SPARSE_IACT" : str(SPARSE_IACT),
                    "SPARSE_WGHT" : str(SPARSE_WGHT),
                    "SEED" : str(SEED),
                    "SPARSITY_EN": "1",
                    "COCOTB_TRACE": "1",
                    "IVERILOG_DUMPER": "fst"},  # Enable FST waveform dumping for Icarus
    )

if __name__ == '__main__':
    # Run pytest programmatically
    pytest.main([__file__, "-v", "-s"])
