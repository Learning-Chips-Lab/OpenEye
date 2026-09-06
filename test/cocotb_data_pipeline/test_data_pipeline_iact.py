# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
"""Runner for the data_pipeline_iact unit testbench."""
import os
import sys

import pytest
import cocotb_test.simulator

from open_eye import hdl_dir, test_dir

tests_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, tests_dir)


@pytest.mark.parametrize("zeros", [
    (),        # dense control
    (0,),      # first activation zero  <- fails at cluster level
    (1,),
    (2,),
    (3,),      # last activation zero   <- passes at cluster level
    (0, 1),
])
def test_iact_positions(zeros, request):
    nodeid = request.node.nodeid.replace("::", "_").replace("/", "_") \
                               .replace("[", "_").replace("]", "_")
    target_dir = os.path.join(test_dir, ".temp", nodeid)
    os.makedirs(target_dir, exist_ok=True)

    cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=[os.path.join(hdl_dir, "data_pipeline_iact.v")],
        toplevel="data_pipeline_iact",
        module="data_pipeline_iact_tb",
        sim_build=target_dir,
        testcase="test_iact_positions",
        # DATA_WIDTH 12 with SECOND_SPAD_DATA 12 gives SECOND_SPAD_DATA_CYCLE
        # of 1: one activation per transfer, which is what the cluster sends for
        # PARALLEL_MACS=1 (TRANS_BITWIDTH_IACT = 1 * (8 + 4)).
        parameters={"DATA_WIDTH": 12, "SECOND_SPAD_DATA": 12,
                    "SECOND_PAYLOAD_WIDTH": 8, "SPARSITY_EN": 1},
        defines={"NO_TRACE": "TRUE"},
        force_compile=True,
        simulator="icarus",
        extra_env={"VALUES_PER_WORD": "1", "CHANNELS": "2", "PER_CHANNEL": "2",
                   "ZEROS": ",".join(str(z) for z in zeros)},
    )
