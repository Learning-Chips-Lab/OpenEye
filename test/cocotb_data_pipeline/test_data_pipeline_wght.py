# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
"""Runner for the data_pipeline_wght unit testbench.

Sweeps zero patterns that the cluster-level tests only reach through a random
mask, so a failure names a specific weight position instead of a PSUM delta.
"""
import os
import sys

import pytest
import cocotb_test.simulator

from open_eye import hdl_dir, test_dir

tests_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, tests_dir)


def run_case(request, zeros, parallel_macs=2, filters_w=6, rows=4):
    nodeid = request.node.nodeid.replace("::", "_").replace("/", "_") \
                               .replace("[", "_").replace("]", "_")
    target_dir = os.path.join(test_dir, ".temp", nodeid)
    os.makedirs(target_dir, exist_ok=True)

    cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=[os.path.join(hdl_dir, "data_pipeline_wght.v")],
        toplevel="data_pipeline_wght",
        module="data_pipeline_wght_tb",
        sim_build=target_dir,
        testcase="test_weight_positions",
        # SECOND_SPAD_DATA is the full SPAD word: PE.v passes WGHT_DATA_DATA =
        # (DATA_WGHT_BITWIDTH + DATA_WGHT_IGNORE_ZEROS) * PARALLEL_MACS, so one
        # SPAD word holds all PARALLEL_MACS sub-words, not just one.
        parameters={"DATA_WIDTH": 12 * parallel_macs,
                    "SECOND_SPAD_DATA": 12 * parallel_macs,
                    "SECOND_PAYLOAD_WIDTH": 8,
                    "SPARSITY_EN": 1},
        defines={"NO_TRACE": "TRUE"},
        force_compile=True,
        simulator="icarus",
        extra_env={"PARALLEL_MACS": str(parallel_macs),
                   "FILTERS_W": str(filters_w),
                   "ROWS": str(rows),
                   "ZEROS": ",".join(str(z) for z in zeros)},
    )


# Patterns chosen from the PE_cluster reproducer: a single zero anywhere is
# handled, but two zeros inside one row are the case that fails there.
@pytest.mark.parametrize("zeros", [
    (),            # dense control
    (0,),          # one zero, first position
    (1,),          # one zero, interior
    (5,),          # one zero, last position of a row
    (13,),         # one zero, second row
    (13, 16),      # two zeros, same row  <- cluster-level failure
    (12, 16),      # two zeros, same row, from the row start
    (13, 21),      # two zeros, different rows (passes at cluster level)
    (0, 1),        # adjacent zeros
    # An entirely zero weight row emits no sub-words, and the pipeline does not
    # advance the first SPAD address for it: the expected boundaries [3, 6, 6, 9]
    # come back as [3, 6, 9, None], so every later row is stored one address too
    # low and the compute engine segments the stream in the wrong places. This is
    # the remaining PE_cluster sparse failure (all 5 left in the 132-case sample
    # are high-sparsity, where empty rows become likely). Reproduce at cluster
    # level with WGHT_ZERO_POS="12,13,14,15,16,17".
    pytest.param((12, 13, 14, 15, 16, 17),
                 marks=pytest.mark.xfail(reason="empty weight row does not advance "
                                                "the first SPAD address", strict=True)),
    pytest.param((0, 1, 2, 3, 4, 5),
                 marks=pytest.mark.xfail(reason="empty weight row does not advance "
                                                "the first SPAD address", strict=True)),
])
def test_wght_positions(zeros, request):
    run_case(request, zeros)
