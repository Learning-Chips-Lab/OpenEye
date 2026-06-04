# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
Pytest runner for the three GEMM approaches on PE_cluster.

Approach 1 – zero RTL changes, software-only iact routing
Approach 2 – gemm_mode_i flag (moderate RTL, compiled into standard cluster)
Approach 3 – systolic pass-through (SYSTOLIC_GEMM_EN=1 override compilation)

Usage:
    pytest test_PE_CLUSTER_GEMM.py -v
    pytest test_PE_CLUSTER_GEMM.py::test_pe_cluster_gemm_approach1 -v
    pytest test_PE_CLUSTER_GEMM.py::test_pe_cluster_gemm_approach3 -v
"""

import os
import sys
import pytest
import cocotb_test.simulator

directory = os.path.abspath(os.getcwd())
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
tests_dir = os.path.abspath(os.path.dirname(__file__))

import pe_cluster_test_utils as pctu
from open_eye import hdl_dir, test_dir

clk_cycle          = 10
clk_cycle_unit     = "ns"
clk_delay_in       = 100
clk_delay_unit_in  = "ps"
clk_delay_out      = 100
clk_delay_unit_out = "ps"

_common_env = {
    "CLOCK_LEN":              str(clk_cycle),
    "CLOCK_UNIT":             clk_cycle_unit,
    "CLOCK_DELAY_INPUT":      str(clk_delay_in),
    "CLOCK_DELAY_UNIT_INPUT": clk_delay_unit_in,
    "CLOCK_DELAY_OUTPUT":     str(clk_delay_out),
    "CLOCK_DELAY_UNIT_OUTPUT": clk_delay_unit_out,
}


# ---------------------------------------------------------------------------
# Approach 1 – pure software mapping (gemm_mode_i low, iact_choose_i routing)
# IACTSIZE_X * IACTSIZE_Y = K (inner dimension); WGHTSIZE_X = N (output columns)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("IACTSIZE_X", [4])
@pytest.mark.parametrize("IACTSIZE_Y", [1])
@pytest.mark.parametrize("WGHTSIZE_X", [6])
@pytest.mark.parametrize("SEED",       [0, 1, 2])
def test_pe_cluster_gemm_approach1(IACTSIZE_X, IACTSIZE_Y, WGHTSIZE_X, SEED, request):
    nodeid     = request.node.nodeid.replace("::", "_").replace("/", "_") \
                                    .replace("[", "_").replace("]", "_")
    target_dir = os.path.join(test_dir, ".temp", nodeid)
    os.makedirs(target_dir, exist_ok=True)

    cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=pctu.get_verilog_sources(hdl_dir),
        toplevel="PE_cluster",
        module="PE_cluster_gemm_approach1_tb",
        sim_build=target_dir,
        testcase="start_test_gemm_approach1",
        defines={"NO_TRACE": "TRUE"},
        force_compile=True,
        simulator="icarus",
        extra_env={
            **_common_env,
            "IACTSIZE_X":  str(IACTSIZE_X),
            "IACTSIZE_Y":  str(IACTSIZE_Y),
            "WGHTSIZE_X":  str(WGHTSIZE_X),
            "SPARSE_IACT": "0",
            "SPARSE_WGHT": "0",
            "SEED":        str(SEED),
        },
    )


# ---------------------------------------------------------------------------
# Approach 2 – gemm_mode_i flag
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("IACTSIZE_X", [4])
@pytest.mark.parametrize("IACTSIZE_Y", [1])
@pytest.mark.parametrize("WGHTSIZE_X", [6])
@pytest.mark.parametrize("SEED",       [0, 1, 2])
def test_pe_cluster_gemm_approach2(IACTSIZE_X, IACTSIZE_Y, WGHTSIZE_X, SEED, request):
    nodeid     = request.node.nodeid.replace("::", "_").replace("/", "_") \
                                    .replace("[", "_").replace("]", "_")
    target_dir = os.path.join(test_dir, ".temp", nodeid)
    os.makedirs(target_dir, exist_ok=True)

    cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=pctu.get_verilog_sources(hdl_dir),
        toplevel="PE_cluster",
        module="PE_cluster_gemm_approach2_tb",
        sim_build=target_dir,
        testcase="start_test_gemm_approach2",
        defines={"NO_TRACE": "TRUE"},
        force_compile=True,
        simulator="icarus",
        extra_env={
            **_common_env,
            "IACTSIZE_X":  str(IACTSIZE_X),
            "IACTSIZE_Y":  str(IACTSIZE_Y),
            "WGHTSIZE_X":  str(WGHTSIZE_X),
            "SPARSE_IACT": "0",
            "SPARSE_WGHT": "0",
            "SEED":        str(SEED),
        },
    )


# ---------------------------------------------------------------------------
# Approach 3 – systolic pass-through (SYSTOLIC_GEMM_EN=1)
# IACTSIZE_X * IACTSIZE_Y = K; WGHTSIZE_X must equal PE_COLUMNS (=4 default)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("IACTSIZE_X", [4])
@pytest.mark.parametrize("IACTSIZE_Y", [1])
@pytest.mark.parametrize("WGHTSIZE_X", [1])
@pytest.mark.parametrize("SEED",       [0, 1, 2])
def test_pe_cluster_gemm_approach3(IACTSIZE_X, IACTSIZE_Y, WGHTSIZE_X, SEED, request):
    """
    Recompiles PE_cluster with SYSTOLIC_GEMM_EN=1 and PE_ROWS=3 / PE_COLUMNS=4
    (the default parameters).  The iact_pass_* ports are then active.
    """
    nodeid     = request.node.nodeid.replace("::", "_").replace("/", "_") \
                                    .replace("[", "_").replace("]", "_")
    target_dir = os.path.join(test_dir, ".temp", nodeid)
    os.makedirs(target_dir, exist_ok=True)

    cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=pctu.get_verilog_sources(hdl_dir),
        toplevel="PE_cluster",
        module="PE_cluster_gemm_approach3_tb",
        sim_build=target_dir,
        testcase="start_test_gemm_approach3",
        parameters={"SYSTOLIC_GEMM_EN": 1, "PARALLEL_MACS": 1},
        defines={"NO_TRACE": "TRUE"},
        force_compile=True,
        simulator="icarus",
        extra_env={
            **_common_env,
            "IACTSIZE_X":  str(IACTSIZE_X),
            "IACTSIZE_Y":  str(IACTSIZE_Y),
            "WGHTSIZE_X":  str(WGHTSIZE_X),
            "SEED":        str(SEED),
        },
    )
