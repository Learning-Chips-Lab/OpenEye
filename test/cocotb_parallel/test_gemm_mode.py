# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
Pytest runner for the output-stationary (GEMM) dataflow plumbing test on
OpenEye_Parallel. See OpenEye_Parallel_gemm_tb.py for the checks performed.

Usage:
    pytest test_gemm_mode.py -v
"""

import os
import sys

import pytest
import cocotb_test.simulator

tests_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.extend([os.path.abspath(os.getcwd()), tests_dir])
hdl_dir = os.path.abspath(os.path.join(tests_dir, os.pardir, os.pardir, "hdl"))

import open_eye.test_utils_main as ptu

clk_cycle = 10
clk_cycle_unit = "ns"


@pytest.mark.parametrize("CLUSTER_ROWS", [2])
def test_gemm_mode_plumbing(CLUSTER_ROWS, request):
    nodeid = request.node.nodeid.replace("::", "_").replace("/", "_") \
                                .replace("[", "_").replace("]", "_")
    target_dir = os.path.join(tests_dir, ".temp", nodeid)
    os.makedirs(target_dir, exist_ok=True)

    # OpenEye_FPGA.v needs a generated parameters.vh; it is not part of the
    # OpenEye_Parallel hierarchy, so exclude it from the compile.
    verilog_sources = [
        src for src in ptu.get_verilog_sources(hdl_dir)
        if os.path.basename(src) != "OpenEye_FPGA.v"
    ]

    cocotb_test.simulator.run(
        python_search=[tests_dir],
        verilog_sources=verilog_sources,
        toplevel="OpenEye_Parallel",
        module="OpenEye_Parallel_gemm_tb",
        sim_build=target_dir,
        includes=[os.path.join(hdl_dir, "include")],
        parameters={"CLUSTER_ROWS": CLUSTER_ROWS},
        defines={"NO_TRACE": "TRUE"},
        force_compile=True,
        simulator="icarus",
        extra_env={
            "CLOCK_LEN": str(clk_cycle),
            "CLOCK_UNIT": clk_cycle_unit,
        },
    )
