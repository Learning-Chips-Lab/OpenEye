# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import sys
import os
directory = (os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])


def get_verilog_sources(hdl_dir):

    verilog_sources =[
    os.path.join(hdl_dir, "PE.v"),
    os.path.join(hdl_dir, "adder.v"),
    os.path.join(hdl_dir, "data_pipeline.v"),
    os.path.join(hdl_dir, "multiplier.v"),
    os.path.join(hdl_dir, "mux2.v"),
    os.path.join(hdl_dir, "demux2.v"),
    os.path.join(hdl_dir, "mux_iact.v"),
    os.path.join(hdl_dir, "SPad_DP_RW.v"),
    os.path.join(hdl_dir, "SPad_SP.v"),
    os.path.join(hdl_dir, "RST_SYNC.v"),
    os.path.join(hdl_dir, "memory/RAM_DP_RW.v"),
    os.path.join(hdl_dir, "memory/RAM_DP.v"),
    os.path.join(hdl_dir, "memory/RAM_SP.v"),
    os.path.join(hdl_dir, "memory/impl/RAM_DP_RW_generic.v"),
    os.path.join(hdl_dir, "memory/impl/RAM_SP_generic.v")
    ]
    return verilog_sources
