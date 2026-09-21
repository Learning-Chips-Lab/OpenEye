# This file is part of the OpenEye project.
# SPDX-License-Identifier: SHL-2.1
"""Check convolution delivery independently of the DMA and MAC datapaths."""
import os

import cocotb_test.simulator
import pytest
from open_eye import hdl_dir, test_dir


@pytest.mark.parametrize("ports", [1, 3])
@pytest.mark.parametrize("channels", [1, 2, 4])
@pytest.mark.parametrize("x_start", [0, 4])
@pytest.mark.parametrize("height", [1, 3, 4])
def test_conv_mapping(ports, channels, x_start, height):
    cocotb_test.simulator.run(
        simulator="icarus",
        verilog_sources=[os.path.join(hdl_dir, name) for name in
                         ("iact_stream_constructor.v", "RAM_SP.v", "RAM_SP_generic.v")],
        toplevel="iact_stream_constructor", module="conv_mapping_tb",
        python_search=[os.path.dirname(__file__)],
        sim_build=os.path.join(test_dir, ".temp", f"conv_mapping_{ports}_{channels}_{x_start}_{height}"),
        parameters={"CLUSTER_ROWS": 2, "NUM_GLB_IACT": ports,
                    "WORD_BITWIDTH": 24 * ports, "ADDRWIDTH": 8},
        extra_env={"CONV_PORTS": str(ports), "CONV_CHANNELS": str(channels),
                   "CONV_X_START": str(x_start), "CONV_HEIGHT": str(height)},
    )
