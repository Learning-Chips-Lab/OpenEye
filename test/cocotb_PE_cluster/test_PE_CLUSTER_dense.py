# Dense-mode test runner for PE_cluster
import logging
import pytest
import os
import sys
from pathlib import Path
import cocotb_test.simulator
logger = logging.getLogger("cocotb")

from open_eye import hdl_dir, test_dir

# pe_cluster_test_utils sits next to this file, not on sys.path by default
sys.path.extend([os.path.abspath(os.getcwd()), os.path.dirname(os.path.realpath(__file__))])
import pe_cluster_test_utils as pctu

# Clock configuration
CLK_CYCLE = 10
CLK_CYCLE_UNIT = "ns"
CLK_DELAY_INPUT = 100
CLK_DELAY_UNIT_INPUT = "ps"
CLK_DELAY_OUTPUT = 100
CLK_DELAY_UNIT_OUTPUT = "ps"

# Small dense-mode test parameters
IACTSIZE_X = 4
IACTSIZE_Y = 3
WGHTSIZE_X = 8
WGHTSIZE_Y = IACTSIZE_X * IACTSIZE_Y
SEED = 0

@pytest.mark.parametrize("PARALLEL_MACS", [1, 2])
def test_pe_cluster_dense(PARALLEL_MACS):
    dut = 'PE_cluster'
    module = 'PE_cluster_tb'
    toplevel = dut
    verilog_sources = pctu.get_verilog_sources(hdl_dir)

    # target dir
    test_name = f"pe_cluster_dense_par{PARALLEL_MACS}_seed{SEED}"
    target_dir = Path(__file__).parent / ".temp" / test_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # set compile-time parameters via env + generate parameters.vh
    os.environ["PARALLEL_MACS"] = str(PARALLEL_MACS)
    os.environ["SPARSITY_EN"] = "0"
    from open_eye import vh_file_creator
    vh_file_creator.create_vh_file_from_envvars(str(target_dir), str(hdl_dir) + "/", toplevel=toplevel)

    extra_env = {
        "CLOCK_LEN": str(CLK_CYCLE),
        "CLOCK_UNIT": CLK_CYCLE_UNIT,
        "CLOCK_DELAY_INPUT": str(CLK_DELAY_INPUT),
        "CLOCK_DELAY_UNIT_INPUT": CLK_DELAY_UNIT_INPUT,
        "CLOCK_DELAY_OUTPUT": str(CLK_DELAY_OUTPUT),
        "CLOCK_DELAY_UNIT_OUTPUT": CLK_DELAY_UNIT_OUTPUT,
        "IACTSIZE_X": str(IACTSIZE_X),
        "IACTSIZE_Y": str(IACTSIZE_Y),
        "WGHTSIZE_X": str(WGHTSIZE_X),
        "WGHTSIZE_Y": str(WGHTSIZE_Y),
        "SPARSE_IACT": "0",
        "SPARSE_WGHT": "0",
        "SEED": str(SEED),
        "PARALLEL_MACS": str(PARALLEL_MACS),
        "SPARSITY_EN": "0",
    }

    results = cocotb_test.simulator.run(
        python_search=[str(test_dir)],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=str(target_dir),
        testcase='start_test_pe',
        defines={"NO_TRACE": "TRUE"},
        force_compile=True,
        waves=True,
        simulator="icarus",
        extra_env=extra_env,
        parameters={"SPARSITY_EN": 0, "PARALLEL_MACS": PARALLEL_MACS}
    )

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])