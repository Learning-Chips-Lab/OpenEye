# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
pytest runner for PE_simple cocotb simulation.

Run with:
    pytest test_PE_simple.py -v
or for a specific parameter set:
    pytest test_PE_simple.py -v -k "B4_C3_M12"
"""

import os
from pathlib import Path
import pytest
import cocotb_test.simulator

from open_eye import hdl_dir, test_dir

# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------
B_VALUES    = [4]    # spatial window size (iact values per channel)
C0_VALUES   = [3]    # input channels
M0_VALUES   = [12]   # output filters
SEED_VALUES = [0, 42]

CLK_CYCLE        = 10
CLK_CYCLE_UNIT   = "ns"
CLK_DELAY_INPUT  = 100
CLK_DELAY_IN_U   = "ps"
CLK_DELAY_OUTPUT = 100
CLK_DELAY_OUT_U  = "ps"


@pytest.mark.parametrize("B",    B_VALUES)
@pytest.mark.parametrize("C0",   C0_VALUES)
@pytest.mark.parametrize("M0",   M0_VALUES)
@pytest.mark.parametrize("SEED", SEED_VALUES)
def test_pe_simple(B, C0, M0, SEED):
    test_name  = f"pe_simple_B{B}_C{C0}_M{M0}_seed{SEED}"
    target_dir = Path(__file__).parent / ".temp" / test_name
    target_dir.mkdir(parents=True, exist_ok=True)

    cocotb_test.simulator.run(
        python_search=[str(Path(__file__).parent)],
        verilog_sources=[str(Path(hdl_dir) / "PE_simple.v")],
        toplevel="PE_simple",
        module="PE_simple_tb",
        sim_build=str(target_dir),
        testcase="test_pe_simple",
        defines={"NO_TRACE": "TRUE"},
        force_compile=True,
        simulator="icarus",
        extra_env={
            "CLOCK_LEN":              str(CLK_CYCLE),
            "CLOCK_UNIT":             CLK_CYCLE_UNIT,
            "CLOCK_DELAY_INPUT":      str(CLK_DELAY_INPUT),
            "CLOCK_DELAY_UNIT_INPUT": CLK_DELAY_IN_U,
            "CLOCK_DELAY_OUTPUT":     str(CLK_DELAY_OUTPUT),
            "CLOCK_DELAY_UNIT_OUTPUT":CLK_DELAY_OUT_U,
            "B":    str(B),
            "C0":   str(C0),
            "M0":   str(M0),
            "SEED": str(SEED),
            "IVERILOG_DUMPER": "fst",
        },
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
