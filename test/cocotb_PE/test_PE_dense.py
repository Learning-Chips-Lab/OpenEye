# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
PE Dense Mode Test Runner

This test runner validates the PE (Processing Element) in dense mode (SPARSITY_EN=0),
where no sparsity exploitation is used. Data is transmitted without overhead bits,
reducing bandwidth by 33% compared to sparse mode.

Test Configuration:
- SPARSITY_EN=0 (dense mode)
- Various dimensions (IACTSIZE_X, IACTSIZE_Y, WGHTSIZE_X)
- 0% sparsity (all values non-zero)
- Tests dense MAC computation without sparse indexing
"""

import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path for OpenEye imports
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

import cocotb_test.simulator
import open_eye.pe_test_utils as pe_test_utils

# Test parameters for dense mode
IACTSIZE_X_VALUES = [2, 3, 4]  # Input activation width
IACTSIZE_Y_VALUES = [1, 2]      # Input channels
WGHTSIZE_X_VALUES = [4, 8]      # Output filters
SEED_VALUES = [0, 42]            # Random seeds

# Clock configuration
CLK_CYCLE = 10
CLK_CYCLE_UNIT = "ns"
CLK_DELAY_INPUT = 100
CLK_DELAY_UNIT_INPUT = "ps"
CLK_DELAY_OUTPUT = 100
CLK_DELAY_UNIT_OUTPUT = "ps"


@pytest.mark.parametrize("IACTSIZE_X", IACTSIZE_X_VALUES)
@pytest.mark.parametrize("IACTSIZE_Y", IACTSIZE_Y_VALUES)
@pytest.mark.parametrize("WGHTSIZE_X", WGHTSIZE_X_VALUES)
@pytest.mark.parametrize("SEED", SEED_VALUES)
def test_pe_dense_mode(IACTSIZE_X, IACTSIZE_Y, WGHTSIZE_X, SEED):
    """
    Test PE in dense mode (SPARSITY_EN=0).

    Validates:
    - Correct MAC computation without sparsity encoding
    - No overhead bits in data transmission
    - Address SPADs are excluded from synthesis
    - Linear psum addressing (no sparse offsets)

    Args:
        IACTSIZE_X: Number of input activation values
        IACTSIZE_Y: Number of input channels
        WGHTSIZE_X: Number of output filters
        SEED: Random seed for reproducibility
    """
    # Get HDL files
    hdl_dir = repo_root / "hdl"
    verilog_sources = pe_test_utils.get_verilog_sources(str(hdl_dir))

    # Test module configuration
    module = "PE_tb"
    toplevel = "PE"

    # Create temporary directory for this test
    test_name = f"test_dense_iact{IACTSIZE_X}x{IACTSIZE_Y}_wght{WGHTSIZE_X}_seed{SEED}"
    target_dir = Path(__file__).parent / ".temp" / test_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Environment variables for test configuration
    extra_env = {
        # Clock timing
        "CLOCK_LEN": str(CLK_CYCLE),
        "CLOCK_UNIT": CLK_CYCLE_UNIT,
        "CLOCK_DELAY_INPUT": str(CLK_DELAY_INPUT),
        "CLOCK_DELAY_UNIT_INPUT": CLK_DELAY_UNIT_INPUT,
        "CLOCK_DELAY_OUTPUT": str(CLK_DELAY_OUTPUT),
        "CLOCK_DELAY_UNIT_OUTPUT": CLK_DELAY_UNIT_OUTPUT,

        # Test dimensions
        "IACTSIZE_X": str(IACTSIZE_X),
        "IACTSIZE_Y": str(IACTSIZE_Y),
        "WGHTSIZE_X": str(WGHTSIZE_X),

        # Dense mode configuration
        "SPARSITY_EN": "0",  # *** DENSE MODE ***
        "SPARSE_IACT": "0",  # 0% sparsity (all values present)
        "SPARSE_WGHT": "0",  # 0% sparsity

        # Random seed
        "SEED": str(SEED)
    }

    # Run simulation with dense mode parameter
    results = cocotb_test.simulator.run(
        python_search=[str(repo_root / "test" / "cocotb_PE"), str(repo_root / "src")],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=str(target_dir),
        testcase="start_test_pe",
        defines={"NO_TRACE": "TRUE"},
        force_compile=True,
        waves=False,  # Disable waveforms for faster testing
        simulator="icarus",
        extra_env=extra_env,
        parameters={"SPARSITY_EN": 0}  # Pass SPARSITY_EN=0 to Verilog
    )


if __name__ == "__main__":
    # Run pytest programmatically
    pytest.main([__file__, "-v", "-s"])
