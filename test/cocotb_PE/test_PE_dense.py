# This file is part of the OpenEye project.
# All rights reserved. © University of Duisburg-Essen.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
PE Dense Mode Test Runner

This test runner validates the PE (Processing Element) in dense mode (SPARSITY_EN=0),
where no sparsity exploitation is used. Data is transmitted without overhead bits,
reducing bandwidth by 33% compared to sparse mode.

Test Configuration:
- SPARSITY_EN=0 (dense mode)
- Various dimensions (U, C0, M0)
- 0% sparsity (all values are assumed to be non-zero/zeros are transmitted as-is)
- Tests dense MAC computation without sparse indexing
"""

import logging
import pytest
import os
import sys
from pathlib import Path

import pytest
import cocotb_test.simulator
logger = logging.getLogger("cocotb")

from open_eye import hdl_dir, test_dir
import pe_test_utils as ptu

# Test parameters for dense mode
U_VALUES = [4]  # Filter width (S in Eyeriss v2) - spatial window dimension
C0_VALUES = [3]  # Input channels per PE (C0 in Eyeriss v2)
M0_VALUES = [12] # Output channels per PE (M0 in Eyeriss v2)
SEED_VALUES = [0]            # Random seeds
USE_DSP_VALUES = [0, 1]      # 0=standard multiplier+adder, 1=DSP48 optimization

# Clock configuration
CLK_CYCLE = 10
CLK_CYCLE_UNIT = "ns"
CLK_DELAY_INPUT = 100
CLK_DELAY_UNIT_INPUT = "ps"
CLK_DELAY_OUTPUT = 100
CLK_DELAY_UNIT_OUTPUT = "ps"


@pytest.mark.parametrize("U", U_VALUES)
@pytest.mark.parametrize("C0", C0_VALUES)
@pytest.mark.parametrize("M0", M0_VALUES)
@pytest.mark.parametrize("SEED", SEED_VALUES)
@pytest.mark.parametrize("USE_DSP", USE_DSP_VALUES)
def test_pe_dense_mode(U, C0, M0, SEED, USE_DSP):
    """
    Test PE in dense mode (SPARSITY_EN=0) with selectable MAC implementation.

    Validates:
    - Correct MAC computation without sparsity encoding
    - No overhead bits in data transmission
    - Address SPADs are excluded from synthesis
    - Linear psum addressing (no sparse offsets)
    - Both standard (USE_DSP=0) and DSP48 (USE_DSP=1) implementations

    Args:
        U: Number of input activation values
        C0: Number of input channels
        M0: Number of output filters
        SEED: Random seed for reproducibility
        USE_DSP: MAC implementation (0=standard multiplier+adder, 1=DSP48 slice)
    """
    # Get HDL files
    from open_eye import hdl_dir
    verilog_sources = ptu.get_verilog_sources(str(hdl_dir))

    # Add dsp_unit.v when USE_DSP=1
    if USE_DSP == 1:
        dsp_unit_path = Path(hdl_dir) / "dsp_unit.v"
        if dsp_unit_path.exists():
            verilog_sources.append(str(dsp_unit_path))

    # Test module configuration
    module = "PE_tb"
    toplevel = "PE"

    # Create temporary directory for this test
    dsp_mode_str = "dsp" if USE_DSP == 1 else "std"
    test_name = f"test_dense_iact{U}x{C0}_wght{M0}_seed{SEED}_{dsp_mode_str}"
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
        "U": str(U),
        "C0": str(C0),
        "M0": str(M0),

        # Dense mode configuration
        "SPARSITY_EN": "0",  # *** DENSE MODE ***
        "SPARSE_IACT": "0",  # 0% sparsity (all values present)
        "SPARSE_WGHT": "0",  # 0% sparsity

        # Random seed
        "SEED": str(SEED),

        # Waveform generation for Icarus Verilog
        "IVERILOG_DUMPER": "fst"  # Enable FST waveform dumping
    }

    # Run simulation with dense mode and USE_DSP parameters
    results = cocotb_test.simulator.run(
        python_search=[str(test_dir)],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=str(target_dir),
        testcase="start_test_pe",
        defines={"NO_TRACE": "TRUE"},
        force_compile=True,
        waves=True,  # Enable waveforms for debugging
        simulator="icarus",
        extra_env=extra_env,
        parameters={"SPARSITY_EN": 0, "USE_DSP": USE_DSP}  # Pass both parameters to Verilog
    )


if __name__ == "__main__":
    # Run pytest programmatically
    pytest.main([__file__, "-v", "-s"])
