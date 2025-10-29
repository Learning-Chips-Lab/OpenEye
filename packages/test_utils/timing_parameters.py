# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Clock and timing parameter management for OpenEye hardware simulation.

This module provides configuration and management of timing parameters for
the OpenEye neural network accelerator simulation. It handles:

- Clock cycle configuration
- Input/output delay timing
- Timing units specification
- Clock domain synchronization parameters

Key capabilities:
- Configurable clock cycle lengths
- Independent input/output delay settings
- Flexible timing unit definitions
- Simulation timing control
"""

class PortTimingParameters(object):
    """Clock timing configuration for OpenEye hardware interfaces.

    This class manages timing parameters for hardware interfaces in the OpenEye
    accelerator simulation. It handles clock cycle definitions and signal
    propagation delays for both input and output paths.

    Attributes:
        clk_cycle (int): Clock cycle duration
        clk_cycle_unit (int): Time unit for clock cycle (e.g., ns, ps)
        clk_delay_in (int): Input path delay duration
        clk_delay_unit_in (int): Time unit for input delay
        clk_delay_out (int): Output path delay duration
        clk_delay_unit_out (int): Time unit for output delay

    Note:
        The name PortTimingParameters reflects its role in managing timing
        for hardware interface ports, including both input and output paths
        and their associated clock parameters.
    """

    def __init__(self) -> None:
        """Initialize timing parameters with default values.
        
        Creates a new timing parameter object with all values initialized
        to zero. Use initiate_params() to set actual timing values.
        """
        self.clk_cycle = 0
        self.clk_cycle_unit = 0
        self.clk_delay_in = 0
        self.clk_delay_unit_in = 0
        self.clk_delay_out = 0
        self.clk_delay_unit_out = 0

    def initiate_params(self, clk_cycle: int, clk_cycle_unit: int, 
                       clk_delay_in: int, clk_delay_unit_in: int, 
                       clk_delay_out: int, clk_delay_unit_out: int) -> None:
        """Configure timing parameters for hardware interface.

        Args:
            clk_cycle: Duration of one clock cycle
            clk_cycle_unit: Time unit for clock cycle (e.g., ns, ps)
            clk_delay_in: Input path propagation delay
            clk_delay_unit_in: Time unit for input delay
            clk_delay_out: Output path propagation delay
            clk_delay_unit_out: Time unit for output delay

        The method sets up all timing parameters needed for accurate
        hardware interface simulation. All time values must be specified
        with their corresponding units for proper timing calculations.
        """
        self.clk_cycle = clk_cycle
        self.clk_cycle_unit = clk_cycle_unit
        self.clk_delay_in = clk_delay_in
        self.clk_delay_unit_in = clk_delay_unit_in
        self.clk_delay_out = clk_delay_out
        self.clk_delay_unit_out = clk_delay_unit_out
        