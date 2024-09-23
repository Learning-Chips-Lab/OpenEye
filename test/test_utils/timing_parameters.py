# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
class PortTimingParameters(object):
    """ TODO: Docu - isn't the name misleading?"""

    def __init__(self) -> None:
        self.clk_cycle = 0
        self.clk_cycle_unit = 0
        self.clk_delay_in = 0
        self.clk_delay_unit_in = 0
        self.clk_delay_out = 0
        self.clk_delay_unit_out = 0

    def initiate_params(self, clk_cycle, clk_cycle_unit, clk_delay_in, clk_delay_unit_in, clk_delay_out, clk_delay_unit_out):
        self.clk_cycle = clk_cycle
        self.clk_cycle_unit = clk_cycle_unit
        self.clk_delay_in = clk_delay_in
        self.clk_delay_unit_in = clk_delay_unit_in
        self.clk_delay_out = clk_delay_out
        self.clk_delay_unit_out = clk_delay_unit_out
        