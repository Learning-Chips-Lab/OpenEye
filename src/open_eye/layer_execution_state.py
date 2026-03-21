# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

class LayerExecutionState(object):

    def __init__(self) -> None:
        # variables to assign the output of the stream to the dram
        # filters (= id of the output feature map)
        self.f_corner_start = 0
        self.f_start = 0
        self.f_end = 0
        self.f_corner_end = 0

        # x and y are the coordinates of the output feature map
        self.x_corner_start = 0
        self.x_last = 0
        self.x_start = 0
        self.x_end = 0
        self.x_corner_end = 0

        self.y_corner_start = 0
        self.y_start = 0
        self.y_end = 0
        self.y_corner_end = 0

        self.current_position = 0
