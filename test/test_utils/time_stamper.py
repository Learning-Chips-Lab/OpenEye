# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import time

class time_stamper(object):
    """ The DRAM contents that is used to store intermediate data during tests.

    The DRAM contents is used for the simulation of OpenEye accelerator. It is used
    during the simulation to store the intermediate data of the accelerator
    between different layers or between different repetitions of the same layer.

    """

    def __init__(self) -> None:
        self.time_last_check = time.time()
        self.time_elapsed = time.time() - self.time_last_check

    def timestamp(self, message, logger):
        self.time_elapsed = time.time() - self.time_last_check
        self.time_last_check = time.time()
        logger.info(message + str(self.time_elapsed))
