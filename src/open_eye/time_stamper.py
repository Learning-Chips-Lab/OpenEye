# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Time measurement utility for OpenEye accelerator testing.

This module provides functionality for measuring and logging elapsed time
between operations during OpenEye accelerator testing. It enables:

- Tracking execution time between checkpoints
- Logging time measurements with custom messages
- Performance profiling of accelerator operations

The module is particularly useful for:
- Measuring layer execution times
- Profiling data transfer operations
- Analyzing system bottlenecks
- Validating timing requirements
"""

import time

class time_stamper(object):
    """Time measurement and logging utility for OpenEye testing.

    This class provides functionality to measure elapsed time between operations
    and log the measurements with descriptive messages. It maintains a running
    timer and calculates time differences between checkpoints.

    Attributes:
        time_last_check (float): Timestamp of the last measurement point
        time_elapsed (float): Time elapsed since last checkpoint
        
    Example:
        ts = time_stamper()
        # ... perform some operation ...
        ts.timestamp("Time for operation: ", logger)
    """

    def __init__(self) -> None:
        """Initialize the time stamper.
        
        Creates a new time stamper instance and initializes the timing
        attributes. The initial elapsed time is set to zero.
        """
        self.time_last_check = time.time()
        self.time_elapsed = time.time() - self.time_last_check

    def timestamp(self, message: str, logger) -> None:
        """Record a timestamp and log the elapsed time.

        Calculates the time elapsed since the last checkpoint, updates the
        checkpoint time, and logs the result with the provided message.

        Args:
            message (str): Descriptive message to prepend to the time value
            logger: Logger object to use for output (must have info method)

        Note:
            The elapsed time is appended to the message automatically.
            The timestamp is recorded in seconds with microsecond precision.
        """
        self.time_elapsed = time.time() - self.time_last_check
        self.time_last_check = time.time()
        logger.info(message + str(self.time_elapsed))
