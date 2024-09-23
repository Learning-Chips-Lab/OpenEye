# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
"""
Dictionaries that are needed for layer classes to create the data stream
"""

stream_serial_dict = {
  "status": 0,
  "router_iact": 1,
  "router_wght": 2,
  "router_psum": 3,
  "iact_data": 4,
  "wght_data": 5,
  "psum_data": 6
}

# dictionary that specifies the position of the data in the list that represents the stream
stream_parallel_dict = {
  "status": 0,
  "iact": 1,
  "wght": 2,
  "psum": 3
}

# further refinement of the status data (index in sub-list)
status_dict = {
  "data_mode": 0,
  "realfactor": 1,
  "autofunction": 2,
  "poolingmode": 3,
  "needed_refreshes": 4,
  "used_X_cluster": 5,
  "used_Y_cluster": 6,
  "needed_Iact_writes": 7,
  "used_psum_per_PE": 8,
  "used_iact_addr_per_PE": 9,
  "used_wght_addr_per_PE": 10,
  "used_iact_per_PE": 11,
  "iact_addr_len": 12,
  "iact_data_len": 13,
  "strideX": 14,
  "strideY": 15,
  "skipIact": 16,
  "skipWght": 17,
  "skipPsum": 18,
  "usePEs": 19,
  "router_iact": 20,
  "router_wght": 21,
  "router_psum": 22,
  "psum_delay": 23
}
