# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
"""
Dictionaries that name the channels of the assembled layer data stream.

A layer stream is a list of channels; these dictionaries map channel names
to list indices. Full documentation of the stream construction lives in
doc/source/architecture/datastream_construction.md.

stream_serial_dict (DMA / OpenEye_FPGA):
    status       - packed 64-bit configuration words (regmap_pack) followed
                   by the PE-enable bitmap words
    router_iact  - iact router modes, 6 bits per router, 10 per word
    router_wght  - wght router modes, 1 bit per router
    router_psum  - psum router modes, 3 bits per router
    iact_data    - raw pixel words (8 x 8-bit values per 64-bit word); the
                   hardware iact_stream_constructor derives the sparse
                   SPad streams from them
    wght_data    - pre-encoded two-level weight SPad image (addr + data)
    psum_data    - bias / initial partial sums

stream_parallel_dict (direct ports / OpenEye_Parallel):
    status       - list indexed by status_dict, driven onto the
                   configuration ports by rtl_test_utils.send_stream
    iact/wght/psum - nested [cluster_x][cluster_y][router] SPad word lists
    quantize     - (mantissa, exponent) pairs for output requantization
    offset       - per-filter output offsets

status_dict:
    Sub-indices of the parallel status channel. Every entry corresponds to
    one OpenEye_Parallel configuration port. gemm_mode selects the dataflow:
    0 = row-stationary (default), 1 = output-stationary GEMM (PE row j
    reads iact GLB bank j, output tile stationary in the PE psum SPad).
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
  "psum": 3,
  "quantize": 4
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
  "kernel_per_pe_cluster": 16,
  "skipIact": 17,
  "skipWght": 18,
  "skipPsum": 19,
  "usePEs": 20,
  "router_iact": 21,
  "router_wght": 22,
  "router_psum": 23,
  "psum_delay": 24,
  "needed_standing_cycles": 25,
  "gemm_mode": 26
}
