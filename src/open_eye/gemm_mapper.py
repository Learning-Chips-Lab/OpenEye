# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""GEMM layer mapper for the output-stationary dataflow of the OpenEye accelerator.

This module provides the GemmMapper class, which maps general matrix
multiplications C = A x B (+ bias) onto the accelerator using the
output-stationary (OS) dataflow. It complements the row-stationary (RS)
mappers (ConvMapper, DWMapper) and generalizes the DenseMapper.

Dataflow background
-------------------
In the row-stationary dataflow each PE holds one filter row and the matching
iact row; psums travel vertically through the PE array. Which iact GLB bank a
PE listens to is decided per PE by the iact_choose pattern (diagonal mapping).

In the output-stationary dataflow the assignment is fixed in hardware: with
gemm_mode = 1 every PE cluster binds iact GLB bank j to PE row j
(see PE_cluster.gemm_mode_i). Each PE keeps a tile of the output matrix C
stationary in its local psum SPad while the inner dimension K streams through:

    PE(row j, column i) accumulates   C_tile += A[:, K_j] x B[K_j, N_i]

where K_j is the K-slice held in iact GLB bank j and N_i the output-column
tile assigned to PE column i. The vertical psum drain afterwards reduces the
partial K-slices across the PE rows, so the full inner product is formed
without any inter-PE iact exchange.

Stream construction
-------------------
The OS GEMM layout reuses the dense stream creators, which already split the
inner dimension across banks (DenseIactStreamMapper places
``router * used_iact_per_PE`` slices in bank ``router``):

- iact stream:  A is flattened; bank j of each cluster receives K-slice j.
- wght stream:  B is tiled per PE row (K-slice) and PE column (N-tile).
- psum stream:  bias values initialize the stationary output tiles.
- status:       identical to the dense configuration, plus gemm_mode = 1
                (DMA register ``gemm_mode`` in serial mode, status_dict entry
                ``gemm_mode`` in parallel mode).

Typical usage:
    >>> mapper = GemmMapper(params, layer_params, repetition, dram_content,
    ...                     sparse_iacts, sparse_wghts)
    >>> mapper.make_stream()
    >>> stream = mapper.get_stream()

The mapper is selected automatically by test_utils_main.write_stream_layer_mp
for layers whose name contains "Gemm", or for dense layers when
params.DATAFLOW == "output_stationary".
"""

import logging
from open_eye.dense_mapper import DenseMapper

logger = logging.getLogger("cocotb")


class GemmMapper(DenseMapper):
    """Mapper for GEMM layers using the output-stationary dataflow.

    The heavy lifting (stream layout, register packing, router programming)
    is shared with DenseMapper; the decisive difference is that gemm_mode is
    forced to 1, which switches the PE clusters into output-stationary
    routing (iact GLB bank j -> PE row j) instead of the iact_choose-driven
    row-stationary routing.

    Args:
        params: Hardware configuration parameters
        layer_params: Layer parameters describing the GEMM (K = input size,
            N = output size; M > 1 is handled through layer repetitions)
        layer_repetition: Current repetition index
        dram_layer_content: (input matrix A, weight matrix B, bias) from DRAM
        sparse_iacts: Sparsity encoding flag for A
        sparse_wghts: Sparsity encoding flag for B
    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts):
        # Force the output-stationary hardware mode for this layer. The flag
        # is picked up by write_working_parameters (serial: gemm_mode DMA
        # register, parallel: status_dict["gemm_mode"]).
        layer_params.gemm_mode = 1
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts)
