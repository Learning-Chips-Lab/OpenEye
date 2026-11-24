# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Partial sum stream mapper for OpenEye accelerator.

This module provides classes for mapping bias values and partial sums to hardware
data streams for different layer types in the OpenEye neural network accelerator.
It handles the initialization of partial sum accumulators with bias values and
manages their distribution across processing elements and clusters.

Key Features:
- Bias value conversion to two's complement format for hardware compatibility
- Partial sum stream generation for serial and parallel communication modes
- Layer-specific mapping for Conv2D, Depthwise, and Dense layers
- Router-based distribution across PE clusters and scratchpad memories

The partial sum stream mapper coordinates with the hardware's accumulator architecture
to ensure bias values are correctly loaded before computation begins. During convolution
and dense operations, partial sums are initialized with bias values and then accumulated
with multiply-accumulate results.

Typical Usage:
    >>> # For standard convolution
    >>> conv_psum_mapper = ConvPsumStreamMapper(params, layer_params,
    ...                                         layer_repetition, dram_bias)
    >>> psum_stream = conv_psum_mapper.get_psum_stream()
    >>>
    >>> # For depthwise convolution
    >>> dw_psum_mapper = DwPsumStreamMapper(params, layer_params,
    ...                                     layer_repetition, dram_bias)
    >>> psum_stream = dw_psum_mapper.get_psum_stream()
"""

import logging
import math
import numpy as np

logger = logging.getLogger("cocotb")


class PsumStreamMapper(object):
    """Base class for mapping partial sum and bias data to hardware streams.

    This class serves as the foundation for layer-specific partial sum mappers.
    It manages the conversion of bias values from DRAM into hardware-compatible
    data streams that initialize partial sum accumulators in the PE clusters.

    Partial sums represent intermediate accumulation results during neural network
    computation. They are initialized with bias values before MAC (multiply-accumulate)
    operations begin. The mapper ensures proper formatting and routing of these
    initial values across the hardware's memory hierarchy.

    Attributes:
        params: OpenEye hardware parameters (cluster configuration, communication mode)
        layer_params: Layer-specific parameters (filters, dimensions, PE allocation)
        layer_repetition (int): Current repetition index for this layer execution
        dram_bias: Bias values loaded from DRAM for the current layer

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        """Initialize the partial sum stream mapper.

        Args:
            params: OpenEye hardware parameters including:
- SERIAL: Communication mode flag (True for serial, False for parallel)
- Clusters_X/Y: Number of clusters in each dimension
- Psum_Routers: Number of partial sum routers per cluster
            layer_params: Layer-specific parameters including:
- filters: Number of output filters
- iact_size_y: Input activation height
- used_psum_per_PE: Number of partial sums per PE
            layer_repetition (int): Repetition index for layer execution scheduling
            dram_layer_content: Bias values from DRAM for this layer

        """
        self.params = params
        self.layer_params = layer_params
        self.layer_repetition = layer_repetition
        self.dram_bias = dram_layer_content

    def get_psum_stream(self):
        """Generate the complete partial sum data stream for the layer.

        Creates the data stream that initializes partial sum accumulators with bias
        values. The format differs based on the communication mode:
- Serial mode: Single linear stream with duplicated bias values
- Parallel mode: 3D structure organized by cluster and router

        Returns:
            Serial mode: List of packed integers containing bias values in two's
                        complement format. Each bias is duplicated to fill block_length
                        positions for different output spatial locations.
            Parallel mode: 3D list [cluster_x][cluster_y][router] of integers ready
                          for distribution to PE scratchpad memories.

        Note:
- Serial mode converts each filter bias once and replicates at C-speed
- Parallel mode delegates to write_psum_data_glb() for per-cluster generation

        """
        if self.params.SERIAL:
            # === SERIAL MODE: LINEAR STREAM GENERATION ===
            bitwidth = int(self.params.DATA_PSUM_BITWIDTH)
            dma_bw = int(self.params.DMA_BITWIDTH)
            words_per_transmission = min(2, dma_bw // bitwidth)
            block_length = (
                (int(self.params.NUM_GLB_PSUM) // words_per_transmission) * int(self.params.Clusters)
            ) // int(self.layer_params.different_kernels_per_calculation)
            total_repetitions = int(self.layer_params.iact_x_line_repetitions) * int(self.layer_params.iact_size_y)
            filters = int(self.layer_params.filters)

            if filters <= 0 or block_length <= 0 or total_repetitions <= 0:
                return []

            raw_biases = np.asarray(self.dram_bias, dtype=np.int64).ravel()[:filters]
            total_bits = bitwidth * words_per_transmission

            if total_bits <= 64:
                mask = np.uint64((1 << bitwidth) - 1)
                bias_bits = raw_biases.astype(np.uint64) & mask
                packed = np.zeros_like(bias_bits)
                for _ in range(words_per_transmission):
                    packed = (packed << np.uint64(bitwidth)) | bias_bits
                packed_biases = np.repeat(packed, block_length).tolist()
            else:
                mask = (1 << bitwidth) - 1
                packed_biases = []
                for b in raw_biases.tolist():
                    bias_bit = int(b) & mask
                    packed_word = 0
                    for _ in range(words_per_transmission):
                        packed_word = (packed_word << bitwidth) | bias_bit
                    packed_biases.extend([packed_word] * block_length)

            return packed_biases * total_repetitions

        # === PARALLEL MODE: CLUSTER-BASED STREAM GENERATION ===
        # write_psum_data_glb already returns flat per-router streams across all cycles
        return [
            [
                [self.write_psum_data_glb(cl_x, cl_y, router) for router in range(self.params.Psum_Routers)]
                for cl_y in range(self.params.Clusters_Y)
            ]
            for cl_x in range(self.params.Clusters_X)
        ]

    def write_psum_data_glb(self, cl_x, cl_y, router):
        """Generate partial sum data for a specific cluster and router.

        Creates the partial sum initialization data for a given cluster's global
        buffer (GLB) router across all required refresh cycles.

        Args:
            cl_x (int): X-coordinate of the target cluster
            cl_y (int): Y-coordinate of the target cluster
            router (int): Router index within the cluster for partial sum distribution

        Returns:
            list: Flat list containing partial sum initialization data across all cycles.

        Note:
- Serial mode: Number of cycles = iact_size_y * 2
- Parallel mode: Cycle count determined by needed_refreshes_mx range
- Multiplies single-cycle storage directly since initialization is cycle-invariant

        """
        if self.params.SERIAL:
            num_cycles = int(self.layer_params.iact_size_y) * 2
        else:
            refresh_bounds = self.layer_params.needed_refreshes_mx[self.layer_repetition]
            num_cycles = math.ceil(refresh_bounds[2] / 2) - math.floor(refresh_bounds[1] / 2)

        if num_cycles <= 0:
            return []

        return self.write_psum_storage(cl_x, cl_y, router, 0) * int(num_cycles)

    def create_complete_psum_stream(self, spad_storage):
        """Format flattened scratchpad storage into a complete partial sum stream.

        Args:
            spad_storage (list): 3D structure [cluster_x][cluster_y][router] containing
                                flattened partial sum data per router.

        Returns:
            Serial mode: List of packed 48-bit words combining data from both X-clusters
            Parallel mode: 3D array [cluster_x][cluster_y][router] as-is

        Note:
- Serial mode packs cluster 0 data in lower 24 bits, cluster 1 in upper 24 bits

        """
        if not self.params.SERIAL:
            return spad_storage

        stream = []
        shift = np.uint64(24)
        for cl_y in range(self.params.Clusters_Y):
            for router in range(self.params.NUM_GLB_PSUM):
                s0 = spad_storage[0][cl_y][router]
                s1 = spad_storage[1][cl_y][router]
                n = min(len(s0), len(s1))
                if n > 0:
                    arr0 = np.asarray(s0[:n], dtype=np.uint64)
                    arr1 = np.asarray(s1[:n], dtype=np.uint64)
                    stream.extend((arr0 | (arr1 << shift)).tolist())
        return stream


class ConvPsumStreamMapper(PsumStreamMapper):
    """Partial sum stream mapper for standard convolutional layers.

    This class extends PsumStreamMapper to provide Conv2D-specific partial sum
    initialization. For standard convolutions, partial sums are initialized
    to zero as bias values are added during the first accumulation cycle.

    Attributes:
        Inherited from PsumStreamMapper (params, layer_params, dram_bias, etc.)

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        """Initialize the Conv2D partial sum stream mapper.

        Args:
            params: OpenEye hardware parameters
            layer_params: Conv2D layer parameters including:
- filters: Number of output filters/channels
- needed_wght_transmissions: Number of weight transmission cycles
- used_psum_per_PE: Partial sums allocated per PE
            layer_repetition (int): Current layer repetition index
            dram_layer_content: Bias values from DRAM for this convolution layer

        """
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def write_psum_storage(self, cl_x, cl_y, router, cycle):
        """Generate partial sum storage data for one cycle in a Conv2D layer.

        Args:
            cl_x (int): X-coordinate of the target cluster
            cl_y (int): Y-coordinate of the target cluster
            router (int): Router index for partial sum distribution
            cycle (int): Current computation cycle index

        Returns:
            list: Zero-initialized partial sum entries bounded by used_psum_per_PE.

        """
        num_entries = int(self.layer_params.filters) // int(self.layer_params.needed_wght_transmissions)
        count = min(num_entries, int(self.layer_params.used_psum_per_PE))
        return [0] * max(0, count)


class DwPsumStreamMapper(PsumStreamMapper):
    """Partial sum stream mapper for depthwise convolutional layers.

    This class extends PsumStreamMapper to provide depthwise convolution-specific
    partial sum initialization. In depthwise convolution, each input channel is
    convolved independently, requiring different PE allocation patterns than
    standard convolution.

    Attributes:
        Inherited from PsumStreamMapper (params, layer_params, dram_bias, etc.)

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        """Initialize the depthwise convolution partial sum stream mapper.

        Args:
            params: OpenEye hardware parameters
            layer_params: Depthwise layer parameters including:
- filters: Number of output channels (matches input channels)
- used_psum_per_PE: Partial sums allocated per PE
- used_Y_cluster: Number of PEs used per column
            layer_repetition (int): Current layer repetition index
            dram_layer_content: Bias values from DRAM for this depthwise layer

        """
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def write_psum_storage(self, cl_x, cl_y, router, cycle):
        """Generate partial sum storage data for one cycle in a depthwise layer.

        Only clusters at PE column boundaries (cl_y % used_Y_cluster == 0) are used,
        resulting in sparse partial sum initialization.

        Args:
            cl_x (int): X-coordinate of the target cluster
            cl_y (int): Y-coordinate of the target cluster
            router (int): Router index for partial sum distribution
            cycle (int): Current computation cycle index

        Returns:
            list: Zero-initialized partial sum entries for active column boundary clusters,
                  or an empty list for unused clusters.

        """
        if (cl_y % int(self.layer_params.used_Y_cluster)) != 0:
            return []

        num_entries = math.ceil(self.layer_params.filters / 2)
        count = min(num_entries, int(self.layer_params.used_psum_per_PE))
        return [0] * max(0, count)


class DensePsumStreamMapper(PsumStreamMapper):
    """Partial sum stream mapper for fully connected (dense) layers.

    This class extends PsumStreamMapper to provide dense layer-specific partial sum
    initialization. Dense layers load bias values directly into partial sum accumulators
    before matrix-vector multiplication begins.

    Attributes:
        Inherited from PsumStreamMapper (params, layer_params, dram_bias, etc.)

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        """Initialize the dense layer partial sum stream mapper.

        Args:
            params: OpenEye hardware parameters
            layer_params: Dense layer parameters including:
- output_shape: Output dimensions determining number of bias values
- used_psum_per_PE: Partial sums allocated per PE
            layer_repetition (int): Current layer repetition index
            dram_layer_content: Bias values from DRAM for this dense layer

        """
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def get_psum_stream(self):
        """Generate the complete partial sum data stream for a dense layer.

        Creates a specialized stream for dense layers that directly loads bias values
        into partial sum accumulators in two's complement format, interleaved across
        cluster columns.

        Returns:
            list: Stream of two's complement bias values interleaved across cluster columns.

        """
        bitwidth = int(self.params.DATA_PSUM_BITWIDTH)
        clusters_x = int(self.params.Clusters_X)
        used_psum = int(self.layer_params.used_psum_per_PE)
        values = math.ceil(self.layer_params.used_psum_per_PE)
        repetitions = int(self.layer_params.needed_psum_transmissions)

        if repetitions <= 0 or values <= 0 or clusters_x <= 0:
            return []

        raw_biases = np.asarray(self.dram_bias, dtype=np.int64).ravel()
        num_biases = raw_biases.size

        if bitwidth <= 64:
            mask = np.uint64((1 << bitwidth) - 1)
            encoded_biases = raw_biases.astype(np.uint64) & mask
        else:
            mask = (1 << bitwidth) - 1
            encoded_biases = np.array([int(b) & mask for b in raw_biases.tolist()], dtype=object)

        # One word per filter index per cluster column, columns interleaved
        # (f0c0, f0c1, f1c0, ...): psum_pipeline's FC GET_BIAS branch writes
        # word k into cluster column k % CLUSTER_COLUMNS and advances the
        # buffer address after every CLUSTER_COLUMNS words. The filters are
        # split across the columns, so column c holds filters
        # c*values .. c*values+values-1.
        rep_offsets = np.arange(repetitions, dtype=np.int64)[:, None, None] * (clusters_x * used_psum)
        val_offsets = np.arange(values, dtype=np.int64)[None, :, None]
        col_offsets = np.arange(clusters_x, dtype=np.int64)[None, None, :] * used_psum
        indices = (rep_offsets + val_offsets + col_offsets).ravel()

        out = np.zeros(indices.size, dtype=encoded_biases.dtype)
        valid = (indices >= 0) & (indices < num_biases)
        out[valid] = encoded_biases[indices[valid]]
        return out.tolist()

    def write_psum_storage(self, cl_x, cl_y, router, cycle):
        """Generate partial sum storage data for one cycle in a dense layer.

        Args:
            cl_x (int): X-coordinate of the target cluster
            cl_y (int): Y-coordinate of the target cluster
            router (int): Router index for partial sum distribution
            cycle (int): Current computation cycle index

        Returns:
            list: Zero-initialized storage of length ceil(used_psum_per_PE / 2)
                  for dual-PE processing.

        """
        num_entries = math.ceil(self.layer_params.used_psum_per_PE / 2)
        return [0] * max(0, num_entries)