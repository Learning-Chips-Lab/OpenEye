# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
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

import math
import logging
import test_utils.generic_test_utils as gtu
import test_utils.stream_dicts as strdic

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
        dram_bias (list): Bias values loaded from DRAM for the current layer
        storage (list): Storage structure for organizing partial sum data streams.
                       Format depends on communication mode (serial vs parallel).

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content):
        """Initialize the partial sum stream mapper.

        Sets up storage structures for partial sum data based on the hardware
        communication mode (serial or parallel).

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
            dram_layer_content (list): Bias values from DRAM for this layer

        """
        self.params = params
        self.layer_params = layer_params
        self.layer_repetition = layer_repetition
        self.dram_bias = dram_layer_content

        # Initialize storage structure based on communication mode
        if (self.params.SERIAL):
            # Serial mode: indexed by stream_serial_dict entries
            self.storage = [[] for _ in range(len(strdic.stream_serial_dict))]
        else:
            # Parallel mode: indexed by stream_parallel_dict entries
            self.storage = [[] for _ in range(len(strdic.stream_parallel_dict))]

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
            Parallel mode: List of integers organized per cluster and router, ready
                          for distribution to PE scratchpad memories.

        Note:
            - Serial mode uses 20-bit two's complement for each bias value
            - Values are packed into 40-bit words (two 20-bit values per word)
            - Parallel mode delegates to write_psum_data_glb() for per-cluster generation

        """
        if (self.params.SERIAL) :
            # === SERIAL MODE: LINEAR STREAM GENERATION ===
            # Calculate how many times to replicate each bias value
            block_length = (2 * self.params.Clusters)//self.layer_params.different_kernels_per_calculation
            psum_stream = []

            # Generate bias stream for each output position and filter
            for j in range(self.layer_params.iact_size_y):
                for i in range(self.layer_params.filters):
                    # Convert bias to 20-bit two's complement
                    bias_20bit = gtu.to_twos_complement(self.dram_bias[i], 20)
                    # Pack two 20-bit values into a 40-bit word and replicate
                    packed_value = bias_20bit + (bias_20bit * 2**20)
                    psum_stream.extend([packed_value] * block_length)
        else :
            # === PARALLEL MODE: CLUSTER-BASED STREAM GENERATION ===
            # Create 3D structure: [cluster_x][cluster_y][router]
            psum_stream = [[[[] for c in range(self.params.Psum_Routers)] for b in range(self.params.Clusters_Y)] for a in range(self.params.Clusters_X)]

            # Generate partial sum data for each cluster and router
            for cl_x in range(self.params.Clusters_X):
                for cl_y in range(self.params.Clusters_Y):
                    for router in range(self.params.Psum_Routers):
                        # Delegate to subclass-specific implementation
                        psum_stream[cl_x][cl_y][router] = self.write_psum_data_glb(cl_x, cl_y, router)

            # Flatten the hierarchical structure into a complete stream
            psum_stream = self.create_complete_psum_stream(psum_stream)
        return psum_stream

    def write_psum_data_glb(self, cl_x, cl_y, router):
        """Generate partial sum data for a specific cluster and router.

        Creates the partial sum initialization data for a given cluster's global
        buffer (GLB) router. The number of cycles depends on the layer's refresh
        requirements and communication mode.

        Args:
            cl_x (int): X-coordinate of the target cluster
            cl_y (int): Y-coordinate of the target cluster
            router (int): Router index within the cluster for partial sum distribution

        Returns:
            list: Storage containing partial sum data for each cycle. Each entry
                  corresponds to one refresh cycle and contains layer-specific
                  partial sum values (implementation in subclasses).

        Note:
            - Serial mode: Number of cycles = iact_size_y * 2
            - Parallel mode: Cycles determined by needed_refreshes_mx range
            - Delegates actual data generation to write_psum_storage()

        """
        storage = []
        if (self.params.SERIAL):
            # Serial mode: fixed number of cycles based on input height
            for cycle in range(self.layer_params.iact_size_y * 2):
                storage.append(self.write_psum_storage(cl_x, cl_y, router, cycle))
        else:
            # Parallel mode: cycle range based on refresh scheduling
            # Extract start and end cycle from refresh matrix for this repetition
            start_cycle = math.floor(self.layer_params.needed_refreshes_mx[self.layer_repetition][1]/2)
            end_cycle = math.ceil(self.layer_params.needed_refreshes_mx[self.layer_repetition][2]/2)
            for cycle in range(start_cycle, end_cycle):
                storage.append(self.write_psum_storage(cl_x, cl_y, router, cycle))
        return storage

    def create_complete_psum_stream(self, spad_storage):
        """Flatten scratchpad storage into a complete partial sum stream.

        Converts the hierarchical scratchpad storage structure into a final stream
        format ready for hardware transmission. Handles cycle-by-cycle data and
        performs mode-specific formatting.

        Args:
            spad_storage (list): 3D structure [cluster_x][cluster_y][router] containing
                                partial sum data organized by cycles within each location.

        Returns:
            Serial mode: List of packed 48-bit words combining data from both X-clusters
            Parallel mode: 3D array [cluster_x][cluster_y][router] with flattened cycles

        Note:
            - Serial mode combines data from two X-clusters into single words
            - Each word packs cluster 0 data in lower 24 bits, cluster 1 in upper 24 bits
            - Parallel mode simply flattens the cycle dimension while maintaining structure

        """
        params = self.params
        # Initialize output stream with same hierarchical structure
        stream = [[[[] for c in range(self.params.Psum_Routers)] for b in range(self.params.Clusters_Y)] for a in range(self.params.Clusters_X)]

        # === FLATTEN CYCLE DATA ===
        # Extend each router's stream with data from all cycles
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Psum_Routers):
                    # Get all cycles for this cluster/router combination
                    current_spad = spad_storage[cl_x][cl_y][router]
                    # Concatenate data from all cycles into a single stream
                    for cycle in range(len(current_spad)):
                        stream[cl_x][cl_y][router].extend(spad_storage[cl_x][cl_y][router][cycle])

        # === MODE-SPECIFIC FORMATTING ===
        if(self.params.SERIAL):
            # Serial mode: combine X-clusters into packed words
            temp_stream = stream
            stream = []

            # Iterate through Y-clusters and routers
            for cl_y in range(self.params.Clusters_Y):
                for router in range(self.params.NUM_GLB_PSUM):
                    # Pack data from both X-clusters into single words
                    for word in range(len(temp_stream[0][cl_y][router])):
                        # Lower 24 bits: cluster 0, Upper 24 bits: cluster 1
                        packed_word = temp_stream[0][cl_y][router][word] + (temp_stream[1][cl_y][router][word] * (2**24))
                        stream.append(packed_word)
        else:
            # Parallel mode: return flattened hierarchical structure as-is
            return stream
        return stream

class ConvPsumStreamMapper(PsumStreamMapper):
    """Partial sum stream mapper for standard convolutional layers.

    This class extends PsumStreamMapper to provide Conv2D-specific partial sum
    initialization. For standard convolutions, partial sums are typically initialized
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
            dram_layer_content (list): Bias values from DRAM for this convolution layer

        """
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def write_psum_storage(self, cl_x, cl_y, router, cycle):
        """Generate partial sum storage data for one cycle in a Conv2D layer.

        Creates the partial sum initialization values for a specific cluster, router,
        and computation cycle. For standard convolution, most partial sums are
        initialized to zero.

        Args:
            cl_x (int): X-coordinate of the target cluster
            cl_y (int): Y-coordinate of the target cluster
            router (int): Router index for partial sum distribution
            cycle (int): Current computation cycle index

        Returns:
            list: Storage containing zeros for partial sum initialization. Length
                  equals filters / needed_wght_transmissions, limited by used_psum_per_PE.

        Note:
            - Partial sums initialized to 0 for Conv2D layers
            - Bias values are typically added during accumulation, not initialization
            - Storage size determined by filter count and transmission scheduling

        """
        storage = []
        # Calculate number of partial sum entries for this cycle
        num_entries = int(self.layer_params.filters / self.layer_params.needed_wght_transmissions)

        for part_data_num in range(num_entries):
            # Initialize partial sums to zero up to the PE allocation limit
            if(part_data_num < self.layer_params.used_psum_per_PE):
                storage.append(0)
            else:
                # Unused entries (should not occur in normal operation)
                line = 0
        return storage

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
                - ceil_used_PE_per_clm: Number of PEs used per column
            layer_repetition (int): Current layer repetition index
            dram_layer_content (list): Bias values from DRAM for this depthwise layer

        """
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def write_psum_storage(self, cl_x, cl_y, router, cycle):
        """Generate partial sum storage data for one cycle in a depthwise layer.

        Creates the partial sum initialization values for depthwise convolution.
        Only specific Y-clusters are used based on PE column allocation, resulting
        in sparse partial sum initialization.

        Args:
            cl_x (int): X-coordinate of the target cluster
            cl_y (int): Y-coordinate of the target cluster
            router (int): Router index for partial sum distribution
            cycle (int): Current computation cycle index

        Returns:
            list: Storage containing zeros for partial sum initialization. Empty list
                  for unused clusters, populated list for active clusters at column starts.

        Note:
            - Only clusters at column boundaries (cl_y % ceil_used_PE_per_clm == 0) are used
            - Storage size is ceil(filters/2) to account for dual-channel processing
            - Partial sums initialized to 0 for depthwise layers

        """
        storage = []
        # Calculate number of partial sum entries (filters divided by 2 for dual processing)
        num_entries = math.ceil(self.layer_params.filters / 2)

        for part_data_num in range(num_entries):
            # Only initialize partial sums for clusters at PE column boundaries
            if((cl_y % self.layer_params.ceil_used_PE_per_clm) == 0):
                # Initialize partial sums to zero up to the PE allocation limit
                if(part_data_num < self.layer_params.used_psum_per_PE):
                    storage.append(0)
                else:
                    # Unused entries (should not occur in normal operation)
                    line = 0
        return storage

class DensePsumStreamMapper(PsumStreamMapper):
    """Partial sum stream mapper for fully connected (dense) layers.

    This class extends PsumStreamMapper to provide dense layer-specific partial sum
    initialization. Dense layers have different data organization than convolutional
    layers, with bias values directly loaded into partial sum accumulators before
    the matrix-vector multiplication begins.

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
            dram_layer_content (list): Bias values from DRAM for this dense layer

        """
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    def get_psum_stream(self):
        """Generate the complete partial sum data stream for a dense layer.

        Creates a specialized stream for dense layers that directly loads bias values
        into partial sum accumulators. Unlike convolutional layers, dense layers
        initialize partial sums with actual bias values in two's complement format.

        Returns:
            list: Stream of 20-bit two's complement bias values organized for dual
                  PE processing. Values are interleaved in blocks of 2.

        Note:
            - Block length is fixed at 2 for dual-PE processing
            - Bias values converted to 20-bit two's complement
            - Values organized in pattern: [bias[0], bias[values], bias[1], bias[values+1], ...]
            - Total stream length = ceil(used_psum_per_PE) * 2

        """
        block_length = 2  # Fixed block size for dual PE processing
        psum_stream = []

        # Calculate number of bias values per PE block
        values = math.ceil(self.layer_params.used_psum_per_PE)

        # Interleave bias values for dual PE processing
        for i in range(values):
            for j in range(2):
                # Convert each bias to 20-bit two's complement and add to stream
                bias_index = i + j * values
                bias_value = gtu.to_twos_complement(self.dram_bias[bias_index], 20)
                psum_stream.extend([bias_value])

        return psum_stream

    def write_psum_storage(self, cl_x, cl_y, router, cycle):
        """Generate partial sum storage data for one cycle in a dense layer.

        Creates the partial sum initialization values for a specific cluster, router,
        and computation cycle. For dense layers, partial sums are initialized to zero
        as bias values are loaded separately via get_psum_stream().

        Args:
            cl_x (int): X-coordinate of the target cluster
            cl_y (int): Y-coordinate of the target cluster
            router (int): Router index for partial sum distribution
            cycle (int): Current computation cycle index

        Returns:
            list: Storage containing zeros for partial sum initialization. Length
                  equals ceil(used_psum_per_PE/2) for dual PE processing.

        Note:
            - Storage initialized to 0 for dense layers
            - Size is half of used_psum_per_PE due to dual processing
            - Actual bias values loaded through separate mechanism in get_psum_stream()

        """
        storage = []
        # Calculate number of partial sum entries (half for dual processing)
        num_entries = math.ceil(self.layer_params.used_psum_per_PE / 2)

        for part_data_num in range(num_entries):
            # Check if this entry is within the PE allocation limit
            if(part_data_num < self.layer_params.used_psum_per_PE):
                # Redundant check (could be simplified) - initialize to zero
                if(part_data_num < self.layer_params.used_psum_per_PE):
                    storage.append(0)
                else:
                    # Unused entries (should not occur in normal operation)
                    line = 0
        return storage


    