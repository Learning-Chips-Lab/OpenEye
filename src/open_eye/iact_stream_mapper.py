# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Input activation stream mapper module for OpenEye accelerator.

This module provides classes for mapping input activations (feature maps) from DRAM
to hardware data streams for the OpenEye neural network accelerator. It handles the
spatial and temporal distribution of activations across processing elements (PEs),
clusters, and Global Buffers (GLBs) for different layer types.

Key Features:
    - Mapping of input activations to PE scratchpad (SPAD) storage
    - Support for sparse activation patterns with overhead encoding
    - Handling of zero-padding for convolution operations
    - Layer-specific mapping strategies (Conv2D, Depthwise, Dense)
    - Support for both serial (DMA) and parallel communication modes
    - Address and data stream generation for PE memory writes

Classes:
    IactStreamMapper: Base class for input activation stream mapping
    ConvIactStreamMapper: Specialized mapper for standard Conv2D layers
    DenseIactStreamMapper: Specialized mapper for fully connected layers
    DwIactStreamMapper: Specialized mapper for depthwise convolution layers

Typical Usage:
    >>> iact_mapper = ConvIactStreamMapper(params, layer_params, layer_repetition,
    ...                                     dram_fmap, sparse_data=False)
    >>> iact_stream = iact_mapper.get_iact_stream()
"""

import numpy as np
import math
import logging
import open_eye.generic_test_utils as gtu
import open_eye.stream_dicts as strdic

logger = logging.getLogger("cocotb")

class IactStreamMapper(object):
    """Base class for mapping input activations to hardware data streams.

    This class provides the core functionality for distributing input activations
    (feature maps) from DRAM to the PE array. It handles the spatial mapping across
    clusters, temporal scheduling, and data stream formatting for both sparse and
    dense activation patterns.

    The mapping process involves:
    1. Determining which activations each PE needs based on output location
    2. Computing DRAM addresses considering stride, padding, and kernel positions
    3. Organizing data into PE scratchpad storage format
    4. Generating bit-packed streams for transmission to hardware

    Attributes:
        params: OpenEye hardware parameters (cluster config, PE counts, bitwidths, etc.)
        layer_params: Layer-specific parameters (kernel size, stride, dimensions, etc.)
        layer_repetition (int): Current repetition index for tiled/repeated layer execution
        dram_fmap: Input feature map data from DRAM (3D array [channels][height][width])
        sparse_data (int): Flag indicating whether to use sparse encoding (1) or dense (0)
        storage (list): Internal storage for organizing streams by communication channel

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_data):
        """Initialize the input activation stream mapper.

        Args:
            params: OpenEye hardware parameters
            layer_params: Layer-specific parameters for the current layer
            layer_repetition (int): Current repetition index
            dram_layer_content: Input feature map data from DRAM
            sparse_data (int): 1 for sparse encoding, 0 for dense

        """
        self.params = params
        self.layer_params = layer_params
        self.layer_repetition = layer_repetition
        self.dram_fmap = dram_layer_content
        self.sparse_data = sparse_data

        # Initialize storage structure based on communication mode
        if (params.SERIAL):
            # Serial mode: organize by DMA streams
            self.storage = [[] for _ in range(len(strdic.stream_serial_dict))]
        else:
            # Parallel mode: organize by parallel channels
            self.storage = [[] for _ in range(len(strdic.stream_parallel_dict))]

    def get_iact_stream(self):
        """Generate the complete input activation data stream for the layer.

        This method orchestrates the creation of activation streams for all clusters
        and Global Buffers. In parallel mode, it creates separate streams for each
        cluster and router. In serial mode, it creates a single DMA-compatible stream.

        Returns:
            Activation stream in one of two formats:
            - Parallel mode: 3D list [cluster_x][cluster_y][router] of bit-packed words
            - Serial mode: 1D list of DMA words, or empty list if skipIact is set

        Note:
            Serial mode uses channel-first ordering with used_channels grouping,
            then packs multiple activation values per DMA word based on bitwidth.

        """
        if (not self.params.SERIAL):
            # === PARALLEL MODE: Create per-cluster streams ===
            # Initialize 3D structure: [cluster_x][cluster_y][router]
            iact_stream = [[[[] for c in range(self.params.NUM_GLB_IACT)] for b in range(self.params.Clusters_Y)] for a in range(self.params.Clusters_X)]

            # Generate SPAD data for each cluster and Global Buffer router
            for cl_x in range(self.params.Clusters_X):
                for cl_y in range(self.params.Clusters_Y):
                    for router in range(self.params.NUM_GLB_IACT):
                        iact_stream[cl_x][cl_y][router] = self.write_iact_data_glb(cl_x, cl_y, router)

            # Convert SPAD storage to final bit-packed stream format
            iact_stream = self.create_complete_iact_stream(iact_stream)
        else:
            # === SERIAL MODE: Create DMA stream ===
            iact_stream = []

            # Skip activation loading if layer parameters indicate it's not needed
            if (self.layer_params.skipIact == 0):
                bitwidth = self.params.IACT_Bitwidth
                dma_bitwidth = self.params.DMA_Bit_AXI
                values_per_word = dma_bitwidth // bitwidth

                # Transpose from [C][H][W] to [H][W][C] for row-major ordering
                values = np.transpose(np.array(self.dram_fmap), axes=[2, 1, 0])
                iact_size_y, iact_size_x, channels = values.shape

                used_channels = self.layer_params.used_channels
                flat_values = []

                # Flatten values with channel-first grouping for used_channels
                # Ordering: iterate over channel groups, then spatial positions, then channels in group
                for c_base in range(0, channels, used_channels):
                    for y in range(iact_size_y):
                        for x in range(iact_size_x):
                            for c_offset in range(used_channels):
                                c = c_base + c_offset
                                if c < channels:
                                    flat_values.append(int(values[y, x, c]))

                # Pack multiple activation values into DMA words
                for i in range(0, len(flat_values), values_per_word):
                    word = 0
                    for j in range(values_per_word):
                        if i + j < len(flat_values):
                            # Convert to two's complement and pack into word
                            val_twos = gtu.to_twos_complement(flat_values[i + j], bitwidth)
                            word |= val_twos << (j * bitwidth)
                    iact_stream.append(word)

        return iact_stream


    def write_iact_data_glb(self, cl_x, cl_y, router):
        """Generate activation data for a specific Global Buffer (GLB) router.

        This method creates the sequence of SPAD writes for a particular GLB router,
        iterating over refresh cycles and activation write cycles. It determines when
        each cluster needs data based on Y-cluster scheduling.

        Args:
            cl_x (int): X-coordinate of the cluster
            cl_y (int): Y-coordinate of the cluster
            router (int): GLB router index within the cluster

        Returns:
            list: Sequence of SPAD data, where each element is either:
                - Dense format: [[value, overhead], ...] for each PE position
                - Sparse format: [[filtered_addrs], [[value, overhead], ...]] after filtering

        Note:
            The modulo check (cl_y - cycle) % used_Y_cluster determines when this
            cluster participates in computation for systolic data flow.

        """
        storage = []

        # Iterate over the refresh cycles assigned to this layer repetition
        for cycle in range(self.layer_params.needed_refreshes_mx[self.layer_repetition][1],
                           self.layer_params.needed_refreshes_mx[self.layer_repetition][2]):
            # Multiple activation writes may be needed per refresh cycle
            for iact_cycle in range(self.layer_params.needed_Iact_writes):
                # Check if this cluster participates in this cycle (systolic scheduling)
                if ((cl_y - cycle) % self.layer_params.used_Y_cluster == 0):
                    # Generate PE SPAD data for this cycle
                    spad = self.write_iact_pe(cl_x, cl_y, router, cycle, iact_cycle)

                    # Apply sparse encoding if enabled
                    if (self.sparse_data == 1):
                        storage.append(self.set_sparse_stream(spad))
                    else:
                        storage.append(spad)
        return storage

    def set_sparse_stream(self, spad):
        """Convert dense SPAD data to sparse format by filtering zeros and compacting addresses.

        This method implements sparse encoding for activations by:
        1. Filtering out zero activation values
        2. Compacting address pointers to reference only non-zero data
        3. Creating a mapping from addresses to filtered data positions

        Args:
            spad (list): Dense SPAD data in format [[addresses], [[value, overhead], ...]]

        Returns:
            list: Sparse SPAD data in format [[filtered_addresses], [[non_zero_value, overhead], ...]]

        Note:
            The sparse encoding reduces memory bandwidth by transmitting only non-zero
            activations along with compact address information. This is particularly
            beneficial for layers with high activation sparsity (e.g., after ReLU).

        """
        layer_params = self.layer_params
        params = self.params

        # Convert address list to numpy array for efficient filtering
        array = np.array(spad[0])

        # Truncate addresses: keep entries before the last used address OR non-zero addresses
        # This removes unused trailing address slots while preserving the address structure
        trunc_addr = array[(np.arange(len(array)) < (math.ceil(layer_params.used_iact_addr_per_PE)-1)) |
            (array != 0)].tolist()

        # Filter data to keep only non-zero activation values
        # Each sublist is [value, overhead], we filter where value != 0
        filtered_data = [sublist for i, sublist in enumerate(spad[1]) if not (sublist[0] == 0)]

        # Build filtered address list that maps to the compacted data
        addr_counter, addr_index = 0, 0
        filtered_addr = []

        # For each non-zero data element, determine which address it belongs to
        for count, data in enumerate(filtered_data):
            # If data's overhead counter is less than current address boundary
            if (data[1] < trunc_addr[addr_index]):
                addr_counter = addr_counter + 1
            else:
                # Reached next address boundary
                filtered_addr.append(addr_counter)
                addr_index = addr_index + 1
                # Check if this data is still before the next address boundary
                if (data[1] < trunc_addr[addr_index]):
                    addr_counter = addr_counter + 1

        # Append final address counter
        filtered_addr.append(addr_counter)

        # Return sparse format: [filtered addresses, filtered data]
        spad = [filtered_addr, filtered_data]
        return spad

    def write_iact_pe(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate complete SPAD data for a PE (Processing Element).

        This is a wrapper method that combines data and address SPAD generation.
        Currently only generates data SPAD; address SPAD generation is disabled.

        Args:
            cl_x (int): Cluster X-coordinate
            cl_y (int): Cluster Y-coordinate
            router (int): GLB router index
            cycle (int): Current refresh cycle
            iact_cycle (int): Current activation write cycle

        Returns:
            SPAD data structure (format depends on subclass implementation)

        """
        # Generate data SPAD for this PE
        data_spad = self.write_iact_data_storage(cl_x, cl_y, router, cycle, iact_cycle)

        # Address SPAD generation currently disabled (would contain address pointers)
        #addr_spad = self.write_iact_addr_storage(cl_x, cl_y, router, cycle, iact_cycle)

        return data_spad

    def write_iact_data_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate activation data SPAD storage for a specific PE location and cycle.

        This is the core method that computes which activation values from DRAM should
        be loaded into a PE's scratchpad for a given computation cycle. It calculates
        the input feature map positions needed based on:
        - Output position being computed (determined by cluster, PE, cycle)
        - Convolution kernel position (word_in_storage)
        - Stride and zero-padding parameters
        - Channel assignment based on layer repetition

        The method uses match/case statements to handle different computation patterns
        (single cluster vs. multi-cluster).

        Args:
            cl_x (int): Cluster X-coordinate
            cl_y (int): Cluster Y-coordinate
            router (int): GLB router index
            cycle (int): Current refresh/computation cycle
            iact_cycle (int): Current activation write cycle

        Returns:
            list: SPAD storage as [[value, overhead_counter], ...] for each PE position.
                  Values are fetched from DRAM or set to 1 for zero-padding regions.

        Note:
            The complex position calculations account for:
            - Tiling across multiple clusters (cl_x, cl_y)
            - Systolic dataflow scheduling (cycle)
            - Kernel window positioning (words_in_storage % kernel_size)
            - Stride for downsampling (strideX, strideY)
            - Zero-padding at feature map boundaries

        """
        layer_params = self.layer_params
        params = self.params
        layer_repetition = self.layer_repetition
        dram_fmap = self.dram_fmap

        overhead_counter = 0
        # Initialize SPAD storage: each entry is [value, overhead_index]
        spad_storage = [[0 for _ in range(2)] for _ in range(params.Iacts_per_PE)]

        # Iterate over each position in the PE's SPAD
        for words_in_storage in range(math.ceil(params.Iacts_per_PE)):
            # Only fill up to the number of activations actually used
            if(words_in_storage < layer_params.used_iact_per_PE):
                # Match on computation pattern: determines how work is distributed across clusters
                match layer_params.single_cluster_computation:
                    case 1:
                        # === CASE 1: Single cluster computation (all work in one cluster) ===

                        # Calculate X position in input feature map
                        # 1. Compute base output X position from cycle count and PEs
                        # 2. Apply stride to map output position to input space
                        # 3. Add router offset for kernel column position
                        # 4. Subtract padding offset to handle zero-padding
                        iact_temp_pos_x = \
                        int((math.floor((cycle * params.PEs_X * math.floor(params.Clusters/layer_params.used_Y_cluster)) * layer_params.strideX) % \
                        ((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)) + \
                        router + iact_cycle * math.floor(params.NUM_GLB_IACT/layer_params.kernel_per_pe_cluster)) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))  # Zero padding offset

                        # Calculate Y position in input feature map
                        # 1. Start with kernel row offset (words_in_storage % kernel_height)
                        # 2. Add output row position (cycle-based) scaled by stride
                        # 3. Subtract padding offset
                        iact_temp_pos_y = \
                        int((words_in_storage % layer_params.kernel_size[1]) + \
                        (layer_params.strideY * \
                        math.floor(((cycle * params.PEs_X) * layer_params.strideX)/((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)))) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))  # Zero padding offset

                        # Calculate input channel index
                        # Channels are distributed across layer repetitions for input channel tiling
                        channel = math.floor(words_in_storage/layer_params.kernel_size[0]) + \
                        ((layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))
                    case 2:
                        # === CASE 2: Multi-cluster computation in X direction ===
                        # Work is distributed across clusters in the X dimension

                        # Calculate X position including cluster X offset
                        iact_temp_pos_x = \
                        int((math.floor(((cl_x * params.PEs_X) + \
                        cycle * params.PEs_X * math.floor(params.Clusters/layer_params.used_Y_cluster)) * layer_params.strideX) % \
                        ((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)) + \
                        router + iact_cycle * math.floor(params.NUM_GLB_IACT/layer_params.kernel_per_pe_cluster)) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))  # Zero padding offset

                        # Calculate Y position (same as case 1, no Y-cluster distribution)
                        iact_temp_pos_y = \
                        int((words_in_storage % layer_params.kernel_size[1]) + \
                        (layer_params.strideY * \
                        math.floor(((cycle * params.PEs_X) * layer_params.strideX)/((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)))) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))  # Zero padding offset

                        # Calculate channel (same as case 1)
                        channel = math.floor(words_in_storage/layer_params.kernel_size[0]) + \
                        ((layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))
                    case _:
                        # === DEFAULT CASE: Full 2D cluster distribution ===
                        # Work is distributed across both X and Y cluster dimensions

                        # Calculate X position including both cl_x and cl_y contributions
                        # cl_y contribution accounts for row groups in 2D cluster layout
                        iact_temp_pos_x = \
                        int((math.floor(((cl_x * params.PEs_X) + \
                        (math.floor(cl_y / layer_params.used_Y_cluster) * params.PEs_X * params.Clusters_X) + \
                        cycle * params.PEs_X * math.floor(params.Clusters/layer_params.used_Y_cluster)) * layer_params.strideX) % \
                        ((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)) + \
                        router + iact_cycle * math.floor(params.NUM_GLB_IACT/layer_params.kernel_per_pe_cluster)) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))  # Zero padding offset

                        # Calculate Y position including cl_y contribution
                        iact_temp_pos_y = \
                        int((words_in_storage % layer_params.kernel_size[1]) + \
                        (layer_params.strideY * \
                        math.floor((((math.floor(cl_y / layer_params.used_Y_cluster) * params.PEs_X * params.Clusters_X) + \
                        cycle * params.PEs_X * math.floor(params.Clusters/layer_params.used_Y_cluster)) * layer_params.strideX)/((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)))) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))  # Zero padding offset

                        # Calculate channel (same as other cases)
                        channel = math.floor(words_in_storage/layer_params.kernel_size[0]) + \
                        ((layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))

                # === BOUNDARY CHECKING AND DATA FETCHING ===

                # Check if this PE is within the active computation region
                # Compute the flat PE index and verify it's within output dimensions
                if(((math.floor(cycle/ self.layer_params.used_Y_cluster) * params.Clusters_Y *  params.Clusters_X * params.PEs_X) + \
                    ((math.floor(cl_y/ self.layer_params.used_Y_cluster) *  params.Clusters_X * params.PEs_X)) + (cl_x * params.PEs_X)) < \
                   (layer_params.output_shape[1] * (layer_params.output_shape[2]+layer_params.add_up))):

                    # Check if calculated position is within valid input feature map bounds
                    if(((iact_temp_pos_x >= 0) & (iact_temp_pos_x < (layer_params.output_shape[1] * layer_params.strideX))) & \
                    ((iact_temp_pos_y) >= 0) & (iact_temp_pos_y < (layer_params.output_shape[2] * layer_params.strideY))):
                        # Valid position: fetch actual data from DRAM
                        spad_storage[words_in_storage][0] = dram_fmap[channel][iact_temp_pos_x][iact_temp_pos_y]
                    else:
                        # Outside bounds: use padding value (1 in this implementation)
                        spad_storage[words_in_storage][0] = 1

                    # Store overhead counter (position index for sparse encoding)
                    spad_storage[words_in_storage][1] = overhead_counter
                    overhead_counter = overhead_counter + 1

        return spad_storage
        
    def write_iact_addr_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate activation address SPAD storage for indirect addressing mode.

        Creates address pointers that divide the activation SPAD into groups based
        on kernel size. Each address pointer marks the boundary of an activation group.

        Args:
            cl_x (int): Cluster X-coordinate
            cl_y (int): Cluster Y-coordinate
            router (int): GLB router index
            cycle (int): Current refresh cycle
            iact_cycle (int): Current activation write cycle

        Returns:
            list: Address pointers, where each entry marks the end position of a
                  kernel-sized group of activations.

        Note:
            Currently disabled in main flow (see write_iact_pe). Address storage is
            used in sparse encoding modes where addresses point to data positions.

        """
        layer_params = self.layer_params
        params = self.params
        line_counter = 0
        spad_storage = [0 for _ in range(self.params.Iacts_Addr_per_PE)]

        # Create address boundaries: each address points to end of a kernel-sized group
        for words_in_storage in range(params.Iacts_Addr_per_PE):
            if(words_in_storage < (math.ceil(layer_params.used_iact_per_PE/layer_params.kernel_size[0]))):
                # Address points to cumulative count of kernel_size[0] elements
                spad_storage[words_in_storage] = (layer_params.kernel_size[0] * (words_in_storage + 1))
            line_counter = line_counter + 1
        return spad_storage
        
    def create_complete_iact_stream(self, spad_storage):
        """Convert SPAD storage to final bit-packed activation streams.

        This method takes the intermediate SPAD representation and converts it to
        the final hardware stream format. It packs multiple activation values and
        overhead bits into transmission words according to hardware bitwidth constraints.

        Args:
            spad_storage (list): 3D structure [cl_x][cl_y][router] containing SPAD data
                                for each cluster and router across all cycles.

        Returns:
            Final stream format:
            - Parallel mode: 3D list [cl_x][cl_y][router] of bit-packed words
            - Serial mode: 1D list combining data from two X-clusters interleaved

        Note:
            In serial mode, data from cl_x=0 and cl_x=1 are combined by placing
            cl_x=1 data in upper bits (shifted by 24 bits).

        """
        params = self.params

        # Initialize output stream structure
        stream = [[[[] for c in range(params.NUM_GLB_IACT)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]

        # Convert SPAD storage to bit-packed streams for each cluster/router
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.NUM_GLB_IACT):
                    current_spad = spad_storage[cl_x][cl_y][router]

                    # Process each cycle's SPAD data
                    for cycle in range(len(current_spad)):
                        # Address stream disabled (would pack address pointers)
                        #stream[cl_x][cl_y][router].extend(self.create_pe_addr_iact_stream(current_spad[cycle]))

                        # Pack activation data and overhead into transmission words
                        stream[cl_x][cl_y][router].extend(self.create_pe_data_iact_stream(current_spad[cycle]))

        # Special handling for serial mode: combine two X-clusters
        if(params.SERIAL):
            temp_stream = stream
            stream = []
            # Interleave data from cl_x=0 (lower bits) and cl_x=1 (upper bits)
            for cl_y in range(params.Clusters_Y):
                for router in range(params.NUM_GLB_IACT):
                    for word in range(len(temp_stream[0][cl_y][router])):
                        # Combine: cl_x=0 in bits [0:23], cl_x=1 in bits [24:47]
                        stream.append(temp_stream[0][cl_y][router][word] + (temp_stream[1][cl_y][router][word] * (2**24)))

        return stream
    
    def create_pe_addr_iact_stream(self, spad):
        """Pack activation address pointers into transmission words.

        Packs multiple address values into transmission words based on hardware bitwidth.
        Currently disabled but available for sparse addressing modes.

        Args:
            spad: SPAD data structure containing addresses in spad[0]

        Returns:
            list: Stream of bit-packed address words

        Note:
            Each transmission word contains multiple addresses packed sequentially.
            The number of addresses per word is determined by transmission bitwidth
            divided by address bitwidth.

        """
        layer_params = self.layer_params
        params = self.params

        # Calculate how many addresses fit in one transmission word
        addr_per_trans = math.floor(params.IACT_Trans_Bitwidth/params.IACT_Addr_Bitwidth)
        line_counter = 0
        stream = []

        # Pack addresses into transmission words
        for spad_addr_trans in range(math.ceil(params.Iacts_Addr_per_PE/addr_per_trans)):
            temp_trans = 0

            # Pack multiple addresses into one word
            for addr_in_trans in range(addr_per_trans):
                try:
                    spad_word = addr_in_trans + spad_addr_trans * addr_per_trans
                    # Pack address at appropriate bit position
                    temp_trans = temp_trans + \
                        (spad[0][spad_word] \
                        << (params.IACT_Addr_Bitwidth * addr_in_trans))
                except:
                    pass  # Ignore out-of-range indices
            stream.append(temp_trans)
            line_counter = line_counter + 1

            # Early exit when all used addresses are packed
            if (line_counter == math.ceil(layer_params.used_iact_addr_per_PE/addr_per_trans)):
                return stream
        return stream
    
    def create_pe_data_iact_stream(self, spad):
        """Pack activation data with overhead bits into transmission words.

        Each activation value is packed together with its 4-bit overhead counter
        to create a "Word with OverHead" (WOH). Multiple WOH values are then packed
        into transmission words.

        Args:
            spad: SPAD data structure where each entry is [value, overhead_counter]

        Returns:
            list: Stream of bit-packed transmission words containing activations

        Note:
            The overhead counter is used for sparse encoding - it tracks the position
            of each activation value. The packed format is:
            [overhead_bits(4) | activation_value(IACT_Bitwidth)] repeated per word.

        """
        layer_params = self.layer_params
        params = self.params

        # Calculate how many activation+overhead pairs fit in one transmission word
        data_per_trans = math.floor(params.IACT_Trans_Bitwidth/params.IACT_WOH_Bitwidth)
        line_counter = 0
        stream = []

        # Pack activation data into transmission words
        for spad_data_trans in range(math.floor(params.Iacts_per_PE/data_per_trans)):
            temp_trans = 0

            # Pack multiple activation+overhead pairs into one word
            for data_in_trans in range(data_per_trans):
                try:
                    number_of_value = (data_in_trans + spad_data_trans * data_per_trans)
                    # Create combined value: 4-bit overhead + activation value
                    # Format: [overhead(4 bits) | value(IACT_Bitwidth bits)]
                    value = gtu.to_twos_complement_string(spad[number_of_value][1], 4) + \
                        gtu.to_twos_complement_string(spad[number_of_value][0], self.params.IACT_Bitwidth)
                    # Pack into transmission word at appropriate bit position
                    temp_trans = temp_trans + (int(value,2) << (data_in_trans * params.IACT_WOH_Bitwidth))
                except:
                    pass  # Ignore out-of-range indices
            stream.append(temp_trans)
            line_counter = line_counter + 1

            # Early exit when all used activations are packed
            # Note: Division by 2 suggests data_per_trans might be 2 in typical config
            if (line_counter == math.ceil(layer_params.used_iact_per_PE/2)):
                break
        return stream
    
class ConvIactStreamMapper(IactStreamMapper):
    """Specialized mapper for standard Conv2D layer input activations.

    This class extends IactStreamMapper with Conv2D-specific logic for mapping
    activations. The key difference from the base class is how channels are
    distributed across routers - Conv2D uses kernel_per_pe_cluster to group
    multiple input channels per router for efficient filter parallelism.

    Attributes:
        Inherited from IactStreamMapper

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_data):
        """Initialize Conv2D activation stream mapper.

        Args:
            params: OpenEye hardware parameters
            layer_params: Conv2D layer-specific parameters
            layer_repetition (int): Current repetition index
            dram_layer_content: Input feature map data from DRAM
            sparse_data (int): 1 for sparse encoding, 0 for dense

        """
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, sparse_data)

    def write_iact_data_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate Conv2D-specific activation data SPAD storage.

        Overrides base class to implement Conv2D-specific channel distribution.
        The key difference is that channels are grouped per router based on
        kernel_per_pe_cluster, allowing multiple input channels to be processed
        in parallel for filter operations.

        Channel Distribution:
        - Each router handles a subset of input channels
        - Channels are divided by (iact_transmissions_pe * kernel_per_pe_cluster)
        - Router index determines which channel subset this PE loads

        Args:
            cl_x (int): Cluster X-coordinate
            cl_y (int): Cluster Y-coordinate
            router (int): GLB router index (determines channel subset)
            cycle (int): Current refresh/computation cycle
            iact_cycle (int): Current activation write cycle

        Returns:
            list: SPAD storage as [[value, overhead_counter], ...] for each PE position

        """
        layer_params = self.layer_params
        params = self.params
        layer_repetition = self.layer_repetition
        dram_fmap = self.dram_fmap

        overhead_counter = 0
        line_counter = 0
        spad_storage = [[0 for _ in range(2)] for _ in range(params.Iacts_per_PE)]

        # === CONV2D-SPECIFIC: Calculate channel range for this router ===
        # Each router handles a subset of input channels based on kernel_per_pe_cluster
        amount_of_channels = ((math.ceil((router+1) * layer_params.input_shape[3]/layer_params.iact_transmissions_pe/layer_params.kernel_per_pe_cluster)) - \
            (math.ceil(router * layer_params.input_shape[3]/layer_params.iact_transmissions_pe/layer_params.kernel_per_pe_cluster)))

        # Total activations = channels assigned to this router * kernel height
        amout_of_iacts = amount_of_channels * layer_params.kernel_size[1]
        for words_in_storage in range(amout_of_iacts):
            if(words_in_storage < layer_params.used_iact_per_PE):
                match layer_params.single_cluster_computation:
                    case 1:
                        iact_temp_pos_x = \
                        int((math.floor((cycle * params.PEs_X) * layer_params.strideX) % \
                        ((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)) + \
                        math.floor(router/layer_params.kernel_per_pe_cluster) + \
                        iact_cycle * math.floor(params.NUM_GLB_IACT/layer_params.kernel_per_pe_cluster)) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))                             #Zero Padding
                        
                        iact_temp_pos_y = \
                        int((words_in_storage % layer_params.kernel_size[1]) + \
                        (layer_params.strideY * \
                        math.floor(((cycle * params.PEs_X) * layer_params.strideX)/((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)))) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))

                        # === CONV2D-SPECIFIC CHANNEL CALCULATION ===
                        # Channel includes router modulo term for kernel_per_pe_cluster distribution
                        # This differs from base class: channels grouped per router for filter parallelism
                        channel = (math.ceil(((router%layer_params.kernel_per_pe_cluster) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))/layer_params.kernel_per_pe_cluster)) + \
                        math.floor(words_in_storage/layer_params.kernel_size[0]) + \
                        ((layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))
                    case 2:
                        iact_temp_pos_x = \
                        int((math.floor(((cl_x * params.PEs_X) + (cycle * params.PEs_X * params.Clusters_X)) * layer_params.strideX) % \
                        ((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)) + \
                        math.floor(router/layer_params.kernel_per_pe_cluster) + \
                        iact_cycle * math.floor(params.NUM_GLB_IACT/layer_params.kernel_per_pe_cluster)) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))                             #Zero Padding
                        
                        iact_temp_pos_y = \
                        int((words_in_storage % layer_params.kernel_size[1]) + \
                        (layer_params.strideY * \
                        math.floor(((cycle * params.PEs_X * params.Clusters_X) * layer_params.strideX)/((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)))) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))      

                        channel = (math.ceil(((router%layer_params.kernel_per_pe_cluster) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))/layer_params.kernel_per_pe_cluster)) + \
                        math.floor(words_in_storage/layer_params.kernel_size[0]) + \
                        ((layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))
                    case _:
                        iact_temp_pos_x = \
                        int((math.floor(((cl_x * params.PEs_X) + \
                        (math.floor(cl_y / layer_params.used_Y_cluster) * params.PEs_X * params.Clusters_X) + \
                        cycle * params.PEs_X * math.floor(params.Clusters/layer_params.used_Y_cluster)) * layer_params.strideX) % \
                        ((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)) + \
                        math.floor(router/layer_params.kernel_per_pe_cluster) + \
                        iact_cycle * math.floor(params.NUM_GLB_IACT/layer_params.kernel_per_pe_cluster)) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))                             #Zero Padding
                        
                        iact_temp_pos_y = \
                        int((words_in_storage % layer_params.kernel_size[1]) + \
                        (layer_params.strideY * \
                        math.floor((((math.floor(cl_y / layer_params.used_Y_cluster) * params.PEs_X * params.Clusters_X) + \
                        (cl_x * params.PEs_X) + \
                        cycle * params.PEs_X * math.floor(params.Clusters/self.layer_params.used_Y_cluster)) * layer_params.strideX)/((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideX)))) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))                             #Zero Padding

                        channel = (math.ceil(((router%layer_params.kernel_per_pe_cluster) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))/layer_params.kernel_per_pe_cluster))+ \
                        math.floor(words_in_storage/layer_params.kernel_size[0]) + \
                        ((layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))


                if((((math.floor(cycle/ self.layer_params.used_Y_cluster) * params.Clusters_Y *  params.Clusters_X * params.PEs_X) + \
                    ((math.floor(cl_y/ self.layer_params.used_Y_cluster) *  params.Clusters_X * params.PEs_X)) + (cl_x * params.PEs_X)) < \
                   (layer_params.output_shape[1] * (layer_params.output_shape[2]+layer_params.add_up))) | (layer_params.single_cluster_computation != 0)):
                    if(((iact_temp_pos_x >= 0) & (iact_temp_pos_x < (layer_params.output_shape[1] * layer_params.strideX))) & \
                    ((iact_temp_pos_y) >= 0) & (iact_temp_pos_y < (layer_params.output_shape[2] * layer_params.strideY))):
                        try:
                            spad_storage[words_in_storage][0] = dram_fmap[channel][iact_temp_pos_x][iact_temp_pos_y]
                        except:
                            spad_storage[words_in_storage][0] = 0

                    else:
                        spad_storage[words_in_storage][0] = 1
                    spad_storage[words_in_storage][1] = overhead_counter
                    overhead_counter = overhead_counter + 1

        return spad_storage
        
    def write_iact_addr_storage(self, cl_x, cl_y, router, cycle, iact_cycle):

        layer_params = self.layer_params
        params = self.params
        line_counter = 0
        spad_storage = [0 for _ in range(self.params.Iacts_Addr_per_PE)]
        if (layer_params.kernel_size[0] == 1):
            spad_storage[0] = layer_params.used_channels
        else:
            for words_in_storage in range(params.Iacts_Addr_per_PE):
                if(words_in_storage < (math.ceil(layer_params.used_iact_per_PE/layer_params.kernel_size[0]))):
                    spad_storage[words_in_storage] = (layer_params.kernel_size[0] * (words_in_storage + 1))
                line_counter = line_counter + 1

        return spad_storage
     
class DenseIactStreamMapper(IactStreamMapper):
    """Specialized mapper for fully connected (Dense) layer input activations.

    This class handles the simpler 1D activation mapping for Dense layers. Unlike
    convolutional layers, Dense layers don't have spatial dimensions or kernels,
    so activations are simply distributed sequentially across routers and PEs.

    Key Differences from Conv:
    - Input is 1D vector instead of 2D/3D feature map
    - No spatial position calculation or padding
    - Sequential distribution across routers without kernel windowing
    - Only first Y-cluster (cl_y==0) loads data (no Y-dimension parallelism)

    Attributes:
        Inherited from IactStreamMapper

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_data):
        """Initialize Dense layer activation stream mapper.

        Args:
            params: OpenEye hardware parameters
            layer_params: Dense layer-specific parameters
            layer_repetition (int): Current repetition index
            dram_layer_content: Input feature vector from DRAM (1D array)
            sparse_data (int): 1 for sparse encoding, 0 for dense

        """
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, sparse_data)

    def get_iact_stream(self):
        layer_params = self.layer_params
        params = self.params
        dram_fmap = self.dram_fmap
        if (not self.params.SERIAL) :
            iact_stream = [[[[] for c in range(params.NUM_GLB_IACT)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]
            for cl_x in range(params.Clusters_X):
                for cl_y in range(params.Clusters_Y):
                    for router in range(params.NUM_GLB_IACT):
                        iact_stream[cl_x][cl_y][router] = write_iact_data_glb(cl_x, cl_y, router)
            iact_stream = create_complete_iact_stream(iact_stream)
        else :
            iact_stream = []
            if (layer_params.skipIact == 0) :
                bitwidth = params.IACT_Bitwidth
                dma_bitwidth = params.DMA_Bit_AXI
                values_per_word = dma_bitwidth // bitwidth
                transmissions = math.ceil((layer_params.input_shape[3] / (params.NUM_GLB_IACT*layer_params.used_iact_per_PE))) * (params.NUM_GLB_IACT*layer_params.used_iact_per_PE)
                transmissions = math.ceil(transmissions/values_per_word)
                for i in range(0, transmissions):
                    word = 0
                    for j in range(values_per_word):
                        if (i * values_per_word) + j < len(dram_fmap):
                            val_twos = gtu.to_twos_complement(dram_fmap[(i * values_per_word) + j], bitwidth)
                            word |= val_twos << (j * bitwidth)
                    iact_stream.append(word)
        return iact_stream

    def write_iact_data_glb(self, cl_x, cl_y, router):
        """Generate GLB activation data for Dense layers.

        Note: Y-cluster scheduling check is disabled (commented out) for Dense layers
        since only cl_y==0 participates in loading activations.

        """
        storage = []
        for cycle in range(self.layer_params.needed_refreshes_mx[self.layer_repetition][1],self.layer_params.needed_refreshes_mx[self.layer_repetition][2]):
            for iact_cycle in range(self.layer_params.needed_Iact_writes):
                # Y-cluster check disabled for Dense - only cl_y==0 loads data
                #if ((cl_y - cycle) % self.layer_params.used_Y_cluster == 0) :
                spad = self.write_iact_pe(cl_x, cl_y, router, cycle, iact_cycle)
                if (self.sparse_data == 1):
                    storage.append(self.set_sparse_stream(spad))
                else:
                    storage.append(spad)
        return storage

    def write_iact_addr_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate address storage for Dense layers.

        For Dense layers, only one address pointer is needed since activations are
        sequential. Only cl_y==0 generates valid addresses.

        """
        layer_params = self.layer_params
        params = self.params
        spad_storage = [0 for _ in range(self.params.Iacts_Addr_per_PE)]

        # Only first Y-cluster generates addresses
        if(cl_y == 0):
            for words_in_storage in range(params.Iacts_Addr_per_PE):
                if(words_in_storage < 1):
                    # Single address pointer to all used activations
                    spad_storage[words_in_storage] = (layer_params.used_iact_per_PE)
        return spad_storage

    def write_iact_data_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate Dense-specific activation data SPAD storage.

        For Dense layers, activations are loaded sequentially from a 1D vector.
        The position is calculated simply as: base + router_offset + repetition_offset.
        Only cl_y==0 loads actual data; other clusters remain empty.

        """
        params = self.params
        layer_params = self.layer_params
        dram_fmap = self.dram_fmap
        layer_repetition = self.layer_repetition

        overhead_counter = 0
        spad_storage = [[0 for _ in range(2)] for _ in range(params.Iacts_per_PE)]

        # Only first Y-cluster loads activations for Dense layers
        if(cl_y == 0):
            for words_in_storage in range(math.ceil(params.Iacts_per_PE)):
                if(words_in_storage < layer_params.used_iact_per_PE):
                    # Simple sequential 1D indexing (no spatial dimensions)
                    # Index = local_offset + router_contribution + repetition_contribution
                    iact_temp_pos_x = words_in_storage + \
                    router * layer_params.used_iact_per_PE + \
                    (layer_repetition % layer_params.iact_transmissions_pe) * params.NUM_GLB_IACT * layer_params.used_iact_per_PE

                    try:
                        # Fetch from 1D DRAM vector
                        spad_storage[words_in_storage][0]= dram_fmap[iact_temp_pos_x]
                    except:
                        # Out of bounds: use 0 (not padding value 1 like Conv)
                        spad_storage[words_in_storage][0]= 0
                    spad_storage[words_in_storage][1]= overhead_counter

                    overhead_counter = overhead_counter + 1

        return spad_storage
    
class DwIactStreamMapper(IactStreamMapper):
    """Specialized mapper for depthwise convolution layer input activations.

    This class handles activation mapping for depthwise convolutions, where each
    input channel is convolved with its own kernel independently (no cross-channel
    mixing). The activation distribution differs from standard Conv2D in how
    channels are assigned to routers and clusters.

    Key Differences from Standard Conv:
    - Each channel processed independently (channel-wise parallelism)
    - Different channel-to-cluster assignment strategy
    - Modified create_pe_data_iact_stream without overhead bits
    - Channel distribution across Y-clusters for depthwise operations

    Attributes:
        Inherited from IactStreamMapper

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_data):
        """Initialize depthwise convolution activation stream mapper.

        Args:
            params: OpenEye hardware parameters
            layer_params: Depthwise convolution layer-specific parameters
            layer_repetition (int): Current repetition index
            dram_layer_content: Input feature map data from DRAM
            sparse_data (int): 1 for sparse encoding, 0 for dense

        """
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, sparse_data)

    def write_iact_addr_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        layer_params = self.layer_params
        params = self.params
        spad_storage = [0 for _ in range(self.params.Iacts_Addr_per_PE)]
        for words_in_storage in range(params.Iacts_Addr_per_PE):
            if(words_in_storage < (math.ceil(layer_params.used_iact_per_PE/layer_params.kernel_size[0]))):
                spad_storage[words_in_storage] = (layer_params.kernel_size[0] * (words_in_storage + 1))
        return spad_storage
    
    def write_iact_data_glb(self, cl_x, cl_y, router):
        storage = []
        for cycle in range(self.layer_params.needed_refreshes_mx[self.layer_repetition][1],self.layer_params.needed_refreshes_mx[self.layer_repetition][2]):
            for iact_cycle in range(self.layer_params.needed_Iact_writes):
                if ((cl_y - cycle) % self.layer_params.used_Y_cluster == 0) :
                    spad = self.write_iact_pe(cl_x, cl_y, router, cycle, iact_cycle)
                    if (self.sparse_data == 1):
                        storage.append(self.set_sparse_stream(spad))
                    else:
                        storage.append(spad)
        return storage
        
    def create_complete_iact_stream(self, spad_storage):
        params = self.params
        stream = [[[[] for c in range(params.NUM_GLB_IACT)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.NUM_GLB_IACT):
                    current_spad = spad_storage[cl_x][cl_y][router]
                    for cycle in range(len(current_spad)):
                        stream[cl_x][cl_y][router].extend(self.create_pe_data_iact_stream(current_spad[cycle]))
        return stream

    def create_pe_data_iact_stream(self, spad):
        layer_params = self.layer_params
        params = self.params

        data_per_trans = math.floor(params.IACT_Trans_Bitwidth/params.IACT_Bitwidth)
        line_counter = 0
        stream = []
        for spad_data_trans in range(math.ceil(params.Iacts_per_PE/data_per_trans)):
            temp_trans = 0
            for data_in_trans in range(data_per_trans):
                try:
                    number_of_value = (data_in_trans + spad_data_trans * data_per_trans)
                    value = gtu.to_twos_complement_string(spad[1][number_of_value][0], self.params.IACT_Bitwidth)
                    temp_trans = temp_trans + (int(value,2) << (data_in_trans * params.IACT_Bitwidth))
                except:
                    pass
            stream.append(temp_trans)
            line_counter = line_counter + 1
            if (line_counter == math.ceil(layer_params.used_iact_per_PE/data_per_trans)):
                break
        return stream
    
    def write_iact_data_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        params = self.params
        layer_params = self.layer_params
        dram_fmap = self.dram_fmap

        iact_temp_pos_x = 0
        iact_temp_pos_y = 0
        overhead_counter = 0
        spad_storage = [[0 for _ in range(2)] for _ in range(params.Iacts_per_PE)]

        for words_in_storage in range(math.ceil(params.Iacts_per_PE)):
            if (words_in_storage < layer_params.used_iact_per_PE):
                match layer_params.single_cluster_computation:
                    case 1:
                        iact_temp_pos_x = int((((words_in_storage) % layer_params.kernel_size[1]) + \
                        ((cycle * params.PEs_X * layer_params.strideX))) % \
                        ((layer_params.output_shape[2]+layer_params.add_up) * layer_params.strideX) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2)) + \
                        (router * params.NUM_GLB_IACT))

                        iact_temp_pos_y = int(iact_cycle + (layer_params.strideY * math.floor( \
                        ((cycle *params.PEs_X) * layer_params.strideX)/ \
                        ((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideY)))) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))                             #Zero Padding

                        
                        channel = (cl_x  + cl_y * params.Clusters_X) + \
                        ((self.layer_repetition * params.Clusters))
                    case 2:
                        iact_temp_pos_x = int(((cl_x * params.PEs_X * layer_params.strideX) + \
                        ((words_in_storage) % layer_params.kernel_size[1]) + \
                        (cycle * params.PEs_X * layer_params.strideX * params.Clusters_X)) % \
                        ((layer_params.output_shape[2]+layer_params.add_up) * layer_params.strideX) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2)) + \
                        (router * params.NUM_GLB_IACT))

                        iact_temp_pos_y = \
                        int(iact_cycle + (layer_params.strideY * math.floor( \
                        (((cl_x * params.PEs_X) + \
                        (cycle * params.PEs_X * params.Clusters_X)) * \
                        layer_params.strideX)/((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideY)))) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))                             #Zero Padding
                        
                        channel = cl_y + ((self.layer_repetition * params.Clusters_Y))
                    case _:
                        iact_temp_pos_x = int(((cl_x * params.PEs_X * layer_params.strideX) + \
                        ((cl_y / layer_params.used_Y_cluster) * params.PEs_X * params.Clusters_X * layer_params.strideX) + \
                        ((words_in_storage) % layer_params.kernel_size[1]) ) % \
                        ((layer_params.output_shape[2]+layer_params.add_up) * layer_params.strideX) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2)) + \
                        (router * params.NUM_GLB_IACT))

                        iact_temp_pos_y = \
                        int(iact_cycle + (layer_params.strideY * math.floor( \
                        ((((cl_y / layer_params.used_Y_cluster) * params.PEs_X * params.Clusters_X) + \
                        ((cl_y % layer_params.used_Y_cluster) * params.Clusters_X * params.PEs_X) + \
                        (cl_x * params.PEs_X) + \
                        cycle * params.PEs_X * params.Clusters) * \
                        layer_params.strideX)/((layer_params.output_shape[2]+layer_params.add_up)*layer_params.strideY)))) - \
                        (math.ceil((layer_params.kernel_size[1]-1)/2))                             #Zero Padding
                        
                        channel = math.floor(words_in_storage/layer_params.kernel_size[0]) + ((self.layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))

                if(((iact_temp_pos_x >= 0) & (iact_temp_pos_x < (layer_params.output_shape[1] * layer_params.strideX))) & \
                ((iact_temp_pos_y) >= 0) & (iact_temp_pos_y < (layer_params.output_shape[2] * layer_params.strideY)) & \
                    (channel < layer_params.output_shape[3])):
                    spad_storage[words_in_storage][0]= dram_fmap[channel][iact_temp_pos_x][iact_temp_pos_y]
                else:
                    spad_storage[words_in_storage][0]= 1
                spad_storage[words_in_storage][1] = overhead_counter
                overhead_counter = overhead_counter + 1

        return spad_storage
    