# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Weight stream mapping for OpenEye accelerator testing.

This module provides classes for mapping neural network weights from DRAM to hardware
data streams for the OpenEye accelerator. It handles the transformation of weight tensors
into formatted bitstreams that can be loaded into PE (Processing Element) scratchpads
across different layer types.

Key Features:
    - Weight data organization for PE scratchpad storage (SPAD)
    - Address pointer generation for weight access patterns
    - Support for sparse weight encoding with offset compression
    - Bitstream packing for efficient serial/parallel transmission
    - Layer-specific weight mapping for Conv2D, Depthwise, and Dense layers

Weight Stream Structure:
    The weight stream consists of two components:
    1. Address SPAD: Pointers indicating where weights start in data SPAD
    2. Data SPAD: Actual weight values with optional sparsity overhead encoding

    For sparse mode, weights are encoded as [value, offset] pairs where offset
    indicates how many zeros were skipped before this non-zero weight.

Typical Usage:
    >>> # For standard convolution
    >>> conv_mapper = ConvWghtStreamMapper(params, layer_params, layer_repetition,
    ...                                     dram_weights, sparse_data=False)
    >>> wght_stream = conv_mapper.get_wght_stream()

    >>> # For depthwise convolution
    >>> dw_mapper = DwWghtStreamMapper(params, layer_params, layer_repetition,
    ...                                 dram_weights, sparse_data=False)
    >>> wght_stream = dw_mapper.get_wght_stream()

    >>> # For dense/fully-connected layers
    >>> dense_mapper = DenseWghtStreamMapper(params, layer_params, layer_repetition,
    ...                                       dram_weights, sparse_data=False)
    >>> wght_stream = dense_mapper.get_wght_stream()
"""

import numpy as np
import math
import logging
import open_eye.generic_test_utils as gtu
import open_eye.stream_dicts as strdic

logger = logging.getLogger("cocotb")


class WghtStreamMapper(object):
    """Base class for weight stream mapping to OpenEye accelerator PEs.

    This class handles the conversion of weights from DRAM format to hardware bitstreams
    suitable for loading into PE scratchpads. It manages both address and data streams,
    with optional support for sparse weight encoding.

    The mapper organizes weights across multiple dimensions:
    - Spatial: X/Y clusters containing PE arrays
    - Temporal: Layer repetitions for weight reuse patterns
    - Logical: Router assignments for data distribution

    Attributes:
        params: Hardware architecture parameters (cluster counts, PE dimensions, bitwidths)
        layer_params: Layer-specific parameters (kernel size, channels, filters, etc.)
        layer_repetition (int): Current iteration index for multi-transmission layers
        dram_weights: Source weight tensor from DRAM storage
        sparse_data (int): Flag indicating sparse encoding (1) or dense (0)
        storage (list): Temporary storage for stream construction

    """

    # Omit all-zero packed words from the data stream. Correct only for the
    # overhead-encoded conv weight format; raw dense payloads must transmit
    # every word (see create_pe_data_wght_stream).
    SKIP_ZERO_WORDS = True

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_data):
        """Initialize the weight stream mapper.

        Args:
            params: OpenEye hardware parameters (clusters, PEs, bitwidths, etc.)
            layer_params: Layer configuration (kernel, channels, strides, etc.)
            layer_repetition (int): Current repetition index for this layer
            dram_layer_content: Weight tensor from DRAM storage
            sparse_data (int): Enable sparse encoding (1) or use dense format (0)

        """
        self.params = params
        self.layer_params = layer_params
        self.layer_repetition = layer_repetition
        self.dram_weights = dram_layer_content
        self.sparse_data = sparse_data

        # Initialize storage structure based on communication mode
        if (params.SERIAL):
            # Serial mode: indexed by stream_serial_dict keys
            self.storage = [[] for _ in range(len(strdic.stream_serial_dict))]
        else:
            # Parallel mode: indexed by stream_parallel_dict keys
            self.storage = [[] for _ in range(len(strdic.stream_parallel_dict))]

    def get_wght_stream(self):
        """Generate complete weight bitstream for all PEs across all clusters.

        This is the main entry point for weight stream generation. It orchestrates the
        process of loading weights from DRAM, organizing them into PE scratchpads,
        optionally applying sparse encoding, and formatting them into transmission-ready
        bitstreams.

        The method handles two transmission modes:
        - Serial: Iterates through all weight transmissions, creating a time-sequenced stream
        - Parallel: Creates a single snapshot of weight data for parallel loading

        Returns:
            list: Complete weight bitstream ready for hardware transmission. Format varies:
                - Serial mode: Flat list of packed integers sequenced by transmission time
                - Parallel mode: Nested structure [cluster_x][cluster_y][router][words]

        Note:
            Only active PEs (indicated by computing_mx == 1) receive weight data.
            Inactive PEs are skipped to save bandwidth and memory.

        """
        # Initialize 3D storage: [cluster_x][cluster_y][router] -> SPAD contents
        storage = [[[[] for c in range(self.params.Wght_Routers)] for b in range(self.params.Clusters_Y)] for a in range(self.params.Clusters_X)]

        if (self.params.SERIAL):
            # === SERIAL MODE: Multiple transmissions over time ===
            wght_stream = []

            # Iterate through each weight transmission cycle
            for layer_repetition_loop in range(self.layer_params.needed_wght_transmissions):
                self.layer_repetition = layer_repetition_loop
                temp_storage = [[[[] for c in range(self.params.Wght_Routers)] for b in range(self.params.Clusters_Y)] for a in range(self.params.Clusters_X)]

                # Populate weight data for each active PE
                for cl_x in range(self.params.Clusters_X):
                    for cl_y in range(self.params.Clusters_Y):
                        for router in range(self.params.Wght_Routers):
                            # Check if this PE is actively computing (bit 0 of computing_mx)
                            if(self.layer_params.computing_mx[cl_x][cl_y][router][0] == 1):
                                # Generate SPAD contents for this PE
                                spad = self.write_wght_pe(cl_x, cl_y, router)

                                # Apply sparse encoding if enabled
                                if (self.sparse_data == 1):
                                    temp_storage[cl_x][cl_y][router] = self.set_sparse_stream(spad)
                                else:
                                    temp_storage[cl_x][cl_y][router] = spad
                # Convert SPAD data to transmission bitstream and append to overall stream
                wght_stream.extend(self.create_complete_wght_stream(temp_storage))
        else:
            # === PARALLEL MODE: Single transmission with all data ===
            for cl_x in range(self.params.Clusters_X):
                for cl_y in range(self.params.Clusters_Y):
                    for router in range(self.params.Wght_Routers):
                        # Check if this PE is actively computing
                        if(self.layer_params.computing_mx[cl_x][cl_y][router][0] == 1):
                            # Generate SPAD contents for this PE
                            spad = self.write_wght_pe(cl_x, cl_y, router)

                            # Apply sparse encoding if enabled
                            if (self.sparse_data == 1):
                                storage[cl_x][cl_y][router] = self.set_sparse_stream(spad)
                            else:
                                storage[cl_x][cl_y][router] = spad
            # Convert SPAD data to transmission bitstream
            wght_stream = self.create_complete_wght_stream(storage)
        chunk = self.params.Clusters * self.params.NUM_GLB_WGHT
        wght_stream = gtu.transform_n_to_m_chunked(wght_stream,self.params.WGHT_Trans_Bitwidth,self.params.DMA_BITWIDTH, chunk)
        n = self.layer_params.wght_cycles_one_word_all_ram
        temp = []
        for i in range(0, len(wght_stream), n):
            part = wght_stream[i : i + n]
            temp.extend(part[::-1])
        wght_stream = temp
        return wght_stream
    
    def set_sparse_stream(self, spad_data):
        """Convert dense SPAD data to sparse format with offset encoding.

        This method transforms weight data from dense representation to a sparse format
        that compresses zero weights using offset encoding. Each non-zero weight is stored
        with an offset field indicating how many zeros preceded it.

        Sparse Format: [weight_value, skip_offset]
        - weight_value: The actual non-zero weight
        - skip_offset: Number of zeros skipped before this weight

        This reduces memory bandwidth and storage for networks with high weight sparsity.

        Args:
            spad_data (list): Dense SPAD data as [addr_spad, data_spad]
                - addr_spad: Address pointers for data regions
                - data_spad: Weight values in dense format

        Returns:
            list: Sparse SPAD data as [new_addr_spad, new_data_spad]
                - new_addr_spad: Updated address pointers for sparse data
                - new_data_spad: Sparse-encoded weights with offset information

        """
        layer_params = self.layer_params
        params = self.params

        # Initialize sparse structures
        addr_spad, data_spad = [0], spad_data[1]
        temp_word, temp_part_list, temp_complete_list, temp_offset, added_words = [], [], [], 0, 0

        # === SPARSE DATA SPAD ENCODING ===
        # Process each address region to compress zero weights
        for addr_spad_count in range(params.Wghts_Addr_per_PE - 1):
            # Only process regions with valid address pointers
            if (spad_data[0][addr_spad_count + 1] != 0):
                # Initialize new address pointer based on previous
                addr_spad.append(addr_spad[addr_spad_count])

                # Process weight data in this address region
                for data_spad_pos in range(spad_data[0][addr_spad_count], spad_data[0][addr_spad_count + 1]):
                    # Collect all parallel MAC values for this position
                    for x in range(params.PARALLEL_MACS):
                        temp_part_list.append(data_spad[data_spad_pos][x][0])

                    # Encode weights with offset compression
                    for x in range(math.ceil(len(temp_part_list) / params.PARALLEL_MACS)):
                        for y in range(params.PARALLEL_MACS):
                            word = (x * params.PARALLEL_MACS) + y
                            try:
                                if (temp_part_list[word] != 0):
                                    # Non-zero weight: store with accumulated offset
                                    temp_word.append([temp_part_list[word], temp_offset])
                                    temp_offset = 0  # Reset offset counter
                                    added_words = added_words + 1
                                else:
                                    # Zero weight: increment skip offset
                                    temp_offset = temp_offset + 1
                            except:
                                # Handle out-of-bounds with zero padding
                                temp_word.append([0, 0])
                                added_words = added_words + 1

                            # Complete a full word when PARALLEL_MACS weights accumulated
                            if (added_words == params.PARALLEL_MACS):
                                temp_complete_list.append(temp_word)
                                addr_spad[addr_spad_count + 1] = addr_spad[addr_spad_count + 1] + 1
                                added_words = 0
                                temp_word = []

                    temp_part_list = []
                temp_offset = 0  # Reset offset for next region

        # Replace dense data with sparse-encoded data
        data_spad = temp_complete_list
        return [addr_spad, data_spad]

    def write_wght_pe(self, cl_x, cl_y, router):
        """Generate complete SPAD contents (address and data) for a single PE.

        This method orchestrates the creation of both address and data SPAD contents
        for a specific PE identified by its cluster coordinates and router ID.

        Args:
            cl_x (int): Cluster X coordinate
            cl_y (int): Cluster Y coordinate
            router (int): Router/PE ID within the cluster

        Returns:
            list: Complete SPAD contents as [addr_spad, data_spad]
                - addr_spad: Address pointers for data regions
                - data_spad: Weight values organized by address regions

        """
        # Generate weight data first
        data_spad = self.write_wght_data_storage(cl_x, cl_y, router)

        # Generate address pointers based on data organization
        return data_spad

    def write_wght_data_storage(self, cl_x, cl_y, router):
        """Populate weight data SPAD from DRAM for a specific PE.

        This method extracts the relevant weights from DRAM and organizes them into
        the PE's weight data scratchpad. It handles the mapping of multi-dimensional
        weight tensors (channels, filters, kernel_y, kernel_x) to the linear SPAD
        storage format.

        The mapping accounts for:
        - Layer repetitions (temporal tiling of large layers)
        - Input channel partitioning across transmissions
        - Filter distribution across PEs
        - Kernel spatial dimensions

        Args:
            cl_x (int): Cluster X coordinate
            cl_y (int): Cluster Y coordinate
            router (int): Router/PE ID within the cluster

        Returns:
            list: Weight data SPAD as 3D array [words][parallel_macs][value, overhead]
                - words: Number of storage words in SPAD
                - parallel_macs: Number of parallel MAC units
                - [0]: Weight value
                - [1]: Overhead/sparsity metadata

        """
        layer_repetition = self.layer_repetition
        layer_params = self.layer_params
        params = self.params
        dram = self.dram_weights

        # Initialize SPAD storage: [words][parallel_macs][value, overhead]
        spad_storage = [[[0 for _ in range(2)] for _ in range(self.params.PARALLEL_MACS)] for _ in range(int(self.params.Wghts_per_PE/self.params.PARALLEL_MACS))]
        overhead_counter = 0
        kernel_x = 0

        # Calculate starting channel for this transmission
        channel = (layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)

        # Determine how many filters each PE processes per input activation
        filters_per_calculation = math.ceil(layer_params.used_wght_per_PE/layer_params.used_iact_per_PE)

        # Calculate starting filter index for this repetition
        start_current_repetition = int((math.floor(layer_repetition/layer_params.iact_transmissions_pe) % layer_params.needed_wght_transmissions) * filters_per_calculation)
        filters = start_current_repetition

        # Populate SPAD with weights from DRAM
        for words_in_storage in range(int(self.params.Wghts_per_PE/self.params.PARALLEL_MACS)):
            # Calculate which kernel row this PE handles
            kernel_row = (cl_y % (layer_params.used_Y_cluster * params.PEs_Y)) * params.PEs_Y + router

            # Only process if within valid kernel dimensions
            if(kernel_row < (layer_params.kernel_size[1] * int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))):
                for spad_val_number in range(self.params.PARALLEL_MACS):
                    # Check if we're still within the channel range for this transmission
                    if(channel != int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe) + (layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)):
                        # Load weight from DRAM: dram[channel][filter][kernel_row][kernel_x]
                        spad_storage[words_in_storage][spad_val_number][0] = dram[channel][filters][kernel_row][kernel_x]
                        spad_storage[words_in_storage][spad_val_number][1] = overhead_counter
                        overhead_counter = 0

                        # Move to next filter
                        filters = filters + 1

                        # Wrap to next kernel position when all filters processed
                        if((filters == (start_current_repetition + filters_per_calculation))):
                            filters = start_current_repetition
                            kernel_x = kernel_x + 1

                        # Wrap to next channel when kernel width exhausted
                        if(kernel_x == layer_params.kernel_size[0]):
                            kernel_x = 0
                            channel = channel + 1

            # Stop when we've filled the used portion of SPAD
            if (words_in_storage == math.ceil(layer_params.used_wght_per_PE/self.params.PARALLEL_MACS)):
                break
        return spad_storage
        
    def write_wght_addr_storage(self, cl_x, cl_y, router, data_spad):
        """Generate address SPAD pointers for weight data access.

        This method creates address pointers that indicate the starting positions of
        different weight regions in the data SPAD. These pointers enable efficient
        indexed access to weights during computation, supporting patterns like
        strided access for different input channels or filter groups.

        Args:
            cl_x (int): Cluster X coordinate (unused in base implementation)
            cl_y (int): Cluster Y coordinate (unused in base implementation)
            router (int): Router/PE ID within cluster (unused in base implementation)
            data_spad: Weight data SPAD (unused in base implementation, used by subclasses)

        Returns:
            list: Address SPAD containing integer pointers to data regions

        """
        layer_params = self.layer_params
        params = self.params

        # Initialize address SPAD with zeros
        spad_storage = [0 for _ in range(self.params.Wghts_Addr_per_PE)]

        # Generate address pointers for each region
        for words_in_storage in range(params.Wghts_Addr_per_PE):
            # Generate pointers for all but the last address
            if((words_in_storage != (self.layer_params.used_wght_addr_per_PE - 1))):
                # Calculate stride: weights per region based on kernel and channel partitioning
                spad_storage[words_in_storage] = \
                    int(words_in_storage * math.ceil(layer_params.used_wght_per_PE/layer_params.kernel_size[0]/self.params.PARALLEL_MACS/ int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)))
            else:
                # Last address entry is implicit (end of SPAD)
                break

        return spad_storage
        
    def create_complete_wght_stream(self, spad_storage):
        """Convert SPAD storage to formatted weight bitstream for all PEs.

        This method transforms the organized SPAD data into a hardware-transmittable
        bitstream. It handles both serial and parallel transmission formats, packing
        weight data according to hardware bitwidth constraints.

        Args:
            spad_storage (list): 3D array of SPAD contents [cluster_x][cluster_y][router]
                                 Each element contains [addr_spad, data_spad] for one PE

        Returns:
            list: Formatted weight bitstream
                - Serial mode: Flat list sequenced by transmission time, may combine
                              data from multiple clusters
                - Parallel mode: Nested structure [cluster_x][cluster_y][router][words]

        """
        params = self.params

        # Initialize stream structure
        stream = [[[[] for c in range(params.Wght_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]

        # Convert each PE's SPAD to bitstream format
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Wght_Routers):
                    current_spad = spad_storage[cl_x][cl_y][router]

                    # Create data stream for this PE (addresses handled separately in subclasses)
                    stream[cl_x][cl_y][router] = self.create_pe_data_wght_stream(current_spad)

        if(params.SERIAL):
            # === SERIAL MODE: Time-multiplex across clusters ===
            # Combine data from multiple clusters into a sequential stream
            temp_stream = stream
            stream = []

            # Interleave words from different PEs for serial transmission
            for word in range(len(temp_stream[0][0][0])):
                for cl_x in range(params.Clusters_X):
                    for cl_y in range(params.Clusters_Y):
                        for router in range(params.NUM_GLB_WGHT):
                            try:
                                stream.append(temp_stream[0][cl_y][router][word])
                            except:
                                try:
                                    stream.append(temp_stream[0][cl_y][router][word])
                                except:
                                    stream.append(0)

        return stream
    
    def create_pe_addr_wght_stream(self, spad):
        """Pack address SPAD into transmission-ready bitstream for a single PE.

        This method packs multiple address pointers into fixed-width transmission words
        according to hardware bitwidth constraints. Multiple addresses are bit-packed
        into each transmission word for efficiency.

        Args:
            spad (list): Complete SPAD data as [addr_spad, data_spad]

        Returns:
            list: Address bitstream as packed integers ready for hardware transmission

        """
        layer_params = self.layer_params
        params = self.params

        # Calculate how many addresses fit in one transmission word
        addr_per_trans = math.floor(params.WGHT_Trans_Bitwidth/params.WGHT_Addr_Bitwidth)
        line_counter = 0
        stream = []

        # Pack addresses into transmission words
        for spad_addr_trans in range(math.ceil(params.Wghts_Addr_per_PE/addr_per_trans)):
            temp_trans = 0

            # Bit-pack multiple addresses into one transmission word
            for addr_in_trans in range(addr_per_trans):
                try:
                    spad_word = addr_in_trans + spad_addr_trans * addr_per_trans
                    # Shift each address to its position in the packed word
                    temp_trans = temp_trans + \
                        (spad[0][spad_word] \
                        << (params.WGHT_Addr_Bitwidth * addr_in_trans))
                except:
                    # Address out of range, leave as zero
                    pass

            # Replicate the packed word for each address it contains
            # (hardware reads one address per cycle from the packed word)
            for write_time in range(addr_per_trans):
                stream.append(temp_trans)
                line_counter = line_counter + 1
                if (line_counter == layer_params.used_wght_addr_per_PE):
                    break

            if (line_counter == layer_params.used_wght_addr_per_PE):
                break

        return stream
    
    def create_pe_data_wght_stream(self, spad):
        """Pack weight data SPAD into transmission-ready bitstream for a single PE.

        This method packs weight values (with optional overhead/sparsity encoding) into
        fixed-width transmission words. Each weight may include both a value field and
        an overhead field for sparse encoding.

        Weight Word Format: [overhead_bits][value_bits]
        - value_bits: Actual weight value in two's complement
        - overhead_bits: Sparsity offset or other metadata

        Args:
            spad (list): Complete SPAD data as [addr_spad, data_spad]

        Returns:
            list: Weight data bitstream as packed integers, zero words are omitted
                  to reduce transmission bandwidth

        """
        layer_params = self.layer_params
        params = self.params

        # Calculate how many weight values fit in one transmission word
        data_per_trans = math.floor(params.WGHT_Trans_Bitwidth/params.WGHT_WOH_Bitwidth)
        line_counter = 0
        stream = []

        # Pack weights into transmission words
        for spad_data_trans in range(math.floor(params.Wghts_per_PE/data_per_trans)):
            temp_trans = 0

            # Bit-pack multiple weights into one transmission word
            for data_in_trans in range(data_per_trans):
                try:
                    spad_word = data_in_trans + data_per_trans * spad_data_trans

                    # Convert weight value to two's complement representation
                    value = gtu.to_twos_complement(spad[spad_data_trans][data_in_trans][0], params.WGHT_Bitwidth)

                    # Convert overhead/sparsity field to two's complement
                    overhead = gtu.to_twos_complement(spad[spad_data_trans][data_in_trans][1], params.WGHT_WOH_Bitwidth - params.WGHT_Bitwidth)

                    # Combine overhead and value: [overhead_bits][value_bits]
                    value = (overhead * (2**params.WGHT_Bitwidth)) + value

                    # Shift to position in packed transmission word
                    temp_trans = temp_trans + (value << (data_in_trans * params.WGHT_WOH_Bitwidth))
                except:
                    # Weight out of range, leave as zero in packed word
                    pass

            # SKIP_ZERO_WORDS: historic behaviour that omits all-zero packed
            # words. Safe only for the overhead-encoded (sparse) conv format,
            # where meaningful words are never all-zero. For raw dense
            # payloads (Dense/GEMM) an all-zero weight pair is legitimate
            # data; dropping it desynchronizes the hardware word count and
            # shifts every following weight to a wrong SPad address (wrong
            # results or a hang in GET_WGHT). Dense mappers therefore
            # disable the skip (see DenseWghtStreamMapper).
            if (temp_trans != 0) or not self.SKIP_ZERO_WORDS:
                stream.append(temp_trans)

            line_counter = line_counter + 1
            # Stop when we've processed all used weights
            if (line_counter == math.ceil(layer_params.used_wght_per_PE/self.params.PARALLEL_MACS)):
                break

        return stream
    
class ConvWghtStreamMapper(WghtStreamMapper):
    """Weight stream mapper specialized for standard 2D convolution layers.

    This class extends WghtStreamMapper with Conv2D-specific weight mapping logic.
    Standard convolutions use full kernels across all input-output channel pairs,
    with weights distributed across PEs based on spatial and channel partitioning.

    Key differences from base class:
    - Supports multi-kernel-per-PE-cluster configurations
    - Handles channel offset calculations for different clusters
    - Accounts for single-cluster vs. multi-cluster computation modes

    Attributes:
        Inherited from WghtStreamMapper

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_data):
        """Initialize the Conv2D weight stream mapper.

        Args:
            params: OpenEye hardware parameters
            layer_params: Conv2D layer parameters
            layer_repetition (int): Current repetition index
            dram_layer_content: Weight tensor from DRAM
            sparse_data (int): Enable sparse encoding (1) or use dense format (0)

        """
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, sparse_data)

    def write_wght_data_storage(self, cl_x, cl_y, router):
        """Populate Conv2D weight data SPAD from DRAM for a specific PE.

        This overridden method handles Conv2D-specific weight distribution patterns,
        including multi-kernel configurations and cluster-based channel partitioning.

        Conv2D weights are 4D: [input_channels][output_filters][kernel_h][kernel_w]
        They are distributed across PEs based on:
        - Kernel spatial position (kernel_h, kernel_w)
        - Input channel assignment
        - Output filter assignment
        - Cluster location and computation mode

        Args:
            cl_x (int): Cluster X coordinate
            cl_y (int): Cluster Y coordinate
            router (int): Router/PE ID within the cluster

        Returns:
            list: Weight data SPAD as 3D array [words][parallel_macs][value, overhead]

        """
        layer_repetition = self.layer_repetition
        layer_params = self.layer_params
        params = self.params
        dram = self.dram_weights

        # Initialize SPAD storage with 2 parallel MACs (Conv2D specific)
        spad_storage = [[[0 for _ in range(2)] for _ in range(params.PARALLEL_MACS)] for _ in range(int(self.params.Wghts_per_PE/params.PARALLEL_MACS))]
        overhead_counter = 0
        kernel_x = 0

        # Calculate channel partitioning for this router/PE
        amount_of_channels = ((math.ceil((router+1) * layer_params.input_shape[3]/layer_params.iact_transmissions_pe/layer_params.kernel_per_pe_cluster)) - \
            (math.ceil(router * layer_params.input_shape[3]/layer_params.iact_transmissions_pe/layer_params.kernel_per_pe_cluster)))
        amount_of_iacts = amount_of_channels * layer_params.kernel_size[1]

        # Calculate starting channel index for this transmission and router
        channel = (layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe) + \
        math.ceil((math.ceil((router%layer_params.kernel_per_pe_cluster) * layer_params.input_shape[3]/layer_params.iact_transmissions_pe/layer_params.kernel_per_pe_cluster)))
        channel_offset = channel

        # Calculate filter assignment accounting for different kernel computations per PE
        filters_per_calculation = math.ceil(layer_params.used_wght_per_PE/layer_params.used_iact_per_PE) * layer_params.different_kernels_per_calculation

        # Determine starting filter for this repetition
        start_current_repetition = int((math.floor(layer_repetition/layer_params.iact_transmissions_pe) % layer_params.needed_wght_transmissions) * filters_per_calculation)

        # Adjust filter start based on cluster allocation
        amount_of_used_clusters = math.ceil((layer_params.used_Y_cluster*math.ceil(layer_params.iact_size_x/layer_params.strideX))/ params.PEs_X)
        channel_offset_in_calculation = (params.Clusters_X * cl_y + cl_x) // amount_of_used_clusters
        start_current_repetition = start_current_repetition + channel_offset_in_calculation
        # Handle different cluster computation modes
        amount_of_words = math.ceil(layer_params.used_wght_per_PE/params.PARALLEL_MACS)
        # Populate SPAD with weights
        filters = start_current_repetition
        spad_position = 0
        for words_in_storage in range(amount_of_words):
            # Calculate kernel row based on cluster Y and router assignment
            kernel_row = ((cl_y % (layer_params.used_Y_cluster)) * params.PEs_Y + (router%layer_params.kernel_size[0]))
            for spad_val_number in range(params.PARALLEL_MACS):
                # Check if still within valid channel range
                if(channel != 1 + int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe) + (layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)):
                    try:
                        # Load weight from DRAM: dram[channel][filter][kernel_row][kernel_x]
                        spad_storage[spad_position//params.PARALLEL_MACS][spad_position%params.PARALLEL_MACS][0] = dram[channel][filters][kernel_x][kernel_row]
                        if (spad_storage[spad_position//params.PARALLEL_MACS][spad_position%params.PARALLEL_MACS][0] == 0):
                            overhead_counter = overhead_counter + 1
                        else:
                            spad_storage[spad_position//params.PARALLEL_MACS][spad_position%params.PARALLEL_MACS][1] = overhead_counter
                            spad_position = spad_position + 1
                            overhead_counter = 0
                    except:
                        # Out of bounds, use zero
                        spad_storage[spad_position//params.PARALLEL_MACS][spad_position%params.PARALLEL_MACS][0] = 0
                        spad_position = spad_position + 1

                    # Advance to next filter (with stride for different kernels)
                    filters = filters + layer_params.different_kernels_per_calculation

                    # Wrap to next channel when filter range exhausted
                    if((filters == (start_current_repetition + filters_per_calculation))):
                        filters = start_current_repetition
                        channel = channel + 1
                        spad_position = spad_position + (spad_position%params.PARALLEL_MACS)

                    # Wrap to next kernel X when channels exhausted
                    if(channel == layer_params.used_channels + channel_offset):
                        channel = channel_offset
                        kernel_x = kernel_x + 1

            # Stop when SPAD is full
            if (words_in_storage == math.ceil(layer_params.used_wght_per_PE/params.PARALLEL_MACS)):
                break
        return spad_storage
        
    def write_wght_addr_storage(self, cl_x, cl_y, router, data_spad):
        """Generate address SPAD pointers for Conv2D weight data access.

        This overridden method creates address pointers specific to Conv2D access patterns,
        accounting for the ratio of weights to input activations per PE.

        Args:
            cl_x (int): Cluster X coordinate (unused)
            cl_y (int): Cluster Y coordinate (unused)
            router (int): Router/PE ID (unused)
            data_spad: Weight data SPAD (unused)

        Returns:
            list: Address SPAD containing integer pointers to weight regions

        """
        layer_params = self.layer_params
        params = self.params
        spad_storage = [0 for _ in range(params.Wghts_Addr_per_PE)]
        temp_value = 0
        data_spad_position = 0
        # Generate address pointers based on Conv2D access stride
        for words_in_storage in range(math.ceil(params.Wghts_Addr_per_PE)):
            # Generate all addresses or all but last depending on configuration
            if((words_in_storage != (self.layer_params.used_wght_addr_per_PE - 1)) | (self.layer_params.used_wght_addr_per_PE == self.params.Wghts_Addr_per_PE)):
                # Calculate address stride based on weights per input activation
                spad_storage[words_in_storage] = \
                    int(words_in_storage * math.ceil(layer_params.used_wght_per_PE/2/int(layer_params.used_iact_per_PE)))
            else:
                break
        return spad_storage


class DenseWghtStreamMapper(WghtStreamMapper):
    """Weight stream mapper specialized for fully-connected (Dense) layers.

    This class extends WghtStreamMapper with Dense layer-specific weight mapping logic.
    Dense layers use 2D weight matrices connecting all input features to all output features,
    distributed across PEs in a different pattern than convolutional layers.

    Key differences from base class:
    - Uses 2D weight matrix instead of 4D convolution kernels
    - Different PE assignment strategy based on output features
    - Always processes all weight transmissions in sequence

    Attributes:
        Inherited from WghtStreamMapper

    """

    # Dense payloads are raw values: an all-zero weight pair is real data
    # and must be transmitted, otherwise the hardware word count
    # desynchronizes (wrong SPad addresses or a GET_WGHT hang).
    SKIP_ZERO_WORDS = False

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_data):
        """Initialize the Dense layer weight stream mapper.

        Args:
            params: OpenEye hardware parameters
            layer_params: Dense layer parameters
            layer_repetition (int): Current repetition index
            dram_layer_content: Weight matrix from DRAM [output_features][input_features]
            sparse_data (int): Enable sparse encoding (1) or use dense format (0)

        """
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, sparse_data)

    def get_wght_stream(self):
        """Generate complete weight bitstream for Dense layer across all PEs.

        This overridden method handles Dense layer-specific transmission patterns.
        Unlike Conv layers, Dense layers always iterate through all weight transmissions.

        Returns:
            list: Complete weight bitstream including both address and data streams

        """
        storage = [[[[] for c in range(self.params.Wght_Routers)] for b in range(self.params.Clusters_Y)] for a in range(self.params.Clusters_X)]

        wght_stream = []
        # Always iterate through all weight transmissions for Dense layers
        for layer_repetition_loop in range(self.layer_params.needed_wght_transmissions):
            self.layer_repetition = layer_repetition_loop
            temp_storage = [[[[] for c in range(self.params.Wght_Routers)] for b in range(self.params.Clusters_Y)] for a in range(self.params.Clusters_X)]

            # Process all X clusters and Y clusters
            for cl_x in range(self.params.Clusters_X):
                for cl_y in range(self.params.Clusters_Y):
                    for router in range(self.params.Wght_Routers):
                        # Check if this PE is active
                        if(self.layer_params.computing_mx[cl_x][cl_y][router][0] == 1):
                            spad = self.write_wght_pe(cl_x, cl_y, router)
                            # Apply sparse encoding if enabled
                            if (self.sparse_data == 1):
                                temp_storage[cl_x][cl_y][router] = self.set_sparse_stream(spad)
                            else:
                                temp_storage[cl_x][cl_y][router] = spad

            # Convert and append this transmission's stream
            wght_stream.extend(self.create_complete_wght_stream(temp_storage))
        # GET_WGHT's shift-pipeline (hdl/OpenEye_FPGA.v) treats the incoming
        # weight stream as a raw byte stream: it shifts WGHT_CYCLES_ONE_WORD_ALL_CELLS
        # DMA words through a buffer and slices off the low
        # TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT bits each time
        # wght_buffer_wr_addr advances. That only produces correct weight
        # rows if the incoming stream is already repacked into this same
        # 24-bit-per-weight-pair -> 64-bit-DMA-word layout - the base
        # WghtStreamMapper.get_wght_stream() (used by ConvWghtStreamMapper)
        # applies this repack; this Dense override predates it and was
        # never updated, so Dense/GEMM layers streamed weights in the old
        # unpacked layout, producing garbage weight data that made the PE's
        # zero-skip SPad range logic (CALCULATING state) loop indefinitely.
        # create_complete_wght_stream's SERIAL-mode combining step packs
        # Clusters_X clusters into each stream element (24 bits per
        # cluster, e.g. temp_stream[0][...] + temp_stream[1][...]*2**24 for
        # Clusters_X==2), so each element here is 24*Clusters_X bits wide,
        # not the base class's flat 24 - passing a plain 24 here silently
        # truncated away every cluster beyond the first via
        # transform_n_to_m_chunked's masking.
        elem_bits = self.params.WGHT_Trans_Bitwidth * self.params.Clusters_X
        wght_stream = gtu.transform_n_to_m_chunked(wght_stream, elem_bits, self.params.DMA_BITWIDTH, 3)
        # layer_params.wght_cycles_one_word_all_ram is only ever set by
        # calculate_transmission_cycles(), a conv-only method Dense never
        # calls; compute the same value directly from params here instead
        # of depending on that (see WghtStreamMapper.get_wght_stream's
        # base-class version for the formula this mirrors).
        n = math.ceil((self.params.Clusters * self.params.NUM_GLB_WGHT * self.params.WGHT_RAM_CELLS_WORD_BITWIDTH) / self.params.DMA_BITWIDTH)
        temp = []
        for i in range(0, len(wght_stream), n):
            part = wght_stream[i : i + n]
            temp.extend(part[::-1])
        wght_stream = temp
        return wght_stream

    def create_complete_wght_stream(self, spad_storage):
        """Convert Dense layer SPAD storage to formatted weight bitstream.

        This overridden method creates bitstreams specific to Dense layers,
        combining both address and data streams.

        Args:
            spad_storage (list): 3D array of SPAD contents [cluster_x][cluster_y][router]

        Returns:
            list: Formatted weight bitstream (address + data combined)

        """
        params = self.params

        # Initialize stream for Dense layers
        stream = [[[[] for c in range(params.Wght_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]

        # Create combined address + data stream for each PE
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Wght_Routers):
                    current_spad = spad_storage[cl_x][cl_y][router]

                    # Append data stream
                    stream[cl_x][cl_y][router].extend(self.create_pe_data_wght_stream(current_spad))

        if(params.SERIAL):
            # === SERIAL MODE: Time-multiplex across clusters ===
            temp_stream = stream
            stream = []

            for word in range(len(temp_stream[0][0][0])):
                for cl_y in range(params.Clusters_Y):
                    for router in range(params.NUM_GLB_WGHT):
                        try:
                            # Combine data from both X-clusters (24-bit shift)
                            stream.append(temp_stream[0][cl_y][router][word] + (temp_stream[1][cl_y][router][word] * (2**self.params.WGHT_Trans_Bitwidth)))
                        except:
                            # Only one cluster or no data
                            stream.append(0)

        return stream

    def write_wght_data_storage(self, cl_x, cl_y, router):
        """Populate Dense layer weight data SPAD from DRAM for a specific PE.

        This overridden method handles Dense layer-specific weight distribution.
        Dense weights are 2D matrices [output_features][input_features] distributed
        across PEs based on both X and Y cluster positions.

        Args:
            cl_x (int): Cluster X coordinate
            cl_y (int): Cluster Y coordinate
            router (int): Router/PE ID within the cluster

        Returns:
            list: Weight data SPAD as 3D array [words][parallel_macs][value, overhead]

        """
        layer_repetition = self.layer_repetition
        layer_params = self.layer_params
        params = self.params
        dram = self.dram_weights

        # Initialize SPAD storage
        spad_storage = [[[0 for _ in range(2)] for _ in range(self.params.PARALLEL_MACS)] for _ in range(int(params.Wghts_per_PE/params.PARALLEL_MACS))]
        overhead_counter = 0

        # Populate SPAD with weights from the 2D weight matrix
        for words_in_storage in range(int(params.Wghts_per_PE/params.PARALLEL_MACS)):
            for spad_val_number in range(params.PARALLEL_MACS):
                # Calculate linear position in weight stream
                position = words_in_storage * self.params.PARALLEL_MACS + spad_val_number
                needed_wghts_in_word = math.ceil(layer_params.used_psum_per_PE/self.params.PARALLEL_MACS)*self.params.PARALLEL_MACS
                # Recalculate filter index with Y-cluster assignment
                filters =  (position%needed_wghts_in_word) + \
                cl_x * layer_params.used_psum_per_PE + \
                (math.floor(layer_repetition/layer_params.iact_transmissions_pe) % layer_params.psum_transmissions_pe) * params.Clusters_X * params.Clusters_Y * layer_params.used_psum_per_PE
                # Recalculate channel with Y-cluster assignment
                #channel = math.floor((layer_repetition%layer_params.iact_transmissions_pe)*params.Wght_Routers*layer_params.used_iact_per_PE) + \
                channel = math.floor((layer_repetition)*(params.NUM_GLB_WGHT * params.Clusters_Y * layer_params.used_iact_per_PE)) + \
                math.floor(position/needed_wghts_in_word) + \
                cl_y * params.NUM_GLB_WGHT * layer_params.used_iact_per_PE + \
                router * layer_params.used_iact_per_PE 

                if ((position % needed_wghts_in_word) < layer_params.used_psum_per_PE) :
                    try:
                        # Load weight from 2D matrix: dram[output_feature][input_feature]
                        spad_storage[words_in_storage][spad_val_number][0] = dram[filters][channel]
                        spad_storage[words_in_storage][spad_val_number][1] = overhead_counter
                    except:
                        # Out of bounds, leave as zero
                        pass

            # Stop when SPAD is full
            if (words_in_storage == math.ceil(layer_params.used_wght_per_PE/self.params.PARALLEL_MACS)):
                break
        return spad_storage

    def write_wght_addr_storage(self, cl_x, cl_y, router, data_spad):
        """Generate address SPAD pointers for Dense layer weight data access.

        This overridden method creates address pointers specific to Dense layer access,
        based on the number of output features (partial sums) per PE.

        Args:
            cl_x (int): Cluster X coordinate (unused)
            cl_y (int): Cluster Y coordinate (unused)
            router (int): Router/PE ID (unused)
            data_spad: Weight data SPAD (unused)

        Returns:
            list: Address SPAD containing integer pointers to weight regions

        """
        layer_params = self.layer_params
        params = self.params

        # Initialize address SPAD
        spad_storage = [0 for _ in range(self.params.Wghts_Addr_per_PE)]

        temp_value = 0

        # Generate address pointers based on output features per PE
        for words_in_storage in range(math.ceil(params.Wghts_Addr_per_PE)):
            if(words_in_storage != (self.layer_params.used_wght_addr_per_PE - 1)):
                # Calculate address stride based on partial sums (output features) per PE
                spad_storage[words_in_storage] = \
                    int(words_in_storage * math.ceil(layer_params.used_psum_per_PE/self.params.PARALLEL_MACS))
            else:
                break

        return spad_storage

class DwWghtStreamMapper(WghtStreamMapper):
    """Weight stream mapper specialized for depthwise convolution layers.

    This class extends WghtStreamMapper with depthwise convolution-specific weight
    mapping logic. Depthwise convolution applies a separate kernel to each input channel
    independently, requiring different weight distribution patterns than standard Conv2D.

    Key differences from base class:
    - One kernel per input channel instead of full cross-channel kernels
    - Simpler weight indexing (no filter dimension)
    - Support for single-cluster computation modes
    - Different kernel row assignment strategy

    Attributes:
        Inherited from WghtStreamMapper

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_data):
        """Initialize the depthwise convolution weight stream mapper.

        Args:
            params: OpenEye hardware parameters
            layer_params: Depthwise convolution layer parameters
            layer_repetition (int): Current repetition index
            dram_layer_content: Weight tensor from DRAM [channels][kernel_h][kernel_w]
            sparse_data (int): Enable sparse encoding (1) or use dense format (0)

        """
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, sparse_data)

    def write_wght_data_storage(self, cl_x, cl_y, router):

        layer_repetition = self.layer_repetition
        layer_params = self.layer_params
        params = self.params
        dram = self.dram_weights

        spad_storage = [[[0 for _ in range(2)] for _ in range(self.params.PARALLEL_MACS)] for _ in range(int(self.params.Wghts_per_PE/self.params.PARALLEL_MACS))]
        overhead_counter = 0
        kernel_x = 0

        match layer_params.single_cluster_computation:
            case 1:
                channel = (cl_x  + cl_y * params.Clusters_X) + ((layer_repetition * params.Clusters))
            case 2:
                channel = cl_y + (layer_repetition * params.Clusters_Y)
            case _:
                channel = (layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)

                    
        if(layer_params.filters == 1):
            values_per_wght_data = 1
        else:
            values_per_wght_data = self.params.PARALLEL_MACS
        for words_in_storage in range(int(self.params.Wghts_per_PE/self.params.PARALLEL_MACS)):
            Mtrx_Row = (cl_y % layer_params.used_Y_cluster) * params.PEs_Y + router
            if(Mtrx_Row < (layer_params.kernel_size[1] * int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe))):
                for spad_val_number in range(values_per_wght_data): 
                    if(channel != int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe) + (layer_repetition % layer_params.iact_transmissions_pe) * math.ceil(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)):
                        spad_storage[words_in_storage][spad_val_number][0] = dram[channel][Mtrx_Row][kernel_x]
                        spad_storage[words_in_storage][spad_val_number][1] = overhead_counter
                        overhead_counter = overhead_counter + 1
                        kernel_x = kernel_x + 1
                        overhead_counter = 0
                        if(kernel_x == layer_params.kernel_size[0]):
                            kernel_x = 0
                            channel = channel + 1
            if (words_in_storage == math.ceil(layer_params.used_wght_per_PE/self.params.PARALLEL_MACS)):
                break
        return spad_storage
        
    def write_wght_addr_storage(self, cl_x, cl_y, router, data_spad):

        layer_params = self.layer_params
        params = self.params

        spad_storage = [0 for _ in range(self.params.Wghts_Addr_per_PE)]

        for words_in_storage in range(math.ceil(params.Wghts_Addr_per_PE)):
            if(words_in_storage != (self.layer_params.used_wght_addr_per_PE - 1)):
                spad_storage[words_in_storage] = \
                    int(words_in_storage * math.ceil(layer_params.used_wght_per_PE/layer_params.kernel_size[0]/self.params.PARALLEL_MACS/ int(layer_params.input_shape[3]/layer_params.iact_transmissions_pe)))
            else:
                break
        return spad_storage
