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

import math
import logging
import open_eye.generic_test_utils as gtu

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

    def _build_transmission_storage(self):
        """Populate 3D PE SPAD storage for the current layer_repetition."""
        clusters_x = self.params.Clusters_X
        clusters_y = self.params.Clusters_Y
        routers = self.params.Wght_Routers
        computing_mx = self.layer_params.computing_mx
        use_sparse = self.sparse_data == 1

        storage = [[[[] for _ in range(routers)] for _ in range(clusters_y)] for _ in range(clusters_x)]

        for cl_x in range(clusters_x):
            comp_x = computing_mx[cl_x]
            for cl_y in range(clusters_y):
                comp_xy = comp_x[cl_y]
                for router in range(routers):
                    if comp_xy[router][0] == 1:
                        spad = self.write_wght_pe(cl_x, cl_y, router)
                        storage[cl_x][cl_y][router] = (
                            self.set_sparse_stream(spad) if use_sparse else spad
                        )
        return storage

    def _finalize_stream(self, raw_stream):
        """Apply bitwidth chunking and RAM word-order reversal to a flat stream."""
        chunk = self.params.Clusters * self.params.NUM_GLB_WGHT
        transformed = gtu.transform_n_to_m_chunked(
            raw_stream,
            self.params.WGHT_Trans_Bitwidth,
            self.params.DMA_BITWIDTH,
            chunk,
        )
        n = self.layer_params.wght_cycles_one_word_all_ram
        if n <= 1:
            return transformed

        result = []
        for i in range(0, len(transformed), n):
            result.extend(transformed[i : i + n][::-1])
        return result

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
        if self.params.SERIAL:
            # === SERIAL MODE: Multiple transmissions over time ===
            wght_stream = []
            for layer_repetition_loop in range(self.layer_params.needed_wght_transmissions):
                self.layer_repetition = layer_repetition_loop
                temp_storage = self._build_transmission_storage()
                wght_stream.extend(self.create_complete_wght_stream(temp_storage))
        else:
            # === PARALLEL MODE: Single transmission with all data ===
            storage = self._build_transmission_storage()
            wght_stream = self.create_complete_wght_stream(storage)

        return self._finalize_stream(wght_stream)

    def set_sparse_stream(self, spad_data):
        """Convert dense SPAD data to sparse format with offset encoding.

        This method transforms weight data from dense representation to a sparse format
        that compresses zero weights using offset encoding. Each non-zero weight is stored
        with an offset field indicating how many zeros preceded it.

        Sparse Format: [weight_value, skip_offset]
        - weight_value: The actual non-zero weight
        - skip_offset: Number of zeros skipped before this weight

        Args:
            spad_data (list): Dense SPAD data as [addr_spad, data_spad]
                - addr_spad: Address pointers for data regions
                - data_spad: Weight values in dense format

        Returns:
            list: Sparse SPAD data as [new_addr_spad, new_data_spad]
                - new_addr_spad: Updated address pointers for sparse data
                - new_data_spad: Sparse-encoded weights with offset information

        """
        parallel_macs = self.params.PARALLEL_MACS
        src_addr, data_spad = spad_data[0], spad_data[1]

        addr_spad = [0]
        temp_word = []
        temp_complete_list = []
        temp_offset = 0

        for addr_spad_count in range(self.params.Wghts_Addr_per_PE - 1):
            region_end = src_addr[addr_spad_count + 1]
            if region_end == 0:
                continue

            word_count = addr_spad[addr_spad_count]
            for data_spad_pos in range(src_addr[addr_spad_count], region_end):
                row = data_spad[data_spad_pos]
                for x in range(parallel_macs):
                    val = row[x][0]
                    if val != 0:
                        temp_word.append([val, temp_offset])
                        temp_offset = 0
                        if len(temp_word) == parallel_macs:
                            temp_complete_list.append(temp_word)
                            word_count += 1
                            temp_word = []
                    else:
                        temp_offset += 1

            addr_spad.append(word_count)
            temp_offset = 0

        return [addr_spad, temp_complete_list]

    def write_wght_pe(self, cl_x, cl_y, router):
        """Generate complete SPAD contents for a single PE.

        Args:
            cl_x (int): Cluster X coordinate
            cl_y (int): Cluster Y coordinate
            router (int): Router/PE ID within the cluster

        Returns:
            list: Weight data SPAD contents

        """
        return self.write_wght_data_storage(cl_x, cl_y, router)

    def write_wght_data_storage(self, cl_x, cl_y, router):
        """Populate weight data SPAD from DRAM for a specific PE.

        This method extracts the relevant weights from DRAM and organizes them into
        the PE's weight data scratchpad. It handles the mapping of multi-dimensional
        weight tensors (channels, filters, kernel_y, kernel_x) to the linear SPAD
        storage format.

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

        parallel_macs = params.PARALLEL_MACS
        total_words = int(params.Wghts_per_PE / parallel_macs)
        max_used_words = math.ceil(layer_params.used_wght_per_PE / parallel_macs)

        spad_storage = [[[0, 0] for _ in range(parallel_macs)] for _ in range(total_words)]
        overhead_counter = 0
        kernel_x = 0

        in_channels = layer_params.input_shape[3]
        iact_trans = layer_params.iact_transmissions_pe
        rep_mod = layer_repetition % iact_trans
        channels_per_trans_ceil = math.ceil(in_channels / iact_trans)
        channels_per_trans_floor = int(in_channels / iact_trans)

        channel = rep_mod * channels_per_trans_ceil
        channel_limit = channels_per_trans_floor + rep_mod * channels_per_trans_ceil

        filters_per_calculation = math.ceil(layer_params.used_wght_per_PE / layer_params.used_iact_per_PE)
        start_current_repetition = int(
            (math.floor(layer_repetition / iact_trans) % layer_params.needed_wght_transmissions)
            * filters_per_calculation
        )
        end_filter = start_current_repetition + filters_per_calculation
        filters = start_current_repetition

        kernel_w = layer_params.kernel_size[0]
        kernel_row = (cl_y % (layer_params.used_Y_cluster * params.PEs_Y)) * params.PEs_Y + router
        valid_kernel_row = kernel_row < (layer_params.kernel_size[1] * channels_per_trans_floor)

        for words_in_storage in range(total_words):
            if valid_kernel_row:
                word_slot = spad_storage[words_in_storage]
                for spad_val_number in range(parallel_macs):
                    if channel != channel_limit:
                        slot = word_slot[spad_val_number]
                        slot[0] = dram[channel][filters][kernel_row][kernel_x]
                        slot[1] = overhead_counter
                        overhead_counter = 0

                        filters += 1
                        if filters == end_filter:
                            filters = start_current_repetition
                            kernel_x += 1

                        if kernel_x == kernel_w:
                            kernel_x = 0
                            channel += 1

            if words_in_storage == max_used_words:
                break

        return spad_storage

    def write_wght_addr_storage(self, cl_x, cl_y, router, data_spad):
        """Generate address SPAD pointers for weight data access.

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

        spad_storage = [0] * params.Wghts_Addr_per_PE
        last_idx = layer_params.used_wght_addr_per_PE - 1
        stride = math.ceil(
            layer_params.used_wght_per_PE
            / layer_params.kernel_size[0]
            / params.PARALLEL_MACS
            / int(layer_params.input_shape[3] / layer_params.iact_transmissions_pe)
        )

        for words_in_storage in range(params.Wghts_Addr_per_PE):
            if words_in_storage != last_idx:
                spad_storage[words_in_storage] = int(words_in_storage * stride)
            else:
                break

        return spad_storage

    def create_complete_wght_stream(self, spad_storage):
        """Convert SPAD storage to formatted weight bitstream for all PEs.

        Args:
            spad_storage (list): 3D array of SPAD contents [cluster_x][cluster_y][router]

        Returns:
            list: Formatted weight bitstream
                - Serial mode: Flat list sequenced by transmission time
                - Parallel mode: Nested structure [cluster_x][cluster_y][router][words]

        """
        params = self.params
        clusters_x = params.Clusters_X
        clusters_y = params.Clusters_Y
        wght_routers = params.Wght_Routers

        stream = [
            [
                [
                    self.create_pe_data_wght_stream(spad_storage[cl_x][cl_y][router])
                    for router in range(wght_routers)
                ]
                for cl_y in range(clusters_y)
            ]
            for cl_x in range(clusters_x)
        ]

        if params.SERIAL:
            # === SERIAL MODE: Time-multiplex across clusters ===
            num_words = len(stream[0][0][0])
            num_glb_wght = params.NUM_GLB_WGHT
            flat_stream = []

            for word in range(num_words):
                for cl_x in range(clusters_x):
                    stream_x = stream[cl_x]
                    for cl_y in range(clusters_y):
                        stream_xy = stream_x[cl_y]
                        for router in range(num_glb_wght):
                            pe_words = stream_xy[router]
                            flat_stream.append(pe_words[word] if word < len(pe_words) else 0)
            return flat_stream

        return stream

    def create_pe_addr_wght_stream(self, spad):
        """Pack address SPAD into transmission-ready bitstream for a single PE.

        Args:
            spad (list): Complete SPAD data as [addr_spad, data_spad]

        Returns:
            list: Address bitstream as packed integers ready for hardware transmission

        """
        layer_params = self.layer_params
        params = self.params

        addr_bitwidth = params.WGHT_Addr_Bitwidth
        addr_per_trans = math.floor(params.WGHT_Trans_Bitwidth / addr_bitwidth)
        used_addr = layer_params.used_wght_addr_per_PE
        addr_list = spad[0]
        addr_len = len(addr_list)
        stream = []

        for spad_addr_trans in range(math.ceil(params.Wghts_Addr_per_PE / addr_per_trans)):
            temp_trans = 0
            base_word = spad_addr_trans * addr_per_trans

            for addr_in_trans in range(addr_per_trans):
                spad_word = base_word + addr_in_trans
                if spad_word < addr_len:
                    temp_trans |= addr_list[spad_word] << (addr_bitwidth * addr_in_trans)

            # Replicate the packed word for each address it contains
            remaining = used_addr - len(stream)
            if remaining <= 0:
                break
            repeat = min(addr_per_trans, remaining)
            stream.extend([temp_trans] * repeat)
            if len(stream) >= used_addr:
                break

        return stream

    def create_pe_data_wght_stream(self, spad):
        """Pack weight data SPAD into transmission-ready bitstream for a single PE.

        Weight Word Format: [overhead_bits][value_bits]
        - value_bits: Actual weight value in two's complement
        - overhead_bits: Sparsity offset or other metadata

        Args:
            spad (list): Weight data SPAD for a single PE

        Returns:
            list: Weight data bitstream as packed integers

        """
        layer_params = self.layer_params
        params = self.params

        wght_bw = params.WGHT_Bitwidth
        woh_bw = params.WGHT_WOH_Bitwidth
        overhead_bw = woh_bw - wght_bw

        val_mask = (1 << wght_bw) - 1
        ovh_mask = (1 << overhead_bw) - 1

        data_per_trans = math.floor(params.WGHT_Trans_Bitwidth / woh_bw)
        total_trans = math.floor(params.Wghts_per_PE / data_per_trans)
        max_lines = math.ceil(layer_params.used_wght_per_PE / params.PARALLEL_MACS)
        keep_zeros = (not self.SKIP_ZERO_WORDS) or (layer_params.needed_wght_transmissions > 1)

        spad_len = len(spad)
        line_counter = 0
        stream = []

        for spad_data_trans in range(total_trans):
            temp_trans = 0

            if spad_data_trans < spad_len:
                row = spad[spad_data_trans]
                row_len = len(row)
                for data_in_trans in range(data_per_trans):
                    if data_in_trans < row_len:
                        entry = row[data_in_trans]
                        # Fast bitwise two's complement equivalent to gtu.to_twos_complement
                        val_bits = int(entry[0]) & val_mask
                        ovh_bits = int(entry[1]) & ovh_mask
                        packed = (ovh_bits << wght_bw) | val_bits
                        temp_trans |= packed << (data_in_trans * woh_bw)

            if temp_trans != 0 or keep_zeros:
                stream.append(temp_trans)

            line_counter += 1
            if line_counter == max_lines:
                break

        return stream


class ConvWghtStreamMapper(WghtStreamMapper):
    """Weight stream mapper specialized for standard 2D convolution layers.

    Key differences from base class:
    - Supports multi-kernel-per-PE-cluster configurations
    - Handles channel offset calculations for different clusters
    - Accounts for single-cluster vs. multi-cluster computation modes

    Attributes:
        Inherited from WghtStreamMapper

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_data):
        """Initialize the Conv2D weight stream mapper."""
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, sparse_data)

    def write_wght_data_storage(self, cl_x, cl_y, router):
        """Populate Conv2D weight data SPAD from DRAM for a specific PE.

        Conv2D weights are 4D: [input_channels][output_filters][kernel_h][kernel_w]

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

        parallel_macs = params.PARALLEL_MACS
        total_words = int(params.Wghts_per_PE / parallel_macs)
        spad_storage = [[[0, 0] for _ in range(parallel_macs)] for _ in range(total_words)]
        overhead_counter = 0
        kernel_x = 0

        in_channels = layer_params.input_shape[3]
        iact_trans = layer_params.iact_transmissions_pe
        kernels_per_cluster = layer_params.kernel_per_pe_cluster
        rep_mod = layer_repetition % iact_trans
        channels_per_trans_ceil = math.ceil(in_channels / iact_trans)
        channels_per_trans_floor = int(in_channels / iact_trans)

        # Calculate starting channel index for this transmission and router
        channel = rep_mod * channels_per_trans_ceil + math.ceil(
            math.ceil((router % kernels_per_cluster) * in_channels / iact_trans / kernels_per_cluster)
        )
        channel_offset = channel
        channel_limit = 1 + channels_per_trans_floor + rep_mod * channels_per_trans_ceil
        used_channels_end = layer_params.used_channels + channel_offset

        # Calculate filter assignment accounting for different kernel computations per PE
        diff_kernels = layer_params.different_kernels_per_calculation
        filters_per_calculation = (
            math.ceil(layer_params.used_wght_per_PE / layer_params.used_iact_per_PE) * diff_kernels
        )

        # Determine starting filter for this repetition and cluster allocation
        start_current_repetition = int(
            (math.floor(layer_repetition / iact_trans) % layer_params.needed_wght_transmissions)
            * filters_per_calculation
        )
        amount_of_used_clusters = math.ceil(
            (layer_params.used_Y_cluster * math.ceil(layer_params.iact_x_add_up / layer_params.strideX))
            / params.PEs_X
        )
        channel_offset_in_calculation = (params.Clusters_X * cl_y + cl_x) // amount_of_used_clusters
        start_current_repetition += channel_offset_in_calculation
        end_filter = start_current_repetition + filters_per_calculation

        amount_of_words = math.ceil(layer_params.used_wght_per_PE / parallel_macs)
        kernel_row = (cl_y % layer_params.used_Y_cluster) * params.PEs_Y + (router % layer_params.kernel_size[0])

        filters = start_current_repetition
        spad_position = 0

        for words_in_storage in range(amount_of_words):
            for _ in range(parallel_macs):
                if channel != channel_limit:
                    w_idx = spad_position // parallel_macs
                    m_idx = spad_position % parallel_macs
                    try:
                        val = dram[channel][filters][kernel_x][kernel_row]
                        spad_storage[w_idx][m_idx][0] = val
                        if val == 0:
                            overhead_counter += 1
                        else:
                            spad_storage[w_idx][m_idx][1] = overhead_counter
                            spad_position += 1
                            overhead_counter = 0
                    except IndexError:
                        spad_storage[w_idx][m_idx][0] = 0
                        spad_position += 1

                    # Advance to next filter (with stride for different kernels)
                    filters += diff_kernels

                    # Wrap to next channel when filter range exhausted
                    if filters == end_filter:
                        filters = start_current_repetition
                        channel += 1
                        spad_position += spad_position % parallel_macs

                    # Wrap to next kernel X when channels exhausted
                    if channel == used_channels_end:
                        channel = channel_offset
                        kernel_x += 1

            if words_in_storage == amount_of_words:
                break

        return spad_storage

    def write_wght_addr_storage(self, cl_x, cl_y, router, data_spad):
        """Generate address SPAD pointers for Conv2D weight data access.

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
        total_addr = math.ceil(params.Wghts_Addr_per_PE)
        spad_storage = [0] * params.Wghts_Addr_per_PE

        used_addr = layer_params.used_wght_addr_per_PE
        all_used = used_addr == params.Wghts_Addr_per_PE
        last_idx = used_addr - 1
        stride = math.ceil(layer_params.used_wght_per_PE / 2 / int(layer_params.used_iact_per_PE))

        for words_in_storage in range(total_addr):
            if (words_in_storage != last_idx) or all_used:
                spad_storage[words_in_storage] = int(words_in_storage * stride)
            else:
                break

        return spad_storage


class DenseWghtStreamMapper(WghtStreamMapper):
    """Weight stream mapper specialized for fully-connected (Dense) layers.

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
        """Initialize the Dense layer weight stream mapper."""
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, sparse_data)

    def get_wght_stream(self):
        """Generate complete weight bitstream for Dense layer across all PEs.

        Returns:
            list: Complete weight bitstream including both address and data streams

        """
        wght_stream = []
        # Always iterate through all weight transmissions for Dense layers
        for layer_repetition_loop in range(self.layer_params.needed_wght_transmissions):
            self.layer_repetition = layer_repetition_loop
            temp_storage = self._build_transmission_storage()
            wght_stream.extend(self.create_complete_wght_stream(temp_storage))
        return wght_stream

    def create_complete_wght_stream(self, spad_storage):
        """Convert Dense layer SPAD storage to formatted weight bitstream.

        Args:
            spad_storage (list): 3D array of SPAD contents [cluster_x][cluster_y][router]

        Returns:
            list: Formatted weight bitstream (address + data combined)

        """
        params = self.params
        clusters_x = params.Clusters_X
        clusters_y = params.Clusters_Y
        wght_routers = params.Wght_Routers

        wght_stream = [
            [
                [
                    self.create_pe_data_wght_stream(spad_storage[cl_x][cl_y][router])
                    for router in range(wght_routers)
                ]
                for cl_y in range(clusters_y)
            ]
            for cl_x in range(clusters_x)
        ]

        if params.SERIAL:
            # === SERIAL MODE: Time-multiplex across clusters ===
            num_words = len(wght_stream[0][0][0])
            num_glb_wght = params.NUM_GLB_WGHT
            shift = params.WGHT_Trans_Bitwidth
            flat_stream = []

            for word in range(num_words):
                for cl_x in range(clusters_x):
                    stream_x = wght_stream[cl_x]
                    for cl_y in range(clusters_y):
                        stream_xy = stream_x[cl_y]
                        for router in range(num_glb_wght):
                            pe_words = stream_xy[router]
                            if word < len(pe_words):
                                val = pe_words[word]
                                # NOTE: both terms index cl_x, so this doubles one
                                # column rather than combining columns 0 and 1 as
                                # the iact/psum mappers do. Emitting a single
                                # value here instead halves the stream and stalls
                                # GET_WGHT, so the word count depends on this
                                # doubling - fixing the value bug needs
                                # trans_cycles_wght corrected in the same change.
                                flat_stream.append(val | (val << shift))
                            else:
                                flat_stream.append(0)
            wght_stream = flat_stream

        return self._finalize_stream(wght_stream)

    def write_wght_data_storage(self, cl_x, cl_y, router):
        """Populate Dense layer weight data SPAD from DRAM for a specific PE.

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

        parallel_macs = params.PARALLEL_MACS
        total_words = int(params.Wghts_per_PE / parallel_macs)
        max_used_words = math.ceil(layer_params.used_wght_per_PE / parallel_macs)
        spad_storage = [[[0, 0] for _ in range(parallel_macs)] for _ in range(total_words)]

        iact_words_per_memory = 2
        used_psum = layer_params.used_psum_per_PE
        used_iact = layer_params.used_iact_per_PE
        num_glb_wght = params.NUM_GLB_WGHT
        needed_wghts_in_word = math.ceil(used_psum / parallel_macs) * parallel_macs

        # Loop-invariant filter and channel base offsets
        filter_base = (
            cl_x * used_psum
            + (math.floor(layer_repetition / layer_params.diff_iact_layer) % layer_params.needed_psum_transmissions)
            * params.Clusters_X
            * used_psum
        )
        channel_base = (
            math.floor((layer_repetition % layer_params.diff_iact_layer) * (num_glb_wght * params.Clusters_Y * used_iact))
            + (cl_y * num_glb_wght * used_iact)
            + router * iact_words_per_memory
        )

        channel_word_stride = iact_words_per_memory * num_glb_wght
        block_size = needed_wghts_in_word * iact_words_per_memory
        dram_rows = len(dram)

        for words_in_storage in range(total_words):
            word_slot = spad_storage[words_in_storage]
            base_pos = words_in_storage * parallel_macs

            for spad_val_number in range(parallel_macs):
                position = base_pos + spad_val_number
                pos_mod = position % needed_wghts_in_word

                if pos_mod < used_psum:
                    filters = int(pos_mod + filter_base)
                    channel = int(
                        channel_base
                        + (math.floor(position / needed_wghts_in_word) % iact_words_per_memory)
                        + (math.floor(position / block_size) * channel_word_stride)
                    )
                    if 0 <= filters < dram_rows:
                        dram_row = dram[filters]
                        if 0 <= channel < len(dram_row):
                            word_slot[spad_val_number][0] = dram_row[channel]

            if words_in_storage == max_used_words:
                break

        return spad_storage

    def write_wght_addr_storage(self, cl_x, cl_y, router, data_spad):
        """Generate address SPAD pointers for Dense layer weight data access.

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

        total_addr = math.ceil(params.Wghts_Addr_per_PE)
        spad_storage = [0] * params.Wghts_Addr_per_PE
        last_idx = layer_params.used_wght_addr_per_PE - 1
        stride = math.ceil(layer_params.used_psum_per_PE / params.PARALLEL_MACS)

        for words_in_storage in range(total_addr):
            if words_in_storage != last_idx:
                spad_storage[words_in_storage] = int(words_in_storage * stride)
            else:
                break

        return spad_storage


class DwWghtStreamMapper(WghtStreamMapper):
    """Weight stream mapper specialized for depthwise 2D convolution layers.

    In depthwise convolutions, each input channel is convolved with its own
    dedicated filter rather than across all input-output channel pairs.

    Attributes:
        Inherited from WghtStreamMapper

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_data):
        """Initialize the Depthwise Conv2D weight stream mapper."""
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, sparse_data)

    def write_wght_data_storage(self, cl_x, cl_y, router):
        """Populate Depthwise Conv2D weight data SPAD from DRAM for a specific PE.

        Depthwise weights are 3D: [channels][kernel_h][kernel_w]

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

        parallel_macs = params.PARALLEL_MACS
        total_words = int(params.Wghts_per_PE / parallel_macs)
        spad_storage = [[[0, 0] for _ in range(parallel_macs)] for _ in range(total_words)]
        overhead_counter = 0
        kernel_x = 0

        in_channels = layer_params.input_shape[3]
        iact_trans = layer_params.iact_transmissions_pe
        kernels_per_cluster = layer_params.kernel_per_pe_cluster
        rep_mod = layer_repetition % iact_trans
        channels_per_trans_ceil = math.ceil(in_channels / iact_trans)
        channels_per_trans_floor = int(in_channels / iact_trans)

        # Calculate starting channel index for this transmission and router
        channel = rep_mod * channels_per_trans_ceil + math.ceil(
            math.ceil((router % kernels_per_cluster) * in_channels / iact_trans / kernels_per_cluster)
        )
        channel_offset = channel
        channel_limit = 1 + channels_per_trans_floor + rep_mod * channels_per_trans_ceil
        used_channels_end = layer_params.used_channels + channel_offset

        amount_of_words = math.ceil(layer_params.used_wght_per_PE / parallel_macs)
        kernel_row = (cl_y % layer_params.used_Y_cluster) * params.PEs_Y + (router % layer_params.kernel_size[0])
        spad_position = 0

        for words_in_storage in range(amount_of_words):
            for _ in range(parallel_macs):
                if channel != channel_limit:
                    w_idx = spad_position // parallel_macs
                    m_idx = spad_position % parallel_macs
                    try:
                        val = dram[channel][kernel_x][kernel_row]
                        spad_storage[w_idx][m_idx][0] = val
                        if val == 0:
                            overhead_counter += 1
                        else:
                            spad_storage[w_idx][m_idx][1] = overhead_counter
                            spad_position += 1
                            overhead_counter = 0
                    except IndexError:
                        spad_storage[w_idx][m_idx][0] = 0
                        spad_position += 1

                    channel += 1
                    spad_position += spad_position % parallel_macs

                    # Wrap to next kernel X when channels exhausted
                    if channel == used_channels_end:
                        channel = channel_offset
                        kernel_x += 1

            if words_in_storage == amount_of_words:
                break

        return spad_storage

    def write_wght_addr_storage(self, cl_x, cl_y, router, data_spad):
        """Generate address SPAD pointers for Depthwise Conv2D weight data access.

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
        total_addr = math.ceil(params.Wghts_Addr_per_PE)
        spad_storage = [0] * params.Wghts_Addr_per_PE

        used_addr = layer_params.used_wght_addr_per_PE
        all_used = used_addr == params.Wghts_Addr_per_PE
        last_idx = used_addr - 1
        stride = math.ceil(layer_params.used_wght_per_PE / 2 / int(layer_params.used_iact_per_PE))

        for words_in_storage in range(total_addr):
            if (words_in_storage != last_idx) or all_used:
                spad_storage[words_in_storage] = int(words_in_storage * stride)
            else:
                break

        return spad_storage