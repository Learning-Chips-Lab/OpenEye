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

import logging
import math
import numpy as np

logger = logging.getLogger("cocotb")


def pack_values_into_words(flat_values, params, layer_params):
    """Pack activation values into DMA-width words, padded to whole cell groups."""
    bitwidth = int(params.IACT_Bitwidth)
    dma_bw = int(params.DMA_BITWIDTH)
    values_per_word = dma_bw // bitwidth
    mask = (1 << bitwidth) - 1
    n = int(layer_params.iact_cycles_one_word_all_ram)

    arr = np.asarray(flat_values, dtype=np.int64).ravel()
    rem = arr.size % values_per_word
    if rem:
        arr = np.pad(arr, (0, values_per_word - rem), constant_values=0)

    if arr.size == 0:
        words = []
    elif dma_bw <= 64:
        grouped = arr.reshape(-1, values_per_word).astype(np.uint64) & np.uint64(mask)
        shifts = np.arange(values_per_word, dtype=np.uint64) * np.uint64(bitwidth)
        words = np.bitwise_or.reduce(grouped << shifts, axis=1).tolist()
    else:
        grouped = arr.reshape(-1, values_per_word) & mask
        words = []
        for row in grouped.tolist():
            word = 0
            for j, val in enumerate(row):
                word |= val << (j * bitwidth)
            words.append(word)

    pad_len = (n - len(words)) % n
    if pad_len:
        words.extend([0] * pad_len)
    return words


def buffer_words_to_dma_stream(buffer_words, layer_params):
    """Reorder a BUFFER_A image into the order the DMA has to send it in.

    GET_IACT shifts each incoming word through a register and commits all
    cells at once, so a group of iact_cycles_one_word_all_ram words has to
    arrive reversed for cell j to end up holding word j of that group.
    """
    n = int(layer_params.iact_cycles_one_word_all_ram)
    padded = list(buffer_words)
    pad_len = (n - len(padded)) % n
    if pad_len:
        padded.extend([0] * pad_len)

    if n <= 1:
        return padded

    stream = []
    for i in range(0, len(padded), n):
        stream.extend(padded[i : i + n][::-1])
    return stream


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

    def _build_parallel_stream(self):
        """Generate and pack per-cluster activation streams for parallel mode."""
        clusters_x = self.params.Clusters_X
        clusters_y = self.params.Clusters_Y
        num_glb_iact = self.params.NUM_GLB_IACT

        iact_stream = [
            [
                [self.write_iact_data_glb(cl_x, cl_y, router) for router in range(num_glb_iact)]
                for cl_y in range(clusters_y)
            ]
            for cl_x in range(clusters_x)
        ]
        return self.create_complete_iact_stream(iact_stream)

    def get_iact_stream(self):
        """Generate the complete input activation data stream for the layer.

        This method orchestrates the creation of activation streams for all clusters
        and Global Buffers. In parallel mode, it creates separate streams for each
        cluster and router. In serial mode, it creates a single DMA-compatible stream.

        Returns:
            Activation stream in one of two formats:
- Parallel mode: 3D list [cluster_x][cluster_y][router] of bit-packed words
- Serial mode: 1D list of DMA words, or empty list if skipIact is set

        """
        if not self.params.SERIAL:
            return self._build_parallel_stream()

        if self.layer_params.skipIact == 0:
            return buffer_words_to_dma_stream(
                pack_values_into_words(self.build_iact_buffer_words(), self.params, self.layer_params),
                self.layer_params,
            )
        return []

    def build_iact_buffer_words(self):
        """Raw 64-bit iact words in BUFFER_A cell order.

        Word k belongs in RAM cell ``k % iact_cycles_one_word_all_ram`` at
        address ``k // iact_cycles_one_word_all_ram``. This is the image the
        buffer holds once a layer's activations are loaded, whether they came
        over DMA or from the psum write-back path, which makes it the single
        definition of the layout for both the stream builder and the
        testbench check in rtl_test_utils.compare_iact_storage.

        get_iact_stream turns this into the DMA order by reversing each group
        of iact_cycles_one_word_all_ram words, because the hardware shifts
        incoming words through a register and commits all cells at once.

        Returns:
            list[int]: activation values ordered for packing.
        """
        # Transpose from [C][X][Y] to [Y][X][C] for row-major ordering
        values = np.transpose(np.asarray(self.dram_fmap), axes=(2, 1, 0))
        iact_size_y, iact_size_x, channels = values.shape
        group_size = int(self.layer_params.cont_iact_channels_glb)

        if channels % group_size == 0:
            # Fast vectorized grouping: [Y][X][num_groups][group_size] -> [num_groups][Y][X][group_size]
            num_groups = channels // group_size
            grouped = values.reshape(iact_size_y, iact_size_x, num_groups, group_size)
            return np.transpose(grouped, (2, 0, 1, 3)).ravel().tolist()

        # Fallback when channel count is not an exact multiple of cont_iact_channels_glb
        chunks = [
            values[:, :, c_base : c_base + group_size].ravel()
            for c_base in range(0, channels, group_size)
        ]
        return np.concatenate(chunks).tolist() if chunks else []

    def write_iact_data_glb(self, cl_x, cl_y, router):
        """Generate activation data for a specific Global Buffer (GLB) router.

        Args:
            cl_x (int): X-coordinate of the cluster
            cl_y (int): Y-coordinate of the cluster
            router (int): GLB router index within the cluster

        Returns:
            list: Sequence of SPAD data across active refresh and write cycles.

        """
        storage = []
        refresh_bounds = self.layer_params.needed_refreshes_mx[self.layer_repetition]
        needed_writes = int(self.layer_params.needed_Iact_writes)
        used_y_cluster = int(self.layer_params.used_Y_cluster)
        use_sparse = self.sparse_data == 1

        for cycle in range(int(refresh_bounds[1]), int(refresh_bounds[2])):
            if (cl_y - cycle) % used_y_cluster == 0:
                for iact_cycle in range(needed_writes):
                    spad = self.write_iact_pe(cl_x, cl_y, router, cycle, iact_cycle)
                    storage.append(self.set_sparse_stream(spad) if use_sparse else spad)
        return storage

    def set_sparse_stream(self, spad):
        """Convert dense SPAD data to sparse format by filtering zeros and compacting addresses.

        Args:
            spad (list): Dense SPAD data in format [[addresses], [[value, overhead], ...]]

        Returns:
            list: Sparse SPAD data in format [[filtered_addresses], [[non_zero_value, overhead], ...]]

        """
        last_used_idx = math.ceil(self.layer_params.used_iact_addr_per_PE) - 1
        trunc_addr = [
            addr for idx, addr in enumerate(spad[0]) if (idx < last_used_idx) or (addr != 0)
        ]

        filtered_data = [entry for entry in spad[1] if entry[0] != 0]

        addr_counter = 0
        addr_index = 0
        filtered_addr = []

        for data in filtered_data:
            if data[1] < trunc_addr[addr_index]:
                addr_counter += 1
            else:
                filtered_addr.append(addr_counter)
                addr_index += 1
                if data[1] < trunc_addr[addr_index]:
                    addr_counter += 1

        filtered_addr.append(addr_counter)
        return [filtered_addr, filtered_data]

    def write_iact_pe(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate complete SPAD data for a PE (Processing Element)."""
        return self.write_iact_data_storage(cl_x, cl_y, router, cycle, iact_cycle)

    def _compute_spatial_indices(self, cl_x, cl_y, router, cycle, iact_cycle, case_2_uses_clusters_x=False):
        """Compute loop-invariant X/Y base coordinates and channel offset for Conv/DW mappings."""
        layer_params = self.layer_params
        params = self.params

        stride_x = layer_params.strideX
        stride_y = layer_params.strideY
        pes_x = params.PEs_X
        clusters_x = params.Clusters_X
        used_y_cluster = layer_params.used_Y_cluster
        kernel_w, kernel_h = int(layer_params.kernel_size[0]), int(layer_params.kernel_size[1])

        out_span_x = (layer_params.output_shape[2] + layer_params.add_up) * stride_x
        pad_offset = math.ceil((kernel_h - 1) / 2)
        router_iact_offset = (
            math.floor(router / layer_params.kernel_per_pe_cluster)
            + iact_cycle * math.floor(params.NUM_GLB_IACT / layer_params.kernel_per_pe_cluster)
            - pad_offset
        )

        match layer_params.single_cluster_computation:
            case 1:
                cycle_step = (cycle * pes_x) * stride_x
                iact_temp_pos_x = int(math.floor(cycle_step) % out_span_x) + router_iact_offset
                base_y = int(stride_y * math.floor(cycle_step / out_span_x)) - pad_offset
            case 2:
                x_factor = clusters_x if case_2_uses_clusters_x else math.floor(params.Clusters / used_y_cluster)
                cycle_step = (cycle * pes_x * x_factor) * stride_x
                iact_temp_pos_x = (
                    int(math.floor((cl_x * pes_x) * stride_x + cycle_step) % out_span_x) + router_iact_offset
                )
                base_y = int(stride_y * math.floor(cycle_step / out_span_x)) - pad_offset
            case _:
                cl_y_offset = math.floor(cl_y / used_y_cluster) * pes_x * clusters_x
                cycle_step = cycle * pes_x * math.floor(params.Clusters / used_y_cluster)
                iact_temp_pos_x = (
                    int(math.floor(((cl_x * pes_x) + cl_y_offset + cycle_step) * stride_x) % out_span_x)
                    + router_iact_offset
                )
                y_num = (cl_y_offset + ((cl_x * pes_x) if case_2_uses_clusters_x else 0) + cycle_step) * stride_x
                base_y = int(stride_y * math.floor(y_num / out_span_x)) - pad_offset

        channel_base = (
            (self.layer_repetition % layer_params.iact_transmissions_pe)
            - math.ceil(layer_params.input_shape[3] / layer_params.iact_transmissions_pe)
        )
        return iact_temp_pos_x, base_y, int(channel_base), kernel_w, kernel_h

    def write_iact_data_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate activation data SPAD storage for a specific PE location and cycle."""
        layer_params = self.layer_params
        params = self.params

        spad_storage = [[0, 0] for _ in range(params.Iacts_per_PE)]
        max_words = min(math.ceil(params.Iacts_per_PE), int(layer_params.used_iact_per_PE))
        if max_words <= 0:
            return spad_storage

        used_y_cluster = layer_params.used_Y_cluster
        pes_x = params.PEs_X
        clusters_x = params.Clusters_X
        in_active_region = (
            (math.floor(cycle / used_y_cluster) * params.Clusters_Y * clusters_x * pes_x)
            + (math.floor(cl_y / used_y_cluster) * clusters_x * pes_x)
            + (cl_x * pes_x)
        ) < (layer_params.output_shape[1] * (layer_params.output_shape[2] + layer_params.add_up))

        if not in_active_region:
            return spad_storage

        stride_x = layer_params.strideX
        stride_y = layer_params.strideY
        kernel_w, kernel_h = int(layer_params.kernel_size[0]), int(layer_params.kernel_size[1])
        out_span_x = (layer_params.output_shape[2] + layer_params.add_up) * stride_x
        pad_offset = math.ceil((kernel_h - 1) / 2)
        router_iact_offset = (
            router
            + iact_cycle * math.floor(params.NUM_GLB_IACT / layer_params.kernel_per_pe_cluster)
            - pad_offset
        )

        match layer_params.single_cluster_computation:
            case 1:
                cycle_step = cycle * pes_x * math.floor(params.Clusters / used_y_cluster)
                iact_temp_pos_x = int(math.floor(cycle_step * stride_x) % out_span_x) + router_iact_offset
                base_y = int(stride_y * math.floor((cycle * pes_x * stride_x) / out_span_x)) - pad_offset
            case 2:
                cycle_step = cycle * pes_x * math.floor(params.Clusters / used_y_cluster)
                iact_temp_pos_x = (
                    int(math.floor(((cl_x * pes_x) + cycle_step) * stride_x) % out_span_x) + router_iact_offset
                )
                base_y = int(stride_y * math.floor((cycle * pes_x * stride_x) / out_span_x)) - pad_offset
            case _:
                cl_y_offset = math.floor(cl_y / used_y_cluster) * pes_x * clusters_x
                cycle_step = cycle * pes_x * math.floor(params.Clusters / used_y_cluster)
                iact_temp_pos_x = (
                    int(math.floor(((cl_x * pes_x) + cl_y_offset + cycle_step) * stride_x) % out_span_x)
                    + router_iact_offset
                )
                base_y = int(stride_y * math.floor(((cl_y_offset + cycle_step) * stride_x) / out_span_x)) - pad_offset

        max_x = layer_params.output_shape[1] * stride_x
        max_y = layer_params.output_shape[2] * stride_y
        valid_x = 0 <= iact_temp_pos_x < max_x
        channel_base = int(
            (self.layer_repetition % layer_params.iact_transmissions_pe)
            * math.ceil(layer_params.input_shape[3] / layer_params.iact_transmissions_pe)
        )

        dram_fmap = self.dram_fmap
        is_ndarray = isinstance(dram_fmap, np.ndarray) and dram_fmap.ndim == 3

        for words_in_storage in range(max_words):
            iact_temp_pos_y = (words_in_storage % kernel_h) + base_y
            slot = spad_storage[words_in_storage]
            if valid_x and (0 <= iact_temp_pos_y < max_y):
                channel = (words_in_storage // kernel_w) + channel_base
                slot[0] = (
                    int(dram_fmap[channel, iact_temp_pos_x, iact_temp_pos_y])
                    if is_ndarray
                    else int(dram_fmap[channel][iact_temp_pos_x][iact_temp_pos_y])
                )
            else:
                slot[0] = 1
            slot[1] = words_in_storage

        return spad_storage

    def _write_spatial_iact_data_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Shared spatial activation SPAD storage builder for Conv2D and DWConv2D."""
        layer_params = self.layer_params
        params = self.params

        spad_storage = [[0, 0] for _ in range(params.Iacts_per_PE)]

        in_channels = layer_params.input_shape[3]
        iact_trans = layer_params.iact_transmissions_pe
        kernels_per_cluster = layer_params.kernel_per_pe_cluster
        kernel_w, kernel_h = int(layer_params.kernel_size[0]), int(layer_params.kernel_size[1])

        amount_of_channels = (
            math.ceil((router + 1) * in_channels / iact_trans / kernels_per_cluster)
            - math.ceil(router * in_channels / iact_trans / kernels_per_cluster)
        )
        max_words = min(amount_of_channels * kernel_h, int(layer_params.used_iact_per_PE))
        if max_words <= 0:
            return spad_storage

        used_y_cluster = layer_params.used_Y_cluster
        pes_x = params.PEs_X
        clusters_x = params.Clusters_X
        in_active_region = (
            (
                (math.floor(cycle / used_y_cluster) * params.Clusters_Y * clusters_x * pes_x)
                + (math.floor(cl_y / used_y_cluster) * clusters_x * pes_x)
                + (cl_x * pes_x)
            )
            < (layer_params.output_shape[1] * (layer_params.output_shape[2] + layer_params.add_up))
        ) or (layer_params.single_cluster_computation != 0)

        if not in_active_region:
            return spad_storage

        iact_temp_pos_x, base_y, rep_channel_base, kernel_w, kernel_h = self._compute_spatial_indices(
            cl_x, cl_y, router, cycle, iact_cycle, case_2_uses_clusters_x=True
        )
        router_channel_offset = math.ceil(
            ((router % kernels_per_cluster) * math.ceil(in_channels / iact_trans)) / kernels_per_cluster
        )
        channel_base = int(router_channel_offset + rep_channel_base)

        max_x = layer_params.output_shape[1] * layer_params.strideX
        max_y = layer_params.output_shape[2] * layer_params.strideY
        valid_x = 0 <= iact_temp_pos_x < max_x

        dram_fmap = self.dram_fmap
        if isinstance(dram_fmap, np.ndarray) and dram_fmap.ndim == 3:
            dram_channels, dim_x, dim_y = dram_fmap.shape
            in_x_bounds = 0 <= iact_temp_pos_x < dim_x
            for words_in_storage in range(max_words):
                iact_temp_pos_y = (words_in_storage % kernel_h) + base_y
                slot = spad_storage[words_in_storage]
                if valid_x and (0 <= iact_temp_pos_y < max_y):
                    channel = (words_in_storage // kernel_w) + channel_base
                    if in_x_bounds and (0 <= channel < dram_channels) and (0 <= iact_temp_pos_y < dim_y):
                        slot[0] = int(dram_fmap[channel, iact_temp_pos_x, iact_temp_pos_y])
                    else:
                        slot[0] = 0
                else:
                    slot[0] = 1
                slot[1] = words_in_storage
            return spad_storage

        dram_channels = len(dram_fmap)
        for words_in_storage in range(max_words):
            iact_temp_pos_y = (words_in_storage % kernel_h) + base_y
            slot = spad_storage[words_in_storage]
            if valid_x and (0 <= iact_temp_pos_y < max_y):
                channel = (words_in_storage // kernel_w) + channel_base
                if 0 <= channel < dram_channels:
                    dram_c = dram_fmap[channel]
                    if 0 <= iact_temp_pos_x < len(dram_c) and 0 <= iact_temp_pos_y < len(dram_c[iact_temp_pos_x]):
                        slot[0] = int(dram_c[iact_temp_pos_x][iact_temp_pos_y])
                    else:
                        slot[0] = 0
                else:
                    slot[0] = 0
            else:
                slot[0] = 1
            slot[1] = words_in_storage

        return spad_storage

    def _write_spatial_iact_addr_storage(self):
        """Shared spatial activation address SPAD builder for Conv2D and DWConv2D."""
        layer_params = self.layer_params
        params = self.params
        spad_storage = [0] * params.Iacts_Addr_per_PE
        kernel_w = int(layer_params.kernel_size[0])

        if kernel_w == 1:
            spad_storage[0] = int(layer_params.used_channels)
        else:
            limit = min(params.Iacts_Addr_per_PE, math.ceil(layer_params.used_iact_per_PE / kernel_w))
            for words_in_storage in range(limit):
                spad_storage[words_in_storage] = kernel_w * (words_in_storage + 1)

        return spad_storage

    def write_iact_addr_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate activation address SPAD storage for indirect addressing mode."""
        layer_params = self.layer_params
        params = self.params
        spad_storage = [0] * params.Iacts_Addr_per_PE
        kernel_w = int(layer_params.kernel_size[0])
        limit = min(params.Iacts_Addr_per_PE, math.ceil(layer_params.used_iact_per_PE / kernel_w))

        for words_in_storage in range(limit):
            spad_storage[words_in_storage] = kernel_w * (words_in_storage + 1)
        return spad_storage

    def create_complete_iact_stream(self, spad_storage):
        """Convert SPAD storage to final bit-packed activation streams."""
        params = self.params
        clusters_x = params.Clusters_X
        clusters_y = params.Clusters_Y
        num_glb_iact = params.NUM_GLB_IACT

        stream = [[[[] for _ in range(num_glb_iact)] for _ in range(clusters_y)] for _ in range(clusters_x)]

        for cl_x in range(clusters_x):
            spad_x = spad_storage[cl_x]
            stream_x = stream[cl_x]
            for cl_y in range(clusters_y):
                spad_xy = spad_x[cl_y]
                stream_xy = stream_x[cl_y]
                for router in range(num_glb_iact):
                    target = stream_xy[router]
                    for cycle_spad in spad_xy[router]:
                        target.extend(self.create_pe_data_iact_stream(cycle_spad))

        if params.SERIAL:
            flat_stream = []
            for cl_y in range(clusters_y):
                for router in range(num_glb_iact):
                    for w0, w1 in zip(stream[0][cl_y][router], stream[1][cl_y][router]):
                        flat_stream.append(int(w0) | (int(w1) << 24))
            return flat_stream

        return stream

    def create_pe_addr_iact_stream(self, spad):
        """Pack activation address pointers into transmission words."""
        layer_params = self.layer_params
        params = self.params

        addr_bw = int(params.IACT_Addr_Bitwidth)
        addr_per_trans = math.floor(params.IACT_Trans_Bitwidth / addr_bw)
        total_trans = math.ceil(params.Iacts_Addr_per_PE / addr_per_trans)
        max_lines = math.ceil(layer_params.used_iact_addr_per_PE / addr_per_trans)
        addr_list = spad[0]
        addr_len = len(addr_list)
        stream = []

        for spad_addr_trans in range(total_trans):
            temp_trans = 0
            base_word = spad_addr_trans * addr_per_trans
            for addr_in_trans in range(addr_per_trans):
                spad_word = base_word + addr_in_trans
                if spad_word < addr_len:
                    temp_trans |= int(addr_list[spad_word]) << (addr_bw * addr_in_trans)
            stream.append(temp_trans)
            if len(stream) == max_lines:
                return stream
        return stream

    def create_pe_data_iact_stream(self, spad):
        """Pack activation data with overhead bits into transmission words.

        Packed format per entry: [overhead(4 bits) | value(IACT_WOH_Bitwidth bits)].
        """
        layer_params = self.layer_params
        params = self.params

        woh_bw = int(params.IACT_WOH_Bitwidth)
        val_mask = (1 << woh_bw) - 1
        ovh_mask = 0xF

        data_per_trans = math.floor(params.IACT_Trans_Bitwidth / woh_bw)
        total_trans = math.floor(params.Iacts_per_PE / data_per_trans)
        max_lines = math.ceil(layer_params.used_iact_per_PE / 2)
        spad_len = len(spad)
        stream = []

        for spad_data_trans in range(total_trans):
            temp_trans = 0
            base_idx = spad_data_trans * data_per_trans
            for data_in_trans in range(data_per_trans):
                idx = base_idx + data_in_trans
                if idx < spad_len:
                    entry = spad[idx]
                    val_bits = int(entry[0]) & val_mask
                    ovh_bits = int(entry[1]) & ovh_mask
                    packed = (ovh_bits << woh_bw) | val_bits
                    temp_trans |= packed << (data_in_trans * woh_bw)
            stream.append(temp_trans)
            if len(stream) == max_lines:
                break
        return stream


class ConvIactStreamMapper(IactStreamMapper):
    """Specialized mapper for standard Conv2D layer input activations.

    Distributes input channels across routers using kernel_per_pe_cluster to
    enable parallel filter computation.

    Attributes:
        Inherited from IactStreamMapper

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_data):
        """Initialize Conv2D activation stream mapper."""
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, sparse_data)

    def write_iact_data_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate Conv2D-specific activation data SPAD storage."""
        return self._write_spatial_iact_data_storage(cl_x, cl_y, router, cycle, iact_cycle)

    def write_iact_addr_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate address SPAD pointers for Conv2D activation access."""
        return self._write_spatial_iact_addr_storage()


class DenseIactStreamMapper(IactStreamMapper):
    """Specialized mapper for fully connected (Dense) layer input activations.

    Key Differences from Conv:
- Input is 1D vector instead of 2D/3D feature map
- No spatial position calculation or padding
- Sequential distribution across routers without kernel windowing
- Only first Y-cluster (cl_y == 0) loads data (no Y-dimension parallelism)

    Attributes:
        Inherited from IactStreamMapper

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_data):
        """Initialize Dense layer activation stream mapper."""
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, sparse_data)

    def get_iact_stream(self):
        """Generate the complete input activation data stream for a Dense layer."""
        if not self.params.SERIAL:
            return self._build_parallel_stream()

        if self.layer_params.skipIact == 0:
            return buffer_words_to_dma_stream(
                pack_values_into_words(self.build_iact_buffer_words(), self.params, self.layer_params),
                self.layer_params,
            )
        return []

    def build_iact_buffer_words(self):
        """Raw activation values in BUFFER_A cell order for a Dense layer."""
        params = self.params
        layer_params = self.layer_params
        values_per_word = params.DMA_BITWIDTH // params.IACT_WOH_Bitwidth

        # iact_size_x is the flattened feature count K; input_shape may be 2D or 4D
        group_size = int(params.NUM_GLB_IACT * layer_params.used_iact_per_PE)
        total_vals = math.ceil(layer_params.iact_size_x / group_size) * group_size
        transmissions = math.ceil(total_vals / values_per_word)
        target_len = transmissions * values_per_word

        flat_dram = np.asarray(self.dram_fmap).ravel()
        if flat_dram.size >= target_len:
            return flat_dram[:target_len].tolist()
        return np.pad(flat_dram, (0, target_len - flat_dram.size), constant_values=0).tolist()

    def write_iact_data_glb(self, cl_x, cl_y, router):
        """Generate GLB activation data for Dense layers."""
        storage = []
        refresh_bounds = self.layer_params.needed_refreshes_mx[self.layer_repetition]
        needed_writes = int(self.layer_params.needed_Iact_writes)
        use_sparse = self.sparse_data == 1

        for cycle in range(int(refresh_bounds[1]), int(refresh_bounds[2])):
            for iact_cycle in range(needed_writes):
                spad = self.write_iact_pe(cl_x, cl_y, router, cycle, iact_cycle)
                storage.append(self.set_sparse_stream(spad) if use_sparse else spad)
        return storage

    def write_iact_addr_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate address storage for Dense layers (cl_y == 0 only)."""
        spad_storage = [0] * self.params.Iacts_Addr_per_PE
        if cl_y == 0 and self.params.Iacts_Addr_per_PE > 0:
            spad_storage[0] = int(self.layer_params.used_iact_per_PE)
        return spad_storage

    def write_iact_data_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate Dense-specific activation data SPAD storage (cl_y == 0 only)."""
        params = self.params
        layer_params = self.layer_params

        spad_storage = [[0, 0] for _ in range(params.Iacts_per_PE)]
        if cl_y != 0:
            return spad_storage

        used_iact = int(layer_params.used_iact_per_PE)
        max_words = min(math.ceil(params.Iacts_per_PE), used_iact)
        if max_words <= 0:
            return spad_storage

        base_offset = int(
            router * used_iact
            + (self.layer_repetition % layer_params.iact_transmissions_pe) * params.NUM_GLB_IACT * used_iact
        )
        flat_dram = np.asarray(self.dram_fmap).ravel()
        dram_len = flat_dram.size

        if base_offset >= 0 and base_offset + max_words <= dram_len:
            vals = flat_dram[base_offset : base_offset + max_words].tolist()
            for i, val in enumerate(vals):
                slot = spad_storage[i]
                slot[0] = val
                slot[1] = i
        else:
            for words_in_storage in range(max_words):
                idx = words_in_storage + base_offset
                slot = spad_storage[words_in_storage]
                slot[0] = int(flat_dram[idx]) if 0 <= idx < dram_len else 0
                slot[1] = words_in_storage

        return spad_storage


class DwIactStreamMapper(IactStreamMapper):
    """Specialized mapper for depthwise 2D convolution (DWConv2D) input activations.

    Distributes input channels across routers using kernel_per_pe_cluster to
    enable parallel channel processing.

    Attributes:
        Inherited from IactStreamMapper

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_data):
        """Initialize DWConv2D activation stream mapper."""
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, sparse_data)

    def write_iact_data_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate DWConv2D-specific activation data SPAD storage."""
        return self._write_spatial_iact_data_storage(cl_x, cl_y, router, cycle, iact_cycle)

    def write_iact_addr_storage(self, cl_x, cl_y, router, cycle, iact_cycle):
        """Generate address SPAD pointers for DWConv2D activation access."""
        return self._write_spatial_iact_addr_storage()