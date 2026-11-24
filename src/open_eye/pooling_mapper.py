# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Pooling layer mapper for OpenEye accelerator.

This module provides the PoolingMapper class which handles the mapping and configuration
of pooling layers (max pooling, average pooling) to the OpenEye hardware architecture.
Pooling operations reduce spatial dimensions by applying a reduction function (max or
average) over local regions of the input feature map.

Key Features:
- Configuration of working parameters for pooling operations
- Router configuration for input activation and partial sum data paths
- Support for both max pooling and average pooling operations
- Support for both serial (DMA) and parallel communication modes
- PE cluster allocation and utilization for pooling computations

Typical Usage:
    >>> pooling_mapper = PoolingMapper(params, layer_params, layer_repetition,
    ...                                dram_layer_content, sparse_iacts, sparse_wghts)
    >>> working_params = pooling_mapper.write_working_parameters(params, layer_params,
    ...                                                           layer_repetition)
"""

import math
import logging
import open_eye.stream_dicts as strdic
from open_eye.generator import pack_registers
from open_eye.layer_mapper import LayerMapper

logger = logging.getLogger("cocotb")


class PoolingMapper(LayerMapper):
    """Mapper for pooling layers in the OpenEye accelerator.

    This class extends LayerMapper to provide specialized mapping for pooling operations.
    Pooling layers downsample the spatial dimensions of feature maps using max or average
    operations over local windows. Unlike convolution layers, pooling operations do not
    use weights but simply apply a reduction function over input activations.

    Attributes:
        Inherited from LayerMapper (params, layer_params, layer_repetition, etc.)

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts):
        """Initialize the pooling layer mapper.

        Creates a mapper for pooling operations. Pooling layers do not require
        weight mappers or bias mappers since they perform spatial reduction without
        trainable parameters.

        Args:
            params: OpenEye hardware parameters (cluster config, PE counts, etc.)
            layer_params: Pooling layer parameters (kernel size, stride, pooling type, etc.)
            layer_repetition (int): Current repetition index for this layer
            dram_layer_content (list): DRAM contents [feature_maps, (unused), (unused)]
            sparse_iacts (bool): Whether input activations are sparse (unused for pooling)
            sparse_wghts (bool): Whether weights are sparse (not applicable for pooling)

        """
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

    @staticmethod
    def _pack_router_values(flat_values, dma_bitwidth, router_bits):
        """Pack a flat list of router configuration codes into DMA-width words."""
        entries_per_word = dma_bitwidth // router_bits
        packed = []
        for i in range(0, len(flat_values), entries_per_word):
            word = 0
            for slot, val in enumerate(flat_values[i : i + entries_per_word]):
                word |= val << (router_bits * slot)
            packed.append(word)
        return packed

    @staticmethod
    def _compute_pe_bitmap(params, layer_params):
        """Build integer bitmap of active PEs across all clusters."""
        computing_pes = 0
        bit_idx = 0
        comp_mx = layer_params.computing_mx

        for x in range(params.Clusters_X):
            comp_x = comp_mx[x]
            for y in range(params.Clusters_Y):
                comp_xy = comp_x[y]
                for pe_y in range(params.PEs_Y):
                    comp_row = comp_xy[pe_y]
                    for pe_x in range(params.PEs_X):
                        if comp_row[pe_x] == 1:
                            computing_pes |= 1 << bit_idx
                        bit_idx += 1
        return computing_pes

    def write_working_parameters(self, params, layer_params, layer_repetition):
        """Generate working parameters and router configurations for the pooling layer.

        This method creates the complete configuration for executing a pooling operation
        on the OpenEye accelerator, including PE allocation, data routing, stride settings,
        and control signals. Supports both serial (DMA-based) and parallel communication modes.

        Args:
            params: OpenEye hardware parameters
            layer_params: Layer-specific parameters for this pooling operation
            layer_repetition (int): Current repetition index

        Returns:
            list: Configuration storage containing all working parameters. Format depends
                  on params.SERIAL mode:
                  - Serial mode: DMA-formatted bitstream as list of packed integers
                  - Parallel mode: List indexed by status_dict keys

        """
        # === PE ALLOCATION BITMAP GENERATION ===
        computing_pes = self._compute_pe_bitmap(params, layer_params)

        # === SERIAL MODE: DMA TRANSMISSION ===
        if params.SERIAL:
            dma_storage = pack_registers({
                "stride_x": layer_params.strideX,
                "stride_y": layer_params.strideY,
                "skipIact_reg": layer_params.skipIact,
                "skipWght_reg": layer_params.skipWght,
                "skipPsum_reg": layer_params.skipPsum,
                "dw_mode": 0,
                "psum_q": layer_params.psum_delay,
                "kernel_per_pe_cluster_reg": layer_params.kernel_per_pe_cluster,
                "kernel_size_x": layer_params.strideX,
                "kernel_size_y": layer_params.strideY,
                "x_lines_reg": layer_params.iact_x_lines,
                "needed_wght_cycles": math.ceil(
                    layer_params.filters
                    / (layer_params.used_psum_per_PE * layer_params.different_kernels_per_calculation)
                ),
                "needed_cycles": layer_params.iact_converter_buffer_addr_max_cycles,
                "trans_cycles_iact": layer_params.trans_cycles_iact,
                "trans_cycles_wght": layer_params.trans_cycles_wght,
                "trans_cycles_psum": layer_params.trans_cycles_psum,
                "iact_converter_buffer_addr_max_cycles": layer_params.needed_cycles,
                "iact_channels_per_pe": layer_params.used_channels,
                "fc_size_reg": 0,
                "iact_size_x": layer_params.iact_size_x,
                "iact_size_y": layer_params.iact_size_y,
                "iact_size_c": layer_params.channels,
                "iact_x_per_cluster": layer_params.iact_x_per_cluster,
                "psum_x_all_cluster": layer_params.psum_x_all_cluster,
                "padding_x": layer_params.padding_x,
                "padding_y": layer_params.padding_y,
                "psum_size_x": math.ceil(layer_params.iact_size_x / layer_params.strideX),
                "psum_size_y": math.ceil(layer_params.iact_size_y / layer_params.strideY),
                "iact_needed_cycles": layer_params.iact_stream_cycles,
                "kernels_per_calc": layer_params.different_kernels_per_calculation,
                "y_lines_per_calc": layer_params.y_lines_per_calculation,
                "store_in_psum": layer_params.store_in_psum,
                "max_pooling": layer_params.max_pooling,
                "fully_connected_layer": layer_params.fully_connected,
                "choose_iact_buffer_output": layer_params.choose_iact_storage_output,
                "choose_iact_buffer_input": layer_params.choose_iact_storage_input,
                "iact_channels_per_pe_next_layer": layer_params.diff_iact_layer_next_layer,
                "needed_psum_storage_cycles_reg": 0,
                "iact_channel_max_cycles": layer_params.diff_iact_layer,
                "input_activations": layer_params.used_iact_per_PE,
                "filters": layer_params.used_psum_per_PE,
                "needed_x_cls_reg": layer_params.used_X_cluster,
                "needed_y_cls_reg": layer_params.used_Y_cluster,
                "needed_iact_cycles_reg": layer_params.needed_Iact_writes,
                "wght_addr_len_reg": layer_params.used_wght_addr_per_PE,
                "send_data_out": layer_params.send_values_out,
                "needed_iact_buffer_words": layer_params.needed_iact_buffer_words,
                "iact_x_line_repetitions_reg": layer_params.iact_x_line_repetitions,
                "buffer_cycles_for_x_iact": layer_params.buffer_cycles_for_x_iact,
                "start_param_array": layer_params.start_param_array,
                "limit_increase": layer_params.limit_increase,
                "initial_upper_limit": layer_params.initial_upper_limit,
                "iteration_for_kernels": layer_params.iteration_for_kernels,
                "fsm_psum_limit": layer_params.fsm_psum_limit,
                "cluster_per_conv_cycle": layer_params.cluster_per_conv_cycle,
                "iact_converter_max_cycles": layer_params.iact_converter_max_cycles,
                "iact_buffer_words_per_write": layer_params.iact_buffer_words_per_write,
                "pooling_mode": layer_params.pooling_mode,
                "used_wght_per_PE": layer_params.used_wght_per_PE,
                "overhang_discrepancy": layer_params.overhang_discrepancy,
                "psum_output_words": layer_params.psum_output_words,
                "iact_read_inc_1": layer_params.iact_read_inc_1,
                "iact_read_inc_2": layer_params.iact_read_inc_2,
                "iact_read_inc_3": layer_params.iact_read_inc_3,
                "iact_read_inc_4": layer_params.iact_read_inc_4,
                "iact_write_inc_1": layer_params.iact_write_inc_1,
                "iact_write_inc_2": layer_params.iact_write_inc_2,
                "iact_write_inc_3": layer_params.iact_write_inc_3,
                "pagu_wght_limit": layer_params.pagu_wght_limit,
                "psum_pagu_loop_limit_0": layer_params.psum_pagu_loop_limit[0],
                "psum_pagu_loop_limit_1": layer_params.psum_pagu_loop_limit[1],
                "psum_pagu_loop_limit_2": layer_params.psum_pagu_loop_limit[2],
                "psum_pagu_loop_limit_3": layer_params.psum_pagu_loop_limit[3],
                "psum_pagu_loop_limit_4": layer_params.psum_pagu_loop_limit[4],
                "psum_pagu_addr_inc_0": layer_params.psum_pagu_addr_inc[0],
                "psum_pagu_addr_inc_1": layer_params.psum_pagu_addr_inc[1],
                "psum_pagu_addr_inc_2": layer_params.psum_pagu_addr_inc[2],
                "psum_pagu_addr_inc_3": layer_params.psum_pagu_addr_inc[3],
                "psum_pagu_addr_inc_4": layer_params.psum_pagu_addr_inc[4],
                "gemm_mode": 0,
            })

            # Slice PE bitmap directly into DMA-width words without binary string conversion
            dma_w = params.DMA_BITWIDTH
            dma_mask = (1 << dma_w) - 1
            num_pe_words = math.ceil(params.PE_Complete / dma_w)
            for x in range(num_pe_words):
                dma_storage.append((computing_pes >> (x * dma_w)) & dma_mask)

            # Append router configurations for all three data paths
            dma_storage.extend(self.write_router_iact(params, layer_params))
            dma_storage.extend(self.write_router_wght(params, layer_params))
            dma_storage.extend(self.write_router_psum(params, layer_params))
            return dma_storage

        # === PARALLEL MODE: DIRECT PARAMETER ASSIGNMENT ===
        sdict = strdic.status_dict
        storage = [[] for _ in range(len(sdict))]

        storage[sdict["data_mode"]] = params.data_mode
        storage[sdict["realfactor"]] = layer_params.realfactor
        storage[sdict["autofunction"]] = params.autofunction
        storage[sdict["poolingmode"]] = params.poolingmode
        storage[sdict["psum_delay"]] = layer_params.psum_delay
        storage[sdict["needed_refreshes"]] = layer_params.needed_refreshes_mx[layer_repetition][0]
        storage[sdict["used_X_cluster"]] = layer_params.used_X_cluster
        storage[sdict["used_Y_cluster"]] = layer_params.used_Y_cluster
        storage[sdict["needed_Iact_writes"]] = layer_params.needed_Iact_writes
        storage[sdict["used_psum_per_PE"]] = layer_params.used_psum_per_PE
        storage[sdict["used_iact_addr_per_PE"]] = layer_params.used_iact_addr_per_PE
        storage[sdict["used_wght_addr_per_PE"]] = layer_params.used_wght_addr_per_PE
        storage[sdict["used_iact_per_PE"]] = layer_params.used_iact_per_PE
        storage[sdict["iact_addr_len"]] = layer_params.iact_addr_len
        storage[sdict["iact_data_len"]] = layer_params.iact_data_len
        storage[sdict["strideX"]] = layer_params.strideX
        storage[sdict["strideY"]] = layer_params.strideY
        storage[sdict["skipIact"]] = layer_params.skipIact
        storage[sdict["skipWght"]] = layer_params.skipWght
        storage[sdict["skipPsum"]] = layer_params.skipPsum
        storage[sdict["usePEs"]] = computing_pes
        storage[sdict["kernel_per_pe_cluster"]] = layer_params.kernel_per_pe_cluster

        storage[sdict["router_iact"]] = self.write_router_iact(params, layer_params)
        storage[sdict["router_wght"]] = self.write_router_wght(params, layer_params)
        storage[sdict["router_psum"]] = self.write_router_psum(params, layer_params)
        return storage

    def write_quant_and_offset(self, params, layer_params, layer_repetition):
        """Generate unified quantization and offset parameters packed for shift-register streaming.

        Layout per Filter (LSB -> MSB):
        - Offset   : OFFSET_WIDTH bits
        - Exponent : EXPONENT_WIDTH bits
        - Mantissa : MANTISSA_WIDTH bits

        Shift behavior:
        Filter 0 lands at the lowest bit-positions (quant_reg[0 +: ENTRY_WIDTH]).
        Incoming DMA words shift from lower to higher indices, so words are emitted
        from MSB down to LSB.
        """
        offset_w = getattr(params, "OFFSET_WIDTH", 8)
        exp_w = getattr(params, "EXPONENT_WIDTH", 7)
        mant_w = getattr(params, "MANTISSA_WIDTH", 25)

        entry_w = offset_w + exp_w + mant_w
        dma_w = params.DMA_BITWIDTH

        offset_mask = (1 << offset_w) - 1
        exp_mask = (1 << exp_w) - 1
        mant_mask = (1 << mant_w) - 1
        mant_shift = offset_w + exp_w

        quantize = layer_params.quantize
        offsets = layer_params.offset
        packed_bitstream = 0

        for f in range(params.QUANT_AMOUNT):
            q_entry = quantize[f]
            filter_entry = (
                (int(offsets[f]) & offset_mask)
                | ((int(q_entry[1]) & exp_mask) << offset_w)
                | ((int(q_entry[0]) & mant_mask) << mant_shift)
            )
            packed_bitstream |= filter_entry << (f * entry_w)

        total_bits = params.QUANT_AMOUNT * entry_w
        num_dma_words = math.ceil(total_bits / dma_w)
        total_padded_bits = num_dma_words * dma_w
        dma_mask = (1 << dma_w) - 1

        dma_storage = []
        for i in range(num_dma_words):
            shift_amount = total_padded_bits - (i + 1) * dma_w
            word = (
                (packed_bitstream >> shift_amount)
                if shift_amount >= 0
                else (packed_bitstream << -shift_amount)
            ) & dma_mask
            dma_storage.append(word)
        return dma_storage

    def write_router_iact(self, params, layer_params):
        """Generate input activation router configuration for pooling operations.

        Note:
            Router value 1 indicates local data processing within each cluster.
            In serial mode, only Y-clusters less than 4 are configured to 1;
            subsequent Y-clusters receive 0.
        """
        clusters_x = params.Clusters_X
        clusters_y = params.Clusters_Y
        num_glb_iact = params.NUM_GLB_IACT

        if params.SERIAL:
            per_y_entries = clusters_x * num_glb_iact
            active_y = min(4, clusters_y)
            inactive_y = max(0, clusters_y - 4)
            flat_values = [1] * (active_y * per_y_entries) + [0] * (inactive_y * per_y_entries)
            return self._pack_router_values(flat_values, params.DMA_BITWIDTH, params.Iact_Router_Bits)

        return [[[1] * num_glb_iact for _ in range(clusters_y)] for _ in range(clusters_x)]

    def write_router_wght(self, params, layer_params):
        """Generate weight router configuration for pooling operations.

        All weight routers are set to 0 since pooling operations do not require
        weight data.
        """
        clusters_x = params.Clusters_X
        clusters_y = params.Clusters_Y
        wght_routers = params.Wght_Routers

        if params.SERIAL:
            total_entries = clusters_x * clusters_y * wght_routers
            entries_per_word = params.DMA_BITWIDTH // params.Wght_Router_Bits
            return [0] * math.ceil(total_entries / entries_per_word)

        return [[[0] * wght_routers for _ in range(clusters_y)] for _ in range(clusters_x)]

    def write_router_psum(self, params, layer_params):
        """Generate partial sum router configuration for pooling operations.

        Routing values:
        - 2: Pass through intermediate cluster (forward to next)
        - 3: Last cluster in chain (receive and finalize)
        - 4: Single PE per column (local handling only)
        - 5: First cluster in chain (receive from bottom and forward up)
        """
        clusters_x = params.Clusters_X
        clusters_y = params.Clusters_Y
        psum_routers = params.Psum_Routers
        used_y = layer_params.used_Y_cluster
        per_y_entries = clusters_x * psum_routers

        if params.SERIAL:
            flat_values = []
            for cl_y in range(clusters_y):
                if used_y == 1:
                    val = 4
                elif (cl_y % used_y) == 0:
                    val = 5
                elif ((cl_y + 1) % used_y) == 0:
                    val = 3
                else:
                    val = 2
                flat_values.extend([val] * per_y_entries)
            return self._pack_router_values(flat_values, params.DMA_BITWIDTH, params.Psum_Router_Bits)

        storage = [[[0] * psum_routers for _ in range(clusters_y)] for _ in range(clusters_x)]
        for cl_y in range(clusters_y):
            if used_y == 1:
                val = 4
            elif (cl_y % used_y) == 0:
                val = 5
            elif ((cl_y + 1) % used_y) == 0:
                val = 3
            else:
                val = 2
            row_vals = [val] * psum_routers
            for cl_x in range(clusters_x):
                storage[cl_x][cl_y] = row_vals.copy()
        return storage

    def write_psum_data_glb(self, params, layer_params, layer_repetition, dram, cl_y, router, cycle):
        """Legacy helper for global buffer partial sum initialization."""
        num_entries = min(
            math.ceil(self.layer_params.used_psum_per_PE / 2),
            int(self.layer_params.used_psum_per_PE),
        )
        count = max(0, num_entries)
        if params.SERIAL:
            return [0] * (count * self.params.Clusters_X)
        return [[0] * count for _ in range(self.params.Clusters_X)]