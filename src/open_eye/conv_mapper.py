# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Convolutional layer mapper for OpenEye accelerator.

This module provides the ConvMapper class which handles the mapping and configuration
of standard 2D convolutional layers to the OpenEye hardware architecture. It orchestrates
input activation, weight, and partial sum stream mappers to generate optimized data
layouts and router configurations for convolution operations.

Key Features:
- Register-based configuration using pack_registers for hardware control
- PE allocation bitmap generation for selective PE activation
- Multi-dimensional router configuration (iact, wght, psum paths)
- Quantization and offset parameter packing
- Support for various kernel sizes, strides, and filter counts
- Serial (DMA) and parallel communication modes
- Layer fusion optimization with skip flags

Architecture Integration:
    ConvMapper coordinates three specialized stream mappers:
    - ConvIactStreamMapper: Handles input activation data layout
    - ConvWghtStreamMapper: Manages weight distribution patterns
    - ConvPsumStreamMapper: Controls partial sum/bias initialization

Typical Usage:
    >>> conv_mapper = ConvMapper(params, layer_params, layer_repetition,
    ...                          dram_layer_content, sparse_iacts, sparse_wghts)
    >>> working_params = conv_mapper.write_working_parameters(params, layer_params,
    ...                                                        layer_repetition)
    >>> stream_data = conv_mapper.make_stream()
"""

import math
import logging
import open_eye.stream_dicts as strdic
from open_eye.generator import pack_registers
from open_eye.layer_mapper import LayerMapper
from open_eye.iact_stream_mapper import ConvIactStreamMapper
from open_eye.wght_stream_mapper import ConvWghtStreamMapper
from open_eye.psum_stream_mapper import ConvPsumStreamMapper

logger = logging.getLogger("cocotb")


class ConvMapper(LayerMapper):
    """Mapper for convolutional layers in the OpenEye accelerator.

    This class handles the mapping and configuration of convolutional layers by coordinating
    input activation (iact), weight (wght), and bias (psum) stream mappers. It generates
    the necessary register configurations, router settings, and data layouts for executing
    convolution operations on the hardware accelerator.

    Args:
        params: Hardware configuration parameters defining the accelerator architecture
        layer_params: Layer-specific parameters including kernel size, stride, filters, etc.
        layer_repetition: Current repetition index for layers that execute multiple times
        dram_layer_content: Tuple containing (input_data, weight_data, bias_data) from DRAM
        sparse_iacts: Sparsity information for input activations
        sparse_wghts: Sparsity information for weights

    Attributes:
        Inherits all attributes from LayerMapper base class, including:
        - input_mapper: ConvIactStreamMapper for input activation data
        - weight_mapper: ConvWghtStreamMapper for weight data
        - bias_mapper: ConvPsumStreamMapper for bias/partial sum data
    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts):
        input_mapper = ConvIactStreamMapper(params, layer_params, layer_repetition, dram_layer_content[0][0], sparse_iacts)
        weight_mapper = ConvWghtStreamMapper(params, layer_params, layer_repetition, dram_layer_content[1], sparse_wghts)
        bias_mapper = ConvPsumStreamMapper(params, layer_params, layer_repetition, dram_layer_content[2])
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, input_mapper, weight_mapper, bias_mapper)

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
        """Generate working parameters and register configuration for the layer.

        Packs all layer configuration parameters into register format for hardware transmission.
        Handles both serial (DMA) and parallel transmission modes.

        Args:
            params: Hardware configuration parameters
            layer_params: Layer-specific parameters containing kernel size, stride, filters, etc.
            layer_repetition: Current repetition index for the layer

        Returns:
            list: Packed register values ready for transmission to hardware.
        """
        # === SKIP FLAG DETERMINATION ===
        # Only load psums at the start of each iact transmission cycle
        layer_params.skipPsum = 0 if (layer_repetition % layer_params.iact_transmissions_pe) == 0 else 1

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
                "kernel_size_x": layer_params.kernel_size[0],
                "kernel_size_y": layer_params.kernel_size[1],
                "x_lines_reg": layer_params.iact_x_lines,
                "needed_wght_cycles": layer_params.needed_wght_cycles,
                "needed_cycles": layer_params.needed_cycles,
                "trans_cycles_iact": layer_params.trans_cycles_iact,
                "trans_cycles_wght": layer_params.trans_cycles_wght,
                "trans_cycles_psum": layer_params.trans_cycles_psum,
                "iact_glb_writing_cycles": layer_params.iact_glb_writing_cycles,
                "iact_converter_buffer_addr_max_cycles": layer_params.iact_converter_buffer_addr_max_cycles,
                "iact_channels_per_pe": layer_params.used_channels,
                "fc_size_reg": 0,
                "iact_size_x": layer_params.iact_size_x,
                "iact_size_y": layer_params.iact_size_y,
                "iact_size_c": layer_params.channels,
                "iact_x_per_cluster": layer_params.iact_x_per_cluster,
                "iact_x_add_up": layer_params.iact_x_add_up,
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
                "needed_psum_storage_cycles_reg": layer_params.psum_storage_cycles,
                "iact_channel_max_cycles": layer_params.diff_iact_layer,
                "input_activations": layer_params.used_iact_per_PE,
                "used_wght_per_PE": math.ceil(layer_params.used_wght_per_PE / params.PARALLEL_MACS),
                "filters": layer_params.used_psum_per_PE,
                "needed_x_cls_reg": layer_params.used_X_cluster,
                "needed_y_cls_reg": layer_params.used_Y_cluster,
                "needed_iact_cycles_reg": layer_params.needed_Iact_writes,
                "wght_addr_len_reg": layer_params.used_wght_addr_per_PE,
                "channel_div_trans": layer_params.channel_div_trans,
                "send_data_out": layer_params.send_values_out,
                "needed_iact_buffer_words": layer_params.needed_iact_buffer_words,
                "iact_x_line_repetitions_reg": layer_params.iact_x_line_repetitions,
                "buffer_cycles_for_x_iact": layer_params.buffer_cycles_for_x_iact,
                "start_param_array": layer_params.start_param_array,
                "limit_increase": layer_params.limit_increase,
                "lower_bound": layer_params.lower_bound,
                "upper_bound": layer_params.upper_bound,
                "initial_upper_limit": layer_params.initial_upper_limit,
                "iteration_for_kernels": layer_params.iteration_for_kernels,
                "fsm_psum_limit": layer_params.fsm_psum_limit,
                "cluster_per_conv_cycle": layer_params.cluster_per_conv_cycle,
                "iact_converter_max_cycles": layer_params.iact_converter_max_cycles,
                "iact_x_pos_inc": layer_params.iact_x_pos_inc,
                "iact_buffer_words_per_write": layer_params.iact_buffer_words_per_write,
                "iact_words_per_compute": layer_params.iact_words_per_compute,
                "pooling_mode": 0,
                "overhang_discrepancy": layer_params.overhang_discrepancy,
                "psum_output_words": layer_params.psum_output_words,
                "iact_read_limit_0": layer_params.iact_read_limit_0,
                "iact_read_limit_1": layer_params.iact_read_limit_1,
                "iact_read_limit_2": layer_params.iact_read_limit_2,
                "iact_read_limit_3": layer_params.iact_read_limit_3,
                "iact_read_limit_4": layer_params.iact_read_limit_4,
                "iact_read_inc_0": layer_params.iact_read_inc_0,
                "iact_read_inc_1": layer_params.iact_read_inc_1,
                "iact_read_inc_2": layer_params.iact_read_inc_2,
                "iact_read_inc_3": layer_params.iact_read_inc_3,
                "iact_read_inc_4": layer_params.iact_read_inc_4,
                "iact_write_limit_0": layer_params.iact_write_limit_0,
                "iact_write_limit_1": layer_params.iact_write_limit_1,
                "iact_write_limit_2": layer_params.iact_write_limit_2,
                "iact_write_inc_0": layer_params.iact_write_inc_0,
                "iact_write_inc_1": layer_params.iact_write_inc_1,
                "iact_write_inc_2": layer_params.iact_write_inc_2,
                "iact_write_inc_3": layer_params.iact_write_inc_3,
                "pagu_wght_limit": layer_params.pagu_wght_limit,
                "psum_pagu_cs_limit_0": layer_params.psum_pagu_cs_limit[0],
                "psum_pagu_cs_limit_1": layer_params.psum_pagu_cs_limit[1],
                "psum_pagu_cs_limit_2": layer_params.psum_pagu_cs_limit[2],
                "psum_pagu_cs_limit_3": layer_params.psum_pagu_cs_limit[3],
                "psum_pagu_cs_inc_0": layer_params.psum_pagu_cs_inc[0],
                "psum_pagu_cs_inc_1": layer_params.psum_pagu_cs_inc[1],
                "psum_pagu_cs_inc_2": layer_params.psum_pagu_cs_inc[2],
                "psum_pagu_cs_inc_3": layer_params.psum_pagu_cs_inc[3],
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
                "gemm_mode": getattr(layer_params, "gemm_mode", 0),
                "concat_limit": layer_params.concat_limit,
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
        storage = [[] for _ in range(len(strdic.status_dict))]
        sdict = strdic.status_dict

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
        storage[sdict["gemm_mode"]] = getattr(layer_params, "gemm_mode", 0)

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
        """Configure input activation (iact) router settings for all clusters.

        Routing values:
        - 1: Single Y cluster mode
        - 9: First cluster in multi-Y-cluster group (multiple PEs per cluster)
        - 17: Last cluster in multi-Y-cluster group (multiple PEs per cluster)
        - 25: Middle cluster in multi-Y-cluster group (multiple PEs per cluster)
        - 3: First cluster (single PE per cluster)
        - 33: Non-first cluster (single PE per cluster)
        """
        clusters_x = params.Clusters_X
        clusters_y = params.Clusters_Y
        num_glb_iact = params.NUM_GLB_IACT

        if clusters_y == 1 and clusters_x == 1:
            return [0] if params.SERIAL else [[[[] for _ in range(num_glb_iact)]]]

        used_y = layer_params.used_Y_cluster
        multi_pe_y = layer_params.used_PEs_Y > 1

        if params.SERIAL:
            flat_values = []
            for cl_y in range(clusters_y):
                if used_y == 1:
                    val = 1
                elif multi_pe_y:
                    if (cl_y % used_y) == 0:
                        val = 9
                    elif ((cl_y + 1) % used_y) == 0:
                        val = 17
                    else:
                        val = 25
                else:
                    val = 3 if cl_y == 0 else 33
                flat_values.extend([val] * (clusters_x * num_glb_iact))
            return self._pack_router_values(flat_values, params.DMA_BITWIDTH, params.Iact_Router_Bits)

        storage = [[[0] * num_glb_iact for _ in range(clusters_y)] for _ in range(clusters_x)]
        for cl_y in range(clusters_y):
            if used_y == 1:
                val = 1
            elif multi_pe_y:
                if (cl_y % used_y) == 0:
                    val = 9
                elif ((cl_y + 1) % used_y) == 0:
                    val = 17
                else:
                    val = 25
            else:
                val = 3 if cl_y == 0 else 33

            row_vals = [val] * num_glb_iact
            for cl_x in range(clusters_x):
                storage[cl_x][cl_y] = row_vals.copy()
        return storage

    def write_router_wght(self, params, layer_params):
        """Configure weight (wght) router settings for all clusters.

        Routing values:
        - 0: Use local weight data (first cluster, single cluster mode, or multi-kernel mode)
        - 1: Forward weight data from previous cluster
        """
        clusters_x = params.Clusters_X
        clusters_y = params.Clusters_Y
        wght_routers = params.Wght_Routers

        if clusters_x == 1:
            return [] if params.SERIAL else [[[[] for _ in range(wght_routers)] for _ in range(clusters_y)]]

        broadcast_all = (layer_params.single_cluster_computation == 1) or (
            layer_params.different_kernels_per_calculation >= 2
        )

        if params.SERIAL:
            flat_values = []
            per_cl_x = clusters_y * wght_routers
            for cl_x in range(clusters_x):
                val = 0 if (cl_x == 0 or broadcast_all) else 1
                flat_values.extend([val] * per_cl_x)
            return self._pack_router_values(flat_values, params.DMA_BITWIDTH, params.Wght_Router_Bits)

        storage = [[[0] * wght_routers for _ in range(clusters_y)] for _ in range(clusters_x)]
        for cl_x in range(clusters_x):
            val = 0 if (cl_x == 0 or broadcast_all) else 1
            if val != 0:
                for cl_y in range(clusters_y):
                    storage[cl_x][cl_y] = [val] * wght_routers
        return storage

    def write_router_psum(self, params, layer_params):
        """Configure partial sum (psum) router settings for all clusters.

        Routing values:
        - 0: No routing (inactive cluster)
        - 2: Pass-through cluster (middle of Y-cluster group)
        - 3: Final accumulation cluster (last in Y-cluster group)
        - 4: Output cluster (single PE per cluster mode)
        - 5: First accumulation cluster (first in Y-cluster group)
        """
        clusters_x = params.Clusters_X
        clusters_y = params.Clusters_Y
        psum_routers = params.Psum_Routers

        if clusters_y == 1:
            return [] if params.SERIAL else [[[[] for _ in range(psum_routers)] for _ in range(clusters_y)] for _ in range(clusters_x)]

        used_y = layer_params.used_Y_cluster
        active_x_limit = (layer_params.iact_size_x + layer_params.add_up) * layer_params.different_kernels_per_calculation
        y_packages_limit = layer_params.Y_Cluster_Packages
        pes_x = params.PEs_X

        if params.SERIAL:
            flat_values = []
            y_package_count = 0
            for cl_y in range(clusters_y):
                for cl_x in range(clusters_x):
                    for router in range(psum_routers):
                        if used_y == 1:
                            val = 4 if (((cl_y * clusters_x) + cl_x) * pes_x < active_x_limit) else 0
                        elif (cl_y % used_y) == 0:
                            if cl_x + router == 0:
                                y_package_count += 1
                            val = 5 if y_package_count <= y_packages_limit else 0
                        elif ((cl_y + 1) % used_y) == 0:
                            val = 3 if y_package_count <= y_packages_limit else 0
                        else:
                            val = 2 if y_package_count <= y_packages_limit else 0
                        flat_values.append(val)
            return self._pack_router_values(flat_values, params.DMA_BITWIDTH, params.Psum_Router_Bits)

        storage = [[[[] for _ in range(psum_routers)] for _ in range(clusters_y)] for _ in range(clusters_x)]
        y_package_count = 0
        for cl_y in range(clusters_y):
            for cl_x in range(clusters_x):
                for router in range(psum_routers):
                    if used_y == 1:
                        storage[cl_x][cl_y][router] = 4
                    elif (cl_y % used_y) == 0:
                        if cl_x + router == 0:
                            y_package_count += 1
                        if y_package_count <= y_packages_limit:
                            storage[cl_x][cl_y][router] = 5
                    elif ((cl_y + 1) % used_y) == 0:
                        if y_package_count <= y_packages_limit:
                            storage[cl_x][cl_y][router] = 3
                    else:
                        if y_package_count <= y_packages_limit:
                            storage[cl_x][cl_y][router] = 2
        return storage

    def write_psum_data_glb(self, params, layer_params, layer_repetition, dram, cl_y, router, cycle):
        """Legacy helper for global buffer partial sum initialization."""
        num_entries = min(
            layer_params.filters // layer_params.needed_wght_transmissions,
            int(layer_params.used_psum_per_PE),
        )
        count = max(0, num_entries)
        if params.SERIAL:
            return [0] * count
        return [[0] * count for _ in range(params.Clusters_X)]