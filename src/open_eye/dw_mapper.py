# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Depthwise convolution layer mapper for OpenEye accelerator.

This module provides the DWMapper class which handles the mapping and configuration
of depthwise convolutional layers to the OpenEye hardware architecture. It manages
the routing of input activations, weights, and partial sums across processing elements
and clusters.

Key Features:
- Configuration of working parameters for depthwise convolution
- Router configuration for activation, weight, and partial sum data paths
- Support for both serial and parallel communication modes
- PE cluster allocation and utilization tracking

Typical Usage:
    >>> dw_mapper = DWMapper(params, layer_params, layer_repetition,
    ...                      dram_layer_content, sparse_iacts, sparse_wghts)
    >>> working_params = dw_mapper.write_working_parameters(params, layer_params,
    ...                                                      layer_repetition)
"""

import logging
import math
import numpy as np
import open_eye.stream_dicts as strdic
from open_eye.generator import pack_registers
from open_eye.iact_stream_mapper import DwIactStreamMapper
from open_eye.layer_mapper import LayerMapper
from open_eye.psum_stream_mapper import DwPsumStreamMapper
from open_eye.wght_stream_mapper import DwWghtStreamMapper

logger = logging.getLogger("cocotb")


class DWMapper(LayerMapper):
    """Mapper for depthwise convolutional layers in the OpenEye accelerator.

    This class extends LayerMapper to provide specialized mapping for depthwise
    convolutions, where each input channel is convolved with its own set of filters
    independently. It configures the data routing and PE allocation for efficient
    depthwise convolution execution.

    Attributes:
        Inherited from LayerMapper (input_mapper, weight_mapper, bias_mapper, etc.)

    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts):
        """Initialize the depthwise convolution mapper.

        Creates stream mappers for input activations, weights, and biases/partial sums
        specific to depthwise convolution operations.

        Args:
            params: OpenEye hardware parameters (cluster config, PE counts, etc.)
            layer_params: Depthwise convolution layer parameters (kernel size, stride, etc.)
            layer_repetition (int): Current repetition index for this layer
            dram_layer_content (list): DRAM contents [feature_maps, weights, biases]
            sparse_iacts (bool): Whether input activations are sparse
            sparse_wghts (bool): Whether weights are sparse

        """
        input_mapper = DwIactStreamMapper(
            params, layer_params, layer_repetition, dram_layer_content[0][0], sparse_iacts
        )
        weight_mapper = DwWghtStreamMapper(
            params, layer_params, layer_repetition, dram_layer_content[1], sparse_wghts
        )
        bias_mapper = DwPsumStreamMapper(params, layer_params, layer_repetition, dram_layer_content[2])
        super().__init__(
            params, layer_params, layer_repetition, dram_layer_content, input_mapper, weight_mapper, bias_mapper
        )

    @staticmethod
    def _pack_router_values(flat_values, dma_bitwidth, router_bits):
        """Pack a flat sequence of router configuration codes into DMA-width words."""
        if not flat_values:
            return []
        entries_per_word = dma_bitwidth // router_bits
        packed = []
        for i in range(0, len(flat_values), entries_per_word):
            word = 0
            for slot, val in enumerate(flat_values[i : i + entries_per_word]):
                word |= int(val) << (router_bits * slot)
            packed.append(word)
        return packed

    @staticmethod
    def _compute_pe_bitmap(params, layer_params):
        """Build integer bitmap of active PEs across all clusters."""
        comp_arr = np.asarray(layer_params.computing_mx)[
            : params.Clusters_X, : params.Clusters_Y, : params.PEs_Y, : params.PEs_X
        ]
        computing_pes = 0
        for bit_idx in np.flatnonzero(comp_arr.ravel() == 1).tolist():
            computing_pes |= 1 << bit_idx
        return computing_pes

    @staticmethod
    def _iact_router_code(cl_y: int, used_y: int, multi_pe_y: bool) -> int:
        """Return the iact router mode code for a Y cluster index."""
        if used_y == 1:
            return 1
        if multi_pe_y:
            if (cl_y % used_y) == 0:
                return 9
            if ((cl_y + 1) % used_y) == 0:
                return 17
            return 25
        return 3 if cl_y == 0 else 33

    def write_working_parameters(self, params, layer_params, layer_repetition):
        """Generate working parameters and router configurations for the depthwise layer.

        This method creates the complete configuration for executing a depthwise convolution
        layer on the OpenEye accelerator, including PE allocation, data routing, and control
        signals. Supports both serial (DMA-based) and parallel communication modes.

        Args:
            params: OpenEye hardware parameters
            layer_params: Layer-specific parameters for this depthwise convolution
            layer_repetition (int): Current repetition index

        Returns:
            list: Configuration storage containing all working parameters. Format depends
                  on params.SERIAL mode:
- Serial mode: DMA-formatted bitstream
- Parallel mode: List indexed by status_dict keys

        """
        # === SKIP FLAG DETERMINATION ===
        # Only load psums at the start of each iact transmission cycle
        layer_params.skipPsum = 0 if (layer_repetition % layer_params.iact_transmissions_pe) == 0 else 1

        # === PE ALLOCATION BITMAP GENERATION ===
        computing_pes = self._compute_pe_bitmap(params, layer_params)

        # === SERIAL MODE: DMA TRANSMISSION ===
        if params.SERIAL:
            cs_limit = [int(v) for v in layer_params.psum_pagu_cs_limit]
            cs_inc = [int(v) for v in layer_params.psum_pagu_cs_inc]
            loop_limit = [int(v) for v in layer_params.psum_pagu_loop_limit]
            addr_inc = [int(v) for v in layer_params.psum_pagu_addr_inc]

            dma_storage = pack_registers({
                "stride_x": int(layer_params.strideX),
                "stride_y": int(layer_params.strideY),
                "skipIact_reg": int(layer_params.skipIact),
                "skipWght_reg": int(layer_params.skipWght),
                "skipPsum_reg": int(layer_params.skipPsum),
                "dw_mode": 1,
                "psum_q": int(layer_params.psum_delay),
                "kernel_per_pe_cluster_reg": int(layer_params.kernel_per_pe_cluster),
                "kernel_size_x": int(layer_params.kernel_size[0]),
                "kernel_size_y": int(layer_params.kernel_size[1]),
                "x_lines_reg": int(layer_params.iact_x_lines),
                "needed_wght_cycles": int(layer_params.needed_wght_cycles),
                "needed_cycles": int(layer_params.needed_cycles),
                "trans_cycles_iact": int(layer_params.trans_cycles_iact),
                "trans_cycles_wght": int(layer_params.trans_cycles_wght),
                "trans_cycles_psum": int(layer_params.trans_cycles_psum),
                "iact_glb_writing_cycles": int(layer_params.iact_glb_writing_cycles),
                "iact_converter_buffer_addr_max_cycles": int(layer_params.iact_converter_buffer_addr_max_cycles),
                "iact_channels_per_pe": int(layer_params.used_channels),
                "fc_size_reg": 0,
                "iact_size_x": int(layer_params.iact_size_x),
                "iact_size_y": int(layer_params.iact_size_y),
                "iact_size_c": int(layer_params.channels),
                "iact_x_per_cluster": int(layer_params.iact_x_per_cluster),
                "iact_x_add_up": int(layer_params.iact_x_add_up),
                "psum_x_all_cluster": int(layer_params.psum_x_all_cluster),
                "padding_x": int(layer_params.padding_x),
                "padding_y": int(layer_params.padding_y),
                "psum_size_x": math.ceil(layer_params.iact_size_x / layer_params.strideX),
                "psum_size_y": math.ceil(layer_params.iact_size_y / layer_params.strideY),
                "iact_needed_cycles": int(layer_params.iact_stream_cycles),
                "kernels_per_calc": int(layer_params.different_kernels_per_calculation),
                "y_lines_per_calc": int(layer_params.y_lines_per_calculation),
                "store_in_psum": int(layer_params.store_in_psum),
                "max_pooling": int(layer_params.max_pooling),
                "fully_connected_layer": int(layer_params.fully_connected),
                "choose_iact_buffer_output": int(layer_params.choose_iact_storage_output),
                "choose_iact_buffer_input": int(layer_params.choose_iact_storage_input),
                "iact_channels_per_pe_next_layer": int(layer_params.diff_iact_layer_next_layer),
                "needed_psum_storage_cycles_reg": 1,
                "iact_channel_max_cycles": 1,
                "input_activations": int(layer_params.used_iact_per_PE),
                "used_wght_per_PE": math.ceil(layer_params.used_wght_per_PE / params.PARALLEL_MACS),
                "filters": int(layer_params.used_channels),
                "needed_x_cls_reg": int(layer_params.used_X_cluster),
                "needed_y_cls_reg": int(layer_params.used_Y_cluster),
                "needed_iact_cycles_reg": int(layer_params.needed_Iact_writes),
                "wght_addr_len_reg": int(layer_params.used_wght_addr_per_PE),
                "channel_div_trans": int(layer_params.channel_div_trans),
                "send_data_out": int(layer_params.send_values_out),
                "needed_iact_buffer_words": int(layer_params.needed_iact_buffer_words),
                "iact_x_line_repetitions_reg": int(layer_params.iact_x_line_repetitions),
                "buffer_cycles_for_x_iact": int(layer_params.buffer_cycles_for_x_iact),
                "start_param_array": int(layer_params.start_param_array),
                "limit_increase": int(layer_params.limit_increase),
                "lower_bound": int(layer_params.lower_bound),
                "upper_bound": int(layer_params.upper_bound),
                "initial_upper_limit": int(layer_params.initial_upper_limit),
                "iteration_for_kernels": int(layer_params.iteration_for_kernels),
                "fsm_psum_limit": int(layer_params.fsm_psum_limit),
                "cluster_per_conv_cycle": int(layer_params.cluster_per_conv_cycle),
                "iact_converter_max_cycles": int(layer_params.iact_converter_max_cycles),
                "iact_x_pos_inc": int(layer_params.iact_x_pos_inc),
                "iact_buffer_words_per_write": int(layer_params.iact_buffer_words_per_write),
                "iact_words_per_compute": int(layer_params.iact_words_per_compute),
                "pooling_mode": 0,
                "overhang_discrepancy": int(layer_params.overhang_discrepancy),
                "psum_output_words": int(layer_params.psum_output_words),
                "iact_read_limit_0": int(layer_params.iact_read_limit_0),
                "iact_read_limit_1": int(layer_params.iact_read_limit_1),
                "iact_read_limit_2": int(layer_params.iact_read_limit_2),
                "iact_read_limit_3": int(layer_params.iact_read_limit_3),
                "iact_read_limit_4": int(layer_params.iact_read_limit_4),
                "iact_read_inc_0": int(layer_params.iact_read_inc_0),
                "iact_read_inc_1": int(layer_params.iact_read_inc_1),
                "iact_read_inc_2": int(layer_params.iact_read_inc_2),
                "iact_read_inc_3": int(layer_params.iact_read_inc_3),
                "iact_read_inc_4": int(layer_params.iact_read_inc_4),
                "iact_write_limit_0": int(layer_params.iact_write_limit_0),
                "iact_write_limit_1": int(layer_params.iact_write_limit_1),
                "iact_write_limit_2": int(layer_params.iact_write_limit_2),
                "iact_write_inc_0": int(layer_params.iact_write_inc_0),
                "iact_write_inc_1": int(layer_params.iact_write_inc_1),
                "iact_write_inc_2": int(layer_params.iact_write_inc_2),
                "iact_write_inc_3": int(layer_params.iact_write_inc_3),
                "pagu_wght_limit": int(layer_params.pagu_wght_limit),
                "psum_pagu_cs_limit_0": cs_limit[0],
                "psum_pagu_cs_limit_1": cs_limit[1],
                "psum_pagu_cs_limit_2": cs_limit[2],
                "psum_pagu_cs_limit_3": cs_limit[3],
                "psum_pagu_cs_inc_0": cs_inc[0],
                "psum_pagu_cs_inc_1": cs_inc[1],
                "psum_pagu_cs_inc_2": cs_inc[2],
                "psum_pagu_cs_inc_3": cs_inc[3],
                "psum_pagu_loop_limit_0": loop_limit[0],
                "psum_pagu_loop_limit_1": loop_limit[1],
                "psum_pagu_loop_limit_2": loop_limit[2],
                "psum_pagu_loop_limit_3": loop_limit[3],
                "psum_pagu_loop_limit_4": loop_limit[4],
                "psum_pagu_addr_inc_0": addr_inc[0],
                "psum_pagu_addr_inc_1": addr_inc[1],
                "psum_pagu_addr_inc_2": addr_inc[2],
                "psum_pagu_addr_inc_3": addr_inc[3],
                "psum_pagu_addr_inc_4": addr_inc[4],
                "gemm_mode": int(getattr(layer_params, "gemm_mode", 0)),
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
        storage[sdict["psum_delay"]] = int(layer_params.psum_delay)
        storage[sdict["needed_refreshes"]] = int(layer_params.needed_refreshes_mx[layer_repetition][0])
        storage[sdict["used_X_cluster"]] = int(layer_params.used_X_cluster)
        storage[sdict["used_Y_cluster"]] = int(layer_params.used_Y_cluster)
        storage[sdict["needed_Iact_writes"]] = int(layer_params.needed_Iact_writes)
        storage[sdict["used_psum_per_PE"]] = int(layer_params.used_psum_per_PE)
        storage[sdict["used_iact_addr_per_PE"]] = int(layer_params.used_iact_addr_per_PE)
        storage[sdict["used_wght_addr_per_PE"]] = int(layer_params.used_wght_addr_per_PE)
        storage[sdict["used_iact_per_PE"]] = int(layer_params.used_iact_per_PE)
        storage[sdict["iact_addr_len"]] = int(layer_params.iact_addr_len)
        storage[sdict["iact_data_len"]] = int(layer_params.iact_data_len)
        storage[sdict["strideX"]] = int(layer_params.strideX)
        storage[sdict["strideY"]] = int(layer_params.strideY)
        storage[sdict["skipIact"]] = int(layer_params.skipIact)
        storage[sdict["skipWght"]] = int(layer_params.skipWght)
        storage[sdict["skipPsum"]] = int(layer_params.skipPsum)
        storage[sdict["usePEs"]] = computing_pes
        storage[sdict["kernel_per_pe_cluster"]] = int(layer_params.kernel_per_pe_cluster)
        storage[sdict["gemm_mode"]] = int(getattr(layer_params, "gemm_mode", 0))

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
        n_quant = params.QUANT_AMOUNT

        offset_mask = (1 << offset_w) - 1
        exp_mask = (1 << exp_w) - 1
        mant_mask = (1 << mant_w) - 1
        mant_shift = offset_w + exp_w

        quantize = layer_params.quantize
        offsets = layer_params.offset

        if isinstance(quantize, dict):
            q_arr = np.asarray([quantize[f] for f in range(n_quant)], dtype=np.int64)
        else:
            q_arr = np.asarray(quantize, dtype=np.int64)[:n_quant]

        if isinstance(offsets, dict):
            o_arr = np.asarray([offsets[f] for f in range(n_quant)], dtype=np.int64)
        else:
            o_arr = np.asarray(offsets, dtype=np.int64)[:n_quant]

        entries = (
            (o_arr.astype(np.uint64) & np.uint64(offset_mask))
            | ((q_arr[:, 1].astype(np.uint64) & np.uint64(exp_mask)) << np.uint64(offset_w))
            | ((q_arr[:, 0].astype(np.uint64) & np.uint64(mant_mask)) << np.uint64(mant_shift))
        )

        packed_bitstream = 0
        for f, entry in enumerate(entries.tolist()):
            packed_bitstream |= entry << (f * entry_w)

        total_bits = n_quant * entry_w
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

        used_y = int(layer_params.used_Y_cluster)
        multi_pe_y = int(layer_params.used_PEs_Y) > 1

        if params.SERIAL:
            flat_values = []
            for cl_y in range(clusters_y):
                val = self._iact_router_code(cl_y, used_y, multi_pe_y)
                flat_values.extend([val] * (clusters_x * num_glb_iact))
            return self._pack_router_values(flat_values, params.DMA_BITWIDTH, params.Iact_Router_Bits)

        storage = [[[0] * num_glb_iact for _ in range(clusters_y)] for _ in range(clusters_x)]
        for cl_y in range(clusters_y):
            val = self._iact_router_code(cl_y, used_y, multi_pe_y)
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
            return (
                []
                if params.SERIAL
                else [[[[] for _ in range(psum_routers)] for _ in range(clusters_y)] for _ in range(clusters_x)]
            )

        used_y = int(layer_params.used_Y_cluster)
        active_x_limit = (layer_params.iact_size_x + layer_params.add_up) * layer_params.different_kernels_per_calculation
        y_packages_limit = int(layer_params.Y_Cluster_Packages)
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