# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Dense (Fully Connected) layer mapper for the OpenEye accelerator.

This module provides the DenseMapper class, which handles the mapping and configuration
of fully connected (dense) layers for execution on the OpenEye hardware accelerator.
It coordinates input activation, weight, and bias stream mappers to generate the
necessary register configurations and data layouts for dense layer operations.

The mapper supports both serial (DMA-based) and parallel transmission modes, and
generates router configurations for distributing data across the accelerator's
cluster array.
"""

import logging
import math
import numpy as np
import open_eye.stream_dicts as strdic
from open_eye.generator import pack_registers
from open_eye.iact_stream_mapper import DenseIactStreamMapper
from open_eye.layer_mapper import LayerMapper
from open_eye.psum_stream_mapper import DensePsumStreamMapper
from open_eye.wght_stream_mapper import DenseWghtStreamMapper

logger = logging.getLogger("cocotb")


class DenseMapper(LayerMapper):
    """Mapper for fully connected (dense) layers in the OpenEye accelerator.

    This class handles the mapping and configuration of fully connected layers by
    coordinating input activation (iact), weight (wght), and bias (psum) stream mappers.
    It generates register configurations, router settings, and data layouts optimized
    for matrix multiplication operations in dense layers.

    Dense layers differ from convolutional layers in that they perform full matrix
    multiplication without spatial convolution, typically used in classification heads
    or fully connected network sections.

    Args:
        params: Hardware configuration parameters defining the accelerator architecture
        layer_params: Layer-specific parameters including input/output dimensions
        layer_repetition: Current repetition index for layers that execute multiple times
        dram_layer_content: Tuple containing (input_data, weight_data, bias_data) from DRAM
        sparse_iacts: Sparsity information for input activations
        sparse_wghts: Sparsity information for weights

    Attributes:
        Inherits all attributes from LayerMapper base class, including:
- input_mapper: DenseIactStreamMapper for input activation data
- weight_mapper: DenseWghtStreamMapper for weight data
- bias_mapper: DensePsumStreamMapper for bias/partial sum data
    """

    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts):
        input_mapper = DenseIactStreamMapper(
            params, layer_params, layer_repetition, dram_layer_content[0][0], sparse_iacts
        )
        weight_mapper = DenseWghtStreamMapper(
            params, layer_params, layer_repetition, dram_layer_content[1], sparse_wghts
        )
        bias_mapper = DensePsumStreamMapper(params, layer_params, layer_repetition, dram_layer_content[2])
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
    def _psum_router_code(cl_y: int, used_y: int) -> int:
        """Return the psum router mode code for a Y cluster index."""
        if used_y == 1:
            return 4
        if (cl_y % used_y) == 0:
            return 5
        if ((cl_y + 1) % used_y) == 0:
            return 3
        return 2

    def write_working_parameters(self, params, layer_params, layer_repetition):
        """Generate working parameters and register configuration for the dense layer.

        Packs all layer configuration parameters into register format for hardware transmission.
        For dense layers, this includes matrix multiplication settings, PE allocation, and
        stream configuration. Handles both serial (DMA) and parallel transmission modes.

        Args:
            params: Hardware configuration parameters
            layer_params: Layer-specific parameters for the dense layer
            layer_repetition: Current repetition index for the layer

        Returns:
            list: Packed register values ready for transmission to hardware. In serial mode,
                  returns DMA words including register configuration, PE enable bitmap, and
                  router configurations. In parallel mode, returns indexed configuration values.
        """
        # Skip partial sum (bias) loading unless at the start of a new iact transmission cycle
        layer_params.skipPsum = 0 if (layer_repetition % layer_params.needed_iact_transmissions) == 0 else 1

        # === PE ALLOCATION BITMAP ===
        computing_pes = self._compute_pe_bitmap(params, layer_params)

        # === SERIAL MODE: DMA BITSTREAM GENERATION ===
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
                "dw_mode": 0,
                "psum_q": int(layer_params.psum_delay),
                "kernel_per_pe_cluster_reg": int(layer_params.kernel_per_pe_cluster),
                "kernel_size_x": 1,
                "kernel_size_y": 1,
                "x_lines_reg": int(layer_params.iact_x_lines),
                "needed_wght_cycles": int(layer_params.needed_wght_transmissions),
                "needed_cycles": int(layer_params.needed_refreshes_mx[layer_repetition][0]),
                "trans_cycles_psum": int(layer_params.trans_cycles_psum),
                # GET_WGHT's exit condition (fsm_cycle == trans_cycles_wght - 1)
                # counts one DMA word per cycle, so this must equal the actual
                # weight stream's word count. Read directly from the weight mapper
                # rather than re-deriving conv's formula for dense's different layout.
                # GET_IACT's exit condition (fsm_cycle == trans_cycles_iact - 1)
                # counts one DMA word per cycle, so this must equal the actual iact
                # stream's word count. Read directly from the iact mapper.
                "trans_cycles_wght": int(
                    layer_params.trans_cycles_wght or len(self.WghtStreamCreator.get_wght_stream())
                ),
                "trans_cycles_iact": int(
                    layer_params.trans_cycles_iact or len(self.IactStreamCreator.get_iact_stream())
                ),
                # Process 5's weight-send phase (hdl/OpenEye_FPGA.v) uses this as
                # the upper bound of its wght_buffer_rd_addr sweep and the
                # window width of the wght_enable_i broadcast pulse to every PE.
                "iact_glb_writing_cycles": int(layer_params.iact_glb_writing_cycles),
                "used_wght_per_PE": math.ceil(layer_params.used_wght_per_PE / params.PARALLEL_MACS),
                "iact_converter_buffer_addr_max_cycles": int(layer_params.iact_converter_buffer_addr_max_cycles),
                "iact_channels_per_pe": int(layer_params.used_iact_per_PE),
                "fc_size_reg": int(layer_params.iact_size_x),
                "iact_size_x": math.ceil(layer_params.iact_size_x / 2),
                "iact_size_y": int(layer_params.iact_size_y),
                "iact_size_c": int(layer_params.iact_size_c),
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
                "iact_channels_per_pe_next_layer": 1,
                "needed_psum_storage_cycles_reg": int(layer_params.psum_storage_cycles),
                "iact_channel_max_cycles": int(layer_params.diff_iact_layer),
                "channel_div_trans": 1,
                "input_activations": int(layer_params.used_iact_per_PE),
                "filters": int(layer_params.used_psum_per_PE),
                "needed_x_cls_reg": int(layer_params.used_X_cluster),
                "needed_y_cls_reg": 1,
                "needed_iact_cycles_reg": int(layer_params.needed_Iact_writes),
                "wght_addr_len_reg": int(layer_params.used_wght_addr_per_PE),
                "send_data_out": int(layer_params.send_values_out),
                "needed_iact_buffer_words": int(layer_params.needed_iact_buffer_words),
                "iact_x_line_repetitions_reg": int(layer_params.iact_x_line_repetitions),
                "buffer_cycles_for_x_iact": int(layer_params.buffer_cycles_for_x_iact),
                "iact_x_per_cluster": int(layer_params.iact_x_per_cluster),
                "iact_x_add_up": int(layer_params.iact_x_add_up),
                "psum_x_all_cluster": int(layer_params.psum_x_all_cluster),
                "overhang_discrepancy": int(layer_params.overhang_discrepancy),
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
                "psum_pagu_cs_inc_0": cs_inc[0],
                "psum_pagu_cs_inc_1": cs_inc[1],
                "psum_pagu_cs_inc_2": cs_inc[2],
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
        needed_refreshes = 1

        storage[sdict["data_mode"]] = params.data_mode
        storage[sdict["realfactor"]] = layer_params.realfactor
        storage[sdict["autofunction"]] = params.autofunction
        storage[sdict["poolingmode"]] = params.poolingmode
        storage[sdict["psum_delay"]] = int(layer_params.psum_delay)
        storage[sdict["needed_refreshes"]] = needed_refreshes
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
        """Configure input activation (iact) router settings for dense layer.

        Routing values:
- 1: Broadcast input data to this cluster
- 0: Single-cluster default
        """
        clusters_x = params.Clusters_X
        clusters_y = params.Clusters_Y
        num_glb_iact = params.NUM_GLB_IACT

        if clusters_y == 1 and clusters_x == 1:
            return [0] if params.SERIAL else [[[[] for _ in range(num_glb_iact)]]]

        if params.SERIAL:
            total_entries = clusters_y * clusters_x * num_glb_iact
            return self._pack_router_values([1] * total_entries, params.DMA_BITWIDTH, params.Iact_Router_Bits)

        return [[[1] * num_glb_iact for _ in range(clusters_y)] for _ in range(clusters_x)]

    def write_router_wght(self, params, layer_params):
        """Configure weight (wght) router settings for dense layer.

        Routing values:
- 0: Use local weight data (all clusters)
        """
        clusters_x = params.Clusters_X
        clusters_y = params.Clusters_Y
        wght_routers = params.Wght_Routers

        if clusters_x == 1:
            return [] if params.SERIAL else [[[[] for _ in range(wght_routers)] for _ in range(clusters_y)]]

        if params.SERIAL:
            total_entries = clusters_x * clusters_y * wght_routers
            entries_per_word = params.DMA_BITWIDTH // params.Wght_Router_Bits
            return [0] * math.ceil(total_entries / entries_per_word)

        return [[[0] * wght_routers for _ in range(clusters_y)] for _ in range(clusters_x)]

    def write_router_psum(self, params, layer_params):
        """Configure partial sum (psum) router settings for dense layer.

        Routing values:
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
        per_y_entries = clusters_x * psum_routers

        if params.SERIAL:
            flat_values = []
            for cl_y in range(clusters_y):
                val = self._psum_router_code(cl_y, used_y)
                flat_values.extend([val] * per_y_entries)
            return self._pack_router_values(flat_values, params.DMA_BITWIDTH, params.Psum_Router_Bits)

        storage = [[[0] * psum_routers for _ in range(clusters_y)] for _ in range(clusters_x)]
        for cl_y in range(clusters_y):
            val = self._psum_router_code(cl_y, used_y)
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