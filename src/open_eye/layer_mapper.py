# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Base layer mapper for OpenEye hardware accelerator.

This module provides the LayerMapper base class, which serves as the foundation for
all layer-specific mappers in the OpenEye accelerator framework. It handles the common
functionality required to map neural network layers to hardware, including stream
generation, data routing, and parameter configuration.

Key Features:
- Abstract base class for layer-specific mappers (Conv, DW, Dense, Pooling)
- Unified interface for generating hardware configuration streams
- Support for both serial (DMA-based) and parallel communication modes
- Management of input activation, weight, and bias/partial sum data streams
- Quantization and offset parameter handling
- Shared PE bitmap and router word packing helpers

Architecture:
    The LayerMapper coordinates three specialized stream mappers:
- IactStreamCreator: Handles input activation data layout
- WghtStreamCreator: Handles weight data layout
- PsumStreamCreator: Handles bias/partial sum data layout

Datastream construction overview:
    make_stream() assembles one list per channel (see stream_dicts.py for
    the channel map). In serial mode the result is the exact 64-bit word
    sequence the OpenEye_FPGA DMA port consumes per layer:
    configuration words (regmap_pack) -> PE-enable bitmap -> router words
    -> iacts -> weights -> bias -> quantization -> offsets. In parallel
    mode the channels hold per-GLB-bank SPad word lists that
    rtl_test_utils.send_stream drives onto the OpenEye_Parallel ports.
    The dataflow is part of the configuration: gemm_mode = 0 keeps the
    row-stationary conv mapping, gemm_mode = 1 selects the
    output-stationary GEMM mapping (see gemm_mapper.py). The complete
    word-level documentation lives in
    doc/source/architecture/datastream_construction.md.

Subclasses:
- ConvMapper: Standard convolution layers
- DWMapper: Depthwise convolution layers
- DenseMapper: Fully connected/dense layers
- PoolingMapper: Pooling layers
"""

import logging
import math
import numpy as np
import open_eye.stream_dicts as strdic

logger = logging.getLogger("cocotb")


def _mp_generate_iact(creator, skip):
    return [] if (skip != 0 or creator is None) else creator.get_iact_stream()


def _mp_generate_wght(creator, skip):
    return [] if (skip != 0 or creator is None) else creator.get_wght_stream()


def _mp_generate_psum(creator, skip):
    return [] if (skip != 0 or creator is None) else creator.get_psum_stream()


class LayerMapper:
    """Abstract base class for mapping neural network layers to OpenEye hardware.

    This class provides the common infrastructure for translating high-level layer
    operations (convolution, pooling, dense, etc.) into low-level hardware streams
    that can be executed on the OpenEye accelerator. It manages the coordination of
    input activations, weights, and biases/partial sums, and generates the necessary
    control signals and data layouts.

    Attributes:
        params: Hardware configuration parameters (cluster layout, PE counts, etc.)
        layer_params: Layer-specific parameters (kernel size, stride, channels, etc.)
        layer_repetition (int): Current iteration index for layers requiring multiple passes
        dram_fmap (array): Feature map data stored in DRAM
        dram_weights (array): Weight data stored in DRAM
        dram_bias (array): Bias data stored in DRAM
        IactStreamCreator: Mapper for input activation streams
        WghtStreamCreator: Mapper for weight streams
        PsumStreamCreator: Mapper for bias/partial sum streams
        storage (list): Multi-channel storage for generated streams

    Note:
        Subclasses must implement:
- write_working_parameters(): Generate layer-specific hardware configuration
    """

    def __init__(
        self,
        params,
        layer_params,
        layer_repetition,
        dram_layer_content,
        inputstream_mapper=None,
        weightstream_mapper=None,
        bias_mapper=None,
    ):
        """Initialize the layer mapper with hardware parameters and stream mappers."""
        self.params = params
        self.layer_params = layer_params
        self.layer_repetition = layer_repetition

        # Extract DRAM data for this layer (safe for NumPy arrays and partial mock inputs in tests)
        if dram_layer_content is not None:
            n_items = len(dram_layer_content)
            fmap_slot = dram_layer_content[0] if n_items > 0 else None
            self.dram_fmap = fmap_slot[0] if (fmap_slot is not None and len(fmap_slot) > 0) else None
            self.dram_weights = dram_layer_content[1] if n_items > 1 else None
            self.dram_bias = dram_layer_content[2] if n_items > 2 else None
        else:
            self.dram_fmap = None
            self.dram_weights = None
            self.dram_bias = None

        # Store stream mapper instances for data path coordination
        self.IactStreamCreator = inputstream_mapper
        self.WghtStreamCreator = weightstream_mapper
        self.PsumStreamCreator = bias_mapper

        # Initialize storage structure based on communication mode
        num_channels = len(strdic.stream_serial_dict) if params.SERIAL else len(strdic.stream_parallel_dict)
        self.storage = [[] for _ in range(num_channels)]

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

    def make_stream(self):
        """Generate all hardware streams and configuration words for this layer pass."""
        pdict = strdic.stream_parallel_dict
        lparams = self.layer_params

        self.storage[pdict["iact"]] = _mp_generate_iact(self.IactStreamCreator, lparams.skipIact)
        self.storage[pdict["wght"]] = _mp_generate_wght(self.WghtStreamCreator, lparams.skipWght)
        self.storage[pdict["psum"]] = _mp_generate_psum(self.PsumStreamCreator, lparams.skipPsum)

        if self.params.SERIAL:
            for attr_name, key in (
                ("trans_cycles_iact", "iact"),
                ("trans_cycles_wght", "wght"),
                ("trans_cycles_psum", "psum"),
            ):
                section = self.storage[pdict[key]]
                if section:
                    setattr(lparams, attr_name, len(section))

        self.storage[pdict["status"]] = self.write_working_parameters(
            self.params, lparams, self.layer_repetition
        )
        self.storage[pdict["quantize"]] = self.write_quant_and_offset(
            self.params, lparams, self.layer_repetition
        )

        logger.info(
            "Stream finished: %d of %d",
            self.layer_repetition + 1,
            lparams.needed_total_transmissions,
        )

    def get_stream(self):
        """Retrieve the generated hardware streams."""
        return self.storage

    def write_working_parameters(self, params, layer_params, layer_repetition):
        """Generate layer-specific working parameters and hardware configuration.

        Must be overridden by concrete layer mapper subclasses.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement write_working_parameters()")

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

    def write_psum_data_glb(self, params, layer_params, layer_repetition, dram, cl_y, router, cycle):
        """Default partial sum GLB initialization stub for subclasses."""
        return self.initialize_storage(params.SERIAL)

    def get_psum_stream(self):
        """Generate partial sum data stream for bias/accumulator initialization."""
        if self.params.SERIAL:
            storage = []
        else:
            storage = [
                [[[] for _ in range(self.params.Psum_Routers)] for _ in range(self.params.Clusters_Y)]
                for _ in range(self.params.Clusters_X)
            ]

        refresh_range = self.layer_params.needed_refreshes_mx[self.layer_repetition]
        start_cycle, end_cycle = int(refresh_range[1]), int(refresh_range[2])

        for cl_y in range(self.params.Clusters_Y):
            for router in range(self.params.Psum_Routers):
                for cycle in range(start_cycle, end_cycle):
                    temp_storage = self.write_psum_data_glb(
                        self.params,
                        self.layer_params,
                        self.layer_repetition,
                        self.dram_bias,
                        cl_y,
                        router,
                        cycle,
                    )
                    if self.params.SERIAL:
                        storage.extend(temp_storage)
                    else:
                        for cl_x in range(self.params.Clusters_X):
                            storage[cl_x][cl_y][router].extend(temp_storage[cl_x])
        return storage

    def initialize_storage(self, serial):
        """Initialize an empty storage structure based on communication mode."""
        if serial:
            return []
        return [[] for _ in range(self.params.Clusters_X)]

    def line_reset(self, serial):
        """Reset a line buffer to its initial state based on communication mode."""
        return 0 if serial else [0, 0]

    def line_to_storage(self, serial, line, storage):
        """Append a completed line buffer to the storage structure."""
        if serial:
            storage.append(line)
        else:
            storage[0].append(line[0])
            storage[1].append(line[1])
        return storage