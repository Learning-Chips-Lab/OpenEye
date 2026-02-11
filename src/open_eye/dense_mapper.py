# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
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

import math
import logging
import open_eye.stream_dicts as strdic
from open_eye.layer_mapper import LayerMapper
from open_eye.iact_stream_mapper import DenseIactStreamMapper
from open_eye.wght_stream_mapper import DenseWghtStreamMapper
from open_eye.psum_stream_mapper import DensePsumStreamMapper

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
        # Create specialized stream mappers for dense layer operations
        input_mapper = DenseIactStreamMapper(params, layer_params, layer_repetition, dram_layer_content[0], sparse_iacts)
        weight_mapper = DenseWghtStreamMapper(params, layer_params, layer_repetition, dram_layer_content[1], sparse_wghts)
        bias_mapper = DensePsumStreamMapper(params, layer_params, layer_repetition, dram_layer_content[2])

        # Initialize parent LayerMapper with the dense-specific mappers
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, input_mapper, weight_mapper, bias_mapper)

    def write_working_parameters(self, params, layer_params, layer_repetition):
        """Generate working parameters and register configuration for the dense layer.

        Packs all layer configuration parameters into register format for hardware transmission.
        For dense layers, this includes matrix multiplication settings, PE allocation, and
        stream configuration. Handles both serial (DMA) and parallel transmission modes.

        The method computes the PE enable bitmap to indicate which processing elements
        are active for this layer, and generates four DMA transmission words containing
        packed configuration data.

        Args:
            params: Hardware configuration parameters
            layer_params: Layer-specific parameters for the dense layer
            layer_repetition: Current repetition index for the layer

        Returns:
            list: Packed register values ready for transmission to hardware. In serial mode,
                  returns DMA words including register configuration, PE enable bitmap, and
                  router configurations. In parallel mode, returns indexed configuration values.
        """
        # Initialize storage structure based on communication mode (serial vs parallel)
        if (params.SERIAL):
            # Serial mode: list indexed by stream_serial_dict
            storage = [[] for b in range(len(strdic.stream_serial_dict))]
        else:
            # Parallel mode: list indexed by status_dict
            storage = [[] for b in range(len(strdic.status_dict))]

        needed_refreshes = 1

        # Determine whether to skip partial sum (bias) loading based on layer repetition
        # Skip if not at the start of a new input activation transmission cycle
        if ((layer_repetition % layer_params.needed_iact_transmissions) == 0) :
            layer_params.skipPsum = 0  # Load biases at the start of new iact cycle
        else :
            layer_params.skipPsum = 1  # Skip bias loading for subsequent repetitions

        counter = 0
        computing_pes = 0

        # === PE ALLOCATION BITMAP ===
        # Create a bitmap indicating which PEs are active for computation
        # Each bit represents one PE across all clusters (X, Y, PE_Y, PE_X)
        for x in range(params.Clusters_X):
            for y in range(params.Clusters_Y):
                for pe_y in range(params.PEs_Y):
                    for pe_x in range(params.PEs_X):
                        # Set bit if this PE is used for computation
                        if(layer_params.computing_mx[x][y][pe_y][pe_x]== 1):
                            computing_pes = computing_pes + 2**(counter)
                        counter = counter + 1

        # Format the PE bitmap as a binary string (reversed for transmission)
        total_bits = params.Clusters_X * params.Clusters_Y * params.PEs_Y * params.PEs_X
        bitstring = format(computing_pes, f"0{total_bits}b")[::-1]
        if (params.SERIAL):
            # === SERIAL MODE: DMA BITSTREAM GENERATION ===
            # Pack all parameters into bit-packed DMA lines for serial transmission
            dma_line = 0
            dma_storage = []
        # === REGISTER PACKING ===
        # Pack all layer configuration parameters into hardware register format
        # Uses pack_registers() utility from regmap_pack module
            from regmap_pack import pack_registers
            words = pack_registers({
            "wght_cycles_reg": layer_params.needed_refreshes_mx[layer_repetition][0],
            "stride_x_reg": layer_params.strideX,
            "stride_y_reg": layer_params.strideY,
            "skipIact_reg": layer_params.skipIact,
            "skipWght_reg": layer_params.skipWght,
            "skipPsum_reg": layer_params.skipPsum,
            "psum_delay_reg": layer_params.psum_delay,
            "kernel_per_pe_cluster_reg": layer_params.kernel_per_pe_cluster,
            "kernel_size": 1,
            "x_lines_reg": layer_params.iact_x_lines,
            "needed_wght_cycles_reg": 1,
            "needed_cycles_reg": layer_params.needed_refreshes_mx[layer_repetition][0],
            "iact_converter_buffer_addr_max_cycles": layer_params.needed_standing_cycles,
            "iact_channels_per_pe": layer_params.used_iact_per_PE,
            "fc_size_reg": layer_params.iact_size_x,
            "iact_size_x": 1,
            "iact_size_y": layer_params.iact_size_y,
            "iact_needed_cycles": layer_params.iact_stream_cycles,
            "kernels_per_calc": layer_params.different_kernels_per_calculation,
            "y_lines_per_calc": layer_params.y_lines_per_calculation,
            "output_cycles": layer_params.output_cycles, 
            "store_in_psum": layer_params.store_in_psum,
            "max_pooling": layer_params.max_pooling,
            "fully_connected_layer": layer_params.fully_connected,
            "choose_iact_buffer_output": layer_params.choose_iact_storage_output,
            "choose_iact_buffer_input": layer_params.choose_iact_storage_input,
            "iact_channels_per_pe_next_layer": 1,
            "needed_psum_storage_cycles_reg": layer_params.psum_storage_cycles,
            "iact_channel_max_cycles": layer_params.diff_iact_layer,
            "input_activations_reg": layer_params.used_iact_per_PE,
            "filters_reg": layer_params.used_psum_per_PE,
            "needed_x_cls_reg": layer_params.used_X_cluster,
            "needed_y_cls_reg": 1,
            "needed_iact_cycles_reg": layer_params.needed_Iact_writes,
            "wght_addr_len_reg": layer_params.used_wght_addr_per_PE,
            "iact_addr_len_reg": layer_params.used_iact_addr_per_PE,
            "send_data_out": layer_params.send_values_out,
            "needed_iact_buffer_words_reg": layer_params.needed_iact_buffer_words,
            "add_up_reg":layer_params.add_up,
            "iact_x_line_repetitions_reg":layer_params.iact_x_line_repetitions
            })
            """
            # === DMA Line 1: Main configuration parameters ===
            # Bit packing: combine multiple parameters into a single 64-bit word
            dma_line = params.data_mode + ((layer_params.realfactor) << 1)      # bits 0-5: data mode and precision
            dma_line = dma_line + (params.autofunction << 6)                    # bit 6: auto-function enable
            dma_line = dma_line + (params.poolingmode << 7)                     # bit 7: pooling mode
            # Note: bits 8-15 reserved for refresh count (commented out)
            dma_line = dma_line + (layer_params.used_X_cluster << 16)           # bits 16-17: active X clusters
            dma_line = dma_line + (layer_params.used_Y_cluster << 18)           # bits 18-21: active Y clusters
            dma_line = dma_line + (layer_params.needed_Iact_writes << 22)       # bits 22-25: iact writes needed
            dma_line = dma_line + (layer_params.used_psum_per_PE << 26)         # bits 26-31: psum values per PE
            dma_line = dma_line + (layer_params.used_iact_addr_per_PE << 32)    # bits 32-35: iact addresses per PE
            dma_line = dma_line + (layer_params.used_wght_addr_per_PE << 36)    # bits 36-40: weight addresses per PE
            dma_line = dma_line + (layer_params.used_iact_per_PE << 41)         # bits 41-45: iact values per PE
            dma_line = dma_line + (layer_params.send_values_out << 46)          # bits 46+: enable output transmission
            dma_storage.append(dma_line)

            # === DMA Line 2: Stride, skip flags, and transmission parameters ===
            dma_line = 0
            dma_line = dma_line + (layer_params.needed_wght_transmissions)      # bits 0-9: weight transmissions needed
            dma_line = dma_line + (layer_params.strideY << 10)                  # bits 10-13: stride Y (also includes X)
            dma_line = dma_line + (layer_params.skipIact << 14)                 # bit 14: skip iact loading flag
            dma_line = dma_line + (layer_params.skipWght << 15)                 # bit 15: skip weight loading flag
            dma_line = dma_line + (layer_params.skipPsum << 16)                 # bit 16: skip psum/bias loading flag
            dma_line = dma_line + (layer_params.psum_delay << 17)               # bits 17-20: psum accumulation delay
            dma_line = dma_line + (layer_params.kernel_per_pe_cluster << 21)    # bits 21-24: kernels per PE cluster
            # Note: bits 25-28 reserved for kernel size (commented out)
            dma_line = dma_line + (layer_params.iact_x_lines << 29)             # bits 29-36: input feature map X lines
            dma_line = dma_line + (math.ceil(layer_params.filters/16) << 37)    # bits 37+: filter count (16-aligned)
            dma_storage.append(dma_line)

            # === DMA Line 3: Input dimensions and streaming parameters ===
            # Using bitwise OR for clarity in this packed word
            dma_line = (layer_params.needed_standing_cycles << 56) | \
                       (layer_params.used_channels << 48) | \
                       (layer_params.iact_size_y << 32) | \
                       (layer_params.iact_size_x << 16) | \
                       layer_params.iact_stream_cycles
            dma_storage.append(dma_line)

            # === DMA Line 4: Layer control and storage selection ===
            dma_line = 0
            dma_line = math.ceil(layer_params.diff_iact_layer)                              # bits 0-7: iact layer difference
            dma_line = dma_line + math.ceil(layer_params.diff_iact_layer_next_layer << 8)   # bits 8-15: next layer iact diff
            dma_line = dma_line + math.ceil(layer_params.choose_iact_storage_input << 16)   # bit 16: input storage bank
            dma_line = dma_line + math.ceil(layer_params.choose_iact_storage_output << 17)  # bit 17: output storage bank
            dma_line = dma_line + math.ceil(layer_params.fully_connected << 18)             # bit 18: fully connected flag
            dma_line = dma_line + math.ceil(layer_params.max_pooling << 19)                 # bit 19: max pooling flag
            dma_line = dma_line + math.ceil(layer_params.store_in_psum << 20)               # bit 20: store in psum buffer
            dma_line = dma_line + math.ceil(layer_params.output_cycles << 21)               # bits 21-28: output cycles
            dma_line = dma_line + math.ceil(layer_params.y_lines_per_calculation << 29)     # bits 29-32: Y lines per calc
            dma_line = dma_line + math.ceil(layer_params.different_kernels_per_calculation << 33)  # bits 33+: kernels per calc
            dma_storage.append(dma_line)"""

        # === SERIAL MODE: DMA TRANSMISSION ===
        if (params.SERIAL):
            # Use modern register packing approach (commented code above is legacy)
            dma_storage = words

            # === PE ENABLE BITMAP TRANSMISSION ===
            # Split PE bitmap into AXI-width segments and append to DMA stream
            for x in range(math.ceil(params.PE_Complete/params.DMA_Bit_AXI)):
                segment = bitstring[x*params.DMA_Bit_AXI:(x+1)*params.DMA_Bit_AXI]
                dma_storage.append(int(segment[::-1], 2))

            # === ROUTER CONFIGURATION TRANSMISSION ===
            # Append router configurations for all three data paths
            dma_storage.extend(self.write_router_iact(params, layer_params))
            dma_storage.extend(self.write_router_wght(params, layer_params))
            dma_storage.extend(self.write_router_psum(params, layer_params))

            storage = dma_storage

        # === PARALLEL MODE: DIRECT PARAMETER ASSIGNMENT ===
        else:
            # === PARALLEL MODE: DICTIONARY-BASED CONFIGURATION ===
            # Store each parameter individually by name for parallel register access
            storage[strdic.status_dict["data_mode"]] = params.data_mode
            storage[strdic.status_dict["realfactor"]] = layer_params.realfactor
            storage[strdic.status_dict["autofunction"]] = params.autofunction
            storage[strdic.status_dict["poolingmode"]] = params.poolingmode
            storage[strdic.status_dict["psum_delay"]] = layer_params.psum_delay
            storage[strdic.status_dict["needed_refreshes"]] = needed_refreshes
            storage[strdic.status_dict["used_X_cluster"]] = layer_params.used_X_cluster
            storage[strdic.status_dict["used_Y_cluster"]] = layer_params.used_Y_cluster
            storage[strdic.status_dict["needed_Iact_writes"]] = layer_params.needed_Iact_writes
            storage[strdic.status_dict["used_psum_per_PE"]] = layer_params.used_psum_per_PE
            storage[strdic.status_dict["used_iact_addr_per_PE"]] = layer_params.used_iact_addr_per_PE
            storage[strdic.status_dict["used_wght_addr_per_PE"]] = layer_params.used_wght_addr_per_PE
            storage[strdic.status_dict["used_iact_per_PE"]] = layer_params.used_iact_per_PE
            storage[strdic.status_dict["iact_addr_len"]] = layer_params.iact_addr_len
            storage[strdic.status_dict["iact_data_len"]] = layer_params.iact_data_len
            storage[strdic.status_dict["strideX"]] = layer_params.strideX
            storage[strdic.status_dict["strideY"]] = layer_params.strideY
            storage[strdic.status_dict["skipIact"]] = layer_params.skipIact
            storage[strdic.status_dict["skipWght"]] = layer_params.skipWght
            storage[strdic.status_dict["skipPsum"]] = layer_params.skipPsum
            storage[strdic.status_dict["usePEs"]] = int(computing_pes,2)
            storage[strdic.status_dict["kernel_per_pe_cluster"]] = layer_params.kernel_per_pe_cluster
            # Router configurations as nested structures for parallel access
            storage[strdic.status_dict["router_iact"]] = self.write_router_iact(params, layer_params)
            storage[strdic.status_dict["router_wght"]] = self.write_router_wght(params, layer_params)
            storage[strdic.status_dict["router_psum"]] = self.write_router_psum(params, layer_params)
        return storage

    def write_quantize(self, params, layer_params, layer_repetition):
        """Generate quantization parameters for the dense layer.

        Packs quantization values into DMA transmission format. Each DMA line contains
        two quantization parameter pairs, with 16 pairs total (32 values). These parameters
        are used for converting between floating-point and fixed-point representations.

        Args:
            params: Hardware configuration parameters
            layer_params: Layer-specific parameters containing quantization values
            layer_repetition: Current repetition index for the layer

        Returns:
            list: 16 DMA words containing packed quantization parameters, where each word
                  contains two quantization parameter pairs at specific bit offsets
                  (0, 25, 32, 57).
        """
        dma_line = 0
        dma_storage = []

        # Pack quantization parameters: 2 parameter pairs per DMA word
        # Each parameter is [scale_factor, zero_point] for quantization
        for f in range(math.ceil(16)):
            dma_line = 0
            # First quantization pair: scale and zero-point
            dma_line = dma_line + (layer_params.quantize[2*f][0] << 0)      # bits 0-24: first scale factor
            dma_line = dma_line + (layer_params.quantize[2*f][1] << 25)     # bits 25-31: first zero-point
            # Second quantization pair: scale and zero-point
            dma_line = dma_line + (layer_params.quantize[2*f+1][0] << 32)   # bits 32-56: second scale factor
            dma_line = dma_line + (layer_params.quantize[2*f+1][1] << 57)   # bits 57-63: second zero-point

            dma_storage.append(dma_line)
        return dma_storage
    
    def write_offset(self, params, layer_params, layer_repetition):
        """Generate offset parameters for the dense layer.

        Packs 32 offset values into DMA transmission format. Each DMA line contains
        8 offset values packed at 8-bit intervals. Offset values are used for
        bias correction or activation adjustments.

        Args:
            params: Hardware configuration parameters
            layer_params: Layer-specific parameters containing offset values
            layer_repetition: Current repetition index for the layer

        Returns:
            list: 4 DMA words containing packed offset parameters, where each word
                  contains 8 consecutive offset values at 8-bit intervals (0, 8, 16,
                  24, 32, 40, 48, 56).
        """
        dma_line = 0
        dma_storage = []

        # Pack 32 offset values into 4 DMA words (8 values per word)
        # Each offset value occupies 8 bits
        for f in range(math.ceil(32/8)):
            dma_line = 0
            # Pack 8 consecutive offset values at 8-bit intervals
            dma_line = dma_line + (layer_params.offset[8*f] << 0)       # bits 0-7: offset 0
            dma_line = dma_line + (layer_params.offset[8*f+1] << 8)     # bits 8-15: offset 1
            dma_line = dma_line + (layer_params.offset[8*f+2] << 16)    # bits 16-23: offset 2
            dma_line = dma_line + (layer_params.offset[8*f+3] << 24)    # bits 24-31: offset 3
            dma_line = dma_line + (layer_params.offset[8*f+4] << 32)    # bits 32-39: offset 4
            dma_line = dma_line + (layer_params.offset[8*f+5] << 40)    # bits 40-47: offset 5
            dma_line = dma_line + (layer_params.offset[8*f+6] << 48)    # bits 48-55: offset 6
            dma_line = dma_line + (layer_params.offset[8*f+7] << 56)    # bits 56-63: offset 7
            dma_storage.append(dma_line)
        return dma_storage
        
    def write_offset(self, params, layer_params, layer_repetition):
        """Generate offset parameters for the layer.

        Packs 32 offset values into DMA transmission format. Each DMA line contains
        8 offset values packed at 8-bit intervals.

        Args:
            params: Hardware configuration parameters
            layer_params: Layer-specific parameters containing offset values
            layer_repetition: Current repetition index for the layer

        Returns:
            list: 4 DMA words containing packed offset parameters, where each word
                  contains 8 consecutive offset values at 8-bit intervals.
        """
        dma_line = 0
        dma_storage = []
        for f in range(math.ceil(32/8)):
            dma_line = 0
            dma_line = dma_line + (layer_params.offset[8*f] << 0)
            dma_line = dma_line + (layer_params.offset[8*f+1] << 8)
            dma_line = dma_line + (layer_params.offset[8*f+2] << 16)
            dma_line = dma_line + (layer_params.offset[8*f+3] << 24)
            dma_line = dma_line + (layer_params.offset[8*f+4] << 32)
            dma_line = dma_line + (layer_params.offset[8*f+5] << 40)
            dma_line = dma_line + (layer_params.offset[8*f+6] << 48)
            dma_line = dma_line + (layer_params.offset[8*f+7] << 56)
            dma_storage.append(dma_line)
        return dma_storage

    def write_router_iact(self, params, layer_params):
        """Configure input activation (iact) router settings for dense layer.

        Generates routing configuration for distributing input activations across the
        cluster array for dense layer operations. For dense layers, routing is simplified
        compared to convolutional layers, using broadcast mode (routing value 1) for
        clusters within the first 4 rows.

        Routing values:
        - 1: Broadcast input data to this cluster (for cl_y < 4)
        - 0: No routing (implicit for cl_y >= 4)

        Args:
            params: Hardware configuration parameters including cluster dimensions
            layer_params: Layer-specific parameters for the dense layer

        Returns:
            In serial mode: list of DMA words with packed router values
            In parallel mode: 3D list [cluster_x][cluster_y][router] of routing values
        """
        line = 0
        # Initialize storage based on mode
        if(params.SERIAL):
            storage = []  # Serial: list of packed DMA words
        else:
            # Parallel: 3D array [cluster_x][cluster_y][router]
            storage = [[[[] for c in range(params.NUM_GLB_IACT)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]

        router_cycle = 0  # Track position within current DMA word

        # Iterate through all clusters and routers
        for cl_y in range(params.Clusters_Y):
            for cl_x in range(params.Clusters_X):
                for router in range(params.NUM_GLB_IACT):
                    if(params.SERIAL):
                        # Routing value 1: broadcast input data to this cluster
                        line = line + (1 << (params.Iact_Router_Bits * router_cycle))
                    else:
                        # Parallel mode: store routing value directly
                        storage[cl_x][cl_y][router] = 1

                    router_cycle = router_cycle + 1

                    # Check if current DMA word is full
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_Bit_AXI/params.Iact_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0

        # Append any remaining partial DMA word
        if(params.SERIAL and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)

        return storage

    def write_router_wght(self, params, layer_params):
        """Configure weight (wght) router settings for dense layer.

        Generates routing configuration for distributing weights across the cluster array
        for dense layer operations. For dense layers, all clusters use local weight data
        (routing value 0), as each cluster typically processes different weight rows.

        Routing values:
        - 0: Use local weight data (all clusters)

        Args:
            params: Hardware configuration parameters including cluster dimensions
            layer_params: Layer-specific parameters for the dense layer

        Returns:
            In serial mode: list of DMA words with packed router values
            In parallel mode: 3D list [cluster_x][cluster_y][router] of routing values
        """
        line = 0
        # Initialize storage based on mode
        if(params.SERIAL):
            storage = []  # Serial: list of packed DMA words
        else:
            # Parallel: 3D array [cluster_x][cluster_y][router]
            storage = [[[[] for c in range(params.Wght_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]

        router_cycle = 0  # Track position within current DMA word

        # Iterate through all clusters and routers
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Wght_Routers):
                    if(params.SERIAL):
                        # Routing value 0: use local weight data (no routing/broadcast)
                        # All clusters load their own weights for dense layers
                        line = line + (0 << (params.Wght_Router_Bits * router_cycle))
                    else:
                        # Parallel mode: store routing value directly
                        storage[cl_x][cl_y][router] = 0

                    router_cycle = router_cycle + 1

                    # Check if current DMA word is full
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_Bit_AXI/params.Wght_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0

        # Append any remaining partial DMA word
        if(params.SERIAL and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)
        return storage

    def write_router_psum(self, params, layer_params):
        """Configure partial sum (psum) router settings for dense layer.

        Generates routing configuration for collecting and forwarding partial sums through
        the cluster array for dense layer operations. The routing depends on PE utilization
        and cluster Y organization, determining how partial sums are accumulated.

        Routing values:
        - 2: Pass-through cluster (middle of Y-cluster group)
        - 3: Final accumulation cluster (last in Y-cluster group)
        - 4: Output cluster (single PE per cluster mode)
        - 5: First accumulation cluster (first in Y-cluster group)

        Args:
            params: Hardware configuration parameters including cluster dimensions
            layer_params: Layer-specific parameters including PE and cluster usage

        Returns:
            In serial mode: list of DMA words with packed router values
            In parallel mode: 3D list [cluster_x][cluster_y][router] of routing values
        """
        line = 0
        # Initialize storage based on mode
        if(params.SERIAL):
            storage = []  # Serial: list of packed DMA words
        else:
            # Parallel: 3D array [cluster_x][cluster_y][router]
            storage = [[[[] for c in range(params.Psum_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]

        router_cycle = 0  # Track position within current DMA word

        # Iterate through all clusters and routers
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Psum_Routers):
                    # Determine routing mode based on PE usage pattern
                    if((layer_params.ceil_used_PE_per_clm == 1) | (params.Clusters_Y==1)):
                        # Single PE per cluster: direct output (no accumulation chain)
                        if(params.SERIAL):
                            line = line + (4 << (params.Psum_Router_Bits * router_cycle))
                        else:
                            storage[cl_x][cl_y][router] = 4
                    else:
                        # Multiple PEs per cluster: build accumulation chain along Y-axis
                        if((cl_y % layer_params.used_Y_cluster) == 0):
                            # First cluster in Y-group: start of accumulation chain
                            if(params.SERIAL):
                                line = line + (5 << (params.Psum_Router_Bits * router_cycle))
                            else:
                                storage[cl_x][cl_y][router] = 5
                        else:
                            if(((cl_y + 1) %  layer_params.used_Y_cluster) == 0):
                                # Last cluster in Y-group: final accumulation and output
                                if(params.SERIAL):
                                    line = line + (3 << (params.Psum_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 3
                            else:
                                # Middle cluster in Y-group: pass-through accumulation
                                if(params.SERIAL):
                                    line = line + (2 << (params.Psum_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 2

                    router_cycle = router_cycle + 1

                    # Check if current DMA word is full
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_Bit_AXI/params.Psum_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0

        # Append any remaining partial DMA word
        if(params.SERIAL and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)
        return storage

    def write_psum_data_glb(self, params, layer_params, layer_repetition, dram, cl_y, router, cycle):
        """Write partial sum data to global buffer format for dense layer.

        Converts partial sum (bias) data from DRAM format to the global buffer format
        expected by the hardware. For dense layers, this typically initializes bias values
        to zero, with scaling based on the configured bitwidths.

        Args:
            params: Hardware configuration parameters including bitwidth settings
            layer_params: Layer-specific parameters including filter/neuron counts
            layer_repetition: Current repetition index for the layer
            dram: Source data from DRAM containing bias values
            cl_y: Cluster Y coordinate (currently unused in implementation)
            router: Router index (currently unused in implementation)
            cycle: Cycle index (currently unused in implementation)

        Returns:
            In serial mode: list of DMA words containing scaled bias values (currently zeros)
            In parallel mode: 2D list [cluster_x][data_index] of scaled bias values
        """
        # Initialize storage structure
        storage, line = self.initialize_storage(params.SERIAL), 0

        # Process partial sum (bias) data for each output neuron/filter
        for part_data_num in range(math.ceil(self.layer_params.used_psum_per_PE/2)):
            if(part_data_num < self.layer_params.used_psum_per_PE):
                # Write bias value for each cluster X
                for cl_x in range(self.params.Clusters_X):
                    if(self.params.SERIAL):
                        # Calculate scaled bias value (currently initialized to 0)
                        # Scaling factor: 2^(IACT_Bitwidth + WGHT_Bitwidth - 1)
                        scaled_bias = int(round(float((2**(params.IACT_Bitwidth + params.WGHT_Bitwidth - 1)) * 0)))
                        line = line + (scaled_bias << (params.PSUM_Bitwidth * cl_x))
                        storage.append(line)
                        line = 0
                    else:
                        # Parallel mode: append directly to cluster-specific storage
                        scaled_bias = int(round(float((2**(params.IACT_Bitwidth + params.WGHT_Bitwidth - 1)) * 0)))
                        storage[cl_x].append(scaled_bias)
            else:
                line = 0
        return storage
