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

The mapper generates configuration data that controls:
    - Which PEs are active for the computation
    - How data is routed through the network-on-chip
    - When to skip loading cached data (iact/wght/psum)
    - Stride and kernel parameters for the convolution
    - Memory addressing and transmission cycles

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

    The mapper supports various convolution configurations including different kernel sizes,
    strides, and cluster distributions. It produces binary-formatted data streams compatible
    with both serial and parallel transmission modes.

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
        input_mapper = ConvIactStreamMapper(params, layer_params, layer_repetition, dram_layer_content[0], sparse_iacts)
        weight_mapper = ConvWghtStreamMapper(params, layer_params, layer_repetition, dram_layer_content[1], sparse_wghts)
        bias_mapper = ConvPsumStreamMapper(params, layer_params, layer_repetition, dram_layer_content[2])
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, input_mapper, weight_mapper, bias_mapper)

    def write_working_parameters(self, params, layer_params, layer_repetition):
        """Generate working parameters and register configuration for the layer.

        Packs all layer configuration parameters into register format for hardware transmission.
        This includes computation settings (stride, skip flags), resource allocation (clusters,
        PEs), and stream configuration. Handles both serial (DMA) and parallel transmission modes.

        Args:
            params: Hardware configuration parameters
            layer_params: Layer-specific parameters containing kernel size, stride, filters, etc.
            layer_repetition: Current repetition index for the layer

        Returns:
            list: Packed register values ready for transmission to hardware. In serial mode,
                  returns DMA words including register configuration, PE enable bitmap, and
                  router configurations. In parallel mode, returns indexed configuration values.
        """
        # Initialize storage structure based on communication mode
        if (params.SERIAL):
            storage = [[] for b in range(len(strdic.stream_serial_dict))]
        else:
            storage = [[] for a in range(len(strdic.status_dict))]

        # === SKIP FLAG DETERMINATION ===
        # Determine if partial sums should be loaded from memory or reused
        # Only load psums at the start of each iact transmission cycle
        if ((layer_repetition % layer_params.iact_transmissions_pe) == 0):
            layer_params.skipPsum = 0  # Load psums from memory
        else:
            layer_params.skipPsum = 1  # Reuse cached psums

        # === PE ALLOCATION BITMAP GENERATION ===
        # Create a bitmap indicating which PEs are active for this layer
        counter = 0
        computing_pes = 0

        # Iterate through all PEs in the accelerator grid
        for x in range(params.Clusters_X):
            for y in range(params.Clusters_Y):
                for pe_y in range(params.PEs_Y):
                    for pe_x in range(params.PEs_X):
                        # Set bit if this PE is used for computation
                        if layer_params.computing_mx[x][y][pe_y][pe_x] == 1:
                            computing_pes |= (1 << counter)
                        counter += 1

        # Convert bitmap to binary string (reversed for hardware consumption)
        total_bits = params.Clusters_X * params.Clusters_Y * params.PEs_Y * params.PEs_X
        bitstring = format(computing_pes, f"0{total_bits}b")[::-1]
        # === REGISTER PACKING ===
        # Pack all layer configuration parameters into hardware register format
        # Uses pack_registers() utility from regmap_pack module
        from regmap_pack import pack_registers
        words = pack_registers({
        "wght_cycles_reg": layer_params.needed_wght_transmissions,
        "stride_x": layer_params.strideX,
        "stride_y": layer_params.strideY,
        "skipIact_reg": layer_params.skipIact,
        "skipWght_reg": layer_params.skipWght,
        "skipPsum_reg": layer_params.skipPsum,
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
        "psum_size_x":math.ceil(layer_params.iact_size_x/layer_params.strideX),
        "psum_size_y": math.ceil(layer_params.iact_size_y/layer_params.strideY),
        "iact_needed_cycles": layer_params.iact_stream_cycles,
        "kernels_per_calc": layer_params.different_kernels_per_calculation,
        "y_lines_per_calc": layer_params.y_lines_per_calculation,
        "output_cycles": layer_params.output_cycles, 
        "store_in_psum": layer_params.store_in_psum,
        "max_pooling": layer_params.max_pooling,
        "fully_connected_layer": layer_params.fully_connected,
        "choose_iact_buffer_output": layer_params.choose_iact_storage_output,
        "choose_iact_buffer_input": layer_params.choose_iact_storage_input,
        "iact_channels_per_pe_next_layer": layer_params.diff_iact_layer_next_layer,
        "needed_psum_storage_cycles_reg": layer_params.psum_storage_cycles,
        "iact_channel_max_cycles": layer_params.diff_iact_layer,
        "input_activations": layer_params.used_iact_per_PE,
        "used_wght_per_PE": math.ceil(layer_params.used_wght_per_PE/params.PARALLEL_MACS),
        "filters": layer_params.used_psum_per_PE,
        "needed_x_cls_reg": layer_params.used_X_cluster,
        "needed_y_cls_reg": layer_params.used_Y_cluster,
        "needed_iact_cycles_reg": layer_params.needed_Iact_writes,
        "wght_addr_len_reg": layer_params.used_wght_addr_per_PE,
        "iact_channels_per_pe": layer_params.used_channels,
        "channel_div_trans": layer_params.channel_div_trans,
        "send_data_out": layer_params.send_values_out,
        "needed_iact_buffer_words": layer_params.needed_iact_buffer_words,
        "add_up_reg":layer_params.add_up,
        "iact_x_line_repetitions_reg":layer_params.iact_x_line_repetitions,
        "buffer_cycles_for_x_iact" : layer_params.buffer_cycles_for_x_iact,
        "start_param_array" : layer_params.start_param_array,
        "limit_increase" : layer_params.limit_increase,
        "lower_bound" : layer_params.lower_bound,
        "upper_bound" : layer_params.upper_bound,
        "initial_upper_limit": layer_params.initial_upper_limit,
        "iteration_for_kernels": layer_params.iteration_for_kernels,
        "fsm_psum_limit": layer_params.fsm_psum_limit,
        "cluster_per_conv_cycle": layer_params.cluster_per_conv_cycle,
        "iact_converter_max_cycles": layer_params.iact_converter_max_cycles,
        "iact_buffer_words_per_write": layer_params.iact_buffer_words_per_write,
        "iact_words_per_compute": layer_params.iact_words_per_compute,
        "pooling_mode": 0,
        "used_wght_per_PE": layer_params.used_wght_per_PE,
        "overhang_discrepancy": layer_params.overhang_discrepancy,
        "psum_output_words": layer_params.psum_output_words,
        "iact_read_limit_0" : layer_params.iact_read_limit_0,
        "iact_read_limit_1" : layer_params.iact_read_limit_1,
        "iact_read_limit_2" : layer_params.iact_read_limit_2,
        "iact_read_limit_3" : layer_params.iact_read_limit_3,
        "iact_read_limit_4" : layer_params.iact_read_limit_4,
        "iact_read_inc_0" : layer_params.iact_read_inc_0,
        "iact_read_inc_1" : layer_params.iact_read_inc_1,
        "iact_read_inc_2" : layer_params.iact_read_inc_2,
        "iact_read_inc_3" : layer_params.iact_read_inc_3,
        "iact_read_inc_4" : layer_params.iact_read_inc_4,
        "iact_write_limit_0" : layer_params.iact_write_limit_0,
        "iact_write_limit_1" : layer_params.iact_write_limit_1,
        "iact_write_limit_2" : layer_params.iact_write_limit_2,
        "iact_write_inc_0" : layer_params.iact_write_inc_0,
        "iact_write_inc_1" : layer_params.iact_write_inc_1,
        "iact_write_inc_2" : layer_params.iact_write_inc_2,
        "pagu_wght_limit" : layer_params.pagu_wght_limit,
        "psum_pagu_loop_limit_0" : layer_params.psum_pagu_loop_limit_0,
        "psum_pagu_loop_limit_1" : layer_params.psum_pagu_loop_limit_1,
        "psum_pagu_loop_limit_2" : layer_params.psum_pagu_loop_limit_2,
        "psum_pagu_loop_limit_3" : layer_params.psum_pagu_loop_limit_3,
        "psum_pagu_loop_limit_4" : layer_params.psum_pagu_loop_limit_4,
        "psum_pagu_addr_inc_0" : layer_params.psum_pagu_addr_inc_0,
        "psum_pagu_addr_inc_1" : layer_params.psum_pagu_addr_inc_1,
        "psum_pagu_addr_inc_2" : layer_params.psum_pagu_addr_inc_2,
        "psum_pagu_addr_inc_3" : layer_params.psum_pagu_addr_inc_3,
        "psum_pagu_addr_inc_4" : layer_params.psum_pagu_addr_inc_4,
        "gemm_mode": getattr(layer_params, "gemm_mode", 0)
        })

        # === SERIAL MODE: DMA TRANSMISSION ===
        if (params.SERIAL):
            # Use modern register packing approach (commented code above is legacy)
            dma_storage = words

            # === PE ENABLE BITMAP TRANSMISSION ===
            # Split PE bitmap into AXI-width segments and append to DMA stream
            for x in range(math.ceil(params.PE_Complete/params.DMA_BITWIDTH)):
                segment = bitstring[x*params.DMA_BITWIDTH:(x+1)*params.DMA_BITWIDTH]
                dma_storage.append(int(segment[::-1], 2))

            # === ROUTER CONFIGURATION TRANSMISSION ===
            # Append router configurations for all three data paths
            dma_storage.extend(self.write_router_iact(params, layer_params))
            dma_storage.extend(self.write_router_wght(params, layer_params))
            dma_storage.extend(self.write_router_psum(params, layer_params))

            storage = dma_storage

        # === PARALLEL MODE: DIRECT PARAMETER ASSIGNMENT ===
        else:
            storage[strdic.status_dict["data_mode"]] = params.data_mode
            storage[strdic.status_dict["realfactor"]] = layer_params.realfactor
            storage[strdic.status_dict["autofunction"]] = params.autofunction
            storage[strdic.status_dict["poolingmode"]] = params.poolingmode
            storage[strdic.status_dict["psum_delay"]] = layer_params.psum_delay
            storage[strdic.status_dict["needed_refreshes"]] = layer_params.needed_refreshes_mx[layer_repetition][0]
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
            storage[strdic.status_dict["gemm_mode"]] = getattr(layer_params, "gemm_mode", 0)

            # Generate and store router configurations for all three data paths
            storage[strdic.status_dict["router_iact"]] = self.write_router_iact(params, layer_params)
            storage[strdic.status_dict["router_wght"]] = self.write_router_wght(params, layer_params)
            storage[strdic.status_dict["router_psum"]] = self.write_router_psum(params, layer_params)

        return storage
    def write_quant_and_offset(self, params, layer_params, layer_repetition):
        """
        Generates unified quantization and offset parameters packed for shift-register streaming.
        
        Layout per Filter (LSB -> MSB):
        - Offset   : OFFSET_WIDTH bits
        - Exponent : EXPONENT_WIDTH bits
        - Mantissa : MANTISSA_WIDTH bits
        
        Shift behavior:
        Filter 0 lands at the lowest bit-positions (quant_reg[0 +: ENTRY_WIDTH]).
        To achieve this with a left-shift register (`quant_reg <= {quant_reg, new_data}`),
        the highest filter blocks (e.g. Filter N-1 down to Filter 0) must be transmitted FIRST,
        or the streaming array must be sliced accordingly.
        """
        offset_w = getattr(params, 'OFFSET_WIDTH', 8)
        exp_w    = getattr(params, 'EXPONENT_WIDTH', 7)
        mant_w   = getattr(params, 'MANTISSA_WIDTH', 25)
        
        entry_w  = offset_w + exp_w + mant_w  # e.g. 8 + 7 + 25 = 40 Bits per filter
        dma_w    = params.DMA_BITWIDTH        # e.g. 64 Bits
        
        total_bits = params.QUANT_AMOUNT * entry_w
        
        # 1. Pack ALL filter data into one giant bitfield (Filter 0 at LSB)
        packed_bitstream = 0
        for f in range(params.QUANT_AMOUNT):
            mant   = layer_params.quantize[f][0]  # Mantissa
            exp    = layer_params.quantize[f][1]  # Exponent
            offset = layer_params.offset[f]       # Zero-Point Offset
            
            # Combine offset | exp | mant for filter `f`
            filter_entry = (offset & ((1 << offset_w) - 1)) | \
                        ((exp    & ((1 << exp_w) - 1)) << offset_w) | \
                        ((mant   & ((1 << mant_w) - 1)) << (offset_w + exp_w))
            
            # Shift into the global bitstream at position `f * entry_w`
            packed_bitstream |= (filter_entry << (f * entry_w))
            
        # 2. Slice the bitstream into DMA words
        # Because your Verilog shifts incoming words from lower to higher indices:
        #   quant_reg[(a+1)*DMA_BITWIDTH +: DMA_BITWIDTH] <= quant_reg[a*DMA_BITWIDTH +: DMA_BITWIDTH]
        # The FIRST word sent will end up at the HIGHEST index.
        # Therefore, we chunk from MSB down to LSB!
        
        num_dma_words = math.ceil(total_bits / dma_w)
        dma_storage = []
        
        # Calculate top padded length
        total_padded_bits = num_dma_words * dma_w
        
        for i in range(num_dma_words):
            # Extract word starting from the top bits down to bottom
            shift_amount = total_padded_bits - (i + 1) * dma_w
            if shift_amount >= 0:
                word = (packed_bitstream >> shift_amount) & ((1 << dma_w) - 1)
            else:
                # Handle edge alignment if total_bits isn't perfectly divisible
                word = (packed_bitstream << abs(shift_amount)) & ((1 << dma_w) - 1)
                
            dma_storage.append(word)
        return dma_storage
    
    def write_router_iact(self, params, layer_params):
        """Configure input activation (iact) router settings for all clusters.

        Generates routing configuration for distributing input activations across the
        cluster array. The routing values determine how data flows through the network-on-chip
        to reach the appropriate PEs. Supports both single and multi-cluster Y configurations.

        Routing values:
        - 1: Single Y cluster mode
        - 9: First cluster in multi-Y-cluster group (multiple PEs per cluster)
        - 17: Last cluster in multi-Y-cluster group (multiple PEs per cluster)
        - 25: Middle cluster in multi-Y-cluster group (multiple PEs per cluster)
        - 3: First cluster (single PE per cluster)
        - 33: Non-first cluster (single PE per cluster)

        Args:
            params: Hardware configuration parameters including cluster dimensions
            layer_params: Layer-specific parameters including used cluster counts

        Returns:
            In serial mode: list of DMA words with packed router values
            In parallel mode: 3D list [cluster_x][cluster_y][router] of routing values
        """
        line = 0
        if(params.SERIAL):
            storage = []
        else:
            storage = [[[[] for c in range(params.NUM_GLB_IACT)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]
        if ((params.Clusters_Y == 1) & (params.Clusters_X == 1)):
            storage.append(line)
            return storage
        router_cycle = 0
        for cl_y in range(params.Clusters_Y):
            for cl_x in range(params.Clusters_X):
                for router in range(params.NUM_GLB_IACT):
                    if(layer_params.used_Y_cluster == 1):
                        if(params.SERIAL):
                            line = line + (1 << (params.Iact_Router_Bits * router_cycle))
                        else:
                            storage[cl_x][cl_y][router] = 1
                    else:
                        if(layer_params.used_PEs_Y > 1):
                            if((cl_y % layer_params.used_Y_cluster) == 0):
                                if(params.SERIAL):
                                    line = line + (9 << (params.Iact_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 9
                            else:
                                if(((cl_y + 1) % layer_params.used_Y_cluster) == 0):
                                    if(params.SERIAL):
                                        line = line + (17 << (params.Iact_Router_Bits * router_cycle))
                                    else:
                                        storage[cl_x][cl_y][router] = 17
                                else:
                                    if(params.SERIAL):
                                        line = line + (25 << (params.Iact_Router_Bits * router_cycle))
                                    else:
                                        storage[cl_x][cl_y][router] = 25
                        else:
                            if(cl_y == 0):
                                if(params.SERIAL):
                                    line = line + (3 << (params.Iact_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 3
                            else:
                                if(params.SERIAL):
                                    line = line + (33 << (params.Iact_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 33
                    router_cycle = router_cycle + 1
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_BITWIDTH/params.Iact_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0
        if(params.SERIAL and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)
        
        return storage

    def write_router_wght(self, params, layer_params):
        """Configure weight (wght) router settings for all clusters.

        Generates routing configuration for distributing weights across the cluster array.
        Routing is simplified compared to iact routing: first cluster or single-cluster
        computations use local routing (0), while subsequent clusters use network routing (1).

        Routing values:
        - 0: Use local weight data (first cluster, single cluster mode, or multi-kernel mode)
        - 1: Forward weight data from previous cluster

        Args:
            params: Hardware configuration parameters including cluster dimensions
            layer_params: Layer-specific parameters including computation mode flags

        Returns:
            In serial mode: list of DMA words with packed router values
            In parallel mode: 3D list [cluster_x][cluster_y][router] of routing values
        """
        line = 0
        if(params.SERIAL):
            storage = []
        else:
            storage = [[[[] for c in range(params.Wght_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]
        if ((params.Clusters_X == 1)):
            return storage
        router_cycle = 0    
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):   
                for router in range(params.Wght_Routers):
                    if((cl_x == 0) | (layer_params.single_cluster_computation == 1) | (layer_params.different_kernels_per_calculation >= 2)):
                        if(params.SERIAL):
                            line = line + (0 << (params.Wght_Router_Bits * router_cycle))
                        else:
                            storage[cl_x][cl_y][router] = 0
                    else:
                        if(params.SERIAL):
                            line = line + (1 << (params.Wght_Router_Bits * router_cycle))
                        else:
                            storage[cl_x][cl_y][router] = 1
                    router_cycle = router_cycle + 1
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_BITWIDTH/params.Wght_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0
        if(params.SERIAL and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)
        return storage

    def write_router_psum(self, params, layer_params):
        """Configure partial sum (psum) router settings for all clusters.

        Generates routing configuration for collecting and forwarding partial sums through
        the cluster array. The routing depends on PE utilization and cluster Y organization.

        Routing values:
        - 0: No routing (inactive cluster)
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
        if(params.SERIAL):
            storage = []
        else:
            storage = [[[[] for c in range(params.Psum_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]
        if (params.Clusters_Y == 1):
            return storage
        router_cycle = 0          
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Psum_Routers):
                    if((layer_params.used_Y_cluster == 1)):
                        if(params.SERIAL):
                            if (((cl_y * params.Clusters_X) + cl_x) * params.PEs_X < (layer_params.iact_size_x + layer_params.add_up) * layer_params.different_kernels_per_calculation) :
                                line = line + (4 << (params.Psum_Router_Bits * router_cycle))
                            else :
                                line = line + (0 << (params.Psum_Router_Bits * router_cycle))
                        else:
                            storage[cl_x][cl_y][router] = 4

                    else:
                        if((cl_y % layer_params.used_Y_cluster) == 0):
                            if(params.SERIAL):
                                line = line + (5 << (params.Psum_Router_Bits * router_cycle))
                            else:
                                storage[cl_x][cl_y][router] = 5
                        else:
                            if(((cl_y + 1) %  layer_params.used_Y_cluster) == 0):
                                if(params.SERIAL):
                                    line = line + (3 << (params.Psum_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 3
                            else:
                                if(params.SERIAL):
                                    line = line + (2 << (params.Psum_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 2
                    router_cycle = router_cycle + 1
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_BITWIDTH/params.Psum_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0
                        
        if(params.SERIAL and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)
        return storage

    def write_psum_data_glb(self, params, layer_params, layer_repetition, dram, cl_y, router, cycle):
        """Write partial sum data to global buffer format.

        Converts partial sum (bias) data from DRAM format to the global buffer format
        expected by the hardware. Scales floating-point values to fixed-point representation
        based on the configured bitwidths.

        Args:
            params: Hardware configuration parameters including bitwidth settings
            layer_params: Layer-specific parameters including filter counts
            layer_repetition: Current repetition index for the layer
            dram: Source data from DRAM containing bias values
            cl_y: Cluster Y coordinate (currently unused in implementation)
            router: Router index (currently unused in implementation)
            cycle: Cycle index (currently unused in implementation)

        Returns:
            In serial mode: list of DMA words containing scaled bias values
            In parallel mode: 2D list [cluster_x][data_index] of scaled bias values
        """
        storage, line = self.initialize_storage(params.SERIAL), 0
        for part_data_num in range(int(layer_params.filters/layer_params.needed_wght_transmissions)):
            if(part_data_num < layer_params.used_psum_per_PE):
                for cl_x in range(params.Clusters_X):
                    if(params.SERIAL):
                        
                        line = line + (int(round(float((2**(params.IACT_Bitwidth + params.WGHT_Bitwidth - 1)) * dram[0][0][part_data_num]))) << (params.DATA_PSUM_BITWIDTH * cl_x))
                        line = 0
                        storage.append(line)
                        line = 0
                    else:

                        storage[cl_x].append(int(round(float((2**(params.IACT_Bitwidth + params.WGHT_Bitwidth - 1)) * 0))))
            else:
                line = 0
        return storage
