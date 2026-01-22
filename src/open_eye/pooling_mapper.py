# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
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

        Creates a mapper for pooling operations. Note that pooling layers do not require
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
        # Initialize parent LayerMapper
        # Pooling layers don't use weight or bias mappers
        super().__init__(params, layer_params, layer_repetition, dram_layer_content)

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
                  - Parallel mode: Dictionary indexed by status_dict keys

        Note:
            - Configures PE usage bitmap indicating which PEs participate in pooling
            - Sets up router configurations for input activations and partial sums
            - No weight routing needed since pooling has no trainable parameters
            - Handles both max pooling and average pooling modes

        """
        # Initialize storage structure based on communication mode
        if (params.SERIAL):
            storage = [[] for b in range(len(strdic.stream_serial_dict))]
        else:
            storage = [[] for a in range(len(strdic.status_dict))]

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
        "stride_x_reg": layer_params.strideX,
        "stride_y_reg": layer_params.strideY,
        "skipIact_reg": layer_params.skipIact,
        "skipWght_reg": layer_params.skipWght,
        "skipPsum_reg": layer_params.skipPsum,
        "psum_delay_reg": layer_params.psum_delay,
        "kernel_per_pe_cluster_reg": layer_params.kernel_per_pe_cluster,
        "kernel_size": 2,#layer_params.kernel_size[1]
        "x_lines_reg": layer_params.iact_x_lines,
        "needed_wght_cycles_reg": math.ceil(layer_params.filters/(layer_params.used_psum_per_PE * layer_params.different_kernels_per_calculation)),
        "needed_cycles_reg": layer_params.needed_standing_cycles,
        "iact_converter_buffer_addr_max_cycles": layer_params.needed_standing_cycles,
        "iact_channels_per_pe": layer_params.used_channels,
        "fc_size_reg":0,
        "iact_size_x":layer_params.iact_size_x,
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
        "iact_channels_per_pe_next_layer": layer_params.diff_iact_layer_next_layer,
        "needed_psum_storage_cycles_reg": 0,#layer_params.psum_storage_cycles
        "iact_channel_max_cycles": layer_params.diff_iact_layer,
        "input_activations_reg": layer_params.used_iact_per_PE,
        "filters_reg": layer_params.used_psum_per_PE,
        "needed_x_cls_reg": layer_params.used_X_cluster,
        "needed_y_cls_reg": layer_params.used_Y_cluster,
        "needed_iact_cycles_reg": layer_params.needed_Iact_writes,
        "wght_addr_len_reg": layer_params.used_wght_addr_per_PE,
        "iact_addr_len_reg": layer_params.used_iact_addr_per_PE,
        "send_data_out": layer_params.send_values_out,
        "needed_iact_buffer_words_reg": layer_params.needed_iact_buffer_words,
        "add_up_reg":layer_params.add_up
        })

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

            # Generate and store router configurations for all three data paths
            storage[strdic.status_dict["router_iact"]] = self.write_router_iact(params, layer_params)
            storage[strdic.status_dict["router_wght"]] = self.write_router_wght(params, layer_params)
            storage[strdic.status_dict["router_psum"]] = self.write_router_psum(params, layer_params)

        return storage
    def write_quantize(self, params, layer_params, layer_repetition):
        """Generate quantization parameters for pooling layer outputs.

        This method would generate quantization parameters for pooling layer outputs,
        but is currently not implemented for pooling operations as they typically don't
        require additional quantization beyond what's already applied to input activations.

        Args:
            params: Hardware parameters
            layer_params: Layer parameters with quantization information
            layer_repetition (int): Current repetition index

        Returns:
            list: Empty list (quantization not used for pooling layers)

        Note:
            The method contains skeleton code for packing quantization parameters
            into DMA lines, but always returns an empty list, indicating that
            quantization is not applied for pooling operations.

        """
        dma_line = 0
        dma_storage = []
        # Pack quantization parameters (2 parameters per line, 16 lines total)
        for f in range(math.ceil(16)):
            dma_line = 0
            dma_line = dma_line + (layer_params.quantize[2*f][0] << 0)    # bits 0-24: first quantize param 0
            dma_line = dma_line + (layer_params.quantize[2*f][1] << 25)   # bits 25-31: first quantize param 1
            dma_line = dma_line + (layer_params.quantize[2*f+1][0] << 32) # bits 32-56: second quantize param 0
            dma_line = dma_line + (layer_params.quantize[2*f+1][1] << 57) # bits 57-63: second quantize param 1
            dma_storage.append(dma_line)
        # Return empty list - quantization not used for pooling
        return []

    def write_offset(self, params, layer_params, layer_repetition):
        """Generate offset parameters for pooling layer.

        This method would generate offset parameters if needed, but pooling operations
        do not require offset configuration.

        Args:
            params: Hardware parameters
            layer_params: Layer parameters
            layer_repetition (int): Current repetition index

        Returns:
            list: Empty list (offsets not used for pooling layers)

        """
        return []
    
    def write_router_iact(self, params, layer_params):
        """Generate input activation router configuration for pooling operations.

        Configures the routing of input activations across clusters and PE arrays for
        pooling operations. For pooling, activations are typically processed locally
        within each cluster without complex inter-cluster routing.

        Args:
            params: Hardware parameters (cluster dimensions, router specs)
            layer_params: Layer parameters (PE usage, cluster allocation)

        Returns:
            Router configuration in serial (list of packed integers) or parallel
            (3D array [cluster_x][cluster_y][router]) format depending on params.SERIAL.

        Note:
            Router value 1 indicates local data processing within each cluster.
            Only Y-clusters less than 4 are configured (hardware limitation).

        """
        line = 0
        if(params.SERIAL):
            storage = []  # Bit-packed router config for serial mode
        else:
            # 3D array for parallel mode: [cluster_x][cluster_y][router_id]
            storage = [[[[] for c in range(params.NUM_GLB_IACT)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]

        router_cycle = 0  # Tracks bit position in serial mode

        # Configure input activation router for each cluster and router instance
        for cl_y in range(params.Clusters_Y):
            for cl_x in range(params.Clusters_X):
                for router in range(params.NUM_GLB_IACT):
                    # Configure routers for Y-clusters 0-3 only
                    if(params.SERIAL):
                        if (cl_y < 4):
                            # Router value 1: use local data only (no inter-cluster routing)
                            line = line + (1 << (params.Iact_Router_Bits * router_cycle))
                    else:
                        # In parallel mode, all routers set to 1 (local processing)
                        storage[cl_x][cl_y][router] = 1

                    router_cycle = router_cycle + 1

                    # In serial mode, flush line when it's full
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_Bit_AXI/params.Iact_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0

        # Append any remaining partial line in serial mode
        if(params.SERIAL and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)

        return storage

    def write_router_wght(self, params, layer_params):
        """Generate weight router configuration for pooling operations.

        Configures the weight routing network for pooling operations. Since pooling
        layers do not use weights, all weight routers are set to 0 (inactive/disabled).

        Args:
            params: Hardware parameters (cluster dimensions, router specs)
            layer_params: Layer parameters (unused for pooling weight routing)

        Returns:
            Router configuration in serial (list of packed integers) or parallel
            (3D array [cluster_x][cluster_y][router]) format depending on params.SERIAL.

        Note:
            All weight routers are set to 0 since pooling operations do not require
            weight data. This effectively disables the weight data path for the layer.

        """
        line = 0
        if(params.SERIAL):
            storage = []  # Bit-packed router config for serial mode
        else:
            # 3D array for parallel mode: [cluster_x][cluster_y][router_id]
            storage = [[[[] for c in range(params.Wght_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]

        router_cycle = 0  # Tracks bit position in serial mode

        # Configure weight routers for all clusters
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Wght_Routers):
                    # Set all weight routers to 0 (disabled/no weight data)
                    if(params.SERIAL):
                        line = line + (0 << (params.Wght_Router_Bits * router_cycle))
                    else:
                        storage[cl_x][cl_y][router] = 0

                    router_cycle = router_cycle + 1

                    # In serial mode, flush line when full
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_Bit_AXI/params.Wght_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0

        # Append any remaining partial line in serial mode
        if(params.SERIAL and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)

        return storage

    def write_router_psum(self, params, layer_params):
        """Generate partial sum router configuration for pooling operations.

        Configures the routing of partial sums (pooling results) across Y-clusters.
        The routing pattern depends on how many PEs per column are used and how the
        computation is distributed across Y-clusters.

        Args:
            params: Hardware parameters (cluster dimensions, router specs)
            layer_params: Layer parameters (PE usage, Y-cluster allocation)

        Returns:
            Router configuration in serial (list of packed integers) or parallel
            (3D array [cluster_x][cluster_y][router]) format depending on params.SERIAL.

        Note:
            Router values indicate different partial sum handling modes:
            - 2: Pass through intermediate cluster (forward to next)
            - 3: Last cluster in chain (receive and finalize)
            - 4: Single PE per column (local handling only)
            - 5: First cluster in chain (receive from bottom and forward up)

        """
        line = 0
        if(params.SERIAL):
            storage = []  # Bit-packed router config for serial mode
        else:
            # 3D array for parallel mode: [cluster_x][cluster_y][router_id]
            storage = [[[[] for c in range(params.Psum_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]

        router_cycle = 0  # Tracks bit position in serial mode

        # Configure partial sum routers based on PE distribution
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Psum_Routers):
                    # Case 1: Single PE per column - use local handling only
                    if((layer_params.ceil_used_PE_per_clm == 1)):
                        if(params.SERIAL):
                            line = line + (4 << (params.Psum_Router_Bits * router_cycle))
                        else:
                            storage[cl_x][cl_y][router] = 4
                    # Case 2: Multiple PEs per column - set up routing chain
                    else:
                        # First cluster in Y-chain: receive from below and forward upward
                        if((cl_y % layer_params.used_Y_cluster) == 0):
                            if(params.SERIAL):
                                line = line + (5 << (params.Psum_Router_Bits * router_cycle))
                            else:
                                storage[cl_x][cl_y][router] = 5
                        else:
                            # Last cluster in Y-chain: receive and finalize
                            if(((cl_y + 1) %  layer_params.used_Y_cluster) == 0):
                                if(params.SERIAL):
                                    line = line + (3 << (params.Psum_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 3
                            # Middle clusters in Y-chain: pass through (forward upward)
                            else:
                                if(params.SERIAL):
                                    line = line + (2 << (params.Psum_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 2

                    router_cycle = router_cycle + 1

                    # In serial mode, flush line when full
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_Bit_AXI/params.Psum_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0

        # Append any remaining partial line in serial mode
        if(params.SERIAL and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)

        return storage

    def write_psum_data_glb(self, params, layer_params, layer_repetition, dram, cl_y, router, cycle):
        """Generate initial partial sum data for global buffer loading.

        Creates initial partial sum values to be loaded into the global buffer before
        pooling computation begins. For pooling operations, partial sums are initialized
        to zero.

        Args:
            params: Hardware parameters (bitwidths, cluster config)
            layer_params: Layer parameters (partial sums per PE)
            layer_repetition (int): Current repetition index
            dram: DRAM contents (unused for pooling initialization)
            cl_y (int): Y-cluster index
            router (int): Router index
            cycle (int): Cycle index

        Returns:
            list: Initial partial sum values in serial (list) or parallel (2D array
                  [cluster_x][data_index]) format depending on params.SERIAL.

        Note:
            All partial sum values are initialized to 0 for pooling operations.
            The bitwidth calculation (2**(IACT_Bitwidth + WGHT_Bitwidth - 1)) * 0
            maintains the correct data format while producing zero values.

        """
        # Initialize storage structure based on communication mode
        storage, line = self.initialize_storage(params.SERIAL), 0

        # Generate zero-initialized partial sums for each PE
        for part_data_num in range(math.ceil(self.layer_params.used_psum_per_PE/2)):
            if(part_data_num < self.layer_params.used_psum_per_PE):
                # Write zero partial sums for each X-cluster
                for cl_x in range(self.params.Clusters_X):
                    if(self.params.SERIAL):
                        # Calculate zero value with correct bitwidth
                        # (2**(IACT_Bitwidth + WGHT_Bitwidth - 1)) ensures proper accumulator size
                        line = line + (int(round(float((2**(params.IACT_Bitwidth + params.WGHT_Bitwidth - 1)) * 0))) << (params.PSUM_Bitwidth * cl_x))
                        storage.append(line)
                        line = 0
                    else:
                        # In parallel mode, append zero to storage for this cluster
                        storage[cl_x].append(int(round(float((2**(params.IACT_Bitwidth + params.WGHT_Bitwidth - 1)) * 0))))
            else:
                # Reset line for next iteration
                line = 0

        return storage
