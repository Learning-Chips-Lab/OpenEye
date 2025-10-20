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
import test_utils.stream_dicts as strdic
from test_utils.layer_mapper import LayerMapper

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
            # Serial mode: list for DMA bitstream transmission
            storage = [[] for b in range(len(strdic.stream_serial_dict))]
        else:
            # Parallel mode: list indexed by status_dict keys
            storage = [[] for b in range(len(strdic.status_dict))]

        # Number of refresh cycles needed for pooling operation
        needed_refreshes = 1

        counter = 0
        computing_pes = 0

        # === PE ALLOCATION BITMAP ===
        # Create a bitmap indicating which PEs are active for pooling computation
        # Each bit represents one PE across all clusters and PE arrays
        for x in range(params.Clusters_X):
            for y in range(params.Clusters_Y):
                for pe_y in range(params.PEs_Y):
                    for pe_x in range(params.PEs_X):
                        # Set bit if this PE is used for computation
                        if(layer_params.computing_mx[x][y][pe_y][pe_x]== 1):
                            computing_pes = computing_pes + 2**(counter)
                        counter = counter + 1

        # Convert PE bitmap to binary string representation
        total_bits = params.Clusters_X * params.Clusters_Y * params.PEs_Y * params.PEs_X
        bitstring = format(computing_pes, f"0{total_bits}b")[::-1]  # Reverse for correct bit order
        if (params.SERIAL):
            # === SERIAL MODE: DMA BITSTREAM GENERATION ===
            # Pack all parameters into bit-packed DMA lines for serial transmission
            dma_line = 0
            dma_storage = []

            # === DMA Transmission 1: Main Configuration Parameters ===
            # Bit packing: combine multiple parameters into a single 64-bit integer
            dma_line = params.data_mode + ((layer_params.realfactor) << 1)   # bits 0-5: data mode and real factor
            dma_line = dma_line + (params.autofunction << 6)                 # bit 6: autofunction enable
            dma_line = dma_line + (params.poolingmode << 7)                  # bit 7: pooling mode (max/avg)
            # Note: refresh calculation commented out for pooling
            #dma_line = dma_line + ((math.ceil(layer_params.needed_refreshes_mx[layer_repetition][0]/layer_params.diff_iact_layer) << 8))
            dma_line = dma_line + (layer_params.used_X_cluster << 16)        # bits 16-17: X clusters used
            dma_line = dma_line + (layer_params.used_Y_cluster << 18)        # bits 18-21: Y clusters used
            dma_line = dma_line + (layer_params.needed_Iact_writes << 22)    # bits 22-25: input activation writes
            dma_line = dma_line + (layer_params.used_psum_per_PE << 26)      # bits 26-31: partial sums per PE
            dma_line = dma_line + (layer_params.used_iact_addr_per_PE << 32) # bits 32-35: input addresses per PE
            dma_line = dma_line + (layer_params.used_wght_addr_per_PE << 36) # bits 36-40: weight addresses per PE (unused for pooling)
            dma_line = dma_line + (layer_params.used_iact_per_PE << 41)      # bits 41-45: input activations per PE
            dma_line = dma_line + (layer_params.send_values_out << 46)       # bit 46+: output control
            dma_storage.append(dma_line)
            dma_line = 0

            # === DMA Transmission 2: Stride and Control Flags ===
            dma_line = dma_line + (layer_params.needed_wght_transmissions)   # bits 0-9: weight transmissions (0 for pooling)
            dma_line = dma_line + (layer_params.strideY << 10)               # bits 10-13: stride Y (also includes stride X)
            dma_line = dma_line + (layer_params.skipIact << 14)              # bit 14: skip input activation loading
            dma_line = dma_line + (layer_params.skipWght << 15)              # bit 15: skip weight loading (always set for pooling)
            dma_line = dma_line + (layer_params.skipPsum << 16)              # bit 16: skip partial sum loading
            dma_line = dma_line + (layer_params.psum_delay << 17)            # bits 17-20: partial sum delay
            dma_line = dma_line + (layer_params.kernel_per_pe_cluster << 21) # bits 21-24: kernels per PE cluster
            #dma_line = dma_line + (layer_params.kernel_size[1] << 25)       # Optional: kernel size Y
            dma_line = dma_line + (1 << 29)                                  # bits 29-36: fixed values
            dma_line = dma_line + (1 << 37)
            #dma_line = dma_line + (math.ceil(layer_params.needed_refreshes_mx[layer_repetition][0]/layer_params.diff_iact_layer) << 45)
            dma_storage.append(dma_line)
            dma_line = 0

            # === DMA Transmission 3: Input Feature Map Dimensions ===
            # Use bitwise OR for clearer bit field assignment
            dma_line = (layer_params.needed_standing_cycles << 56) | \
                       (layer_params.used_channels << 48) | \
                       (layer_params.iact_size_y << 32) | \
                       (layer_params.iact_size_x << 16) | \
                       layer_params.iact_stream_cycles
            dma_storage.append(dma_line)
            dma_line = 0

            # === DMA Transmission 4: Layer Configuration Flags ===
            dma_line = math.ceil(layer_params.diff_iact_layer)                              # bits 0-7: input activation layer difference
            dma_line = dma_line + math.ceil(layer_params.diff_iact_layer_next_layer << 8)   # bits 8-15: next layer difference
            dma_line = dma_line + math.ceil(layer_params.choose_iact_storage_input << 16)   # bit 16: input storage selection
            dma_line = dma_line + math.ceil(layer_params.choose_iact_storage_output << 17)  # bit 17: output storage selection
            dma_line = dma_line + math.ceil(layer_params.fully_connected << 18)             # bit 18: fully connected flag
            dma_line = dma_line + math.ceil(layer_params.max_pooling << 19)                 # bit 19: max pooling flag
            dma_line = dma_line + math.ceil(layer_params.store_in_psum << 20)               # bit 20: store in partial sum
            dma_line = dma_line + math.ceil(layer_params.output_cycles << 21)               # bits 21-28: output cycles
            dma_line = dma_line + math.ceil(layer_params.y_lines_per_calculation << 29)     # bits 29-32: Y lines per calculation
            dma_line = dma_line + math.ceil(layer_params.different_kernels_per_calculation << 33)  # bits 33+: kernels per calc
            dma_storage.append(dma_line)
            dma_line = 0

            # === DMA Transmissions 5+: PE Usage Bitmap ===
            # Split the PE bitmap into segments that fit in DMA_Bit_AXI width
            for x in range(math.ceil(params.PE_Complete/params.DMA_Bit_AXI)):
                segment = bitstring[x*params.DMA_Bit_AXI:(x+1)*params.DMA_Bit_AXI]
                dma_storage.append(int(segment[::-1], 2))  # Reverse segment for correct bit order

            # === Router Configurations ===
            # Append router configurations for all data paths
            dma_storage.extend(self.write_router_iact(params, layer_params))  # Input activation routers
            dma_storage.extend(self.write_router_wght(params, layer_params))  # Weight routers (all zeros for pooling)
            dma_storage.extend(self.write_router_psum(params, layer_params))  # Partial sum routers
            storage = dma_storage
        else:
            # === PARALLEL MODE: DIRECT PARAMETER ASSIGNMENT ===
            # Store each parameter separately in a dictionary-style structure
            storage[strdic.status_dict["data_mode"]] = params.data_mode
            storage[strdic.status_dict["realfactor"]] = layer_params.realfactor
            storage[strdic.status_dict["autofunction"]] = params.autofunction
            storage[strdic.status_dict["poolingmode"]] = params.poolingmode        # Max or average pooling
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
            storage[strdic.status_dict["usePEs"]] = int(computing_pes,2)          # Convert binary PE bitmap to integer
            storage[strdic.status_dict["kernel_per_pe_cluster"]] = layer_params.kernel_per_pe_cluster

            # Generate and store router configurations for all data paths
            storage[strdic.status_dict["router_iact"]] = self.write_router_iact(params, layer_params)
            storage[strdic.status_dict["router_wght"]] = self.write_router_wght(params)
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
