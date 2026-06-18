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

import math
import logging
import open_eye.stream_dicts as strdic
from open_eye.layer_mapper import LayerMapper
from open_eye.iact_stream_mapper import DwIactStreamMapper
from open_eye.wght_stream_mapper import DwWghtStreamMapper
from open_eye.psum_stream_mapper import DwPsumStreamMapper

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
        # Create specialized stream mappers for depthwise convolution
        input_mapper = DwIactStreamMapper(params, layer_params, layer_repetition, dram_layer_content[0], sparse_iacts)
        weight_mapper = DwWghtStreamMapper(params, layer_params, layer_repetition, dram_layer_content[1], sparse_wghts)
        bias_mapper = DwPsumStreamMapper(params, layer_params, layer_repetition, dram_layer_content[2])

        # Initialize parent LayerMapper with the depthwise-specific mappers
        super().__init__(params, layer_params, layer_repetition, dram_layer_content, input_mapper, weight_mapper, bias_mapper)

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
                  - Parallel mode: Dictionary indexed by status_dict keys

        Note:
            - Configures PE usage bitmap, stride settings, skip flags
            - Sets up router configurations for iact, wght, and psum paths
            - Handles refresh counters and transmission scheduling

        """
        # Initialize storage structure based on communication mode
        if (params.SERIAL):
            # Serial mode: list indexed by stream_serial_dict
            storage = [[] for b in range(len(strdic.stream_serial_dict))]
        else:
            # Parallel mode: list indexed by status_dict
            storage = [[] for a in range(len(strdic.status_dict))]

        # Initialize skip flag for partial sum loading
        layer_params.skipPsum = 0
        counter = 0
        computing_pes = 0

        # === PE ALLOCATION BITMAP ===
        # Create a bitmap indicating which PEs are active for computation
        # Each bit represents one PE across all clusters
        for x in range(params.Clusters_X):
            for y in range(params.Clusters_Y):
                for pe_y in range(params.PEs_Y):
                    for pe_x in range(params.PEs_X):
                        # Set bit if this PE is used for computation
                        if(layer_params.computing_mx[x][y][pe_y][pe_x]== 1):
                            computing_pes = computing_pes + 2**(counter)
                        counter = counter + 1

        # Format the PE bitmap as a binary string
        formating = "0" + str(params.Clusters_X * params.Clusters_Y * params.PEs_Y) + "b"
        computing_pes = format(computing_pes, formating)

        if (params.SERIAL):
            # === SERIAL MODE: DMA BITSTREAM GENERATION ===
            # Pack all parameters into bit-packed DMA lines for serial transmission
            dma_line = 0
            dma_storage = []

            # DMA Line 1: Main configuration parameters
            # Bit packing: combine multiple parameters into a single integer
            dma_line = params.data_mode + ((layer_params.realfactor) << 1)  # bits 0-5
            dma_line = dma_line + (params.autofunction << 6)                 # bit 6
            dma_line = dma_line + (params.poolingmode << 7)                  # bit 7
            dma_line = dma_line + ((layer_params.needed_refreshes_mx[layer_repetition][0] << 8))  # bits 8-15
            dma_line = dma_line + (layer_params.used_X_cluster << 16)        # bits 16-17
            dma_line = dma_line + (layer_params.used_Y_cluster << 18)        # bits 18-21
            dma_line = dma_line + (layer_params.needed_Iact_writes << 22)    # bits 22-25
            dma_line = dma_line + (layer_params.used_psum_per_PE << 26)      # bits 26-31
            dma_line = dma_line + (layer_params.used_iact_addr_per_PE << 32) # bits 32-35
            dma_line = dma_line + (layer_params.used_wght_addr_per_PE << 36) # bits 36-40
            dma_line = dma_line + (layer_params.used_iact_per_PE << 41)      # bits 41+
            dma_storage.append(dma_line)
            # DMA Line 2: Stride, address/data lengths, and skip flags
            dma_line = 0
            dma_line = dma_line + (layer_params.iact_addr_len)              # bits 0-1
            dma_line = dma_line + (layer_params.iact_data_len << 2)         # bits 2-5
            dma_line = dma_line + (layer_params.strideX << 6)               # bits 6-9
            dma_line = dma_line + (layer_params.strideY << 10)              # bits 10-13

            # Check if weight refresh is needed (based on refresh/transmission ratio)
            if ((layer_params.needed_refreshes_mx[layer_repetition][0] % layer_params.needed_wght_transmissions) == 0):
                dma_line = dma_line + (0 << 14)  # No refresh needed
            else:
                dma_line = dma_line + (1 << 14)  # Refresh needed

            dma_line = dma_line + (layer_params.skipWght << 15)             # bit 15: skip weight loading
            dma_line = dma_line + (layer_params.skipPsum << 16)             # bit 16: skip psum loading
            dma_storage.append(dma_line)

            # DMA Lines 3-6: PE usage bitmap (split into 4 lines of 48 bits each)
            for x in reversed(range(4)):
                dma_line = int(computing_pes[x*48:(x+1)*48],2)
                dma_storage.append(dma_line)

            # Store the complete DMA configuration
            storage[strdic.stream_serial_dict["status"]] = dma_storage
        else:
            # === PARALLEL MODE: DIRECT PARAMETER ASSIGNMENT ===
            # Store each parameter separately in a dictionary-style structure
            storage[strdic.status_dict["data_mode"]] = 1
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

            # Generate and store router configurations for all data paths
            storage[strdic.status_dict["router_iact"]] = self.write_router_iact(params, layer_params)
            storage[strdic.status_dict["router_wght"]] = self.write_router_wght(params, layer_params)
            storage[strdic.status_dict["router_psum"]] = self.write_router_psum(params, layer_params)
            return storage

    def write_router_iact(self, params, layer_params):
        """Generate input activation router configuration for depthwise convolution.

        Configures the routing of input activations across clusters and PE arrays.
        For depthwise convolution, routing depends on whether operations span multiple
        Y-clusters and how PEs are utilized per column.

        Args:
            params: Hardware parameters (cluster dimensions, router specs)
            layer_params: Layer parameters (PE usage, cluster allocation)

        Returns:
            Router configuration in serial (list of packed integers) or parallel
            (3D array [cluster_x][cluster_y][router]) format depending on params.SERIAL.

        Note:
            Router values indicate data path:
            - 1: Use local data only
            - 3: Forward to next cluster
            - 9: Receive from previous and use local
            - 33: Pass through from previous to next
            - 41: Receive, use local, and forward

        """
        line = 0
        if(params.SERIAL):
            storage = []  # Bit-packed router config for serial mode
        else:
            # 3D array for parallel mode: [cluster_x][cluster_y][router_id]
            storage = [[[[] for c in range(params.NUM_GLB_IACT)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]

        router_cycle = 0  # Tracks bit position in serial mode

        # Configure router for each cluster and router instance
        for cl_y in range(params.Clusters_Y):
            for cl_x in range(params.Clusters_X):
                for router in range(params.NUM_GLB_IACT):
                    # Single Y-cluster case: all routers use local data only
                    if(layer_params.used_Y_cluster == 1):
                        if(params.SERIAL):
                            line = line + (1 << (params.Iact_Router_Bits * router_cycle))
                        else:
                            storage[cl_x][cl_y][router] = 1
                    else:
                        # Multi Y-cluster case: configure routing chain
                        if(layer_params.used_PEs_Y > 1):
                            # First cluster in chain: receive and use
                            if((cl_y % layer_params.used_Y_cluster) == 0):
                                if(params.SERIAL):
                                    line = line + (9 << (params.Iact_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 9
                            else:
                                # Last cluster in chain: pass through
                                if((cl_y % layer_params.used_Y_cluster) + 1 == layer_params.used_Y_cluster):
                                    if(params.SERIAL):
                                        line = line + (33 << (params.Iact_Router_Bits * router_cycle))
                                    else:
                                        storage[cl_x][cl_y][router] = 33
                                # Middle clusters: receive, use, and forward
                                else:
                                    if(params.SERIAL):
                                        line = line + (41 << (params.Iact_Router_Bits * router_cycle))
                                    else:
                                        storage[cl_x][cl_y][router] = 41
                        else:
                            # Single PE per column case
                            if(cl_y == 0):
                                # First cluster: forward
                                if(params.SERIAL):
                                    line = line + (3 << (params.Iact_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 3
                            else:
                                # Other clusters: pass through
                                if(params.SERIAL):
                                    line = line + (33 << (params.Iact_Router_Bits * router_cycle))
                                else:
                                    storage[cl_x][cl_y][router] = 33

                    router_cycle = router_cycle + 1

                    # In serial mode, flush line when it's full
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_Bit_AXI/params.Iact_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0

        # Append any remaining partial line in serial mode
        if((params.SERIAL) and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)
        return storage

    def write_router_wght(self, params, layer_params):
        """Generate weight router configuration for depthwise convolution.

        Configures the routing of weights across X-clusters. For depthwise convolution,
        weights are typically broadcast horizontally across clusters.

        Args:
            params: Hardware parameters (cluster dimensions, router specs)
            layer_params: Layer parameters (includes single_cluster_computation flag)

        Returns:
            Router configuration in serial (list of packed integers) or parallel
            (3D array [cluster_x][cluster_y][router]) format depending on params.SERIAL.

        Note:
            Router values:
            - 0: Use data from DMA/memory
            - 1: Use data from previous cluster (horizontal pass-through)

        """
        line = 0
        if(params.SERIAL):
            storage = []  # Bit-packed router config for serial mode
        else:
            # 3D array for parallel mode: [cluster_x][cluster_y][router_id]
            storage = [[[[] for c in range(params.Wght_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]

        router_cycle = 0  # Tracks bit position in serial mode

        # Configure weight routers for horizontal (X-direction) data flow
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Wght_Routers):
                    # First X-cluster or single-cluster mode: load from memory
                    if((cl_x == 0) | (layer_params.single_cluster_computation == 1)):
                        if(params.SERIAL):
                            line = line + (0 << (params.Wght_Router_Bits * router_cycle))
                        else:
                            storage[cl_x][cl_y][router] = 0
                    # Subsequent X-clusters: receive from previous cluster
                    else:
                        if(params.SERIAL):
                            line = line + (1 << (params.Wght_Router_Bits * router_cycle))
                        else:
                            storage[cl_x][cl_y][router] = 1

                    router_cycle = router_cycle + 1

                    # In serial mode, flush line when full
                    if(params.SERIAL and (router_cycle == math.floor(params.DMA_Bit_AXI/params.Wght_Router_Bits))):
                        router_cycle = 0
                        storage.append(line)
                        line = 0

        # Append any remaining partial line in serial mode
        if((params.SERIAL) and (router_cycle != 0)):
            router_cycle = 0
            storage.append(line)
        return storage

    def write_router_psum(self, params, layer_params):
        """Generate partial sum router configuration for depthwise convolution.

        Configures the routing of partial sums. For depthwise convolution, partial
        sums typically remain local to each PE without inter-cluster routing.

        Args:
            params: Hardware parameters (cluster dimensions, router specs)
            layer_params: Layer parameters

        Returns:
            Router configuration in serial (list of packed integers) or parallel
            (3D array [cluster_x][cluster_y][router]) format depending on params.SERIAL.

        Note:
            Router value 4 indicates local partial sum handling without routing
            to other clusters.

        """
        line = 0
        if(params.SERIAL):
            storage = []  # Bit-packed router config for serial mode
        else:
            # 3D array for parallel mode: [cluster_x][cluster_y][router_id]
            storage = [[[[] for c in range(params.Psum_Routers)] for b in range(params.Clusters_Y)] for a in range(params.Clusters_X)]

        router_cycle = 0  # Tracks bit position in serial mode

        # Configure partial sum routers for all clusters
        for cl_x in range(params.Clusters_X):
            for cl_y in range(params.Clusters_Y):
                for router in range(params.Psum_Routers):
                    # All routers set to value 4 (local partial sum handling)
                    if(params.SERIAL):
                        line = line + (4 << (params.Psum_Router_Bits * router_cycle))
                    else:
                        storage[cl_x][cl_y][router] = 4

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
