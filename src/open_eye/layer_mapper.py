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

Typical Usage:
    LayerMapper is not instantiated directly. Instead, use layer-specific subclasses:

    >>> # For convolution layers
    >>> conv_mapper = ConvMapper(params, layer_params, repetition, dram_content,
    ...                          sparse_iacts, sparse_wghts)
    >>> conv_mapper.make_stream()
    >>> stream = conv_mapper.get_stream()

    >>> # For depthwise convolution layers
    >>> dw_mapper = DWMapper(params, layer_params, repetition, dram_content,
    ...                      sparse_iacts, sparse_wghts)
    >>> dw_mapper.make_stream()
    >>> stream = dw_mapper.get_stream()

Subclasses:
    - ConvMapper: Standard convolution layers
    - DWMapper: Depthwise convolution layers
    - DenseMapper: Fully connected/dense layers
    - PoolingMapper: Pooling layers
"""

import logging
import open_eye.stream_dicts as strdic

logger = logging.getLogger("cocotb")


class LayerMapper(object):
    """Abstract base class for mapping neural network layers to OpenEye hardware.

    This class provides the common infrastructure for translating high-level layer
    operations (convolution, pooling, dense, etc.) into low-level hardware streams
    that can be executed on the OpenEye accelerator. It manages the coordination of
    input activations, weights, and biases/partial sums, and generates the necessary
    control signals and data layouts.

    The LayerMapper serves as a template for layer-specific implementations, providing:
    - Stream storage management for multi-channel data paths
    - Default implementations for common stream generation patterns
    - Utilities for serial vs. parallel communication modes
    - Integration with DRAM for loading layer data

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
        This is an abstract base class. Subclasses must implement:
        - write_working_parameters(): Generate layer-specific hardware configuration
        - write_quant_and_offset(): Generate quantization parameters
    """
    def __init__(self, params, layer_params, layer_repetition, dram_layer_content, inputstream_mapper = None, weightstream_mapper = None, bias_mapper = None):
        """Initialize the layer mapper with hardware parameters and stream mappers.

        Sets up the base infrastructure for layer mapping, including DRAM data references,
        stream mapper instances, and storage structures for generated streams.

        Args:
            params: Hardware configuration parameters defining the accelerator architecture
                (cluster dimensions, PE counts, router configurations, communication mode)
            layer_params: Layer-specific parameters (kernel dimensions, strides, channel counts,
                padding, etc.) that define the neural network operation
            layer_repetition (int): Index of the current layer repetition (for layers that
                require multiple transmission passes to complete)
            dram_layer_content (list): Three-element list containing DRAM data:
                [0] Feature map/input activation data
                [1] Weight/kernel data
                [2] Bias/partial sum data
            inputstream_mapper (optional): Stream mapper instance for input activations.
                Subclasses typically provide layer-specific implementations.
            weightstream_mapper (optional): Stream mapper instance for weights.
                Subclasses typically provide layer-specific implementations.
            bias_mapper (optional): Stream mapper instance for biases/partial sums.
                Subclasses typically provide layer-specific implementations.

        Note:
            The storage structure is initialized based on params.SERIAL:
            - Serial mode: Uses stream_serial_dict indexing for DMA-based transmission
            - Parallel mode: Uses stream_parallel_dict indexing for direct port access
        """
        # Store hardware and layer configuration
        self.params = params
        self.layer_params = layer_params
        self.layer_repetition = layer_repetition

        # Extract DRAM data for this layer
        self.dram_fmap = dram_layer_content[0]      # Input feature maps
        self.dram_weights = dram_layer_content[1]   # Layer weights/kernels
        self.dram_bias = dram_layer_content[2]      # Biases/partial sums

        # Store stream mapper instances for data path coordination
        self.IactStreamCreator = inputstream_mapper  # Input activation mapper
        self.WghtStreamCreator = weightstream_mapper # Weight mapper
        self.PsumStreamCreator = bias_mapper         # Bias/partial sum mapper

        # Initialize storage structure based on communication mode
        if (params.SERIAL):
            # Serial mode: DMA-based transmission with bit-packed streams
            self.storage = [[] for _ in range(len(strdic.stream_serial_dict))]
        else:
            # Parallel mode: Direct port access with separate channels
            self.storage = [[] for _ in range(len(strdic.stream_parallel_dict))]

    def make_stream(self):
        """Generate all data streams for the layer execution.

        This is the main orchestration method that coordinates the generation of all
        required hardware streams for layer execution. It populates the storage structure
        with working parameters, input activations, weights, biases/partial sums,
        quantization parameters, and offset values.

        The method respects skip flags (skipIact, skipWght, skipPsum) to avoid redundant
        data transfers when data can be reused from previous transmissions.

        Storage Channels Populated:
            - status: Working parameters and hardware configuration
            - iact: Input activation data (if skipIact == 0)
            - wght: Weight data (if skipWght == 0)
            - psum: Bias/partial sum data (if skipPsum == 0)
            - quantize: Quantization parameters for output
            - offset: Offset values for activation functions

        Note:
            This method delegates to layer-specific implementations:
            - write_working_parameters() for hardware configuration
            - IactStreamCreator.get_iact_stream() for input activations
            - WghtStreamCreator.get_wght_stream() for weights
            - PsumStreamCreator.get_psum_stream() for biases/partial sums
            - write_quant_and_offset() for post-processing parameters
        """
        # Generate working parameters and hardware configuration
        self.storage[strdic.stream_parallel_dict["status"]] = self.write_working_parameters(self.params, self.layer_params, self.layer_repetition)

        # Generate input activation stream (skip if data can be reused)
        if (self.layer_params.skipIact == 0) :
            self.storage[strdic.stream_parallel_dict["iact"]] = self.IactStreamCreator.get_iact_stream()
        else :
            self.storage[strdic.stream_parallel_dict["iact"]] = []

        # Generate weight stream (skip if weights can be reused)
        if (self.layer_params.skipWght == 0) :
            self.storage[strdic.stream_parallel_dict["wght"]] = self.WghtStreamCreator.get_wght_stream()
        else :
            self.storage[strdic.stream_parallel_dict["wght"]] = []

        # Generate bias/partial sum stream (skip if not needed)
        if (self.layer_params.skipPsum == 0) :
            self.storage[strdic.stream_parallel_dict["psum"]] = self.PsumStreamCreator.get_psum_stream()
        else :
            self.storage[strdic.stream_parallel_dict["psum"]] = []

        # Generate quantization and offset parameters
        self.storage[strdic.stream_parallel_dict["quantize"]] = self.write_quant_and_offset(self.params, self.layer_params, self.layer_repetition)
        # Log progress for this layer transmission
        logger.info("Stream finished: " + str(self.layer_repetition + 1) + " of " + str(self.layer_params.needed_total_transmissions))

    def get_stream(self):
        """Retrieve the generated hardware streams.

        Returns:
            list: The complete storage structure containing all generated streams.
                The format depends on communication mode (serial vs. parallel) and
                includes status, iact, wght, psum, quantize, and offset channels.
        """
        return self.storage

    def write_working_parameters(self, params, layer_params, layer_repetition):
        """Generate layer-specific working parameters and hardware configuration.

        Abstract method that must be implemented by subclasses to generate the
        hardware configuration specific to their layer type (Conv, DW, Dense, Pooling).

        Args:
            params: Hardware configuration parameters
            layer_params: Layer-specific parameters
            layer_repetition (int): Current repetition index

        Returns:
            Configuration data in format appropriate for the communication mode

        Note:
            This method should be overridden by all subclasses to provide
            layer-specific implementations.
        """
        pass

    def get_psum_stream(self):
        """Generate partial sum data stream for bias/accumulator initialization.

        Creates the data stream for loading biases or partial sums into the accelerator.
        The method iterates through Y-clusters and routers, writing partial sum data
        for the cycles specified in the layer's refresh schedule.

        Returns:
            Storage structure in serial (flat list) or parallel (3D array) format:
            - Serial mode: List of bit-packed values
            - Parallel mode: [cluster_x][cluster_y][router] nested list structure

        Note:
            This is a default implementation that may be overridden by subclasses.
            It delegates to write_psum_data_glb() which should be implemented by
            subclasses to define layer-specific partial sum formatting.
        """
        # Initialize storage structure based on communication mode
        if(self.params.SERIAL):
            storage = []  # Flat list for serial mode
        else:
            # 3D array for parallel mode: [cluster_x][cluster_y][router]
            storage = [[[[] for c in range(self.params.Psum_Routers)] for b in range(self.params.Clusters_Y)] for a in range(self.params.Clusters_X)]

        # Iterate through Y-clusters and routers to generate partial sum data
        for cl_y in range(self.params.Clusters_Y):
            for router in range(self.params.Psum_Routers):
                # Process cycles specified in the refresh schedule for this repetition
                for cycle in range(self.layer_params.needed_refreshes_mx[self.layer_repetition][1],self.layer_params.needed_refreshes_mx[self.layer_repetition][2]):
                    # Generate partial sum data for this cluster/router/cycle
                    temp_storage = self.write_psum_data_glb(self.params, self.layer_params, self.layer_repetition, self.dram_weights, cl_y, router, cycle)

                    # Store data according to communication mode
                    if(self.params.SERIAL):
                        storage.extend(temp_storage)
                    else:
                        storage[0][cl_y][router].extend(temp_storage[0])
                        storage[1][cl_y][router].extend(temp_storage[1])
        return storage

    def initialize_storage(self, serial):
        """Initialize an empty storage structure based on communication mode.

        Utility method to create storage structures that adapt to serial or parallel
        communication modes. Used by stream generation methods to prepare data buffers.

        Args:
            serial (bool): True for serial/DMA mode, False for parallel mode

        Returns:
            list: Empty storage structure:
                - Serial mode: Empty flat list []
                - Parallel mode: List with Clusters_X empty sublists
        """
        if(serial):
            storage = []  # Flat list for serial DMA transmission
        else:
            # Separate lists for each X-cluster in parallel mode
            storage = [[] for a in range(self.params.Clusters_X)]
        return storage

    def line_reset(self, serial):
        """Reset a line buffer to its initial state based on communication mode.

        Utility method to create a fresh line buffer for accumulating data before
        writing to storage.

        Args:
            serial (bool): True for serial/DMA mode, False for parallel mode

        Returns:
            int or list: Reset line value:
                - Serial mode: 0 (integer for bit accumulation)
                - Parallel mode: [0, 0] (separate buffers for data/control)
        """
        if(serial):
            return 0  # Single accumulator for bit-packed serial data
        else:
            return [0, 0]  # Separate buffers for parallel mode [data, control]

    def line_to_storage(self, serial, line, storage):
        """Append a completed line buffer to the storage structure.

        Utility method to add accumulated data to the storage in the format
        appropriate for the communication mode.

        Args:
            serial (bool): True for serial/DMA mode, False for parallel mode
            line: The line data to append (int for serial, list for parallel)
            storage (list): The storage structure to append to

        Returns:
            list: The updated storage structure with the new line appended

        Note:
            In parallel mode, line[0] is appended to storage[0] (data channel)
            and line[1] is appended to storage[1] (control channel).
        """
        if(serial):
            storage.append(line)  # Append single value in serial mode
        else:
            # Append to separate channels in parallel mode
            storage[0].append(line[0])  # Data channel
            storage[1].append(line[1])  # Control channel
        return storage