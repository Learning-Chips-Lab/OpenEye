# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Layer parameter computation and configuration for OpenEye accelerator.

This module provides the LayerParameters class which computes and stores all necessary
configuration parameters for executing neural network layers on the OpenEye hardware
accelerator. It handles Conv2D, Depthwise Convolution, Dense (Fully Connected), and
Pooling layers by calculating PE allocation, memory transmissions, computation cycles,
and data routing requirements.

Key Features:
    - Automatic PE (Processing Element) allocation based on layer dimensions
    - Computation of memory transmission requirements for activations, weights, and partial sums
    - Calculation of execution cycles and refresh patterns for iterative processing
    - Generation of computing matrices that define which PEs are active
    - Support for various optimization modes (serial, single-cluster, multi-cluster)
    - FPGA-specific parameter computation for hardware implementation

Typical Usage:
    >>> layer_params = LayerParameters(layer_parameters, keras_layer, hw_params,
    ...                                layer_num, total_layers)
    >>> print(f"PEs required: {layer_params.used_PEs_X} x {layer_params.used_PEs_Y}")
    >>> print(f"Total cycles: {layer_params.Used_refreshes}")
"""

import math
import logging
logger = logging.getLogger("cocotb")

class LayerParameters(object):
    """Configuration parameters for a single neural network layer on OpenEye accelerator.

    This class computes and stores all hardware configuration parameters required to execute
    a neural network layer on the OpenEye accelerator. It determines PE allocation, memory
    bandwidth requirements, execution cycles, and data routing for Conv2D, Depthwise Conv,
    Dense, and Pooling layers.

    The class automatically analyzes layer dimensions and computes:
    - PE utilization and cluster allocation
    - Memory transmission counts (IACT/WGHT/PSUM to/from GLB and PEs)
    - Execution cycle requirements and refresh patterns
    - Computing matrices defining active PEs
    - Data streaming parameters and address/data lengths

    Attributes:
        Layer Identification:
            layer_name (str): Type of layer ("Convolution2D", "DepthwiseConvolution", "Dense", "Pooling")
            fully_connected (int): 1 if Dense layer, 0 otherwise
            max_pooling (int): 1 if Pooling layer, 0 otherwise

        Shape Information:
            input_shape (list): Input tensor dimensions [batch, height, width, channels]
            output_shape (list): Output tensor dimensions [batch, height, width, channels/filters]
            kernel_shape (list): Weight tensor dimensions (varies by layer type)
            kernel_size (list): [kernel_height, kernel_width] for convolution layers
            filters (int): Number of output filters/channels
            channels (int): Number of input channels

        PE Allocation:
            used_PEs_X (int): Number of PEs used in X dimension
            used_PEs_Y (int): Number of PEs used in Y dimension
            used_X_cluster (int): Number of clusters used in X dimension
            used_Y_cluster (int): Number of clusters used in Y dimension
            computing_mx (list): 4D binary matrix [clx][cly][pey][pex] indicating active PEs
            kernel_per_pe_cluster (int): Number of kernels processed per PE cluster

        Memory Usage per PE:
            used_iact_per_PE (int): Input activations stored per PE
            used_wght_per_PE (int): Weights stored per PE
            used_psum_per_PE (int): Partial sums stored per PE
            used_iact_addr_per_PE (int): Number of activation address entries per PE
            used_wght_addr_per_PE (int): Number of weight address entries per PE

        Transmission Counts (PE Level):
            iact_transmissions_pe (int): Activation transmission iterations at PE level
            wght_transmissions_pe (int): Weight transmission iterations at PE level
            psum_transmissions_pe (int): Partial sum transmission iterations at PE level
            all_transmissions_of_pe (int): Product of all PE-level transmissions

        Transmission Counts (GLB Level):
            iact_transmissions_glb (int): Activation transmission iterations at GLB level
            wght_transmissions_glb (int): Weight transmission iterations at GLB level
            psum_transmissions_glb (int): Partial sum transmission iterations at GLB level
            needed_iact_transmissions (int): Total activation transmissions (PE * GLB)
            needed_wght_transmissions (int): Total weight transmissions (PE * GLB)
            needed_psum_transmissions (int): Total partial sum transmissions (PE * GLB)
            needed_total_transmissions (int): Product of all transmission types

        Execution Cycles:
            Used_refreshes (int): Total number of execution cycles/refreshes needed
            needed_refreshes_mx (list): 2D matrix of refresh counts per transmission iteration
            output_cycles (int): Number of output production cycles
            iact_converter_buffer_addr_max_cycles (int): FPGA-specific standing/idle cycles

        Computation Parameters:
            total_computations (int): Total number of convolution operations
            calc_X (int): Number of output positions in X dimension
            calc_Y (int): Number of output positions in Y dimension
            single_cluster_computation (int): Mode flag (0=multi-cluster, 1=single, 2=single-Y)
            y_lines_per_calculation (int): Number of Y lines processed per cycle
            different_kernels_per_calculation (int): Number of kernels processed simultaneously

        Stride and Padding:
            strideX (int): Convolution stride in X dimension
            strideY (int): Convolution stride in Y dimension
            padding (str): Padding mode ("same" or "valid")

        Data Streaming:
            iact_size_x (int): Input activation width
            iact_size_y (int): Input activation height
            iact_addr_len (int): Length of activation address fields
            iact_data_len (int): Length of activation data fields
            iact_stream_cycles (int): Cycles needed to stream input activations
            needed_Iact_writes (int): Number of activation write operations required
            iact_x_lines (int): FPGA-specific X-line count for activation storage

        Channel Processing:
            used_channels (int): Number of channels processed per iteration
            channel_repetition (int): Channel processing repetition factor
            diff_iact_layer (list): Channel difference metrics for current layer
            diff_iact_layer_next_layer (int): Channel metrics for next layer

        Control Flags:
            skipIact (int): 1 to skip activation loading (reuse from previous layer)
            skipWght (int): 1 to skip weight loading
            skipPsum (int): 1 to skip partial sum loading
            send_values_out (int): 1 to send outputs to DRAM (last layer)
            store_in_psum (int): 1 to store results in partial sum memory

        Storage Selection:
            choose_iact_storage_input (int): Flag for input activation storage location
            choose_iact_storage_output (int): Flag for output activation storage location

        Timing and Delays:
            psum_delay (int): Partial sum accumulation delay cycles
            psum_storage_cycles (int): Cycles for storing partial sums

        Quantization:
            quantize (list): 2D array of quantization parameters per filter
            offset (list): Quantization offset values per filter

        Miscellaneous:
            data_mode (int): Data processing mode (0=standard, 1=depthwise)
            realfactor (int): Real number scaling factor
            current_highest_number (int): Tracking variable for internal calculations
            add_up (int): Padding adjustment for non-aligned dimensions
            complete_iacts_in_design (int): Flag for complete activation availability

    Args:
        layer_parameters (list): List of previously computed layer parameters (for inter-layer dependencies)
        layer: A Keras/TensorFlow layer object with weights, shape, and configuration
        params: OpenEye hardware parameters (PE counts, memory sizes, cluster config, etc.)
        layer_number (int): Index of this layer in the network (0-indexed)
        max_layers (int): Total number of layers in the network

    Raises:
        ValueError: If layer type is unsupported or kernel size exceeds hardware constraints

    Note:
        The __init__ method automatically dispatches to specialized initialization methods
        based on layer type: write_conv2d_layer, write_convdw_layer, write_dense_layer,
        or write_pooling_layer.
    """
    def __init__(self, layer_parameters, layer, params, layer_number, max_layers):
        """Initialize layer parameters by dispatching to layer-type-specific methods.

        This constructor initializes all parameter attributes to default values, then
        delegates to specialized methods based on the layer type detected in layer.name.

        The initialization process:
        1. Sets default values for all configuration parameters
        2. Detects layer type from layer.name
        3. Calls appropriate write_*_layer method to compute actual parameters
        4. Raises ValueError if layer type is not supported
        """
        # === Layer Identification ===
        self.layer_name = ""
        # === Pooling Identification ===
        self.pooling_mode = 0

        # === PE Allocation Parameters ===
        self.used_PEs_X = 1                    # PEs used in X dimension
        self.used_PEs_Y = 0                    # PEs used in Y dimension
        self.used_X_cluster = 1                # Clusters used in X dimension
        self.used_Y_cluster = 1                # Clusters used in Y dimension
        self.Used_refreshes = 0                # Total execution cycles needed
        self.current_input_X = 0               # Current input X position
        self.current_input_Y = 0               # Current input Y position
        self.padding = "same"                  # Padding mode for convolution
        self.needed_refreshes_mx = []          # Refresh pattern matrix
        self.calc_X = 0                        # Output positions in X
        self.calc_Y = 0                        # Output positions in Y
        self.total_computations = 0            # Total MAC operations

        # === Memory Usage per PE ===
        self.used_iact_per_PE = []             # Activations per PE
        self.used_wght_per_PE = []             # Weights per PE
        self.used_psum_per_PE = []             # Partial sums per PE
        self.diff_iact_layer = []              # Channel difference metrics
        self.diff_iact_layer_next_layer = 0    # Next layer's channel metrics
        self.used_PEs_per_col = 0              # Ceiling of PEs per column

        # === Data Transmission Parameters ===
        self.needed_Iact_writes = 0            # Number of activation write cycles
        self.needed_iact_buffer_words = 0      # Amount of words per convolution in iact buffer

        # === Miscellaneous Calculation Parameters ===
        self.current_highest_number = 0        # Internal calculation tracker
        self.realfactor = 0                    # Real number scaling factor
        self.used_iact_addr_per_PE = 2         # Activation address entries per PE
        self.used_wght_addr_per_PE = 5         # Weight address entries per PE
        self.iact_addr_len = 1                 # Length of activation address field
        self.iact_data_len = 3                 # Length of activation data field
        self.choose_iact_storage_input = 0     # Input storage location selector
        self.choose_iact_storage_output = 0    # Output storage location selector

        # === Convolution Parameters ===
        self.strideX = 1                       # Stride in X dimension
        self.strideY = 1                       # Stride in Y dimension
        self.add_up = 0                        # Padding adjustment for alignment
        self.complete_iacts_in_design = 0      # Flag for complete activation availability
        self.max_pooling = 0                   # Pooling layer flag
        self.output_cycles = 0                 # Output generation cycles

        # === Layer Shape Information ===
        self.filters = 1                       # Number of output filters
        self.input_shape = []                  # Input tensor shape
        self.kernel_shape = []                 # Weight tensor shape
        self.output_shape = []                 # Output tensor shape
        self.kernel_size = []                  # Convolution kernel dimensions
        self.padding_x = 0
        self.padding_y = 0
        self.kernel_per_pe_cluster = 1         # Kernels per PE cluster
        self.used_channels = 1                 # Channels processed per iteration
        self.channel_repetition = 4            # Channel processing repetition factor
        self.single_cluster_computation = 0    # Cluster computation mode
        self.iact_size_x = 1                   # Input activation width
        self.iact_size_y = 1                   # Input activation height
        self.psum_size_x = 1                   # Input activation width
        self.psum_size_y = 1                   # Input activation height

        # === Transmission Count Defaults ===
        self.iact_transmissions_pe = 1         # Activation transmissions (PE level)
        self.wght_transmissions_pe = 1         # Weight transmissions (PE level)
        self.psum_transmissions_pe = 1         # Partial sum transmissions (PE level)
        self.all_transmissions_of_pe = 1       # Product of PE transmissions
        self.iact_transmissions_glb = 1        # Activation transmissions (GLB level)
        self.wght_transmissions_glb = 1        # Weight transmissions (GLB level)
        self.psum_transmissions_glb = 1        # Partial sum transmissions (GLB level)
        self.used_iact_per_PE = 1              # Activations stored per PE
        self.used_wght_per_PE = 1              # Weights stored per PE
        self.used_psum_per_PE = 1              # Partial sums stored per PE
        self.psum_storage_cycles = 1
        self.needed_wght_transmissions = 1     # Total weight transmissions
        self.needed_total_transmissions = 1    # Total all transmissions
        self.psum_delay = 0                    # Partial sum delay cycles
        self.fully_connected = 0               # Dense layer flag
        self.store_in_psum = 0                 # Store in psum memory flag
        self.limit_increase = 0                # Amount of Iact Storages, that incrase adresses
        self.limit_increase_mod = 0            # Module amount of Iact Storages, that incrase adresses
        self.iteration_for_kernels = 1         # Amount of iterations per kernel
        self.needed_wght_cycles = 1            # Number of cycles for wght
        self.fsm_psum_limit = 1                # Number of cycles for psum
        self.cluster_per_conv_cycle = 4        # Number of clusters, that IActs are written to per single cycle
        self.iact_converter_max_cycles = 1     # Needed cycles for iact converter to write data to buffer
        self.iact_buffer_words_per_write = 12  # Needed words per write in Iact Cycles
        self.needed_cycles = 1                 # Total needed cycles of rewritting of PEs
        self.trans_cycles_iact = 1             # Needed transmissions cycles for iact
        self.trans_cycles_wght = 1             # Needed transmissions cycles for wght
        self.trans_cycles_psum = 1             # Needed transmissions cycles for psum
        self.iact_cycles_one_word_all_ram = 1

        # === Control Flags ===
        self.send_values_out = 1               # Send outputs to DRAM
        self.skipIact = 0                      # Skip activation loading flag
        self.skipWght = 0                      # Skip weight loading flag
        self.skipPsum = 0                      # Skip psum loading flag
        self.iact_stream_cycles = 1            # Activation streaming cycles

        # === Computing Matrix and Mode ===
        self.computing_mx = 0                  # 4D matrix of active PEs
        self.data_mode = 0                     # Data processing mode
        self.y_lines_per_calculation = 1       # Y lines per computation cycle
        self.iact_x_line_repetitions = 1       # Cycles needed for computing a single x iact line
        self.different_kernels_per_calculation = 1 # Kernels per cycle
        self.buffer_cycles_for_x_iact = 1      # Needed repetitions in IACT GLB to cycle to one line
        self.start_param_array = 1             # Representation of 1s and 0s for every cycle to write into the clusters
        self.limit_increase = 0                # Amound of Iact Clusters, that switch their addresses in one writing cycle
        self.initial_upper_limit = 0           # 

        # === FPGA-Specific Parameters ===
        self.iact_converter_buffer_addr_max_cycles = 0    # FPGA standing/idle cycles
        self.iact_x_lines = 3              # X-lines for activation storage
        self.quantize = [[0 for _ in range(2)]for _ in range(params.QUANT_AMOUNT)]  # Quantization params
        self.offset =  [0 for _ in range(params.QUANT_AMOUNT)]  # Quantization offsets

        # === Layer Type Detection and Dispatch ===
        # Detect layer type from name and call appropriate initialization method
        if "depthwise_conv2d" in layer.name:
            logger.debug("Depthwise Convolution Layer")
            self.write_convdw_layer(layer, params)

        elif "conv2d" in layer.name:
            logger.debug("2D Convolution Layer")
            self.write_conv2d_layer(layer_parameters, layer, params, layer_number, max_layers)

        elif "dense" in layer.name:
            logger.debug("Dense Layer")
            self.write_dense_layer(layer_parameters, layer, params, layer_number, max_layers)

        elif "pooling2d" in layer.name:
            logger.debug("Pooling Layer")
            self.write_pooling_layer(layer_parameters, layer, params, layer_number, max_layers)

        else:
            # Unsupported layer type
            logger.debug(f"Layer type for {layer.name} not supported.")
            raise ValueError("Layer type not supported.")


    def check_for_multiple_lines_per_computation(self, params):
        """Determine how many Y lines and kernels can be processed simultaneously.

        Analyzes the hardware resources (clusters and PEs) to determine if multiple
        input lines or multiple kernels can be computed in parallel during a single
        computation cycle. This optimization can improve throughput by maximizing
        PE utilization.

        Args:
            params: Hardware parameters containing cluster and PE counts

        Note:
            Sets self.different_kernels_per_calculation and self.y_lines_per_calculation.
            Currently hardcoded to set y_lines_per_calculation = 1 (disabled optimization).
        """
        #Can it all be mapped in one single compuation Cycle?

        # Calculate PE usage in Y dimension
        self.used_PEs_Y = self.kernel_size[0]*self.kernel_per_pe_cluster
        used_PEs_per_clm = self.used_PEs_Y/params.PEs_Y
        self.used_Y_cluster = math.ceil(used_PEs_per_clm)
        usable_pes = params.PEs_X*math.floor(params.Clusters/self.used_Y_cluster)
        if (self.output_shape[1] >  usable_pes): # No
            self.iact_x_line_repetitions = math.ceil(self.output_shape[1]/usable_pes)
            self.y_lines_per_calculation = 1
            self.different_kernels_per_calculation = 1
        else : # Yes
            self.iact_x_line_repetitions = 1
            # Calculate how many different kernels can be processed simultaneously
            # based on available PE resources divided by input width
            self.different_kernels_per_calculation = usable_pes//self.output_shape[1]
            # Limit to at most ceil(output_channels/8) kernels
            self.different_kernels_per_calculation = min(self.different_kernels_per_calculation, math.ceil(self.output_shape[3]/4))
            # Calculate how many Y lines can be processed per computation
            self.y_lines_per_calculation = math.floor(usable_pes/self.output_shape[1]/self.different_kernels_per_calculation)
            # Limit by available Y clusters and input height
            self.y_lines_per_calculation = min(self.y_lines_per_calculation,params.Clusters_Y,self.output_shape[2])
            # Currently disabled - process one line at a time
            self.y_lines_per_calculation = 1

    def calculate_used_Y_cluster(self, params):
        """Calculate the number of Y-direction clusters needed for this layer.

        Determines how many clusters in the Y dimension are required based on the
        number of PEs needed in Y and the available PEs per cluster. The calculation
        involves rounding up, then distributing evenly across available clusters.

        Args:
            params: Hardware parameters with Clusters_Y and PEs_Y specifications

        Note:
            Updates self.used_Y_cluster with the final cluster count.
        """
        # Calculate how many PE rows (clusters in Y) are needed
        self.used_Y_cluster = (math.ceil(self.used_PEs_Y/params.PEs_Y))
        # Distribute evenly across available Y clusters
        self.used_Y_cluster = (math.floor(params.Clusters_Y/self.used_Y_cluster))
        # Round up to get final cluster count
        self.used_Y_cluster = (math.ceil(params.Clusters_Y/self.used_Y_cluster))

    def calculate_total_computations(self):
        """Calculate the total number of output positions for the convolution operation.

        Computes how many output positions (windows) the convolution kernel will produce
        based on input dimensions, kernel size, and padding mode. For "same" padding, the
        output dimensions match the input. For "valid" padding, output is reduced by
        (kernel_size - 1) in each dimension.

        Sets:
            self.calc_X: Number of output positions in X dimension
            self.calc_Y: Number of output positions in Y dimension
            self.total_computations: Total output positions (calc_X * calc_Y)
            self.output_cycles: Number of cycles to produce all outputs (equals calc_Y)

        Note:
            This calculates spatial output dimensions, not the total MAC operations.
            Total MACs would be total_computations * kernel_area * input_channels.
        """
        if(self.padding == "same"):
            # Same padding: output size matches input size
            self.calc_X = self.input_shape[1]
            self.calc_Y = self.input_shape[2]
        else:
            # Valid padding: output size reduced by (kernel_size - 1)
            self.calc_X = self.input_shape[1] - self.kernel_size[0] + 1
            self.calc_Y = self.input_shape[2] - self.kernel_size[1] + 1

        # Total number of output positions
        self.total_computations = self.calc_X * self.calc_Y

    def calculate_iact_transmissions(self, params):
        """Calculate the number of activation write operations needed per computation.

        Determines how many write cycles are required to load input activations into
        the global buffer (GLB). This depends on the sliding window pattern created by
        convolution stride and the number of GLB activation banks available.

        The calculation accounts for:
        - Overlap between adjacent PE computations (affected by stride)
        - Multiple kernels per PE cluster
        - Number of global activation memory banks

        Args:
            params: Hardware parameters with PEs_X, NUM_GLB_IACT specifications

        Sets:
            self.needed_Iact_writes: Number of activation write cycles required
        """
        # Calculate total activation width needed:
        # (PEs_X - 1) * stride * kernels_per_cluster + kernel_width * kernels_per_cluster
        # Then divide by number of GLB activation banks and round up
        self.needed_Iact_writes = math.ceil(((params.PEs_X - 1) * self.strideX * self.kernel_per_pe_cluster + (self.kernel_size[1]*self.kernel_per_pe_cluster)) / (params.NUM_GLB_IACT))

    def calculate_computing_matrix(self, params):
        """Generate a 4D binary matrix indicating which PEs are active for this layer.

        Creates a computing matrix [cluster_x][cluster_y][pe_y][pe_x] where each element
        is 1 if that PE participates in computation, or 0 if it's idle. This matrix is used
        to configure the hardware by:
        - Enabling only necessary PEs to save power
        - Defining data routing patterns
        - Handling non-aligned output dimensions

        The matrix accounts for:
        - Kernel dimensions that don't fully utilize all PEs
        - Output dimensions that don't align with PE array size
        - Multiple Y-clusters for tall kernels
        - Padding adjustments for partial PE utilization

        Args:
            params: Hardware parameters (Clusters_X/Y, PEs_X/Y, Iacts_per_PE)

        Sets:
            self.computing_mx: 4D binary matrix indicating active PEs
            self.add_up: Number of padding positions added for alignment

        Raises:
            ValueError: If kernel size exceeds hardware constraints

        Note:
            This function performs three main masking operations:
            1. Mask PEs beyond required output width (horizontal masking)
            2. Mask PEs beyond required kernel height (vertical masking)
            3. Handle multi-cluster vertical distribution for tall kernels
        """

        # Verify that kernel fits within hardware constraints
        if((self.kernel_size[0] <= params.Iacts_per_PE) & (self.kernel_size[1] <= params.PEs_Y * params.Clusters_Y)):
            # === Initialize all PEs as active ===
            self.computing_mx = [[[[1 for _ in range(params.PEs_X)]
                                            for _ in range(params.PEs_Y)]
                                            for _ in range(params.Clusters_Y)]
                                            for _ in range(params.Clusters_X)]
            amount_of_psum_per_cycle = min(params.Clusters_Y*params.Clusters_X*params.PEs_X,8)
            # === MASKING PHASE 1: Handle non-aligned output width ===
            # If output width doesn't evenly divide by PEs_X, some PEs will be unused
            if(((self.output_shape[1]) % amount_of_psum_per_cycle) != 0):
                # Calculate padding needed to align to PE arrays
                if (self.different_kernels_per_calculation == 1):
                    self.add_up = (amount_of_psum_per_cycle - ((self.output_shape[1]) % amount_of_psum_per_cycle))
                else:
                    self.add_up = 8 - (self.output_shape[1] % 8)
                #if ((self.output_shape[1] % (params.PEs_X)) != 0):
                #    self.add_up = (params.PEs_X)- (self.output_shape[1] % params.PEs_X)
                x_count       = 0
                kernel_number = 0
                # Disable PEs in the partial row that exceed output width
                for y_cluster in range(params.Clusters_Y):
                    for x_cluster in range(params.Clusters_X):
                        if ((x_count >= self.output_shape[1]) & (kernel_number < self.different_kernels_per_calculation - 1)):
                            x_count = 0
                            kernel_number = kernel_number + 1
                        for x_pe in range(params.PEs_X):
                            if (x_count >= self.output_shape[1]):
                                for y_pe in range(params.PEs_Y):
                                    self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0
                            x_count = x_count + 1
            else:
                # Output width perfectly aligned - no padding needed
                self.add_up = 0
                if (self.iact_x_line_repetitions != 1):
                    self.add_up = (self.output_shape[1]) % (params.Clusters_Y * params.Clusters_X * params.PEs_X)
                if (self.different_kernels_per_calculation != 1):
                    self.add_up = (self.output_shape[1]) % (params.Clusters_Y * params.Clusters_X * params.PEs_X)
            # === MASKING PHASE 2: Eliminate PEs beyond computation requirements ===
            # Calculate total number of X positions needed per computation cycle
            x_values_per_cycle = (self.calc_X + self.add_up) * self.y_lines_per_calculation * self.different_kernels_per_calculation
            for x_cluster in range(params.Clusters_X):
                for y_cluster in range(params.Clusters_Y):
                    for y_pe in range(params.PEs_Y):
                        for x_pe in range(params.PEs_X):
                            # Calculate linear position of this PE in the flattened array
                            x_pos_in_pes = x_cluster * params.PEs_X + (y_cluster//self.used_Y_cluster) * params.PEs_X * params.Clusters_X + x_pe
                            # Disable PEs beyond the required computation width
                            if(x_pos_in_pes >= x_values_per_cycle):
                                self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0

            # === MASKING PHASE 3: Handle kernel height vs PE height ===
            if((self.kernel_size[0]*self.kernel_per_pe_cluster) < params.PEs_Y):
                # Kernel is smaller than PE array height - disable excess PEs
                for x_cluster in range(params.Clusters_X):
                    for y_cluster in range(params.Clusters_Y):
                        for y_pe in range(params.PEs_Y):
                            for x_pe in range(params.PEs_X):
                                # Disable PEs beyond kernel height
                                if((1 + y_pe) > (self.kernel_size[0]*self.kernel_per_pe_cluster)):
                                    self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0
            else:
                # Kernel spans multiple Y-clusters - need distributed masking
                for x_cluster in range(params.Clusters_X):
                    for y_cluster in range(params.Clusters_Y):
                        for y_pe in range(params.PEs_Y):
                            for x_pe in range(params.PEs_X):
                                # Calculate absolute Y position accounting for cluster offset
                                # Disable PEs beyond the distributed kernel height
                                if((1 + y_pe + (y_cluster % self.used_Y_cluster) * params.PEs_Y) > (self.kernel_size[0]*self.kernel_per_pe_cluster)):
                                    self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0
        else:
            # Kernel dimensions exceed hardware capacity
            logger.error("Can't fit model, kernel size must be adjusted.")
            raise ValueError("Can't fit model, kernel size must be adjusted.")
    
    def output_logger(self):
        """Log all computed parameters for debugging purposes.

        Outputs comprehensive debug information about layer parameters including
        refresh counts, channel usage, memory transmissions at both PE and GLB
        levels, and total transmission requirements.

        Note:
            Uses logger.debug() so output only appears when debug logging is enabled.
        """
        logger.debug("Refreshes: " + str(self.Used_refreshes))
        logger.debug("Used complete new descriptions: " + str(self.Used_refreshes))
        logger.debug("self.used_channels : " + str(self.used_channels))
        logger.debug("layer_params.needed_Iact_writes : " + str(self.needed_Iact_writes))
        logger.debug("Used_refreshes : " + str(self.Used_refreshes))
        logger.debug("layer_params.needed_psum_transmissions : " + str(self.needed_psum_transmissions))
        logger.debug("Needed transmissions: " + str(self.needed_iact_transmissions))
        logger.debug("Needed transmissions: " + str(self.needed_wght_transmissions))
        logger.debug("Needed transmissions: " + str(self.needed_psum_transmissions))
        logger.debug("Needed transmissions: " + str(self.needed_total_transmissions))

        logger.debug("Needed transmissions IACT PE : " + str(self.iact_transmissions_pe))
        logger.debug("Needed transmissions WGHT PE : " + str(self.wght_transmissions_pe))
        logger.debug("Needed transmissions PSUM PE : " + str(self.psum_transmissions_pe))
        logger.debug("Needed transmissions IACT    : " + str(self.needed_iact_transmissions))
        logger.debug("Needed transmissions WGHT    : " + str(self.needed_wght_transmissions))
        logger.debug("Needed transmissions PSUM    : " + str(self.needed_psum_transmissions))
        logger.debug("Needed transmissions TOTAL   : " + str(self.needed_total_transmissions))

    def read_data_from_layer(self, layer):
        """Extract shape and configuration parameters from a Conv2D Keras layer.

        Reads layer properties including input/output shapes, kernel dimensions,
        strides, and optional quantization parameters. Also sets control flags
        for partial sum handling and quantization.

        Args:
            layer: A Keras Conv2D layer object with attributes:
                - filters, input.shape, output.shape, kernel.shape
                - kernel_size, strides
                - Optional: store_in_psum, skip_psum, quantization_factor

        Sets:
            Layer shape attributes (input_shape, output_shape, kernel_size, etc.)
            Control flags (store_in_psum, skipPsum, choose_iact_storage_input)
            Quantization parameters (quantize array with [factor, bits] per filter)
        """
        self.layer_name = "Convolution2D"
        self.fully_connected = 0

        # Extract basic layer dimensions
        self.filters = layer.filters
        self.input_shape = layer.input.shape
        self.kernel_shape = layer.kernel.shape
        self.output_shape = layer.output.shape
        self.kernel_size = layer.kernel_size
        if(self.padding == "same"):
            self.padding_x = layer.kernel_size[0]//2
            self.padding_y = layer.kernel_size[1]//2
        self.strideX = layer.strides[0]
        self.strideY = layer.strides[1]

        # Check for optional partial sum storage flag
        if hasattr(layer, 'store_in_psum'):
            self.store_in_psum = layer.store_in_psum

        # Check for partial sum skip flag (for layer fusion)
        if hasattr(layer, 'skip_psum'):
            self.skipPsum = layer.skip_psum
            self.choose_iact_storage_input = 1

        # Read quantization parameters if provided, otherwise use defaults
        if hasattr(layer, 'quantization_factor'):
            for f in range(self.filters):
                self.quantize[f][0] = layer.quantization_factor[f][0]  # Scaling factor
                self.quantize[f][1] = layer.quantization_factor[f][1]  # Bit width
        else:
            # Default quantization: factor=1, 9 bits
            for f in range(self.filters):
                self.quantize[f][0] = 1
                self.quantize[f][1] = 7

        # Store input activation dimensions
        self.iact_size_x = self.input_shape[1]
        self.iact_size_y = self.input_shape[2]
        self.psum_size_x = self.output_shape[1]
        self.psum_size_y = self.output_shape[2]
        self.channels = self.input_shape[3]

    def choose_iact_location(self, layer_number, max_layers):
        """Configure activation storage and data flow flags based on layer position.

        Determines whether to load activations from DRAM or reuse from previous layer,
        and whether to send outputs back to DRAM. This enables layer fusion optimization
        by keeping intermediate activations in on-chip memory.

        Args:
            layer_number (int): Current layer index (0-indexed)
            max_layers (int): Total number of layers in the network

        Sets:
            self.choose_iact_storage_input: 1 to load from DRAM, 0 to reuse from prev layer
            self.send_values_out: 1 to write outputs to DRAM (last layer), 0 otherwise
            self.skipIact: 1 to skip activation loading (not first layer), 0 otherwise
        """
        self.choose_iact_storage_input = 1

        # Last layer must send outputs to DRAM
        if (layer_number == max_layers - 1):
            self.send_values_out = 1
        else:
            # Intermediate layers keep outputs on-chip
            self.send_values_out = 0

        # Non-first layers can skip loading activations (reuse from previous)
        if (layer_number != 0):
            self.skipIact = 1
            self.choose_iact_storage_input = 0

    def calculate_used_channels(self, params):
        """Calculate how many input channels are processed per iteration.

        Determines the channel blocking factor based on PE memory capacity and
        kernel dimensions. Channels may need to be processed in multiple iterations
        if they don't all fit in PE memory simultaneously.

        Args:
            params: Hardware parameters (Iacts_per_PE, Iacts_Addr_per_PE, PEs_Y)

        Sets:
            self.used_channels: Number of channels processed per iteration

        Note:
            The calculation considers:
            - Total activation memory per PE (Iacts_per_PE)
            - Kernel height (affects memory per channel)
            - Power-of-2 alignment for efficient addressing
        """
        # Simple case: all channels fit in PE memory
        if((self.channels*self.kernel_size[1])<params.Iacts_per_PE):
            self.used_channels = math.floor(self.channels)
        else:
            self.used_channels = 8

        # Case: kernel height allows multiple kernels per PE cluster
        if(2*self.kernel_size[0] <= params.PEs_Y):
            temp = math.ceil(self.channels/self.kernel_per_pe_cluster)
            if (self.kernel_size[0] >= 2):
                divisor = math.ceil(temp / params.Iacts_per_PE)
            else:
                divisor = math.ceil(temp / params.Iacts_Addr_per_PE)
            self.used_channels = math.ceil(temp/divisor)
            self.used_channels = 4  # Hardcoded override
        else:
            # Large kernel case: limit channels and round down to power of 2
            self.used_channels = 16//self.kernel_size[1]
            self.used_channels = 1 << (self.used_channels.bit_length() - 1)  # Round down to 2^n
            if (self.used_channels == 0):
                assert False
        self.used_channels = min(self.channels,self.used_channels)

    def calculate_needed_refreshes_mx(self, params):
        """Generate a matrix of refresh cycle counts for each transmission iteration.

        Creates a 2D matrix where each row represents one transmission iteration
        (combination of IACT/WGHT/PSUM transmission indices) and contains three values:
        [0] = incremental refreshes for this iteration
        [1] = cumulative refreshes at start of iteration
        [2] = cumulative refreshes at end of iteration

        This matrix is used during execution to track which refresh cycle range
        corresponds to each data transmission phase.

        Args:
            params: Hardware parameters (SERIAL flag)

        Sets:
            self.needed_total_transmissions: Total number of transmission iterations
            self.needed_refreshes_mx: 2D array [iteration][start/end/delta] of refresh counts

        Note:
            In SERIAL mode, there's only one transmission iteration.
            In parallel mode, iterations = iact_trans * wght_trans * psum_trans.
        """
        if (params.SERIAL == 1):
            # Serial mode: single transmission iteration
            self.needed_total_transmissions = 1
            self.needed_refreshes_mx = [[1 for _ in range(3)] for _ in range(1)]
            self.needed_refreshes_mx[0][2] = self.Used_refreshes  # End refresh count
            self.needed_refreshes_mx[0][1] = 0                    # Start refresh count
            self.needed_refreshes_mx[0][0] = self.Used_refreshes  # Delta refreshes
        else :
            # Parallel mode: multiple transmission iterations
            self.needed_total_transmissions = self.needed_psum_transmissions * \
                                                self.needed_wght_transmissions * \
                                                self.needed_iact_transmissions
            self.needed_refreshes_mx = [[1 for _ in range(3)] for _ in range(self.needed_total_transmissions)]

            for layer_repetition in range(self.needed_total_transmissions):
                # Calculate cumulative refresh count at end of this iteration
                self.needed_refreshes_mx[layer_repetition][2] = math.floor(((math.floor(math.floor(layer_repetition/self.iact_transmissions_pe)/self.needed_wght_transmissions)+1)/ \
                    self.needed_total_transmissions) * self.Used_refreshes)
                # Align to cluster boundary
                self.needed_refreshes_mx[layer_repetition][2] = self.needed_refreshes_mx[layer_repetition][2] - (self.needed_refreshes_mx[layer_repetition][2]%self.used_Y_cluster)

                # Calculate cumulative refresh count at start of this iteration
                self.needed_refreshes_mx[layer_repetition][1] = math.floor((math.floor(math.floor(layer_repetition/self.iact_transmissions_pe)/self.needed_wght_transmissions)/ \
                    self.needed_total_transmissions) * self.Used_refreshes)
                # Align to cluster boundary
                self.needed_refreshes_mx[layer_repetition][1] = self.needed_refreshes_mx[layer_repetition][1] - (self.needed_refreshes_mx[layer_repetition][1]%self.used_Y_cluster)

                # Delta: refreshes executed in this iteration
                self.needed_refreshes_mx[layer_repetition][0] = self.needed_refreshes_mx[layer_repetition][2] - self.needed_refreshes_mx[layer_repetition][1]

        self.needed_cycles = math.ceil(self.needed_refreshes_mx[0][0]/self.diff_iact_layer)*math.ceil(32/self.output_shape[1])
        self.needed_cycles = self.Used_refreshes

    def calculate_used_refreshes(self, params):
        """Calculate total number of refresh cycles needed to execute this layer.

        Computes how many execution cycles (refreshes) are required based on:
        - Output dimensions and number of output positions
        - Available PE resources
        - Transmission iterations needed for IACT/WGHT/PSUM
        - Single vs multi-cluster computation mode

        Args:
            params: Hardware parameters (PEs_X, Clusters_X, etc.)

        Sets:
            self.Used_refreshes: Total refresh cycles needed for layer execution

        Note:
            Uses match/case on single_cluster_computation mode:
            - Mode 1: Single cluster computation
            - Mode 2: Single X-cluster computation
            - Default: Full multi-cluster computation
        """
        self.Used_refreshes = math.ceil(self.output_shape[2] * math.ceil(self.output_shape[1]/((params.Clusters//self.different_kernels_per_calculation)*params.PEs_X)))
        self.Used_refreshes = math.ceil(self.used_Y_cluster * self.iact_transmissions_pe * self.Used_refreshes * math.ceil(math.ceil(self.filters/self.different_kernels_per_calculation)/self.used_psum_per_PE))

    def calculate_single_cluster_computation(self, params):
        """Determine if layer can use single-cluster optimization mode.

        Analyzes output dimensions to determine if the layer is small enough to
        execute within a single cluster or single X-cluster group. This optimization
        reduces routing complexity and may improve performance for small layers.

        Args:
            params: Hardware parameters (SERIAL flag)

        Sets:
            self.single_cluster_computation:
                0 = multi-cluster (default)
                1 = single cluster (output width <= 4)
                2 = single X-cluster (output width <= 8)

        Note:
            Only applies in parallel mode (SERIAL == 0).
        """
        if (params.SERIAL == 0):
            # Check if output fits in single X-cluster
            if (self.output_shape[1] <= 8):
                self.single_cluster_computation = 2
            # Check if output fits in single cluster
            if (self.output_shape[1] <= 4):
                self.single_cluster_computation = 1

    def calculate_pe_transmissions(self, params):
        """Calculate transmission iterations needed at the PE level for IACT/WGHT/PSUM.

        Determines how many times data must be transmitted to PEs based on:
        - PE memory capacity for activations, weights, and partial sums
        - Number of input channels and output filters
        - Computation mode (single vs multi-cluster)

        This is a critical calculation that affects:
        - Memory bandwidth requirements
        - Total execution cycles
        - Data reuse patterns

        Args:
            params: Hardware parameters (PEs_Y, Wghts_per_PE, Psums_per_PE, etc.)

        Sets:
            self.iact_transmissions_pe: Activation transmission iterations
            self.wght_transmissions_pe: Weight transmission iterations
            self.psum_transmissions_pe: Partial sum transmission iterations
            self.used_wght_per_PE: Weights stored per PE
            self.used_psum_per_PE: Partial sums stored per PE
            self.all_transmissions_of_pe: Product of all PE transmissions

        Note:
            Complex logic with multiple cases based on whether weights/psums fit
            in PE memory and the computation mode.
        """
        # === Calculate Activation Transmissions ===
        # Depends on computation mode and how channels are distributed
        match self.single_cluster_computation:
            case 1:
                # Single cluster: divide channels by used_channels and PEs_Y
                self.iact_transmissions_pe = math.ceil(self.input_shape[3]/(self.used_channels*params.PEs_Y))
            case 2:
                # Single X-cluster: same calculation as case 1
                self.iact_transmissions_pe = math.ceil(self.input_shape[3]/(self.used_channels*params.PEs_Y))
            case _:
                # Multi-cluster: use channel difference metric
                self.iact_transmissions_pe = math.ceil(self.diff_iact_layer/self.kernel_per_pe_cluster)

        # === Calculate Weight and Psum Transmissions ===
        # Two main cases: weights fit in PE memory vs. weights require multiple loads
        if(((self.filters * self.used_iact_per_PE)//self.different_kernels_per_calculation) <= params.Wghts_per_PE):
            # CASE 1: All weights fit in PE memory
            if (self.filters//self.different_kernels_per_calculation <= 16):
                # Small number of filters: use all at once
                self.used_psum_per_PE = math.ceil(self.filters/self.different_kernels_per_calculation)
                self.wght_transmissions_pe = math.ceil(self.channels/(self.used_channels*self.kernel_per_pe_cluster))
            else :
                # Many filters: limit to 16 psums per PE, double weight transmissions
                self.used_psum_per_PE = 16
                self.wght_transmissions_pe = 2 * math.ceil(self.channels/self.used_channels)
            # Weights per PE = psums * activations
            self.used_wght_per_PE = self.used_psum_per_PE*self.used_iact_per_PE
        else:
            # CASE 2: Weights don't fit - need to tile/split
            # Calculate tiling factor based on kernel size
            if (self.kernel_size[0] == 5):
                wght_factor = math.ceil((self.filters*self.used_iact_per_PE)/160)
            else:
                wght_factor = math.ceil((self.filters*self.used_iact_per_PE)/params.Wghts_per_PE)
            # Calculate weights and psums per PE based on computation mode
            match self.single_cluster_computation:
                case 1:
                    # Single cluster: divide filters across all clusters
                    self.used_wght_per_PE = math.ceil(self.filters/params.Clusters)*self.used_iact_per_PE
                    self.used_psum_per_PE = int(self.used_wght_per_PE/self.used_iact_per_PE)
                case 2:
                    # Single X-cluster: divide across Y-clusters only
                    self.used_wght_per_PE = math.ceil(self.filters/params.Clusters_Y)*self.used_iact_per_PE
                    self.used_psum_per_PE = int(self.used_wght_per_PE/self.used_iact_per_PE)
                case _:
                    # Multi-cluster: use tiling factor
                    self.used_wght_per_PE = math.ceil(self.filters/wght_factor/params.PARALLEL_MACS)*self.used_iact_per_PE
                    self.used_psum_per_PE = int(self.filters/wght_factor)

            # Calculate weight transmission iterations based on mode
            self.wght_transmissions_pe = math.ceil(self.channels/(self.used_channels*self.kernel_per_pe_cluster)) * math.ceil(self.filters * self.used_iact_per_PE / self.used_wght_per_PE / self.different_kernels_per_calculation)

        # === Calculate Partial Sum Transmissions ===
        if(math.ceil(self.used_iact_per_PE/self.used_wght_per_PE) <= params.Psums_per_PE):
            # Partial sums fit in PE memory - single transmission
            self.psum_transmissions_pe = 1
        else:
            # Partial sums exceed PE memory - need multiple transmissions
            self.psum_transmissions_pe = math.ceil(self.filters / params.Psums_per_PE)
            logger.debug("Error Code 5, Overused PSUM per PE, not implemented flow yet")

        # Total transmission iterations = product of all three types
        self.all_transmissions_of_pe = self.iact_transmissions_pe * self.wght_transmissions_pe * self.psum_transmissions_pe

    def calculate_glb_transmissions(self, params):
        """Calculate transmission iterations needed at the Global Buffer (GLB) level.

        Determines how many times data must be transmitted between DRAM and GLB
        based on GLB memory capacity and the amount of data to be processed.
        Works in conjunction with PE-level transmissions to compute total bandwidth.

        Args:
            params: Hardware parameters (NUM_GLB_PSUM, NUM_GLB_IACT, memory sizes, etc.)

        Sets:
            self.psum_transmissions_glb: Partial sum GLB transmission iterations
            self.wght_transmissions_glb: Weight GLB transmission iterations (typically 1)
            self.iact_transmissions_glb: Activation GLB transmission iterations
            self.needed_psum_transmissions: Total psum transmissions (PE * GLB)
            self.needed_wght_transmissions: Total weight transmissions (PE * GLB)
            self.needed_iact_transmissions: Total activation transmissions (PE * GLB)

        Note:
            GLB transmissions depend on:
            - GLB memory capacity (Psum_Mem_Addr_Words, Iact_Mem_Addr_Words)
            - Number of GLB banks (NUM_GLB_PSUM, NUM_GLB_IACT)
            - Output dimensions and PE-level transmission requirements
        """
        # === Calculate Partial Sum GLB Transmissions ===
        # Based on output size, GLB banks, clusters, and memory addressing
        self.psum_transmissions_glb = math.ceil(((math.ceil(self.output_shape[1]/params.NUM_GLB_PSUM) * \
                                    self.output_shape[2] * math.ceil(self.output_shape[3]/ self.wght_transmissions_pe)) / \
                                    params.Clusters_X / 2 / params.Clusters_Y) \
                                    / self.psum_transmissions_pe / self.wght_transmissions_pe / params.Psum_Mem_Addr_Words)

        # === Weight GLB Transmissions ===
        # Typically 1 - weights loaded once and reused
        self.wght_transmissions_glb = 1

        # === Calculate Total Transmissions (PE * GLB) ===
        self.needed_psum_transmissions = self.psum_transmissions_pe * self.psum_transmissions_glb
        self.needed_wght_transmissions = self.wght_transmissions_pe * self.wght_transmissions_glb

        # === Calculate Activation GLB Transmissions ===
        if (params.SERIAL == 1):
            # Serial mode: single GLB transmission
            self.iact_transmissions_glb = 1
        else :
            # Parallel mode: complex calculation based on:
            # - Total refresh cycles
            # - Memory address capacity
            # - Channels and kernel size
            # - Required activation writes
            self.iact_transmissions_glb = \
            math.ceil(math.ceil(self.Used_refreshes/self.wght_transmissions_pe/self.needed_psum_transmissions/self.iact_transmissions_pe)/math.floor(params.Iact_Mem_Addr_Words/\
            ((math.ceil((self.used_channels*self.kernel_size[0])/2) + (math.ceil((self.used_channels + 1)/6)))* self.needed_Iact_writes)))

        # Total activation transmissions
        self.needed_iact_transmissions = self.iact_transmissions_pe * self.iact_transmissions_glb

    def calculate_fpga_parameters(self, params):
        """Calculate FPGA-specific timing and buffering parameters.

        Computes additional parameters needed for FPGA implementation including
        standing cycles (idle/setup time) and activation line buffering requirements.
        These parameters affect the FPGA datapath timing and memory organization.

        Args:
            params: Hardware parameters (Clusters, PEs_X)

        Sets:
            self.iact_converter_buffer_addr_max_cycles: Number of standing/idle cycles needed
            self.iact_x_lines: Number of activation lines to buffer in X dimension

        Note:
            Standing cycles account for pipeline fill time and setup overhead.
            X-lines buffering is sized to hold the sliding window for convolution.
        """
        # Set X-line buffer size for sliding window (kernel height + input height - 1)
        self.iact_x_lines = self.kernel_size[1] + self.iact_size_y - 1
        # Update standing cycles based on channel packing
        if (self.buffer_cycles_for_x_iact != 1):
            self.iact_converter_buffer_addr_max_cycles = math.ceil(self.used_channels / 2) * self.needed_Iact_writes
        else:
            self.iact_converter_buffer_addr_max_cycles = (self.kernel_size[1] + self.iact_size_y - 1) * self.iact_x_line_repetitions * math.ceil(self.used_channels / 2) * self.needed_Iact_writes
        self.iact_converter_buffer_addr_max_cycles = math.ceil(self.used_channels / 2) * self.needed_Iact_writes

    def  calculate_transmission_cycles(self, params):
        self.iact_cycles_one_word_all_ram = math.ceil((params.IACT_RAM_CELLS*params.IACT_RAM_CELLS_WORD_BITWIDTH)/params.DMA_Bit_AXI)
        self.trans_cycles_iact = math.ceil(self.iact_size_x*self.iact_size_y*self.channels / params.IACT_WORDS_IN_RAM)
        missing_cycles = self.iact_cycles_one_word_all_ram - (self.trans_cycles_iact % self.iact_cycles_one_word_all_ram)
        self.trans_cycles_iact =  self.trans_cycles_iact + missing_cycles

        self.wght_cycles_one_word_all_ram = math.ceil((params.Clusters*params.NUM_GLB_WGHT*params.WGHT_RAM_CELLS_WORD_BITWIDTH)/params.DMA_Bit_AXI)
        print("LP")
        print(self.wght_cycles_one_word_all_ram)
        self.trans_cycles_wght = self.needed_wght_transmissions * (self.used_iact_per_PE * (math.ceil(self.filters / params.PARALLEL_MACS))) * math.ceil(params.NUM_GLB_WGHT * params.Clusters * 24 / params.DMA_Bit_AXI)
        self.trans_cycles_wght = self.trans_cycles_wght
        print(self.trans_cycles_wght)
        
        temp = math.ceil(params.NUM_GLB_PSUM * (params.DATA_PSUM_BITWIDTH/params.DMA_Bit_AXI))
        self.trans_cycles_psum = self.needed_wght_cycles * self.filters * self.iact_size_y * self.iact_x_line_repetitions * temp

    def write_conv2d_layer(self, layer_parameters, layer, params, layer_number, max_layers):
        """Compute all configuration parameters for a Conv2D layer.

        This is the main orchestrator method for Conv2D layers. It calls a sequence
        of specialized calculation methods to determine all hardware configuration
        parameters needed to execute the convolution on the OpenEye accelerator.

        The calculation flow:
        1. Read layer dimensions and properties
        2. Determine data flow (load from DRAM vs reuse)
        3. Calculate computation requirements
        4. Determine PE allocation and clustering
        5. Calculate transmission requirements (PE and GLB levels)
        6. Generate computing matrix (active PE bitmap)
        7. Calculate timing parameters (delays, cycles)
        8. Calculate FPGA-specific parameters

        Args:
            layer_parameters (list): Previously computed parameters for other layers
            layer: Keras Conv2D layer object with shape and configuration
            params: OpenEye hardware parameters
            layer_number (int): Index of this layer (0-indexed)
            max_layers (int): Total number of layers in network

        Note:
            This method updates all instance attributes with computed values.
            See class docstring for complete list of attributes set.
        """
        # === Phase 1: Read layer configuration ===
        self.read_data_from_layer(layer)
        self.choose_iact_location(layer_number, max_layers)

        # === Phase 2: Calculate computation requirements ===
        self.calculate_total_computations()
        self.check_for_multiple_lines_per_computation(params)
        self.calculate_single_cluster_computation(params)

        # === Phase 3: Calculate PE allocation ===
        # Determine if multiple kernels fit per PE cluster
        if (math.floor(params.PEs_Y/self.kernel_size[0]) > 1):
            self.kernel_per_pe_cluster = math.floor(params.PEs_Y/self.kernel_size[0])
            self.kernel_per_pe_cluster = 1 << (self.kernel_per_pe_cluster.bit_length() - 1) #Round down to power of 2
            self.kernel_per_pe_cluster = 1 #Harcoded to 1 kernel per cluster
        # === Phase 4: Calculate activation handling ===
        self.calculate_iact_transmissions(params)
        self.calculate_used_channels(params)
        # Calculate activation address entries based on kernel size
        if (self.kernel_size[0] == 1):
            self.used_iact_addr_per_PE = self.used_channels  # 1x1 convolution: minimal addressing
        else:
            self.used_iact_addr_per_PE = self.used_channels

        # Calculate DMA streaming cycles for input data
        self.iact_stream_cycles = math.ceil(self.input_shape[1] * self.input_shape[2] * self.input_shape[3] / params.NUM_BUFFER / (params.DMA_Bit_AXI//params.IACT_Bitwidth))

        # Calculate channel iteration metrics
        self.diff_iact_layer = math.ceil(self.input_shape[3]/self.used_channels)
        if (layer_number != max_layers - 1):
            # Store next layer's channel requirements for inter-layer optimization
            self.diff_iact_layer_next_layer = layer_parameters[max_layers - layer_number - 2].used_channels
            if (layer_parameters[max_layers - layer_number - 2].layer_name == "Dense"):
                self.diff_iact_layer_next_layer = 1

        # Total activations per PE = kernel height * channels per iteration
        self.used_iact_per_PE = self.kernel_size[1] * self.used_channels

        # === Phase 5: Calculate transmission requirements ===
        self.calculate_pe_transmissions(params)

        # === Phase 6: Calculate clustering and cycles ===
        self.calculate_used_Y_cluster(params)
        self.calculate_used_refreshes(params)
        self.calculate_computing_matrix(params)
        # === Phase 7: Calculate timing parameters ===
        # Partial sum delay for accumulation pipeline
        if (params.SERIAL):
            self.psum_delay = int(max([(math.ceil(self.used_psum_per_PE) - 2) - (self.used_Y_cluster * params.PEs_Y * 2),0]))
        else :
            self.psum_delay = int(max([(math.ceil(self.needed_refreshes_mx[layer_repetition][0]/2) - 2) - (self.used_Y_cluster * params.PEs_Y * 2),0]))

        # Check if X-cluster usage is aligned
        if((self.output_shape[2] % params.PEs_X)== 0):
            self.used_X_cluster = 1

        # Calculate data field lengths for DMA
        self.iact_addr_len = 1
        self.iact_data_len = math.ceil(self.used_iact_per_PE/(math.ceil(params.DMA_Bit_AXI/params.Clusters_X)/params.IACT_WOH_Bitwidth))

        # === Phase 8: Calculate GLB transmissions ===
        self.calculate_glb_transmissions(params)

        # Calculate weight address entries based on computation mode
        
        self.used_wght_addr_per_PE = (math.ceil(self.kernel_size[0] * self.channels/self.kernel_per_pe_cluster / self.iact_transmissions_pe)) + 2

        # Clamp weight addresses to hardware limit
        if(self.used_wght_addr_per_PE == (params.Wghts_Addr_per_PE + 1)):
            self.used_wght_addr_per_PE = self.used_wght_addr_per_PE - 1

        # Cycles needed to produce all output rows
        self.output_cycles = math.ceil(math.ceil(self.calc_Y/self.strideY) * self.iact_x_line_repetitions)

        # Calculate partial sum storage requirements
        self.psum_storage_cycles = self.diff_iact_layer * self.used_Y_cluster
        if (self.choose_iact_storage_output):
            self.psum_storage_cycles = self.diff_iact_layer
        temp1 = math.ceil(self.strideX * self.iact_size_x/((params.IACT_RAM_CELLS*8)//4))
        if ((self.iact_size_x/((params.IACT_RAM_CELLS*8)//4) >= 1) & (self.iact_x_line_repetitions >= 2)):
            temp2 = 2
        else:
            temp2 = 1
        self.buffer_cycles_for_x_iact = max(temp1,temp2)
        if (self.iact_x_line_repetitions != 1):
            addition = self.kernel_size[1]
        else:
            addition = 0
        self.buffer_cycles_for_x_iact = math.ceil((self.strideX*(addition+(params.Clusters_X*params.Clusters_Y*params.PEs_X)))/(params.IACT_RAM_CELLS*(8//4)))
        if (self.buffer_cycles_for_x_iact == 1):
            self.needed_iact_buffer_words = self.iact_x_line_repetitions*self.needed_Iact_writes*math.ceil((self.used_channels*(self.kernel_size[1]+self.iact_size_y-1)/2))
        else :
            self.needed_iact_buffer_words = self.iact_x_line_repetitions*self.needed_Iact_writes*math.ceil((self.used_channels/2))
        if (self.buffer_cycles_for_x_iact == 1):
            self.start_param_array = (1 << math.ceil((self.different_kernels_per_calculation * self.y_lines_per_calculation * self.used_Y_cluster * math.ceil(self.iact_size_x / params.NUM_GLB_PSUM)))) - 1
        else:
            self.start_param_array = (1 << math.ceil((params.Clusters_X*params.Clusters_Y)/self.buffer_cycles_for_x_iact)) - 1
        # Calculated the limit of iact buffers that need activations
        if (self.used_channels == 1):
            self.limit_increase = math.floor((self.iact_size_x*2)/(2*4))
            self.initial_upper_limit = self.limit_increase
        else:
            words_per_iact_glb = 8
            if (self.buffer_cycles_for_x_iact == 1):
                if (self.iact_x_line_repetitions == 1):
                    self.limit_increase = (self.iact_size_x*self.used_channels*self.strideX)//words_per_iact_glb
                    if (self.limit_increase == params.IACT_RAM_CELLS):
                        self.initial_upper_limit = 0
                    else:
                        self.initial_upper_limit = self.limit_increase + self.strideX
                else:
                    #self.limit_increase = (self.iact_size_x*self.used_channels*self.strideX)//words_per_iact_glb
                    #self.limit_increase = math.ceil(self.limit_increase/self.iact_x_line_repetitions)
                    self.limit_increase = (params.PEs_X*params.Clusters*self.strideX)//2
                    self.initial_upper_limit = self.limit_increase + self.strideX

            else :
                self.limit_increase = math.ceil(((params.Clusters_X*params.Clusters_Y*params.PEs_X*self.strideX)//self.buffer_cycles_for_x_iact)/(words_per_iact_glb//self.used_channels))
                self.initial_upper_limit = self.limit_increase + 1

        self.iteration_for_kernels = math.ceil(self.diff_iact_layer_next_layer / self.different_kernels_per_calculation)
        self.needed_wght_cycles = math.ceil(self.filters/(self.used_psum_per_PE * self.different_kernels_per_calculation))
        if ((params.Clusters_X * params.NUM_GLB_PSUM) == 4):
            psum_cycles = 2
        else:
            psum_cycles = 1
        self.fsm_psum_limit = (((self.iact_size_x + self.add_up) * psum_cycles * self.different_kernels_per_calculation * self.needed_wght_cycles * self.used_psum_per_PE * self.iact_size_y)//8) + 12
        if (self.channels == 1):
            self.iact_converter_max_cycles = math.ceil(self.buffer_cycles_for_x_iact*self.iact_x_line_repetitions*(self.iact_size_y +  self.kernel_size[1])/2)
        else:
            """if (self.iact_x_line_repetitions != 1):
                self.iact_converter_max_cycles = self.iact_x_line_repetitions*((self.iact_size_y + self.kernel_size[1]) - 1)
            else:"""
            self.iact_converter_max_cycles = self.buffer_cycles_for_x_iact*self.iact_x_line_repetitions*((self.iact_size_y + self.kernel_size[1]) - 1)
        if (self.buffer_cycles_for_x_iact == 1):
            self.iact_buffer_words_per_write = self.needed_Iact_writes * self.iact_x_line_repetitions*((self.iact_size_y + self.kernel_size[1]) - 1) * (4//2) 
        else:
            self.iact_buffer_words_per_write = self.needed_Iact_writes * (4//2)
        
        # === Phase 9: Finalize calculations ===
        self.calculate_transmission_cycles(params)
        self.calculate_needed_refreshes_mx(params)
        self.calculate_fpga_parameters(params)
        self.output_logger()
    def write_convdw_layer(self, layer, params):
        """Compute all configuration parameters for a Depthwise Convolution layer.

        Depthwise convolution applies a separate filter to each input channel independently,
        unlike standard convolution which combines all channels. This requires different
        PE allocation and data routing strategies.

        Args:
            layer: Keras DepthwiseConv2D layer object
            params: OpenEye hardware parameters

        Note:
            Sets data_mode = 1 to indicate depthwise mode.
            Depthwise layers typically have filters = kernel_size[0].
        """
        # Identify layer type
        self.layer_name = "DepthwiseConvolution"
                        
            
        self.strideX = layer.strides[0]
        self.strideY = layer.strides[1]
        self.data_mode = 1
        self.filters = layer.kernel_size[0]
        self.input_shape = layer.input.shape
        self.kernel_shape = layer.kernel.shape
        self.output_shape = layer.output.shape
        self.kernel_size = layer.kernel_size
        self.calculate_total_computations()

        self.needed_Iact_writes = 3
        self.kernel_per_pe_cluster = 1
        # Calculate the number of refreshes needed for the layer
        if((self.kernel_size[0] <= params.Iacts_per_PE) & (self.kernel_size[1] <= params.PEs_Y * params.Clusters_Y)):

            self.computing_mx = [[[[1 for _ in range(params.PEs_X)]
                                            for _ in range(params.PEs_Y)]
                                            for _ in range(params.Clusters_Y)]
                                            for _ in range(params.Clusters_X)]
            if(self.kernel_size[0] < 3):
                for x_cluster in range(params.Clusters_X):
                    for y_cluster in range(params.Clusters_Y):
                        for y_pe in range(params.PEs_Y):
                            for x_pe in range(params.PEs_X):
                                if((1 + y_pe) > self.kernel_size[1]):
                                    self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0

            if(((self.output_shape[1]) % (params.PEs_X * params.Clusters_X)) != 0):
                if((self.output_shape[1] < 8) | ((self.output_shape[1] > 12) & (self.output_shape[1] < 16))):
                    self.add_up = (params.PEs_X * params.Clusters_X)- (self.output_shape[1] % (params.PEs_X * params.Clusters_X))
                    yc_step = math.ceil(self.output_shape[1]/(params.PEs_X*params.Clusters_X))
                    yc_start = yc_step - 1
                    yc_end = params.Clusters_Y
                    for y_cluster in range(yc_start,yc_end,yc_step):
                        for x_cluster in range(math.floor((self.output_shape[1]%(params.Clusters_X*params.PEs_X)) / params.PEs_X),params.Clusters_X):
                            for x_pe in range(self.output_shape[1] % params.PEs_X,params.PEs_X):
                                for y_pe in range(params.PEs_Y):
                                    self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0
                else:
                    assert False, "Kernel cant be caclulated"
            else:
                self.add_up = 0

            self.used_channels = 1

            self.diff_iact_layer = math.ceil(self.input_shape[3]/self.used_channels)
            self.used_iact_per_PE = self.kernel_size[0] * self.used_channels
            self.iact_transmissions_pe = self.diff_iact_layer
            logger.debug("used_iact_per_PE " + str(self.used_iact_per_PE))
            logger.debug("iact_transmissions_pe " + str(self.iact_transmissions_pe))
            self.used_wght_per_PE = 2*self.used_iact_per_PE
            self.used_psum_per_PE = self.filters
            self.wght_transmissions_pe = 1
            if (self.output_shape[1] <= 16):
                self.single_cluster_computation = 2
            if (self.output_shape[1] <= 8):
                self.single_cluster_computation = 1

            match self.single_cluster_computation:
                case 1:
                    self.psum_transmissions_pe = math.ceil(self.output_shape[1] * self.output_shape[2]/(params.PEs_X*params.Clusters*params.Psums_per_PE))
                    self.iact_transmissions_pe = math.ceil(self.iact_transmissions_pe/params.Clusters)
                case 2:
                    self.psum_transmissions_pe = math.ceil(self.output_shape[1] * self.output_shape[2]/(params.PEs_X*params.Clusters_Y*params.Psums_per_PE))
                    self.iact_transmissions_pe = math.ceil(self.iact_transmissions_pe/params.Clusters_Y)
                case _:
                    self.psum_transmissions_pe = math.ceil(self.output_shape[1] * self.output_shape[2]/(params.PEs_X*params.Clusters*params.Psums_per_PE))
            
            #Calculation of seperate PE-Cluster
            self.used_PEs_Y    = self.kernel_size[1]
            used_PEs_per_clm     = self.used_PEs_Y/params.PEs_Y
            self.used_Y_cluster = math.ceil(used_PEs_per_clm)

            self.used_Y_cluster = (math.ceil(self.used_PEs_Y/params.PEs_Y))
            if((self.output_shape[2] % params.PEs_X)== 0):
                self.used_X_cluster = 1
            
            self.iact_data_len = math.ceil(self.used_iact_per_PE/(math.ceil(params.DMA_Bit_AXI/2)/params.IACT_WOH_Bitwidth))

            self.psum_transmissions_glb = math.ceil(((math.ceil(self.output_shape[1]/params.NUM_GLB_PSUM) * \
                                    self.output_shape[2]) / \
                                    params.Clusters_X / 2 / params.Clusters_Y) \
                                    / self.psum_transmissions_pe / self.wght_transmissions_pe / params.Psum_Mem_Addr_Words)
            
            self.needed_psum_transmissions = self.psum_transmissions_pe * self.psum_transmissions_glb
            self.wght_transmissions_glb = 1
            self.needed_wght_transmissions = self.wght_transmissions_pe * self.wght_transmissions_glb


            match self.single_cluster_computation:
                case 1:
                    self.Used_refreshes = math.ceil(math.ceil(self.output_shape[1] * self.output_shape[2])/params.PEs_X)*math.ceil(self.output_shape[3] / params.Clusters)
                case 2:
                    self.Used_refreshes = math.ceil(math.ceil(self.output_shape[1] * self.output_shape[2])/(params.PEs_X * params.Clusters_X))*math.ceil(self.output_shape[3] / params.Clusters_Y)
                case _:
                    self.Used_refreshes = math.ceil(self.output_shape[1] * self.output_shape[2]* self.output_shape[3]/(params.PEs_X*params.Clusters))

            self.used_iact_addr_per_PE = self.used_channels + 1
            logger.debug("Refreshes: " + str(self.Used_refreshes))
            logger.debug("self.used_channels : " + str(self.used_channels))
            logger.debug("self.kernel_size[0] : " + str(self.kernel_size[0]))
            logger.debug("self.needed_Iact_writes : " + str(self.needed_Iact_writes))
            logger.debug("self.needed_psum_transmissions : " + str(self.needed_psum_transmissions))
            self.iact_transmissions_glb = \
                math.ceil((self.needed_Iact_writes * math.ceil(self.Used_refreshes/self.needed_psum_transmissions/self.iact_transmissions_pe))/ \
                    math.floor(params.Iact_Mem_Addr_Words/(math.ceil((self.used_channels*self.kernel_size[0])/2))))

            match self.single_cluster_computation:
                case 1:
                    self.iact_transmissions_glb = math.ceil(self.iact_transmissions_glb/params.Clusters)
                case 2:
                    self.iact_transmissions_glb = math.ceil(self.iact_transmissions_glb/params.Clusters_Y)
                case _:
                    self.iact_transmissions_glb = self.iact_transmissions_glb


            self.needed_iact_transmissions = self.iact_transmissions_pe * self.iact_transmissions_glb

            self.needed_total_transmissions = self.needed_psum_transmissions * \
                                                        self.needed_wght_transmissions * \
                                                        self.needed_iact_transmissions
            self.needed_refreshes_mx = [[1 for _ in range(3)]
                                    for _ in range(self.needed_total_transmissions)]
            for layer_repetition in range(self.needed_total_transmissions):
                self.needed_refreshes_mx[layer_repetition][2] = math.floor(((math.floor(layer_repetition/self.iact_transmissions_pe)+1)/ \
                    self.needed_total_transmissions) * self.Used_refreshes)
                self.needed_refreshes_mx[layer_repetition][1] = math.floor((math.floor(layer_repetition/self.iact_transmissions_pe)/ \
                    self.needed_total_transmissions) * self.Used_refreshes)
                self.needed_refreshes_mx[layer_repetition][0] = self.needed_refreshes_mx[layer_repetition][2] - self.needed_refreshes_mx[layer_repetition][1]
            if (params.SERIAL == 1):
                self.psum_delay = int(max([(math.ceil(self.filters) - 4) - (self.used_Y_cluster * params.PEs_Y * 2),0]))
            else :
                self.psum_delay = int(max([(math.ceil(self.needed_refreshes_mx[layer_repetition][0]/2) - 2) - (self.used_Y_cluster * params.PEs_Y * 2),0]))
            logger.debug("Cycles: " + str(self.needed_refreshes_mx))
            logger.debug("Needed transmissions: " + str(self.needed_iact_transmissions))
            logger.debug("Needed transmissions: " + str(self.needed_wght_transmissions))
            logger.debug("Needed transmissions: " + str(self.needed_psum_transmissions))
            logger.debug("Needed transmissions: " + str(self.needed_total_transmissions))
            logger.debug("Needed transmissions IACT PE : " + str(self.iact_transmissions_pe))
            logger.debug("Needed transmissions WGHT PE : " + str(self.wght_transmissions_pe))
            logger.debug("Needed transmissions PSUM PE : " + str(self.psum_transmissions_pe))
            logger.debug("Needed transmissions IACT    : " + str(self.needed_iact_transmissions))
            logger.debug("Needed transmissions WGHT    : " + str(self.needed_wght_transmissions))
            logger.debug("Needed transmissions PSUM    : " + str(self.needed_psum_transmissions))
            logger.debug("Needed transmissions TOTAL   : " + str(self.needed_total_transmissions))
        else:
            logger.error("Can't fit model, kernel size must be adjusted.")
        return
    
    def write_dense_layer(self, layer_parameters, layer, params, layer_number, max_layers):
        """Compute all configuration parameters for a Dense (Fully Connected) layer.

        Dense layers perform matrix multiplication between flattened input and weight matrix.
        This requires different PE utilization patterns compared to convolutional layers,
        typically using only a subset of PEs in specific positions.

        Args:
            layer_parameters (list): Previously computed parameters for other layers
            layer: Keras Dense layer object
            params: OpenEye hardware parameters
            layer_number (int): Index of this layer (0-indexed)
            max_layers (int): Total number of layers in network

        Note:
            Sets fully_connected = 1 flag.
            Dense layers typically use only PE[0][0] in each cluster.
        """
        self.layer_name = "Dense"
        try:
            self.iact_size_x = layer.input.shape[3]
        except:
            self.iact_size_x = layer.input.shape[1]
        try:
            self.filters = layer.output.shape[3]
        except:
            self.filters = layer.output.shape[1]
        for f in range(self.filters):
            self.quantize[f][0] = 1
            self.quantize[f][1] = 4
        self.fully_connected = 1
        self.output_cycles = 1
        self.y_lines_per_calculation = 1
        self.kernel_size = [0]
        self.used_iact_addr_per_PE = 15
        if (layer_number == max_layers - 1):
            self.send_values_out = 1
        else:
            self.send_values_out = 0
        for f in range(self.filters):
            self.quantize[f][0] = 1
            self.quantize[f][1] = 9
        if (layer_number != 0): 
            self.skipIact = 1
        self.input_shape = layer.input.shape
        self.kernel_shape = layer.kernel.shape
        self.output_shape = layer.output.shape
            
        self.used_channels = params.NUM_GLB_IACT*math.ceil(self.iact_size_x/(params.Clusters_Y*params.NUM_GLB_IACT))
        # Calculate Iact Cycles
        self.needed_Iact_writes = math.ceil(params.PEs_Y/params.NUM_GLB_IACT)

        # Calculate the number of refreshes needed for the layer
        
        temp = math.ceil(self.iact_size_x/(params.Clusters_Y*params.PEs_Y))
        temp = min(temp, 12)
        temp = math.ceil(self.iact_size_x/(params.PEs_Y*temp))
        self.needed_wght_transmissions = math.ceil(temp/params.Clusters_Y)
        self.used_iact_per_PE = math.ceil(self.iact_size_x/(self.needed_wght_transmissions*params.Clusters_Y*params.PEs_Y))
        self.used_wght_per_PE = self.used_iact_per_PE * math.ceil(self.filters/params.Clusters_X/params.PARALLEL_MACS)*params.PARALLEL_MACS
        self.diff_iact_layer = math.ceil(self.iact_size_x/(params.NUM_GLB_WGHT*self.used_iact_per_PE))
        self.used_psum_per_PE = math.ceil(self.used_wght_per_PE/self.used_iact_per_PE)
        self.used_psum_per_PE = math.ceil(self.filters/params.Clusters_X)
        self.fsm_psum_limit = self.used_psum_per_PE + 3
        #self.needed_wght_transmissions = self.needed_wght_transmissions * 1
        
        self.used_Y_cluster = params.Clusters_Y
        self.used_X_cluster = 1
        self.kernel_per_pe_cluster = 1
        self.needed_iact_buffer_words = math.ceil(self.used_iact_per_PE/2)
        self.iact_converter_buffer_addr_max_cycles = self.needed_iact_buffer_words

        self.computing_mx = [[[[1 for _ in range(params.PEs_X)]
                                        for _ in range(params.PEs_Y)]
                                        for _ in range(params.Clusters_Y)]
                                        for _ in range(params.Clusters_X)]
        
        for x_cluster in range(params.Clusters_X):
            for y_cluster in range(params.Clusters_Y):
                for y_pe in range(params.PEs_Y):
                    for x_pe in range(params.PEs_X):
                        if(x_pe != 0):
                            self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0
        
        self.psum_transmissions_pe = math.ceil(1/16)
        self.psum_transmissions_glb = 1

        self.iact_transmissions_pe = math.ceil(1/(self.used_iact_per_PE * params.PEs_Y))
        self.iact_transmissions_glb = 1

        self.needed_psum_transmissions = self.psum_transmissions_pe * self.psum_transmissions_glb
        self.wght_transmissions_pe = 1
        self.wght_transmissions_glb = 1
        #self.needed_wght_transmissions = self.wght_transmissions_pe * self.wght_transmissions_glb
        self.needed_iact_transmissions = self.iact_transmissions_pe * self.iact_transmissions_glb
        self.Used_refreshes = self.iact_transmissions_pe * self.wght_transmissions_pe * self.psum_transmissions_pe

        self.iact_data_len = math.ceil(self.used_iact_per_PE/(math.ceil(params.DMA_Bit_AXI/2)/params.IACT_WOH_Bitwidth))
        self.psum_delay = int(max([((self.used_wght_per_PE/2/self.used_iact_per_PE) - 2) - (self.used_Y_cluster * params.PEs_Y * 2),0]))
        self.used_wght_addr_per_PE = (self.used_iact_per_PE) + 2
        if (self.used_wght_addr_per_PE >= 16):
            self.used_wght_addr_per_PE = 16

        self.needed_total_transmissions = 1
        self.needed_refreshes_mx = [[1 for _ in range(3)] for _ in range(self.needed_total_transmissions)]
        for layer_repetition in range(self.needed_total_transmissions):
            self.needed_refreshes_mx[layer_repetition][2] = math.floor(((math.floor(math.floor(layer_repetition/self.iact_transmissions_pe))+1)/ \
                self.needed_total_transmissions) * self.Used_refreshes)
            self.needed_refreshes_mx[layer_repetition][1] = math.floor((math.floor(math.floor(layer_repetition/self.iact_transmissions_pe))/ \
                self.needed_total_transmissions) * self.Used_refreshes)
            self.needed_refreshes_mx[layer_repetition][0] = self.needed_refreshes_mx[layer_repetition][2] - self.needed_refreshes_mx[layer_repetition][1]
            self.needed_refreshes_mx[layer_repetition][0] = math.ceil(self.diff_iact_layer/params.Clusters_Y)
        self.psum_storage_cycles = self.needed_wght_transmissions
        if (params.Clusters_Y == 1):
            self.psum_delay = 5
        self.limit_increase = math.floor((params.NUM_GLB_WGHT*self.used_iact_per_PE)/8)
        #self.limit_increase = math.floor((params.NUM_GLB_WGHT*self.used_iact_per_PE)%8)
        self.initial_upper_limit = self.limit_increase + 9
        self.iteration_for_kernels = math.ceil(self.diff_iact_layer_next_layer / self.different_kernels_per_calculation)
        self.buffer_cycles_for_x_iact = params.Clusters_Y
        self.iact_converter_max_cycles = 1
        print(self.used_iact_per_PE)
        self.iact_buffer_words_per_write = math.ceil(self.used_iact_per_PE/2) * self.needed_Iact_writes
        
        self.iact_size_c = self.used_iact_per_PE * params.NUM_GLB_WGHT * self.diff_iact_layer

        self.trans_cycles_iact = math.ceil(self.iact_size_x*self.iact_size_y*self.channels / params.IACT_WORDS_IN_RAM)
        self.trans_cycles_wght = params.NUM_GLB_WGHT * params.Clusters_Y * self.needed_wght_transmissions * (self.used_iact_per_PE * (math.ceil(self.filters / params.PARALLEL_MACS)))
        self.trans_cycles_psum = self.filters
        logger.debug("Needed transmissions: " + str(self.needed_wght_transmissions))
        logger.debug("Needed transmissions: " + str(self.needed_psum_transmissions))
        logger.debug("Needed transmissions: " + str(self.needed_total_transmissions))
            
        logger.debug("Needed transmissions IACT PE : " + str(self.iact_transmissions_pe))
        logger.debug("Needed transmissions WGHT PE : " + str(self.wght_transmissions_pe))
        logger.debug("Needed transmissions PSUM PE : " + str(self.psum_transmissions_pe))
        logger.debug("Needed transmissions IACT    : " + str(self.needed_iact_transmissions))
        logger.debug("Needed transmissions WGHT    : " + str(self.needed_wght_transmissions))
        logger.debug("Needed transmissions PSUM    : " + str(self.needed_psum_transmissions))
        logger.debug("Needed transmissions TOTAL   : " + str(self.needed_total_transmissions))
        return

    def write_pooling_layer(self, layer_parameters, layer, params, layer_number, max_layers):
        """Compute all configuration parameters for a Max Pooling layer.

        Pooling layers reduce spatial dimensions by taking the maximum value in each
        pooling window. Unlike other layers, pooling doesn't require weight loading
        and has simpler PE requirements.

        Args:
            layer_parameters (list): Previously computed parameters for other layers
            layer: Keras MaxPooling2D layer object
            params: OpenEye hardware parameters
            layer_number (int): Index of this layer (0-indexed)
            max_layers (int): Total number of layers in network

        Note:
            Sets max_pooling = 1 flag.
            Sets skipIact, skipWght, skipPsum = 1 (no weight/bias operations).
            Always sends output values to DRAM (send_values_out = 1).
        """
        self.layer_name = "Pooling"
        if ("max" in layer.name):
            self.pooling_mode = 0
        else:
            self.pooling_mode = 1
        self.input_shape = layer.input.shape
        self.output_shape = layer.output.shape
        self.skipIact = 1
        self.skipWght = 1
        self.skipPsum = 1
        self.channels = self.input_shape[3]
        self.max_pooling = 1
        self.send_values_out = 1
        self.strideX = layer.strides[0]
        self.strideY = layer.strides[1]
        self.iact_size_x = self.input_shape[1]
        self.iact_size_y = self.input_shape[2]
        self.computing_mx = [[[[1 for _ in range(params.PEs_X)]
                                        for _ in range(params.PEs_Y)]
                                        for _ in range(params.Clusters_Y)]
                                        for _ in range(params.Clusters_X)]
        self.diff_iact_layer = self.input_shape[3]
        if ((layer_parameters[max_layers - layer_number - 2].layer_name == "Dense") & (self.pooling_mode == 0)):
            self.used_channels = 1
        else:
            self.used_channels = 4
        self.diff_iact_layer_next_layer = layer_parameters[max_layers - layer_number - 2].used_channels
        self.iact_converter_buffer_addr_max_cycles = math.ceil((((self.iact_size_x*self.iact_size_y*self.diff_iact_layer)/32)/2)/2)
        return

    def print_layer_parameters(self, debug_file):
        """Print summary of computed layer parameters for debugging/analysis.

        Outputs key parameter values including PE allocation, memory usage per PE,
        and execution cycle counts. Useful for analyzing layer mapping and
        identifying performance bottlenecks.

        Args:
            debug_file: Debug file handle (currently unused, output goes to logger)

        Note:
            Uses logger.info() for output, so appears at INFO log level.
        """
        logger.info("params.PEs X: " + str(self.used_PEs_X) + "\n")
        logger.info("params.PEs Y: " + str(self.used_PEs_Y) + "\n")
        logger.info("Iact PE: " + str(self.used_iact_per_PE) + "\n")
        logger.info("Wght PE: " + str(self.used_wght_per_PE) + "\n")
        logger.info("Psum PE: " + str(self.used_psum_per_PE) + "\n")
        logger.info("Needed Cycles: " + str(self.Used_refreshes) + "\n")
        logger.info("Factor: " + str(self.current_highest_number) + "\n")
        logger.info("Real Factor: " + str(self.realfactor) + "\n")
        logger.info("End of Layer")
