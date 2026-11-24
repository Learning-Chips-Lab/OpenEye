# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Layer parameter computation and configuration for OpenEye accelerator.

This module provides the LayerParameters class which computes and stores all necessary
configuration parameters for executing neural network layers on the OpenEye hardware
accelerator. It handles Conv2D, Depthwise Convolution, Dense (Fully Connected), GEMM,
and Pooling layers by calculating PE allocation, memory transmissions, computation cycles,
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
    ...                                rel_layers, layer_num)
    >>> print(f"PEs required: {layer_params.used_PEs_X} x {layer_params.used_PEs_Y}")
    >>> print(f"Total cycles: {layer_params.Used_refreshes}")
"""

import math
import logging

logger = logging.getLogger("cocotb")


class LayerParameters:
    """Configuration parameters for a single neural network layer on OpenEye accelerator.

    This class computes and stores all hardware configuration parameters required to execute
    a neural network layer on the OpenEye accelerator. It determines PE allocation, memory
    bandwidth requirements, execution cycles, and data routing for Conv2D, Depthwise Conv,
    Dense, GEMM, and Pooling layers.

    Args:
        layer_parameters (list): Previously computed layer parameters (for inter-layer dependencies)
        layer: A Keras/TensorFlow layer object with weights, shape, and configuration
        params: OpenEye hardware parameters (PE counts, memory sizes, cluster config, etc.)
        rel_layers (dict): Graph connectivity dictionary ("Previous_Layer", "Next_Layer")
        layer_number (int): Index of the layer in the model

    Raises:
        ValueError: If layer type is unsupported or kernel size exceeds hardware constraints
    """

    def __init__(self, layer_parameters, layer, params, rel_layers, layer_number):
        # === Layer Identification ===
        self.use_layer_on_hw = True
        self.rel_layers = rel_layers
        self.prev_layers = []
        self.next_layers = []
        self.layer_number_in_model = layer_number
        self.layer_name = layer.name
        self.pooling_mode = 0

        # === PE Allocation Parameters ===
        self.used_PEs_X = 1
        self.used_PEs_Y = 0
        self.used_X_cluster = 1
        self.used_Y_cluster = 1
        self.Used_refreshes = 0
        self.current_input_X = 0
        self.current_input_Y = 0
        self.padding = "same"
        self.needed_refreshes_mx = []
        self.calc_X = 0
        self.calc_Y = 0
        self.total_computations = 0
        self.Y_Cluster_Packages = 1
        self.available_X_Cluster_Packages = params.Clusters_X

        # === Memory Usage per PE ===
        self.used_iact_per_PE = 1
        self.used_wght_per_PE = 1
        self.used_psum_per_PE = 1
        self.diff_iact_layer = []
        self.diff_iact_layer_next_layer = 0
        self.used_PEs_per_col = 0

        # === Data Transmission Parameters ===
        self.needed_Iact_writes = 0
        self.needed_iact_buffer_words = 0
        self.cont_iact_channels_glb = 2

        # === Miscellaneous Calculation Parameters ===
        self.current_highest_number = 0
        self.realfactor = 0
        self.used_iact_addr_per_PE = 2
        self.used_wght_addr_per_PE = 5
        self.iact_addr_len = 1
        self.iact_data_len = 3
        self.choose_iact_storage_input = 0
        self.choose_iact_storage_output = 0

        # === Convolution Parameters ===
        self.strideX = 1
        self.strideY = 1
        self.add_up = 0
        self.psum_add_up = 0
        self.complete_iacts_in_design = 0
        self.max_pooling = 0
        self.output_cycles = 0

        # === Layer Shape Information ===
        self.filters = 1
        self.channels = 1
        self.input_shape = []
        self.kernel_shape = []
        self.output_shape = []
        self.kernel_size = []
        self.padding_x = 0
        self.padding_y = 0
        self.kernel_per_pe_cluster = 1
        self.used_channels = 1
        self.channel_repetition = 4
        self.single_cluster_computation = 0
        self.iact_size_x = 1
        self.iact_size_y = 1
        self.iact_size_c = 1
        self.psum_size_x = 1
        self.psum_size_y = 1

        # === Transmission Count Defaults ===
        self.iact_transmissions_pe = 1
        self.wght_transmissions_pe = 1
        self.psum_transmissions_pe = 1
        self.all_transmissions_of_pe = 1
        self.iact_transmissions_glb = 1
        self.wght_transmissions_glb = 1
        self.psum_transmissions_glb = 1
        self.psum_storage_cycles = 1
        self.needed_iact_transmissions = 1
        self.needed_wght_transmissions = 1
        self.needed_psum_transmissions = 1
        self.needed_total_transmissions = 1
        self.psum_delay = 0
        self.fully_connected = 0
        self.gemm_mode = 0
        self.store_in_psum = 0
        self.limit_increase = 0
        self.limit_increase_mod = 0
        self.lower_bound = 0
        self.upper_bound = 0
        self.iteration_for_kernels = 1
        self.needed_wght_cycles = 1
        self.fsm_psum_limit = 1
        self.cluster_per_conv_cycle = 4
        self.iact_converter_max_cycles = 1
        self.iact_x_pos_inc = 1
        self.iact_buffer_words_per_write = 12
        self.iact_words_per_compute = 0
        self.needed_cycles = 1
        self.trans_cycles_iact = 1
        self.trans_cycles_wght = 1
        self.trans_cycles_psum = 1
        self.iact_cycles_one_word_all_ram = 1
        self.iact_repetitions_per_write = 0
        self.overhang_discrepancy = 0
        self.psum_output_words = 1
        self.wght_cycles_one_word_all_ram = 0
        self.iact_glb_writing_cycles = 0
        self.concat_limit = 0

        # === Control Flags ===
        self.send_values_out = 1
        self.skipIact = 0
        self.skipWght = 0
        self.skipPsum = 0
        self.iact_stream_cycles = 1

        # === Computing Matrix and Mode ===
        self.computing_mx = 0
        self.data_mode = 0
        self.iact_x_per_cluster = 0
        self.iact_x_add_up = 0
        self.psum_x_all_cluster = 0
        self.y_lines_per_calculation = 1
        self.iact_x_line_repetitions = 1
        self.different_kernels_per_calculation = 1
        self.buffer_cycles_for_x_iact = 1
        self.start_param_array = 1
        self.initial_upper_limit = 0
        self.channel_div_trans = 0

        # === FPGA-Specific Parameters ===
        self.iact_converter_buffer_addr_max_cycles = 0
        self.iact_x_lines = 3
        self.quantize = [[0, 0] for _ in range(params.QUANT_AMOUNT)]
        self.offset = [0] * params.QUANT_AMOUNT

        # === Programmable Address Generation Unit ===
        self.iact_read_limit_0 = 0
        self.iact_read_limit_1 = 0
        self.iact_read_limit_2 = 0
        self.iact_read_limit_3 = 0
        self.iact_read_limit_4 = 0
        self.iact_read_inc_0 = 0
        self.iact_read_inc_1 = 0
        self.iact_read_inc_2 = 0
        self.iact_read_inc_3 = 0
        self.iact_read_inc_4 = 0
        self.iact_write_limit_0 = 0
        self.iact_write_limit_1 = 0
        self.iact_write_limit_2 = 0
        self.iact_write_limit_3 = 0
        self.iact_write_limit_4 = 0
        self.iact_write_inc_0 = 0
        self.iact_write_inc_1 = 0
        self.iact_write_inc_2 = 0
        self.iact_write_inc_3 = 0
        self.pagu_wght_limit = 0
        self.psum_pagu_cs_limit = [0, 0, 0, 0, 0, 0]
        self.psum_pagu_cs_inc = [0, 0, 0, 0, 0, 0]
        self.psum_pagu_loop_limit = [0, 0, 0, 0, 0, 0]
        self.psum_pagu_addr_inc = [0, 0, 0, 0, 0, 0]

        # === Layer Type Detection and Dispatch ===
        name = layer.name
        if "depthwise_conv2d" in name:
            logger.debug("Depthwise Convolution Layer")
            self.write_convdw_layer(layer_parameters, layer, params)

        elif "conv2d" in name:
            logger.debug("2D Convolution Layer")
            self.write_conv2d_layer(layer_parameters, layer, params)

        elif "dense" in name:
            logger.debug("Dense Layer")
            self.write_dense_layer(layer_parameters, layer, params)

        elif "gemm" in name or "matmul" in name:
            logger.debug("GEMM Layer (output-stationary)")
            self.write_gemm_layer(layer_parameters, layer, params)

        elif "pooling2d" in name:
            logger.debug("Pooling Layer")
            self.write_pooling_layer(layer_parameters, layer, params)

        elif "add" in name:
            logger.debug("Adding Layer")
            self.output_shape = layer.output.shape
            self.use_layer_on_hw = False

        elif "concat" in name:
            logger.debug("Concat Layer")
            self.output_shape = layer.output.shape
            self.use_layer_on_hw = False

        elif "input" in name:
            logger.debug("Input Layer")
            self.use_layer_on_hw = False

        else:
            logger.debug("Layer type for %s not supported.", name)
            raise ValueError(f"Layer type for '{name}' not supported.")

    def __repr__(self):
        return (
            f"LayerParameters(name={self.layer_name!r}, hw={self.use_layer_on_hw}, "
            f"in={tuple(self.input_shape)}, out={tuple(self.output_shape)}, "
            f"PEs={self.used_PEs_X}x{self.used_PEs_Y}, refreshes={self.Used_refreshes})"
        )

    def _get_prev_output_slot(self, prev_name):
        """Return the output storage slot index used by a preceding layer."""
        prev_next = self.rel_layers["Next_Layer"][prev_name]
        if len(prev_next) > 1:
            return prev_next.index(self.layer_name)
        prev_prev_list = self.rel_layers["Previous_Layer"].get(prev_name, [])
        if prev_prev_list and "input_layer" not in prev_prev_list[0]:
            prev_prev = prev_prev_list[0]
            siblings = self.rel_layers["Next_Layer"][prev_prev]
            if len(siblings) > 1:
                return siblings.index(prev_name)
        return 0

    def check_for_multiple_lines_per_computation(self, params):
        """Determine how many Y lines and kernels can be processed simultaneously."""
        self.used_PEs_Y = self.kernel_size[0] * self.kernel_per_pe_cluster
        self.used_Y_cluster = math.ceil(self.used_PEs_Y / params.PEs_Y)
        usable_pes = params.PEs_X * math.floor(params.Clusters / self.used_Y_cluster)

        if math.ceil(self.output_shape[1] / params.PEs_X) >= math.ceil(usable_pes / params.PEs_X):
            self.iact_x_line_repetitions = math.ceil(self.output_shape[1] / usable_pes)
            self.y_lines_per_calculation = 1
            self.different_kernels_per_calculation = 1
        else:
            self.iact_x_line_repetitions = 1
            aligned_psum_x = math.ceil(self.psum_size_x / params.NUM_GLB_PSUM) * params.NUM_GLB_PSUM
            kernels = usable_pes // aligned_psum_x
            if kernels == 3:
                kernels = 2
            elif kernels >= 4:
                kernels = 4 * (kernels // 4)
            self.different_kernels_per_calculation = min(kernels, math.ceil(self.filters / 4))
            # Currently disabled - process one Y line at a time
            self.y_lines_per_calculation = 1

    def calculate_used_Y_cluster(self, params):
        """Calculate the number of Y-direction clusters needed for this layer."""
        self.used_Y_cluster = math.ceil(self.used_PEs_Y / params.PEs_Y)
        self.Y_Cluster_Packages = math.floor(params.Clusters_Y / self.used_Y_cluster)
        self.used_Y_cluster += math.floor(
            (params.Clusters_Y - (self.used_Y_cluster * self.Y_Cluster_Packages)) / self.Y_Cluster_Packages
        )
        self.available_X_Cluster_Packages = self.Y_Cluster_Packages * params.Clusters_X

    def calculate_total_computations(self):
        """Calculate the total number of spatial output positions for the convolution."""
        if self.padding == "same":
            self.calc_X = self.input_shape[1]
            self.calc_Y = self.input_shape[2]
        else:
            self.calc_X = self.input_shape[1] - self.kernel_size[0] + 1
            self.calc_Y = self.input_shape[2] - self.kernel_size[1] + 1
        self.total_computations = self.calc_X * self.calc_Y

    def calculate_iact_transmissions(self, params):
        """Calculate the number of activation write operations needed per computation."""
        total_iact_width = (
            (params.PEs_X - 1) * self.strideX * self.kernel_per_pe_cluster
            + (self.kernel_size[1] * self.kernel_per_pe_cluster)
        )
        self.needed_Iact_writes = math.ceil(total_iact_width / params.NUM_GLB_IACT)

    def calculate_computing_matrix(self, params):
        """Generate a 4D binary matrix [clx][cly][pey][pex] indicating which PEs are active."""
        kernel_h = self.kernel_size[0]
        kernel_w = self.kernel_size[1]
        pes_x = params.PEs_X
        pes_y = params.PEs_Y
        clusters_x = params.Clusters_X
        clusters_y = params.Clusters_Y

        if not (kernel_h <= params.Iacts_per_PE and kernel_w <= pes_y * clusters_y):
            logger.error("Can't fit model, kernel size must be adjusted.")
            raise ValueError("Can't fit model, kernel size must be adjusted.")

        # === MASKING PHASE 1: Handle non-aligned output width ===
        needed_cluster_rows = math.ceil(self.psum_size_x / clusters_x * pes_x)
        if clusters_y >= needed_cluster_rows:
            total_x_slots = needed_cluster_rows * clusters_x * pes_x
            self.psum_add_up = (total_x_slots - (self.output_shape[1] % total_x_slots)) % total_x_slots
        else:
            total_x_slots = clusters_y * pes_x * clusters_x
            self.psum_add_up = (total_x_slots - (self.psum_size_x % total_x_slots)) % total_x_slots

        self.add_up = self.psum_add_up * self.strideX
        self.iact_x_add_up = self.iact_size_x + self.add_up

        # === MASKING PHASE 2 & 3: Single-pass active PE construction ===
        x_values_per_cycle = (
            self.iact_x_add_up * self.y_lines_per_calculation * self.different_kernels_per_calculation
        )
        kernel_h_total = kernel_h * self.kernel_per_pe_cluster
        used_y_cl = self.used_Y_cluster
        row_stride_x = pes_x * clusters_x

        self.computing_mx = [
            [
                [
                    [
                        1
                        if (
                            (x_cluster * pes_x + (y_cluster // used_y_cl) * row_stride_x + x_pe)
                            < x_values_per_cycle
                            and (1 + y_pe + (y_cluster % used_y_cl) * pes_y) <= kernel_h_total
                        )
                        else 0
                        for x_pe in range(pes_x)
                    ]
                    for y_pe in range(pes_y)
                ]
                for y_cluster in range(clusters_y)
            ]
            for x_cluster in range(clusters_x)
        ]

    def output_logger(self):
        """Log computed parameters when DEBUG logging is enabled."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug("Refreshes: %s", self.Used_refreshes)
        logger.debug("Used complete new descriptions: %s", self.Used_refreshes)
        logger.debug("self.used_channels : %s", self.used_channels)
        logger.debug("layer_params.needed_Iact_writes : %s", self.needed_Iact_writes)
        logger.debug("Used_refreshes : %s", self.Used_refreshes)
        logger.debug("layer_params.needed_psum_transmissions : %s", self.needed_psum_transmissions)
        logger.debug("Needed transmissions IACT PE : %s", self.iact_transmissions_pe)
        logger.debug("Needed transmissions WGHT PE : %s", self.wght_transmissions_pe)
        logger.debug("Needed transmissions PSUM PE : %s", self.psum_transmissions_pe)
        logger.debug("Needed transmissions IACT    : %s", self.needed_iact_transmissions)
        logger.debug("Needed transmissions WGHT    : %s", self.needed_wght_transmissions)
        logger.debug("Needed transmissions PSUM    : %s", self.needed_psum_transmissions)
        logger.debug("Needed transmissions TOTAL   : %s", self.needed_total_transmissions)

    def read_data_from_layer(self, layer):
        """Extract shape and configuration parameters from a Keras Conv2D/DW layer."""
        self.fully_connected = 0

        self.input_shape = layer.input.shape
        self.kernel_shape = layer.kernel.shape
        self.output_shape = layer.output.shape
        self.kernel_size = layer.kernel_size

        self.iact_size_x = self.input_shape[1]
        self.iact_size_y = self.input_shape[2]
        self.psum_size_x = self.output_shape[1]
        self.psum_size_y = self.output_shape[2]
        self.channels = self.input_shape[3]
        self.filters = getattr(layer, "filters", self.channels)

        if self.padding == "same":
            self.padding_x = self.kernel_size[0] // 2
            self.padding_y = self.kernel_size[1] // 2
        self.strideX = layer.strides[0]
        self.strideY = layer.strides[1]

        if hasattr(layer, "store_in_psum"):
            self.store_in_psum = layer.store_in_psum
        if hasattr(layer, "skip_psum"):
            self.skipPsum = layer.skip_psum

        self.prev_layers = self.rel_layers["Previous_Layer"][self.layer_name]
        self.next_layers = self.rel_layers["Next_Layer"][self.layer_name]

        # 1. Determine input storage slot
        if not self.prev_layers or "input_layer" in self.prev_layers[0]:
            self.choose_iact_storage_input = 0
        elif len(self.prev_layers) > 1:
            self.choose_iact_storage_input = [
                self.rel_layers["Next_Layer"][prev].index(self.layer_name)
                if len(self.rel_layers["Next_Layer"][prev]) > 1
                else self._get_prev_output_slot(prev)
                for prev in self.prev_layers
            ]
        else:
            prev = self.prev_layers[0]
            prev_next = self.rel_layers["Next_Layer"][prev]
            if len(prev_next) > 1:
                self.choose_iact_storage_input = len(prev_next) - 1
            else:
                prev_prev = self.rel_layers["Previous_Layer"][prev][0]
                if len(self.rel_layers["Next_Layer"][prev_prev]) > 1:
                    self.choose_iact_storage_input = self.rel_layers["Next_Layer"][prev_prev].index(prev)
                else:
                    self.choose_iact_storage_input = 0

        # 2. Determine output storage slot
        if len(self.next_layers) > 1:
            self.choose_iact_storage_output = len(self.next_layers) - 1
        elif self.prev_layers and "input_layer" not in self.prev_layers[0]:
            prev = self.prev_layers[0]
            siblings = self.rel_layers["Next_Layer"][prev]
            self.choose_iact_storage_output = siblings.index(self.layer_name) if len(siblings) > 1 else 0
        else:
            self.choose_iact_storage_output = 0

        # 3. Quantization parameters
        num_q = min(self.filters, len(self.quantize))
        if hasattr(layer, "quantization_factor"):
            q_factors = layer.quantization_factor
            for f in range(num_q):
                self.quantize[f][0] = q_factors[f][0]
                self.quantize[f][1] = q_factors[f][1]
        else:
            for f in range(num_q):
                self.quantize[f][0] = 1
                self.quantize[f][1] = 7

    def choose_iact_location(self):
        """Configure activation storage and data flow flags based on layer position."""
        self.send_values_out = 1 if len(self.rel_layers["Next_Layer"][self.layer_name]) == 0 else 0
        if self.rel_layers["Previous_Layer"][self.layer_name] != ["input_layer"]:
            self.skipIact = 1

    def calculate_used_channels(self, params):
        """Calculate how many input channels are processed per iteration."""
        if 2 * self.kernel_size[0] <= params.PEs_Y:
            temp = math.ceil(self.channels / self.kernel_per_pe_cluster)
            capacity = params.Iacts_per_PE if self.kernel_size[0] >= 2 else params.Iacts_Addr_per_PE
            divisor = math.ceil(temp / capacity)
            self.used_channels = math.ceil(temp / divisor)
        else:
            raw_channels = 16 // self.kernel_size[1]
            assert raw_channels > 0, "Kernel width exceeds channel capacity"
            self.used_channels = 1 << (raw_channels.bit_length() - 1)

        self.used_channels = min(self.channels, self.used_channels)
        self.channel_div_trans = math.ceil(self.used_channels / params.MAX_CHANNELS_IACT_GLB)
        self.cont_iact_channels_glb = min(self.used_channels, params.MAX_CHANNELS_IACT_GLB)

    def calculate_needed_refreshes_mx(self, params):
        """Generate a matrix of refresh cycle counts [delta, start, end] per transmission iteration."""
        if params.SERIAL == 1:
            self.needed_total_transmissions = 1
            self.needed_refreshes_mx = [[self.Used_refreshes, 0, self.Used_refreshes]]
        else:
            total = (
                self.needed_psum_transmissions
                * self.needed_wght_transmissions
                * self.needed_iact_transmissions
            )
            self.needed_total_transmissions = total
            used_y_cl = self.used_Y_cluster
            used_ref = self.Used_refreshes
            step_div = self.iact_transmissions_pe * self.needed_wght_transmissions

            mx = []
            for rep in range(total):
                block = rep // step_div
                end_ref = math.floor(((block + 1) / total) * used_ref)
                end_ref -= end_ref % used_y_cl

                start_ref = math.floor((block / total) * used_ref)
                start_ref -= start_ref % used_y_cl

                mx.append([end_ref - start_ref, start_ref, end_ref])
            self.needed_refreshes_mx = mx

        self.needed_cycles = self.Used_refreshes

    def calculate_used_refreshes(self, params):
        """Calculate total number of refresh cycles needed to execute this layer."""
        pes_per_line = ((params.Clusters // self.used_Y_cluster) // self.different_kernels_per_calculation) * params.PEs_X
        spatial_refreshes = math.ceil(self.output_shape[2] * math.ceil(self.output_shape[1] / pes_per_line))
        filter_passes = math.ceil(math.ceil(self.filters / self.different_kernels_per_calculation) / self.used_psum_per_PE)
        self.Used_refreshes = math.ceil(self.iact_transmissions_pe * spatial_refreshes * filter_passes)

    def calculate_single_cluster_computation(self, params):
        """Determine if layer can use single-cluster optimization mode in parallel mode."""
        if params.SERIAL == 0:
            if self.output_shape[1] <= 8:
                self.single_cluster_computation = 2
            if self.output_shape[1] <= 4:
                self.single_cluster_computation = 1

    def calculate_pe_transmissions(self, params):
        """Calculate transmission iterations needed at the PE level for IACT/WGHT/PSUM."""
        self.iact_transmissions_pe = math.ceil(self.diff_iact_layer / self.kernel_per_pe_cluster)

        if ((self.filters * self.used_iact_per_PE) // self.different_kernels_per_calculation) <= params.Wghts_per_PE:
            if self.filters // self.different_kernels_per_calculation <= 32:
                self.used_psum_per_PE = math.ceil(self.filters / self.different_kernels_per_calculation)
                self.wght_transmissions_pe = math.ceil(
                    self.channels / (self.used_channels * self.kernel_per_pe_cluster)
                )
            else:
                self.used_psum_per_PE = 32
                self.wght_transmissions_pe = math.ceil(self.channels / self.channel_div_trans)
            self.used_wght_per_PE = self.used_psum_per_PE * self.used_iact_per_PE
        else:
            capacity = 160 if self.kernel_size[0] == 5 else params.Wghts_per_PE
            wght_factor = math.ceil((self.filters * self.used_iact_per_PE) / capacity)
            self.used_wght_per_PE = math.ceil(self.filters / wght_factor) * self.used_iact_per_PE
            self.used_psum_per_PE = int(self.filters / wght_factor)
            self.wght_transmissions_pe = (
                math.ceil(self.channels / (self.used_channels * self.kernel_per_pe_cluster))
                * math.ceil(
                    self.filters
                    * self.used_iact_per_PE
                    / self.used_wght_per_PE
                    / self.different_kernels_per_calculation
                )
            )

        if math.ceil(self.used_iact_per_PE / self.used_wght_per_PE) <= params.Psums_per_PE:
            self.psum_transmissions_pe = 1
        else:
            self.psum_transmissions_pe = math.ceil(self.filters / params.Psums_per_PE)
            logger.debug("Error Code 5, Overused PSUM per PE, not implemented flow yet")

        self.all_transmissions_of_pe = (
            self.iact_transmissions_pe * self.wght_transmissions_pe * self.psum_transmissions_pe
        )

    def calculate_glb_transmissions(self, params):
        """Calculate transmission iterations needed at the Global Buffer (GLB) level."""
        self.psum_transmissions_glb = math.ceil(
            (
                (
                    math.ceil(self.output_shape[1] / params.NUM_GLB_PSUM)
                    * self.output_shape[2]
                    * math.ceil(self.output_shape[3] / self.wght_transmissions_pe)
                )
                / params.Clusters_X
                / 2
                / params.Clusters_Y
            )
            / self.psum_transmissions_pe
            / self.wght_transmissions_pe
            / params.Psum_Mem_Addr_Words
        )
        self.wght_transmissions_glb = 1

        self.needed_psum_transmissions = self.psum_transmissions_pe * self.psum_transmissions_glb
        self.needed_wght_transmissions = self.wght_transmissions_pe * self.wght_transmissions_glb

        if params.SERIAL == 1:
            self.iact_transmissions_glb = 1
        else:
            words_per_pass = (
                math.ceil(self.channel_div_trans * self.kernel_size[0])
                + math.ceil((self.used_channels + 1) / 6)
            ) * self.needed_Iact_writes
            self.iact_transmissions_glb = math.ceil(
                math.ceil(
                    self.Used_refreshes
                    / self.wght_transmissions_pe
                    / self.needed_psum_transmissions
                    / self.iact_transmissions_pe
                )
                / math.floor(params.Iact_Mem_Addr_Words / words_per_pass)
            )

        self.needed_iact_transmissions = self.iact_transmissions_pe * self.iact_transmissions_glb

    def calculate_fpga_parameters(self, params):
        """Calculate FPGA-specific timing and activation line buffering parameters."""
        self.iact_x_lines = self.kernel_size[1] + self.iact_size_y - 1
        self.iact_converter_buffer_addr_max_cycles = self.channel_div_trans * self.needed_Iact_writes

    def get_pagu_values(self, params):
        """Configure Programmable Address Generation Unit (PAGU) limits and increments."""
        is_depthwise = "depthwise" in self.layer_name
        channel_group = 4 if is_depthwise else 2

        # === IACT Read PAGU ===
        self.iact_read_limit_0 = (self.channel_div_trans * self.needed_Iact_writes) - 1
        self.iact_read_limit_1 = math.ceil(self.diff_iact_layer) - 1
        self.iact_read_limit_2 = 0 if is_depthwise else (self.needed_wght_cycles - 1)
        self.iact_read_limit_3 = self.iact_x_line_repetitions - 1
        self.iact_read_limit_4 = math.ceil(self.iact_size_y / self.strideY) - 1

        self.iact_read_inc_0 = self.iact_repetitions_per_write
        self.iact_read_inc_1 = math.ceil(self.iact_buffer_words_per_write / self.diff_iact_layer)
        self.iact_read_inc_2 = 0

        if self.Y_Cluster_Packages == 1 and params.Clusters_X == 1:
            self.iact_read_inc_3 = self.channel_div_trans * params.NUM_GLB_PSUM * self.strideX
        else:
            self.iact_read_inc_3 = self.channel_div_trans * self.needed_Iact_writes * params.NUM_GLB_IACT
            cluster_span = (params.Clusters_X * self.Y_Cluster_Packages) - 1
            if cluster_span * (params.NUM_GLB_PSUM * self.strideX) < self.kernel_shape[0] - 1:
                self.iact_read_inc_3 += (
                    cluster_span * ((params.NUM_GLB_PSUM + 1) * self.strideX) - self.kernel_shape[0]
                )
        self.iact_read_inc_4 = self.iact_read_inc_0 * self.strideY

        # === IACT Write PAGU ===
        self.iact_write_limit_0 = (
            math.ceil(self.iact_buffer_words_per_write / self.channel_div_trans / self.diff_iact_layer) - 1
        )
        self.iact_write_limit_1 = self.channel_div_trans - 1
        self.iact_write_limit_2 = self.diff_iact_layer - 1
        self.iact_write_limit_3 = 0
        self.iact_write_limit_4 = 0

        self.iact_write_inc_0 = self.channel_div_trans
        self.iact_write_inc_1 = 1
        self.iact_write_inc_2 = math.ceil(self.iact_buffer_words_per_write / self.diff_iact_layer)
        self.iact_write_inc_3 = 0

        # === WGHT PAGU ===
        self.pagu_wght_limit = (
            int((self.iact_x_add_up % params.NUM_GLB_PSUM) + self.iact_x_add_up)
            * self.y_lines_per_calculation
            * self.different_kernels_per_calculation
            * self.used_Y_cluster
        )

        # === PSUM PAGU ===
        max_channels = min(self.different_kernels_per_calculation, channel_group)
        self.psum_pagu_cs_limit[0] = max_channels - 1
        self.psum_pagu_cs_limit[1] = math.ceil(channel_group / max_channels) - 1
        self.psum_pagu_cs_limit[2] = 0
        self.psum_pagu_cs_limit[3] = (
            math.ceil(self.Y_Cluster_Packages * params.Clusters_X / self.different_kernels_per_calculation) - 1
        )
        self.psum_pagu_cs_limit[4] = self.iact_x_line_repetitions * self.psum_size_y - 1
        self.psum_pagu_cs_limit[5] = math.floor(self.different_kernels_per_calculation / max_channels) - 1

        if self.different_kernels_per_calculation == 1:
            self.psum_pagu_cs_inc[0] = 0
        else:
            self.psum_pagu_cs_inc[0] = math.ceil(self.used_X_cluster / self.different_kernels_per_calculation)
        self.psum_pagu_cs_inc[1] = 0
        self.psum_pagu_cs_inc[2] = 0
        self.psum_pagu_cs_inc[3] = 1
        self.psum_pagu_cs_inc[4] = 0
        self.psum_pagu_cs_inc[5] = max_channels * (
            self.used_Y_cluster * math.floor(self.used_X_cluster / self.different_kernels_per_calculation)
        )

        self.compact_and_pad_in_place(self.psum_pagu_cs_limit, self.psum_pagu_cs_inc)

        self.psum_pagu_loop_limit[0] = max_channels - 1
        self.psum_pagu_addr_inc[0] = 0

        self.psum_pagu_loop_limit[1] = math.ceil(channel_group / max_channels) - 1
        self.psum_pagu_addr_inc[1] = 1

        self.psum_pagu_loop_limit[2] = math.ceil(self.used_X_cluster / self.different_kernels_per_calculation) - 1
        self.psum_pagu_addr_inc[2] = 0

        self.psum_pagu_loop_limit[3] = self.iact_x_line_repetitions * self.psum_size_y - 1
        self.psum_pagu_addr_inc[3] = math.ceil(self.filters / self.different_kernels_per_calculation)

        self.psum_pagu_loop_limit[4] = math.floor(self.different_kernels_per_calculation / max_channels) - 1
        self.psum_pagu_addr_inc[4] = 0

        self.psum_pagu_loop_limit[5] = math.ceil(self.filters / channel_group) - 1
        self.psum_pagu_addr_inc[5] = math.ceil(channel_group / max_channels)

        self.compact_and_pad_in_place(self.psum_pagu_loop_limit, self.psum_pagu_addr_inc)

    @staticmethod
    def compact_and_pad_in_place(primary_list: list, secondary_list: list) -> None:
        """Compact non-zero entries in primary_list (and corresponding secondary_list) to the front."""
        write_idx = 0
        n = len(primary_list)

        for read_idx in range(n):
            if primary_list[read_idx] != 0:
                primary_list[write_idx] = primary_list[read_idx]
                secondary_list[write_idx] = secondary_list[read_idx]
                write_idx += 1

        while write_idx < n:
            primary_list[write_idx] = 0
            secondary_list[write_idx] = 0
            write_idx += 1

    def calculate_transmission_cycles(self, params):
        """Calculate DMA transmission cycle counts for IACT, WGHT, and PSUM streams."""
        self.iact_cycles_one_word_all_ram = math.ceil(
            (params.IACT_RAM_CELLS * params.IACT_RAM_CELLS_WORD_BITWIDTH) / params.DMA_BITWIDTH
        )
        raw_iact_cycles = math.ceil(
            params.IACT_Bitwidth * self.iact_size_x * self.iact_size_y * self.channels / params.DMA_BITWIDTH
        )
        missing_cycles = (
            self.iact_cycles_one_word_all_ram - (raw_iact_cycles % self.iact_cycles_one_word_all_ram)
        ) % self.iact_cycles_one_word_all_ram
        self.trans_cycles_iact = raw_iact_cycles + missing_cycles

        self.wght_cycles_one_word_all_ram = math.ceil(
            (params.Clusters * params.NUM_GLB_WGHT * params.WGHT_RAM_CELLS_WORD_BITWIDTH) / params.DMA_BITWIDTH
        )
        self.trans_cycles_wght = (
            self.needed_wght_transmissions
            * math.ceil(self.used_wght_per_PE / params.PARALLEL_MACS)
            * math.ceil(params.NUM_GLB_WGHT * params.Clusters * params.WGHT_Trans_Bitwidth / params.DMA_BITWIDTH)
        )

        psum_words_per_line = math.ceil(
            params.NUM_GLB_PSUM * (params.Clusters * params.DATA_PSUM_BITWIDTH / params.DMA_BITWIDTH)
        )
        self.trans_cycles_psum = self.filters * self.iact_size_y * self.iact_x_line_repetitions * psum_words_per_line

    def _write_conv_common(self, layer_parameters, layer, params, is_depthwise):
        """Shared parameter computation pipeline for Conv2D and Depthwise Conv2D layers."""
        # === Phase 1: Read layer configuration ===
        self.read_data_from_layer(layer)
        self.choose_iact_location()

        # === Phase 2: Calculate computation requirements ===
        self.calculate_total_computations()
        self.check_for_multiple_lines_per_computation(params)
        self.calculate_single_cluster_computation(params)
        self.iact_x_per_cluster = self.strideX * params.NUM_GLB_PSUM

        # === Phase 3: Calculate PE allocation ===
        if math.floor(params.PEs_Y / self.kernel_size[0]) > 1:
            self.kernel_per_pe_cluster = 1  # Hardcoded to 1 kernel per cluster

        # === Phase 4: Calculate activation handling ===
        self.calculate_iact_transmissions(params)
        self.calculate_used_channels(params)
        self.used_iact_addr_per_PE = self.used_channels

        self.iact_stream_cycles = math.ceil(
            self.input_shape[1]
            * self.input_shape[2]
            * self.input_shape[3]
            / params.IACT_RAM_CELLS
            / (params.DMA_BITWIDTH // params.IACT_Bitwidth)
        )

        self.diff_iact_layer = math.ceil(self.input_shape[3] / self.used_channels)
        next_layers = self.rel_layers["Next_Layer"][self.layer_name]
        if next_layers:
            if is_depthwise:
                self.diff_iact_layer_next_layer = layer_parameters[next_layers[0]].used_channels
            else:
                self.diff_iact_layer_next_layer = 2

        self.used_iact_per_PE = self.kernel_size[1] * self.used_channels

        # === Phase 5: Calculate transmission requirements ===
        self.calculate_pe_transmissions(params)

        # === Phase 6: Calculate clustering and cycles ===
        self.calculate_used_Y_cluster(params)
        self.calculate_used_refreshes(params)
        self.calculate_computing_matrix(params)

        x_size = self.iact_x_add_up + 2 * self.padding_x - self.strideX + 1
        y_size = self.iact_size_y + 2 * self.padding_y - self.strideY + 1
        self.iact_glb_writing_cycles = self.diff_iact_layer * y_size * x_size * self.channel_div_trans + 1
        if not is_depthwise and self.prev_layers and "concat" in self.prev_layers[0]:
            self.concat_limit = 3

        # === Phase 7: Calculate timing parameters ===
        self.psum_delay = int(max((math.ceil(self.used_psum_per_PE) - 2) - (self.used_Y_cluster * params.PEs_Y * 2), 0))
        self.used_X_cluster = params.Clusters_X * self.Y_Cluster_Packages
        self.iact_addr_len = 1
        self.iact_data_len = math.ceil(
            self.used_iact_per_PE / (math.ceil(params.DMA_BITWIDTH / params.Clusters_X) / params.IACT_WOH_Bitwidth)
        )

        # === Phase 8: Calculate GLB transmissions ===
        self.calculate_glb_transmissions(params)

        self.used_wght_addr_per_PE = (
            math.ceil(self.kernel_size[0] * self.channels / self.kernel_per_pe_cluster / self.iact_transmissions_pe) + 2
        )
        if self.used_wght_addr_per_PE == (params.Wghts_Addr_per_PE + 1):
            self.used_wght_addr_per_PE -= 1

        self.psum_storage_cycles = self.diff_iact_layer
        self.buffer_cycles_for_x_iact = 1

        self.needed_iact_buffer_words = (
            self.iact_x_line_repetitions
            * self.needed_Iact_writes
            * math.ceil(self.channel_div_trans * (self.kernel_size[1] + self.iact_size_y - self.strideY))
            * 2
        )
        self.start_param_array = (
            1
            << math.ceil(
                self.different_kernels_per_calculation
                * self.y_lines_per_calculation
                * self.used_Y_cluster
                * math.ceil(self.iact_x_add_up / params.NUM_GLB_PSUM)
            )
        ) - 1

        if self.used_channels == 1:
            self.limit_increase = math.floor((self.iact_size_x * 2) / (2 * 4))
            self.initial_upper_limit = self.limit_increase
            self.overhang_discrepancy = (self.iact_size_x * 2) % (params.MAX_CHANNELS_IACT_GLB * 4)
        else:
            words_per_iact_glb = params.MAX_CHANNELS_IACT_GLB * 4
            if self.iact_x_line_repetitions == 1:
                self.limit_increase = (self.iact_size_x * self.used_channels * self.strideX) // words_per_iact_glb
                self.overhang_discrepancy = (self.iact_size_x * self.used_channels) % words_per_iact_glb
                if self.limit_increase == params.IACT_RAM_CELLS:
                    self.initial_upper_limit = 0
                else:
                    self.initial_upper_limit = self.limit_increase + self.strideX
            else:
                raw_limit = (self.iact_size_x * self.used_channels * self.strideX) / words_per_iact_glb
                self.overhang_discrepancy = math.floor(
                    words_per_iact_glb * ((raw_limit / self.iact_x_line_repetitions) % 1)
                )
                self.limit_increase = math.floor(raw_limit / self.iact_x_line_repetitions)
                self.initial_upper_limit = self.limit_increase + self.strideX + (2 if self.overhang_discrepancy != 0 else 0)

        self.iteration_for_kernels = math.ceil(self.diff_iact_layer_next_layer / self.different_kernels_per_calculation)
        wght_denom = self.used_channels if is_depthwise else self.used_psum_per_PE
        self.needed_wght_cycles = math.ceil(self.filters / (wght_denom * self.different_kernels_per_calculation))

        psum_cycles = 2 if (params.Clusters_X * params.NUM_GLB_PSUM) == 4 else 1
        if next_layers:
            self.fsm_psum_limit = math.ceil(self.iact_x_add_up * self.filters * self.iact_size_y / 4) + 11
        else:
            self.fsm_psum_limit = (
                (
                    self.iact_x_add_up
                    * psum_cycles
                    * self.different_kernels_per_calculation
                    * self.needed_wght_cycles
                    * self.used_psum_per_PE
                    * self.iact_size_y
                )
                // 8
            ) + 12

        if self.channels == 1:
            self.iact_converter_max_cycles = math.ceil(
                self.buffer_cycles_for_x_iact * self.iact_x_line_repetitions * (self.iact_size_y + self.kernel_size[1]) / 2
            )
        else:
            self.iact_converter_max_cycles = (
                self.buffer_cycles_for_x_iact
                * self.iact_x_line_repetitions
                * ((self.iact_size_y + self.kernel_size[1]) - 1)
            )

        full_temp = self.needed_Iact_writes * self.channel_div_trans
        dilation_pes = (self.kernel_shape[0] - self.strideX) - (
            (math.floor(params.Clusters / self.used_Y_cluster) - 1) * params.NUM_GLB_PSUM * self.strideX
        )
        dilation_pes = max(0, dilation_pes) if is_depthwise else math.ceil(max(0, dilation_pes) / params.NUM_GLB_IACT)
        less_temp = full_temp - (dilation_pes * self.channel_div_trans)
        self.iact_repetitions_per_write = (self.iact_x_line_repetitions - 1) * less_temp + full_temp
        self.iact_buffer_words_per_write = (
            self.iact_repetitions_per_write * ((self.iact_size_y + self.kernel_size[1]) - self.strideY) * self.diff_iact_layer
        )
        if is_depthwise:
            self.iact_buffer_words_per_write *= params.NUM_GLB_IACT

        self.iact_x_pos_inc = self.strideX * (params.NUM_GLB_PSUM * (self.Y_Cluster_Packages * params.Clusters_X))
        self.iact_words_per_compute = ((self.needed_Iact_writes * self.kernel_size[1]) * self.used_channels) + 1

        self.lower_bound = (self.padding_y * self.buffer_cycles_for_x_iact * self.iact_x_line_repetitions) - 1
        self.upper_bound = (self.padding_y + self.iact_size_y) * self.buffer_cycles_for_x_iact * self.iact_x_line_repetitions

        self.output_cycles = (
            math.ceil(self.calc_Y / self.strideY) * self.iact_x_line_repetitions / self.different_kernels_per_calculation
        )
        self.psum_output_words = int(
            (self.output_cycles * self.filters * params.Clusters * params.NUM_GLB_PSUM) / math.ceil(params.DMA_BITWIDTH / 32)
        )
        self.psum_x_all_cluster = self.different_kernels_per_calculation * self.iact_x_add_up
        if is_depthwise:
            self.used_wght_per_PE = self.used_iact_per_PE

        # === Phase 9: Finalize calculations ===
        self.calculate_transmission_cycles(params)
        self.calculate_needed_refreshes_mx(params)
        self.calculate_fpga_parameters(params)
        self.get_pagu_values(params)
        self.output_logger()

    def write_conv2d_layer(self, layer_parameters, layer, params):
        """Compute all configuration parameters for a standard Conv2D layer."""
        self._write_conv_common(layer_parameters, layer, params, is_depthwise=False)

    def write_convdw_layer(self, layer_parameters, layer, params):
        """Compute all configuration parameters for a Depthwise Conv2D layer."""
        self._write_conv_common(layer_parameters, layer, params, is_depthwise=True)

    def write_dense_layer(self, layer_parameters, layer, params):
        """Compute all configuration parameters for a Dense (Fully Connected) layer."""
        in_shape = layer.input.shape
        out_shape = layer.output.shape
        self.iact_size_x = in_shape[3] if len(in_shape) > 3 else in_shape[1]
        self.filters = out_shape[3] if len(out_shape) > 3 else out_shape[1]

        num_q = min(self.filters, len(self.quantize))
        for f in range(num_q):
            self.quantize[f][0] = 1
            self.quantize[f][1] = 9

        self.fully_connected = 1
        if getattr(params, "DATAFLOW", "row_stationary") == "output_stationary":
            self.gemm_mode = 1
        self.y_lines_per_calculation = 1
        self.kernel_size = [0]
        self.used_iact_addr_per_PE = 15
        self.channels = 1

        next_layers = self.rel_layers["Next_Layer"][self.layer_name]
        self.prev_layers = self.rel_layers["Previous_Layer"][self.layer_name]
        self.next_layers = next_layers
        self.send_values_out = 1 if len(next_layers) == 0 else 0
        if self.prev_layers != ["input_layer"]:
            self.skipIact = 1

        self.input_shape = in_shape
        self.kernel_shape = layer.kernel.shape
        self.output_shape = out_shape

        self.used_channels = params.NUM_GLB_IACT * math.ceil(
            self.iact_size_x / (params.Clusters_Y * params.NUM_GLB_IACT)
        )
        self.needed_Iact_writes = math.ceil(params.PEs_Y / params.NUM_GLB_IACT)
        self.iact_x_add_up = self.iact_size_x
        self.calculate_transmission_cycles(params)

        temp = math.ceil(self.iact_size_x / (params.Clusters_Y * params.PEs_Y))
        temp = min(temp, 12)
        temp = math.ceil(self.iact_size_x / (params.Clusters_Y * params.PEs_Y * temp))
        self.used_iact_per_PE = math.ceil(self.iact_size_x / (temp * params.Clusters_Y * params.PEs_Y))
        self.used_iact_per_PE = max(2, self.used_iact_per_PE + (self.used_iact_per_PE % 2))

        self.needed_psum_transmissions = math.ceil(self.filters / (16 * params.Clusters_X))
        self.diff_iact_layer = math.ceil(
            self.iact_size_x / (params.NUM_GLB_WGHT * self.used_iact_per_PE * params.Clusters_Y)
        )
        self.needed_wght_transmissions = self.diff_iact_layer * self.needed_psum_transmissions
        self.output_cycles = self.needed_psum_transmissions
        self.used_wght_per_PE = (
            self.used_iact_per_PE
            * math.ceil(self.filters / params.Clusters_X / params.PARALLEL_MACS / self.needed_psum_transmissions)
            * params.PARALLEL_MACS
        )
        self.used_psum_per_PE = math.ceil(self.used_wght_per_PE / self.used_iact_per_PE)
        self.fsm_psum_limit = self.filters + (8 if next_layers else 3)
        self.x_pos_inc = params.Clusters_Y * self.used_iact_per_PE * params.NUM_GLB_WGHT

        self.iact_read_limit_0 = self.needed_Iact_writes * math.ceil(self.used_iact_per_PE / 2)
        self.iact_read_inc_0 = 1
        self.iact_read_limit_1 = self.diff_iact_layer - 1
        self.iact_read_inc_1 = math.ceil(self.needed_Iact_writes * self.used_iact_per_PE / 2)
        self.iact_read_limit_2 = self.diff_iact_layer - 1
        self.iact_read_inc_2 = 0
        self.iact_write_limit_0 = 255
        self.iact_write_inc_0 = 1
        self.iact_write_limit_1 = 255
        self.iact_write_inc_1 = 256

        self.psum_pagu_loop_limit[0] = 255
        self.psum_pagu_loop_limit[1] = 0
        self.psum_pagu_loop_limit[2] = 0
        self.psum_pagu_loop_limit[3] = 0
        self.psum_pagu_loop_limit[4] = 0
        self.psum_pagu_addr_inc[0] = 1
        self.psum_pagu_addr_inc[1] = 0
        self.psum_pagu_addr_inc[2] = 0
        self.psum_pagu_addr_inc[3] = 0
        self.psum_pagu_addr_inc[4] = 0

        self.iact_words_per_compute = math.ceil(params.NUM_GLB_WGHT / params.NUM_GLB_IACT) * self.used_iact_per_PE + 1
        self.used_Y_cluster = params.Clusters_Y
        self.used_X_cluster = 1
        self.kernel_per_pe_cluster = 1
        self.needed_iact_buffer_words = math.ceil(self.used_iact_per_PE / 2) - 1
        self.iact_converter_buffer_addr_max_cycles = self.needed_iact_buffer_words

        # Dense layers activate PE[0] along X in each PE row
        self.computing_mx = [
            [
                [[1 if x_pe == 0 else 0 for x_pe in range(params.PEs_X)] for _ in range(params.PEs_Y)]
                for _ in range(params.Clusters_Y)
            ]
            for _ in range(params.Clusters_X)
        ]

        self.psum_transmissions_pe = 1
        self.iact_transmissions_pe = math.ceil(1 / (self.used_iact_per_PE * params.PEs_Y))
        self.iact_transmissions_glb = 1
        self.wght_transmissions_pe = 1
        self.wght_transmissions_glb = 1
        self.needed_iact_transmissions = self.iact_transmissions_pe * self.iact_transmissions_glb
        self.Used_refreshes = self.iact_transmissions_pe * self.wght_transmissions_pe * self.psum_transmissions_pe
        self.needed_cycles = self.Used_refreshes

        self.iact_data_len = math.ceil(
            self.used_iact_per_PE / (math.ceil(params.DMA_BITWIDTH / 2) / params.IACT_WOH_Bitwidth)
        )
        self.psum_delay = (
            5
            if params.Clusters_Y == 1
            else int(max(((self.used_wght_per_PE / 2 / self.used_iact_per_PE) - 2) - (self.used_Y_cluster * params.PEs_Y * 2), 0))
        )
        self.used_wght_addr_per_PE = min(self.used_iact_per_PE + 2, 16)

        self.needed_total_transmissions = 1
        total_dense_refreshes = self.diff_iact_layer * self.needed_psum_transmissions
        self.needed_refreshes_mx = [[total_dense_refreshes, 0, total_dense_refreshes]]
        self.psum_storage_cycles = self.diff_iact_layer

        self.limit_increase = math.floor((params.NUM_GLB_WGHT * self.used_iact_per_PE) / 8)
        self.initial_upper_limit = self.limit_increase + 9
        self.iteration_for_kernels = math.ceil(self.diff_iact_layer_next_layer / self.different_kernels_per_calculation)
        self.buffer_cycles_for_x_iact = params.Clusters_Y
        self.iact_converter_max_cycles = 1

        self.iact_size_c = (self.used_iact_per_PE * params.PEs_Y * params.Clusters_Y) - self.diff_iact_layer
        self.iact_buffer_words_per_write = (
            self.diff_iact_layer * math.ceil(self.used_iact_per_PE / self.cont_iact_channels_glb) * params.NUM_GLB_WGHT
        )
        self.iact_glb_writing_cycles = self.iact_size_c // 2 + 1

        self.trans_cycles_wght = (
            self.needed_wght_transmissions
            * params.Clusters_Y
            * params.NUM_GLB_WGHT
            * math.ceil(self.used_wght_per_PE / params.PARALLEL_MACS)
            * 3
        )
        self.trans_cycles_psum = self.used_psum_per_PE * params.Clusters_X
        self.lower_bound = (self.padding_y * self.buffer_cycles_for_x_iact * self.iact_x_line_repetitions) - 1
        self.upper_bound = (self.padding_y + self.iact_size_y) * self.buffer_cycles_for_x_iact * self.iact_x_line_repetitions
        self.overhang_discrepancy = (self.used_channels * params.NUM_GLB_WGHT) % (params.MAX_CHANNELS_IACT_GLB * 4)
        self.psum_output_words = int(
            (self.output_cycles * self.used_psum_per_PE) - (params.Clusters_X * self.needed_wght_cycles)
        )
        self.iact_x_pos_inc = math.ceil(params.NUM_GLB_WGHT * self.used_iact_per_PE / 2) * params.Clusters_Y

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Needed transmissions WGHT    : %s", self.needed_wght_transmissions)
            logger.debug("Needed transmissions PSUM    : %s", self.needed_psum_transmissions)
            logger.debug("Needed transmissions TOTAL   : %s", self.needed_total_transmissions)
            logger.debug("Needed transmissions IACT PE : %s", self.iact_transmissions_pe)
            logger.debug("Needed transmissions WGHT PE : %s", self.wght_transmissions_pe)
            logger.debug("Needed transmissions PSUM PE : %s", self.psum_transmissions_pe)
            logger.debug("Needed transmissions IACT    : %s", self.needed_iact_transmissions)

    def write_gemm_layer(self, layer_parameters, layer, params):
        """Compute configuration parameters for an output-stationary GEMM layer."""
        self.write_dense_layer(layer_parameters, layer, params)
        self.gemm_mode = 1

    def write_pooling_layer(self, layer_parameters, layer, params):
        """Compute all configuration parameters for a Pooling layer."""
        self.pooling_mode = 0 if "max" in layer.name else 1
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

        self.computing_mx = [
            [[[1] * params.PEs_X for _ in range(params.PEs_Y)] for _ in range(params.Clusters_Y)]
            for _ in range(params.Clusters_X)
        ]
        self.diff_iact_layer = self.input_shape[3]

        self.prev_layers = self.rel_layers["Previous_Layer"][self.layer_name]
        next_layers = self.rel_layers["Next_Layer"][self.layer_name]
        self.next_layers = next_layers

        if next_layers:
            next_layer_params = layer_parameters[next_layers[0]]
            if ("dense" in next_layer_params.layer_name) and (self.pooling_mode == 0):
                self.used_channels = 1
            else:
                self.used_channels = 4
            self.diff_iact_layer_next_layer = next_layer_params.used_channels
        else:
            self.used_channels = 4

        if self.pooling_mode == 0:
            self.iact_converter_buffer_addr_max_cycles = math.ceil(
                (self.iact_size_y * self.diff_iact_layer) / (self.strideY * 2)
            )
        else:
            self.iact_converter_buffer_addr_max_cycles = math.ceil(self.diff_iact_layer / 2)

    def print_layer_parameters(self, debug_file=None):
        """Print summary of computed layer parameters for debugging/analysis."""
        logger.info("params.PEs X: %s", self.used_PEs_X)
        logger.info("params.PEs Y: %s", self.used_PEs_Y)
        logger.info("Iact PE: %s", self.used_iact_per_PE)
        logger.info("Wght PE: %s", self.used_wght_per_PE)
        logger.info("Psum PE: %s", self.used_psum_per_PE)
        logger.info("Needed Cycles: %s", self.Used_refreshes)
        logger.info("Factor: %s", self.current_highest_number)
        logger.info("Real Factor: %s", self.realfactor)
        logger.info("End of Layer")