# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import math
import logging
logger = logging.getLogger("cocotb")

class LayerParameters(object):
    """
    Parameter class for a single layer.
    
    It contains the parameters that are computed during the generation of the output files
    for subsequent usage and insepction.

    Args:
        layer: A Keras layer.
        params: The parameters of the OpenEye.
        filename: The filename of the output file.
    """
    def __init__(self, layer_parameters, layer, params, layer_number, max_layers):
        self.layer_name = ""

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

        self.used_iact_per_PE = []
        self.used_wght_per_PE = []
        self.used_psum_per_PE = []
        self.diff_iact_layer = []
        self.diff_iact_layer_next_layer = 0
        self.ceil_used_PE_per_clm = 0

        self.needed_Iact_writes = 0

        self.current_highest_number = 0
        self.realfactor = 0
        self.used_iact_addr_per_PE = 2
        self.used_wght_addr_per_PE = 5

        self.iact_addr_len = 1
        self.iact_data_len = 3
        self.choose_iact_storage_input = 0
        self.choose_iact_storage_output = 0
        
        self.strideX = 1
        self.strideY = 1
        self.add_up = 1
        self.complete_iacts_in_design = 0
        self.max_pooling = 0
        self.output_cycles = 0

        self.filters = 1
        self.input_shape = []
        self.kernel_shape = []
        self.output_shape = []
        self.kernel_size = []
        self.kernel_per_pe_cluster = 1
        self.used_channels = 1
        self.channel_repetition = 4
        self.single_cluster_computation = 0
        self.iact_size_x = 0
        self.iact_size_y = 0
        self.iact_transmissions_pe = 1
        self.wght_transmissions_pe = 1
        self.psum_transmissions_pe = 1
        self.iact_transmissions_glb = 1
        self.wght_transmissions_glb = 1
        self.psum_transmissions_glb = 1
        self.used_iact_per_PE = 1
        self.used_wght_per_PE = 1
        self.used_psum_per_PE = 1
        self.needed_wght_transmissions = 1
        self.needed_total_transmissions = 1
        self.psum_delay = 0
        self.fully_connected = 0
        self.store_in_psum = 0

        self.send_values_out = 1
        self.skipIact = 0
        self.skipWght = 0
        self.skipPsum = 0
        self.iact_stream_cycles = 1

        self.computing_mx = 0
        self.data_mode = 0

        #FPGA parameters
        self.needed_standing_cycles = 0
        self.direct_cycling = 0
        self.iact_x_lines = 3
        self.quantize = [[0 for _ in range(2)]for _ in range(256)]

        match layer.name:
            case "depthwise_conv2d":
                logger.debug("Depthwise Convolution Layer")
                self.write_convdw_layer(layer, params)

            case "conv2d":
                logger.debug("2D Convolution Layer")
                self.write_conv2d_layer(layer_parameters, layer, params, layer_number, max_layers)
                
            case "dense":
                logger.debug("Dense Layer")
                self.write_dense_layer(layer_parameters, layer, params, layer_number, max_layers)

            case "max_pooling2d":
                logger.debug("Pooling Layer")
                self.write_pooling_layer(layer_parameters, layer, params, layer_number, max_layers)
            
            case default:
                logger.debug("Layer type for " + str(layer) + " not supported.")
                print(str(layer))
                raise ValueError("Layer type not supported.")

    def compute_total_computations(self):
        """ Compute the total number of computations for the layer.

        TODO: find a better name
        """
        if(self.padding == "same"):
            self.calc_X = self.input_shape[1]
            self.calc_Y = self.input_shape[2]
        else:
            self.calc_X = self.input_shape[1] - self.kernel_size[0] + 1
            self.calc_Y = self.input_shape[2] - self.kernel_size[1] + 1
        self.total_computations = self.calc_X * self.calc_Y

    def calculate_iact_transmissions(self, params):
        #Calculate Iact Cycles
        self.needed_Iact_writes = math.ceil(((params.PEs_X - 1) * self.strideX * self.kernel_per_pe_cluster + (self.kernel_size[1]*self.kernel_per_pe_cluster)) / (params.NUM_GLB_IACT))

    def calculate_computing_matrix(self, params):
        """ TODO: Docu - explain why this function exists"""

        if((self.kernel_size[0] <= params.Iacts_per_PE) & (self.kernel_size[1] <= params.PEs_Y * params.Clusters_Y)):
            self.computing_mx = [[[[1 for _ in range(params.PEs_X)]
                                            for _ in range(params.PEs_Y)]
                                            for _ in range(params.Clusters_Y)]
                                            for _ in range(params.Clusters_X)]
            for x_cluster in range(params.Clusters_X):
                for y_cluster in range(params.Clusters_Y):
                    for y_pe in range(params.PEs_Y):
                        for x_pe in range(params.PEs_X):
                            x_pos_in_pes = x_cluster * 4 + y_cluster * 8 + x_pe
                            x_values_per_cycle = self.calc_X * self.calc_Y
                            if(x_pos_in_pes >= x_values_per_cycle):
                                self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0
            if((self.kernel_size[0]*self.kernel_per_pe_cluster) < params.PEs_Y):
                for x_cluster in range(params.Clusters_X):
                    for y_cluster in range(params.Clusters_Y):
                        for y_pe in range(params.PEs_Y):
                            for x_pe in range(params.PEs_X):
                                if((1 + y_pe) > (self.kernel_size[0]*self.kernel_per_pe_cluster)):
                                    self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0
            else:
                for x_cluster in range(params.Clusters_X):
                    for y_cluster in range(params.Clusters_Y):
                        for y_pe in range(params.PEs_Y):
                            for x_pe in range(params.PEs_X):
                                if((1 + y_pe + (y_cluster % self.used_Y_cluster) * params.PEs_Y) > (self.kernel_size[0]*self.kernel_per_pe_cluster)):
                                    self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0

            if((self.output_shape[1] % params.PEs_X) != 0):
                    self.add_up = params.PEs_X - (self.output_shape[1] % params.PEs_X)
                    yc_step = math.ceil(self.output_shape[1]/(params.PEs_X*params.Clusters_X))
                    yc_start = yc_step - 1
                    yc_end = params.Clusters_Y
                    for y_cluster in range(yc_start,yc_end,yc_step):
                        for x_cluster in range(math.floor((self.output_shape[1]%(params.Clusters_X*params.PEs_X)) / params.PEs_X),params.Clusters_X):
                            for x_pe in range(self.output_shape[1] % params.PEs_X,params.PEs_X):
                                for y_pe in range(params.PEs_Y):
                                    self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0
            else:
                self.add_up = 0
        else:
            logger.error("Can't fit model, kernel size must be adjusted.")
            raise ValueError("Can't fit model, kernel size must be adjusted.")
    def write_conv2d_layer(self, layer_parameters, layer, params, layer_number, max_layers):
        self.layer_name = "Convolution2D"
        self.choose_iact_storage_input = 1
        self.filters = layer.filters
        self.fully_connected = 0
        if (layer_number == max_layers - 1) :
            self.send_values_out = 1
        else:
            self.send_values_out = 0
        if hasattr(layer, 'quantization_factor'):
            for f in range(self.filters):
                self.quantize[f][0] = layer.quantization_factor[f][0]
                self.quantize[f][1] = layer.quantization_factor[f][1]
        else:
            for f in range(self.filters):
                self.quantize[f][0] = 1
                self.quantize[f][1] = 9
        if (layer_number != 0) : 
            self.skipIact = 1
            self.choose_iact_storage_input = 0
        if hasattr(layer, 'store_in_psum'):
            self.store_in_psum = layer.store_in_psum
        if hasattr(layer, 'skip_psum'):
            self.skipPsum = layer.skip_psum
            self.choose_iact_storage = 1
        self.input_shape = layer.input.shape
        self.kernel_shape = layer.kernel.shape
        self.output_shape = layer.output.shape
        self.kernel_size = layer.kernel_size
        self.strideX = layer.strides[0]
        self.strideY = layer.strides[1]
        self.compute_total_computations()
        self.output_cycles = self.calc_Y
        if (params.SERIAL == 0) :
            if (self.output_shape[1] <= 8):
                self.single_cluster_computation = 2
            if (self.output_shape[1] <= 4):
                self.single_cluster_computation = 1

        if (math.floor(params.PEs_Y/self.kernel_size[0]) > 1):
            self.kernel_per_pe_cluster = math.floor(params.PEs_Y/self.kernel_size[0])
        self.calculate_iact_transmissions(params)
        self.channels = self.input_shape[3]
        match self.single_cluster_computation:
            case 1:
                self.complete_iacts_in_design = math.floor((self.input_shape[1]*self.input_shape[2])/ \
                    (params.PEs_X))
            case 2:
                self.complete_iacts_in_design = math.floor((self.input_shape[1]*self.input_shape[2])/ \
                    (params.PEs_X*params.Clusters_X))
            case _:
                self.complete_iacts_in_design = math.floor((self.input_shape[1]*self.input_shape[2])/ \
                    (params.Clusters * params.PEs_X))
        # Calculate the number of refreshes needed for the layer
        if((self.input_shape[3]*self.kernel_size[0])<params.Iacts_per_PE):
            self.used_channels = math.floor(self.input_shape[3])
        else:
            self.used_channels = 8
        if(2*self.kernel_size[0] <= params.PEs_Y):
            temp = math.ceil(self.input_shape[3]/self.kernel_per_pe_cluster)
            if (self.kernel_size[0] >= 2):
                divisor = math.ceil(temp / params.Iacts_per_PE)
            else:
                divisor = math.ceil(temp / params.Iacts_Addr_per_PE)
            self.used_channels = math.ceil(temp/divisor)
            self.used_channels = 6
        elif(self.kernel_size[0] == 3):
            if (self.used_channels >= 4):
                self.used_channels = 4
                if(self.used_channels >= self.input_shape[3]):
                    self.used_channels = self.input_shape[3]

        elif(self.kernel_size[0] == 5):
            if(self.used_channels >= 2):
                self.used_channels = 2

        elif(self.kernel_size[0] >= 8):
            if(self.used_channels >= 2):
                self.used_channels = 1
        else: 
            assert False
        self.iact_stream_cycles = math.ceil(self.input_shape[1] * self.input_shape[2] * self.input_shape[3] / params.NUM_BUFFER / (params.DMA_Bit_AXI//params.IACT_Bitwidth))
        self.diff_iact_layer = math.ceil(self.input_shape[3]/self.used_channels)
        if (layer_number != max_layers - 1) :
            self.diff_iact_layer_next_layer = layer_parameters[max_layers - layer_number - 2].used_channels
        self.used_iact_per_PE = self.kernel_size[0] * self.used_channels

        match self.single_cluster_computation:
            case 1:
                self.iact_transmissions_pe = math.ceil(self.input_shape[3]/(self.used_channels*params.PEs_Y))
            case 2:
                self.iact_transmissions_pe = math.ceil(self.input_shape[3]/(self.used_channels*params.PEs_Y))
            case _:
                self.iact_transmissions_pe = math.ceil(self.diff_iact_layer /self.kernel_per_pe_cluster)
        logger.debug("used_iact_per_PE " + str(self.used_iact_per_PE))
        logger.debug("iact_transmissions_pe " + str(self.iact_transmissions_pe))
            
        if((self.filters * self.used_iact_per_PE) <= params.Wghts_per_PE):
            if (self.filters <= 16) :
                self.used_psum_per_PE = self.filters
                self.wght_transmissions_pe = math.ceil(self.channels/self.used_channels)
            else :
                self.used_psum_per_PE = 16
                self.wght_transmissions_pe = 2 * math.ceil(self.channels/self.used_channels)
            self.used_wght_per_PE = self.used_psum_per_PE*self.used_iact_per_PE

        else:
            if (self.kernel_size[0] == 5):
                wght_factor = math.ceil((self.filters*self.used_iact_per_PE)/160)
            else:
                wght_factor = math.ceil((self.filters*self.used_iact_per_PE)/params.Wghts_per_PE)
            match self.single_cluster_computation:
                case 1:
                    self.used_wght_per_PE = math.ceil(self.filters/params.Clusters)*self.used_iact_per_PE
                    self.used_psum_per_PE = int(self.used_wght_per_PE/self.used_iact_per_PE)
                case 2:
                    self.used_wght_per_PE = math.ceil(self.filters/params.Clusters_Y)*self.used_iact_per_PE
                    self.used_psum_per_PE = int(self.used_wght_per_PE/self.used_iact_per_PE)
                case _:
                    self.used_wght_per_PE = math.ceil(self.filters/wght_factor)*self.used_iact_per_PE
                    self.used_psum_per_PE = int(self.filters/wght_factor)

            match self.single_cluster_computation:
                case 1:
                    self.wght_transmissions_pe = math.ceil(self.filters/(32*params.PEs_X*params.Clusters))
                case 2:
                    self.wght_transmissions_pe = math.ceil(self.filters/(32*params.PEs_X*params.Clusters_Y))
                case _:
                    self.wght_transmissions_pe = math.ceil(self.channels/self.used_channels) * math.ceil(self.filters * self.used_iact_per_PE / self.used_wght_per_PE)

        if(math.ceil(self.used_iact_per_PE/self.used_wght_per_PE) <= params.Psums_per_PE):
            self.psum_transmissions_pe = 1
        else:    
            self.psum_transmissions_pe = math.ceil(self.filters / params.Psums_per_PE)
            logger.debug("Error Code 5, Overused PSUM per PE, not implemented flow yet")
        #Calculation of seperate PE-Cluster
        self.used_PEs_Y    = self.kernel_size[1]*self.kernel_per_pe_cluster
        used_PEs_per_clm     = self.used_PEs_Y/params.PEs_Y
        self.ceil_used_PE_per_clm = math.ceil(used_PEs_per_clm)

        self.used_Y_cluster = (math.ceil(self.used_PEs_Y/params.PEs_Y))
        self.used_Y_cluster = (math.floor(params.Clusters_Y/self.used_Y_cluster))
        self.used_Y_cluster = (math.ceil(params.Clusters_Y/self.used_Y_cluster))
        self.calculate_computing_matrix(params)
        if (params.SERIAL) :
            self.psum_delay = int(max([(math.ceil(self.used_psum_per_PE) - 2) - (self.used_Y_cluster * params.PEs_Y * 2),0]))
        else :
            self.psum_delay = int(max([(math.ceil(self.needed_refreshes_mx[layer_repetition][0]/2) - 2) - (self.used_Y_cluster * params.PEs_Y * 2),0]))
        if((self.output_shape[2] % params.PEs_X)== 0):
            self.used_X_cluster = 1

        self.iact_addr_len = math.ceil((self.used_channels)/(math.ceil(params.DMA_Bit_AXI/params.Clusters_X)/params.IACT_Addr_Bitwidth))
        self.iact_addr_len = 1
        self.iact_data_len = math.ceil(self.used_iact_per_PE/(math.ceil(params.DMA_Bit_AXI/params.Clusters_X)/params.IACT_WOH_Bitwidth))

        self.psum_transmissions_glb = math.ceil(((math.ceil(self.output_shape[1]/params.NUM_GLB_PSUM) * \
                                    self.output_shape[2] * math.ceil(self.output_shape[3]/ self.wght_transmissions_pe)) / \
                                    params.Clusters_X / 2 / params.Clusters_Y) \
                                    / self.psum_transmissions_pe / self.wght_transmissions_pe / params.Psum_Mem_Addr_Words)
        self.needed_psum_transmissions = self.psum_transmissions_pe * self.psum_transmissions_glb
        self.wght_transmissions_glb = 1
        self.needed_wght_transmissions = self.wght_transmissions_pe * self.wght_transmissions_glb
        
        all_transmissions_of_pe = self.iact_transmissions_pe * self.wght_transmissions_pe * self.psum_transmissions_pe
        match self.single_cluster_computation:
            case 1:
                self.Used_refreshes = math.ceil(all_transmissions_of_pe * math.ceil(self.output_shape[1] * self.output_shape[2]/(params.PEs_X)))
            case 2:
                self.Used_refreshes = math.ceil(all_transmissions_of_pe * math.ceil(self.output_shape[1] * self.output_shape[2]/(params.PEs_X*params.Clusters_X)))
            case _:
                self.Used_refreshes = math.ceil(self.used_Y_cluster* all_transmissions_of_pe * math.ceil(self.output_shape[1] * self.output_shape[2]/(params.PEs_X*params.Clusters)))
        if (self.kernel_size[0] == 1):
            self.used_iact_addr_per_PE = 1
        else:
            self.used_iact_addr_per_PE = self.used_channels
        logger.debug("Refreshes: " + str(self.Used_refreshes))
        logger.debug("Used complete new descriptions: " + str(self.Used_refreshes))
        logger.debug("self.used_channels : " + str(self.used_channels))
        logger.debug("layer_params.needed_Iact_writes : " + str(self.needed_Iact_writes))
        logger.debug("Used_refreshes : " + str(self.Used_refreshes))
        logger.debug("layer_params.needed_psum_transmissions : " + str(self.needed_psum_transmissions))
        if (params.SERIAL == 1) :
            self.iact_transmissions_glb = 1
        else :
            self.iact_transmissions_glb = \
            math.ceil(math.ceil(self.Used_refreshes/self.wght_transmissions_pe/self.needed_psum_transmissions/self.iact_transmissions_pe)/math.floor(params.Iact_Mem_Addr_Words/\
            ((math.ceil((self.used_channels*self.kernel_size[0])/2) + (math.ceil((self.used_channels + 1)/6)))* self.needed_Iact_writes)))
        self.needed_iact_transmissions = self.iact_transmissions_pe * self.iact_transmissions_glb
        match self.single_cluster_computation:
            case 1:
                self.used_wght_addr_per_PE = (math.ceil(self.kernel_size[0] * self.input_shape[3]/self.kernel_per_pe_cluster / self.iact_transmissions_pe/ self.wght_transmissions_pe)) + 2
            case 2:
                self.used_wght_addr_per_PE = (math.ceil(self.kernel_size[0] * self.input_shape[3]/self.kernel_per_pe_cluster / self.iact_transmissions_pe/ self.wght_transmissions_pe)) + 2
            case _:
                self.used_wght_addr_per_PE = (math.ceil(self.kernel_size[0] * self.input_shape[3]/self.kernel_per_pe_cluster / self.iact_transmissions_pe)) + 2

        if(self.used_wght_addr_per_PE == (params.Wghts_Addr_per_PE + 1)):
            self.used_wght_addr_per_PE = self.used_wght_addr_per_PE - 1

        if (params.SERIAL == 1) :
            self.needed_total_transmissions = 1
            self.needed_refreshes_mx = [[1 for _ in range(3)] for _ in range(1)]
            self.needed_refreshes_mx[0][2] = self.Used_refreshes
            self.needed_refreshes_mx[0][1] = 0
            self.needed_refreshes_mx[0][0] = self.Used_refreshes
        else :
            self.needed_total_transmissions = self.needed_psum_transmissions * \
                                                self.needed_wght_transmissions * \
                                                self.needed_iact_transmissions
            self.needed_refreshes_mx = [[1 for _ in range(3)] for _ in range(self.needed_total_transmissions)]

            for layer_repetition in range(self.needed_total_transmissions):
                self.needed_refreshes_mx[layer_repetition][2] = math.floor(((math.floor(math.floor(layer_repetition/self.iact_transmissions_pe)/self.needed_wght_transmissions)+1)/ \
                    self.needed_total_transmissions) * self.Used_refreshes)
                self.needed_refreshes_mx[layer_repetition][2] = self.needed_refreshes_mx[layer_repetition][2] - (self.needed_refreshes_mx[layer_repetition][2]%self.used_Y_cluster)
                self.needed_refreshes_mx[layer_repetition][1] = math.floor((math.floor(math.floor(layer_repetition/self.iact_transmissions_pe)/self.needed_wght_transmissions)/ \
                    self.needed_total_transmissions) * self.Used_refreshes)
                self.needed_refreshes_mx[layer_repetition][1] = self.needed_refreshes_mx[layer_repetition][1] - (self.needed_refreshes_mx[layer_repetition][1]%self.used_Y_cluster)
                self.needed_refreshes_mx[layer_repetition][0] = self.needed_refreshes_mx[layer_repetition][2] - self.needed_refreshes_mx[layer_repetition][1]

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

        self.iact_size_x = self.input_shape[1]
        self.iact_size_y = self.input_shape[2]
        #FPGA parameters
        self.needed_standing_cycles = math.ceil(params.Clusters/math.floor((params.Clusters*params.PEs_X)/self.iact_size_x))
        if (self.iact_size_x == (params.Clusters*params.PEs_X)) :
            self.direct_cycling = 1
            self.iact_x_lines = self.kernel_size[1] + self.iact_size_y - 1
            self.needed_standing_cycles = ((self.used_channels + 1) // 2) * self.needed_Iact_writes
        else :
            self.needed_standing_cycles = max(self.needed_standing_cycles,4)          #Change later for interchangebility
        self.direct_cycling = 1
        self.iact_x_lines = self.kernel_size[1] + self.iact_size_y - 1
        self.needed_standing_cycles = ((self.used_channels + 1) // 2) * self.needed_Iact_writes
 
    def write_convdw_layer(self, layer, params):
        """ Write the weights and bias of a Conv2D layer to a file. """
        
        # Get the weights and bias of the layer
        self.layer_name = "DepthwiseConvolution"
                        
            
        self.strideX = layer.strides[0]
        self.strideY = layer.strides[1]
        self.data_mode = 1
        self.filters = layer.kernel_size[0]
        self.input_shape = layer.input.shape
        self.kernel_shape = layer.kernel.shape
        self.output_shape = layer.output.shape
        self.kernel_size = layer.kernel_size
        self.compute_total_computations()

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

            if((self.output_shape[1] % params.PEs_X) != 0):
                if((self.output_shape[1] < 8) | ((self.output_shape[1] > 12) & (self.output_shape[1] < 16))):
                    self.add_up = params.PEs_X - (self.output_shape[1] % params.PEs_X)
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
            self.ceil_used_PE_per_clm = math.ceil(used_PEs_per_clm)

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
            if (params.SERIAL == 1) :
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
        """ Write the weights and bias of a Conv2D layer to a file. """
            
        self.layer_name = "Dense"
        self.iact_size_x = 1
        self.iact_size_y = layer.input.shape[2]
        self.filters = layer.output.shape[3]
        self.fully_connected = 1
        if (layer_number == max_layers - 1) :
            self.send_values_out = 1
        else:
            self.send_values_out = 0
        for f in range(self.filters):
            self.quantize[f][0] = 1
            self.quantize[f][1] = 9
        if (layer_number != 0) : 
            self.skipIact = 1
        self.input_shape = layer.input.shape
        self.kernel_shape = layer.kernel.shape
        self.output_shape = layer.output.shape
            
        self.used_channels = math.ceil(self.input_shape[3]/4)
        #Calculate Iact Cycles
        self.needed_Iact_writes = math.ceil(params.PEs_Y/params.NUM_GLB_IACT)

        # Calculate the number of refreshes needed for the layer
        
        self.used_iact_per_PE = math.ceil(self.input_shape[3]/4)
        self.used_wght_per_PE = math.ceil(self.input_shape[3]*self.output_shape[3]/(4*params.Clusters_X))
        self.used_psum_per_PE = math.ceil(self.output_shape[3]/params.Clusters_X)

        self.used_Y_cluster = params.Clusters_Y
        self.used_X_cluster = 1
        self.kernel_per_pe_cluster = 1

        self.computing_mx = [[[[1 for _ in range(params.PEs_X)]
                                        for _ in range(params.PEs_Y)]
                                        for _ in range(params.Clusters_Y)]
                                        for _ in range(params.Clusters_X)]
        
        for x_cluster in range(params.Clusters_X):
            for y_cluster in range(params.Clusters_Y):
                for y_pe in range(params.PEs_Y):
                    for x_pe in range(params.PEs_X):
                        if((x_pe != 0) | (y_pe != 0) | (y_cluster >= 4)):
                            self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0
        
        self.psum_transmissions_pe = math.ceil(self.output_shape[1]/(self.used_psum_per_PE * params.Clusters_X * params.Clusters_Y))
        self.psum_transmissions_glb = 1

        self.iact_transmissions_pe = math.ceil(self.input_shape[1]/(self.used_iact_per_PE * params.PEs_Y))
        self.iact_transmissions_glb = 1

        self.needed_psum_transmissions = self.psum_transmissions_pe * self.psum_transmissions_glb
        self.wght_transmissions_pe = 1
        self.wght_transmissions_glb = 1
        self.needed_wght_transmissions = self.wght_transmissions_pe * self.wght_transmissions_glb
        self.needed_iact_transmissions = self.iact_transmissions_pe * self.iact_transmissions_glb
        self.Used_refreshes = self.iact_transmissions_pe * self.wght_transmissions_pe * self.psum_transmissions_pe
        
        self.used_iact_per_PE = math.ceil(self.input_shape[3]/4)
        self.iact_data_len = math.ceil(self.used_iact_per_PE/(math.ceil(params.DMA_Bit_AXI/2)/params.IACT_WOH_Bitwidth))
        logger.debug("Refreshes: " + str(self.Used_refreshes))
        logger.debug("Used complete new descriptions: " + str(self.Used_refreshes))
        logger.debug("self.needed_Iact_writes : " + str(self.needed_Iact_writes))
        logger.debug("Used_refreshes : " + str(self.Used_refreshes))
        self.psum_delay = int(max([((self.used_wght_per_PE/2/self.used_iact_per_PE) - 2) - (self.used_Y_cluster * params.PEs_Y * 2),0]))
        self.used_wght_addr_per_PE = (self.used_iact_per_PE) + 2
        if (self.used_wght_addr_per_PE >= 16):
            self.used_wght_addr_per_PE = 16

        self.needed_total_transmissions = self.needed_psum_transmissions * \
                                                    self.needed_wght_transmissions * \
                                                    self.needed_iact_transmissions
        self.needed_refreshes_mx = [[1 for _ in range(3)]
                                    for _ in range(self.needed_total_transmissions)]
        for layer_repetition in range(self.needed_total_transmissions):
            self.needed_refreshes_mx[layer_repetition][2] = math.floor(((math.floor(math.floor(layer_repetition/self.iact_transmissions_pe))+1)/ \
                self.needed_total_transmissions) * self.Used_refreshes)
            self.needed_refreshes_mx[layer_repetition][1] = math.floor((math.floor(math.floor(layer_repetition/self.iact_transmissions_pe))/ \
                self.needed_total_transmissions) * self.Used_refreshes)
            self.needed_refreshes_mx[layer_repetition][0] = self.needed_refreshes_mx[layer_repetition][2] - self.needed_refreshes_mx[layer_repetition][1]
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
        self.layer_name = "Pooling"
        self.input_shape = layer.input.shape
        self.output_shape = layer.output.shape
        self.skipIact = 1
        self.skipWght = 1
        self.skipPsum = 1
        self.channels = self.input.shape[3]
        self.max_pooling = 1
        self.send_values_out = 1
        self.iact_size_x = self.input.shape[1]
        self.iact_size_y = self.input.shape[2]
        self.computing_mx = [[[[1 for _ in range(params.PEs_X)]
                                        for _ in range(params.PEs_Y)]
                                        for _ in range(params.Clusters_Y)]
                                        for _ in range(params.Clusters_X)]
        self.used_channels = 1
        self.diff_iact_layer = self.input.shape[3]
        return

    def print_layer_parameters(self, debug_file):
        logger.info("params.PEs X: " + str(self.used_PEs_X) + "\n")
        logger.info("params.PEs Y: " + str(self.used_PEs_Y) + "\n")
        logger.info("Iact PE: " + str(self.used_iact_per_PE) + "\n")
        logger.info("Wght PE: " + str(self.used_wght_per_PE) + "\n")
        logger.info("Psum PE: " + str(self.used_psum_per_PE) + "\n")
        logger.info("Needed Cycles: " + str(self.Used_refreshes) + "\n")
        logger.info("Factor: " + str(self.current_highest_number) + "\n")
        logger.info("Real Factor: " + str(self.realfactor) + "\n")
        logger.info("End of Layer")
