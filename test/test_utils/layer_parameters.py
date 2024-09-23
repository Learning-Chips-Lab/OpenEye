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
    def __init__(self, layer, params):
        self.used_PEs_X = 1
        self.used_PEs_Y = 0
        self.used_X_cluster = 1
        self.used_Y_cluster = 1
        self.Used_refreshes = 0
        self.current_input_X = 0
        self.current_input_Y = 0
        self.needed_refreshes_mx = []

        self.used_iact_per_PE = []
        self.used_wght_per_PE = []
        self.used_psum_per_PE = []
        self.diff_iact_layer = []
        self.ceil_used_PE_per_clm = 0

        self.needed_Iact_writes = 0

        self.current_highest_number = 0
        self.realfactor = 0
        self.used_iact_addr_per_PE = 2
        self.used_wght_addr_per_PE = 5

        self.iact_addr_len = 1
        self.iact_data_len = 3
        
        self.strideX = 1
        self.strideY = 1
        self.add_up = 1

        self.filters = 1
        self.input_shape = []
        self.output_shape = []
        self.kernel_size = []

        self.iact_transmissions_pe = 1
        self.wght_transmissions_pe = 1
        self.psum_transmissions_pe = 1
        self.iact_transmissions_glb = 1
        self.wght_transmissions_glb = 1
        self.psum_transmissions_glb = 1
        self.used_iact_per_PE = 1
        self.used_wght_per_PE = 1
        self.used_psum_per_PE = 1
        self.needed_total_transmissions = 1

        self.skipIact = 0
        self.skipWght = 0
        self.skipPsum = 0

        self.computing_mx = 0

        params = params


        if "Depthwise" in str(layer):
            logger.debug("Depthwise Convolution Layer")
            self.write_convdw_layer(layer, params)

        elif "Conv2D" in str(layer):
            logger.debug("2D Convolution Layer")
            self.write_conv2d_layer(layer, params)
                
        elif "Dense" in str(layer):
            logger.debug("Dense Layer")
            self.write_dense_layer(layer, params)
            
        else:
            logger.debug("Layer type for " + str(layer) + " not supported.")
            raise ValueError("Layer type not supported.")
        
    def get_realfactor(self, layer):
        """ TODO: Docu - explain why this function exists
        
        """
        # Get the weights and bias of the layer
        max_weight = 0
        for current_number_x in range(layer.kernel_size[0]):
            for current_number_y in range(layer.kernel_size[1]):
                for current_number_z in range(len(layer.weights[0][current_number_x][current_number_y])):
                    for current_number_c in range(len(layer.weights[0][current_number_x][current_number_y][current_number_z])):
                        if(abs(layer.weights[0][current_number_x][current_number_y][current_number_z][current_number_c]) >= self.current_highest_number):
                            max_weight = abs(layer.weights[0][current_number_x][current_number_y][current_number_z][current_number_c])
        realfactor = math.floor(abs(math.log2(abs(max_weight))))
        return realfactor

    def compute_total_computations(self, layer):
        """ Compute the total number of computations for the layer.

        TODO: find a better name
        """
        if(layer.padding == "same"):
            calc_X = layer.input.shape[1]
            calc_Y = layer.input.shape[2]
        else:
            calc_X = layer.input.shape[1] - layer.kernel_size[0] + 1
            calc_Y = layer.input.shape[2] - layer.kernel_size[1] + 1
        self.total_computations = calc_X * calc_Y

    def calculate_iact_transmissions(self, layer, params):
        #Calculate Iact Cycles
        self.needed_Iact_writes = math.ceil(((params.PEs_X - 1) * self.strideX + layer.kernel_size[1]) /(params.Iact_Routers))
       
    def calculate_computing_matrix(self, layer, params):
        """ TODO: Docu - explain why this function exists"""

        if((layer.kernel_size[0] <= params.Iacts_per_PE) & (layer.kernel_size[1] <= params.PEs_Y * params.Clusters_Y)):
            self.computing_mx = [[[[1 for _ in range(params.PEs_X)]
                                            for _ in range(params.PEs_Y)]
                                            for _ in range(params.Clusters_Y)]
                                            for _ in range(params.Clusters_X)]
            if(layer.kernel_size[0] < 3):
                for x_cluster in range(params.Clusters_X):
                    for y_cluster in range(params.Clusters_Y):
                        for y_pe in range(params.PEs_Y):
                            for x_pe in range(params.PEs_X):
                                if((1 + y_pe) > layer.kernel_size[1]):
                                    self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0
            else:
                for x_cluster in range(params.Clusters_X):
                    for y_cluster in range(params.Clusters_Y):
                        for y_pe in range(params.PEs_Y):
                            for x_pe in range(params.PEs_X):
                                if((1 + y_pe + (y_cluster % math.ceil(layer.kernel_size[0]/params.PEs_Y)) * params.PEs_Y) > layer.kernel_size[1]):
                                    self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0

            if((layer.output.shape[1] % params.PEs_X) != 0):
                if((layer.output.shape[1] < 8) | ((layer.output.shape[1] > 12) & (layer.output.shape[1] < 16))):
                    self.add_up = params.PEs_X - (layer.output.shape[1] % params.PEs_X)
                    yc_step = math.ceil(layer.output.shape[1]/(params.PEs_X*params.Clusters_X))
                    yc_start = yc_step - 1
                    yc_end = params.Clusters_Y
                    for y_cluster in range(yc_start,yc_end,yc_step):
                        for x_cluster in range(math.floor((layer.output.shape[1]%(params.Clusters_X*params.PEs_X)) / params.PEs_X),params.Clusters_X):
                            for x_pe in range(layer.output.shape[1] % params.PEs_X,params.PEs_X):
                                for y_pe in range(params.PEs_Y):
                                    self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0
                else:
                    assert False, "Kernel cant be caclulated"
            else:
                self.add_up = 0
        else:
            logger.error("Can't fit model, kernel size must be adjusted.")
            raise ValueError("Can't fit model, kernel size must be adjusted.")

    def write_conv2d_layer(self, layer, params):
        """ """
        
        realfactor = self.get_realfactor(layer)

        self.compute_total_computations(layer)
            
        self.strideX = layer.strides[0]
        self.strideY = layer.strides[1]
        self.filters = layer.filters
        self.input_shape = layer.input.shape
        self.output_shape = layer.output.shape
        self.kernel_size = layer.kernel_size

        self.calculate_iact_transmissions(layer,params)

        # Calculate the number of refreshes needed for the layer
        self.calculate_computing_matrix(layer, params)
            
        if((layer.input.shape[3]*layer.kernel_size[0])<params.Iacts_per_PE):
            used_channels = math.floor(layer.input.shape[3])
        else:
            used_channels = 8
        if(layer.kernel_size[0] == 1):
            if (used_channels >= 12):
                used_channels = 12
        elif(layer.kernel_size[0] == 3):
            if (used_channels >= 4): #TODO: Make 5 Iacts * 3 per PE possible
                used_channels = 4
                if(used_channels >= layer.input.shape[3]):
                    used_channels = layer.input.shape[3]

        elif(layer.kernel_size[0] == 5):
            used_channels = 2

        elif(layer.kernel_size[0] >= 8):
            if(used_channels >= 2):
                used_channels = 1
        else: 
            assert False

        self.diff_iact_layer = math.ceil(layer.input.shape[3]/used_channels)
        self.used_iact_per_PE = layer.kernel_size[0] * used_channels
        self.iact_transmissions_pe = self.diff_iact_layer
        logger.debug("used_iact_per_PE " + str(self.used_iact_per_PE))
        logger.debug("iact_transmissions_pe " + str(self.iact_transmissions_pe))
            
        if((layer.filters * self.used_iact_per_PE) <= params.Wghts_per_PE):
            self.used_wght_per_PE = layer.filters*self.used_iact_per_PE
            self.used_psum_per_PE = layer.filters
            self.wght_transmissions_pe = 1
        else:
            if (layer.kernel_size[0] == 5) :
                wght_factor = math.ceil((layer.filters*self.used_iact_per_PE)/160)
            else :
                wght_factor = math.ceil((layer.filters*self.used_iact_per_PE)/params.Wghts_per_PE)
            self.used_wght_per_PE = math.ceil((layer.filters)/wght_factor)*self.used_iact_per_PE
            self.used_psum_per_PE = int(layer.filters/wght_factor)
            self.wght_transmissions_pe = math.ceil(layer.filters * self.used_iact_per_PE / self.used_wght_per_PE)

        if(math.ceil(self.used_iact_per_PE/self.used_wght_per_PE) <= params.Psums_per_PE):
            self.psum_transmissions_pe = 1
        else:    
            self.psum_transmissions_pe = math.ceil(layer.filters / params.Psums_per_PE)
            logger.debug("Error Code 5, Overused PSUM per PE, not implemented flow yet")
        #Calculation of seperate PE-Cluster
        self.used_PEs_Y    = layer.kernel_size[1]
        used_PEs_per_clm     = self.used_PEs_Y/params.PEs_Y
        self.ceil_used_PE_per_clm = math.ceil(used_PEs_per_clm)

        self.used_Y_cluster = (math.ceil(self.used_PEs_Y/params.PEs_Y))
        self.psum_delay = int(max([((self.used_wght_per_PE/2/self.used_iact_per_PE) - 2) - (self.used_Y_cluster * params.PEs_Y * 2),0]))
        if((layer.output.shape[2] % params.PEs_X)== 0):
            self.used_X_cluster = 1

        self.iact_addr_len = math.ceil((used_channels+1)/(math.ceil(params.DMA_Bits/2)/params.IACT_Addr_Bitwidth))
        self.iact_data_len = math.ceil(self.used_iact_per_PE/(math.ceil(params.DMA_Bits/2)/params.IACT_WOH_Bitwidth))

        self.psum_transmissions_glb = math.ceil(((math.ceil(layer.output.shape[1]/params.NUM_GLB_PSUM) * \
                                    layer.output.shape[2] * math.ceil(layer.output.shape[3]/ self.wght_transmissions_pe)) / \
                                    params.Clusters_X / 2 / params.Clusters_Y) \
                                    / self.psum_transmissions_pe / self.wght_transmissions_pe / params.Psum_Mem_Addr_Words)
        self.needed_psum_transmissions = self.psum_transmissions_pe * self.psum_transmissions_glb
        self.wght_transmissions_glb = 1
        self.needed_wght_transmissions = self.wght_transmissions_pe * self.wght_transmissions_glb
        
        all_transmissions_of_pe = self.iact_transmissions_pe * self.wght_transmissions_pe * self.psum_transmissions_pe
        self.Used_refreshes = math.ceil(self.used_Y_cluster* all_transmissions_of_pe * math.ceil(layer.output.shape[1] * layer.output.shape[2]/(params.PEs_X*params.Clusters)))
        
        self.used_iact_addr_per_PE = used_channels + 1
        logger.debug("Refreshes: " + str(self.Used_refreshes))
        logger.debug("Used complete new descriptions: " + str(self.Used_refreshes))
        logger.debug("used_channels : " + str(used_channels))
        logger.debug("layer.kernel_size[0] : " + str(layer.kernel_size[0]))
        logger.debug("layer_params.needed_Iact_writes : " + str(self.needed_Iact_writes))
        logger.debug("Used_refreshes : " + str(self.Used_refreshes))
        logger.debug("layer_params.needed_psum_transmissions : " + str(self.needed_psum_transmissions))
        self.iact_transmissions_glb = \
            math.ceil(((math.ceil((used_channels*layer.kernel_size[0])/2) + (math.ceil((used_channels + 1)/6))) \
            * self.needed_Iact_writes * math.ceil(self.Used_refreshes/self.wght_transmissions_pe/self.needed_psum_transmissions/self.iact_transmissions_pe))/params.Iact_Mem_Addr_Words)
        self.needed_iact_transmissions = self.iact_transmissions_pe * self.iact_transmissions_glb

        self.used_wght_addr_per_PE = (int(layer.kernel_size[0] * layer.input.shape[3] / self.iact_transmissions_pe)) + 2
        if(self.used_wght_addr_per_PE == (params.Wghts_Addr_per_PE + 1)):
            self.used_wght_addr_per_PE = self.used_wght_addr_per_PE - 1

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
 
    def write_convdw_layer(self, layer, params):
        """ Write the weights and bias of a Conv2D layer to a file. """
        
        # Get the weights and bias of the layer
        realfactor = self.get_realfactor(layer)
                        
        self.compute_total_computations(layer)
            
        self.strideX = layer.strides[0]
        self.strideY = layer.strides[1]
        self.filters = 1
        self.input_shape = layer.input.shape
        self.output_shape = layer.output.shape
        self.kernel_size = layer.kernel_size

        self.calculate_iact_transmissions(layer, params)

        # Calculate the number of refreshes needed for the layer
        if((layer.kernel_size[0] <= params.Iacts_per_PE) & (layer.kernel_size[1] <= params.PEs_Y * params.Clusters_Y)):

            self.computing_mx = [[[[1 for _ in range(params.PEs_X)]
                                            for _ in range(params.PEs_Y)]
                                            for _ in range(params.Clusters_Y)]
                                            for _ in range(params.Clusters_X)]
            if(layer.kernel_size[0] < 3):
                for x_cluster in range(params.Clusters_X):
                    for y_cluster in range(params.Clusters_Y):
                        for y_pe in range(params.PEs_Y):
                            for x_pe in range(params.PEs_X):
                                if((1 + y_pe) > layer.kernel_size[1]):
                                    self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0

            if((layer.output.shape[1] % params.PEs_X) != 0):
                if((layer.output.shape[1] < 8) | ((layer.output.shape[1] > 12) & (layer.output.shape[1] < 16))):
                    self.add_up = params.PEs_X - (layer.output.shape[1] % params.PEs_X)
                    yc_step = math.ceil(layer.output.shape[1]/(params.PEs_X*params.Clusters_X))
                    yc_start = yc_step - 1
                    yc_end = params.Clusters_Y
                    for y_cluster in range(yc_start,yc_end,yc_step):
                        for x_cluster in range(math.floor((layer.output.shape[1]%(params.Clusters_X*params.PEs_X)) / params.PEs_X),params.Clusters_X):
                            for x_pe in range(layer.output.shape[1] % params.PEs_X,params.PEs_X):
                                for y_pe in range(params.PEs_Y):
                                    self.computing_mx[x_cluster][y_cluster][y_pe][x_pe] = 0
                else:
                    assert False, "Kernel cant be caclulated"
            else:
                self.add_up = 0
            if((layer.input.shape[3]*layer.kernel_size[0])<params.Iacts_per_PE):
                used_channels = math.floor(layer.input.shape[3]*layer.kernel_size[0])
            else:
                used_channels = 8

            if(layer.kernel_size[0] == 3):
                if (used_channels >= 4):
                    used_channels = 4
            elif(layer.kernel_size[0] == 1):
                if(used_channels >= 8):
                    used_channels = 8
            else: 
                assert False

            used_channels = 1

            self.diff_iact_layer = math.ceil(layer.input.shape[3]/used_channels)
            self.used_iact_per_PE = layer.kernel_size[0] * used_channels
            self.iact_transmissions_pe = self.diff_iact_layer
            logger.debug("used_iact_per_PE " + str(self.used_iact_per_PE))
            logger.debug("iact_transmissions_pe " + str(self.iact_transmissions_pe))
                
            self.used_wght_per_PE = 2*self.used_iact_per_PE
            self.used_psum_per_PE = self.filters
            self.wght_transmissions_pe = 1

            self.psum_transmissions_pe = 1

            #Calculation of seperate PE-Cluster
            self.used_PEs_Y    = layer.kernel_size[1]
            used_PEs_per_clm     = self.used_PEs_Y/params.PEs_Y
            self.ceil_used_PE_per_clm = math.ceil(used_PEs_per_clm)

            self.used_Y_cluster = (math.ceil(self.used_PEs_Y/params.PEs_Y))
            self.psum_delay = int(max([((self.used_wght_per_PE/2/self.used_iact_per_PE) - 2) - (self.used_Y_cluster * params.PEs_Y * 2),0]))
            if((layer.output.shape[2] % params.PEs_X)== 0):
                self.used_X_cluster = 1
            
            self.iact_addr_len = math.ceil((used_channels+1)/(math.ceil(params.DMA_Bits/2)/params.IACT_Addr_Bitwidth))
            self.iact_data_len = math.ceil(self.used_iact_per_PE/(math.ceil(params.DMA_Bits/2)/params.IACT_WOH_Bitwidth))

            self.psum_transmissions_glb = math.ceil(((math.ceil(layer.output.shape[1]/params.NUM_GLB_PSUM) * \
                                    layer.output.shape[2]) / \
                                    params.Clusters_X / 2 / params.Clusters_Y) \
                                    / self.psum_transmissions_pe / self.wght_transmissions_pe / params.Psum_Mem_Addr_Words)
            self.needed_psum_transmissions = self.psum_transmissions_pe * self.psum_transmissions_glb
            self.wght_transmissions_glb = 1
            self.needed_wght_transmissions = self.wght_transmissions_pe * self.wght_transmissions_glb
            self.Used_refreshes = math.ceil(self.iact_transmissions_pe * self.psum_transmissions_pe * math.ceil(layer.output.shape[1] * layer.output.shape[2]/(params.PEs_X*params.Clusters)))
            
            self.used_iact_addr_per_PE = used_channels + 1
            logger.debug("Refreshes: " + str(self.Used_refreshes))
            logger.debug("Used complete new descriptions: " + str(self.Used_refreshes))
            logger.debug("used_channels : " + str(used_channels))
            logger.debug("layer.kernel_size[0] : " + str(layer.kernel_size[0]))
            logger.debug("self.needed_Iact_writes : " + str(self.needed_Iact_writes))
            logger.debug("Used_refreshes : " + str(self.Used_refreshes))
            logger.debug("self.needed_psum_transmissions : " + str(self.needed_psum_transmissions))
            self.iact_transmissions_glb = \
                math.ceil((self.needed_Iact_writes * math.ceil(self.Used_refreshes/self.needed_psum_transmissions/self.iact_transmissions_pe))/ \
                    math.floor(params.Iact_Mem_Addr_Words/(math.ceil((used_channels*layer.kernel_size[0])/2) + (math.ceil((used_channels + 1)/6)))))
            self.needed_iact_transmissions = self.iact_transmissions_pe * self.iact_transmissions_glb

            self.used_wght_addr_per_PE = (int(layer.kernel_size[0] * layer.input.shape[3] / self.iact_transmissions_pe)) + 2

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
    
    def write_dense_layer(self, layer, params):
        """ Write the weights and bias of a Conv2D layer to a file. """
        
        # Get the weights and bias of the layer
        realfactor = 1
                        
        # Calculate the number of PEs needed for the layer
        self.complete_computations = layer.input.shape[1] * layer.output.shape[1]
            
        self.filters = layer.output.shape[1]
        #Calculate Iact Cycles
        self.needed_Iact_writes = math.ceil(params.PEs_Y  /params.Iact_Routers)

        # Calculate the number of refreshes needed for the layer
        
        self.used_iact_per_PE = 12
        self.used_wght_per_PE = 192
        self.used_psum_per_PE = 16

        self.used_Y_cluster = 8
        self.used_X_cluster = 1

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
        
        self.psum_transmissions_pe = math.ceil(layer.output.shape[1]/(self.used_psum_per_PE * params.Clusters_X * params.Clusters_Y))
        self.psum_transmissions_glb = 1

        self.iact_transmissions_pe = math.ceil(layer.input.shape[1]/(self.used_iact_per_PE * params.PEs_Y))
        self.iact_transmissions_glb = 1

        self.needed_psum_transmissions = self.psum_transmissions_pe * self.psum_transmissions_glb
        self.wght_transmissions_pe = 1
        self.wght_transmissions_glb = 1
        self.needed_wght_transmissions = self.wght_transmissions_pe * self.wght_transmissions_glb
        self.needed_iact_transmissions = self.iact_transmissions_pe * self.iact_transmissions_glb
        self.Used_refreshes = self.iact_transmissions_pe * self.wght_transmissions_pe * self.psum_transmissions_pe
        
        self.used_iact_addr_per_PE = 1 + 1
        self.iact_data_len = math.ceil(self.used_iact_per_PE/(math.ceil(params.DMA_Bits/2)/params.IACT_WOH_Bitwidth))
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
