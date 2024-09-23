# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import math
import numpy as np

class DRAMContents(object):
    """ The DRAM contents that is used to store intermediate data during tests.

    The DRAM contents is used for the simulation of OpenEye accelerator. It is used
    during the simulation to store the intermediate data of the accelerator
    between different layers or between different repetitions of the same layer.

    """

    def __init__(self, model) -> None:
        dram_fmap = []
        dram_weights = []
        dram_bias = []
        for i in range(len(model.layers)):
            if "Depthwise" in str(model.layers[i]):
                dram_weights.append([[[0 for l in range(model.layers[i].kernel_size[1])]
                                    for k in range(model.layers[i].kernel_size[0])]
                                    for j in range(model.layers[i].input.shape[3])])
            elif "Conv" in str(model.layers[i]):
                dram_weights.append([[[[0 for m in range(model.layers[i].kernel_size[1])]
                                    for l in range(model.layers[i].kernel_size[0])]
                                    for k in range(model.layers[i].filters)]
                                    for j in range(model.layers[i].input.shape[3])])
            elif "Dense" in str(model.layers[i]):
                dram_weights.append([[0 for m in range(model.layers[i].input.shape[1])]
                                    for l in range(model.layers[i].output.shape[1])])
            elif "Flat" in str(model.layers[i]):
                dram_weights.append([0])

            if "Depthwise" in str(model.layers[i]):
                dram_bias.append([0 for k in range(model.layers[i].kernel_size[0])])
            elif "Conv" in str(model.layers[i]):
                dram_bias.append([0 for m in range(model.layers[i].filters)])
            elif "Dense" in str(model.layers[i]):
                dram_bias.append([[0 for m in range(model.layers[i].input.shape[1])]
                                    for l in range(model.layers[i].output.shape[1])])

            if "Conv" in str(model.layers[i]):
                dram_fmap.append([[[0 for l in range(model.layers[i].input.shape[2])]
                                for k in range(model.layers[i].input.shape[1])]
                                for j in range(model.layers[i].input.shape[3])])
            elif "Dense" in str(model.layers[i]):
                dram_fmap.append([0 for j in range(model.layers[i].input.shape[1])])
            elif "Flat" in str(model.layers[i]):
                dram_fmap.append([[[0 for l in range(model.layers[i].input.shape[2])]
                                for k in range(model.layers[i].input.shape[1])]
                                for j in range(model.layers[i].input.shape[3])])

        for i in [len(model.layers)-1]:
            if "Conv" in str(model.layers[i]):
                dram_fmap.append([[[0 for l in range(model.layers[i].output.shape[2])]
                                    for k in range(model.layers[i].output.shape[1])]
                                    for j in range(model.layers[i].output.shape[3])])
            elif "Dense" in str(model.layers[i]):
                dram_fmap.append([0 for l in range(model.layers[i].output.shape[1])])
        self.fmap = dram_fmap
        self.weights = dram_weights
        self.bias = dram_bias

    def write_initial_data_to_dram(self, model):
        """ TODO: Docu"""
        for l in range(len(model.layers)):
            if "Depthwise" in str(model.layers[l]):
                for c in range(model.layers[l].input.shape[3]):
                    for x in range(model.layers[l].kernel_size[0]):
                        for y in range(model.layers[l].kernel_size[1]):
                            self.weights[l][c][x][y] = int(math.floor(float(127*model.layers[l].weights[0][x][y][c])))
            elif "Conv" in str(model.layers[l]):
                for c in range(model.layers[l].input.shape[3]):
                    for f in range(model.layers[l].filters):
                        for x in range(model.layers[l].kernel_size[0]):
                            for y in range(model.layers[l].kernel_size[1]):
                                self.weights[l][c][f][x][y] = int(math.floor(float(127*model.layers[l].weights[0][x][y][c][f])))
            elif "Dense" in str(model.layers[l]):
                for c in range(model.layers[l].input.shape[1]):
                    for x in range(model.layers[l].output.shape[1]):
                            self.weights[l][x][c] = np.random.randint(-128, 127)

        for l in range(len(model.layers)):
            if "Depthwise" in str(model.layers[l]):
                if (model.layers[l].kernel_size[0] != 1):
                    for x in range(model.layers[l].kernel_size[0]):
                        self.bias[l][x] = int(math.floor(float(model.layers[l].weights[1][x])))

                else:
                    self.bias[l][0] = int(math.floor(float(model.layers[l].weights[1])))
            elif "Conv" in str(model.layers[l]):
                for x in range(model.layers[l].filters):
                    self.bias[l][x] = int(math.floor(float(model.layers[l].weights[1][x])))
            elif "Dense" in str(model.layers[l]):
                for c in range(model.layers[l].input.shape[1]):
                    for x in range(model.layers[l].output.shape[1]):
                            self.bias[l][x][c] = np.random.randint(-128, 127)

        if "Conv" in str(model.layers[0]):
            for c in range(model.layers[0].input.shape[3]):
                for x in range(model.layers[0].input.shape[1]):
                    for y in range(model.layers[0].input.shape[2]):
                        self.fmap[0][c][x][y] = np.random.randint(-128, 127)
        elif "Dense" in str(model.layers[0]):
            for c in range(model.layers[l].input.shape[1]):
                self.fmap[0][c] = np.random.randint(-128, 127)
