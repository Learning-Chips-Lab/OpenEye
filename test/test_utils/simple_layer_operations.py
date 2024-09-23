# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import math


def pool(dram, layer, layer_number):
    temp_number = 0

    if "Average" in str(layer):
        for oc in range(layer.output.shape[3]):
            for ox in range(layer.output.shape[2]):
                for oy in range(layer.output.shape[1]):
                    for ix in range(layer.strides[1]):
                        for iy in range(layer.strides[0]):
                            temp_number = temp_number + dram.fmap[layer_number][oc][ix][iy]
                    dram.fmap[layer_number + 1][oc][ox][oy] = int(temp_number/ (layer.strides[0]*layer.strides[1]))

    elif "Max" in str(layer):
        for oc in range(layer.output.shape[3]):
            for ox in range(layer.output.shape[2]):
                for oy in range(layer.output.shape[1]):
                    for ix in range(layer.strides[1]):
                        for iy in range(layer.strides[0]):
                            if(dram.fmap[layer_number][oc][ix][iy] > temp_number):
                                temp_number = dram.fmap[layer_number][oc][ix][iy]

                    dram.fmap[layer_number + 1][oc][ox][oy] = temp_number

def flat(dram, layer, layer_number):
    for ic in range(layer.input.shape[3]):
        for iy in range(layer.input.shape[1]):
            for ix in range(layer.input.shape[2]):
                    dram.fmap[layer_number + 1][ic * layer.input.shape[2] * layer.input.shape[1]+ iy * layer.input.shape[2] + ix] = dram.fmap[layer_number][ic][ix][iy]

def batchnorm_output(layer, divide_value, layer_number, dram):
    if "Conv" in str(layer):
        for f in range(len(dram.fmap[1 + layer_number])):
            for x in range(len(dram.fmap[1 + layer_number][f])):
                for y in range(len(dram.fmap[1 + layer_number][f][x])):
                    dram.fmap[1 + layer_number][f][x][y] = math.floor(dram.fmap[1 + layer_number][f][x][y]/divide_value)
    elif "Dense" in str(layer):
        for f in range(len(dram.fmap[1 + layer_number])):
            dram.fmap[1 + layer_number][f] = math.floor(dram.fmap[1 + layer_number][f]/divide_value)

