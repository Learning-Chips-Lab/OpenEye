# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Layer operation utilities for the OpenEye neural network accelerator.

This module provides implementation of basic neural network layer operations
that are performed on the host side or used for reference calculations.
It includes operations for:
- Pooling layers (max and average pooling)
- Flattening operations
- Batch normalization and quantization

The operations work directly with the DRAM data structures used by OpenEye
and handle the data format transformations needed between layers.
"""

import math

def pool(dram, layer, layer_number):
    """Perform pooling operation on feature maps.
    
    This function implements both average and max pooling operations on the input
    feature maps. The type of pooling is determined by the layer configuration.
    
    Args:
        dram: Memory object containing feature maps
        layer: Layer configuration object containing:
            - output.shape: Output dimensions (batch, height, width, channels)
            - strides: Pooling window size and stride
            - Type indicator ("Average" or "Max" in layer string representation)
        layer_number: Index of the current layer (used for DRAM indexing)
    
    The function operates in-place on the DRAM object, writing results to
    layer_number + 1 index. For average pooling, results are integer-divided
    by the pooling window area.
    """
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
    """Flatten multi-dimensional feature maps to 1D arrays.
    
    This function transforms 3D feature maps (channels, height, width) into
    a 1D array, which is typically used before fully connected layers. The
    flattening preserves the data order required by subsequent operations.
    
    Args:
        dram: Memory object containing feature maps
        layer: Layer configuration object containing:
            - input.shape: Input dimensions (batch, height, width, channels)
        layer_number: Index of the current layer
        
    The function performs the following transformations:
    - Reads 3D input from dram[layer_number]
    - Writes flattened 1D output to dram[layer_number + 1]
    - Maintains channel-major ordering (channels before spatial dimensions)
    """
    for ic in range(layer.input.shape[3]):
        for iy in range(layer.input.shape[1]):
            for ix in range(layer.input.shape[2]):
                    dram.fmap[layer_number + 1][ic * layer.input.shape[2] * layer.input.shape[1]+ iy * layer.input.shape[2] + ix] = dram.fmap[layer_number][ic][ix][iy]

def batchnorm_output(layer_parameters, divide_value, layer_number, dram):
    """Apply batch normalization and quantization to layer outputs.
    
    This function performs post-processing on layer outputs, applying both
    batch normalization and quantization. It handles both convolutional and
    dense layer outputs differently due to their distinct data organizations.
    
    Args:
        layer_parameters: Layer configuration containing:
            - layer_name: Type of layer ("Conv" or "Dense")
            - quantize: Per-channel quantization parameters [scale, shift]
        divide_value: Division factor for dense layer normalization
        layer_number: Index of the current layer
        dram: Memory object containing feature maps
        
    The function handles two cases:
    1. Convolutional layers: 
       - Applies per-channel quantization with scale and shift
       - Uses floor division to maintain integer arithmetic
    2. Dense layers:
       - Applies simple division by divide_value
       - Uses floor division for result
    
    All operations are performed in-place on dram[layer_number + 1].
    """
    #if "Conv" in str(layer_parameters.layer_name):
    try:
        for f in range(len(dram.fmap[1 + layer_number])):
            for x in range(len(dram.fmap[1 + layer_number][f])):
                for y in range(len(dram.fmap[1 + layer_number][f][x])):
                    dram.fmap[1 + layer_number][f][x][y] = math.floor((layer_parameters.quantize[f][0]*dram.fmap[1 + layer_number][f][x][y])/(2**layer_parameters.quantize[f][1]))
    #elif "Dense" in str(layer_parameters.layer_name):
    except:
        for f in range(len(dram.fmap[1 + layer_number])):
            dram.fmap[1 + layer_number][f] = math.floor(dram.fmap[1 + layer_number][f]/(2**layer_parameters.quantize[0][1]))

