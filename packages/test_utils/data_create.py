# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import tensorflow as tf
import logging
import subprocess
import os
import tempfile
import math

logger = logging.getLogger("cocotb")

def create_layer(layer_mode, filters, kernelsize, inputsize_x, inputsize_y, strides, channels, outputsize):
    """Create a TensorFlow/Keras neural network layer based on the specified configuration.

    Constructs a Keras Sequential model containing the requested layer type with the
    given parameters. Supports various layer types including convolution, depthwise
    convolution, fully connected (dense), and pooling layers.

    Args:
        layer_mode (str): Type of layer to create. Supported values:
            - "Convolution": Standard 2D convolution layer
            - "Depthwise_Convolution": Depthwise 2D convolution layer
            - "FC": Fully connected (Dense) layer
            - "Pooling": Conv2D followed by MaxPooling2D and another Conv2D
            - "Pooling_OLD": Conv2D, MaxPooling2D, and Dense layer sequence
        filters (int): Number of output filters/channels for convolution layers
        kernelsize (int): Size of the convolution kernel (assumes square kernel)
        inputsize_x (int): Width of the input tensor
        inputsize_y (int): Height of the input tensor
        strides (int or tuple): Stride value(s) for convolution operations
        channels (int): Number of input channels
        outputsize (int): Number of output units for fully connected layers

    Returns:
        tf.keras.models.Sequential: Compiled Keras Sequential model containing the
            specified layer(s). The model is compiled with Adam optimizer and
            sparse categorical crossentropy loss.

    Note:
        - All convolution layers use "SAME" padding
        - The "Pooling" and "Pooling_OLD" modes use hardcoded dimensions for
          specific network architectures
        - Invalid layer_mode values will log an error and return an empty model
    """
    logger.debug("Start compiling.")
    model = tf.keras.models.Sequential()
    match layer_mode:
        case "Convolution":
            model.add(tf.keras.layers.Conv2D(filters, (kernelsize, kernelsize), padding="SAME", input_shape=(inputsize_x, inputsize_y, channels), strides = strides))
            #model.add(tf.keras.layers.Conv2D(filters, (kernelsize, kernelsize), padding="SAME", input_shape=(inputsize_x, inputsize_y, filters), strides = strides))

        case "Depthwise_Convolution":
            model.add(tf.keras.layers.DepthwiseConv2D((kernelsize, kernelsize), padding="SAME", input_shape=(inputsize_x, inputsize_y, channels), strides = strides))
        case "FC":
            model.add(tf.keras.layers.Dense(input_shape=(1,1,inputsize_x), units=outputsize, use_bias = True))
        case "Pooling":
            channels = 4
            x_axis = 14
            y_axis = 14
            filters = 32
            model.add(tf.keras.layers.Conv2D(filters, (3, 3), padding="SAME", input_shape=(x_axis, y_axis, channels), strides = strides))
            
            pool_x_axis = 2
            pool_y_axis = 2
            model.add(tf.keras.layers.MaxPooling2D(pool_size = (pool_x_axis, pool_y_axis), strides=(2,2), padding="valid"))
            
            channels = filters
            x_axis   = math.ceil(x_axis/pool_x_axis)
            y_axis   = math.ceil(y_axis/pool_y_axis)
            filters  = 32
            model.add(tf.keras.layers.Conv2D(filters, (3, 3), padding="SAME", input_shape=(x_axis, y_axis, channels), strides = strides))
            
        case "Pooling_OLD":
            channels = 4
            x_axis = 64
            y_axis = 1
            pool_x_axis = 64
            pool_y_axis = 1
            filters = 8
            outputvalue = 20
            model.add(tf.keras.layers.Conv2D(filters, (3, 3), padding="SAME", input_shape=(x_axis, y_axis, channels), strides = strides))
            model.add(tf.keras.layers.MaxPooling2D(pool_size = (pool_x_axis, pool_y_axis), strides=(1,1), padding="valid"))
            model.add(tf.keras.layers.Dense(input_shape=(filters), units=outputvalue, use_bias = True))

        case _:
            logger.error("Layer not detected!")

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logger.debug("Model compiled.")
    return model
