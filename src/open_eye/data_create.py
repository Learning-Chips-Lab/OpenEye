# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import logging
import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tempfile
import math
import tensorflow as tf

logger = logging.getLogger("cocotb")

def create_layer(layer_mode, filters, kernelsize_x, kernelsize_y, inputsize_x, inputsize_y, strides, channels, outputsize):
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
        - All convolution layers use "same" padding
        - The "Pooling" and "Pooling_OLD" modes use hardcoded dimensions for
          specific network architectures
        - Invalid layer_mode values will log an error and return an empty model
    """
    logger.debug("Start compiling.")
    # Deterministic weight initialization when a seed is provided. Without
    # this, Keras draws fresh random weights every run, which makes GEMM/Dense
    # layer tests non-reproducible (a given run may pass or fail purely on the
    # random draw). Set OPENEYE_LAYER_SEED to fix the weights across runs.
    _seed = os.environ.get("OPENEYE_LAYER_SEED")
    if _seed is not None:
        import numpy as np
        tf.keras.utils.set_random_seed(int(_seed))
        np.random.seed(int(_seed))
    model = tf.keras.models.Sequential()
    match layer_mode:
        case "Convolution":
            model.add(tf.keras.layers.Conv2D(filters, (kernelsize_x, kernelsize_y), padding="same", input_shape=(inputsize_x, inputsize_y, channels), strides = strides))
            #model.add(tf.keras.layers.Conv2D(filters, (kernelsize_x, kernelsize_y), padding="same", input_shape=(inputsize_x, inputsize_y, filters), strides = strides))
            #model.add(tf.keras.layers.Conv2D(filters, (kernelsize_x, kernelsize_y), padding="same", input_shape=(64, 4, filters), strides = 1))
            #model.add(tf.keras.layers.Conv2D(filters, (kernelsize_x, kernelsize_y), padding="same", input_shape=(64, 4, filters), strides = 1))
            #model.add(tf.keras.layers.Flatten())
            #model.add(tf.keras.layers.Dense(units=outputsize, use_bias = True))

        case "Depthwise_Convolution":
            model.add(tf.keras.layers.DepthwiseConv2D((kernelsize_x, kernelsize_y), padding="same", input_shape=(inputsize_x, inputsize_y, channels), strides = strides))
        case "GEMM":
            # Pure matrix multiplication C = A x B (+ bias), realized as a single
            # Dense layer. Combined with DATAFLOW="output_stationary" (or the
            # GemmMapper) it runs on the output-stationary GEMM datapath
            # (gemm_mode=1) instead of the row-stationary conv routing.
            model.add(tf.keras.layers.Dense(input_shape=(inputsize_x,), units=outputsize, use_bias=True))
        case "FC":
            model.add(tf.keras.layers.Conv2D(filters, (kernelsize_x, kernelsize_y), padding="same", input_shape=(inputsize_x, inputsize_y, channels), strides = strides))
            model.add(tf.keras.layers.Flatten())
            #model.add(tf.keras.layers.Dense(units=10, use_bias = True))
            model.add(tf.keras.layers.Dense(units=outputsize, use_bias = True))
            #model.add(tf.keras.layers.Dense(input_shape=(1,1,inputsize_x), units=outputsize, use_bias = True))
        case "MNIST":
            channels = 4
            x_axis = 28
            y_axis = 28
            channels = 2
            filters = 16
            pool_x_axis = 2
            pool_y_axis = 2
            
            model.add(tf.keras.layers.Conv2D(filters, (3, 3), padding="same", input_shape=(x_axis, y_axis, channels), strides = strides))
            model.add(tf.keras.layers.MaxPooling2D(pool_size = (pool_x_axis, pool_y_axis), strides=(pool_x_axis,pool_y_axis), padding="valid"))
            channels = filters
            x_axis   = math.ceil(x_axis/pool_x_axis)
            y_axis   = math.ceil(y_axis/pool_y_axis)
            filters  = 16
            model.add(tf.keras.layers.Conv2D(filters, (3, 3), padding="same", input_shape=(x_axis, y_axis, channels), strides = strides))
            model.add(tf.keras.layers.MaxPooling2D(pool_size = (pool_x_axis, pool_y_axis), strides=(pool_x_axis,pool_y_axis), padding="valid"))
            channels = filters
            x_axis   = math.ceil(x_axis/pool_x_axis)
            y_axis   = math.ceil(y_axis/pool_y_axis)
            filters  = 16
            model.add(tf.keras.layers.Conv2D(filters, (3, 3), padding="same", input_shape=(x_axis, y_axis, channels), strides = strides))
            model.add(tf.keras.layers.Flatten())
            output_size  = 10
            model.add(tf.keras.layers.Dense(units=output_size, use_bias = True))
            
        case "Pooling":
            x_axis = inputsize_x
            y_axis = inputsize_y

            model.add(tf.keras.layers.Conv2D(
                filters, 
                (3, 3), 
                padding="same",
                input_shape=(x_axis, y_axis, channels), 
                strides=strides
            ))

            pool_x_axis = 2
            pool_y_axis = 2
            model.add(tf.keras.layers.MaxPooling2D(
                pool_size=(pool_x_axis, pool_y_axis), 
                strides=(pool_x_axis, pool_y_axis), 
                padding="valid"
            ))
            x_axis = math.ceil(x_axis/pool_x_axis)
            y_axis = math.ceil(y_axis/pool_y_axis)
            model.add(tf.keras.layers.Conv2D(
                filters, 
                (3, 3), 
                padding="same",
                input_shape=(x_axis, y_axis, channels), 
                strides=strides
            ))
            """
            model.add(tf.keras.layers.Flatten())

            outputsize = 32
            model.add(tf.keras.layers.Dense(units=outputsize, use_bias=True))"""
        case _:
            logger.error("Layer not detected!")

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logger.debug("Model compiled.")
    return model
