# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import tensorflow as tf
import logging

logger = logging.getLogger("cocotb")

def create_layer(layer_mode, filters, kernelsize, inputsize_x, inputsize_y, strides, channels,outputsize):
    logger.debug("Start compiling.")
    model = tf.keras.models.Sequential()
    match layer_mode:
        case "Convolution":
            model.add(tf.keras.layers.Conv2D(filters, (kernelsize, kernelsize), padding="SAME", input_shape=(inputsize_x, inputsize_y, channels), strides = strides))
            model.add(tf.keras.layers.Conv2D(filters, (kernelsize, kernelsize), padding="SAME", input_shape=(inputsize_x, inputsize_y, filters), strides = strides))
        case "Depthwise_Convolution":
            model.add(tf.keras.layers.DepthwiseConv2D((kernelsize, kernelsize), padding="SAME", input_shape=(inputsize_x, inputsize_y, channels), strides = strides))
        case "FC":
            model.add(tf.keras.layers.Dense(input_shape=(1,1,inputsize_x), units=outputsize, use_bias = True))
        case "Pooling":
            channels = 1
            x_axis = 28
            y_axis = 2
            filters = 32
            model.add(tf.keras.layers.Conv2D(filters, (3, 3), padding="SAME", input_shape=(x_axis, y_axis, channels), strides = strides))

            pool_x_axis = 2
            pool_y_axis = 2
            model.add(tf.keras.layers.MaxPooling2D(pool_size = (pool_x_axis, pool_y_axis), strides=(2,2), padding="valid"))

            channels = 32
            x_axis = 14
            y_axis = 1
            filters = 32
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
            logger.debug("Layer not detected!")

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logger.debug("Model compiled.")
    return model
