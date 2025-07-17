# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import h5py
import tensorflow as tf
import numpy as np

hdf5_file_path = 'model_info.h5'

def create_model_from_hdf5(hdf5_file_path):
    filters = {}
    weights = {}
    bias = {}
    dilations = {}
    group = {}
    kernel_shape = {}
    pads = {}
    strides = {}

    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        for layer_key in hdf5_file.keys():
            
            filters[layer_key] = hdf5_file[layer_key]['filters'][()]  # Filter
            weights[layer_key] = hdf5_file[layer_key]['weights'][()]  # Weights
            bias[layer_key] = hdf5_file[layer_key]['bias'][()]  # Bias
            dilations[layer_key] = hdf5_file[layer_key]['dilations'][()]  # Dilations
            group[layer_key] = hdf5_file[layer_key]['group'][()].item()  # Group
            kernel_shape[layer_key] = tuple(hdf5_file[layer_key]['kernel_shape'][()])  # Kernel-Form
            pads[layer_key] = tuple(hdf5_file[layer_key]['pads'][()])  # Pads
            strides[layer_key] = tuple(hdf5_file[layer_key]['strides'][()])  # Strides

            kernel_shape[layer_key] = tuple(map(int, kernel_shape[layer_key]))
            strides[layer_key] = tuple(map(int, strides[layer_key]))

            weights_data = hdf5_file[layer_key]['weights'][:]
            print("layer",[layer_key] )
            print("Shape of weights:", weights_data.shape)
            print("filters",filters[layer_key])
            print("kernel shape",kernel_shape[layer_key])
            print("strides",strides[layer_key])

    print('1:',weights['1'])
    print('3:',weights['3'])


    model = tf.keras.Sequential()
    conv_layer_1 = tf.keras.layers.Conv2D(filters=filters['1'], kernel_size=kernel_shape['1'], strides=strides['1'],
                                    padding='same', input_shape=(28, 28, 3))
    conv_layer_1.build((None, None, None, 3))  
    conv_layer_1.set_weights([weights['1'], bias['1']])
    model.add(conv_layer_1)

    output_shape_1 = conv_layer_1.compute_output_shape((None, 28, 28, 3))
    print("Shape of output from conv_layer_1:", output_shape_1)


    conv_layer_3 = tf.keras.layers.Conv2D(filters=filters['3'], kernel_size=kernel_shape['3'], strides=strides['3'],
                                    padding='same')  

    dummy_input = tf.zeros((1,) + model.layers[0].output_shape[1:])
    print("Shape of dummy_input:", dummy_input.shape)
    print("dummy_input:", dummy_input)
    conv_layer_3(dummy_input)
    conv_layer_3.set_weights([weights['3'], bias['3']])   
    model.add(conv_layer_3)



    print(model.summary())
    return model


def print_parameter(conv_layer):
    model = tf.keras.models.Sequential()
    model.add(conv_layer)
    print("Convolutional Layer Details:")
    example_filters = conv_layer.filters
    filters_data_type = type(np.array([example_filters])[0])
    print("Number of Filters:", example_filters, ", Datentyp:", filters_data_type)
    example_kernel_size = conv_layer.kernel_size
    kernel_size_data_type = type(np.array([example_kernel_size])[0])
    print("Kernel Size:", example_kernel_size, ", Datentyp:", kernel_size_data_type)

    print("Kernel Size Data Type:", type(example_kernel_size[0]))

    print("Strides:", conv_layer.strides, ", Datentyp:", type(conv_layer.strides))
    print("Padding:", conv_layer.padding, ", Datentyp:", type(conv_layer.padding))
    print("Dilations:", conv_layer.dilation_rate, ", Datentyp:", type(conv_layer.dilation_rate))
    print("Groups:", conv_layer.groups, ", Datentyp:", type(conv_layer.groups))

if __name__ == "__main__":
    model = create_model_from_hdf5(hdf5_file_path)
