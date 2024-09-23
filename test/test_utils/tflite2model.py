# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import tensorflow as tf
import numpy as np
from pathlib import Path
from io import BytesIO
import tarfile
import requests

def create_model_from_tflite(use_random):
#tflite model needed for bias and weights
    script_dir = Path(__file__).resolve().parent
    tflite_model_path = Path.joinpath(script_dir, 'mobilenet_v1_0.5_128_quant.tflite')

    if not tflite_model_path.exists():
        mobilenet_v1_url = 'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128_quant.tgz'
        response = requests.get(mobilenet_v1_url)
        if response.status_code == 200:
            data = response.content
            tar_file = tarfile.open(fileobj=BytesIO(data))
            tar_file.extract('./mobilenet_v1_0.5_128_quant.tflite', path=script_dir)
            tar_file.close()

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()


#manual model parameter
    layer_type = ["Convolution", "Depthwise_Convolution","Convolution", "Depthwise_Convolution",\
                    "Convolution", "Depthwise_Convolution","Convolution", "Depthwise_Convolution",\
                    "Convolution", "Depthwise_Convolution","Convolution", "Depthwise_Convolution",\
                    "Convolution", "Depthwise_Convolution","Convolution", "Depthwise_Convolution",\
                    "Convolution", "Depthwise_Convolution","Convolution", "Depthwise_Convolution",\
                    "Convolution", "Depthwise_Convolution","Convolution", "Depthwise_Convolution",\
                    "Convolution", "Depthwise_Convolution","Convolution", "FC"]

    filter_array = [16,16,32,32,\
                64,64,64,64,\
                128,128,128,128,\
                256,256,256,256,\
                256,256,256,256,\
                256,256,256,256,\
                512,512,512,1001
                ]

    stride_array = [2,1,1,2,\
                1,1,1,2,\
                1,1,1,2,\
                1,1,1,1,\
                1,1,1,1,\
                1,1,1,1,\
                1,1,1,2,\
                1,2,1,1
                ]

    kernel_size_array = [3,3,1,3,\
                     1,3,1,3,\
                     1,3,1,3,\
                     1,3,1,3,\
                     1,3,1,3,\
                     1,3,1,3,\
                     1,3,1,1001
                     ]

    input_size_array =  [128,64,64,64,\
                    32,32,32,32,\
                    16,16,16,16,\
                    8,8,8,8,\
                    8,8,8,8,\
                    8,8,8,8,\
                    4,4,4,512,\
                        ]

    input_channel_array =   [3,16,16,32,\
                        32,64,64,64,\
                        64,128,128,128,\
                        128,256,256,256,\
                        256,256,256,256,\
                        256,256,256,256,\
                        256,512,512,512,\
                            ]


    bias_array =     [6,34,36,40,\
                 42,46,48,52,\
                 54,58,60,64,\
                 66,70,72,76,\
                 78,82,84,10,\
                 12,16,18,22,\
                 24,28,30,2]

    weight_array =  [8,35,38,41,\
                44,47,50,53,\
                56,59,62,65,\
                68,71,74,77,\
                80,83,86,11,\
                14,17,20,23,\
                26,29,32,3]



    conv_layer = []
    model = tf.keras.models.Sequential()

    for i in range(len(input_size_array)-27):
        match layer_type[i]:
            case "Convolution":
                conv_layer.append(tf.keras.layers.Conv2D(filter_array[i], (kernel_size_array[i], kernel_size_array[i]), padding="SAME",\
                                                        input_shape=(input_size_array[i], input_size_array[i], input_channel_array[i]), strides = stride_array[i]))
                model.add(conv_layer[i])

            case "Depthwise_Convolution":
                conv_layer.append(tf.keras.layers.DepthwiseConv2D((kernel_size_array[i], kernel_size_array[i]), padding="SAME",\
                                                                input_shape=(input_size_array[i], input_size_array[i], input_channel_array[i])))
                model.add(conv_layer[i])

            case "FC":
                conv_layer.append(tf.keras.Input(shape =(input_size_array[i],)))
                model.add(conv_layer[i])
                conv_layer.append(tf.keras.layers.Dense(1, use_bias = True))
                model.add(conv_layer[i])



        if use_random == 0:
            weights = interpreter.get_tensor(weight_array[i])
            reshaped_weights = np.transpose(weights, (1, 2, 3, 0))
            bias = interpreter.get_tensor(bias_array[i])
            conv_layer[i].set_weights([reshaped_weights, bias])


        show_weights, show_biases = conv_layer[i].get_weights()
        print(f"Layer {i} Gewichte Form:", show_weights.shape)
        print(f"Layer {i} Gewichte Werte:", show_weights)

        print(f"Layer {i} Biases Form:", show_biases.shape)
    print(f"Layer {i} Biases Werte:", show_biases)
    print("use random:",use_random)
    model.summary()
    return model

if __name__ == "__main__":
    use_random = 0
    model = create_model_from_tflite(use_random)