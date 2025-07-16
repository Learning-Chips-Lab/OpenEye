# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import math
import tensorflow as tf
import numpy as np
from pathlib import Path
from io import BytesIO
import tarfile
import requests

class TFLite_layer(object):
    def __init__(self, name, idx_in, idx_out, input_shape, output_shape):
        self.name = name
        self.idx_in = idx_in
        self.idx_out = idx_out

        self.input_shape = input_shape
        self.output_shape = output_shape

class TFLite_conv2d(TFLite_layer):
    weights = []
    store_in_psum = 0
    skip_psum = 0

    def __init__(self, idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp, relu=False, bn=False):
        super().__init__('conv2d', idx_in, idx_out, input_shape, output_shape)
        
        self.weights = [weights, bias]

        self.filters = len(bias)
        self.kernel_size = weights.shape[:2]
        self.kernel = weights

        self.relu = relu
        self.batchnorm = bn
        self.strides = (1, 1)

        # quantization
        self.quantization_factor = qf
        self.zero_point = zp

class TFLite_max_pooling2d(TFLite_layer):
    def __init__(self, idx_in, idx_out, input_shape, output_shape):
        super().__init__('max_pooling2d', idx_in, idx_out, input_shape, output_shape)

class TFLite_dense(TFLite_layer):
    weights = []
    qf = None

    def __init__(self, idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp):
        super().__init__('dense', idx_in, idx_out, input_shape, output_shape)

        self.weights = [weights, bias]

        # quantization
        self.quantization_factor = qf
        self.zero_point = zp

class TFLite_model(object):
    layers = []
    def __init__(self):
        pass

    def add_conv2d(self, idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp, relu=None, bn=None):
        layer = TFLite_conv2d(idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp, relu, bn)
        self.layers.append(layer)

    def add_max_pooling2d(self, idx_in, idx_out, input_shape, output_shape):
        layer = TFLite_max_pooling2d(idx_in, idx_out, input_shape, output_shape)
        self.layers.append(layer)

    def add_dense(self, idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp):
        layer = TFLite_dense(idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp)
        self.layers.append(layer)

def quantize_scale(scale):
    if scale == 0:
        return 0, 0

    m, e = math.frexp(scale)
    q = int(round(m * (1 << 31)))
    
    if q == (1 << 31):
        q //= 2
        e += 1

    shift = -e
    return q, shift

def create_model_from_tflite(use_random=False, tflite_model_path=None, model_name='resnet'):
    #tflite model needed for bias and weights
    script_dir = Path(__file__).resolve().parent.parent / 'cocotb_fpga'

    if tflite_model_path:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        interpreter.allocate_tensors()
        tensor_details = interpreter.get_tensor_details()
        graph = interpreter._get_ops_details()

        input_layer_idx = -1
        tflite_model = TFLite_model()

        for op in graph:
            match op['op_name']:
                case 'QUANTIZE':
                    input_layer_idx = op['outputs'][0]
                case 'CONV_2D':
                    idx_in, idx_w, idx_b = (op['inputs'])
                    input_shape = tensor_details[idx_in]['shape']
                    w = interpreter.get_tensor(idx_w)
                    w = np.transpose(w, (1, 2, 3, 0))
                    b = interpreter.get_tensor(idx_b)
                    
                    idx_out = op['outputs'][0]
                    output_shape = tensor_details[idx_out]['shape']
                    
                    qf_i, zp_i = tensor_details[idx_in]['quantization']
                    qf_w = tensor_details[idx_w]['quantization_parameters']['scales']
                    zp_w = tensor_details[idx_w]['quantization_parameters']['zero_points']
                    qf_o, zp_o = tensor_details[idx_out]['quantization']

                    qf = []
                    for qf_n in qf_w:
                        mult, shift = quantize_scale(qf_i * qf_n / qf_o)
                        qf.append((mult, shift+31))

                    relu = False
                    if 'Relu' in tensor_details[idx_b]['name']:
                        relu = True

                    bn = False
                    if 'FusedBatchNorm' in tensor_details[idx_b]['name']:
                        bn = True

                    tflite_model.add_conv2d(idx_in, idx_out, input_shape, output_shape, w, b, qf, zp_o, relu, bn)
                case 'ADD':
                    pass
                case 'MUL':
                    pass
                case 'REDUCE_MAX':
                    idx_in = op['inputs'][0]
                    input_shape = tensor_details[idx_in]['shape']
                    idx_out = op['outputs'][0]
                    output_shape = tensor_details[idx_out]['shape']

                    # tflite_model.add_max_pooling2d(idx_in, idx_out, input_shape, output_shape)

                case 'FULLY_CONNECTED':
                    idx_in, idx_w, idx_b = (op['inputs'])
                    input_shape = tensor_details[idx_in]['shape']
                    w = interpreter.get_tensor(idx_w)
                    w = np.transpose(w, (1, 0))
                    b = interpreter.get_tensor(idx_b)
                    
                    idx_out = op['outputs'][0]
                    output_shape = tensor_details[idx_out]['shape']
                    
                    qf_i, zp_i = tensor_details[idx_in]['quantization']
                    qf_w = tensor_details[idx_w]['quantization_parameters']['scales']
                    zp_w = tensor_details[idx_w]['quantization_parameters']['zero_points']
                    qf_o, zp_o = tensor_details[idx_out]['quantization']

                    qf = []
                    for qf_n in qf_w:
                        mult, shift = quantize_scale(qf_i * qf_n / qf_o)
                        qf.append((mult, shift+31))

                    # tflite_model.add_dense(idx_in, idx_out, input_shape, output_shape, w, b, qf, zp_o)

                case default:
                    pass
        
        return tflite_model

    elif model_name:
        match model_name:
            case 'mobilenet':
                if (False):
                    tflite_model_path = Path.joinpath(script_dir, 'mobilenet_v1_0.5_128_quant.tflite')

                    if not tflite_model_path.exists():
                        mobilenet_v1_url = 'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128_quant.tgz'
                        response = requests.get(mobilenet_v1_url)
                        if response.status_code == 200:
                            data = response.content
                            tar_file = tarfile.open(fileobj=BytesIO(data))
                            tar_file.extract('./mobilenet_v1_0.5_128_quant.tflite', path=script_dir)
                            tar_file.close()
                    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
                    interpreter.allocate_tensors()
                else:
                    tflite_model_path = Path.joinpath(script_dir, 'tflite_net')

                    #if not tflite_model_path.exists():
                    mobilenet_v1_url = 'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128.tgz'
                    response = requests.get(mobilenet_v1_url)
                    if response.status_code == 200:
                        data = response.content
                        tar_file = tarfile.open(fileobj=BytesIO(data))
                        tar_file.extractall(path=tflite_model_path)
                        tar_file.close()
                    model = tf.saved_model.load(tflite_model_path)
                    model.summary()

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
                            512,512,512,1000
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
                                1,3,1,1000
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
            case 'resnet':
                tflite_model_path = Path.joinpath(script_dir, 'resnet_quantized.tflite')

                interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
                interpreter.allocate_tensors()

                layer_type = ['Convolution', 'Convolution', 'Convolution',]

                bias_array = [24, 22, 20]
                weight_array = [25, 23, 21]

                stride_array = [1, 1, 1]

                input_channel_array = []
                input_size_x_array = [48, 48, 48]
                input_size_y_array = [64, 64, 64]

                filter_array = []
                kernel_size_array = []

                for idx in weight_array:
                    w_shape = interpreter.get_tensor(idx).shape
                    input_channel_array.append(w_shape[3])
                    filter_array.append(w_shape[0])
                    kernel_size_array.append(w_shape[1])


        conv_layer = []
        model = tf.keras.models.Sequential()

        for i in range(len(layer_type)):
            match layer_type[i]:
                case "Convolution":
                    conv_layer.append(tf.keras.layers.Conv2D(filter_array[i], (kernel_size_array[i], kernel_size_array[i]), padding="SAME",\
                                                            input_shape=(input_size_x_array[i], input_size_y_array[i], input_channel_array[i]), strides = stride_array[i]))
                    model.add(conv_layer[i])

                case "Depthwise_Convolution":
                    conv_layer.append(tf.keras.layers.DepthwiseConv2D((kernel_size_array[i], kernel_size_array[i]), padding="SAME",\
                                                                    input_shape=(input_size_x_array[i], input_size_y_array[i], input_channel_array[i])))
                    model.add(conv_layer[i])

                case "FC":
                    conv_layer.append(tf.keras.Input(shape =(input_size_array[i],)))
                    model.add(conv_layer[i])
                    conv_layer.append(tf.keras.layers.Dense(1, use_bias = True))
                    model.add(conv_layer[i])



            if not use_random:
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
    model = create_model_from_tflite()