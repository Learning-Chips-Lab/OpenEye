# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""TFLite model converter and interface for OpenEye neural network accelerator.

This module provides functionality to convert TensorFlow Lite models to a format
compatible with the OpenEye neural network accelerator. It includes:

- Classes for representing different neural network layer types
- Model conversion from TFLite format
- Quantization parameter handling
- Support for common layer types (Conv2D, Dense, MaxPooling)
- Utilities for model inspection and manipulation

Key Classes:
- TFLite_layer: Base class for all layer types
- TFLite_conv2d: Convolutional layer representation  
- TFLite_max_pooling2d: Max pooling layer representation
- TFLite_dense: Fully connected layer representation
- TFLite_model: Container class for the complete model

The module supports both direct TFLite model loading and creation of 
predefined model architectures like MobileNet and ResNet.
"""

import math
import tensorflow as tf
import numpy as np
from pathlib import Path
from io import BytesIO
import tarfile
import requests

class TFLite_layer(object):
    """Base class for TensorFlow Lite layers in OpenEye.

    This class serves as the foundation for all layer types supported by
    the OpenEye accelerator. It provides common attributes and interfaces
    for layer representation.

    Attributes:
        name (str): Name/type of the layer
        idx_in (int): Input tensor index in TFLite model
        idx_out (int): Output tensor index in TFLite model
        input_shape (tuple): Shape of input tensor
        output_shape (tuple): Shape of output tensor
    """
    def __init__(self, name, idx_in, idx_out, input_shape, output_shape):
        """Initialize a TFLite layer.

        Args:
            name: Identifier/type of the layer
            idx_in: Input tensor index
            idx_out: Output tensor index
            input_shape: Shape of input tensor
            output_shape: Shape of output tensor
        """
        self.name = name
        self.idx_in = idx_in
        self.idx_out = idx_out

        self.input_shape = input_shape
        self.output_shape = output_shape

    @property
    def input(self):
        return tf.TensorSpec(shape=tuple(self.input_shape))

    @property
    def output(self):
        return tf.TensorSpec(shape=tuple(self.output_shape))

class TFLite_conv2d(TFLite_layer):
    """2D Convolutional layer implementation for TFLite models.

    This class represents a 2D convolutional layer with support for:
    - Weight and bias parameters
    - ReLU activation
    - Batch normalization
    - Quantization parameters
    - Configurable stride and kernel size

    Attributes:
        weights (list): List containing weights and bias tensors
        store_in_psum (int): Flag for partial sum storage
        skip_psum (int): Flag for skipping partial sum computation
        filters (int): Number of output filters
        kernel_size (tuple): Size of convolution kernel (height, width)
        kernel (ndarray): Convolution kernel weights
        relu (bool): Whether ReLU activation is applied
        batchnorm (bool): Whether batch normalization is applied
        strides (tuple): Convolution stride in (height, width)
        quantization_factor (float): Scale factor for quantization
        zero_point (int): Zero point for quantization
    """
    weights = []
    store_in_psum = 0
    skip_psum = 0

    def __init__(self, idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp, relu=False, bn=False):
        """Initialize a 2D convolutional layer.

        Args:
            idx_in (int): Input tensor index
            idx_out (int): Output tensor index
            input_shape (tuple): Shape of input tensor
            output_shape (tuple): Shape of output tensor
            weights (ndarray): Convolution kernel weights
            bias (ndarray): Bias terms
            qf (float): Quantization scale factor
            zp (int): Quantization zero point
            relu (bool, optional): Apply ReLU activation. Defaults to False
            bn (bool, optional): Apply batch normalization. Defaults to False
        """
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
    """2D Max Pooling layer implementation for TFLite models.

    This class represents a 2D max pooling layer that downsamples the input
    by taking the maximum value in sliding windows.

    The layer maintains the dimensionality reduction information through its
    input and output shapes but doesn't require weights or additional parameters.
    """
    def __init__(self, idx_in, idx_out, input_shape, output_shape):
        """Initialize a 2D max pooling layer.

        Args:
            idx_in (int): Input tensor index
            idx_out (int): Output tensor index
            input_shape (tuple): Shape of input tensor
            output_shape (tuple): Shape of output tensor
        """
        super().__init__('max_pooling2d', idx_in, idx_out, input_shape, output_shape)

class TFLite_dense(TFLite_layer):
    """Dense (fully connected) layer implementation for TFLite models.

    This class represents a dense layer that performs a matrix multiplication
    with learnable weights and biases. It supports quantization for efficient
    computation on hardware.

    Attributes:
        weights (list): List containing weight matrix and bias vector
        qf (None): Default quantization factor
        quantization_factor (float): Scale factor for quantization
        zero_point (int): Zero point for quantization
    """
    weights = []
    qf = None

    def __init__(self, idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp):
        """Initialize a dense layer.

        Args:
            idx_in (int): Input tensor index
            idx_out (int): Output tensor index
            input_shape (tuple): Shape of input tensor
            output_shape (tuple): Shape of output tensor
            weights (ndarray): Weight matrix
            bias (ndarray): Bias vector
            qf (float): Quantization scale factor
            zp (int): Quantization zero point
        """
        super().__init__('dense', idx_in, idx_out, input_shape, output_shape)

        self.weights = [weights, bias]

        # quantization
        self.quantization_factor = qf
        self.zero_point = zp

class TFLite_model(object):
    """Container class for TensorFlow Lite models in OpenEye.

    This class manages a collection of neural network layers and provides
    methods to add different types of layers. It serves as the high-level
    representation of a complete neural network model.

    Attributes:
        layers (list): List of layer objects in the model
    """
    layers = []
    
    def __init__(self):
        """Initialize an empty TFLite model."""
        pass

    def add_conv2d(self, idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp, relu=None, bn=None):
        """Add a 2D convolutional layer to the model.

        Args:
            idx_in (int): Input tensor index
            idx_out (int): Output tensor index
            input_shape (tuple): Shape of input tensor
            output_shape (tuple): Shape of output tensor
            weights (ndarray): Convolution kernel weights
            bias (ndarray): Bias terms
            qf (float): Quantization scale factor
            zp (int): Quantization zero point
            relu (bool, optional): Apply ReLU activation
            bn (bool, optional): Apply batch normalization
        """
        layer = TFLite_conv2d(idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp, relu, bn)
        self.layers.append(layer)

    def add_max_pooling2d(self, idx_in, idx_out, input_shape, output_shape):
        """Add a 2D max pooling layer to the model.

        Args:
            idx_in (int): Input tensor index
            idx_out (int): Output tensor index
            input_shape (tuple): Shape of input tensor
            output_shape (tuple): Shape of output tensor
        """
        layer = TFLite_max_pooling2d(idx_in, idx_out, input_shape, output_shape)
        self.layers.append(layer)

    def add_dense(self, idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp):
        """Add a dense (fully connected) layer to the model.

        Args:
            idx_in (int): Input tensor index
            idx_out (int): Output tensor index
            input_shape (tuple): Shape of input tensor
            output_shape (tuple): Shape of output tensor
            weights (ndarray): Weight matrix
            bias (ndarray): Bias vector
            qf (float): Quantization scale factor
            zp (int): Quantization zero point
        """
        layer = TFLite_dense(idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp)
        self.layers.append(layer)

def quantize_scale(scale):
    """Calculate quantization parameters for a given scale factor.

    This function converts a floating-point scale factor into fixed-point
    representation suitable for hardware implementation. It decomposes the
    scale into a multiplier and shift value.

    Args:
        scale (float): Scale factor to quantize

    Returns:
        tuple: (q, shift) where:
            - q (int): Fixed-point multiplier
            - shift (int): Required bit shift

    Implementation Details:
        - Uses frexp to decompose float into mantissa and exponent
        - Handles special case of zero scale
        - Ensures no overflow in fixed-point representation
        - Adjusts for maximum precision while avoiding overflow
    """
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
    """Create an OpenEye model from a TensorFlow Lite model.

    This function takes a TFLite model file or a predefined architecture name
    and creates an OpenEye-compatible model representation. It supports both
    custom models and standard architectures like ResNet and MobileNet.

    Args:
        use_random (bool, optional): Use random weights instead of pretrained.
            Defaults to False.
        tflite_model_path (str, optional): Path to TFLite model file.
            If None, uses predefined architecture. Defaults to None.
        model_name (str, optional): Name of predefined architecture to use
            when tflite_model_path is None. Defaults to 'resnet'.

    Returns:
        TFLite_model: OpenEye model representation

    Features:
        - Direct TFLite model loading
        - Predefined architecture support (ResNet, MobileNet)
        - Weight quantization handling
        - Layer type conversion
        - Tensor shape tracking
        - Optional random weight initialization
        
    Supported Layer Types:
        - Conv2D
        - MaxPooling2D
        - Dense (Fully Connected)
        - Add, Multiply operations
        - Batch Normalization
    """
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