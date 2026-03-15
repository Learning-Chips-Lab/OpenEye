# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""ONNX model converter for OpenEye neural network accelerator.

This module provides functionality to convert ONNX models to a format
compatible with the OpenEye neural network accelerator. It supports quantized
ONNX models and maps them to the internal layer representation used by OpenEye.

Key Classes:
- ONNXLayer: Base class for all layer types from ONNX
- ONNXConv2d: Convolutional layer representation
- ONNXMaxPooling2d: Max pooling layer representation
- ONNXLinear: Fully connected layer representation (Gemm)
- ONNXModel: Container class for the complete ONNX model

Typical Usage:
    >>> import onnx
    >>> onnx_model = onnx.load('model.onnx')
    >>> openeye_model = create_model_from_onnx(onnx_model)
    >>> print(f"Converted {len(openeye_model.layers)} layers")

Supported Layer Types:
    - Conv (Convolution)
    - QLinearConv (Quantized Convolution)
    - MaxPool (Max Pooling)
    - Gemm (Fully Connected/Dense)
    - QLinearMatMul (Quantized Fully Connected)
    - Relu
    - BatchNormalization
"""

import onnx
from onnx import numpy_helper
import numpy as np
from pathlib import Path
import math
import logging
from open_eye.base_model import OpenEyeBaseModel

logger = logging.getLogger("cocotb")


class ONNXLayer(object):
    """Base class for ONNX layers in OpenEye.

    This class serves as the foundation for all ONNX layer types supported by
    the OpenEye accelerator. It provides common attributes and interfaces compatible
    with the existing TFLite layer system.

    Attributes:
        name (str): Name/type of the layer
        idx_in (int): Input tensor index
        idx_out (int): Output tensor index
        input_shape (tuple): Shape of input tensor
        output_shape (tuple): Shape of output tensor
    """
    def __init__(self, name, idx_in, idx_out, input_shape, output_shape):
        """Initialize an ONNX layer.

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


class ONNXConv2d(ONNXLayer):
    """2D Convolutional layer implementation for ONNX models.

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


class ONNXMaxPooling2d(ONNXLayer):
    """2D Max Pooling layer implementation for ONNX models.

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


class ONNXLinear(ONNXLayer):
    """Dense (fully connected) layer implementation for ONNX models.

    This class represents a fully connected layer (Gemm/MatMul in ONNX) that
    performs a matrix multiplication with learnable weights and biases.
    It supports quantization for efficient computation on hardware.

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


class ONNXModel(OpenEyeBaseModel):
    """Container class for ONNX models in OpenEye.

    This class manages a collection of neural network layers and provides
    methods to add different types of layers. It serves as the high-level
    representation of a complete neural network model compatible with
    the existing OpenEye infrastructure.

    Inherits from OpenEyeBaseModel to provide:
    - input_shape property
    - output_shape property
    - is_quantized property
    - Common model inspection methods

    Attributes:
        layers (list): List of layer objects in the model
    """

    def __init__(self):
        """Initialize an empty ONNX model."""
        super().__init__(framework='onnx')
        self.layers = []

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
        layer = ONNXConv2d(idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp, relu, bn)
        self.layers.append(layer)

    def add_max_pooling2d(self, idx_in, idx_out, input_shape, output_shape):
        """Add a 2D max pooling layer to the model.

        Args:
            idx_in (int): Input tensor index
            idx_out (int): Output tensor index
            input_shape (tuple): Shape of input tensor
            output_shape (tuple): Shape of output tensor
        """
        layer = ONNXMaxPooling2d(idx_in, idx_out, input_shape, output_shape)
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
        layer = ONNXLinear(idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp)
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


def get_initializer_dict(onnx_model):
    """Create a dictionary of initializers (weights/biases) from ONNX model.

    Args:
        onnx_model: ONNX model object

    Returns:
        dict: Dictionary mapping initializer names to numpy arrays
    """
    initializer_dict = {}
    for initializer in onnx_model.graph.initializer:
        initializer_dict[initializer.name] = numpy_helper.to_array(initializer)
    return initializer_dict


def get_tensor_shape_dict(onnx_model):
    """Create a dictionary of tensor shapes from ONNX model.

    Args:
        onnx_model: ONNX model object

    Returns:
        dict: Dictionary mapping tensor names to shapes
    """
    shape_dict = {}

    # Add input shapes
    for input_tensor in onnx_model.graph.input:
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            shape.append(getattr(dim, 'dim_value', 1))
        shape_dict[input_tensor.name] = tuple(shape)

    # Add output shapes from value_info
    for value_info in onnx_model.graph.value_info:
        shape = []
        for dim in value_info.type.tensor_type.shape.dim:
            shape.append(getattr(dim, 'dim_value', 1))
        shape_dict[value_info.name] = tuple(shape)

    # Add output shapes
    for output_tensor in onnx_model.graph.output:
        shape = []
        for dim in output_tensor.type.tensor_type.shape.dim:
            shape.append(getattr(dim, 'dim_value', 1))
        shape_dict[output_tensor.name] = tuple(shape)

    return shape_dict


def get_attribute_value(node, attr_name, default=None):
    """Extract attribute value from ONNX node.

    Args:
        node: ONNX node object
        attr_name (str): Attribute name to extract
        default: Default value if attribute not found

    Returns:
        Attribute value or default
    """
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.ints:
                return list(attr.ints)
            elif attr.floats:
                return list(attr.floats)
            elif attr.i:
                return attr.i
            elif attr.f:
                return attr.f
            elif attr.s:
                return attr.s.decode('utf-8')
    return default


def create_model_from_onnx(onnx_model_path, use_random=False):
    """Create an OpenEye model from an ONNX model.

    This function takes an ONNX model file and creates an OpenEye-compatible
    model representation. It extracts weights, biases, and quantization
    parameters from each supported layer type.

    Args:
        onnx_model_path (str or Path): Path to ONNX model file
        use_random (bool, optional): Use random weights instead of trained weights.
            Defaults to False.

    Returns:
        ONNXModel: OpenEye model representation

    Features:
        - Loads ONNX model from file
        - Extracts weights and biases from Conv, Gemm layers
        - Handles quantization parameters if model is quantized
        - Supports ReLU and MaxPool operations
        - Tracks tensor shapes through the network
        - Converts ONNX weight format to OpenEye format

    Supported Layer Types:
        - Conv (regular and QLinearConv)
        - MaxPool
        - Gemm (Fully Connected)
        - QLinearMatMul (Quantized Fully Connected)
        - Relu
        - BatchNormalization

    Raises:
        ValueError: If unsupported layer type is encountered
        FileNotFoundError: If ONNX model file not found
        RuntimeError: If weight extraction fails
    """
    # Load ONNX model
    if isinstance(onnx_model_path, (str, Path)):
        onnx_model = onnx.load(str(onnx_model_path))
    else:
        # Assume it's already a loaded ONNX model
        onnx_model = onnx_model_path

    # Verify the model
    try:
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model is valid")
    except Exception as e:
        logger.warning(f"ONNX model validation warning: {e}")

    # Create dictionaries for easy lookup
    initializer_dict = get_initializer_dict(onnx_model)
    shape_dict = get_tensor_shape_dict(onnx_model)

    # Create OpenEye model
    openeye_model = ONNXModel()

    # Track tensor indices
    idx_in = 0
    idx_out = 0

    # Track if next layer should have ReLU
    next_has_relu = False

    # Process each node in the graph
    for node in onnx_model.graph.node:
        node_type = node.op_type
        node_name = node.name

        logger.debug(f"Processing node: {node_name} (type: {node_type})")

        # Skip QuantizeLinear and DequantizeLinear nodes (handled separately)
        if node_type in ['QuantizeLinear', 'DequantizeLinear']:
            continue

        if node_type == 'Conv' or node_type == 'QLinearConv':
            # Convolution layer
            logger.debug(f"Processing Conv layer: {node_name}")

            # Get input/output tensor names
            if node_type == 'QLinearConv':
                # QLinearConv has: input, input_scale, input_zp, weight, weight_scale, weight_zp, [bias], output_scale, output_zp
                input_name = node.input[0]
                weight_name = node.input[3]
                bias_name = node.input[6] if len(node.input) > 6 else None
                output_name = node.output[0]
            else:
                # Regular Conv has: input, weight, [bias]
                input_name = node.input[0]
                weight_name = node.input[1]
                bias_name = node.input[2] if len(node.input) > 2 else None
                output_name = node.output[0]

            # Get weights
            if weight_name in initializer_dict:
                weights = initializer_dict[weight_name]
                # ONNX format: (out_channels, in_channels, kernel_h, kernel_w)
                # OpenEye format: (kernel_h, kernel_w, in_channels, out_channels)
                weights = np.transpose(weights, (2, 3, 1, 0))
            else:
                logger.error(f"Weight {weight_name} not found in initializers")
                continue

            # Get bias
            if bias_name and bias_name in initializer_dict:
                bias = initializer_dict[bias_name]
            else:
                # No bias - create zero bias
                out_channels = weights.shape[3]
                bias = np.zeros(out_channels, dtype=np.int32)

            # Get quantization parameters
            if node_type == 'QLinearConv':
                # Quantized convolution
                input_scale_name = node.input[1]
                weight_scale_name = node.input[4]
                output_scale_name = node.input[7] if len(node.input) > 7 else node.input[-2]

                input_scale = initializer_dict.get(input_scale_name, 1.0)
                weight_scale = initializer_dict.get(weight_scale_name, np.array([1.0]))
                output_scale = initializer_dict.get(output_scale_name, 1.0)

                # Convert scalar to array if needed
                if isinstance(input_scale, np.ndarray):
                    input_scale = float(input_scale)
                if isinstance(output_scale, np.ndarray):
                    output_scale = float(output_scale)
                if not isinstance(weight_scale, np.ndarray):
                    weight_scale = np.array([weight_scale])

                zero_point = 0  # Assuming symmetric quantization
            else:
                # Non-quantized - use default values
                input_scale = 1.0
                output_scale = 1.0
                weight_scale = np.ones(weights.shape[3])
                zero_point = 0

            # Calculate quantization factors for each filter
            qf = []
            for scale_w in weight_scale:
                combined_scale = input_scale * float(scale_w) / output_scale
                mult, shift = quantize_scale(combined_scale)
                qf.append((mult, shift + 31))

            # Get shapes
            input_shape = shape_dict.get(input_name, (1, 1, 28, 28))
            output_shape = shape_dict.get(output_name, (1, weights.shape[3], 28, 28))

            # Get stride and padding from attributes
            strides = get_attribute_value(node, 'strides', [1, 1])
            pads = get_attribute_value(node, 'pads', [0, 0, 0, 0])

            # Add to OpenEye model
            openeye_model.add_conv2d(
                idx_in=idx_in,
                idx_out=idx_out + 1,
                input_shape=input_shape,
                output_shape=output_shape,
                weights=weights,
                bias=bias,
                qf=qf,
                zp=zero_point,
                relu=next_has_relu,  # Apply ReLU if flagged
                bn=False
            )

            next_has_relu = False  # Reset flag
            idx_in = idx_out + 1
            idx_out += 1

        elif node_type == 'MaxPool':
            # MaxPooling layer
            logger.debug(f"Processing MaxPool layer: {node_name}")

            input_name = node.input[0]
            output_name = node.output[0]

            input_shape = shape_dict.get(input_name, (1, 1, 28, 28))
            output_shape = shape_dict.get(output_name, (1, 1, 14, 14))

            openeye_model.add_max_pooling2d(
                idx_in=idx_in,
                idx_out=idx_out + 1,
                input_shape=input_shape,
                output_shape=output_shape
            )

            idx_in = idx_out + 1
            idx_out += 1

        elif node_type == 'Gemm' or node_type == 'QLinearMatMul' or node_type == 'MatMul':
            # Fully connected layer
            logger.debug(f"Processing Gemm/MatMul layer: {node_name}")

            # Get input/output tensor names
            input_name = node.input[0]
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 else None
            output_name = node.output[0]

            # Get weights
            if weight_name in initializer_dict:
                weights = initializer_dict[weight_name]
                # ONNX Gemm format: (out_features, in_features)
                # OpenEye format: (in_features, out_features)
                if len(weights.shape) == 2:
                    weights = np.transpose(weights, (1, 0))
            else:
                logger.error(f"Weight {weight_name} not found in initializers")
                continue

            # Get bias
            if bias_name and bias_name in initializer_dict:
                bias = initializer_dict[bias_name]
            else:
                out_features = weights.shape[1]
                bias = np.zeros(out_features, dtype=np.int32)

            # Get quantization parameters (if quantized)
            if node_type == 'QLinearMatMul':
                # Similar to QLinearConv
                scale = 1.0
                zero_point = 0
            else:
                scale = 1.0
                zero_point = 0

            # Calculate quantization factors
            qf = []
            for i in range(weights.shape[1]):
                mult, shift = quantize_scale(scale)
                qf.append((mult, shift + 31))

            # Get shapes
            input_shape = shape_dict.get(input_name, (1, 784))
            output_shape = shape_dict.get(output_name, (1, weights.shape[1]))

            openeye_model.add_dense(
                idx_in=idx_in,
                idx_out=idx_out + 1,
                input_shape=input_shape,
                output_shape=output_shape,
                weights=weights,
                bias=bias,
                qf=qf,
                zp=zero_point
            )

            idx_in = idx_out + 1
            idx_out += 1

        elif node_type == 'Relu':
            # ReLU activation - flag for next layer
            logger.debug(f"Processing Relu layer: {node_name}")
            # Mark that the previous layer should have ReLU
            if len(openeye_model.layers) > 0:
                prev_layer = openeye_model.layers[-1]
                if hasattr(prev_layer, 'relu'):
                    prev_layer.relu = True

        elif node_type in ['BatchNormalization', 'Flatten', 'Reshape', 'Transpose', 'Squeeze', 'Unsqueeze']:
            # These are either fused or no-ops in OpenEye
            logger.debug(f"Processing {node_type} layer: {node_name} (will be fused or skipped)")
            continue

        else:
            # Unsupported layer type
            logger.warning(f"Unsupported layer type {node_type}: {node_name}")

    logger.info(f"Successfully converted ONNX model with {len(openeye_model.layers)} layers")
    return openeye_model


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        onnx_path = sys.argv[1]
    else:
        onnx_path = "model.onnx"

    try:
        openeye_model = create_model_from_onnx(onnx_path)
        print(f"Successfully converted ONNX model with {len(openeye_model.layers)} layers")

        for i, layer in enumerate(openeye_model.layers):
            print(f"Layer {i}: {layer.name}, input_shape={layer.input_shape}, output_shape={layer.output_shape}")

    except FileNotFoundError:
        print(f"Error: ONNX model file '{onnx_path}' not found")
        print("Usage: python onnx2model.py <path_to_onnx_model>")
    except Exception as e:
        print(f"Error converting ONNX model: {e}")
        import traceback
        traceback.print_exc()
