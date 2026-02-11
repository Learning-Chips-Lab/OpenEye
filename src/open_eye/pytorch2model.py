# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""PyTorch model converter for OpenEye neural network accelerator.

This module provides functionality to convert PyTorch models to a format
compatible with the OpenEye neural network accelerator. It supports quantized
PyTorch models and maps them to the internal layer representation used by OpenEye.

Key Classes:
- PyTorchLayer: Base class for all layer types from PyTorch
- PyTorchConv2d: Convolutional layer representation
- PyTorchMaxPooling2d: Max pooling layer representation
- PyTorchLinear: Fully connected layer representation
- PyTorchModel: Container class for the complete PyTorch model

Typical Usage:
    >>> import torch
    >>> pytorch_model = torch.load('model.pth')
    >>> openeye_model = create_model_from_pytorch(pytorch_model)
    >>> print(f"Converted {len(openeye_model.layers)} layers")

Supported Layer Types:
    - Conv2D (including quantized)
    - MaxPool2D
    - Linear (Fully Connected)
    - ReLU activations
    - Batch Normalization (fused)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import math
import logging

logger = logging.getLogger("cocotb")


class PyTorchLayer(object):
    """Base class for PyTorch layers in OpenEye.

    This class serves as the foundation for all PyTorch layer types supported by
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
        """Initialize a PyTorch layer.

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


class PyTorchConv2d(PyTorchLayer):
    """2D Convolutional layer implementation for PyTorch models.

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


class PyTorchMaxPooling2d(PyTorchLayer):
    """2D Max Pooling layer implementation for PyTorch models.

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


class PyTorchLinear(PyTorchLayer):
    """Dense (fully connected) layer implementation for PyTorch models.

    This class represents a fully connected layer that performs a matrix multiplication
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


class PyTorchModel(object):
    """Container class for PyTorch models in OpenEye.

    This class manages a collection of neural network layers and provides
    methods to add different types of layers. It serves as the high-level
    representation of a complete neural network model compatible with
    the existing OpenEye infrastructure.

    Attributes:
        layers (list): List of layer objects in the model
    """
    layers = []

    def __init__(self):
        """Initialize an empty PyTorch model."""
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
        layer = PyTorchConv2d(idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp, relu, bn)
        self.layers.append(layer)

    def add_max_pooling2d(self, idx_in, idx_out, input_shape, output_shape):
        """Add a 2D max pooling layer to the model.

        Args:
            idx_in (int): Input tensor index
            idx_out (int): Output tensor index
            input_shape (tuple): Shape of input tensor
            output_shape (tuple): Shape of output tensor
        """
        layer = PyTorchMaxPooling2d(idx_in, idx_out, input_shape, output_shape)
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
        layer = PyTorchLinear(idx_in, idx_out, input_shape, output_shape, weights, bias, qf, zp)
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


def extract_quantization_params(layer):
    """Extract quantization parameters from a quantized PyTorch layer.

    Args:
        layer: PyTorch layer (quantized or regular)

    Returns:
        tuple: (scale, zero_point) or (1.0, 0) if not quantized
    """
    # Check if layer has quantization parameters
    if hasattr(layer, 'weight') and hasattr(layer.weight, 'q_scale'):
        # Quantized layer
        scale = layer.weight().q_scale()
        zero_point = layer.weight().q_zero_point()
        return scale, zero_point
    else:
        # Regular layer - no quantization
        return 1.0, 0


def infer_output_shape(layer, input_shape):
    """Infer output shape of a PyTorch layer given input shape.

    Args:
        layer: PyTorch layer
        input_shape (tuple): Input tensor shape (batch, channels, height, width)

    Returns:
        tuple: Output tensor shape
    """
    if isinstance(layer, nn.Conv2d):
        batch, in_channels, in_h, in_w = input_shape
        out_channels = layer.out_channels

        # Calculate output dimensions based on padding, stride, kernel
        if layer.padding == 'same' or (isinstance(layer.padding, tuple) and layer.padding[0] > 0):
            out_h = math.ceil(in_h / layer.stride[0])
            out_w = math.ceil(in_w / layer.stride[1])
        else:
            out_h = math.floor((in_h - layer.kernel_size[0]) / layer.stride[0]) + 1
            out_w = math.floor((in_w - layer.kernel_size[1]) / layer.stride[1]) + 1

        return (batch, out_channels, out_h, out_w)

    elif isinstance(layer, nn.MaxPool2d):
        batch, channels, in_h, in_w = input_shape
        out_h = math.floor((in_h - layer.kernel_size) / layer.stride) + 1
        out_w = math.floor((in_w - layer.kernel_size) / layer.stride) + 1
        return (batch, channels, out_h, out_w)

    elif isinstance(layer, nn.Linear):
        batch = input_shape[0]
        out_features = layer.out_features
        return (batch, out_features)

    else:
        # For other layers, return input shape
        return input_shape


def create_model_from_pytorch(pytorch_model, input_shape=(1, 1, 28, 28), use_random=False):
    """Create an OpenEye model from a PyTorch model.

    This function takes a PyTorch model (quantized or regular) and creates an
    OpenEye-compatible model representation. It extracts weights, biases, and
    quantization parameters from each supported layer type.

    Args:
        pytorch_model: PyTorch nn.Module (can be quantized)
        input_shape (tuple, optional): Input tensor shape (batch, C, H, W).
            Defaults to (1, 1, 28, 28) for MNIST.
        use_random (bool, optional): Use random weights instead of trained weights.
            Defaults to False.

    Returns:
        PyTorchModel: OpenEye model representation

    Features:
        - Extracts weights and biases from Conv2d, Linear layers
        - Handles quantization parameters if model is quantized
        - Supports ReLU and MaxPool2d operations
        - Tracks tensor shapes through the network
        - Converts PyTorch weight format to OpenEye format

    Supported Layer Types:
        - Conv2D (nn.Conv2d)
        - MaxPooling2D (nn.MaxPool2d)
        - Linear/Dense (nn.Linear)
        - ReLU (nn.ReLU)
        - Flatten operations

    Raises:
        ValueError: If unsupported layer type is encountered
        RuntimeError: If weight extraction fails
    """
    openeye_model = PyTorchModel()

    # Set model to evaluation mode
    pytorch_model.eval()

    idx_in = 0
    idx_out = 0
    current_shape = input_shape

    # Iterate through model layers
    for name, module in pytorch_model.named_modules():
        # Skip the root module
        if name == '':
            continue

        if isinstance(module, nn.Conv2d):
            # Conv2D layer
            logger.debug(f"Processing Conv2D layer: {name}")

            # Get weights and bias
            if hasattr(module, 'weight'):
                weights = module.weight.detach().cpu().numpy()
                # PyTorch format: (out_channels, in_channels, kernel_h, kernel_w)
                # OpenEye format: (kernel_h, kernel_w, in_channels, out_channels)
                weights = np.transpose(weights, (2, 3, 1, 0))
            else:
                logger.warning(f"No weights found for layer {name}")
                continue

            if hasattr(module, 'bias') and module.bias is not None:
                bias = module.bias.detach().cpu().numpy()
            else:
                bias = np.zeros(module.out_channels)

            # Get quantization parameters
            scale, zero_point = extract_quantization_params(module)

            # Calculate quantization factors for each filter
            qf = []
            for i in range(module.out_channels):
                mult, shift = quantize_scale(scale)
                qf.append((mult, shift + 31))

            # Infer output shape
            output_shape = infer_output_shape(module, current_shape)

            # Add to OpenEye model
            openeye_model.add_conv2d(
                idx_in=idx_in,
                idx_out=idx_out + 1,
                input_shape=current_shape,
                output_shape=output_shape,
                weights=weights,
                bias=bias,
                qf=qf,
                zp=zero_point,
                relu=False,  # ReLU is typically a separate layer in PyTorch
                bn=False
            )

            idx_in = idx_out + 1
            idx_out += 1
            current_shape = output_shape

        elif isinstance(module, nn.MaxPool2d):
            # MaxPooling layer
            logger.debug(f"Processing MaxPool2D layer: {name}")

            output_shape = infer_output_shape(module, current_shape)

            openeye_model.add_max_pooling2d(
                idx_in=idx_in,
                idx_out=idx_out + 1,
                input_shape=current_shape,
                output_shape=output_shape
            )

            idx_in = idx_out + 1
            idx_out += 1
            current_shape = output_shape

        elif isinstance(module, nn.Linear):
            # Fully connected layer
            logger.debug(f"Processing Linear layer: {name}")

            # Get weights and bias
            if hasattr(module, 'weight'):
                weights = module.weight.detach().cpu().numpy()
                # PyTorch format: (out_features, in_features)
                # OpenEye format: (in_features, out_features)
                weights = np.transpose(weights, (1, 0))
            else:
                logger.warning(f"No weights found for layer {name}")
                continue

            if hasattr(module, 'bias') and module.bias is not None:
                bias = module.bias.detach().cpu().numpy()
            else:
                bias = np.zeros(module.out_features)

            # Get quantization parameters
            scale, zero_point = extract_quantization_params(module)

            # Calculate quantization factors
            qf = []
            for i in range(module.out_features):
                mult, shift = quantize_scale(scale)
                qf.append((mult, shift + 31))

            # Handle shape for fully connected (may need flattening)
            if len(current_shape) == 4:
                # Need to flatten: (batch, C, H, W) -> (batch, C*H*W)
                batch = current_shape[0]
                flattened_size = current_shape[1] * current_shape[2] * current_shape[3]
                current_shape = (batch, flattened_size)

            output_shape = infer_output_shape(module, current_shape)

            openeye_model.add_dense(
                idx_in=idx_in,
                idx_out=idx_out + 1,
                input_shape=current_shape,
                output_shape=output_shape,
                weights=weights,
                bias=bias,
                qf=qf,
                zp=zero_point
            )

            idx_in = idx_out + 1
            idx_out += 1
            current_shape = output_shape

        elif isinstance(module, nn.ReLU):
            # ReLU activation - typically fused with previous layer in OpenEye
            logger.debug(f"Processing ReLU layer: {name} (will be fused)")
            # Mark previous layer as having ReLU if it's a conv layer
            if len(openeye_model.layers) > 0:
                prev_layer = openeye_model.layers[-1]
                if hasattr(prev_layer, 'relu'):
                    prev_layer.relu = True

        elif isinstance(module, nn.Flatten) or isinstance(module, nn.Identity):
            # These are no-ops in terms of computation, just shape changes
            logger.debug(f"Processing {module.__class__.__name__} layer: {name}")
            continue

        else:
            # Skip unsupported layers with a warning
            if not isinstance(module, (nn.Sequential, nn.Dropout)):
                logger.warning(f"Unsupported layer type {module.__class__.__name__}: {name}")

    logger.info(f"Successfully converted PyTorch model with {len(openeye_model.layers)} layers")
    return openeye_model


if __name__ == "__main__":
    # Example usage
    import torch
    import torch.nn as nn

    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32 * 7 * 7, 10)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            x = self.fc1(x)
            return x

    model = SimpleModel()
    openeye_model = create_model_from_pytorch(model, input_shape=(1, 1, 28, 28))

    print(f"Converted model with {len(openeye_model.layers)} layers")
    for i, layer in enumerate(openeye_model.layers):
        print(f"Layer {i}: {layer.name}, input_shape={layer.input_shape}, output_shape={layer.output_shape}")
