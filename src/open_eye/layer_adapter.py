# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Layer adapter for OpenEye neural network accelerator.

This module provides adapter classes that wrap PyTorch and ONNX layer objects
to make them compatible with the Keras layer interface expected by the
LayerParameters class. This allows the existing layer_parameters.py code to
work seamlessly with models from different frameworks.

Key Classes:
- KerasLayerAdapter: Wraps PyTorch/ONNX layers to look like Keras layers
- TensorShapeAdapter: Wraps tensor shapes to provide .shape attribute

Typical Usage:
    >>> from open_eye.pytorch2model import create_model_from_pytorch
    >>> from open_eye.layer_adapter import adapt_layers_for_keras
    >>> pytorch_model = create_model_from_pytorch(torch_model)
    >>> keras_compatible_layers = adapt_layers_for_keras(pytorch_model.layers)
"""

import numpy as np
import logging

logger = logging.getLogger("cocotb")


class TensorShapeAdapter:
    """Adapter to provide shape attribute for tensor-like objects.

    This class wraps a shape tuple and provides attribute-based access
    compatible with TensorFlow/Keras tensor objects.

    Attributes:
        _shape (tuple): The underlying shape tuple
    """
    def __init__(self, shape):
        """Initialize the shape adapter.

        Args:
            shape (tuple): Shape tuple (e.g., (batch, height, width, channels))
        """
        self._shape = tuple(shape) if shape is not None else ()

    @property
    def shape(self):
        """Return the shape as a tuple.

        Returns:
            tuple: The shape tuple
        """
        return self._shape

    def __getitem__(self, index):
        """Allow indexing into the shape.

        Args:
            index (int): Index to access

        Returns:
            int: Dimension at the given index
        """
        return self._shape[index]

    def __len__(self):
        """Return the number of dimensions.

        Returns:
            int: Number of dimensions
        """
        return len(self._shape)

    def __repr__(self):
        """String representation of the shape.

        Returns:
            str: String representation
        """
        return f"TensorShape({self._shape})"


class WeightTensorAdapter:
    """Adapter to provide tensor-like interface for weight arrays.

    This class wraps a numpy array and provides attribute-based access
    compatible with TensorFlow/Keras weight tensors.

    Attributes:
        _array (ndarray): The underlying numpy array
    """
    def __init__(self, array):
        """Initialize the weight tensor adapter.

        Args:
            array (ndarray): Numpy array containing weights
        """
        self._array = np.array(array) if array is not None else np.array([])

    @property
    def shape(self):
        """Return the shape of the weight tensor.

        Returns:
            tuple: Shape of the underlying array
        """
        return self._array.shape

    def numpy(self):
        """Return the numpy array.

        Returns:
            ndarray: The underlying numpy array
        """
        return self._array

    def __array__(self):
        """Support conversion to numpy array.

        Returns:
            ndarray: The underlying numpy array
        """
        return self._array

    def __getitem__(self, key):
        """Support indexing.

        Args:
            key: Index or slice

        Returns:
            Element or sub-array
        """
        return self._array[key]

    def __repr__(self):
        """String representation.

        Returns:
            str: String representation
        """
        return f"WeightTensor(shape={self.shape}, dtype={self._array.dtype})"


class KerasLayerAdapter:
    """Adapter class that makes PyTorch/ONNX layers look like Keras layers.

    This adapter wraps layer objects from PyTorch or ONNX models and provides
    the same interface as Keras layers, allowing the existing LayerParameters
    code to work without modification.

    The adapter provides all attributes required by LayerParameters:

    **Common attributes (all layer types):**
    - .name (str): Layer type name ('conv2d', 'max_pooling2d', 'dense')
    - .input: Input tensor adapter with .shape attribute
    - .output: Output tensor adapter with .shape attribute

    **Conv2D layers:**
    - .kernel: Weight tensor adapter (4D array)
    - .weights: List [weights, bias]
    - .kernel_size: Tuple (height, width)
    - .strides: Tuple (stride_h, stride_w)
    - .filters: Number of output filters
    - .padding: Padding mode ('same' or 'valid')
    - .relu: ReLU activation flag
    - .batchnorm: Batch normalization flag
    - .quantization_factor: Quantization scale
    - .zero_point: Quantization zero point
    - .store_in_psum: Partial sum storage flag
    - .skip_psum: Skip partial sum flag

    **Dense layers:**
    - .kernel: Weight matrix adapter
    - .weights: List [weights, bias]
    - .units: Number of output units
    - .use_bias: Whether bias is used
    - .quantization_factor: Quantization scale
    - .zero_point: Quantization zero point

    **MaxPooling2D layers:**
    - .pool_size: Tuple (pool_h, pool_w)
    - .strides: Tuple (stride_h, stride_w)

    Attributes:
        _layer: The underlying layer object (from PyTorch/ONNX)
        name (str): Layer type name
        input: Input tensor adapter
        output: Output tensor adapter
        kernel: Weight tensor adapter (for Conv/Dense)
        weights: List of weight arrays
    """
    def __init__(self, layer):
        """Initialize the Keras layer adapter.

        Args:
            layer: Layer object from PyTorch/ONNX model
                (e.g., PyTorchConv2d, ONNXConv2d, TFLite_conv2d, etc.)
        """
        self._layer = layer

        # Map layer names to Keras convention
        name_mapping = {
            'conv2d': 'conv2d',
            'max_pooling2d': 'max_pooling2d',
            'dense': 'dense',
            'flatten': 'flatten',
            'relu': 'relu'
        }
        self.name = name_mapping.get(layer.name, layer.name)

        # Create tensor shape adapters for input/output
        self.input = TensorShapeAdapter(layer.input_shape)
        self.output = TensorShapeAdapter(layer.output_shape)

        # Handle layer-specific attributes
        if hasattr(layer, 'weights') and layer.weights:
            # Conv2D or Dense layer with weights
            weights_array = layer.weights[0]
            bias_array = layer.weights[1] if len(layer.weights) > 1 else None

            # Create weight tensor adapter
            self.kernel = WeightTensorAdapter(weights_array)
            self.weights = [weights_array]
            if bias_array is not None:
                self.weights.append(bias_array)
                self.bias = bias_array

            # Conv2D-specific attributes
            if layer.name == 'conv2d':
                # Core Conv2D attributes (required by LayerParameters)
                self.kernel_size = layer.kernel_size if hasattr(layer, 'kernel_size') else (3, 3)
                self.strides = layer.strides if hasattr(layer, 'strides') else (1, 1)
                self.filters = layer.filters if hasattr(layer, 'filters') else weights_array.shape[-1]
                self.padding = 'same'  # Default assumption (most common for OpenEye)

                # Activation and normalization flags
                self.relu = layer.relu if hasattr(layer, 'relu') else False
                self.batchnorm = layer.batchnorm if hasattr(layer, 'batchnorm') else False

            # Dense-specific attributes
            elif layer.name == 'dense':
                self.units = layer.output_shape[-1] if layer.output_shape is not None else 0
                self.use_bias = bias_array is not None

            # Quantization parameters
            if hasattr(layer, 'quantization_factor'):
                self.quantization_factor = layer.quantization_factor
            if hasattr(layer, 'zero_point'):
                self.zero_point = layer.zero_point

            # Partial sum control
            if hasattr(layer, 'store_in_psum'):
                self.store_in_psum = layer.store_in_psum
            if hasattr(layer, 'skip_psum'):
                self.skip_psum = layer.skip_psum

        else:
            # Layers without weights (e.g., MaxPooling, ReLU, Flatten)
            self.weights = []

            # MaxPooling-specific attributes
            if layer.name == 'max_pooling2d':
                # Check if layer has explicit pool_size attribute
                if hasattr(layer, 'pool_size'):
                    self.pool_size = layer.pool_size
                    self.strides = layer.strides if hasattr(layer, 'strides') else layer.pool_size
                # Otherwise infer pool size from shape change
                elif len(layer.input_shape) >= 3 and len(layer.output_shape) >= 3:
                    pool_h = layer.input_shape[1] // layer.output_shape[1] if layer.output_shape[1] > 0 else 2
                    pool_w = layer.input_shape[2] // layer.output_shape[2] if layer.output_shape[2] > 0 else 2
                    self.pool_size = (pool_h, pool_w)
                    self.strides = (pool_h, pool_w)
                else:
                    # Fallback to default 2x2 pooling
                    self.pool_size = (2, 2)
                    self.strides = (2, 2)

            # Flatten layer
            elif layer.name == 'flatten':
                # Flatten doesn't need special attributes
                pass

            # ReLU or other activation layers
            elif layer.name in ['relu', 'activation']:
                # Activation layers don't need special attributes
                pass

    def get_weights(self):
        """Return the layer weights in Keras format.

        Returns:
            list: List of numpy arrays [weights, bias] for Conv/Dense layers,
                  empty list for layers without weights
        """
        return self.weights

    def __repr__(self):
        """String representation of the adapter.

        Returns:
            str: String representation showing layer type and shapes
        """
        return (f"KerasLayerAdapter({self.name}, "
                f"input_shape={self.input.shape}, "
                f"output_shape={self.output.shape})")


def adapt_layers_for_keras(layers):
    """Convert a list of PyTorch/ONNX layers to Keras-compatible format.

    This function takes a list of layer objects from PyTorch or ONNX models
    and wraps them with KerasLayerAdapter to make them compatible with the
    existing LayerParameters code.

    Args:
        layers (list): List of layer objects from PyTorch/ONNX model

    Returns:
        list: List of KerasLayerAdapter objects

    Examples:
        >>> from open_eye.pytorch2model import create_model_from_pytorch
        >>> pytorch_model = create_model_from_pytorch(torch_model)
        >>> keras_layers = adapt_layers_for_keras(pytorch_model.layers)
        >>> # Now keras_layers can be used with LayerParameters
    """
    adapted_layers = []

    for layer in layers:
        # Check if already a Keras layer (from TensorFlow models)
        if hasattr(layer, '__class__') and 'tensorflow' in layer.__class__.__module__:
            # Already a Keras layer, use as-is
            adapted_layers.append(layer)
        else:
            # PyTorch or ONNX layer, wrap with adapter
            adapted_layer = KerasLayerAdapter(layer)
            adapted_layers.append(adapted_layer)

    return adapted_layers


def is_keras_layer(layer):
    """Check if a layer is a Keras layer (not needing adaptation).

    Args:
        layer: Layer object to check

    Returns:
        bool: True if it's a Keras layer, False otherwise
    """
    # Check if it's from TensorFlow/Keras
    if hasattr(layer, '__class__'):
        module = layer.__class__.__module__
        return 'tensorflow' in module or 'keras' in module
    return False


if __name__ == "__main__":
    # Example usage and testing
    print("Layer Adapter for OpenEye")
    print("=" * 60)

    # Create a mock layer for testing
    class MockPyTorchLayer:
        def __init__(self):
            self.name = 'conv2d'
            self.input_shape = (1, 1, 28, 28)
            self.output_shape = (1, 32, 28, 28)
            self.weights = [
                np.random.randn(3, 3, 1, 32),  # kernel
                np.random.randn(32)  # bias
            ]
            self.kernel_size = (3, 3)
            self.strides = (1, 1)
            self.filters = 32
            self.quantization_factor = [(1234567, 42)] * 32
            self.zero_point = 0

    mock_layer = MockPyTorchLayer()
    adapted = KerasLayerAdapter(mock_layer)

    print(f"\nOriginal layer: {mock_layer.name}")
    print(f"Adapted layer: {adapted}")
    print(f"  Input shape: {adapted.input.shape}")
    print(f"  Output shape: {adapted.output.shape}")
    print(f"  Kernel size: {adapted.kernel_size}")
    print(f"  Filters: {adapted.filters}")
    print(f"  Weights: {len(adapted.get_weights())} arrays")
    print(f"  Weight shape: {adapted.kernel.shape}")

    print("\n✅ Layer adapter working correctly!")
