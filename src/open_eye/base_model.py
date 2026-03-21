# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Base model class for OpenEye multi-framework support.

This module provides a unified base class that all framework-specific model
classes (TFLite, ONNX, PyTorch, Keras) inherit from. This ensures a consistent
interface across all supported frameworks.

The base class provides:
- Common attributes (input_shape, output_shape, is_quantized)
- Unified interface for model inspection
- Helper methods for layer analysis
- Compatibility with existing OpenEye code

Key Classes:
- OpenEyeBaseModel: Abstract base class for all model formats
- ModelMetadata: Container for model metadata and properties

Typical Usage:
    All framework-specific model classes inherit from OpenEyeBaseModel:

    >>> class TFLite_model(OpenEyeBaseModel):
    ...     def __init__(self):
    ...         super().__init__(framework='tflite')
    ...
    >>> model = TFLite_model()
    >>> print(model.input_shape)  # Works for all frameworks
"""

from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger("cocotb")


class ModelMetadata:
    """Container for model metadata and properties.

    This class stores common metadata that applies to all model formats,
    making it easy to inspect and compare models from different frameworks.

    Attributes:
        framework (str): Source framework ('tflite', 'onnx', 'pytorch', 'keras')
        version (str): Framework version if available
        quantized (bool): Whether model is quantized
        quantization_type (str): Type of quantization (e.g., 'int8', 'int16')
        input_dtype (str): Input data type
        output_dtype (str): Output data type
        model_size_bytes (int): Approximate model size in bytes
        param_count (int): Total number of parameters
    """

    def __init__(self):
        """Initialize empty metadata."""
        self.framework = "unknown"
        self.version = None
        self.quantized = False
        self.quantization_type = None
        self.input_dtype = "float32"
        self.output_dtype = "float32"
        self.model_size_bytes = 0
        self.param_count = 0

    def __repr__(self):
        """String representation of metadata."""
        return (f"ModelMetadata(framework={self.framework}, "
                f"quantized={self.quantized}, "
                f"params={self.param_count:,})")


class OpenEyeBaseModel(ABC):
    """Abstract base class for all OpenEye model formats.

    This class provides a unified interface for models from different frameworks
    (TensorFlow Lite, ONNX, PyTorch, Keras). All framework-specific model classes
    should inherit from this class to ensure compatibility.

    The base class ensures that all models have:
    - A layers attribute (list of layer objects)
    - Input/output shape properties
    - Quantization status information
    - Framework metadata
    - Common utility methods

    Attributes:
        layers (list): List of layer objects
        metadata (ModelMetadata): Model metadata and properties
        _input_shape (tuple): Cached input shape
        _output_shape (tuple): Cached output shape
        _is_quantized (bool): Cached quantization status

    Example:
        >>> class MyFrameworkModel(OpenEyeBaseModel):
        ...     def __init__(self):
        ...         super().__init__(framework='myframework')
        ...         self.layers = []
        ...
        >>> model = MyFrameworkModel()
        >>> model.input_shape  # (1, 3, 224, 224)
    """

    def __init__(self, framework: str = "unknown"):
        """Initialize the base model.

        Args:
            framework: Name of the source framework
        """
        self.layers = []
        self.metadata = ModelMetadata()
        self.metadata.framework = framework

        # Cached properties
        self._input_shape = None
        self._output_shape = None
        self._is_quantized = None

    @property
    def input_shape(self) -> Optional[Tuple[int, ...]]:
        """Get the input shape of the model.

        Returns the shape of the first layer's input, or None if no layers.
        The shape is typically (batch, channels, height, width) for CNNs.

        Returns:
            Tuple of integers representing input dimensions, or None

        Example:
            >>> model.input_shape
            (1, 1, 28, 28)  # MNIST input
        """
        if self._input_shape is not None:
            return self._input_shape

        if not self.layers:
            return None

        first_layer = self.layers[0]
        if hasattr(first_layer, 'input_shape'):
            self._input_shape = tuple(first_layer.input_shape)
            return self._input_shape
        elif hasattr(first_layer, 'input'):
            # For TensorFlow-style layers
            self._input_shape = tuple(first_layer.input.shape)
            return self._input_shape

        return None

    @input_shape.setter
    def input_shape(self, shape: Tuple[int, ...]):
        """Set the input shape explicitly.

        Args:
            shape: Tuple of integers representing input dimensions
        """
        self._input_shape = tuple(shape) if shape else None

    @property
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        """Get the output shape of the model.

        Returns the shape of the last layer's output, or None if no layers.

        Returns:
            Tuple of integers representing output dimensions, or None

        Example:
            >>> model.output_shape
            (1, 10)  # 10 class probabilities
        """
        if self._output_shape is not None:
            return self._output_shape

        if not self.layers:
            return None

        last_layer = self.layers[-1]
        if hasattr(last_layer, 'output_shape'):
            self._output_shape = tuple(last_layer.output_shape)
            return self._output_shape
        elif hasattr(last_layer, 'output'):
            # For TensorFlow-style layers
            self._output_shape = tuple(last_layer.output.shape)
            return self._output_shape

        return None

    @output_shape.setter
    def output_shape(self, shape: Tuple[int, ...]):
        """Set the output shape explicitly.

        Args:
            shape: Tuple of integers representing output dimensions
        """
        self._output_shape = tuple(shape) if shape else None

    @property
    def is_quantized(self) -> bool:
        """Check if the model is quantized.

        A model is considered quantized if it has quantization parameters
        (scale factors, zero points) in its layers.

        Returns:
            True if model is quantized, False otherwise

        Example:
            >>> model.is_quantized
            True  # Model uses INT8 weights
        """
        if self._is_quantized is not None:
            return self._is_quantized

        # Check if metadata already set
        if self.metadata.quantized:
            self._is_quantized = True
            return True

        # Check layers for quantization parameters
        for layer in self.layers:
            # Check for common quantization attributes
            if hasattr(layer, 'quantization_factor') and layer.quantization_factor is not None:
                self._is_quantized = True
                self.metadata.quantized = True
                return True
            if hasattr(layer, 'scale') and layer.scale is not None:
                self._is_quantized = True
                self.metadata.quantized = True
                return True
            if hasattr(layer, 'zero_point') and layer.zero_point is not None:
                self._is_quantized = True
                self.metadata.quantized = True
                return True

        self._is_quantized = False
        return False

    @is_quantized.setter
    def is_quantized(self, value: bool):
        """Set the quantization status explicitly.

        Args:
            value: True if model is quantized, False otherwise
        """
        self._is_quantized = value
        self.metadata.quantized = value

    @property
    def num_layers(self) -> int:
        """Get the total number of layers in the model.

        Returns:
            Number of layers
        """
        return len(self.layers)

    @property
    def framework(self) -> str:
        """Get the source framework name.

        Returns:
            Framework name string ('tflite', 'onnx', 'pytorch', 'keras')
        """
        return self.metadata.framework

    def get_layer_types(self) -> List[str]:
        """Get a list of layer types in the model.

        Returns:
            List of layer type names

        Example:
            >>> model.get_layer_types()
            ['Conv2D', 'ReLU', 'MaxPool2D', 'Dense']
        """
        layer_types = []
        for layer in self.layers:
            if hasattr(layer, 'name'):
                layer_types.append(layer.name)
            else:
                layer_types.append(layer.__class__.__name__)
        return layer_types

    def get_conv_layers(self) -> List:
        """Get all convolutional layers in the model.

        Returns:
            List of Conv2D layer objects
        """
        conv_layers = []
        for layer in self.layers:
            layer_name = layer.name if hasattr(layer, 'name') else layer.__class__.__name__
            if 'conv' in layer_name.lower() or 'Conv' in layer_name:
                conv_layers.append(layer)
        return conv_layers

    def count_parameters(self) -> int:
        """Count total number of trainable parameters.

        Returns:
            Total parameter count
        """
        total_params = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                for weight in layer.weights:
                    if hasattr(weight, 'size'):
                        total_params += weight.size
                    elif hasattr(weight, 'shape'):
                        total_params += np.prod(weight.shape)
            elif hasattr(layer, 'kernel') and layer.kernel is not None:
                total_params += np.prod(layer.kernel.shape)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    total_params += np.prod(layer.bias.shape)

        self.metadata.param_count = total_params
        return total_params

    def summary(self) -> str:
        """Generate a summary string of the model.

        Returns:
            Multi-line string with model summary

        Example:
            >>> print(model.summary())
            Model Summary
            =============
            Framework: TFLite
            Layers: 8
            Input: (1, 1, 28, 28)
            Output: (1, 10)
            Quantized: Yes
            Parameters: 123,456
        """
        summary_lines = [
            "Model Summary",
            "=" * 50,
            f"Framework: {self.framework}",
            f"Layers: {self.num_layers}",
            f"Input shape: {self.input_shape}",
            f"Output shape: {self.output_shape}",
            f"Quantized: {'Yes' if self.is_quantized else 'No'}",
            f"Parameters: {self.count_parameters():,}",
            "",
            "Layer Structure:",
            "-" * 50
        ]

        for i, layer in enumerate(self.layers):
            layer_name = layer.name if hasattr(layer, 'name') else layer.__class__.__name__
            output_shape = layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'
            summary_lines.append(f"  {i:2d}: {layer_name:20s} → {str(output_shape)}")

        return "\n".join(summary_lines)

    def __repr__(self):
        """String representation of the model."""
        return (f"OpenEyeModel(framework={self.framework}, "
                f"layers={self.num_layers}, "
                f"quantized={self.is_quantized})")

    def __len__(self):
        """Return number of layers."""
        return len(self.layers)

    def __getitem__(self, idx):
        """Get layer by index."""
        return self.layers[idx]

    def __iter__(self):
        """Iterate over layers."""
        return iter(self.layers)


# Convenience type alias for backwards compatibility
BaseModel = OpenEyeBaseModel


if __name__ == "__main__":
    # Example usage and testing
    print("OpenEye Base Model Class")
    print("=" * 60)
    print("\nThis module provides the base class for all model formats.")
    print("\nSupported frameworks:")
    print("  - TensorFlow Lite (tflite)")
    print("  - ONNX (onnx)")
    print("  - PyTorch (pytorch)")
    print("  - Keras (keras)")
    print("\nAll framework-specific models inherit from OpenEyeBaseModel")
    print("to provide a unified interface.")
    print("\nCommon properties:")
    print("  - input_shape: Input tensor dimensions")
    print("  - output_shape: Output tensor dimensions")
    print("  - is_quantized: Quantization status")
    print("  - num_layers: Number of layers")
    print("  - framework: Source framework name")