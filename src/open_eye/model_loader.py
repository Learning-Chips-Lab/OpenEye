# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Unified model loader interface for OpenEye neural network accelerator.

This module provides a unified interface for loading models from different
deep learning frameworks (TensorFlow/Keras, PyTorch, ONNX) and converting
them to the OpenEye-compatible format.

Key Functions:
- load_model: Main entry point for loading models from any supported format
- detect_model_format: Auto-detect the format of a model file
- convert_to_keras_format: Helper to ensure compatibility with existing code

Typical Usage:
    >>> from open_eye.model_loader import load_model
    >>> model = load_model('my_model.onnx')
    >>> print(f"Loaded {len(model.layers)} layers")

    Or with explicit format:
    >>> model = load_model('my_model.pth', format='pytorch')

Supported Formats:
    - TensorFlow/Keras (.h5, .keras, saved_model directory)
    - TensorFlow Lite (.tflite)
    - PyTorch (.pt, .pth)
    - ONNX (.onnx)
"""

from pathlib import Path
import logging
from typing import Union, Optional

logger = logging.getLogger("cocotb")


def detect_model_format(model_path):
    """Auto-detect the format of a model file based on file extension.

    Args:
        model_path (str or Path): Path to the model file

    Returns:
        str: Detected format ('keras', 'tflite', 'pytorch', 'onnx', or 'unknown')

    Examples:
        >>> detect_model_format('model.onnx')
        'onnx'
        >>> detect_model_format('model.pth')
        'pytorch'
    """
    path = Path(model_path)

    # Check if it's a directory (could be SavedModel)
    if path.is_dir():
        # Check for TensorFlow SavedModel
        if (path / 'saved_model.pb').exists() or (path / 'keras_metadata.pb').exists():
            return 'keras'
        return 'unknown'

    # Check file extension
    suffix = path.suffix.lower()

    if suffix in ['.h5', '.keras']:
        return 'keras'
    elif suffix == '.tflite':
        return 'tflite'
    elif suffix in ['.pt', '.pth']:
        return 'pytorch'
    elif suffix == '.onnx':
        return 'onnx'
    else:
        return 'unknown'


def load_model(model_path: Union[str, Path],
               format: Optional[str] = None,
               input_shape: Optional[tuple] = None,
               use_random: bool = False):
    """Load a model from any supported format and convert to OpenEye format.

    This is the main entry point for loading models. It automatically detects
    the model format (or uses the explicitly provided format) and delegates to
    the appropriate converter.

    Args:
        model_path (str or Path): Path to the model file
        format (str, optional): Model format ('keras', 'tflite', 'pytorch', 'onnx').
            If None, format is auto-detected from file extension.
        input_shape (tuple, optional): Input shape for the model (batch, C, H, W).
            Required for PyTorch models. Defaults to (1, 1, 28, 28).
        use_random (bool, optional): Use random weights instead of trained weights.
            Defaults to False.

    Returns:
        Model object compatible with OpenEye layer parameter system.
        The exact type depends on the source format but all have a compatible
        .layers attribute.

    Raises:
        ValueError: If model format is unsupported or cannot be detected
        FileNotFoundError: If model file doesn't exist
        ImportError: If required framework library is not installed

    Examples:
        >>> # Auto-detect format
        >>> model = load_model('mnist_model.onnx')
        >>>
        >>> # Explicit format
        >>> model = load_model('model.pth', format='pytorch', input_shape=(1, 3, 224, 224))
        >>>
        >>> # TFLite model
        >>> model = load_model('quantized_model.tflite')
    """
    model_path = Path(model_path)

    # Check if file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Auto-detect format if not provided
    if format is None:
        format = detect_model_format(model_path)
        if format == 'unknown':
            raise ValueError(f"Cannot detect model format from file: {model_path}. "
                           "Please specify format explicitly using the 'format' parameter.")
        logger.info(f"Auto-detected model format: {format}")

    # Set default input shape if not provided
    if input_shape is None:
        input_shape = (1, 1, 28, 28)  # Default MNIST shape

    # Load model based on format
    if format == 'tflite':
        return load_tflite_model(model_path, use_random)

    elif format == 'keras':
        return load_keras_model(model_path, use_random)

    elif format == 'pytorch':
        return load_pytorch_model(model_path, input_shape, use_random)

    elif format == 'onnx':
        return load_onnx_model(model_path, use_random)

    else:
        raise ValueError(f"Unsupported model format: {format}. "
                       "Supported formats are: 'keras', 'tflite', 'pytorch', 'onnx'")


def load_tflite_model(model_path, use_random=False):
    """Load a TensorFlow Lite model.

    Args:
        model_path (Path): Path to .tflite file
        use_random (bool): Use random weights

    Returns:
        TFLite_model object

    Raises:
        ImportError: If TensorFlow is not installed
    """
    try:
        from open_eye.tflite2model import create_model_from_tflite
    except ImportError:
        raise ImportError("TensorFlow is required to load TFLite models. "
                        "Install it with: pip install tensorflow")

    logger.info(f"Loading TFLite model from {model_path}")
    model = create_model_from_tflite(
        use_random=use_random,
        tflite_model_path=str(model_path)
    )
    logger.info(f"Successfully loaded TFLite model with {len(model.layers)} layers")
    return model


def load_keras_model(model_path, use_random=False):
    """Load a Keras/TensorFlow SavedModel.

    Args:
        model_path (Path): Path to .h5, .keras file or SavedModel directory
        use_random (bool): Use random weights

    Returns:
        Keras model object

    Raises:
        ImportError: If TensorFlow is not installed
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required to load Keras models. "
                        "Install it with: pip install tensorflow")

    logger.info(f"Loading Keras model from {model_path}")

    # Load the Keras model
    if model_path.is_dir():
        # SavedModel format
        model = tf.saved_model.load(str(model_path))
    else:
        # .h5 or .keras format
        model = tf.keras.models.load_model(str(model_path))

    if use_random:
        logger.warning("use_random=True is not implemented for Keras models")

    logger.info(f"Successfully loaded Keras model")
    return model


def load_pytorch_model(model_path, input_shape=(1, 1, 28, 28), use_random=False):
    """Load a PyTorch model.

    Args:
        model_path (Path): Path to .pt or .pth file
        input_shape (tuple): Input shape for the model
        use_random (bool): Use random weights

    Returns:
        PyTorchModel object compatible with OpenEye

    Raises:
        ImportError: If PyTorch is not installed
    """
    try:
        import torch
        from open_eye.pytorch2model import create_model_from_pytorch
    except ImportError:
        raise ImportError("PyTorch is required to load PyTorch models. "
                        "Install it with: pip install torch")

    logger.info(f"Loading PyTorch model from {model_path}")

    # Load the PyTorch model
    checkpoint = torch.load(str(model_path), map_location='cpu')

    # Try to extract the model from checkpoint
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            pytorch_model = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            # Need to reconstruct the model - this requires knowing the architecture
            logger.error("Cannot load PyTorch model: model_state_dict found but model architecture unknown. "
                       "Please pass the model object directly or save the full model.")
            raise ValueError("PyTorch checkpoint contains only state_dict. "
                           "Please save the full model or provide the model architecture.")
        else:
            # Assume the checkpoint is a state_dict
            logger.error("Cannot determine PyTorch model structure from checkpoint. "
                       "Please pass the model object directly.")
            raise ValueError("Cannot determine model structure from checkpoint")
    else:
        # Assume it's a model object
        pytorch_model = checkpoint

    # Convert to OpenEye format
    model = create_model_from_pytorch(
        pytorch_model,
        input_shape=input_shape,
        use_random=use_random
    )

    logger.info(f"Successfully loaded PyTorch model with {len(model.layers)} layers")
    return model


def load_onnx_model(model_path, use_random=False):
    """Load an ONNX model.

    Args:
        model_path (Path): Path to .onnx file
        use_random (bool): Use random weights

    Returns:
        ONNXModel object compatible with OpenEye

    Raises:
        ImportError: If ONNX is not installed
    """
    try:
        import onnx
        from open_eye.onnx2model import create_model_from_onnx
    except ImportError:
        raise ImportError("ONNX is required to load ONNX models. "
                        "Install it with: pip install onnx")

    logger.info(f"Loading ONNX model from {model_path}")

    # Convert to OpenEye format
    model = create_model_from_onnx(
        str(model_path),
        use_random=use_random
    )

    logger.info(f"Successfully loaded ONNX model with {len(model.layers)} layers")
    return model


# Convenience function for backwards compatibility
def load_model_auto(model_path, **kwargs):
    """Convenience function that auto-detects format and loads model.

    This is an alias for load_model() with format=None (auto-detect).

    Args:
        model_path (str or Path): Path to model file
        **kwargs: Additional arguments passed to load_model()

    Returns:
        Model object compatible with OpenEye
    """
    return load_model(model_path, format=None, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        try:
            model = load_model(model_path)
            print(f"✅ Successfully loaded model from {model_path}")
            print(f"   Format: {detect_model_format(model_path)}")
            print(f"   Layers: {len(model.layers)}")

            for i, layer in enumerate(model.layers):
                print(f"   Layer {i}: {layer.name}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Model Loader for OpenEye")
        print("=" * 60)
        print("\nUsage: python model_loader.py <path_to_model>")
        print("\nSupported formats:")
        print("  - TensorFlow/Keras: .h5, .keras, saved_model/")
        print("  - TensorFlow Lite: .tflite")
        print("  - PyTorch: .pt, .pth")
        print("  - ONNX: .onnx")
        print("\nExample:")
        print("  python model_loader.py mnist_model.onnx")
