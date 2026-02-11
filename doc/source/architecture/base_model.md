(base_model)=
# Base Model Class - Unified Interface

## Overview

The **OpenEyeBaseModel** class provides a unified interface for models from all supported frameworks (TensorFlow Lite, ONNX, PyTorch, Keras). This base class ensures that regardless of which framework you use to train your model, it will have a consistent API when loaded into OpenEye.

## Motivation

Different deep learning frameworks have different ways of representing models:
- TensorFlow uses Keras API with `.input` and `.output` attributes
- PyTorch uses `nn.Module` with forward passes
- ONNX uses a graph-based representation
- TFLite uses a flatbuffer format

The **OpenEyeBaseModel** abstracts these differences and provides a single, consistent interface.

## Architecture

```
                  ┌──────────────────────┐
                  │  OpenEyeBaseModel    │
                  │  (Abstract Base)     │
                  └──────────┬───────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼──────┐  ┌─────▼─────┐  ┌──────▼──────┐
    │ TFLite_model │  │ ONNXModel │  │ PyTorchModel│
    └──────────────┘  └───────────┘  └─────────────┘
```

All framework-specific model classes inherit from `OpenEyeBaseModel` and gain:
- Unified property access (`.input_shape`, `.output_shape`, `.is_quantized`)
- Common inspection methods (`.summary()`, `.get_layer_types()`)
- Framework metadata (`.framework`, `.metadata`)
- Parameter counting (`.count_parameters()`)

## Key Properties

### Common Properties (Available on All Models)

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `input_shape` | `tuple` | Input tensor dimensions | `(1, 1, 28, 28)` |
| `output_shape` | `tuple` | Output tensor dimensions | `(1, 10)` |
| `is_quantized` | `bool` | Quantization status | `True` |
| `framework` | `str` | Source framework name | `'tflite'` |
| `num_layers` | `int` | Total number of layers | `8` |
| `layers` | `list` | List of layer objects | `[conv1, relu1, ...]` |

### Example Usage

```python
from open_eye.model_loader import load_model

# Load models from different frameworks
tflite_model = load_model('model.tflite')
onnx_model = load_model('model.onnx')
pytorch_model = load_model('model.pth', input_shape=(1, 1, 28, 28))

# All have the same interface!
print(tflite_model.input_shape)   # (1, 1, 28, 28)
print(onnx_model.input_shape)     # (1, 1, 28, 28)
print(pytorch_model.input_shape)  # (1, 1, 28, 28)

print(tflite_model.is_quantized)  # True
print(onnx_model.is_quantized)    # True
print(pytorch_model.is_quantized) # True

print(tflite_model.framework)     # 'tflite'
print(onnx_model.framework)       # 'onnx'
print(pytorch_model.framework)    # 'pytorch'
```

## Methods

### `.summary()` - Model Summary

Generates a human-readable summary of the model:

```python
model = load_model('mnist.tflite')
print(model.summary())
```

Output:
```
Model Summary
==================================================
Framework: tflite
Layers: 8
Input shape: (1, 1, 28, 28)
Output shape: (1, 10)
Quantized: Yes
Parameters: 123,456

Layer Structure:
--------------------------------------------------
   0: conv2d              → (1, 32, 28, 28)
   1: max_pooling2d       → (1, 32, 14, 14)
   2: conv2d              → (1, 64, 14, 14)
   3: max_pooling2d       → (1, 64, 7, 7)
   4: conv2d              → (1, 64, 7, 7)
   5: flatten             → (1, 3136)
   6: dense               → (1, 64)
   7: dense               → (1, 10)
```

### `.get_layer_types()` - Layer Type List

Returns a list of layer type names:

```python
model = load_model('mnist.onnx')
layer_types = model.get_layer_types()
print(layer_types)
# Output: ['Conv2D', 'ReLU', 'MaxPool2D', 'Conv2D', 'ReLU', 'MaxPool2D', 'Flatten', 'Dense']
```

### `.get_conv_layers()` - Convolutional Layers

Retrieves all convolutional layers:

```python
model = load_model('resnet.tflite')
conv_layers = model.get_conv_layers()
print(f"Found {len(conv_layers)} convolutional layers")

for i, layer in enumerate(conv_layers):
    print(f"Conv {i}: {layer.filters} filters, kernel {layer.kernel_size}")
```

### `.count_parameters()` - Parameter Count

Counts total trainable parameters:

```python
model = load_model('vgg.pth', input_shape=(1, 3, 224, 224))
params = model.count_parameters()
print(f"Model has {params:,} parameters")
# Output: Model has 138,357,544 parameters
```

## ModelMetadata Class

Each model has a `.metadata` attribute containing additional information:

```python
class ModelMetadata:
    framework: str          # 'tflite', 'onnx', 'pytorch', 'keras'
    version: str            # Framework version
    quantized: bool         # Quantization status
    quantization_type: str  # e.g., 'int8', 'int16'
    input_dtype: str        # Input data type
    output_dtype: str       # Output data type
    model_size_bytes: int   # Approximate size
    param_count: int        # Parameter count
```

Example:
```python
model = load_model('model.onnx')
print(f"Framework: {model.metadata.framework}")
print(f"Quantized: {model.metadata.quantized}")
print(f"Quantization type: {model.metadata.quantization_type}")
print(f"Input dtype: {model.metadata.input_dtype}")
```

## Implementation Details

### Property Caching

The base class caches property lookups for efficiency:

```python
# First access: computed from layers
shape = model.input_shape  # Scans layers to find input

# Subsequent accesses: returned from cache
shape = model.input_shape  # Instant return
shape = model.input_shape  # Instant return
```

### Quantization Detection

The `is_quantized` property automatically detects quantization by checking layers for:
- `quantization_factor` attribute
- `scale` attribute
- `zero_point` attribute

If any layer has these attributes, the model is considered quantized.

### Framework-Specific Differences

While the base class provides a unified interface, some framework-specific features remain:

#### TFLite Models
- Full quantization support (INT8)
- TensorSpec-based input/output
- Optimized for mobile deployment

#### ONNX Models
- Graph-based representation
- QLinearConv for quantized convolutions
- Operator-level quantization

#### PyTorch Models
- Requires explicit input_shape
- State dict or full model loading
- Dynamic computation graphs

#### Keras Models
- Native TensorFlow integration
- Layer-based architecture
- SavedModel format support

## Backwards Compatibility

The base class is fully backwards compatible with existing OpenEye code:

**Old code (still works):**
```python
from open_eye.tflite2model import create_model_from_tflite

model = create_model_from_tflite(tflite_model_path='model.tflite')
# model is TFLite_model, has .layers attribute
for layer in model.layers:
    process(layer)
```

**New code (recommended):**
```python
from open_eye.model_loader import load_model

model = load_model('model.tflite')
# model is TFLite_model(OpenEyeBaseModel), has .layers + new properties
print(f"Input: {model.input_shape}")
print(f"Quantized: {model.is_quantized}")
for layer in model.layers:
    process(layer)
```

## Benefits

### 1. Unified API
Write code once, works with all frameworks:

```python
def analyze_model(model_path):
    model = load_model(model_path)  # Works for any format!

    print(f"Framework: {model.framework}")
    print(f"Input: {model.input_shape}")
    print(f"Output: {model.output_shape}")
    print(f"Quantized: {model.is_quantized}")
    print(f"Layers: {model.num_layers}")

    return model
```

### 2. Framework Transparency
Switch frameworks without changing code:

```python
# Works with TFLite
model = analyze_model('model.tflite')

# Works with ONNX
model = analyze_model('model.onnx')

# Works with PyTorch
model = analyze_model('model.pth')
```

### 3. Easy Comparison
Compare models from different frameworks:

```python
models = {
    'TFLite': load_model('model.tflite'),
    'ONNX': load_model('model.onnx'),
    'PyTorch': load_model('model.pth', input_shape=(1, 1, 28, 28))
}

for name, model in models.items():
    print(f"{name}: {model.num_layers} layers, quantized={model.is_quantized}")
```

### 4. Type Hinting Support
Better IDE support with common interface:

```python
from open_eye.base_model import OpenEyeBaseModel

def process_model(model: OpenEyeBaseModel):
    """Process any OpenEye model format."""
    # IDE knows about .input_shape, .is_quantized, etc.
    if model.is_quantized:
        print(f"Processing quantized {model.framework} model")
        print(f"Input: {model.input_shape}")
```

## Creating Custom Model Classes

To create a custom model class for a new framework:

```python
from open_eye.base_model import OpenEyeBaseModel

class MyFrameworkModel(OpenEyeBaseModel):
    """Custom model for MyFramework."""

    def __init__(self):
        super().__init__(framework='myframework')
        self.layers = []

    def add_layer(self, layer):
        """Add a layer to the model."""
        self.layers.append(layer)

    # input_shape, output_shape, is_quantized are automatically available!
```

## See Also

- [Multi-Framework Support](../tutorial/multi_framework_support.rst) - Framework conversion guide
- [Model Loader](model_loader.md) - Unified model loading
- [Layer Adapter](layer_adapter.md) - Layer compatibility system
- [Architecture](index.rst) - OpenEye architecture overview

## References

- `src/open_eye/base_model.py` - Base class implementation
- `src/open_eye/tflite2model.py` - TFLite model class
- `src/open_eye/onnx2model.py` - ONNX model class
- `src/open_eye/pytorch2model.py` - PyTorch model class
- `src/open_eye/model_loader.py` - Unified loader