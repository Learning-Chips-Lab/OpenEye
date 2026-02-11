.. _multi_framework_support:

Multi-Framework Support
========================

Overview
--------

OpenEye supports multiple deep learning frameworks for model conversion and deployment. You can train your models in your preferred framework and seamlessly deploy them on the OpenEye accelerator.

Supported Frameworks
--------------------

.. list-table:: Framework Support Matrix
   :header-rows: 1
   :widths: 20 20 15 25

   * - Framework
     - Format
     - Status
     - Quantization Support
   * - **TensorFlow/Keras**
     - ``.h5``, ``.keras``, SavedModel
     - ✅ Full
     - ✅ TFLite
   * - **TensorFlow Lite**
     - ``.tflite``
     - ✅ Full
     - ✅ Native
   * - **PyTorch**
     - ``.pt``, ``.pth``
     - ✅ Full
     - ✅ torch.ao.quantization
   * - **ONNX**
     - ``.onnx``
     - ✅ Full
     - ✅ onnxruntime

Installation
------------

Install Framework Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install additional dependencies for multi-framework support:

.. code-block:: bash

    # Install PyTorch support
    pip install torch torchvision

    # Install ONNX support
    pip install onnx onnxruntime onnxscript onnxoptimizer

    # Or install everything from requirements
    pip install -r requirements.txt

Verify Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Verify PyTorch
    import torch
    print(f"PyTorch version: {torch.__version__}")

    # Verify ONNX
    import onnx
    print(f"ONNX version: {onnx.__version__}")

Quick Start
-----------

Basic Model Loading
~~~~~~~~~~~~~~~~~~~

The unified model loader provides a single interface for all frameworks:

.. code-block:: python

    from open_eye.model_loader import load_model

    # Load any supported model format (auto-detection)
    model = load_model('my_model.onnx')
    model = load_model('my_model.pth', format='pytorch', input_shape=(1, 3, 224, 224))
    model = load_model('my_model.h5', format='keras')
    model = load_model('my_model.tflite', format='tflite')

    print(f"Loaded {len(model.layers)} layers")

Format Detection
~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_eye.model_loader import detect_model_format

    # Auto-detect format from file extension
    format = detect_model_format('model.onnx')
    print(f"Detected format: {format}")  # Output: onnx

    # Load with detected format
    model = load_model('model.onnx')

Architecture
------------

Conversion Flow
~~~~~~~~~~~~~~~

The multi-framework support uses a modular architecture:

.. code-block:: text

    ┌─────────────┐
    │   PyTorch   │──┐
    │   .pt/.pth  │  │
    └─────────────┘  │
                     │    ┌──────────────────┐    ┌─────────────┐
    ┌─────────────┐  │    │  model_loader.py │    │   Layer     │
    │    ONNX     │──┼───▶│  (auto-detect)   │───▶│  Adapter    │
    │    .onnx    │  │    └──────────────────┘    └─────────────┘
    └─────────────┘  │                                    │
                     │                                    ▼
    ┌─────────────┐  │                           ┌────────────────┐
    │   TFLite    │──┤                           │  OpenEye       │
    │   .tflite   │  │                           │  Internal      │
    └─────────────┘  │                           │  Format        │
                     │                           └────────────────┘
    ┌─────────────┐  │                                    │
    │    Keras    │──┘                                    ▼
    │  .h5/.keras │                              ┌────────────────┐
    └─────────────┘                              │ LayerParameters│
                                                 │ & Hardware     │
                                                 │ Mapping        │
                                                 └────────────────┘

Module Structure
~~~~~~~~~~~~~~~~

The framework support is organized into specialized modules:

.. code-block:: text

    src/open_eye/
    ├── model_loader.py       # Unified interface for all formats
    ├── pytorch2model.py      # PyTorch converter
    ├── onnx2model.py         # ONNX converter
    ├── tflite2model.py       # TFLite converter
    ├── get_model_struct.py   # Keras converter
    ├── layer_adapter.py      # Layer compatibility adapter
    └── layer_parameters.py   # Hardware mapping

Key Features
------------

1. Unified Model Loader
~~~~~~~~~~~~~~~~~~~~~~~~

Single entry point for all model formats with automatic format detection:

.. code-block:: python

    from open_eye.model_loader import load_model, detect_model_format

    # Auto-detect and load
    model = load_model('path/to/model.onnx')

    # Explicit format specification
    model = load_model('path/to/model', format='keras')

2. Framework-Specific Converters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each framework has a dedicated converter module:

**PyTorch Converter** (``pytorch2model.py``):

.. code-block:: python

    from open_eye.pytorch2model import create_model_from_pytorch
    import torch

    pytorch_model = torch.load('model.pth')
    openeye_model = create_model_from_pytorch(
        pytorch_model,
        input_shape=(1, 1, 28, 28)
    )

**ONNX Converter** (``onnx2model.py``):

.. code-block:: python

    from open_eye.onnx2model import create_model_from_onnx

    openeye_model = create_model_from_onnx('model.onnx')

**TFLite Converter** (``tflite2model.py``):

.. code-block:: python

    from open_eye.tflite2model import create_model_from_tflite

    openeye_model = create_model_from_tflite('model.tflite')

3. Layer Adapter System
~~~~~~~~~~~~~~~~~~~~~~~~

Makes all frameworks compatible with OpenEye's internal layer representation:

.. code-block:: python

    from open_eye.layer_adapter import adapt_layers_for_keras

    # Convert PyTorch/ONNX layers to Keras-compatible format
    keras_compatible_layers = adapt_layers_for_keras(model.layers)

    # Now you can use with existing OpenEye code
    from open_eye.layer_parameters import LayerParameters
    # ... continue with normal OpenEye workflow ...

Complete Examples
-----------------

Example 1: PyTorch Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train and deploy a PyTorch model on OpenEye:

.. code-block:: python

    import torch
    import torch.nn as nn
    from open_eye.model_loader import load_model
    from open_eye.layer_adapter import adapt_layers_for_keras

    # 1. Define PyTorch model
    class MNISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 7 * 7, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = self.fc1(x)
            return x

    # 2. Train model (training code omitted)
    model = MNISTNet()
    # ... train model ...

    # 3. Save model
    torch.save(model, 'mnist.pth')

    # 4. Load with OpenEye
    openeye_model = load_model(
        'mnist.pth',
        format='pytorch',
        input_shape=(1, 1, 28, 28)
    )

    # 5. Adapt layers for hardware mapping
    adapted_layers = adapt_layers_for_keras(openeye_model.layers)

    # 6. Continue with OpenEye workflow
    print(f"Ready for OpenEye with {len(adapted_layers)} layers")

Example 2: ONNX with Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ONNX quantization for efficient deployment:

.. code-block:: python

    import torch
    import onnx
    from onnxruntime.quantization import quantize_static, QuantType
    from open_eye.model_loader import load_model

    # 1. Export PyTorch to ONNX
    pytorch_model = torch.load('model.pth')
    pytorch_model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        pytorch_model,
        dummy_input,
        'model.onnx',
        input_names=['input'],
        output_names=['output']
    )

    # 2. Quantize ONNX model
    class CalibrationReader:
        def __init__(self, dataset):
            self.dataset = dataset
            self.iter = iter(dataset)

        def get_next(self):
            try:
                data, _ = next(self.iter)
                return {'input': data.numpy()}
            except StopIteration:
                return None

    calibration_reader = CalibrationReader(calibration_dataset)

    quantize_static(
        'model.onnx',
        'model_quantized.onnx',
        calibration_reader,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8
    )

    # 3. Load quantized model with OpenEye
    openeye_model = load_model('model_quantized.onnx')

    print(f"Quantized model loaded: {len(openeye_model.layers)} layers")

Example 3: TensorFlow to ONNX to OpenEye
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert TensorFlow models via ONNX intermediate format:

.. code-block:: python

    import tensorflow as tf
    import tf2onnx
    from open_eye.model_loader import load_model

    # 1. Create TensorFlow model
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])

    # 2. Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(
        tf_model,
        input_signature=[tf.TensorSpec((None, 28, 28, 1), tf.float32, name='input')],
        opset=13
    )

    # 3. Save ONNX
    with open('tf_to_onnx.onnx', 'wb') as f:
        f.write(onnx_model.SerializeToString())

    # 4. Load with OpenEye
    openeye_model = load_model('tf_to_onnx.onnx')

    print(f"TF→ONNX→OpenEye successful: {len(openeye_model.layers)} layers")

Supported Layer Types
---------------------

All frameworks must use these layer types for OpenEye compatibility:

.. list-table:: Layer Type Mapping
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Layer Type
     - TensorFlow
     - PyTorch
     - ONNX
     - Notes
   * - **Conv2D**
     - ``Conv2D``
     - ``Conv2d``
     - ``Conv``
     - 2D convolution
   * - **MaxPool**
     - ``MaxPooling2D``
     - ``MaxPool2d``
     - ``MaxPool``
     - 2D max pooling
   * - **Dense**
     - ``Dense``
     - ``Linear``
     - ``Gemm``
     - Fully connected
   * - **ReLU**
     - ``ReLU``
     - ``ReLU``
     - ``Relu``
     - Activation
   * - **Flatten**
     - ``Flatten``
     - ``Flatten``
     - ``Reshape``
     - Reshape to 1D

.. note::
   Unsupported layers will generate warnings and may cause errors during conversion.

Quantization Guide
------------------

Why Quantize?
~~~~~~~~~~~~~

Quantization converts floating-point weights to 8-bit integers, providing:

- **4× smaller model size**
- **Faster inference** on integer hardware
- **Lower power consumption**
- **Maintained accuracy** (typically <1% loss)

Per-Framework Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch Quantization
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from torch.ao.quantization import get_default_qconfig, prepare, convert

    # Prepare model
    model.eval()
    model.qconfig = get_default_qconfig('x86')
    model_prepared = prepare(model)

    # Calibrate
    for data, _ in calibration_loader:
        model_prepared(data)

    # Quantize
    model_quantized = convert(model_prepared)
    torch.save(model_quantized, 'model_quantized.pth')

ONNX Quantization
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from onnxruntime.quantization import quantize_static, QuantType

    quantize_static(
        'model.onnx',
        'model_quantized.onnx',
        calibration_data_reader,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8
    )

TFLite Quantization
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]

    tflite_model = converter.convert()
    with open('model_quantized.tflite', 'wb') as f:
        f.write(tflite_model)

Testing
-------

Each module includes standalone testing capabilities:

.. code-block:: bash

    # Test PyTorch conversion
    python src/open_eye/pytorch2model.py

    # Test ONNX conversion
    python src/open_eye/onnx2model.py

    # Test unified loader
    python src/open_eye/model_loader.py model.onnx

    # Test layer adapter
    python src/open_eye/layer_adapter.py

Common Issues & Solutions
-------------------------

Issue: "Cannot detect model format"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Unknown file extension

**Solution**: Specify format explicitly:

.. code-block:: python

    model = load_model('my_model', format='keras')

Issue: "Input shape required for PyTorch"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: PyTorch needs explicit shape for inference

**Solution**: Provide ``input_shape``:

.. code-block:: python

    model = load_model('model.pth', format='pytorch', input_shape=(1, 3, 224, 224))

Issue: "Unsupported layer type"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Model uses unsupported operations

**Solution**: Modify model to use only supported layers:

- Conv2D
- MaxPool2D
- Dense/Linear
- ReLU
- Flatten

Performance Tips
----------------

1. **Always quantize** models before deployment
2. **Use BatchNorm fusion** during quantization
3. **Keep kernel sizes** ≤ 5×5 for optimal hardware usage
4. **Prefer SAME padding** over VALID padding
5. **Avoid skip connections** (not yet supported)

Backward Compatibility
----------------------

All changes are **fully backward compatible**:

- Existing TFLite/Keras code continues to work unchanged
- New frameworks are opt-in (require explicit import)
- No changes to core ``layer_parameters.py`` logic
- Adapter pattern ensures transparent operation

Migration Example
~~~~~~~~~~~~~~~~~

**Old code (still works):**

.. code-block:: python

    from open_eye.tflite2model import create_model_from_tflite
    model = create_model_from_tflite(tflite_model_path='model.tflite')

**New code (recommended):**

.. code-block:: python

    from open_eye.model_loader import load_model
    model = load_model('model.tflite')  # Same result, simpler API

Roadmap
-------

Future enhancements planned:

.. hlist::
   :columns: 2

   * Support for depthwise separable convolutions
   * Batch normalization layer support
   * Skip connections (ResNet-style)
   * Additional quantization schemes
   * Automatic model optimization
   * Support for TorchScript
   * JAX framework support

Contributing
------------

To add support for a new framework:

1. Create ``<framework>2model.py`` converter
2. Define layer classes compatible with OpenEye
3. Add format detection to ``model_loader.py``
4. Update ``layer_adapter.py`` if needed
5. Add tests and documentation

See ``CONTRIBUTING.md`` for detailed guidelines.

References
----------

- `TensorFlow Lite Quantization <https://www.tensorflow.org/lite/performance/post_training_quantization>`_
- `PyTorch Quantization <https://pytorch.org/docs/stable/quantization.html>`_
- `ONNX Runtime Quantization <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>`_
- :ref:`architecture` - OpenEye Architecture Overview
- :ref:`sparsity_modes` - Sparse and Dense Mode Support

See Also
--------

- :ref:`tutorial` - Getting started with OpenEye
- :ref:`architecture` - Architecture documentation
- :ref:`test` - Testing and validation
