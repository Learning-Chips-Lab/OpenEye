.. _model_formats:

Model Format Guide
==================

This guide provides detailed information on loading and converting models from each supported framework.

TensorFlow/Keras Models
-----------------------

Supported Formats
~~~~~~~~~~~~~~~~~

OpenEye supports three TensorFlow/Keras formats:

- ``.h5`` - HDF5 format (legacy Keras)
- ``.keras`` - Keras 3.0+ native format
- ``SavedModel`` - TensorFlow SavedModel directory

Loading Keras Models
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_eye.model_loader import load_model

    # Load .h5 file
    model = load_model('mnist_model.h5', format='keras')

    # Load .keras file
    model = load_model('mnist_model.keras', format='keras')

    # Load SavedModel directory
    model = load_model('saved_model/', format='keras')

Creating a Keras Model
~~~~~~~~~~~~~~~~~~~~~~

Complete example of creating and saving a Keras model for OpenEye:

.. code-block:: python

    import tensorflow as tf

    # Define your model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train your model
    model.fit(train_data, train_labels, epochs=5)

    # Save for OpenEye
    model.save('my_model.h5')

TensorFlow Lite Models
----------------------

Supported Formats
~~~~~~~~~~~~~~~~~

- ``.tflite`` - TensorFlow Lite quantized or float models

Loading TFLite Models
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_eye.model_loader import load_model

    model = load_model('quantized_model.tflite', format='tflite')

Converting Keras to TFLite
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert a Keras model to TFLite with INT8 quantization for optimal OpenEye performance:

.. code-block:: python

    import tensorflow as tf

    # Load your Keras model
    keras_model = tf.keras.models.load_model('my_model.h5')

    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]

    # Perform conversion
    tflite_model = converter.convert()

    # Save the quantized model
    with open('quantized_model.tflite', 'wb') as f:
        f.write(tflite_model)

.. note::
   TFLite quantization automatically handles scale and zero-point parameters,
   which OpenEye extracts during model loading.

PyTorch Models
--------------

Supported Formats
~~~~~~~~~~~~~~~~~

- ``.pt`` - PyTorch model file
- ``.pth`` - PyTorch checkpoint file

Loading PyTorch Models
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_eye.model_loader import load_model

    # Load PyTorch model
    # IMPORTANT: You must specify input_shape for PyTorch models
    model = load_model(
        'mnist_model.pth',
        format='pytorch',
        input_shape=(1, 1, 28, 28)  # (batch, channels, height, width)
    )

.. important::
   PyTorch models require explicit ``input_shape`` parameter because PyTorch
   doesn't store shape information in saved models.

Creating a PyTorch Model
~~~~~~~~~~~~~~~~~~~~~~~~

Complete example with proper model structure:

.. code-block:: python

    import torch
    import torch.nn as nn

    class MNISTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 7 * 7, 10)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            x = self.fc1(x)
            return x

    # Create and train model
    model = MNISTModel()
    # ... training code ...

    # IMPORTANT: Save the FULL model (not just state_dict)
    torch.save(model, 'mnist_model.pth')

Saving PyTorch Models
~~~~~~~~~~~~~~~~~~~~~

**Option 1: Save Complete Model (Recommended)**

.. code-block:: python

    # Save full model with architecture
    torch.save(model, 'mnist_model.pth')

**Option 2: Save Checkpoint with Metadata**

.. code-block:: python

    # Save as checkpoint (requires model architecture to load)
    torch.save({
        'model': model,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, 'mnist_checkpoint.pth')

.. warning::
   Do NOT save only the state_dict: ``torch.save(model.state_dict(), 'model.pth')``

   OpenEye requires the full model architecture. If you only have state_dict,
   you must provide the model class definition separately.

PyTorch Quantization
~~~~~~~~~~~~~~~~~~~~

Complete quantization workflow for PyTorch models:

.. code-block:: python

    import torch
    from torch.ao.quantization import get_default_qconfig, prepare, convert

    # Prepare model for quantization
    model.eval()
    model.qconfig = get_default_qconfig('x86')
    model_prepared = prepare(model)

    # Calibrate with representative data
    with torch.no_grad():
        for data, _ in calibration_loader:
            model_prepared(data)

    # Convert to quantized model
    model_quantized = convert(model_prepared)

    # Save quantized model
    torch.save(model_quantized, 'mnist_quantized.pth')

ONNX Models
-----------

Supported Formats
~~~~~~~~~~~~~~~~~

- ``.onnx`` - ONNX model file (regular or quantized)

.. note::
   Recommend using ONNX opset version 13 or higher for best compatibility.

Loading ONNX Models
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from open_eye.model_loader import load_model

    model = load_model('model.onnx', format='onnx')

Converting PyTorch to ONNX
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export a PyTorch model to ONNX format:

.. code-block:: python

    import torch

    # Load your PyTorch model
    pytorch_model = torch.load('mnist_model.pth')
    pytorch_model.eval()

    # Create dummy input with correct shape
    dummy_input = torch.randn(1, 1, 28, 28)

    # Export to ONNX
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        'mnist_model.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=13  # Use opset 13 or higher
    )

Converting TensorFlow to ONNX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert a TensorFlow/Keras model to ONNX using tf2onnx:

.. code-block:: python

    import tensorflow as tf
    import tf2onnx

    # Load your TensorFlow model
    keras_model = tf.keras.models.load_model('my_model.h5')

    # Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=[
            tf.TensorSpec(
                shape=(None, 28, 28, 1),
                dtype=tf.float32,
                name='input'
            )
        ],
        opset=13
    )

    # Save ONNX model
    with open('model.onnx', 'wb') as f:
        f.write(onnx_model.SerializeToString())

.. tip::
   Use tf2onnx as an intermediate format to convert TensorFlow models to
   OpenEye. This provides better quantization support than direct TFLite.

ONNX Quantization
~~~~~~~~~~~~~~~~~

Quantize ONNX models using ONNX Runtime:

.. code-block:: python

    from onnxruntime.quantization import quantize_static, QuantType

    # Define calibration data reader
    class CalibrationDataReader:
        def __init__(self, dataset):
            self.dataset = dataset
            self.iterator = iter(dataset)

        def get_next(self):
            try:
                batch = next(self.iterator)
                return {'input': batch[0].numpy()}
            except StopIteration:
                return None

    # Create calibration reader
    calibration_reader = CalibrationDataReader(calibration_dataset)

    # Quantize the model
    quantize_static(
        model_input='model.onnx',
        model_output='model_quantized.onnx',
        calibration_data_reader=calibration_reader,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8
    )

Model Requirements
------------------

Input Shape Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenEye supports two tensor format conventions:

.. list-table:: Input Shape Formats
   :header-rows: 1
   :widths: 30 35 35

   * - Framework
     - Channels First (NCHW)
     - Channels Last (NHWC)
   * - **PyTorch**
     - ``(batch, channels, height, width)``
     - Not typically used
   * - **TensorFlow**
     - Configurable
     - ``(batch, height, width, channels)``
   * - **ONNX**
     - ``(batch, channels, height, width)``
     - Configurable

Common Input Shapes
^^^^^^^^^^^^^^^^^^^

.. list-table:: Standard Dataset Shapes
   :header-rows: 1
   :widths: 20 40 40

   * - Dataset
     - Channels First (NCHW)
     - Channels Last (NHWC)
   * - **MNIST**
     - ``(1, 1, 28, 28)``
     - ``(1, 28, 28, 1)``
   * - **CIFAR-10**
     - ``(1, 3, 32, 32)``
     - ``(1, 32, 32, 3)``
   * - **ImageNet**
     - ``(1, 3, 224, 224)``
     - ``(1, 224, 224, 3)``

Supported Layer Types
~~~~~~~~~~~~~~~~~~~~~

All frameworks must use these supported operations:

- **Convolution**: 2D convolution (Conv2D) with optional ReLU activation
- **MaxPooling**: 2D max pooling (MaxPool2D)
- **Dense/Linear**: Fully connected layers
- **Flatten**: Reshape to 1D (implicit in some frameworks)
- **ReLU**: Rectified Linear Unit activation

Quantization Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For optimal hardware performance, models should be quantized to 8-bit integers:

.. list-table:: Quantization Tools
   :header-rows: 1
   :widths: 25 35 40

   * - Framework
     - Tool
     - Key Features
   * - **TensorFlow**
     - TFLite quantization
     - ``tf.lite.Optimize.DEFAULT``
   * - **PyTorch**
     - ``torch.ao.quantization``
     - Static/dynamic quantization
   * - **ONNX**
     - ``onnxruntime.quantization``
     - Static quantization with calibration

Troubleshooting
---------------

Error: "Cannot detect model format"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: The file extension is not recognized.

**Solution**: Specify the format explicitly:

.. code-block:: python

    # For SavedModel directory (no extension)
    model = load_model('my_model', format='keras')

    # For non-standard extensions
    model = load_model('custom.model', format='pytorch', input_shape=(1, 3, 224, 224))

Error: "PyTorch checkpoint contains only state_dict"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: The PyTorch file only contains weights, not the full model architecture.

**Cause**: Model was saved incorrectly using:

.. code-block:: python

    # WRONG - Only saves weights
    torch.save(model.state_dict(), 'model.pth')

**Solution**: Save the complete model:

.. code-block:: python

    # CORRECT - Saves full model with architecture
    torch.save(model, 'model.pth')

If you only have a state_dict file, you must provide the model class definition
and reconstruct the model:

.. code-block:: python

    # Load state_dict into model architecture
    model = MNISTModel()  # Your model class
    model.load_state_dict(torch.load('model.pth'))

    # Now save the full model
    torch.save(model, 'model_full.pth')

Error: "Input shape required for PyTorch models"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: PyTorch models need explicit input shape for shape inference.

**Solution**: Provide the ``input_shape`` parameter:

.. code-block:: python

    model = load_model(
        'model.pth',
        format='pytorch',
        input_shape=(1, 3, 224, 224)  # Specify input dimensions
    )

Error: "Unsupported layer type"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: The model contains a layer type not supported by OpenEye.

**Unsupported Operations**:
- Concatenation layers
- Skip connections (ResNet-style)
- Batch normalization (must be fused)
- Dilated convolutions
- 3D convolutions
- Recurrent layers (LSTM, GRU)

**Solution**: Modify your model to use only supported layers (Conv2D, MaxPool2D, Dense, ReLU, Flatten).

Warning: "ONNX model validation warning"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: The ONNX model has minor validation issues but may still work.

**Solution**: This warning is usually safe to ignore. The model will likely work correctly.

If you encounter actual errors (not just warnings), try:

1. Re-export your ONNX model with opset version 13 or higher:

   .. code-block:: python

       torch.onnx.export(
           model, dummy_input, 'model.onnx',
           opset_version=13  # or 14, 15, etc.
       )

2. Validate the ONNX model:

   .. code-block:: python

       import onnx

       model = onnx.load('model.onnx')
       onnx.checker.check_model(model)

Complete Examples
-----------------

Example 1: MNIST with PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

End-to-end workflow for MNIST classification with PyTorch:

.. code-block:: python

    import torch
    from open_eye.model_loader import load_model
    from open_eye.layer_adapter import adapt_layers_for_keras
    from open_eye.layer_parameters import LayerParameters

    # Step 1: Load quantized PyTorch model
    model = load_model(
        'mnist_quantized.pth',
        format='pytorch',
        input_shape=(1, 1, 28, 28)
    )

    print(f"Loaded {len(model.layers)} layers")

    # Step 2: Adapt layers for OpenEye processing
    adapted_layers = adapt_layers_for_keras(model.layers)

    # Step 3: Process with OpenEye hardware parameters
    layer_params = []
    for i, layer in enumerate(adapted_layers):
        params = LayerParameters(
            layer_parameters=layer_params,
            layer=layer,
            params=hardware_params,
            layer_number=i,
            max_layers=len(adapted_layers)
        )
        layer_params.append(params)

    # Step 4: Continue with OpenEye workflow
    # ... hardware configuration and execution ...

Example 2: ImageNet with ONNX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load and process a ResNet18 model for ImageNet classification:

.. code-block:: python

    from open_eye.model_loader import load_model
    from open_eye.layer_adapter import adapt_layers_for_keras

    # Step 1: Load quantized ONNX model
    model = load_model('resnet18_quantized.onnx', format='onnx')

    print(f"Model loaded: {len(model.layers)} layers")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # Step 2: Adapt for OpenEye
    adapted_layers = adapt_layers_for_keras(model.layers)

    # Step 3: Verify layer compatibility
    for i, layer in enumerate(adapted_layers):
        print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")
        if hasattr(layer, 'kernel'):
            print(f"  - Kernel shape: {layer.kernel.shape}")

    # Step 4: Continue with OpenEye processing
    # ... rest of OpenEye workflow ...

Next Steps
----------

After successfully loading your model, follow these steps to deploy on OpenEye:

1. **Adapt Layers**

   Use ``layer_adapter.adapt_layers_for_keras()`` to make layers compatible with OpenEye:

   .. code-block:: python

       from open_eye.layer_adapter import adapt_layers_for_keras
       adapted_layers = adapt_layers_for_keras(model.layers)

2. **Create Layer Parameters**

   Generate hardware-specific parameters for each layer:

   .. code-block:: python

       from open_eye.layer_parameters import LayerParameters

       layer_params = []
       for i, layer in enumerate(adapted_layers):
           params = LayerParameters(
               layer_parameters=layer_params,
               layer=layer,
               params=hardware_params,
               layer_number=i,
               max_layers=len(adapted_layers)
           )
           layer_params.append(params)

3. **Generate Hardware Configuration**

   Create stream mappers and router configurations:

   .. code-block:: python

       from open_eye.conv_mapper import ConvMapper

       mapper = ConvMapper(hardware_params)
       # ... generate data streams ...

4. **Run Inference**

   Execute on OpenEye accelerator hardware or simulator.

See Also
--------

- :ref:`multi_framework_support` - Multi-framework architecture and overview
- :ref:`architecture` - OpenEye hardware architecture
- :ref:`software_integration` - Software stack details
- :ref:`test` - Testing and validation

References
----------

- `PyTorch Model Saving <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_
- `TensorFlow Lite Quantization <https://www.tensorflow.org/lite/performance/post_training_quantization>`_
- `ONNX Format Specification <https://onnx.ai/onnx/>`_
- `tf2onnx Converter <https://github.com/onnx/tensorflow-onnx>`_
