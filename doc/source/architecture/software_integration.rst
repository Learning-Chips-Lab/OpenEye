.. _software_integration:

Software Integration
====================

Overview
--------

The OpenEye accelerator includes a comprehensive software stack that bridges high-level deep learning frameworks with the hardware implementation. This software layer handles model conversion, layer mapping, and hardware configuration.

Software Architecture
---------------------

Layer Structure
~~~~~~~~~~~~~~~

The software stack is organized into distinct functional layers:

.. code-block:: text

    ┌─────────────────────────────────────────────┐
    │  Deep Learning Frameworks                   │
    │  (PyTorch, ONNX, TensorFlow, Keras)        │
    └─────────────────┬───────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────┐
    │  Model Loader & Format Detection            │
    │  - Auto-detection (model_loader.py)         │
    │  - Format-specific converters               │
    └─────────────────┬───────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────┐
    │  Layer Adapter                              │
    │  - Framework-to-OpenEye translation         │
    │  - Keras-compatible interface               │
    └─────────────────┬───────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────┐
    │  Layer Parameters                           │
    │  - Hardware constraint mapping              │
    │  - Memory allocation                        │
    │  - Data layout optimization                 │
    └─────────────────┬───────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────┐
    │  Stream Mappers                             │
    │  - IACT/WGHT/PSUM stream generation        │
    │  - Sparse encoding                          │
    │  - Router configuration                     │
    └─────────────────┬───────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────┐
    │  Hardware (PE Array, GLB, NoC)             │
    └─────────────────────────────────────────────┘

Module Organization
-------------------

Core Modules
~~~~~~~~~~~~

The OpenEye software is structured into specialized modules:

**Model Conversion Layer**

.. code-block:: text

    src/open_eye/
    ├── model_loader.py          # Unified model loading interface
    ├── pytorch2model.py         # PyTorch → OpenEye converter
    ├── onnx2model.py            # ONNX → OpenEye converter
    ├── tflite2model.py          # TFLite → OpenEye converter
    └── get_model_struct.py      # Keras → OpenEye converter

**Compatibility Layer**

.. code-block:: text

    src/open_eye/
    └── layer_adapter.py         # Framework-agnostic adapter

**Hardware Mapping Layer**

.. code-block:: text

    src/open_eye/
    ├── layer_parameters.py      # Layer-to-hardware mapping
    ├── iact_stream_mapper.py    # Input activation streams
    ├── wght_stream_mapper.py    # Weight streams
    ├── psum_stream_mapper.py    # Partial sum streams
    └── conv_mapper.py           # Convolution orchestration

Framework Converters
--------------------

PyTorch Converter
~~~~~~~~~~~~~~~~~

**Module**: ``pytorch2model.py``

**Functionality**:

- Extracts layer definitions from PyTorch ``nn.Module``
- Converts weight tensors from PyTorch layout to OpenEye layout
- Supports quantized and non-quantized models
- Performs shape inference through network

**Key Classes**:

.. code-block:: python

    class PyTorchLayer:
        """Base class for PyTorch layer representation"""
        name: str
        layer_type: str
        input_shape: tuple
        output_shape: tuple

    class PyTorchConv2d(PyTorchLayer):
        """Convolutional layer with weights, biases, parameters"""
        weights: np.ndarray      # [out_ch, in_ch, kh, kw]
        biases: np.ndarray       # [out_ch]
        kernel_size: tuple
        strides: tuple
        padding: str

    class PyTorchModel:
        """Container for complete model"""
        layers: List[PyTorchLayer]
        input_shape: tuple
        output_shape: tuple

**Weight Layout Conversion**:

PyTorch stores convolution weights as ``[out_channels, in_channels, kernel_h, kernel_w]``, which is directly compatible with OpenEye's expected format.

ONNX Converter
~~~~~~~~~~~~~~

**Module**: ``onnx2model.py``

**Functionality**:

- Parses ONNX graph representation
- Extracts initializers (weights and biases)
- Tracks tensor shapes through value_info
- Supports regular and QLinear quantized operations

**Key Classes**:

.. code-block:: python

    class ONNXLayer:
        """Base class for ONNX layer representation"""
        name: str
        op_type: str  # Conv, Gemm, MaxPool, etc.
        inputs: List[str]
        outputs: List[str]
        attributes: Dict[str, Any]

    class ONNXModel:
        """Container for ONNX model graph"""
        layers: List[ONNXLayer]
        initializers: Dict[str, np.ndarray]
        value_info: Dict[str, TensorShape]

**Graph Traversal**:

ONNX models are represented as computational graphs. The converter performs topological traversal to extract sequential layer ordering.

TFLite Converter
~~~~~~~~~~~~~~~~

**Module**: ``tflite2model.py``

**Functionality**:

- Reads TensorFlow Lite flatbuffer format
- Extracts quantized tensors
- Handles TFLite-specific quantization parameters
- Supports both quantized and float32 models

**Quantization Handling**:

TFLite uses per-tensor quantization with scale and zero-point:

.. math::

   \text{real\_value} = (\text{quantized\_value} - \text{zero\_point}) \times \text{scale}

The converter extracts these parameters and stores them for hardware configuration.

Layer Adapter System
--------------------

Purpose
~~~~~~~

The layer adapter (``layer_adapter.py``) provides a unified interface for all frameworks, making PyTorch/ONNX/TFLite layers compatible with the existing ``LayerParameters`` module designed for Keras.

Design Pattern
~~~~~~~~~~~~~~

The adapter uses the **Adapter Pattern** to wrap framework-specific layers:

.. code-block:: python

    class KerasLayerAdapter:
        """
        Wraps PyTorch/ONNX layer to look like Keras layer.

        Provides Keras-compatible attributes:
        - .input.shape
        - .output.shape
        - .get_weights()
        - .kernel
        - .bias
        """

        def __init__(self, framework_layer):
            self._layer = framework_layer

        @property
        def input(self):
            """Returns TensorShapeAdapter"""
            return TensorShapeAdapter(self._layer.input_shape)

        @property
        def kernel(self):
            """Returns WeightTensorAdapter"""
            return WeightTensorAdapter(self._layer.weights)

Adapter Classes
~~~~~~~~~~~~~~~

**TensorShapeAdapter**:

Wraps shape tuples to provide Keras-compatible ``.shape`` attribute:

.. code-block:: python

    class TensorShapeAdapter:
        def __init__(self, shape_tuple):
            self._shape = shape_tuple

        @property
        def shape(self):
            return self._shape

**WeightTensorAdapter**:

Wraps NumPy arrays to provide Keras-compatible tensor interface:

.. code-block:: python

    class WeightTensorAdapter:
        def __init__(self, numpy_array):
            self._array = numpy_array

        @property
        def shape(self):
            return self._array.shape

        def numpy(self):
            return self._array

**KerasLayerAdapter**:

Main adapter that makes framework layers compatible:

.. code-block:: python

    adapted_layers = adapt_layers_for_keras(pytorch_model.layers)

    # Now these work with existing code:
    for layer in adapted_layers:
        input_shape = layer.input.shape
        weights = layer.kernel.numpy()
        # ... existing LayerParameters code ...

Layer Parameters
----------------

**Module**: ``layer_parameters.py``

Purpose
~~~~~~~

The ``LayerParameters`` class maps high-level layer descriptions to hardware constraints and configuration:

- PE allocation (which PEs compute which outputs)
- Memory allocation (GLB, SPAD sizing)
- Data layout (row-stationary dataflow)
- Router configuration
- Precision handling (quantization scales)

Key Responsibilities
~~~~~~~~~~~~~~~~~~~~

1. **Dimension Mapping**:

   - Map layer input/output dimensions to PE array dimensions
   - Calculate required iterations for large layers
   - Determine tiling strategy for memory constraints

2. **Memory Allocation**:

   - Calculate IACT SPAD requirements
   - Calculate WGHT SPAD requirements
   - Calculate PSUM SPAD requirements
   - Verify against hardware limits

3. **Data Layout**:

   - Row-stationary dataflow configuration
   - Weight distribution across PE rows
   - Activation distribution across PE array
   - Partial sum accumulation paths

4. **Precision Configuration**:

   - Extract quantization parameters
   - Configure fixed-point precision
   - Set up scale/shift for post-processing

Stream Mappers
--------------

Input Activation Stream Mapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Module**: ``iact_stream_mapper.py``

**Functionality**:

- Generates input activation data streams for PE array
- Handles sliding window operations
- Supports sparse encoding (overhead bits)
- Configures IACT GLB and SPAD loading

**Key Methods**:

.. code-block:: python

    class ConvIactStreamMapper:
        def create_pe_data_iact_stream(self, layer, input_data):
            """
            Generate IACT stream for convolution.

            Returns:
                - IACT data stream (packed with overhead if sparse)
                - IACT address stream (for sparse addressing)
            """

Weight Stream Mapper
~~~~~~~~~~~~~~~~~~~~

**Module**: ``wght_stream_mapper.py``

**Functionality**:

- Generates weight data streams
- Distributes weights row-wise across PEs
- Supports sparse encoding
- Handles weight reuse across input windows

**Key Methods**:

.. code-block:: python

    class ConvWghtStreamMapper:
        def create_pe_data_wght_stream(self, layer, weights):
            """
            Generate WGHT stream for convolution.

            Returns:
                - WGHT data stream (packed for PARALLEL_MACS)
                - WGHT address stream (for sparse indexing)
            """

Partial Sum Stream Mapper
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Module**: ``psum_stream_mapper.py``

**Functionality**:

- Generates partial sum initialization (biases)
- Configures accumulation paths
- Handles output channel distribution

**Key Methods**:

.. code-block:: python

    class ConvPsumStreamMapper:
        def create_pe_data_psum_stream(self, layer, biases):
            """
            Generate PSUM stream (bias initialization).

            Returns:
                - PSUM data stream (bias values)
            """

Data Flow Example
-----------------

Complete Model Execution
~~~~~~~~~~~~~~~~~~~~~~~~

End-to-end data flow for a Conv2D layer:

.. code-block:: python

    # 1. Load model (any framework)
    from open_eye.model_loader import load_model
    model = load_model('model.onnx')

    # 2. Adapt layers
    from open_eye.layer_adapter import adapt_layers_for_keras
    adapted_layers = adapt_layers_for_keras(model.layers)

    # 3. Create layer parameters
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

    # 4. Generate data streams
    from open_eye.conv_mapper import ConvMapper
    mapper = ConvMapper(hardware_params)

    for params in layer_params:
        # Generate IACT stream
        iact_stream = mapper.create_iact_stream(params)

        # Generate WGHT stream
        wght_stream = mapper.create_wght_stream(params)

        # Generate PSUM stream (biases)
        psum_stream = mapper.create_psum_stream(params)

        # Configure routers
        router_config = mapper.create_router_config(params)

        # Send to hardware
        # ... hardware execution ...

Memory Management
-----------------

GLB Allocation
~~~~~~~~~~~~~~

Global Buffer (GLB) allocation strategy:

.. code-block:: text

    GLB Organization (per cluster):
    ┌────────────────────────────────┐
    │ IACT_GLB: 512 words × 8-bit    │  ← Input activations
    ├────────────────────────────────┤
    │ WGHT: Direct forward (no GLB)  │  ← Weights (direct to PE)
    ├────────────────────────────────┤
    │ PSUM_GLB: 768 words × 20-bit   │  ← Partial sums
    └────────────────────────────────┘

**Allocation Rules**:

1. IACT data loaded in sliding windows
2. Weights bypass GLB, go directly to PE SPADs
3. PSUMs accumulated in GLB before writeback

SPAD Allocation
~~~~~~~~~~~~~~~

Per-PE scratchpad allocation (default configuration):

.. code-block:: text

    PE Memory (per PE):
    ┌────────────────────────────────┐
    │ IACT_Addr_SPad: 9 × 4-bit      │  ← Sparse addressing
    ├────────────────────────────────┤
    │ IACT_Data_SPad: 16 × 12-bit    │  ← Input activations
    ├────────────────────────────────┤
    │ WGHT_Addr_SPad: 16 × 7-bit     │  ← Weight addressing
    ├────────────────────────────────┤
    │ WGHT_Data_SPad: 96 × 24-bit    │  ← Weights (stationary)
    ├────────────────────────────────┤
    │ PSUM_SPad: 32 × 20-bit         │  ← Partial sums
    └────────────────────────────────┘
    Total: ~410.5 bytes per PE

**Note**: In dense mode (``SPARSITY_EN=0``), address SPADs are excluded, saving ~144 bits per PE.

Quantization Support
--------------------

Quantization Flow
~~~~~~~~~~~~~~~~~

The software stack supports INT8 quantization:

.. code-block:: text

    Float32 Model
         │
         ▼
    Quantization Tool
    (Framework-specific)
         │
         ▼
    INT8 Quantized Model
         │
         ▼
    OpenEye Converter
         │
         ▼
    Hardware Configuration
    (scale, zero_point, etc.)

Per-Framework Support
~~~~~~~~~~~~~~~~~~~~~

**PyTorch**:

.. code-block:: python

    import torch
    from torch.ao.quantization import quantize_dynamic

    model_quantized = quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )

**ONNX**:

.. code-block:: python

    from onnxruntime.quantization import quantize_static

    quantize_static(
        'model.onnx',
        'model_int8.onnx',
        calibration_data_reader
    )

**TFLite**:

.. code-block:: python

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized = converter.convert()

Design Patterns
---------------

Adapter Pattern
~~~~~~~~~~~~~~~

Used in ``layer_adapter.py`` to make framework-specific layers compatible with Keras-expecting code without modifying the core codebase.

**Benefits**:

- Backward compatibility
- Clean separation of concerns
- Easy to extend for new frameworks

Strategy Pattern
~~~~~~~~~~~~~~~~

Used in stream mappers to handle different layer types:

.. code-block:: python

    class StreamMapper:
        def get_mapper_for_layer(self, layer_type):
            mappers = {
                'Conv2D': ConvIactStreamMapper,
                'Dense': DenseIactStreamMapper,
                'MaxPool': PoolIactStreamMapper
            }
            return mappers[layer_type]()

Factory Pattern
~~~~~~~~~~~~~~~

Used in ``model_loader.py`` to create appropriate converter based on format:

.. code-block:: python

    def load_model(path, format=None):
        if format is None:
            format = detect_model_format(path)

        loaders = {
            'pytorch': load_pytorch_model,
            'onnx': load_onnx_model,
            'tflite': load_tflite_model,
            'keras': load_keras_model
        }

        return loaders[format](path)

Performance Considerations
--------------------------

Conversion Overhead
~~~~~~~~~~~~~~~~~~~

Model conversion is performed once at deployment time:

.. list-table:: Conversion Time
   :header-rows: 1
   :widths: 20 30 30

   * - Framework
     - Small Model (<10 layers)
     - Large Model (50+ layers)
   * - PyTorch
     - <100ms
     - ~500ms
   * - ONNX
     - <200ms
     - ~1s
   * - TFLite
     - <50ms
     - ~300ms

Memory Efficiency
~~~~~~~~~~~~~~~~~

The adapter pattern adds minimal overhead:

- **TensorShapeAdapter**: ~40 bytes per layer
- **WeightTensorAdapter**: Zero-copy (wraps existing NumPy array)
- **KerasLayerAdapter**: ~200 bytes per layer

Total adapter overhead: ~250 bytes per layer (~12.5 KB for 50-layer model)

Limitations
-----------

Current Limitations
~~~~~~~~~~~~~~~~~~~

1. **No skip connections** (ResNet-style) - Sequential models only
2. **No depthwise separable convolutions** - Standard Conv2D only
3. **BatchNorm must be fused** - Not supported as standalone layer
4. **Fixed PE array size** - Compile-time configuration only
5. **No dynamic shapes** - Input dimensions must be known at conversion

Unsupported Operations
~~~~~~~~~~~~~~~~~~~~~~

Operations that will generate errors:

- Concatenation (except via sequential stacking)
- Split/Branch operations
- Attention mechanisms
- Recurrent layers (LSTM, GRU)
- 3D convolutions
- Dilated convolutions (dilation > 1)

Future Enhancements
-------------------

Planned features for software integration:

1. **Dynamic shape support** - Runtime shape inference
2. **Skip connection handling** - ResNet/DenseNet support
3. **Depthwise convolution** - MobileNet support
4. **Auto-optimization** - Automatic layer fusion and optimization
5. **Model compression** - Pruning and knowledge distillation integration
6. **Mixed precision** - INT4/INT8/INT16 mixed precision support

See Also
--------

- :ref:`multi_framework_support` - Multi-framework usage guide
- :ref:`architecture` - Overall architecture
- :ref:`sparsity_modes` - Sparse and dense mode support
- :ref:`configuration` - Hardware configuration details
