.. _sparsity_modes:

Sparsity Modes
==============

Overview
--------

The OpenEye accelerator supports two operating modes for handling data: **Sparse Mode** (default) with sparsity exploitation, and **Dense Mode** for efficient processing of dense (non-sparse) neural networks. This flexibility is controlled by the ``SPARSITY_EN`` parameter, allowing the hardware to optimize for different network characteristics.

The choice between sparse and dense mode affects:

* Data format and bandwidth requirements (33% reduction in dense mode)
* Hardware resource utilization (address SPADs excluded in dense mode)
* Processing logic (simplified addressing in dense mode)
* Energy consumption (reduced data movement overhead)

Parameter: SPARSITY_EN
----------------------

The ``SPARSITY_EN`` parameter is a compile-time configuration that propagates through the entire hardware hierarchy from the top-level module down to individual Processing Elements (PEs).

**Default Value:** 1 (Sparse Mode enabled)

Sparse Mode (SPARSITY_EN = 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Active:** Sparsity exploitation with Compressed Sparse Column (CSC) format support

**Data Format:**
::

    ┌─────────────┬─────────────┐
    │ Overhead    │  Value      │
    │ (4 bits)    │  (8 bits)   │
    └─────────────┴─────────────┘
    Total: 12 bits per data element

**Hardware Features:**

* Address SPADs instantiated for indirect addressing
* Overhead-based zero-skipping logic active
* CSC format processing with count vectors
* Sparse offset calculation for partial sum addressing

**Use Cases:**

* Pruned neural networks (> 30% sparsity)
* Sparse convolutional layers
* Networks optimized with structured sparsity
* Models with significant zero-valued activations/weights

Dense Mode (SPARSITY_EN = 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Active:** No sparsity exploitation, sequential processing

**Data Format:**
::

    ┌─────────────┐
    │  Value      │
    │  (8 bits)   │
    └─────────────┘
    Total: 8 bits per data element

**Hardware Features:**

* Address SPADs **not instantiated** (hardware resource savings)
* Linear, sequential MAC computation
* No overhead bits in data stream
* Simplified control logic

**Benefits:**

* **33% data bandwidth reduction** (8 bits vs. 12 bits per value)
* **Hardware resource savings** (~25 address SPAD entries per PE eliminated)
* **Simplified addressing** (no sparse offset calculations)
* **Lower power consumption** for dense networks

**Use Cases:**

* Dense neural networks (< 30% sparsity)
* Fully connected layers
* Networks without pruning
* Depthwise separable convolutions (typically dense)

Resource Comparison
-------------------

The following table summarizes the hardware differences between sparse and dense modes:

+----------------------------+------------------------+----------------------+
| Resource                   | Sparse Mode (=1)       | Dense Mode (=0)      |
+============================+========================+======================+
| **IACT Data Width**        | 12 bits (8 + 4)        | 8 bits               |
+----------------------------+------------------------+----------------------+
| **WGHT Data Width**        | 12 bits/MAC (8 + 4)    | 8 bits/MAC           |
+----------------------------+------------------------+----------------------+
| **Address SPADs**          | ✓ Instantiated         | ✗ Excluded           |
+----------------------------+------------------------+----------------------+
| **Overhead per Value**     | 4 bits (33%)           | 0 bits (0%)          |
+----------------------------+------------------------+----------------------+
| **Data Bandwidth**         | 100% (baseline)        | 67% (33% reduction)  |
+----------------------------+------------------------+----------------------+
| **Addressing Logic**       | Overhead-based         | Sequential           |
+----------------------------+------------------------+----------------------+

**Estimated Resource Savings in Dense Mode:**

* **33% less data bandwidth** (8-bit instead of 12-bit transport)
* **~25 SPAD entries saved per PE** (9 iact_addr + 16 wght_addr entries eliminated)
* **Simplified address logic** (no indirect addressing computation)

Hardware Implementation
-----------------------

Module Hierarchy
~~~~~~~~~~~~~~~~

The ``SPARSITY_EN`` parameter propagates through the following hardware modules:

1. **PE.v** (Processing Element) - Line ~475
2. **data_pipeline_iact.v** - Line ~88
3. **data_pipeline_wght.v** - Line ~93
4. **PE_cluster.v** - Line ~135
5. **OpenEye_Cluster.v** - Line ~118
6. **OpenEye_Parallel.v** - Line ~138
7. **OpenEye_FPGA.v** - Line ~110

Each module receives the parameter from its parent and forwards it to child modules, ensuring consistent operation mode throughout the hierarchy.

PE.v Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Processing Element (PE.v) contains the core sparsity mode logic:

**Conditional Data Widths (Lines ~475-485):**

.. code-block:: verilog

    localparam integer IACT_DATA_DATA = SPARSITY_EN ?
        (DATA_IACT_BITWIDTH + DATA_IACT_OVERHEAD) : DATA_IACT_BITWIDTH;

    localparam integer WGHT_DATA_DATA = SPARSITY_EN ?
        ((DATA_WGHT_BITWIDTH + DATA_WGHT_IGNORE_ZEROS) * PARALLEL_MACS) :
        (DATA_WGHT_BITWIDTH * PARALLEL_MACS);

These localparams automatically adjust data widths based on the sparsity mode.

**Generate Blocks for Data Unpacking (Lines ~795-820):**

.. code-block:: verilog

    generate
        if (SPARSITY_EN) begin
            // Sparse mode: Extract overhead and payload
            assign iact_overhead = iact_data_r[IACT_DATA_DATA-1 : DATA_IACT_BITWIDTH];
            assign iact_payload  = iact_data_r[DATA_IACT_BITWIDTH-1 : 0];
        end else begin
            // Dense mode: No overhead, entire word is payload
            assign iact_overhead = 0;
            assign iact_payload  = iact_data_r;
        end
    endgenerate

**Generate Blocks for Psum Addressing (Lines ~870-890):**

.. code-block:: verilog

    generate
        if (SPARSITY_EN) begin
            // Sparse mode: Use overhead bits for sparse offset calculation
            assign psum_address = base_addr + sparse_offset;
        end else begin
            // Dense mode: Linear sequential addressing
            assign psum_address = base_addr + linear_counter;
        end
    endgenerate

**Conditional SPAD Instantiation (Lines ~1827-1880):**

.. code-block:: verilog

    generate
        if (SPARSITY_EN) begin
            // Instantiate iact address SPAD
            SPad_SP #(.DATA(4), .WORDS(IACT_ADDR_WORDS)) iact_addr_SPad (...);

            // Instantiate wght address SPAD
            SPad_SP #(.DATA(7), .WORDS(WGHT_ADDR_WORDS)) wght_addr_SPad (...);
        end else begin
            // Dense mode: Tie outputs to zero
            assign iact_addr_SPad_data_r = 0;
            assign wght_addr_SPad_data_r = 0;
        end
    endgenerate

Software/Test Integration
--------------------------

Python Testbench Support
~~~~~~~~~~~~~~~~~~~~~~~~

The cocotb testbench infrastructure (``test/cocotb_PE/PE_tb.py``) supports both modes:

**Global Variable (Line ~134):**

.. code-block:: python

    sparsity_en = True  # Default sparse mode

**Environment Variable Reading (Line ~193):**

.. code-block:: python

    sparsity_en = (os.environ.get('SPARSITY_EN', '1') == '1')

**generate_spad() Function Enhancement:**

.. code-block:: python

    def generate_spad(data, bitwidth=8, overhead_bits=4, sparsity_en=True, mode='SISD'):
        """
        Generate SPAD data with optional sparsity encoding.

        Args:
            data: Input data values
            bitwidth: Bits per value (default: 8)
            overhead_bits: Sparsity metadata bits (default: 4)
            sparsity_en: Enable sparsity encoding (default: True)
            mode: Packing mode ('SISD' or 'SIMD')

        Returns:
            Packed SPAD data ready for hardware transmission
        """
        spad_data = []

        if sparsity_en:
            # Sparse mode: Pack [overhead | value]
            for idx, value in enumerate(data):
                overhead = calculate_zero_count(data, idx)
                packed_word = (overhead << bitwidth) | (value & ((1 << bitwidth) - 1))
                spad_data.append(packed_word)
        else:
            # Dense mode: Pack raw values only
            for value in data:
                spad_data.append(value & ((1 << bitwidth) - 1))

        return spad_data

**Function Call Updates:**

.. code-block:: python

    # Input activations
    send_iact(iact_data, sparsity_en=sparsity_en)

    # Weights with adjusted offset calculation
    send_wght(wght_data, sparsity_en=sparsity_en, offset=wght_offset)

    # Partial sums (never use sparsity encoding)
    send_bias(psum_data, sparsity_en=False)

Dedicated Dense Mode Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Test File:** ``test/cocotb_PE/test_PE_dense.py``

**Test Configuration:**

.. code-block:: python

    @pytest.mark.parametrize("iact_x, iact_y, wght_x", [
        (3, 2, 8),
        (5, 3, 10),
        (7, 4, 12),
    ])
    def test_PE_dense_mode(iact_x, iact_y, wght_x):
        """
        Test PE with SPARSITY_EN=0 (dense mode).
        """
        # Environment variables
        os.environ['SPARSITY_EN'] = '0'
        os.environ['SPARSE_IACT'] = '0'  # 0% sparsity (all non-zero)
        os.environ['SPARSE_WGHT'] = '0'  # 0% sparsity
        os.environ['IACTSIZE_X'] = str(iact_x)
        os.environ['IACTSIZE_Y'] = str(iact_y)
        os.environ['WGHTSIZE_X'] = str(wght_x)

        # Verilog parameters
        verilog_params = {
            "SPARSITY_EN": 0,
            "DATA_IACT_BITWIDTH": 8,
            "DATA_WGHT_BITWIDTH": 8,
            "PARALLEL_MACS": 2,
        }

        # Run test
        run_cocotb_test(verilog_params)

**Execution:**

.. code-block:: bash

    cd test/cocotb_PE
    pytest test_PE_dense.py -v

Layer Type Compatibility
-------------------------

Convolutional Layers (Conv2D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mappers:** ``ConvIactStreamMapper``, ``ConvWghtStreamMapper``, ``ConvPsumStreamMapper``

**Operation with SPARSITY_EN=0:**

* ✓ Input activations: Raw 8-bit values without overhead
* ✓ Weights: Raw 8-bit values, packed for PARALLEL_MACS
* ✓ Psum addressing: Linear (no sparse offsets)
* ⚠ CALCULATING state logic uses ``wght_addr_SPad_data_r`` (tied to 0 in dense mode)

**Functional Status:** Works correctly for both modes

Dense/Fully Connected Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mappers:** ``DenseIactStreamMapper``, ``DenseWghtStreamMapper``, ``DensePsumStreamMapper``

**Operation with SPARSITY_EN=0:**

* ✓ Input activations: 1D vector, 8-bit values
* ✓ Weights: Row-wise packed, 8-bit values
* ✓ Psum addressing: Linear
* ⚠ Same CALCULATING state logic considerations

**Functional Status:** Works correctly for both modes

Hardware Instantiation Examples
--------------------------------

Sparse Mode (Default)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: verilog

    PE #(
        .SPARSITY_EN(1),               // Sparse mode (or omit, default=1)
        .DATA_IACT_BITWIDTH(8),
        .DATA_WGHT_BITWIDTH(8),
        .DATA_IACT_OVERHEAD(4),        // 4-bit overhead for zero-skipping
        .DATA_WGHT_IGNORE_ZEROS(4),    // 4-bit weight sparsity metadata
        .PARALLEL_MACS(2)
    ) sparse_pe (
        .clk_i(clk),
        .rst_ni(rst_n),
        // ... port connections
    );

Dense Mode
~~~~~~~~~~

.. code-block:: verilog

    PE #(
        .SPARSITY_EN(0),               // Dense mode
        .DATA_IACT_BITWIDTH(8),
        .DATA_WGHT_BITWIDTH(8),
        .PARALLEL_MACS(2)
        // Overhead parameters ignored in dense mode
    ) dense_pe (
        .clk_i(clk),
        .rst_ni(rst_n),
        // ... port connections
    );

Full System Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: verilog

    OpenEye_FPGA #(
        .SPARSITY_EN(0),               // Propagates to all modules
        .PE_ROWS(3),
        .PE_COLUMNS(4),
        .PARALLEL_MACS(2),
        .DATA_IACT_BITWIDTH(8),
        .DATA_WGHT_BITWIDTH(8),
        .DATA_PSUM_BITWIDTH(20)
    ) openeye_dense (
        .clk_i(clk),
        .rst_ni(rst_n),
        // ... system interfaces
    );

Cocotb Testing
--------------

Dense Mode Test Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Environment Variables:**

.. code-block:: bash

    export SPARSITY_EN=0           # Enable dense mode
    export SPARSE_IACT=0           # 0% input activation sparsity
    export SPARSE_WGHT=0           # 0% weight sparsity
    export IACTSIZE_X=3
    export IACTSIZE_Y=2
    export WGHTSIZE_X=8
    export SEED=42

**Makefile Execution:**

.. code-block:: bash

    cd test/cocotb_PE
    make

Sparse Mode Test Environment (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Environment Variables:**

.. code-block:: bash

    export SPARSITY_EN=1           # Enable sparse mode (or omit)
    export SPARSE_IACT=50          # 50% input activation sparsity
    export SPARSE_WGHT=30          # 30% weight sparsity
    export IACTSIZE_X=5
    export IACTSIZE_Y=3
    export WGHTSIZE_X=10

**Pytest Execution:**

.. code-block:: bash

    cd test/cocotb_PE
    pytest test_PE.py -v

Backward Compatibility
----------------------

✓ **Fully Backward Compatible**

* ``SPARSITY_EN`` defaults to 1 (sparse mode)
* All existing tests run without modification
* No behavioral changes when ``SPARSITY_EN=1``
* Hardware synthesis with default parameters produces identical sparse mode behavior

Existing sparse mode tests continue to function correctly:

* ``test/cocotb_PE/test_PE.py`` - Sparse PE tests
* ``test/cocotb_PE_cluster/test_PE_CLUSTER.py`` - Sparse PE cluster tests
* All existing testbenches remain compatible

Implementation Status
---------------------

Fully Implemented Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**HDL Modules:**

* ✓ ``hdl/PE.v``: Conditional data unpacking, psum addressing, SPAD instantiation
* ✓ ``hdl/data_pipeline_iact.v``: SPARSITY_EN parameter propagation
* ✓ ``hdl/data_pipeline_wght.v``: SPARSITY_EN parameter propagation
* ✓ ``hdl/PE_cluster.v``: Parameter forwarding to all PEs
* ✓ ``hdl/OpenEye_Cluster.v``: Parameter forwarding to PE clusters
* ✓ ``hdl/OpenEye_Parallel.v``: Parameter forwarding to clusters
* ✓ ``hdl/OpenEye_FPGA.v``: Top-level parameter configuration

**Test Infrastructure:**

* ✓ ``test/cocotb_PE/PE_tb.py``: ``generate_spad()`` supports dense mode
* ✓ ``test/cocotb_PE/test_PE_dense.py``: Dedicated dense mode tests

**Documentation:**

* ✓ ``SPARSITY_EN_README.md``: Comprehensive implementation documentation
* ✓ ``ANALYSIS_SPARSITY_LAYERS.md``: Layer compatibility analysis

Optional Future Enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Python Stream Mappers:**

* ⚪ ``src/iact_stream_mapper.py``: Skip ``set_sparse_stream()`` when ``sparsity_en=0``
* ⚪ ``src/wght_stream_mapper.py``: Dense mode data packing optimization
* ⚪ ``create_pe_data_*_stream()``: Packing without overhead bits

**Extended Testing:**

* ⚪ ``test/cocotb_PE_cluster/test_PE_CLUSTER.py``: SPARSITY_EN=0 cluster tests
* ⚪ ``test/cocotb_fpga/``: Full system dense mode tests
* ⚪ ``test/cocotb_parallel/``: Multi-cluster dense mode tests

**Hardware Optimization:**

* ⚪ Conditional ``wght_addr_SPad_en_r`` in CALCULATING state (PE.v line ~1378)
* ⚪ Further conditional logic for sparse-specific operations

Known Limitations and Considerations
-------------------------------------

CALCULATING State Logic
~~~~~~~~~~~~~~~~~~~~~~~~

**Issue:** In ``PE.v`` CALCULATING state (line ~1378), the weight address SPAD enable signal is unconditionally activated:

.. code-block:: verilog

    wght_addr_SPad_en_r <= 1;    // Always enabled, even when SPARSITY_EN=0

**Impact:**

* ⚠ Simulation: Functions correctly (SPAD output tied to 0 when not instantiated)
* ⚠ Synthesis: May generate warnings about unused signals
* ✓ Functional: Correct behavior (output=0 when SPAD excluded)

**Recommended Fix:**

.. code-block:: verilog

    wght_addr_SPad_en_r <= SPARSITY_EN ? 1 : 0;  // Conditional enable

Overhead-Based Addressing
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Observation:** When ``SPARSITY_EN=0``, overhead values are always 0, causing the hardware to automatically select the sequential addressing path:

.. code-block:: verilog

    if (iact_oh_delay_1 <= iact_oh_delay_2 + 1) begin
        // Sequential path (taken when overhead=0)
        wght_addr_vec <= wght_addr_vec + 1;
    end else begin
        // Sparse path (not taken when overhead=0)
        wght_addr_vec <= iact_oh_delay_1 + 1;
    end

**Result:** Dense mode automatically uses the correct sequential path without additional logic modifications.

Test Coverage Gaps
~~~~~~~~~~~~~~~~~~~

Current testing focuses on PE-level validation. Extended testing recommended:

* ⚪ Conv2D layers with various kernel sizes (1×1, 3×3, 5×5)
* ⚪ Different stride configurations (stride=1, stride=2)
* ⚪ Multi-channel convolutions (e.g., 32 input channels)
* ⚪ Dense/FC layers with large dimensions (e.g., 128 → 10)
* ⚪ PE cluster-level dense mode operation
* ⚪ Full system integration tests

Performance Characteristics
---------------------------

Dense Mode Benefits
~~~~~~~~~~~~~~~~~~~

**Data Bandwidth Reduction:**

* Sparse mode: 12 bits per value (8-bit data + 4-bit overhead)
* Dense mode: 8 bits per value (raw data only)
* **Bandwidth savings: 33%**

**Memory Footprint Reduction:**

* Address SPADs eliminated: ~25 entries per PE (9 iact + 16 wght)
* Per-PE savings: ~144 bits (9×4 + 16×7 = 36 + 112 = 148 bits)
* For 192-PE system: ~27.6 Kbits total savings

**Power Consumption:**

* Reduced data movement energy (33% less traffic)
* Address SPADs not accessed (zero dynamic power)
* Simplified control logic (lower switching activity)

When to Use Each Mode
~~~~~~~~~~~~~~~~~~~~~~

**Use Sparse Mode (SPARSITY_EN=1) when:**

* Network sparsity > 30-40%
* Pruned/quantized models
* Structured sparsity patterns
* Activation sparsity significant (ReLU-heavy networks)

**Use Dense Mode (SPARSITY_EN=0) when:**

* Network sparsity < 30%
* Fully connected (dense) layers
* Depthwise separable convolutions
* Non-pruned baseline models
* Bandwidth-constrained systems (prioritize data reduction)

Synthesis Considerations
------------------------

FPGA Resource Utilization
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Sparse Mode (SPARSITY_EN=1):**

* Register file usage: Higher (address SPADs instantiated)
* Logic utilization: Higher (overhead processing logic)
* BRAM/distributed RAM: Higher (larger data widths)

**Dense Mode (SPARSITY_EN=0):**

* Register file usage: Lower (no address SPADs)
* Logic utilization: Lower (simplified addressing)
* BRAM/distributed RAM: Lower (8-bit vs. 12-bit data paths)

**Recommendation:** For area-constrained FPGA designs, use ``SPARSITY_EN=0`` to reduce resource usage by ~10-15% per PE.

Timing Considerations
~~~~~~~~~~~~~~~~~~~~~

* Dense mode: Simplified logic may improve maximum clock frequency
* Sparse mode: Overhead calculation may be critical path in some designs
* Both modes: 7-stage pipeline depth remains constant

Design Trade-offs
~~~~~~~~~~~~~~~~~

+------------------------+-------------------------+------------------------+
| Aspect                 | Sparse Mode             | Dense Mode             |
+========================+=========================+========================+
| **Area**               | Higher (address SPADs)  | Lower (no addr SPADs)  |
+------------------------+-------------------------+------------------------+
| **Bandwidth**          | Higher (12-bit data)    | Lower (8-bit data)     |
+------------------------+-------------------------+------------------------+
| **Flexibility**        | Handles sparse/dense    | Dense networks only    |
+------------------------+-------------------------+------------------------+
| **Energy**             | Lower for sparse data   | Lower for dense data   |
+------------------------+-------------------------+------------------------+
| **Complexity**         | Higher (CSC processing) | Lower (sequential)     |
+------------------------+-------------------------+------------------------+

References
----------

* Chen et al., "Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices", IEEE JSSC 2019
* Han et al., "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding", ICLR 2016
* Parashar et al., "SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks", ISCA 2017

See Also
--------

* :ref:`architecture` - Overall architecture overview
* :ref:`PE_cluster` - Processing Element cluster details
* :ref:`dataflow` - Row stationary dataflow description
* :ref:`configuration` - System configuration and data encoding
