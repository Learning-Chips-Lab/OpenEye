.. _dma_regmap:

DMA Register Map System
=======================

Overview
--------

The DMA Register Map is a critical component of the OpenEye accelerator that enables efficient transfer of layer configuration parameters from software to hardware. This system implements a bit-packing protocol that reduces DMA bus traffic by ~90% compared to sending individual parameters.

The Problem
-----------

Hardware Challenge
~~~~~~~~~~~~~~~~~~

Each neural network layer requires approximately 40 configuration parameters to control execution:

.. list-table:: Layer Configuration Parameters
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter Category
     - Count
     - Examples
   * - Execution Control
     - 12
     - ``wght_cycles_reg``, ``stride_x_reg``, ``stride_y_reg``, ``skipIact_reg``
   * - Buffer Management
     - 8
     - ``iact_channels_per_pe``, ``fc_size_reg``, ``iact_size_x``, ``iact_size_y``
   * - Advanced Control
     - 17
     - ``store_in_psum``, ``max_pooling``, ``filters_reg``, ``needed_x_cls_reg``
   * - Layer Type Flags
     - 2
     - ``send_data_out``, ``needed_iact_buffer_words_reg``

**Total**: 39+ parameters per layer

Efficiency Problem
~~~~~~~~~~~~~~~~~~

Sending these parameters individually would require:

.. code-block:: text

    40 parameters × 1 DMA transaction each = 40 bus transactions per layer

    For a 10-layer network:
      40 transactions/layer × 10 layers = 400 total DMA transactions

    Each DMA transaction has overhead:
      - Address setup: ~2 cycles
      - Bus arbitration: ~1 cycle
      - Data transfer: 1 cycle
      - Acknowledge: ~1 cycle
      ──────────────────────────────
      Total: ~5 cycles per parameter

    Total cycles: 400 transactions × 5 cycles = 2000 cycles

This represents significant latency before layer execution can begin.

The Solution
------------

Bit-Packing Protocol
~~~~~~~~~~~~~~~~~~~~

The register map system packs multiple parameters into 64-bit DMA words:

.. code-block:: text

    Traditional Approach:            Bit-Packed Approach:
    ┌─────────────────┐             ┌───────────────────────────────┐
    │ Transaction 1   │             │ Transaction 1 (64 bits)       │
    │ wght_cycles: 9  │             │ ┌────┬──┬──┬─┬─┬─┬────┬────┐ │
    └─────────────────┘             │ │wgt │sx│sy│i│w│p│psum│kern│ │
    ┌─────────────────┐             │ │cyc │  │  │a│g│s│dly │_sz │ │
    │ Transaction 2   │      ══>    │ │8bit│3b│3b│1│1│1│4bit│4bit│ │
    │ stride_x: 1     │             │ └────┴──┴──┴─┴─┴─┴────┴────┘ │
    └─────────────────┘             │  ...more params... (64 bits)  │
    ┌─────────────────┐             └───────────────────────────────┘
    │ Transaction 3   │             ┌───────────────────────────────┐
    │ stride_y: 1     │             │ Transaction 2 (64 bits)       │
    └─────────────────┘             │  ...buffer params...          │
    ...                             └───────────────────────────────┘
    (40 transactions)               (4 transactions total)

**Result**: 40 transactions → 4 transactions (90% reduction)

Architecture
------------

Component Overview
~~~~~~~~~~~~~~~~~~

The register map system consists of four components:

.. code-block:: text

    ┌────────────────────────────────────────────────────────┐
    │                   regmap.yaml                          │
    │    (Human-editable specification)                      │
    │    - Register names and bit widths                     │
    │    - Order determines packing                          │
    └─────────────────┬──────────────────────────────────────┘
                      │
                      ▼
    ┌────────────────────────────────────────────────────────┐
    │                 generator.py                           │
    │    (Code generator)                                    │
    │    - Calculates bit positions                          │
    │    - Determines transmission boundaries                │
    │    - Generates 3 output files                          │
    └────┬─────────────┬─────────────┬────────────────────────┘
         │             │             │
         ▼             ▼             ▼
    ┌─────────┐  ┌──────────┐  ┌─────────────┐
    │regmap_  │  │regmap_   │  │dma_storage.v│
    │params.vh│  │pack.py   │  │             │
    └─────────┘  └──────────┘  └─────────────┘
         │             │             │
         └─────────────┴─────────────┘
                      │
            Used by hardware and software

YAML Specification (regmap.yaml)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The YAML file defines the register layout:

.. code-block:: yaml

    dma_bitwidth: 64
    registers:
      # Execution control parameters
      - {name: wght_cycles_reg, width: 8}
      - {name: stride_x_reg, width: 3}
      - {name: stride_y_reg, width: 3}
      - {name: skipIact_reg, width: 1}
      - {name: skipWght_reg, width: 1}

      # Buffer management
      - {name: iact_channels_per_pe, width: 8}
      - {name: fc_size_reg, width: 12}

      # Advanced control
      - {name: store_in_psum, width: 1}
      - {name: max_pooling, width: 1}

      # ... more parameters ...

**Key Properties**:

- **Order matters**: Parameters are packed in the order specified
- **Width expressions**: Can use expressions like ``ceil(log2(CLUSTER_ROWS))``
- **No gaps**: Parameters are packed tightly with no padding bits

Generator (generator.py)
~~~~~~~~~~~~~~~~~~~~~~~~

The generator reads the YAML and produces three files:

**Algorithm**:

.. code-block:: python

    transmissions = []
    current_transmission = []
    bit_position = 0

    for register in yaml_registers:
        if bit_position + register.width > 64:
            # Start new transmission
            transmissions.append(current_transmission)
            current_transmission = []
            bit_position = 0

        current_transmission.append({
            'name': register.name,
            'width': register.width,
            'position': bit_position
        })
        bit_position += register.width

    # Generate output files using transmission layout

**Execution**:

.. code-block:: bash

    # Generate for main codebase
    python generator.py /path/to/regmap_dir

    # Generate for test environment
    python generator.py /path/to/test/cocotb_fpga

Generated Files
~~~~~~~~~~~~~~~

1. **regmap_params.vh** (Verilog Parameters)

   Defines bit positions for Verilog extraction:

   .. code-block:: verilog

       parameter DMA_BITWIDTH = 64;
       parameter TRANSMISSIONS = 4;

       // Transmission 0 Offsets
       parameter PARAMETER_POS_0_0 = 0;              // wght_cycles_reg
       parameter PARAMETER_POS_0_1 = PARAMETER_POS_0_0 + 8;   // stride_x_reg
       parameter PARAMETER_POS_0_2 = PARAMETER_POS_0_1 + 3;   // stride_y_reg
       // ...

2. **regmap_pack.py** (Python Pack/Unpack)

   Provides functions for software to pack parameters:

   .. code-block:: python

       REGISTERS = [
           {'name': 'wght_cycles_reg', 'width': 8, 'trans': 0, 'pos': 0},
           {'name': 'stride_x_reg', 'width': 3, 'trans': 0, 'pos': 8},
           # ...
       ]

       def pack_registers(values):
           """Pack parameter dict into 64-bit words"""
           words = [0] * TRANSMISSIONS
           for reg in REGISTERS:
               val = values[reg['name']]
               mask = (1 << reg['width']) - 1
               words[reg['trans']] |= (val & mask) << reg['pos']
           return words

       def unpack_registers(words):
           """Unpack 64-bit words into parameter dict"""
           values = {}
           for reg in REGISTERS:
               mask = (1 << reg['width']) - 1
               values[reg['name']] = (words[reg['trans']] >> reg['pos']) & mask
           return values

3. **dma_storage.v** (Verilog Unpacking Module)

   Hardware module that receives DMA data and extracts parameters:

   .. code-block:: verilog

       module dma_storage (
           input  wire clk_i,
           input  wire rst_ni,
           input  wire write_en,
           input  wire [$clog2(TRANSMISSIONS)-1:0] write_addr,
           input  wire [DMA_BITWIDTH-1:0] dma_data_i,
           output reg [7:0] wght_cycles_reg,
           output reg [2:0] stride_x_reg,
           // ... all other parameters ...
       );

       always @(posedge clk_i, negedge rst_ni) begin
           if (!rst_ni) begin
               // Reset all registers
               wght_cycles_reg <= 8'd0;
               stride_x_reg <= 3'd0;
               // ...
           end else if (write_en) begin
               case (write_addr)
                   0: begin  // Transmission 0
                       wght_cycles_reg <= dma_data_i[PARAMETER_POS_0_0 + 7 : PARAMETER_POS_0_0];
                       stride_x_reg    <= dma_data_i[PARAMETER_POS_0_1 + 2 : PARAMETER_POS_0_1];
                       stride_y_reg    <= dma_data_i[PARAMETER_POS_0_2 + 2 : PARAMETER_POS_0_2];
                       // ...
                   end
                   // ... other transmissions ...
               endcase
           end
       end
       endmodule

Data Flow
---------

Complete Flow for One Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following diagram shows how layer parameters flow from Python to hardware:

.. code-block:: text

    ┌────────────────────────────────────────────────────┐
    │ Python (layer_parameters.py)                       │
    │                                                    │
    │ 1. Calculate layer parameters:                    │
    │    params = {                                      │
    │        'wght_cycles_reg': 9,                       │
    │        'stride_x_reg': 1,                          │
    │        'stride_y_reg': 1,                          │
    │        'skipIact_reg': 0,                          │
    │        ...  (40 parameters total)                  │
    │    }                                               │
    └─────────────────┬──────────────────────────────────┘
                      │
                      ▼
    ┌────────────────────────────────────────────────────┐
    │ Python (regmap_pack.py)                            │
    │                                                    │
    │ 2. Pack parameters into 64-bit words:              │
    │    from regmap_pack import pack_registers          │
    │    words = pack_registers(params)                  │
    │                                                    │
    │    Result: [0x0000012345678901,  # Trans 0        │
    │             0xABCDEF0123456789,  # Trans 1        │
    │             0x9876543210ABCDEF,  # Trans 2        │
    │             0x0000000000000123]  # Trans 3        │
    └─────────────────┬──────────────────────────────────┘
                      │
                      ▼
    ┌────────────────────────────────────────────────────┐
    │ DMA Controller / Testbench                         │
    │                                                    │
    │ 3. Write each word over DMA:                       │
    │    for addr, word in enumerate(words):             │
    │        dma_write(addr, word)                       │
    │                                                    │
    │    Signals:                                        │
    │      write_en = 1                                  │
    │      write_addr = 0, 1, 2, 3  (transmission ID)    │
    │      dma_data_i = 64-bit word                      │
    └─────────────────┬──────────────────────────────────┘
                      │
                      ▼
    ┌────────────────────────────────────────────────────┐
    │ Hardware (dma_storage.v)                           │
    │                                                    │
    │ 4. Extract parameters using bit slicing:           │
    │    always @(posedge clk_i) begin                   │
    │        case (write_addr)                           │
    │            0: begin                                │
    │                wght_cycles_reg <= dma_data_i[7:0]; │
    │                stride_x_reg <= dma_data_i[10:8];   │
    │                ...                                 │
    │            end                                     │
    │        endcase                                     │
    │    end                                             │
    │                                                    │
    │ Output: Individual register values                 │
    └─────────────────┬──────────────────────────────────┘
                      │
                      ▼
    ┌────────────────────────────────────────────────────┐
    │ Other Hardware Modules                             │
    │ (PE_cluster, GLB_cluster, etc.)                    │
    │                                                    │
    │ 5. Read configuration registers:                   │
    │    - PE array uses stride_x_reg, stride_y_reg      │
    │    - GLB uses buffer size registers                │
    │    - Router uses skip flags                        │
    │    - Control logic uses layer type flags           │
    └────────────────────────────────────────────────────┘

Bit-Packing Details
-------------------

Packing Algorithm
~~~~~~~~~~~~~~~~~

Parameters are packed using bitwise operations:

.. code-block:: python

    # Packing (Python)
    word = 0
    for param in transmission_params:
        value = param_values[param.name]
        mask = (1 << param.width) - 1      # Create width-bit mask
        masked_value = value & mask         # Mask to width bits
        word |= masked_value << param.pos   # Shift to position and OR

**Example**: Packing 3 parameters into transmission 0:

.. code-block:: python

    # Parameter values
    wght_cycles = 9     # 8 bits
    stride_x = 1        # 3 bits
    stride_y = 1        # 3 bits

    # Packing
    word = 0
    word |= (9 & 0xFF) << 0    # Bits [7:0]   = 0x09
    word |= (1 & 0x07) << 8    # Bits [10:8]  = 0x1
    word |= (1 & 0x07) << 11   # Bits [13:11] = 0x1

    # Result: word = 0x0909 (binary: 0000...0001001000001001)
    #                                        ^^^^--^^^^---^^^^
    #                                         sy    sx   wght

Unpacking Algorithm
~~~~~~~~~~~~~~~~~~~

Parameters are extracted using bit slicing:

.. code-block:: verilog

    // Unpacking (Verilog)
    wght_cycles_reg <= dma_data_i[7:0];      // Extract bits [7:0]
    stride_x_reg    <= dma_data_i[10:8];     // Extract bits [10:8]
    stride_y_reg    <= dma_data_i[13:11];    // Extract bits [13:11]

Or in Python for verification:

.. code-block:: python

    # Unpacking (Python)
    word = 0x0909
    wght_cycles = (word >> 0) & 0xFF    # Shift right 0, mask 8 bits → 9
    stride_x    = (word >> 8) & 0x07    # Shift right 8, mask 3 bits → 1
    stride_y    = (word >> 11) & 0x07   # Shift right 11, mask 3 bits → 1

Transmission Layout
~~~~~~~~~~~~~~~~~~~

Typical layout for OpenEye accelerator:

**Transmission 0** (Core execution parameters):

.. code-block:: text

    Bit Position:  0      8  11  14 15 16 17   21   25   29      37      45         63
                   │──────│──│──│─│─│─│───│───│───│───────│───────│─────────────│
    Parameters:    │wght_ │sx│sy│i│w│p│psum│ker│ker│x_lines│needed_│needed_cycles│
                   │cycles│  │  │a│g│s│_dly│_pe│_sz│  _reg │wght_cy│    _reg     │
                   │ _reg │  │  │c│h│u│    │_cl│   │       │cles_rg│  [7:0]      │
    Bit Width:     │  8   │3 │3 │1│1│1│ 4  │ 4 │ 4 │  8    │  8    │    8        │
                   └──────┴──┴──┴─┴─┴─┴────┴───┴───┴───────┴───────┴─────────────┘

**Transmission 1** (Buffer and sizing):

.. code-block:: text

    Bit Position:  0         8        16           28       36       44            55   60
                   │────────│────────│───────────│───────│────────│────────────│───│───│
    Parameters:    │iact_buf│iact_ch │fc_size_reg│n_cycle│iact_sz │iact_sz_y   │iad│ker│
                   │_max_cyc│_per_pe │           │ [17:8]│ _x     │            │_nc│_pc│
    Bit Width:     │   8    │   8    │    12     │   8   │   8    │     11     │ 5 │ 4 │
                   └────────┴────────┴───────────┴───────┴────────┴────────────┴───┴───┘

**Transmission 2** (Advanced control):

.. code-block:: text

    Contains 17 parameters including flags (max_pooling, store_in_psum),
    buffer configuration, and GLB addressing parameters.

**Transmission 3** (Final configuration):

.. code-block:: text

    Contains output flags and buffer word count.

Usage Examples
--------------

Generating the Register Map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When adding a new parameter or modifying widths:

.. code-block:: bash

    # 1. Edit regmap.yaml
    vim hdl/config/regmap.yaml

    # 2. Add your new parameter
    #    registers:
    #      ...
    #      - {name: my_new_param_reg, width: 6}

    # 3. Regenerate files
    cd src/open_eye
    python generator.py ../../hdl/config

    # Files updated:
    #   - hdl/config/include/regmap_params.vh
    #   - hdl/config/dma_storage.v
    #   - src/open_eye/regmap_pack.py

Using in Python Code
~~~~~~~~~~~~~~~~~~~~

From layer_parameters.py or test code:

.. code-block:: python

    from open_eye.regmap_pack import pack_registers, unpack_registers

    # Prepare parameter dictionary
    params = {
        'wght_cycles_reg': layer.weight_cycles,
        'stride_x_reg': layer.stride[0],
        'stride_y_reg': layer.stride[1],
        'skipIact_reg': 0 if layer.needs_iact else 1,
        # ... all 40 parameters ...
    }

    # Pack into DMA words
    dma_words = pack_registers(params)
    # Result: [0x..., 0x..., 0x..., 0x...]  (4 words)

    # Send to hardware
    for trans_id, word in enumerate(dma_words):
        dut.write_en.value = 1
        dut.write_addr.value = trans_id
        dut.dma_data_i.value = word
        await RisingEdge(dut.clk_i)

    # Verify (optional)
    readback = unpack_registers(dma_words)
    assert readback['wght_cycles_reg'] == layer.weight_cycles

Using in Verilog Code
~~~~~~~~~~~~~~~~~~~~~

From PE_cluster.v or other hardware modules:

.. code-block:: verilog

    `include "regmap_params.vh"

    module my_hardware_module (
        input wire clk_i,
        input wire [7:0] wght_cycles_reg,    // From dma_storage
        input wire [2:0] stride_x_reg,       // From dma_storage
        // ...
    );

    // Use the parameters
    always @(posedge clk_i) begin
        if (cycle_counter < wght_cycles_reg) begin
            // Process weight cycles
            stride_offset = pixel_x * stride_x_reg;
            // ...
        end
    end

    endmodule

Performance Analysis
--------------------

Bus Traffic Reduction
~~~~~~~~~~~~~~~~~~~~~

Comparison of traditional vs. bit-packed approach:

.. list-table:: DMA Traffic Comparison
   :header-rows: 1
   :widths: 30 25 25 20

   * - Metric
     - Traditional (Individual)
     - Bit-Packed
     - Improvement
   * - Parameters per layer
     - 40
     - 40
     - --
   * - Bits per parameter
     - 64 (padded)
     - 1-18 (actual)
     - Varies
   * - DMA transactions/layer
     - 40
     - 4
     - **90% fewer**
   * - Bits transferred/layer
     - 2560
     - 256
     - **90% less**
   * - Setup overhead/layer
     - 160 cycles
     - 16 cycles
     - **90% less**

For a 10-layer network:

.. code-block:: text

    Traditional:  400 transactions × 5 cycles = 2000 cycles
    Bit-packed:    40 transactions × 5 cycles =  200 cycles

    Speedup: 10× faster parameter loading

Bit Utilization
~~~~~~~~~~~~~~~~

How efficiently are the 64-bit words utilized?

.. code-block:: text

    Transmission 0: 45 bits used / 64 bits = 70% utilization
    Transmission 1: 60 bits used / 64 bits = 94% utilization
    Transmission 2: 63 bits used / 64 bits = 98% utilization
    Transmission 3: 13 bits used / 64 bits = 20% utilization

    Overall: 181 bits used / 256 bits = 71% average utilization

This is excellent considering parameters have varying widths and cannot be split across transmissions.

Timing Analysis
~~~~~~~~~~~~~~~

Critical path considerations:

.. code-block:: text

    DMA Write Path:
    ┌──────────────────────────────────────────────┐
    │ DMA data input → Register                    │  Tcq ~ 0.5ns
    │ (No combinational logic)                     │
    └──────────────────────────────────────────────┘

    Read Path (from other modules):
    ┌──────────────────────────────────────────────┐
    │ Register output → Module input               │  Wire delay ~ 0.1ns
    │ (Direct connection)                          │
    └──────────────────────────────────────────────┘

The bit slicing is purely structural (no gates), so there's zero impact on timing.

Design Considerations
---------------------

Parameter Ordering
~~~~~~~~~~~~~~~~~~

The order of parameters in regmap.yaml affects packing efficiency:

**Good Practice**: Group related parameters together

.. code-block:: yaml

    # Good: Related parameters grouped
    registers:
      # Stride parameters (6 bits total)
      - {name: stride_x_reg, width: 3}
      - {name: stride_y_reg, width: 3}

      # Skip flags (3 bits total)
      - {name: skipIact_reg, width: 1}
      - {name: skipWght_reg, width: 1}
      - {name: skipPsum_reg, width: 1}

**Bad Practice**: Mixing large and small parameters

.. code-block:: yaml

    # Bad: Inefficient packing
    registers:
      - {name: large_param, width: 61}      # Uses almost all of trans 0
      - {name: tiny_flag, width: 1}         # Only 3 bits left in trans 0
      - {name: medium_param, width: 8}      # Forced to trans 1 (wastes trans 0 bits)

Parameter Width Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

Choose widths carefully based on actual range:

.. code-block:: yaml

    # Calculate minimum required bits

    # stride can be 1, 2, or 3 → need ceil(log2(4)) = 2 bits
    # BUT: Allow headroom for stride=4 → use 3 bits
    - {name: stride_x_reg, width: 3}

    # kernel_size ranges 1-9 → need ceil(log2(10)) = 4 bits
    - {name: kernel_size, width: 4}

    # Boolean flags always 1 bit
    - {name: max_pooling, width: 1}

Dynamic Width Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Some widths depend on hardware parameters:

.. code-block:: yaml

    # Width depends on CLUSTER_ROWS parameter
    - {name: needed_y_cls_reg, width: ceil(log2(int(CLUSTER_ROWS)+1))}

    # If CLUSTER_ROWS=4: width = ceil(log2(5)) = 3 bits
    # If CLUSTER_ROWS=8: width = ceil(log2(9)) = 4 bits

The generator evaluates these expressions at generation time.

Limitations and Trade-offs
---------------------------

Current Limitations
~~~~~~~~~~~~~~~~~~~

1. **No parameter spanning**: Parameters cannot split across transmission boundaries

   .. code-block:: text

       ✗ NOT ALLOWED:
       Trans 0: [...........................] [param_start]
       Trans 1: [param_end] [...........................]

   This wastes some bits but simplifies hardware extraction.

2. **Fixed transmission count**: Determined at generation time, not runtime

3. **No compression**: Unused bits in transmissions are wasted (zero-padded)

4. **Sequential packing**: Parameters must be packed in YAML order

Design Trade-offs
~~~~~~~~~~~~~~~~~

.. list-table:: Design Decisions
   :header-rows: 1
   :widths: 30 35 35

   * - Decision
     - Benefit
     - Cost
   * - No parameter spanning
     - Simple hardware extraction
     - ~10% bit wastage
   * - Fixed packing order
     - Deterministic layout
     - Less packing optimization
   * - No compression
     - Zero hardware overhead
     - Unused bits wasted
   * - 64-bit words
     - Standard bus width
     - May not fit all params perfectly

Future Enhancements
-------------------

Potential improvements:

1. **Automatic optimization**: Reorder parameters for best packing
2. **Compression**: Huffman coding for frequently-zero parameters
3. **Delta encoding**: Only send changed parameters between layers
4. **Variable word size**: Allow 32-bit or 128-bit transmissions
5. **Runtime configuration**: Reconfigurable packing based on layer types

See Also
--------

- :ref:`software_integration` - Software architecture overview
- :ref:`architecture` - Overall hardware architecture
- Generator source: ``src/open_eye/generator.py``
- Example YAML: ``hdl/config/regmap.yaml``
- Pack/unpack code: ``src/open_eye/regmap_pack.py``
