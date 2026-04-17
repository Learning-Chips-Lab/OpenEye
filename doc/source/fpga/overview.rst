.. _fpga_overview:

Purpose and Module Hierarchy
============================

Purpose and Overview
--------------------

**File:** ``hdl/OpenEye_FPGA.v``

**License:** SHL-2.1 — © Fachhochschule Dortmund – University of Applied Sciences and Arts
(until 2025), Universität Duisburg-Essen (since 2025)

``OpenEye_FPGA`` is the **top-level FPGA wrapper** for the OpenEye neural-network inference
accelerator. It adapts the core ``OpenEye_Parallel`` module—which was designed for ASIC—to
the constraints of an FPGA platform (limited port count, block-RAM availability, DMA burst
interfaces).

Responsibilities:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Responsibility
     - Mechanism
   * - Receive raw layer parameters, activations, weights, biases, and quantization
       coefficients from a host CPU
     - DMA input port (``data_dma_i`` / ``enable_dma_i`` / ``ready_dma_o``)
   * - Buffer weight words in a single-port SRAM (``wght_buffer_SP``)
     - Controlled by ``wght_buffer_SP_en_w/r``
   * - Buffer raw iact pixels across 32 double-buffered SRAM cells
       (``iact_converter_buffer_SP``)
     - Controlled by ``buffer_SP_en_w/r``
   * - Convert stored raw iact pixels into packed sparse streams per PE cluster
     - ``iact_stream_constructor`` (one per cluster position)
   * - Drive ``OpenEye_Parallel`` (the compute core) with correctly formatted iact, wght,
       and psum data
     - Via ``iact_data_i_oep_w``, ``wght_data_i_w``, ``psum_data_i_reg`` wires/registers
   * - Collect partial sums from ``OpenEye_Parallel``, store them in PSUM SRAMs, apply
       quantization, and optionally route them back as iact for the next layer
     - PSUM FSM + ``psum_buffer_SP`` array
   * - Send final 20-bit PSUM words (or 8-bit quantized activations) back to the host
     - DMA output port (``data_dma_o`` / ``enable_dma_o`` / ``ready_dma_i``)
   * - Perform 2×2 max-pooling on-chip before sending results
     - ``MAXPOOLING_READ/SEND`` states

Module Hierarchy and Instantiated Submodules
--------------------------------------------

.. code-block:: text

   OpenEye_FPGA
   ├── RST_SYNC                              (reset synchronizer, 1 instance)
   ├── dma_storage                           (register-map decoder, 1 instance)
   ├── RAM_SP [×32]   (BUFFER_A)            (iact double-buffer cells)
   ├── iact_stream_constructor [CLUSTER_COLUMNS × CLUSTER_ROWS]   (one per cluster)
   ├── RAM_SP (wght_buffer_SP)              (weight staging buffer)
   ├── RAM_SP [CLUSTER_COLUMNS × CLUSTER_ROWS × NUM_GLB_PSUM/2]  (psum staging buffers)
   └── OpenEye_Parallel                      (compute core)

RST_SYNC
~~~~~~~~

**File:** ``hdl/RST_SYNC.v``

Provides a 2-FF synchronizer for the active-low reset ``rst_ni``. The output ``rst_n`` is
used by all internal sequential logic so that reset de-assertion is always synchronous to
``clk_i``, preventing metastability.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Port
     - Direction
     - Description
   * - ``clk_i``
     - input
     - System clock
   * - ``rst_ni``
     - input
     - Asynchronous, active-low reset from board
   * - ``rst_no``
     - output
     - Synchronized active-low reset

**Operation:** On the falling edge of ``rst_ni``, both synchronizer FFs are forced to ``0``.
On clock edges while ``rst_ni`` is high, the chain shifts a ``1`` through so that ``rst_no``
asserts (goes high) two cycles after the board reset releases.

dma_storage
~~~~~~~~~~~

**File:** ``hdl/dma_storage.v`` (auto-generated from ``hdl/config/regmap.yaml``)

Decodes the first few DMA words received during ``GET_PARAMETERS`` state into named
configuration registers. Acts as a write-only register file with address-based write enable.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Input
     - Description
   * - ``write_en``
     - Pulse to latch one DMA word
   * - ``write_addr[1:0]``
     - Register bank address (0–3)
   * - ``dma_data_i[63:0]``
     - DMA word to decode

Key output registers decoded from the first 4 DMA words (word 0–3):

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Register
     - Width
     - Meaning
   * - ``needed_cycles_reg``
     - 18
     - Total computation cycles expected
   * - ``iact_size_x/y``
     - 8
     - Input feature map spatial dimensions
   * - ``iact_channels_per_pe``
     - 8
     - Channel depth assigned per PE
   * - ``kernels_per_calc``
     - 5
     - Filters computed per cycle batch
   * - ``y_lines_per_calc``
     - 4
     - Output rows computed per batch
   * - ``filters_reg``
     - 6
     - Number of output filters
   * - ``wght_cycles_reg``
     - 8
     - Clock cycles to stream all weights
   * - ``kernel_size``
     - 4
     - Convolution kernel spatial dimension (e.g. 3 for 3×3)
   * - ``stride_x_reg`` / ``stride_y_reg``
     - 3
     - Convolution strides
   * - ``skipIact_reg``
     - 1
     - Skip iact DMA loading phase
   * - ``skipWght_reg``
     - 1
     - Skip weight DMA loading phase
   * - ``skipPsum_reg``
     - 1
     - Skip bias/psum DMA loading phase
   * - ``fully_connected_layer``
     - 1
     - FC-mode flag
   * - ``max_pooling``
     - 1
     - Enable max-pooling post-processing
   * - ``send_data_out``
     - 1
     - Result goes directly out (vs. looped back as next-layer iact)
   * - ``store_in_psum``
     - 1
     - Keep psums internally for further accumulation
   * - ``iact_channels_per_pe_next_layer``
     - 4
     - Channel count for subsequent layer (used when routing psums→iact)
   * - ``needed_psum_storage_cycles_reg``
     - 8
     - How many PSUM buffer passes before final output
   * - ``iact_channel_max_cycles``
     - 8
     - How many iact channel batches per spatial position
   * - ``input_activations``
     - 5
     - Number of input activations per MAC cycle
   * - ``needed_x_cls_reg`` / ``needed_y_cls_reg``
     - 2/4
     - Number of cluster columns/rows active
   * - ``fc_size_reg``
     - 12
     - FC layer input size
   * - ``add_up``
     - 2
     - Extra overlap pixels to pad the iact width
   * - ``iact_x_line_repetitions``
     - 8
     - Number of times a horizontal iact line repeats
   * - ``choose_iact_buffer_input/output``
     - 1
     - Which half of the double-buffer is used for loading vs. reading
   * - ``psum_delay_reg``
     - 4
     - Pipeline delay through PSUM cluster

RAM_SP (×32, iact double-buffer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** ``hdl/RAM_SP.v``

**Instance prefix:** ``BUFFER_A[j_gen].iact_converter_buffer_SP``

32 identical single-port SRAMs form the **iact ping-pong buffer**. Each cell is:

- ``DataWidth = RAM_CELLS_WORD_BITWIDTH = 64`` bits (8 bytes = 8 INT8 pixels)
- ``AddrWidth = RAM_CELLS_ADDR_WIDTH = 12`` bits (4096 addresses per cell; address bit [11]
  selects ping vs. pong half via ``choose_iact_buffer``)
- ``Pipelined = 1`` (read data appears one clock after address)

The 32 cells are addressed in a rotating fashion. During ``GET_IACT``, incoming DMA data is
written to successive cells via ``current_buffer_n``. During ``CONVERT_IACT``, all 32 cells
are read in parallel and their data is fed to the ``iact_stream_constructor`` instances.

iact_stream_constructor (CLUSTER_COLUMNS × CLUSTER_ROWS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** ``hdl/iact_stream_constructor.v``

**Instance array:** ``IACT_CONVERTER_X[cc].IACT_CONVERTER_Y[cr]``

One per cluster position. Reads raw pixel data from the shared 32-cell iact SRAM bank and
produces the packed, sparse-encoded activation streams (``iact_data_o``, ``iact_enable_o``,
``iact_choose_o``) expected by the corresponding PE cluster in ``OpenEye_Parallel``.

Key ports:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Port
     - Direction
     - Description
   * - ``storage_i``
     - input
     - Concatenated read data from all 32 iact RAM cells
   * - ``params``
     - input
     - 36-bit configuration word: ``[35:32]`` row-offset, ``[31:24]`` x-coordinate,
       ``[23:16]`` y-coordinate, ``[15:8]`` iact_size_x, ``[7:0]`` channel-start
   * - ``enable_config``
     - input
     - Latch ``params`` into internal registers
   * - ``enable_store``
     - input
     - Run one store (write into internal FIFO/buffer)
   * - ``enable_converter``
     - input
     - Run one encode cycle (push data to PE)
   * - ``ready_o``
     - output
     - High when converter has drained its internal buffer
   * - ``iact_data_o``
     - output
     - Packed iact words for ``NUM_GLB_IACT`` GLBs
   * - ``iact_enable_o``
     - output
     - Per-GLB valid signals
   * - ``iact_choose_o``
     - output
     - Per-PE source-selection signal

RAM_SP (wght_buffer_SP)
~~~~~~~~~~~~~~~~~~~~~~~

Single-port SRAM buffering all weight words for the current layer.

- ``DataWidth = TRANS_BITWIDTH_WGHT × CLUSTERS × NUM_GLB_WGHT = 24 × 16 × 3 = 1152`` bits
- ``AddrWidth = BUFFER_WIDTH + 1 = 13`` bits

During ``GET_WGHT``, the main FSM assembles one row of weight data from successive DMA words
and writes it at ``wght_buffer_SP_wr_addr``. During ``WAIT_FOR_RESULTS``, the data-flow
process reads from ``wght_buffer_SP_rd_addr`` and drives ``wght_enable_i_reg`` to stream
weights into ``OpenEye_Parallel``.

RAM_SP (psum_buffer_SP, CLUSTER_COLUMNS × CLUSTER_ROWS × NUM_GLB_PSUM/2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An array of single-port SRAMs that buffer partial sums between the compute core and the
PSUM FSM.

- ``DataWidth = TRANS_BITWIDTH_PSUM × 2 = 40`` bits (two 20-bit psum values)
- ``AddrWidth = BUFFER_WIDTH = 12`` bits

Addressed via ``psum_buffer_SP_addr_array[cc][cr][g]``. Used to:

1. Pre-load bias values (received via DMA in ``GET_BIAS``)
2. Feed accumulated psums back into ``OpenEye_Parallel`` during ``CALCULATE_PSUM``
3. Capture output psums from ``OpenEye_Parallel`` in ``PSUM_GET_RESULTS``
4. Supply data for DMA output (``PSUM_SEND_RESULTS``) or quantization
   (``SEND_PSUM_TO_IACT``)

OpenEye_Parallel
~~~~~~~~~~~~~~~~

**File:** ``hdl/OpenEye_Parallel.v``

The actual compute core: a 2D array of PE clusters performing MAC operations on iact/weight
data and accumulating partial sums. ``OpenEye_FPGA`` feeds it all required data and
configuration and collects results.

Relevant connection summary:

.. list-table::
   :header-rows: 1
   :widths: 45 15 40

   * - Signal group
     - Direction
     - Description
   * - ``compute_reg``
     - → core
     - Single-cycle pulse to start a computation batch
   * - ``iact_data_i_oep_w`` / ``iact_enable_i_oep_w`` / ``iact_choose_i_oep_w``
     - → core
     - Activation stream from iact_stream_constructors
   * - ``iact_ready_o_oep_w``
     - ← core
     - Back-pressure signal per iact GLB
   * - ``wght_data_i_w`` / ``wght_enable_i_reg`` / ``wght_ready_o_reg``
     - ↔ core
     - Weight stream from wght_buffer_SP
   * - ``psum_data_i_reg`` / ``psum_enable_i_reg`` / ``psum_ready_o_reg``
     - → core
     - Bias/psum input from psum_buffer_SP
   * - ``psum_data_o_w`` / ``psum_enable_o`` / ``psum_ready_i_reg``
     - ← core
     - Result psum output back to psum_buffer_SP
   * - ``status_reg_enable_reg``
     - → core
     - Enable configuration register writes during ``GET_PARAMETERS``
   * - ``router_mode_iact/wght/psum``
     - → core
     - Router configuration loaded during ``GET_ROUTER_CONFIG``
   * - ``compute_mask_reg_port``
     - → core
     - Bit mask of active PE instances

varlenFIFO
~~~~~~~~~~

**File:** ``hdl/varlenFIFO.v``

A simple circular FIFO with configurable ``DATA_WIDTH`` and ``DEPTH``, ``wr_en`` /
``rd_en`` strobes, a ``new_stream_i`` input to reset pointers, and ``empty`` / ``full``
status flags.

In ``OpenEye_FPGA``, the FIFO wires are initialized but the FIFO instance is not explicitly
shown in the portion of the file read; these signals (``fifo_data_i``, ``fifo_read_i``,
``fifo_write_i``) are used for optional output buffering of DMA results.
