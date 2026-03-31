.. _fpga_internals:

Internal Registers and Wires
=============================

DMA Input Pipeline Registers
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Signal
     - Type
     - Description
   * - ``data_dma_i_reg[63:0]``
     - reg
     - One-cycle registered version of ``data_dma_i``
   * - ``enable_dma_i_reg``
     - reg
     - One-cycle registered version of ``enable_dma_i``

These registers add one pipeline stage, ensuring timing closure on the DMA interface.

Configuration Registers (from dma_storage)
-------------------------------------------

These wires connect to outputs of the ``dma_storage`` module. They are set once per layer
and remain stable during computation:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Signal
     - Description
   * - ``needed_cycles_reg[17:0]``
     - Total iteration count until computation complete
   * - ``needed_x_cls_reg[1:0]``, ``needed_y_cls_reg[3:0]``
     - Active cluster grid size
   * - ``needed_iact_cycles_reg[3:0]``
     - Iact router broadcast cycles per position
   * - ``filters_reg[5:0]``
     - Output filters per PE batch
   * - ``iact_addr_len_reg``, ``wght_addr_len_reg``
     - SPad address lengths
   * - ``kernel_size[3:0]``
     - Kernel spatial size
   * - ``stride_x_reg[2:0]``, ``stride_y_reg[2:0]``
     - Convolution strides
   * - ``skipIact_reg``, ``skipWght_reg``, ``skipPsum_reg``
     - Phase-skip flags
   * - ``wght_cycles_reg[7:0]``
     - Weight streaming cycles
   * - ``input_activations[4:0]``
     - Activations per PE MAC cycle
   * - ``iact_size_x[7:0]``, ``iact_size_y[7:0]``
     - Feature map dimensions
   * - ``iact_channels_per_pe[7:0]``
     - Channels in this PE batch
   * - ``iact_channel_max_cycles[7:0]``
     - Total channel batches
   * - ``kernels_per_calc[4:0]``
     - Filters calculated simultaneously
   * - ``y_lines_per_calc[3:0]``
     - Output rows per calculation
   * - ``iact_x_line_repetitions[7:0]``
     - Repetitions of each x-line
   * - ``fully_connected_layer``
     - FC mode flag
   * - ``max_pooling``
     - Max-pool mode flag
   * - ``store_in_psum``
     - Keep psums for further iterations
   * - ``send_data_out``
     - Send results directly out (vs. loop back)
   * - ``iact_channels_per_pe_next_layer[3:0]``
     - Channel count for next layer
   * - ``needed_psum_storage_cycles_reg[7:0]``
     - PSUM buffer passes
   * - ``fc_size_reg[11:0]``
     - FC layer input length
   * - ``iact_needed_cycles[10:0]``
     - Iact streaming cycle count
   * - ``output_cycles[7:0]``
     - Output buffer read cycles
   * - ``needed_iact_buffer_words_reg``
     - Words in iact stream buffer
   * - ``add_up[2:0]``
     - Extra overlap columns
   * - ``iact_converter_buffer_addr_max_cycles[7:0]``
     - Max address cycles in iact converter
   * - ``choose_iact_buffer_input/output``
     - Double-buffer selector

Hyperparameter Registers (set by main FSM)
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Signal
     - Type
     - Description
   * - ``data_mode_reg``
     - reg
     - Fixed-point vs. floating-point mode
   * - ``fraction_bit_reg[4:0]``
     - reg
     - Fractional bit position
   * - ``padding_reg[3:0]``
     - reg
     - Zero-padding size = ``(kernel_size-1)/2``
   * - ``bano_cluster_mode_reg``
     - reg
     - Batch-norm mode per psum GLB
   * - ``af_cluster_mode_reg[1:0]``
     - reg
     - Activation function mode
   * - ``compute_mask_reg``
     - reg
     - Bitmask of active PEs (one bit per PE per cluster)

Main FSM Counters
-----------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Signal
     - Type
     - Description
   * - ``fsm_current_state[3:0]``
     - reg
     - Current state of the main FSM
   * - ``fsm_last_state[3:0]``
     - reg
     - Previous state (for debugging)
   * - ``fsm_cycle[31:0]``
     - reg
     - Cycle counter within the current state
   * - ``fsm_x_cl[clog2(CLUSTER_COLUMNS)-1:0]``
     - reg
     - Current cluster column being processed
   * - ``fsm_y_cl[clog2(CLUSTER_ROWS)-1:0]``
     - reg
     - Current cluster row being processed
   * - ``fsm_iact_r[clog2(NUM_GLB_IACT)-1:0]``
     - reg
     - Current iact GLB index being loaded
   * - ``fsm_wght_r[clog2(NUM_GLB_WGHT)-1:0]``
     - reg
     - Current weight GLB index being loaded

Cycle Tracking Registers
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Signal
     - Type
     - Description
   * - ``current_cycle[19:0]``
     - reg
     - Counts iact delivery iterations during ``WAIT_FOR_RESULTS``
   * - ``iact_cycle_count[15:0]``
     - reg
     - Sub-counter tracking weight cycling period
   * - ``iact_router_counter[3:0]``
     - reg
     - Counts cluster-row sweeps per iact push
   * - ``single_iteration``
     - reg
     - Flag: currently in the "active part" of one iact delivery
   * - ``single_iteration2``
     - reg
     - Delayed version of ``single_iteration`` (for edge detection)
   * - ``single_iteration3``
     - reg
     - Single-cycle pulse: first cycle of active iact delivery
   * - ``finished_cycles_iact[19:0]``
     - reg
     - Total iact delivery iterations completed
   * - ``finished_cycles_psum[19:0]``
     - reg
     - Total psum output iterations completed
   * - ``reset_cycle``
     - reg
     - Pulse to reset all cycle counters

Iact Double-Buffer Control
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Signal
     - Type
     - Description
   * - ``choose_iact_buffer``
     - reg
     - Selects which half (0/1) of the double-buffer is active
   * - ``current_buffer_n[7:0]``
     - reg
     - Current write cell index (0–31)
   * - ``current_buffer_n_1[7:0]``
     - reg
     - Previous write cell index (for address update pipeline)
   * - ``current_buffer_addr[10:0]``
     - reg
     - Current write address within a cell
   * - ``buffer_SP_en_r_reg[32]``
     - reg array
     - Read enables per iact RAM cell
   * - ``buffer_SP_en_w_reg[32]``
     - reg array
     - Write enables per iact RAM cell
   * - ``buffer_SP_addr_reg[32][10:0]``
     - reg array
     - Addresses per iact RAM cell
   * - ``buffer_SP_data_w_reg[32][63:0]``
     - reg array
     - Write data per iact RAM cell
   * - ``buffer_SP_addr_upper_limit[7:0]``
     - reg
     - Upper boundary of the active cell window
   * - ``buffer_SP_addr_lower_limit[7:0]``
     - reg
     - Lower boundary of the active cell window
   * - ``limit_increase_reg[7:0]``
     - reg
     - How much to advance the boundary per cycle
   * - ``overhang[1:0]`` / ``overhang_delay``
     - reg
     - Extra boundary increment when accumulated fractional part ≥ 1
   * - ``overhang_counter[3:0]``
     - reg
     - Fractional accumulator for overhang detection
   * - ``overhang_discrepancy[3:0]``
     - reg
     - Fractional part of the number of cells per cycle

Iact Converter Control
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Signal
     - Type
     - Description
   * - ``iact_converter_params_reg[CC][CR][35:0]``
     - reg
     - Packed {row_offset, x, y, size_x, channel} config per converter
   * - ``iact_converter_en_cfg_reg[CC][CR]``
     - reg
     - Pulse to latch params into converter
   * - ``iact_converter_en_store_reg[CC][CR]``
     - reg
     - Enable one store step in converter
   * - ``iact_converter_en_enc_reg[CC][CR]``
     - reg
     - Enable one encode step in converter
   * - ``iact_converter_ready_w[CC][CR]``
     - wire
     - Converter has finished current batch
   * - ``iact_converter_max_cycles[7:0]``
     - reg
     - Total y-line cycles including kernel overlap
   * - ``min_standing_cycles[7:0]``
     - reg
     - Minimum cycles a row must remain active
   * - ``iact_converter_cycles[7:0]``
     - reg
     - Current y-position counter
   * - ``iact_converter_buffer_addr_cycles[7:0]``
     - reg
     - Sub-counter for buffer address cycling
   * - ``iact_converter_params_enable``
     - reg
     - Gate signal for param FSM
   * - ``iact_converter_enc_enable``
     - reg
     - Gate signal for encode FSM
   * - ``converters_ready``
     - reg
     - AND of all ``iact_converter_ready_w`` outputs

Iact Converter Traversal State
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Signal
     - Type
     - Description
   * - ``fsm_iact_params[7:0]``
     - reg
     - Countdown: remaining param slots to configure
   * - ``fsm_iact_params_y_line[7:0]``
     - reg
     - Current y-line index in the param sweep
   * - ``fsm_iact_params_kernel[7:0]``
     - reg
     - Current filter kernel index in the param sweep
   * - ``iact_converter_x[7:0]``
     - reg
     - X start coordinate for the current converter config
   * - ``iact_converter_y[7:0]``
     - reg
     - Y start coordinate
   * - ``iact_converter_c[7:0]``
     - reg
     - Channel start index
   * - ``fsm_row[clog2(CLUSTER_ROWS+1)-1:0]``
     - reg
     - Target cluster row for the current param write
   * - ``fsm_row_offset[clog2(CLUSTER_ROWS+1)-1:0]``
     - reg
     - Base row offset cycling within ``needed_y_cls_reg``
   * - ``param_array_reg[CLUSTERS-1:0]``
     - reg
     - Bitmask: which clusters receive a config event
   * - ``conv_array_reg[CLUSTERS-1:0]``
     - reg
     - Bitmask: which clusters receive a store event

Weight Buffer Control
----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Signal
     - Type
     - Description
   * - ``wght_buffer_SP_en_r`` / ``_en_w``
     - reg
     - Read/write enables for the weight SRAM
   * - ``wght_buffer_SP_wr_addr[BUFFER_WIDTH:0]``
     - reg
     - Current write address
   * - ``wght_buffer_SP_rd_addr[BUFFER_WIDTH:0]``
     - reg
     - Current read address
   * - ``wght_buffer_SP_rd_addr_storage[BUFFER_WIDTH:0]``
     - reg
     - Saved read address for rewind after a channel batch
   * - ``wght_buffer_SP_data_w``
     - reg
     - Write data (assembled from two DMA words per cycle)
   * - ``wght_data_i_w``
     - wire
     - Read data (drives ``OpenEye_Parallel`` weight input)
   * - ``wght_cnt[BUFFER_WIDTH:0]``
     - reg
     - Total weight words loaded (minus 1)
   * - ``wght_enable_i_reg[CLUSTERS×NUM_GLB_WGHT-1:0]``
     - reg
     - Per-GLB valid signals for weight stream

Dataflow / Sending State
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Signal
     - Type
     - Description
   * - ``sending_data``
     - reg
     - True while streaming data to OpenEye_Parallel
   * - ``send_data_reg``
     - reg
     - One-cycle pulse to initiate data sending
   * - ``fsm_sending_cycle[12:0]``
     - reg
     - Cycle counter within the send phase
   * - ``wght_sendable``
     - reg
     - Weight stream is permitted to start
   * - ``compute_reg``
     - reg
     - One-cycle ``compute`` pulse for OpenEye_Parallel
   * - ``new_stream``
     - reg
     - Pulse to reset ``varlenFIFO``
   * - ``early_stream_start``
     - reg
     - Host sent data before ``ready_dma_o`` was high

PSUM FSM State and Counters
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Signal
     - Type
     - Description
   * - ``fsm_psum_current_state[3:0]``
     - reg
     - Current state of the PSUM FSM
   * - ``fsm_psum_last_state[3:0]``
     - reg
     - Previous PSUM FSM state
   * - ``fsm_psum_cycle[15:0]``
     - reg
     - Cycle counter within the current PSUM state
   * - ``fsm_x_cl_psum[clog2(CLUSTER_COLUMNS)-1:0]``
     - reg
     - Current output cluster column
   * - ``fsm_y_cl_psum[clog2(CLUSTER_ROWS+1)-1:0]``
     - reg
     - Current output cluster row
   * - ``fsm_psum_r[clog2(NUM_GLB_PSUM)-1:0]``
     - reg
     - Current output GLB index
   * - ``fsm_psum_r_q``, ``fsm_x_cl_psum_q``, ``fsm_y_cl_psum_q``
     - reg
     - One-cycle delayed versions (pipeline compensation)
   * - ``fsm_y_cl_psum_delay1/2/3``
     - reg
     - 3-stage delay chain for ``fsm_y_cl_psum`` (quantization pipeline)
   * - ``psum_buffer_SP_addr_array[CC][CR][g]``
     - reg
     - Per-psum-buffer read/write address
   * - ``psum_buffer_SP_addr_storage``
     - reg
     - Base address for the next psum output page
   * - ``psum_buffer_SP_en_r/w``
     - reg
     - Per-buffer read/write enable
   * - ``psum_enable_i_reg``
     - reg
     - Enable signals fed into OpenEye_Parallel psum input
   * - ``psum_ready_i_reg``
     - reg
     - Ready signals fed into OpenEye_Parallel psum input
   * - ``psum_transmitted``
     - reg
     - Flag: psums have been sent into OpenEye_Parallel
   * - ``results_ready``
     - reg
     - Local flag: all PE outputs are available
   * - ``storage_cycles[7:0]``
     - reg
     - Counts how many iterations have been stored
   * - ``psum_cnt[BUFFER_WIDTH-1:0]``
     - reg
     - Total psum words per output pass
   * - ``start_new_cycle``
     - reg
     - Trigger next computation cycle
   * - ``finished_cycles_psum[19:0]``
     - reg
     - Total psum output iterations done
   * - ``psum_router_set_reg``
     - reg
     - Tracks whether psum router is configured
   * - ``iact_channel_counter_reg[7:0]``
     - reg
     - Channel counter for psum router update
   * - ``last_data`` / ``last_data_reg`` / ``last_data_o``
     - reg
     - Last-data signaling chain

Quantization and Psum→Iact Conversion
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Signal
     - Type
     - Description
   * - ``quant_exp[32][6:0]``
     - reg
     - Quantization exponent per filter (7 bits)
   * - ``quant_mant[32][24:0]``
     - reg
     - Quantization mantissa per filter (25 bits)
   * - ``quant_offset[32][7:0]``
     - reg
     - Per-filter zero-point offset
   * - ``quantized_value_reg[8][7:0]``
     - reg
     - 8 output bytes from the current quantization
   * - ``current_filter[7:0]``
     - reg
     - Index of the filter being quantized
   * - ``psum_to_iact_state``
     - reg
     - Toggle: which half-batch of psums is being quantized

**Quantization formula** (for each filter ``f``, psum value ``p``):

.. code-block:: text

   quantized = (quant_mant[f] * (p + quant_offset[f])) >>> quant_exp[f]

This implements per-channel fixed-point linear quantization.

Max-Pooling Pipeline
---------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Signal
     - Type
     - Description
   * - ``pooling_regs[32][7:0]``
     - reg
     - Accumulates running max per output pixel
   * - ``pooling_stage_1[8][7:0]``
     - reg
     - Stage 1: raw 2×2 input pixels
   * - ``pooling_stage_2[4][7:0]``
     - reg
     - Stage 2: max of adjacent pairs
   * - ``pooling_stage_3[2][7:0]``
     - reg
     - Stage 3: max of pairs of pairs
   * - ``pooling_stage_4[7:0]``
     - reg
     - Final max (currently unused in some paths)

**Pipeline operation:**

.. code-block:: text

   stage_2[a] = max(stage_1[2a], stage_1[2a+1])   // for a=0..3
   stage_3[a] = max(stage_2[2a], stage_2[2a+1])   // for a=0..1
   result = max(stage_3[0], stage_3[1])            // scalar max

Router Mode Registers
----------------------

.. list-table::
   :header-rows: 1
   :widths: 55 15 30

   * - Signal
     - Type
     - Description
   * - ``router_mode_iact[CLUSTERS×NUM_GLB_IACT×ROUTER_MODES_IACT-1:0]``
     - reg
     - 6-bit config per iact router
   * - ``router_mode_iact_storage[…]``
     - reg
     - Saved initial iact router config (for restoration after row sweep)
   * - ``router_mode_wght[CLUSTERS×NUM_GLB_WGHT×ROUTER_MODES_WGHT-1:0]``
     - reg
     - 1-bit config per weight router
   * - ``router_mode_psum[CLUSTERS×NUM_GLB_PSUM×ROUTER_MODES_PSUM-1:0]``
     - reg
     - 3-bit config per psum router
   * - ``psum_choose_i_reg[CLUSTERS×NUM_GLB_PSUM-1:0]``
     - reg
     - Per-psum-GLB PE-select bitmask

PSUM Send-Phase Bookkeeping
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Signal
     - Type
     - Description
   * - ``psum_sending_counter[7:0]``
     - reg
     - Counts psum words sent to iact buffer
   * - ``sending_clusters[3:0]``
     - reg
     - Number of cluster GLBs active per send step
   * - ``sending_cluster_rows[3:0]``
     - reg
     - Number of cluster rows active per send step
   * - ``iteration_for_kernels_reg[3:0]``
     - reg
     - Iterations needed for all kernel groups
   * - ``psum_cycle_buffer_1/2/3/4[7:0]``
     - reg
     - Nested loop counters: y-line, kernel, channel, output tile
   * - ``pcb_1/2/3[11:0]``
     - reg
     - Base address for psum buffer when each pcb counter wraps
   * - ``fsm_psum_row_offset[3:0]``
     - reg
     - Starting cluster row for psum scanning
   * - ``fsm_psum_limit[16:0]``
     - reg
     - Total fsm_psum_cycle count before ``SEND_PSUM_TO_IACT`` exits

RAM/DMA Write-Back Registers
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Signal
     - Type
     - Description
   * - ``write_dma_en``
     - reg
     - Enable signal for ``dma_storage`` write port
   * - ``write_dma_addr[1:0]``
     - reg
     - Address in ``dma_storage`` (0–3)
   * - ``dma_data_i[63:0]``
     - reg
     - Registered copy of incoming DMA word for ``dma_storage``
   * - ``select_ram_counter[15:0]``
     - reg
     - Current cell index during psum→iact packing
   * - ``ram_counter_storage[15:0]``
     - reg
     - Saved reference cell for each output row
   * - ``select_ram_offset[7:0]``
     - reg
     - Additional offset into cell
   * - ``ram_iact_modulo[7:0]``
     - reg
     - ``iact_size_x % 8`` for handling non-multiples of 8
