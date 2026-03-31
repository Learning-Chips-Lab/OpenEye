.. _fpga_processes:

Always Blocks / Processes
==========================

Process 1: DMA Input Registration (lines 248–256)
--------------------------------------------------

**Trigger:** ``posedge clk_i, negedge rst_n``

Registers ``data_dma_i`` and ``enable_dma_i`` into ``data_dma_i_reg`` /
``enable_dma_i_reg`` on each clock. On reset: both cleared to 0.

Process 2: Cycle Counting (lines 514–570)
------------------------------------------

**Trigger:** ``posedge clk_i, negedge rst_n``

Manages the iteration counters used in ``WAIT_FOR_RESULTS`` / ``RECEIVE_PSUMS_TO_IACT``:

- **``single_iteration3``**: single-cycle pulse that fires on the first clock of each new
  iact delivery (detected when ``iact_ready_o_oep_w != all-ones`` and
  ``single_iteration == 0``)
- **``current_cycle``**: increments on each ``single_iteration3`` pulse
- **``iact_router_counter``**: increments after each channel batch finishes; wraps at
  ``needed_y_cls_reg``
- **``iact_cycle_count``**: increments when both channel and row counters wrap; tracks
  which weight-reuse iteration we are on
- Reset all counters when ``fsm_current_state == GET_PARAMETERS`` or ``reset_cycle``
  asserted

Process 3: Iact Converter Parameter Distribution (lines 582–794)
------------------------------------------------------------------

**Trigger:** ``posedge clk_i, negedge rst_n``

Distributes spatial parameters to each ``iact_stream_constructor``:

- Computes ``iact_converter_x``, ``iact_converter_y``, ``iact_converter_c`` by iterating
  over all cluster columns, rows, kernel indices, and y-lines
- On each iteration: writes a 36-bit params word
  ``{row_offset[3:0], x[7:0], y[7:0], iact_size_x[7:0], c[7:0]}`` to
  ``iact_converter_params_reg[col][row]`` and pulses ``iact_converter_en_cfg_reg``
- Manages ``param_array_reg`` bitmask to select which clusters get updated
- During ``GET_WGHT`` or ``GET_IACT``: sweeps over all clusters; during
  ``CONVERT_IACT``: updates in lock-step with the encoding loop

Process 4: Iact Converter Store Enable (lines 797–848)
-------------------------------------------------------

**Trigger:** ``posedge clk_i, negedge rst_n``

Controls ``iact_converter_en_store_reg`` and ``conv_array_reg``:

- ``conv_array_reg`` = bitmask of converters to activate for storing; initialized to
  ``start_param_array`` in ``GET_ROUTER_CONFIG``
- When ``iact_converter_enc_enable`` is asserted: enables all converters whose bit in
  ``conv_array_reg`` is set and advances the bitmask (circular shift for FC mode)
- For FC layers: ``conv_array_reg`` rotates left by 2 each cycle

Process 5: Data-Flow to OpenEye_Parallel (lines 851–996)
----------------------------------------------------------

**Trigger:** ``posedge clk_i, negedge rst_n``

Manages weight streaming and compute trigger:

- When ``send_data_reg`` or ``sending_data`` is high: enters the "send" mode

  1. On first cycle: pulses ``iact_converter_en_enc_reg`` for all converters
  2. When ``wght_sendable`` and PE weight interfaces are ready: starts weight buffer read
     (``wght_buffer_SP_en_r``)
  3. Reads ``wght_cnt + 1`` words from ``wght_buffer_SP`` by advancing
     ``wght_buffer_SP_rd_addr``
  4. On each weight word: builds ``wght_enable_i_reg`` mask indicating which cluster
     weight GLBs should receive data
  5. After all weights sent: if ``current_cycle == 0``, fires ``compute_reg`` once

- On subsequent ``current_cycle`` iterations: re-triggers ``iact_converter_en_enc_reg``
  and optionally rewinds ``wght_buffer_SP_rd_addr`` for weight reuse
- In ``GET_PARAMETERS``: full reset of all send-phase registers; ``wght_sendable <= 1``

Process 6: Main FSM (lines 1090–1964)
--------------------------------------

**Trigger:** ``posedge clk_i, negedge rst_n``

The main case-statement FSM described fully in :ref:`fpga_main_fsm`. Drives all loading
phases, converter control, and transitions between computation modes.

Process 7: Router Mode Configuration (lines 1966–2178)
-------------------------------------------------------

**Trigger:** ``posedge clk_i, negedge rst_n``

Manages ``router_mode_iact``, ``router_mode_wght``, ``router_mode_psum``, and
``psum_choose_i_reg``:

- **During** ``GET_ROUTER_CONFIG``: unpacks DMA words into the three router mode registers
- **On** ``compute_reg``: sets ``psum_choose_i_reg`` based on ``needed_y_cls_reg`` (which
  cluster rows receive psum enables)
- **During** ``WAIT_FOR_RESULTS`` / ``RECEIVE_PSUMS_TO_IACT``: on each
  ``single_iteration3`` pulse, updates ``router_mode_iact`` to advance the iact source
  one row down the cluster array (for the case where data flows through multiple rows),
  and updates ``router_mode_psum`` bit[2] (storage-accumulate flag) rotating through the
  cluster rows

Process 8: PSUM FSM (lines 2181–2731)
--------------------------------------

**Trigger:** ``posedge clk_i, negedge rst_n``

Full case-statement PSUM FSM described in :ref:`fpga_psum_fsm`. Also handles the
``status_reg_enable_reg`` override path that resets the entire PSUM state machine when a
new layer starts.
