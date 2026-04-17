.. _fpga_data_paths:

Key Computations and Data Paths
================================

Iact Loading (host → double-buffer)
-------------------------------------

.. code-block:: text

   enable_dma_i
        │
   data_dma_i ──reg──► data_dma_i_reg
                             │
                       buffer_SP_data_w_reg[current_buffer_n % 32]
                             │
                       RAM_SP[current_buffer_n % 32]
                       addr = {choose_iact_buffer, current_buffer_addr}

64-bit DMA words are written to successive RAM cells, rotating through cells 0–31 with
address ``current_buffer_addr`` incrementing every 32 writes.

Iact Streaming (double-buffer → iact_stream_constructor → OpenEye_Parallel)
-----------------------------------------------------------------------------

All 32 RAM cells are read in parallel every clock during ``CONVERT_IACT``:

.. code-block:: text

   RAM_SP[0..31].data_o ──► buffer_SP_data_r[0..32×64-1]
                                 │
                       iact_stream_constructor[cc][cr].storage_i
                                 │ (format, sparse-encode, select)
                       iact_data_o, iact_enable_o, iact_choose_o
                                 │
                       OpenEye_Parallel.iact_data_i / enable_i / choose_i

Weight Loading (host → wght_buffer_SP)
----------------------------------------

Two 24-bit rows per 64-bit DMA word; assembled into a word covering all cluster rows:

.. code-block:: text

   data_dma_i_reg[23:0]  → wght_buffer_SP_data_w[row][glb]
   data_dma_i_reg[47:24] → wght_buffer_SP_data_w[row + CLUSTER_ROWS][glb]
                                       │
               wght_buffer_SP (written at wght_buffer_SP_wr_addr)

Weight Streaming (wght_buffer_SP → OpenEye_Parallel)
------------------------------------------------------

.. code-block:: text

   wght_buffer_SP.data_o ──► wght_data_i_w ──► OpenEye_Parallel.wght_data_i
   wght_enable_i_reg     ────────────────────► OpenEye_Parallel.wght_enable_i

``wght_enable_i_reg`` is a per-GLB mask: bit ``a + b×NUM_GLB_WGHT`` is set when cluster
row ``a`` is within the range of active rows for the current spatial tile.

Partial Sum Path (OpenEye_Parallel → psum_buffer_SP → host / iact buffer)
---------------------------------------------------------------------------

.. code-block:: text

   OpenEye_Parallel.psum_data_o ──► psum_buffer_SP.data_i   (in PSUM_GET_RESULTS)
   psum_buffer_SP.data_o        ──► psum_buffer_SP_data_r
       │
       ├──► (PSUM_SEND_RESULTS) ──► data_dma_o ──► DMA output
       │
       └──► (SEND_PSUM_TO_IACT)
                quantization:
                q = (quant_mant[f] × (p + quant_offset[f])) >>> quant_exp[f]
                               ──► quantized_value_reg[0..7]
                               ──► buffer_SP_data_w_reg  (iact double-buffer write)

Address Window Advancement During CONVERT_IACT
------------------------------------------------

The 32 iact RAM cells are partitioned into a sliding window that advances one "row" per
``iact_converter_buffer_addr_cycles`` wrap:

.. code-block:: text

   window size = limit_increase_reg (+ overhang when fractional part accumulates ≥ 1)
   upper_limit advances by limit_increase_reg each step
   lower_limit advances by limit_increase_reg (+ overhang_delay) each step

Cells in ``[lower_limit, upper_limit)`` (mod 32) have their address incremented. This
allows the 32 cells to represent a circular sliding window over the 2D feature map,
covering ``kernel_size`` rows at a time.

``start_param_array`` Computation
-----------------------------------

.. code-block:: verilog

   start_param_array = (1 << N) - 1
   where N = ((kernels_per_calc × y_lines_per_calc × ((iact_size_x-1+NUM_GLB_PSUM)/NUM_GLB_PSUM) × NUM_GLB_PSUM)
              + NUM_GLB_PSUM - 1) / NUM_GLB_PSUM

This is the number of clusters that need to receive activation data for one spatial tile.
The initial value of ``param_array_reg`` is set to this and then shifted/rotated as tiles
progress.
