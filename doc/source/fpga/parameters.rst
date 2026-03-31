.. _fpga_parameters:

Parameters
==========

Architecture Parameters
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - ``IS_TOPLEVEL``
     - ``1``
     - ``1`` = this is the chip top; enables simulation dump
   * - ``SERIAL``
     - ``1``
     - ``1`` = PE uses serial MAC; ``0`` = parallel
   * - ``PARALLEL_MACS``
     - ``2``
     - Number of MAC units operating in parallel per PE
   * - ``SPARSITY_EN``
     - ``1``
     - ``1`` = sparse mode; ``0`` = dense mode
   * - ``CLUSTER_ROWS``
     - ``8``
     - Rows of PE clusters in the array
   * - ``CLUSTER_COLUMNS``
     - ``2``
     - Columns of PE clusters
   * - ``CLUSTERS``
     - ``CLUSTER_COLUMNS × CLUSTER_ROWS``
     - Total clusters

Data-Width Parameters
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - ``DATA_IACT_BITWIDTH``
     - ``8``
     - Bit width of one iact value (INT8)
   * - ``DATA_PSUM_BITWIDTH``
     - ``20``
     - Bit width of partial sum accumulator
   * - ``DATA_WGHT_BITWIDTH``
     - ``8``
     - Bit width of one weight value
   * - ``DATA_IACT_OVERHEAD``
     - ``4``
     - Sparsity metadata bits per iact word
   * - ``TRANS_BITWIDTH_IACT``
     - ``24``
     - Iact bus width (24 = 3×8-bit values or 2×12-bit)
   * - ``TRANS_BITWIDTH_WGHT``
     - ``24``
     - Weight bus width
   * - ``TRANS_BITWIDTH_PSUM``
     - ``20``
     - PSUM bus width
   * - ``DMA_BITWIDTH``
     - ``64``
     - Width of the DMA AXI-stream interface

Memory Sizing Parameters
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - ``NUM_GLB_IACT``
     - ``3``
     - Iact GLBs per cluster row
   * - ``NUM_GLB_WGHT``
     - ``3``
     - Weight GLBs per cluster (= PE rows per cluster)
   * - ``NUM_GLB_PSUM``
     - ``4``
     - Psum GLBs per cluster (= PE columns per cluster)
   * - ``PES``
     - ``NUM_GLB_PSUM × NUM_GLB_WGHT``
     - PEs per cluster
   * - ``IACT_PER_PE``
     - ``16``
     - Max iact words in PE scratchpad
   * - ``WGHT_PER_PE``
     - ``96``
     - Max weight words in PE scratchpad
   * - ``PSUM_PER_PE``
     - ``32``
     - Max psum words in PE scratchpad
   * - ``IACT_ADDR_PER_PE``
     - ``9``
     - Address words in iact address SPad
   * - ``WGHT_ADDR_PER_PE``
     - ``16``
     - Address words in weight address SPad
   * - ``IACT_MEM_ADDR_WORDS``
     - ``512``
     - IACT GLB depth (words)
   * - ``PSUM_MEM_ADDR_WORDS``
     - ``768``
     - PSUM GLB depth (words)
   * - ``RAM_CELLS``
     - ``32``
     - Number of iact double-buffer SRAM cells
   * - ``RAM_CELLS_WORD_BITWIDTH``
     - ``64``
     - Width of one cell word (bits)
   * - ``RAM_CELLS_ADDR_WIDTH``
     - ``12``
     - Address width of cell (11 bits usable + 1 ping/pong bit)
   * - ``BUFFER_WIDTH``
     - ``12``
     - Address width of psum/weight staging buffers

Router Configuration Parameters
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - ``ROUTER_MODES_IACT``
     - ``6``
     - Bits per iact router entry
   * - ``ROUTER_MODES_WGHT``
     - ``1``
     - Bits per weight router entry
   * - ``ROUTER_MODES_PSUM``
     - ``3``
     - Bits per psum router entry
   * - ``BANO_MODES``
     - ``2``
     - Batch-normalization mode count
   * - ``AF_MODES``
     - ``4``
     - Activation-function mode count

Derived / Computed Parameters
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Parameter
     - Formula
     - Description
   * - ``IACT_WORDS_IN_RAM``
     - ``RAM_CELLS_WORD_BITWIDTH / DATA_IACT_BITWIDTH`` = 8
     - INT8 pixels per 64-bit RAM word
   * - ``WORDS_PER_CYCLE``
     - 2
     - Iact conversion processes 2 words per cycle
   * - ``PSUM_TO_IACT_CYCLES``
     - 2 if ``CLUSTER_COLUMNS×NUM_GLB_PSUM == 4``, else 1
     - Cycles needed per spatial position for psum→iact routing
   * - ``FSM_CEIL_IACT_RTR_CCLS``
     - ``⌈(CLUSTERS×NUM_GLB_IACT) / (DMA_BITWIDTH/ROUTER_MODES_IACT)⌉``
     - DMA words needed for iact router config
   * - ``FSM_CEIL_WGHT_RTR_CCLS``
     - ``⌈(CLUSTERS×NUM_GLB_WGHT) / (DMA_BITWIDTH/ROUTER_MODES_WGHT)⌉``
     - DMA words needed for weight router config
   * - ``FSM_CEIL_PSUM_RTR_CCLS``
     - ``⌈(CLUSTERS×NUM_GLB_PSUM) / (DMA_BITWIDTH/ROUTER_MODES_PSUM)⌉``
     - DMA words needed for psum router config
   * - ``EXTENDEDBITS``
     - ``48 - NUM_GLB_WGHT``
     - Zero-pad width when building ``flat_help_var_send`` mask
