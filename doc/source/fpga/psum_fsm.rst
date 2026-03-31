.. _fpga_psum_fsm:

PSUM FSM
========

A separate FSM manages the partial-sum pipeline from compute-trigger through output,
operating concurrently with the main FSM.

States
------

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - State
     - Value
     - Description
   * - ``PSUM_IDLE``
     - 0
     - Idle; also handles bias loading when main FSM is in ``GET_BIAS``
   * - ``WAIT_TO_SEND_READY_SIGNAL``
     - 1
     - Waits for wgt/iact enables to go to 0, then sends ready to PEs
   * - ``CALCULATE_PSUM``
     - 2
     - Feeds bias data from psum buffer into OpenEye_Parallel; waits for
       ``psum_ready_o``
   * - ``PSUM_GET_RESULTS``
     - 3
     - Collects output psums from OpenEye_Parallel into ``psum_buffer_SP``
   * - ``WAIT_FOR_SENDING_RESULTS``
     - 4
     - Waits 2 cycles then decides: send to host or loop to iact
   * - ``PSUM_SEND_RESULTS``
     - 5
     - Streams psum buffer data to DMA output; sequentially iterates over all
       clusters/GLBs/filters
   * - ``SEND_PSUM_TO_IACT``
     - 6
     - Quantizes psums and drives ``quantized_value_reg`` for iact repacking

PSUM FSM Detailed Flow
-----------------------

PSUM_IDLE
~~~~~~~~~

- Clears ``enable_dma_o``, ``psum_buffer_SP_en_w``, etc.
- During ``GET_BIAS`` in main FSM: absorbs bias values into ``psum_buffer_SP``
- On ``compute_reg`` pulse: resets psum buffer addresses; transitions to
  ``WAIT_TO_SEND_READY_SIGNAL``

WAIT_TO_SEND_READY_SIGNAL
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Waits until ``wght_enable_i_reg == 0`` and ``iact_enable_i_oep_w == 0``
  (OpenEye_Parallel finished receiving)
- Counts 16 cycles then asserts ``psum_ready_i_reg`` (all ones) and transitions to
  ``CALCULATE_PSUM``

CALCULATE_PSUM
~~~~~~~~~~~~~~

- Reads bias data from ``psum_buffer_SP`` (via ``psum_buffer_SP_en_r``)
- Drives ``psum_data_i_reg <= psum_buffer_SP_data_r`` and ``psum_enable_i_reg`` to feed
  psums into OpenEye_Parallel
- Increments ``psum_buffer_SP_addr_array`` for each active GLB
- After ``filters_reg`` cycles: transitions to ``PSUM_GET_RESULTS``

PSUM_GET_RESULTS
~~~~~~~~~~~~~~~~

- Captures ``psum_data_o_w`` (from OpenEye_Parallel) into ``psum_buffer_SP`` (via
  ``psum_buffer_SP_en_w``)
- Increments psum buffer addresses as results come in
- Checks ``results_ready`` (AND of ``psum_enable_o`` for all active GLBs)
- When ``fsm_psum_cycle >= filters_reg``:

  - If more iterations remain: go back to ``WAIT_TO_SEND_READY_SIGNAL``; manage
    ``storage_cycles`` to select the correct psum buffer page
  - If all iterations done: go to ``WAIT_FOR_SENDING_RESULTS``

WAIT_FOR_SENDING_RESULTS
~~~~~~~~~~~~~~~~~~~~~~~~~

- If ``send_data_out == 1``: transition to ``PSUM_SEND_RESULTS`` after 2 cycles
- If ``send_data_out == 0``: transition to ``SEND_PSUM_TO_IACT`` (quantize for next
  layer) or ``PSUM_IDLE`` (if ``store_in_psum``)

PSUM_SEND_RESULTS
~~~~~~~~~~~~~~~~~

- Streams psum buffer contents to DMA output
- Iterates over: ``fsm_psum_r`` (GLB), ``fsm_x_cl_psum`` (column),
  ``fsm_y_cl_psum`` (row), ``fsm_psum_cycle`` (filter/row tile)
- ``data_dma_o`` is assigned the 40-bit slice from psum buffer read data
  (2 ``PARALLEL_MACS`` × ``TRANS_BITWIDTH_PSUM`` bits)
- ``enable_dma_o`` asserted when ``ready_dma_i`` is high and not at position 0
- Sets ``last_data`` when the last address is read; returns to ``PSUM_IDLE`` after
  ``last_data_o``

SEND_PSUM_TO_IACT
~~~~~~~~~~~~~~~~~

Reads from psum buffer and applies the quantization formula:

.. code-block:: text

   quantized = (quant_mant[f] * (psum + quant_offset[f])) >>> quant_exp[f]

- Stores 8 quantized bytes into ``quantized_value_reg[0..7]``
- Three hardware paths depending on topology:
  ``CLUSTER_COLUMNS×NUM_GLB_PSUM >= 8`` (large),
  ``CLUSTERS×NUM_GLB_PSUM >= 8`` (medium), or small
- Advances ``psum_buffer_SP_addr_array`` through the address sequence
- ``fsm_psum_limit`` terminates the state after all required outputs
