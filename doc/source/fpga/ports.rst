.. _fpga_ports:

Ports
=====

Clock and Reset
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 10 50

   * - Port
     - Direction
     - Width
     - Description
   * - ``clk_i``
     - input
     - 1
     - System clock. All FFs are positive-edge triggered
   * - ``rst_ni``
     - input
     - 1
     - Active-low, asynchronous global reset

DMA Input Interface (host → accelerator)
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Port
     - Direction
     - Width
     - Description
   * - ``ready_dma_o``
     - output reg
     - 1
     - High when the module can accept a new DMA word
   * - ``data_dma_i``
     - input
     - ``DMA_BITWIDTH``
     - 64-bit data word from DMA
   * - ``enable_dma_i``
     - input
     - 1
     - Valid signal: ``data_dma_i`` is valid this cycle

The protocol is a simple handshake: a transfer occurs on any cycle where both
``enable_dma_i`` and ``ready_dma_o`` are high. After reset, the FSM waits in ``IDLE``
until the host asserts ``enable_dma_i``.

DMA Output Interface (accelerator → host)
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Port
     - Direction
     - Width
     - Description
   * - ``ready_dma_i``
     - input
     - 1
     - High when the downstream sink can accept data
   * - ``data_dma_o``
     - output reg
     - ``DMA_BITWIDTH``
     - 64-bit result word
   * - ``enable_dma_o``
     - output reg
     - 1
     - Valid signal for ``data_dma_o``
   * - ``last_data_o``
     - output reg
     - 1
     - Pulses high on the last output word of a layer

Debug Outputs
-------------

These signals expose internal SRAM and FSM state for simulation / SignalTap analysis and
are not used in production:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Port
     - Description
   * - ``debug_iact_we/re``
     - Write/read enable to the first iact RAM cell
   * - ``debug_iact_addr[11:0]``
     - Address driven to the first iact RAM cell
   * - ``debug_iact_data_i/o[63:0]``
     - Data driven into / read from first iact RAM cell
   * - ``debug_psum_we/re``
     - Write/read enable to psum buffer cell [0][0][0]
   * - ``debug_psum_addr[11:0]``
     - Address for above
   * - ``debug_psum_data_i/o[39:0]``
     - Data for above
   * - ``debug_skip_iact_o``
     - Reflects ``skipIact_reg`` from ``dma_storage``
   * - ``debug_data_dma_stream_o[63:0]``
     - Registered DMA input word (``data_dma_i_reg``)
   * - ``debug_enable_dma_stream_o``
     - Registered ``enable_dma_i_reg``
   * - ``debug_fsm_cycle_o[3:0]``
     - Low 4 bits of ``fsm_cycle`` counter
   * - ``debug_fsm_current_state[3:0]``
     - Current main FSM state
   * - ``debug_fsm_psum_state[2:0]``
     - Current PSUM FSM state
