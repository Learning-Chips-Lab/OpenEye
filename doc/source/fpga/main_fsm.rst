.. _fpga_main_fsm:

Main FSM
========

The main FSM (``fsm_current_state``) drives the overall control flow: loading parameters,
activations, weights, biases, running the accelerator, and collecting output.

State Diagram (simplified)
---------------------------

.. code-block:: text

   IDLE
     │ (enable_dma_i)
     ▼
   GET_PARAMETERS ──────────────────────────────────────────────►
     │ (fsm_cycle == param words done)
     ▼
   GET_ROUTER_CONFIG
     │ (fsm_cycle == all router config words done)
     ├─── skipIact ──────────────────────────────►
     │                                            │
     ▼ (!skipIact)                                │
   GET_IACT                                       │
     │ (all iact words received)                  │
     ▼                                            │
   GET_WGHT ◄──────────────────────────────────────
     │ (all weight words received)
     ├─── skipPsum ──────────────────────────────►
     │                                            │
     ▼ (!skipPsum)                                │
   GET_BIAS                                       │
     │ (psum_cnt != 0)                            │
     ▼                                            │
   GET_QUANTIZE ◄──────────────────────────────────
     │ (16 quantize words received)
     ▼
   GET_OFFSET
     │ (4 offset words received)    (max_pooling)
     ├────────────────────────────────────────────► MAXPOOLING_READ → MAXPOOLING_SEND → GET_PARAMETERS
     ▼ (!max_pooling)
   START_CONVERTER
     │ (all converters ready)
     ▼
   CONVERT_IACT
     │ (all channel batches processed)
     ▼
   WAIT_CYCLE
     │ (16 extra cycles)
     ├──── send_data_out ────────────────────────► WAIT_FOR_RESULTS → GET_PARAMETERS
     ▼ (!send_data_out)
   RECEIVE_PSUMS_TO_IACT → (PSUM_IDLE in psum FSM) → GET_PARAMETERS

State Descriptions
------------------

IDLE (0)
~~~~~~~~

Waits for any ``enable_dma_i_reg`` pulse. Transitions to ``GET_PARAMETERS`` and initializes
``fifo`` signals.

GET_PARAMETERS (1)
~~~~~~~~~~~~~~~~~~

Receives the layer configuration. Actions per cycle:

- ``ready_dma_o <= 1`` (accepting data)
- ``status_reg_enable_reg <= 1`` (allows ``OpenEye_Parallel`` to latch config from DMA)
- ``reset_cycle <= 1`` (resets all cycle counters to 0)
- Initializes pooling pipeline registers to -128 (minimum signed 8-bit)
- Initializes quantization registers to 0
- **Cycles 0–3:** ``write_dma_en`` strobed to write 4 DMA words to ``dma_storage`` at
  consecutive addresses. At cycle 3: ``padding_reg <= (kernel_size-1)/2``
- **Cycles 4–N:** Fills ``compute_mask_reg`` one DMA word at a time (64 bits per cycle,
  covering ``PES × CLUSTERS`` bits)
- **Exit:** When all mask words received → transition to ``GET_ROUTER_CONFIG``; latch
  ``choose_iact_buffer`` from ``choose_iact_buffer_input``

GET_ROUTER_CONFIG (2)
~~~~~~~~~~~~~~~~~~~~~

Receives all router mode vectors. Actions:

- ``ready_dma_o <= 1``
- ``new_stream <= 1`` (resets FIFO)
- Computes ``iact_channels`` and ``iact_x_with_add_up``
- On each ``enable_dma_i_reg``:

  - **If** ``fsm_cycle < FSM_CEIL_IACT_RTR_CCLS``: Unpacks IACT router modes from DMA word
    into ``router_mode_iact``. Each word carries ``DMA_BITWIDTH/ROUTER_MODES_IACT = 64/6 ≈
    10`` router entries.
  - **If** ``fsm_cycle < FSM_CEIL_IACT_RTR_CCLS + FSM_CEIL_WGHT_RTR_CCLS``: Unpacks WGHT
    router modes.
  - **Else:** Unpacks PSUM router modes.
  - At last word: compute ``fsm_psum_limit``; transition to ``GET_IACT`` (or ``GET_WGHT``
    if ``skipIact_reg``)

- If ``max_pooling``: skip straight to ``GET_OFFSET``

GET_IACT (3)
~~~~~~~~~~~~

Loads raw input activation pixels into the 32-cell double-buffer. Actions per enabled DMA
word:

- Increment ``current_buffer_n`` (wraps at 31)
- Write ``data_dma_i_reg`` into ``buffer_SP_data_w_reg[current_buffer_n]``
- Strobe ``buffer_SP_en_w_reg[current_buffer_n]``
- Advance ``current_buffer_addr`` (shared across all cells since they step in lock-step)
- **Exit:** When ``fsm_cycle == (iact_size_x × iact_size_y × iact_channels /
  IACT_WORDS_IN_RAM) - 1`` → ``GET_WGHT``

GET_WGHT (4)
~~~~~~~~~~~~

Loads weight data into ``wght_buffer_SP``. The DMA provides pairs of 24-bit weight rows
(two rows per 64-bit DMA word). Actions per enabled DMA word:

- Set ``iact_converter_max_cycles``: ``iact_size_y + kernel_size - 1`` (or ``/2`` for
  single-channel, ``2`` for FC)
- Set ``min_standing_cycles``
- Assemble ``wght_buffer_SP_data_w``: lower 24 bits go to ``[fsm_y_cl][fsm_wght_r]``,
  upper 24 bits to ``[fsm_y_cl + CLUSTER_ROWS][fsm_wght_r]``
- Increment ``fsm_wght_r``; when all weight GLBs done, increment ``fsm_y_cl`` and write
  one RAM word
- Compute ``wght_cnt`` = (wght_cycles × input_activations × ⌈filters/PARALLEL_MACS⌉) − 1
- **Exit:** When all weight words written → ``GET_BIAS`` (or ``GET_QUANTIZE`` if
  ``skipPsum_reg``)

GET_BIAS (5)
~~~~~~~~~~~~

Loads initial bias values into ``psum_buffer_SP``. Actions per enabled DMA word:

- Write ``data_dma_i_reg[39:0]`` (40 bits = two 20-bit psum values) to the current PSUM
  buffer entry
- Step through ``fsm_psum_r``, ``fsm_y_cl_psum``, ``fsm_x_cl_psum``
- Strobe ``psum_buffer_SP_en_w``
- When all clusters done: check if complete, set ``psum_cnt``
- **Exit:** When complete → ``GET_QUANTIZE``; also compute ``limit_increase_reg`` and
  ``overhang_discrepancy`` for subsequent iact packing

GET_QUANTIZE (6)
~~~~~~~~~~~~~~~~

Loads per-filter quantization (mantissa + exponent) parameters. 32 filters need 16 DMA
words (2 filters per word):

- Each DMA word: ``data_dma_i_reg[31:25]`` → ``quant_exp[2*cycle]``, ``[24:0]`` →
  ``quant_mant[2*cycle]``; bits ``[63:57]`` → ``quant_exp[2*cycle+1]``, ``[56:32]`` →
  ``quant_mant[2*cycle+1]``
- **Exit:** After 16 words → ``GET_OFFSET``

GET_OFFSET (7)
~~~~~~~~~~~~~~

Loads per-filter zero-point offsets. 4 DMA words × 8 bytes = 32 offsets:

- Each DMA word unpacked into 8 consecutive ``quant_offset[8*cycle + 0..7]`` entries
- **Exit:** After 4 words → ``START_CONVERTER`` (or ``MAXPOOLING_READ`` if
  ``max_pooling``); updates ``buffer_SP_addr_upper_limit``

START_CONVERTER (8)
~~~~~~~~~~~~~~~~~~~

Waits for all ``iact_stream_constructor`` instances to be ready. Polls ``converters_ready``
(AND of all ``iact_converter_ready_w``). When all are ready:

- Transition to ``CONVERT_IACT``
- Initialize RAM addresses to 0; set ``past_padding`` if single-channel

CONVERT_IACT (9)
~~~~~~~~~~~~~~~~

Drives the iact stream constructors through all y-line cycles, coordinate updates, and
channel batches. Each cycle:

- Increments ``iact_converter_buffer_addr_cycles``; wraps at
  ``iact_converter_buffer_addr_max_cycles``
- When the buffer-address sub-cycle wraps: increments ``iact_converter_cycles``
  (y-line counter)
- When y-line counter wraps: increments ``iact_channels_counter``
- Manages the ``buffer_SP_addr_reg`` sliding window: each time a new row is needed, the
  active RAM cell range shifts forward by ``limit_increase_reg`` (+ overhang correction)
- On ``iact_buffer_next_addr`` event: triggers ``iact_converter_enc_enable`` and
  ``iact_converter_params_enable``
- **Exit:** When all channel batches done → ``WAIT_CYCLE``

WAIT_CYCLE (10)
~~~~~~~~~~~~~~~

Allows the converter pipeline to drain (4 × 4 = 16 extra cycles):

- ``iact_buffer_SP_data_w <= iact_out_reg`` (not actively used — legacy holdover)
- ``send_data_reg <= 1`` (triggers the data-flow always-block to start sending weights to
  OpenEye_Parallel)
- **Exit:** After 16 cycles → ``WAIT_FOR_RESULTS`` or ``RECEIVE_PSUMS_TO_IACT``

WAIT_FOR_RESULTS (11)
~~~~~~~~~~~~~~~~~~~~~

Waits while ``OpenEye_Parallel`` computes. Tracks iterations via ``current_cycle`` and
``single_iteration`` logic. When ``last_data_o`` pulses → done, return to
``GET_PARAMETERS``.

RECEIVE_PSUMS_TO_IACT (12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used when ``send_data_out == 0``: the computed psums are quantized and written back into
the iact double-buffer to serve as activations for the next layer. The PSUM FSM sub-state
``SEND_PSUM_TO_IACT`` drives the quantization; this state assembles the resulting bytes
into ``buffer_SP_data_w_reg``. Three channel-count paths:

- ``iact_channels_per_pe_next_layer == 4``: complex interleaved packing with overhang
  handling
- ``iact_channels_per_pe_next_layer == 2``: simpler 2-channel packing
- ``iact_channels_per_pe_next_layer == 1``: one channel per output pixel (FC or
  single-channel)
- **Exit:** When ``fsm_psum_current_state == PSUM_IDLE`` → ``GET_PARAMETERS``

MAXPOOLING_READ (13)
~~~~~~~~~~~~~~~~~~~~

Reads a 2×2 region from the iact buffer and computes the max via the 3-stage pipelined
comparator tree. Updates ``buffer_SP_addr_reg`` to traverse the input feature map.
Accumulates results into ``pooling_regs[0..31]``. Exits when all pixels scanned →
``MAXPOOLING_SEND``.

MAXPOOLING_SEND (14)
~~~~~~~~~~~~~~~~~~~~

Writes max-pool results from ``pooling_regs`` back into the iact double-buffer cells at
the appropriate addresses so that subsequent layers can read them. Returns to
``GET_PARAMETERS`` when done.
