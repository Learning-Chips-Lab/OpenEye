.. _fpga_generate_blocks:

Generate Blocks and Instantiations
====================================

gen_RAM_wires (genvar k_gen, lines 2734–2741)
----------------------------------------------

Wires up the ``buffer_SP_en_r/w/addr/data_w/data_r`` arrays to the corresponding ``_reg``
registers. Also assigns ``buffer_SP_data_r`` from the low half of the concatenated
double-port read output ``buffer_SP_data_r_w``.

UNPACKED_TRACES generate (lines 2744–2768)
-------------------------------------------

If ``UNPACKED_TRACES_ENABLED == 1``, generates named wire aliases for each element of the
unpacked arrays (``pooling_stage_1/2/3/4``, ``quantized_value_reg``). This is purely for
simulation visibility — waveform viewers cannot directly display unpacked arrays, so this
exposes them as individual named wires.

BUFFER_A generate (j_gen, lines 2773–2786)
-------------------------------------------

Generates 32 ``RAM_SP`` instances for the iact double-buffer. Each instance:

- Read enable: ``buffer_SP_en_r[j] & !buffer_SP_en_w[j]`` (priority write)
- Write enable: ``buffer_SP_en_w[j]``
- Address: ``{choose_iact_buffer, buffer_SP_addr[j]}`` — MSB selects ping/pong half

IACT_CONVERTER_X/Y generate (i_gen/j_gen, lines 2788–2834)
------------------------------------------------------------

Generates ``CLUSTER_COLUMNS × CLUSTER_ROWS`` instances of ``iact_stream_constructor``.
Connects:

- Shared ``storage_i`` from all 32 iact RAM cells (read in parallel)
- Individual ``params``, ``enable_config/store/converter``, ``ready_o`` per instance
- Local wires ``iact_data_w``, ``iact_enable_w``, ``iact_choose_w`` are then connected
  to the global ``iact_*_oep_w`` buses (lines 3024–3046)

Weight buffer instance (lines 2836–2847)
-----------------------------------------

Single ``RAM_SP`` for weights. Address multiplexed:
``wght_buffer_SP_wr_addr | wght_buffer_SP_rd_addr`` (one is always 0 when the other is
active since read/write are mutually exclusive).

PSUM_RAM_X/Y/GLB generate (lines 2849–2874)
--------------------------------------------

Generates ``CLUSTER_COLUMNS × CLUSTER_ROWS × NUM_GLB_PSUM/2`` ``RAM_SP`` instances. Each
stores 2 PSUM values (40-bit word). Also assigns debug signals for cell [0][0][0].

dma_storage instance (lines 2880–2928)
----------------------------------------

Connects the register-map decoder. All configuration wire outputs feed directly into main
FSM logic and into ``OpenEye_Parallel``.

OpenEye_Parallel instance (lines 2931–3023)
--------------------------------------------

The compute core. All parameter passes match the FPGA wrapper's parameter set. Key
interface signals:

- ``compute_i <= compute_reg`` (one-cycle trigger)
- Iact: from ``iact_stream_constructor`` outputs via ``iact_*_oep_w`` wires
- Wght: from ``wght_buffer_SP_data_r`` via ``wght_data_i_w``
- Psum in/out: ``psum_buffer_SP_data_r`` / ``psum_data_o_w``
- Config: all ``dma_storage`` outputs plus ``router_mode_*`` and ``compute_mask_reg_port``
