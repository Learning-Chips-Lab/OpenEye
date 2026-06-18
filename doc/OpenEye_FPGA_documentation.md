# OpenEye_FPGA Module — Detailed Technical Documentation

**File:** `hdl/OpenEye_FPGA.v`
**License:** SHL-2.1 — © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025)

---

## Table of Contents

1. [Purpose and Overview](#1-purpose-and-overview)
2. [Module Hierarchy and Instantiated Submodules](#2-module-hierarchy-and-instantiated-submodules)
3. [Parameters](#3-parameters)
4. [Ports](#4-ports)
5. [Internal Registers and Wires](#5-internal-registers-and-wires)
6. [Main FSM (fsm_current_state)](#6-main-fsm-fsm_current_state)
7. [PSUM FSM (fsm_psum_current_state)](#7-psum-fsm-fsm_psum_current_state)
8. [Always Blocks / Processes](#8-always-blocks--processes)
9. [Generate Blocks and Instantiations](#9-generate-blocks-and-instantiations)
10. [Key Computations and Data Paths](#10-key-computations-and-data-paths)
11. [Waveform / Simulation Facilities](#11-waveform--simulation-facilities)

---

## 1. Purpose and Overview

`OpenEye_FPGA` is the **top-level FPGA wrapper** for the OpenEye neural-network inference accelerator. It adapts the core `OpenEye_Parallel` module—which was designed for ASIC—to the constraints of an FPGA platform (limited port count, block-RAM availability, DMA burst interfaces).

Responsibilities:

| Responsibility | Mechanism |
|---|---|
| Receive raw layer parameters, activations, weights, biases, and quantization coefficients from a host CPU | DMA input port (`data_dma_i` / `enable_dma_i` / `ready_dma_o`) |
| Buffer weight words in a single-port SRAM (`wght_buffer_SP`) | Controlled by `wght_buffer_SP_en_w/r` |
| Buffer raw iact pixels across 32 double-buffered SRAM cells (`iact_converter_buffer_SP`) | Controlled by `buffer_SP_en_w/r` |
| Convert stored raw iact pixels into packed sparse streams per PE cluster | `iact_stream_constructor` (one per cluster position) |
| Drive `OpenEye_Parallel` (the compute core) with correctly formatted iact, wght, and psum data | Via `iact_data_i_oep_w`, `wght_data_i_w`, `psum_data_i_reg` wires/registers |
| Collect partial sums from `OpenEye_Parallel`, store them in PSUM SRAMs, apply quantization, and optionally route them back as iact for the next layer | PSUM FSM + `psum_buffer_SP` array |
| Send final 20-bit PSUM words (or 8-bit quantized activations) back to the host | DMA output port (`data_dma_o` / `enable_dma_o` / `ready_dma_i`) |
| Perform 2×2 max-pooling on-chip before sending results | `MAXPOOLING_READ/SEND` states |

---

## 2. Module Hierarchy and Instantiated Submodules

```
OpenEye_FPGA
├── RST_SYNC                              (reset synchronizer, 1 instance)
├── dma_storage                           (register-map decoder, 1 instance)
├── RAM_SP [×32]   (BUFFER_A)            (iact double-buffer cells)
├── iact_stream_constructor [CLUSTER_COLUMNS × CLUSTER_ROWS]   (one per cluster)
├── RAM_SP (wght_buffer_SP)              (weight staging buffer)
├── RAM_SP [CLUSTER_COLUMNS × CLUSTER_ROWS × NUM_GLB_PSUM/2]  (psum staging buffers)
└── OpenEye_Parallel                      (compute core)
```

### 2.1 RST_SYNC

**File:** `hdl/RST_SYNC.v`

Provides a 2-FF synchronizer for the active-low reset `rst_ni`. The output `rst_n` is used by all internal sequential logic so that reset de-assertion is always synchronous to `clk_i`, preventing metastability.

| Port | Direction | Description |
|---|---|---|
| `clk_i` | input | System clock |
| `rst_ni` | input | Asynchronous, active-low reset from board |
| `rst_no` | output | Synchronized active-low reset |

**Operation:** On the falling edge of `rst_ni`, both synchronizer FFs are forced to `0`. On clock edges while `rst_ni` is high, the chain shifts a `1` through so that `rst_no` asserts (goes high) two cycles after the board reset releases.

---

### 2.2 dma_storage

**File:** `hdl/dma_storage.v`  (auto-generated from `hdl/config/regmap.yaml`)

Decodes the first few DMA words received during `GET_PARAMETERS` state into named configuration registers. Acts as a write-only register file with address-based write enable.

| Input | Description |
|---|---|
| `write_en` | Pulse to latch one DMA word |
| `write_addr[1:0]` | Register bank address (0–3) |
| `dma_data_i[63:0]` | DMA word to decode |

Key output registers decoded from the first 4 DMA words (word 0–3):

| Register | Width | Meaning |
|---|---|---|
| `needed_cycles_reg` | 18 | Total computation cycles expected |
| `iact_size_x/y` | 8 | Input feature map spatial dimensions |
| `iact_channels_per_pe` | 8 | Channel depth assigned per PE |
| `kernels_per_calc` | 5 | Filters computed per cycle batch |
| `y_lines_per_calc` | 4 | Output rows computed per batch |
| `filters_reg` | 6 | Number of output filters |
| `wght_cycles_reg` | 8 | Clock cycles to stream all weights |
| `kernel_size` | 4 | Convolution kernel spatial dimension (e.g. 3 for 3×3) |
| `stride_x_reg` / `stride_y_reg` | 3 | Convolution strides |
| `skipIact_reg` | 1 | Skip iact DMA loading phase |
| `skipWght_reg` | 1 | Skip weight DMA loading phase |
| `skipPsum_reg` | 1 | Skip bias/psum DMA loading phase |
| `fully_connected_layer` | 1 | FC-mode flag |
| `max_pooling` | 1 | Enable max-pooling post-processing |
| `send_data_out` | 1 | Result goes directly out (vs. looped back as next-layer iact) |
| `store_in_psum` | 1 | Keep psums internally for further accumulation |
| `iact_channels_per_pe_next_layer` | 4 | Channel count for subsequent layer (used when routing psums→iact) |
| `needed_psum_storage_cycles_reg` | 8 | How many PSUM buffer passes before final output |
| `iact_channel_max_cycles` | 8 | How many iact channel batches per spatial position |
| `input_activations` | 5 | Number of input activations per MAC cycle |
| `needed_x_cls_reg` / `needed_y_cls_reg` | 2/4 | Number of cluster columns/rows active |
| `fc_size_reg` | 12 | FC layer input size |
| `add_up` | 2 | Extra overlap pixels to pad the iact width |
| `iact_x_line_repetitions` | 8 | Number of times a horizontal iact line repeats |
| `choose_iact_buffer_input/output` | 1 | Which half of the double-buffer is used for loading vs. reading |
| `psum_delay_reg` | 4 | Pipeline delay through PSUM cluster |

---

### 2.3 RAM_SP (×32, iact double-buffer)

**File:** `hdl/RAM_SP.v`
**Instance prefix:** `BUFFER_A[j_gen].iact_converter_buffer_SP`

32 identical single-port SRAMs form the **iact ping-pong buffer**. Each cell is:
- `DataWidth = RAM_CELLS_WORD_BITWIDTH = 64` bits (8 bytes = 8 INT8 pixels)
- `AddrWidth = RAM_CELLS_ADDR_WIDTH = 12` bits (4096 addresses per cell; address bit [11] selects ping vs. pong half via `choose_iact_buffer`)
- `Pipelined = 1` (read data appears one clock after address)

The 32 cells are addressed in a rotating fashion. During `GET_IACT`, incoming DMA data is written to successive cells via `current_buffer_n`. During `CONVERT_IACT`, all 32 cells are read in parallel and their data is fed to the `iact_stream_constructor` instances.

---

### 2.4 iact_stream_constructor (CLUSTER_COLUMNS × CLUSTER_ROWS)

**File:** `hdl/iact_stream_constructor.v`
**Instance array:** `IACT_CONVERTER_X[cc].IACT_CONVERTER_Y[cr]`

One per cluster position. Reads raw pixel data from the shared 32-cell iact SRAM bank and produces the packed, sparse-encoded activation streams (`iact_data_o`, `iact_enable_o`, `iact_choose_o`) expected by the corresponding PE cluster in `OpenEye_Parallel`.

Key ports:

| Port | Direction | Description |
|---|---|---|
| `storage_i` | input | Concatenated read data from all 32 iact RAM cells |
| `params` | input | 36-bit configuration word: `[35:32]` row-offset, `[31:24]` x-coordinate, `[23:16]` y-coordinate, `[15:8]` iact_size_x, `[7:0]` channel-start |
| `enable_config` | input | Latch `params` into internal registers |
| `enable_store` | input | Run one store (write into internal FIFO/buffer) |
| `enable_converter` | input | Run one encode cycle (push data to PE) |
| `ready_o` | output | High when converter has drained its internal buffer |
| `iact_data_o` | output | Packed iact words for `NUM_GLB_IACT` GLBs |
| `iact_enable_o` | output | Per-GLB valid signals |
| `iact_choose_o` | output | Per-PE source-selection signal |

---

### 2.5 RAM_SP (wght_buffer_SP)

Single-port SRAM buffering all weight words for the current layer.

- `DataWidth = TRANS_BITWIDTH_WGHT × CLUSTERS × NUM_GLB_WGHT = 24 × 16 × 3 = 1152` bits
- `AddrWidth = BUFFER_WIDTH + 1 = 13` bits

During `GET_WGHT`, the main FSM assembles one row of weight data from successive DMA words and writes it at `wght_buffer_SP_wr_addr`. During `WAIT_FOR_RESULTS`, the data-flow process reads from `wght_buffer_SP_rd_addr` and drives `wght_enable_i_reg` to stream weights into `OpenEye_Parallel`.

---

### 2.6 RAM_SP (psum_buffer_SP, CLUSTER_COLUMNS × CLUSTER_ROWS × NUM_GLB_PSUM/2)

An array of single-port SRAMs that buffer partial sums between the compute core and the PSUM FSM.

- `DataWidth = TRANS_BITWIDTH_PSUM × 2 = 40` bits (two 20-bit psum values)
- `AddrWidth = BUFFER_WIDTH = 12` bits

Addressed via `psum_buffer_SP_addr_array[cc][cr][g]`. Used to:
1. Pre-load bias values (received via DMA in `GET_BIAS`)
2. Feed accumulated psums back into `OpenEye_Parallel` during `CALCULATE_PSUM`
3. Capture output psums from `OpenEye_Parallel` in `PSUM_GET_RESULTS`
4. Supply data for DMA output (`PSUM_SEND_RESULTS`) or quantization (`SEND_PSUM_TO_IACT`)

---

### 2.7 OpenEye_Parallel

**File:** `hdl/OpenEye_Parallel.v`

The actual compute core: a 2D array of PE clusters performing MAC operations on iact/weight data and accumulating partial sums. `OpenEye_FPGA` feeds it all required data and configuration and collects results.

Relevant connection summary:

| Signal group | Direction | Description |
|---|---|---|
| `compute_reg` | → core | Single-cycle pulse to start a computation batch |
| `iact_data_i_oep_w` / `iact_enable_i_oep_w` / `iact_choose_i_oep_w` | → core | Activation stream from iact_stream_constructors |
| `iact_ready_o_oep_w` | ← core | Back-pressure signal per iact GLB |
| `wght_data_i_w` / `wght_enable_i_reg` / `wght_ready_o_reg` | ↔ core | Weight stream from wght_buffer_SP |
| `psum_data_i_reg` / `psum_enable_i_reg` / `psum_ready_o_reg` | → core | Bias/psum input from psum_buffer_SP |
| `psum_data_o_w` / `psum_enable_o` / `psum_ready_i_reg` | ← core | Result psum output back to psum_buffer_SP |
| `status_reg_enable_reg` | → core | Enable configuration register writes during `GET_PARAMETERS` |
| `router_mode_iact/wght/psum` | → core | Router configuration loaded during `GET_ROUTER_CONFIG` |
| `compute_mask_reg_port` | → core | Bit mask of active PE instances |

---

### 2.8 varlenFIFO

**File:** `hdl/varlenFIFO.v`
*(Defined in the code but appears to be available for use; the main FPGA wrapper uses `fifo_data_i`, `fifo_read_i`, `fifo_write_i` registers that target this interface.)*

A simple circular FIFO with:
- Configurable `DATA_WIDTH` and `DEPTH`
- `wr_en` / `rd_en` strobes
- `new_stream_i` to reset pointers
- `empty` / `full` status flags

In `OpenEye_FPGA`, the FIFO wires are initialized but the FIFO instance is not explicitly shown in the portion of the file read; these signals are used for optional output buffering of DMA results.

---

## 3. Parameters

### 3.1 Architecture Parameters

| Parameter | Default | Description |
|---|---|---|
| `IS_TOPLEVEL` | `1` | `1` = this is the chip top; enables simulation dump |
| `SERIAL` | `1` | `1` = PE uses serial MAC; `0` = parallel |
| `PARALLEL_MACS` | `2` | Number of MAC units operating in parallel per PE |
| `SPARSITY_EN` | `1` | `1` = sparse mode; `0` = dense mode |
| `CLUSTER_ROWS` | `8` | Rows of PE clusters in the array |
| `CLUSTER_COLUMNS` | `2` | Columns of PE clusters |
| `CLUSTERS` | `CLUSTER_COLUMNS × CLUSTER_ROWS` | Total clusters |

### 3.2 Data-Width Parameters

| Parameter | Default | Description |
|---|---|---|
| `DATA_IACT_BITWIDTH` | `8` | Bit width of one iact value (INT8) |
| `DATA_PSUM_BITWIDTH` | `20` | Bit width of partial sum accumulator |
| `DATA_WGHT_BITWIDTH` | `8` | Bit width of one weight value |
| `DATA_IACT_OVERHEAD` | `4` | Sparsity metadata bits per iact word |
| `TRANS_BITWIDTH_IACT` | `24` | Iact bus width (24 = 3×8-bit values or 2×12-bit) |
| `TRANS_BITWIDTH_WGHT` | `24` | Weight bus width |
| `TRANS_BITWIDTH_PSUM` | `20` | PSUM bus width |
| `DMA_BITWIDTH` | `64` | Width of the DMA AXI-stream interface |

### 3.3 Memory Sizing Parameters

| Parameter | Default | Description |
|---|---|---|
| `NUM_GLB_IACT` | `3` | Iact GLBs per cluster row |
| `NUM_GLB_WGHT` | `3` | Weight GLBs per cluster (= PE rows per cluster) |
| `NUM_GLB_PSUM` | `4` | Psum GLBs per cluster (= PE columns per cluster) |
| `PES` | `NUM_GLB_PSUM × NUM_GLB_WGHT` | PEs per cluster |
| `IACT_PER_PE` | `16` | Max iact words in PE scratchpad |
| `WGHT_PER_PE` | `96` | Max weight words in PE scratchpad |
| `PSUM_PER_PE` | `32` | Max psum words in PE scratchpad |
| `IACT_ADDR_PER_PE` | `9` | Address words in iact address SPad |
| `WGHT_ADDR_PER_PE` | `16` | Address words in weight address SPad |
| `IACT_MEM_ADDR_WORDS` | `512` | IACT GLB depth (words) |
| `PSUM_MEM_ADDR_WORDS` | `768` | PSUM GLB depth (words) |
| `RAM_CELLS` | `32` | Number of iact double-buffer SRAM cells |
| `RAM_CELLS_WORD_BITWIDTH` | `64` | Width of one cell word (bits) |
| `RAM_CELLS_ADDR_WIDTH` | `12` | Address width of cell (11 bits usable + 1 ping/pong bit) |
| `BUFFER_WIDTH` | `12` | Address width of psum/weight staging buffers |

### 3.4 Router Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `ROUTER_MODES_IACT` | `6` | Bits per iact router entry |
| `ROUTER_MODES_WGHT` | `1` | Bits per weight router entry |
| `ROUTER_MODES_PSUM` | `3` | Bits per psum router entry |
| `BANO_MODES` | `2` | Batch-normalization mode count |
| `AF_MODES` | `4` | Activation-function mode count |

### 3.5 Derived / Computed Parameters

| Parameter | Formula | Description |
|---|---|---|
| `IACT_WORDS_IN_RAM` | `RAM_CELLS_WORD_BITWIDTH / DATA_IACT_BITWIDTH` = 8 | INT8 pixels per 64-bit RAM word |
| `WORDS_PER_CYCLE` | 2 | Iact conversion processes 2 words per cycle |
| `PSUM_TO_IACT_CYCLES` | 2 if `CLUSTER_COLUMNS×NUM_GLB_PSUM == 4`, else 1 | Cycles needed per spatial position for psum→iact routing |
| `FSM_CEIL_IACT_RTR_CCLS` | `⌈(CLUSTERS×NUM_GLB_IACT) / (DMA_BITWIDTH/ROUTER_MODES_IACT)⌉` | DMA words needed for iact router config |
| `FSM_CEIL_WGHT_RTR_CCLS` | `⌈(CLUSTERS×NUM_GLB_WGHT) / (DMA_BITWIDTH/ROUTER_MODES_WGHT)⌉` | DMA words needed for weight router config |
| `FSM_CEIL_PSUM_RTR_CCLS` | `⌈(CLUSTERS×NUM_GLB_PSUM) / (DMA_BITWIDTH/ROUTER_MODES_PSUM)⌉` | DMA words needed for psum router config |
| `EXTENDEDBITS` | `48 - NUM_GLB_WGHT` | Zero-pad width when building `flat_help_var_send` mask |

---

## 4. Ports

### 4.1 Clock and Reset

| Port | Direction | Width | Description |
|---|---|---|---|
| `clk_i` | input | 1 | System clock. All FFs are positive-edge triggered |
| `rst_ni` | input | 1 | Active-low, asynchronous global reset |

### 4.2 DMA Input Interface (host → accelerator)

| Port | Direction | Width | Description |
|---|---|---|---|
| `ready_dma_o` | output reg | 1 | High when the module can accept a new DMA word |
| `data_dma_i` | input | `DMA_BITWIDTH` | 64-bit data word from DMA |
| `enable_dma_i` | input | 1 | Valid signal: `data_dma_i` is valid this cycle |

The protocol is a simple handshake: a transfer occurs on any cycle where both `enable_dma_i` and `ready_dma_o` are high. After reset, the FSM waits in `IDLE` until the host asserts `enable_dma_i`.

### 4.3 DMA Output Interface (accelerator → host)

| Port | Direction | Width | Description |
|---|---|---|---|
| `ready_dma_i` | input | 1 | High when the downstream sink can accept data |
| `data_dma_o` | output reg | `DMA_BITWIDTH` | 64-bit result word |
| `enable_dma_o` | output reg | 1 | Valid signal for `data_dma_o` |
| `last_data_o` | output reg | 1 | Pulses high on the last output word of a layer |

### 4.4 Debug Outputs

These signals expose internal SRAM and FSM state for simulation / SignalTap analysis and are not used in production:

| Port | Description |
|---|---|
| `debug_iact_we/re` | Write/read enable to the first iact RAM cell |
| `debug_iact_addr[11:0]` | Address driven to the first iact RAM cell |
| `debug_iact_data_i/o[63:0]` | Data driven into / read from first iact RAM cell |
| `debug_psum_we/re` | Write/read enable to psum buffer cell [0][0][0] |
| `debug_psum_addr[11:0]` | Address for above |
| `debug_psum_data_i/o[39:0]` | Data for above |
| `debug_skip_iact_o` | Reflects `skipIact_reg` from `dma_storage` |
| `debug_data_dma_stream_o[63:0]` | Registered DMA input word (`data_dma_i_reg`) |
| `debug_enable_dma_stream_o` | Registered `enable_dma_i_reg` |
| `debug_fsm_cycle_o[3:0]` | Low 4 bits of `fsm_cycle` counter |
| `debug_fsm_current_state[3:0]` | Current main FSM state |
| `debug_fsm_psum_state[2:0]` | Current PSUM FSM state |

---

## 5. Internal Registers and Wires

### 5.1 DMA Input Pipeline Registers

| Signal | Type | Description |
|---|---|---|
| `data_dma_i_reg[63:0]` | reg | One-cycle registered version of `data_dma_i` |
| `enable_dma_i_reg` | reg | One-cycle registered version of `enable_dma_i` |

These registers add one pipeline stage, ensuring timing closure on the DMA interface.

### 5.2 Configuration Registers (from dma_storage)

These wires connect to outputs of the `dma_storage` module. They are set once per layer and remain stable during computation:

| Signal | Description |
|---|---|
| `needed_cycles_reg[17:0]` | Total iteration count until computation complete |
| `needed_x_cls_reg[1:0]`, `needed_y_cls_reg[3:0]` | Active cluster grid size |
| `needed_iact_cycles_reg[3:0]` | Iact router broadcast cycles per position |
| `filters_reg[5:0]` | Output filters per PE batch |
| `iact_addr_len_reg`, `wght_addr_len_reg` | SPad address lengths |
| `kernel_size[3:0]` | Kernel spatial size |
| `stride_x_reg[2:0]`, `stride_y_reg[2:0]` | Convolution strides |
| `skipIact_reg`, `skipWght_reg`, `skipPsum_reg` | Phase-skip flags |
| `wght_cycles_reg[7:0]` | Weight streaming cycles |
| `input_activations[4:0]` | Activations per PE MAC cycle |
| `iact_size_x[7:0]`, `iact_size_y[7:0]` | Feature map dimensions |
| `iact_channels_per_pe[7:0]` | Channels in this PE batch |
| `iact_channel_max_cycles[7:0]` | Total channel batches |
| `kernels_per_calc[4:0]` | Filters calculated simultaneously |
| `y_lines_per_calc[3:0]` | Output rows per calculation |
| `iact_x_line_repetitions[7:0]` | Repetitions of each x-line |
| `fully_connected_layer` | FC mode flag |
| `max_pooling` | Max-pool mode flag |
| `store_in_psum` | Keep psums for further iterations |
| `send_data_out` | Send results directly out (vs. loop back) |
| `iact_channels_per_pe_next_layer[3:0]` | Channel count for next layer |
| `needed_psum_storage_cycles_reg[7:0]` | PSUM buffer passes |
| `fc_size_reg[11:0]` | FC layer input length |
| `iact_needed_cycles[10:0]` | Iact streaming cycle count |
| `output_cycles[7:0]` | Output buffer read cycles |
| `needed_iact_buffer_words_reg` | Words in iact stream buffer |
| `add_up[2:0]` | Extra overlap columns |
| `iact_converter_buffer_addr_max_cycles[7:0]` | Max address cycles in iact converter |
| `choose_iact_buffer_input/output` | Double-buffer selector |

### 5.3 Hyperparameter Registers (set by main FSM)

| Signal | Type | Description |
|---|---|---|
| `data_mode_reg` | reg | Fixed-point vs. floating-point mode |
| `fraction_bit_reg[4:0]` | reg | Fractional bit position |
| `padding_reg[3:0]` | reg | Zero-padding size = `(kernel_size-1)/2` |
| `bano_cluster_mode_reg` | reg | Batch-norm mode per psum GLB |
| `af_cluster_mode_reg[1:0]` | reg | Activation function mode |
| `compute_mask_reg` | reg | Bitmask of active PEs (one bit per PE per cluster) |

### 5.4 Main FSM Counters

| Signal | Type | Description |
|---|---|---|
| `fsm_current_state[3:0]` | reg | Current state of the main FSM |
| `fsm_last_state[3:0]` | reg | Previous state (for debugging) |
| `fsm_cycle[31:0]` | reg | Cycle counter within the current state |
| `fsm_x_cl[clog2(CLUSTER_COLUMNS)-1:0]` | reg | Current cluster column being processed |
| `fsm_y_cl[clog2(CLUSTER_ROWS)-1:0]` | reg | Current cluster row being processed |
| `fsm_iact_r[clog2(NUM_GLB_IACT)-1:0]` | reg | Current iact GLB index being loaded |
| `fsm_wght_r[clog2(NUM_GLB_WGHT)-1:0]` | reg | Current weight GLB index being loaded |

### 5.5 Cycle Tracking Registers

| Signal | Type | Description |
|---|---|---|
| `current_cycle[19:0]` | reg | Counts iact delivery iterations during `WAIT_FOR_RESULTS` |
| `iact_cycle_count[15:0]` | reg | Sub-counter tracking weight cycling period |
| `iact_router_counter[3:0]` | reg | Counts cluster-row sweeps per iact push |
| `single_iteration` | reg | Flag: currently in the "active part" of one iact delivery |
| `single_iteration2` | reg | Delayed version of `single_iteration` (for edge detection) |
| `single_iteration3` | reg | Single-cycle pulse: first cycle of active iact delivery |
| `finished_cycles_iact[19:0]` | reg | Total iact delivery iterations completed |
| `finished_cycles_psum[19:0]` | reg | Total psum output iterations completed |
| `reset_cycle` | reg | Pulse to reset all cycle counters |

### 5.6 Iact Double-Buffer Control

| Signal | Type | Description |
|---|---|---|
| `choose_iact_buffer` | reg | Selects which half (0/1) of the double-buffer is active |
| `current_buffer_n[7:0]` | reg | Current write cell index (0–31) |
| `current_buffer_n_1[7:0]` | reg | Previous write cell index (for address update pipeline) |
| `current_buffer_addr[10:0]` | reg | Current write address within a cell |
| `buffer_SP_en_r_reg[32]` | reg array | Read enables per iact RAM cell |
| `buffer_SP_en_w_reg[32]` | reg array | Write enables per iact RAM cell |
| `buffer_SP_addr_reg[32][10:0]` | reg array | Addresses per iact RAM cell |
| `buffer_SP_data_w_reg[32][63:0]` | reg array | Write data per iact RAM cell |
| `buffer_SP_addr_upper_limit[7:0]` | reg | Upper boundary of the active cell window |
| `buffer_SP_addr_lower_limit[7:0]` | reg | Lower boundary of the active cell window |
| `limit_increase_reg[7:0]` | reg | How much to advance the boundary per cycle |
| `overhang[1:0]` / `overhang_delay` | reg | Extra boundary increment when accumulated fractional part ≥ 1 |
| `overhang_counter[3:0]` | reg | Fractional accumulator for overhang detection |
| `overhang_discrepancy[3:0]` | reg | Fractional part of the number of cells per cycle |

### 5.7 Iact Converter Control

| Signal | Type | Description |
|---|---|---|
| `iact_converter_params_reg[CC][CR][35:0]` | reg | Packed {row_offset, x, y, size_x, channel} config per converter |
| `iact_converter_en_cfg_reg[CC][CR]` | reg | Pulse to latch params into converter |
| `iact_converter_en_store_reg[CC][CR]` | reg | Enable one store step in converter |
| `iact_converter_en_enc_reg[CC][CR]` | reg | Enable one encode step in converter |
| `iact_converter_ready_w[CC][CR]` | wire | Converter has finished current batch |
| `iact_converter_max_cycles[7:0]` | reg | Total y-line cycles including kernel overlap |
| `min_standing_cycles[7:0]` | reg | Minimum cycles a row must remain active |
| `iact_converter_cycles[7:0]` | reg | Current y-position counter |
| `iact_converter_buffer_addr_cycles[7:0]` | reg | Sub-counter for buffer address cycling |
| `iact_converter_params_enable` | reg | Gate signal for param FSM |
| `iact_converter_enc_enable` | reg | Gate signal for encode FSM |
| `converters_ready` | reg | AND of all `iact_converter_ready_w` outputs |

### 5.8 Iact Converter Traversal State

| Signal | Type | Description |
|---|---|---|
| `fsm_iact_params[7:0]` | reg | Countdown: remaining param slots to configure |
| `fsm_iact_params_y_line[7:0]` | reg | Current y-line index in the param sweep |
| `fsm_iact_params_kernel[7:0]` | reg | Current filter kernel index in the param sweep |
| `iact_converter_x[7:0]` | reg | X start coordinate for the current converter config |
| `iact_converter_y[7:0]` | reg | Y start coordinate |
| `iact_converter_c[7:0]` | reg | Channel start index |
| `fsm_row[clog2(CLUSTER_ROWS+1)-1:0]` | reg | Target cluster row for the current param write |
| `fsm_row_offset[clog2(CLUSTER_ROWS+1)-1:0]` | reg | Base row offset cycling within `needed_y_cls_reg` |
| `param_array_reg[CLUSTERS-1:0]` | reg | Bitmask: which clusters receive a config event |
| `conv_array_reg[CLUSTERS-1:0]` | reg | Bitmask: which clusters receive a store event |

### 5.9 Weight Buffer Control

| Signal | Type | Description |
|---|---|---|
| `wght_buffer_SP_en_r` / `_en_w` | reg | Read/write enables for the weight SRAM |
| `wght_buffer_SP_wr_addr[BUFFER_WIDTH:0]` | reg | Current write address |
| `wght_buffer_SP_rd_addr[BUFFER_WIDTH:0]` | reg | Current read address |
| `wght_buffer_SP_rd_addr_storage[BUFFER_WIDTH:0]` | reg | Saved read address for rewind after a channel batch |
| `wght_buffer_SP_data_w` | reg | Write data (assembled from two DMA words per cycle) |
| `wght_data_i_w` | wire | Read data (drives `OpenEye_Parallel` weight input) |
| `wght_cnt[BUFFER_WIDTH:0]` | reg | Total weight words loaded (minus 1) |
| `wght_enable_i_reg[CLUSTERS×NUM_GLB_WGHT-1:0]` | reg | Per-GLB valid signals for weight stream |

### 5.10 Dataflow / Sending State

| Signal | Type | Description |
|---|---|---|
| `sending_data` | reg | True while streaming data to OpenEye_Parallel |
| `send_data_reg` | reg | One-cycle pulse to initiate data sending |
| `fsm_sending_cycle[12:0]` | reg | Cycle counter within the send phase |
| `wght_sendable` | reg | Weight stream is permitted to start |
| `compute_reg` | reg | One-cycle `compute` pulse for OpenEye_Parallel |
| `new_stream` | reg | Pulse to reset `varlenFIFO` |
| `early_stream_start` | reg | Host sent data before `ready_dma_o` was high |

### 5.11 PSUM FSM State and Counters

| Signal | Type | Description |
|---|---|---|
| `fsm_psum_current_state[3:0]` | reg | Current state of the PSUM FSM |
| `fsm_psum_last_state[3:0]` | reg | Previous PSUM FSM state |
| `fsm_psum_cycle[15:0]` | reg | Cycle counter within the current PSUM state |
| `fsm_x_cl_psum[clog2(CLUSTER_COLUMNS)-1:0]` | reg | Current output cluster column |
| `fsm_y_cl_psum[clog2(CLUSTER_ROWS+1)-1:0]` | reg | Current output cluster row |
| `fsm_psum_r[clog2(NUM_GLB_PSUM)-1:0]` | reg | Current output GLB index |
| `fsm_psum_r_q`, `fsm_x_cl_psum_q`, `fsm_y_cl_psum_q` | reg | One-cycle delayed versions (pipeline compensation) |
| `fsm_y_cl_psum_delay1/2/3` | reg | 3-stage delay chain for `fsm_y_cl_psum` (quantization pipeline) |
| `psum_buffer_SP_addr_array[CC][CR][g]` | reg | Per-psum-buffer read/write address |
| `psum_buffer_SP_addr_storage` | reg | Base address for the next psum output page |
| `psum_buffer_SP_en_r/w` | reg | Per-buffer read/write enable |
| `psum_enable_i_reg` | reg | Enable signals fed into OpenEye_Parallel psum input |
| `psum_ready_i_reg` | reg | Ready signals fed into OpenEye_Parallel psum input |
| `psum_transmitted` | reg | Flag: psums have been sent into OpenEye_Parallel |
| `results_ready` | reg | Local flag: all PE outputs are available |
| `storage_cycles[7:0]` | reg | Counts how many iterations have been stored |
| `psum_cnt[BUFFER_WIDTH-1:0]` | reg | Total psum words per output pass |
| `start_new_cycle` | reg | Trigger next computation cycle |
| `finished_cycles_psum[19:0]` | reg | Total psum output iterations done |
| `psum_router_set_reg` | reg | Tracks whether psum router is configured |
| `iact_channel_counter_reg[7:0]` | reg | Channel counter for psum router update |
| `last_data` / `last_data_reg` / `last_data_o` | reg | Last-data signaling chain |

### 5.12 Quantization and Psum→Iact Conversion

| Signal | Type | Description |
|---|---|---|
| `quant_exp[32][6:0]` | reg | Quantization exponent per filter (7 bits) |
| `quant_mant[32][24:0]` | reg | Quantization mantissa per filter (25 bits) |
| `quant_offset[32][7:0]` | reg | Per-filter zero-point offset |
| `quantized_value_reg[8][7:0]` | reg | 8 output bytes from the current quantization |
| `current_filter[7:0]` | reg | Index of the filter being quantized |
| `psum_to_iact_state` | reg | Toggle: which half-batch of psums is being quantized |

**Quantization formula** (for each filter `f`, psum value `p`):
```
quantized = (quant_mant[f] * (p + quant_offset[f])) >>> quant_exp[f]
```
This implements per-channel fixed-point linear quantization.

### 5.13 Max-Pooling Pipeline

| Signal | Type | Description |
|---|---|---|
| `pooling_regs[32][7:0]` | reg | Accumulates running max per output pixel |
| `pooling_stage_1[8][7:0]` | reg | Stage 1: raw 2×2 input pixels |
| `pooling_stage_2[4][7:0]` | reg | Stage 2: max of adjacent pairs |
| `pooling_stage_3[2][7:0]` | reg | Stage 3: max of pairs of pairs |
| `pooling_stage_4[7:0]` | reg | Final max (currently unused in some paths) |

**Pipeline operation:**
```
stage_2[a] = max(stage_1[2a], stage_1[2a+1])   // for a=0..3
stage_3[a] = max(stage_2[2a], stage_2[2a+1])   // for a=0..1
result = max(stage_3[0], stage_3[1])            // scalar max
```

### 5.14 Router Mode Registers

| Signal | Type | Description |
|---|---|---|
| `router_mode_iact[CLUSTERS×NUM_GLB_IACT×ROUTER_MODES_IACT-1:0]` | reg | 6-bit config per iact router |
| `router_mode_iact_storage[…]` | reg | Saved initial iact router config (for restoration after row sweep) |
| `router_mode_wght[CLUSTERS×NUM_GLB_WGHT×ROUTER_MODES_WGHT-1:0]` | reg | 1-bit config per weight router |
| `router_mode_psum[CLUSTERS×NUM_GLB_PSUM×ROUTER_MODES_PSUM-1:0]` | reg | 3-bit config per psum router |
| `psum_choose_i_reg[CLUSTERS×NUM_GLB_PSUM-1:0]` | reg | Per-psum-GLB PE-select bitmask |

### 5.15 PSUM Send-Phase Bookkeeping

| Signal | Type | Description |
|---|---|---|
| `psum_sending_counter[7:0]` | reg | Counts psum words sent to iact buffer |
| `sending_clusters[3:0]` | reg | Number of cluster GLBs active per send step |
| `sending_cluster_rows[3:0]` | reg | Number of cluster rows active per send step |
| `iteration_for_kernels_reg[3:0]` | reg | Iterations needed for all kernel groups |
| `psum_cycle_buffer_1/2/3/4[7:0]` | reg | Nested loop counters: y-line, kernel, channel, output tile |
| `pcb_1/2/3[11:0]` | reg | Base address for psum buffer when each pcb counter wraps |
| `fsm_psum_row_offset[3:0]` | reg | Starting cluster row for psum scanning |
| `fsm_psum_limit[16:0]` | reg | Total fsm_psum_cycle count before `SEND_PSUM_TO_IACT` exits |

### 5.16 RAM/DMA Write-Back Registers

| Signal | Type | Description |
|---|---|---|
| `write_dma_en` | reg | Enable signal for `dma_storage` write port |
| `write_dma_addr[1:0]` | reg | Address in `dma_storage` (0–3) |
| `dma_data_i[63:0]` | reg | Registered copy of incoming DMA word for `dma_storage` |
| `select_ram_counter[15:0]` | reg | Current cell index during psum→iact packing |
| `ram_counter_storage[15:0]` | reg | Saved reference cell for each output row |
| `select_ram_offset[7:0]` | reg | Additional offset into cell |
| `ram_iact_modulo[7:0]` | reg | `iact_size_x % 8` for handling non-multiples of 8 |

---

## 6. Main FSM (`fsm_current_state`)

The main FSM drives the overall control flow: loading parameters, activations, weights, biases, running the accelerator, and collecting output.

### State Diagram (simplified)

```
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
```

### State Descriptions

#### `IDLE` (0)
Waits for any `enable_dma_i_reg` pulse. Transitions to `GET_PARAMETERS` and initializes `fifo` signals.

#### `GET_PARAMETERS` (1)
Receives the layer configuration. Actions per cycle:
- `ready_dma_o <= 1` (accepting data)
- `status_reg_enable_reg <= 1` (allows `OpenEye_Parallel` to latch config from DMA)
- `reset_cycle <= 1` (resets all cycle counters to 0)
- Initializes pooling pipeline registers to -128 (minimum signed 8-bit)
- Initializes quantization registers to 0
- **Cycles 0–3:** `write_dma_en` strobed to write 4 DMA words to `dma_storage` at consecutive addresses. At cycle 3: `padding_reg <= (kernel_size-1)/2`
- **Cycles 4–N:** Fills `compute_mask_reg` one DMA word at a time (64 bits per cycle, covering `PES × CLUSTERS` bits)
- **Exit:** When all mask words received → transition to `GET_ROUTER_CONFIG`; latch `choose_iact_buffer` from `choose_iact_buffer_input`

#### `GET_ROUTER_CONFIG` (2)
Receives all router mode vectors. Actions:
- `ready_dma_o <= 1`
- `new_stream <= 1` (resets FIFO)
- Computes `iact_channels` and `iact_x_with_add_up`
- On each `enable_dma_i_reg`:
  - **If `fsm_cycle < FSM_CEIL_IACT_RTR_CCLS`:** Unpacks IACT router modes from DMA word into `router_mode_iact`. Each word carries `DMA_BITWIDTH/ROUTER_MODES_IACT = 64/6 ≈ 10` router entries.
  - **If `fsm_cycle < FSM_CEIL_IACT_RTR_CCLS + FSM_CEIL_WGHT_RTR_CCLS`:** Unpacks WGHT router modes.
  - **Else:** Unpacks PSUM router modes.
  - At last word: compute `fsm_psum_limit`; transition to `GET_IACT` (or `GET_WGHT` if `skipIact_reg`)
- If `max_pooling`: skip straight to `GET_OFFSET`

#### `GET_IACT` (3)
Loads raw input activation pixels into the 32-cell double-buffer. Actions per enabled DMA word:
- Increment `current_buffer_n` (wraps at 31)
- Write `data_dma_i_reg` into `buffer_SP_data_w_reg[current_buffer_n]`
- Strobe `buffer_SP_en_w_reg[current_buffer_n]`
- Advance `current_buffer_addr` (shared across all cells since they step in lock-step)
- **Exit:** When `fsm_cycle == (iact_size_x × iact_size_y × iact_channels / IACT_WORDS_IN_RAM) - 1` → `GET_WGHT`

#### `GET_WGHT` (4)
Loads weight data into `wght_buffer_SP`. The DMA provides pairs of 24-bit weight rows (two rows per 64-bit DMA word). Actions per enabled DMA word:
- Set `iact_converter_max_cycles`: `iact_size_y + kernel_size - 1` (or `/2` for single-channel, `2` for FC)
- Set `min_standing_cycles`
- Assemble `wght_buffer_SP_data_w`: lower 24 bits go to `[fsm_y_cl][fsm_wght_r]`, upper 24 bits to `[fsm_y_cl + CLUSTER_ROWS][fsm_wght_r]`
- Increment `fsm_wght_r`; when all weight GLBs done, increment `fsm_y_cl` and write one RAM word
- Compute `wght_cnt` = (wght_cycles × input_activations × ⌈filters/PARALLEL_MACS⌉) − 1
- **Exit:** When all weight words written → `GET_BIAS` (or `GET_QUANTIZE` if `skipPsum_reg`)

#### `GET_BIAS` (5)
Loads initial bias values into `psum_buffer_SP`. Actions per enabled DMA word:
- Write `data_dma_i_reg[39:0]` (40 bits = two 20-bit psum values) to the current PSUM buffer entry
- Step through `fsm_psum_r`, `fsm_y_cl_psum`, `fsm_x_cl_psum`
- Strobe `psum_buffer_SP_en_w`
- When all clusters done: check if complete, set `psum_cnt`
- **Exit:** When complete → `GET_QUANTIZE`; also compute `limit_increase_reg` and `overhang_discrepancy` for subsequent iact packing

#### `GET_QUANTIZE` (6)
Loads per-filter quantization (mantissa + exponent) parameters. 32 filters need 16 DMA words (2 filters per word):
- Each DMA word: `data_dma_i_reg[31:25]` → `quant_exp[2*cycle]`, `[24:0]` → `quant_mant[2*cycle]`; bits `[63:57]` → `quant_exp[2*cycle+1]`, `[56:32]` → `quant_mant[2*cycle+1]`
- **Exit:** After 16 words → `GET_OFFSET`

#### `GET_OFFSET` (7)
Loads per-filter zero-point offsets. 4 DMA words × 8 bytes = 32 offsets:
- Each DMA word unpacked into 8 consecutive `quant_offset[8*cycle + 0..7]` entries
- **Exit:** After 4 words → `START_CONVERTER` (or `MAXPOOLING_READ` if `max_pooling`); updates `buffer_SP_addr_upper_limit`

#### `START_CONVERTER` (8)
Waits for all `iact_stream_constructor` instances to be ready. Polls `converters_ready` (AND of all `iact_converter_ready_w`). When all are ready:
- Transition to `CONVERT_IACT`
- Initialize RAM addresses to 0; set `past_padding` if single-channel

#### `CONVERT_IACT` (9)
Drives the iact stream constructors through all y-line cycles, coordinate updates, and channel batches. Each cycle:
- Increments `iact_converter_buffer_addr_cycles`; wraps at `iact_converter_buffer_addr_max_cycles`
- When the buffer-address sub-cycle wraps: increments `iact_converter_cycles` (y-line counter)
- When y-line counter wraps: increments `iact_channels_counter`
- Manages the `buffer_SP_addr_reg` sliding window: each time a new row is needed, the active RAM cell range shifts forward by `limit_increase_reg` (+ overhang correction)
- On `iact_buffer_next_addr` event: triggers `iact_converter_enc_enable` and `iact_converter_params_enable`
- **Exit:** When all channel batches done → `WAIT_CYCLE`

#### `WAIT_CYCLE` (10)
Allows the converter pipeline to drain (4 × 4 = 16 extra cycles):
- `iact_buffer_SP_data_w <= iact_out_reg` (not actively used — legacy holdover)
- `send_data_reg <= 1` (triggers the data-flow always-block to start sending wgts to OpenEye_Parallel)
- **Exit:** After 16 cycles → `WAIT_FOR_RESULTS` or `RECEIVE_PSUMS_TO_IACT`

#### `WAIT_FOR_RESULTS` (11)
Waits while `OpenEye_Parallel` computes. Tracks iterations via `current_cycle` and `single_iteration` logic. When `last_data_o` pulses → done, return to `GET_PARAMETERS`.

#### `RECEIVE_PSUMS_TO_IACT` (12)
Used when `send_data_out == 0`: the computed psums are quantized and written back into the iact double-buffer to serve as activations for the next layer. The PSUM FSM sub-state `SEND_PSUM_TO_IACT` drives the quantization; this state assembles the resulting bytes into `buffer_SP_data_w_reg`. Three channel-count paths:
- `iact_channels_per_pe_next_layer == 4`: complex interleaved packing with overhang handling
- `iact_channels_per_pe_next_layer == 2`: simpler 2-channel packing
- `iact_channels_per_pe_next_layer == 1`: one channel per output pixel (FC or single-channel)
- **Exit:** When `fsm_psum_current_state == PSUM_IDLE` → `GET_PARAMETERS`

#### `MAXPOOLING_READ` (13)
Reads a 2×2 region from the iact buffer and computes the max via the 3-stage pipelined comparator tree. Updates `buffer_SP_addr_reg` to traverse the input feature map. Accumulates results into `pooling_regs[0..31]`. Exits when all pixels scanned → `MAXPOOLING_SEND`.

#### `MAXPOOLING_SEND` (14)
Writes max-pool results from `pooling_regs` back into the iact double-buffer cells at the appropriate addresses so that subsequent layers can read them. Returns to `GET_PARAMETERS` when done.

---

## 7. PSUM FSM (`fsm_psum_current_state`)

A separate FSM manages the partial-sum pipeline from compute-trigger through output, operating concurrently with the main FSM.

### States

| State | Value | Description |
|---|---|---|
| `PSUM_IDLE` | 0 | Idle; also handles bias loading when main FSM is in `GET_BIAS` |
| `WAIT_TO_SEND_READY_SIGNAL` | 1 | Waits for wgt/iact enables to go to 0, then sends ready to PEs |
| `CALCULATE_PSUM` | 2 | Feeds bias data from psum buffer into OpenEye_Parallel; waits for psum_ready_o |
| `PSUM_GET_RESULTS` | 3 | Collects output psums from OpenEye_Parallel into psum_buffer_SP |
| `WAIT_FOR_SENDING_RESULTS` | 4 | Waits 2 cycles then decides: send to host or loop to iact |
| `PSUM_SEND_RESULTS` | 5 | Streams psum buffer data to DMA output; sequentially iterates over all clusters/GLBs/filters |
| `SEND_PSUM_TO_IACT` | 6 | Quantizes psums and drives `quantized_value_reg` for iact repacking |

### PSUM FSM Detailed Flow

**`PSUM_IDLE`:**
- Clears `enable_dma_o`, `psum_buffer_SP_en_w`, etc.
- During `GET_BIAS` in main FSM: absorbs bias values into `psum_buffer_SP`
- On `compute_reg` pulse: resets psum buffer addresses; transitions to `WAIT_TO_SEND_READY_SIGNAL`

**`WAIT_TO_SEND_READY_SIGNAL`:**
- Waits until `wght_enable_i_reg == 0` and `iact_enable_i_oep_w == 0` (OpenEye_Parallel finished receiving)
- Counts 16 cycles then asserts `psum_ready_i_reg` (all ones) and transitions to `CALCULATE_PSUM`

**`CALCULATE_PSUM`:**
- Reads bias data from `psum_buffer_SP` (via `psum_buffer_SP_en_r`)
- Drives `psum_data_i_reg <= psum_buffer_SP_data_r` and `psum_enable_i_reg` to feed psums into OpenEye_Parallel
- Increments `psum_buffer_SP_addr_array` for each active GLB
- After `filters_reg` cycles: transitions to `PSUM_GET_RESULTS`

**`PSUM_GET_RESULTS`:**
- Captures `psum_data_o_w` (from OpenEye_Parallel) into `psum_buffer_SP` (via `psum_buffer_SP_en_w`)
- Increments psum buffer addresses as results come in
- Checks `results_ready` (AND of `psum_enable_o` for all active GLBs)
- When `fsm_psum_cycle >= filters_reg`:
  - If more iterations remain: go back to `WAIT_TO_SEND_READY_SIGNAL`; manage `storage_cycles` to select the correct psum buffer page
  - If all iterations done: go to `WAIT_FOR_SENDING_RESULTS`

**`WAIT_FOR_SENDING_RESULTS`:**
- If `send_data_out == 1`: transition to `PSUM_SEND_RESULTS` after 2 cycles
- If `send_data_out == 0`: transition to `SEND_PSUM_TO_IACT` (quantize for next layer) or `PSUM_IDLE` (if `store_in_psum`)

**`PSUM_SEND_RESULTS`:**
- Streams psum buffer contents to DMA output
- Iterates over: `fsm_psum_r` (GLB), `fsm_x_cl_psum` (column), `fsm_y_cl_psum` (row), `fsm_psum_cycle` (filter/row tile)
- `data_dma_o` is assigned the 40-bit slice from psum buffer read data (2 PARALLEL_MACS × TRANS_BITWIDTH_PSUM bits)
- `enable_dma_o` asserted when `ready_dma_i` is high and not at position 0
- Sets `last_data` when the last address is read; returns to `PSUM_IDLE` after `last_data_o`

**`SEND_PSUM_TO_IACT`:**
- Reads from psum buffer and applies quantization formula:
  `quantized = (quant_mant[f] * (psum + quant_offset[f])) >>> quant_exp[f]`
- Stores 8 quantized bytes into `quantized_value_reg[0..7]`
- Three hardware paths depending on topology: `CLUSTER_COLUMNS×NUM_GLB_PSUM >= 8` (large), `CLUSTERS×NUM_GLB_PSUM >= 8` (medium), or small
- Advances `psum_buffer_SP_addr_array` through the address sequence
- `fsm_psum_limit` terminates the state after all required outputs

---

## 8. Always Blocks / Processes

### Process 1: DMA Input Registration (`lines 248–256`)
**Trigger:** `posedge clk_i, negedge rst_n`

Registers `data_dma_i` and `enable_dma_i` into `data_dma_i_reg` / `enable_dma_i_reg` on each clock. On reset: both cleared to 0.

---

### Process 2: Cycle Counting (`lines 514–570`)
**Trigger:** `posedge clk_i, negedge rst_n`

Manages the iteration counters used in `WAIT_FOR_RESULTS` / `RECEIVE_PSUMS_TO_IACT`:
- **`single_iteration3`**: single-cycle pulse that fires on the first clock of each new iact delivery (detected when `iact_ready_o_oep_w != all-ones` and `single_iteration == 0`)
- **`current_cycle`**: increments on each `single_iteration3` pulse
- **`iact_router_counter`**: increments after each channel batch finishes; wraps at `needed_y_cls_reg`
- **`iact_cycle_count`**: increments when both channel and row counters wrap; tracks which weight-reuse iteration we are on
- Reset all counters when `fsm_current_state == GET_PARAMETERS` or `reset_cycle` asserted

---

### Process 3: Iact Converter Parameter Distribution (`lines 582–794`)
**Trigger:** `posedge clk_i, negedge rst_n`

Distributes spatial parameters to each `iact_stream_constructor`:
- Computes `iact_converter_x`, `iact_converter_y`, `iact_converter_c` by iterating over all cluster columns, rows, kernel indices, and y-lines
- On each iteration: writes a 36-bit params word `{row_offset[3:0], x[7:0], y[7:0], iact_size_x[7:0], c[7:0]}` to `iact_converter_params_reg[col][row]` and pulses `iact_converter_en_cfg_reg`
- Manages `param_array_reg` bitmask to select which clusters get updated
- During `GET_WGHT` or `GET_IACT`: sweeps over all clusters; during `CONVERT_IACT`: updates in lock-step with the encoding loop

---

### Process 4: Iact Converter Store Enable (`lines 797–848`)
**Trigger:** `posedge clk_i, negedge rst_n`

Controls `iact_converter_en_store_reg` and `conv_array_reg`:
- `conv_array_reg` = bitmask of converters to activate for storing; initialized to `start_param_array` in `GET_ROUTER_CONFIG`
- When `iact_converter_enc_enable` is asserted: enables all converters whose bit in `conv_array_reg` is set and advances the bitmask (circular shift for FC mode)
- For FC layers: `conv_array_reg` rotates left by 2 each cycle

---

### Process 5: Data-Flow to OpenEye_Parallel (`lines 851–996`)
**Trigger:** `posedge clk_i, negedge rst_n`

Manages weight streaming and compute trigger:
- When `send_data_reg` or `sending_data` is high: enters the "send" mode
  1. On first cycle: pulses `iact_converter_en_enc_reg` for all converters
  2. When `wght_sendable` and PE weight interfaces are ready: starts weight buffer read (`wght_buffer_SP_en_r`)
  3. Reads `wght_cnt + 1` words from `wght_buffer_SP` by advancing `wght_buffer_SP_rd_addr`
  4. On each weight word: builds `wght_enable_i_reg` mask indicating which cluster weight GLBs should receive data
  5. After all weights sent: if `current_cycle == 0`, fires `compute_reg` once
- On subsequent `current_cycle` iterations: re-triggers `iact_converter_en_enc_reg` and optionally rewinds `wght_buffer_SP_rd_addr` for weight reuse
- In `GET_PARAMETERS`: full reset of all send-phase registers; `wght_sendable <= 1`

---

### Process 6: Main FSM (`lines 1090–1964`)
**Trigger:** `posedge clk_i, negedge rst_n`

The main case-statement FSM described fully in Section 6 above. Drives all loading phases, converter control, and transitions between computation modes.

---

### Process 7: Router Mode Configuration (`lines 1966–2178`)
**Trigger:** `posedge clk_i, negedge rst_n`

Manages `router_mode_iact`, `router_mode_wght`, `router_mode_psum`, and `psum_choose_i_reg`:
- **During `GET_ROUTER_CONFIG`**: unpacks DMA words into the three router mode registers
- **On `compute_reg`**: sets `psum_choose_i_reg` based on `needed_y_cls_reg` (which cluster rows receive psum enables)
- **During `WAIT_FOR_RESULTS` / `RECEIVE_PSUMS_TO_IACT`**: on each `single_iteration3` pulse, updates `router_mode_iact` to advance the iact source one row down the cluster array (for the case where data flows through multiple rows), and updates `router_mode_psum` bit[2] (storage-accumulate flag) rotating through the cluster rows

---

### Process 8: PSUM FSM (`lines 2181–2731`)
**Trigger:** `posedge clk_i, negedge rst_n`

Full case-statement PSUM FSM described in Section 7 above. Also handles the `status_reg_enable_reg` override path that resets the entire PSUM state machine when a new layer starts.

---

## 9. Generate Blocks and Instantiations

### 9.1 `gen_RAM_wires` (genvar k_gen, lines 2734–2741)
Wires up the `buffer_SP_en_r/w/addr/data_w/data_r` arrays to the corresponding `_reg` registers. Also assigns `buffer_SP_data_r` from the low half of the concatenated double-port read output `buffer_SP_data_r_w`.

### 9.2 `UNPACKED_TRACES` generate (lines 2744–2768)
If `UNPACKED_TRACES_ENABLED == 1`, generates named wire aliases for each element of the unpacked arrays (`pooling_stage_1/2/3/4`, `quantized_value_reg`). This is purely for simulation visibility — waveform viewers cannot directly display unpacked arrays, so this exposes them as individual named wires.

### 9.3 `BUFFER_A` generate (j_gen, lines 2773–2786)
Generates 32 `RAM_SP` instances for the iact double-buffer. Each instance:
- Read enable: `buffer_SP_en_r[j] & !buffer_SP_en_w[j]` (priority write)
- Write enable: `buffer_SP_en_w[j]`
- Address: `{choose_iact_buffer, buffer_SP_addr[j]}` — MSB selects ping/pong half

### 9.4 `IACT_CONVERTER_X/Y` generate (i_gen/j_gen, lines 2788–2834)
Generates `CLUSTER_COLUMNS × CLUSTER_ROWS` instances of `iact_stream_constructor`. Connects:
- Shared `storage_i` from all 32 iact RAM cells (read in parallel)
- Individual `params`, `enable_config/store/converter`, `ready_o` per instance
- Local wires `iact_data_w`, `iact_enable_w`, `iact_choose_w` are then connected to the global `iact_*_oep_w` buses (lines 3024–3046)

### 9.5 Weight buffer instance (lines 2836–2847)
Single `RAM_SP` for weights. Address multiplexed: `wght_buffer_SP_wr_addr | wght_buffer_SP_rd_addr` (one is always 0 when the other is active since read/write are mutually exclusive).

### 9.6 `PSUM_RAM_X/Y/GLB` generate (lines 2849–2874)
Generates `CLUSTER_COLUMNS × CLUSTER_ROWS × NUM_GLB_PSUM/2` `RAM_SP` instances. Each stores 2 PSUM values (40-bit word). Also assigns debug signals for cell [0][0][0].

### 9.7 `dma_storage` instance (lines 2880–2928)
Connects the register-map decoder. All configuration wire outputs feed directly into main FSM logic and into `OpenEye_Parallel`.

### 9.8 `OpenEye_Parallel` instance (lines 2931–3023)
The compute core. All parameter passes match the FPGA wrapper's parameter set. Key interface signals:
- `compute_i <= compute_reg` (one-cycle trigger)
- Iact: from `iact_stream_constructor` outputs via `iact_*_oep_w` wires
- Wght: from `wght_buffer_SP_data_r` via `wght_data_i_w`
- Psum in/out: `psum_buffer_SP_data_r` / `psum_data_o_w`
- Config: all `dma_storage` outputs plus `router_mode_*` and `compute_mask_reg_port`

---

## 10. Key Computations and Data Paths

### 10.1 Iact Loading (host → double-buffer)

```
enable_dma_i
     │
data_dma_i ──reg──► data_dma_i_reg
                          │
                    buffer_SP_data_w_reg[current_buffer_n % 32]
                          │
                    RAM_SP[current_buffer_n % 32]
                    addr = {choose_iact_buffer, current_buffer_addr}
```

64-bit DMA words are written to successive RAM cells, rotating through cells 0–31 with address `current_buffer_addr` incrementing every 32 writes.

### 10.2 Iact Streaming (double-buffer → iact_stream_constructor → OpenEye_Parallel)

All 32 RAM cells are read in parallel every clock during `CONVERT_IACT`:
```
RAM_SP[0..31].data_o ──► buffer_SP_data_r[0..32×64-1]
                              │
                    iact_stream_constructor[cc][cr].storage_i
                              │ (format, sparse-encode, select)
                    iact_data_o, iact_enable_o, iact_choose_o
                              │
                    OpenEye_Parallel.iact_data_i / enable_i / choose_i
```

### 10.3 Weight Loading (host → wght_buffer_SP)

Two 24-bit rows per 64-bit DMA word; assembled into a word covering all cluster rows:
```
data_dma_i_reg[23:0]  → wght_buffer_SP_data_w[row][glb]
data_dma_i_reg[47:24] → wght_buffer_SP_data_w[row + CLUSTER_ROWS][glb]
                                    │
            wght_buffer_SP (written at wght_buffer_SP_wr_addr)
```

### 10.4 Weight Streaming (wght_buffer_SP → OpenEye_Parallel)

```
wght_buffer_SP.data_o ──► wght_data_i_w ──► OpenEye_Parallel.wght_data_i
wght_enable_i_reg     ────────────────────► OpenEye_Parallel.wght_enable_i
```
`wght_enable_i_reg` is a per-GLB mask: bit `a + b×NUM_GLB_WGHT` is set when cluster row `a` is within the range of active rows for the current spatial tile.

### 10.5 Partial Sum Path (OpenEye_Parallel → psum_buffer_SP → host / iact buffer)

```
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
```

### 10.6 Address Window Advancement During CONVERT_IACT

The 32 iact RAM cells are partitioned into a sliding window that advances one "row" per `iact_converter_buffer_addr_cycles` wrap:

```
window size = limit_increase_reg (+ overhang when fractional part accumulates ≥ 1)
upper_limit advances by limit_increase_reg each step
lower_limit advances by limit_increase_reg (+ overhang_delay) each step
```

Cells in `[lower_limit, upper_limit)` (mod 32) have their address incremented. This allows the 32 cells to represent a circular sliding window over the 2D feature map, covering `kernel_size` rows at a time.

### 10.7 `start_param_array` Computation

```verilog
start_param_array = (1 << N) - 1
where N = ((kernels_per_calc × y_lines_per_calc × ((iact_size_x-1+NUM_GLB_PSUM)/NUM_GLB_PSUM) × NUM_GLB_PSUM)
           + NUM_GLB_PSUM - 1) / NUM_GLB_PSUM
```

This is the number of clusters that need to receive activation data for one spatial tile. The initial value of `param_array_reg` is set to this and then shifted/rotated as tiles progress.

---

## 11. Waveform / Simulation Facilities

### `fst_path` and `$dumpvars`

```verilog
`ifndef NO_TRACE
  initial begin
    if ($value$plusargs("FST_PATH=%s", fst_path)) begin
      $dumpfile(fst_path);
      $dumpvars(0, OpenEye_FPGA);
    end else begin
      $dumpfile("OpenEye_FPGA.fst");
      $dumpvars(0, OpenEye_FPGA);
    end
  end
`endif
```

When `NO_TRACE` is not defined, the simulation dumps all signals in `OpenEye_FPGA` to an FST file. The path can be overridden with `+FST_PATH=<path>` on the simulator command line.

### `UNPACKED_TRACES_ENABLED`

When set to `1` (default), generates named wire aliases for all arrays that would otherwise be invisible in VCD/FST waveform viewers:
- `pooling_stage_1_traces[0..7]`
- `pooling_stage_2_traces[0..3]`
- `pooling_stage_3_traces[0..1]`
- `pooling_stage_out_trace[0..31]` (pooling_regs)
- `quantized_out_trace[0..7]` (quantized_value_reg)

### Debug Ports

As listed in Section 4.4, several debug output ports expose internal SRAM control signals for external ChipScope/SignalTap logic analyzers on FPGA.
