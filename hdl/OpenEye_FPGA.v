// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: OpenEye_FPGA
///
/// Overview:
/// The OpenEye_FPGA module adapts the OpenEye accelerator architecture for FPGA implementation,
/// addressing the port limitations typical in FPGA platforms. It encapsulates the OpenEye_Parallel
/// module and manages data flow through DMA interfaces with buffering capabilities.
///
/// Key Features:
/// 1. FPGA-Optimized Interface:
///    - Implements DMA-based data transfer
///    - Handles port limitations through efficient buffering
///    - Uses handshake protocol for reliable data transfer
///
/// 2. Data Management:
///    - Variable-length FIFO buffering for output data
///    - Handles timing variations and data delays
///    - Manages data synchronization between host and accelerator
///
/// 3. Memory Organization:
///    - Global Buffer (GLB) management for activations, weights, and partial sums
///    - Configurable memory depths and widths
///    - Efficient data routing between memory and processing elements
///
/// 4. Processing Control:
///    - Configurable processing parameters
///    - Multiple operation modes support
///    - Flexible computation scheduling
///
/// Operation Modes:
/// 1. Data Loading:
///    - DMA transfers for input activations and weights
///    - Buffering in respective memory banks
///    - Handshake-based transfer control
///
/// 2. Computation:
///    - Orchestrates data movement to processing elements
///    - Manages computation timing and synchronization
///    - Handles partial results accumulation
///
/// 3. Result Collection:
///    - Buffers computed results
///    - Manages output data transfer
///    - Implements flow control for host interface
///
///
/// Parameters:
///   CLUSTER_ROWS           - Number of cluster rows
///   CLUSTER_COLUMNS        - Number of cluster columns
///   NUM_GLB_IACT           - Number of activation global buffers
///   NUM_GLB_WGHT           - Number of weight global buffers
///   NUM_GLB_PSUM           - Number of partial-sum global buffers
///   IACT_RAM_CELLS             - Number of RAM cells used for buffering
///   BRANCHES              - Number of branch paths in the storage structure
///   BUFFER_WIDTH          - Width of the buffering datapath
///   QUANT_AMOUNT          - Quantization-related parameter count
///   DATA_PSUM_BITWIDTH    - Width of the partial-sum data path
///   TRANS_WORDS           - Number of DMA transfer words
///   IS_TOPLEVEL           - Selects top-level behavior for this instance
///   SERIAL                - Enables serial execution mode
///   PARALLEL_MACS         - Number of parallel MAC units
///   SPARSITY_EN           - Enables sparse-mode operation
///   ADDR_IACT_BITWIDTH    - Address width for activation memories
///   ADDR_WGHT_BITWIDTH    - Address width for weight memories
///   DATA_IACT_BITWIDTH    - Width of activation data
///   DATA_WGHT_BITWIDTH    - Width of weight data
///   TRANSMISSIONS         - Number of DMA transmissions
///   TRANS_BITWIDTH_IACT   - DMA bit width for activation data
///   TRANS_BITWIDTH_WGHT   - DMA bit width for weight data
///   TRANS_BITWIDTH_PSUM   - DMA bit width for partial-sum data
///   DATA_IACT_OVERHEAD    - Overhead bits appended to activation data
///   PES                   - Total number of processing elements
///   CLUSTERS              - Total number of clusters
///   IACT_ADDR_PER_PE      - Activation address words per PE
///   WGHT_ADDR_PER_PE      - Weight address words per PE
///   IACT_PER_PE           - Activation words per PE
///   PSUM_PER_PE           - Partial-sum words per PE
///   WGHT_PER_PE           - Weight words per PE
///   IACT_MEM_ADDR_WORDS   - Number of activation memory words
///   PSUM_MEM_ADDR_WORDS  - Number of partial-sum memory words
///   IACT_MEM_ADDR_BITS    - Bit width of activation memory addresses
///   PSUM_MEM_ADDR_BITS    - Bit width of partial-sum memory addresses
///   ROUTER_MODES_IACT     - Router mode count for activation traffic
///   ROUTER_MODES_WGHT     - Router mode count for weight traffic
///   ROUTER_MODES_PSUM     - Router mode count for partial-sum traffic
///   DMA_BITWIDTH          - DMA data width
///   BANO_MODES            - Batch-normalization modes
///   AF_MODES              - Activation-function modes
///
/// Ports:
///   clk_i                 - System clock input
///   rst_ni                - Active-low reset input
///   ready_dma_o           - DMA-ready signal from FPGA to host
///   data_dma_i            - DMA input data from host
///   enable_dma_i          - DMA input-enable signal from host
///   ready_dma_i           - DMA-ready signal from host to FPGA
///   data_dma_o            - DMA output data from FPGA
///   enable_dma_o          - DMA output-enable signal from FPGA
///   last_data_o           - Asserted on the final output data word
///
module OpenEye_FPGA #(
    //Set parameters
  `ifdef USE_INTERNAL_PARAMS
      parameter CLUSTER_ROWS  = 2,
      parameter NUM_GLB_IACT  = 3,
      parameter NUM_GLB_PSUM  = 4,
      parameter NUM_GLB_WGHT  = 3,
      parameter IACT_RAM_CELLS= 8,
      parameter BRANCHES      = 1,
      parameter BUFFER_WIDTH  = 10,
      parameter BUFFER_WIDTH_PSUM = 14,
      parameter BUFFER_WIDTH_WGHT = 12,
      parameter QUANT_AMOUNT  = 32,
      parameter DATA_PSUM_BITWIDTH = 32,
      parameter CLUSTER_COLUMNS = 2,
      parameter TRANS_WORDS = 4,
      parameter DMA_BITWIDTH  = 64,
      parameter TRANSMISSIONS = 8,
      parameter PARALLEL_MACS = 2,
      parameter SPARSITY_EN   = 1,  // 1=sparse mode (default), 0=dense mode
      parameter TRANS_BITWIDTH_IACT = 24,
      parameter TRANS_BITWIDTH_WGHT = 8,
  `else
    `include "parameters_FPGA.vh"
      // Defaultvalues
  `endif
    parameter IS_TOPLEVEL   = 1,
    parameter SERIAL        = 1,

    parameter ADDR_IACT_BITWIDTH = 4,
    parameter ADDR_WGHT_BITWIDTH = 8,

    parameter DATA_IACT_BITWIDTH = 8,
    parameter DATA_WGHT_BITWIDTH = 8,

    // Number of 64-bit DMA words gating GET_PARAMETERS's write-cycle loop
    // into dma_storage's status-word shift register (see dma_storage.v's
    // "Pipelined Shift Register Architecture": every transmission must be
    // written or the whole chain misaligns by one word, corrupting every
    // field). Must equal the generated regmap_params.vh's own TRANSMISSIONS
    // constant (currently 8, from test/cocotb_fpga/regmap.yaml's field
    // list) - OpenEye_FPGA.v does not `include regmap_params.vh (it only
    // consumes dma_storage's decoded output wires), so this count has to be
    // kept in sync by hand whenever fields are added to regmap.yaml.
    // Per-PE psum transfer width. Must match PE.v's internal
    // TRANS_BITWIDTH_PSUM = DATA_PSUM_BITWIDTH * (SERIAL ? 1 : PARALLEL_MACS);
    // a fixed value (was 32) mismatches the generated DATA_PSUM_BITWIDTH (e.g.
    // 20) and prunes the psum bus feeding each PE, corrupting results.
    parameter TRANS_BITWIDTH_PSUM = DATA_PSUM_BITWIDTH * (SERIAL == 1 ? 1 : PARALLEL_MACS),
    parameter PSUM_BUFFER_WIDTH = NUM_GLB_PSUM == 1 ? DATA_PSUM_BITWIDTH : DATA_PSUM_BITWIDTH * 2,
    parameter DATA_IACT_OVERHEAD  = 4,

    parameter PES = NUM_GLB_PSUM * NUM_GLB_WGHT,

    parameter CLUSTERS = CLUSTER_COLUMNS * CLUSTER_ROWS,

    parameter IACT_ADDR_PER_PE = 9,
    parameter WGHT_ADDR_PER_PE = 16,

    parameter IACT_PER_PE = 16,
    parameter PSUM_PER_PE = 32,
    parameter WGHT_PER_PE = 96,

    parameter IACT_MEM_ADDR_WORDS = 512,
    parameter IACT_FSM_CYCL_WORDS = IACT_PER_PE + IACT_ADDR_PER_PE,
    parameter WGHT_FSM_CYCL_WORDS = WGHT_PER_PE + WGHT_ADDR_PER_PE,
    parameter PSUM_MEM_ADDR_WORDS = 768,

    parameter IACT_MEM_ADDR_BITS = $clog2(IACT_MEM_ADDR_WORDS),
    parameter PSUM_MEM_ADDR_BITS = $clog2(PSUM_MEM_ADDR_WORDS),

    parameter ROUTER_MODES_IACT = 6,
    parameter ROUTER_MODES_WGHT = 1,
    parameter ROUTER_MODES_PSUM = 3,


    parameter BUFFER_WIDTH_IACT_STREAM_CONSTRUCTOR = BUFFER_WIDTH,
    parameter real FSM_IACT_RTR_CCLS_A = (CLUSTERS * NUM_GLB_IACT),
    parameter real FSM_IACT_RTR_CCLS_B = DMA_BITWIDTH / ROUTER_MODES_IACT,
    parameter real FSM_IACT_RTR_CCLS = FSM_IACT_RTR_CCLS_A / FSM_IACT_RTR_CCLS_B,
    parameter real FSM_WGHT_RTR_CCLS_A = (CLUSTERS * NUM_GLB_WGHT),
    parameter real FSM_WGHT_RTR_CCLS_B = DMA_BITWIDTH / ROUTER_MODES_WGHT,
    parameter real FSM_WGHT_RTR_CCLS = FSM_WGHT_RTR_CCLS_A / FSM_WGHT_RTR_CCLS_B,
    parameter real fsm_psum_rTR_CCLS_A = (CLUSTERS * NUM_GLB_PSUM),
    parameter real fsm_psum_rTR_CCLS_B = DMA_BITWIDTH / ROUTER_MODES_PSUM,
    parameter integer fsm_psum_rTR_CCLS_C = DMA_BITWIDTH - (DMA_BITWIDTH % ROUTER_MODES_PSUM),
    parameter real fsm_psum_rTR_CCLS = fsm_psum_rTR_CCLS_A / fsm_psum_rTR_CCLS_B,

    parameter integer FSM_CEIL_IACT_RTR_CCLS = $rtoi($ceil(FSM_IACT_RTR_CCLS)),
    parameter integer FSM_CEIL_WGHT_RTR_CCLS = $rtoi($ceil(FSM_WGHT_RTR_CCLS)),
    parameter integer FSM_CEIL_PSUM_RTR_CCLS = $rtoi($ceil(fsm_psum_rTR_CCLS)),


    //Number of Words per PE
    parameter BANO_MODES = 2,
    parameter AF_MODES   = 4,

    // Converter
    parameter BITWIDTH_IACT = TRANS_BITWIDTH_IACT / 2,

    //Storage RAMs
    parameter BRANCHES_CLOG = $clog2(BRANCHES),
    parameter BRANCHES_WIDTH = BUFFER_WIDTH - BRANCHES_CLOG,

    //Enable Traces for unpacked arrays
    parameter UNPACKED_TRACES_ENABLED = 1,

    //Pooling Features
    parameter AVERAGE_POOLING = 0,

    //Pooling Features
    parameter STRIDE_ENABLED = 0,

    //Channels per word
    parameter CHANNELS_PER_WORD = 4,

    //Quantization Width
    parameter integer OFFSET_WIDTH = 8,
    parameter integer EXPONENT_WIDTH = 7,  
    parameter integer MANTISSA_WIDTH = 25,

    localparam FINAL_DATA_IACT_BITWIDTH = SPARSITY_EN ? 4 + DATA_IACT_BITWIDTH : DATA_IACT_BITWIDTH,
    localparam FINAL_DATA_WGHT_BITWIDTH = SPARSITY_EN ? 4 + DATA_WGHT_BITWIDTH : DATA_WGHT_BITWIDTH,

    localparam integer IACT_RAM_CELLS_VALUES_PER_WORD = 8,
    localparam integer IACT_RAM_CELLS_WORD_BITWIDTH = DATA_IACT_BITWIDTH * IACT_RAM_CELLS_VALUES_PER_WORD,
    localparam integer WGHT_RAM_CELLS_WORD_BITWIDTH = FINAL_DATA_WGHT_BITWIDTH * PARALLEL_MACS * NUM_GLB_WGHT,
    localparam integer PSUM_RAM_CELLS_WORD_BITWIDTH = DATA_PSUM_BITWIDTH * 2,

    localparam IACT_WORDS_IN_RAM = IACT_RAM_CELLS_WORD_BITWIDTH / DATA_IACT_BITWIDTH,
    localparam WORDS_PER_CYCLE   = 2,
    localparam PSUM_TO_IACT_CYCLES = (CLUSTER_COLUMNS * NUM_GLB_PSUM) == 4 ? 2 : 1,
    localparam integer IACT_CYCLES_ONE_WORD_ALL_CELLS = ((IACT_RAM_CELLS * IACT_RAM_CELLS_WORD_BITWIDTH) + DMA_BITWIDTH - 1) / DMA_BITWIDTH,
    localparam IACT_CELL_INPUT_WIDTH = IACT_CYCLES_ONE_WORD_ALL_CELLS * DMA_BITWIDTH,
    localparam integer IACT_ONE_WORD_ALL_RAM = $ceil(IACT_RAM_CELLS*IACT_RAM_CELLS_WORD_BITWIDTH/DMA_BITWIDTH),

    localparam WGHT_RAM_CELLS = CLUSTERS * NUM_GLB_WGHT,
    localparam integer WGHT_CYCLES_ONE_WORD_ALL_CELLS = ((WGHT_RAM_CELLS * WGHT_RAM_CELLS_WORD_BITWIDTH) + DMA_BITWIDTH - 1) / DMA_BITWIDTH,
    localparam WGHT_CELL_INPUT_WIDTH = WGHT_CYCLES_ONE_WORD_ALL_CELLS * DMA_BITWIDTH,
    localparam integer WGHT_ONE_WORD_ALL_RAM = ((CLUSTERS*WGHT_RAM_CELLS_WORD_BITWIDTH)+DMA_BITWIDTH-1)/DMA_BITWIDTH,

    localparam integer PSUM_RAM_CELLS = CLUSTERS * (NUM_GLB_PSUM+1)/2,
    localparam integer PSUM_CYCLES_ONE_WORD_ALL_CELLS = ((PSUM_RAM_CELLS * PSUM_RAM_CELLS_WORD_BITWIDTH) + DMA_BITWIDTH - 1) / DMA_BITWIDTH,
    localparam         PSUM_CELL_INPUT_WIDTH = PSUM_CYCLES_ONE_WORD_ALL_CELLS * DMA_BITWIDTH,
    localparam integer PSUM_ONE_WORD_ALL_RAM = ((CLUSTERS*PSUM_RAM_CELLS_WORD_BITWIDTH)+DMA_BITWIDTH-1)/DMA_BITWIDTH,
    localparam integer QUANT_TRANS_CYCLES = (((OFFSET_WIDTH + EXPONENT_WIDTH + MANTISSA_WIDTH) * QUANT_AMOUNT) + DMA_BITWIDTH - 1)/DMA_BITWIDTH,
    localparam integer QUANT_TRANS_WIDTH = QUANT_TRANS_CYCLES * DMA_BITWIDTH,

    localparam integer TEMP_WORDS_PSUM_TO_IACT = IACT_RAM_CELLS_VALUES_PER_WORD/TRANS_WORDS,

    localparam integer TEMP_BITS_PSUM_TO_IACT = IACT_RAM_CELLS_VALUES_PER_WORD * DATA_IACT_BITWIDTH * TEMP_WORDS_PSUM_TO_IACT,

    localparam integer PSUM_OUTPUT_WORDS = DMA_BITWIDTH == 64? 2 : 1
) (
    //Clock and Reset
    input clk_i,
    input rst_ni,

    //Input DMA
    output reg                      ready_dma_o,
    input      [DMA_BITWIDTH-1 : 0] data_dma_i,
    input                           enable_dma_i,

    //Output DMA
    input                           ready_dma_i,
    output reg [DMA_BITWIDTH-1 : 0] data_dma_o,
    output reg                      enable_dma_o,

    output reg last_data_o

);

  //#######################
  // Reset Synchronization
  // RST_SYNC instantiates a 2-FF synchronizer so that the de-assertion of rst_ni is
  // aligned to clk_i before it fans out to all sequential logic.  The synchronized
  // output rst_n is used everywhere inside this module instead of rst_ni directly.
  //#######################
  wire rst_n; // Synchronized, active-low reset; driven by RST_SYNC below.

  RST_SYNC rst_sync_wrapper (
      .clk_i (clk_i),
      .rst_ni(rst_ni),
      .rst_no(rst_n)
  );

  //#######################
  // DMA Input Pipeline Register
  // One register stage on the DMA input interface.  This breaks the combinatorial
  // path from the DMA bus into the FSM and improves timing closure on FPGA.
  // A transfer on the DMA input occurs when enable_dma_i_reg AND ready_dma_o are
  // both high in the same cycle (simple valid/ready handshake).
  //#######################

  reg [DMA_BITWIDTH-1:0] data_dma_i_reg;  // Registered DMA input data word (64 b). Sampled each cycle.
  reg                    enable_dma_i_reg; // Registered DMA input valid flag.

  // Process: DMA input registration
  // Latches data_dma_i and enable_dma_i on every rising clock edge so that
  // downstream logic always sees a stable, timing-closed copy of the bus.
  always @(posedge clk_i, negedge rst_n) begin
    if (rst_n == 0) begin
      data_dma_i_reg   <= 0;
      enable_dma_i_reg <= 0;
    end else begin
      data_dma_i_reg   <= data_dma_i;
      enable_dma_i_reg <= enable_dma_i;
    end
  end


reg [1023:0] fst_path;
`ifndef NO_TRACE
  initial begin
    // Read the path from the command line argument
    if ($value$plusargs("FST_PATH=%s", fst_path)) begin
      $dumpfile(fst_path);
      $dumpvars(0, OpenEye_FPGA);
    end else begin
      // Fallback for when the argument is not provided
      $dumpfile("OpenEye_FPGA.fst");
      $dumpvars(0, OpenEye_FPGA);
    end
  end
`endif


  // -----------------------------------------------------------------------
  // Layer Hyperparameter Registers
  // These registers are set once per layer during GET_PARAMETERS / GET_ROUTER_CONFIG
  // and remain stable throughout the entire compute phase.
  // Most are wires driven by the dma_storage register-map decoder; a few are
  // local regs that the main FSM derives from the decoded values.
  // -----------------------------------------------------------------------
  reg data_mode_reg;                              // 0 = fixed-point, 1 = floating-point computation mode passed to OpenEye_Parallel.
  reg [$clog2(DATA_PSUM_BITWIDTH)-1:0] fraction_bit_reg; // Position of the binary point in fixed-point psums (passed to OpenEye_Parallel).
  wire [17:0] needed_cycles;                 // Total iact-delivery iterations required to complete this layer (from dma_storage).
  wire [$clog2(CLUSTER_COLUMNS+1)-1:0] needed_x_cls_reg;// Number of active cluster columns for this layer (from dma_storage).
  wire [$clog2(CLUSTER_ROWS+1)-1:0] needed_y_cls_reg; // Number of active cluster rows for this layer (from dma_storage).
  wire [3:0] needed_iact_cycles_reg;             // Number of iact router broadcast cycles per spatial position (from dma_storage).
  wire [$clog2(PSUM_PER_PE+1)-1:0] filters; // Output filters per PE batch; also the psum address stride (from dma_storage).
  wire [$clog2(WGHT_ADDR_PER_PE+1)-1:0] wght_addr_len_reg; // Weight address scratchpad length per PE (from dma_storage).
  reg [$clog2(BANO_MODES)*NUM_GLB_PSUM-1:0] bano_cluster_mode_reg; // Batch-normalisation mode bits per psum GLB (set in GET_PARAMETERS, forwarded to OpenEye_Parallel).
  reg [$clog2(AF_MODES)-1:0] af_cluster_mode_reg;  // Activation-function mode (set in GET_PARAMETERS, forwarded to OpenEye_Parallel).
  wire [4:0] input_activations;                  // Number of non-zero activations per MAC cycle (from dma_storage).
  wire [7:0] wght_cycles_reg;                    // DMA cycles needed to stream one complete set of weights (from dma_storage).
  wire [2:0] stride_x_dma;                       // Convolution stride in the x-direction (from dma_storage).
  wire [2:0] stride_y_dma;                       // Convolution stride in the y-direction (from dma_storage).
  wire [2:0] stride_x;                           // Convolution stride in the x-direction (from dma_storage).
  wire [2:0] stride_y;                           // Convolution stride in the y-direction (from dma_storage).
  assign stride_x = STRIDE_ENABLED == 1 ? stride_x_dma : 1;
  assign stride_y = STRIDE_ENABLED == 1 ? stride_y_dma : 1;
  wire skipIact_reg;                             // When 1: skip GET_IACT phase (activations already in buffer from previous layer).
  wire skipWght_reg;                             // When 1: skip GET_WGHT phase (weights unchanged from previous layer).
  wire skipPsum_reg;                             // When 1: skip GET_BIAS / GET_QUANTIZE phases (no bias to load).
  wire [4-1:0] kernel_per_pe_cluster_reg;        // Number of filter kernels mapped to a single PE cluster (from dma_storage).
  wire [5:0] kernel_size_x;                      // Spatial kernel size (e.g. 3 for 3×3 conv); used to compute padding and iact_converter_max_cycles.
  wire [3:0] kernel_size_y;                      // Spatial kernel size (e.g. 3 for 3×3 conv); used to compute padding and iact_converter_max_cycles.
  reg [3:0] padding_x;                           // Zero-padding amount = (kernel_sizeX-1)/2.
  reg [3:0] padding_y;                           // Zero-padding amount = (kernel_sizeY-1)/2.
  reg [DMA_BITWIDTH-1 : 0] fifo_data_i;          // Data word written into the output varlenFIFO (currently driven to 0).
  reg fifo_read_i;                               // Read strobe for the output varlenFIFO (currently driven to 0).
  reg fifo_write_i;                              // Write strobe for the output varlenFIFO (currently driven to 0).
  wire [3:0] psum_q;                     // Pipeline delay cycles through the psum GLB cluster (from dma_storage, forwarded to OpenEye_Parallel).
  reg [CLUSTERS-1:0] conv_array_reg;             // Bitmask: which clusters are enabled to receive a store-enable pulse in the current cycle. Rotated each iact-encode step for FC layers.
  reg [CLUSTERS-1:0] param_array_reg;            // Bitmask: which clusters receive a config-enable pulse. Initialised to start_param_array at GET_ROUTER_CONFIG.
  wire [CLUSTERS-1:0] start_param_array;         // Initial bitmask for param_array_reg; encodes how many clusters need data for one spatial tile.
                                                 // Formula: (1 << N) - 1 where N = ceil((kernels * y_lines * ceil(iact_size_x/NUM_GLB_PSUM)) / 1)
  wire [7:0]needed_psum_storage_cycles_reg;      // How many PSUM accumulation passes are required before the final result is complete (from dma_storage).
  wire      pooling_mode;
  wire      gemm_mode;                    // Dataflow select from dma_storage: 0 = row-stationary conv, 1 = output-stationary GEMM.
  reg [7:0] debug_reg;                           // Scratch debug register; written with small integer literals inside RECEIVE_PSUMS_TO_IACT to mark which packing branch was taken.

  // -----------------------------------------------------------------------
  // Main FSM State and Cycle Counters
  // -----------------------------------------------------------------------
  reg [32-1:0] fsm_cycle;                              // General-purpose per-state cycle counter; reset to 0 on every state transition.
  wire [15:0]                       trans_cycles_iact; // Register for deciding, how many cycles are needed for transmissions for iact
  wire [15:0]                       trans_cycles_wght; // Register for deciding, how many cycles are needed for transmissions for wght
  wire [15:0]                       trans_cycles_psum; // Register for deciding, how many cycles are needed for transmissions for psum
  reg [$clog2(CLUSTER_COLUMNS)-1:0] fsm_x_cl;          // Current cluster column being processed during weight loading.
  reg [$clog2(CLUSTER_ROWS)-1:0] fsm_y_cl;             // Current cluster row being processed during weight loading.
  reg [$clog2(NUM_GLB_IACT)-1:0] fsm_iact_r;           // Current iact GLB index during iact loading (unused after refactor but kept for compatibility).
  reg [$clog2(NUM_GLB_WGHT)-1:0] fsm_wght_r;           // Current weight GLB index; increments each DMA word in GET_WGHT, wraps at NUM_GLB_WGHT.
  reg [$clog2(NUM_GLB_PSUM)-1:0] fsm_psum_r;           // Current psum GLB index during psum output; used by PSUM FSM.
  reg [$clog2(NUM_GLB_PSUM)-1:0] fsm_psum_r_q;         // One-cycle delayed fsm_psum_r; compensates for the pipelined RAM read in PSUM_SEND_RESULTS.
  reg results_ready;                                   // Local flag (combinatorial inside always block): AND of all active psum_enable_o or psum_ready_o signals.
  reg [19:0] finished_cycles_iact;                     // Count of completed iact delivery iterations; used as address base in MAXPOOLING_SEND.
  reg [19:0] finished_cycles_psum;                     // Count of completed psum output passes; determines when last_data fires.
  reg reset_cycle;                                     // Pulse that resets all iteration counters (current_cycle, iact_cycle_count, etc.) to 0.
  wire send_data_out;                                  // When 1: final results go directly to DMA output (PSUM_SEND_RESULTS). When 0: quantize and loop back as next-layer iact.
  wire [2:0] add_up;                                   // Extra overlap columns beyond iact_size_x needed for the sliding iact window (from dma_storage).
  wire [7:0] iact_x_line_repetitions;                  // How many times each iact x-line is reused across cluster columns (from dma_storage).
  wire [7:0] buffer_cycles_for_x_iact;                 // How many times the Iact GLBs need to cycle for a given iact_x_size.
  reg [7:0] psum_x_with_add_up;
  wire [16:0] fsm_psum_limit;                          // Total fsm_psum_cycle count before SEND_PSUM_TO_IACT returns to PSUM_IDLE.
  wire [$clog2(CLUSTERS+1)-1:0] cluster_per_conv_cycle;// Amount of clusters, that are written IACTs in parallel 

  // -----------------------------------------------------------------------
  // PAGU Values
  // -----------------------------------------------------------------------
  wire [ 7:0] iact_read_limit_0;
  wire [ 7:0] iact_read_limit_1;
  wire [ 7:0] iact_read_limit_2;
  wire [ 7:0] iact_read_limit_3;
  wire [ 7:0] iact_read_limit_4;
  wire [11:0] iact_read_inc_0;
  wire [11:0] iact_read_inc_1;
  wire [11:0] iact_read_inc_2;
  wire [11:0] iact_read_inc_3;
  wire [11:0] iact_read_inc_4;
  wire [11:0] iact_write_limit_0;
  wire [ 7:0] iact_write_limit_1;
  wire [ 7:0] iact_write_limit_2;
  wire [11:0] iact_write_inc_0;
  wire [11:0] iact_write_inc_1;
  wire [11:0] iact_write_inc_2;
  wire [ 9:0] pagu_wght_limit;
  wire [8-1:0]psum_pagu_loop_limit_0;
  wire [8-1:0]psum_pagu_loop_limit_1;
  wire [8-1:0]psum_pagu_loop_limit_2;
  wire [8-1:0]psum_pagu_loop_limit_3;
  wire [8-1:0]psum_pagu_loop_limit_4;
  wire [8-1:0]psum_pagu_addr_inc_0;
  wire [8-1:0]psum_pagu_addr_inc_1;
  wire [8-1:0]psum_pagu_addr_inc_2;
  wire [8-1:0]psum_pagu_addr_inc_3;
  wire [8-1:0]psum_pagu_addr_inc_4;


  // -----------------------------------------------------------------------
  // Iact Double-Buffer Control
  // The 32 RAM_SP cells (BUFFER_A) form a circular sliding window over the
  // input feature map.  During GET_IACT the host writes raw INT8 pixels in;
  // during CONVERT_IACT the cells are read in parallel and fed to the
  // iact_stream_constructors.  A single address MSB (choose_iact_buffer)
  // selects which physical half (ping / pong) of each cell is active,
  // allowing one half to be written while the other is being read.
  // -----------------------------------------------------------------------
  reg          buffer_select;       // Legacy / unused; choose_iact_buffer is the active double-buffer selector.
  reg [8-1:0]  buffer_addr_upper_limit; // Index of the cell just past the upper edge of the active write window (mod IACT_RAM_CELLS).
  reg [8-1:0]  buffer_addr_lower_limit; // Index of the first cell in the active write window (mod IACT_RAM_CELLS).
  wire [8-1:0] limit_increase;            // How many cells the active window advances per y-line step.
  wire [8-1:0] initial_upper_limit;       // Initial amount of Iact Clusters, that change in the first iteration
  reg [8-1:0]  limit_increase_reg;         // How many cells the active window advances per y-line step in the psum to iact operation.
  wire [9:0]   lower_bound;
  wire [9:0]   upper_bound;
  wire [4-1:0] overhang_discrepancy;       // Fractional part of (cells_per_line); accumulated to detect when an extra cell (+overhang) is needed.
  reg [4-1:0]  overhang_counter;           // Running total of fractional increments; generates overhang pulse when >= WORDS_PER_CYCLE*4.
  reg          overhang;                   // Extra +1 added to upper_limit in the current step when overhang_counter wraps.
  reg          overhang_delay;             // One-cycle delayed version of overhang; applied to lower_limit one cycle after upper_limit.

  // -----------------------------------------------------------------------
  // Weight Staging Buffer Control
  // A single wide RAM_SP holds all weight words for the current layer.
  // During GET_WGHT the host fills it; during the send phase it is read
  // sequentially and forwarded to OpenEye_Parallel via wght_data_i_w.
  // Read and write addresses are OR-combined on the RAM address port
  // (only one is non-zero at any time).
  // -----------------------------------------------------------------------
  reg                                                  wght_buffer_en_r;            // Read enable for the weight staging RAM.
  reg                                                  wght_buffer_en_w;            // Write enable for the weight staging RAM.
  reg [BUFFER_WIDTH_WGHT-1:0]                          wght_buffer_wr_addr;         // Write-address pointer; incremented each time a full weight row is assembled.
  reg [BUFFER_WIDTH_WGHT-1:0]                          wght_buffer_rd_addr;         // Read-address pointer; incremented each clock during the send phase.
  reg [BUFFER_WIDTH_WGHT-1:0]                          wght_buffer_rd_addr_storage; // Saved read address to rewind to the start of the current weight block after each channel batch.
  reg [WGHT_CELL_INPUT_WIDTH-1:0]                      wght_buffer_data_w;          // Write-data bus: assembled from pairs of DMA words, one row per cycle.
  wire [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_buffer_data_r;          // Read-data bus; directly assigned to wght_data_i_w → OpenEye_Parallel.

  // -----------------------------------------------------------------------
  // Psum Staging Buffer Control
  // An array of RAM_SP instances (one per cluster column × row × GLB pair)
  // stores bias values (loaded in GET_BIAS), accumulates partial sums written
  // by OpenEye_Parallel (in PSUM_GET_RESULTS), and supplies them for DMA
  // output (PSUM_SEND_RESULTS) or quantization (SEND_PSUM_TO_IACT).
  // -----------------------------------------------------------------------
  reg  [CLUSTERS*(NUM_GLB_PSUM+1)/2-1:0] psum_buffer_en_r;           // Per-buffer read-enable vector (one bit per RAM instance).
  reg  [CLUSTERS*(NUM_GLB_PSUM+1)/2-1:0] psum_buffer_en_w;           // Per-buffer write-enable vector.
  wire [BUFFER_WIDTH_PSUM*CLUSTERS*(NUM_GLB_PSUM+1)/2-1:0] psum_buffer_addr; // Flattened address bus; driven from psum_buffer_addr_array.
  reg  [BUFFER_WIDTH_PSUM-1:0] psum_buffer_addr_storage;              // Base address for the start of the current output page; advances by filters after each complete psum pass.
  reg  [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_buffer_data_w;  // Write-data bus to all psum buffers; MUXed between bias-load and PE-output paths.
  wire [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_buffer_data_r;  // Read-data bus from all psum buffers; fed into OpenEye_Parallel or into the quantizer.

  wire [BUFFER_WIDTH_WGHT-1:0] wghts_per_pe;   // Total weight words loaded; used as the upper-bound of the weight-send loop.
  reg  [BUFFER_WIDTH_PSUM-1:0] psum_cnt; // Total psum words in one output pass; set at end of GET_BIAS from the bias buffer depth.

  // -----------------------------------------------------------------------
  // Iact Stream Loading Registers
  // Used during GET_IACT to distribute incoming DMA words across the 32
  // double-buffer cells in a round-robin fashion.
  // -----------------------------------------------------------------------
  // Shared between GET_IACT (wraps at IACT_ONE_WORD_ALL_RAM) and GET_WGHT
  // (wraps at WGHT_ONE_WORD_ALL_RAM) - must be wide enough for whichever
  // modulus is larger, or the counter silently overflows/wraps early and
  // wght_buffer_en_w/iact_buffer_en_w never assert (weight/iact staging
  // RAM never written, downstream reads return X forever). A fixed 4-bit
  // width happens to fit the default configs but silently breaks for any
  // parameter set where either modulus exceeds 16.
  reg [$clog2(IACT_ONE_WORD_ALL_RAM > WGHT_ONE_WORD_ALL_RAM ? IACT_ONE_WORD_ALL_RAM : WGHT_ONE_WORD_ALL_RAM)-1:0] current_buffer;   // Index of the cell currently being written; increments each DMA word.
  wire [11:0] iact_size_x;       // Feature map width in pixels (from dma_storage).
  wire [ 7:0] iact_size_y;       // Feature map height in pixels (from dma_storage).
  wire [11:0] iact_size_c;      // Total input channel count for this PE batch; computed in GET_ROUTER_CONFIG as iact_channels_per_pe * iact_channel_max_cycles.
  wire [11:0] psum_size_x;       // Output map width in pixels (from dma_storage).
  wire [ 7:0] psum_size_y;       // Output map height in pixels (from dma_storage).
  wire [ 1:0] channel_div_trans;
  wire [ 3:0] iact_channels_per_pe;             // Channels assigned to one PE (from dma_storage).
  wire [ 3:0] iact_channels_per_pe_next_layer;  // Channel count for the next layer (from dma_storage); used when routing psums back as iact.
  reg  [ 4:0] iact_channels_counter;             // Counts which channel batch [0..iact_channel_max_cycles-1] is currently being processed.
  wire [ 7:0] iact_channel_max_cycles;          // Total number of channel batches per layer pass (from dma_storage).
  wire [10:0] iact_needed_cycles;               // Number of iact streaming cycles for one spatial position (from dma_storage).
  wire [15:0] psum_x_all_cluster;               // Psum X with multiple channels counting seperate
  wire [ 3:0] iact_x_per_cluster;
  wire [11:0] iact_x_add_up;

  reg [$clog2(CLUSTER_ROWS+1)-1:0] iact_router_counter; // Counts how many cluster-row sweeps have been performed within the current iact delivery; wraps at needed_y_cls_reg.

  // -----------------------------------------------------------------------
  // Iact Double-Buffer Cell Arrays
  // These per-cell register arrays drive the 32 RAM_SP instances in BUFFER_A.
  // The gen_RAM_wires generate block connects these regs to the RAM ports.
  // -----------------------------------------------------------------------
  reg [BRANCHES_CLOG == 0 ? 0 : BRANCHES_CLOG-1:0] choose_iact_buffer;                                       // Selects the active half of the double-buffer (address MSB); toggled between layers via choose_iact_buffer_input/output.
  wire [BRANCHES_CLOG == 0 ? 0 : BRANCHES_CLOG-1:0] choose_iact_buffer_input;                                // Value choose_iact_buffer should take when the host is loading new activations (from dma_storage).
  wire [BRANCHES_CLOG == 0 ? 0 : BRANCHES_CLOG-1:0] choose_iact_buffer_output;                               // Value choose_iact_buffer should take when psums are being written back as activations (from dma_storage).
  wire fully_connected_layer;                                   // When 1: layer is a fully-connected (FC) layer; modifies iact packing and weight/psum addressing (from dma_storage).
  wire max_pooling;                                             // When 1: skip compute; instead run a 2×2 max-pool on the iact buffer (from dma_storage).
  reg [           BUFFER_WIDTH-1:0] iact_buffer_addr_reg      [IACT_RAM_CELLS-1:0]; // Per-cell read/write address; advanced by the sliding-window logic in CONVERT_IACT.
  reg [           BUFFER_WIDTH-1:0] buffer_addr_temp_reg; // Saved address snapshot used to restart a cell's address in MAXPOOLING_SEND.

  // -----------------------------------------------------------------------
  // Iact Stream Constructor Control
  // One iact_stream_constructor per cluster position reads pixels from the
  // shared 32-cell buffer and produces the packed, sparse-encoded iact
  // streams for the corresponding PE cluster in OpenEye_Parallel.
  // -----------------------------------------------------------------------
  reg [37:0] iact_converter_params_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0]; // Packed config word per converter: [35:32]=row_offset, [31:24]=x, [23:16]=y, [15:8]=size_x, [7:0]=channel_start.
  reg iact_converter_en_cfg_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];        // Pulse: latch iact_converter_params_reg into the converter's internal registers.
  reg iact_converter_en_store_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];      // Pulse: run one store step inside the converter (reads from shared buffer, writes to internal FIFO).
  reg iact_converter_en_enc_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];        // Pulse: run one encode step (outputs one word to the iact GLB interface).
  wire [7:0] x_lines_reg;                                                       // Number of x-lines per iact pass (from dma_storage); forwarded to iact_stream_constructor.
  reg send_data_reg;                                                             // One-cycle pulse that triggers the data-flow process to begin streaming to OpenEye_Parallel.
  wire store_in_psum;                                                            // When 1: keep psums in psum_buffer for further accumulation instead of sending them out (from dma_storage).
  wire iact_converter_ready_w[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];          // Per-converter ready signal; high when the converter has finished its current batch.

  reg converters_ready; // Combinatorial flag set in START_CONVERTER: AND of all iact_converter_ready_w signals. Transitions to CONVERT_IACT when high.

  // Legacy pipeline-stage registers for the old converter path (superseded by iact_stream_constructor; retained for potential future use)
  reg buffer_r_en_reg[IACT_RAM_CELLS-1:0];
  reg [2:0] iact_converter_n_1_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [3:0] iact_converter_mem_off_1_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [2:0] iact_converter_n_2_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [3:0] iact_converter_mem_off_2_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [2:0] iact_converter_n_3_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [3:0] iact_converter_mem_off_3_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [10-1:0] iact_to_psum_x_pos_counter;
  reg [4-1:0] iact_to_psum_trans_counter;
  reg [$clog2(IACT_RAM_CELLS*(8/TRANS_WORDS))-1:0] iact_to_psum_storage_counter;
  reg [DATA_IACT_BITWIDTH*TRANS_WORDS-1:0] iact_to_psum_shift_reg;
  reg [DATA_IACT_BITWIDTH*TRANS_WORDS-1:0] iact_to_psum_mux_reg;
  reg iact_to_psum_start_shifting;

  reg [CLUSTERS*NUM_GLB_IACT*TRANS_BITWIDTH_IACT - 1:0] iact_out_reg; // Legacy assembled iact output register (not driven in current path; kept for compatibility).
  reg iact_ready;                     // Legacy iact-ready flag (not used in current path).
  reg iact_converter_enc_enable;      // Global gate: when high, enables the enc-enable pulse distribution to all converters.
  reg iact_converter_params_enable;   // Global gate: when high, enables the params-enable pulse distribution to all converters.

  // -----------------------------------------------------------------------
  // Iact Converter Timing Counters
  // -----------------------------------------------------------------------
  wire [7:0] iact_converter_max_cycles;             // Total y-lines to process including kernel overlap = iact_size_y + kernel_size - 1 (set in GET_WGHT).
  wire [11:0] iact_buffer_words_per_write;          // Words written per cycle into iact buffer per writing cycle
  wire [7:0] iact_words_per_compute;
  wire [7:0] iact_converter_buffer_addr_max_cycles; // Maximum value of iact_converter_buffer_addr_cycles (from dma_storage).
  reg [7:0] iact_converter_cycles;                  // Current y-line counter [0..iact_converter_max_cycles-1]; drives the sliding window advancement.
  reg [7:0] iact_converter_buffer_addr_cycles;      // Sub-counter [0..iact_converter_buffer_addr_max_cycles-1]; controls how often the buffer address advances.


  // -----------------------------------------------------------------------
  // OpenEye_Parallel Configuration and Interface Signals
  // -----------------------------------------------------------------------
  reg status_reg_enable_reg; // When 1: enables OpenEye_Parallel to accept configuration data from the DMA stream (active during GET_PARAMETERS).

  reg compute_reg;           // One-cycle pulse sent to OpenEye_Parallel.compute_i to trigger the start of a computation batch.
  reg  [CLUSTERS * PES -1:0] compute_mask_reg; // Bitmask of active PE instances (one bit per PE × cluster); loaded from DMA in GET_PARAMETERS words 4+.
  wire [CLUSTERS * PES -1:0] compute_mask_reg_port;
  assign compute_mask_reg_port = compute_mask_reg[CLUSTERS * PES -1:0]; // Wire alias for compute_mask_reg; passed to OpenEye_Parallel.compute_mask_i.

  // Router mode vectors: loaded from DMA in GET_ROUTER_CONFIG and updated dynamically
  // during WAIT_FOR_RESULTS when data flows through multiple cluster rows.
  // Encoding: ROUTER_MODES_IACT=6 bits per iact GLB, ROUTER_MODES_WGHT=1 bit per wght GLB, ROUTER_MODES_PSUM=3 bits per psum GLB.
  // These signals are now driven by the router_configurator instance (see below).
  wire [ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0] router_mode_iact; // Iact router configuration; bit layout: [cc*CLUSTER_ROWS*NUM_GLB_IACT*6 + cr*NUM_GLB_IACT*6 + g*6 +: 6].
  wire [ROUTER_MODES_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] router_mode_wght; // Weight router configuration (1 bit per GLB: 0=pass, 1=source).
  wire [ROUTER_MODES_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] router_mode_psum; // Psum router configuration; bit [2] selects storage-accumulate mode per GLB.
  wire [ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0] router_mode_iact_storage; // Saved iact router vector from router_configurator.
  wire [7:0] storage_cycles_router;      // Psum router shift cycle counter from router_configurator.
  wire first_cycle;                      // First compute iteration flag from router_configurator.
  wire [CLUSTERS*NUM_GLB_PSUM-1:0] psum_choose_i_reg;         // Source-select bus for psum routing (from router_configurator).
  wire [7:0] iact_channels_counter_psum_router; // Local channel counter copy from router_configurator.

  // Weight data path: wght_buffer read data flows directly to OpenEye_Parallel.
  wire [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_data_i_w;
  assign wght_data_i_w = wght_buffer_data_r; // Direct wire: weight buffer output → accelerator weight input.
  reg [CLUSTERS*NUM_GLB_WGHT-1:0] wght_enable_i_reg;   // Per-GLB weight valid signals; built each cycle in the data-flow process to select which cluster rows receive data.
  wire [CLUSTERS*NUM_GLB_WGHT-1:0] wght_ready_o_reg;   // Back-pressure from OpenEye_Parallel weight inputs; all bits high when the PE array is ready for the next weight word.

  // Legacy iact wires (superseded by iact_*_oep_w wires driven from iact_stream_constructor)
  wire [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_data_i_wire;
  wire [CLUSTERS*NUM_GLB_IACT-1:0] iact_enable_i_wire;
  wire [CLUSTERS*NUM_GLB_IACT-1:0] iact_ready_o_wire;
  wire [PES*CLUSTERS*$clog2(NUM_GLB_IACT+1)-1:0] iact_choose_i;

  // Psum input to OpenEye_Parallel: bias values and previously accumulated psums are fed in from psum_buffer_.
  reg [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_i_reg;  // Registered copy of psum_buffer_data_r; driven into OpenEye_Parallel in CALCULATE_PSUM.
  reg [CLUSTERS*NUM_GLB_PSUM-1:0] psum_enable_i_reg;                    // Per-GLB enable signals for the psum input; set in CALCULATE_PSUM and PSUM_GET_RESULTS.
  wire [CLUSTERS*NUM_GLB_PSUM-1:0] psum_ready_o_reg;                    // Back-pressure: PE psum input is ready to accept data.

  // Psum output from OpenEye_Parallel: accumulated results flow back to psum_buffer_.
  wire [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_o_w;   // Result psum bus from OpenEye_Parallel; written to psum_buffer in PSUM_GET_RESULTS.
  wire [CLUSTERS*NUM_GLB_PSUM-1:0] psum_enable_o;                       // Valid signal for psum_data_o_w; used to gate psum_buffer writes.
  reg [CLUSTERS*NUM_GLB_PSUM-1:0] psum_ready_i_reg;                     // Ready signal sent back to OpenEye_Parallel; asserted all-ones in WAIT_TO_SEND_READY_SIGNAL.
  wire [5-1:0] kernels_per_calc; // Filters computed per calculation batch (from dma_storage).
  wire [4-1:0] y_lines_per_calc; // Output rows calculated per batch (from dma_storage).

  wire [ 7:0] needed_wght_cycles;                                           // Weight cycling period (from dma_storage); how many iact batches share the same weights.
  wire [13:0] fc_size_reg;                                                  // FC-layer input size (from dma_storage).
  wire [BUFFER_WIDTH_IACT_STREAM_CONSTRUCTOR-1:0] needed_iact_buffer_words; // Words required in the iact stream constructor's internal buffer (from dma_storage).

  //Output Registers for counting and delaying output (AXI Stream)

  reg [DMA_BITWIDTH-1 : 0] data_dma_o_q1;
  reg [DMA_BITWIDTH-1 : 0] data_dma_o_q2;
  reg [             2 : 0] data_dma_o_counter;
  //#######################
  // Main FSM State Encoding
  // The main FSM (fsm_current_state) drives the overall layer-execution
  // sequence.  Transitions follow the order shown in the diagram below;
  // several phases are skippable via skip* flags from dma_storage.
  //
  //   IDLE
  //    └─(enable_dma_i)──► GET_PARAMETERS
  //                          └─► GET_ROUTER_CONFIG
  //                               ├─(skipIact)──────────► GET_WGHT
  //                               └─► GET_IACT ──────────► GET_WGHT
  //                                                          ├─(skipPsum)──► GET_QUANTIZE
  //                                                          └─► GET_BIAS ──► GET_QUANTIZE
  //                                                                             └─► GET_OFFSET
  //                                                                                  ├─(max_pooling)─► MAXPOOLING_READ ─► MAXPOOLING_SEND ─► GET_PARAMETERS
  //                                                                                  └─► START_CONVERTER ─► CONVERT_IACT ─► WAIT_CYCLE
  //                                                                                                                             ├─(send_data_out)─► WAIT_FOR_RESULTS ─► GET_PARAMETERS
  //                                                                                                                             └─► RECEIVE_PSUMS_TO_IACT ─(PSUM_IDLE)─► GET_PARAMETERS
  //#######################

  localparam IDLE                 = 4'd0;  // Wait for first enable_dma_i pulse from host.
  localparam GET_PARAMETERS       = 4'd1;  // Receive 4+ DMA words: first 4 go to dma_storage (layer config); remainder fill compute_mask_reg.
  localparam GET_ROUTER_CONFIG    = 4'd2;  // Receive FSM_CEIL_IACT/WGHT/PSUM_RTR_CCLS DMA words; unpack into router_mode_iact/wght/psum.
  localparam GET_IACT             = 4'd3;  // Receive raw iact pixel words from host into the 32-cell double-buffer RAM array.
  localparam GET_WGHT             = 4'd4;  // Receive weight words from host into wght_buffer_; also compute iact_converter_max_cycles.
  localparam GET_BIAS             = 4'd5;  // Receive bias (initial psum) values from host into psum_buffer_.
  localparam GET_QUANTIZE         = 4'd6;  // Receive 16 DMA words of per-filter quantization (mantissa + exponent) into quant_mant / quant_exp.
  localparam START_CONVERTER      = 4'd7;  // Poll converters_ready (AND of all iact_converter_ready_w); transition to CONVERT_IACT when all are idle.
  localparam CONVERT_IACT         = 4'd8;  // Drive iact_stream_constructors through all y-line / channel cycles; advance the buffer address sliding window.
  localparam WAIT_CYCLE           = 4'd9; // Allow the iact converter pipeline to drain (16 extra cycles); pulse send_data_reg to start the weight-send process.
  localparam WAIT_FOR_RESULTS     = 4'd10; // Wait while OpenEye_Parallel computes; count iterations via current_cycle; transition to GET_PARAMETERS on last_data_o.
  localparam RECEIVE_PSUMS_TO_IACT= 4'd11; // Receive quantized psums from the PSUM FSM and write them into the iact double-buffer as next-layer activations.
  localparam MAXPOOLING_READ      = 4'd12; // Read 2×2 pixel groups from the iact buffer; accumulate running max in pooling_regs via a 3-stage comparator tree.
  localparam MAXPOOLING_SEND      = 4'd13; // Write max-pooled results from pooling_regs back into the iact buffer cells for subsequent processing.
  localparam MAXPOOLING_WAIT      = 4'd14;

  // PSUM FSM State Encoding
  // The PSUM FSM (fsm_psum_current_state) runs concurrently with the main FSM
  // and manages the partial-sum pipeline from compute-trigger to final output.
  localparam PSUM_IDLE                 = 0; // Idle; absorbs bias writes (GET_BIAS) and waits for compute_reg.
  localparam WAIT_TO_SEND_READY_SIGNAL = 1; // Waits until wght/iact enables go to 0, then asserts psum_ready_i_reg after 16 cycles.
  localparam CALCULATE_PSUM            = 2; // Feeds bias data from psum_buffer into OpenEye_Parallel; waits for psum_ready_o to confirm receipt.
  localparam PSUM_GET_RESULTS          = 3; // Captures psum_data_o_w from OpenEye_Parallel into psum_buffer_; counts until filters results collected.
  localparam WAIT_FOR_SENDING_RESULTS  = 4; // Decides next action: PSUM_SEND_RESULTS (send_data_out=1) or SEND_PSUM_TO_IACT / PSUM_IDLE.
  localparam PSUM_SEND_RESULTS         = 5; // Streams psum buffer contents to data_dma_o; iterates over all clusters, GLBs, and filter positions.
  localparam SEND_PSUM_TO_IACT         = 6; // Quantizes psum values and writes results into quantized_value_reg for the main FSM to pack into the iact buffer.

  reg [3:0] fsm_current_state;  // Current state of the main FSM.
  reg [3:0] fsm_last_state;     // Previous main FSM state; useful for tracing transitions in simulation.
  //assign debug_fsm_current_state = fsm_current_state; // Expose current state on debug port.

  //#######################
  // Process Variable Declarations
  //#######################

  // --- Iteration / cycle-tracking variables (Process 2: Cycle Counting) ---
  reg [19:0] current_cycle;    // Counts iact-delivery iterations completed in WAIT_FOR_RESULTS / RECEIVE_PSUMS_TO_IACT; compared against needed_cycles.
  reg [15:0] iact_cycle_count; // Tracks which weight-reuse iteration we are on [0..needed_wght_cycles-1]; advances after each full cluster-row sweep.
  reg        single_iteration; // Level flag: high from the first clock of an iact delivery until iact_ready_o_oep_w goes all-ones again (delivery complete).
  reg        single_iteration2;// Delayed version of single_iteration (one cycle); used to detect the falling edge of single_iteration.
  reg        single_iteration3;// Single-cycle pulse on the rising edge of single_iteration; triggers current_cycle increment and router updates.

  // --- Data-flow send-phase variables (Process 5: Data-Flow to OpenEye_Parallel) ---
  reg        sending_data;     // Level flag: high throughout the entire weight+iact send phase.
  reg        wght_sendable;    // When 1: the weight buffer is allowed to start streaming (gated by PE weight-ready back-pressure).
  reg [12:0] fsm_sending_cycle;// Cycle counter within the send phase; drives weight buffer read pointer and wght_enable_i_reg mask.
  reg [CLUSTERS*NUM_GLB_WGHT-1:0] flat_help_var_send; // Temporary variable for building the wght_enable_i_reg bitmask (blocking assignment).
  reg [CLUSTERS*NUM_GLB_WGHT-1:0] temp_var;           // Scratch variable used when computing the weight-enable bitmask for partial cluster rows.
  reg [63:0] prepared_iact [31:0]; // Pre-assembled iact words (one per buffer cell); populated in Process 5 for later streaming (currently unused in main path).
  localparam EXTENDEDBITS = 48 - NUM_GLB_WGHT; // Zero-padding width when building flat_help_var_send from a per-row weight-enable mask.

  wire [CLUSTERS*NUM_GLB_IACT-1:0] iact_ready_o_oep_w; // Back-pressure bus from OpenEye_Parallel iact inputs; all-ones when every iact GLB is ready for more data.
  
  // -----------------------------------------------------------------------
  // Process 2: Cycle Counting
  // Tracks how many iact-delivery iterations have been completed during the
  // WAIT_FOR_RESULTS and RECEIVE_PSUMS_TO_IACT states.
  //
  // Key signals updated here:
  //   current_cycle     - total iact delivery iterations done; compared to needed_cycles.
  //   iact_cycle_count  - sub-counter tracking the weight-reuse period.
  //   iact_router_counter - counts cluster-row sweeps within one iact batch.
  //   single_iteration  - level flag: high while iact delivery is in progress.
  //   single_iteration3 - single-cycle pulse on delivery start; triggers increments.
  //
  // Delivery detection: a new delivery starts when iact_ready_o_oep_w != all-ones
  // (i.e. at least one PE is still consuming data) and single_iteration is not yet set.
  // -----------------------------------------------------------------------
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin
      current_cycle       <= 0;
      iact_cycle_count    <= 0;
      iact_router_counter <= 0;
      single_iteration    <= 0;
      single_iteration2   <= 0;
      single_iteration3   <= 0;
    end else begin
      single_iteration3 <= 0;
      if (current_cycle <= needed_cycles - 1) begin
        if (iact_ready_o_oep_w != {CLUSTERS*NUM_GLB_IACT{1'b1}}) begin
          if (!single_iteration) begin
            single_iteration  <= 1;
            single_iteration3 <= 1;
            if (iact_channels_counter == iact_channel_max_cycles -1) begin
              if (iact_router_counter == needed_y_cls_reg - 1) begin
                iact_cycle_count <= iact_cycle_count + 1;
                if (iact_cycle_count == {{8 {1'd0}},needed_wght_cycles} - 1) begin
                  iact_cycle_count <= 0;
                end
              end
            end
          end
        end else begin
          single_iteration <= 0;
        end
      end
      if (fsm_current_state == WAIT_FOR_RESULTS | fsm_current_state == RECEIVE_PSUMS_TO_IACT) begin
        if (single_iteration3) begin
          current_cycle <= current_cycle + 1;
          if (iact_channels_counter == iact_channel_max_cycles -1) begin
            iact_router_counter <= iact_router_counter + 1;
            if (iact_router_counter == needed_y_cls_reg - 1) begin
              iact_router_counter  <= 0;
            end
          end
        end
      end
      if ((RECEIVE_PSUMS_TO_IACT == fsm_current_state) | (fsm_current_state == WAIT_FOR_RESULTS)) begin
        if (single_iteration) begin
          single_iteration2 <= 1;
        end
        if (single_iteration == 0) begin
          single_iteration2 <= 0;
        end
      end
      if ((fsm_current_state == GET_PARAMETERS) |
        reset_cycle) begin
        current_cycle       <= 0;
        iact_cycle_count    <= 0;
        iact_router_counter <= 0;
        single_iteration    <= 0;
        single_iteration3   <= 0;
      end
    end
  end
  // -----------------------------------------------------------------------
  // Process 3: Iact Converter Parameter Distribution
  // Computes and broadcasts the per-converter spatial parameters
  // (x-offset, y-offset, channel, strip width) to every iact_stream_constructor
  // instance before each iact encoding run.
  //
  // Registers local to this process:
  //   fsm_iact_params       - countdown: number of param-write cycles remaining in
  //                           the current parameter push (initialised to CLUSTER_ROWS
  //                           on GET_ROUTER_CONFIG; decremented each cycle).
  //   fsm_iact_params_y_line - which y-line (output row) within kernels_per_calc we are
  //                           currently configuring.
  //   fsm_iact_params_kernel - kernel index within the current y-line batch.
  //   iact_converter_x      - current x-pixel start coordinate assigned to column 0;
  //                           each column adds a*NUM_GLB_PSUM.
  //   iact_converter_y      - current y-pixel coordinate (input row) being loaded.
  //   iact_converter_c      - current input channel being loaded.
  //   fsm_row               - which CLUSTER_ROW slot currently receives params;
  //                           advances by needed_y_cls_reg each step.
  //   fsm_row_offset        - base offset for row interleaving; cycles 0..needed_y_cls_reg-1.
  //   a, b, word, line      - loop variables (integer).
  //
  // Key behaviour:
  //   - During GET_ROUTER_CONFIG: loads start_param_array into param_array_reg and
  //     resets fsm_iact_params to CLUSTER_ROWS.
  //   - During GET_WGHT / GET_IACT: iterates over rows and columns, writing
  //     iact_converter_params_reg[col][row][35:0] with:
  //       [35:32] fsm_row_offset  (strip offset for multi-row interleaving)
  //       [31:24] iact_converter_x + col*NUM_GLB_PSUM  (x start)
  //       [23:16] iact_converter_y  (y start, incremented at strip boundaries)
  //       [15: 8] iact_size_x       (full input width)
  //       [ 7: 0] iact_converter_c  (channel index)
  //     and asserts iact_converter_en_cfg_reg[col][row] for one cycle.
  //   - During CONVERT_IACT (iact_converter_params_enable high): rotates
  //     param_array_reg by kernels_per_calc*(iact_size_x…) each cycle to
  //     select which converters get updated next.
  //   - reset_cycle clears all state.
  //   - FC (fully_connected_layer) mode skips x-boundary checks.
  // -----------------------------------------------------------------------
  reg [7:0] fsm_iact_params;       // Countdown: param writes remaining this push.
  reg [7:0] fsm_iact_params_y_line;// Y-line index within current kernel/y-line batch.
  reg [7:0] fsm_iact_params_kernel;// Kernel index within current y-line.
  reg [7:0] iact_converter_x;      // X-pixel origin for column 0; col k gets +k*NUM_GLB_PSUM.
  reg [7:0] iact_converter_y;      // Y-pixel (row) coordinate currently being configured.
  reg [7:0] iact_converter_c;      // Input channel index currently being configured.
  // additional register for fsm
  reg [$clog2(CLUSTER_ROWS+1)-1:0] fsm_row;        // Current CLUSTER_ROW target; steps by needed_y_cls_reg.
  reg [$clog2(CLUSTER_ROWS+1)-1:0] fsm_row_offset; // Interleave offset; cycles 0..needed_y_cls_reg-1.
  integer a, b, word, line;
  always @(posedge clk_i, negedge rst_n) begin
  if (!rst_n) begin
    param_array_reg               <= 0;
    fsm_iact_params               <= 0;
    fsm_iact_params_y_line        <= 0;
    fsm_iact_params_kernel        <= 0;
    iact_converter_x              <= 0;
    iact_converter_y              <= 0;
    iact_converter_c              <= 0;
    fsm_row                       <= 0;
    fsm_row_offset                <= 0;
    for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
      for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
        iact_converter_params_reg[a][b] <= 0;
        iact_converter_en_cfg_reg[a][b] <= 0;
      end
    end
  end else begin
    for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
      for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
        iact_converter_en_cfg_reg[a][b] <= 0;
      end
    end

    if (GET_PARAMETERS == fsm_last_state) begin
      param_array_reg <= start_param_array;
      fsm_iact_params <= CLUSTER_ROWS;
    end else begin
      if (!((GET_WGHT == fsm_current_state) | (GET_IACT == fsm_current_state))) begin
        if (iact_converter_params_enable) begin
          for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
            for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
              if (param_array_reg[(a+(b*CLUSTER_COLUMNS))] == 1) begin
                if (iact_converter_cycles <= ((iact_converter_max_cycles - 1))) begin
                  iact_converter_en_cfg_reg[a][b] <= 1;
                end
              end
            end
          end
          param_array_reg <= ((kernels_per_calc * iact_size_x / NUM_GLB_PSUM) | param_array_reg>>(CLUSTERS-(kernels_per_calc * iact_size_x / NUM_GLB_PSUM)));
          param_array_reg <= ((param_array_reg << 2) | param_array_reg>>(CLUSTERS-2));
          
          if (fsm_current_state == CONVERT_IACT) begin
            fsm_iact_params <= fsm_iact_params + kernels_per_calc * ((iact_size_x - 1 + (2 * NUM_GLB_PSUM)) / (2 * NUM_GLB_PSUM));
          end
        end else begin
          param_array_reg <= conv_array_reg;
        end
      end
      if ( ((GET_WGHT == fsm_current_state) | (GET_IACT == fsm_current_state)) ? 
           (fsm_iact_params > 0) : 
           (iact_converter_params_enable | (fsm_iact_params > 0)) ) begin
        
        if (fsm_iact_params > 0) begin
          fsm_iact_params <= fsm_iact_params - 1;
        end

        for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin 
          iact_converter_params_reg[a][fsm_row][37:32] <= fsm_row_offset;
          iact_converter_params_reg[a][fsm_row][15:8]  <= iact_size_x;
          iact_converter_params_reg[a][fsm_row][7:0]   <= iact_converter_c;

          if ((GET_WGHT == fsm_current_state) | (GET_IACT == fsm_current_state)) begin
            iact_converter_en_cfg_reg[a][fsm_row] <= 1;
          end

          if (fully_connected_layer) begin
            iact_converter_params_reg[a][fsm_row][31:24] <= iact_converter_x;
            iact_converter_params_reg[a][fsm_row][23:16] <= iact_converter_y;
          end else begin
            if ((a != 0) & ((iact_converter_x + a[7:0] * iact_x_per_cluster) >= iact_size_x)) begin
              iact_converter_params_reg[a][fsm_row][31:24] <= 0;
            end else begin
              iact_converter_params_reg[a][fsm_row][31:24] <= iact_converter_x + a[7:0] * iact_x_per_cluster;
            end
            
            if ((a != 0) & ((iact_converter_x + a[7:0] * iact_x_per_cluster) >= iact_size_x) & (fsm_iact_params_kernel == kernels_per_calc - 1)) begin
              iact_converter_params_reg[a][fsm_row][23:16] <= iact_converter_y + 1;
            end else begin
              iact_converter_params_reg[a][fsm_row][23:16] <= iact_converter_y;
            end
          end
        end
        fsm_row <= fsm_row + needed_y_cls_reg;
        if (fsm_row + needed_y_cls_reg >= CLUSTER_ROWS) begin
          fsm_row        <= fsm_row_offset + 1;
          fsm_row_offset <= fsm_row_offset + NUM_GLB_IACT;
          if (fsm_row_offset == NUM_GLB_IACT * (needed_y_cls_reg - 1)) begin
            fsm_row        <= 0;
            fsm_row_offset <= 0;
          end
        end
        iact_converter_x <= iact_converter_x + (CLUSTER_COLUMNS * iact_x_per_cluster);
        if (((((iact_converter_x + (CLUSTER_COLUMNS * iact_x_per_cluster)) * iact_x_line_repetitions) >= iact_size_x * needed_y_cls_reg) | (fully_connected_layer))) begin
          iact_converter_x <= 0;
          if (((iact_converter_x + iact_x_per_cluster) >= iact_size_x) & (!fully_connected_layer) & (kernels_per_calc != 1) & (iact_x_per_cluster < iact_size_x)) begin
            iact_converter_x <= iact_x_per_cluster;
          end
          fsm_iact_params_kernel <= fsm_iact_params_kernel + 1;
          if (fsm_iact_params_kernel == kernels_per_calc - 1) begin
            fsm_iact_params_kernel <= 0;
            fsm_iact_params_y_line <= fsm_iact_params_y_line + 1;
            if (fsm_iact_params_y_line == y_lines_per_calc - 1) begin
              fsm_iact_params_y_line <= 0;
              fsm_row_offset         <= 0;
              if ((GET_WGHT == fsm_current_state) | (GET_IACT == fsm_current_state)) begin
                iact_converter_c <= iact_converter_c + 1;
              end else begin
                fsm_iact_params <= 0;
              end
              if (!fully_connected_layer) begin
                if ((GET_WGHT == fsm_current_state) | (GET_IACT == fsm_current_state)) begin
                  fsm_iact_params <= 0;
                end
                fsm_row <= 0;
                if ((GET_WGHT == fsm_current_state) | (GET_IACT == fsm_current_state)) begin
                   iact_converter_c <= iact_converter_c + iact_channels_per_pe;
                end
              end
            end
            
            if (!((GET_WGHT == fsm_current_state) | (GET_IACT == fsm_current_state))) begin
               iact_converter_c <= iact_converter_c + iact_channels_per_pe;
               if (fully_connected_layer) begin
                iact_converter_c <= iact_converter_c + 1;
               end
            end

            // Channel Wrap-around
            if (iact_converter_c + iact_channels_per_pe == iact_size_c) begin
              iact_converter_c <= 0;
              iact_converter_y <= iact_converter_y + 1;
              if (iact_converter_y + 1 >= iact_size_y) begin
                iact_converter_y <= 0;
              end
            end
          end
        end
      end
    end
    
    // 4. Reset Cycle auswerten
    if (reset_cycle) begin
      param_array_reg  <= 0;
      fsm_iact_params  <= 0;
      iact_converter_x <= 0;
      iact_converter_y <= 0;
      iact_converter_c <= 0;
      fsm_row          <= 0;
      fsm_row_offset   <= 0;
      for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
        for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
          iact_converter_params_reg[a][b] <= 0;
          iact_converter_en_cfg_reg[a][b] <= 0;
        end
      end
    end
    
  end
end

  // -----------------------------------------------------------------------
  // Process 4: Iact Converter Store Enable
  // Controls when each iact_stream_constructor is allowed to write an
  // encoded iact word into the shared buffer RAM cells.
  //
  // Registers local to this process:
  //   conv_array_reg             - rotating bitmask (CLUSTERS bits) that tracks
  //                                which converters currently hold active (non-padding)
  //                                strip data and should fire en_store.
  //   iact_converter_en_store_reg[col][row] - one-cycle pulse telling converter [col][row]
  //                                to commit its current encoded output to the buffer.
  //
  // Key behaviour:
  //   - GET_PARAMETERS: clears conv_array_reg and all en_store signals.
  //   - GET_ROUTER_CONFIG: initialises conv_array_reg to start_param_array
  //     (FC mode forces it to 3 = both column clusters active).
  //   - Each cycle: all en_store_reg outputs default to 0; then, if
  //     iact_converter_enc_enable is high, any converter whose bit in
  //     conv_array_reg is set AND whose iact_converter_cycles counter hasn't
  //     exceeded iact_converter_max_cycles gets en_store asserted for one cycle.
  //   - FC mode: rotates conv_array_reg left by 2 each encoding step so the
  //     active-converter window walks across all cluster pairs.
  //   - reset_cycle: zeros everything.
  // -----------------------------------------------------------------------
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin
      conv_array_reg <= 0;
      for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
        for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
          iact_converter_en_store_reg[a][b] <= 0;
        end
      end
    end else begin
      if (fsm_current_state == GET_PARAMETERS) begin
        conv_array_reg <= 0;
        for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
          for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
            iact_converter_en_store_reg[a][b] <= 0;
          end
        end
      end
      if (fsm_last_state == GET_PARAMETERS) begin
        conv_array_reg <= start_param_array;
        if (fully_connected_layer) begin
          conv_array_reg <= 3;
        end
      end
      for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
        for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
          iact_converter_en_store_reg[a][b] <= 0;
        end
      end
      if (iact_converter_enc_enable) begin
        for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
          for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
            if (conv_array_reg[(a+(b*CLUSTER_COLUMNS))] == 1) begin
              if (iact_converter_cycles <= ((iact_converter_max_cycles - 1))) begin
                iact_converter_en_store_reg[a][b] <= 1;
              end
            end
          end
        end
        conv_array_reg <= (conv_array_reg<<cluster_per_conv_cycle | conv_array_reg>>(CLUSTERS-cluster_per_conv_cycle));
        if (fully_connected_layer) begin
          conv_array_reg <= (conv_array_reg<<2 | conv_array_reg>>(CLUSTERS-2));
        end
      end
      if (reset_cycle) begin
        conv_array_reg <= 0;
        for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
          for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
            iact_converter_en_store_reg[a][b] <= 0;
          end
        end
      end
    end
  end
  
  // -----------------------------------------------------------------------
  // Process 5: Data-Flow to OpenEye_Parallel (Weight + Iact Send Phase)
  // Orchestrates the entire forward-pass data delivery: starts the
  // iact encoders, then streams weights from wght_buffer_, and finally
  // triggers computation inside OpenEye_Parallel.
  //
  // Key signals driven here:
  //   sending_data               - level flag; high from the first cycle of a send
  //                                phase until current_cycle reaches needed_cycles.
  //   fsm_sending_cycle          - cycle counter within the send phase [0..wghts_per_pe+2+];
  //                                drives wght_buffer_rd_addr and wght_enable_i_reg.
  //   iact_converter_en_enc_reg  - asserted for all converters on cycle 0 of the send
  //                                phase (start encoding) and on each subsequent
  //                                iact delivery event.
  //   wght_sendable              - becomes 1 when a new iact delivery begins; cleared
  //                                once wght_ready_o_reg is all-ones (PEs ready for weights).
  //   wght_buffer_en_r        - read enable for the weight staging buffer; active
  //                                while fsm_sending_cycle <= wghts_per_pe.
  //   wght_buffer_rd_addr     - advances by 1 each sending cycle; reset to storage
  //                                pointer at weight-reuse boundaries; fully reset when
  //                                iact_cycle_count wraps.
  //   wght_enable_i_reg          - per-GLB weight-enable bitmask for OpenEye_Parallel;
  //                                built from the flat_help_var_send shift-OR procedure
  //                                that maps active CLUSTER_ROWS to the wght bus.
  //   compute_reg                - single-cycle pulse to start computation; asserted
  //                                after the last weight word is sent AND current_cycle==0.
  //
  // Weight-enable mask construction (fsm_sending_cycle > 2):
  //   For each row r in [0..CLUSTER_ROWS-1], if r*NUM_GLB_PSUM*CLUSTER_COLUMNS is
  //   within the strip range, the corresponding NUM_GLB_WGHT bits are set in
  //   flat_help_var_send; the mask is then mirrored to both CLUSTER_ROWS halves.
  //   FC mode: all bits set unconditionally.
  //
  // reset_cycle / GET_PARAMETERS: fully resets send-phase state.
  // -----------------------------------------------------------------------
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin
      //Reset Registers
      sending_data                <= 0;
      fsm_sending_cycle           <= 0;
      wght_enable_i_reg           <= 0;
      wght_buffer_en_r            <= 0;
      wght_buffer_rd_addr         <= 0;
      wght_buffer_rd_addr_storage <= 0;
      compute_reg                 <= 0;
      wght_sendable               <= 0;
      flat_help_var_send           = 0;
      for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
        prepared_iact[a] <= 0;
      end
      for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
        for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
          iact_converter_en_enc_reg[a][b] <= 0;
        end
      end
    end else begin
      //Set Registers to 0
      for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
        for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
          iact_converter_en_enc_reg[a][b] <= 0;
        end
      end
      compute_reg <= 0;
      if (send_data_reg | sending_data) begin
        sending_data <= 1;
        fsm_sending_cycle <= fsm_sending_cycle + PARALLEL_MACS;
        if (!sending_data) begin
          for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
            for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
              iact_converter_en_enc_reg[a][b] <= 1;
            end
          end
        end
        if (wght_sendable & (wght_ready_o_reg == {CLUSTERS*NUM_GLB_WGHT{1'b1}})) begin
          wght_sendable     <= 0;
          fsm_sending_cycle <= 1;
          wght_buffer_en_r  <= 1;
        end
        if (wght_buffer_en_r) begin
          if (fsm_sending_cycle <= wghts_per_pe) begin
            wght_buffer_rd_addr <= wght_buffer_rd_addr + 1;
          end
          if (fsm_sending_cycle > 1) begin
            flat_help_var_send = 0;
            for (a = 0; a < CLUSTER_ROWS; a=a+1) begin
              if ((a * (NUM_GLB_PSUM * CLUSTER_COLUMNS)) <= pagu_wght_limit - 1) begin
                temp_var = {{EXTENDEDBITS{1'b0}}, {NUM_GLB_WGHT{1'b1}}};
                flat_help_var_send = flat_help_var_send + (temp_var << (a * NUM_GLB_WGHT));
                temp_var = 0;
              end
            end
            flat_help_var_send = flat_help_var_send + (flat_help_var_send << (CLUSTER_ROWS * NUM_GLB_WGHT));
            wght_enable_i_reg <= flat_help_var_send[CLUSTERS*NUM_GLB_WGHT-1:0];
            flat_help_var_send = 0;
            if (fully_connected_layer) begin
              wght_enable_i_reg <= {{CLUSTERS{{NUM_GLB_WGHT{1'b1}}}}};
            end
          end
          if (fsm_sending_cycle > wghts_per_pe + 2) begin
            fsm_sending_cycle <= fsm_sending_cycle;
            wght_buffer_en_r  <= 0;
            wght_enable_i_reg <= 0;
            if (current_cycle == 0) begin
              compute_reg <= 1;
            end
          end
        end
        if (current_cycle < needed_cycles - 1) begin
          if (iact_ready_o_oep_w != {CLUSTERS*NUM_GLB_IACT{1'b1}}) begin
            if (!single_iteration) begin
              for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
                for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
                  iact_converter_en_enc_reg[a][b] <= 1;
                end
              end
              wght_sendable <= 1;
              if ((iact_channel_max_cycles == 1) & (needed_wght_cycles == 1)) begin
                wght_sendable <= 0;
              end
              if (iact_channels_counter == iact_channel_max_cycles -1) begin
                if (iact_channel_max_cycles != 1) begin
                  wght_buffer_rd_addr <= wght_buffer_rd_addr_storage;
                end
                if (iact_router_counter == needed_y_cls_reg - 1) begin
                  wght_buffer_rd_addr <= wght_buffer_rd_addr;
                  wght_buffer_rd_addr_storage <= wght_buffer_rd_addr;
                  if (iact_cycle_count == {{8 {1'd0}},needed_wght_cycles} - 1) begin
                    wght_buffer_rd_addr_storage <= 0;
                    wght_buffer_rd_addr         <= 0;
                  end
                end
              end
            end
          end
        end
        if (current_cycle == needed_cycles) begin
          fsm_sending_cycle <= 0;
        end
      end else begin
        //Set Registers to 0
        fsm_sending_cycle <= 0;
        wght_enable_i_reg <= 0;
        wght_buffer_en_r  <= 0;
      end
      if (fsm_current_state == GET_PARAMETERS) begin
        sending_data        <= 0;
        fsm_sending_cycle   <= 0;
        wght_enable_i_reg   <= 0;
        wght_buffer_en_r    <= 0;
        wght_buffer_rd_addr <= 0;
        compute_reg         <= 0;
        wght_sendable       <= 1;
        flat_help_var_send = 0;
        for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
          prepared_iact[a] <= 0;
        end
        for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
          for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
            iact_converter_en_enc_reg[a][b] <= 0;
          end
        end
      end
      if (reset_cycle) begin
        sending_data                   <= 0;
        fsm_sending_cycle              <= 0;
        wght_enable_i_reg              <= 0;
        wght_buffer_en_r               <= 0;
        wght_buffer_rd_addr            <= 0;
        wght_buffer_rd_addr_storage    <= 0;
        compute_reg                    <= 0;
        wght_sendable                  <= 1;
        flat_help_var_send              = 0;
        for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
          for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
            iact_converter_en_enc_reg[a][b] <= 0;
          end
        end
      end
    end
    temp_var = 0;
  end

  // --- DMA output staging registers (main FSM Process 6) ---
  reg       write_dma_en;             // High for one cycle when a result word is ready on data_dma_o.
  reg [2:0] write_dma_addr;           // Selects which of the 4 dma_storage target registers to write.
  reg [DMA_BITWIDTH-1:0] dma_data_i;  // Data word being sent back to the host via the DMA interface.

  // --- Iact buffer address counters (main FSM Process 6) ---
  reg [15:0] select_ram_counter;  // Counts which of the 32 RAM_SP cells is currently being addressed
  reg [15:0] select_ram_counter2;  // Counts which of the 32 RAM_SP cells is currently being addressed
                                  // during GET_IACT, CONVERT_IACT, RECEIVE_PSUMS_TO_IACT, and MAXPOOLING.
  reg [ 7:0] select_ram_offset;   // Additional offset into the active RAM cell address (within one cell).
  reg [$clog2(IACT_RAM_CELLS)-1:0] select_iact_glb_addr_inc;     // Modulo counter tracking position within a 64-bit RAM word
                                  // (used when packing multiple iact bytes into one word).

  // --- Max-pooling pipeline registers (MAXPOOLING_READ state) ---
  // A 3-stage pipelined comparator tree reduces a 2×2 input region (32 candidate bytes)
  // to a single maximum byte value stored back into the iact buffer.
  //   pooling_regs[0..31]  - running-maximum accumulators; one per output pixel;
  //                          initialised to -128 at reset and updated each read cycle.
  //   pooling_stage_1[0..7]- first pipeline compare level: 32→8 values.
  //   pooling_stage_2[0..3]- second level: 8→4 values.
  //   pooling_stage_3[0..1]- third level: 4→2 values.
  //   pooling_stage_4      - final output: 2→1 maximum value.
  reg signed  [                   7:0] pooling_regs          [31:0];
  reg                                  pooling_buffer_enable;
  reg                                  set_pointer_start;
  reg signed  [DATA_IACT_BITWIDTH-1:0] pooling_buffer_new    [CHANNELS_PER_WORD-1:0];
  wire signed [DATA_IACT_BITWIDTH-1:0] pooling_buffer_old    [CHANNELS_PER_WORD-1:0];
  reg signed  [DATA_IACT_BITWIDTH-1:0] pooling_buffer_old_q  [CHANNELS_PER_WORD-1:0];
  reg signed  [                   7:0] pooling_stage_1       [7:0];  // Pipeline stage 1: 32→8 max.
  reg signed  [                   7:0] pooling_stage_2       [3:0];  // Pipeline stage 2: 8→4 max.
  reg signed  [                   7:0] pooling_stage_3       [1:0];  // Pipeline stage 3: 4→2 max.
  reg signed  [                   7:0] pooling_stage_4;         // Pipeline stage 4: 2→1 max (final result).
  reg [IACT_RAM_CELLS_WORD_BITWIDTH*IACT_RAM_CELLS-1:0] iact_buffer_data_pool_temp;             // Temp reg from iact_buffer_data_r.

  // --- Per-filter quantization parameters (loaded in GET_QUANTIZE / GET_OFFSET) ---
  // Quantization formula applied in SEND_PSUM_TO_IACT state:
  //   q[f] = (quant_mant[f] * (psum + quant_offset[f])) >>> quant_exp[f]
  // Result is clamped to signed 8-bit before writing back to the iact buffer.
  reg  [QUANT_TRANS_WIDTH-1:0] quant_reg;
  wire [     OFFSET_WIDTH-1:0] quant_offset [QUANT_AMOUNT-1:0]; // Per-filter zero-point offset (8-bit, added to raw psum).
  wire [   EXPONENT_WIDTH-1:0] quant_exp    [QUANT_AMOUNT-1:0]; // Per-filter right-shift exponent (7-bit; applied after multiply).
  wire [   MANTISSA_WIDTH-1:0] quant_mant   [QUANT_AMOUNT-1:0]; // Per-filter scale mantissa (25-bit; multiplied with shifted psum).
  genvar quant_count;
  for (quant_count = 0; quant_count < QUANT_AMOUNT; quant_count=quant_count+1) begin
    assign quant_offset[quant_count] = quant_reg[(quant_count*(OFFSET_WIDTH+EXPONENT_WIDTH+MANTISSA_WIDTH))+:OFFSET_WIDTH];
    assign quant_exp[quant_count]    = quant_reg[OFFSET_WIDTH+(quant_count*(OFFSET_WIDTH+EXPONENT_WIDTH+MANTISSA_WIDTH))+:OFFSET_WIDTH];
    assign quant_mant[quant_count]   = quant_reg[OFFSET_WIDTH+EXPONENT_WIDTH+(quant_count*(OFFSET_WIDTH+EXPONENT_WIDTH+MANTISSA_WIDTH))+:OFFSET_WIDTH];
  end
  //#######################
  // Wires: Iact Buffer Interface (iact_stream_constructor → RAM_SP cells)
  // These wires are the unpacked per-cell signals that the generate blocks below
  // route between the iact_stream_constructor instances and the RAM_SP cells.
  // Packing/unpacking is done in the UNPACKED_TRACES generate block.
  //#######################
  reg                                                     iact_buffer_en_r   [IACT_RAM_CELLS-1:0]; // Per-cell read enable from FSM or converter.
  reg                                                     iact_buffer_en_w   [IACT_RAM_CELLS-1:0]; // Per-cell write enable from FSM or converter.
  reg [                                 BUFFER_WIDTH-1:0] iact_buffer_addr   [IACT_RAM_CELLS-1:0]; // Per-cell address (half-width; MSB is buffer_select).
  reg [                        IACT_CELL_INPUT_WIDTH-1:0] iact_buffer_data_w;                      // Per-cell write data (64 bits).
  wire [ IACT_RAM_CELLS_WORD_BITWIDTH*IACT_RAM_CELLS-1:0] iact_buffer_data_r;                     // Active-half read data: unpacked from iact_buffer_data_r.

  // --- PSUM send-phase registers (PSUM FSM Process 8) ---
  // These registers coordinate the multi-cycle sequence of reading psum results
  // from psum_buffer_, optionally quantizing them, and streaming to the DMA output
  // (PSUM_SEND_RESULTS) or writing back to the iact buffer (SEND_PSUM_TO_IACT).
  reg [7:0] psum_cycle_buffer_0;         // Counts GLB positions within a single cluster during result streaming.
  reg [7:0] psum_cycle_buffer_1;         // Pipelined copy of psum_cycle_buffer_0 (1-cycle delay).
  reg [7:0] psum_cycle_buffer_2;         // Pipelined copy of psum_cycle_buffer_0 (2-cycle delay).
  reg [7:0] psum_cycle_buffer_3;         // Pipelined copy of psum_cycle_buffer_0 (3-cycle delay).
  reg [7:0] psum_cycle_buffer_4;         // Pipelined copy of psum_cycle_buffer_0 (4-cycle delay); aligns with RAM read latency.
  reg [3:0] sending_cluster_rows;        // Row-cluster index being read in PSUM_SEND_RESULTS.
  wire [3:0] iteration_for_kernels;      // Tracks which kernel group is currently being output (multi-kernel layers).
  reg [11:0] pcb_1;                      // Composite psum buffer address word, stage 1: {sending_clusters, psum_cycle_buffer_1}.
  reg [11:0] pcb_2;                      // Composite psum buffer address word, stage 2: {sending_clusters, psum_cycle_buffer_2}.
  reg [11:0] pcb_3;                      // Composite psum buffer address word, stage 3: {sending_clusters, psum_cycle_buffer_3}.
  reg [ 7:0] psum_cycle_inc_0;           // Counts GLB positions within a single cluster during result streaming.
  reg [ 7:0] psum_cycle_inc_1;           // Pipelined copy of psum_cycle_inc_0 (1-cycle delay).
  reg [ 7:0] psum_cycle_inc_2;           // Pipelined copy of psum_cycle_inc_0 (2-cycle delay).
  reg [ 7:0] psum_cycle_inc_3;           // Pipelined copy of psum_cycle_inc_0 (3-cycle delay).
  reg [ 7:0] psum_cycle_inc_4;           // Pipelined copy of psum_cycle_inc_0 (4-cycle delay); aligns with RAM read latency.
  reg [ 7:0] psum_cycle_limit_0;         // Counts GLB positions within a single cluster during result streaming.
  reg [ 7:0] psum_cycle_limit_1;         // Pipelined copy of psum_cycle_limit_0 (1-cycle delay).
  reg [ 7:0] psum_cycle_limit_2;         // Pipelined copy of psum_cycle_limit_0 (2-cycle delay).
  reg [ 7:0] psum_cycle_limit_3;         // Pipelined copy of psum_cycle_limit_0 (3-cycle delay).
  reg [ 7:0] psum_cycle_limit_4;         // Pipelined copy of psum_cycle_limit_0 (4-cycle delay); aligns with RAM read latency.
  reg [11:0] pcb_inc_1;                  // Composite psum buffer address word, stage 1: {sending_clusters, psum_cycle_buffer_1}.
  reg [11:0] pcb_inc_2;                  // Composite psum buffer address word, stage 2: {sending_clusters, psum_cycle_buffer_2}.
  reg [11:0] pcb_inc_3;                  // Composite psum buffer address word, stage 3: {sending_clusters, psum_cycle_buffer_3}.
  reg [3:0] fsm_psum_row_offset;         // Row offset used when iterating over cluster rows in PSUM_GET_RESULTS.

  // --- iact buffer next-address combinational signal ---
  // Asserted when the iact_stream_constructor is about to need the next buffer address
  // (one cycle before the current address window runs out).
  reg signed [                                 12-1:0] current_x;
  reg signed [                                  8-1:0] current_y;
  reg        [           $clog2(2*IACT_RAM_CELLS)-1:0] current_pos_in_iact_glb;
  reg        [                                  3-1:0] relative_pos;
  reg        [                                    1:0] iact_channel_sending_cycle1;
  reg        [                                    1:0] iact_channel_sending_cycle2;
  reg        [(8*DATA_IACT_BITWIDTH*NUM_GLB_IACT)-1:0] iact_input_window;
  reg        [(2*DATA_IACT_BITWIDTH*NUM_GLB_IACT)-1:0] iact_input_window_q;

  reg [ 7:0] iact_channel_counter_reg;  // Registered copy of iact_channels_counter for cross-process use.

  // --- PSUM FSM state registers ---
  reg [15:0] fsm_psum_cycle;            // Cycle counter within the current PSUM FSM state.
  reg [ 3:0] fsm_psum_last_state;       // Previous PSUM FSM state; used for transition tracing in simulation.
  reg [ 3:0] fsm_psum_current_state;    // Current PSUM FSM state (one of PSUM_IDLE … SEND_PSUM_TO_IACT).
  //assign debug_fsm_psum_state = fsm_psum_current_state; // Expose PSUM state on debug port.

  reg        psum_transmitted;           // Handshake flag: 1 once psum_ready_o from OpenEye_Parallel confirms receipt.
  reg        psum_router_set_reg;        // High once psum router modes have been updated for the output phase.
  reg        start_new_cycle;            // Single-cycle pulse: triggers a new compute iteration (transitions PSUM_IDLE→WAIT_TO_SEND_READY_SIGNAL).

  // --- PSUM FSM cluster sweep pointers ---
  reg [$clog2(CLUSTER_COLUMNS)-1:0]   fsm_x_cl_psum;    // Column-cluster index during result sweep.
  reg [$clog2(CLUSTER_COLUMNS)-1:0]   fsm_x_cl_psum_q1;  // 1-cycle delayed fsm_y_cl_psum (pipeline alignment).
  reg [$clog2(CLUSTER_COLUMNS)-1:0]   fsm_x_cl_psum_q2;  // 1-cycle delayed fsm_y_cl_psum (pipeline alignment).
  reg [$clog2(CLUSTER_COLUMNS)-1:0]   fsm_x_cl_psum_q3;  // 1-cycle delayed fsm_y_cl_psum (pipeline alignment).
  reg [$clog2(CLUSTER_ROWS+1)-1:0]    fsm_y_cl_psum_offset;
  reg [$clog2(CLUSTER_ROWS+1)-1:0]    fsm_y_cl_psum;    // Row-cluster index during result sweep.
  reg [$clog2(CLUSTER_ROWS+1)-1:0]    fsm_y_cl_psum_q1; // 1-cycle delayed fsm_y_cl_psum (pipeline alignment).
  reg [$clog2(CLUSTER_ROWS+1)-1:0]    fsm_y_cl_psum_q2; // 2-cycle delayed fsm_y_cl_psum.
  reg [$clog2(CLUSTER_ROWS+1)-1:0]    fsm_y_cl_psum_q3; // 3-cycle delayed fsm_y_cl_psum.

  // --- Quantization output staging (SEND_PSUM_TO_IACT) ---
  // After quantization, the 8 output bytes (one per filter in the current group)
  // are held here for packing into the iact buffer.
  reg [7:0] quantized_value_reg [TRANS_WORDS-1:0]; // Quantized output bytes [0..7]; one per parallel filter.

  reg [7:0] current_filter; // Index of the filter group currently being quantized/output [0..filters-1].
  wire[15:0]psum_output_words; // Amoutn of Output words for streaming

  // --- Wires from iact_stream_constructor instances to OpenEye_Parallel ---
  // These buses aggregate the per-instance outputs from all CLUSTER_COLUMNS×CLUSTER_ROWS
  // iact_stream_constructor modules into flat vectors for the OpenEye_Parallel port.
  wire [      $clog2(NUM_GLB_IACT+1)*CLUSTERS*PES-1:0] iact_choose_i_oep_w;  // Iact-source selector for each PE in each cluster.
  wire [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_data_i_oep_w;    // Packed iact data from all converters.
  wire [                    CLUSTERS*NUM_GLB_IACT-1:0] iact_enable_i_oep_w;  // Per-GLB iact-valid flags from converters.

  // -----------------------------------------------------------------------
  // Process 6: Main FSM
  // The central sequencing process.  It steps through the 15 states defined
  // by the localparams above, driving virtually every other signal in this
  // module.  The process is a single large always block; the case statement
  // dispatches to per-state logic.
  //
  // Loop variables used inside this process:
  //   cr  - cluster-row loop index (integer)
  //   cc  - cluster-column loop index (integer)
  //   g   - general-purpose loop index (integer)
  //
  // Summary of what each state does inside this always block:
  //
  //  IDLE
  //    Waits for the first enable_dma_i_reg pulse from the host.
  //    Transitions to GET_PARAMETERS unconditionally on reset-release
  //    (reset puts the FSM directly into GET_PARAMETERS).
  //
  //  GET_PARAMETERS  (fsm_cycle 0..3+)
  //    Receives DMA words one per cycle while enable_dma_i_reg is high.
  //    Cycles trough states of the dma_storage decoder (write_dma_en).
  //    Remaining cycles fill compute_mask_reg (one word = one cluster-column
  //    enable bit).  ready_dma_o stays high throughout.
  //    Transitions to GET_ROUTER_CONFIG once the expected word count is done.
  //
  //  GET_ROUTER_CONFIG  (fsm_cycle 0..FSM_CEIL_IACT/WGHT/PSUM_RTR_CCLS-1)
  //    Receives router-mode vectors from the DMA and unpacks them into
  //    router_mode_iact, router_mode_wght, router_mode_psum arrays.
  //    Also latches layer geometry from dma_storage outputs (iact_size_x/y,
  //    needed_cycles, filters ...).
  //    Transitions to GET_IACT.
  //
  //  GET_IACT  (select_ram_counter walks 0..IACT_RAM_CELLS-1)
  //    Writes raw iact pixel words from the DMA bus directly into the
  //    inactive half of the double-buffered iact RAM (controlled by
  //    buffer_select XOR choose_iact_buffer).
  //    iact_buffer_en_w asserted each cycle; address and data taken
  //    from the DMA word.
  //    Transitions to GET_WGHT when select_ram_counter wraps.
  //
  //  GET_WGHT  (wght_buffer_wr_addr walks 0..wghts_per_pe)
  //    Writes weight words from the DMA bus into wght_buffer_.
  //    from layer geometry.
  //    Transitions to GET_BIAS.
  //
  //  GET_BIAS  (psum_buffer_addr_array iterates over all clusters/GLBs)
  //    Loads initial bias values from DMA into psum_buffer_.
  //    Transitions to GET_QUANTIZE.
  //
  //  GET_QUANTIZE  (fsm_cycle 0..15)
  //    Receives 16 DMA words containing per-filter quantization parameters;
  //    unpacks quant_mant and quant_exp from each 64-bit word (32-bit each).
  //    Transitions to START_CONVERTER.
  //
  //  START_CONVERTER
  //    Polls converters_ready (AND-tree of all iact_converter_ready_w).
  //    Toggles buffer_select (swaps ping/pong) and computes sliding-window
  //    address limits (buffer_addr_upper/lower_limit) once ready.
  //    Transitions to CONVERT_IACT when all converters are idle.
  //
  //  CONVERT_IACT
  //    Drives iact_stream_constructors through the full iact encoding pass:
  //      - Updates iact_channels_counter and iact_converter_buffer_addr_cycles.
  //      - Manages the sliding address window (overhang, limit_increase_reg).
  //    Transitions to WAIT_CYCLE when the full set of addresses is consumed.
  //
  //  WAIT_CYCLE  (16-cycle drain)
  //    Holds for 16 cycles to flush the converter pipeline.
  //    Asserts send_data_reg (one pulse) to kick Process 5.
  //    Transitions to WAIT_FOR_RESULTS.
  //
  //  WAIT_FOR_RESULTS
  //    Idle while OpenEye_Parallel computes.
  //    Monitors current_cycle vs needed_cycles.
  //    On last_data_o (from PSUM FSM): decides next state:
  //      -> GET_PARAMETERS  if this was the final layer iteration.
  //      -> RECEIVE_PSUMS_TO_IACT  if results feed the next layer.
  //      -> MAXPOOLING_READ  if max-pooling is enabled.
  //
  //  RECEIVE_PSUMS_TO_IACT
  //    Reads quantized results from quantized_value_reg (written by PSUM FSM)
  //    and packs them back into the iact double-buffer for the next layer.
  //    Manages select_ram_counter and byte-packing within 64-bit words.
  //    Transitions back to GET_PARAMETERS (or loops for multi-pass layers).
  //
  //  MAXPOOLING_READ
  //    Reads the current iact buffer contents through the pooling pipeline:
  //      pooling_stage_1/2/3/4 -> pooling_regs[].
  //    Advances select_ram_counter over all active cells.
  //    Transitions to MAXPOOLING_SEND when all cells read.
  //
  //  MAXPOOLING_SEND
  //    Writes pooling_regs[] results back into the iact buffer at the
  //    reduced (pooled) addresses.
  //    Transitions to GET_PARAMETERS when done.
  //
  // reset_cycle: a flag set by any state when a layer-internal iteration
  //   needs to restart (e.g. multi-pass weight reuse); clears most counters.
  // -----------------------------------------------------------------------
  integer cr, cc, g;
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin
      status_reg_enable_reg              <= 0;
      data_mode_reg                      <= 0;
      fraction_bit_reg                   <= 0;
      fsm_cycle                          <= 0;
      fsm_last_state                     <= IDLE;
      fsm_current_state                  <= GET_PARAMETERS;
      fsm_x_cl                           <= 0;
      fsm_y_cl                           <= 0;
      fsm_iact_r                         <= 0;
      fsm_wght_r                         <= 0;
      buffer_select                      <= 0;
      fifo_data_i                        <= 0;
      fifo_read_i                        <= 0;
      fifo_write_i                       <= 0;
      bano_cluster_mode_reg              <= 0;
      af_cluster_mode_reg                <= 0;
      compute_mask_reg                   <= 0;
      ready_dma_o                        <= 0;
      iact_buffer_data_w                 <= 0;
      wght_buffer_wr_addr                <= 0;
      wght_buffer_en_w                   <= 0;
      wght_buffer_data_w                 <= 0;
      finished_cycles_iact               <= 0;
      // iact converter
      iact_out_reg                       <= 0;
      iact_ready                         <= 0;
      buffer_addr_upper_limit            <= 0;
      limit_increase_reg                 <= 0;
      overhang_counter                   <= 0;
      overhang                           <= 0;
      overhang_delay                     <= 0;
      buffer_addr_lower_limit            <= 0;
      current_buffer                     <= 0;
      iact_channels_counter              <= 0;
      reset_cycle                        <= 0;
      select_ram_counter                 <= 0;
      select_ram_counter2                <= 0;
      psum_x_with_add_up                 <= 0;
      iact_converter_cycles              <= 0;
      iact_converter_buffer_addr_cycles  <= 0;
      send_data_reg                      <= 0;
      write_dma_en                       <= 0;
      dma_data_i                         <= 0;
      // Pooling
      iact_buffer_data_pool_temp         <= 0;
      pooling_buffer_enable              <= 0;
      for (a = 0; a < CHANNELS_PER_WORD; a = a + 1) begin
        pooling_buffer_new[a] <= 0;
        pooling_buffer_old_q[a] <= 0;
      end
      for (a = 0; a < 32; a = a + 1) begin
        pooling_regs[a] <= -128;
      end
      for (a = 0; a < 8; a = a + 1) begin
        pooling_stage_1[a] <= 0;
      end
      for (a = 0; a < 4; a = a + 1) begin
        pooling_stage_2[a] <= 0;
      end
      for (a = 0; a < 2; a = a + 1) begin
        pooling_stage_3[a] <= 0;
      end
      pooling_stage_4 <= 0;
      quant_reg       <= 0;

      for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
        iact_buffer_en_w[a]      <= 0;
        iact_buffer_en_r[a]      <= 0;
        iact_buffer_en_r[a]      <= 0;
        iact_buffer_en_w[a]      <= 0;
        iact_buffer_addr_reg[a]  <= 0;
      end
      buffer_addr_temp_reg         <= 0;
      iact_buffer_data_w           <= 0;
      choose_iact_buffer           <= 0;
      converters_ready              = 0;
      iact_converter_enc_enable    <= 0;
      iact_converter_params_enable <= 0;
      select_iact_glb_addr_inc     <= 0;
      iact_to_psum_shift_reg       <= 0;
      iact_to_psum_mux_reg         <= 0;
      iact_to_psum_x_pos_counter   <= 0;
      iact_to_psum_trans_counter   <= 0;
      iact_to_psum_storage_counter <= 0;
      iact_to_psum_start_shifting  <= 0;
      set_pointer_start            <= 0;
      current_x                    <= 0;
      current_y                    <= 0;
      relative_pos                 <= 0;
      current_pos_in_iact_glb      <= 0;
      iact_input_window            <= 0;
      iact_input_window_q          <= 0;
      iact_channel_sending_cycle1  <= 0;
      iact_channel_sending_cycle2  <= 0;
    end else begin
      for (a = 0; a < CHANNELS_PER_WORD; a = a + 1) begin
        pooling_buffer_old_q[a] <= pooling_buffer_old[a];
      end
      case (fsm_current_state)

        // -------------------------------------------------------------------
        // IDLE
        // Waiting state entered only briefly at reset-release.
        // Because reset initialises fsm_current_state = GET_PARAMETERS,
        // IDLE is normally never reached in the expected boot sequence.
        // If somehow entered, waits for the first enable_dma_i_reg pulse
        // (the host signalling that it is ready) and immediately transitions
        // to GET_PARAMETERS.
        // -------------------------------------------------------------------
        IDLE: begin
          if (enable_dma_i_reg) begin
            fsm_last_state    <= IDLE;
            fsm_current_state <= GET_PARAMETERS;
            fifo_data_i       <= 0;
            fifo_read_i       <= 0;
            fifo_write_i      <= 0;
          end
        end

        // -------------------------------------------------------------------
        // GET_PARAMETERS
        // Receives the per-layer configuration burst from the host DMA.
        // - Asserts ready_dma_o and status_reg_enable_reg.
        // - Sets reset_cycle on the first cycle to clear counters from the
        //   previous layer; clears it once DMA data arrives.
        // - Clears pooling_regs, quant arrays, buffer addresses.
        // - Cycles 0-3 (fsm_cycle 0-3): writes each 64-bit DMA word to
        //   dma_storage via write_dma_en.
        //   Cycle 3 also latches padding = (kernel_size-1)/2.
        // - Cycles 4+: fills compute_mask_reg one 64-bit slice per cycle;
        //   each bit in the DMA word enables one (PE, cluster) pair.
        // - After the last expected word (4 + ceil(PES*CLUSTERS / 64)):
        //   snaps choose_iact_buffer to choose_iact_buffer_input,
        //   resets fsm_cycle, and transitions to GET_ROUTER_CONFIG.
        //   (FC layers also zero padding here.)
        // -------------------------------------------------------------------
        GET_PARAMETERS: begin
          fifo_data_i                  <= 0;
          fifo_read_i                  <= 0;
          fifo_write_i                 <= 0;
          status_reg_enable_reg        <= 1;
          ready_dma_o                  <= 1;
          reset_cycle                  <= 1;
          buffer_addr_lower_limit      <= 0;
          buffer_addr_upper_limit      <= 0;
          set_pointer_start            <= 0;
          limit_increase_reg           <= 0;
          iact_to_psum_trans_counter   <= 0;
          iact_to_psum_x_pos_counter   <= 0;
          iact_to_psum_storage_counter <= 0;
          for (a = 0; a < IACT_RAM_CELLS; a = a + 1) begin
            iact_buffer_en_r[a]      <= 0;
            iact_buffer_en_w[a]      <= 0;
            iact_buffer_addr_reg[a]      <= 0;
          end
          buffer_addr_temp_reg    <= 0;
          iact_buffer_data_w <= 0;
          for (a = 0; a < 32; a = a + 1) begin
            pooling_regs[a] <= -128;
          end
          for (a = 0; a < 8; a = a + 1) begin
            pooling_stage_1[a] <= -128;
          end
          for (a = 0; a < 4; a = a + 1) begin
            pooling_stage_2[a] <= -128;
          end
          for (a = 0; a < 2; a = a + 1) begin
            pooling_stage_3[a] <= -128;
          end
          pooling_stage_4 <= -128;
          quant_reg       <= 0;
          write_dma_en    <= 0;
          if (enable_dma_i_reg) begin
            if (ready_dma_o) begin
              fsm_cycle   <= fsm_cycle + 1;
              reset_cycle <= 0;
              dma_data_i  <= data_dma_i_reg;
              // write_dma_en is itself a registered (nonblocking) signal,
              // so it takes effect one cycle after this condition is
              // checked - by the time it deasserts, dma_storage's shift
              // register has already received exactly TRANSMISSIONS
              // pushes (fsm_cycle 0..TRANSMISSIONS-1). Using
              // TRANSMISSIONS+1 here (as the other TRANSMISSIONS+1 uses
              // below do, for the separate compute_mask_reg/PE-bitmap
              // phase boundary) pushes one extra, stale word into the
              // shift chain, permanently misaligning every decoded field
              // by one transmission slot - harmless on the old
              // non-shifting dma_storage architecture (an extra write to
              // an already-terminal per-field register is a no-op), but
              // corrupting on the new pipelined shift-register one.
              if (fsm_cycle < TRANSMISSIONS) begin
                  write_dma_en   <= 1;
              end
              // Shifted from TRANSMISSIONS+1 to TRANSMISSIONS to match the
              // write_dma_en fix above: dma_storage now correctly
              // receives exactly TRANSMISSIONS words (fsm_cycle
              // 1..TRANSMISSIONS), so the PE-bitmap phase starts one
              // cycle earlier than before. Leaving this at TRANSMISSIONS+1
              // read the bitmap from the wrong word and delayed the
              // GET_PARAMETERS->GET_ROUTER_CONFIG transition by one DMA
              // word, which then made GET_ROUTER_CONFIG (which reads
              // data_dma_i_reg directly, not through dma_storage's shift
              // chain) skip the first router_mode_iact word entirely.
              for (a = 0; a < PES * CLUSTERS; a = a + 1) begin
                if (fsm_cycle >= TRANSMISSIONS & (((fsm_cycle - TRANSMISSIONS) * DMA_BITWIDTH <= a) & ((fsm_cycle - 4) * DMA_BITWIDTH > a))) begin
                  compute_mask_reg[a] <= data_dma_i_reg[a%DMA_BITWIDTH];
                end
              end
              if (fsm_cycle == (TRANSMISSIONS + (((PES * CLUSTERS) - 1)/DMA_BITWIDTH))) begin
                fsm_last_state <= GET_PARAMETERS;
                fsm_current_state  <= GET_ROUTER_CONFIG;
              end
            end
          end
        end
        
        // -------------------------------------------------------------------
        // GET_ROUTER_CONFIG
        // Receives the router-mode burst and latches all remaining layer
        // geometry parameters from the now-stable dma_storage outputs.
        //
        // On entry each cycle:
        // - Asserts ready_dma_o and new_stream.
        // - Latches iact_size_c = iact_channels_per_pe * iact_channel_max_cycles
        //   (FC: multiplied by NUM_GLB_WGHT for full row sweep).
        // - Latches psum_x_with_add_up = psum_size_x + add_up.
        // - max_pooling mode: asserts read-enable for all buffer cells so
        //   the converter buffer is pre-warmed.
        //
        // While enable_dma_i_reg: counts fsm_cycle over the expected router
        // word count (FSM_CEIL_IACT_RTR_CCLS + WGHT + PSUM - 1).
        // On the last word:
        // - Clears fsm_cycle, deasserts status_reg_enable_reg.
        // - Transitions: GET_IACT normally; GET_WGHT if skipIact; GET_OFFSET
        //   if max_pooling (no iact load needed, jump straight to pooling).
        // -------------------------------------------------------------------
        GET_ROUTER_CONFIG: begin
          status_reg_enable_reg <= 0;
          ready_dma_o           <= 1;
          psum_x_with_add_up    <= psum_size_x + add_up;
          if (enable_dma_i_reg) begin
            fsm_cycle <= fsm_cycle + 1;
            if(fsm_cycle == FSM_CEIL_IACT_RTR_CCLS + FSM_CEIL_WGHT_RTR_CCLS + FSM_CEIL_PSUM_RTR_CCLS - 1 | (CLUSTERS == 1)) begin
              fsm_cycle             <= 0;
              fsm_last_state        <= GET_ROUTER_CONFIG;
              if (!skipIact_reg) begin
                fsm_current_state <= GET_IACT;
                for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
                  iact_buffer_addr_reg[a] <= ~0;
                end
                wght_buffer_wr_addr <= ~0;
              end else begin
                fsm_current_state   <= GET_WGHT;
                wght_buffer_wr_addr <= ~0;
              end
              if (max_pooling) begin
                ready_dma_o       <= 0;
                fsm_current_state <= GET_QUANTIZE;
              end
            end
          end
        end

        // -------------------------------------------------------------------
        // GET_IACT
        // Streams raw iact pixel data from the host into the inactive half
        // of the 32-cell double-buffer (BUFFER_A).
        //
        // - Asserts ready_dma_o; clears all buffer write enables at the
        //   top of each cycle (en_w gated per-cell below).
        // - On each enable_dma_i_reg pulse:
        //   * Advances the round-robin cell pointer current_buffer
        //     (wraps at IACT_RAM_CELLS-1).
        //   * Asserts iact_buffer_en_w for the current cell.
        //   * Writes data_dma_i_reg into iact_buffer_data_w.
        //   * Updates iact_buffer_addr_reg for the previous cell pointer
        //   * Advances current_buffer_addr: the address within the RAM
        //     cell (rows increase by 1 per full IACT_RAM_CELLS-wide word cycle).
        // - Exit condition: fsm_cycle reaches
        //   ceil(iact_size_x * iact_size_y * iact_size_c / IACT_WORDS_IN_RAM) - 1.
        //   Resets fsm_cycle and transitions to GET_WGHT.
        // - When !enable_dma_i_reg: de-asserts all write enables.
        // -------------------------------------------------------------------
        GET_IACT: begin
          status_reg_enable_reg <= 0;
          ready_dma_o           <= 1;
          for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
            iact_buffer_en_w[a] <= 0;
          end
          if (enable_dma_i_reg) begin
            fsm_cycle        <= fsm_cycle + 1;
            current_buffer   <= current_buffer + 1;
            iact_buffer_data_w[0+:DMA_BITWIDTH] <= data_dma_i_reg;
            for (a = 0; a < IACT_CYCLES_ONE_WORD_ALL_CELLS-1; a=a+1) begin
              iact_buffer_data_w[DMA_BITWIDTH*(a+1)+:DMA_BITWIDTH]
              <= iact_buffer_data_w[DMA_BITWIDTH*a+:DMA_BITWIDTH];
            end
            if (current_buffer == 0) begin
              for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
                iact_buffer_addr_reg[a] <= iact_buffer_addr_reg[a] + 1;
              end
            end
            if (current_buffer == IACT_ONE_WORD_ALL_RAM - 1) begin
              current_buffer <= 0;
              for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
                iact_buffer_en_w[a] <= 1;
              end
            end
            if (fsm_cycle == trans_cycles_iact - 1) begin
              fsm_cycle          <= 0;
              wght_buffer_wr_addr <= ~0;
              fsm_current_state  <= GET_WGHT;
              fsm_last_state     <= GET_IACT;
            end
          end else begin
            for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
              iact_buffer_en_w[a] <= 0;
            end
          end
        end

        // -------------------------------------------------------------------
        // GET_WGHT
        // Streams weight data from the host into wght_buffer and
        // computes two key timing parameters used by the converter later.
        //
        // On each enable_dma_i_reg pulse:
        // - Computes iact_converter_max_cycles:
        //     normal: iact_size_y + kernel_size - 1 (rows the converter must scan).
        //     1-channel: half of the above (two rows packed per word).
        //     FC: fixed at 2.
        //   / WORDS_PER_CYCLE (minimum cycles the converter holds each buffer line).
        // - Unpacks two weight words from the 64-bit DMA word:
        //     bits [TRANS_BITWIDTH_WGHT-1:0] -> row fsm_y_cl, GLB fsm_wght_r.
        //     bits [2*TRANS_BITWIDTH_WGHT-1:TRANS_BITWIDTH_WGHT] -> mirrored row
        //     (second half of the cluster, offset by CLUSTER_ROWS).
        // - Advances fsm_wght_r; when it wraps at NUM_GLB_WGHT, advances fsm_y_cl.
        // - When fsm_y_cl wraps at CLUSTER_ROWS (one full weight word assembled):
        //   * Asserts wght_buffer_en_w, advances wght_buffer_wr_addr.
        //   * Recomputes wghts_per_pe (total weight depth).
        //   * Exit: when fsm_cycle reaches the last expected weight word,
        //     resets fsm_cycle, wghts_per_pe to final value, and transitions
        //     to GET_BIAS (or GET_QUANTIZE if skipPsum is set).
        // -------------------------------------------------------------------
        GET_WGHT: begin
          status_reg_enable_reg <= 0;
          ready_dma_o           <= 1;
          for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
            iact_buffer_en_w[a] <= 0;
          end
          wght_buffer_en_w <= 0;
          if (enable_dma_i_reg) begin
            fsm_cycle <= fsm_cycle + 1;
            current_buffer   <= current_buffer + 1;
            wght_buffer_data_w[0+:DMA_BITWIDTH] <= data_dma_i_reg;
            for (a = 0; a < WGHT_CYCLES_ONE_WORD_ALL_CELLS-1; a=a+1) begin
              wght_buffer_data_w[DMA_BITWIDTH*(a+1)+:DMA_BITWIDTH]
              <= wght_buffer_data_w[DMA_BITWIDTH*a+:DMA_BITWIDTH];
            end
            if (current_buffer == 0) begin
                wght_buffer_wr_addr <= wght_buffer_wr_addr + 1;
            end
            if (current_buffer == WGHT_ONE_WORD_ALL_RAM - 1) begin
              current_buffer   <= 0;
              wght_buffer_en_w <= 1;
            end
            if(fsm_cycle == trans_cycles_wght - 1)begin
              fsm_cycle      <= 0;
              fsm_last_state <= GET_WGHT;
              if (!skipPsum_reg) begin
                fsm_current_state <= GET_BIAS;
              end else begin
                fsm_current_state <= GET_QUANTIZE;
              end
            end
          end
        end

        // -------------------------------------------------------------------
        // GET_BIAS
        // Transitions immediately to GET_QUANTIZE (no DMA words consumed
        // here) while computing the iact sliding-window address parameters
        // from the layer geometry.
        //
        // Triggered when psum_cnt != 0 (bias was already written to
        // psum_buffer by the PSUM FSM PSUM_IDLE handler during the
        // previous GET_BIAS phase; this state just computes addresses):
        // - Resets wght_buffer_wr_addr to 0.
        // - Computes limit_increase_reg: how many RAM cells the upper
        //   address limit advances per iact converter row.
        //     normal: (iact_size_x * iact_channels_per_pe) / (WORDS_PER_CYCLE*4).
        //     1-channel: uses iact_size_x*2 (two rows packed together).
        //     FC: uses NUM_GLB_WGHT * iact_channels_per_pe / 8.
        // - Computes overhang_discrepancy: the remainder bytes that cause
        //   a fractional extra cell every overhang_counter cycles.
        // - Clears overhang, overhang_delay, fsm_cycle.
        // - Transitions to GET_QUANTIZE.
        // Note: when psum_cnt == 0, the PSUM FSM has not yet run; the
        // GET_BIAS->GET_QUANTIZE transition still fires since the state
        // is entered only after psum_cnt is set.
        // -------------------------------------------------------------------
        GET_BIAS: begin
          status_reg_enable_reg <= 0;
          ready_dma_o           <= 1;
          wght_buffer_en_w      <= 0;
          overhang              <= 0;
          overhang_delay        <= 0;
          if (fsm_psum_cycle == trans_cycles_psum - 1) begin
            fsm_last_state         <= GET_BIAS;
            fsm_current_state      <= GET_QUANTIZE;
            fsm_cycle              <= 0;
            wght_buffer_wr_addr    <= 0;
            if (fully_connected_layer) begin
              overhang             <= 4;
            end
          end
        end

        // -------------------------------------------------------------------
        // GET_QUANTIZE
        // Receives 16 DMA words containing per-filter quantization scale
        // factors (mantissa + exponent) for up to 32 output filters.
        //
        // - Initialises overhang_counter = overhang_discrepancy (fractional
        //   cell accumulator seeded from GET_BIAS computation).
        // - Asserts ready_dma_o; clears wght_buffer_en_w.
        // - max_pooling: deasserts ready_dma_o and asserts all buffer read
        //   enables (pooling pass doesn't need DMA input here).
        // - On each enable_dma_i_reg pulse:
        //   * Increments fsm_cycle.
        //   * Unpacks two quant entries from the 64-bit DMA word:
        //       quant_exp [2*fsm_cycle]   = bits[31:25]  (7-bit)
        //       quant_mant[2*fsm_cycle]   = bits[24:0]   (25-bit)
        //       quant_exp [2*fsm_cycle+1] = bits[63:57]
        //       quant_mant[2*fsm_cycle+1] = bits[56:32]
        // - After 16 words (fsm_cycle == 15): clears fsm_cycle,
        //   transitions to GET_OFFSET.
        // -------------------------------------------------------------------
        GET_QUANTIZE: begin
          choose_iact_buffer <= choose_iact_buffer_input;
          overhang_counter   <= overhang_discrepancy;
          ready_dma_o        <= 1;
          wght_buffer_en_w   <= 0;
          pooling_buffer_enable <= 1;
          if (pooling_mode == 0) begin
            for (a = 0; a < CHANNELS_PER_WORD; a = a + 1) begin
              pooling_buffer_new[a] <= -128;
            end
          end else begin
            for (a = 0; a < CHANNELS_PER_WORD; a = a + 1) begin
              pooling_buffer_new[a] <= 0;
            end
          end
          if (enable_dma_i_reg) begin
            fsm_cycle <= fsm_cycle + 1;
            quant_reg[0+:DMA_BITWIDTH] <= data_dma_i_reg;
            for (a = 0; a < QUANT_TRANS_CYCLES - 1; a=a+1) begin
              quant_reg[(a+1)*DMA_BITWIDTH+:DMA_BITWIDTH] <= quant_reg[a*DMA_BITWIDTH+:DMA_BITWIDTH];
            end
          end
          if (max_pooling) begin
            ready_dma_o <= 0;
            for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
              iact_buffer_en_r[a] <= 1;
            end
          end
          if ((fsm_cycle == QUANT_TRANS_CYCLES - 1) & enable_dma_i_reg) begin
            ready_dma_o           <= 0;
            fsm_cycle             <= 0;
            pooling_buffer_enable <= 0;
            fsm_last_state        <= GET_QUANTIZE;
            fsm_current_state     <= START_CONVERTER;
            buffer_addr_upper_limit <= initial_upper_limit;
            if (max_pooling) begin
              fsm_cycle           <= 2;
              fsm_current_state   <= MAXPOOLING_READ;
              select_ram_counter  <= 0;
              select_ram_counter2 <= 0;
              buffer_addr_temp_reg <= -1;
              iact_buffer_data_w   <= 0;
            end
          end
        end
        
        // -------------------------------------------------------------------
        // START_CONVERTER
        // Waits for all iact_stream_constructor instances to become idle
        // (combinational AND of iact_converter_ready_w[col][row]).
        //
        // Each cycle:
        // - Recomputes converters_ready = AND of all ready flags.
        // - When all ready:
        //   * Transitions to CONVERT_IACT.
        //   * Resets select_ram_counter to 0.
        //   * Enables all buffers read lines (iact_buffer_en_r all 1).
        //   * Resets all buffer address pointers to 0.
        //
        // This state may spin for multiple cycles if any converter is still
        // finishing a previous encoding run from the last iact batch.
        // -------------------------------------------------------------------
        START_CONVERTER: begin
          converters_ready  = 1;
          for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
            iact_buffer_addr_reg[a] <= 0;
          end
          for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
            iact_buffer_en_r[a] <= 1;
          end
          for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
            for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
              converters_ready = converters_ready & iact_converter_ready_w[a][b];
            end
          end
          fsm_cycle        <= fsm_cycle + 1;
          if (fsm_cycle == 3) begin
            fsm_cycle <= fsm_cycle;
            if (converters_ready == 1) begin
              fsm_cycle                   <= 0;
              iact_input_window           <= iact_buffer_data_r[0+:2*8*4];
              iact_channel_sending_cycle1 <= 0;
              iact_channel_sending_cycle2 <= 0;
              iact_converter_enc_enable   <= 1;
              select_iact_glb_addr_inc    <= 0;
              current_x                   <= -padding_x;
              current_y                   <= -padding_y;
              current_pos_in_iact_glb     <= 1;
              relative_pos                <= 0;
              iact_input_window_q         <= 0;
              fsm_current_state           <= CONVERT_IACT;
              select_ram_counter          <= 0;
            end
          end
        end

        // -------------------------------------------------------------------
        // CONVERT_IACT
        // Drives the iact_stream_constructor pipeline through a full
        // encoding pass over all input rows and channel batches.
        //
        // Counter hierarchy (innermost to outermost):
        //   iact_converter_buffer_addr_cycles [0..max_cycles-1]
        //     -> per address-window step (how many cycles each RAM row is held)
        //   iact_converter_cycles [0..iact_converter_max_cycles-1]
        //     -> one spatial row of the input feature map
        //   iact_channels_counter [0..iact_channel_max_cycles-1]
        //     -> one channel batch
        //        //
        //
        // Exit: when iact_channels_counter wraps at iact_channel_max_cycles,
        //   clears all counters, and goes to WAIT_CYCLE.
        // -------------------------------------------------------------------
        CONVERT_IACT: begin
          //TODO: Add pointer for multiple iact_glbs
          fsm_cycle                 <= fsm_cycle + 1;
          iact_converter_enc_enable <= 0;
          if (fsm_cycle >= 2) begin
            iact_channel_sending_cycle1 <= iact_channel_sending_cycle1 + 1;
            if (iact_channel_sending_cycle1 == 1 - 1) begin
              iact_channel_sending_cycle1 <= 0;
              iact_channel_sending_cycle2 <= iact_channel_sending_cycle2 + 1;
              if (iact_channel_sending_cycle2 == channel_div_trans - 1) begin
                iact_channel_sending_cycle2 <= 0;
                current_x <= current_x + NUM_GLB_IACT;
                if (current_x == iact_x_add_up + padding_x - 1) begin
                  current_x <= -padding_x;
                  current_y <= current_y + 1;
                  if (current_y == iact_size_y + padding_y - 1) begin
                    current_y <= -padding_y;
                  end
                end
              end 
              iact_input_window_q <= 0;
              if ((current_x >= 0) & (current_y >= 0) & (current_x < iact_size_x) & (current_y < iact_size_y)) begin
                iact_input_window       <= iact_input_window >> 16;
                iact_input_window_q     <= iact_input_window[15:0];
                relative_pos            <= relative_pos + 1;
                if (relative_pos == IACT_RAM_CELLS - 1) begin
                  relative_pos             <= 0;
                  current_pos_in_iact_glb  <= current_pos_in_iact_glb + 1;
                  if (current_pos_in_iact_glb == IACT_RAM_CELLS - 1) begin
                    current_pos_in_iact_glb  <= 0;
                  end
                  iact_input_window        <= iact_buffer_data_r[current_pos_in_iact_glb*2*8*4+:2*8*4];
                  select_iact_glb_addr_inc <= current_pos_in_iact_glb;
                  for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
                    if (select_iact_glb_addr_inc == a) begin
                      iact_buffer_addr_reg[a] <= iact_buffer_addr_reg[a] + 1;
                    end
                  end
                end
              end
            end
          end
          if (fsm_cycle == iact_buffer_words_per_write) begin
            fsm_current_state     <= WAIT_CYCLE;
            fsm_cycle             <= 0;
            for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
              iact_buffer_en_r[a] <= 0;
            end
          end
        end

        // -------------------------------------------------------------------
        // WAIT_CYCLE
        // Pipeline drain state: holds for 4*4 = 16 cycles to let the iact
        // converter's internal pipeline flush before any computation starts.
        //
        // - Copies iact_out_reg into iact_buffer_data_w (legacy, unused
        //   in the current data path).
        // - Deasserts iact_ready and iact_converter_enc_enable.
        // - Counts fsm_cycle; at cycle 16:
        //   * Clears fsm_cycle and iact_converter_cycles.
        //   * Resets buffer pointers (current_buffer/n_1/addr, all addr_reg).
        //   * Asserts send_data_reg for one cycle, triggering Process 5 to
        //     begin the weight + iact send phase.
        //   * Routing decision based on send_data_out from dma_storage:
        //       send_data_out = 1  -> WAIT_FOR_RESULTS (results go to host).
        //       send_data_out = 0  -> RECEIVE_PSUMS_TO_IACT (results loop back).
        //         Also initialises select_iact_glb_addr_inc, select_ram_counter,
        //         buffer address window parameters, and
        //         clears all buffer data write registers.
        // -------------------------------------------------------------------
        WAIT_CYCLE: begin
          iact_buffer_data_w        <= iact_out_reg;
          fsm_cycle                 <= fsm_cycle + 1;
          iact_ready                <= 0;
          iact_converter_enc_enable <= 0;
          if (fsm_cycle == (8 * 4 * 2)) begin
            fsm_cycle                <= 0;
            send_data_reg            <= 1;
            fsm_last_state           <= WAIT_CYCLE;
            if (send_data_out) begin
              fsm_current_state <= WAIT_FOR_RESULTS;
            end else begin
              fsm_current_state           <= RECEIVE_PSUMS_TO_IACT;
              iact_to_psum_start_shifting <= 0;
              select_ram_offset           <= 0;
              select_ram_counter          <= 0;
              if (iact_channels_per_pe_next_layer == 4) begin
                buffer_addr_upper_limit <= (((iact_size_x+1)/2))%IACT_RAM_CELLS;
              end else begin
                buffer_addr_upper_limit <= (((iact_size_x+1)/8))%IACT_RAM_CELLS;
              end
              buffer_addr_lower_limit <= 0;

              iact_buffer_data_w <= 0;
            end
            iact_converter_cycles <= 0;
            current_buffer      <= 0;
            for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
              iact_buffer_addr_reg[a] <= 0;
            end
          end
        end

        // -------------------------------------------------------------------
        // RECEIVE_PSUMS_TO_IACT
        // Packs the quantized output bytes produced by the PSUM FSM
        // (SEND_PSUM_TO_IACT) back into the iact double-buffer so they
        // can serve as input activations for the next layer.
        //
        // - When PSUM FSM reaches WAIT_FOR_SENDING_RESULTS: switches
        //   choose_iact_buffer to choose_iact_buffer_output (target half).
        // - Tracks iact_channels_counter (rising edge of single_iteration):
        //   increments and wraps at iact_channel_max_cycles.
        // - Each cycle: clears buffer write enables; if any were set in the
        //   previous cycle, advances that cell's read address by 1.
        // - When PSUM FSM is in SEND_PSUM_TO_IACT and fsm_psum_cycle >= 3
        //   (quantized_value_reg is valid)
        //   Packs quantized bytes into iact_buffer_data_w cells using one
        //   of three sub-modes determined by iact_channels_per_pe_next_layer:
        //     4: 4 bytes per pixel (normal conv next layer); uses select_ram_counter,
        //        wrapping-window logic, and overhang_discrepancy for odd iact_size_x.
        //     2: 2 bytes per pixel; simpler counter walking 16 cells.
        //     1: 1 byte per pixel (single-channel or FC); packs into 8-byte words
        //        using overhang_discrepancy to track sub-word offset.
        // - Exit: when PSUM FSM returns to PSUM_IDLE (all results written):
        //   Resets select_ram_counter, fsm_cycle; enables all buffer writes
        //   for one cycle (to flush); transitions to GET_PARAMETERS.
        // -------------------------------------------------------------------
        RECEIVE_PSUMS_TO_IACT: begin
          if (fsm_psum_current_state == WAIT_FOR_SENDING_RESULTS) begin
            choose_iact_buffer <= choose_iact_buffer_output;
          end
          if (single_iteration & (single_iteration2 == 0)) begin
            iact_channels_counter <= iact_channels_counter + 1;
            if (iact_channels_counter == iact_channel_max_cycles - 1) begin
              iact_channels_counter <= 0;
            end
          end
          for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
            iact_buffer_en_w[a] <= 0;
            if (iact_buffer_en_w[a] == 1) begin
              iact_buffer_addr_reg[a] <= iact_buffer_addr_reg[a] + 1;
            end
          end
          if (((fsm_psum_current_state == SEND_PSUM_TO_IACT) & (fsm_psum_cycle >= 7)) | (fsm_cycle != 0)) begin
            fsm_cycle                  <= fsm_cycle + 1;
            iact_to_psum_x_pos_counter <= 0;
            iact_to_psum_trans_counter <= iact_to_psum_trans_counter + 1;

            if (iact_to_psum_start_shifting) begin
              iact_to_psum_x_pos_counter <= iact_to_psum_x_pos_counter + 1;
              if (iact_to_psum_x_pos_counter < iact_size_x) begin
                iact_to_psum_storage_counter <= iact_to_psum_storage_counter + 1;
                if (iact_to_psum_storage_counter == ((8/TRANS_WORDS)*IACT_RAM_CELLS)-1) begin
                  iact_to_psum_storage_counter <= 0;
                  for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
                    iact_buffer_en_w[a] <= 1;
                  end
                end

                
                iact_buffer_data_w[(((8/TRANS_WORDS)*IACT_RAM_CELLS)-1)*(DATA_IACT_BITWIDTH*TRANS_WORDS)+:(DATA_IACT_BITWIDTH*TRANS_WORDS)] <= iact_to_psum_shift_reg[0+:(DATA_IACT_BITWIDTH*TRANS_WORDS)];
                iact_to_psum_shift_reg[(TRANS_WORDS-1)*(DATA_IACT_BITWIDTH*TRANS_WORDS)+:(DATA_IACT_BITWIDTH*TRANS_WORDS)] <= 0;
                for (b = 0; b < TRANS_WORDS-1; b=b+1) begin
                  iact_to_psum_shift_reg[b*(DATA_IACT_BITWIDTH*TRANS_WORDS)+:(DATA_IACT_BITWIDTH*TRANS_WORDS)] <=
                  iact_to_psum_shift_reg[(b+1)*(DATA_IACT_BITWIDTH*TRANS_WORDS)+:(DATA_IACT_BITWIDTH*TRANS_WORDS)];
                end
                for (b = 0; b < ((8/TRANS_WORDS)*IACT_RAM_CELLS)-1; b=b+1) begin
                  iact_buffer_data_w[(((8/TRANS_WORDS)*IACT_RAM_CELLS)-1-(b+1))*(DATA_IACT_BITWIDTH*TRANS_WORDS)+:(DATA_IACT_BITWIDTH*TRANS_WORDS)] <=
                  iact_buffer_data_w[(((8/TRANS_WORDS)*IACT_RAM_CELLS)-1-b)*(DATA_IACT_BITWIDTH*TRANS_WORDS)+:(DATA_IACT_BITWIDTH*TRANS_WORDS)];
                end
              end
            end

            if (iact_to_psum_trans_counter == TRANS_WORDS) begin
              iact_to_psum_trans_counter  <= 1;
              iact_to_psum_shift_reg      <= iact_to_psum_mux_reg;
              iact_to_psum_start_shifting <= 1;
              if (iact_to_psum_x_pos_counter >= iact_size_x - 1) begin
                iact_to_psum_x_pos_counter <= 0;
              end
            end            
            for (a = 0; a < TRANS_WORDS; a=a+1) begin
              iact_to_psum_mux_reg[a*(DATA_IACT_BITWIDTH*TRANS_WORDS)+(DATA_IACT_BITWIDTH*(TRANS_WORDS-1))+:DATA_IACT_BITWIDTH] <= quantized_value_reg[a];
            end
            for (a = 0; a < TRANS_WORDS; a=a+1) begin
              for (b = 0; b < TRANS_WORDS-1; b=b+1) begin
                iact_to_psum_mux_reg[a*(DATA_IACT_BITWIDTH*TRANS_WORDS)+(DATA_IACT_BITWIDTH*(TRANS_WORDS-1)-((b+1)*DATA_IACT_BITWIDTH))+:DATA_IACT_BITWIDTH] <=
                iact_to_psum_mux_reg[a*(DATA_IACT_BITWIDTH*TRANS_WORDS)+(DATA_IACT_BITWIDTH*(TRANS_WORDS-1)-(b*DATA_IACT_BITWIDTH))+:DATA_IACT_BITWIDTH];
              end
            end
          end
          if ((fsm_psum_current_state == PSUM_IDLE) & (fsm_cycle >= 1)) begin
            iact_to_psum_storage_counter <= iact_to_psum_storage_counter + 1;
            if (iact_to_psum_storage_counter == ((8/TRANS_WORDS)*IACT_RAM_CELLS)-1) begin
              iact_to_psum_storage_counter <= 0;
              for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
                iact_buffer_en_w[a] <= 1;
              end
            end
            for (b = 0; b < ((8/TRANS_WORDS)*IACT_RAM_CELLS)-1; b=b+1) begin
              iact_buffer_data_w[(((8/TRANS_WORDS)*IACT_RAM_CELLS)-1-(b+1))*(DATA_IACT_BITWIDTH*TRANS_WORDS)+:(DATA_IACT_BITWIDTH*TRANS_WORDS)] <=
              iact_buffer_data_w[(((8/TRANS_WORDS)*IACT_RAM_CELLS)-1-b)*(DATA_IACT_BITWIDTH*TRANS_WORDS)+:(DATA_IACT_BITWIDTH*TRANS_WORDS)];
            end
            if (iact_to_psum_storage_counter == 0) begin            
              select_ram_counter          <= 0;
              fsm_cycle                   <= 0;
              fsm_current_state           <= GET_PARAMETERS;
              fsm_last_state              <= RECEIVE_PSUMS_TO_IACT;
              send_data_reg               <= 0;
              iact_channels_counter       <= 0;
              iact_to_psum_start_shifting <= 0;
            end
          end
        end

        // -------------------------------------------------------------------
        // WAIT_FOR_RESULTS
        // Idle state while OpenEye_Parallel processes the layer.
        // The PSUM FSM runs independently during this state.
        //
        // - When PSUM FSM reaches WAIT_FOR_SENDING_RESULTS: switches
        //   choose_iact_buffer to choose_iact_buffer_output (ping/pong swap).
        // - Tracks iact_channels_counter (single_iteration rising edge):
        //   increments and wraps at iact_channel_max_cycles.
        // - Increments fsm_cycle each clock (elapsed time counter).
        // - When last_data_o is asserted by the PSUM FSM (all results sent
        //   or passed to the iact buffer):
        //   * Transitions to GET_PARAMETERS (the host will send the next layer).
        //   * Clears send_data_reg, fsm_cycle, iact_channels_counter.
        // -------------------------------------------------------------------
        WAIT_FOR_RESULTS: begin
          if (fsm_psum_current_state == WAIT_FOR_SENDING_RESULTS) begin
            choose_iact_buffer       <= choose_iact_buffer_output;
          end
          if (single_iteration & (single_iteration2 == 0)) begin
            iact_channels_counter <= iact_channels_counter + 1;
            if (iact_channels_counter == iact_channel_max_cycles - 1) begin
              iact_channels_counter <= 0;
            end
          end
          fsm_cycle <= fsm_cycle + 1;
          if (last_data_o & enable_dma_o & ready_dma_i) begin
            fsm_last_state        <= WAIT_FOR_RESULTS;
            fsm_current_state     <= GET_PARAMETERS;
            send_data_reg         <= 0;
            fsm_cycle             <= 0;
            iact_channels_counter <= 0;
          end
        end

        // -------------------------------------------------------------------
        // MAXPOOLING_READ
        // Implements a 2×2 max-pool pass directly over the iact buffer.
        // Each "pixel group" is compared through a 3-stage pipelined tree:
        //   stage_1[0..7] -> stage_2[0..3] -> stage_3[0..1] -> pooling_regs
        //
        // Entered with fsm_cycle = 2 (pre-loaded by GET_OFFSET).
        //
        // Per 5-cycle loop (one pixel group at position iact_converter_cycles):
        // - Cycles 2-4: advance select_ram_counter (0..1) and
        //   (0..iact_size_x-1).
        // - Feeds pooling_stage_1[0..7] from iact_buffer_data_r using a
        //   2×2 spatial window: word × line indexing with stride iact_size_x.
        // - Pipeline stages 1-3 execute every cycle (registered):
        //     stage_2[a] = max(stage_1[2a], stage_1[2a+1])
        //     stage_3[a] = max(stage_2[2a], stage_2[2a+1])
        // - At cycle 5 (fsm_cycle == 5): compare stage_3[0] and stage_3[1]
        //   against pooling_regs[iact_converter_cycles+b] (running max update).
        //   Also advance iact_converter_cycles by stride_x.
        // - Address window update (cycles >= 3, select_ram_counter == 0 or 1):
        //   shifts buffer_addr_upper_limit and individual cell addresses.
        //   Saved in buffer_addr_temp_reg for MAXPOOLING_SEND.
        // - Exit: when iact_converter_cycles reaches 32-2 (all positions scanned),
        //   resets counters and transitions to MAXPOOLING_SEND.
        // -------------------------------------------------------------------
        MAXPOOLING_READ: begin
          //Counting and setting inputs for reading values

          //Ending Condition
          if (iact_converter_cycles == (psum_size_x * 2)) begin
            fsm_cycle               <= 0;
            iact_converter_cycles   <= 0;
            fsm_last_state          <= MAXPOOLING_READ;
            fsm_current_state       <= MAXPOOLING_SEND;
            buffer_addr_temp_reg    <= iact_buffer_addr_reg[0];
            for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
              iact_buffer_addr_reg[a] <= buffer_addr_temp_reg;
            end
            for (a = 0; a < 4; a = a + 1) begin
                pooling_buffer_new[a] <= -128;
            end
          end else begin
            fsm_cycle <= fsm_cycle + 1;
            if (fsm_cycle == 5) begin
              fsm_cycle             <= fsm_cycle;
              iact_converter_cycles <= iact_converter_cycles + 1;
              pooling_buffer_enable <= 1;
            end
            iact_buffer_data_pool_temp <= iact_buffer_data_pool_temp >> IACT_RAM_CELLS_WORD_BITWIDTH;
            select_ram_counter         <= select_ram_counter - 1;
            if (select_ram_counter == 0) begin
              select_ram_counter         <= IACT_RAM_CELLS - 1;
              iact_buffer_data_pool_temp <= iact_buffer_data_r;
              for (a = 0; a < IACT_RAM_CELLS; a = a + 1) begin
                iact_buffer_addr_reg[a] <= iact_buffer_addr_reg[a] + 1;
              end
            end
            for (word = 0; word < 2; word = word + 1) begin
              for (a = 0; a < 4; a = a + 1) begin
                pooling_stage_1[(2*a)+word] <= iact_buffer_data_pool_temp[DATA_IACT_BITWIDTH*(a+(word*4))+:DATA_IACT_BITWIDTH];
              end
            end
            if (pooling_mode == 0) begin 
              for (a = 0; a < 4; a = a + 1) begin
                if (pooling_stage_1[2*a] >= pooling_stage_1[2*a+1]) begin
                  pooling_stage_2[a] <= pooling_stage_1[2*a];
                end else begin
                  pooling_stage_2[a] <= pooling_stage_1[2*a+1];
                end
                if (pooling_stage_2[a] > pooling_buffer_old[a]) begin
                  pooling_buffer_new[a] <= pooling_stage_2[a];
                end else begin
                  pooling_buffer_new[a] <= pooling_buffer_old[a];
                end
              end
            end else begin
              if (AVERAGE_POOLING == 1) begin
                for (a = 0; a < 4; a = a + 1) begin
                  pooling_stage_2[a] <= pooling_stage_1[2*a] + pooling_stage_1[2*a+1];
                end
                for (a = 0; a < 2; a = a + 1) begin
                  pooling_stage_3[a] <= pooling_stage_2[2*a] >= pooling_stage_2[2*a+1];
                end
                for (a = 0; a < 32; a = a + 1) begin
                  for (b = 0; b < 2; b = b + 1) begin
                    if (a == iact_converter_cycles + b & (fsm_cycle == 5)) begin
                      pooling_regs[a] <= pooling_regs[a] + pooling_stage_3[b];
                    end
                  end
                end
              end
            end
          end
        end

        // -------------------------------------------------------------------
        // MAXPOOLING_SEND
        // Writes the 32 max-pooled results from pooling_regs[] back into
        // the iact buffer so subsequent layers see the pooled output.
        //
        // - Sets all cell addresses to finished_cycles_iact / 8 (word offset
        //   for the current pass within the output address space).
        // - Cycles 0-3 (fsm_cycle <= 3): for each cycle c,
        //     enables the write for cell (c + finished_cycles_iact*4) % IACT_RAM_CELLS.
        //   (4 cells written per pass, IACT_WORDS_IN_RAM bytes per cell.)
        // - After all 32/IACT_WORDS_IN_RAM cycles: clears fsm_cycle,
        //   resets all pooling_regs to -128 (minimum, ready for next pass).
        // - If finished_cycles_iact == needed_cycles - 1 (last pass):
        //   * Resets finished_cycles_iact.
        //   * Clears iact_channels_counter.
        //   * Transitions to GET_PARAMETERS.
        // - Otherwise:
        //   * Increments finished_cycles_iact.
        //   * Restores iact_buffer_addr_reg from buffer_addr_temp_reg
        //     (saved in MAXPOOLING_READ for the next 2×2 window).
        //   * Loops back to MAXPOOLING_READ.
        // -------------------------------------------------------------------
        MAXPOOLING_SEND: begin
          fsm_cycle <= fsm_cycle + 1;
          pooling_buffer_enable   <= 1;
          for (a = 0; a < IACT_RAM_CELLS; a = a + 1) begin
            iact_buffer_en_w[a] <= 0;
          end
          for (a = 0; a < 4; a = a + 1) begin
            iact_buffer_data_w[((IACT_RAM_CELLS-1)*IACT_WORDS_IN_RAM*DATA_IACT_BITWIDTH)+((DATA_IACT_BITWIDTH*IACT_WORDS_IN_RAM)/2)+(8*a)+:8] <= pooling_buffer_old_q[a];
          end
          iact_buffer_data_w[((IACT_RAM_CELLS-1)*IACT_WORDS_IN_RAM*DATA_IACT_BITWIDTH)+(DATA_IACT_BITWIDTH*IACT_WORDS_IN_RAM/2)-1:0] <= iact_buffer_data_w >> 32;
          
          select_ram_counter2 <= select_ram_counter2 + 1;
          if (select_ram_counter2 == 8 - 1) begin
            select_ram_counter2 <= 0;
            for (a = 0; a < IACT_RAM_CELLS; a = a + 1) begin
              iact_buffer_en_w[a] <= 1;
              iact_buffer_addr_reg[a] <= iact_buffer_addr_reg[a] + 1;
            end
          end
          if (fsm_cycle >= psum_size_x - 1) begin
            if (finished_cycles_iact == needed_cycles-1) begin
              if (select_ram_counter2 == 8 - 1) begin
                fsm_cycle             <= 0;
                finished_cycles_iact  <= 0;
                fsm_last_state        <= MAXPOOLING_SEND;
                fsm_current_state     <= GET_PARAMETERS;
                pooling_buffer_enable <= 0;
                set_pointer_start     <= 1;
              end
            end else begin
              fsm_cycle            <= 0;
              finished_cycles_iact <= finished_cycles_iact + 1;
              fsm_cycle            <= 0;
              fsm_last_state       <= MAXPOOLING_SEND;
              fsm_current_state    <= MAXPOOLING_WAIT;
            end
          end
        end
        MAXPOOLING_WAIT: begin
          fsm_cycle            <= fsm_cycle + 1;
          for (a = 0; a < IACT_RAM_CELLS; a = a + 1) begin
            iact_buffer_en_w[a] <= 0;
          end
          if (fsm_cycle == 0) begin
            buffer_addr_temp_reg <= iact_buffer_addr_reg[0];
            for (a = 0; a < IACT_RAM_CELLS; a=a+1) begin
              iact_buffer_addr_reg[a] <= buffer_addr_temp_reg;
            end
          end
          if (fsm_cycle == 2) begin
            fsm_cycle         <= 5;
            fsm_last_state    <= MAXPOOLING_WAIT;
            fsm_current_state <= MAXPOOLING_READ;
          end
        end
        default: begin
        end
      endcase
    end
  end
  reg [7:0] storage_cycles;     
  wire [ (8*QUANT_AMOUNT)-1:0] quant_offset_flat;
  wire [ (7*QUANT_AMOUNT)-1:0] quant_exp_flat;
  wire [(25*QUANT_AMOUNT)-1:0] quant_mant_flat;
  wire [  (8*TRANS_WORDS)-1:0] quantized_value_flat;

  genvar quant_flat_gen;
  for (quant_flat_gen = 0; quant_flat_gen < QUANT_AMOUNT; quant_flat_gen=quant_flat_gen+1) begin : gen_flat_quant_wires
    assign quant_offset_flat[8*quant_flat_gen+:8] = quant_offset[quant_flat_gen];
    assign quant_exp_flat[7*quant_flat_gen+:7] = quant_exp[quant_flat_gen];
    assign quant_mant_flat[25*quant_flat_gen+:25] = quant_mant[quant_flat_gen];
  end
  for (quant_flat_gen = 0; quant_flat_gen < TRANS_WORDS; quant_flat_gen=quant_flat_gen+1) begin : gen_trans_quant_wires
    assign quantized_value_reg[quant_flat_gen] = quantized_value_flat[8*quant_flat_gen+:8];
  end
  psum_pipeline #(
      .CLUSTER_ROWS(CLUSTER_ROWS),
      .CLUSTER_COLUMNS(CLUSTER_COLUMNS),
      .NUM_GLB_IACT(NUM_GLB_IACT),
      .NUM_GLB_PSUM(NUM_GLB_PSUM),
      .NUM_GLB_WGHT(NUM_GLB_WGHT),
      .CLUSTERS(CLUSTERS),
      .PARALLEL_MACS(PARALLEL_MACS),
      .TRANS_WORDS(TRANS_WORDS),
      .TRANS_BITWIDTH_PSUM(TRANS_BITWIDTH_PSUM),
      .BUFFER_WIDTH(BUFFER_WIDTH_PSUM),
      .DMA_BITWIDTH(DMA_BITWIDTH),
      .ROUTER_MODES_PSUM(ROUTER_MODES_PSUM),
      .QUANT_AMOUNT(QUANT_AMOUNT),
      .DATA_PSUM_BITWIDTH(DATA_PSUM_BITWIDTH),
      .PSUM_PER_PE(PSUM_PER_PE),
      .PSUM_CYCLES_ONE_WORD_ALL_CELLS(PSUM_CYCLES_ONE_WORD_ALL_CELLS),
      .PSUM_OUTPUT_WORDS(PSUM_OUTPUT_WORDS)
  ) psum_pipeline_inst (
      .clk_i(clk_i),
      .rst_n(rst_n),
      .data_dma_i_reg(data_dma_i_reg),
      .enable_dma_i_reg(enable_dma_i_reg),
      .ready_dma_i(ready_dma_i),
      .psum_ready_o_reg(psum_ready_o_reg),
      .psum_data_o_w(psum_data_o_w),
      .psum_enable_o(psum_enable_o),
      .router_mode_psum(router_mode_psum),
      .wght_enable_i_reg(wght_enable_i_reg),
      .iact_enable_i_oep_w(iact_enable_i_oep_w),
      .compute_reg(compute_reg),
      .status_reg_enable_reg(status_reg_enable_reg),
      .fsm_current_state(fsm_current_state),
      .psum_size_x(psum_size_x),
      .psum_size_y(psum_size_y),
      .iact_x_line_repetitions(iact_x_line_repetitions),
      .kernels_per_calc(kernels_per_calc),
      .needed_wght_cycles(needed_wght_cycles),
      .needed_y_cls_reg(needed_y_cls_reg),
      .needed_psum_storage_cycles_reg(needed_psum_storage_cycles_reg),
      .iact_channel_max_cycles(iact_channel_max_cycles),
      .iact_channels_per_pe_next_layer(iact_channels_per_pe_next_layer),
      .filters(filters),
      .psum_x_with_add_up(psum_x_with_add_up),
      .psum_x_all_cluster(psum_x_all_cluster),
      .iteration_for_kernels(iteration_for_kernels),
      .needed_cycles(needed_cycles),
      .trans_cycles_psum(trans_cycles_psum),
      .iact_size_y(iact_size_y),
      .iact_size_x(iact_size_x),
      .fully_connected_layer(fully_connected_layer),
      .store_in_psum(store_in_psum),
      .send_data_out(send_data_out),
      .quant_offset_flat(quant_offset_flat),
      .quant_exp_flat(quant_exp_flat),
      .quant_mant_flat(quant_mant_flat),
      .fsm_psum_limit(fsm_psum_limit),
      .psum_buffer_en_r(psum_buffer_en_r),
      .psum_buffer_en_w(psum_buffer_en_w),
      .psum_buffer_addr(psum_buffer_addr),
      .psum_buffer_data_w(psum_buffer_data_w),
      .psum_buffer_data_r(psum_buffer_data_r),
      .psum_data_i_reg(psum_data_i_reg),
      .psum_enable_i_reg(psum_enable_i_reg),
      .psum_ready_i_reg(psum_ready_i_reg),
      .quantized_value_flat(quantized_value_flat),
      .data_dma_o(data_dma_o),
      .enable_dma_o(enable_dma_o),
      .last_data_o(last_data_o),
      .fsm_psum_last_state(fsm_psum_last_state),
      .fsm_psum_current_state(fsm_psum_current_state),
      .fsm_psum_cycle(fsm_psum_cycle),
      .psum_transmitted(psum_transmitted),
      .psum_router_set_reg(psum_router_set_reg),
      .start_new_cycle(start_new_cycle),
      .psum_cnt(psum_cnt),
      .current_filter(current_filter),
      .output_words(psum_output_words),
      .psum_cycle_loop_limit_0(psum_pagu_loop_limit_0),
      .psum_cycle_loop_limit_1(psum_pagu_loop_limit_1),
      .psum_cycle_loop_limit_2(psum_pagu_loop_limit_2),
      .psum_cycle_loop_limit_3(psum_pagu_loop_limit_3),
      .psum_cycle_loop_limit_4(psum_pagu_loop_limit_4),
      .psum_cycle_addr_inc_0(psum_pagu_addr_inc_0),
      .psum_cycle_addr_inc_1(psum_pagu_addr_inc_1),
      .psum_cycle_addr_inc_2(psum_pagu_addr_inc_2),
      .psum_cycle_addr_inc_3(psum_pagu_addr_inc_3),
      .psum_cycle_addr_inc_4(psum_pagu_addr_inc_4)
  );

  genvar k_gen;
  wire [CHANNELS_PER_WORD*DATA_IACT_BITWIDTH-1:0]pooling_buffer_new_flat;
  wire [CHANNELS_PER_WORD*DATA_IACT_BITWIDTH-1:0]pooling_buffer_old_flat;
  wire [4:0]limit_i;
  assign limit_i = psum_size_x-1;
  for (k_gen = 0; k_gen < CHANNELS_PER_WORD; k_gen=k_gen+1) begin : gen_ring_buffer_wires
    assign pooling_buffer_old[k_gen] = pooling_buffer_old_flat[(k_gen*DATA_IACT_BITWIDTH)+:DATA_IACT_BITWIDTH];
    assign pooling_buffer_new_flat[(k_gen*DATA_IACT_BITWIDTH)+:DATA_IACT_BITWIDTH] = pooling_buffer_new[k_gen];
  end
  
  ring_buffer_pipelined #(
      .DATA_WIDTH(DATA_IACT_BITWIDTH*CHANNELS_PER_WORD),
      .ADDR_WIDTH(5)
  ) ring_buffer_pipelined_inst (
      .clk_i(clk_i),
      .rst_n(rst_n),
      .set_pointer_start_i(set_pointer_start),
      .limit_i(limit_i),
      .ready_i(pooling_buffer_enable),
      .data_i(pooling_buffer_new_flat),
      .data_o(pooling_buffer_old_flat),
      .valid_o()
  );

  // -----------------------------------------------------------------------
  // Generate Block: gen_RAM_wires
  // Connects the registered per-cell control signals (buffer_*_reg arrays,
  // driven by the main FSM and the iact_stream_constructor enable logic) to
  // the wire arrays that are fed into the BUFFER_A RAM_SP instances below.
  //
  // Also collapses the wide iact_buffer_data_r packed read bus (2× width,
  // providing both buffer halves) down to the active half: iact_buffer_data_r
  // always exposes the lower half of iact_buffer_data_r (bits [IACT_RAM_CELLS*W-1:0]).
  // The upper half (bits [2*IACT_RAM_CELLS*W-1:IACT_RAM_CELLS*W]) is unused because
  // choose_iact_buffer selects the active half at the RAM address level.
  // -----------------------------------------------------------------------
  for (k_gen = 0; k_gen < IACT_RAM_CELLS; k_gen=k_gen+1) begin : gen_RAM_wires
    assign iact_buffer_addr[k_gen] = BRANCHES != 1 ? {choose_iact_buffer,iact_buffer_addr_reg[k_gen]} : iact_buffer_addr_reg[k_gen];
  end
  // -----------------------------------------------------------------------
  // Generate Block: UNPACKED_TRACES  (conditional, UNPACKED_TRACES_ENABLED)
  // Creates named per-element wire aliases for every array-typed register
  // that simulators cannot easily display as individual signals.
  // Only elaborated when the UNPACKED_TRACES_ENABLED parameter is 1 (default).
  // This adds zero hardware; it is purely for waveform visibility in
  // Icarus Verilog / GTKWave / ModelSim.
  //
  // Sub-loops created:
  //   pooling_stage_1_traces[0..7]   - 8 wires, one per pooling_stage_1 element.
  //   pooling_stage_2_traces[0..3]   - 4 wires for pooling_stage_2.
  //   pooling_stage_3_traces[0..1]   - 2 wires for pooling_stage_3.
  //   pooling_stage_traces[0..31]    - 32 wires for pooling_regs (final accumulators).
  //   quant_stage_traces[0..7]       - 8 wires for quantized_value_reg outputs.
  // -----------------------------------------------------------------------
  generate
    genvar i_trace;
    if (UNPACKED_TRACES_ENABLED) begin  : UNPACKED_TRACES
      for (i_trace = 0; i_trace < 8; i_trace = i_trace + 1) begin : pooling_stage_1_traces
        wire [7:0] pooling_stage_1_trace; 
        assign pooling_stage_1_trace = pooling_stage_1[i_trace]; 
      end
      for (i_trace = 0; i_trace < 4; i_trace = i_trace + 1) begin : pooling_stage_2_traces
        wire [7:0] pooling_stage_2_trace; 
        assign pooling_stage_2_trace = pooling_stage_2[i_trace]; 
      end
      for (i_trace = 0; i_trace < 2; i_trace = i_trace + 1) begin : pooling_stage_3_traces
        wire [7:0] pooling_stage_3_trace; 
        assign pooling_stage_3_trace = pooling_stage_3[i_trace]; 
      end
      for (i_trace = 0; i_trace < 32; i_trace = i_trace + 1) begin : pooling_stage_traces
        wire [7:0] pooling_stage_out_trace; 
        assign pooling_stage_out_trace = pooling_regs[i_trace]; 
      end
      for (i_trace = 0; i_trace < TRANS_WORDS; i_trace = i_trace + 1) begin : quant_stage_traces
        wire [7:0] quantized_out_trace; 
        assign quantized_out_trace = quantized_value_reg[i_trace]; 
      end
      for (i_trace = 0; i_trace < 8; i_trace = i_trace + 1) begin : offset_stage_traces
        wire [7:0] offset_out_trace; 
        assign offset_out_trace = quant_offset[i_trace]; 
      end
      for (i_trace = 0; i_trace < 8; i_trace = i_trace + 1) begin : exp_stage_traces
        wire [6:0] exp_out_trace; 
        assign exp_out_trace = quant_exp[i_trace]; 
      end
      for (i_trace = 0; i_trace < 8; i_trace = i_trace + 1) begin : mant_stage_traces
        wire [24:0] mant_out_trace; 
        assign mant_out_trace = quant_mant[i_trace]; 
      end
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Generate Block: Main instantiation block
  // Contains all sub-module instances: BUFFER_A, IACT_CONVERTER_X/Y,
  // wght_buffer_, PSUM_RAM_X/Y/GLB, dma_storage, OpenEye_Parallel,
  // and the cross-wiring loops that connect converter outputs to the
  // accelerator core inputs.
  // -----------------------------------------------------------------------
  generate
    genvar i_gen, j_gen, g_gen;

    // -------------------------------------------------------------------
    // BUFFER_A: Iact Double-Buffer (32 × RAM_SP)
    // Instantiates IACT_RAM_CELLS single-port SRAMs, each 64 bits wide
    // and BUFFER_WIDTH deep.  The MSB of addr_i is choose_iact_buffer,
    // implementing ping/pong double-buffering: the host writes into one half
    // while the converters read from the other.
    // Pipelined=1 means the read data appears one cycle after rd_en_i.
    // -------------------------------------------------------------------
    for (j_gen = 0; j_gen < IACT_RAM_CELLS; j_gen=j_gen+1) begin : BUFFER_A
        RAM_SP #(
            .DataWidth(IACT_RAM_CELLS_WORD_BITWIDTH),
            .AddrWidth(BUFFER_WIDTH),
            .Pipelined(1)
        ) iact_layer_buffer (
            .clk_i(clk_i),
            .rd_en_i(iact_buffer_en_r[j_gen] & !iact_buffer_en_w[j_gen]),
            .wr_en_i(iact_buffer_en_w[j_gen]),
            .addr_i(iact_buffer_addr[j_gen]),
            .data_i(iact_buffer_data_w[j_gen*IACT_RAM_CELLS_WORD_BITWIDTH+:IACT_RAM_CELLS_WORD_BITWIDTH]),
            .data_o(iact_buffer_data_r[j_gen*IACT_RAM_CELLS_WORD_BITWIDTH+:IACT_RAM_CELLS_WORD_BITWIDTH])
        );
    end

    // -------------------------------------------------------------------
    // IACT_CONVERTER_X / IACT_CONVERTER_Y: Iact Stream Constructors
    // Instantiates CLUSTER_COLUMNS × CLUSTER_ROWS iact_stream_constructor
    // modules (one per cluster position).  Each instance reads raw pixel
    // data from the shared BUFFER_A cells, applies zero-run-length encoding,
    // and outputs a sparse iact stream (iact_data_w / iact_enable_w /
    // iact_choose_w) for the corresponding PE cluster in OpenEye_Parallel.
    //
    // Local wires per instance:
    //   iact_data_w   - encoded iact data bus (TRANS_BITWIDTH_IACT × NUM_GLB_IACT).
    //   iact_choose_w - per-PE source selector ($clog2(NUM_GLB_IACT+1) × PES).
    //   iact_ready_w  - back-pressure from the PE cluster (NUM_GLB_IACT bits);
    //                   all-ones means the cluster can accept the next word.
    //   iact_enable_w - valid flags from the converter (NUM_GLB_IACT bits).
    //
    // Key parameters forwarded:
    //   params                  - 36-bit spatial config word from Process 3.
    //   enable_config           - one-cycle pulse: latch params.
    //   enable_store            - one-cycle pulse: commit encoded word to BUFFER_A.
    //   enable_converter        - enable encoding for this cycle.
    //   needed_y_cls_i          - active cluster rows.
    //   needed_iact_channel_cycles_i - channel batches per pass.
    //   iact_size_x/y_i         - feature map dimensions.
    //   fully_connected_i       - selects FC vs conv addressing mode.
    // -------------------------------------------------------------------
    for (i_gen = 0; i_gen < CLUSTER_COLUMNS; i_gen=i_gen+1) begin : IACT_CONVERTER_X
      for (j_gen = 0; j_gen < CLUSTER_ROWS; j_gen=j_gen+1) begin : IACT_CONVERTER_Y
        wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] iact_data_w;
        wire [      $clog2(NUM_GLB_IACT+1)*PES-1:0] iact_choose_w;
        wire [                    NUM_GLB_IACT-1:0] iact_ready_w;
        wire [                    NUM_GLB_IACT-1:0] iact_enable_w;
        iact_stream_constructor #(
            .CLUSTER_COLUMNS   (CLUSTER_COLUMNS),
            .CLUSTER_ROWS      (CLUSTER_ROWS),
            .SPARSITY_EN       (SPARSITY_EN),
            .NUM_GLB_IACT      (NUM_GLB_IACT),
            .PE_X              (NUM_GLB_PSUM),
            .PE_Y              (NUM_GLB_WGHT),
            .DATA_IACT_BITWIDTH(DATA_IACT_BITWIDTH),
            .DATA_IACT_OVERHEAD(DATA_IACT_OVERHEAD),
            .RAM_CELLS         (IACT_RAM_CELLS),
            .WORD_BITWIDTH     (TRANS_BITWIDTH_IACT * NUM_GLB_IACT),
            .ADDRWIDTH         (13)
        ) iact_stream_constructor (
            .clk_i                       (clk_i),
            .rst_ni                      (rst_n),
            .storage_i                   (iact_input_window_q),
            .reset_cycle_i               (reset_cycle),
            .params                      (iact_converter_params_reg[i_gen][j_gen]),
            .enable_config               (iact_converter_en_cfg_reg[i_gen][j_gen]),
            .enable_store                (iact_converter_en_store_reg[i_gen][j_gen]),
            .enable_converter            (iact_converter_en_enc_reg[i_gen][j_gen]),
            .ready_o                     (iact_converter_ready_w[i_gen][j_gen]),
            .iact_ready_i                (iact_ready_w),
            .iact_data_o                 (iact_data_w),
            .iact_enable_o               (iact_enable_w),
            .iact_choose_o               (iact_choose_w),
            .needed_y_cls_i              (needed_y_cls_reg),
            .needed_iact_channel_cycles_i(iact_channel_max_cycles),
            .fc_size_i                   (fc_size_reg),
            .iact_size_x_i               (iact_size_x),
            .iact_size_y_i               (iact_size_y),
            .iact_channels_per_pe_i      (iact_channels_per_pe),
            .channel_div_trans           (channel_div_trans),
            .x_lines_i                   (iact_x_line_repetitions),
            .needed_wght_cycles_i        (needed_wght_cycles),
            .needed_iact_router_cycles_i (needed_iact_cycles_reg),
            .wght_size_x_i               (kernel_size_x),
            .wght_size_y_i               (kernel_size_y),
            .stride_x_i                  (stride_x),
            .stride_y_i                  (stride_y),
            .iact_x_per_cluster_i        (iact_x_per_cluster),
            .y_lines_per_calc            (y_lines_per_calc),
            .fully_connected_i           (fully_connected_layer),
            .needed_iact_buffer_words_i  (iact_buffer_words_per_write),
            .iact_words_per_compute(iact_words_per_compute),
            .rd_loop_limit_0             (iact_read_limit_0),
            .rd_loop_limit_1             (iact_read_limit_1),
            .rd_loop_limit_2             (iact_read_limit_2),
            .rd_loop_limit_3             (iact_read_limit_3),
            .rd_loop_limit_4             (iact_read_limit_4),
            .rd_addr_inc_0               (iact_read_inc_0),
            .rd_addr_inc_1               (iact_read_inc_1),
            .rd_addr_inc_2               (iact_read_inc_2),
            .rd_addr_inc_3               (iact_read_inc_3),
            .rd_addr_inc_4               (iact_read_inc_4),
            .wr_loop_limit_0             (iact_write_limit_0),
            .wr_loop_limit_1             (iact_write_limit_1),
            .wr_loop_limit_2             (iact_write_limit_2),
            .wr_addr_inc_0               (iact_write_inc_0),
            .wr_addr_inc_1               (iact_write_inc_1),
            .wr_addr_inc_2               (iact_write_inc_2),
            .padding_x                   (padding_x),
            .padding_y                   (padding_y)
        );
      end
    end

    // -------------------------------------------------------------------
    // wght_buffer_: Weight Staging Buffer (single RAM_SP)
    // Stores the full weight tensor for the current layer.  Depth is
    // BUFFER_WIDTH+1 address bits; width is TRANS_BITWIDTH_WGHT × CLUSTERS
    // × NUM_GLB_WGHT bits (all clusters, all wght GLBs in one wide word).
    // Written sequentially by Process 6 (GET_WGHT state) using
    // wght_buffer_wr_addr; read sequentially by Process 5 using
    // wght_buffer_rd_addr during the send phase.
    // The read address is held or rewound by Process 5 for weight reuse
    // across multiple iact batches (needed_wght_cycles iterations).
    // Read and write are mutually exclusive (rd_en_i gated by !wr_en_i).
    // -------------------------------------------------------------------
    RAM_SP #(
        .DataWidth(TRANS_BITWIDTH_WGHT * CLUSTERS * NUM_GLB_WGHT),
        .AddrWidth(BUFFER_WIDTH_WGHT),
        .Pipelined(1)
    ) wght_buffer (
        .clk_i  (clk_i),
        .rd_en_i(wght_buffer_en_r & !wght_buffer_en_w),
        .wr_en_i(wght_buffer_en_w),
        .addr_i (wght_buffer_wr_addr | wght_buffer_rd_addr),
        .data_i (wght_buffer_data_w[TRANS_BITWIDTH_WGHT * CLUSTERS * NUM_GLB_WGHT-1:0]),
        .data_o (wght_buffer_data_r)
    );

    // -------------------------------------------------------------------
    // PSUM_RAM_X / PSUM_RAM_Y / PSUM_RAM_GLB: Psum Staging Buffers
    // Instantiates CLUSTER_COLUMNS × CLUSTER_ROWS × ((NUM_GLB_PSUM+1)/2) RAM_SP
    // cells.  Each cell is TRANS_BITWIDTH_PSUM×2 bits wide (holds two psum
    // values per word) and BUFFER_WIDTH address bits deep.
    //
    // Usage:
    //   GET_BIAS:          Written by Process 6 with initial bias values.
    //   CALCULATE_PSUM:    Read by PSUM FSM; data fed into psum_data_i_reg
    //                      → OpenEye_Parallel psum_data_i port.
    //   PSUM_GET_RESULTS:  Written by PSUM FSM with psum_data_o_w results.
    //   PSUM_SEND_RESULTS: Read by PSUM FSM for DMA output.
    //   SEND_PSUM_TO_IACT: Read by PSUM FSM for quantization + iact writeback.
    //
    // Address arbitration: psum_buffer_addr_array[cc][cr][g] holds the
    // current address for each cell; updated by the PSUM FSM.
    // -------------------------------------------------------------------
    for (i_gen = 0; i_gen < CLUSTER_COLUMNS; i_gen=i_gen+1) begin : PSUM_RAM_X
      for (j_gen = 0; j_gen < CLUSTER_ROWS; j_gen=j_gen+1) begin : PSUM_RAM_Y
        for (g_gen = 0; g_gen < (NUM_GLB_PSUM+1)/2; g_gen=g_gen+1) begin : PSUM_RAM_GLB
          RAM_SP #(
              .DataWidth(PSUM_BUFFER_WIDTH),
              .AddrWidth(BUFFER_WIDTH_PSUM),
              .Pipelined(1)
          ) psum_buffer (
              .clk_i  (clk_i),
              .rd_en_i(psum_buffer_en_r[i_gen*CLUSTER_ROWS*(NUM_GLB_PSUM+1)/2+j_gen*(NUM_GLB_PSUM+1)/2+g_gen] & !psum_buffer_en_w[i_gen*CLUSTER_ROWS*(NUM_GLB_PSUM+1)/2+j_gen*(NUM_GLB_PSUM+1)/2+g_gen]),
              .wr_en_i(psum_buffer_en_w[i_gen*CLUSTER_ROWS*(NUM_GLB_PSUM+1)/2+j_gen*(NUM_GLB_PSUM+1)/2+g_gen]),
              .addr_i (psum_buffer_addr[i_gen*BUFFER_WIDTH_PSUM*CLUSTER_ROWS*(NUM_GLB_PSUM+1)/2+j_gen*BUFFER_WIDTH_PSUM*(NUM_GLB_PSUM+1)/2+g_gen*BUFFER_WIDTH_PSUM+:BUFFER_WIDTH_PSUM]),
              // The data buses are declared in TRANS_BITWIDTH_PSUM units
              // (TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM), so the per-column
              // and per-row strides stay in those units. Only the slot inside
              // one cluster is PSUM_BUFFER_WIDTH wide - using it for the
              // strides too doubles them for NUM_GLB_PSUM>1 and runs off the
              // end of the bus.
              .data_i (psum_buffer_data_w[i_gen*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+j_gen*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+g_gen*PSUM_BUFFER_WIDTH+:PSUM_BUFFER_WIDTH]),
              .data_o (psum_buffer_data_r[i_gen*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+j_gen*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+g_gen*PSUM_BUFFER_WIDTH+:PSUM_BUFFER_WIDTH])
          );
        end
      end
    end
    /*assign debug_psum_re     = psum_buffer_en_r[0] & !psum_buffer_en_w[0];
    assign debug_psum_we     = psum_buffer_en_w[0];
    assign debug_psum_addr   = psum_buffer_addr[0+:BUFFER_WIDTH];
    assign debug_psum_data_i = psum_buffer_data_w[0+:TRANS_BITWIDTH_PSUM*PARALLEL_MACS];
    assign debug_psum_data_o = psum_buffer_data_r[0+:TRANS_BITWIDTH_PSUM*PARALLEL_MACS];

    assign debug_skip_iact_o = skipIact_reg;
    assign debug_data_dma_stream_o = data_dma_i_reg;
    assign debug_enable_dma_stream_o = enable_dma_i_reg;
    assign debug_fsm_cycle_o = fsm_cycle[3:0];*/

    // -------------------------------------------------------------------
    // dma_storage: Register-Map Decoder Instance
    // Auto-generated module (from hdl/config/regmap.yaml) that decodes the
    // first 4 DMA configuration words into ~30 named layer-parameter outputs.
    // Connected via:
    //   write_en   - asserted by Process 6 for the first 4 GET_PARAMETERS words.
    //   write_addr - 2-bit address selecting which of the 4 config registers
    //                to write (decoded from fsm_cycle).
    //   dma_data_i - the 64-bit DMA word to decode.
    // All output ports (wght_cycles_reg, iact_size_x, kernels_per_calc, etc.)
    // are combinational functions of the stored register contents.
    // -------------------------------------------------------------------
    dma_storage  #(

    ) dma_storage (
        .clk_i(clk_i),
        .rst_ni(rst_n),
        .write_en(write_dma_en),
        .dma_data_i(dma_data_i),
        .wght_cycles_reg(wght_cycles_reg),
        .stride_x(stride_x_dma),
        .stride_y(stride_y_dma),
        .skipIact_reg(skipIact_reg),
        .skipWght_reg(skipWght_reg),
        .skipPsum_reg(skipPsum_reg),
        .psum_q(psum_q),
        .kernel_per_pe_cluster_reg(kernel_per_pe_cluster_reg),
        .kernel_size_x(kernel_size_x),
        .kernel_size_y(kernel_size_y),
        .x_lines_reg(x_lines_reg),
        .needed_wght_cycles(needed_wght_cycles),
        .needed_cycles(needed_cycles),
        .trans_cycles_iact(trans_cycles_iact),
        .trans_cycles_wght(trans_cycles_wght),
        .trans_cycles_psum(trans_cycles_psum),
        .iact_converter_buffer_addr_max_cycles(iact_converter_buffer_addr_max_cycles),
        .iact_channels_per_pe(iact_channels_per_pe),
        .channel_div_trans(channel_div_trans),
        .fc_size_reg(fc_size_reg),
        .iact_size_x(iact_size_x),
        .iact_size_y(iact_size_y),
        .iact_size_c(iact_size_c),
        .iact_x_per_cluster(iact_x_per_cluster),
        .iact_x_add_up(iact_x_add_up),
        .padding_x(padding_x),
        .padding_y(padding_y),
        .psum_size_x(psum_size_x),
        .psum_size_y(psum_size_y),
        .psum_x_all_cluster(psum_x_all_cluster),
        .iact_needed_cycles(iact_needed_cycles),
        .kernels_per_calc(kernels_per_calc),
        .y_lines_per_calc(y_lines_per_calc),
        .store_in_psum(store_in_psum),
        .max_pooling(max_pooling),
        .fully_connected_layer(fully_connected_layer),
        .choose_iact_buffer_output(choose_iact_buffer_output),
        .choose_iact_buffer_input(choose_iact_buffer_input),
        .iact_channels_per_pe_next_layer(iact_channels_per_pe_next_layer),
        .needed_psum_storage_cycles_reg(needed_psum_storage_cycles_reg),
        .iact_channel_max_cycles(iact_channel_max_cycles),
        .input_activations(input_activations),
        .used_wght_per_PE(wghts_per_pe),
        .filters(filters),
        .needed_x_cls_reg(needed_x_cls_reg),
        .needed_y_cls_reg(needed_y_cls_reg),
        .needed_iact_cycles_reg(needed_iact_cycles_reg),
        .wght_addr_len_reg(wght_addr_len_reg),
        .send_data_out(send_data_out),
        .needed_iact_buffer_words(needed_iact_buffer_words),
        .add_up_reg(add_up),
        .iact_x_line_repetitions_reg(iact_x_line_repetitions),
        .buffer_cycles_for_x_iact(buffer_cycles_for_x_iact),
        .start_param_array(start_param_array),
        .limit_increase(limit_increase),
        .lower_bound(lower_bound),
        .upper_bound(upper_bound),
        .initial_upper_limit(initial_upper_limit),
        .iteration_for_kernels(iteration_for_kernels),
        .fsm_psum_limit(fsm_psum_limit),
        .cluster_per_conv_cycle(cluster_per_conv_cycle),
        .iact_converter_max_cycles(iact_converter_max_cycles),
        .iact_buffer_words_per_write(iact_buffer_words_per_write),
        .iact_words_per_compute(iact_words_per_compute),
        .pooling_mode(pooling_mode),
        .overhang_discrepancy(overhang_discrepancy),
        .psum_output_words(psum_output_words),
        .gemm_mode(gemm_mode),
        .iact_read_limit_0(iact_read_limit_0),
        .iact_read_limit_1(iact_read_limit_1),
        .iact_read_limit_2(iact_read_limit_2),
        .iact_read_limit_3(iact_read_limit_3),
        .iact_read_limit_4(iact_read_limit_4),
        .iact_read_inc_0(iact_read_inc_0),
        .iact_read_inc_1(iact_read_inc_1),
        .iact_read_inc_2(iact_read_inc_2),
        .iact_read_inc_3(iact_read_inc_3),
        .iact_read_inc_4(iact_read_inc_4),
        .iact_write_limit_0(iact_write_limit_0),
        .iact_write_limit_1(iact_write_limit_1),
        .iact_write_limit_2(iact_write_limit_2),
        .iact_write_inc_0(iact_write_inc_0),
        .iact_write_inc_1(iact_write_inc_1),
        .iact_write_inc_2(iact_write_inc_2),
        .pagu_wght_limit(pagu_wght_limit),
        .psum_pagu_loop_limit_0(psum_pagu_loop_limit_0),
        .psum_pagu_loop_limit_1(psum_pagu_loop_limit_1),
        .psum_pagu_loop_limit_2(psum_pagu_loop_limit_2),
        .psum_pagu_loop_limit_3(psum_pagu_loop_limit_3),
        .psum_pagu_loop_limit_4(psum_pagu_loop_limit_4),
        .psum_pagu_addr_inc_0(psum_pagu_addr_inc_0),
        .psum_pagu_addr_inc_1(psum_pagu_addr_inc_1),
        .psum_pagu_addr_inc_2(psum_pagu_addr_inc_2),
        .psum_pagu_addr_inc_3(psum_pagu_addr_inc_3),
        .psum_pagu_addr_inc_4(psum_pagu_addr_inc_4)
    );


    // -------------------------------------------------------------------
    // router_configurator: Dynamic Router Configuration Instance
    // Extracted from Process 7. Manages runtime updates to router_mode_iact,
    // router_mode_wght, and router_mode_psum during computation.
    // -------------------------------------------------------------------
    router_configurator #(
        .CLUSTER_COLUMNS(CLUSTER_COLUMNS),
        .CLUSTER_ROWS   (CLUSTER_ROWS),
        .NUM_GLB_IACT   (NUM_GLB_IACT),
        .NUM_GLB_WGHT   (NUM_GLB_WGHT),
        .NUM_GLB_PSUM   (NUM_GLB_PSUM),
        .CLUSTERS       (CLUSTERS),
        .DMA_BITWIDTH   (DMA_BITWIDTH),
        .ROUTER_MODES_IACT(ROUTER_MODES_IACT),
        .ROUTER_MODES_WGHT(ROUTER_MODES_WGHT),
        .ROUTER_MODES_PSUM(ROUTER_MODES_PSUM),
        .FSM_CEIL_IACT_RTR_CCLS(FSM_CEIL_IACT_RTR_CCLS),
        .FSM_CEIL_WGHT_RTR_CCLS(FSM_CEIL_WGHT_RTR_CCLS),
        .FSM_CEIL_PSUM_RTR_CCLS(FSM_CEIL_PSUM_RTR_CCLS),
        .fsm_psum_rTR_CCLS_C   (fsm_psum_rTR_CCLS_C)
    ) router_configurator_inst (
        .clk_i                        (clk_i),
        .rst_n                        (rst_n),
        .compute_reg_i                (compute_reg),
        .fully_connected_layer_i      (fully_connected_layer),
        .needed_y_cls_reg_i           (needed_y_cls_reg),
        .fsm_current_state_i          (fsm_current_state),
        .enable_dma_i_reg_i           (enable_dma_i_reg),
        .fsm_cycle_i                  (fsm_cycle),
        .data_dma_i_reg_i             (data_dma_i_reg),
        .single_iteration3_i          (single_iteration3),
        .iact_channels_counter_i      (iact_channels_counter),
        .iact_channel_max_cycles_i    (iact_channel_max_cycles),
        .iact_router_counter_i        (iact_router_counter),
        .needed_psum_storage_cycles_i (needed_psum_storage_cycles_reg),
        .reset_cycle_i                (reset_cycle),
        .router_mode_iact_o           (router_mode_iact),
        .router_mode_wght_o           (router_mode_wght),
        .router_mode_psum_o           (router_mode_psum),
        .router_mode_iact_storage_o   (router_mode_iact_storage),
        .storage_cycles_router_o      (storage_cycles_router),
        .first_cycle_o                (first_cycle),
        .psum_choose_i_reg_o          (psum_choose_i_reg),
        .iact_channels_counter_psum_router_o(iact_channels_counter_psum_router)
    );

    // -------------------------------------------------------------------
    // OpenEye_Parallel: Main Compute Core Instance
    // The ASIC-style systolic array accelerator.  This is the heart of
    // the design; all other logic in this file exists to feed it data
    // and extract results.
    //
    // Key interface groups:
    //   iact_*   - Iact data from iact_stream_constructors (via oep_w wires).
    //              iact_ready_o feeds back to all converters to signal acceptance.
    //   wght_*   - Weight data from wght_buffer read port; enable mask from
    //              Process 5; ready back-pressure to Process 5.
    //   psum_*_i - Bias / previous psum input from psum_buffer (PSUM FSM).
    //   psum_*_o - Accumulated result output, written back to psum_buffer_
    //              by the PSUM FSM in PSUM_GET_RESULTS.
    //   compute_i - Single-cycle trigger from Process 5 (compute_reg).
    //   router_mode_*_i - Dynamic routing vectors from router_configurator.
    //   status_reg_enable_i - Enables config reception during GET_PARAMETERS.
    //   compute_mask_i - Bitmask enabling specific PEs for this layer.
    // -------------------------------------------------------------------
    OpenEye_Parallel #(
        .IS_TOPLEVEL(0),
        .SERIAL     (SERIAL),
        .SPARSITY_EN(SPARSITY_EN),

        .DATA_IACT_BITWIDTH(DATA_IACT_BITWIDTH),
        .DATA_PSUM_BITWIDTH(DATA_PSUM_BITWIDTH),
        .DATA_WGHT_BITWIDTH(DATA_WGHT_BITWIDTH),

        .TRANS_BITWIDTH_IACT(TRANS_BITWIDTH_IACT),
        .TRANS_BITWIDTH_WGHT(TRANS_BITWIDTH_WGHT),
        .TRANS_BITWIDTH_PSUM(TRANS_BITWIDTH_PSUM),
        .DATA_IACT_OVERHEAD (DATA_IACT_OVERHEAD),

        .PE_COLUMNS(NUM_GLB_PSUM),
        .PE_ROWS   (NUM_GLB_WGHT),

        .NUM_GLB_IACT(NUM_GLB_IACT),
        .NUM_GLB_WGHT(NUM_GLB_WGHT),
        .NUM_GLB_PSUM(NUM_GLB_PSUM),

        .CLUSTER_COLUMNS(CLUSTER_COLUMNS),
        .CLUSTER_ROWS   (CLUSTER_ROWS),

        .IACT_ADDR_PER_PE(IACT_ADDR_PER_PE),
        .WGHT_ADDR_PER_PE(WGHT_ADDR_PER_PE),

        .IACT_PER_PE(IACT_PER_PE),
        .WGHT_PER_PE(WGHT_PER_PE),
        .PSUM_PER_PE(PSUM_PER_PE),

        .IACT_MEM_ADDR_WORDS(IACT_MEM_ADDR_WORDS),
        .PSUM_MEM_ADDR_WORDS(PSUM_MEM_ADDR_WORDS),

        .ROUTER_MODES_IACT(ROUTER_MODES_IACT),
        .ROUTER_MODES_WGHT(ROUTER_MODES_WGHT),
        .ROUTER_MODES_PSUM(ROUTER_MODES_PSUM),

        .PARALLEL_MACS(PARALLEL_MACS)

    ) OpenEye_Parallel (
        //Clock and Reset Ports
        .clk_i    (clk_i),
        .rst_ni   (rst_n),
        .compute_i(compute_reg),

        //Ports for GLBs and PEs
        .iact_choose_i(iact_choose_i_oep_w),
        .iact_data_i  (iact_data_i_oep_w),
        .iact_enable_i(iact_enable_i_oep_w),
        .iact_ready_o (iact_ready_o_oep_w),

        .wght_data_i  (wght_data_i_w),
        .wght_enable_i(wght_enable_i_reg),
        .wght_ready_o (wght_ready_o_reg),

        .psum_choose_i(psum_choose_i_reg),
        .psum_data_i  (psum_buffer_data_r),
        .psum_enable_i(psum_enable_i_reg),
        .psum_ready_o (psum_ready_o_reg),

        .psum_data_o  (psum_data_o_w),
        .psum_enable_o(psum_enable_o),
        .psum_ready_i (psum_ready_i_reg),

        //Ports for Hyperparameters
        .status_reg_enable_i          (status_reg_enable_reg),
        .data_mode_i                  (data_mode_reg),
        .gemm_mode_i                  (gemm_mode),
        // Dense/FC and GEMM layers stream raw (uncompressed) weights;
        // all-zero weight words must then be stored, not skipped.
        .raw_wght_i                   (fully_connected_layer | gemm_mode),
        .fraction_bit_i               (fraction_bit_reg),
        .needed_cycles_i              (needed_cycles),
        .needed_x_cls_i               (needed_x_cls_reg),
        .needed_y_cls_i               (needed_y_cls_reg),
        .needed_iact_cycles_i         (needed_iact_cycles_reg),
        .iact_size_x_i                (iact_size_x),
        .filters_i                    (filters),
        .iact_channels_per_pe_i       (iact_channels_per_pe),
        .wght_addr_len_i              (wght_addr_len_reg),
        .bano_cluster_mode_i          (bano_cluster_mode_reg),
        .af_cluster_mode_i            (af_cluster_mode_reg),
        .pooling_cluster_mode_i       ({NUM_GLB_PSUM{1'd0}}),
        .kernel_per_pe_cluster_i      (kernel_per_pe_cluster_reg[$clog2(NUM_GLB_WGHT)-1:0]),
        .iact_x_line_repetitions_i    (iact_x_line_repetitions[3:0]),
        .kernel_size_y_i              (kernel_size_y),
        .input_activations_i          (input_activations),
        .delay_psum_glb_i             (psum_q),
        .compute_mask_i               (compute_mask_reg_port),
        .router_mode_iact_i           (router_mode_iact),
        .router_mode_wght_i           (router_mode_wght),
        .router_mode_psum_i           (router_mode_psum),
        .needed_psum_storage_cycles_i (needed_psum_storage_cycles_reg),
        .needed_iact_channel_cycles_i (iact_channel_max_cycles),
        .psum_transmitted_i           (psum_transmitted)
    );
    // -------------------------------------------------------------------
    // Iact Converter → OpenEye_Parallel Cross-Wiring
    // Connects the hierarchical per-instance wire arrays declared inside
    // each IACT_CONVERTER_X[cc].IACT_CONVERTER_Y[cr] scope to the flat
    // packed vectors expected by the OpenEye_Parallel port.
    //
    // Three signal groups are wired here:
    //
    // 1. iact_ready_w (converter input, from OpenEye_Parallel output):
    //    Each converter's iact_ready_w bus is asserted when ALL converters'
    //    corresponding GLBs are simultaneously ready (global all-ones check).
    //    Note: the per-converter address assignment (commented out) was the
    //    original approach; the current implementation broadcasts a single
    //    global ready so all converters advance together.
    //
    // 2. iact_enable_i_oep_w (OpenEye_Parallel input, from converter output):
    //    Packed from IACT_CONVERTER_X[cc].IACT_CONVERTER_Y[cr].iact_enable_w
    //    using the flat index: cc*CLUSTER_ROWS*NUM_GLB_IACT + cr*NUM_GLB_IACT + g.
    //
    // 3. iact_data_i_oep_w (OpenEye_Parallel input, from converter output):
    //    Packed from iact_data_w using the corresponding TRANS_BITWIDTH_IACT-
    //    wide slice at the same flat index × TRANS_BITWIDTH_IACT.
    //
    // 4. iact_choose_i_oep_w (OpenEye_Parallel input, from converter output):
    //    Packed from iact_choose_w; each PE gets IACT_CHOOSE_DATAWIDTH bits
    //    = $clog2(NUM_GLB_IACT+1) bits.
    // -------------------------------------------------------------------
    genvar cc_gen, cr_gen, pe_gen;
    for (cc_gen = 0; cc_gen < CLUSTER_COLUMNS; cc_gen = cc_gen + 1) begin
      for (cr_gen = 0; cr_gen < CLUSTER_ROWS; cr_gen = cr_gen + 1) begin
        for (g_gen = 0; g_gen < NUM_GLB_IACT; g_gen = g_gen + 1) begin
          /*assign IACT_CONVERTER_X[cc_gen].IACT_CONVERTER_Y[cr_gen].iact_ready_w[g_gen] =
                iact_ready_o_oep_w[cc_gen * CLUSTER_ROWS * NUM_GLB_IACT + cr_gen * NUM_GLB_IACT + g_gen];*/
          assign IACT_CONVERTER_X[cc_gen].IACT_CONVERTER_Y[cr_gen].iact_ready_w[g_gen] =
                (iact_ready_o_oep_w == (2**(CLUSTER_COLUMNS*CLUSTER_ROWS*NUM_GLB_IACT))-1);
          assign iact_enable_i_oep_w[cc_gen * CLUSTER_ROWS * NUM_GLB_IACT + cr_gen * NUM_GLB_IACT + g_gen] =
                IACT_CONVERTER_X[cc_gen].IACT_CONVERTER_Y[cr_gen].iact_enable_w[g_gen];
          assign iact_data_i_oep_w[cc_gen * CLUSTER_ROWS * NUM_GLB_IACT * TRANS_BITWIDTH_IACT +
                                  cr_gen * NUM_GLB_IACT * TRANS_BITWIDTH_IACT +
                                  g_gen * TRANS_BITWIDTH_IACT +:TRANS_BITWIDTH_IACT] =
                IACT_CONVERTER_X[cc_gen].IACT_CONVERTER_Y[cr_gen].iact_data_w[g_gen * TRANS_BITWIDTH_IACT+:TRANS_BITWIDTH_IACT];
        end
        localparam IACT_CHOOSE_DATAWIDTH = $clog2(NUM_GLB_IACT + 1);
        for (pe_gen = 0; pe_gen < PES; pe_gen = pe_gen + 1) begin
          assign  iact_choose_i_oep_w[cc_gen * CLUSTER_ROWS * PES * IACT_CHOOSE_DATAWIDTH +
                                  cr_gen * PES * IACT_CHOOSE_DATAWIDTH + pe_gen * IACT_CHOOSE_DATAWIDTH +:IACT_CHOOSE_DATAWIDTH] =
                IACT_CONVERTER_X[cc_gen].IACT_CONVERTER_Y[cr_gen].iact_choose_w[pe_gen * IACT_CHOOSE_DATAWIDTH +:IACT_CHOOSE_DATAWIDTH];
        end
      end
    end
  endgenerate
endmodule
