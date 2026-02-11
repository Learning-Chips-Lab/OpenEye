// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
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
///   IS_TOPLEVEL            - Decides, wether modul is topmodul or not
///   DATA_IACT_BITWIDTH     - Width of input activation data
///   DATA_WGHT_BITWIDTH     - Width of weight data
///   DATA_PSUM_BITWIDTH     - Width of partial sum data, used in internal accumulator
///   TRANS_BITWIDTH_IACT    - Width of iact input port 
///   TRANS_BITWIDTH_WGHT    - Width of weight input port
///   TRANS_BITWIDTH_PSUM    - Width of partial sum input port
///   NUM_GLB_IACT           - Number of input activation global buffers
///   NUM_GLB_WGHT           - Number of rows of PEs in cluster (No WGHT GLBs)
///   NUM_GLB_PSUM           - Number of partial sum global buffers
///   CLUSTER_ROWS           - Amount of rows of clusters
///   CLUSTER_COLUMNS        - Amount of columns of clusters
///   IACT_PER_PE            - Maximum Iact Words in process element
///   WGHT_PER_PE            - Maximum Wght Words in process element
///   PSUM_PER_PE            - Maximum Psum Words in process element
///   IACT_PER_PE            - Iact words in PE
///   PSUM_PER_PE            - Psum words in PE
///   WGHT_PER_PE            - Wght words in PE
///   IACT_MEM_ADDR_WORDS    - Number of words in IACT GLB
///   PSUM_MEM_ADDR_WORDS    - Number of words in PSUM GLB
///   IACT_MEM_ADDR_BITS     - Width of words in IACT GLB
///   PSUM_MEM_ADDR_BITS     - Width of words in PSUM GLB
///   ROUTER_MODES_IACT      - Amount of bits in the Router for IACT
///   ROUTER_MODES_WGHT      - Amount of bits in the Router for WGHT
///   ROUTER_MODES_PSUM      - Amount of bits in the Router for PSUM
///   BANO_MODES             - Amount of Modes in Batch Normalization
///   AF_MODES               - Amount of Modes in AutoFunction CLuster
///   CLUSTERS               - Amount of clusters
///   PES                    - Amount of process elements
///   PARAMETER_POS indicates, in which word and at which position the required parameter ist stored
/// 
/// Ports:
///   ready_dma_o            - Ready Port Out
///   data_dma_i             - Data Port In
///   enable_dma_i           - Enable Port In
///
///   ready_dma_i            - Ready Port Out
///   data_dma_o             - Data Port In
///   enable_dma_o           - Enable Port In
///   last_data_o            - Signals the last output data word
///                 

module OpenEye_FPGA #(
    //Set parameters
  `ifdef USE_INTERNAL_PARAMS
      parameter CLUSTER_ROWS  = 8,
      parameter NUM_GLB_IACT  = 3,
      parameter NUM_GLB_PSUM  = 4,
      parameter NUM_GLB_WGHT  = 3,
  `else
    `include "parameters.vh"
      // Defaultvalues
  `endif
    parameter IS_TOPLEVEL   = 1,
    parameter SERIAL        = 1,
    parameter PARALLEL_MACS = 2,
    parameter SPARSITY_EN   = 1,  // 1=sparse mode (default), 0=dense mode

    parameter ADDR_IACT_BITWIDTH = 4,
    parameter ADDR_WGHT_BITWIDTH = 8,

    parameter DATA_IACT_BITWIDTH = 8,
    parameter DATA_PSUM_BITWIDTH = 20,
    parameter DATA_WGHT_BITWIDTH = 8,

    parameter TRANS_BITWIDTH_IACT = 24,
    parameter TRANS_BITWIDTH_WGHT = 24,
    parameter TRANS_BITWIDTH_PSUM = 20,
    parameter DATA_IACT_OVERHEAD  = 4,

    parameter PES = NUM_GLB_PSUM * NUM_GLB_WGHT,

    parameter CLUSTER_COLUMNS = 2,
    parameter CLUSTERS        = CLUSTER_COLUMNS * CLUSTER_ROWS,

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

    parameter DMA_BITWIDTH  = 64,

    parameter BUFFER_WIDTH = 12,
    parameter BUFFER_WIDTH_IACT_STREAM_CONSTRUCTOR = BUFFER_WIDTH + 1,
    parameter real FSM_IACT_RTR_CCLS_A = (CLUSTERS * NUM_GLB_IACT),
    parameter real FSM_IACT_RTR_CCLS_B = DMA_BITWIDTH / ROUTER_MODES_IACT,
    parameter real FSM_IACT_RTR_CCLS = FSM_IACT_RTR_CCLS_A / FSM_IACT_RTR_CCLS_B,
    parameter real FSM_WGHT_RTR_CCLS_A = (CLUSTERS * NUM_GLB_WGHT),
    parameter real FSM_WGHT_RTR_CCLS_B = DMA_BITWIDTH / ROUTER_MODES_WGHT,
    parameter real FSM_WGHT_RTR_CCLS = FSM_WGHT_RTR_CCLS_A / FSM_WGHT_RTR_CCLS_B,
    parameter real FSM_PSUM_RTR_CCLS_A = (CLUSTERS * NUM_GLB_PSUM),
    parameter real FSM_PSUM_RTR_CCLS_B = DMA_BITWIDTH / ROUTER_MODES_PSUM,
    parameter integer FSM_PSUM_RTR_CCLS_C = DMA_BITWIDTH - (DMA_BITWIDTH % ROUTER_MODES_PSUM),
    parameter real FSM_PSUM_RTR_CCLS = FSM_PSUM_RTR_CCLS_A / FSM_PSUM_RTR_CCLS_B,

    parameter integer FSM_CEIL_IACT_RTR_CCLS = $rtoi($ceil(FSM_IACT_RTR_CCLS)),
    parameter integer FSM_CEIL_WGHT_RTR_CCLS = $rtoi($ceil(FSM_WGHT_RTR_CCLS)),
    parameter integer FSM_CEIL_PSUM_RTR_CCLS = $rtoi($ceil(FSM_PSUM_RTR_CCLS)),


    //Number of Words per PE
    parameter BANO_MODES = 2,
    parameter AF_MODES   = 4,

    // Converter
    parameter BITWIDTH_IACT = TRANS_BITWIDTH_IACT / 2,

    //Storage RAMs
    parameter RAM_CELLS = 32,
    parameter RAM_CELLS_CLOG2 = $clog2(RAM_CELLS),
    parameter RAM_CELLS_ADDR_WIDTH = 12,
    parameter RAM_CELLS_WORD_BITWIDTH = 64,

    //Enable Traces for unpacked arrays
    parameter UNPACKED_TRACES_ENABLED = 1,

    localparam IACT_WORDS_IN_RAM = RAM_CELLS_WORD_BITWIDTH / DATA_IACT_BITWIDTH,
    localparam WORDS_PER_CYCLE   = 2,
    localparam PSUM_TO_IACT_CYCLES = (CLUSTER_COLUMNS * NUM_GLB_PSUM) == 4 ? 2 : 1

) (
    //Input DMA
    input clk_i,
    input rst_ni,

    //DEBUG OUTPUT IACT
    output                                debug_iact_we,
    output                                debug_iact_re,
    output  [RAM_CELLS_ADDR_WIDTH-1:0]    debug_iact_addr,
    output  [RAM_CELLS_WORD_BITWIDTH-1:0] debug_iact_data_i,
    output  [RAM_CELLS_WORD_BITWIDTH-1:0] debug_iact_data_o,

    //DEBUG OUTPUT PSUM1

    output                              debug_psum_we,
    output                              debug_psum_re,
    output  [BUFFER_WIDTH-1:0]          debug_psum_addr,
    output  [TRANS_BITWIDTH_PSUM*2-1:0] debug_psum_data_i,
    output  [TRANS_BITWIDTH_PSUM*2-1:0] debug_psum_data_o,

    //DEBUG OUTPUT PSUM2
    output                              debug_skip_iact_o,
    output  [DMA_BITWIDTH-1 : 0]        debug_data_dma_stream_o,
    output                              debug_enable_dma_stream_o,
    output  [3:0]                       debug_fsm_cycle_o,
    output  [3:0]                       debug_fsm_current_state,
    output  [2:0]                       debug_fsm_psum_state,

    output reg                      ready_dma_o,
    input      [DMA_BITWIDTH-1 : 0] data_dma_i,
    input                           enable_dma_i,

    //Output DMA
    input                           ready_dma_i,
    output reg [DMA_BITWIDTH-1 : 0] data_dma_o,
    output reg                      enable_dma_o,

    output reg last_data_o

);
reg last_data;
  //#######################
  //reset synchronization
  //#######################
  wire rst_n;

  RST_SYNC rst_sync_wrapper (
      .clk_i (clk_i),
      .rst_ni(rst_ni),
      .rst_no(rst_n)
  );

  //#######################
  //Register
  //#######################

  reg [DMA_BITWIDTH-1:0] data_dma_i_reg;
  reg                    enable_dma_i_reg;

  always @(posedge clk_i, negedge rst_n) begin
    if (rst_n == 1'b0) begin
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


  //Register, that occupy hyperparameters
  reg data_mode_reg;
  reg [$clog2(DATA_PSUM_BITWIDTH)-1:0] fraction_bit_reg;
  wire [17:0] needed_cycles_reg;
  wire [1:0] needed_x_cls_reg;
  wire [$clog2(CLUSTER_ROWS+1)-1:0] needed_y_cls_reg;
  wire [3:0] needed_iact_cycles_reg;
  wire [$clog2(PSUM_PER_PE+1)-1:0] filters_reg;
  wire [$clog2(IACT_ADDR_PER_PE+1)-1:0] iact_addr_len_reg;
  wire [$clog2(WGHT_ADDR_PER_PE+1)-1:0] wght_addr_len_reg;
  reg [$clog2(BANO_MODES)*NUM_GLB_PSUM-1:0] bano_cluster_mode_reg;
  reg [$clog2(AF_MODES)-1:0] af_cluster_mode_reg;
  wire [4:0] input_activations;
  wire [7:0] wght_cycles_reg;
  wire [2:0] stride_x_reg;
  wire [2:0] stride_y_reg;
  wire skipIact_reg;
  wire skipWght_reg;
  wire skipPsum_reg;
  wire [4-1:0] kernel_per_pe_cluster_reg;
  wire [3:0] kernel_size;
  reg [3:0] padding_reg;
  reg [DMA_BITWIDTH-1 : 0] fifo_data_i;
  reg fifo_read_i;
  reg fifo_write_i;
  wire [3:0] psum_delay_reg;
  reg [CLUSTERS-1:0] conv_array_reg;
  reg [CLUSTERS-1:0] param_array_reg;
  wire [CLUSTERS-1:0] start_param_array;
  wire [7:0]needed_psum_storage_cycles_reg;
  reg [7:0] debug_reg;

  //Register for the FSM
  reg [32-1:0] fsm_cycle;
  reg [$clog2(CLUSTER_COLUMNS)-1:0] fsm_x_cl;
  reg [$clog2(CLUSTER_ROWS)-1:0] fsm_y_cl;
  reg [$clog2(NUM_GLB_IACT)-1:0] fsm_iact_r;
  reg [$clog2(NUM_GLB_WGHT)-1:0] fsm_wght_r;
  reg [$clog2(NUM_GLB_PSUM)-1:0] fsm_psum_r;
  reg [$clog2(NUM_GLB_PSUM)-1:0] fsm_psum_r_q;
  reg results_ready;
  reg [19:0] finished_cycles_iact;
  reg [19:0] finished_cycles_psum;
  reg new_stream;
  reg reset_cycle;
  wire send_data_out;
  wire [2:0] add_up;
  wire [7:0] iact_x_line_repetitions;
  reg [7:0] iact_x_with_add_up;
  reg [16:0] fsm_psum_limit;
  reg early_stream_start;

  // Register for the Buffer
  reg buffer_select;
  reg iact_buffer_SP_en_r;
  reg iact_buffer_SP_en_w;
  reg [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_buffer_SP_data_w;
  reg [8-1:0] buffer_SP_addr_upper_limit;
  reg [8-1:0] buffer_SP_addr_lower_limit;
  reg [8-1:0] limit_increase_reg;
  reg [4-1:0] overhang_discrepancy;
  reg [4-1:0] overhang_counter;
  reg         overhang;
  reg         overhang_delay;

  reg wght_buffer_SP_en_r;
  reg wght_buffer_SP_en_w;
  reg [BUFFER_WIDTH:0] wght_buffer_SP_wr_addr;
  reg [BUFFER_WIDTH:0] wght_buffer_SP_rd_addr;
  reg [BUFFER_WIDTH:0] wght_buffer_SP_rd_addr_storage;
  reg [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_buffer_SP_data_w;
  wire [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_buffer_SP_data_r;

  reg  [CLUSTERS*NUM_GLB_PSUM/2-1:0]psum_buffer_SP_en_r;
  reg  [CLUSTERS*NUM_GLB_PSUM/2-1:0]psum_buffer_SP_en_w;
  wire  [BUFFER_WIDTH*CLUSTERS*NUM_GLB_PSUM/2-1:0] psum_buffer_SP_addr;
  reg  [BUFFER_WIDTH-1:0] psum_buffer_SP_addr_array [NUM_GLB_PSUM-1:0][CLUSTER_ROWS-1:0][CLUSTER_COLUMNS-1:0];
  reg  [BUFFER_WIDTH-1:0] psum_buffer_SP_addr_storage;
  reg  [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_buffer_SP_data_w;
  wire [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_buffer_SP_data_r;

  reg [BUFFER_WIDTH:0] wght_cnt;
  reg [BUFFER_WIDTH-1:0] psum_cnt;

  // Register for IACT Stream
  reg [RAM_CELLS_ADDR_WIDTH-2:0] current_buffer_addr;
  reg [ 7:0] current_buffer_n;
  reg [ 7:0] current_buffer_n_1;
  wire [ 7:0] iact_size_x;
  wire [ 7:0] iact_size_y;
  reg [11:0] iact_channels;
  wire [ 7:0] iact_channels_per_pe;
  wire [ 3:0] iact_channels_per_pe_next_layer;
  reg [ 7:0] iact_channels_counter;
  wire [ 7:0] iact_channel_max_cycles;
  wire [10:0] iact_needed_cycles;

  reg [ 3:0] iact_router_counter;

  // Register for IACT Converter Buffer
  reg buffer_SP_en_r_reg[RAM_CELLS-1:0];
  reg buffer_SP_en_w_reg[RAM_CELLS-1:0];
  reg choose_iact_buffer;
  wire choose_iact_buffer_input;
  wire choose_iact_buffer_output;
  wire fully_connected_layer;
  wire max_pooling;
  reg [RAM_CELLS_ADDR_WIDTH-2:0] buffer_SP_addr_reg[RAM_CELLS-1:0];
  reg [RAM_CELLS_ADDR_WIDTH-2:0] buffer_SP_addr_temp_reg[RAM_CELLS-1:0];
  reg [RAM_CELLS_WORD_BITWIDTH-1:0] buffer_SP_data_w_reg[RAM_CELLS-1:0];

  // Registers for IACT Converter
  reg [35:0] iact_converter_params_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg iact_converter_en_cfg_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg iact_converter_en_store_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg iact_converter_en_enc_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  wire [7:0] x_lines_reg;
  reg send_data_reg;
  wire store_in_psum;
  wire iact_converter_ready_w[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [2:0] iact_converter_n_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [BUFFER_WIDTH-1:0] iact_converter_mem_addr_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [3:0] iact_converter_mem_off_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];

  reg converters_ready;

  // Register for converting IACTS
  reg buffer_r_en_reg[RAM_CELLS-1:0];
  reg [2:0] iact_converter_n_1_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [3:0] iact_converter_mem_off_1_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [2:0] iact_converter_n_2_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [3:0] iact_converter_mem_off_2_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [2:0] iact_converter_n_3_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [3:0] iact_converter_mem_off_3_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];

  reg [CLUSTERS*NUM_GLB_IACT*TRANS_BITWIDTH_IACT - 1:0] iact_out_reg;
  reg iact_ready;
  reg iact_converter_enc_enable;
  reg iact_converter_params_enable;


  //New Iact Converter
  reg [7:0] iact_converter_max_cycles;
  reg [7:0] min_standing_cycles;
  wire [7:0] iact_converter_buffer_addr_max_cycles;
  reg [7:0] iact_converter_cycles;
  reg [7:0] iact_converter_buffer_addr_cycles;


  // Register, that configure the chip

  reg status_reg_enable_reg;

  reg compute_reg;
  reg  [ CLUSTERS * PES-1:0] compute_mask_reg; //Clusters * PEs in Cluster
  wire [CLUSTERS * PES -1:0] compute_mask_reg_port;
  assign compute_mask_reg_port = compute_mask_reg[CLUSTERS * PES -1:0];
  reg [ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0] router_mode_iact;
  reg [ROUTER_MODES_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] router_mode_wght;
  reg [ROUTER_MODES_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] router_mode_psum;

  wire [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_data_i_w;
  assign wght_data_i_w = wght_buffer_SP_data_r;
  reg [CLUSTERS*NUM_GLB_WGHT-1:0] wght_enable_i_reg;
  wire [CLUSTERS*NUM_GLB_WGHT-1:0] wght_ready_o_reg;

  wire [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_data_i_wire;
  wire [CLUSTERS*NUM_GLB_IACT-1:0] iact_enable_i_wire;
  wire [CLUSTERS*NUM_GLB_IACT-1:0] iact_ready_o_wire;
  wire [PES*CLUSTERS*$clog2(NUM_GLB_IACT+1)-1:0] iact_choose_i;

  reg [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_i_reg;
  reg [CLUSTERS*NUM_GLB_PSUM-1:0] psum_enable_i_reg;
  wire [CLUSTERS*NUM_GLB_PSUM-1:0] psum_ready_o_reg;

  wire [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_o_w;
  wire [CLUSTERS*NUM_GLB_PSUM-1:0] psum_enable_o;
  reg [CLUSTERS*NUM_GLB_PSUM-1:0] psum_ready_i_reg;
  wire [                    8-1:0] output_cycles;
  wire [                    5-1:0] kernels_per_calc;
  wire [                    4-1:0] y_lines_per_calc;

  wire [ 7:0] needed_wght_cycles_reg;
  wire [11:0] fc_size_reg;
  wire [BUFFER_WIDTH_IACT_STREAM_CONSTRUCTOR-1:0] needed_iact_buffer_words_reg;
  assign start_param_array = (1 << (((kernels_per_calc * y_lines_per_calc * ((iact_size_x-1+NUM_GLB_PSUM)/NUM_GLB_PSUM)*NUM_GLB_PSUM) + NUM_GLB_PSUM - 1)/NUM_GLB_PSUM)) - 1;
  //#######################
  //States of the FSM
  //#######################

  localparam IDLE = 4'd0;
  localparam GET_PARAMETERS = 4'd1;
  localparam GET_ROUTER_CONFIG = 4'd2;
  localparam GET_IACT = 4'd3;
  localparam GET_WGHT = 4'd4;
  localparam GET_BIAS = 4'd5;
  localparam GET_QUANTIZE = 4'd6;
  localparam GET_OFFSET = 4'd7;
  localparam START_CONVERTER = 4'd8;
  localparam CONVERT_IACT = 4'd9;
  localparam WAIT_CYCLE = 4'd10;
  localparam WAIT_FOR_RESULTS = 4'd11;
  localparam RECEIVE_PSUMS_TO_IACT = 4'd12;
  localparam MAXPOOLING_READ = 4'd13;
  localparam MAXPOOLING_SEND = 4'd14;
  
  localparam PSUM_IDLE = 0;
  localparam WAIT_TO_SEND_READY_SIGNAL = 1;
  localparam CALCULATE_PSUM = 2;
  localparam PSUM_GET_RESULTS = 3;
  localparam WAIT_FOR_SENDING_RESULTS = 4;
  localparam PSUM_SEND_RESULTS = 5;
  localparam SEND_PSUM_TO_IACT = 6;

  reg [3:0] fsm_current_state;
  reg [3:0] fsm_last_state;
  assign debug_fsm_current_state = fsm_current_state;

  //#######################
  //Process
  //#######################
  //Process for manging counter regs
  //reg single_iteration;
  reg [                     19:0] current_cycle;
  reg [                     15:0] iact_cycle_count;
  reg                             single_iteration;
  reg                             single_iteration2;
  reg                             single_iteration3;
  reg                             sending_data;
  reg                             wght_sendable;
  reg [                     12:0] fsm_sending_cycle;
  reg [CLUSTERS*NUM_GLB_WGHT-1:0] flat_help_var_send;
  reg [CLUSTERS*NUM_GLB_WGHT-1:0] temp_var;
  reg [                     63:0] prepared_iact [31:0];
  reg                             psum_to_iact_state;
  localparam EXTENDEDBITS = 48 - NUM_GLB_WGHT;
  //Process for sending data to OpenEye
  wire [CLUSTERS*NUM_GLB_IACT-1:0] iact_ready_o_oep_w;
  
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
      if (current_cycle <= needed_cycles_reg - 1) begin
        if (iact_ready_o_oep_w != {CLUSTERS*NUM_GLB_IACT{1'b1}}) begin
          if (!single_iteration) begin
            single_iteration  <= 1;
            single_iteration3 <= 1;
            if (iact_channels_counter == iact_channel_max_cycles -1) begin
              if (iact_router_counter == needed_y_cls_reg - 1) begin
                iact_cycle_count <= iact_cycle_count + 1;
                if (iact_cycle_count == {{8 {1'd0}},needed_wght_cycles_reg} - 1) begin
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
  //Process for sending parameters to Iact Converters
  reg [7:0] fsm_iact_params;
  reg [7:0] fsm_iact_params_y_line;
  reg [7:0] fsm_iact_params_kernel;
  reg [7:0] iact_converter_x;
  reg [7:0] iact_converter_y;
  reg [7:0] iact_converter_c;
  // additional register for fsm
  reg [$clog2(CLUSTER_ROWS+1)-1:0] fsm_row;
  reg [$clog2(CLUSTER_ROWS+1)-1:0] fsm_row_offset;
  integer a, b, word, line;
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin
      param_array_reg                 <= 0;
      fsm_iact_params                 <= 0;
      fsm_iact_params_y_line          <= 0;
      fsm_iact_params_kernel          <= 0;
      iact_converter_x                <= 0;
      iact_converter_y                <= 0;
      iact_converter_c                <= 0;
      fsm_row                         <= 0;
      fsm_row_offset                  <= 0;
      for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
        for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
          iact_converter_params_reg[a][b] <= 0;
          iact_converter_en_cfg_reg[a][b] <= 0;
        end
      end
    end else begin
      if (GET_ROUTER_CONFIG == fsm_current_state) begin
        param_array_reg <= start_param_array;
        fsm_iact_params <= CLUSTER_ROWS;
      end else begin
        for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
          for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
            iact_converter_en_cfg_reg[a][b] <= 0;
          end
        end
        if ((GET_WGHT == fsm_current_state) | (GET_IACT == fsm_current_state)) begin
          if (fsm_iact_params > 0) begin
            fsm_iact_params  <= fsm_iact_params - 1;
            for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin 
              iact_converter_params_reg[a
              ][fsm_row][35:32] <= fsm_row_offset;
              if (fully_connected_layer) begin
                iact_converter_params_reg[a
                ][fsm_row][31:24] <= iact_converter_x;
                  iact_converter_params_reg[a
                  ][fsm_row][23:16] <= iact_converter_y;
              end else begin
                if ((a != 0) & ((iact_converter_x + a[7:0] * NUM_GLB_PSUM[7:0]) >= iact_size_x)) begin
                  iact_converter_params_reg[a
                  ][fsm_row][31:24] <= 0;
                end else begin
                  iact_converter_params_reg[a
                  ][fsm_row][31:24] <= iact_converter_x + a[7:0] * NUM_GLB_PSUM[7:0];
                end
                if ((a != 0) & ((iact_converter_x + a[7:0] * NUM_GLB_PSUM[7:0]) >= iact_size_x) & (fsm_iact_params_kernel == kernels_per_calc - 1)) begin
                  iact_converter_params_reg[a
                  ][fsm_row][23:16] <= iact_converter_y + 1;
                end else begin
                  iact_converter_params_reg[a
                  ][fsm_row][23:16] <= iact_converter_y;
                end
              end

              iact_converter_params_reg[a
              ][fsm_row][15:8] <= iact_size_x;

              iact_converter_params_reg[a
              ][fsm_row][7:0] <= iact_converter_c;

              iact_converter_en_cfg_reg[a
              ][fsm_row] <= 1;
            end
            fsm_row <= fsm_row + needed_y_cls_reg;
            if (fsm_row + needed_y_cls_reg >= CLUSTER_ROWS) begin
              fsm_row <= fsm_row_offset + 1;
              fsm_row_offset <= fsm_row_offset + 1;
              if (fsm_row_offset == needed_y_cls_reg - 1) begin
                fsm_row        <= 0;
                fsm_row_offset <= 0;
              end
            end
            iact_converter_x <= iact_converter_x + (NUM_GLB_PSUM * CLUSTER_COLUMNS);
            if ((!fully_connected_layer & (((iact_converter_x + (NUM_GLB_PSUM * CLUSTER_COLUMNS)) * iact_x_line_repetitions) >= iact_size_x))
             | (fully_connected_layer)) begin
              iact_converter_x <= 0;
              if (((iact_converter_x + NUM_GLB_PSUM[7:0]) >= iact_size_x) & (!fully_connected_layer) & (kernels_per_calc != 1)) begin
                iact_converter_x <= NUM_GLB_PSUM[7:0];
              end
              fsm_iact_params_kernel <= fsm_iact_params_kernel + 1;
              if (fsm_iact_params_kernel == kernels_per_calc - 1) begin
                fsm_iact_params_kernel <= 0;
                fsm_iact_params_y_line <= fsm_iact_params_y_line + 1;
                if (fsm_iact_params_y_line == y_lines_per_calc - 1) begin
                  fsm_iact_params_y_line <= 0;
                  fsm_row_offset         <= 0;
                  iact_converter_c       <= iact_converter_c + 1;
                  if (!fully_connected_layer) begin
                    fsm_iact_params  <= 0;
                    fsm_row          <= 0;
                    iact_converter_c <= iact_converter_c + iact_channels_per_pe;
                  end
                end
                if (iact_converter_c + iact_channels_per_pe == iact_channels) begin
                  iact_converter_c <= 0;
                  iact_converter_y <= iact_converter_y + 1;
                  if (iact_converter_y + 1 >= iact_size_y) begin
                  iact_converter_y <= 0;
                  end
                end
              end
            end
          end
        end else begin
          for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
            for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
              iact_converter_en_cfg_reg[a][b] <= 0;
            end
          end
          if (iact_converter_params_enable) begin
            for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
              for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
                if (param_array_reg[(a+(b*CLUSTER_COLUMNS))] == 1) begin
                  if (iact_converter_cycles <= ((iact_converter_max_cycles - 1))) begin //Include Padding
                    iact_converter_en_cfg_reg[a][b] <= 1;
                  end
                end
              end
            end
            param_array_reg <= ((kernels_per_calc * iact_size_x / NUM_GLB_PSUM) | param_array_reg>>(CLUSTERS-(kernels_per_calc * iact_size_x / NUM_GLB_PSUM)));
          end
          if ((iact_converter_params_enable) | (fsm_iact_params > 0)) begin
            if (fsm_iact_params > 0) begin
              fsm_iact_params <= fsm_iact_params - 1;
            end
            if (iact_converter_params_enable & (fsm_current_state == CONVERT_IACT)) begin
              fsm_iact_params <= fsm_iact_params + kernels_per_calc * ((iact_size_x - 1 + (2 * NUM_GLB_PSUM)) / (2 * NUM_GLB_PSUM));
            end
            if (fsm_iact_params > 0) begin
              for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin 
                iact_converter_params_reg[a
                ][fsm_row][31:24] <= iact_converter_x + a[7:0] * NUM_GLB_PSUM[7:0];
                if ((a != 0) & ((iact_converter_x + a[7:0] * NUM_GLB_PSUM[7:0]) >= iact_size_x) & (fsm_iact_params_kernel == kernels_per_calc - 1)) begin
                  iact_converter_params_reg[a
                  ][fsm_row][23:16] <= iact_converter_y + 1;
                end else begin
                  iact_converter_params_reg[a
                  ][fsm_row][23:16] <= iact_converter_y;
                end
                if ((a != 0) & ((iact_converter_x + a[7:0] * NUM_GLB_PSUM[7:0]) >= iact_size_x)) begin
                  iact_converter_params_reg[a
                  ][fsm_row][31:24] <= 0;
                end else begin
                  iact_converter_params_reg[a
                  ][fsm_row][31:24] <= iact_converter_x + a[7:0] * NUM_GLB_PSUM[7:0];
                end
                iact_converter_params_reg[a
                ][fsm_row][7:0] <= iact_converter_c;
              end
              fsm_row <= fsm_row + needed_y_cls_reg;
              if (fsm_row + needed_y_cls_reg >= CLUSTER_ROWS) begin
                fsm_row <= fsm_row_offset + 1;
                fsm_row_offset <= fsm_row_offset + 1;
                if (fsm_row_offset == needed_y_cls_reg - 1) begin
                  fsm_row        <= 0;
                  fsm_row_offset <= 0;
                end
              end
              iact_converter_x <= iact_converter_x + (NUM_GLB_PSUM * CLUSTER_COLUMNS);
              if (((iact_converter_x + (NUM_GLB_PSUM * CLUSTER_COLUMNS)) * iact_x_line_repetitions >= iact_size_x)) begin
                iact_converter_x <= 0;
                if ((iact_converter_x + NUM_GLB_PSUM[7:0]) >= iact_size_x & (!fully_connected_layer) & (kernels_per_calc != 1)) begin
                  iact_converter_x <= NUM_GLB_PSUM[7:0];
                end
                fsm_iact_params_kernel <= fsm_iact_params_kernel + 1;
                if (fsm_iact_params_kernel == kernels_per_calc - 1) begin
                  fsm_iact_params_kernel <= 0;
                  fsm_iact_params_y_line <= fsm_iact_params_y_line + 1;
                  if (fsm_iact_params_y_line == y_lines_per_calc - 1) begin
                    fsm_iact_params_y_line <= 0;
                    fsm_iact_params        <= 0;
                    if (!fully_connected_layer) begin
                      fsm_row                <= 0;
                    end
                    fsm_row_offset         <= 0;
                  end
                  iact_converter_c <= iact_converter_c + iact_channels_per_pe;
                  if (fully_connected_layer) begin
                    iact_converter_c <= iact_converter_c + 1;
                  end
                  if (iact_converter_c == iact_channels - iact_channels_per_pe) begin
                    iact_converter_c <= 0;
                    iact_converter_y <= iact_converter_y + 1;
                    if (iact_converter_y >= iact_size_y - 1) begin
                    iact_converter_y <= 0;
                    end
                  end
                end
              end
            end
          end else begin
            param_array_reg <= conv_array_reg;
          end
        end
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
  end

  //Process for sending data to Iact Converters
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
      if (fsm_current_state == GET_ROUTER_CONFIG) begin
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
              if (iact_converter_cycles <= ((iact_converter_max_cycles - 1))) begin //Include Padding
                iact_converter_en_store_reg[a][b] <= 1;
              end
            end
          end
        end
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
  
  //Change here dataflow
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin
      //Reset Registers
      sending_data                   <= 0;
      fsm_sending_cycle              <= 0;
      wght_enable_i_reg              <= 0;
      wght_buffer_SP_en_r            <= 0;
      wght_buffer_SP_rd_addr         <= 0;
      wght_buffer_SP_rd_addr_storage <= 0;
      compute_reg                    <= 0;
      wght_sendable                  <= 0;
      flat_help_var_send              = 0;
      for (a = 0; a < RAM_CELLS; a=a+1) begin
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
        fsm_sending_cycle <= fsm_sending_cycle + 1;
        if (!sending_data) begin
          for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
            for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
              iact_converter_en_enc_reg[a][b] <= 1;
            end
          end
        end
        if (wght_sendable & (wght_ready_o_reg == {CLUSTERS*NUM_GLB_WGHT{1'b1}})) begin
          wght_sendable       <= 0;
          fsm_sending_cycle   <= 1;
          wght_buffer_SP_en_r <= 1;
        end
        if (wght_buffer_SP_en_r) begin
          if (fsm_sending_cycle <= wght_cnt) begin
            wght_buffer_SP_rd_addr <= wght_buffer_SP_rd_addr + 1;
          end
          if (fsm_sending_cycle > 2) begin
            flat_help_var_send = 0;
            for (a = 0; a < CLUSTER_ROWS; a=a+1) begin
              if ((a * (NUM_GLB_PSUM * CLUSTER_COLUMNS)) <= (((iact_size_x%NUM_GLB_PSUM)+iact_size_x) * y_lines_per_calc * kernels_per_calc) - 1) begin
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
          if (fsm_sending_cycle > wght_cnt + 2) begin
            fsm_sending_cycle   <= fsm_sending_cycle;
            wght_buffer_SP_en_r <= 0;
            wght_enable_i_reg   <= 0;
            if (current_cycle == 0) begin
              compute_reg <= 1;
            end
          end
        end
        if (current_cycle < needed_cycles_reg - 1) begin
          if (iact_ready_o_oep_w != {CLUSTERS*NUM_GLB_IACT{1'b1}}) begin
            if (!single_iteration) begin
              for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
                for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
                  iact_converter_en_enc_reg[a][b] <= 1;
                end
              end
              wght_sendable <= 1;
              if ((iact_channel_max_cycles == 1) & (needed_wght_cycles_reg == 1)) begin
                wght_sendable <= 0;
              end
              if (iact_channels_counter == iact_channel_max_cycles -1) begin
                if (iact_channel_max_cycles != 1) begin
                  wght_buffer_SP_rd_addr <= wght_buffer_SP_rd_addr_storage;
                end
                if (iact_router_counter == needed_y_cls_reg - 1) begin
                  wght_buffer_SP_rd_addr <= wght_buffer_SP_rd_addr;
                  wght_buffer_SP_rd_addr_storage <= wght_buffer_SP_rd_addr;
                  if (iact_cycle_count == {{8 {1'd0}},needed_wght_cycles_reg} - 1) begin
                    wght_buffer_SP_rd_addr_storage <= 0;
                    wght_buffer_SP_rd_addr         <= 0;
                  end
                end
              end
            end
          end
        end
        if (current_cycle == needed_cycles_reg) begin
          fsm_sending_cycle <= 0;
        end
      end else begin
        //Set Registers to 0
        fsm_sending_cycle   <= 0;
        wght_enable_i_reg   <= 0;
        wght_buffer_SP_en_r <= 0;
      end
      if (fsm_current_state == GET_PARAMETERS) begin
        sending_data           <= 0;
        fsm_sending_cycle      <= 0;
        wght_enable_i_reg      <= 0;
        wght_buffer_SP_en_r    <= 0;
        wght_buffer_SP_rd_addr <= 0;
        compute_reg            <= 0;
        wght_sendable          <= 1;
        flat_help_var_send = 0;
        for (a = 0; a < RAM_CELLS; a=a+1) begin
          prepared_iact[a]             <= 0;
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
        wght_buffer_SP_en_r            <= 0;
        wght_buffer_SP_rd_addr         <= 0;
        wght_buffer_SP_rd_addr_storage <= 0;
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

  reg       write_dma_en;
  reg [1:0] write_dma_addr;
  reg [DMA_BITWIDTH-1:0] dma_data_i;
  reg [15:0] select_ram_counter;
  reg [15:0] ram_counter_storage;
  reg [ 7:0] select_ram_offset;
  reg [ 7:0] ram_iact_modulo;
  reg signed [ 7:0] pooling_regs    [31:0];
  wire signed [ 7:0] debug_pooling_regs0;
  assign debug_pooling_regs0 = pooling_regs[0];
  wire signed [ 7:0] debug_pooling_regs1;
  assign debug_pooling_regs1 = pooling_regs[1];
  reg signed [ 7:0] pooling_stage_1 [7:0];
  reg signed [ 7:0] pooling_stage_2 [3:0];
  reg signed [ 7:0] pooling_stage_3 [1:0];
  reg signed [ 7:0] pooling_stage_4;

  reg [ 7:0] quant_offset [31:0];
  reg [ 6:0] quant_exp    [31:0];
  reg [24:0] quant_mant   [31:0];
 
  //#######################
  //Wires
  //#######################


  wire                                           buffer_SP_en_r     [RAM_CELLS-1:0];
  wire                                           buffer_SP_en_w     [RAM_CELLS-1:0];
  wire [             RAM_CELLS_ADDR_WIDTH-2:0]   buffer_SP_addr     [RAM_CELLS-1:0];
  wire [          RAM_CELLS_WORD_BITWIDTH-1:0]   buffer_SP_data_w   [RAM_CELLS-1:0];
  wire [2*RAM_CELLS_WORD_BITWIDTH*RAM_CELLS-1:0] buffer_SP_data_r_w;
  wire [RAM_CELLS_WORD_BITWIDTH*RAM_CELLS-1:0]   buffer_SP_data_r; 
  
reg [7:0] psum_cycle_buffer_1;
reg [7:0] psum_cycle_buffer_2;
reg [7:0] psum_cycle_buffer_3;
reg [7:0] psum_cycle_buffer_4;
reg [7:0] psum_sending_counter;
reg [3:0] sending_clusters;
reg [3:0] sending_cluster_rows;
reg [3:0] iteration_for_kernels_reg;
reg [11:0] pcb_1;
reg [11:0] pcb_2;
reg [11:0] pcb_3;
reg [3:0] fsm_psum_row_offset;
reg       past_padding;
wire      iact_buffer_next_addr;
reg [ 7:0] iact_channel_counter_reg;
reg [15:0] fsm_psum_cycle;
reg [ 3:0] fsm_psum_last_state;
reg [ 3:0] fsm_psum_current_state;
assign debug_fsm_psum_state = fsm_psum_current_state;
reg        psum_transmitted;
reg psum_router_set_reg;
reg start_new_cycle;
reg last_data_reg;
reg [$clog2(CLUSTER_COLUMNS)-1:0] fsm_x_cl_psum;
reg [$clog2(CLUSTER_COLUMNS)-1:0] fsm_x_cl_psum_q;
reg [   $clog2(CLUSTER_ROWS+1)-1:0] fsm_y_cl_psum;
reg [   $clog2(CLUSTER_ROWS+1)-1:0] fsm_y_cl_psum_q;
reg [   $clog2(CLUSTER_ROWS+1)-1:0] fsm_y_cl_psum_delay1;
reg [   $clog2(CLUSTER_ROWS+1)-1:0] fsm_y_cl_psum_delay2;
reg [   $clog2(CLUSTER_ROWS+1)-1:0] fsm_y_cl_psum_delay3;
reg [7:0] quantized_value_reg [7:0];
wire [7:0] testquant1;
wire [7:0] testquant2;
wire [7:0] testquant3;
wire [7:0] testquant4;
wire [7:0] testquant5;
wire [7:0] testquant6;
wire [7:0] testquant7;
wire [7:0] testquant8;
assign testquant1 = quantized_value_reg[0];
assign testquant2 = quantized_value_reg[1];
assign testquant3 = quantized_value_reg[2];
assign testquant4 = quantized_value_reg[3];
assign testquant5 = quantized_value_reg[4];
assign testquant6 = quantized_value_reg[5];
assign testquant7 = quantized_value_reg[6];
assign testquant8 = quantized_value_reg[7];
reg [7:0] current_filter;

wire [      $clog2(NUM_GLB_IACT+1)*CLUSTERS*PES-1:0] iact_choose_i_oep_w;
wire [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_data_i_oep_w;
wire [                    CLUSTERS*NUM_GLB_IACT-1:0] iact_enable_i_oep_w;

assign iact_buffer_next_addr = (((iact_converter_buffer_addr_cycles + 2 == (iact_converter_buffer_addr_max_cycles)) |
          (iact_converter_buffer_addr_max_cycles == 1 & (iact_converter_buffer_addr_cycles == 0))) &
          (iact_converter_cycles == 0) & 
          (iact_channels_counter != (iact_channel_max_cycles)));

  integer cr, cc, g;
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin
      status_reg_enable_reg                 <= 0;
      data_mode_reg                         <= 0;
      fraction_bit_reg                      <= 0;
      padding_reg                           <= 0;
      new_stream                            <= 0;
      fsm_cycle                             <= 0;
      fsm_last_state                        <= IDLE;
      fsm_current_state                     <= GET_PARAMETERS;
      fsm_x_cl                              <= 0;
      fsm_y_cl                              <= 0;
      fsm_iact_r                            <= 0;
      fsm_wght_r                            <= 0;
      buffer_select                         <= 0;
      early_stream_start                    <= 0;
      fifo_data_i                           <= 0;
      fifo_read_i                           <= 0;
      fifo_write_i                          <= 0;
      bano_cluster_mode_reg                 <= 0;
      af_cluster_mode_reg                   <= 0;
      compute_mask_reg                      <= 0;
      psum_data_i_reg                       <= 0;
      ready_dma_o                           <= 0;
      iact_buffer_SP_en_r                   <= 0;
      iact_buffer_SP_en_w                   <= 0;
      iact_buffer_SP_data_w                 <= 0;
      wght_buffer_SP_wr_addr                <= 0;
      wght_buffer_SP_en_w                   <= 0;
      wght_buffer_SP_data_w                 <= 0;
      wght_cnt                              <= 0;
      finished_cycles_iact                  <= 0;
      // iact converter
      iact_out_reg                          <= 0;
      iact_ready                            <= 0;
      buffer_SP_addr_upper_limit            <= 0;
      limit_increase_reg                    <= 0;
      overhang_discrepancy                  <= 0;
      overhang_counter                      <= 0;
      overhang                              <= 0;
      overhang_delay                        <= 0;
      buffer_SP_addr_lower_limit            <= 0;
      current_buffer_n                      <= 0;
      current_buffer_n_1                    <= 0;
      current_buffer_addr                   <= 0;
      iact_channels                         <= 0;
      iact_channels_counter                 <= 0;
      reset_cycle                       <= 0;
      select_ram_counter                    <= 0;
      ram_counter_storage                   <= 0;
      iact_x_with_add_up                    <= 0;
      fsm_psum_limit                        <= 0;
      //new iact regs
      iact_converter_max_cycles             <= 0;
      min_standing_cycles                   <= 0;
      iact_converter_cycles                 <= 0;
      iact_converter_buffer_addr_cycles     <= 0;
      send_data_reg                         <= 0;
      write_dma_en                          <= 0;
      write_dma_addr                        <= ~0;
      dma_data_i                            <= 0;
      past_padding                          <= 0;
      // Pooling
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
      for (a = 0; a < 32; a = a + 1) begin
        quant_offset[a] <= 0;
        quant_exp[a]    <= 0;
        quant_mant[a]   <= 0;
      end

      for (a = 0; a < RAM_CELLS; a=a+1) begin
        buffer_SP_en_r_reg[a]       <= 0;
        buffer_SP_en_w_reg[a]       <= 0;
        buffer_SP_addr_reg[a]       <= 0;
        buffer_SP_addr_temp_reg [a] <= 0;
        buffer_SP_data_w_reg[a]     <= 0;
      end
      choose_iact_buffer           <= 0;
      converters_ready              = 0;
      iact_converter_enc_enable    <= 0;
      iact_converter_params_enable <= 0;
      ram_iact_modulo              <= 0;

    end else begin
      case (fsm_current_state)

        IDLE: begin
          if (enable_dma_i_reg) begin
            fsm_last_state    <= IDLE;
            fsm_current_state <= GET_PARAMETERS;
            fifo_data_i       <= 0;
            fifo_read_i       <= 0;
            fifo_write_i      <= 0;
          end
        end

        GET_PARAMETERS: begin
          fifo_data_i                <= 0;
          fifo_read_i                <= 0;
          fifo_write_i               <= 0;
          status_reg_enable_reg      <= 1;
          ready_dma_o                <= 1;
          reset_cycle            <= 1;
          buffer_SP_addr_lower_limit <= 0;
          buffer_SP_addr_upper_limit <= 0;
          limit_increase_reg         <= 0;
          for (a = 0; a < RAM_CELLS; a = a + 1) begin
            buffer_SP_en_r_reg[a]      <= 0;
            buffer_SP_en_w_reg[a]      <= 0;
            buffer_SP_addr_reg[a]      <= 0;
            buffer_SP_addr_temp_reg[a] <= 0;
            buffer_SP_data_w_reg[a]    <= 0;
          end
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
          for (a = 0; a < 32; a = a + 1) begin
            quant_offset[a] <= 0;
            quant_exp[a]    <= 0;
            quant_mant[a]   <= 0;
          end
          write_dma_en <= 0;
          if (enable_dma_i_reg) begin
            if (!ready_dma_o) begin
              early_stream_start <= 1;
            end else begin
              early_stream_start <= 0;
              if (!early_stream_start) begin
                fsm_cycle       <= fsm_cycle + 1;
                reset_cycle <= 0;
                dma_data_i      <= data_dma_i_reg;
                case (fsm_cycle)
                  32'd0: begin
                    write_dma_addr  <= write_dma_addr + 1;
                    write_dma_en    <= 1;
                  end
                  32'd1: begin
                    write_dma_addr  <= write_dma_addr + 1;
                    write_dma_en    <= 1;
                  end
                  32'd2: begin
                    write_dma_addr <= write_dma_addr + 1;
                    write_dma_en   <= 1;
                  end
                  32'd3: begin
                    write_dma_addr <= write_dma_addr + 1;
                    write_dma_en   <= 1;
                    padding_reg    <= (kernel_size-1)/2;
                  end
                  default: begin

                  end
                endcase
                for (a = 0; a < PES * CLUSTERS; a = a + 1) begin
                  if (fsm_cycle >= 4 & (((fsm_cycle - 4) * DMA_BITWIDTH <= a) & ((fsm_cycle - 3) * DMA_BITWIDTH > a))) begin
                    compute_mask_reg[a] <= data_dma_i_reg[a%DMA_BITWIDTH];
                  end
                end
                if (fsm_cycle == (4 + (((PES * CLUSTERS) - 1)/64))) begin
                  choose_iact_buffer <= choose_iact_buffer_input;
                  fsm_last_state     <= GET_PARAMETERS;
                  fsm_current_state  <= GET_ROUTER_CONFIG;
                  fsm_cycle          <= 0;
                  write_dma_addr     <= ~0;
                  if (fully_connected_layer) begin
                    padding_reg <= 0;
                  end
                end
              end
            end
          end
        end
        
        GET_ROUTER_CONFIG: begin
          ready_dma_o   <= 1;
          new_stream    <= 1;
          iact_channels <= iact_channels_per_pe * iact_channel_max_cycles;
          iact_x_with_add_up <= iact_size_x + add_up;
          if (fully_connected_layer) begin
            iact_channels <= iact_channels_per_pe * NUM_GLB_WGHT * iact_channel_max_cycles; //iact_channel contains CLUSTER_Y
          end
          if (max_pooling) begin
            for (a = 0; a < RAM_CELLS; a=a+1) begin
              buffer_SP_en_r_reg[a] <= 1;
            end
          end
          if (enable_dma_i_reg) begin
            fsm_cycle <= fsm_cycle + 1;
            if(fsm_cycle == FSM_CEIL_IACT_RTR_CCLS + FSM_CEIL_WGHT_RTR_CCLS + FSM_CEIL_PSUM_RTR_CCLS - 1) begin
                fsm_psum_limit <= ((iact_x_with_add_up * PSUM_TO_IACT_CYCLES * kernels_per_calc * needed_wght_cycles_reg * filters_reg * iact_size_y)/8)+ 12;
              if (fully_connected_layer) begin
                fsm_psum_limit <= filters_reg + 2;
              end
              fsm_cycle             <= 0;
              fsm_last_state        <= GET_ROUTER_CONFIG;
              status_reg_enable_reg <= 0;
              if (!skipIact_reg) begin
                fsm_current_state <= GET_IACT;
              end else begin
                fsm_current_state <= GET_WGHT;
              end
              if (max_pooling) begin
                ready_dma_o       <= 0;
                fsm_current_state <= GET_OFFSET;
              end
            end
          end
        end

        GET_IACT: begin
          ready_dma_o         <= 1;
          for (a = 0; a < RAM_CELLS; a=a+1) begin
            buffer_SP_en_w_reg[a] <= 0;
          end
          if (enable_dma_i_reg) begin
            fsm_cycle          <= fsm_cycle + 1;
            current_buffer_n   <= current_buffer_n + 1;
            current_buffer_n_1 <= current_buffer_n;
            // get iact params
            buffer_SP_en_w_reg[current_buffer_n[RAM_CELLS_CLOG2-1:0]]   <= 1;
            buffer_SP_data_w_reg[current_buffer_n[RAM_CELLS_CLOG2-1:0]] <= data_dma_i_reg;
            buffer_SP_addr_reg[current_buffer_n_1[RAM_CELLS_CLOG2-1:0]] <= current_buffer_addr;
            current_buffer_addr                                         <= buffer_SP_addr_reg[current_buffer_n[RAM_CELLS_CLOG2-1:0]] + 1;
            if (current_buffer_n == RAM_CELLS - 1) begin
              current_buffer_n <= 0;
            end
            if (fsm_cycle == ((iact_size_x*iact_size_y*iact_channels + IACT_WORDS_IN_RAM - 1)/IACT_WORDS_IN_RAM) - 1) begin
              fsm_cycle         <= 0;
              fsm_current_state <= GET_WGHT;
              fsm_last_state    <= GET_IACT;
            end
          end else begin
            for (a = 0; a < RAM_CELLS; a=a+1) begin
              buffer_SP_en_w_reg[a] <= 0;
            end
          end
        end

        GET_WGHT: begin
          ready_dma_o <= 1;
          for (a = 0; a < RAM_CELLS; a=a+1) begin
            buffer_SP_en_w_reg[a] <= 0;
          end
          wght_buffer_SP_en_w <= 0;
          if (enable_dma_i_reg) begin
            iact_converter_max_cycles  <= (iact_size_y + {{4{1'd0}}, kernel_size}) - 8'b00000001;
            if (iact_channels == 1) begin
              iact_converter_max_cycles <= (iact_size_y + {{4{1'd0}}, kernel_size})/2;
            end
            if (fully_connected_layer) begin
              iact_converter_max_cycles <= 2;
            end
            min_standing_cycles <= (needed_iact_cycles_reg * iact_channels_per_pe) / WORDS_PER_CYCLE[8-1:0];
            for (b = 0; b < TRANS_BITWIDTH_WGHT; b = b + 1) begin
              wght_buffer_SP_data_w[fsm_y_cl*TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT+fsm_wght_r*TRANS_BITWIDTH_WGHT+b]
              <= data_dma_i_reg[b];
              wght_buffer_SP_data_w[CLUSTER_ROWS*TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT+fsm_y_cl*TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT+fsm_wght_r*TRANS_BITWIDTH_WGHT+b]
              <= data_dma_i_reg[TRANS_BITWIDTH_WGHT+b];
            end
            if (fsm_wght_r != NUM_GLB_WGHT - 1) begin
              fsm_wght_r <= fsm_wght_r + 1;
            end else begin
              fsm_wght_r <= 0;
              fsm_y_cl <= fsm_y_cl + 1;
              if (fsm_y_cl == CLUSTER_ROWS - 1) begin
                fsm_y_cl               <= 0;
                fsm_cycle              <= fsm_cycle + 1;
                wght_buffer_SP_en_w    <= 1;
                wght_buffer_SP_wr_addr <= wght_buffer_SP_wr_addr + 1;
                wght_cnt               <= ({7'd0,wght_cycles_reg} * ({7'd0,input_activations} * ((PARALLEL_MACS+filters_reg-1) / PARALLEL_MACS))) - 1;
                if(fsm_cycle == (wght_cycles_reg * (input_activations * ((PARALLEL_MACS+filters_reg-1) / PARALLEL_MACS))) - 1)begin
                  wght_cnt       <= (input_activations * ((PARALLEL_MACS+filters_reg-1) / PARALLEL_MACS));
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
          end
        end

        GET_BIAS: begin
          ready_dma_o         <= 1;
          wght_buffer_SP_en_w <= 0;
          if (psum_cnt != 0) begin
            fsm_last_state         <= GET_BIAS;
            fsm_current_state      <= GET_QUANTIZE;
            wght_buffer_SP_wr_addr <= 0;
            if (iact_channels_per_pe == 1) begin
              limit_increase_reg     <= (iact_size_x*2)/(WORDS_PER_CYCLE[7:0]*4);
              overhang_discrepancy   <= (iact_size_x*2)%(WORDS_PER_CYCLE[7:0]*4);
            end else begin
              limit_increase_reg     <= (iact_size_x*iact_channels_per_pe)/(WORDS_PER_CYCLE[7:0]*4);
              overhang_discrepancy   <= (iact_size_x*iact_channels_per_pe)%(WORDS_PER_CYCLE[7:0]*4);

            end
            overhang               <= 0;
            overhang_delay         <= 0;
            if (fully_connected_layer) begin
              overhang_discrepancy <= ((iact_size_x*iact_channels_per_pe*NUM_GLB_WGHT)%(WORDS_PER_CYCLE[7:0]*4));
              limit_increase_reg   <= ((NUM_GLB_WGHT*iact_channels_per_pe)/8);
              overhang             <= 0;
            end
            fsm_cycle <= 0;
          end
        end

        GET_QUANTIZE: begin
          overhang_counter    <= overhang_discrepancy;
          ready_dma_o         <= 1;
          wght_buffer_SP_en_w <= 0;
          if (enable_dma_i_reg) begin
            fsm_cycle <= fsm_cycle + 1;
            quant_exp[2*fsm_cycle]    <= data_dma_i_reg[31:25];
            quant_mant[2*fsm_cycle]   <= data_dma_i_reg[24:0];
            quant_exp[2*fsm_cycle+1]  <= data_dma_i_reg[63:57];
            quant_mant[2*fsm_cycle+1] <= data_dma_i_reg[56:32];
          end
          if (max_pooling) begin
            ready_dma_o <= 0;
            for (a = 0; a < RAM_CELLS; a=a+1) begin
              buffer_SP_en_r_reg[a] <= 1;
            end
          end
          if (fsm_cycle == 16 - 1) begin
            fsm_cycle         <= 0;
            fsm_last_state    <= GET_QUANTIZE;
            fsm_current_state <= GET_OFFSET;
          end
        end

        GET_OFFSET: begin
          ready_dma_o         <= 1;
          wght_buffer_SP_en_w <= 0;
          if (enable_dma_i_reg) begin
            fsm_cycle <= fsm_cycle + 1;
            quant_offset[8*fsm_cycle]   <= data_dma_i_reg[7:0];
            quant_offset[8*fsm_cycle+1] <= data_dma_i_reg[15:8];
            quant_offset[8*fsm_cycle+2] <= data_dma_i_reg[23:16];
            quant_offset[8*fsm_cycle+3] <= data_dma_i_reg[31:24];
            quant_offset[8*fsm_cycle+4] <= data_dma_i_reg[39:32];
            quant_offset[8*fsm_cycle+5] <= data_dma_i_reg[47:40];
            quant_offset[8*fsm_cycle+6] <= data_dma_i_reg[55:48];
            quant_offset[8*fsm_cycle+7] <= data_dma_i_reg[63:56];
          end
          if (max_pooling) begin
            ready_dma_o         <= 0;
            fsm_cycle           <= 2;
            fsm_last_state      <= GET_OFFSET;
            fsm_current_state   <= MAXPOOLING_READ;
            select_ram_counter  <= 0;
            ram_counter_storage <= 0;
          end
          if (fsm_cycle == 4 - 1) begin
            fsm_cycle         <= 0;
            ready_dma_o       <= 0;
            fsm_last_state    <= GET_OFFSET;
            fsm_current_state <= START_CONVERTER;
            buffer_SP_addr_upper_limit <= (buffer_SP_addr_upper_limit + limit_increase_reg);
            if (max_pooling) begin
              fsm_cycle           <= 2;
              fsm_current_state   <= MAXPOOLING_READ;
              select_ram_counter  <= 0;
              ram_counter_storage <= 0;
            end
          end
        end
        
        START_CONVERTER: begin
          converters_ready         = 1;
          for (a = 0; a < CLUSTER_COLUMNS; a=a+1) begin
            for (b = 0; b < CLUSTER_ROWS; b=b+1) begin
              converters_ready = converters_ready & iact_converter_ready_w[a][b];
            end
          end
          if (converters_ready == 1) begin
            fsm_current_state          <= CONVERT_IACT;
            past_padding               <= 0;
            select_ram_counter         <= 0;
            ram_counter_storage        <= 0;
            for (a = 0; a < RAM_CELLS; a=a+1) begin
              buffer_SP_en_r_reg[a] <= 1;
            end
            if (iact_channels == 1) begin
              past_padding <= 1;
              for (a = 0; a < RAM_CELLS; a=a+1) begin
                buffer_SP_addr_reg[a] <= 0;
              end
            end else begin
              for (a = 0; a < RAM_CELLS; a=a+1) begin
                buffer_SP_addr_reg[a] <= 0;
              end
            end
          end
        end

        CONVERT_IACT: begin
          fsm_cycle           <= fsm_cycle + 1;
          select_ram_counter  <= 0;
          if (past_padding & (select_ram_counter < iact_converter_buffer_addr_max_cycles)) begin
            select_ram_counter <= select_ram_counter + 1;
          end
          if ((select_ram_counter >= iact_converter_buffer_addr_max_cycles - 1)) begin
            select_ram_counter <= 0;
          end
          if ((select_ram_counter == iact_converter_buffer_addr_max_cycles - 1) | iact_buffer_next_addr) begin
            if ((iact_converter_cycles > {{4{1'd0}},padding_reg}) & (iact_converter_cycles < {{4{1'd0}},padding_reg} + iact_size_y + 1)) begin
              past_padding <= 1;
              for (a = 0; a < RAM_CELLS; a=a+1) begin
                if (buffer_SP_addr_upper_limit > buffer_SP_addr_lower_limit | (limit_increase_reg == 0)) begin
                  if (((a >= buffer_SP_addr_lower_limit) & (a < buffer_SP_addr_upper_limit))) begin
                    buffer_SP_addr_reg[a] <= buffer_SP_addr_reg[a] + 1;
                  end
                end else begin
                  if (((a >= buffer_SP_addr_lower_limit) | (a < buffer_SP_addr_upper_limit))) begin
                    buffer_SP_addr_reg[a] <= buffer_SP_addr_reg[a] + 1;
                  end
                end
              end
              buffer_SP_addr_upper_limit <= ((buffer_SP_addr_upper_limit + limit_increase_reg + overhang)%RAM_CELLS);
              buffer_SP_addr_lower_limit <= ((buffer_SP_addr_lower_limit + limit_increase_reg + overhang_delay)%RAM_CELLS);
              overhang                   <= 0;
              overhang_delay             <= overhang;
              overhang_counter           <= overhang_counter + overhang_discrepancy;
              if (overhang_counter + overhang_discrepancy >= (WORDS_PER_CYCLE[7:0]*4)) begin
                overhang_counter <= overhang_counter + overhang_discrepancy - (WORDS_PER_CYCLE[7:0]*4);
                overhang         <= 1;
              end
            end
          end
          iact_converter_enc_enable    <= 0;
          iact_converter_params_enable <= 0;
          if (iact_buffer_next_addr) begin
            iact_converter_enc_enable    <= 1;
            iact_converter_params_enable <= 1;
            select_ram_counter           <= 0;
            past_padding                 <= 1;
          end
          iact_converter_buffer_addr_cycles <= iact_converter_buffer_addr_cycles + 1;
          if (iact_converter_buffer_addr_cycles == (iact_converter_buffer_addr_max_cycles - 1)) begin
            iact_converter_buffer_addr_cycles <= 0;
            iact_converter_cycles <= iact_converter_cycles + 1;
            if (iact_converter_cycles == (iact_converter_max_cycles - 1)) begin
              iact_converter_cycles <= 0;
              iact_channels_counter <= iact_channels_counter + 1;
              if (iact_channels_counter == (iact_channel_max_cycles - 1)) begin
                fsm_current_state     <= WAIT_CYCLE;
                overhang_discrepancy  <= 0;
                iact_channels_counter <= 0;
                fsm_cycle             <= 0;
                past_padding          <= 0;
              end
            end
          end
        end

        WAIT_CYCLE: begin
          iact_buffer_SP_data_w     <= iact_out_reg;
          fsm_cycle                 <= fsm_cycle + 1;
          iact_ready                <= 0;
          iact_converter_enc_enable <= 0;
          if (fsm_cycle == (4 * 4)) begin
            fsm_cycle                <= 0;
            send_data_reg            <= 1;
            fsm_last_state           <= WAIT_CYCLE;
            if (send_data_out) begin
              fsm_current_state <= WAIT_FOR_RESULTS;
            end else begin
              fsm_current_state          <= RECEIVE_PSUMS_TO_IACT;
              ram_iact_modulo            <= iact_size_x % 8;
              select_ram_offset          <= 0;
              ram_counter_storage        <= 0;
              select_ram_counter         <= 0;
              if (iact_channels_per_pe_next_layer == 4) begin
                buffer_SP_addr_upper_limit <= (((iact_size_x+1)/2))%RAM_CELLS;
              end else begin
                buffer_SP_addr_upper_limit <= (((iact_size_x+1)/8))%RAM_CELLS;
              end
              buffer_SP_addr_lower_limit <= 0;
              limit_increase_reg         <= 0;
              overhang_discrepancy       <= 0;

              for (a = 0; a < RAM_CELLS; a=a+1) begin
                buffer_SP_data_w_reg[a] <= 0;
              end
            end
            iact_converter_cycles <= 0;
            current_buffer_n      <= 0;
            current_buffer_n_1    <= 0;
            current_buffer_addr   <= 0;
            for (a = 0; a < RAM_CELLS; a=a+1) begin
              buffer_SP_addr_reg[a] <= 0;
            end
          end
        end

        RECEIVE_PSUMS_TO_IACT: begin
          if (fsm_psum_current_state == WAIT_FOR_SENDING_RESULTS) begin
            choose_iact_buffer       <= choose_iact_buffer_output;
          end
          if (single_iteration & (single_iteration2 == 0)) begin
            iact_channels_counter <= iact_channels_counter + 1;
            if (iact_channels_counter == iact_channel_max_cycles - 1) begin
              iact_channels_counter <= 0;
            end
          end
          for (a = 0; a < RAM_CELLS; a=a+1) begin
            buffer_SP_en_w_reg[a] <= 0;
            if (buffer_SP_en_w_reg[a] == 1) begin
              buffer_SP_addr_reg[a] <= buffer_SP_addr_reg[a] + 1;
            end
          end
          if ((fsm_psum_current_state == SEND_PSUM_TO_IACT) & (fsm_psum_cycle >= 3) & (psum_to_iact_state == 0)) begin
            fsm_cycle <= fsm_cycle + 1;
            if (fsm_cycle >= 1) begin
              if (iact_channels_per_pe_next_layer == 4) begin
                select_ram_counter <= select_ram_counter + 1;
                select_ram_counter <= select_ram_counter + iact_channels_per_pe_next_layer;
                if (select_ram_counter >= ram_counter_storage + (iact_x_with_add_up/2) - iact_channels_per_pe_next_layer) begin
                  select_ram_counter <= select_ram_counter + iact_channels_per_pe_next_layer - (iact_x_with_add_up/2);
                  iact_channels_counter <= iact_channels_counter + 1;
                  if (iact_channels_counter == {4'd0,iact_channels_per_pe_next_layer} - 1) begin
                    iact_channels_counter      <= 0;
                    ram_counter_storage        <= (select_ram_counter + iact_channels_per_pe_next_layer - (add_up/2) - limit_increase_reg) % RAM_CELLS;
                    select_ram_counter         <= (select_ram_counter + iact_channels_per_pe_next_layer - (add_up/2) - limit_increase_reg) % RAM_CELLS;
                    buffer_SP_addr_upper_limit <= (buffer_SP_addr_upper_limit + (iact_size_x/2) + limit_increase_reg) % RAM_CELLS;
                    buffer_SP_addr_lower_limit <= buffer_SP_addr_upper_limit;
                    for (a = 0; a < RAM_CELLS; a=a+1) begin
                      if (buffer_SP_addr_upper_limit == buffer_SP_addr_lower_limit) begin
                        buffer_SP_en_w_reg[a] <= 1;
                      end else begin
                        if (buffer_SP_addr_upper_limit > buffer_SP_addr_lower_limit) begin
                          if ((a >= buffer_SP_addr_lower_limit) & (a < buffer_SP_addr_upper_limit)) begin
                            buffer_SP_en_w_reg[a] <= 1;
                          end
                        end else begin
                          if ((a >= buffer_SP_addr_lower_limit) | (a < buffer_SP_addr_upper_limit)) begin
                            buffer_SP_en_w_reg[a] <= 1;
                          end
                        end
                      end
                      if (iact_size_x % 2) begin
                        if ((a == (buffer_SP_addr_upper_limit - 1)%RAM_CELLS) & !limit_increase_reg) begin
                            buffer_SP_en_w_reg[a] <= 0;
                        end
                        limit_increase_reg <= (limit_increase_reg + 1)%2;
                        if (!limit_increase_reg) begin
                          buffer_SP_addr_lower_limit <= (buffer_SP_addr_upper_limit - 1)%RAM_CELLS;
                        end
                      end
                    end
                  end
                end
                for (a = 0; a < RAM_CELLS; a=a+1) begin
                  for (word = 0; word < 8; word=word+1) begin
                    //Wrapping around higher and lower edge
                     if (word == 4 & a == 0) begin debug_reg <= 0; end
                    if ((select_ram_counter - ram_counter_storage + iact_channels_per_pe_next_layer  > (iact_x_with_add_up/2) ) &
                      (((a >= (ram_counter_storage - ((1+limit_increase_reg)/2))) & (a < ram_counter_storage + ram_iact_modulo/2)) |
                      (((a > (select_ram_counter + 1)%RAM_CELLS) | (a < (select_ram_counter + ram_iact_modulo)%RAM_CELLS)) & (((select_ram_counter)%32) + ram_iact_modulo >= RAM_CELLS))
                      )) begin
                      if (word == 4 & a == 0) begin debug_reg <= 1; end
                      if ((word == (4 + {{24{1'd0}},iact_channels_counter} + limit_increase_reg*4) % 8 |
                          (word == (    {{24{1'd0}},iact_channels_counter} + limit_increase_reg*4) % 8)) &
                          !((a == buffer_SP_addr_lower_limit) & (limit_increase_reg) & word >= 4)) begin
                        if (word == 4 & a == 0) begin debug_reg <= 2; end
                        if (a < buffer_SP_addr_lower_limit + select_ram_counter + 4 - (iact_x_with_add_up/2)) begin
                          if (word == 4 & a == 0) begin debug_reg <= 3; end
                          buffer_SP_data_w_reg[a][8*((word+1)%8)+:8] <= quantized_value_reg[((word / 4) + limit_increase_reg + ((a-ram_counter_storage) * 2) + overhang_discrepancy + 4)%8];
                        end else begin
                          buffer_SP_data_w_reg[a][8*word+:8] <= quantized_value_reg[((word / 4) + ((a-ram_counter_storage) * 2) + overhang_discrepancy + 4)%8];
                        end
                      overhang_discrepancy <= (overhang_discrepancy + 4)%8;
                      end
                    //Regular
                    end else begin
                      if (word == 4 & a == 0) begin debug_reg <= 4; end
                      if (
                      ((iact_size_x % 2 == 0) &
                      ((a >= (select_ram_counter%RAM_CELLS)) & (a < (select_ram_counter%RAM_CELLS) + iact_channels_per_pe_next_layer)) |
                      //(((a >= (select_ram_counter%RAM_CELLS)) | (a < (select_ram_counter%RAM_CELLS) + iact_channels_per_pe_next_layer)) & ((select_ram_counter + iact_channels_per_pe_next_layer)%RAM_CELLS < (select_ram_counter%RAM_CELLS))) |
                      (((a >= (select_ram_counter)%RAM_CELLS) | (a < (select_ram_counter + iact_channels_per_pe_next_layer)%RAM_CELLS)) & ((select_ram_counter%RAM_CELLS) >= RAM_CELLS - 4 + 1)))
                      
                      |
                      
                      ((iact_size_x % 2 == 1) &
                      (((a >= buffer_SP_addr_lower_limit) & (a < (buffer_SP_addr_upper_limit))) | ((a >= buffer_SP_addr_lower_limit) |
                        (a < (buffer_SP_addr_upper_limit)) & (buffer_SP_addr_lower_limit > buffer_SP_addr_upper_limit)))) |
                      (((a >= (select_ram_counter)%RAM_CELLS) | (a < (select_ram_counter + iact_channels_per_pe_next_layer)%RAM_CELLS)) & ((select_ram_counter%RAM_CELLS) >= RAM_CELLS - 4 + 1))
                      ) begin
                        if (word == 4 & a == 0) begin debug_reg <= 5; end

                        if ((word == (iact_channels_per_pe_next_layer + {{24{1'd0}},iact_channels_counter})) | (word == {{24{1'd0}},iact_channels_counter})
                          & !((a == buffer_SP_addr_lower_limit) & (limit_increase_reg) & word <= 3)
                        ) begin
                          if (word == 4 & a == 0) begin debug_reg <= 6; end

                          buffer_SP_data_w_reg[a][8*word+:8] <= quantized_value_reg[((word / 4) + limit_increase_reg + ((a-ram_counter_storage) * 2) - overhang_discrepancy)%8];
                        
                        end
                        if (select_ram_counter - ram_counter_storage + iact_channels_per_pe_next_layer == (iact_size_x/2)) begin
                          overhang_discrepancy <= 0;
                        end
                      end
                    end
                  end
                end
              end else if (iact_channels_per_pe_next_layer == 2) begin
                select_ram_counter <= select_ram_counter + 1;
                if (select_ram_counter == 16 - 1) begin
                  select_ram_counter <= 0;
                end
                if (select_ram_counter == (CLUSTER_ROWS + ram_counter_storage - 1)%16) begin
                  select_ram_counter    <= ram_counter_storage;
                  iact_channels_counter <= iact_channels_counter + 1;
                  if (iact_channels_counter == iact_channels_per_pe_next_layer - 1) begin
                    iact_channels_counter <= 0;
                    ram_counter_storage   <= select_ram_counter + 1;
                    select_ram_counter    <= select_ram_counter + 1;
                    if (select_ram_counter >= (RAM_CELLS/2) - 1) begin
                      select_ram_counter  <= 0;
                      ram_counter_storage <= 0;
                    end
                    for (a = 0; a < RAM_CELLS; a=a+1) begin
                      if (buffer_SP_addr_upper_limit == buffer_SP_addr_lower_limit) begin
                        buffer_SP_en_w_reg[a] <= 1;
                      end else begin
                        if (buffer_SP_addr_upper_limit > buffer_SP_addr_lower_limit) begin
                          if ((a >= buffer_SP_addr_lower_limit) & (a < buffer_SP_addr_upper_limit)) begin
                            buffer_SP_en_w_reg[a] <= 1;
                          end
                        end else begin
                          if ((a >= buffer_SP_addr_lower_limit) | (a < buffer_SP_addr_upper_limit)) begin
                            buffer_SP_en_w_reg[a] <= 1;
                          end
                        end
                      end
                    end
                    buffer_SP_addr_lower_limit <= buffer_SP_addr_upper_limit;
                    buffer_SP_addr_upper_limit <= (buffer_SP_addr_upper_limit + CLUSTER_ROWS * 2) % 32;
                  end
                end
                for (a = 0; a < RAM_CELLS; a=a+1) begin
                  for (word = 0; word < 8; word=word+1) begin
                    if ((a >= select_ram_counter * 2) & (a < (select_ram_counter + 1) * 2)) begin
                      if ((word == (6 + iact_channels_counter)) |
                          (word == (4 + iact_channels_counter)) |
                          (word == (2 + iact_channels_counter)) |
                          (word == iact_channels_counter)) begin
                        buffer_SP_data_w_reg[a][8*word+:8] <= quantized_value_reg[word / 2 + (a%2) * 4];
                      end
                    end
                  end
                end
              end else if (iact_channels_per_pe_next_layer == 1) begin
                overhang_discrepancy <= overhang_discrepancy + iact_size_x;
                if (overhang_discrepancy >= 8 - iact_size_x) begin
                  overhang_discrepancy <= overhang_discrepancy + iact_size_x - 8;
                  select_ram_counter   <= select_ram_counter + 1;
                  if (select_ram_counter == RAM_CELLS - 1) begin
                    select_ram_counter <= 0;
                  end
                  buffer_SP_en_w_reg[select_ram_counter] <= 1;
                end
                
                for (a = 0; a < RAM_CELLS; a=a+1) begin
                  for (word = 0; word < 8; word=word+1) begin
                    if (!fully_connected_layer) begin
                      if ((((iact_size_x + overhang_discrepancy+select_ram_counter*8) > (word+8*a)) &
                      ((overhang_discrepancy+select_ram_counter*8) <= (word+8*a))) |
                      (((iact_size_x + overhang_discrepancy+select_ram_counter*8) > RAM_CELLS*8) &
                      ((iact_size_x + overhang_discrepancy+select_ram_counter*8) > (RAM_CELLS*8) + (word+8*a)))
                      ) begin
                        buffer_SP_data_w_reg[a][8*word+:8] <= quantized_value_reg[((word+8*a)-(overhang_discrepancy+select_ram_counter*8))%8];
                      end
                    end else begin
                      if (((word+8*a) == overhang_discrepancy + (select_ram_counter * 8))) begin
                        buffer_SP_data_w_reg[a][8*word+:8] <= quantized_value_reg[0];
                      end
                      if (((word+8*a) == filters_reg + overhang_discrepancy + (select_ram_counter * 8))) begin
                        buffer_SP_data_w_reg[a][8*word+:8] <= quantized_value_reg[1];
                      end
                    end
                  end
                end
              end
            end
          end
          if ((fsm_psum_current_state == PSUM_IDLE) & (fsm_cycle >= 1)) begin
            select_ram_counter <= 0;
            fsm_cycle          <= 0;
            for (a = 0; a < RAM_CELLS; a=a+1) begin
              buffer_SP_en_w_reg[a] <= 1;
            end
            fsm_current_state     <= GET_PARAMETERS;
            fsm_last_state        <= RECEIVE_PSUMS_TO_IACT;
            send_data_reg         <= 0;
            iact_channels_counter <= 0;
            write_dma_addr        <= ~0;
          end
        end

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
          if (last_data_o) begin
            fsm_last_state        <= WAIT_FOR_RESULTS;
            fsm_current_state     <= GET_PARAMETERS;
            send_data_reg         <= 0;
            fsm_cycle             <= 0;
            iact_channels_counter <= 0;
            write_dma_addr        <= ~0;
          end
        end

        MAXPOOLING_READ: begin
          //Counting and setting inputs for reading values
          fsm_cycle <= fsm_cycle + 1;
          if (fsm_cycle == 5) begin
            fsm_cycle             <= fsm_cycle;
            iact_converter_cycles <= iact_converter_cycles + stride_x_reg;
          end
          if (fsm_cycle >= 2) begin
            select_ram_counter    <= select_ram_counter + 1;
            if (select_ram_counter == 2 - 1) begin
              select_ram_counter  <= 0;
              ram_counter_storage <= ram_counter_storage + 1;
            end
            iact_channels_counter <= iact_channels_counter + 1;
            if (iact_channels_counter == iact_size_x - 1) begin
              iact_channels_counter <= 0;
              ram_counter_storage <= ram_counter_storage + 1 + (iact_size_x/2);
            end
          end
          for (word = 0; word < 2; word = word + 1) begin
            for (line = 0; line < 2; line = line + 1) begin
              for (a = 0; a < 2; a = a + 1) begin
                pooling_stage_1[a+2*line+4*word] <= buffer_SP_data_r[(8*((a*4)+(line*iact_size_x*4)+word+(select_ram_counter*2)+(ram_counter_storage*8)))%(32*64)+:8];
              end
            end
          end
          for (a = 0; a < 4; a = a + 1) begin
            if (pooling_stage_1[2*a] >= pooling_stage_1[2*a+1]) begin
              pooling_stage_2[a] <= pooling_stage_1[2*a];
            end else begin
              pooling_stage_2[a] <= pooling_stage_1[2*a+1];
            end
          end
          for (a = 0; a < 2; a = a + 1) begin
            if (pooling_stage_2[2*a] >= pooling_stage_2[2*a+1]) begin
              pooling_stage_3[a] <= pooling_stage_2[2*a];
            end else begin
              pooling_stage_3[a] <= pooling_stage_2[2*a+1];
            end
          end
          for (a = 0; a < 32; a = a + 1) begin
            for (b = 0; b < 2; b = b + 1) begin
              if (a == iact_converter_cycles + b & (fsm_cycle == 5)) begin
                if (pooling_regs[a] < pooling_stage_3[b]) begin
                  pooling_regs[a] <= pooling_stage_3[b];
                end
              end
            end
          end

          //Ending Condition
          if (fsm_cycle == 5) begin
            if (iact_converter_cycles >= 28 - 2) begin
              iact_channels_counter <= iact_channels_counter;
            end
            if (iact_converter_cycles == 32 - 2) begin
              select_ram_counter    <= 0;
              ram_counter_storage   <= ram_counter_storage - 1;
              fsm_cycle             <= 0;
              iact_converter_cycles <= 0;
              fsm_last_state        <= MAXPOOLING_READ;
              fsm_current_state     <= MAXPOOLING_SEND;
            end
          end

          if ((iact_converter_cycles < 30 - 2) & (fsm_cycle >= 3)) begin
            if (select_ram_counter == 1) begin
              buffer_SP_addr_upper_limit <= (buffer_SP_addr_upper_limit + (iact_size_x/2))%RAM_CELLS;
            end else begin
              buffer_SP_addr_upper_limit <= (buffer_SP_addr_upper_limit - (iact_size_x/2) + 1)%RAM_CELLS;
              if (iact_channels_counter == 0) begin
                buffer_SP_addr_upper_limit <= (buffer_SP_addr_upper_limit + 1)%RAM_CELLS;
              end
            end
            for (a = 0; a < RAM_CELLS; a=a+1) begin
              if (a == buffer_SP_addr_upper_limit) begin
                buffer_SP_addr_reg[a]      <= buffer_SP_addr_reg[a] + 1;
                buffer_SP_addr_temp_reg[a] <= buffer_SP_addr_temp_reg[a] + 1;
              end
            end
          end
        end

        MAXPOOLING_SEND: begin
          fsm_cycle <= fsm_cycle + 1;
          for (a = 0; a < RAM_CELLS; a=a+1) begin
            buffer_SP_addr_reg[a] <= finished_cycles_iact/8;
          end
          for (a = 0; a < RAM_CELLS; a=a+1) begin
            buffer_SP_en_w_reg[a] <= 0;
            if (fsm_cycle <= 3) begin
              if ((fsm_cycle +(finished_cycles_iact*4))%RAM_CELLS == a) begin
                buffer_SP_en_w_reg[a] <= 1;
                for (b = 0; b < IACT_WORDS_IN_RAM; b=b+1) begin
                  buffer_SP_data_w_reg[a][8*b+:8] <= pooling_regs[(8*fsm_cycle) + b];
                end
              end
            end
          end
          if (fsm_cycle >= 32/IACT_WORDS_IN_RAM) begin
            fsm_cycle <= 0;
            for (a = 0; a < 32; a = a + 1) begin
                pooling_regs[a] <= -128;
            end
            if (finished_cycles_iact == needed_cycles_reg-1) begin
              finished_cycles_iact   <= 0;
              fsm_last_state    <= MAXPOOLING_SEND;
              fsm_current_state <= GET_PARAMETERS;
              write_dma_addr    <= ~0;
              iact_channels_counter <= 0;
            end else begin
              finished_cycles_iact  <= finished_cycles_iact + 1;
              fsm_cycle             <= 0;
              for (a = 0; a < RAM_CELLS; a=a+1) begin
                buffer_SP_addr_reg[a] <= buffer_SP_addr_temp_reg[a];
              end
              fsm_last_state    <= MAXPOOLING_SEND;
              fsm_current_state <= MAXPOOLING_READ;
            end
          end
        end
        default: begin
        end
      endcase
    end
  end

  reg [ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0] router_mode_iact_storage;
  reg [                                        7:0] storage_cycles;
  reg [                                        7:0] storage_cycles_router;
  reg                                               first_cycle;
  reg [                  CLUSTERS*NUM_GLB_PSUM-1:0] psum_choose_i_reg;
  reg [7:0] iact_channels_counter_psum_router;
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin
      router_mode_iact                  <= 0;
      router_mode_iact_storage          <= 0;
      router_mode_wght                  <= 0;
      router_mode_psum                  <= 0;
      storage_cycles_router             <= 0;
      first_cycle                       <= 1;
      psum_choose_i_reg                 <= 0;
      iact_channels_counter_psum_router <= 0;
    end else begin
      if (compute_reg) begin
        if (fully_connected_layer) begin
          psum_choose_i_reg <= {CLUSTER_COLUMNS{{NUM_GLB_PSUM{1'b1}}, {((CLUSTER_ROWS - 1) * NUM_GLB_PSUM){1'b0}}}};
        end else begin
          if (needed_y_cls_reg == 1) begin
            psum_choose_i_reg <= (2 ** (CLUSTER_ROWS * CLUSTER_COLUMNS * NUM_GLB_PSUM) - 1);
          end else begin
            if (needed_y_cls_reg == 2) begin
              psum_choose_i_reg <= {CLUSTER_ROWS{{NUM_GLB_PSUM{1'b1}}, {NUM_GLB_PSUM{1'b0}}}};
            end else begin
              if (CLUSTER_ROWS == 8) begin
                psum_choose_i_reg <= {((CLUSTER_ROWS+1)/2){{NUM_GLB_PSUM{1'b1}},{NUM_GLB_PSUM{3'd0}}}};
              end else begin
                psum_choose_i_reg <= {(4){{NUM_GLB_PSUM{1'b1}},{NUM_GLB_PSUM{2'd0}}}};
              end
            end
          end
        end
      end
      if (fsm_current_state == GET_ROUTER_CONFIG) begin
        storage_cycles_router  <= 0;
        first_cycle            <= 1;
        iact_channels_counter_psum_router <= 0;
        if (enable_dma_i_reg) begin
          if (fsm_cycle < FSM_CEIL_IACT_RTR_CCLS) begin
            for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
              for (cr = 0; cr < CLUSTER_ROWS; cr = cr + 1) begin
                for (g = 0; g < NUM_GLB_IACT; g = g + 1) begin
                  if(((cc*NUM_GLB_IACT + cr*CLUSTER_COLUMNS*NUM_GLB_IACT + g)>=(fsm_cycle    *(DMA_BITWIDTH/ROUTER_MODES_IACT)))
                    &((cc*NUM_GLB_IACT + cr*CLUSTER_COLUMNS*NUM_GLB_IACT + g)< ((fsm_cycle+1)*(DMA_BITWIDTH/ROUTER_MODES_IACT))))begin
                    router_mode_iact[cc * CLUSTER_ROWS * NUM_GLB_IACT * ROUTER_MODES_IACT +
                                        cr * NUM_GLB_IACT * ROUTER_MODES_IACT + 
                                        g * ROUTER_MODES_IACT +:ROUTER_MODES_IACT] <=
                    data_dma_i_reg[(cc*NUM_GLB_IACT+cr*CLUSTER_COLUMNS*NUM_GLB_IACT+g-fsm_cycle*(DMA_BITWIDTH/ROUTER_MODES_IACT))
                    *ROUTER_MODES_IACT+:ROUTER_MODES_IACT];
                  end
                end
              end
            end
          end else begin
            if (fsm_cycle < FSM_CEIL_IACT_RTR_CCLS + FSM_CEIL_WGHT_RTR_CCLS) begin
              for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                for (cr = 0; cr < CLUSTER_ROWS; cr = cr + 1) begin
                  for (g = 0; g < NUM_GLB_WGHT; g = g + 1) begin
                    if(((cc*CLUSTER_ROWS*NUM_GLB_WGHT+cr*NUM_GLB_WGHT+g)>=((fsm_cycle-FSM_CEIL_IACT_RTR_CCLS)  *(DMA_BITWIDTH/ROUTER_MODES_WGHT)))
                      &((cc*CLUSTER_ROWS*NUM_GLB_WGHT+cr*NUM_GLB_WGHT+g)< ((fsm_cycle+1-FSM_CEIL_IACT_RTR_CCLS)*(DMA_BITWIDTH/ROUTER_MODES_WGHT))))begin
                      router_mode_wght[cc * CLUSTER_ROWS * NUM_GLB_WGHT * ROUTER_MODES_WGHT +
                                          cr * NUM_GLB_WGHT * ROUTER_MODES_WGHT + 
                                          g * ROUTER_MODES_WGHT+: ROUTER_MODES_WGHT] <=
                      data_dma_i_reg[(cc*CLUSTER_ROWS*NUM_GLB_WGHT+cr*NUM_GLB_WGHT+g-(fsm_cycle-FSM_CEIL_IACT_RTR_CCLS)*(DMA_BITWIDTH/ROUTER_MODES_WGHT))+:ROUTER_MODES_WGHT];
                    end
                  end
                end
              end
            end else begin
              for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                for (cr = 0; cr < CLUSTER_ROWS; cr = cr + 1) begin
                  for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
                    if(((cc*CLUSTER_ROWS*NUM_GLB_PSUM+cr*NUM_GLB_PSUM+g)>=((fsm_cycle-FSM_CEIL_IACT_RTR_CCLS-FSM_CEIL_WGHT_RTR_CCLS)  *(DMA_BITWIDTH/ROUTER_MODES_PSUM)))
                      &((cc*CLUSTER_ROWS*NUM_GLB_PSUM+cr*NUM_GLB_PSUM+g)< ((fsm_cycle+1-FSM_CEIL_IACT_RTR_CCLS-FSM_CEIL_WGHT_RTR_CCLS)*(DMA_BITWIDTH/ROUTER_MODES_PSUM))))begin
                      router_mode_psum[cc * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM +
                                          cr * NUM_GLB_PSUM * ROUTER_MODES_PSUM + 
                                          g * ROUTER_MODES_PSUM +:ROUTER_MODES_PSUM] <=
                      data_dma_i_reg[(cc*CLUSTER_ROWS*NUM_GLB_PSUM*ROUTER_MODES_PSUM+cr*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM-(fsm_cycle-FSM_CEIL_IACT_RTR_CCLS-FSM_CEIL_WGHT_RTR_CCLS)*FSM_PSUM_RTR_CCLS_C)+:ROUTER_MODES_PSUM];
                    end
                  end
                end
              end
            end
          end
        end
        router_mode_iact_storage <= router_mode_iact;
      end else begin
        if (fsm_current_state == WAIT_FOR_RESULTS | fsm_current_state == RECEIVE_PSUMS_TO_IACT) begin
          if (single_iteration3) begin
            if (iact_channels_counter == iact_channel_max_cycles -1) begin
              if (iact_router_counter == needed_y_cls_reg - 1) begin
                router_mode_iact <= router_mode_iact_storage;
              end else begin
                for (cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                  for (g=0; g<NUM_GLB_IACT; g=g+1) begin
                    router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+g*ROUTER_MODES_IACT+3] <= 0;
                    router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+g*ROUTER_MODES_IACT+4] <= 1;
                    router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+g*ROUTER_MODES_IACT+5] <= 1;
                  end
                end
                for (cr=1; cr<CLUSTER_ROWS; cr=cr+1) begin
                  for (cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                    for (g=0; g<NUM_GLB_IACT; g=g+1) begin
                      // If router is not on top of source already
                      if (!((router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT] == 1) &
                              (router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] == 1) &
                              (router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+2] == 0) &
                              (router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+3] == 0) &
                              (router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
                              (router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 1))) begin
                        // If router is source
                        if ((router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 0) &
                              (router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 0)) begin
                              // If router above is already destination
                              if ((router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
                                  (router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 1)) begin
                                router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] <= 1;
                              end
                              router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+3] <= 0;
                              router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] <= 1;
                              router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] <= 1;
                        end else begin
                          if ((router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 0) &
                              (router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 0)) begin
                            router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] <= 1;
                            router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] <= 0;
                            router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] <= 0;
                          end else begin
                            if ((router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
                                (router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 1)) begin
                              router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] <= 1;
                              router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+3] <= 0;
                              router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] <= 1;
                              router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] <= 1;
                            end else begin
                              if ((router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+3] == 1) &
                                  (router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
                                  (router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 0)) begin

                                  router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] <= 1;
                                  router_mode_iact[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] <= 0;
                              end
                            end
                          end
                        end
                      end
                    end
                  end
                end
              end
            end
            //PSUM Router
            first_cycle    <= 0;
            if (first_cycle == 0) begin
              if ((needed_y_cls_reg >= 2)) begin
                iact_channels_counter_psum_router <= iact_channels_counter_psum_router + 1;
                if ((iact_channels_counter_psum_router == iact_channel_max_cycles - 1)) begin
                  iact_channels_counter_psum_router <= 0;
                  for (cr = 1; cr < CLUSTER_ROWS; cr = cr + 1) begin
                    for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                      for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
                        router_mode_psum[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2] <=
                        router_mode_psum[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+(cr-1)*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2];
                      end
                    end
                  end
                  if (storage_cycles_router != (needed_psum_storage_cycles_reg - 1)) begin
                    storage_cycles_router <= storage_cycles_router + 1;
                    for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                      for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
                        router_mode_psum[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+g*ROUTER_MODES_PSUM+2] <= 0;
                      end
                    end
                    for (cr = 1; cr < CLUSTER_ROWS; cr = cr + 1) begin
                      for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                        for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
                          router_mode_psum[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2] <=
                          router_mode_psum[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+(cr-1)*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2];
                        end
                      end
                    end
                  end else begin
                    for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                      for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
                        router_mode_psum[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+g*ROUTER_MODES_PSUM+2] <= 1;
                      end
                    end
                    storage_cycles_router <= 0;
                  end
                end
                if (storage_cycles_router == needed_psum_storage_cycles_reg - 1) begin
                  storage_cycles_router <= 0;
                end else begin
                  storage_cycles_router <= storage_cycles_router + 1;
                end
              end
            end
          end
        end
      end
      if (reset_cycle) begin
        router_mode_iact              <= 0;
        router_mode_iact_storage          <= 0;
        storage_cycles_router             <= 0;
        first_cycle                       <= 1;
        psum_choose_i_reg                 <= 0;
        iact_channels_counter_psum_router <= 0;
      end
    end
  end

  integer g_psum, b_psum, cc_psum, cr_psum;
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin  ///Reset
      psum_transmitted            <= 0;
      fsm_psum_cycle              <= 0;
      fsm_psum_last_state         <= PSUM_IDLE;
      fsm_psum_current_state      <= PSUM_IDLE;
      for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
        for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
          for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
            psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= ~0;
          end
        end
      end
      psum_buffer_SP_addr_storage <= 0;
      psum_enable_i_reg           <= 0;
      psum_ready_i_reg            <= 0;
      storage_cycles              <= 0;
      psum_router_set_reg         <= 1;
      iact_channel_counter_reg    <= 0;
      results_ready                = 0;
      psum_cnt                    <= 0;
      start_new_cycle             <= 0;
      enable_dma_o                <= 0;
      data_dma_o                  <= 0;
      last_data_reg               <= 0;
      psum_to_iact_state          <= 0;
      fsm_x_cl_psum               <= 0;
      fsm_x_cl_psum_q             <= 0;
      fsm_y_cl_psum               <= 0;
      fsm_y_cl_psum_q             <= 0;
      fsm_psum_r                  <= 0;
      fsm_psum_r_q                <= 0;
      fsm_y_cl_psum_delay1        <= 0;
      fsm_y_cl_psum_delay2        <= 0;
      fsm_y_cl_psum_delay3        <= 0;
      psum_buffer_SP_en_r         <= 0;
      last_data                   <= 0;
      last_data_o                 <= 0;
      current_filter              <= 0;
      iteration_for_kernels_reg   <= 0;
      psum_cycle_buffer_1         <= 0;
      psum_cycle_buffer_2         <= 0;
      psum_cycle_buffer_3         <= 0;
      psum_cycle_buffer_4         <= 0;
      psum_sending_counter        <= 0;
      sending_clusters            <= 0;
      sending_cluster_rows        <= 0;
      pcb_1                       <= 0;
      pcb_2                       <= 0;
      pcb_3                       <= 0;
      fsm_psum_row_offset         <= 0;
      for (cr_psum = 0; cr_psum < 8; cr_psum = cr_psum + 1) begin
        quantized_value_reg[cr_psum] <= 0;
      end
      finished_cycles_psum        <= 0;
      psum_buffer_SP_data_w       <= 0;
    end else begin
      fsm_y_cl_psum_delay1        <= fsm_y_cl_psum;
      fsm_y_cl_psum_delay2        <= fsm_y_cl_psum_delay1;
      fsm_y_cl_psum_delay3        <= fsm_y_cl_psum_delay2;
      case (fsm_psum_current_state)
        PSUM_IDLE: begin
          enable_dma_o         <= 0;
          last_data            <= 0;
          last_data_o          <= 0;
          psum_buffer_SP_en_w  <= 0;
          psum_enable_i_reg    <= 0;
          last_data_reg        <= 0;
          current_filter       <= 0;
          psum_sending_counter <= 0;
          if (GET_BIAS == fsm_current_state) begin
            if (enable_dma_i_reg) begin
              psum_buffer_SP_data_w[fsm_x_cl_psum*CLUSTER_ROWS*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+fsm_y_cl_psum*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+fsm_psum_r*TRANS_BITWIDTH_PSUM+:TRANS_BITWIDTH_PSUM*PARALLEL_MACS]<= data_dma_i_reg[39:0];

              fsm_psum_r <= fsm_psum_r + PARALLEL_MACS;
              if ((fsm_psum_r == NUM_GLB_PSUM - PARALLEL_MACS)  | fully_connected_layer) begin
                fsm_psum_r <= 0;
                fsm_y_cl_psum <= fsm_y_cl_psum + 1;
                if ((fsm_y_cl_psum == CLUSTER_ROWS - 1) | fully_connected_layer) begin
                  fsm_y_cl_psum <= 0;
                  fsm_x_cl_psum <= fsm_x_cl_psum + 1;
                  if (fsm_x_cl_psum == (CLUSTER_COLUMNS - 1)) begin
                    fsm_x_cl_psum       <= 0;
                    fsm_psum_cycle      <= fsm_psum_cycle + 1;
                    psum_buffer_SP_en_w <= {((CLUSTERS * NUM_GLB_PSUM/2)){1'b1}};

                    for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                      for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                        for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                          psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] + 1;
                        end
                      end
                    end
                    if (((fsm_psum_cycle == (needed_wght_cycles_reg * filters_reg * iact_size_y * iact_x_line_repetitions) - 1) & (!fully_connected_layer))
                      | (fully_connected_layer & (fsm_psum_cycle == filters_reg - 1))) begin
                      fsm_psum_cycle <= 0;
                      psum_cnt       <= psum_buffer_SP_addr_array[0][0][0] + 1;
                    end
                  end
                end
              end
            end else begin
              psum_data_i_reg   <= 0;
            end
          end
          psum_transmitted    <= 0;
          fsm_psum_last_state <= PSUM_IDLE;
          storage_cycles      <= 0;
          if (compute_reg) begin
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                  psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= 0;
                end
              end
            end
            fsm_psum_last_state    <= PSUM_IDLE;
            fsm_psum_current_state <= WAIT_TO_SEND_READY_SIGNAL;
            fsm_psum_cycle         <= 0;
          end
        end
        
        WAIT_TO_SEND_READY_SIGNAL : begin
          results_ready = 0;
          if ((wght_enable_i_reg == 0) & (iact_enable_i_oep_w == 0)) begin
            fsm_psum_cycle <= fsm_psum_cycle + 1;
          end
          psum_transmitted <= 1;
          psum_buffer_SP_en_r    <= {(NUM_GLB_PSUM*CLUSTERS/2){1'd1}};
          if (fsm_psum_cycle >= 16) begin
            fsm_psum_cycle         <= 0;
            psum_ready_i_reg       <= {(NUM_GLB_PSUM*CLUSTER_ROWS*CLUSTER_COLUMNS){1'd1}};
            fsm_psum_last_state    <= WAIT_TO_SEND_READY_SIGNAL;
            fsm_psum_current_state <= CALCULATE_PSUM;
            psum_transmitted       <= 0;
          end
        end
        CALCULATE_PSUM : begin
          if (psum_ready_i_reg != 0) begin
            psum_ready_i_reg <= psum_ready_i_reg;
          end
          psum_buffer_SP_en_r <= {(NUM_GLB_PSUM*CLUSTERS/2){1'd1}};
          if ((results_ready == 0) & (psum_ready_i_reg != 0)) begin
            results_ready = 1;
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                  results_ready = results_ready & (psum_ready_o_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum*2] |
                   (router_mode_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM * 2 + 2] == 0));
                end
              end
            end
          end
          if (results_ready & (psum_ready_i_reg != 0)) begin
            fsm_psum_cycle <= fsm_psum_cycle + 1;
            psum_data_i_reg <= psum_buffer_SP_data_r;
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                  if (router_mode_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM * 2 + 2] == 1) begin
                    psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] + 1;
                  end
                  if ((fsm_psum_cycle != 0) & (router_mode_psum[(cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM) + (cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM) + (g_psum * ROUTER_MODES_PSUM) + 2] == 1)) begin
                    psum_enable_i_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum * 2] <= 1;
                    psum_enable_i_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum * 2 + 1] <= 1;
                  end
                end
              end
            end
            if (fsm_psum_cycle >= {{10{1'd0}},filters_reg}) begin
              psum_buffer_SP_en_r    <= 0;
              fsm_psum_last_state    <= CALCULATE_PSUM;
              fsm_psum_current_state <= PSUM_GET_RESULTS;
              results_ready           = 0;
              fsm_psum_cycle         <= 0;
              psum_enable_i_reg      <= {(NUM_GLB_PSUM*CLUSTER_ROWS*CLUSTER_COLUMNS){1'd1}};
              for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                  for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                    if (router_mode_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM * 2 + 2] == 1) begin
                      psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_SP_addr_storage;
                    end
                  end
                end
              end
            end
          end
        end
        PSUM_GET_RESULTS: begin
          psum_enable_i_reg <= 0;
          results_ready      = 1;
          psum_ready_i_reg  <= psum_ready_i_reg;
          psum_buffer_SP_data_w <= psum_data_o_w;
          for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
            for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
              for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                if (psum_buffer_SP_en_w[cc_psum*NUM_GLB_PSUM/2*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM/2+g_psum]) begin
                  psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] + 1;
                end
                results_ready = results_ready & (psum_enable_o[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum * 2] | 
                (router_mode_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM * 2 + 2] == 0));
                if (psum_enable_o[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum * 2] != 0) begin
                  psum_buffer_SP_en_w[cc_psum*NUM_GLB_PSUM/2*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM/2+g_psum] <= 1;
                end else begin
                  psum_buffer_SP_en_w[cc_psum*NUM_GLB_PSUM/2*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM/2+g_psum] <= 0;
                end
              end
            end
          end
          if (results_ready) begin
            psum_router_set_reg <= 0;
            fsm_psum_cycle <= fsm_psum_cycle + 1;
          end
          if (fsm_psum_cycle[$clog2(PSUM_PER_PE+1 )-1:0] >= filters_reg) begin
            psum_transmitted       <= 1;
            if ((finished_cycles_psum == needed_cycles_reg - 1)) begin
              for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                  for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                    psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= 0;
                  end
                end
              end
              fsm_psum_last_state    <= PSUM_GET_RESULTS;
              fsm_psum_current_state <= WAIT_FOR_SENDING_RESULTS;
              psum_ready_i_reg       <= 0;
              fsm_psum_cycle         <= 0;
              psum_buffer_SP_en_w    <= 0;
              psum_buffer_SP_en_r    <= {(NUM_GLB_PSUM/2*CLUSTER_ROWS*CLUSTER_COLUMNS){1'd1}};
              psum_enable_i_reg      <= 0;
            end else begin
              finished_cycles_psum <= finished_cycles_psum + 1;
              if ((needed_y_cls_reg >= 2) & !psum_router_set_reg) begin
                psum_router_set_reg <= 1;
                iact_channel_counter_reg <= iact_channel_counter_reg + 1;
                if ((iact_channel_counter_reg == iact_channel_max_cycles - 1)) begin
                  iact_channel_counter_reg <= 0;
                  if (storage_cycles == (needed_psum_storage_cycles_reg - 1)) begin
                    storage_cycles <= 0;
                  end
                end
              end
              fsm_psum_last_state    <= PSUM_GET_RESULTS;
              fsm_psum_current_state <= WAIT_TO_SEND_READY_SIGNAL;
              fsm_psum_cycle         <= 0;
              psum_buffer_SP_en_w    <= 0;
              if (storage_cycles == needed_psum_storage_cycles_reg - 1) begin
                storage_cycles <= 0;
                psum_buffer_SP_addr_storage <= psum_buffer_SP_addr_storage + {{6{1'd0}}, filters_reg};
                for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                  for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                    for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                      psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_SP_addr_storage + {{6{1'd0}}, filters_reg};
                    end
                  end
                end
              end else begin
                storage_cycles <= storage_cycles + 1;
                for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                  for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                    for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                      psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_SP_addr_storage;
                    end
                  end
                end
              end
              psum_ready_i_reg <= 0;
            end
          end
        end
        WAIT_FOR_SENDING_RESULTS: begin
          fsm_psum_cycle       <= fsm_psum_cycle + 1;
          fsm_psum_last_state  <= WAIT_FOR_SENDING_RESULTS;
          psum_sending_counter <= 0;
          if (send_data_out) begin
            if (fsm_psum_cycle == 1) begin
              fsm_psum_cycle         <= 0;
              fsm_psum_current_state <= PSUM_SEND_RESULTS;
              psum_buffer_SP_en_r    <= {(NUM_GLB_PSUM/2*CLUSTER_ROWS*CLUSTER_COLUMNS){1'd1}};
            end
          end else begin
            fsm_y_cl_psum          <= 0;
            if (fsm_psum_cycle >= 2) begin
              if (store_in_psum == 0) begin
                fsm_psum_current_state    <= SEND_PSUM_TO_IACT;

                iteration_for_kernels_reg <= (iact_channels_per_pe_next_layer+kernels_per_calc-1) / kernels_per_calc;
                psum_sending_counter      <= 0;
                if (CLUSTER_COLUMNS * NUM_GLB_PSUM < 8) begin
                  sending_clusters     <= 2 * CLUSTER_COLUMNS * NUM_GLB_PSUM;
                  sending_cluster_rows <= 2;
                end else begin
                  sending_clusters     <= CLUSTER_COLUMNS * NUM_GLB_PSUM;
                  sending_cluster_rows <= 1;
                end
              end else begin
                fsm_psum_current_state <= PSUM_IDLE;
              end
              fsm_psum_cycle       <= 0;
              psum_cycle_buffer_1  <= 0;
              psum_cycle_buffer_2  <= 0;
              psum_cycle_buffer_3  <= 0;
              psum_cycle_buffer_4  <= 0;
              pcb_1                <= 0;
              pcb_2                <= 0;
              pcb_3                <= 0;
              fsm_psum_row_offset  <= 0;
            end
          end
          fsm_psum_r    <= 0;
          fsm_y_cl_psum <= 0;
          fsm_x_cl_psum <= 0;
        end
        PSUM_SEND_RESULTS: begin
          psum_buffer_SP_en_r <= 0;
          fsm_psum_r_q <= fsm_psum_r;
          fsm_x_cl_psum_q <= fsm_x_cl_psum;
          fsm_y_cl_psum_q <= fsm_y_cl_psum;
          last_data_o     <= last_data;
          if (ready_dma_i == 1) begin
            if (fsm_psum_r | fsm_x_cl_psum | fsm_y_cl_psum | fsm_psum_cycle) begin
              enable_dma_o <= 1;
            end
            data_dma_o   <= psum_buffer_SP_data_r[fsm_x_cl_psum_q*CLUSTER_ROWS*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+fsm_y_cl_psum_q*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+fsm_psum_r_q*PARALLEL_MACS*TRANS_BITWIDTH_PSUM+:TRANS_BITWIDTH_PSUM * PARALLEL_MACS];
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                  if ((fsm_psum_r == g_psum) & (fsm_x_cl_psum == cc_psum) & (fsm_y_cl_psum == cr_psum)) begin
                    psum_buffer_SP_en_r[cc_psum*CLUSTER_ROWS*NUM_GLB_PSUM/2+cr_psum*NUM_GLB_PSUM/2+g_psum] <= 1;
                    psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] + 1;
                  end
                end
              end
            end
            fsm_psum_r <= fsm_psum_r + 1;
            if ((fsm_psum_r == (NUM_GLB_PSUM/2) - 1) | fully_connected_layer) begin
              fsm_psum_r <= 0;
              fsm_x_cl_psum <= fsm_x_cl_psum + 1;
              if (fsm_x_cl_psum == CLUSTER_COLUMNS - 1) begin
                fsm_x_cl_psum <= 0;
                fsm_y_cl_psum <= fsm_y_cl_psum + 1;
                if ((fsm_y_cl_psum >= CLUSTER_ROWS - 1) | fully_connected_layer) begin
                  fsm_y_cl_psum <= 0;
                  fsm_psum_cycle <= fsm_psum_cycle + 1;
                  if (fsm_psum_cycle == (needed_wght_cycles_reg * filters_reg * output_cycles) - 1) begin
                    fsm_psum_cycle      <= 0;
                    for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                      for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                        for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                          psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= 0;
                        end
                      end
                    end
                    psum_buffer_SP_en_r <= 0;
                    last_data           <= 1;
                  end
                end
              end
            end
            if (last_data_o) begin
              psum_buffer_SP_en_r    <= 0;
              last_data              <= 0;
              last_data_o            <= 0;
              enable_dma_o           <= 0;
              last_data_reg          <= 0;
              finished_cycles_psum   <= 0;
              fsm_psum_last_state    <= PSUM_SEND_RESULTS;
              fsm_psum_current_state <= PSUM_IDLE;
              fsm_psum_r             <= 0;
              fsm_y_cl_psum          <= 0;
              fsm_x_cl_psum          <= 0;
            end
          end
        end
        SEND_PSUM_TO_IACT: begin
          if (fsm_psum_cycle >= 3) begin
            if (!fully_connected_layer) begin
              if (CLUSTER_COLUMNS*NUM_GLB_PSUM>= 8) begin
                for (cr_psum = 0; cr_psum < 4; cr_psum = cr_psum + 1) begin
                  quantized_value_reg[2*cr_psum]     <= (quant_mant[current_filter] *
                  (psum_buffer_SP_data_r[(cr_psum/2)*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+fsm_y_cl_psum_delay3*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+(cr_psum%2)*TRANS_BITWIDTH_PSUM*2+:TRANS_BITWIDTH_PSUM]
                  + quant_offset[current_filter]))
                  >>> quant_exp[current_filter];
                  quantized_value_reg[2*cr_psum + 1] <= (quant_mant[current_filter] *
                  (psum_buffer_SP_data_r[(cr_psum/2)*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+fsm_y_cl_psum_delay3*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+(cr_psum%2)*TRANS_BITWIDTH_PSUM*2+TRANS_BITWIDTH_PSUM+:TRANS_BITWIDTH_PSUM]
                  + quant_offset[current_filter]))
                  >>> quant_exp[current_filter];
                end
              end else begin
                if (CLUSTERS*NUM_GLB_PSUM>= 8) begin
                  for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                    for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                      quantized_value_reg[4*cr_psum+2*cc_psum]     <= (quant_mant[current_filter] *
                      (psum_buffer_SP_data_r[cc_psum*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+(cr_psum+(fsm_y_cl_psum_delay3))*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+:TRANS_BITWIDTH_PSUM]
                      + quant_offset[current_filter]))
                      >>> quant_exp[current_filter];
                      quantized_value_reg[4*cr_psum+2*cc_psum + 1] <= (quant_mant[current_filter] *
                      (psum_buffer_SP_data_r[cc_psum*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+(cr_psum+(fsm_y_cl_psum_delay3))*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+TRANS_BITWIDTH_PSUM+:TRANS_BITWIDTH_PSUM]
                      + quant_offset[current_filter]))
                      >>> quant_exp[current_filter];
                    end
                  end
                end else begin
                  psum_to_iact_state <= ~psum_to_iact_state;
                  if (psum_to_iact_state == 0) begin
                    for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                      quantized_value_reg[2*cc_psum]     <= (quant_mant[current_filter] *
                      (psum_buffer_SP_data_r[cc_psum*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+:TRANS_BITWIDTH_PSUM]
                      + quant_offset[current_filter]))
                      >>> quant_exp[current_filter];
                      quantized_value_reg[2*cc_psum + 1] <= (quant_mant[current_filter] *
                      (psum_buffer_SP_data_r[cc_psum*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+TRANS_BITWIDTH_PSUM+:TRANS_BITWIDTH_PSUM]
                      + quant_offset[current_filter]))
                      >>> quant_exp[current_filter];
                    end
                  end else begin
                    for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                      quantized_value_reg[4+(2*cc_psum)]     <= (quant_mant[current_filter] *
                      (psum_buffer_SP_data_r[(cc_psum*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM) +:TRANS_BITWIDTH_PSUM]
                      + quant_offset[current_filter]))
                      >>> quant_exp[current_filter];
                      quantized_value_reg[4+(2*cc_psum) + 1] <= (quant_mant[current_filter] *
                      (psum_buffer_SP_data_r[(cc_psum*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM) + TRANS_BITWIDTH_PSUM+:TRANS_BITWIDTH_PSUM]
                      + quant_offset[current_filter]))
                      >>> quant_exp[current_filter];
                    end
                  end
                end
              end
            end else begin
                quantized_value_reg[0]     <= (quant_mant[current_filter] *
                (psum_buffer_SP_data_r[0+:TRANS_BITWIDTH_PSUM]
                + quant_offset[current_filter]))
                >>> quant_exp[current_filter];
                quantized_value_reg[1] <= (quant_mant[current_filter] *
                (psum_buffer_SP_data_r[TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+:TRANS_BITWIDTH_PSUM]
                + quant_offset[current_filter]))
                >>> quant_exp[current_filter];
            end
          end
          for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
            for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
              for (g_psum = 0; g_psum < (NUM_GLB_PSUM/2); g_psum = g_psum + 1) begin
                if (((cr_psum >= fsm_y_cl_psum) & (cr_psum < fsm_y_cl_psum+sending_cluster_rows)) | (CLUSTERS*NUM_GLB_PSUM <= 8)) begin
                  psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] + filters_reg * needed_wght_cycles_reg;
                  if (psum_cycle_buffer_1 == 0) begin
                    psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= pcb_1;
                    if (psum_cycle_buffer_2 == 0)  begin
                      psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= pcb_2;
                      if (psum_cycle_buffer_3 == 0) begin
                        psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= pcb_3;
                      end
                    end
                  end
                end
              end
            end 
          end
          psum_sending_counter <= psum_sending_counter + sending_clusters * iact_x_line_repetitions;
          if ((psum_sending_counter + (sending_clusters * iact_x_line_repetitions) >= kernels_per_calc * iact_x_with_add_up) | (iact_channels_per_pe_next_layer == 1)) begin
            psum_sending_counter <= 0;
            psum_cycle_buffer_1 <= psum_cycle_buffer_1 + 1;
            if (psum_cycle_buffer_1 == iact_x_line_repetitions - 1) begin
              psum_cycle_buffer_1 <= 0;
              pcb_1               <= pcb_1 + 1;
              psum_cycle_buffer_2 <= psum_cycle_buffer_2 + kernels_per_calc;
              if (psum_cycle_buffer_2 + kernels_per_calc>= iact_channels_per_pe_next_layer) begin
                psum_cycle_buffer_2 <= 0;
                pcb_1               <= pcb_2 + (filters_reg * iact_x_line_repetitions * needed_wght_cycles_reg);
                pcb_2               <= pcb_2 + (filters_reg * iact_x_line_repetitions * needed_wght_cycles_reg);
                psum_cycle_buffer_3 <= psum_cycle_buffer_3 + 1;
                if (psum_cycle_buffer_3 == iact_size_y - 1) begin
                  psum_cycle_buffer_3 <= 0;
                  pcb_1               <= pcb_3 + iteration_for_kernels_reg;
                  pcb_2               <= pcb_3 + iteration_for_kernels_reg;
                  pcb_3               <= pcb_3 + iteration_for_kernels_reg;
                  psum_cycle_buffer_4 <= psum_cycle_buffer_4 + 1;
                  //if (psum_cycle_buffer_4 == kernels_per_calc - 1) begin
                  //  psum_cycle_buffer_4 <= 0;
                  //  pcb_1 <= pcb_3 + iteration_for_kernels_reg;
                  //  pcb_2 <= pcb_3 + iteration_for_kernels_reg;
                  //end
                end
              end
            end
          end
          //if (iact_size_x > 8) begin
          if (iact_size_x > NUM_GLB_PSUM * CLUSTER_COLUMNS) begin
            fsm_y_cl_psum <= fsm_y_cl_psum + sending_cluster_rows;
            if (fsm_y_cl_psum + sending_cluster_rows >= CLUSTER_ROWS | ((fsm_y_cl_psum+sending_cluster_rows)*(NUM_GLB_PSUM*CLUSTER_COLUMNS) >= kernels_per_calc * iact_x_with_add_up)) begin
              fsm_y_cl_psum <= 0;
            end
          end else begin
            if (!fully_connected_layer) begin
              if (psum_cycle_buffer_3 == iact_size_y - 1) begin
                fsm_y_cl_psum <= fsm_y_cl_psum + sending_cluster_rows;
                if (fsm_y_cl_psum + sending_cluster_rows>= CLUSTER_ROWS ) begin
                  fsm_y_cl_psum <= 0;
                end
              end
            end
          end
          
          fsm_psum_cycle <= fsm_psum_cycle + 1;
          if (fsm_psum_cycle == fsm_psum_limit) begin
            fsm_psum_cycle              <= 0;
            fsm_psum_last_state         <= PSUM_SEND_RESULTS;
            fsm_psum_current_state      <= PSUM_IDLE;
            fsm_psum_r                  <= 0;
            fsm_y_cl_psum               <= 0;
            fsm_x_cl_psum               <= 0;
            psum_sending_counter        <= 0;
            psum_to_iact_state          <= 0;
          end
        end
        default: begin
        end
      endcase
      if (status_reg_enable_reg) begin
        psum_transmitted            <= 0;
        for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
          for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
            for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
              psum_buffer_SP_addr_array[cc_psum][cr_psum][g_psum] <= {(BUFFER_WIDTH*CLUSTERS*NUM_GLB_PSUM/2){1'd1}};
            end
          end
        end
        psum_enable_i_reg           <= 0;
        psum_ready_i_reg            <= 0;
        fsm_psum_last_state         <= PSUM_SEND_RESULTS;
        fsm_psum_current_state      <= PSUM_IDLE;
        fsm_psum_cycle              <= 0;
        psum_buffer_SP_addr_storage <= 0;
        storage_cycles              <= 0;
        psum_router_set_reg         <= 1;
        iact_channel_counter_reg    <= 0;
        results_ready                = 0;
        psum_cnt                    <= 0;
        start_new_cycle             <= 0;
        enable_dma_o                <= 0;
        data_dma_o                  <= 0;
        last_data_reg               <= 0;
        fsm_x_cl_psum               <= 0;
        fsm_y_cl_psum               <= 0;
        psum_buffer_SP_en_r         <= 0;
        finished_cycles_psum        <= 0;
        fsm_psum_r                  <= 0;
        fsm_x_cl_psum               <= 0;
      end
    end
  end


  genvar k_gen;
  for (k_gen = 0; k_gen < RAM_CELLS; k_gen=k_gen+1) begin : gen_RAM_wires
    assign buffer_SP_en_r[k_gen] = buffer_SP_en_r_reg[k_gen];
    assign buffer_SP_en_w[k_gen] = buffer_SP_en_w_reg[k_gen];
    assign buffer_SP_addr[k_gen] = buffer_SP_addr_reg[k_gen];
    assign buffer_SP_data_w[k_gen] = buffer_SP_data_w_reg[k_gen];
    assign buffer_SP_data_r =   buffer_SP_data_r_w[0+:RAM_CELLS_WORD_BITWIDTH*RAM_CELLS];
  end
  
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
      for (i_trace = 0; i_trace < 8; i_trace = i_trace + 1) begin : quant_stage_traces
        wire [7:0] quantized_out_trace; 
        assign quantized_out_trace = quantized_value_reg[i_trace]; 
      end
      
    end
  endgenerate

  generate
    genvar i_gen, j_gen, g_gen;
    // IACT Buffer
    for (j_gen = 0; j_gen < RAM_CELLS; j_gen=j_gen+1) begin : BUFFER_A
        RAM_SP #(
            .DataWidth(RAM_CELLS_WORD_BITWIDTH),
            .AddrWidth(RAM_CELLS_ADDR_WIDTH),
            .Pipelined(1)
        ) iact_converter_buffer_SP (
            .clk_i(clk_i),
            .rd_en_i(buffer_SP_en_r[j_gen] & !buffer_SP_en_w[j_gen]),
            .wr_en_i(buffer_SP_en_w[j_gen]),
            .addr_i({choose_iact_buffer,buffer_SP_addr[j_gen]}),
            .data_i(buffer_SP_data_w[j_gen]),
            .data_o(buffer_SP_data_r_w[j_gen*RAM_CELLS_WORD_BITWIDTH+:RAM_CELLS_WORD_BITWIDTH])
        );
    end

    // IACT Converter
    for (i_gen = 0; i_gen < CLUSTER_COLUMNS; i_gen=i_gen+1) begin : IACT_CONVERTER_X
      for (j_gen = 0; j_gen < CLUSTER_ROWS; j_gen=j_gen+1) begin : IACT_CONVERTER_Y
        wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] iact_data_w;
        wire [      $clog2(NUM_GLB_IACT+1)*PES-1:0] iact_choose_w;
        wire [                    NUM_GLB_IACT-1:0] iact_ready_w;
        wire [                    NUM_GLB_IACT-1:0] iact_enable_w;
        iact_stream_constructor #(
            .CLUSTER_ROWS      (CLUSTER_ROWS),
            .NUM_GLB_IACT      (NUM_GLB_IACT),
            .PE_X              (NUM_GLB_PSUM),
            .PE_Y              (NUM_GLB_WGHT),
            .DATA_IACT_BITWIDTH(DATA_IACT_BITWIDTH),
            .DATA_IACT_OVERHEAD(DATA_IACT_OVERHEAD),
            .RAM_CELLS         (RAM_CELLS),
            .WORD_BITWIDTH     (TRANS_BITWIDTH_IACT * NUM_GLB_IACT),
            .ADDRWIDTH         (BUFFER_WIDTH_IACT_STREAM_CONSTRUCTOR)
        ) iact_stream_constructor (
            .clk_i                       (clk_i),
            .rst_ni                      (rst_ni),
            .storage_i                   (buffer_SP_data_r),
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
            .iact_channels_i             (iact_channels_per_pe),
            .x_lines_i                   (iact_x_line_repetitions),
            .needed_wght_cycles_i        (needed_wght_cycles_reg),
            .needed_iact_router_cycles_i (needed_iact_cycles_reg),
            .wght_size_i                 (kernel_size),
            .y_lines_per_calc            (y_lines_per_calc),
            .fully_connected_i           (fully_connected_layer),
            .needed_iact_buffer_words_i  (needed_iact_buffer_words_reg)
        );
      end
    end

    RAM_SP #(
        .DataWidth(TRANS_BITWIDTH_WGHT * CLUSTERS * NUM_GLB_WGHT),
        .AddrWidth(BUFFER_WIDTH + 1),
        .Pipelined(1)
    ) wght_buffer_SP (
        .clk_i  (clk_i),
        .rd_en_i(wght_buffer_SP_en_r & !wght_buffer_SP_en_w),
        .wr_en_i(wght_buffer_SP_en_w),
        .addr_i (wght_buffer_SP_wr_addr | wght_buffer_SP_rd_addr),
        .data_i (wght_buffer_SP_data_w),
        .data_o (wght_buffer_SP_data_r)
    );

    for (i_gen = 0; i_gen < CLUSTER_COLUMNS; i_gen=i_gen+1) begin : PSUM_RAM_X
      for (j_gen = 0; j_gen < CLUSTER_ROWS; j_gen=j_gen+1) begin : PSUM_RAM_Y
        for (g_gen = 0; g_gen < NUM_GLB_PSUM/2; g_gen=g_gen+1) begin : PSUM_RAM_GLB
          assign psum_buffer_SP_addr[i_gen*BUFFER_WIDTH*CLUSTER_ROWS*NUM_GLB_PSUM/2+j_gen*BUFFER_WIDTH*NUM_GLB_PSUM/2+g_gen*BUFFER_WIDTH+:BUFFER_WIDTH]
          = psum_buffer_SP_addr_array[i_gen][j_gen][g_gen];
          RAM_SP #(
              .DataWidth(TRANS_BITWIDTH_PSUM*2),
              .AddrWidth(BUFFER_WIDTH),
              .Pipelined(1)
          ) psum_buffer_SP (
              .clk_i  (clk_i),
              .rd_en_i(psum_buffer_SP_en_r[i_gen*CLUSTER_ROWS*NUM_GLB_PSUM/2+j_gen*NUM_GLB_PSUM/2+g_gen] & !psum_buffer_SP_en_w[i_gen*CLUSTER_ROWS*NUM_GLB_PSUM/2+j_gen*NUM_GLB_PSUM/2+g_gen]),
              .wr_en_i(psum_buffer_SP_en_w[i_gen*CLUSTER_ROWS*NUM_GLB_PSUM/2+j_gen*NUM_GLB_PSUM/2+g_gen]),
              .addr_i (psum_buffer_SP_addr[i_gen*BUFFER_WIDTH*CLUSTER_ROWS*NUM_GLB_PSUM/2+j_gen*BUFFER_WIDTH*NUM_GLB_PSUM/2+g_gen*BUFFER_WIDTH+:BUFFER_WIDTH]),
              .data_i (psum_buffer_SP_data_w[i_gen*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+j_gen*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+g_gen*TRANS_BITWIDTH_PSUM*2+:TRANS_BITWIDTH_PSUM*2]),
              .data_o (psum_buffer_SP_data_r[i_gen*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+j_gen*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+g_gen*TRANS_BITWIDTH_PSUM*2+:TRANS_BITWIDTH_PSUM*2])
          );
        end
      end
    end
    assign debug_psum_re     = psum_buffer_SP_en_r[0] & !psum_buffer_SP_en_w[0];
    assign debug_psum_we     = psum_buffer_SP_en_w[0];
    assign debug_psum_addr   = psum_buffer_SP_addr[0+:BUFFER_WIDTH];
    assign debug_psum_data_i = psum_buffer_SP_data_w[0+:TRANS_BITWIDTH_PSUM*2];
    assign debug_psum_data_o = psum_buffer_SP_data_r[0+:TRANS_BITWIDTH_PSUM*2];

    assign debug_skip_iact_o = skipIact_reg;
    assign debug_data_dma_stream_o = data_dma_i_reg;
    assign debug_enable_dma_stream_o = enable_dma_i_reg;
    assign debug_fsm_cycle_o = fsm_cycle[3:0];

    dma_storage  #(
      
    ) dma_storage (
        .clk_i(clk_i),
        .rst_ni(rst_n),
        .write_en(write_dma_en),
        .write_addr(write_dma_addr),
        .dma_data_i(dma_data_i),
        .wght_cycles_reg(wght_cycles_reg),
        .stride_x_reg(stride_x_reg),
        .stride_y_reg(stride_y_reg),
        .skipIact_reg(skipIact_reg),
        .skipWght_reg(skipWght_reg),
        .skipPsum_reg(skipPsum_reg),
        .psum_delay_reg(psum_delay_reg),
        .kernel_per_pe_cluster_reg(kernel_per_pe_cluster_reg),
        .kernel_size(kernel_size),
        .x_lines_reg(x_lines_reg),
        .needed_wght_cycles_reg(needed_wght_cycles_reg),
        .needed_cycles_reg(needed_cycles_reg),
        .iact_converter_buffer_addr_max_cycles(iact_converter_buffer_addr_max_cycles),
        .iact_channels_per_pe(iact_channels_per_pe),
        .fc_size_reg(fc_size_reg),
        .iact_size_x(iact_size_x),
        .iact_size_y(iact_size_y),
        .iact_needed_cycles(iact_needed_cycles),
        .kernels_per_calc(kernels_per_calc),
        .y_lines_per_calc(y_lines_per_calc),
        .output_cycles(output_cycles),
        .store_in_psum(store_in_psum),
        .max_pooling(max_pooling),
        .fully_connected_layer(fully_connected_layer),
        .choose_iact_buffer_output(choose_iact_buffer_output),
        .choose_iact_buffer_input(choose_iact_buffer_input),
        .iact_channels_per_pe_next_layer(iact_channels_per_pe_next_layer),
        .needed_psum_storage_cycles_reg(needed_psum_storage_cycles_reg),
        .iact_channel_max_cycles(iact_channel_max_cycles),
        .input_activations_reg(input_activations),
        .filters_reg(filters_reg),
        .needed_x_cls_reg(needed_x_cls_reg),
        .needed_y_cls_reg(needed_y_cls_reg),
        .needed_iact_cycles_reg(needed_iact_cycles_reg),
        .wght_addr_len_reg(wght_addr_len_reg),
        .iact_addr_len_reg(iact_addr_len_reg),
        .send_data_out(send_data_out),
        .needed_iact_buffer_words_reg(needed_iact_buffer_words_reg),
        .add_up_reg(add_up),
        .iact_x_line_repetitions_reg(iact_x_line_repetitions)
    );


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
        .ROUTER_MODES_PSUM(ROUTER_MODES_PSUM)

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
        .psum_data_i  (psum_buffer_SP_data_r),
        .psum_enable_i(psum_enable_i_reg),
        .psum_ready_o (psum_ready_o_reg),

        .psum_data_o  (psum_data_o_w),
        .psum_enable_o(psum_enable_o),
        .psum_ready_i (psum_ready_i_reg),

        //Ports for Hyperparameters
        .status_reg_enable_i          (status_reg_enable_reg),
        .data_mode_i                  (data_mode_reg),
        .fraction_bit_i               (fraction_bit_reg),
        .needed_cycles_i              (needed_cycles_reg),
        .needed_x_cls_i               (needed_x_cls_reg),
        .needed_y_cls_i               (needed_y_cls_reg),
        .needed_iact_cycles_i         (needed_iact_cycles_reg),
        .iact_size_x_i                (iact_size_x),
        .filters_i                    (filters_reg),
        .iact_addr_len_i              (iact_addr_len_reg),
        .wght_addr_len_i              (wght_addr_len_reg),
        .bano_cluster_mode_i          (bano_cluster_mode_reg),
        .af_cluster_mode_i            (af_cluster_mode_reg),
        .pooling_cluster_mode_i       ({NUM_GLB_PSUM{1'd0}}),
        .kernel_per_pe_cluster_i      (kernel_per_pe_cluster_reg[$clog2(NUM_GLB_WGHT)-1:0]),
        .iact_x_line_repetitions_i    (iact_x_line_repetitions[3:0]),
        .kernel_size_i                (kernel_size),
        .input_activations_i          (input_activations),
        .stride_x_i                   (stride_x_reg),
        .stride_y_i                   (stride_y_reg),
        .delay_psum_glb_i             (psum_delay_reg),
        .compute_mask_i               (compute_mask_reg_port),
        .router_mode_iact_i           (router_mode_iact),
        .router_mode_wght_i           (router_mode_wght),
        .router_mode_psum_i           (router_mode_psum),
        .needed_psum_storage_cycles_i (needed_psum_storage_cycles_reg),
        .needed_iact_channel_cycles_i (iact_channel_max_cycles),
        .psum_transmitted_i           (psum_transmitted)
    );
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
