// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: OpenEye_FPGA
///
/// The OpenEye_FPGA is used for implementation, that are limited in their ports. One example is
/// the use of an FPGA. It uses OpenEye_Parallel.v as a submodul and communicates via handshake
/// protocol. As delay of data can occur, the varlenFIFO will buffer the output data.
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
///   PE_ROWS                - Amount of rows of process elements
///   PE_COLUMNS             - Amount of columns of process elements
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
      parameter PE_COLUMNS    = 4,
      parameter NUM_GLB_PSUM  = 4,
      parameter NUM_GLB_WGHT  = 3,
      parameter PE_ROWS       = 3,
  `else
    `include "parameters.vh"
      // Defaultwerte
  `endif
    parameter IS_TOPLEVEL   = 1,
    parameter SERIAL        = 1,
    parameter PARALLEL_MACS = 2,

    parameter ADDR_IACT_BITWIDTH = 4,
    parameter ADDR_WGHT_BITWIDTH = 8,

    parameter DATA_IACT_BITWIDTH = 8,
    parameter DATA_PSUM_BITWIDTH = 20,
    parameter DATA_WGHT_BITWIDTH = 8,

    parameter TRANS_BITWIDTH_IACT = 24,
    parameter TRANS_BITWIDTH_WGHT = 24,
    parameter TRANS_BITWIDTH_PSUM = 20,
    parameter DATA_IACT_OVERHEAD  = 4,

    parameter PES = PE_COLUMNS * PE_ROWS,

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
    parameter FSM_CYCLE_MAX = 4294967295,

    parameter BUFFER_WIDTH = 12,
    parameter BUFFER_WIDTH_IACT_STREAM_CONSTRUCTOR = (CLUSTER_ROWS == 8) ? BUFFER_WIDTH + 1 : BUFFER_WIDTH + 2,
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

    parameter PARAMETER_POS_1_0  = 0,
    parameter PARAMETER_POS_1_1  = 1 + PARAMETER_POS_1_0,
    parameter PARAMETER_POS_1_2  = $clog2(DATA_PSUM_BITWIDTH) + PARAMETER_POS_1_1,
    parameter PARAMETER_POS_1_3  = 1 + PARAMETER_POS_1_2,
    parameter PARAMETER_POS_1_4  = 1 + PARAMETER_POS_1_3,
    parameter PARAMETER_POS_1_5  = 8 + PARAMETER_POS_1_4,
    parameter PARAMETER_POS_1_6  = 2 + PARAMETER_POS_1_5,
    parameter PARAMETER_POS_1_7  = 4 + PARAMETER_POS_1_6,
    parameter PARAMETER_POS_1_8  = 4 + PARAMETER_POS_1_7,
    parameter PARAMETER_POS_1_9  = $clog2(PSUM_PER_PE + 1) + PARAMETER_POS_1_8,
    parameter PARAMETER_POS_1_10 = $clog2(IACT_ADDR_PER_PE + 1) + PARAMETER_POS_1_9,
    parameter PARAMETER_POS_1_11 = $clog2(WGHT_ADDR_PER_PE + 1) + PARAMETER_POS_1_10,
    parameter PARAMETER_POS_1_12 = $clog2(IACT_PER_PE + 1) + PARAMETER_POS_1_11,

    parameter PARAMETER_POS_2_0  = 0,
    parameter PARAMETER_POS_2_1  = 2 + PARAMETER_POS_2_0,
    parameter PARAMETER_POS_2_2  = 4 + PARAMETER_POS_2_1,
    parameter PARAMETER_POS_2_3  = 4 + PARAMETER_POS_2_2,
    parameter PARAMETER_POS_2_4  = 4 + PARAMETER_POS_2_3,
    parameter PARAMETER_POS_2_5  = 1 + PARAMETER_POS_2_4,
    parameter PARAMETER_POS_2_6  = 1 + PARAMETER_POS_2_5,
    parameter PARAMETER_POS_2_7  = 1 + PARAMETER_POS_2_6,
    parameter PARAMETER_POS_2_8  = 4 + PARAMETER_POS_2_7,
    parameter PARAMETER_POS_2_9  = 4 + PARAMETER_POS_2_8,
    parameter PARAMETER_POS_2_10 = 4 + PARAMETER_POS_2_9,
    parameter PARAMETER_POS_2_11 = 8 + PARAMETER_POS_2_10,
    parameter PARAMETER_POS_2_12 = 8 + PARAMETER_POS_2_11,

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

    localparam IACT_WORDS_IN_RAM = RAM_CELLS_WORD_BITWIDTH / DATA_IACT_BITWIDTH,
    localparam WORDS_PER_CYCLE   = 2

) (
    //Input DMA
    input clk_i,
    input rst_ni,

    //DEBUG OUTPUT STATES
    output reg [4-1:0] debug_fsm_current_state,
    output reg [4-1:0] debug_fsm_psum_state,

    //DEBUG OUTPUT IACT
    output reg                               debug_iact_we,
    output reg                               debug_iact_re,
    output reg [RAM_CELLS_ADDR_WIDTH-1:0]    debug_iact_addr,
    output reg [RAM_CELLS_WORD_BITWIDTH-1:0] debug_iact_data_i,
    output reg [RAM_CELLS_WORD_BITWIDTH-1:0] debug_iact_data_o,

    //DEBUG OUTPUT PSUM1

    output reg                             debug_psum_we,
    output reg                             debug_psum_re,
    output reg [BUFFER_WIDTH-1:0]          debug_psum_addr,
    output reg [TRANS_BITWIDTH_PSUM*2-1:0] debug_psum_data_i,
    output reg [TRANS_BITWIDTH_PSUM*2-1:0] debug_psum_data_o,

    //DEBUG OUTPUT PSUM2

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

`ifdef COCOTB_SIM
  initial begin
    if (IS_TOPLEVEL) begin
      $dumpfile("sim_build/OpenEye_FPGA.fst");
      $dumpvars(0, OpenEye_FPGA);
    end
  end
`endif

  //Register, that occupy hyperparameters
  reg data_mode_reg;
  reg [$clog2(DATA_PSUM_BITWIDTH)-1:0] fraction_bit_reg;
  reg [19:0] needed_cycles_reg;
  reg [$clog2(CLUSTER_COLUMNS+1)-1:0] needed_x_cls_reg;
  reg [$clog2(CLUSTER_ROWS+1)-1:0] needed_y_cls_reg;
  reg [3:0] needed_iact_cycles_reg;
  reg [$clog2(PSUM_PER_PE+1)-1:0] filters_reg;
  reg [$clog2(IACT_ADDR_PER_PE+1)-1:0] iact_addr_len_reg;
  reg [$clog2(WGHT_ADDR_PER_PE)-1:0] wght_addr_len_reg;
  reg [$clog2(BANO_MODES)*NUM_GLB_PSUM-1:0] bano_cluster_mode_reg;
  reg [$clog2(AF_MODES)-1:0] af_cluster_mode_reg;
  reg [$clog2(IACT_PER_PE+1)-1:0] input_activations_reg;
  reg [7:0] wght_cycles_reg;
  reg [2:0] stride_x_reg;
  reg [2:0] stride_y_reg;
  reg skipIact_reg;
  reg skipWght_reg;
  reg skipPsum_reg;
  reg [4-1:0] kernel_per_pe_cluster_reg;
  reg [3:0] kernel_size;
  reg [3:0] padding_reg;
  reg [DMA_BITWIDTH-1 : 0] fifo_data_i;
  reg fifo_read_i;
  reg fifo_write_i;
  reg [3:0] psum_delay_reg;
  reg [CLUSTERS-1:0] conv_array_reg;
  reg [CLUSTERS-1:0] param_array_reg;
  reg [7:0]needed_psum_storage_cycles_reg;
  //Register for the FSM
  reg [32-1:0] fsm_cycle;
  reg [$clog2(CLUSTER_COLUMNS)-1:0] fsm_x_cl;
  reg [$clog2(CLUSTER_ROWS)-1:0] fsm_y_cl;
  reg [$clog2(NUM_GLB_IACT)-1:0] fsm_iact_r;
  reg [$clog2(NUM_GLB_WGHT)-1:0] fsm_wght_r;
  reg [$clog2(NUM_GLB_PSUM)-1:0] fsm_psum_r;
  reg results_ready;
  reg [19:0] finished_cycles;
  reg new_stream;
  reg reset_cycle_reg;
  reg send_data_out;


  // Register for the Buffer
  reg buffer_select;
  reg iact_buffer_SP_en_r;
  reg iact_buffer_SP_en_w;
  reg [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_buffer_SP_data_w;
  reg [8-1:0] buffer_SP_addr_upper_limit;
  reg [8-1:0] buffer_SP_addr_lower_limit;
  reg [8-1:0] limit_increase_reg;
  wire [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_buffer_SP_data_r;

  reg wght_buffer_SP_en_r;
  reg wght_buffer_SP_en_w;
  reg [BUFFER_WIDTH:0] wght_buffer_SP_wr_addr;
  reg [BUFFER_WIDTH:0] wght_buffer_SP_rd_addr;
  reg [BUFFER_WIDTH:0] wght_buffer_SP_rd_addr_storage;
  reg [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_buffer_SP_data_w;
  reg [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_buffer_SP_data_r;

  reg [CLUSTERS*NUM_GLB_PSUM/2-1:0]psum_buffer_SP_en_r;
  reg [CLUSTERS*NUM_GLB_PSUM/2-1:0]psum_buffer_SP_en_w;
  reg [BUFFER_WIDTH*CLUSTERS*NUM_GLB_PSUM/2-1:0] psum_buffer_SP_addr;
  reg [BUFFER_WIDTH-1:0] psum_buffer_SP_addr_storage;
  reg [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_buffer_SP_data_w;
  wire [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_buffer_SP_data_r;

  reg [BUFFER_WIDTH:0] wght_cnt;
  reg [BUFFER_WIDTH-1:0] psum_cnt;

  // Register for IACT Stream
  reg [RAM_CELLS_ADDR_WIDTH-2:0] current_buffer_addr;
  reg [ 7:0] current_buffer_n;
  reg [ 7:0] current_buffer_n_1;
  reg [ 7:0] current_channel;
  reg [ 7:0] iact_size_x;
  reg [ 7:0] iact_size_y;
  reg [ 7:0] iact_channels;
  reg [ 7:0] iact_channels_per_pe;
  reg [ 3:0] iact_channels_per_pe_next_layer;
  reg [ 7:0] iact_channels_counter;
  reg [ 7:0] iact_channel_max_cycles;
  reg [10:0] iact_needed_cycles;

  reg [ 3:0] iact_router_counter;

  // Register for IACT Converter Buffer
  reg buffer_SP_en_r_reg[RAM_CELLS-1:0];
  reg buffer_SP_en_w_reg[RAM_CELLS-1:0];
  reg choose_iact_buffer;
  reg fully_connected_layer;
  reg max_pooling;
  reg [RAM_CELLS_ADDR_WIDTH-2:0] buffer_SP_addr_reg[RAM_CELLS-1:0];
  reg [RAM_CELLS_WORD_BITWIDTH-1:0] buffer_SP_data_w_reg[RAM_CELLS-1:0];

  // Registers for IACT Converter
  reg [35:0] iact_converter_params_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg iact_converter_en_cfg_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg iact_converter_en_store_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg iact_converter_en_enc_reg[CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [7:0] x_lines_reg;
  reg direct_cycling_reg;
  reg send_data_reg;
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

  reg [5:0] converter_needed_cycles;

  //New Iact Converter
  reg [7:0] iact_converter_max_cycles;
  reg [7:0] min_standing_cycles;
  reg [7:0] iact_converter_buffer_addr_max_cycles;
  reg [7:0] iact_converter_cycles;
  reg [7:0] iact_converter_buffer_addr_cycles;


  // Register, that configure the chip

  reg status_reg_enable_reg;

  reg compute_reg;
  reg [192-1:0] compute_mask_reg; //Clusters * PEs in Cluster
  wire [CLUSTERS * PES -1:0]compute_mask_reg_port;
  assign compute_mask_reg_port = compute_mask_reg[CLUSTERS * PES -1:0];
  reg [ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0] router_mode_iact_reg;
  reg [ROUTER_MODES_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] router_mode_wght_reg;
  reg [ROUTER_MODES_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] router_mode_psum_reg;

  wire [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_data_i_w;
  assign wght_data_i_w = wght_buffer_SP_data_r;
  reg [CLUSTERS*NUM_GLB_WGHT-1:0] wght_enable_i_reg;
  reg [CLUSTERS*NUM_GLB_WGHT-1:0] wght_ready_o_reg;

  wire [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_data_i_wire;
  wire [CLUSTERS*NUM_GLB_IACT-1:0] iact_enable_i_wire;
  wire [CLUSTERS*NUM_GLB_IACT-1:0] iact_ready_o_wire;
  wire [PES*CLUSTERS*$clog2(NUM_GLB_IACT+1)-1:0] iact_choose_i;

  reg [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_i_reg;
  reg [CLUSTERS*NUM_GLB_PSUM-1:0] psum_enable_i_reg;
  reg [CLUSTERS*NUM_GLB_PSUM-1:0] psum_ready_o_reg;

  wire [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_o_w;
  reg [CLUSTERS*NUM_GLB_PSUM-1:0] psum_enable_o_reg;
  reg [CLUSTERS*NUM_GLB_PSUM-1:0] psum_ready_i_reg;

  reg [7:0] needed_wght_cycles_reg;
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
  localparam START_CONVERTER = 4'd7;
  localparam CONVERT_IACT = 4'd8;
  localparam WAIT_CYCLE = 4'd9;
  localparam WAIT_FOR_RESULTS = 4'd10;
  localparam RECEIVE_PSUMS_TO_IACT = 4'd11;
  localparam MAXPOOLING_READ = 4'd12;
  localparam MAXPOOLING_SEND = 4'd13;

  reg [3:0] fsm_current_state;
  reg [3:0] fsm_last_state;

  //#######################
  //Process
  //#######################
  //Process for manging counter regs
  /*
  reg single_iteration;
  reg single_iteration2;
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin
      iact_channels_counter <= 0;
      single_iteration2     <= 0;
    end else begin
      case (fsm_current_state)

        CONVERT_IACT: begin
          if (iact_converter_buffer_addr_cycles == (iact_converter_buffer_addr_max_cycles - 1)) begin
            if (iact_converter_cycles == (iact_converter_max_cycles - 1)) begin
              iact_channels_counter <= iact_channels_counter + 1;
              if (iact_channels_counter == (iact_channel_max_cycles - 1)) begin
                iact_channels_counter  <= 0;
              end
            end
          end
        end
        WAIT_FOR_RESULTS: begin
          if (single_iteration & (single_iteration2 == 0)) begin
            single_iteration2 <= 1;
            iact_channels_counter <= iact_channels_counter + 1;
            if (iact_channels_counter == iact_channel_max_cycles - 1) begin
              iact_channels_counter <= 0;
            end
          end
          if (single_iteration == 0) begin
            single_iteration2     <= 0;
          end 
        end
        default: begin
          iact_channels_counter <= iact_channels_counter;
          single_iteration2     <= single_iteration2;
        end
      endcase
    end
  end
  */
  //Process for sending parameters to Iact Converters
  reg [7:0] fsm_iact_params;
  reg [7:0] iact_converter_x;
  reg [7:0] iact_converter_y;
  reg [7:0] iact_converter_c;
  // additional register for fsm
  reg [$clog2(CLUSTER_ROWS+1)-1:0]    fsm_row;
  reg [$clog2(CLUSTER_ROWS+1)-1:0]    fsm_row_offset;
  integer a, b, word;
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin
      param_array_reg                 <= 0;
      fsm_iact_params                 <= 0;
      iact_converter_x                <= 0;
      iact_converter_y                <= 0;
      iact_converter_c                <= 0;
      fsm_row                         <= 0;
      fsm_row_offset                  <= 0;
      for (a = 0; a < CLUSTER_COLUMNS; a++) begin
        for (b = 0; b < CLUSTER_ROWS; b++) begin
          iact_converter_params_reg[a][b] <= 0;
          iact_converter_en_cfg_reg[a][b] <= 0;
        end
      end
    end else begin
      if (GET_ROUTER_CONFIG == fsm_current_state) begin
        param_array_reg <= (1 << (iact_size_x / PE_COLUMNS)) - 1;
        fsm_iact_params <= CLUSTER_ROWS;
      end else begin
        if ((GET_WGHT == fsm_current_state) | (GET_IACT == fsm_current_state)) begin
          if (fsm_iact_params > 0) begin
            fsm_iact_params  <= fsm_iact_params - 1;
            iact_converter_x <= iact_converter_x + (PE_COLUMNS * 2);
            if (iact_converter_x + (PE_COLUMNS * 2) >= iact_size_x) begin
              iact_converter_x <= 0;
              iact_converter_c <= iact_converter_c + iact_channels_per_pe;
              if (iact_converter_c + iact_channels_per_pe == iact_channels) begin
                iact_converter_c <= 0;
                iact_converter_y <= iact_converter_y + 1;
                if (iact_converter_y + 1 >= iact_size_y) begin
                iact_converter_y <= 0;
                end
              end
            end
            for (a = 0; a < CLUSTER_COLUMNS; a++) begin 
              iact_converter_params_reg[a
              ][fsm_row[$clog2(
                  CLUSTER_ROWS
              )-1:0]][35:32] <= fsm_row_offset;
              if (fully_connected_layer) begin
                iact_converter_params_reg[a
                ][fsm_row[$clog2(
                    CLUSTER_ROWS
                )-1:0]][31:24] <= iact_converter_x;
              end else begin
                iact_converter_params_reg[a
                ][fsm_row[$clog2(
                    CLUSTER_ROWS
                )-1:0]][31:24] <= iact_converter_x + a[7:0] * PE_COLUMNS[7:0];
              end

              iact_converter_params_reg[a
              ][fsm_row[$clog2(
                  CLUSTER_ROWS
              )-1:0]][23:16] <= iact_converter_y;

              iact_converter_params_reg[a
              ][fsm_row[$clog2(
                  CLUSTER_ROWS
              )-1:0]][15:8] <= iact_size_x;

              iact_converter_params_reg[a
              ][fsm_row[$clog2(
                  CLUSTER_ROWS
              )-1:0]][7:0] <= iact_converter_c;

              iact_converter_en_cfg_reg[a
              ][fsm_row[$clog2(
                  CLUSTER_ROWS
              )-1:0]] <= 1;
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
          end
        end else begin
          for (a = 0; a < CLUSTER_COLUMNS; a++) begin
            for (b = 0; b < CLUSTER_ROWS; b++) begin
              iact_converter_en_cfg_reg[a][b] <= 0;
            end
          end
          if (iact_converter_params_enable) begin
            for (a = 0; a < CLUSTER_COLUMNS; a++) begin
              for (b = 0; b < CLUSTER_ROWS; b++) begin
                if (param_array_reg[(a+(b*CLUSTER_COLUMNS))] == 1) begin
                  if (iact_converter_cycles < ((iact_converter_max_cycles - 1))) begin //Include Padding
                    iact_converter_en_cfg_reg[a][b] <= 1;
                  end
                end
              end
            end
            param_array_reg <= (param_array_reg<<(iact_size_x/PE_COLUMNS) | param_array_reg>>(CLUSTERS-(iact_size_x/PE_COLUMNS)));
          end
          if ((iact_converter_params_enable) | (fsm_iact_params > 0)) begin
            if (fsm_iact_params > 0) begin
              fsm_iact_params <= fsm_iact_params - 1;
            end
            if (iact_converter_params_enable & (fsm_current_state == CONVERT_IACT)) begin
              fsm_iact_params <= fsm_iact_params + (iact_size_x / (2 * PE_COLUMNS));
            end
            if (fsm_iact_params > 0) begin
              iact_converter_x <= iact_converter_x + (PE_COLUMNS * 2);
              if (iact_converter_x >= iact_size_x - (PE_COLUMNS * 2)) begin
                iact_converter_x <= 0;
                iact_converter_c <= iact_converter_c + iact_channels_per_pe;
                if (iact_converter_c == iact_channels - iact_channels_per_pe) begin
                  iact_converter_c <= 0;
                  iact_converter_y <= iact_converter_y + 1;
                  if (iact_converter_y >= iact_size_y - 1) begin
                  iact_converter_y <= 0;
                  end
                end
              end
              for (a = 0; a < CLUSTER_COLUMNS; a++) begin 
                iact_converter_params_reg[a
                ][fsm_row[$clog2(
                    CLUSTER_ROWS
                )-1:0]][31:24] <= iact_converter_x + a[7:0] * PE_COLUMNS[7:0];

                iact_converter_params_reg[a
                ][fsm_row[$clog2(
                    CLUSTER_ROWS
                )-1:0]][23:16] <= iact_converter_y;

                iact_converter_params_reg[a
                ][fsm_row[$clog2(
                    CLUSTER_ROWS
                )-1:0]][15:8] <= iact_size_x;

                iact_converter_params_reg[a
                ][fsm_row[$clog2(
                    CLUSTER_ROWS
                )-1:0]][7:0] <= iact_converter_c;
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
            end
          end else begin
            param_array_reg <= conv_array_reg;
          end
        end
        if (reset_cycle_reg) begin
          param_array_reg  <= 0;
          fsm_iact_params  <= 0;
          iact_converter_x <= 0;
          iact_converter_y <= 0;
          iact_converter_c <= 0;
          fsm_row          <= 0;
          fsm_row_offset   <= 0;
          for (a = 0; a < CLUSTER_COLUMNS; a++) begin
            for (b = 0; b < CLUSTER_ROWS; b++) begin
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
      for (a = 0; a < CLUSTER_COLUMNS; a++) begin
        for (b = 0; b < CLUSTER_ROWS; b++) begin
          iact_converter_en_store_reg[a][b] <= 0;
        end
      end
    end else begin
      if (fsm_current_state == GET_PARAMETERS) begin
        conv_array_reg <= 0;
        for (a = 0; a < CLUSTER_COLUMNS; a++) begin
          for (b = 0; b < CLUSTER_ROWS; b++) begin
            iact_converter_en_store_reg[a][b] <= 0;
          end
        end
      end
      if (GET_ROUTER_CONFIG == fsm_current_state) begin
        conv_array_reg <= (1 << (iact_size_x/PE_COLUMNS)) - 1;
        if (fully_connected_layer) begin
          conv_array_reg <= ~0;
        end
      end
      for (a = 0; a < CLUSTER_COLUMNS; a++) begin
        for (b = 0; b < CLUSTER_ROWS; b++) begin
          iact_converter_en_store_reg[a][b] <= 0;
        end
      end
      if (iact_converter_enc_enable) begin
        for (a = 0; a < CLUSTER_COLUMNS; a++) begin
          for (b = 0; b < CLUSTER_ROWS; b++) begin
            if (conv_array_reg[(a+(b*CLUSTER_COLUMNS))] == 1) begin
              if (iact_converter_cycles < ((iact_converter_max_cycles - 1))) begin //Include Padding
                iact_converter_en_store_reg[a][b] <= 1;
              end
            end
          end
        end
        conv_array_reg <= (conv_array_reg<<(iact_size_x/PE_COLUMNS) | conv_array_reg>>(CLUSTERS-(iact_size_x/PE_COLUMNS)));
      end
      if (reset_cycle_reg) begin
        conv_array_reg <= 0;
        for (a = 0; a < CLUSTER_COLUMNS; a++) begin
          for (b = 0; b < CLUSTER_ROWS; b++) begin
            iact_converter_en_store_reg[a][b] <= 0;
          end
        end
      end
    end
  end
  reg                             sending_data;
  reg                             wght_sendable;
  reg                             single_iteration;
  reg                             single_iteration3;
  reg [                     19:0] current_cycle;
  reg [                     15:0] iact_cycle_count;
  reg [                     12:0] fsm_sending_cycle;
  reg [CLUSTERS*NUM_GLB_WGHT-1:0] flat_help_var_send;
  reg [CLUSTERS*NUM_GLB_WGHT-1:0] temp_var;
  reg [7:0] loop_debug;
  reg [63:0] prepared_iact [31:0];
  localparam EXTENDEDBITS = 48 - NUM_GLB_WGHT;
  //Process for sending data to OpenEye
  wire [CLUSTERS*NUM_GLB_IACT-1:0] iact_ready_o_oep_w;
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin
      //Reset Registers
      loop_debug <= 0;
      sending_data                   <= 0;
      single_iteration               <= 0;
      single_iteration3              <= 0;
      current_cycle                  <= 0;
      fsm_sending_cycle              <= 0;
      wght_enable_i_reg              <= 0;
      wght_buffer_SP_en_r            <= 0;
      wght_buffer_SP_rd_addr         <= 0;
      wght_buffer_SP_rd_addr_storage <= 0;
      compute_reg                    <= 0;
      wght_sendable                  <= 0;
      iact_cycle_count               <= 0;
      flat_help_var_send              = 0;
      for (a = 0; a < RAM_CELLS; a++) begin
        prepared_iact[a]             <= 0;
      end
      for (a = 0; a < CLUSTER_COLUMNS; a++) begin
        for (b = 0; b < CLUSTER_ROWS; b++) begin
          iact_converter_en_enc_reg[a][b] <= 0;
        end
      end
    end else begin
      //Set Registers to 0
      for (a = 0; a < CLUSTER_COLUMNS; a++) begin
        for (b = 0; b < CLUSTER_ROWS; b++) begin
          iact_converter_en_enc_reg[a][b] <= 0;
        end
      end
      compute_reg <= 0;
      if (send_data_reg | sending_data) begin
        sending_data <= 1;
        fsm_sending_cycle <= fsm_sending_cycle + 1;
        if (!sending_data) begin
          for (a = 0; a < CLUSTER_COLUMNS; a++) begin
            for (b = 0; b < CLUSTER_ROWS; b++) begin
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
            for (a = 0; a < CLUSTER_ROWS; a++) begin
              if ((a * (PE_COLUMNS * CLUSTER_COLUMNS)) <= (iact_size_x * iact_size_y) - 1) begin
                temp_var = {{EXTENDEDBITS{1'b0}}, {NUM_GLB_WGHT{1'b1}}};
                flat_help_var_send = flat_help_var_send + (temp_var << (a * NUM_GLB_WGHT));
                temp_var = 0;
              end
            end
            flat_help_var_send    = flat_help_var_send + (flat_help_var_send << (CLUSTER_ROWS * NUM_GLB_WGHT));
            wght_enable_i_reg <= flat_help_var_send[CLUSTERS*NUM_GLB_WGHT-1:0];
            flat_help_var_send = 0;
            if (fully_connected_layer) begin
              wght_enable_i_reg <= {
                  {(CLUSTER_ROWS-4){{NUM_GLB_WGHT{1'b0}}}},
                  {4{3'b1}},
                  {(CLUSTER_ROWS-4){{NUM_GLB_WGHT{1'b0}}}},
                  {4{3'b1}}
              };
            end
          end
          if (fsm_sending_cycle > wght_cnt + 2) begin
            fsm_sending_cycle      <= fsm_sending_cycle;
            wght_buffer_SP_en_r    <= 0;
            wght_enable_i_reg      <= 0;
            if (current_cycle == 0) begin
              compute_reg <= 1;
            end
          end
        end

        loop_debug <= 0;
        single_iteration3 <= 0;
        if (current_cycle < needed_cycles_reg - 1) begin
          if (iact_ready_o_oep_w == 0) begin
            if (!single_iteration) begin
              single_iteration  <= 1;
              single_iteration3 <= 1;
              current_cycle     <= current_cycle + 1;

              loop_debug <= 1;
              for (a = 0; a < CLUSTER_COLUMNS; a++) begin
                for (b = 0; b < CLUSTER_ROWS; b++) begin
                  iact_converter_en_enc_reg[a][b] <= 1;
                end
              end
              wght_sendable <= 1;
              if (iact_channel_max_cycles == 1) begin
                wght_sendable <= 0;
              end
              if (iact_channels_counter == iact_channel_max_cycles -1) begin
                loop_debug <= 2;
                if (iact_channel_max_cycles != 1) begin
                  wght_buffer_SP_rd_addr <= wght_buffer_SP_rd_addr_storage;
                end
                if (iact_router_counter == needed_y_cls_reg - 1) begin
                  wght_buffer_SP_rd_addr <= wght_buffer_SP_rd_addr;
                  wght_buffer_SP_rd_addr_storage <= wght_buffer_SP_rd_addr;
                  loop_debug <= 3;
                  iact_cycle_count <= iact_cycle_count + 1;
                  wght_sendable    <= 1;
                  if (iact_cycle_count == {{8 {1'd0}},needed_wght_cycles_reg} - 1) begin
                    loop_debug                     <= 4;
                    wght_buffer_SP_rd_addr_storage <= 0;
                    wght_buffer_SP_rd_addr         <= 0;
                    iact_cycle_count               <= 0;
                  end
                end
              end
            end

          end else begin
            single_iteration <= 0;
          end
        end else begin
          if (current_cycle == needed_cycles_reg - 1) begin
            if (iact_ready_o_oep_w == 0) begin
              if (!single_iteration) begin
                single_iteration  <= 1;
                single_iteration3 <= 1;
                current_cycle     <= current_cycle + 1;
              end
            end else begin
              single_iteration <= 0;
            end
          end
        end
        if (current_cycle == needed_cycles_reg) begin
          //sending_data      <= 0;
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
        single_iteration       <= 0;
        current_cycle          <= 0;
        fsm_sending_cycle      <= 0;
        wght_enable_i_reg      <= 0;
        wght_buffer_SP_en_r    <= 0;
        wght_buffer_SP_rd_addr <= 0;
        compute_reg            <= 0;
        wght_sendable          <= 1;
        flat_help_var_send = 0;
        for (a = 0; a < RAM_CELLS; a++) begin
          prepared_iact[a]             <= 0;
        end
        for (a = 0; a < CLUSTER_COLUMNS; a++) begin
          for (b = 0; b < CLUSTER_ROWS; b++) begin
            iact_converter_en_enc_reg[a][b] <= 0;
          end
        end
      end
      if (reset_cycle_reg) begin
        loop_debug <= 0;
        sending_data                   <= 0;
        single_iteration               <= 0;
        single_iteration3              <= 0;
        current_cycle                  <= 0;
        fsm_sending_cycle              <= 0;
        wght_enable_i_reg              <= 0;
        wght_buffer_SP_en_r            <= 0;
        wght_buffer_SP_rd_addr         <= 0;
        wght_buffer_SP_rd_addr_storage <= 0;
        compute_reg                    <= 0;
        wght_sendable                  <= 1;
        iact_cycle_count               <= 0;
        flat_help_var_send = 0;
        for (a = 0; a < CLUSTER_COLUMNS; a++) begin
          for (b = 0; b < CLUSTER_ROWS; b++) begin
            iact_converter_en_enc_reg[a][b] <= 0;
          end
        end
      end
    end
    temp_var = 0;
  end
  reg single_iteration2;
  reg [15:0] select_ram_counter;
  reg [15:0] ram_counter_storage;
  reg [ 7:0] select_ram_offset;
  reg signed [ 7:0] pooling_regs    [31:0];
  reg signed [ 7:0] pooling_stage_1 [7:0];
  reg signed [ 7:0] pooling_stage_2 [3:0];
  reg signed [ 7:0] pooling_stage_3 [1:0];
  reg signed [ 7:0] pooling_stage_4;

  wire [7:0] debug_pooling_regs0;
  wire [7:0] debug_pooling_regs1;
  wire [7:0] debug_pooling_regs2;
  wire [7:0] debug_pooling_regs3;
  wire [7:0] debug_pooling_regs4;
  wire [7:0] debug_pooling_regs5;
  wire [7:0] debug_pooling_regs6;
  wire [7:0] debug_pooling_regs7;
  wire [7:0] debug_pooling_regs12;
  wire [7:0] debug_pooling_regs15;
  wire [7:0] debug_pooling_regs31;
  wire [7:0] debug_pooling_stage_1;
  wire [7:0] debug_pooling_stage_2;
  wire [7:0] debug_pooling_stage_3;
  wire [7:0] debug_pooling_stage_4;

  assign debug_pooling_regs0 = pooling_regs[0];
  assign debug_pooling_regs1 = pooling_regs[1];
  assign debug_pooling_regs2 = pooling_regs[2];
  assign debug_pooling_regs3 = pooling_regs[3];
  assign debug_pooling_regs4 = pooling_regs[4];
  assign debug_pooling_regs5 = pooling_regs[5];
  assign debug_pooling_regs6 = pooling_regs[6];
  assign debug_pooling_regs7 = pooling_regs[7];
  assign debug_pooling_regs12 = pooling_regs[12];
  assign debug_pooling_regs15 = pooling_regs[15];
  assign debug_pooling_regs31 = pooling_regs[31];
  assign debug_pooling_stage_1 = pooling_stage_1[0];
  assign debug_pooling_stage_2 = pooling_stage_2[0];
  assign debug_pooling_stage_3 = pooling_stage_3[0];
  assign debug_pooling_stage_4 = pooling_stage_4;

  reg [ 6:0] quant_exp  [31:0];
  reg [24:0] quant_mant [31:0];
  integer cr, cc, g;
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin
      status_reg_enable_reg     <= 0;
      data_mode_reg             <= 0;
      fraction_bit_reg          <= 0;
      needed_cycles_reg         <= 0;
      needed_x_cls_reg          <= 0;
      needed_y_cls_reg          <= 0;
      needed_iact_cycles_reg    <= 0;
      filters_reg               <= 0;
      iact_addr_len_reg         <= 0;
      wght_addr_len_reg         <= 0;
      stride_x_reg              <= 0;
      stride_y_reg              <= 0;
      kernel_per_pe_cluster_reg <= 0;
      kernel_size               <= 0;
      padding_reg               <= 0;
      new_stream                <= 0;
      fsm_cycle                 <= 0;
      single_iteration2         <= 0;
      fsm_last_state            <= IDLE;
      fsm_current_state         <= GET_PARAMETERS;
      fsm_x_cl                  <= 0;
      fsm_y_cl                  <= 0;
      fsm_iact_r                <= 0;
      fsm_wght_r                <= 0;
      skipIact_reg              <= 0;
      skipWght_reg              <= 0;
      skipPsum_reg              <= 0;
      buffer_select             <= 0;
      needed_wght_cycles_reg    <= 0;
      fifo_data_i                       <= 0;
      fifo_read_i                       <= 0;
      fifo_write_i                      <= 0;
      bano_cluster_mode_reg             <= 0;
      af_cluster_mode_reg               <= 0;
      compute_mask_reg                  <= 0;
      psum_data_i_reg                   <= 0;
      psum_delay_reg                    <= 0;
      ready_dma_o                       <= 0;
      iact_buffer_SP_en_r               <= 0;
      iact_buffer_SP_en_w               <= 0;
      iact_buffer_SP_data_w             <= 0;
      wght_buffer_SP_wr_addr            <= 0;
      wght_buffer_SP_en_w               <= 0;
      wght_buffer_SP_data_w             <= 0;
      wght_cnt                          <= 0;
      iact_converter_buffer_addr_max_cycles <= 0;
      x_lines_reg                       <= 0;
      direct_cycling_reg                <= 0;
      wght_cycles_reg                   <= 0;
      // iact converter
      iact_out_reg                      <= 0;
      iact_ready                        <= 0;
      buffer_SP_addr_upper_limit        <= 0;
      limit_increase_reg                <= 0;
      buffer_SP_addr_lower_limit        <= 0;
      current_buffer_n                  <= 0;
      current_buffer_n_1                <= 0;
      current_buffer_addr               <= 0;
      current_channel                   <= 0;
      iact_size_x                       <= 0;
      iact_size_y                       <= 0;
      iact_channels                     <= 0;
      iact_channel_max_cycles           <= 0;
      iact_channels_per_pe              <= 0;
      iact_channels_per_pe_next_layer   <= 0;
      iact_channels_counter             <= 0;
      iact_needed_cycles                <= 1;  // params
      reset_cycle_reg                   <= 0;
      needed_psum_storage_cycles_reg    <= 0;
      select_ram_counter                <= 0;
      ram_counter_storage               <= 0;
      send_data_out                     <= 0;
      //new iact regs
      iact_converter_max_cycles       <= 0;
      min_standing_cycles               <= 0;
      iact_converter_cycles          <= 0;
      iact_converter_buffer_addr_cycles <= 0;
      send_data_reg                     <= 0;
      // Pooling
      for (a = 0; a < 32; a = a + 1) begin
        pooling_regs[a] <= 0;
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
        quant_exp[a]  <= 0;
        quant_mant[a] <= 0;
      end

      for (a = 0; a < RAM_CELLS; a++) begin
        buffer_SP_en_r_reg[a]   <= 0;
        buffer_SP_en_w_reg[a]   <= 0;
        buffer_SP_addr_reg[a]   <= 0;
        buffer_SP_data_w_reg[a] <= 0;
      end
      choose_iact_buffer    <= 0;
      fully_connected_layer <= 0;
      max_pooling           <= 0;
      converters_ready = 0;
      converter_needed_cycles      <= 0;
      iact_converter_enc_enable    <= 0;
      iact_converter_params_enable <= 0;

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
          reset_cycle_reg            <= 1;
          buffer_SP_addr_lower_limit <= 0;
          buffer_SP_addr_upper_limit <= 0;
          limit_increase_reg         <= 0;
          for (a = 0; a < 32; a = a + 1) begin
            buffer_SP_en_r_reg[a]   <= 0;
            buffer_SP_en_w_reg[a]   <= 0;
            buffer_SP_addr_reg[a]   <= 0;
            buffer_SP_data_w_reg[a] <= 0;
          end
          for (a = 0; a < 32; a = a + 1) begin
            pooling_regs[a] <= 0;
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
            quant_exp[a]  <= 0;
            quant_mant[a] <= 0;
          end
          if (enable_dma_i_reg) begin
            fsm_cycle       <= fsm_cycle + 1;
            reset_cycle_reg <= 0;
            case (fsm_cycle)
              32'd0: begin
                data_mode_reg <= data_dma_i_reg[PARAMETER_POS_1_0];
                fraction_bit_reg <= data_dma_i_reg[$clog2(
                    DATA_PSUM_BITWIDTH
                )-1+PARAMETER_POS_1_1:PARAMETER_POS_1_1];
                af_cluster_mode_reg <= {1'd0, data_dma_i_reg[PARAMETER_POS_1_2]};
                //needed_cycles_reg <= data_dma_i_reg[7+PARAMETER_POS_1_4:PARAMETER_POS_1_4]; //8 Bit free
                needed_x_cls_reg <= data_dma_i_reg[$clog2(
                    CLUSTER_COLUMNS+1
                )-1+PARAMETER_POS_1_5:PARAMETER_POS_1_5];
                needed_y_cls_reg <= data_dma_i_reg[$clog2(
                    CLUSTER_ROWS+1
                )-1+PARAMETER_POS_1_6:PARAMETER_POS_1_6];
                needed_iact_cycles_reg <= data_dma_i_reg[3+PARAMETER_POS_1_7:PARAMETER_POS_1_7];
                filters_reg <= data_dma_i_reg[$clog2(
                    PSUM_PER_PE+1
                )-1+PARAMETER_POS_1_8:PARAMETER_POS_1_8];
                iact_addr_len_reg <= data_dma_i_reg[3+PARAMETER_POS_1_9:PARAMETER_POS_1_9];
                wght_addr_len_reg <= data_dma_i_reg[3+PARAMETER_POS_1_10:PARAMETER_POS_1_10];
                input_activations_reg <= data_dma_i_reg[$clog2(
                    IACT_PER_PE+1
                )-1+PARAMETER_POS_1_11:PARAMETER_POS_1_11];
                send_data_out <= data_dma_i_reg[PARAMETER_POS_1_12];
              end
              32'd1: begin
                wght_cycles_reg           <= data_dma_i_reg[7+PARAMETER_POS_2_0:PARAMETER_POS_2_0];
                stride_x_reg              <= data_dma_i_reg[2+PARAMETER_POS_2_3:PARAMETER_POS_2_3];
                stride_y_reg              <= data_dma_i_reg[2+PARAMETER_POS_2_3:PARAMETER_POS_2_3];
                skipIact_reg              <= data_dma_i_reg[PARAMETER_POS_2_4:PARAMETER_POS_2_4];
                skipWght_reg              <= data_dma_i_reg[PARAMETER_POS_2_5:PARAMETER_POS_2_5];
                skipPsum_reg              <= data_dma_i_reg[PARAMETER_POS_2_6:PARAMETER_POS_2_6];
                psum_delay_reg            <= data_dma_i_reg[3+PARAMETER_POS_2_7:PARAMETER_POS_2_7];
                kernel_per_pe_cluster_reg <= data_dma_i_reg[3+PARAMETER_POS_2_8:PARAMETER_POS_2_8];
                kernel_size               <= data_dma_i_reg[3+PARAMETER_POS_2_9:PARAMETER_POS_2_9];
                x_lines_reg               <= data_dma_i_reg[7+PARAMETER_POS_2_10:PARAMETER_POS_2_10];
                needed_wght_cycles_reg    <= data_dma_i_reg[PARAMETER_POS_2_11+:8];
                needed_cycles_reg         <= {{2{1'd0}},data_dma_i_reg[PARAMETER_POS_2_12+:18]};
              end
              32'd2: begin
                padding_reg                           <= (kernel_size-1)/2;
                iact_converter_buffer_addr_max_cycles <= data_dma_i_reg[63:56];
                iact_channels_per_pe                  <= data_dma_i_reg[55:48];
                iact_size_y                           <= data_dma_i_reg[39:32];
                iact_size_x                           <= data_dma_i_reg[23:16];
                iact_needed_cycles                    <= data_dma_i_reg[10:0];
              end
              32'd3: begin
                max_pooling                     <= data_dma_i_reg[18];
                fully_connected_layer           <= data_dma_i_reg[17];
                choose_iact_buffer              <= data_dma_i_reg[16];
                iact_channels_per_pe_next_layer <= data_dma_i_reg[11:8];
                needed_psum_storage_cycles_reg  <= data_dma_i_reg[7:0] * needed_y_cls_reg;
                if (data_dma_i_reg[17]) begin
                  needed_psum_storage_cycles_reg <= data_dma_i_reg[7:0];
                end
                iact_channel_max_cycles         <= data_dma_i_reg[7:0];
              end
              32'd4: begin
                iact_channels <= iact_channels_per_pe * iact_channel_max_cycles;
                if (fully_connected_layer) begin
                  iact_channels <= iact_channels_per_pe * 4;
                end
                compute_mask_reg[DMA_BITWIDTH-1:0] <= data_dma_i_reg[DMA_BITWIDTH-1:0];
              end
              32'd5: begin
                compute_mask_reg[2*DMA_BITWIDTH-1:DMA_BITWIDTH] <= data_dma_i_reg[DMA_BITWIDTH-1:0];
              end
              32'd6: begin
                compute_mask_reg[3*DMA_BITWIDTH-1:2*DMA_BITWIDTH] <= data_dma_i_reg[DMA_BITWIDTH-1:0];
                fsm_last_state <= GET_PARAMETERS;
                fsm_current_state <= GET_ROUTER_CONFIG;
                fsm_cycle <= 0;
                if (fully_connected_layer) begin
                  iact_channel_max_cycles               <= 1;
                  kernel_size                           <= 1;
                  iact_converter_buffer_addr_max_cycles <= 2;
                  needed_y_cls_reg                      <= 1;
                  padding_reg                           <= 0;
                  needed_cycles_reg                     <= 1;
                end
              end
              default: begin
                fsm_last_state    <= GET_PARAMETERS;
                fsm_current_state <= GET_ROUTER_CONFIG;
                fsm_cycle         <= 0;
              end
            endcase
          end
        end

        GET_ROUTER_CONFIG: begin
          ready_dma_o <= 1;
          new_stream  <= 1;
          if (enable_dma_i_reg) begin
            fsm_cycle <= fsm_cycle + 1;
            if(fsm_cycle == FSM_CEIL_IACT_RTR_CCLS + FSM_CEIL_WGHT_RTR_CCLS + FSM_CEIL_PSUM_RTR_CCLS - 1) begin
              fsm_cycle             <= 0;
              fsm_last_state        <= GET_ROUTER_CONFIG;
              status_reg_enable_reg <= 0;
              if (!skipIact_reg) begin
                fsm_current_state <= GET_IACT;
              end else begin
                fsm_current_state <= GET_WGHT;
              end
              if (max_pooling) begin
                fsm_current_state <= GET_QUANTIZE;
              end
            end
          end
        end

        GET_IACT: begin
          ready_dma_o         <= 1;
          for (a = 0; a < RAM_CELLS; a++) begin
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

            if (fsm_cycle == ((iact_size_x*iact_size_y*iact_channels)/IACT_WORDS_IN_RAM) - 1) begin
              fsm_cycle         <= 0;
              fsm_current_state <= GET_WGHT;
              fsm_last_state    <= GET_IACT;
            end
          end else begin
            for (a = 0; a < RAM_CELLS; a++) begin
              buffer_SP_en_w_reg[a] <= 0;
            end
          end
        end
        GET_WGHT: begin
          ready_dma_o <= 1;
          for (a = 0; a < RAM_CELLS; a++) begin
            buffer_SP_en_w_reg[a] <= 0;
          end
          wght_buffer_SP_en_w <= 0;
          if (enable_dma_i_reg) begin
            iact_converter_max_cycles  <= (iact_size_y + {{4{1'd0}}, kernel_size}) - 8'b00000001;
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
              if ((fsm_y_cl == CLUSTER_ROWS - 1) | (fully_connected_layer & (fsm_y_cl == 4 - 1))) begin
                fsm_y_cl               <= 0;
                fsm_cycle              <= fsm_cycle + 1;
                wght_buffer_SP_en_w    <= 1;
                wght_buffer_SP_wr_addr <= wght_buffer_SP_wr_addr + 1;
                wght_cnt               <= ({7'd0,wght_cycles_reg} * ({9'd0,wght_addr_len_reg} + {7'd0,input_activations_reg} * ({7'd0,filters_reg} / PARALLEL_MACS[12:0]))) - 1;
                if(fsm_cycle == (wght_cycles_reg * ({27'd0,wght_addr_len_reg} + input_activations_reg * ({26'd0,filters_reg} / PARALLEL_MACS))) - 1)begin
                  wght_cnt  <= ({9'd0,wght_addr_len_reg} + input_activations_reg * ({7'd0,filters_reg} / PARALLEL_MACS[12:0]));
                  fsm_cycle <= 0;
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
          if (((psum_cnt != 0) & !fully_connected_layer) |
            (fully_connected_layer & (fsm_psum_cycle == 10 - 1) & (fsm_x_cl_psum == (CLUSTER_COLUMNS - 1)))) begin
            fsm_last_state         <= GET_BIAS;
            fsm_current_state      <= GET_QUANTIZE;
            wght_buffer_SP_wr_addr <= 0;
            limit_increase_reg     <= ((iact_size_x*iact_channels_per_pe)/(WORDS_PER_CYCLE[7:0]*4));
            if (fully_connected_layer) begin
              limit_increase_reg <= iact_channels_per_pe/2;
            end
            fsm_cycle              <= 0;
          end
        end

        GET_QUANTIZE: begin
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
            for (a = 0; a < RAM_CELLS; a++) begin
              buffer_SP_en_r_reg[a] <= 1;
            end
          end
          if (fsm_cycle == 16 - 1) begin
            fsm_cycle <= 0;
            ready_dma_o       <= 0;
            fsm_last_state    <= GET_QUANTIZE;
            fsm_current_state <= START_CONVERTER;
            if (max_pooling) begin
              fsm_cycle          <= -4;
              fsm_current_state  <= MAXPOOLING_READ;
              select_ram_counter <= 0;
            end
          end
        end
        
        START_CONVERTER: begin
          converters_ready     = 1;
          for (a = 0; a < CLUSTER_COLUMNS; a++) begin
            for (b = 0; b < CLUSTER_ROWS; b++) begin
              converters_ready = converters_ready & iact_converter_ready_w[a][b];
            end
          end
          if (converters_ready == 1) begin
            buffer_SP_addr_upper_limit <= (buffer_SP_addr_upper_limit + limit_increase_reg);
            fsm_current_state          <= CONVERT_IACT;
            select_ram_counter         <= 0;
            ram_counter_storage        <= 0;
            for (a = 0; a < RAM_CELLS; a++) begin
              buffer_SP_addr_reg[a] <= ~0;
            end
          end
          converter_needed_cycles <= kernel_size * iact_channels[6-1:0];
        end

        CONVERT_IACT: begin
          fsm_cycle <= fsm_cycle + 1;
          select_ram_counter  <= 0;
          if (select_ram_counter > 0) begin
            select_ram_counter <= select_ram_counter - 1;
          end
          if (select_ram_counter == 1) begin
            select_ram_counter <= {{8{1'd0}},iact_converter_buffer_addr_max_cycles};
            if ((iact_converter_cycles >= {{4{1'd0}},padding_reg}) & (iact_converter_cycles <= {{4{1'd0}},padding_reg} + iact_size_y - 1)) begin
              for (a = 0; a < RAM_CELLS; a++) begin
                buffer_SP_en_r_reg[a] <= 1;
                if (buffer_SP_addr_upper_limit > buffer_SP_addr_lower_limit) begin
                  if (((a >= buffer_SP_addr_lower_limit) & (a < buffer_SP_addr_upper_limit))) begin
                    buffer_SP_addr_reg[a] <= buffer_SP_addr_reg[a] + 1;
                  end
                end else begin
                  if (((a >= buffer_SP_addr_lower_limit) | (a < buffer_SP_addr_upper_limit))) begin
                    buffer_SP_addr_reg[a] <= buffer_SP_addr_reg[a] + 1;
                  end
                end
              end
              buffer_SP_addr_upper_limit <= ((buffer_SP_addr_upper_limit + limit_increase_reg)%RAM_CELLS);
              buffer_SP_addr_lower_limit <= ((buffer_SP_addr_lower_limit + limit_increase_reg)%RAM_CELLS);
            end
          end
          iact_converter_enc_enable    <= 0;
          iact_converter_params_enable <= 0;
          if ((iact_converter_buffer_addr_cycles + 2 == (iact_converter_buffer_addr_max_cycles)) & 
          (iact_converter_cycles == 0) & 
          (iact_channels_counter != (iact_channel_max_cycles))) begin
            iact_converter_enc_enable    <= 1;
            iact_converter_params_enable <= 1;
            select_ram_counter           <= 1;
          end
          iact_converter_buffer_addr_cycles <= iact_converter_buffer_addr_cycles + 1;
          if (iact_converter_buffer_addr_cycles == (iact_converter_buffer_addr_max_cycles - 1)) begin
            iact_converter_buffer_addr_cycles <= 0;
            iact_converter_cycles             <= iact_converter_cycles + 1;
            if (iact_converter_cycles == (iact_converter_max_cycles - 1)) begin
              iact_converter_cycles <= 0;
              iact_channels_counter <= iact_channels_counter + 1;
              if (iact_channels_counter == (iact_channel_max_cycles - 1)) begin
                fsm_current_state     <= WAIT_CYCLE;
                iact_channels_counter <= 0;
                fsm_cycle             <= 0;
              end
            end
          end
        end

        WAIT_CYCLE: begin
          iact_buffer_SP_data_w     <= iact_out_reg;
          fsm_cycle                 <= fsm_cycle + 1;
          iact_ready                <= 0;
          iact_converter_enc_enable <= 0;
          if (fsm_cycle == (4 * 2)) begin
            fsm_cycle                <= 0;
            send_data_reg            <= 1;
            fsm_last_state           <= WAIT_CYCLE;
            if (send_data_out) begin
              fsm_current_state <= WAIT_FOR_RESULTS;
            end else begin
              fsm_current_state          <= RECEIVE_PSUMS_TO_IACT;
              select_ram_offset          <= 0;
              ram_counter_storage        <= 0;
              select_ram_counter         <= 0;
              buffer_SP_addr_upper_limit <= (CLUSTER_ROWS * iact_channels_per_pe_next_layer)%32;
              buffer_SP_addr_lower_limit <= 0;
              for (a = 0; a < RAM_CELLS; a++) begin
                buffer_SP_data_w_reg[a] <= 0;
              end
            end
            iact_converter_cycles <= 0;
            current_buffer_n      <= 0;
            current_buffer_n_1    <= 0;
            current_buffer_addr   <= 0;
            current_channel       <= 0;
            for (a = 0; a < RAM_CELLS; a++) begin
              buffer_SP_addr_reg[a] <= 0;
            end
          end
        end

        RECEIVE_PSUMS_TO_IACT: begin
          if (fsm_psum_current_state == WAIT_FOR_SENDING_RESULTS) begin
            choose_iact_buffer       <= 0;
          end
          if (single_iteration & (single_iteration2 == 0)) begin
            single_iteration2 <= 1;
            iact_channels_counter <= iact_channels_counter + 1;
            if (iact_channels_counter == iact_channel_max_cycles - 1) begin
              iact_channels_counter <= 0;
            end
          end
          if (single_iteration == 0) begin
            single_iteration2 <= 0;
          end
          for (a = 0; a < RAM_CELLS; a++) begin
            buffer_SP_en_w_reg[a] <= 0;
            if (buffer_SP_en_w_reg[a] == 1) begin
              buffer_SP_addr_reg[a] <= buffer_SP_addr_reg[a] + 1;
            end
          end
          if (fsm_psum_current_state == SEND_PSUM_TO_IACT) begin
            fsm_cycle <= fsm_cycle + 1;
            if (fsm_cycle >= 1) begin
              select_ram_counter <= select_ram_counter + 1;
              if (iact_channels_per_pe_next_layer == 4) begin
                if (select_ram_counter == 8 - 1) begin
                  select_ram_counter <= 0;
                end
                if (select_ram_counter == (CLUSTER_ROWS + ram_counter_storage - 1)%8) begin
                  select_ram_counter <= ram_counter_storage;
                  iact_channels_counter <= iact_channels_counter + 1;
                  if (iact_channels_counter == {4'd0,iact_channels_per_pe_next_layer} - 1) begin
                    iact_channels_counter <= 0;
                    ram_counter_storage   <= select_ram_counter + 1;
                    select_ram_counter    <= select_ram_counter + 1;
                    if (select_ram_counter >= (32/4) - 1) begin
                      select_ram_counter  <= 0;
                      ram_counter_storage <= 0;
                    end
                    for (a = 0; a < RAM_CELLS; a++) begin
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
                    buffer_SP_addr_upper_limit <= (buffer_SP_addr_upper_limit + CLUSTER_ROWS * 4) % 32;
                  end
                end
                for (a = 0; a < RAM_CELLS; a++) begin
                  for (word = 0; word < 8; word++) begin
                    if ((a >= select_ram_counter * 4) & (a < 4 * select_ram_counter + 4)) begin
                      if ((word == (4 + {{24{1'd0}},iact_channels_counter})) | (word == {{24{1'd0}},iact_channels_counter})) begin
                        buffer_SP_data_w_reg[a][8*word+:8] <= quantized_value_reg[word / 4 + (a%4) * 2];
                      end
                    end
                  end
                end
              end
              if (iact_channels_per_pe_next_layer == 2) begin
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
                    if (select_ram_counter >= (32/2) - 1) begin
                      select_ram_counter  <= 0;
                      ram_counter_storage <= 0;
                    end
                    for (a = 0; a < RAM_CELLS; a++) begin
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
                for (a = 0; a < RAM_CELLS; a++) begin
                  for (word = 0; word < 8; word++) begin
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
              end
              if (iact_channels_per_pe_next_layer == 1) begin
                if (select_ram_counter == 32 - 1) begin
                  select_ram_counter <= 0;
                end
                if (select_ram_counter == (CLUSTER_ROWS + ram_counter_storage - 1)%32) begin
                  select_ram_counter    <= ram_counter_storage;
                  iact_channels_counter <= iact_channels_counter + 1;
                  if (iact_channels_counter == iact_channels_per_pe_next_layer - 1) begin
                    iact_channels_counter <= 0;
                    ram_counter_storage   <= select_ram_counter + 1;
                    select_ram_counter    <= select_ram_counter + 1;
                    if (select_ram_counter >= 32 - 1) begin
                      select_ram_counter  <= 0;
                      ram_counter_storage <= 0;
                    end
                    for (a = 0; a < RAM_CELLS; a++) begin
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
                    buffer_SP_addr_upper_limit <= (buffer_SP_addr_upper_limit + CLUSTER_ROWS) % 32;
                  end
                end
                for (a = 0; a < RAM_CELLS; a++) begin
                  for (word = 0; word < 8; word++) begin
                    if (a[15:0] == select_ram_counter) begin
                      buffer_SP_data_w_reg[a][8*word+:8] <= quantized_value_reg[word];
                    end
                  end
                end
              end
            end
          end
          if ((fsm_psum_current_state == PSUM_IDLE) & (fsm_cycle >= 1)) begin
            select_ram_counter <= 0;
            fsm_cycle          <= 0;
            for (a = 0; a < RAM_CELLS; a++) begin
              buffer_SP_en_w_reg[a] <= 0;
            end
            fsm_current_state     <= GET_PARAMETERS;
            fsm_last_state        <= RECEIVE_PSUMS_TO_IACT;
            send_data_reg         <= 0;
            iact_channels_counter <= 0;
          end
        end
        WAIT_FOR_RESULTS: begin
          if (fsm_psum_current_state == WAIT_FOR_SENDING_RESULTS) begin
            choose_iact_buffer       <= 0;
          end
          if (single_iteration & (single_iteration2 == 0)) begin
            single_iteration2 <= 1;
            iact_channels_counter <= iact_channels_counter + 1;
            if (iact_channels_counter == iact_channel_max_cycles - 1) begin
              iact_channels_counter <= 0;
            end
          end
          if (single_iteration == 0) begin
            single_iteration2 <= 0;
          end
          fsm_cycle <= fsm_cycle + 1;
          if (last_data_o) begin
            fsm_last_state        <= WAIT_FOR_RESULTS;
            fsm_current_state     <= GET_PARAMETERS;
            send_data_reg         <= 0;
            fsm_cycle             <= 0;
            iact_channels_counter <= 0;
          end
        end
        MAXPOOLING_READ: begin
          select_ram_counter <= select_ram_counter + 1;
          if (select_ram_counter == RAM_CELLS - 1) begin
            select_ram_counter <= 0;
          end
          fsm_cycle <= fsm_cycle + 1;
          if (fsm_cycle == iact_size_y * CLUSTER_ROWS - 1) begin
            fsm_cycle <= 0;
            iact_converter_cycles <= iact_converter_cycles + 1;
          end
          if (((select_ram_counter+2)%32) == 0) begin
            for (a = 0; a < RAM_CELLS; a++) begin
              buffer_SP_addr_reg[a] <= buffer_SP_addr_reg[a] + 1;
            end
          end
          for (a = 0; a < 8; a = a + 1) begin
            pooling_stage_1[a] <= buffer_SP_data_r[select_ram_counter*RAM_CELLS_WORD_BITWIDTH+8*a+:8];
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
          if (pooling_stage_3[0] >= pooling_stage_3[1]) begin
            pooling_stage_4 <= pooling_stage_3[0];
          end else begin
            pooling_stage_4 <= pooling_stage_3[1];
          end
          for (a = 0; a < 32; a = a + 1) begin
            if (a == iact_converter_cycles) begin
              if (pooling_regs[a] < pooling_stage_4) begin
                pooling_regs[a] <= pooling_stage_4;
              end
            end
          end
          if ((fsm_cycle == 1) & (iact_converter_cycles == (iact_channels))) begin
            select_ram_counter    <= 0;
            fsm_cycle             <= 0;
            iact_converter_cycles <= 0;
            fsm_last_state        <= MAXPOOLING_READ;
            fsm_current_state     <= MAXPOOLING_SEND;
          end
        end
        MAXPOOLING_SEND: begin
          fsm_cycle <= fsm_cycle + 1;
          for (a = 0; a < RAM_CELLS; a++) begin
            buffer_SP_addr_reg[a] <= 0;
          end
          for (a = 0; a < RAM_CELLS; a++) begin
            buffer_SP_en_w_reg[a] <= 0;
            if (fsm_cycle == a) begin
              buffer_SP_en_w_reg[a] <= 1;
              for (b = 0; b < IACT_WORDS_IN_RAM; b++) begin
                buffer_SP_data_w_reg[a][8*b+:8] <= pooling_regs[8*a + b];
              end
            end
          end
          if (fsm_cycle >= (iact_channel_max_cycles)/IACT_WORDS_IN_RAM - 1) begin
            fsm_cycle         <= 0;
            fsm_last_state    <= MAXPOOLING_SEND;
            fsm_current_state <= GET_PARAMETERS;
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
      router_mode_iact_reg              <= 0;
      router_mode_iact_storage          <= 0;
      router_mode_wght_reg              <= 0;
      router_mode_psum_reg              <= 0;
      iact_router_counter               <= 0;
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
                    router_mode_iact_reg[cc * CLUSTER_ROWS * NUM_GLB_IACT * ROUTER_MODES_IACT +
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
                      router_mode_wght_reg[cc * CLUSTER_ROWS * NUM_GLB_WGHT * ROUTER_MODES_WGHT +
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
                      router_mode_psum_reg[cc * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM +
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
        router_mode_iact_storage <= router_mode_iact_reg;
      end else begin
        if (fsm_current_state == WAIT_FOR_RESULTS | fsm_current_state == RECEIVE_PSUMS_TO_IACT) begin
          if (single_iteration3) begin
            if (iact_channels_counter == iact_channel_max_cycles -1) begin
              iact_router_counter      <= iact_router_counter + 1;
              if (iact_router_counter == needed_y_cls_reg - 1) begin
                iact_router_counter  <= 0;
                router_mode_iact_reg <= router_mode_iact_storage;
              end else begin
                for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                  for (int g=0; g<NUM_GLB_IACT; g=g+1) begin
                    router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+g*ROUTER_MODES_IACT+3] <= 0;
                    router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+g*ROUTER_MODES_IACT+4] <= 1;
                    router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+g*ROUTER_MODES_IACT+5] <= 1;
                  end
                end
                for (int cr=1; cr<CLUSTER_ROWS; cr=cr+1) begin
                  for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                    for (int g=0; g<NUM_GLB_IACT; g=g+1) begin
                      // If router is not on top of source already
                      if (!((router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT] == 1) &
                              (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] == 1) &
                              (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+2] == 0) &
                              (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+3] == 0) &
                              (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
                              (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 1))) begin
                        // If router is source
                        if ((router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 0) &
                              (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 0)) begin
                              // If router above is already destination
                              if ((router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
                                  (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 1)) begin
                                router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] <= 1;
                              end
                              router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+3] <= 0;
                              router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] <= 1;
                              router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] <= 1;
                        end else begin
                          if ((router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 0) &
                              (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 0)) begin
                            router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] <= 1;
                            router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] <= 0;
                            router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] <= 0;
                          end else begin
                            if ((router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
                                (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 1)) begin
                              router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] <= 1;
                              router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+3] <= 0;
                              router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] <= 1;
                              router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] <= 1;
                            end else begin
                              if ((router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+3] == 1) &
                                  (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
                                  (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 0)) begin

                                  router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] <= 1;
                                  router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] <= 0;
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
                        router_mode_psum_reg[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2] <=
                        router_mode_psum_reg[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+(cr-1)*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2];
                      end
                    end
                  end
                  if (storage_cycles_router != (needed_psum_storage_cycles_reg - 1)) begin
                    storage_cycles_router <= storage_cycles_router + 1;
                    for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                      for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
                        router_mode_psum_reg[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+g*ROUTER_MODES_PSUM+2] <= 0;
                      end
                    end
                    for (cr = 1; cr < CLUSTER_ROWS; cr = cr + 1) begin
                      for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                        for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
                          router_mode_psum_reg[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2] <=
                          router_mode_psum_reg[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+(cr-1)*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2];
                        end
                      end
                    end
                  end else begin
                    for (cc = 0; cc < CLUSTER_COLUMNS; cc = cc + 1) begin
                      for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
                        router_mode_psum_reg[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+g*ROUTER_MODES_PSUM+2] <= 1;
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
      if (reset_cycle_reg) begin
        router_mode_iact_reg              <= 0;
        router_mode_iact_storage          <= 0;
        iact_router_counter               <= 0;
        storage_cycles_router             <= 0;
        first_cycle                       <= 1;
        psum_choose_i_reg                 <= 0;
        iact_channels_counter_psum_router <= 0;
      end
    end
  end

localparam PSUM_IDLE = 0;
localparam WAIT_TO_SEND_READY_SIGNAL = 1;
localparam CALCULATE_PSUM = 2;
localparam PSUM_GET_RESULTS = 3;
localparam WAIT_FOR_SENDING_RESULTS = 4;
localparam PSUM_SEND_RESULTS = 5;
localparam SEND_PSUM_TO_IACT = 6;

reg [7:0]test_reg1;
reg [7:0]test_reg2;
reg [7:0]psum_sending_counter;
reg [7:0]psum_filter_offset;
reg [3:0]fsm_psum_row_offset;

reg [ 7:0] iact_channel_counter_reg;
reg [15:0] fsm_psum_cycle;
reg [ 3:0] fsm_psum_last_state;
reg [ 3:0] fsm_psum_current_state;
assign debug_fsm_psum_state = fsm_psum_current_state;
assign debug_fsm_current_state = fsm_current_state;
reg        psum_transmitted;
reg psum_router_set_reg;
reg start_new_cycle;
reg last_data_reg;
reg [$clog2(CLUSTER_COLUMNS)-1:0] fsm_x_cl_psum;
reg [   $clog2(CLUSTER_ROWS)-1:0] fsm_y_cl_psum;
reg [7:0] quantized_value_reg [7:0];
wire [7:0] testquant;
assign testquant = quantized_value_reg[0];
reg [7:0] current_filter;
  integer g_psum, b_psum, cc_psum, cr_psum;
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin  ///Reset
      psum_transmitted            <= 0;
      fsm_psum_cycle              <= 0;
      fsm_psum_last_state         <= PSUM_IDLE;
      fsm_psum_current_state      <= PSUM_IDLE;
      psum_buffer_SP_addr         <= ~0;
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
      fsm_x_cl_psum               <= 0;
      fsm_y_cl_psum               <= 0;
      psum_buffer_SP_en_r         <= 0;
      last_data_o                 <= 0;
      current_filter              <= 0;
      test_reg1                   <= 0;
      test_reg2                   <= 0;
      psum_sending_counter        <= 0;
      psum_filter_offset          <= 0;
      fsm_psum_row_offset         <= 0;
      for (cr_psum = 0; cr_psum < 8; cr_psum = cr_psum + 1) begin
        quantized_value_reg[cr_psum] <= 0;
      end
      fsm_psum_r                  <= 0;
      finished_cycles             <= 0;
      psum_buffer_SP_data_w       <= 0;
    end else begin
      case (fsm_psum_current_state)
        PSUM_IDLE: begin
          enable_dma_o         <= 0;
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
                if (((fsm_y_cl_psum) == CLUSTER_ROWS - 1) | fully_connected_layer) begin
                  fsm_y_cl_psum <= 0;
                  fsm_x_cl_psum <= fsm_x_cl_psum + 1;
                  if (fsm_x_cl_psum == (CLUSTER_COLUMNS - 1)) begin
                    fsm_x_cl_psum       <= 0;
                    fsm_psum_cycle      <= fsm_psum_cycle + 1;
                    psum_buffer_SP_en_w <= {((CLUSTERS * NUM_GLB_PSUM/2)){1'b1}};

                    for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                      for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                        for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                          psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH]
                          <= psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH] + 1;
                        end
                      end
                    end
                    if (((fsm_psum_cycle == (needed_wght_cycles_reg * filters_reg * iact_size_y) - 1) & (!fully_connected_layer))
                      | (fully_connected_layer & (fsm_psum_cycle == 10 - 1))) begin
                      fsm_psum_cycle <= 0;
                      psum_cnt       <= psum_buffer_SP_addr[11:0] + 1;
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
            psum_buffer_SP_addr    <= 0;
            fsm_psum_last_state    <= PSUM_IDLE;
            fsm_psum_current_state <= WAIT_TO_SEND_READY_SIGNAL;
            fsm_psum_cycle         <= 0;
          end
        end
        WAIT_TO_SEND_READY_SIGNAL: begin
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
        CALCULATE_PSUM: begin
          if (psum_ready_i_reg != 0) begin
            psum_ready_i_reg <= psum_ready_i_reg;
          end
          psum_buffer_SP_en_r <= {(NUM_GLB_PSUM*CLUSTERS/2){1'd1}};
          if (results_ready == 0 & (psum_ready_i_reg != 0)) begin
            results_ready = 1;
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                  results_ready = results_ready & (psum_ready_o_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum*2] |
                   (router_mode_psum_reg[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM * 2 + 2] == 0));
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
                  if ((fsm_psum_cycle != 0) & (router_mode_psum_reg[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM * 2 + 2] == 1)) begin
                    psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH]
                    <= psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH] + 1;
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
                    if (router_mode_psum_reg[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM * 2 + 2] == 1) begin
                      psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH] <= psum_buffer_SP_addr_storage;
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
                  psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH]
                  <= psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH] + 1;
                end
                results_ready = results_ready & (psum_enable_o_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum * 2] | 
                (router_mode_psum_reg[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM * 2 + 2] == 0));
                if (psum_enable_o_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum * 2] != 0) begin
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
            if ((finished_cycles == needed_cycles_reg - 1)) begin
              fsm_psum_last_state    <= PSUM_GET_RESULTS;
              fsm_psum_current_state <= WAIT_FOR_SENDING_RESULTS;
              psum_buffer_SP_addr    <= 0;
              psum_ready_i_reg       <= 0;
              fsm_psum_cycle         <= 0;
              psum_buffer_SP_en_w    <= 0;
              psum_buffer_SP_en_r    <= {(NUM_GLB_PSUM/2*CLUSTER_ROWS*CLUSTER_COLUMNS){1'd1}};
              psum_enable_i_reg      <= 0;
            end else begin
              finished_cycles <= finished_cycles + 1;
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
                      psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH] <= psum_buffer_SP_addr_storage + {{6{1'd0}}, filters_reg};
                    end
                  end
                end
              end else begin
                storage_cycles <= storage_cycles + 1;
                for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                  for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                    for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                      psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH] <= psum_buffer_SP_addr_storage;
                    end
                  end
                end
              end
              psum_ready_i_reg <= 0;
            end
          end
        end
        WAIT_FOR_SENDING_RESULTS: begin
          fsm_psum_cycle      <= 0;
          fsm_psum_last_state <= WAIT_FOR_SENDING_RESULTS;
          if (send_data_out) begin
            fsm_psum_current_state <= PSUM_SEND_RESULTS;
            psum_buffer_SP_en_r    <= {(NUM_GLB_PSUM/2*CLUSTER_ROWS*CLUSTER_COLUMNS){1'd1}};
          end else begin
            fsm_psum_cycle         <= fsm_psum_cycle + 1;
            fsm_y_cl_psum          <= 0;
            if (fsm_psum_cycle >= 2) begin
              fsm_psum_current_state <= SEND_PSUM_TO_IACT;
              fsm_psum_cycle         <= 0;
              test_reg1              <= 0;
              test_reg2              <= 0;
              psum_sending_counter   <= 0;
              psum_filter_offset     <= iact_channels_per_pe_next_layer;
              fsm_psum_row_offset    <= 0;
            end
          end
          fsm_psum_r             <= 0;
          fsm_y_cl_psum          <= 0;
          fsm_x_cl_psum          <= 0;
        end
        PSUM_SEND_RESULTS: begin
          psum_buffer_SP_en_r <= 0;
          if (ready_dma_i == 1) begin
            enable_dma_o <= 1;
            data_dma_o <= psum_buffer_SP_data_r[fsm_x_cl_psum*CLUSTER_ROWS*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+fsm_y_cl_psum*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+fsm_psum_r*PARALLEL_MACS*TRANS_BITWIDTH_PSUM+:TRANS_BITWIDTH_PSUM * PARALLEL_MACS];
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                  if ((fsm_psum_r == g_psum) & (fsm_x_cl_psum == cc_psum) & (fsm_y_cl_psum == cr_psum)) begin
                    psum_buffer_SP_en_r[cc_psum*CLUSTER_ROWS*NUM_GLB_PSUM/2+cr_psum*NUM_GLB_PSUM/2+g_psum] <= 1;
                    psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH]
                    <= psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH] + 1;
                  end
                end
              end
            end
            fsm_psum_r <= fsm_psum_r + 1;
            if ((fsm_psum_r == NUM_GLB_PSUM - 3) | fully_connected_layer) begin //NUM_GLB_PSUM / PARALLEL_MACS - 1
              fsm_psum_r <= 0;
              fsm_x_cl_psum <= fsm_x_cl_psum + 1;
              if (fsm_x_cl_psum == CLUSTER_COLUMNS - 1) begin
                fsm_x_cl_psum <= 0;
                fsm_y_cl_psum <= fsm_y_cl_psum + 1;
                if ((fsm_y_cl_psum == CLUSTER_ROWS - 1) | fully_connected_layer) begin
                  fsm_y_cl_psum <= 0;
                  fsm_psum_cycle <= fsm_psum_cycle + 1;
                  if (fsm_psum_cycle == needed_wght_cycles_reg * filters_reg * iact_size_y - 1) begin
                    fsm_psum_cycle      <= 0;
                    psum_buffer_SP_addr <= 0;
                    psum_buffer_SP_en_r <= 0;
                    last_data_o         <= 1;
                  end
                end
              end
            end
            if (last_data_o) begin
              psum_buffer_SP_en_r    <= 0;
              last_data_o            <= 0;
              enable_dma_o           <= 0;
              last_data_reg          <= 0;
              finished_cycles        <= 0;
              fsm_psum_last_state    <= PSUM_SEND_RESULTS;
              fsm_psum_current_state <= PSUM_IDLE;
              fsm_psum_r             <= 0;
              fsm_y_cl_psum          <= 0;
              fsm_x_cl_psum          <= 0;
            end
          end
        end
        SEND_PSUM_TO_IACT: begin
          for (cr_psum = 0; cr_psum < 4; cr_psum = cr_psum + 1) begin
            quantized_value_reg[2*cr_psum]     <= (quant_mant[current_filter] * psum_buffer_SP_data_r[(cr_psum/2)*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+fsm_y_cl_psum*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+(cr_psum%2)*TRANS_BITWIDTH_PSUM*2+:TRANS_BITWIDTH_PSUM]) >>> quant_exp[current_filter];
            quantized_value_reg[2*cr_psum + 1] <= (quant_mant[current_filter] * psum_buffer_SP_data_r[(cr_psum/2)*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+fsm_y_cl_psum*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+(cr_psum%2)*TRANS_BITWIDTH_PSUM*2+TRANS_BITWIDTH_PSUM+:TRANS_BITWIDTH_PSUM]) >>> quant_exp[current_filter];
          end
          psum_sending_counter <= psum_sending_counter + 1;
          if (psum_sending_counter == CLUSTER_ROWS - 1) begin
            psum_sending_counter        <= 0;
          end
          if (psum_sending_counter == CLUSTER_ROWS - 2) begin
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                  test_reg1 <= test_reg1 + 1;
                  if (test_reg1[3:0] != iact_channels_per_pe_next_layer - 1) begin
                    psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH]
                    <= psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH] + 1;
                  end else begin
                    test_reg1 <= 0;
                    test_reg2 <= test_reg2 + 1;
                    psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH]
                    <= psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH] + (needed_wght_cycles_reg * {6'd0,filters_reg} - {8'd0,iact_channels_per_pe_next_layer}) + 1;
                    if (test_reg2 == iact_size_y - 1) begin
                      test_reg2 <= 0;
                      psum_buffer_SP_addr[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM/2 * BUFFER_WIDTH + cr_psum * NUM_GLB_PSUM/2 * BUFFER_WIDTH + g_psum * BUFFER_WIDTH +: BUFFER_WIDTH]
                      <= {4'd0,psum_filter_offset};
                      psum_filter_offset <= psum_filter_offset + {4'd0,iact_channels_per_pe_next_layer};
                    end
                  end
                end
              end
            end
          end

          fsm_y_cl_psum <= fsm_y_cl_psum[2:0] + needed_y_cls_reg[2:0];
          if (fsm_y_cl_psum + needed_y_cls_reg >= CLUSTER_ROWS) begin
            fsm_y_cl_psum       <= fsm_psum_row_offset[2:0] + 1;
            fsm_psum_row_offset <= fsm_psum_row_offset + 1;
            if (fsm_psum_row_offset == needed_y_cls_reg - 1) begin
              fsm_y_cl_psum       <= 0;
              fsm_psum_row_offset <= 0;
            end
          end
          fsm_psum_cycle <= fsm_psum_cycle + 1;
          if (fsm_psum_cycle == 8 * needed_wght_cycles_reg * filters_reg * iact_size_y) begin
            fsm_psum_cycle              <= 0;
            fsm_psum_last_state         <= PSUM_SEND_RESULTS;
            fsm_psum_current_state      <= PSUM_IDLE;
            fsm_psum_r                  <= 0;
            fsm_y_cl_psum               <= 0;
            fsm_x_cl_psum               <= 0;
            psum_sending_counter        <= 0;
          end
        end
        default: begin
        end
      endcase
      if (status_reg_enable_reg) begin
        psum_transmitted            <= 0;
        psum_buffer_SP_addr         <= {(BUFFER_WIDTH*CLUSTERS*NUM_GLB_PSUM/2){1'd1}};
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
        finished_cycles             <= 0;
        fsm_psum_r                  <= 0;
        fsm_y_cl_psum               <= 0;
        fsm_x_cl_psum               <= 0;
      end
    end
  end

  //#######################
  //Wires
  //#######################


  wire                                           buffer_SP_en_r     [RAM_CELLS-1:0];
  wire                                           buffer_SP_en_w     [RAM_CELLS-1:0];
  wire [             RAM_CELLS_ADDR_WIDTH-2:0]   buffer_SP_addr     [RAM_CELLS-1:0];
  wire [          RAM_CELLS_WORD_BITWIDTH-1:0]   buffer_SP_data_w   [RAM_CELLS-1:0];
  wire [2*RAM_CELLS_WORD_BITWIDTH*RAM_CELLS-1:0] buffer_SP_data_r_w;
  wire [RAM_CELLS_WORD_BITWIDTH*RAM_CELLS-1:0]   buffer_SP_data_r;

  genvar k_gen;
  for (k_gen = 0; k_gen < RAM_CELLS; k_gen++) begin : gen_RAM_wires
    assign buffer_SP_en_r[k_gen] = buffer_SP_en_r_reg[k_gen];
    assign buffer_SP_en_w[k_gen] = buffer_SP_en_w_reg[k_gen];
    assign buffer_SP_addr[k_gen] = buffer_SP_addr_reg[k_gen];
    assign buffer_SP_data_w[k_gen] = buffer_SP_data_w_reg[k_gen];
    assign buffer_SP_data_r =   buffer_SP_data_r_w[0+:RAM_CELLS_WORD_BITWIDTH*RAM_CELLS];
  end

  generate
    genvar i_gen, j_gen, g_gen;
    // Converter Buffer
    for (j_gen = 0; j_gen < RAM_CELLS; j_gen++) begin : BUFFER_A
        RAM_SP #(
            .DataWidth(RAM_CELLS_WORD_BITWIDTH),
            .AddrWidth(RAM_CELLS_ADDR_WIDTH)
        ) iact_converter_buffer_SP (
            .clk_i(clk_i),
            .rd_en_i(buffer_SP_en_r[j_gen] & !buffer_SP_en_w[j_gen]),
            .wr_en_i(buffer_SP_en_w[j_gen]),
            .addr_i({choose_iact_buffer,buffer_SP_addr[j_gen]}),
            .data_i(buffer_SP_data_w[j_gen]),
            .data_o(buffer_SP_data_r_w[j_gen*RAM_CELLS_WORD_BITWIDTH+:RAM_CELLS_WORD_BITWIDTH])
        );
    end

    assign debug_iact_we     = buffer_SP_en_r[0] & !buffer_SP_en_w[0];
    assign debug_iact_re     = buffer_SP_en_w[0];
    assign debug_iact_addr   = {choose_iact_buffer,buffer_SP_addr[0]};
    assign debug_iact_data_i = buffer_SP_data_w[0];
    assign debug_iact_data_o = buffer_SP_data_r_w[0+:RAM_CELLS_WORD_BITWIDTH];

    // IACT Converter
    for (i_gen = 0; i_gen < CLUSTER_COLUMNS; i_gen++) begin : IACT_CONVERTER_X
      for (j_gen = 0; j_gen < CLUSTER_ROWS; j_gen++) begin : IACT_CONVERTER_Y
        wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] iact_data_w;
        wire [      $clog2(NUM_GLB_IACT+1)*PES-1:0] iact_choose_w;
        wire [                    NUM_GLB_IACT-1:0] iact_ready_w;
        wire [                    NUM_GLB_IACT-1:0] iact_enable_w;
        iact_stream_constructor #(
            .CLUSTER_ROWS   (CLUSTER_ROWS),
            .NUM_GLB_IACT      (NUM_GLB_IACT),
            .PE_X              (PE_COLUMNS),
            .PE_Y              (PE_ROWS),
            .DATA_IACT_BITWIDTH(DATA_IACT_BITWIDTH),
            .DATA_IACT_OVERHEAD(DATA_IACT_OVERHEAD),
            .RAM_CELLS         (RAM_CELLS),
            .WORD_BITWIDTH     (TRANS_BITWIDTH_IACT * NUM_GLB_IACT),
            .ADDRWIDTH         (BUFFER_WIDTH_IACT_STREAM_CONSTRUCTOR)
        ) iact_stream_constructor (
            .clk_i                       (clk_i),
            .rst_ni                      (rst_ni),
            .storage_i                   (buffer_SP_data_r),
            .reset_cycle_i               (reset_cycle_reg),
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
            .needed_cycles_i             (iact_converter_buffer_addr_max_cycles[3:0]),
            .needed_iact_channel_cycles_i(iact_channel_max_cycles),
            .iact_size_x_i               (iact_size_x),
            .iact_size_y_i               (iact_size_y),
            .iact_channels_i             (iact_channels_per_pe),
            .x_lines_i                   (x_lines_reg),
            .needed_wght_cycles_i        (needed_wght_cycles_reg),
            .needed_iact_router_cycles_i (needed_iact_cycles_reg),
            .wght_size_i                 (kernel_size),
            .fully_connected_i           (fully_connected_layer)
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



    for (i_gen = 0; i_gen < CLUSTER_COLUMNS; i_gen++) begin : PSUM_RAM_X
      for (j_gen = 0; j_gen < CLUSTER_ROWS; j_gen++) begin : PSUM_RAM_Y
        for (g_gen = 0; g_gen < NUM_GLB_PSUM/2; g_gen++) begin : PSUM_RAM_GLB

          RAM_SP #(
              //.DataWidth(TRANS_BITWIDTH_PSUM * CLUSTERS * NUM_GLB_PSUM),
              //.AddrWidth(BUFFER_WIDTH)
              .DataWidth(TRANS_BITWIDTH_PSUM*2),
              .AddrWidth(BUFFER_WIDTH)
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

    wire [      $clog2(NUM_GLB_IACT+1)*CLUSTERS*PES-1:0] iact_choose_i_oep_w;
    wire [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_data_i_oep_w;
    wire [                    CLUSTERS*NUM_GLB_IACT-1:0] iact_enable_i_oep_w;

    OpenEye_Parallel #(
        .IS_TOPLEVEL(0),
        .SERIAL     (SERIAL),

        .DATA_IACT_BITWIDTH(DATA_IACT_BITWIDTH),
        .DATA_PSUM_BITWIDTH(DATA_PSUM_BITWIDTH),
        .DATA_WGHT_BITWIDTH(DATA_WGHT_BITWIDTH),

        .TRANS_BITWIDTH_IACT(TRANS_BITWIDTH_IACT),
        .TRANS_BITWIDTH_WGHT(TRANS_BITWIDTH_WGHT),
        .TRANS_BITWIDTH_PSUM(TRANS_BITWIDTH_PSUM),
        .DATA_IACT_OVERHEAD (DATA_IACT_OVERHEAD),

        .PE_COLUMNS(PE_COLUMNS),
        .PE_ROWS   (PE_ROWS),

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
        .psum_enable_o(psum_enable_o_reg),
        .psum_ready_i (psum_ready_i_reg),

        //Ports for Hyperparameters
        .status_reg_enable_i          (status_reg_enable_reg),
        .data_mode_i                  (data_mode_reg),
        .fraction_bit_i               (fraction_bit_reg),
        .needed_cycles_i              (needed_cycles_reg),
        .needed_x_cls_i               (needed_x_cls_reg),
        .needed_y_cls_i               (needed_y_cls_reg),
        .needed_iact_cycles_i         (needed_iact_cycles_reg),
        .filters_i                    (filters_reg),
        .iact_addr_len_i              (iact_addr_len_reg),
        .wght_addr_len_i              (wght_addr_len_reg),
        .bano_cluster_mode_i          (bano_cluster_mode_reg),
        .af_cluster_mode_i            (af_cluster_mode_reg),
        .pooling_cluster_mode_i       (4'd0),
        .kernel_per_pe_cluster_i      (kernel_per_pe_cluster_reg[$clog2(PE_ROWS)-1:0]),
        .input_activations_i          (input_activations_reg),
        .stride_x_i                   (stride_x_reg),
        .stride_y_i                   (stride_y_reg),
        .delay_psum_glb_i             (psum_delay_reg),
        .compute_mask_i               (compute_mask_reg_port),
        .router_mode_iact_i           (router_mode_iact_reg),
        .router_mode_wght_i           (router_mode_wght_reg),
        .router_mode_psum_i           (router_mode_psum_reg),
        .needed_psum_storage_cycles_i (needed_psum_storage_cycles_reg),
        .needed_iact_channel_cycles_i (iact_channel_max_cycles),
        .psum_transmitted_i           (psum_transmitted)
    );

    genvar cc_gen, cr_gen, pe_gen;
    for (cc_gen = 0; cc_gen < CLUSTER_COLUMNS; cc_gen = cc_gen + 1) begin
      for (cr_gen = 0; cr_gen < CLUSTER_ROWS; cr_gen = cr_gen + 1) begin
        for (g_gen = 0; g_gen < NUM_GLB_IACT; g_gen = g_gen + 1) begin
          assign IACT_CONVERTER_X[cc_gen].IACT_CONVERTER_Y[cr_gen].iact_ready_w[g_gen] =
                iact_ready_o_oep_w[cc_gen * CLUSTER_ROWS * NUM_GLB_IACT + cr_gen * NUM_GLB_IACT + g_gen];
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
