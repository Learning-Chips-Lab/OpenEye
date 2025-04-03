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
///   FSM_STATES             - Amount of fsm states
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

module OpenEye_FPGA
#(
  //Set parameters
  parameter IS_TOPLEVEL         = 1,
  parameter SERIAL              = 1,
  parameter PARALLEL_MACS       = 2,
  parameter PADDING             = 1,

  parameter ADDR_IACT_BITWIDTH  = 4,
  parameter ADDR_WGHT_BITWIDTH  = 8,

  parameter DATA_IACT_BITWIDTH  = 8,
  parameter DATA_PSUM_BITWIDTH  = 20,
  parameter DATA_WGHT_BITWIDTH  = 8,

  parameter TRANS_BITWIDTH_IACT = 24,
  parameter TRANS_BITWIDTH_WGHT = 24,
  parameter TRANS_BITWIDTH_PSUM = 20,
  parameter DATA_IACT_OVERHEAD  = 4,

  parameter PE_COLUMNS          = 4,
  parameter PE_ROWS             = 3,
  parameter PES                 = PE_COLUMNS * PE_ROWS,
  
  parameter NUM_GLB_IACT        = 3,
  parameter NUM_GLB_WGHT        = 3,
  parameter NUM_GLB_PSUM        = 4,
  
  parameter CLUSTER_COLUMNS     = 2,
  parameter CLUSTER_ROWS        = 8,
  parameter CLUSTERS            = CLUSTER_COLUMNS * CLUSTER_ROWS,

  parameter IACT_ADDR_PER_PE    = 9,
  parameter WGHT_ADDR_PER_PE    = 16,

  parameter IACT_PER_PE         = 16,
  parameter PSUM_PER_PE         = 32,
  parameter WGHT_PER_PE         = 96,
  
  parameter IACT_MEM_ADDR_WORDS = 512,
  parameter IACT_FSM_CYCL_WORDS = IACT_PER_PE + IACT_ADDR_PER_PE,
  parameter WGHT_FSM_CYCL_WORDS = WGHT_PER_PE + WGHT_ADDR_PER_PE,
  parameter PSUM_MEM_ADDR_WORDS = 384,

  parameter IACT_MEM_ADDR_BITS  = $clog2(IACT_MEM_ADDR_WORDS),
  parameter PSUM_MEM_ADDR_BITS  = $clog2(PSUM_MEM_ADDR_WORDS),

  parameter ROUTER_MODES_IACT   = 6,
  parameter ROUTER_MODES_WGHT   = 1,
  parameter ROUTER_MODES_PSUM   = 3,

  parameter DMA_BITWIDTH        = 64,
  parameter FSM_CYCLE_MAX       = 4294967295,
  parameter FSM_STATES          = 10,

  parameter BUFFER_WIDTH = 12,
  parameter KERNEL_SIZE = 3, //Remoe later
  parameter real FSM_IACT_RTR_CCLS_A = (CLUSTERS*NUM_GLB_IACT),
  parameter real FSM_IACT_RTR_CCLS_B = $floor(DMA_BITWIDTH/ROUTER_MODES_IACT),
  parameter real FSM_IACT_RTR_CCLS   = FSM_IACT_RTR_CCLS_A/FSM_IACT_RTR_CCLS_B,
  parameter real FSM_WGHT_RTR_CCLS_A = (CLUSTERS*NUM_GLB_WGHT),
  parameter real FSM_WGHT_RTR_CCLS_B = $floor(DMA_BITWIDTH/ROUTER_MODES_WGHT),
  parameter real FSM_WGHT_RTR_CCLS   = FSM_WGHT_RTR_CCLS_A/FSM_WGHT_RTR_CCLS_B,
  parameter real FSM_PSUM_RTR_CCLS_A = (CLUSTERS*NUM_GLB_PSUM),
  parameter real FSM_PSUM_RTR_CCLS_B = $floor(DMA_BITWIDTH/ROUTER_MODES_PSUM),
  parameter integer FSM_PSUM_RTR_CCLS_C = DMA_BITWIDTH - (DMA_BITWIDTH % ROUTER_MODES_PSUM),
  parameter real FSM_PSUM_RTR_CCLS   = FSM_PSUM_RTR_CCLS_A/FSM_PSUM_RTR_CCLS_B,

  parameter integer FSM_CEIL_IACT_RTR_CCLS = $rtoi($ceil(FSM_IACT_RTR_CCLS)),
  parameter integer FSM_CEIL_WGHT_RTR_CCLS = $rtoi($ceil(FSM_WGHT_RTR_CCLS)),
  parameter integer FSM_CEIL_PSUM_RTR_CCLS = $rtoi($ceil(FSM_PSUM_RTR_CCLS)),

  parameter PARAMETER_POS_1_0     = 0,
  parameter PARAMETER_POS_1_1     = 1 + PARAMETER_POS_1_0,
  parameter PARAMETER_POS_1_2     = $clog2(DATA_PSUM_BITWIDTH) + PARAMETER_POS_1_1,
  parameter PARAMETER_POS_1_3     = 1 + PARAMETER_POS_1_2,
  parameter PARAMETER_POS_1_4     = 1 + PARAMETER_POS_1_3,
  parameter PARAMETER_POS_1_5     = 8 + PARAMETER_POS_1_4,
  parameter PARAMETER_POS_1_6     = 2 + PARAMETER_POS_1_5,
  parameter PARAMETER_POS_1_7     = $clog2(CLUSTER_ROWS+1) + PARAMETER_POS_1_6,
  parameter PARAMETER_POS_1_8     = 4 + PARAMETER_POS_1_7,
  parameter PARAMETER_POS_1_9     = $clog2(PSUM_PER_PE+1) + PARAMETER_POS_1_8,
  parameter PARAMETER_POS_1_10    = $clog2(IACT_ADDR_PER_PE+1) + PARAMETER_POS_1_9,
  parameter PARAMETER_POS_1_11    = $clog2(WGHT_ADDR_PER_PE+1) + PARAMETER_POS_1_10,

  parameter PARAMETER_POS_2_0     = 0,
  parameter PARAMETER_POS_2_1     = 2 + PARAMETER_POS_2_0,
  parameter PARAMETER_POS_2_2     = 4 + PARAMETER_POS_2_1,
  parameter PARAMETER_POS_2_3     = 4 + PARAMETER_POS_2_2,
  parameter PARAMETER_POS_2_4     = 4 + PARAMETER_POS_2_3,
  parameter PARAMETER_POS_2_5     = 1 + PARAMETER_POS_2_4,
  parameter PARAMETER_POS_2_6     = 1 + PARAMETER_POS_2_5,
  parameter PARAMETER_POS_2_7     = 1 + PARAMETER_POS_2_6,
  parameter PARAMETER_POS_2_8     = 4 + PARAMETER_POS_2_7,
  parameter PARAMETER_POS_2_9     = 4 + PARAMETER_POS_2_8,
  parameter PARAMETER_POS_2_10    = 1 + PARAMETER_POS_2_9,
  parameter PARAMETER_POS_2_11    = 1 + PARAMETER_POS_2_10,
  parameter PARAMETER_POS_2_12    = 4 + PARAMETER_POS_2_11,
  
  //Number of Words per PE
  parameter BANO_MODES            = 2,
  parameter AF_MODES              = 2,

  // Converter
  parameter BITWIDTH_IACT = TRANS_BITWIDTH_IACT / 2,
  parameter WGHT_SIZE = 3,

  //Storage RAMs
  parameter RAM_CELLS = 32,
  parameter RAM_CELLS_ADDR_WIDTH = 12,
  parameter RAM_CELLS_WORD_BITWIDTH = 64
    
) (
  //Input DMA
  input                            clk_i,
  input                            rst_ni,

  output reg                       ready_dma_o,
  input  [DMA_BITWIDTH-1 : 0]      data_dma_i,
  input                            enable_dma_i,
  
  //Output DMA
  input                            ready_dma_i,
  output reg [DMA_BITWIDTH-1 : 0]  data_dma_o,
  output reg                       enable_dma_o,

  output reg                       last_data_o
    
);
  //#######################
  //reset synchronization
  //#######################
  wire rst_n;

  RST_SYNC rst_sync_wrapper (
    .clk_i        (clk_i),
    .rst_ni       (rst_ni),
    .rst_no       (rst_n)
  );

  //#######################
  //Register
  //#######################

  reg [DMA_BITWIDTH-1:0]   data_dma_i_reg;
  reg                      enable_dma_i_reg;

  always @(posedge clk_i, negedge rst_n) begin
    if (rst_n == 1'b0) begin
      data_dma_i_reg <= 0;
      enable_dma_i_reg <= 0;
    end else begin
      data_dma_i_reg <= data_dma_i;
      enable_dma_i_reg <= enable_dma_i;
    end
  end

  `ifdef COCOTB_SIM
    initial begin
      if(IS_TOPLEVEL) begin
        $dumpfile ("sim_build/OpenEye_FPGA.fst");
        $dumpvars (0, OpenEye_FPGA);
      end
    end
  `endif

  //Register, that occupy hyperparameters
  reg                                           data_mode_reg;
  reg  [$clog2(DATA_PSUM_BITWIDTH)-1:0]         fraction_bit_reg;
  reg  [7:0]                                    needed_cycles_reg;
  reg  [$clog2(CLUSTER_COLUMNS+1)-1:0]          needed_x_cls_reg;
  reg  [$clog2(CLUSTER_ROWS+1)-1:0]             needed_y_cls_reg;
  reg  [3:0]                                    needed_iact_cycles_reg;
  reg  [$clog2(PSUM_PER_PE+1)-1:0]              filters_reg;
  reg  [$clog2(IACT_ADDR_PER_PE+1)-1:0]         iact_addr_len_reg;
  reg  [$clog2(WGHT_ADDR_PER_PE)-1:0]           wght_addr_len_reg;
  reg  [$clog2(BANO_MODES)*NUM_GLB_PSUM-1:0]    bano_cluster_mode_reg;
  reg  [$clog2(AF_MODES)*NUM_GLB_PSUM-1:0]      af_cluster_mode_reg;
  reg  [$clog2(IACT_PER_PE+1)-1:0]              input_activations_reg;
  reg  [1:0]                                    iact_write_addr_t_reg;
  reg  [3:0]                                    iact_write_data_t_reg;
  reg  [2:0]                                    stride_x_reg;
  reg  [2:0]                                    stride_y_reg;
  reg                                           skipIact_reg;
  reg                                           skipWght_reg;
  reg                                           skipPsum_reg;
  reg [$clog2(PE_ROWS)-1:0]                     kernel_per_pe_cluster_reg;
  reg [DMA_BITWIDTH-1 : 0]                      fifo_data_i;
  reg                                           fifo_read_i;
  reg                                           fifo_write_i;
  reg [3:0]                                     psum_delay_reg;

  //Register for the FSM
  reg  [32-1:0]                        fsm_cycle;
  reg  [6:0]                           fsm_cycle_mod1;
  reg  [11:0]                          fsm_cycle_div;
  reg  [6:0]                           fsm_cycle_div_cnt;
  // reg  [$clog2(FSM_STATES)-1:0]        fsm_last_state;
  // reg  [$clog2(FSM_STATES)-1:0]        fsm_current_state;
  reg  [$clog2(CLUSTER_COLUMNS)-1:0]    fsm_x_cl;
  reg  [$clog2(CLUSTER_COLUMNS)-1:0]    fsm_x_cl1;
  reg  [$clog2(CLUSTER_COLUMNS)-1:0]    fsm_x_cl2;
  reg  [$clog2(CLUSTER_COLUMNS)-1:0]    fsm_x_cl3;
  reg  [$clog2(CLUSTER_COLUMNS)-1:0]    fsm_x_cl4;
  reg  [$clog2(CLUSTER_COLUMNS)-1:0]    fsm_x_cl5;
  reg  [$clog2(CLUSTER_ROWS)-1:0]      fsm_y_cl;
  reg  [$clog2(CLUSTER_ROWS)-1:0]      fsm_y_cl1;
  reg  [$clog2(CLUSTER_ROWS)-1:0]      fsm_y_cl2;
  reg  [$clog2(CLUSTER_ROWS)-1:0]      fsm_y_cl3;
  reg  [$clog2(CLUSTER_ROWS)-1:0]      fsm_y_cl4;
  reg  [$clog2(CLUSTER_ROWS)-1:0]      fsm_y_cl5;
  reg  [$clog2(NUM_GLB_IACT)-1:0]      fsm_iact_r;
  reg  [$clog2(NUM_GLB_WGHT)-1:0]      fsm_wght_r;
  reg  [$clog2(NUM_GLB_PSUM)-1:0]      fsm_psum_r;
  reg  [$clog2(NUM_GLB_PSUM)-1:0]      fsm_psum_r1;
  reg  [$clog2(NUM_GLB_PSUM)-1:0]      fsm_psum_r2;
  reg  [$clog2(NUM_GLB_PSUM)-1:0]      fsm_psum_r3;
  reg  [$clog2(NUM_GLB_PSUM)-1:0]      fsm_psum_r4;
  reg  [$clog2(NUM_GLB_PSUM)-1:0]      fsm_psum_r5;
  reg  [DMA_BITWIDTH-1:0]              flat_help_var_1;
  reg  [DMA_BITWIDTH-1:0]              flat_help_var_2;
  reg                                  results_ready;
  reg  [2:0]                           loop_mod;
  reg  [7:0]                           finished_cycles;
  reg                                  new_stream;


  // Register for the Buffer

  reg                                                  iact_buffer_SP_en_r;
  reg                                                  iact_buffer_SP_en_w;
  reg  [BUFFER_WIDTH-1:0]                              iact_buffer_SP_addr;
  reg  [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_buffer_SP_data_w;
  wire [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_buffer_SP_data_r;

  reg                                                  wght_buffer_SP_en_r;
  reg                                                  wght_buffer_SP_en_w;
  reg  [BUFFER_WIDTH-1:0]                              wght_buffer_SP_addr;
  reg  [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_buffer_SP_data_w;
  reg  [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_buffer_SP_data_r;

  reg                                                 psum_buffer_SP_en_r;
  reg                                                 psum_buffer_SP_en_w;
  reg [BUFFER_WIDTH-1:0]                              psum_buffer_SP_addr;
  reg [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_buffer_SP_data_w;
  reg [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_buffer_SP_data_r;

  reg [BUFFER_WIDTH-1:0]                              iact_cnt;
  reg [BUFFER_WIDTH-1:0]                              wght_cnt;
  reg [BUFFER_WIDTH-1:0]                              psum_cnt;

  // Register for IACT Stream
  reg [7:0] current_buffer_n;
  reg [7:0] current_buffer_n_1;
  reg [RAM_CELLS_ADDR_WIDTH-1:0] current_buffer_addr;
  reg [7:0] current_channel;
  reg [7:0] iact_size;
  reg [7:0] iact_channels;
  reg [10:0] iact_needed_cycles;

  // Register for IACT Converter Buffer
  reg                               buffer_SP_en_r_reg   [RAM_CELLS-1:0];
  reg                               buffer_SP_en_w_reg   [RAM_CELLS-1:0];
  reg [RAM_CELLS_ADDR_WIDTH-1:0]    buffer_SP_addr_reg   [RAM_CELLS-1:0];
  reg [RAM_CELLS_WORD_BITWIDTH-1:0] buffer_SP_data_w_reg [RAM_CELLS-1:0];
  reg [RAM_CELLS_WORD_BITWIDTH-1:0] buffer_SP_data_r_reg [RAM_CELLS-1:0];

  // Registers for IACT Converter
  reg [31:0] iact_converter_params_reg   [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg iact_converter_en_cfg_reg          [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg iact_converter_en_enc_reg          [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg iact_converter_ready_reg           [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  wire iact_converter_n_en_w             [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [2:0] iact_converter_n_reg         [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [BUFFER_WIDTH-1:0] iact_converter_mem_addr_reg [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [3:0] iact_converter_mem_off_reg   [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];

  reg [7:0] iact_converter_x;
  reg [7:0] iact_converter_y;
  reg converters_ready;

  // Register for converting IACTS
  reg       buffer_r_en_reg              [RAM_CELLS-1:0];
  reg [2:0] iact_converter_n_1_reg       [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [3:0] iact_converter_mem_off_1_reg [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [2:0] iact_converter_n_2_reg       [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [3:0] iact_converter_mem_off_2_reg [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [2:0] iact_converter_n_3_reg       [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
  reg [3:0] iact_converter_mem_off_3_reg [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];

  reg [CLUSTERS*NUM_GLB_IACT*TRANS_BITWIDTH_IACT - 1:0] iact_out_reg;
  reg iact_ready;
  reg [2:0] iact_last;

  reg [5:0] converter_needed_cycles;

  //New Iact Converter
  reg [5:0] new_converter_needed_cycles;
  reg [6:0] new_converter_standing_cycles;
  reg [5:0] current_converter_cycles;
  reg [6:0] current_converter_standing_cycles;

  // additional register for fsm
  reg [$clog2(CLUSTER_COLUMNS)-1:0] fsm_col;
  reg [$clog2(CLUSTER_ROWS)-1:0]    fsm_row;
  reg [31:0]                        fsm_cycle_converter;
  reg [31:0]                        fsm_cycle_converter_1;
  reg [31:0]                        fsm_cycle_converter_2;
  reg [31:0]                        fsm_cycle_converter_3;
  reg [7:0]                         fsm_iact_n;
  reg [7:0]                         fsm_iact_n_1;
  reg [7:0]                         fsm_iact_n_2;
  reg [7:0]                         fsm_iact_n_3;


  // Register, that configure the chip

  reg                                                  status_reg_enable_reg;

  reg                                                  compute_reg;
  reg  [CLUSTERS*PES-1:0]                              compute_mask_reg;

  reg  [ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0]   router_mode_iact_reg;
  reg  [ROUTER_MODES_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0]   router_mode_wght_reg;
  reg  [ROUTER_MODES_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0]   router_mode_psum_reg;

  reg  [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_data_i_reg;
  reg  [CLUSTERS*NUM_GLB_WGHT-1:0]                     wght_enable_i_reg;
  reg  [CLUSTERS*NUM_GLB_WGHT-1:0]                     wght_ready_o_reg;

  reg  [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_data_i_reg;
  reg  [CLUSTERS*NUM_GLB_IACT-1:0]                     iact_enable_i_reg;
  reg  [CLUSTERS*NUM_GLB_IACT-1:0]                     iact_ready_o_reg;

  reg  [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_i_reg;
  reg  [CLUSTERS*NUM_GLB_PSUM-1:0]                     psum_enable_i_reg;
  reg  [CLUSTERS*NUM_GLB_PSUM-1:0]                     psum_ready_o_reg;

  reg  [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_o_reg;
  reg  [CLUSTERS*NUM_GLB_PSUM-1:0]                     psum_enable_o_reg;
  reg  [CLUSTERS*NUM_GLB_PSUM-1:0]                     psum_ready_i_reg;


  //#######################
  //States of the FSM
  //#######################

  typedef enum logic [4:0] {
    IDLE               = 0,
    GET_PARAMETERS     = 1,
    GET_ROUTER_CONFIG  = 2,
    GET_IACT           = 3,
    GET_WGHT           = 4,
    GET_BIAS           = 5,
  
    INIT_CONVERTER     = 6,
    START_CONVERTER    = 7,
    CONVERT_IACT       = 8,
    WAIT_CYCLE         = 9,
    WAIT_CYCLE_2       = 10,

    WRITE_IACT         = 11,
    WRITE_WGHT         = 12,
    WRITE_PSUM         = 13,

    WAIT_FOR_RESULTS   = 14,

    GET_RESULTS        = 15,
    SEND_RESULTS       = 16,
    LAST_CYCLE         = 17
  } state_t;

  state_t fsm_current_state;
  state_t fsm_last_state;

  //#######################
  //Process
  //#######################

  always@(posedge clk_i, negedge rst_n) begin
    if(!rst_n)begin
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
      new_stream                <= 0;

      fsm_cycle                 <= 0;
      fsm_cycle_mod1            <= 0;
      fsm_cycle_div             <= 0;
      fsm_cycle_div_cnt         <= 0;
      fsm_last_state            <= IDLE;
      fsm_current_state         <= GET_PARAMETERS;
      fsm_x_cl                  <= 0;
      fsm_x_cl1                 <= 0;
      fsm_x_cl2                 <= 0;
      fsm_x_cl3                 <= 0;
      fsm_x_cl4                 <= 0;
      fsm_x_cl5                 <= 0;
      fsm_y_cl                  <= 0;
      fsm_y_cl1                 <= 0;
      fsm_y_cl2                 <= 0;
      fsm_y_cl3                 <= 0;
      fsm_y_cl4                 <= 0;
      fsm_y_cl5                 <= 0;
      fsm_iact_r                <= 0;
      fsm_wght_r                <= 0;
      fsm_psum_r                <= 0;
      fsm_psum_r1               <= 0;
      fsm_psum_r2               <= 0;
      fsm_psum_r3               <= 0;
      fsm_psum_r4               <= 0;
      fsm_psum_r5               <= 0;
      finished_cycles           <= 0;

      skipIact_reg              <= 0;
      skipWght_reg              <= 0;
      skipPsum_reg              <= 0;

      results_ready              = 0;
      loop_mod                   = 0;
      flat_help_var_1            = 0;
      flat_help_var_2            = 0;
      enable_dma_o              <= 0;
      fifo_data_i               <= 0;
      fifo_read_i               <= 0;
      fifo_write_i              <= 0;

      bano_cluster_mode_reg     <= 0;
      af_cluster_mode_reg       <= 0;
      compute_reg               <= 0;
      compute_mask_reg          <= 0;
      router_mode_iact_reg      <= 0;
      router_mode_wght_reg      <= 0;
      router_mode_psum_reg      <= 0;

      iact_data_i_reg           <= 0;
      iact_enable_i_reg         <= 0;
      wght_data_i_reg           <= 0;
      wght_enable_i_reg         <= 0;
      psum_data_i_reg           <= 0;
      psum_enable_i_reg         <= 0;
      psum_ready_i_reg          <= 0;
      psum_delay_reg            <= 0;

      ready_dma_o               <= 0;
      last_data_o               <= 0;

      iact_buffer_SP_en_r       <= 0;
      iact_buffer_SP_en_w       <= 0;
      iact_buffer_SP_addr       <= 0;
      iact_buffer_SP_data_w     <= 0;
      wght_buffer_SP_en_r       <= 0;
      wght_buffer_SP_en_w       <= 0;
      wght_buffer_SP_addr       <= 0;
      wght_buffer_SP_data_w     <= 0;
      psum_buffer_SP_en_r       <= 0;
      psum_buffer_SP_en_w       <= 0;
      psum_buffer_SP_addr       <= 0;
      psum_buffer_SP_data_w     <= 0;

      iact_cnt <= 0;
      wght_cnt <= 0;
      psum_cnt <= 0;

      // iact converter
      fsm_col                   <= 0;
      fsm_row                   <= 0;
      fsm_cycle_converter       <= 0;
      fsm_cycle_converter_1     <= 0;
      fsm_iact_n                <= 0;
      fsm_iact_n_1              <= 0;
      fsm_iact_n_2              <= 0;
      fsm_iact_n_3              <= 0;

      iact_out_reg              <= 0;
      iact_ready                <= 0;
      iact_last                 <= 0;

      current_buffer_n          <= 0;
      current_buffer_n_1        <= 0;
      current_buffer_addr       <= 0;
      current_channel           <= 0;
      iact_size                 <= 0;
      iact_channels             <= 0;
      iact_needed_cycles        <= 1; // params

      //new iact regs
      new_converter_needed_cycles       <= 0;
      new_converter_standing_cycles     <= 0;
      current_converter_cycles          <= 0;
      current_converter_standing_cycles <= 0;
      for (int a=0; a<RAM_CELLS; a++) begin
        buffer_SP_en_r_reg[a] <= 0;
        buffer_SP_en_w_reg[a] <= 0;
        buffer_SP_addr_reg[a] <= 0;
        buffer_SP_data_w_reg[a] <= 0;
      end

      iact_converter_x <= 0;
      iact_converter_y <= 0;
      for (int a=0; a<CLUSTER_COLUMNS; a++) begin
        for (int b=0; b<CLUSTER_ROWS; b++) begin
          iact_converter_params_reg [a][b] <= 0;
          iact_converter_en_cfg_reg [a][b] <= 0;
          iact_converter_en_enc_reg [a][b] <= 0;
        end
      end

      converters_ready <= 0;
      converter_needed_cycles <= 0;

    end else begin
      case(fsm_current_state)

        IDLE : begin
          if(enable_dma_i_reg)begin
            fsm_last_state       <= IDLE;
            fsm_current_state    <= fsm_last_state;
            enable_dma_o         <= 0;
            fifo_data_i          <= 0;
            fifo_read_i          <= 0;
            fifo_write_i         <= 0;
          end
        end

        GET_PARAMETERS : begin
          enable_dma_o          <= 0;
          fifo_data_i           <= 0;
          fifo_read_i           <= 0;
          fifo_write_i          <= 0;
          status_reg_enable_reg <= 1;
          ready_dma_o           <= 1;
          if(enable_dma_i_reg) begin

            fsm_cycle <= fsm_cycle + 1;
            case(fsm_cycle)
              32'd0 : begin
                data_mode_reg          <= data_dma_i_reg[PARAMETER_POS_1_0];
                fraction_bit_reg       <= data_dma_i_reg[$clog2(DATA_PSUM_BITWIDTH)-1+PARAMETER_POS_1_1:PARAMETER_POS_1_1];
                af_cluster_mode_reg    <= data_dma_i_reg[PARAMETER_POS_1_2];
                needed_cycles_reg      <= data_dma_i_reg[7+PARAMETER_POS_1_4:PARAMETER_POS_1_4];
                needed_x_cls_reg       <= data_dma_i_reg[$clog2(CLUSTER_COLUMNS+1)-1+PARAMETER_POS_1_5:PARAMETER_POS_1_5];
                needed_y_cls_reg       <= data_dma_i_reg[$clog2(CLUSTER_ROWS+1)-1+PARAMETER_POS_1_6:PARAMETER_POS_1_6];
                needed_iact_cycles_reg <= data_dma_i_reg[3+PARAMETER_POS_1_7:PARAMETER_POS_1_7];
                filters_reg            <= data_dma_i_reg[$clog2(PSUM_PER_PE+1)-1+PARAMETER_POS_1_8:PARAMETER_POS_1_8];
                iact_addr_len_reg      <= data_dma_i_reg[3+PARAMETER_POS_1_9:PARAMETER_POS_1_9];
                wght_addr_len_reg      <= data_dma_i_reg[4+PARAMETER_POS_1_10:PARAMETER_POS_1_10];
                input_activations_reg  <= data_dma_i_reg[$clog2(IACT_PER_PE+1)-1+PARAMETER_POS_1_11:PARAMETER_POS_1_11];
              end
              32'd1 : begin
                iact_write_addr_t_reg <= data_dma_i_reg[1+PARAMETER_POS_2_0:PARAMETER_POS_2_0];
                iact_write_data_t_reg <= data_dma_i_reg[3+PARAMETER_POS_2_1:PARAMETER_POS_2_1];
                stride_x_reg          <= data_dma_i_reg[2+PARAMETER_POS_2_2:PARAMETER_POS_2_2];
                stride_y_reg          <= data_dma_i_reg[2+PARAMETER_POS_2_3:PARAMETER_POS_2_3];
                skipIact_reg          <= data_dma_i_reg[PARAMETER_POS_2_4:PARAMETER_POS_2_4];
                skipWght_reg          <= data_dma_i_reg[PARAMETER_POS_2_5:PARAMETER_POS_2_5];
                skipPsum_reg          <= data_dma_i_reg[PARAMETER_POS_2_6:PARAMETER_POS_2_6];
                psum_delay_reg        <= data_dma_i_reg[3+PARAMETER_POS_2_7:PARAMETER_POS_2_7];
                kernel_per_pe_cluster_reg <= data_dma_i_reg[1+PARAMETER_POS_2_8:PARAMETER_POS_2_8];
              end
              32'd2 : begin
                iact_channels      <= data_dma_i_reg[55:48];
                iact_size          <= data_dma_i_reg[39:32];
                iact_needed_cycles <= data_dma_i_reg[10:0];
              end
              32'd3 : begin
                compute_mask_reg[DMA_BITWIDTH-1:0] <= data_dma_i_reg[DMA_BITWIDTH-1:0];
              end
              32'd4 : begin
                compute_mask_reg[2*DMA_BITWIDTH-1:DMA_BITWIDTH] <= data_dma_i_reg[DMA_BITWIDTH-1:0];
              end
              32'd5 : begin
                compute_mask_reg[3*DMA_BITWIDTH-1:2*DMA_BITWIDTH] <= data_dma_i_reg[DMA_BITWIDTH-1:0];
                fsm_last_state    <= GET_PARAMETERS;
                fsm_current_state <= GET_ROUTER_CONFIG;
                fsm_cycle         <= 0;
              end
              default : begin
                fsm_last_state    <= GET_PARAMETERS;
                fsm_current_state <= GET_ROUTER_CONFIG;
                fsm_cycle         <= 0;
              end
            endcase
          end
        end

        GET_ROUTER_CONFIG : begin
          ready_dma_o <= 1;
          new_stream  <= 1;
          if(enable_dma_i_reg) begin
            fsm_cycle         <= fsm_cycle + 1;
            if(fsm_cycle < FSM_CEIL_IACT_RTR_CCLS)begin
              for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
                  for (int g=0; g<NUM_GLB_IACT; g=g+1) begin
                    if(((cc*NUM_GLB_IACT + cr*CLUSTER_COLUMNS*NUM_GLB_IACT + g)>=(fsm_cycle    *$floor(DMA_BITWIDTH/ROUTER_MODES_IACT)))
                      &((cc*NUM_GLB_IACT + cr*CLUSTER_COLUMNS*NUM_GLB_IACT + g)< ((fsm_cycle+1)*$floor(DMA_BITWIDTH/ROUTER_MODES_IACT))))begin
                      for(int i = 0; i < ROUTER_MODES_IACT; i = i + 1)begin
                        router_mode_iact_reg[cc * CLUSTER_ROWS * NUM_GLB_IACT * ROUTER_MODES_IACT +
                                           cr * NUM_GLB_IACT * ROUTER_MODES_IACT + 
                                           g * ROUTER_MODES_IACT + i] <=
                        data_dma_i_reg[(cc*NUM_GLB_IACT+cr*CLUSTER_COLUMNS*NUM_GLB_IACT+g-fsm_cycle*(DMA_BITWIDTH/ROUTER_MODES_IACT))
                        *ROUTER_MODES_IACT+i];
                      end
                    end
                  end
                end
              end
            end else begin

              if(fsm_cycle < FSM_CEIL_IACT_RTR_CCLS + FSM_CEIL_WGHT_RTR_CCLS)begin
                for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                  for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
                    for (int g=0; g<NUM_GLB_WGHT; g=g+1) begin
                      if(((cc*CLUSTER_ROWS*NUM_GLB_WGHT+cr*NUM_GLB_WGHT+g)>=((fsm_cycle-FSM_CEIL_IACT_RTR_CCLS)  *$floor(DMA_BITWIDTH/ROUTER_MODES_WGHT)))
                        &((cc*CLUSTER_ROWS*NUM_GLB_WGHT+cr*NUM_GLB_WGHT+g)< ((fsm_cycle+1-FSM_CEIL_IACT_RTR_CCLS)*$floor(DMA_BITWIDTH/ROUTER_MODES_WGHT))))begin
                        for(int i = 0; i < ROUTER_MODES_WGHT; i = i + 1)begin
                          router_mode_wght_reg[cc * CLUSTER_ROWS * NUM_GLB_WGHT * ROUTER_MODES_WGHT +
                                             cr * NUM_GLB_WGHT * ROUTER_MODES_WGHT + 
                                             g * ROUTER_MODES_WGHT + i] <=
                          data_dma_i_reg[(cc*CLUSTER_ROWS*NUM_GLB_WGHT+cr*NUM_GLB_WGHT+g-(fsm_cycle-FSM_CEIL_IACT_RTR_CCLS)*(DMA_BITWIDTH/ROUTER_MODES_WGHT))
                          +i];
                        end
                      end
                    end
                  end
                end
              end else begin
                for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                  for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
                    for (int g=0; g<NUM_GLB_PSUM; g=g+1) begin
                      if(((cc*NUM_GLB_PSUM+cr*CLUSTER_COLUMNS*NUM_GLB_PSUM+g)>=((fsm_cycle-FSM_CEIL_IACT_RTR_CCLS-FSM_CEIL_WGHT_RTR_CCLS)  *$floor(DMA_BITWIDTH/ROUTER_MODES_PSUM)))
                        &((cc*NUM_GLB_PSUM+cr*CLUSTER_COLUMNS*NUM_GLB_PSUM+g)< ((fsm_cycle+1-FSM_CEIL_IACT_RTR_CCLS-FSM_CEIL_WGHT_RTR_CCLS)*$floor(DMA_BITWIDTH/ROUTER_MODES_PSUM))))begin
                        for(int i = 0; i < ROUTER_MODES_PSUM; i = i + 1)begin
                          router_mode_psum_reg[cc * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM +
                                             cr * NUM_GLB_PSUM * ROUTER_MODES_PSUM + 
                                             g * ROUTER_MODES_PSUM + i] <=
                          data_dma_i_reg[(cc*NUM_GLB_PSUM*ROUTER_MODES_PSUM+cr*CLUSTER_COLUMNS*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM-(fsm_cycle-FSM_CEIL_IACT_RTR_CCLS-FSM_CEIL_WGHT_RTR_CCLS)*FSM_PSUM_RTR_CCLS_C)
                          +i];
                        end
                      end
                    end
                  end
                end
                if(fsm_cycle == FSM_CEIL_IACT_RTR_CCLS + FSM_CEIL_WGHT_RTR_CCLS + FSM_CEIL_PSUM_RTR_CCLS - 1) begin
                  fsm_cycle         <= 0;
                  fsm_last_state    <= GET_ROUTER_CONFIG;
                  status_reg_enable_reg <= 0;
                  if (!skipIact_reg) begin
                    fsm_current_state <= GET_IACT;
                  end else begin
                    fsm_current_state <= GET_WGHT;
                  end
                end
              end
            end
          end
        end

        GET_IACT : begin
          ready_dma_o         <= 1;
          psum_buffer_SP_en_w <= 0;

          for (int a=0; a<RAM_CELLS; a++) begin
            buffer_SP_en_w_reg[a] <= 0;
          end

          if(enable_dma_i_reg) begin
            fsm_cycle                         <= fsm_cycle + 1;
            current_buffer_n                  <= current_buffer_n + 1;
            current_buffer_n_1                <= current_buffer_n;
          // get iact params
            new_converter_needed_cycles       <= iact_size + ((KERNEL_SIZE-1));
            new_converter_standing_cycles     <= needed_iact_cycles_reg * iact_channels;

            buffer_SP_en_w_reg[current_buffer_n]   <= 1;
            buffer_SP_data_w_reg[current_buffer_n] <= data_dma_i_reg;
            buffer_SP_addr_reg[current_buffer_n_1] <= current_buffer_addr;
            current_buffer_addr                    <= buffer_SP_addr_reg[current_buffer_n] + 1; 

            if (current_buffer_n == RAM_CELLS - 1) begin
              current_buffer_n <= 0;
            end

            if (fsm_cycle == (iact_needed_cycles * RAM_CELLS) - 1) begin
              fsm_cycle         <= 0;
              fsm_current_state <= GET_WGHT;
              fsm_last_state    <= GET_IACT;
            end
          end else begin
            for (int a=0; a<RAM_CELLS; a++) begin
              buffer_SP_en_w_reg[a] <= 0;
            end
          end
        end

        GET_WGHT : begin
          ready_dma_o <= 1;
          for (int a=0; a<RAM_CELLS; a++) begin
              buffer_SP_en_w_reg[a] <= 0;
          end

          wght_buffer_SP_en_w <= 0;

          if(enable_dma_i_reg) begin
            for(int b=0; b<TRANS_BITWIDTH_WGHT; b=b+1)begin
              wght_buffer_SP_data_w[fsm_y_cl*TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT+fsm_wght_r*TRANS_BITWIDTH_WGHT+b]
              <= data_dma_i_reg[b];
              wght_buffer_SP_data_w[CLUSTER_ROWS*TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT+fsm_y_cl*TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT+fsm_wght_r*TRANS_BITWIDTH_WGHT+b] 
              <= data_dma_i_reg[TRANS_BITWIDTH_WGHT+b];
            end
            wght_enable_i_reg <= 0;
            
            if(fsm_wght_r != NUM_GLB_WGHT - 1)begin
              fsm_wght_r <= fsm_wght_r + 1;
            end else begin
              fsm_wght_r <= 0;
              if((fsm_y_cl + 1) != CLUSTER_ROWS)begin
                fsm_y_cl <= fsm_y_cl + 1;
              end else begin
                fsm_y_cl <= 0;
                fsm_cycle <= fsm_cycle + 1;

                wght_buffer_SP_en_w <= 1;
                wght_buffer_SP_addr <= wght_buffer_SP_addr + 1;

                if(fsm_cycle == ({27'd0,wght_addr_len_reg} + input_activations_reg * ({26'd0,filters_reg}/PARALLEL_MACS)) - 1)begin
                  fsm_cycle <= 0;
                  fsm_last_state    <= GET_WGHT;
                  if (!skipPsum_reg) begin
                    fsm_current_state <= GET_BIAS;
                  end else begin
                    fsm_current_state <= WAIT_FOR_RESULTS;
                  end
                end
              end
            end
          end else begin
            wght_enable_i_reg <= 0;
            wght_data_i_reg   <= 0;
          end
        end

        GET_BIAS : begin  
          ready_dma_o         <= 1;
          wght_enable_i_reg   <= 0;
          wght_buffer_SP_en_w <= 0;
          psum_buffer_SP_en_w <= 0;
          if(enable_dma_i_reg) begin    
            //fsm_cycle_mod1: fsm_cycle%CLUSTER_COLUMNS
            if(fsm_cycle_mod1 == (CLUSTER_COLUMNS-1))begin
              fsm_cycle_mod1 <= 0;
            end else begin
              fsm_cycle_mod1 <= fsm_cycle_mod1 + 1;
            end
            for(int b=0; b<TRANS_BITWIDTH_PSUM*2; b=b+1)begin
              psum_data_i_reg[fsm_cycle_mod1[0]*CLUSTER_ROWS*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+fsm_y_cl*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+fsm_psum_r*TRANS_BITWIDTH_PSUM+b]
              <= data_dma_i_reg[b];
            end
            psum_enable_i_reg <= 0;
            if((fsm_psum_r + 2) != NUM_GLB_PSUM)begin
              fsm_psum_r <= fsm_psum_r + 2;
            end else begin
              fsm_psum_r <= 0;
              if((fsm_y_cl + 1) != CLUSTER_ROWS)begin
                fsm_y_cl <= fsm_y_cl + 1;
              end else begin
                fsm_y_cl          <= 0;
                fsm_cycle         <= fsm_cycle + 1;
                psum_enable_i_reg <= {((CLUSTERS*NUM_GLB_PSUM)){1'b1}};
                psum_buffer_SP_en_w <= 1;
                psum_buffer_SP_addr <= psum_buffer_SP_addr + 1;
                if(fsm_cycle == (filters_reg*needed_cycles_reg) - 1)begin
                  fsm_cycle      <= 0;
                  fsm_cycle_mod1 <= 0;

                  fsm_last_state    <= GET_BIAS;
                  fsm_current_state <= INIT_CONVERTER;
                  ready_dma_o       <= 0;
                  iact_cnt <= iact_buffer_SP_addr;
                  iact_buffer_SP_addr <= 0;
                  wght_cnt <= wght_buffer_SP_addr;
                  wght_buffer_SP_addr <= 0;
                  psum_cnt <= psum_buffer_SP_addr + 1;
                end
              end
            end
          end else begin
            psum_data_i_reg   <= 0;
            psum_enable_i_reg <= 0;
          end
        end

        INIT_CONVERTER : begin
          fsm_cycle <= fsm_cycle + 1;
          psum_enable_i_reg <= 0;
          psum_buffer_SP_en_w <= 0;

          psum_buffer_SP_addr <= 0;
          if (iact_converter_x >= iact_size - PE_COLUMNS) begin
            iact_converter_x <= 0;
            iact_converter_y <= iact_converter_y + 1;
          end else begin
            iact_converter_x <= iact_converter_x + PE_COLUMNS;
          end

          iact_converter_params_reg[fsm_col][fsm_row][31:24] <= iact_converter_x;
          iact_converter_params_reg[fsm_col][fsm_row][23:16] <= iact_converter_y;
          iact_converter_params_reg[fsm_col][fsm_row][15:8]  <= iact_size;
          iact_converter_params_reg[fsm_col][fsm_row][7:0]   <= iact_channels;

          fsm_col <= fsm_col + 1;
          if (fsm_col >= CLUSTER_COLUMNS-1) begin
            fsm_col <= 0;
            fsm_row <= fsm_row + 1;
          end

          if (fsm_cycle >= CLUSTERS-1) begin
            fsm_current_state <= START_CONVERTER;
            fsm_last_state    <= INIT_CONVERTER;
            fsm_cycle         <= 0;

            for (int a=0; a<CLUSTER_COLUMNS; a++) begin
              for (int b=0; b<CLUSTER_ROWS; b++) begin
                iact_converter_en_cfg_reg[a][b] <= 1;
              end
            end
          end
        end
        
        START_CONVERTER : begin
          converters_ready = 1;
          for (int a=0; a<CLUSTER_COLUMNS; a++) begin
            for (int b=0; b<CLUSTER_ROWS; b++) begin
              iact_converter_en_cfg_reg[a][b] <= 0;
              converters_ready = converters_ready & iact_converter_ready_reg[a][b];
            end
          end

          if (converters_ready == 1) begin
            fsm_current_state <= CONVERT_IACT;
            for (int a=0; a<RAM_CELLS; a++) begin
              buffer_SP_addr_reg[a] <= ~0;
            end
          end

          converter_needed_cycles <= WGHT_SIZE * iact_channels;
        end

        CONVERT_IACT : begin
          fsm_cycle <= fsm_cycle + 1;
            /*
            if ((
            a < (current_converter_cycles*16/2)
            )&(
            a > (((current_converter_cycles - 3)*16/2) - 1))) begin
            */
          if (fsm_cycle == 0) begin
            for (int a=0; a<RAM_CELLS; a++) begin
              buffer_SP_en_r_reg[a] <= 0;
              if ((current_converter_cycles % 4) == (a/8)) begin
                buffer_SP_addr_reg[a] <= buffer_SP_addr_reg[a] + 1;  //Austauschen
              end
                buffer_SP_en_r_reg[a] <= 1;
            end
          end
          current_converter_standing_cycles <= current_converter_standing_cycles + 1;
          if (current_converter_standing_cycles == (new_converter_standing_cycles-1)) begin
            current_converter_standing_cycles <= 0;
            current_converter_cycles <= current_converter_cycles + 1;
            if (current_converter_cycles == (new_converter_needed_cycles - 1)) begin
              current_converter_cycles = 0;
              fsm_current_state <= WAIT_CYCLE;
            end
          end
          
          converters_ready = 1;
          for (int a=0; a<CLUSTER_COLUMNS; a++) begin
            for (int b=0; b<CLUSTER_ROWS; b++) begin
              converters_ready = converters_ready & iact_converter_ready_reg[a][b];
            end
          end

          if (converters_ready == 0) begin
            iact_last <= iact_last + 1;
          end

          if (fsm_cycle == 4) begin
            for (int a=0; a<CLUSTER_COLUMNS; a++) begin
              for (int b=0; b<CLUSTER_ROWS; b++) begin
                if (((b/CLUSTER_COLUMNS) == (current_converter_cycles%4)) & (current_converter_cycles < (new_converter_needed_cycles - 2))) begin
                  iact_converter_en_enc_reg[a][b] <= 1;
                end
              end
            end
          end
          if (fsm_cycle == 5) begin
            for (int a=0; a<CLUSTER_COLUMNS; a++) begin
              for (int b=0; b<CLUSTER_ROWS; b++) begin
                iact_converter_en_enc_reg[a][b] <= 0;
              end
            end
          end
          if (fsm_cycle == new_converter_standing_cycles-1) begin
            fsm_cycle <= 0;
          end
        end

        WAIT_CYCLE : begin
          iact_buffer_SP_data_w <= iact_out_reg;
          fsm_cycle             <= fsm_cycle + 1;
          iact_ready            <= 0;
          if (fsm_cycle == 8) begin
            fsm_cycle             <= 0;
            fsm_last_state        <= WAIT_CYCLE;
            fsm_current_state     <= WAIT_CYCLE_2;
            for (int a=0; a<CLUSTER_COLUMNS; a++) begin
              for (int b=0; b<CLUSTER_ROWS; b++) begin
                iact_converter_en_cfg_reg[a][b] <= 1;
              end
            end
          end
        end

        WAIT_CYCLE_2 : begin
          // reset IACT Stream Signals
          current_buffer_n    <= 0;
          current_buffer_n_1  <= 0;
          current_buffer_addr <= 0;
          current_channel     <= 0;
          iact_cnt            <= iact_buffer_SP_addr;
          for (int a=0; a<RAM_CELLS; a++) begin
            buffer_SP_addr_reg[a] <= 0;
          end

          // reset converter Signals
          fsm_iact_n            <= 0;
          fsm_iact_n_1          <= 0;
          fsm_iact_n_2          <= 0;
          fsm_iact_n_3          <= 0;
          fsm_cycle_converter   <= 0;
          fsm_cycle_converter_1 <= 0;
          fsm_cycle_converter_2 <= 0;
          fsm_cycle_converter_3 <= 0;
          fsm_col               <= 0;
          fsm_row               <= 0;
          iact_converter_x      <= 0;
          iact_converter_y      <= 0;
          iact_last             <= 0;
          iact_buffer_SP_en_w   <= 0;
          iact_cnt              <= iact_converter_mem_addr_reg[0][0];
          iact_buffer_SP_addr   <= ~0;
          fsm_cycle             <= 0;
          fsm_last_state        <= WAIT_CYCLE_2;
          fsm_current_state     <= WRITE_IACT;
        end

        WRITE_IACT : begin
          iact_buffer_SP_en_r <= 1;
          iact_buffer_SP_addr <= iact_buffer_SP_addr + 1;
          fsm_cycle           <= fsm_cycle + 1;

          if (fsm_cycle > 1) begin
            iact_data_i_reg   <= iact_buffer_SP_data_r;
            iact_enable_i_reg <= {((CLUSTERS*NUM_GLB_IACT)){1'b1}};
            if (iact_buffer_SP_addr >= iact_cnt + 1) begin
              // next cycle?
              iact_buffer_SP_en_r <= 0;
              iact_enable_i_reg   <= 0;
              iact_buffer_SP_addr <= 0;
              fsm_cycle           <= 0;
              fsm_current_state   <= WRITE_WGHT;
              fsm_last_state      <= WRITE_IACT;
            end
          end
        end

        WRITE_WGHT : begin
          wght_buffer_SP_en_r <= 1;
          wght_buffer_SP_addr <= wght_buffer_SP_addr + 1;

          fsm_cycle <= fsm_cycle + 1;

          if (fsm_cycle > 1) begin
            wght_data_i_reg <= wght_buffer_SP_data_r;
            wght_enable_i_reg <= {((CLUSTERS*NUM_GLB_WGHT)){1'b1}};
          end

          if (wght_buffer_SP_addr > wght_cnt  + 1) begin
            // next cycle?
            wght_buffer_SP_en_r <= 0;
            wght_enable_i_reg   <= 0;
            wght_buffer_SP_addr <= 0;
            fsm_cycle           <= 0;
            compute_reg         <= 1;
            fsm_current_state   <= WAIT_FOR_RESULTS;
            fsm_last_state      <= WRITE_WGHT;
          end
        end
        
        WAIT_FOR_RESULTS : begin
          compute_reg <= 0;

          status_reg_enable_reg <= 0;
          results_ready          = 1;
          for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
            for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
              for (int g=0; g<NUM_GLB_PSUM; g=g+1) begin
                flat_help_var_1 = 0;
                for(int b=0; b<ROUTER_MODES_PSUM; b=b+1)begin
                  flat_help_var_1[b] = router_mode_psum_reg[cc * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g * ROUTER_MODES_PSUM + b];
                end
                results_ready = results_ready & (psum_ready_o_reg[cc*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM+g] | (flat_help_var_1 == 2) |(flat_help_var_1 == 3));
              end
            end
          end
          if(results_ready)begin
            fsm_x_cl            <= 0;
            fsm_y_cl            <= 0;
            fsm_psum_r          <= 0;
            fsm_last_state      <= WAIT_FOR_RESULTS;
            fsm_current_state   <= GET_RESULTS;

            fsm_cycle           <= 1;
            psum_enable_i_reg   <= {(CLUSTERS*NUM_GLB_PSUM){1'b1}};
            psum_buffer_SP_addr <= {(BUFFER_WIDTH){1'b1}};
          end
          results_ready = 0;
        end

        GET_RESULTS : begin
          fsm_cycle <= fsm_cycle + 1;
          if (fsm_cycle == psum_cnt) begin
            psum_enable_i_reg <= 0;
          end
          if (psum_enable_o_reg != 0) begin
            psum_buffer_SP_en_w <= 1;
            psum_buffer_SP_addr <= psum_buffer_SP_addr + 1;
            psum_buffer_SP_data_w <= psum_data_o_reg;
          end
          if (psum_buffer_SP_addr ==  psum_cnt - 1) begin
            psum_buffer_SP_en_w <= 0;
            psum_buffer_SP_addr <= 0;
            psum_buffer_SP_en_r <= 1;
            psum_enable_i_reg   <= 0;
            fsm_cycle           <= 0;
            fsm_current_state   <= SEND_RESULTS;
            fsm_last_state      <= GET_RESULTS;
          end
        end

        SEND_RESULTS : begin
          if (fsm_psum_r != 0) begin
            enable_dma_o <= 1;
          end

          psum_buffer_SP_en_r <= 1;

          if (ready_dma_i == 1) begin
            fsm_psum_r1  <= fsm_psum_r;
            fsm_y_cl1    <= fsm_y_cl;
            fsm_x_cl1    <= fsm_x_cl;

            data_dma_o = 0;
            for(int b=0; b<TRANS_BITWIDTH_PSUM*2; b=b+1) begin
              data_dma_o[b] = psum_buffer_SP_data_r[fsm_x_cl1*CLUSTER_ROWS*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+fsm_y_cl1*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+fsm_psum_r1*TRANS_BITWIDTH_PSUM+b];
            end

              
            if((fsm_psum_r + 2) != NUM_GLB_PSUM)begin
              fsm_psum_r <= fsm_psum_r + 2;
            end else begin
              fsm_psum_r <= 0;
              if((fsm_x_cl + 1) != CLUSTER_COLUMNS)begin
                fsm_x_cl <= fsm_x_cl + 1;
              end else begin
                fsm_x_cl          <= 0;

                if((fsm_y_cl + 1) != CLUSTER_ROWS)begin
                  fsm_y_cl <= fsm_y_cl + 1;
                end else begin
                  fsm_y_cl <= 0;
                  
                  fsm_cycle <= fsm_cycle + 1;
                  psum_buffer_SP_addr <= psum_buffer_SP_addr + 1;

                  if(fsm_cycle == (filters_reg*needed_cycles_reg) - 1)begin
                    fsm_cycle           <= 0;
                    psum_buffer_SP_addr <= 0;
                    psum_buffer_SP_en_r <= 0;
                    fsm_last_state      <= SEND_RESULTS;
                    fsm_current_state   <= LAST_CYCLE;
                  end
                end
              end
            end
          end
        end

        LAST_CYCLE : begin
          if (ready_dma_i == 1) begin
            fsm_cycle <= fsm_cycle + 1;
            last_data_o <= 1;
            data_dma_o = 0;
            for(int b=0; b<TRANS_BITWIDTH_PSUM*2; b=b+1) begin
              data_dma_o[b] = psum_buffer_SP_data_r[fsm_x_cl1*CLUSTER_ROWS*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+fsm_y_cl1*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+fsm_psum_r1*TRANS_BITWIDTH_PSUM+b];
            end

            if (fsm_cycle == 1) begin
              enable_dma_o <= 0;
              last_data_o <= 0;
              fsm_cycle <= 0;
              fsm_last_state <= LAST_CYCLE;
              fsm_current_state <= GET_PARAMETERS;
            end
          end
        end

        default : begin
        end
      endcase
    
    end
  end

  //#######################
  //Wires
  //#######################

  wire                               buffer_SP_en_r   [RAM_CELLS-1:0];
  wire                               buffer_SP_en_w   [RAM_CELLS-1:0];
  wire [RAM_CELLS_ADDR_WIDTH-1:0]    buffer_SP_addr   [RAM_CELLS-1:0];
  wire [RAM_CELLS_WORD_BITWIDTH-1:0] buffer_SP_data_w [RAM_CELLS-1:0];
  wire [RAM_CELLS_WORD_BITWIDTH-1:0] buffer_SP_data_r [RAM_CELLS-1:0];

  assign buffer_SP_en_r = buffer_SP_en_r_reg;
  assign buffer_SP_en_w = buffer_SP_en_w_reg;
  assign buffer_SP_addr = buffer_SP_addr_reg;
  assign buffer_SP_data_w = buffer_SP_data_w_reg;
  assign buffer_SP_data_r_reg = buffer_SP_data_r;

  genvar i,j;
  generate
    // IACT Converter Buffer
    for (i = 0; i < RAM_CELLS; i++) begin : BUFFER_A
      RAM_SP #(
        .DataWidth(RAM_CELLS_WORD_BITWIDTH),
        .AddrWidth(RAM_CELLS_ADDR_WIDTH)
      ) iact_converter_buffer_SP ( 
        .clk_i   (clk_i),
        .rd_en_i (buffer_SP_en_r[i] & !buffer_SP_en_w[i]),
        .wr_en_i (buffer_SP_en_w[i]),
        .addr_i  (buffer_SP_addr[i]),
        .data_i  (buffer_SP_data_w[i]),
        .data_o  (buffer_SP_data_r[i])
      );
    end

  wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] iact_converter_mem_data_reg [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
    // IACT Converter
    for (i = 0; i < CLUSTER_COLUMNS; i++) begin : IACT_CONVERTER_X
      for (j = 0; j < CLUSTER_ROWS; j++) begin : IACT_CONVERTER_Y
        iact_stream_constructor #(
          .NUM_GLB_IACT       (NUM_GLB_IACT),
          .DATA_IACT_BITWIDTH (DATA_IACT_BITWIDTH),
          .DATA_IACT_OVERHEAD (DATA_IACT_OVERHEAD),
          .RAM_CELLS          (RAM_CELLS),
          .WORD_BITWIDTH      (TRANS_BITWIDTH_IACT*NUM_GLB_IACT),
          .ADDRWIDTH          (BUFFER_WIDTH)
        ) iact_stream_constructor (
          .clk_i            (clk_i),
          .rst_ni           (rst_ni),
          .params           (iact_converter_params_reg[i][j]),
          .enable_config    (iact_converter_en_cfg_reg[i][j]),
          .enable_converter (iact_converter_en_enc_reg[i][j]),
          .storage_i        (buffer_SP_data_r),
          .ready_o          (iact_converter_ready_reg[i][j]),
          .mem_en_o         (iact_converter_n_en_w[i][j]),
          .mem_addr_o       (iact_converter_mem_addr_reg[i][j]),
          .mem_data_o       (iact_converter_mem_data_reg[i][j])
        );
      end
    end

    wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] iact_buffer_SP_data_wi [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];
    wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] iact_buffer_SP_data_ro [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0];

    // other Buffers
    for (i = 0; i < CLUSTER_COLUMNS; i++) begin : IACT_CLUSTER_X
      for (j = 0; j < CLUSTER_ROWS; j++) begin : IACT_CLUSTER_Y

        assign iact_buffer_SP_data_wi[i][j] = iact_buffer_SP_data_w[((1+j+i*CLUSTER_ROWS)*NUM_GLB_IACT*TRANS_BITWIDTH_IACT)-1:(j+i*CLUSTER_ROWS)*NUM_GLB_IACT*TRANS_BITWIDTH_IACT];
        assign iact_buffer_SP_data_r[((1+j+i*CLUSTER_ROWS)*NUM_GLB_IACT*TRANS_BITWIDTH_IACT)-1:(j+i*CLUSTER_ROWS)*NUM_GLB_IACT*TRANS_BITWIDTH_IACT] = iact_buffer_SP_data_ro[i][j];

        RAM_SP #(
          .DataWidth(TRANS_BITWIDTH_IACT*NUM_GLB_IACT),
          .AddrWidth(BUFFER_WIDTH)
        ) iact_buffer_SP (
          .clk_i   (clk_i), 
          .rd_en_i (iact_buffer_SP_en_r & !iact_buffer_SP_en_w),
          .wr_en_i (iact_converter_n_en_w[i][j]), 
          .addr_i  (iact_buffer_SP_addr | iact_converter_mem_addr_reg[i][j]),
          .data_i  (iact_converter_mem_data_reg[i][j]),
          .data_o  (iact_buffer_SP_data_ro[i][j])
        );
      end
    end

    RAM_SP #(
      .DataWidth(TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT),
      .AddrWidth(BUFFER_WIDTH)
    ) wght_buffer_SP ( 
      .clk_i   (clk_i), 
      .rd_en_i (wght_buffer_SP_en_r & !wght_buffer_SP_en_w),
      .wr_en_i (wght_buffer_SP_en_w), 
      .addr_i  (wght_buffer_SP_addr),
      .data_i  (wght_buffer_SP_data_w),
      .data_o  (wght_buffer_SP_data_r)
    );

    RAM_SP #(
      .DataWidth(TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM),
      .AddrWidth(BUFFER_WIDTH)
    ) psum_buffer_SP ( 
      .clk_i   (clk_i), 
      .rd_en_i (psum_buffer_SP_en_r & !psum_buffer_SP_en_w),
      .wr_en_i (psum_buffer_SP_en_w), 
      .addr_i  (psum_buffer_SP_addr),
      .data_i  (psum_buffer_SP_data_w),
      .data_o  (psum_buffer_SP_data_r)
    );

    OpenEye_Parallel #(
      .IS_TOPLEVEL         (0),
      .SERIAL              (SERIAL),

      .DATA_IACT_BITWIDTH  (DATA_IACT_BITWIDTH),
      .DATA_PSUM_BITWIDTH  (DATA_PSUM_BITWIDTH),
      .DATA_WGHT_BITWIDTH  (DATA_WGHT_BITWIDTH),

      .TRANS_BITWIDTH_IACT (TRANS_BITWIDTH_IACT),
      .TRANS_BITWIDTH_WGHT (TRANS_BITWIDTH_WGHT),
      .TRANS_BITWIDTH_PSUM (TRANS_BITWIDTH_PSUM),
      .DATA_IACT_OVERHEAD  (DATA_IACT_OVERHEAD),

      .PE_COLUMNS          (PE_COLUMNS),
      .PE_ROWS             (PE_ROWS),

      .NUM_GLB_IACT        (NUM_GLB_IACT),
      .NUM_GLB_WGHT        (NUM_GLB_WGHT),
      .NUM_GLB_PSUM        (NUM_GLB_PSUM),

      .CLUSTER_COLUMNS     (CLUSTER_COLUMNS),
      .CLUSTER_ROWS        (CLUSTER_ROWS),

      .IACT_ADDR_PER_PE    (IACT_ADDR_PER_PE),
      .WGHT_ADDR_PER_PE    (WGHT_ADDR_PER_PE),

      .IACT_PER_PE         (IACT_PER_PE),
      .WGHT_PER_PE         (WGHT_PER_PE),
      .PSUM_PER_PE         (PSUM_PER_PE),

      .IACT_MEM_ADDR_WORDS (IACT_MEM_ADDR_WORDS),
      .PSUM_MEM_ADDR_WORDS (PSUM_MEM_ADDR_WORDS),

      .ROUTER_MODES_IACT   (ROUTER_MODES_IACT),
      .ROUTER_MODES_WGHT   (ROUTER_MODES_WGHT),
      .ROUTER_MODES_PSUM   (ROUTER_MODES_PSUM),

      .FSM_STATES          (FSM_STATES)
    )OpenEye_Parallel(
      //Clock and Reset Ports

      .clk_i                  (clk_i),
      .rst_ni                 (rst_n),
      .compute_i              (compute_reg),

      //Ports for GLBs and PEs

      .wght_data_i            (wght_data_i_reg),
      .wght_enable_i          (wght_enable_i_reg),
      .wght_ready_o           (wght_ready_o_reg),

      .iact_data_i            (iact_data_i_reg),
      .iact_enable_i          (iact_enable_i_reg),
      .iact_ready_o           (iact_ready_o_reg),

      .psum_data_i            (psum_data_i_reg),
      .psum_enable_i          (psum_enable_i_reg),
      .psum_ready_o           (psum_ready_o_reg),

      .psum_data_o            (psum_data_o_reg),
      .psum_enable_o          (psum_enable_o_reg),
      .psum_ready_i           (psum_ready_i_reg),

      //Ports for Hyperparameters
      .status_reg_enable_i    (status_reg_enable_reg),
      .data_mode_i            (data_mode_reg),
      .fraction_bit_i         (fraction_bit_reg),
      .needed_cycles_i        (needed_cycles_reg),
      .needed_x_cls_i         (needed_x_cls_reg),
      .needed_y_cls_i         (needed_y_cls_reg),
      .needed_iact_cycles_i   (needed_iact_cycles_reg),
      .filters_i              (filters_reg),
      .iact_addr_len_i        (iact_addr_len_reg),
      .wght_addr_len_i        (wght_addr_len_reg),
      .bano_cluster_mode_i    (bano_cluster_mode_reg),
      .af_cluster_mode_i      (af_cluster_mode_reg),
      .pooling_cluster_mode_i (4'd0),
      .kernel_per_pe_cluster_i(kernel_per_pe_cluster_reg),
      .input_activations_i    (input_activations_reg),
      .iact_write_addr_t_i    (iact_write_addr_t_reg),
      .iact_write_data_t_i    (iact_write_data_t_reg),
      .stride_x_i             (stride_x_reg),
      .stride_y_i             (stride_y_reg),
      .delay_psum_glb_i       (psum_delay_reg),
      .compute_mask_i         (compute_mask_reg),
      .router_mode_iact_i     (router_mode_iact_reg),
      .router_mode_wght_i     (router_mode_wght_reg),
      .router_mode_psum_i     (router_mode_psum_reg)
    );

  endgenerate

endmodule
