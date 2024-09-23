// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: OpenEye_Parallel
///
/// The OpenEye_Parallel represents the most sophisticated module for the implementation of the
/// OpenEye in the field of application-specific integrated circuits (ASIC). It incorporates a
/// multitude of finite state machine (FSM) units, which are responsible for the initialization of
/// the calculation process and the storage of the data within the global buffers. Upon completion
/// of the calculation, the data is transferred through the PSUM ports.
///
/// The Status FSM is the principal component that oversees the data flow. In its default state,
/// data can be written to the GLBs or the PEs. Upon receipt of the compute signal, both other FSMs
/// commence transmitting data from the GLBs to the PEs.
///
/// The IACT FSM accepts data from the input ports of the module and controls the transmission to
/// the Iact GLBs. Upon receipt of the compute signal by the Status FSM, the IACT FSM initiates the
/// transmission of the input activations. It then pauses until the SUM FSM commences the subsequent
/// computation cycle.
///
/// The PSUM FSM is responsible for accepting data from the input ports of the module and
/// controlling the transmission to the PSUM GLBs. Upon receipt of the compute signal by the Status
/// FSM, the PSUM FSM initiates the transmission of the bias. Once the bias has been sent, the FSM
/// awaits the results, which are then sent back to the GLBs. Following this, the state machine
/// initiates the next computation cycle.
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
/// 
/// Ports:
///   compute_i               - Port for starting the computation
///   wght_data_i             - Wght Data Port In
///   wght_enable_i           - Wght Enable Port In
///   wght_ready_o            - Wght Ready Port Out
///   iact_data_i             - Iact Data Port In
///   iact_enable_i           - Iact Enable Port In
///   iact_ready_o            - Iact Ready Port Out
///   psum_data_i             - Psum Data Port In
///   psum_enable_i           - Psum Enable Port In
///   psum_ready_o            - Psum Ready Port Out
///   psum_data_o             - Psum Data Port Out
///   psum_enable_o           - Psum Enable Port Out
///   psum_ready_i            - Psum Ready Port In
///   data_mode_i             - Specify if data is compressed or not, 0:compressed, 1:not compressed
///   router_mode_iact_i      - Configure Router Iact
///   router_mode_wght_i      - Configure Router Wght
///   router_mode_psum_i      - Configure Router Psum
///
///   bano_cluster_mode_i     - Chooses mode for the batch normalization
///   af_cluster_mode_i       - Chooses mode for the activation function
///   delay_psum_glb_i        - Chooses the needed delay for the psum data
///   fraction_bit_i          - Fraction bit for fixed point arithmetic
///

module OpenEye_Parallel 
#(

  ///Set parameters
  parameter IS_TOPLEVEL         = 1,

  parameter DATA_IACT_BITWIDTH  = 8,
  parameter DATA_PSUM_BITWIDTH  = 20,
  parameter DATA_WGHT_BITWIDTH  = 8,

  parameter TRANS_BITWIDTH_IACT = 24,
  parameter TRANS_BITWIDTH_WGHT = 24,
  parameter TRANS_BITWIDTH_PSUM = 40,

  parameter PE_COLUMNS          = 4,
  parameter PE_ROWS             = 3,
  
  parameter NUM_GLB_IACT        = 3,
  parameter NUM_GLB_WGHT        = 3,
  parameter NUM_GLB_PSUM        = 4,
  
  parameter CLUSTER_ROWS        = 8,
  parameter CLUSTER_COLUMNS      = 2,

  parameter IACT_PER_PE         = 16,
  parameter PSUM_PER_PE         = 32,
  parameter WGHT_PER_PE         = 96,

  parameter IACT_ADDR_PER_PE    = 9,
  parameter WGHT_ADDR_PER_PE    = 16,
  
  parameter IACT_MEM_ADDR_WORDS = 512,
  parameter PSUM_MEM_ADDR_WORDS = 384,

  parameter ROUTER_MODES_IACT   = 6,
  parameter ROUTER_MODES_WGHT   = 1,
  parameter ROUTER_MODES_PSUM   = 3,

  parameter FSM_STATES          = 7,
  parameter BANO_MODES            = 2,
  parameter AF_MODES              = 2,

  localparam IACT_MEM_ADDR_BITS    = $clog2(IACT_MEM_ADDR_WORDS),
  localparam PSUM_MEM_ADDR_BITS    = $clog2(PSUM_MEM_ADDR_WORDS),

  localparam IACT_FSM_CYCL_WORDS   = IACT_PER_PE + IACT_ADDR_PER_PE,
  localparam WGHT_FSM_CYCL_WORDS   = WGHT_PER_PE + WGHT_ADDR_PER_PE,

  localparam PES                   = PE_COLUMNS * PE_ROWS,

  localparam CLUSTERS              = CLUSTER_COLUMNS * CLUSTER_ROWS
    
) (
  ///Clock and Reset Ports
  
  input                                                          clk_i,
  input                                                          rst_ni,
  input                                                          compute_i,

  ///Ports for GLBs and PEs

  input      [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0]     wght_data_i,
  input      [CLUSTERS*NUM_GLB_WGHT-1:0]                         wght_enable_i,
  output reg [CLUSTERS*NUM_GLB_WGHT-1:0]                         wght_ready_o,

  input      [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0]     iact_data_i,
  input      [CLUSTERS*NUM_GLB_IACT-1:0]                         iact_enable_i,
  output reg [CLUSTERS*NUM_GLB_IACT-1:0]                         iact_ready_o,

  input      [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0]     psum_data_i,
  input      [CLUSTERS*NUM_GLB_PSUM-1:0]                         psum_enable_i,
  output reg [CLUSTERS*NUM_GLB_PSUM-1:0]                         psum_ready_o,

  output reg [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0]     psum_data_o,
  output reg [CLUSTERS*NUM_GLB_PSUM-1:0]                         psum_enable_o,
  input      [CLUSTERS*NUM_GLB_PSUM-1:0]                         psum_ready_i,

  ///Ports for Hyperparameters
  input                                                          status_reg_enable_i,
  input                                                          data_mode_i,
  input      [$clog2(DATA_PSUM_BITWIDTH)-1:0]                    fraction_bit_i,
  input      [7:0]                                               needed_cycles_i,
  input      [$clog2(CLUSTER_COLUMNS+1)-1:0]                      needed_x_cls_i,
  input      [$clog2(CLUSTER_ROWS+1)-1:0]                        needed_y_cls_i,
  input      [3:0]                                               needed_iact_cycles_i,
  input      [$clog2(PSUM_PER_PE+1)-1:0]                         filters_i,
  input      [$clog2(IACT_ADDR_PER_PE+1)-1:0]                    iact_addr_len_i,
  input      [$clog2(WGHT_ADDR_PER_PE+1)-1:0]                    wght_addr_len_i,
  input      [$clog2(BANO_MODES)*NUM_GLB_PSUM-1:0]               bano_cluster_mode_i,
  input      [$clog2(AF_MODES)*NUM_GLB_PSUM-1:0]                 af_cluster_mode_i,
  input      [NUM_GLB_PSUM-1:0]                                  pooling_cluster_mode_i,
  input      [3:0]                                               delay_psum_glb_i,
  input      [$clog2(IACT_PER_PE+1)-1:0]                         input_activations_i,
  input      [1:0]                                               iact_write_addr_t_i,
  input      [3:0]                                               iact_write_data_t_i,
  input      [3:0]                                               stride_x_i,
  input      [3:0]                                               stride_y_i,
  input      [CLUSTERS*PES-1:0]                                  compute_mask_i,
  input      [ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0]       router_mode_iact_i,
  input      [ROUTER_MODES_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0]       router_mode_wght_i,
  input      [ROUTER_MODES_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0]       router_mode_psum_i
  
    
);
  ///#######################
  ///Reset synchronization
  ///#######################
  wire rst_n;

  RST_SYNC rst_sync_top (
    .clk_i        (clk_i),
    .rst_ni       (rst_ni),
    .rst_no       (rst_n)
  );



  `ifdef COCOTB_SIM
  initial begin
      if(IS_TOPLEVEL) begin
        $dumpvars (0, OpenEye_Parallel);
      end
    end
  `endif

  ///#######################
  ///Register
  ///#######################

  ///Register, that occupy hyperparameters
  reg                                           data_mode_reg;
  reg  [$clog2(DATA_PSUM_BITWIDTH)-1:0]         fraction_bit_reg;
  reg  [7:0]                                    needed_cycles_reg;
  reg  [$clog2(CLUSTER_COLUMNS+1)-1:0]           needed_x_cls_reg;
  reg  [$clog2(CLUSTER_ROWS+1)-1:0]             needed_y_cls_reg;
  reg  [3:0]                                    needed_iact_cycles_reg;
  reg  [$clog2(PSUM_PER_PE+1)-1:0]              filters_reg;
  reg  [$clog2(IACT_ADDR_PER_PE+1)-1:0]         iact_addr_len_reg;
  reg  [$clog2(WGHT_ADDR_PER_PE+1)-1:0]         wght_addr_len_reg;
  reg  [$clog2(BANO_MODES)*NUM_GLB_PSUM-1:0]    bano_cluster_mode_reg;
  reg  [$clog2(AF_MODES)*NUM_GLB_PSUM-1:0]      af_cluster_mode_reg;
  reg  [NUM_GLB_PSUM-1:0]                       pooling_cluster_mode_reg;
  reg  [3:0]                                    delay_psum_glb_reg;
  reg  [$clog2(IACT_PER_PE+1)-1:0]              input_activations_reg;
  reg  [1:0]                                    iact_write_addr_t_reg;
  reg  [3:0]                                    iact_write_data_t_reg;
  reg  [3:0]                                    stride_x_reg;
  reg  [3:0]                                    stride_y_reg;
  reg  [CLUSTERS*PES-1:0]                       compute_cluster_i_reg;
  reg  [CLUSTERS*PES-1:0]                       compute_mask_reg;

  //Register for the main FSM
  reg  [32-1:0]                        fsm_cycle;
  reg  [$clog2(FSM_STATES)-1:0]        fsm_last_state;
  reg  [$clog2(FSM_STATES)-1:0]        fsm_current_state;

  ///Register for the iact FSM
  reg  [32-1:0]                        fsm_iact_cycle;
  reg  [6:0]                           fsm_iact_cycle_mod1;
  reg  [6:0]                           fsm_iact_cycle_mod2;
  reg  [6:0]                           fsm_iact_cycle_mod3;
  reg  [11:0]                          fsm_iact_cycle_div;
  reg  [6:0]                           fsm_iact_cycle_div_cnt;
  reg  [$clog2(FSM_STATES)-1:0]        fsm_iact_last_state;
  reg  [$clog2(FSM_STATES)-1:0]        fsm_iact_current_state;

  ///Register for the psum FSM
  reg  [32-1:0]                        fsm_psum_cycle;
  reg  [$clog2(FSM_STATES)-1:0]        fsm_psum_last_state;
  reg  [$clog2(FSM_STATES)-1:0]        fsm_psum_current_state;

  reg                                  data_write_enable;
  reg                                  data_write_enable_iact;
  reg  [64-1:0]                        flat_help_var;
  reg                                  results_ready;
  reg  [2:0]                           loop_mod;
  reg  [7:0]                           finished_cycles;
  reg  [$clog2(CLUSTER_ROWS)-1:0]      storage_cycles;

  ///Register, that configure the chip
  reg  [$clog2(NUM_GLB_IACT)*CLUSTERS*PES-1:0]         iact_choose_i;
  reg  [CLUSTERS*NUM_GLB_PSUM-1:0]                     psum_choose_i;
  reg  [ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0]   router_mode_iact_reg;
  reg  [ROUTER_MODES_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0]   router_mode_wght_reg;
  reg  [ROUTER_MODES_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0]   router_mode_psum_reg;
  reg                                                  computing;

  reg  [CLUSTERS*NUM_GLB_IACT-1:0]                     iact_enable_comp_reg;
  reg  [IACT_MEM_ADDR_BITS*CLUSTERS*NUM_GLB_IACT-1:0]  mem_addr_iact;
  reg  [PSUM_MEM_ADDR_BITS*CLUSTERS*NUM_GLB_PSUM-1:0]  mem_addr_psum;
  reg  [PSUM_MEM_ADDR_BITS-1:0]                        mem_addr_psum_storage;

  reg  [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_i_reg;
  reg  [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_o_reg;
  reg  [CLUSTERS*NUM_GLB_PSUM-1:0]                     psum_enable_delay;
  reg  [CLUSTERS*NUM_GLB_PSUM-1:0]                     psum_enable_i_reg;
  reg  [CLUSTERS*NUM_GLB_PSUM-1:0]                     psum_enable_o_reg;
  reg  [CLUSTERS*NUM_GLB_PSUM-1:0]                     psum_ready_i_reg;
  reg  [CLUSTERS*NUM_GLB_PSUM-1:0]                     psum_cluster_enable_o_reg;
  reg  [CLUSTERS*NUM_GLB_PSUM-1:0]                     psum_ready_o_reg;
  reg  [CLUSTERS*NUM_GLB_PSUM-1:0]                     psum_ready_o_cluster_reg;


  ///Wires and Regs for Ports

  wire  [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_data_i_w;
  wire  [CLUSTERS*NUM_GLB_IACT-1:0]                     iact_enable_i_w;
  wire  [CLUSTERS*NUM_GLB_IACT-1:0]                     iact_ready_o_w;
  wire  [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_data_i_w;
  wire  [CLUSTERS*NUM_GLB_WGHT-1:0]                     wght_enable_i_w;
  wire  [CLUSTERS*NUM_GLB_WGHT-1:0]                     wght_ready_o_w;

  wire                                                  compute_i_w;

  reg                                                  iact_transmitted;
  reg                                                  psum_transmitted;
  reg                                                  start_new_cycle;

  wire                                                  status_reg_enable_i_w;
  wire                                                  data_mode_i_w;
  wire  [$clog2(DATA_PSUM_BITWIDTH)-1:0]                fraction_bit_i_w;
  wire  [7:0]                                           needed_cycles_i_w;
  wire  [$clog2(CLUSTER_COLUMNS+1)-1:0]                  needed_x_cls_i_w;
  wire  [$clog2(CLUSTER_ROWS+1)-1:0]                    needed_y_cls_i_w;
  wire  [3:0]                                           needed_iact_cycles_i_w;
  wire  [$clog2(PSUM_PER_PE+1)-1:0]                     filters_i_w;
  wire  [$clog2(IACT_ADDR_PER_PE+1)-1:0]                iact_addr_len_i_w;
  wire  [$clog2(WGHT_ADDR_PER_PE+1)-1:0]                wght_addr_len_i_w;
  wire  [$clog2(BANO_MODES)*NUM_GLB_PSUM-1:0]           bano_cluster_mode_i_w;
  wire  [$clog2(AF_MODES)*NUM_GLB_PSUM-1:0]             af_cluster_mode_i_w;
  wire  [NUM_GLB_PSUM-1:0]                              pooling_cluster_mode_i_w;
  wire  [3:0]                                           delay_psum_glb_i_w;
  wire  [$clog2(IACT_PER_PE+1)-1:0]                     input_activations_i_w;
  wire  [1:0]                                           iact_write_addr_t_i_w;
  wire  [3:0]                                           iact_write_data_t_i_w;
  wire  [3:0]                                           stride_x_i_w;
  wire  [3:0]                                           stride_y_i_w;
  wire  [CLUSTERS*PES-1:0]                              compute_mask_i_w;
  wire  [ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0]   router_mode_iact_i_w;
  wire  [ROUTER_MODES_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0]   router_mode_wght_i_w;
  wire  [ROUTER_MODES_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0]   router_mode_psum_i_w;

  reg  [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_data_i_reg;
  reg  [CLUSTERS*NUM_GLB_IACT-1:0]                     iact_enable_i_reg;
  reg  [CLUSTERS*NUM_GLB_IACT-1:0]                     iact_ready_o_reg;
  reg  [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_data_i_reg;
  reg  [CLUSTERS*NUM_GLB_WGHT-1:0]                     wght_enable_i_reg;
  reg  [CLUSTERS*NUM_GLB_WGHT-1:0]                     wght_ready_o_reg;

  reg                                                  compute_i_reg;
  reg                                                  status_reg_enable_i_reg;
  reg                                                  data_mode_i_reg;
  reg  [$clog2(DATA_PSUM_BITWIDTH)-1:0]                fraction_bit_i_reg;
  reg  [7:0]                                           needed_cycles_i_reg;
  reg  [$clog2(CLUSTER_COLUMNS+1)-1:0]                  needed_x_cls_i_reg;
  reg  [$clog2(CLUSTER_ROWS+1)-1:0]                    needed_y_cls_i_reg;
  reg  [3:0]                                           needed_iact_cycles_i_reg;
  reg  [$clog2(PSUM_PER_PE+1)-1:0]                     filters_i_reg;
  reg  [$clog2(IACT_ADDR_PER_PE+1)-1:0]                iact_addr_len_i_reg;
  reg  [$clog2(WGHT_ADDR_PER_PE+1)-1:0]                wght_addr_len_i_reg;
  reg  [$clog2(BANO_MODES)*NUM_GLB_PSUM-1:0]           bano_cluster_mode_i_reg;
  reg  [$clog2(AF_MODES)*NUM_GLB_PSUM-1:0]             af_cluster_mode_i_reg;
  reg  [NUM_GLB_PSUM-1:0]                              pooling_cluster_mode_i_reg;
  reg  [$clog2(IACT_PER_PE+1)-1:0]                     input_activations_i_reg;
  reg  [1:0]                                           iact_write_addr_t_i_reg;
  reg  [3:0]                                           iact_write_data_t_i_reg;
  reg  [3:0]                                           stride_x_i_reg;
  reg  [3:0]                                           stride_y_i_reg;
  reg  [CLUSTERS*PES-1:0]                              compute_mask_i_reg;
  reg  [ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0]   router_mode_iact_i_reg;
  reg  [ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0]   router_mode_iact_storage;
  reg  [ROUTER_MODES_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0]   router_mode_wght_i_reg;
  reg  [ROUTER_MODES_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0]   router_mode_psum_i_reg;
  reg  [4*CLUSTERS-1:0]                                iact_router_offset;
  reg                                                  enable_stream_reg;
  reg  [7:0]                                           data_stream_reg;


  ///#######################
  ///States of the FSM
  ///#######################

  enum bit [2:0] {
    MAIN_IDLE          = 0,
    COMPUTING          = 1
  } fsm_mode;

  enum bit [2:0] {
    IACT_IDLE          = 0,
    CALCULATE_IACT     = 1,
    WAIT               = 2
  } fsm_iact_mode;

  enum bit [2:0] {
    PSUM_IDLE          = 0,
    CALCULATE_PSUM     = 1,
    GET_RESULTS        = 2,
    WAIT_FOR_RESULTS   = 3,
    SEND_RESULTS       = 4
  } fsm_psum_mode;

  ///#######################
  ///Process
  ///#######################

  always@(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin  ///Reset
      start_new_cycle            <= 0;
      data_mode_reg              <= 0;
      fraction_bit_reg           <= 0;
      needed_cycles_reg          <= 0;
      needed_x_cls_reg           <= 0;
      needed_y_cls_reg           <= 0;
      needed_iact_cycles_reg     <= 0;
      filters_reg                <= 0;
      iact_addr_len_reg          <= 0;
      wght_addr_len_reg          <= 0;
      stride_x_reg               <= 0;
      stride_y_reg               <= 0;
      iact_addr_len_reg          <= 0;
      wght_addr_len_reg          <= 0;
      bano_cluster_mode_reg      <= 0;
      af_cluster_mode_reg        <= 0;
      pooling_cluster_mode_reg   <= 0;
      delay_psum_glb_reg         <= 0;
      input_activations_reg      <= 0;
      iact_write_addr_t_reg      <= 0;
      iact_write_data_t_reg      <= 0;
      compute_mask_reg           <= 0;
      router_mode_wght_reg       <= 0;

      fsm_cycle                  <= 0;
      fsm_last_state             <= MAIN_IDLE;
      fsm_current_state          <= MAIN_IDLE;

      data_write_enable          <= 1;
      finished_cycles            <= 0;
      compute_mask_i_reg         <= 0;
      compute_cluster_i_reg      <= 0;

      results_ready               = 0;
      loop_mod                    = 0;
      flat_help_var               = 0;

      computing                  <= 0;
      wght_enable_i_reg          <= 0;
      wght_data_i_reg            <= 0;
      wght_ready_o               <= 0;
      compute_i_reg              <= 0;
      status_reg_enable_i_reg    <= 0;
      data_mode_i_reg            <= 0;
      fraction_bit_i_reg         <= 0;
      needed_cycles_i_reg        <= 0;
      needed_x_cls_i_reg         <= 0;
      needed_y_cls_i_reg         <= 0;
      needed_iact_cycles_i_reg   <= 0;
      filters_i_reg              <= 0;
      iact_addr_len_i_reg        <= 0;
      wght_addr_len_i_reg        <= 0;
      bano_cluster_mode_i_reg    <= 0;
      af_cluster_mode_i_reg      <= 0;
      pooling_cluster_mode_i_reg <= 0;
      input_activations_i_reg    <= 0;
      iact_write_addr_t_i_reg    <= 0;
      iact_write_data_t_i_reg    <= 0;
      stride_x_i_reg             <= 0;
      stride_y_i_reg             <= 0;
      compute_mask_i_reg         <= 0;
      router_mode_wght_i_reg     <= 0;
      router_mode_psum_i_reg     <= 0;
      storage_cycles             <= 0;
      enable_stream_reg          <= 1;
      data_stream_reg            <= 0;
      
    end else begin

      ///Regs for ports
      wght_data_i_reg            <= wght_data_i;
      wght_enable_i_reg          <= wght_enable_i;
      wght_ready_o               <= wght_ready_o_reg;  
      compute_i_reg              <= compute_i;
      status_reg_enable_i_reg    <= status_reg_enable_i;
      data_mode_i_reg            <= data_mode_i;
      fraction_bit_i_reg         <= fraction_bit_i;
      needed_cycles_i_reg        <= needed_cycles_i;
      needed_x_cls_i_reg         <= needed_x_cls_i;
      needed_y_cls_i_reg         <= needed_y_cls_i;
      needed_iact_cycles_i_reg   <= needed_iact_cycles_i;
      filters_i_reg              <= filters_i;
      iact_addr_len_i_reg        <= iact_addr_len_i;
      wght_addr_len_i_reg        <= wght_addr_len_i;
      bano_cluster_mode_i_reg    <= bano_cluster_mode_i;
      af_cluster_mode_i_reg      <= af_cluster_mode_i;
      pooling_cluster_mode_i_reg <= pooling_cluster_mode_i;
      input_activations_i_reg    <= input_activations_i;
      iact_write_addr_t_i_reg    <= iact_write_addr_t_i;
      iact_write_data_t_i_reg    <= iact_write_data_t_i;
      stride_x_i_reg             <= stride_x_i;
      stride_y_i_reg             <= stride_y_i;
      compute_mask_i_reg         <= compute_mask_i;

      case(fsm_current_state)

        MAIN_IDLE : begin
          data_write_enable    <= 1;
          computing            <= 0;
          storage_cycles       <= 0;
          router_mode_wght_reg <= router_mode_wght_i;
          if (compute_i_w) begin
            fsm_last_state    <= MAIN_IDLE;
            fsm_current_state <= COMPUTING;
            data_write_enable <= 0;
            computing         <= 1;
            mem_addr_psum     <= 0;
            finished_cycles   <= 0;
            enable_stream_reg <= 1;
            data_stream_reg   <= 0;
          end

          if (status_reg_enable_i_w) begin
            data_mode_reg            <= data_mode_i_w;
            fraction_bit_reg         <= fraction_bit_i_w;
            needed_cycles_reg        <= needed_cycles_i_w;
            needed_x_cls_reg         <= needed_x_cls_i_w;
            needed_y_cls_reg         <= needed_y_cls_i_w;
            needed_iact_cycles_reg   <= needed_iact_cycles_i_w;
            filters_reg              <= filters_i_w;
            iact_addr_len_reg        <= iact_addr_len_i_w;
            wght_addr_len_reg        <= wght_addr_len_i_w;
            bano_cluster_mode_reg    <= bano_cluster_mode_i_w;
            af_cluster_mode_reg      <= af_cluster_mode_i_w;
            pooling_cluster_mode_reg <= pooling_cluster_mode_i_w;
            delay_psum_glb_reg       <= delay_psum_glb_i;
            input_activations_reg    <= input_activations_i_w;
            iact_write_addr_t_reg    <= iact_write_addr_t_i_w;
            iact_write_data_t_reg    <= iact_write_data_t_i_w;
            stride_x_reg             <= stride_x_i_w;
            stride_y_reg             <= stride_y_i_w;
            compute_mask_reg         <= compute_mask_i_w;
            router_mode_iact_reg     <= router_mode_iact_i_w;
            router_mode_iact_storage <= router_mode_iact_i;
            router_mode_wght_reg     <= router_mode_wght_i_w;
            router_mode_psum_reg     <= router_mode_psum_i_w;
          end
          flat_help_var = 0;
        end

        COMPUTING : begin

          start_new_cycle <= 0;
          compute_cluster_i_reg <= 0;
          if (iact_transmitted & psum_transmitted) begin
            start_new_cycle <= 1;
            if (start_new_cycle != 1) begin
              finished_cycles <= finished_cycles + 1;
            end
            compute_cluster_i_reg <= compute_mask_reg;
          end
          if (fsm_psum_last_state == SEND_RESULTS) begin
            fsm_last_state    <= COMPUTING;
            fsm_current_state <= MAIN_IDLE;
          end

        end

        default : begin
        end
      endcase
    end
  end


  always@(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin  ///Reset
      iact_transmitted         <= 0;
      fsm_iact_cycle           <= 0;
      fsm_iact_cycle_mod1      <= 0;
      fsm_iact_cycle_mod2      <= 0;
      fsm_iact_cycle_mod3      <= 0;
      fsm_iact_cycle_div       <= 0;
      fsm_iact_cycle_div_cnt   <= 0;
      data_write_enable_iact   <= 1;
      fsm_iact_last_state      <= IACT_IDLE;
      fsm_iact_current_state   <= IACT_IDLE;
      router_mode_iact_i_reg   <= 0;
      router_mode_iact_storage <= 0;
      mem_addr_iact            <= 0;
      iact_choose_i            <= {CLUSTERS*PES{2'b11}};
      iact_enable_comp_reg     <= 0;
      iact_data_i_reg          <= 0;
      iact_ready_o             <= 0;
      iact_enable_i_reg        <= 0;
      iact_router_offset       <= 0;
      router_mode_iact_reg     <= 0;

    end else begin
      case(fsm_iact_current_state)       
        IACT_IDLE : begin
          data_write_enable_iact <= 1;
          iact_data_i_reg        <= iact_data_i;
          iact_enable_i_reg      <= iact_enable_i;
          iact_ready_o           <= iact_ready_o_reg;

          fsm_iact_cycle         <= 0;
          fsm_iact_cycle_mod1    <= 0;
          fsm_iact_cycle_mod2    <= 0;
          fsm_iact_cycle_mod3    <= 0;
          fsm_iact_cycle_div     <= 0;
          fsm_iact_cycle_div_cnt <= 0;

          if (compute_i_w) begin
            mem_addr_iact <= 0;
            fsm_iact_last_state    <= IACT_IDLE;
            fsm_iact_current_state <= CALCULATE_IACT;
          end

          if (status_reg_enable_i_w) begin
            mem_addr_iact <= 0;
          end

          router_mode_iact_reg     <= router_mode_iact_i;
          for (int g=0; g<CLUSTERS*NUM_GLB_IACT; g=g+1) begin
            if (iact_enable_i_w[g]) begin
              for (int b=0; b<IACT_MEM_ADDR_BITS; b=b+1) begin
                flat_help_var[b] = mem_addr_iact[g*IACT_MEM_ADDR_BITS+b];
              end
              flat_help_var = flat_help_var + 1;
              for (int b=0; b<IACT_MEM_ADDR_BITS; b=b+1) begin
                 mem_addr_iact[g*IACT_MEM_ADDR_BITS+b] <= flat_help_var[b];
              end
              flat_help_var = 0;
            end
          end

          flat_help_var      = 0;
          if (needed_y_cls_i_reg == 2) begin
            for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
              for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
                if ((cr == 1) | (cr == 3) | (cr == 5) | (cr == 7)) begin
                  flat_help_var = 3;
                end else begin
                  flat_help_var = 0;
                end
                for (int b=0; b<4; b=b+1) begin
                  iact_router_offset[cc*CLUSTER_ROWS*4+cr*4+b]
                  <= flat_help_var[b];
                end
              end
            end
          end else begin
            if (needed_y_cls_i_reg == 4) begin
              for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
                  if ((cr == 1) | (cr == 5)) begin
                    flat_help_var = 3;
                  end else begin
                    if ((cr == 2) | (cr == 6)) begin
                      flat_help_var = 6;
                    end else begin
                      if ((cr == 3) | (cr == 7)) begin
                        flat_help_var = 9;
                      end
                    end
                  end
                  for (int b=0; b<4; b=b+1) begin
                    iact_router_offset[cc*CLUSTER_ROWS*4+cr*4+b]
                    <= flat_help_var[b];
                  end
                  flat_help_var = 0;
                end
              end
            end
          end
        end

        CALCULATE_IACT : begin
          data_write_enable_iact <= 0;
          if (iact_ready_o_reg == ((2**(CLUSTERS*NUM_GLB_IACT))-1)) begin
            fsm_iact_cycle <= fsm_iact_cycle + 1;
            //fsm_iact_cycle_mod3: (fsm_iact_cycle_mod1-IACT_ADDR_PER_PE)%(TRANS_BITWIDTH_IACT/12)
            if (fsm_iact_cycle_mod3 == ((TRANS_BITWIDTH_IACT/12)-1)) begin
              fsm_iact_cycle_mod3 <= 0;
            end else begin
              if (fsm_iact_cycle_mod1[5:0] >= 6'(iact_addr_len_reg)) begin
                fsm_iact_cycle_mod3 <= fsm_iact_cycle_mod3 + 1;
              end
            end
            ///fsm_iact_cycle_mod2: fsm_iact_cycle_mod1%(TRANS_BITWIDTH_IACT/8)
            if (fsm_iact_cycle_mod2 == ((TRANS_BITWIDTH_IACT/4)-1)) begin
              fsm_iact_cycle_mod2 <= 0;
            end else begin
              fsm_iact_cycle_mod2 <= fsm_iact_cycle_mod2 + 1;
            end
            ///fsm_iact_cycle_mod1: fsm_iact_cycle%IACT_FSM_CYCL_WORDS
            if (fsm_iact_cycle_mod1 == ((7'(iact_addr_len_reg) + 7'(input_activations_reg))-1)) begin
              fsm_iact_cycle_mod1 <= 0;
              fsm_iact_cycle_mod2 <= 0;
            end else begin
              fsm_iact_cycle_mod1 <= fsm_iact_cycle_mod1 + 1;
            end
            ///fsm_iact_cycle_div: ((fsm_iact_cycle - 1)/IACT_FSM_CYCL_WORDS)
            if (fsm_iact_cycle_div_cnt == ((7'(iact_addr_len_reg) + 7'(input_activations_reg))-1)) begin
              fsm_iact_cycle_div <= fsm_iact_cycle_div + 1;
              fsm_iact_cycle_div_cnt <= 0;
            end else begin
              if (fsm_iact_cycle >= 2) begin
                fsm_iact_cycle_div_cnt <= fsm_iact_cycle_div_cnt + 1;
              end
            end
            
            if (fsm_iact_last_state == IACT_IDLE) begin
              mem_addr_psum <= 0;
            end

            for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
              for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
                ///loop_mod: cr%needed_y_cls
                if (loop_mod + 1 == (needed_y_cls_reg)) begin
                  loop_mod = 0;
                end
                for (int pec=0; pec<PE_COLUMNS; pec=pec+1) begin
                  for (int per=0; per<PE_ROWS; per=per+1) begin
                    for (int b=0; b<4; b=b+1) begin
                      flat_help_var[b] = iact_router_offset[cc*CLUSTER_ROWS*4+cr*4+b];
                    end
                    if ((32'(32'(flat_help_var) + 32'(loop_mod) * PE_ROWS + 32'(pec * stride_x_reg) + per) >=  NUM_GLB_IACT *  32'(fsm_iact_cycle_div)) 
                    &  (32'(32'(flat_help_var) + 32'(loop_mod) * PE_ROWS + 32'(pec * stride_x_reg) + per)  <  NUM_GLB_IACT * (32'(fsm_iact_cycle_div) + 1))
                    &  (compute_mask_reg[cc * PES * CLUSTER_ROWS + cr * PES + per * PE_COLUMNS + pec] == 1)) begin
                      flat_help_var   = (flat_help_var + loop_mod * PE_ROWS + pec * stride_x_reg + 64'(per) - NUM_GLB_IACT * fsm_iact_cycle_div);
                      for (int b=0; b<$clog2(NUM_GLB_IACT); b=b+1) begin
                        iact_choose_i[cc*PES*CLUSTER_ROWS*$clog2(NUM_GLB_IACT)+cr*PES*$clog2(NUM_GLB_IACT)+per*PE_COLUMNS*$clog2(NUM_GLB_IACT)+pec*$clog2(NUM_GLB_IACT)+b]
                        <= flat_help_var[b];
                      end
                    end else begin
                      flat_help_var = NUM_GLB_IACT;
                      for (int b=0; b<$clog2(NUM_GLB_IACT); b=b+1) begin
                        iact_choose_i[cc*PES*CLUSTER_ROWS*$clog2(NUM_GLB_IACT)+cr*PES*$clog2(NUM_GLB_IACT)+per*PE_COLUMNS*$clog2(NUM_GLB_IACT)+pec*$clog2(NUM_GLB_IACT)+b]
                        <= flat_help_var[b];
                      end
                    end
                    flat_help_var = 0;
                  end
                end
              end
            end
            loop_mod = 0;

            flat_help_var = 0;
            if (((fsm_iact_cycle != 0) & (6'(fsm_iact_cycle_mod1) < 6'(iact_addr_len_reg))
            & (0 == fsm_iact_cycle_mod2))  
            | ((6'(fsm_iact_cycle_mod1) >= 6'(iact_addr_len_reg))
            & (0 == fsm_iact_cycle_mod3))) begin
              for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
                  for (int g=0; g<NUM_GLB_IACT; g=g+1) begin
                    flat_help_var = 0;
                    for (int b=0; b<2; b=b+1) begin
                      flat_help_var[b] = router_mode_iact_reg[cc*CLUSTER_ROWS*NUM_GLB_IACT*ROUTER_MODES_IACT+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT + b + 4];
                    end
                    if ((flat_help_var[0] == 0) & (flat_help_var[1] == 0)) begin
                      for (int b=0; b<IACT_MEM_ADDR_BITS; b=b+1) begin
                        flat_help_var[b] = mem_addr_iact[cc*CLUSTER_ROWS*NUM_GLB_IACT*IACT_MEM_ADDR_BITS+cr*NUM_GLB_IACT*IACT_MEM_ADDR_BITS+g*IACT_MEM_ADDR_BITS + b];
                      end
                      flat_help_var = flat_help_var  +1;
                      for (int b=0; b<IACT_MEM_ADDR_BITS; b=b+1) begin
                        mem_addr_iact[cc*CLUSTER_ROWS*NUM_GLB_IACT*IACT_MEM_ADDR_BITS+cr*NUM_GLB_IACT*IACT_MEM_ADDR_BITS+g*IACT_MEM_ADDR_BITS + b]
                        <= flat_help_var[b];
                      end
                      flat_help_var = 0;
                    end
                  end
                end
              end
            end
            if (fsm_iact_cycle < ((32'(iact_addr_len_reg) + 32'(input_activations_reg))*needed_iact_cycles_reg)) begin
              for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
                  for (int g=0; g<NUM_GLB_IACT; g=g+1) begin
                    iact_enable_comp_reg[cc*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT+g] <= 1;
                  end
                end
              end
            end else begin
              iact_enable_comp_reg   <= 0;
              loop_mod                = 0;
              fsm_iact_cycle         <= 0;
              fsm_iact_cycle_mod1    <= 0;
              fsm_iact_cycle_mod2    <= 0;
              fsm_iact_cycle_div     <= 0;
              fsm_iact_cycle_div_cnt <= 0;
              fsm_iact_last_state    <= CALCULATE_IACT;
              fsm_iact_current_state <= WAIT;
            end
          end
        end

        WAIT : begin
          iact_transmitted <= 1;
          if (start_new_cycle) begin
            iact_transmitted       <= 0;
            fsm_iact_cycle         <= 0;
            fsm_iact_last_state    <= WAIT;
            fsm_iact_current_state <= CALCULATE_IACT;
            if (finished_cycles == needed_cycles_reg) begin
              fsm_iact_current_state <= IACT_IDLE;
              mem_addr_iact <= 0;
            end
          end
        end

        default : begin
        end
      endcase
    end
  end

  always@(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin  ///Reset
      psum_transmitted         <= 0;
      fsm_psum_cycle           <= 0;
      fsm_psum_last_state      <= PSUM_IDLE;
      fsm_psum_current_state   <= PSUM_IDLE;
      mem_addr_psum            <= {PSUM_MEM_ADDR_BITS*CLUSTERS*NUM_GLB_PSUM{1'b1}};
      mem_addr_psum_storage    <= 0;
      psum_enable_delay        <= 0;
      psum_enable_i_reg        <= 0;
      psum_enable_o_reg        <= 0;
      psum_ready_i_reg         <= 0;
      psum_enable_o            <= 0;
      psum_ready_o             <= 0;
      psum_ready_o_reg         <= 0;
      psum_data_i_reg          <= 0;
      psum_data_o              <= 0;
      psum_choose_i            <= 0;
      router_mode_psum_reg     <= 0;

    end else begin
      psum_enable_o     <= psum_enable_o_reg;
      psum_enable_i_reg <= psum_enable_i;
      psum_data_i_reg   <= psum_data_i;
      psum_data_o       <= psum_data_o_reg;
      psum_ready_i_reg  <= psum_ready_i;
      psum_ready_o      <= psum_ready_o_reg;
      if (start_new_cycle == 1) begin
        psum_transmitted <= 0;
      end
      case(fsm_psum_current_state)       
        PSUM_IDLE : begin
          psum_transmitted <= 0;
          fsm_psum_cycle   <= 0;

          if (compute_i_w) begin
            mem_addr_psum          <= 0;
            fsm_psum_last_state    <= PSUM_IDLE;
            fsm_psum_current_state <= CALCULATE_PSUM;
          end

          if (status_reg_enable_i_w) begin
            mem_addr_psum <= 0;
          end

          router_mode_psum_reg     <= router_mode_psum_i;
          for (int g=0; g<CLUSTERS*NUM_GLB_PSUM; g=g+1) begin
            if (psum_enable_i_reg[g]) begin
              for (int b=0; b<PSUM_MEM_ADDR_BITS; b=b+1) begin
                flat_help_var[b] = mem_addr_psum[g*PSUM_MEM_ADDR_BITS+b];
              end
              flat_help_var = flat_help_var + 1;
              for (int b=0; b<PSUM_MEM_ADDR_BITS; b=b+1) begin
                 mem_addr_psum[g*PSUM_MEM_ADDR_BITS+b] <= flat_help_var[b];
              end
              flat_help_var = 0;
            end
          end
          if (compute_i_w) begin
            if (needed_y_cls_reg == 1) begin
              psum_choose_i <= (2**(CLUSTER_ROWS*CLUSTER_COLUMNS*NUM_GLB_PSUM)-1);
            end else begin 
              if (needed_y_cls_reg == 2) begin
              psum_choose_i <= 64'b1111000011110000111100001111000011110000111100001111000011110000;
              end else begin
                if (needed_y_cls_reg == 4) begin
                  psum_choose_i <= 64'b1111000000000000111100000000000011110000000000001111000000000000;
                end else begin
                  psum_choose_i <= (2**(CLUSTER_ROWS*CLUSTER_COLUMNS*NUM_GLB_PSUM)-1);
                end
              end
            end
          end
        end

        CALCULATE_PSUM : begin
          if (finished_cycles == 0) begin
            psum_transmitted <= 1;
          end
          for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
            for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
              for (int g=0; g<NUM_GLB_PSUM; g=g+1) begin
                psum_ready_i_reg[cc*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM+g] <= 1;
              end
            end
          end
          if (results_ready == 0) begin
            results_ready = 1;
            for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
              for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
                for (int g=0; g<NUM_GLB_PSUM; g=g+1) begin
                  flat_help_var = 0;
                  flat_help_var = 64'(router_mode_psum_reg[cc * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g * ROUTER_MODES_PSUM + 2]);
                  results_ready = results_ready & (psum_ready_o_cluster_reg[cc*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM+g] | (flat_help_var == 0));
                  flat_help_var = 0;
                end
              end
            end
          end
          if (results_ready) begin
            fsm_psum_cycle <= fsm_psum_cycle + 1;
            for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
              for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
                for (int g=0; g<NUM_GLB_PSUM; g=g+1) begin
                  if ((fsm_psum_cycle != 0) & (router_mode_psum_reg[cc * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g * ROUTER_MODES_PSUM + 2] == 1)) begin
                    for (int b=0; b<PSUM_MEM_ADDR_BITS; b=b+1) begin
                      flat_help_var[b] = mem_addr_psum[cc * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + cr * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + g * PSUM_MEM_ADDR_BITS + b];
                    end
                    flat_help_var = flat_help_var + 1;
                    for (int b=0; b<PSUM_MEM_ADDR_BITS; b=b+1) begin
                      mem_addr_psum[cc * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + cr * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + g * PSUM_MEM_ADDR_BITS + b] <= flat_help_var[b];
                    end
                    flat_help_var = 0;
                  end
                  psum_enable_i_reg[cc*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM+g] <= 1;
                end
              end
            end
            if (fsm_psum_cycle >= ((32'(filters_reg)+1)/2)) begin
              fsm_psum_last_state    <= CALCULATE_PSUM;
              fsm_psum_current_state <= GET_RESULTS;
              results_ready           = 0;
              fsm_psum_cycle         <= 0;
              psum_enable_i_reg      <= 0;
              for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
                  for (int g=0; g<NUM_GLB_PSUM; g=g+1) begin
                    flat_help_var = 64'(mem_addr_psum_storage);
                    for (int b=0; b<PSUM_MEM_ADDR_BITS; b=b+1) begin
                      if (router_mode_psum_reg[cc * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g * ROUTER_MODES_PSUM + 2] == 1) begin
                        mem_addr_psum[cc * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + cr * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + g * PSUM_MEM_ADDR_BITS + b] <= 
                                      flat_help_var[b];
                      end
                    end
                    flat_help_var = 0;
                  end
                end
              end
            end
          end
        end

        GET_RESULTS : begin
          results_ready     = 1;
          psum_ready_i_reg <= psum_ready_i_reg;
          for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
            for (int cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin
              for (int g=0; g<NUM_GLB_PSUM; g=g+1) begin
                if (psum_cluster_enable_o_reg[cc*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM+g]) begin
                  flat_help_var = 0;
                  for (int b=0; b<PSUM_MEM_ADDR_BITS; b=b+1) begin
                    flat_help_var[b] = mem_addr_psum[cc * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + cr * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + g * PSUM_MEM_ADDR_BITS + b];
                  end
                  flat_help_var   = flat_help_var   + 1;
                  for (int b=0; b<PSUM_MEM_ADDR_BITS; b=b+1) begin
                    mem_addr_psum[cc * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + cr * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + g * PSUM_MEM_ADDR_BITS + b]
                    <= flat_help_var[b];
                  end
                end
                flat_help_var = 0;
                flat_help_var = 64'(router_mode_psum_reg[cc * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g * ROUTER_MODES_PSUM + 2]);

                results_ready = results_ready & (psum_cluster_enable_o_reg[cc*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM+g] | (flat_help_var == 0));
              end
            end
          end
          if (results_ready) begin
            fsm_psum_cycle <= fsm_psum_cycle + 1;
            if (fsm_psum_cycle == (32'((32'(filters_reg)+1)/2) - 1)) begin
              psum_transmitted       <= 1;
            end
          end
          if (fsm_psum_cycle == 32'((32'(filters_reg)+1)/2)) begin
            fsm_psum_cycle       <= 0;
            if ((finished_cycles == needed_cycles_reg) | (fsm_current_state == MAIN_IDLE)) begin
              fsm_psum_last_state    <= GET_RESULTS;
              fsm_psum_current_state <= WAIT_FOR_RESULTS;
              mem_addr_psum          <= 0;
              psum_ready_i_reg       <= 0;
              
            end else begin
              if (needed_y_cls_reg >= 2) begin
                for (int cr=1; cr<CLUSTER_ROWS; cr=cr+1) begin
                  for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                    for (int g=0; g<NUM_GLB_PSUM; g=g+1) begin
                      router_mode_psum_reg[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2] <=
                      router_mode_psum_reg[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+(cr-1)*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2];
                    end
                  end
                end
                if (storage_cycles != 3'(32'(needed_y_cls_reg) - 1)) begin
                  for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                    for (int g=0; g<NUM_GLB_PSUM; g=g+1) begin
                      router_mode_psum_reg[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+g*ROUTER_MODES_PSUM+2] <= 0;
                    end
                  end

                  for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                    for (int g=0; g<NUM_GLB_IACT; g=g+1) begin
                      router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+g*ROUTER_MODES_IACT+3] <= 0;
                      router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+g*ROUTER_MODES_IACT+4] <= 1;
                      router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+g*ROUTER_MODES_IACT+5] <= 1;
                    end
                  end

                  storage_cycles <= storage_cycles + 1;

                  for (int cr=1; cr<CLUSTER_ROWS; cr=cr+1) begin
                    for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                      for (int g=0; g<NUM_GLB_PSUM; g=g+1) begin
                        router_mode_psum_reg[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2] <=
                        router_mode_psum_reg[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+(cr-1)*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g*ROUTER_MODES_PSUM+2];
                      end
                    end
                  end

                  for (int cr=1; cr<CLUSTER_ROWS; cr=cr+1) begin
                    for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                      for (int g=0; g<NUM_GLB_IACT; g=g+1) begin
                        if (!((router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT] == 1) &
                            (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] == 1) &
                            (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+2] == 0) &
                            (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+3] == 0) &
                            (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
                            (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 1))) begin
                          if ((router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 0) &
                                (router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+5] == 0)) begin
                                router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+1] <= 1;
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
                                if ((router_mode_iact_reg[cc*ROUTER_MODES_IACT*NUM_GLB_IACT*CLUSTER_ROWS+(cr-1)*NUM_GLB_IACT*ROUTER_MODES_IACT+g*ROUTER_MODES_IACT+4] == 1) &
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
                end else begin
                  for (int cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
                    for (int g=0; g<NUM_GLB_PSUM; g=g+1) begin
                      router_mode_psum_reg[cc*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+g*ROUTER_MODES_PSUM+2] <= 1;
                    end
                  end
                  storage_cycles       <= 0;
                  router_mode_iact_reg <= router_mode_iact_storage;
                end
              end
              fsm_psum_last_state    <= GET_RESULTS;
              fsm_psum_current_state <= CALCULATE_PSUM;
              mem_addr_psum_storage  <= 9'(64'(mem_addr_psum_storage) + 64'(64'(48'(48'(filters_reg)+48'(1)))>>48'(1)));
              psum_ready_i_reg       <= 0;
            end
          end
        end

        WAIT_FOR_RESULTS : begin

          fsm_psum_cycle         <= fsm_psum_cycle + 1;
          if (fsm_psum_cycle == 2) begin
            fsm_psum_cycle         <= 0;
            fsm_psum_last_state    <= WAIT_FOR_RESULTS;
            fsm_psum_current_state <= SEND_RESULTS;
          end
        end

        SEND_RESULTS : begin
          for (int g=0; g<NUM_GLB_PSUM*CLUSTERS; g=g+1) begin
            psum_ready_i_reg[g]  <= 0;
            psum_ready_o_reg[g]  <= 1;
            psum_enable_delay[g] <= psum_enable_i_reg[g];
            psum_enable_o_reg[g] <= psum_enable_delay[g];
            if (psum_enable_i_reg[g] == 1) begin
              for (int b=0; b<PSUM_MEM_ADDR_BITS; b=b+1) begin
                flat_help_var[b] = mem_addr_psum[g * PSUM_MEM_ADDR_BITS + b];
              end
              flat_help_var = flat_help_var   + 1;
              for (int b=0; b<PSUM_MEM_ADDR_BITS; b=b+1) begin
                mem_addr_psum[g * PSUM_MEM_ADDR_BITS + b] <= flat_help_var[b];
              end
            end
          end
          flat_help_var   = 0;
          if (status_reg_enable_i) begin
            mem_addr_psum          <= 0;
            psum_enable_i_reg      <= 0;
            psum_ready_i_reg       <= 0;
            psum_ready_o_reg       <= 0;
            fsm_psum_last_state    <= SEND_RESULTS;
            fsm_psum_current_state <= PSUM_IDLE;
            fsm_psum_cycle         <= 0;
            mem_addr_psum_storage  <= 0;
          end
        end

        default : begin
        end
      endcase
    end
  end


  ///#######################
  ///Wires
  ///#######################

  genvar clusters_y;
  genvar clusters_x;
  generate
    for (clusters_x = 0; clusters_x < CLUSTER_COLUMNS; clusters_x = clusters_x + 1) begin: gen_x
      for (clusters_y = 0; clusters_y < CLUSTER_ROWS; clusters_y = clusters_y + 1) begin: gen_y

        ///Wires
        ////////////////////////////////
          ///Selection
          //////////////////////////////////
        wire  [$clog2(NUM_GLB_IACT)*PES-1:0]       iact_choose_cluster_i_w;
        wire  [NUM_GLB_PSUM-1:0]                   psum_choose_cluster_i_w;
        wire  [PES-1:0]                            compute_cluster_i_w;

          ///Router Modes
          //////////////////////////////////
        wire  [ROUTER_MODES_IACT*NUM_GLB_IACT-1:0] router_mode_iact_i_w;
        wire  [NUM_GLB_WGHT-1:0]                   router_mode_wght_i_w;
        wire  [ROUTER_MODES_PSUM*NUM_GLB_PSUM-1:0] router_mode_psum_i_w;

          ///IACT Connection
          //////////////////////////////////
          ///Connect above
        wire  [NUM_GLB_IACT-1:0]                     enable_src_top_iact_cluster_w;
        wire  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] data_src_top_iact_cluster_w;
        wire  [NUM_GLB_IACT-1:0]                     ready_src_top_iact_cluster_w;

        wire  [NUM_GLB_IACT-1:0]                     enable_dst_top_iact_cluster_w;
        wire  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] data_dst_top_iact_cluster_w;
        wire  [NUM_GLB_IACT-1:0]                     ready_dst_top_iact_cluster_w;

          ///Connect other side
        wire  [NUM_GLB_IACT-1:0]                     enable_src_side_iact_cluster_w;
        wire  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] data_src_side_iact_cluster_w;
        wire  [NUM_GLB_IACT-1:0]                     ready_src_side_iact_cluster_w;

        wire  [NUM_GLB_IACT-1:0]                     enable_dst_side_iact_cluster_w;
        wire  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] data_dst_side_iact_cluster_w;
        wire  [NUM_GLB_IACT-1:0]                     ready_dst_side_iact_cluster_w;

          ///Connect below
        wire  [NUM_GLB_IACT-1:0]                     enable_src_bottom_iact_cluster_w;
        wire  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] data_src_bottom_iact_cluster_w;
        wire  [NUM_GLB_IACT-1:0]                     ready_src_bottom_iact_cluster_w;

        wire  [NUM_GLB_IACT-1:0]                     enable_dst_bottom_iact_cluster_w;
        wire  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] data_dst_bottom_iact_cluster_w;
        wire  [NUM_GLB_IACT-1:0]                     ready_dst_bottom_iact_cluster_w;

          ///Memory
        wire  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] iact_data_i_cluster_w;
        wire  [IACT_MEM_ADDR_BITS*NUM_GLB_IACT-1:0]  iact_addr_i_cluster_w;
        wire  [NUM_GLB_IACT-1:0]                     iact_enable_i_cluster_w;
        wire  [NUM_GLB_IACT-1:0]                     iact_ready_o_cluster_w;

          ///WGHT Connection
          //////////////////////////////////
          ///Connect other side
        wire  [NUM_GLB_WGHT-1:0]                     enable_src_side_wght_cluster_w;
        wire  [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0] data_src_side_wght_cluster_w;
        wire  [NUM_GLB_WGHT-1:0]                     ready_src_side_wght_cluster_w;

        wire  [NUM_GLB_WGHT-1:0]                     enable_dst_side_wght_cluster_w;
        wire  [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0] data_dst_side_wght_cluster_w;
        wire  [NUM_GLB_WGHT-1:0]                     ready_dst_side_wght_cluster_w;

          ///Memory
        wire  [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0] wght_data_i_cluster_w;
        wire  [NUM_GLB_WGHT-1:0]                     wght_enable_i_cluster_w;
        wire  [NUM_GLB_WGHT-1:0]                     wght_ready_o_cluster_w;

          ///PSUM Connection
          //////////////////////////////////
          ///Connect above
        wire  [NUM_GLB_PSUM-1:0]                     enable_src_top_psum_cluster_w;
        wire  [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] data_src_top_psum_cluster_w;
        wire  [NUM_GLB_PSUM-1:0]                     ready_src_top_psum_cluster_w;

        wire  [NUM_GLB_PSUM-1:0]                     enable_dst_top_psum_cluster_w;
        wire  [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] data_dst_top_psum_cluster_w;
        wire  [NUM_GLB_PSUM-1:0]                     ready_dst_top_psum_cluster_w;

          ///Connect below
        wire  [NUM_GLB_PSUM-1:0]                     enable_src_bottom_psum_cluster_w;
        wire  [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] data_src_bottom_psum_cluster_w;
        wire  [NUM_GLB_PSUM-1:0]                     ready_src_bottom_psum_cluster_w;

        wire  [NUM_GLB_PSUM-1:0]                     enable_dst_bottom_psum_cluster_w;
        wire  [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] data_dst_bottom_psum_cluster_w;
        wire  [NUM_GLB_PSUM-1:0]                     ready_dst_bottom_psum_cluster_w;

          ///Memory 1
        wire  [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] psum_data_i_cluster_w;
        wire  [PSUM_MEM_ADDR_BITS*NUM_GLB_PSUM-1:0]  psum_addr_i_cluster_w;
        wire  [NUM_GLB_PSUM-1:0]                     psum_enable_i_cluster_w;
        wire  [NUM_GLB_PSUM-1:0]                     psum_ready_o_cluster_w;

          ///Memory 2
        wire  [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] psum_data_o_cluster_w;
        wire  [NUM_GLB_PSUM-1:0]                     psum_enable_o_cluster_w;
        wire  [NUM_GLB_PSUM-1:0]                     psum_ready_i_cluster_w;

        OpenEye_Cluster #(
          .DATA_IACT_BITWIDTH (DATA_IACT_BITWIDTH),
          .DATA_PSUM_BITWIDTH (DATA_PSUM_BITWIDTH),
          .DATA_WGHT_BITWIDTH (DATA_WGHT_BITWIDTH),

          .TRANS_BITWIDTH_IACT(TRANS_BITWIDTH_IACT),
          .TRANS_BITWIDTH_PSUM(TRANS_BITWIDTH_PSUM),
          .TRANS_BITWIDTH_WGHT(TRANS_BITWIDTH_WGHT),

          .NUM_GLB_IACT       (NUM_GLB_IACT),
          .NUM_GLB_WGHT       (NUM_GLB_WGHT),
          .NUM_GLB_PSUM       (NUM_GLB_PSUM),

          .PE_ROWS            (PE_ROWS),
          .PE_COLUMNS         (PE_COLUMNS),

          .IACT_PER_PE        (IACT_PER_PE),
          .PSUM_PER_PE        (PSUM_PER_PE),
          .WGHT_PER_PE        (WGHT_PER_PE),

          .IACT_MEM_ADDR_WORDS(IACT_MEM_ADDR_WORDS),
          .PSUM_MEM_ADDR_WORDS(PSUM_MEM_ADDR_WORDS),
            
          .LEFT_CLUSTER       (clusters_x == 0),
          .TOP_CLUSTER        (clusters_y == 0),
          .BOTTOM_CLUSTER     (clusters_y == CLUSTER_ROWS - 1)
        )OpenEye_Cluster(
          ///Clock and Reset Line
          //////////////////////////////////
          .clk_i                   (clk_i),
          .rst_ni                  (rst_n),
          .data_write_enable_iact_i(data_write_enable_iact),
          .data_write_enable_i     (data_write_enable),

          ///Selection
          //////////////////////////////////
          .iact_choose_i           (iact_choose_cluster_i_w),
          .psum_choose_i           (psum_choose_cluster_i_w),
          .compute_i               (compute_cluster_i_w),

          ///Router Modes
          //////////////////////////////////
          .router_mode_iact_i      (router_mode_iact_i_w),
          .router_mode_wght_i      (router_mode_wght_i_w),
          .router_mode_psum_i      (router_mode_psum_i_w),

          ///IACT Connection
          //////////////////////////////////
          //Connect above
          .enable_src_top_iact     (enable_src_top_iact_cluster_w),
          .data_src_top_iact       (data_src_top_iact_cluster_w),
          .ready_src_top_iact      (ready_src_top_iact_cluster_w),

          .enable_dst_top_iact     (enable_dst_top_iact_cluster_w),
          .data_dst_top_iact       (data_dst_top_iact_cluster_w),
          .ready_dst_top_iact      (ready_dst_top_iact_cluster_w),

          ///Connect other side
          .enable_src_side_iact    (enable_src_side_iact_cluster_w),
          .data_src_side_iact      (data_src_side_iact_cluster_w),
          .ready_src_side_iact     (ready_src_side_iact_cluster_w),

          .enable_dst_side_iact    (enable_dst_side_iact_cluster_w),
          .data_dst_side_iact      (data_dst_side_iact_cluster_w),
          .ready_dst_side_iact     (ready_dst_side_iact_cluster_w),

          ///Connect below
          .enable_src_bottom_iact  (enable_src_bottom_iact_cluster_w),
          .data_src_bottom_iact    (data_src_bottom_iact_cluster_w),
          .ready_src_bottom_iact   (ready_src_bottom_iact_cluster_w),

          .enable_dst_bottom_iact  (enable_dst_bottom_iact_cluster_w),
          .data_dst_bottom_iact    (data_dst_bottom_iact_cluster_w),
          .ready_dst_bottom_iact   (ready_dst_bottom_iact_cluster_w),

          ///Memory
          .ext_mem_iact_data_i     (iact_data_i_cluster_w),
          .ext_mem_iact_addr_i     (iact_addr_i_cluster_w),
          .ext_mem_iact_enable_i   (iact_enable_i_cluster_w),
          .ext_mem_iact_ready_o    (iact_ready_o_cluster_w),

          ///WGHT Connection
          //////////////////////////////////
          ///Connect other side
          .enable_src_side_wght    (enable_src_side_wght_cluster_w),
          .data_src_side_wght      (data_src_side_wght_cluster_w),
          .ready_src_side_wght     (ready_src_side_wght_cluster_w),

          .enable_dst_side_wght    (enable_dst_side_wght_cluster_w),
          .data_dst_side_wght      (data_dst_side_wght_cluster_w),
          .ready_dst_side_wght     (ready_dst_side_wght_cluster_w),

          ///Memory
          .ext_mem_wght_data_i     (wght_data_i_cluster_w),
          .ext_mem_wght_enable_i   (wght_enable_i_cluster_w),
          .ext_mem_wght_ready_o    (wght_ready_o_cluster_w),

          ///PSUM Connection
          //////////////////////////////////
          ///Connect above
          .enable_src_top_psum     (enable_src_top_psum_cluster_w),
          .data_src_top_psum       (data_src_top_psum_cluster_w),
          .ready_src_top_psum      (ready_src_top_psum_cluster_w),

          .enable_dst_top_psum     (enable_dst_top_psum_cluster_w),
          .data_dst_top_psum       (data_dst_top_psum_cluster_w),
          .ready_dst_top_psum      (ready_dst_top_psum_cluster_w),

          ///Connect below
          .enable_src_bottom_psum  (enable_src_bottom_psum_cluster_w),
          .data_src_bottom_psum    (data_src_bottom_psum_cluster_w),
          .ready_src_bottom_psum   (ready_src_bottom_psum_cluster_w),

          .enable_dst_bottom_psum  (enable_dst_bottom_psum_cluster_w),
          .data_dst_bottom_psum    (data_dst_bottom_psum_cluster_w),
          .ready_dst_bottom_psum   (ready_dst_bottom_psum_cluster_w),

          ///Memory 1
          .ext_mem_psum_data_i     (psum_data_i_cluster_w),
          .ext_mem_psum_addr_i     (psum_addr_i_cluster_w),
          .ext_mem_psum_enable_i   (psum_enable_i_cluster_w),
          .ext_mem_psum_ready_o    (psum_ready_o_cluster_w),

          ///Memory 2
          .ext_mem_psum_data_o     (psum_data_o_cluster_w),
          .ext_mem_psum_enable_o   (psum_enable_o_cluster_w),
          .ext_mem_psum_ready_i    (psum_ready_i_cluster_w),

          .bano_cluster_mode_i     (bano_cluster_mode_reg),
          .af_cluster_mode_i       (af_cluster_mode_reg),
          .delay_psum_glb_i        (delay_psum_glb_reg),
          .enable_stream_i         (enable_stream_reg),
          .data_stream_i           (data_stream_reg)
        );

      end
    end

  genvar cr,cc,pec,per,g,b;

  if (IS_TOPLEVEL) begin

    assign iact_data_i_w            = iact_data_i_reg;
    assign iact_enable_i_w          = iact_enable_i_reg;
    assign wght_data_i_w            = wght_data_i_reg;
    assign wght_enable_i_w          = wght_enable_i_reg;

    assign compute_i_w              = compute_i_reg;
    assign status_reg_enable_i_w    = status_reg_enable_i_reg;
    assign data_mode_i_w            = data_mode_i_reg;
    assign fraction_bit_i_w         = fraction_bit_i_reg;
    assign needed_cycles_i_w        = needed_cycles_i_reg;
    assign needed_x_cls_i_w         = needed_x_cls_i_reg;
    assign needed_y_cls_i_w         = needed_y_cls_i_reg;
    assign needed_iact_cycles_i_w   = needed_iact_cycles_i_reg;
    assign filters_i_w              = filters_i_reg;
    assign iact_addr_len_i_w        = iact_addr_len_i_reg;
    assign wght_addr_len_i_w        = wght_addr_len_i_reg;
    assign bano_cluster_mode_i_w    = bano_cluster_mode_i_reg;
    assign af_cluster_mode_i_w      = af_cluster_mode_i_reg;
    assign pooling_cluster_mode_i_w = pooling_cluster_mode_i_reg;
    assign delay_psum_glb_i_w       = delay_psum_glb_reg;
    assign input_activations_i_w    = input_activations_i_reg;
    assign iact_write_addr_t_i_w    = iact_write_addr_t_i_reg;
    assign iact_write_data_t_i_w    = iact_write_data_t_i_reg;
    assign stride_x_i_w             = stride_x_i_reg;
    assign stride_y_i_w             = stride_y_i_reg;
    assign compute_mask_i_w         = compute_mask_i_reg;
    assign router_mode_iact_i_w     = router_mode_iact_i_reg;
    assign router_mode_wght_i_w     = router_mode_wght_i_reg;
    assign router_mode_psum_i_w     = router_mode_psum_i_reg;

  end else begin

  
    assign iact_data_i_w            = iact_data_i;
    assign iact_enable_i_w          = iact_enable_i;
    assign wght_data_i_w            = wght_data_i;
    assign wght_enable_i_w          = wght_enable_i;

    assign compute_i_w              = compute_i;
    assign status_reg_enable_i_w    = status_reg_enable_i;
    assign data_mode_i_w            = data_mode_i;
    assign fraction_bit_i_w         = fraction_bit_i;
    assign needed_cycles_i_w        = needed_cycles_i;
    assign needed_x_cls_i_w         = needed_x_cls_i;
    assign needed_y_cls_i_w         = needed_y_cls_i;
    assign needed_iact_cycles_i_w   = needed_iact_cycles_i;
    assign filters_i_w              = filters_i;
    assign iact_addr_len_i_w        = iact_addr_len_i;
    assign wght_addr_len_i_w        = wght_addr_len_i;
    assign bano_cluster_mode_i_w    = bano_cluster_mode_i;
    assign af_cluster_mode_i_w      = af_cluster_mode_i;
    assign pooling_cluster_mode_i_w = pooling_cluster_mode_i;
    assign delay_psum_glb_i_w       = delay_psum_glb_i;
    assign input_activations_i_w    = input_activations_i;
    assign iact_write_addr_t_i_w    = iact_write_addr_t_i;
    assign iact_write_data_t_i_w    = iact_write_data_t_i;
    assign stride_x_i_w             = stride_x_i;
    assign stride_y_i_w             = stride_y_i;
    assign compute_mask_i_w         = compute_mask_i;
    assign router_mode_iact_i_w     = router_mode_iact_i;
    assign router_mode_wght_i_w     = router_mode_wght_i;
    assign router_mode_psum_i_w     = router_mode_psum_i;

  end

  
  for (cc=0; cc<CLUSTER_COLUMNS; cc=cc+1) begin
    for (cr=0; cr<CLUSTER_ROWS; cr=cr+1) begin

      ///IACT ASSIGNMENTS
      for (g=0; g<NUM_GLB_IACT; g=g+1) begin
        for (b=0; b<ROUTER_MODES_IACT; b=b+1) begin
          assign gen_x[cc].gen_y[cr].router_mode_iact_i_w[g*ROUTER_MODES_IACT+b] =
          router_mode_iact_reg[cc * CLUSTER_ROWS * NUM_GLB_IACT * ROUTER_MODES_IACT +
                             cr * NUM_GLB_IACT * ROUTER_MODES_IACT +
                             g * ROUTER_MODES_IACT + b];
        end
        if (cc == 0) begin
          assign gen_x[(cc + 1) % CLUSTER_COLUMNS].gen_y[cr].enable_src_side_iact_cluster_w[g] = gen_x[cc].gen_y[cr].enable_dst_side_iact_cluster_w[g];
          assign gen_x[(cc + 1) % CLUSTER_COLUMNS].gen_y[cr].ready_dst_side_iact_cluster_w[g]  = gen_x[cc].gen_y[cr].ready_src_side_iact_cluster_w[g];
          for (b=0; b<TRANS_BITWIDTH_IACT; b=b+1) begin
            assign gen_x[(cc + 1) % CLUSTER_COLUMNS].gen_y[cr].data_src_side_iact_cluster_w[g*TRANS_BITWIDTH_IACT+b] 
            = gen_x[cc].gen_y[cr].data_dst_side_iact_cluster_w[g * TRANS_BITWIDTH_IACT + b];
          end
        end
        if (cr != CLUSTER_ROWS - 1) begin
          for (b=0; b<TRANS_BITWIDTH_IACT; b=b+1) begin
            assign gen_x[cc].gen_y[cr + 1].data_src_top_iact_cluster_w[g*TRANS_BITWIDTH_IACT+b] = 
            gen_x[cc].gen_y[cr].data_dst_bottom_iact_cluster_w[g*TRANS_BITWIDTH_IACT+b];
          end
          assign gen_x[cc].gen_y[cr + 1].enable_src_top_iact_cluster_w[g] = gen_x[cc].gen_y[cr].enable_dst_bottom_iact_cluster_w[g];
          assign gen_x[cc].gen_y[cr + 1].ready_dst_top_iact_cluster_w[g]  = gen_x[cc].gen_y[cr].ready_src_bottom_iact_cluster_w[g];
        end

        if (cr != 0) begin
          for (b=0; b<TRANS_BITWIDTH_IACT; b=b+1) begin
            assign gen_x[cc].gen_y[cr - 1].data_src_bottom_iact_cluster_w[g*TRANS_BITWIDTH_IACT+b] = 
            gen_x[cc].gen_y[cr].data_dst_top_iact_cluster_w[g*TRANS_BITWIDTH_IACT+b];
          end
          assign gen_x[cc].gen_y[cr - 1].ready_dst_bottom_iact_cluster_w[g]  = gen_x[cc].gen_y[cr].ready_src_top_iact_cluster_w[g];
          assign gen_x[cc].gen_y[cr - 1].enable_src_bottom_iact_cluster_w[g] = gen_x[cc].gen_y[cr].enable_dst_top_iact_cluster_w[g];
        end

        for (b=0; b<TRANS_BITWIDTH_IACT; b=b+1) begin
          assign gen_x[cc].gen_y[cr].iact_data_i_cluster_w[g * TRANS_BITWIDTH_IACT+b] =
                  iact_data_i_w[cc * CLUSTER_ROWS * NUM_GLB_IACT * TRANS_BITWIDTH_IACT +
                                  cr * NUM_GLB_IACT * TRANS_BITWIDTH_IACT + 
                                  g * TRANS_BITWIDTH_IACT + b];
        end
        for (b=0; b<IACT_MEM_ADDR_BITS; b=b+1) begin
          assign gen_x[cc].gen_y[cr].iact_addr_i_cluster_w[g * IACT_MEM_ADDR_BITS + b]   =
          mem_addr_iact[cc * CLUSTER_ROWS * NUM_GLB_IACT * IACT_MEM_ADDR_BITS +
                        cr * NUM_GLB_IACT * IACT_MEM_ADDR_BITS +
                        g * IACT_MEM_ADDR_BITS + b];
        end
        assign gen_x[cc].gen_y[cr].iact_enable_i_cluster_w[g] = (iact_enable_i_w[cc*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT+g] |
                                                        iact_enable_comp_reg[cc*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT+g]);
        assign iact_ready_o_reg[cc*NUM_GLB_IACT*CLUSTER_ROWS+cr*NUM_GLB_IACT+g] = gen_x[cc].gen_y[cr].iact_ready_o_cluster_w[g];
      end
      for (pec=0; pec<PE_COLUMNS; pec=pec+1) begin
        for (per=0; per<PE_ROWS; per=per+1) begin
          for (b=0; b<$clog2(NUM_GLB_IACT); b=b+1) begin
            assign gen_x[cc].gen_y[cr].iact_choose_cluster_i_w[per*PE_COLUMNS*$clog2(NUM_GLB_IACT)+pec*$clog2(NUM_GLB_IACT)+b]
            = iact_choose_i[cc*CLUSTER_ROWS*PES*$clog2(NUM_GLB_IACT)+cr*PES*$clog2(NUM_GLB_IACT)+per*PE_COLUMNS*$clog2(NUM_GLB_IACT)+pec*$clog2(NUM_GLB_IACT)+b];
          end
          assign gen_x[cc].gen_y[cr].compute_cluster_i_w[pec*PE_ROWS+per] = compute_cluster_i_reg[cc*CLUSTER_ROWS*PES+cr*PES+pec*PE_ROWS+per];
        end
      end
      
      ///WGHT ASSIGNMENTS
      for (g=0; g<NUM_GLB_WGHT; g=g+1) begin
        for (b=0; b<ROUTER_MODES_WGHT; b=b+1) begin
          assign gen_x[cc].gen_y[cr].router_mode_wght_i_w[g*ROUTER_MODES_WGHT+b] =
                  router_mode_wght_reg[cc * CLUSTER_ROWS * NUM_GLB_WGHT * ROUTER_MODES_WGHT +
                                     cr * NUM_GLB_WGHT * ROUTER_MODES_WGHT + 
                                     g * ROUTER_MODES_WGHT + b];
        end

        assign gen_x[(cc + 1) % CLUSTER_COLUMNS].gen_y[cr].enable_src_side_wght_cluster_w[g] = gen_x[cc].gen_y[cr].enable_dst_side_wght_cluster_w[g];
        assign gen_x[(cc + 1) % CLUSTER_COLUMNS].gen_y[cr].ready_dst_side_wght_cluster_w[g]  = gen_x[cc].gen_y[cr].ready_src_side_wght_cluster_w[g];

        for (b=0; b<TRANS_BITWIDTH_WGHT; b=b+1) begin
          assign gen_x[(cc + 1) % CLUSTER_COLUMNS].gen_y[cr].data_src_side_wght_cluster_w[g*TRANS_BITWIDTH_WGHT+b] =
          gen_x[cc].gen_y[cr].data_dst_side_wght_cluster_w[g*TRANS_BITWIDTH_WGHT+b];
          assign gen_x[cc].gen_y[cr].wght_data_i_cluster_w[g * TRANS_BITWIDTH_WGHT+b] = 
                  wght_data_i_w[cc * CLUSTER_ROWS * NUM_GLB_WGHT * TRANS_BITWIDTH_WGHT +
                                  cr * NUM_GLB_WGHT * TRANS_BITWIDTH_WGHT + 
                                  g * TRANS_BITWIDTH_WGHT + b];
        end
        assign gen_x[cc].gen_y[cr].wght_enable_i_cluster_w[g] = wght_enable_i_w[cc*NUM_GLB_WGHT*CLUSTER_ROWS+cr*NUM_GLB_WGHT+g];
        assign wght_ready_o_reg[cc*NUM_GLB_WGHT*CLUSTER_ROWS+cr*NUM_GLB_WGHT+g] = gen_x[cc].gen_y[cr].wght_ready_o_cluster_w[g];
      end

      ///PSUM ASSIGNMENTS
      for (g=0; g<NUM_GLB_PSUM; g=g+1) begin
        for (b=0; b<ROUTER_MODES_PSUM; b=b+1) begin
          assign gen_x[cc].gen_y[cr].router_mode_psum_i_w[g*ROUTER_MODES_PSUM+b] =
          router_mode_psum_reg[cc * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM +
                             cr * NUM_GLB_PSUM * ROUTER_MODES_PSUM +
                             g * ROUTER_MODES_PSUM + b];
        end
        assign gen_x[cc].gen_y[cr].psum_choose_cluster_i_w[g] = psum_choose_i[cc * CLUSTER_ROWS * NUM_GLB_PSUM + cr * NUM_GLB_PSUM + g];

        if (cr != CLUSTER_ROWS - 1) begin
          for (b=0; b<TRANS_BITWIDTH_PSUM; b=b+1) begin
            assign gen_x[cc].gen_y[cr + 1].data_src_top_psum_cluster_w[g*TRANS_BITWIDTH_PSUM+b] =
            gen_x[cc].gen_y[cr].data_dst_bottom_psum_cluster_w[g*TRANS_BITWIDTH_PSUM+b];
          end
          assign gen_x[cc].gen_y[cr + 1].enable_src_top_psum_cluster_w[g] = gen_x[cc].gen_y[cr].enable_dst_bottom_psum_cluster_w[g];
          assign gen_x[cc].gen_y[cr + 1].ready_dst_top_psum_cluster_w[g]  = gen_x[cc].gen_y[cr].ready_src_bottom_psum_cluster_w[g];
        end

        if (cr != 0) begin
          for (b=0; b<TRANS_BITWIDTH_PSUM; b=b+1) begin
            assign gen_x[cc].gen_y[cr - 1].data_src_bottom_psum_cluster_w[g*TRANS_BITWIDTH_PSUM+b] = 
            gen_x[cc].gen_y[cr].data_dst_top_psum_cluster_w[g*TRANS_BITWIDTH_PSUM+b];
          end
          assign gen_x[cc].gen_y[cr - 1].ready_dst_bottom_psum_cluster_w[g]  = gen_x[cc].gen_y[cr].ready_src_top_psum_cluster_w[g];
          assign gen_x[cc].gen_y[cr - 1].enable_src_bottom_psum_cluster_w[g] = gen_x[cc].gen_y[cr].enable_dst_top_psum_cluster_w[g];
        end
        
        if (cr == CLUSTER_ROWS - 1) begin
          for (b=0; b<TRANS_BITWIDTH_PSUM; b=b+1) begin
            assign gen_x[cc].gen_y[cr].data_src_bottom_psum_cluster_w[g*TRANS_BITWIDTH_PSUM+b] = 0;
          end
          assign gen_x[cc].gen_y[cr].enable_src_bottom_psum_cluster_w[g] = 0;
        end

        for (b=0; b<TRANS_BITWIDTH_PSUM; b=b+1) begin
          assign gen_x[cc].gen_y[cr].psum_data_i_cluster_w[g * TRANS_BITWIDTH_PSUM + b] = 
          psum_data_i_reg[cc * CLUSTER_ROWS * NUM_GLB_PSUM * TRANS_BITWIDTH_PSUM +
                          cr * NUM_GLB_PSUM * TRANS_BITWIDTH_PSUM + 
                          g * TRANS_BITWIDTH_PSUM + b];
          assign psum_data_o_reg[cc * CLUSTER_ROWS * NUM_GLB_PSUM * TRANS_BITWIDTH_PSUM +
                                 cr * NUM_GLB_PSUM * TRANS_BITWIDTH_PSUM + 
                                 g * TRANS_BITWIDTH_PSUM + b]
          = gen_x[cc].gen_y[cr].psum_data_o_cluster_w[g * TRANS_BITWIDTH_PSUM + b];
        end
        for (b=0; b<PSUM_MEM_ADDR_BITS; b=b+1) begin
          assign gen_x[cc].gen_y[cr].psum_addr_i_cluster_w[g * PSUM_MEM_ADDR_BITS + b] =
          mem_addr_psum[cc * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS +
                        cr * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS +
                        g * PSUM_MEM_ADDR_BITS + b];
        end
        assign gen_x[cc].gen_y[cr].psum_enable_i_cluster_w[g] = psum_enable_i_reg[cc*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM+g];

        assign psum_ready_o_cluster_reg[cc*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM+g] = gen_x[cc].gen_y[cr].psum_ready_o_cluster_w[g];
      
        assign psum_cluster_enable_o_reg[cc*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM+g] = gen_x[cc].gen_y[cr].psum_enable_o_cluster_w[g];
        assign gen_x[cc].gen_y[cr].psum_ready_i_cluster_w[g] = psum_ready_i_reg[cc*NUM_GLB_PSUM*CLUSTER_ROWS+cr*NUM_GLB_PSUM+g];
      end
    end
  end
  endgenerate

endmodule