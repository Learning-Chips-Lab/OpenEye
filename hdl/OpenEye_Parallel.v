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

module OpenEye_Parallel #(
`ifdef USE_EXTERNAL_PARAMS
    `include "parameters.vh"
`else
    // Defaultwerte
    parameter  CLUSTER_ROWS        = 8,
    parameter  NUM_GLB_IACT        = 3,
    parameter  PE_COLUMNS          = 4,
    parameter  NUM_GLB_PSUM        = 4,
    parameter  NUM_GLB_WGHT        = 3,
    parameter  PE_ROWS             = 3,
`endif
    ///Set parameters
    parameter  IS_TOPLEVEL         = 1,
    parameter  SERIAL              = 0,
    parameter  DATA_IACT_BITWIDTH  = 8,
    parameter  DATA_PSUM_BITWIDTH  = 20,
    parameter  DATA_WGHT_BITWIDTH  = 8,
    parameter  TRANS_BITWIDTH_IACT = 24,
    parameter  TRANS_BITWIDTH_WGHT = 24,
    parameter  TRANS_BITWIDTH_PSUM = 40,
    parameter  DATA_IACT_OVERHEAD  = 4,
    parameter  CLUSTER_COLUMNS     = 2,
    parameter  IACT_PER_PE         = 16,
    parameter  PSUM_PER_PE         = 32,
    parameter  WGHT_PER_PE         = 96,
    parameter  IACT_ADDR_PER_PE    = 9,
    parameter  WGHT_ADDR_PER_PE    = 16,
    parameter  IACT_MEM_ADDR_WORDS = 512,
    parameter  PSUM_MEM_ADDR_WORDS = 384,
    parameter  ROUTER_MODES_IACT   = 6,
    parameter  ROUTER_MODES_WGHT   = 1,
    parameter  ROUTER_MODES_PSUM   = 3,
    parameter  BANO_MODES          = 2,
    parameter  AF_MODES            = 4,
    localparam IACT_MEM_ADDR_BITS  = $clog2(IACT_MEM_ADDR_WORDS),
    localparam PSUM_MEM_ADDR_BITS  = $clog2(PSUM_MEM_ADDR_WORDS),
    localparam PES                 = PE_COLUMNS * PE_ROWS,
    localparam CLUSTERS            = CLUSTER_COLUMNS * CLUSTER_ROWS

) (
    ///Clock and Reset Ports

    input                                                      clk_i,
    input                                                      rst_ni,
    input                                                      compute_i,
    ///Ports for GLBs and PEs
    input      [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_data_i,
    input      [                    CLUSTERS*NUM_GLB_IACT-1:0] iact_enable_i,
    output     [                    CLUSTERS*NUM_GLB_IACT-1:0] iact_ready_o,
    input      [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_data_i,
    input      [                    CLUSTERS*NUM_GLB_WGHT-1:0] wght_enable_i,
    output reg [                    CLUSTERS*NUM_GLB_WGHT-1:0] wght_ready_o,
    input      [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_i,
    input      [                    CLUSTERS*NUM_GLB_PSUM-1:0] psum_enable_i,
    output reg [                    CLUSTERS*NUM_GLB_PSUM-1:0] psum_ready_o,
    output reg [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_o,
    output reg [                    CLUSTERS*NUM_GLB_PSUM-1:0] psum_enable_o,
    input      [                    CLUSTERS*NUM_GLB_PSUM-1:0] psum_ready_i,
    input                                                      status_reg_enable_i,
    input                                                      data_mode_i,
    input      [               $clog2(DATA_PSUM_BITWIDTH)-1:0] fraction_bit_i,
    input      [                                         15:0] needed_cycles_i,
    input      [                $clog2(CLUSTER_COLUMNS+1)-1:0] needed_x_cls_i,
    input      [                   $clog2(CLUSTER_ROWS+1)-1:0] needed_y_cls_i,
    input      [                                          3:0] needed_iact_cycles_i,
    input      [                    $clog2(PSUM_PER_PE+1)-1:0] filters_i,
    input      [               $clog2(IACT_ADDR_PER_PE+1)-1:0] iact_addr_len_i,
    input      [                 $clog2(WGHT_ADDR_PER_PE)-1:0] wght_addr_len_i,
    input      [          $clog2(BANO_MODES)*NUM_GLB_PSUM-1:0] bano_cluster_mode_i,
    input      [                         $clog2(AF_MODES)-1:0] af_cluster_mode_i,
    input      [                             NUM_GLB_PSUM-1:0] pooling_cluster_mode_i,
    input      [                                          3:0] delay_psum_glb_i,
    input      [                    $clog2(IACT_PER_PE+1)-1:0] input_activations_i,
    input      [                                          2:0] stride_x_i,
    input      [                                          2:0] stride_y_i,
    input      [                          $clog2(PE_ROWS)-1:0] kernel_per_pe_cluster_i,
    input      [                             CLUSTERS*PES-1:0] compute_mask_i,
    input      [      $clog2(NUM_GLB_IACT+1)*CLUSTERS*PES-1:0] iact_choose_i,
    input      [  ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0] router_mode_iact_i,
    input      [  ROUTER_MODES_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] router_mode_wght_i,
    input      [  ROUTER_MODES_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] router_mode_psum_i,
    input      [                                        8-1:0] needed_psum_storage_cycles_i,
    input      [                                        8-1:0] needed_iact_channel_cycles_i


);
  ///#######################
  ///Reset synchronization
  ///#######################
  wire rst_n;

  RST_SYNC rst_sync_top (
      .clk_i (clk_i),
      .rst_ni(rst_ni),
      .rst_no(rst_n)
  );

`ifdef COCOTB_SIM
  initial begin
    if (IS_TOPLEVEL) begin
      $dumpvars(0, OpenEye_Parallel);
    end
  end
`endif

  ///#######################
  ///Register
  ///#######################

  ///Register, that occupy hyperparameters
  reg                                                  data_mode_reg;
  reg  [               $clog2(DATA_PSUM_BITWIDTH)-1:0] fraction_bit_reg;
  reg  [                                         15:0] needed_cycles_reg;
  reg  [                $clog2(CLUSTER_COLUMNS+1)-1:0] needed_x_cls_reg;
  reg  [                   $clog2(CLUSTER_ROWS+1)-1:0] needed_y_cls_reg;
  reg  [                                          3:0] needed_iact_cycles_reg;
  reg  [                    $clog2(PSUM_PER_PE+1)-1:0] filters_reg;
  reg  [                 $clog2(IACT_ADDR_PER_PE)-1:0] iact_addr_len_reg;
  reg  [                 $clog2(WGHT_ADDR_PER_PE)-1:0] wght_addr_len_reg;
  reg  [          $clog2(BANO_MODES)*NUM_GLB_PSUM-1:0] bano_cluster_mode_reg;
  reg  [            $clog2(AF_MODES)*NUM_GLB_PSUM-1:0] af_cluster_mode_reg;
  reg  [                             NUM_GLB_PSUM-1:0] pooling_cluster_mode_reg;
  reg  [                                          3:0] delay_psum_glb_reg;
  reg  [                    $clog2(IACT_PER_PE+1)-1:0] input_activations_reg;
  reg  [                                          2:0] stride_x_reg;
  reg  [                                          2:0] stride_y_reg;
  reg  [                             CLUSTERS*PES-1:0] compute_cluster_i_reg;
  reg  [                             CLUSTERS*PES-1:0] compute_mask_reg;
  ///Register for the psum FSM
  reg  [                                         15:0] fsm_psum_cycle;
  reg                                                  data_write_enable;
  reg                                                  results_ready;
  reg  [                                         15:0] finished_cycles;
  reg  [                   $clog2(CLUSTER_ROWS+1)-1:0] storage_cycles;

  ///Register, that configure the chip
  reg  [      $clog2(NUM_GLB_IACT+1)*CLUSTERS*PES-1:0] iact_choose_reg;
  reg  [                    CLUSTERS*NUM_GLB_PSUM-1:0] psum_choose_reg;
  reg  [  ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0] router_mode_iact_reg;
  reg  [  ROUTER_MODES_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] router_mode_wght_reg;
  reg  [  ROUTER_MODES_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] router_mode_psum_reg;
  reg                                                  psum_router_set_reg;
  reg                                                  computing;
  reg  [ IACT_MEM_ADDR_BITS*CLUSTERS*NUM_GLB_IACT-1:0] mem_addr_iact;
  reg  [ PSUM_MEM_ADDR_BITS*CLUSTERS*NUM_GLB_PSUM-1:0] mem_addr_psum;
  reg  [                       PSUM_MEM_ADDR_BITS-1:0] mem_addr_psum_storage;
  reg  [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_i_reg;
  wire [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_o_reg;
  reg  [                    CLUSTERS*NUM_GLB_PSUM-1:0] psum_enable_delay;
  reg  [                    CLUSTERS*NUM_GLB_PSUM-1:0] psum_enable_i_reg;
  reg  [                    CLUSTERS*NUM_GLB_PSUM-1:0] psum_enable_o_reg;
  reg  [                    CLUSTERS*NUM_GLB_PSUM-1:0] psum_ready_i_reg;
  wire [                    CLUSTERS*NUM_GLB_PSUM-1:0] psum_cluster_enable_o_reg;
  reg  [                    CLUSTERS*NUM_GLB_PSUM-1:0] psum_ready_o_reg;
  wire [                    CLUSTERS*NUM_GLB_PSUM-1:0] psum_ready_o_cluster_reg;
  ///Wires and Regs for Ports
  wire [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_data_i_w;
  wire [                    CLUSTERS*NUM_GLB_IACT-1:0] iact_enable_i_w;
  wire [                    CLUSTERS*NUM_GLB_IACT-1:0] iact_ready_o_w;
  wire [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_data_i_w;
  wire [                    CLUSTERS*NUM_GLB_WGHT-1:0] wght_enable_i_w;
  wire [                    CLUSTERS*NUM_GLB_WGHT-1:0] wght_ready_o_w;
  wire                                                 compute_i_w;
  reg                                                  iact_transmitted;
  reg                                                  psum_transmitted;
  reg                                                  start_new_cycle;
  wire                                                 status_reg_enable_i_w;
  wire                                                 data_mode_i_w;
  wire [               $clog2(DATA_PSUM_BITWIDTH)-1:0] fraction_bit_i_w;
  wire [                                         15:0] needed_cycles_i_w;
  wire [                $clog2(CLUSTER_COLUMNS+1)-1:0] needed_x_cls_i_w;
  wire [                   $clog2(CLUSTER_ROWS+1)-1:0] needed_y_cls_i_w;
  wire [                                          3:0] needed_iact_cycles_i_w;
  wire [                    $clog2(PSUM_PER_PE+1)-1:0] filters_i_w;
  wire [               $clog2(IACT_ADDR_PER_PE+1)-1:0] iact_addr_len_i_w;
  wire [                 $clog2(WGHT_ADDR_PER_PE)-1:0] wght_addr_len_i_w;
  wire [          $clog2(BANO_MODES)*NUM_GLB_PSUM-1:0] bano_cluster_mode_i_w;
  wire [            $clog2(AF_MODES)*NUM_GLB_PSUM-1:0] af_cluster_mode_i_w;
  wire [                             NUM_GLB_PSUM-1:0] pooling_cluster_mode_i_w;
  wire [                                          3:0] delay_psum_glb_i_w;
  wire [                    $clog2(IACT_PER_PE+1)-1:0] input_activations_i_w;
  wire [                                          2:0] stride_x_i_w;
  wire [                                          2:0] stride_y_i_w;
  wire [                          $clog2(PE_ROWS)-1:0] kernel_per_pe_cluster_i_w;
  wire [                             CLUSTERS*PES-1:0] compute_mask_i_w;
  reg  [TRANS_BITWIDTH_IACT*CLUSTERS*NUM_GLB_IACT-1:0] iact_data_i_reg;
  reg  [                    CLUSTERS*NUM_GLB_IACT-1:0] iact_enable_i_reg;
  reg  [                    CLUSTERS*NUM_GLB_IACT-1:0] iact_ready_o_reg;
  reg  [TRANS_BITWIDTH_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] wght_data_i_reg;
  reg  [                    CLUSTERS*NUM_GLB_WGHT-1:0] wght_enable_i_reg;
  wire [                    CLUSTERS*NUM_GLB_WGHT-1:0] wght_ready_o_reg;
  reg                                                  compute_i_reg;
  reg                                                  status_reg_enable_i_reg;
  reg                                                  data_mode_i_reg;
  reg  [               $clog2(DATA_PSUM_BITWIDTH)-1:0] fraction_bit_i_reg;
  reg  [                                         15:0] needed_cycles_i_reg;
  reg  [                $clog2(CLUSTER_COLUMNS+1)-1:0] needed_x_cls_i_reg;
  reg  [                   $clog2(CLUSTER_ROWS+1)-1:0] needed_y_cls_i_reg;
  reg  [                                          3:0] needed_iact_cycles_i_reg;
  reg  [                    $clog2(PSUM_PER_PE+1)-1:0] filters_i_reg;
  reg  [               $clog2(IACT_ADDR_PER_PE+1)-1:0] iact_addr_len_i_reg;
  reg  [                 $clog2(WGHT_ADDR_PER_PE)-1:0] wght_addr_len_i_reg;
  reg  [          $clog2(BANO_MODES)*NUM_GLB_PSUM-1:0] bano_cluster_mode_i_reg;
  reg  [            $clog2(AF_MODES)*NUM_GLB_PSUM-1:0] af_cluster_mode_i_reg;
  reg  [                             NUM_GLB_PSUM-1:0] pooling_cluster_mode_i_reg;
  reg  [                    $clog2(IACT_PER_PE+1)-1:0] input_activations_i_reg;
  reg  [                                          2:0] stride_x_i_reg;
  reg  [                                          2:0] stride_y_i_reg;
  reg  [                          $clog2(PE_ROWS)-1:0] kernel_per_pe_cluster_i_reg;
  reg  [                             CLUSTERS*PES-1:0] compute_mask_i_reg;
  reg  [  ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0] router_mode_iact_i_reg;
  reg  [  ROUTER_MODES_IACT*CLUSTERS*NUM_GLB_IACT-1:0] router_mode_iact_storage;
  reg  [  ROUTER_MODES_WGHT*CLUSTERS*NUM_GLB_WGHT-1:0] router_mode_wght_i_reg;
  reg  [  ROUTER_MODES_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] router_mode_psum_i_reg;
  reg  [                               4*CLUSTERS-1:0] iact_router_offset;
  reg                                                  enable_stream_reg;
  reg  [                                          7:0] data_stream_reg;
  reg  [                                          4:0] iact_pes_per_router;


  ///#######################
  ///States of the FSM
  ///#######################

  localparam IDLE_TRANSMI = 0;
  localparam FIRST_PARAMS = 1;
  localparam SECOND_PARAMS = 2;

  reg [1:0] fsm_transmission_state;

  localparam MAIN_IDLE = 0;
  localparam COMPUTING = 1;

  reg fsm_last_state;
  reg fsm_current_state;

  localparam IACT_IDLE = 0;
  localparam CALCULATE_IACT = 1;
  localparam WAIT = 2;

  reg [1:0] fsm_iact_last_state;
  reg [1:0] fsm_iact_current_state;

  localparam WGHT_READY = 0;
  localparam WGHT_BUSY = 1;

  reg fsm_wght_current_state;

  localparam PSUM_IDLE = 0;
  localparam CALCULATE_PSUM = 1;
  localparam GET_RESULTS = 2;
  localparam WAIT_FOR_RESULTS = 3;
  localparam SEND_RESULTS = 4;

  reg [2:0] fsm_psum_last_state;
  reg [2:0] fsm_psum_current_state;
  ///#######################
  ///Process
  ///#######################

  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin  ///Reset
      iact_data_i_reg <= 0;
    end else begin
      iact_data_i_reg <= iact_data_i;
    end

  end
  ///#######################
  ///Process
  ///#######################

  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin  ///Reset
      enable_stream_reg      <= 0;
      enable_stream_reg      <= 0;
      fsm_transmission_state <= FIRST_PARAMS;
      data_stream_reg        <= 0;
    end else begin
      enable_stream_reg <= 0;
      data_stream_reg   <= 0;
      case (fsm_transmission_state)
        IDLE_TRANSMI: begin
          if (status_reg_enable_i_reg & (!status_reg_enable_i)) begin
            fsm_transmission_state <= FIRST_PARAMS;
          end
        end
        FIRST_PARAMS: begin
          enable_stream_reg      <= 1;
          data_stream_reg        <= {wght_addr_len_i_reg, stride_x_i_reg, data_mode_i_reg};
          fsm_transmission_state <= SECOND_PARAMS;
        end
        SECOND_PARAMS: begin
          enable_stream_reg      <= 1;
          data_stream_reg        <= {{4{1'd0}}, {iact_addr_len_i_reg}};
          fsm_transmission_state <= IDLE_TRANSMI;
        end
        default: begin
        end
      endcase
    end
  end
  reg [7:0] cycle_break_counter;
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin  ///Reset
      start_new_cycle             <= 0;
      data_mode_reg               <= 0;
      fraction_bit_reg            <= 0;
      needed_cycles_reg           <= 0;
      needed_x_cls_reg            <= 0;
      needed_y_cls_reg            <= 0;
      needed_iact_cycles_reg      <= 0;
      filters_reg                 <= 0;
      iact_addr_len_reg           <= 0;
      stride_x_reg                <= 0;
      stride_y_reg                <= 0;
      bano_cluster_mode_reg       <= 0;
      af_cluster_mode_reg         <= 0;
      pooling_cluster_mode_reg    <= 0;
      delay_psum_glb_reg          <= 0;
      input_activations_reg       <= 0;
      compute_mask_reg            <= 0;
      fsm_last_state              <= MAIN_IDLE;
      fsm_current_state           <= MAIN_IDLE;

      data_write_enable           <= 1;
      finished_cycles             <= 0;
      compute_mask_i_reg          <= 0;
      compute_cluster_i_reg       <= 0;
      computing                   <= 0;
      compute_i_reg               <= 0;
      status_reg_enable_i_reg     <= 0;
      data_mode_i_reg             <= 0;
      fraction_bit_i_reg          <= 0;
      needed_cycles_i_reg         <= 0;
      needed_x_cls_i_reg          <= 0;
      needed_y_cls_i_reg          <= 0;
      needed_iact_cycles_i_reg    <= 0;
      filters_i_reg               <= 0;
      iact_addr_len_i_reg         <= 0;
      bano_cluster_mode_i_reg     <= 0;
      af_cluster_mode_i_reg       <= 0;
      pooling_cluster_mode_i_reg  <= 0;
      input_activations_i_reg     <= 0;
      stride_x_i_reg              <= 0;
      stride_y_i_reg              <= 0;
      kernel_per_pe_cluster_i_reg <= 0;
      router_mode_wght_reg        <= 0;
      router_mode_wght_i_reg      <= 0;
      router_mode_psum_i_reg      <= 0;
      iact_pes_per_router         <= 5;
      wght_addr_len_i_reg         <= 0;
      wght_addr_len_reg           <= 0;
      router_mode_iact_reg        <= 0;
      cycle_break_counter         <= 0;

    end else begin

      ///Regs for ports
      compute_i_reg               <= compute_i;
      status_reg_enable_i_reg     <= status_reg_enable_i;
      data_mode_i_reg             <= data_mode_i;
      fraction_bit_i_reg          <= fraction_bit_i;
      needed_cycles_i_reg         <= needed_cycles_i;
      needed_x_cls_i_reg          <= needed_x_cls_i;
      needed_y_cls_i_reg          <= needed_y_cls_i;
      needed_iact_cycles_i_reg    <= needed_iact_cycles_i;
      filters_i_reg               <= filters_i;
      iact_addr_len_i_reg         <= iact_addr_len_i;
      bano_cluster_mode_i_reg     <= bano_cluster_mode_i;
      af_cluster_mode_i_reg       <= {NUM_GLB_PSUM{af_cluster_mode_i}};
      pooling_cluster_mode_i_reg  <= pooling_cluster_mode_i;
      input_activations_i_reg     <= input_activations_i;
      stride_x_i_reg              <= stride_x_i;
      stride_y_i_reg              <= stride_y_i;
      kernel_per_pe_cluster_i_reg <= kernel_per_pe_cluster_i;
      wght_addr_len_i_reg         <= wght_addr_len_i;
      compute_mask_i_reg          <= compute_mask_i;

      if (status_reg_enable_i_w) begin
        cycle_break_counter         <= 0;
        data_mode_reg               <= data_mode_i_w;
        fraction_bit_reg            <= fraction_bit_i_w;
        needed_cycles_reg           <= needed_cycles_i_w;
        needed_x_cls_reg            <= needed_x_cls_i_w;
        needed_y_cls_reg            <= needed_y_cls_i_w;
        needed_iact_cycles_reg      <= needed_iact_cycles_i_w;
        filters_reg                 <= filters_i_w;
        iact_addr_len_reg           <= 0;
        wght_addr_len_reg           <= wght_addr_len_i_w;
        bano_cluster_mode_reg       <= bano_cluster_mode_i_w;
        af_cluster_mode_reg         <= af_cluster_mode_i_w;
        pooling_cluster_mode_reg    <= pooling_cluster_mode_i_w;
        delay_psum_glb_reg          <= delay_psum_glb_i;
        input_activations_reg       <= input_activations_i_w;
        stride_x_reg                <= stride_x_i_w;
        stride_y_reg                <= stride_y_i_w;
        kernel_per_pe_cluster_i_reg <= kernel_per_pe_cluster_i_w;
        compute_mask_reg            <= compute_mask_i_w;
        router_mode_iact_reg        <= router_mode_iact_i;
        router_mode_iact_storage    <= router_mode_iact_i;
        router_mode_wght_reg        <= router_mode_wght_i;
      end
      case (fsm_current_state)

        MAIN_IDLE: begin
          data_write_enable    <= 1;
          computing            <= 0;
          cycle_break_counter  <= 0;
          router_mode_wght_reg <= router_mode_wght_i;
          if (compute_i_w) begin
            fsm_last_state    <= MAIN_IDLE;
            fsm_current_state <= COMPUTING;
            data_write_enable <= 0;
            computing         <= 1;
          end
          if (compute_i) begin
            finished_cycles <= 0;
          end
        end

        COMPUTING: begin
          start_new_cycle       <= 0;
          compute_cluster_i_reg <= 0;
          cycle_break_counter   <= 0;
          if (psum_transmitted & (iact_enable_i == 0) & (wght_enable_i == 0)) begin
            cycle_break_counter <= cycle_break_counter + 1;
            if (cycle_break_counter >= needed_y_cls_i_reg * 2) begin 
              cycle_break_counter <= 0;
              start_new_cycle     <= 1;
              if (start_new_cycle != 1) begin
                finished_cycles <= finished_cycles + 1;
              end
              if (finished_cycles < needed_cycles_i_reg) begin
                compute_cluster_i_reg <= compute_mask_reg;
              end
            end
          end
          if (fsm_psum_current_state == SEND_RESULTS) begin
            computing <= 0;
          end
          if (fsm_psum_last_state == SEND_RESULTS) begin
            fsm_last_state    <= COMPUTING;
            fsm_current_state <= MAIN_IDLE;
          end

        end

        default: begin
        end
      endcase
    end
  end

  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin  ///Reset
      fsm_wght_current_state <= WGHT_READY;
      wght_enable_i_reg      <= 0;
      wght_data_i_reg        <= 0;
      wght_ready_o           <= 0;
    end else begin
      wght_data_i_reg   <= wght_data_i;
      wght_enable_i_reg <= wght_enable_i;
      wght_ready_o      <= wght_ready_o_reg;
      case (fsm_wght_current_state)

        WGHT_READY: begin
          if (compute_i) begin
            fsm_wght_current_state <= WGHT_BUSY;
          end
        end

        WGHT_BUSY: begin
          if ((compute_cluster_i_reg != 0) & (finished_cycles == needed_cycles_i_reg)) begin
            fsm_wght_current_state <= WGHT_READY;
          end
          if ((data_mode_reg) & (finished_cycles == 1)) begin
            fsm_wght_current_state <= WGHT_READY;
          end
        end

        default: begin
        end
      endcase
    end
  end
reg [7:0]needed_psum_storage_cycles_reg;
reg [7:0] iact_channel_counter_reg;
  integer g_psum, b_psum, cc_psum, cr_psum;
  always @(posedge clk_i, negedge rst_n) begin
    if (!rst_n) begin  ///Reset
      psum_transmitted       <= 0;
      fsm_psum_cycle         <= 0;
      fsm_psum_last_state    <= PSUM_IDLE;
      fsm_psum_current_state <= PSUM_IDLE;
      mem_addr_psum          <= {PSUM_MEM_ADDR_BITS * CLUSTERS * NUM_GLB_PSUM{1'b1}};
      mem_addr_psum_storage  <= 0;
      psum_enable_delay      <= 0;
      psum_enable_i_reg      <= 0;
      psum_enable_o_reg      <= 0;
      psum_ready_i_reg       <= 0;
      psum_enable_o          <= 0;
      psum_ready_o           <= 0;
      psum_ready_o_reg       <= 0;
      psum_data_i_reg        <= 0;
      psum_data_o            <= 0;
      psum_choose_reg        <= 0;
      storage_cycles         <= 0;
      router_mode_psum_reg   <= 0;
      psum_router_set_reg    <= 1;
      iact_channel_counter_reg               <= 0;
      results_ready = 0;
      needed_psum_storage_cycles_reg <= 0;

    end else begin
      if (status_reg_enable_i_w) begin
        router_mode_psum_reg <= router_mode_psum_i;
        needed_psum_storage_cycles_reg <= needed_psum_storage_cycles_i;
      end
      if (compute_i_w) begin
        mem_addr_psum <= 0;
      end
      psum_enable_o     <= psum_enable_o_reg;
      psum_enable_i_reg <= psum_enable_i;
      psum_data_i_reg   <= psum_data_i;
      psum_data_o       <= psum_data_o_reg;
      psum_ready_i_reg  <= psum_ready_i;
      psum_ready_o      <= psum_ready_o_reg;
      if (start_new_cycle == 1) begin
        psum_transmitted <= 0;
      end
      if (computing) begin
        if (needed_y_cls_reg == 1) begin
          psum_choose_reg <= (2 ** (CLUSTER_ROWS * CLUSTER_COLUMNS * NUM_GLB_PSUM) - 1);
        end else begin
          if (needed_y_cls_reg == 2) begin
            psum_choose_reg <= {CLUSTER_ROWS{{NUM_GLB_PSUM{1'b1}}, {NUM_GLB_PSUM{1'b0}}}};
          end else begin
            if (needed_y_cls_reg == 4 & (CLUSTER_ROWS >= 4)) begin
              psum_choose_reg <= {((CLUSTER_ROWS+1)/2){{NUM_GLB_PSUM{1'b1}},{NUM_GLB_PSUM{3'b000}}}};
            end else begin
              psum_choose_reg <= (2 ** (CLUSTER_ROWS * CLUSTER_COLUMNS * NUM_GLB_PSUM) - 1);
            end
          end
        end
      end
      case (fsm_psum_current_state)
        PSUM_IDLE: begin
          psum_transmitted    <= 0;
          fsm_psum_cycle      <= 0;
          fsm_psum_last_state <= PSUM_IDLE;
          storage_cycles      <= 0;

          if (computing) begin
            mem_addr_psum          <= 0;
            fsm_psum_last_state    <= PSUM_IDLE;
            fsm_psum_current_state <= CALCULATE_PSUM;
          end
          if (status_reg_enable_i_w) begin
            mem_addr_psum <= 0;
          end
          router_mode_psum_reg <= router_mode_psum_i;
          for (g_psum = 0; g_psum < CLUSTERS * NUM_GLB_PSUM; g_psum = g_psum + 1) begin
            if (psum_enable_i_reg[g_psum]) begin
              mem_addr_psum[g_psum*PSUM_MEM_ADDR_BITS +: PSUM_MEM_ADDR_BITS] <=
                mem_addr_psum[g_psum*PSUM_MEM_ADDR_BITS +: PSUM_MEM_ADDR_BITS] + 1;
            end
          end
        end

        CALCULATE_PSUM: begin
          if (psum_ready_i_reg != 0) begin
            psum_ready_i_reg <= psum_ready_i_reg;
          end
          if (finished_cycles == 0) begin
            psum_transmitted <= 1;
          end
          if (results_ready == 0 & (psum_ready_i_reg != 0)) begin
            results_ready = 1;
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM; g_psum = g_psum + 1) begin
                  results_ready = results_ready & (psum_ready_o_cluster_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum] |
                   (router_mode_psum_reg[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM + 2] == 0));
                end
              end
            end
          end
          if (results_ready & (psum_ready_i_reg != 0)) begin
            fsm_psum_cycle <= fsm_psum_cycle + 1;
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM; g_psum = g_psum + 1) begin
                  if ((fsm_psum_cycle != 0) & (router_mode_psum_reg[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM + 2] == 1)) begin

                    mem_addr_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + cr_psum * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + g_psum * PSUM_MEM_ADDR_BITS +: PSUM_MEM_ADDR_BITS] <=
                      mem_addr_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + cr_psum * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + g_psum * PSUM_MEM_ADDR_BITS +: PSUM_MEM_ADDR_BITS] + 1;


                  end
                  psum_enable_i_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum] <= 1;
                end
              end
            end
            if ((!SERIAL & (((fsm_psum_cycle[$clog2(
                    PSUM_PER_PE+1
                )-1:0] >= (filters_reg + 1) / 2) & !data_mode_reg) |
                    ((fsm_psum_cycle >= (needed_cycles_i_reg + 1) / 2) & data_mode_reg))) |
                    (SERIAL & (((fsm_psum_cycle >= {{10{1'd0}},filters_reg}) & !data_mode_reg) |
                               ((fsm_psum_cycle >= (needed_cycles_i_reg + 1) / 2) &
                                data_mode_reg)))) begin
              results_ready = 0;
              fsm_psum_cycle         <= 0;
              psum_enable_i_reg      <= 0;
              fsm_psum_last_state    <= CALCULATE_PSUM;
              fsm_psum_current_state <= GET_RESULTS;
              for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                  for (g_psum = 0; g_psum < NUM_GLB_PSUM; g_psum = g_psum + 1) begin
                    if (router_mode_psum_reg[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM + 2] == 1) begin
                      mem_addr_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + cr_psum * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + g_psum * PSUM_MEM_ADDR_BITS +: PSUM_MEM_ADDR_BITS] <=
                        mem_addr_psum_storage;

                    end
                  end
                end
              end
            end
          end else begin
            if (start_new_cycle & (psum_ready_i_reg == 0)) begin
              fsm_psum_cycle <= 0;
              for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                  for (g_psum = 0; g_psum < NUM_GLB_PSUM; g_psum = g_psum + 1) begin
                    psum_ready_i_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum] <= 1;
                  end
                end
              end
            end
          end
        end

        GET_RESULTS: begin
          results_ready = 1;
          psum_ready_i_reg <= psum_ready_i_reg;
          for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
            for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
              for (g_psum = 0; g_psum < NUM_GLB_PSUM; g_psum = g_psum + 1) begin
                if (psum_cluster_enable_o_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum]) begin
                  mem_addr_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + cr_psum * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + g_psum * PSUM_MEM_ADDR_BITS +: PSUM_MEM_ADDR_BITS] <=
                      mem_addr_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + cr_psum * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + g_psum * PSUM_MEM_ADDR_BITS +: PSUM_MEM_ADDR_BITS] + 1;
                end
                results_ready = results_ready & (psum_cluster_enable_o_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum] | 
                (router_mode_psum_reg[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM + 2] == 0));
              end
            end
          end
          if (results_ready) begin
            psum_router_set_reg <= 0;
            fsm_psum_cycle <= fsm_psum_cycle + 1;
          end
          if ((!SERIAL & (((fsm_psum_cycle[$clog2(
                  PSUM_PER_PE+1
              )-1:0] >= (filters_reg + 1) / 2) & !data_mode_reg) |
                  ((fsm_psum_cycle >= (needed_cycles_i_reg + 1) / 2) & data_mode_reg))) |
                  (SERIAL & (((fsm_psum_cycle[$clog2(
                  PSUM_PER_PE+1
              )-1:0] >= filters_reg) & !data_mode_reg) |
                  ((fsm_psum_cycle >= (needed_cycles_i_reg + 1) / 2) & data_mode_reg)))) begin
            if ((finished_cycles == needed_cycles_reg) | (fsm_current_state == MAIN_IDLE) | data_mode_reg) begin
              fsm_psum_last_state    <= GET_RESULTS;
              fsm_psum_current_state <= WAIT_FOR_RESULTS;
              mem_addr_psum          <= 0;
              psum_ready_i_reg       <= 0;
              fsm_psum_cycle         <= 0;
              psum_transmitted       <= 1;
            end else begin
              if ((needed_y_cls_reg >= 2) & !psum_router_set_reg) begin
                psum_router_set_reg <= 1;
                iact_channel_counter_reg <= iact_channel_counter_reg + 1;
                if ((iact_channel_counter_reg == needed_iact_channel_cycles_i - 1)) begin
                  iact_channel_counter_reg <= 0;
                  for (cr_psum = 1; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                    for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                      for (g_psum = 0; g_psum < NUM_GLB_PSUM; g_psum = g_psum + 1) begin
                        router_mode_psum_reg[cc_psum*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g_psum*ROUTER_MODES_PSUM+2] <=
                        router_mode_psum_reg[cc_psum*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+(cr_psum-1)*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g_psum*ROUTER_MODES_PSUM+2];
                      end
                    end
                  end
                  if (storage_cycles != (needed_psum_storage_cycles_reg[3:0] - 1)) begin
                    for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                      for (g_psum = 0; g_psum < NUM_GLB_PSUM; g_psum = g_psum + 1) begin
                        router_mode_psum_reg[cc_psum*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+g_psum*ROUTER_MODES_PSUM+2] <= 0;
                      end
                    end

                    storage_cycles <= storage_cycles + 1;

                    for (cr_psum = 1; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                      for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                        for (g_psum = 0; g_psum < NUM_GLB_PSUM; g_psum = g_psum + 1) begin
                          router_mode_psum_reg[cc_psum*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g_psum*ROUTER_MODES_PSUM+2] <=
                          router_mode_psum_reg[cc_psum*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+(cr_psum-1)*NUM_GLB_PSUM*ROUTER_MODES_PSUM+g_psum*ROUTER_MODES_PSUM+2];
                        end
                      end
                    end
                  end else begin
                    for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                      for (g_psum = 0; g_psum < NUM_GLB_PSUM; g_psum = g_psum + 1) begin
                        router_mode_psum_reg[cc_psum*ROUTER_MODES_PSUM*NUM_GLB_PSUM*CLUSTER_ROWS+g_psum*ROUTER_MODES_PSUM+2] <= 1;
                      end
                    end
                    storage_cycles <= 0;
                  end
                end
              end
              psum_transmitted       <= 1;
              fsm_psum_last_state    <= GET_RESULTS;
              fsm_psum_current_state <= CALCULATE_PSUM;
              fsm_psum_cycle         <= 0;
              if (storage_cycles == needed_psum_storage_cycles_reg[3:0] - 1) begin
                storage_cycles <= 0;
                if (SERIAL) begin
                  mem_addr_psum_storage <= mem_addr_psum_storage +
                      {{(PSUM_MEM_ADDR_BITS - $clog2(PSUM_PER_PE + 1)) {1'd0}}, filters_reg};
                  for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                    for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                      for (g_psum = 0; g_psum < NUM_GLB_PSUM; g_psum = g_psum + 1) begin
                        mem_addr_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + cr_psum * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + g_psum * PSUM_MEM_ADDR_BITS +: PSUM_MEM_ADDR_BITS] <=
                            mem_addr_psum_storage + {{(PSUM_MEM_ADDR_BITS - $clog2(PSUM_PER_PE + 1)) {1'd0}}, filters_reg};
                      end
                    end
                  end
                end else begin
                  mem_addr_psum_storage <= mem_addr_psum_storage +
                      ({{(PSUM_MEM_ADDR_BITS - $clog2(PSUM_PER_PE + 1)) {1'd0}}, filters_reg} + 1) /
                      2;
                end
              end else begin
                storage_cycles <= storage_cycles + 1;
                for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                  for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                    for (g_psum = 0; g_psum < NUM_GLB_PSUM; g_psum = g_psum + 1) begin
                      mem_addr_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + cr_psum * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS + g_psum * PSUM_MEM_ADDR_BITS +: PSUM_MEM_ADDR_BITS] <=
                          mem_addr_psum_storage;
                    end
                  end
                end
              end
              psum_ready_i_reg <= 0;
            end
          end
        end
        WAIT_FOR_RESULTS: begin

          fsm_psum_cycle <= fsm_psum_cycle + 1;
          if (fsm_psum_cycle == 0) begin
            fsm_psum_cycle         <= 0;
            fsm_psum_last_state    <= WAIT_FOR_RESULTS;
            fsm_psum_current_state <= SEND_RESULTS;
          end
        end
        SEND_RESULTS: begin
          for (g_psum = 0; g_psum < NUM_GLB_PSUM * CLUSTERS; g_psum = g_psum + 1) begin
            psum_ready_i_reg[g_psum]  <= 0;
            psum_ready_o_reg[g_psum]  <= 1;
            psum_enable_delay[g_psum] <= psum_enable_i_reg[g_psum];
            psum_enable_o_reg[g_psum] <= psum_enable_delay[g_psum];
            if (psum_enable_i_reg[g_psum] == 1) begin
              mem_addr_psum[g_psum * PSUM_MEM_ADDR_BITS +: PSUM_MEM_ADDR_BITS] <=
                      mem_addr_psum[g_psum * PSUM_MEM_ADDR_BITS +: PSUM_MEM_ADDR_BITS] + 1;
            end
          end
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

        default: begin
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
    for (clusters_x = 0; clusters_x < CLUSTER_COLUMNS; clusters_x = clusters_x + 1) begin : gen_x
      for (clusters_y = 0; clusters_y < CLUSTER_ROWS; clusters_y = clusters_y + 1) begin : gen_y

        ///Wires
        ////////////////////////////////
        ///Selection
        //////////////////////////////////
        wire [      $clog2(NUM_GLB_IACT+1)*PES-1:0] iact_choose_cluster_i_w;
        wire [                    NUM_GLB_PSUM-1:0] psum_choose_cluster_i_w;
        wire [                             PES-1:0] compute_cluster_i_w;

        ///Router Modes
        //////////////////////////////////
        wire [  ROUTER_MODES_IACT*NUM_GLB_IACT-1:0] router_mode_iact_i_w;
        wire [                    NUM_GLB_WGHT-1:0] router_mode_wght_i_w;
        wire [  ROUTER_MODES_PSUM*NUM_GLB_PSUM-1:0] router_mode_psum_i_w;

        ///IACT Connection
        //////////////////////////////////
        ///Connect above
        wire [                    NUM_GLB_IACT-1:0] enable_src_top_iact_cluster_w;
        wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] data_src_top_iact_cluster_w;
        wire [                    NUM_GLB_IACT-1:0] ready_src_top_iact_cluster_w;

        wire [                    NUM_GLB_IACT-1:0] enable_dst_top_iact_cluster_w;
        wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] data_dst_top_iact_cluster_w;
        wire [                    NUM_GLB_IACT-1:0] ready_dst_top_iact_cluster_w;

        ///Connect other side
        wire [                    NUM_GLB_IACT-1:0] enable_src_side_iact_cluster_w;
        wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] data_src_side_iact_cluster_w;
        wire [                    NUM_GLB_IACT-1:0] ready_src_side_iact_cluster_w;

        wire [                    NUM_GLB_IACT-1:0] enable_dst_side_iact_cluster_w;
        wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] data_dst_side_iact_cluster_w;
        wire [                    NUM_GLB_IACT-1:0] ready_dst_side_iact_cluster_w;

        ///Connect below
        wire [                    NUM_GLB_IACT-1:0] enable_src_bottom_iact_cluster_w;
        wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] data_src_bottom_iact_cluster_w;
        wire [                    NUM_GLB_IACT-1:0] ready_src_bottom_iact_cluster_w;

        wire [                    NUM_GLB_IACT-1:0] enable_dst_bottom_iact_cluster_w;
        wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] data_dst_bottom_iact_cluster_w;
        wire [                    NUM_GLB_IACT-1:0] ready_dst_bottom_iact_cluster_w;

        ///Memory
        wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] iact_data_i_cluster_w;
        wire [ IACT_MEM_ADDR_BITS*NUM_GLB_IACT-1:0] iact_addr_i_cluster_w;
        wire [                    NUM_GLB_IACT-1:0] iact_enable_i_cluster_w;
        wire [                    NUM_GLB_IACT-1:0] iact_ready_o_cluster_w;

        ///WGHT Connection
        //////////////////////////////////
        ///Connect other side
        wire [                    NUM_GLB_WGHT-1:0] enable_src_side_wght_cluster_w;
        wire [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0] data_src_side_wght_cluster_w;
        wire [                    NUM_GLB_WGHT-1:0] ready_src_side_wght_cluster_w;

        wire [                    NUM_GLB_WGHT-1:0] enable_dst_side_wght_cluster_w;
        wire [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0] data_dst_side_wght_cluster_w;
        wire [                    NUM_GLB_WGHT-1:0] ready_dst_side_wght_cluster_w;

        ///Memory
        wire [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0] wght_data_i_cluster_w;
        wire [                    NUM_GLB_WGHT-1:0] wght_enable_i_cluster_w;
        wire [                    NUM_GLB_WGHT-1:0] wght_ready_o_cluster_w;

        ///PSUM Connection
        //////////////////////////////////
        ///Connect above
        wire [                    NUM_GLB_PSUM-1:0] enable_src_top_psum_cluster_w;
        wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] data_src_top_psum_cluster_w;
        wire [                    NUM_GLB_PSUM-1:0] ready_src_top_psum_cluster_w;

        wire [                    NUM_GLB_PSUM-1:0] enable_dst_top_psum_cluster_w;
        wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] data_dst_top_psum_cluster_w;
        wire [                    NUM_GLB_PSUM-1:0] ready_dst_top_psum_cluster_w;

        ///Connect below
        wire [                    NUM_GLB_PSUM-1:0] enable_src_bottom_psum_cluster_w;
        wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] data_src_bottom_psum_cluster_w;
        wire [                    NUM_GLB_PSUM-1:0] ready_src_bottom_psum_cluster_w;

        wire [                    NUM_GLB_PSUM-1:0] enable_dst_bottom_psum_cluster_w;
        wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] data_dst_bottom_psum_cluster_w;
        wire [                    NUM_GLB_PSUM-1:0] ready_dst_bottom_psum_cluster_w;

        ///Memory 1
        wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] psum_data_i_cluster_w;
        wire [ PSUM_MEM_ADDR_BITS*NUM_GLB_PSUM-1:0] psum_addr_i_cluster_w;
        wire [                    NUM_GLB_PSUM-1:0] psum_enable_i_cluster_w;
        wire [                    NUM_GLB_PSUM-1:0] psum_ready_o_cluster_w;

        ///Memory 2
        wire [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0] psum_data_o_cluster_w;
        wire [                    NUM_GLB_PSUM-1:0] psum_enable_o_cluster_w;
        wire [                    NUM_GLB_PSUM-1:0] psum_ready_i_cluster_w;

        OpenEye_Cluster #(
            .IS_TOPLEVEL       (0),
            .SERIAL            (SERIAL),
            .DATA_IACT_BITWIDTH(DATA_IACT_BITWIDTH),
            .DATA_PSUM_BITWIDTH(DATA_PSUM_BITWIDTH),
            .DATA_WGHT_BITWIDTH(DATA_WGHT_BITWIDTH),

            .TRANS_BITWIDTH_IACT(TRANS_BITWIDTH_IACT),
            .TRANS_BITWIDTH_PSUM(TRANS_BITWIDTH_PSUM),
            .TRANS_BITWIDTH_WGHT(TRANS_BITWIDTH_WGHT),
            .DATA_IACT_OVERHEAD (DATA_IACT_OVERHEAD),

            .NUM_GLB_IACT(NUM_GLB_IACT),
            .NUM_GLB_WGHT(NUM_GLB_WGHT),
            .NUM_GLB_PSUM(NUM_GLB_PSUM),

            .PE_ROWS   (PE_ROWS),
            .PE_COLUMNS(PE_COLUMNS),

            .IACT_PER_PE(IACT_PER_PE),
            .PSUM_PER_PE(PSUM_PER_PE),
            .WGHT_PER_PE(WGHT_PER_PE),

            .IACT_MEM_ADDR_WORDS(IACT_MEM_ADDR_WORDS),
            .PSUM_MEM_ADDR_WORDS(PSUM_MEM_ADDR_WORDS),

            .LEFT_CLUSTER  (clusters_x == 0),
            .TOP_CLUSTER   (clusters_y == 0),
            .BOTTOM_CLUSTER(clusters_y == CLUSTER_ROWS - 1)
        ) OpenEye_Cluster (
            ///Clock and Reset Line
            //////////////////////////////////
            .clk_i                   (clk_i),
            .rst_ni                  (rst_n),
            .data_write_enable_iact_i(1'd0),
            .data_write_enable_i     (data_write_enable),

            ///Selection
            //////////////////////////////////
            .iact_choose_i(iact_choose_cluster_i_w),
            .psum_choose_i(psum_choose_cluster_i_w),
            .compute_i    (compute_cluster_i_w),

            ///Router Modes
            //////////////////////////////////
            .router_mode_iact_i(router_mode_iact_i_w),
            .router_mode_wght_i(router_mode_wght_i_w),
            .router_mode_psum_i(router_mode_psum_i_w),

            ///IACT Connection
            //////////////////////////////////
            //Connect above
            .enable_src_top_iact(enable_src_top_iact_cluster_w),
            .data_src_top_iact  (data_src_top_iact_cluster_w),
            .ready_src_top_iact (ready_src_top_iact_cluster_w),

            .enable_dst_top_iact(enable_dst_top_iact_cluster_w),
            .data_dst_top_iact  (data_dst_top_iact_cluster_w),
            .ready_dst_top_iact (ready_dst_top_iact_cluster_w),

            ///Connect other side
            .enable_src_side_iact(enable_src_side_iact_cluster_w),
            .data_src_side_iact  (data_src_side_iact_cluster_w),
            .ready_src_side_iact (ready_src_side_iact_cluster_w),

            .enable_dst_side_iact(enable_dst_side_iact_cluster_w),
            .data_dst_side_iact  (data_dst_side_iact_cluster_w),
            .ready_dst_side_iact (ready_dst_side_iact_cluster_w),

            ///Connect below
            .enable_src_bottom_iact(enable_src_bottom_iact_cluster_w),
            .data_src_bottom_iact  (data_src_bottom_iact_cluster_w),
            .ready_src_bottom_iact (ready_src_bottom_iact_cluster_w),

            .enable_dst_bottom_iact(enable_dst_bottom_iact_cluster_w),
            .data_dst_bottom_iact  (data_dst_bottom_iact_cluster_w),
            .ready_dst_bottom_iact (ready_dst_bottom_iact_cluster_w),

            ///Memory
            .ext_mem_iact_data_i  (iact_data_i_cluster_w),
            .ext_mem_iact_addr_i  (iact_addr_i_cluster_w),
            .ext_mem_iact_enable_i(iact_enable_i_cluster_w),
            .ext_mem_iact_ready_o (iact_ready_o_cluster_w),

            ///WGHT Connection
            //////////////////////////////////
            ///Connect other side
            .enable_src_side_wght(enable_src_side_wght_cluster_w),
            .data_src_side_wght  (data_src_side_wght_cluster_w),
            .ready_src_side_wght (ready_src_side_wght_cluster_w),

            .enable_dst_side_wght(enable_dst_side_wght_cluster_w),
            .data_dst_side_wght  (data_dst_side_wght_cluster_w),
            .ready_dst_side_wght (ready_dst_side_wght_cluster_w),

            ///Memory
            .ext_mem_wght_data_i  (wght_data_i_cluster_w),
            .ext_mem_wght_enable_i(wght_enable_i_cluster_w),
            .ext_mem_wght_ready_o (wght_ready_o_cluster_w),

            ///PSUM Connection
            //////////////////////////////////
            ///Connect above
            .enable_src_top_psum(enable_src_top_psum_cluster_w),
            .data_src_top_psum  (data_src_top_psum_cluster_w),
            .ready_src_top_psum (ready_src_top_psum_cluster_w),

            .enable_dst_top_psum(enable_dst_top_psum_cluster_w),
            .data_dst_top_psum  (data_dst_top_psum_cluster_w),
            .ready_dst_top_psum (ready_dst_top_psum_cluster_w),

            ///Connect below
            .enable_src_bottom_psum(enable_src_bottom_psum_cluster_w),
            .data_src_bottom_psum  (data_src_bottom_psum_cluster_w),
            .ready_src_bottom_psum (ready_src_bottom_psum_cluster_w),

            .enable_dst_bottom_psum(enable_dst_bottom_psum_cluster_w),
            .data_dst_bottom_psum  (data_dst_bottom_psum_cluster_w),
            .ready_dst_bottom_psum (ready_dst_bottom_psum_cluster_w),

            ///Memory 1
            .ext_mem_psum_data_i  (psum_data_i_cluster_w),
            .ext_mem_psum_addr_i  (psum_addr_i_cluster_w),
            .ext_mem_psum_enable_i(psum_enable_i_cluster_w),
            .ext_mem_psum_ready_o (psum_ready_o_cluster_w),

            ///Memory 2
            .ext_mem_psum_data_o  (psum_data_o_cluster_w),
            .ext_mem_psum_enable_o(psum_enable_o_cluster_w),
            .ext_mem_psum_ready_i (psum_ready_i_cluster_w),

            .bano_cluster_mode_i(bano_cluster_mode_reg),
            .af_cluster_mode_i  (af_cluster_mode_reg),
            .delay_psum_glb_i   (delay_psum_glb_reg),
            .enable_stream_i    (enable_stream_reg),
            .data_stream_i      (data_stream_reg)
        );

      end
    end

    genvar cr_gen, cc_gen, pec_gen, per_gen, g_gen, b_gen;

    if (IS_TOPLEVEL) begin : gen_pipelined_ports

      assign iact_data_i_w             = iact_data_i_reg;
      assign iact_enable_i_w           = iact_enable_i_reg;
      assign wght_data_i_w             = wght_data_i_reg;
      assign wght_enable_i_w           = wght_enable_i_reg;

      assign compute_i_w               = compute_i_reg;
      assign status_reg_enable_i_w     = status_reg_enable_i_reg;
      assign data_mode_i_w             = data_mode_i_reg;
      assign fraction_bit_i_w          = fraction_bit_i_reg;
      assign needed_cycles_i_w         = needed_cycles_i_reg;
      assign needed_x_cls_i_w          = needed_x_cls_i_reg;
      assign needed_y_cls_i_w          = needed_y_cls_i_reg;
      assign needed_iact_cycles_i_w    = needed_iact_cycles_i_reg;
      assign filters_i_w               = filters_i_reg;
      assign iact_addr_len_i_w         = iact_addr_len_i_reg;
      assign wght_addr_len_i_w         = wght_addr_len_i_reg;
      assign bano_cluster_mode_i_w     = bano_cluster_mode_i_reg;
      assign af_cluster_mode_i_w       = af_cluster_mode_i_reg;
      assign pooling_cluster_mode_i_w  = pooling_cluster_mode_i_reg;
      assign delay_psum_glb_i_w        = delay_psum_glb_reg;
      assign input_activations_i_w     = input_activations_i_reg;
      assign stride_x_i_w              = stride_x_i_reg;
      assign stride_y_i_w              = stride_y_i_reg;
      assign kernel_per_pe_cluster_i_w = kernel_per_pe_cluster_i_reg;
      assign compute_mask_i_w          = compute_mask_i_reg;

    end else begin : gen_non_pipelined_ports

      assign iact_data_i_w             = iact_data_i;
      assign iact_enable_i_w           = iact_enable_i;
      assign wght_data_i_w             = wght_data_i;
      assign wght_enable_i_w           = wght_enable_i;

      assign compute_i_w               = compute_i;
      assign status_reg_enable_i_w     = status_reg_enable_i;
      assign data_mode_i_w             = data_mode_i;
      assign fraction_bit_i_w          = fraction_bit_i;
      assign needed_cycles_i_w         = needed_cycles_i;
      assign needed_x_cls_i_w          = needed_x_cls_i;
      assign needed_y_cls_i_w          = needed_y_cls_i;
      assign needed_iact_cycles_i_w    = needed_iact_cycles_i;
      assign filters_i_w               = filters_i;
      assign iact_addr_len_i_w         = iact_addr_len_i;
      assign wght_addr_len_i_w         = wght_addr_len_i;
      assign bano_cluster_mode_i_w     = bano_cluster_mode_i;
      assign af_cluster_mode_i_w       = {NUM_GLB_PSUM{af_cluster_mode_i}};
      assign pooling_cluster_mode_i_w  = pooling_cluster_mode_i;
      assign delay_psum_glb_i_w        = delay_psum_glb_i;
      assign input_activations_i_w     = input_activations_i;
      assign stride_x_i_w              = stride_x_i;
      assign stride_y_i_w              = stride_y_i;
      assign kernel_per_pe_cluster_i_w = kernel_per_pe_cluster_i;
      assign compute_mask_i_w          = compute_mask_i;
    end


    for (cc_gen = 0; cc_gen < CLUSTER_COLUMNS; cc_gen = cc_gen + 1) begin
      for (cr_gen = 0; cr_gen < CLUSTER_ROWS; cr_gen = cr_gen + 1) begin

        ///IACT ASSIGNMENTS
        for (g_gen = 0; g_gen < NUM_GLB_IACT; g_gen = g_gen + 1) begin
          if (SERIAL) begin : gen_serial_router_mode_iact
              assign gen_x[cc_gen].gen_y[cr_gen].router_mode_iact_i_w[g_gen*ROUTER_MODES_IACT+: ROUTER_MODES_IACT] =
            router_mode_iact_i[cc_gen * CLUSTER_ROWS * NUM_GLB_IACT * ROUTER_MODES_IACT +
                              cr_gen * NUM_GLB_IACT * ROUTER_MODES_IACT +
                              g_gen * ROUTER_MODES_IACT +: ROUTER_MODES_IACT];
          end else begin : gen_parallel_router_mode_iact
              assign gen_x[cc_gen].gen_y[cr_gen].router_mode_iact_i_w[g_gen*ROUTER_MODES_IACT+: ROUTER_MODES_IACT] =
            router_mode_iact_reg[cc_gen * CLUSTER_ROWS * NUM_GLB_IACT * ROUTER_MODES_IACT +
                              cr_gen * NUM_GLB_IACT * ROUTER_MODES_IACT +
                              g_gen * ROUTER_MODES_IACT +: ROUTER_MODES_IACT];
          end
          assign gen_x[cc_gen].gen_y[cr_gen].enable_src_side_iact_cluster_w[g_gen] = gen_x[(cc_gen + 1) % CLUSTER_COLUMNS].gen_y[cr_gen].enable_dst_side_iact_cluster_w[g_gen];
          assign gen_x[cc_gen].gen_y[cr_gen].ready_dst_side_iact_cluster_w[g_gen]  = gen_x[(cc_gen + 1) % CLUSTER_COLUMNS].gen_y[cr_gen].ready_src_side_iact_cluster_w[g_gen];
          assign gen_x[cc_gen].gen_y[cr_gen].data_src_side_iact_cluster_w[g_gen*TRANS_BITWIDTH_IACT+: TRANS_BITWIDTH_IACT] 
          = gen_x[(cc_gen + 1) % CLUSTER_COLUMNS].gen_y[cr_gen].data_dst_side_iact_cluster_w[g_gen * TRANS_BITWIDTH_IACT +: TRANS_BITWIDTH_IACT];
          if (cr_gen != CLUSTER_ROWS - 1) begin : gen_router_iact_bottom_connect
              assign gen_x[cc_gen].gen_y[cr_gen + 1].data_src_top_iact_cluster_w[g_gen*TRANS_BITWIDTH_IACT+: TRANS_BITWIDTH_IACT] = 
            gen_x[cc_gen].gen_y[cr_gen].data_dst_bottom_iact_cluster_w[g_gen*TRANS_BITWIDTH_IACT+: TRANS_BITWIDTH_IACT];
            assign gen_x[cc_gen].gen_y[cr_gen + 1].enable_src_top_iact_cluster_w[g_gen] = gen_x[cc_gen].gen_y[cr_gen].enable_dst_bottom_iact_cluster_w[g_gen];
            assign gen_x[cc_gen].gen_y[cr_gen + 1].ready_dst_top_iact_cluster_w[g_gen]  = gen_x[cc_gen].gen_y[cr_gen].ready_src_bottom_iact_cluster_w[g_gen];
          end

          if (cr_gen != 0) begin : gen_router_iact_top_connect
            assign gen_x[cc_gen].gen_y[cr_gen - 1].data_src_bottom_iact_cluster_w[g_gen*TRANS_BITWIDTH_IACT+: TRANS_BITWIDTH_IACT] = 
            gen_x[cc_gen].gen_y[cr_gen].data_dst_top_iact_cluster_w[g_gen*TRANS_BITWIDTH_IACT+: TRANS_BITWIDTH_IACT];
            assign gen_x[cc_gen].gen_y[cr_gen - 1].ready_dst_bottom_iact_cluster_w[g_gen]  = gen_x[cc_gen].gen_y[cr_gen].ready_src_top_iact_cluster_w[g_gen];
            assign gen_x[cc_gen].gen_y[cr_gen - 1].enable_src_bottom_iact_cluster_w[g_gen] = gen_x[cc_gen].gen_y[cr_gen].enable_dst_top_iact_cluster_w[g_gen];
          end
          if (cr_gen == 0) begin : gen_router_iact_top_edge
            assign gen_x[cc_gen].gen_y[cr_gen].enable_src_top_iact_cluster_w[g_gen] = 0;
            assign gen_x[cc_gen].gen_y[cr_gen].ready_dst_top_iact_cluster_w[g_gen]  = 0;
          end
          if (cr_gen == CLUSTER_ROWS - 1) begin : gen_router_iact_bottom_edge
            assign gen_x[cc_gen].gen_y[cr_gen].ready_dst_bottom_iact_cluster_w[g_gen]  = 0;
            assign gen_x[cc_gen].gen_y[cr_gen].enable_src_bottom_iact_cluster_w[g_gen] = 0;
          end

          assign gen_x[cc_gen].gen_y[cr_gen].iact_data_i_cluster_w[g_gen * TRANS_BITWIDTH_IACT+: TRANS_BITWIDTH_IACT] =
                iact_data_i_w[cc_gen * CLUSTER_ROWS * NUM_GLB_IACT * TRANS_BITWIDTH_IACT +
                                cr_gen * NUM_GLB_IACT * TRANS_BITWIDTH_IACT + 
                                g_gen * TRANS_BITWIDTH_IACT +: TRANS_BITWIDTH_IACT];
          assign gen_x[cc_gen].gen_y[cr_gen].iact_addr_i_cluster_w[g_gen * IACT_MEM_ADDR_BITS +: IACT_MEM_ADDR_BITS]   =
          mem_addr_iact[cc_gen * CLUSTER_ROWS * NUM_GLB_IACT * IACT_MEM_ADDR_BITS +
                        cr_gen * NUM_GLB_IACT * IACT_MEM_ADDR_BITS +
                        g_gen * IACT_MEM_ADDR_BITS +: IACT_MEM_ADDR_BITS];
          assign gen_x[cc_gen].gen_y[cr_gen].iact_enable_i_cluster_w[g_gen] = iact_enable_i_w[cc_gen*NUM_GLB_IACT*CLUSTER_ROWS+cr_gen*NUM_GLB_IACT+g_gen];
          assign iact_ready_o[cc_gen*NUM_GLB_IACT*CLUSTER_ROWS+cr_gen*NUM_GLB_IACT+g_gen] = gen_x[cc_gen].gen_y[cr_gen].iact_ready_o_cluster_w[g_gen];
        end
        for (pec_gen = 0; pec_gen < PE_COLUMNS; pec_gen = pec_gen + 1) begin
          for (per_gen = 0; per_gen < PE_ROWS; per_gen = per_gen + 1) begin
              assign gen_x[cc_gen].gen_y[cr_gen].iact_choose_cluster_i_w[per_gen*PE_COLUMNS*$clog2(
                  NUM_GLB_IACT+1
              )+pec_gen*$clog2(
                  NUM_GLB_IACT+1
              )+:$clog2(NUM_GLB_IACT + 1)] = iact_choose_i[cc_gen*CLUSTER_ROWS*PES*$clog2(
                  NUM_GLB_IACT+1
              )+cr_gen*PES*$clog2(
                  NUM_GLB_IACT+1
              )+per_gen*PE_COLUMNS*$clog2(
                  NUM_GLB_IACT+1
              )+pec_gen*$clog2(
                  NUM_GLB_IACT+1
              )+: $clog2(NUM_GLB_IACT + 1)];
            assign gen_x[cc_gen].gen_y[cr_gen].compute_cluster_i_w[pec_gen*PE_ROWS+per_gen] = compute_cluster_i_reg[cc_gen*CLUSTER_ROWS*PES+cr_gen*PES+pec_gen*PE_ROWS+per_gen];
          end
        end

        ///WGHT ASSIGNMENTS
        for (g_gen = 0; g_gen < NUM_GLB_WGHT; g_gen = g_gen + 1) begin
          assign gen_x[cc_gen].gen_y[cr_gen].router_mode_wght_i_w[g_gen*ROUTER_MODES_WGHT+: ROUTER_MODES_WGHT] =
                router_mode_wght_reg[cc_gen * CLUSTER_ROWS * NUM_GLB_WGHT * ROUTER_MODES_WGHT +
                                    cr_gen * NUM_GLB_WGHT * ROUTER_MODES_WGHT + 
                                    g_gen * ROUTER_MODES_WGHT +: ROUTER_MODES_WGHT];
          assign gen_x[(cc_gen + 1) % CLUSTER_COLUMNS].gen_y[cr_gen].enable_src_side_wght_cluster_w[g_gen] = gen_x[cc_gen].gen_y[cr_gen].enable_dst_side_wght_cluster_w[g_gen];
          assign gen_x[(cc_gen + 1) % CLUSTER_COLUMNS].gen_y[cr_gen].ready_dst_side_wght_cluster_w[g_gen]  = gen_x[cc_gen].gen_y[cr_gen].ready_src_side_wght_cluster_w[g_gen];

          assign gen_x[(cc_gen + 1) % CLUSTER_COLUMNS].gen_y[cr_gen].data_src_side_wght_cluster_w[g_gen*TRANS_BITWIDTH_WGHT+: TRANS_BITWIDTH_WGHT] =
          gen_x[cc_gen].gen_y[cr_gen].data_dst_side_wght_cluster_w[g_gen*TRANS_BITWIDTH_WGHT+: TRANS_BITWIDTH_WGHT];
          assign gen_x[cc_gen].gen_y[cr_gen].wght_data_i_cluster_w[g_gen * TRANS_BITWIDTH_WGHT+: TRANS_BITWIDTH_WGHT] = 
                  wght_data_i_w[cc_gen * CLUSTER_ROWS * NUM_GLB_WGHT * TRANS_BITWIDTH_WGHT +
                                  cr_gen * NUM_GLB_WGHT * TRANS_BITWIDTH_WGHT + 
                                  g_gen * TRANS_BITWIDTH_WGHT +:TRANS_BITWIDTH_WGHT];
          assign gen_x[cc_gen].gen_y[cr_gen].wght_enable_i_cluster_w[g_gen] = wght_enable_i_w[cc_gen*NUM_GLB_WGHT*CLUSTER_ROWS+cr_gen*NUM_GLB_WGHT+g_gen];
          assign wght_ready_o_reg[cc_gen*NUM_GLB_WGHT*CLUSTER_ROWS+cr_gen*NUM_GLB_WGHT+g_gen] = gen_x[cc_gen].gen_y[cr_gen].wght_ready_o_cluster_w[g_gen];
        end

        ///PSUM ASSIGNMENTS
        for (g_gen = 0; g_gen < NUM_GLB_PSUM; g_gen = g_gen + 1) begin
          assign gen_x[cc_gen].gen_y[cr_gen].router_mode_psum_i_w[g_gen*ROUTER_MODES_PSUM+: ROUTER_MODES_PSUM] =
          router_mode_psum_reg[cc_gen * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM +
                             cr_gen * NUM_GLB_PSUM * ROUTER_MODES_PSUM +
                             g_gen * ROUTER_MODES_PSUM+: ROUTER_MODES_PSUM];
          assign gen_x[cc_gen].gen_y[cr_gen].psum_choose_cluster_i_w[g_gen] = psum_choose_reg[cc_gen * CLUSTER_ROWS * NUM_GLB_PSUM + cr_gen * NUM_GLB_PSUM + g_gen];

          if (cr_gen != CLUSTER_ROWS - 1) begin : gen_router_iact_bottom_connect
            assign gen_x[cc_gen].gen_y[cr_gen + 1].data_src_top_psum_cluster_w[g_gen*TRANS_BITWIDTH_PSUM+: TRANS_BITWIDTH_PSUM] =
          gen_x[cc_gen].gen_y[cr_gen].data_dst_bottom_psum_cluster_w[g_gen*TRANS_BITWIDTH_PSUM+: TRANS_BITWIDTH_PSUM];
            assign gen_x[cc_gen].gen_y[cr_gen + 1].enable_src_top_psum_cluster_w[g_gen] = gen_x[cc_gen].gen_y[cr_gen].enable_dst_bottom_psum_cluster_w[g_gen];
            assign gen_x[cc_gen].gen_y[cr_gen + 1].ready_dst_top_psum_cluster_w[g_gen]  = gen_x[cc_gen].gen_y[cr_gen].ready_src_bottom_psum_cluster_w[g_gen];
          end

          if (cr_gen != 0) begin : gen_router_iact_top_connect
            assign gen_x[cc_gen].gen_y[cr_gen - 1].data_src_bottom_psum_cluster_w[g_gen*TRANS_BITWIDTH_PSUM+: TRANS_BITWIDTH_PSUM] = 
            gen_x[cc_gen].gen_y[cr_gen].data_dst_top_psum_cluster_w[g_gen*TRANS_BITWIDTH_PSUM+: TRANS_BITWIDTH_PSUM];
            assign gen_x[cc_gen].gen_y[cr_gen - 1].ready_dst_bottom_psum_cluster_w[g_gen]  = gen_x[cc_gen].gen_y[cr_gen].ready_src_top_psum_cluster_w[g_gen];
            assign gen_x[cc_gen].gen_y[cr_gen - 1].enable_src_bottom_psum_cluster_w[g_gen] = gen_x[cc_gen].gen_y[cr_gen].enable_dst_top_psum_cluster_w[g_gen];
          end

          if (cr_gen == 0) begin : gen_router_iact_top_edge
            assign gen_x[cc_gen].gen_y[cr_gen].enable_src_top_psum_cluster_w[g_gen] = 0;
          end
          if (cr_gen == CLUSTER_ROWS - 1) begin : gen_router_iact_bottom_edge
            assign gen_x[cc_gen].gen_y[cr_gen].data_src_bottom_psum_cluster_w[g_gen*TRANS_BITWIDTH_PSUM+: TRANS_BITWIDTH_PSUM] = 0;
            assign gen_x[cc_gen].gen_y[cr_gen].ready_dst_bottom_psum_cluster_w[g_gen]  = 0;
            assign gen_x[cc_gen].gen_y[cr_gen].enable_src_bottom_psum_cluster_w[g_gen] = 0;
          end

          assign gen_x[cc_gen].gen_y[cr_gen].psum_data_i_cluster_w[g_gen * TRANS_BITWIDTH_PSUM +: TRANS_BITWIDTH_PSUM] = 
          psum_data_i_reg[cc_gen * CLUSTER_ROWS * NUM_GLB_PSUM * TRANS_BITWIDTH_PSUM +
                          cr_gen * NUM_GLB_PSUM * TRANS_BITWIDTH_PSUM + 
                          g_gen * TRANS_BITWIDTH_PSUM +: TRANS_BITWIDTH_PSUM];
          assign psum_data_o_reg[cc_gen * CLUSTER_ROWS * NUM_GLB_PSUM * TRANS_BITWIDTH_PSUM +
                                 cr_gen * NUM_GLB_PSUM * TRANS_BITWIDTH_PSUM + 
                                 g_gen * TRANS_BITWIDTH_PSUM +: TRANS_BITWIDTH_PSUM]
          = gen_x[cc_gen].gen_y[cr_gen].psum_data_o_cluster_w[g_gen * TRANS_BITWIDTH_PSUM +: TRANS_BITWIDTH_PSUM];
          assign gen_x[cc_gen].gen_y[cr_gen].psum_addr_i_cluster_w[g_gen * PSUM_MEM_ADDR_BITS +:PSUM_MEM_ADDR_BITS] =
          mem_addr_psum[cc_gen * CLUSTER_ROWS * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS +
                        cr_gen * NUM_GLB_PSUM * PSUM_MEM_ADDR_BITS +
                        g_gen * PSUM_MEM_ADDR_BITS +: PSUM_MEM_ADDR_BITS];
          assign gen_x[cc_gen].gen_y[cr_gen].psum_enable_i_cluster_w[g_gen] = psum_enable_i_reg[cc_gen*NUM_GLB_PSUM*CLUSTER_ROWS+cr_gen*NUM_GLB_PSUM+g_gen];

          assign psum_ready_o_cluster_reg[cc_gen*NUM_GLB_PSUM*CLUSTER_ROWS+cr_gen*NUM_GLB_PSUM+g_gen] = gen_x[cc_gen].gen_y[cr_gen].psum_ready_o_cluster_w[g_gen];

          assign psum_cluster_enable_o_reg[cc_gen*NUM_GLB_PSUM*CLUSTER_ROWS+cr_gen*NUM_GLB_PSUM+g_gen] = gen_x[cc_gen].gen_y[cr_gen].psum_enable_o_cluster_w[g_gen];
          assign gen_x[cc_gen].gen_y[cr_gen].psum_ready_i_cluster_w[g_gen] = psum_ready_i_reg[cc_gen*NUM_GLB_PSUM*CLUSTER_ROWS+cr_gen*NUM_GLB_PSUM+g_gen];
        end
      end
    end
  endgenerate

endmodule
