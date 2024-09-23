// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: PE
///
/// OpenEye Processing Element (PE). The PE can perform multiply-accumulate (MAC) operations
/// on the input data. The PE can be configured to exploit sparsity in the input data
/// by ignoring zeros in the input data. The PE can also be configured to perform fixed-point
/// arithmetic. 
///
/// In order to operate, the Iact and Wght SPad memory must first be filled. After the memory has
/// been filled and a compute signal has been received, the internal FSM starts to prepare the
/// computation by reading the Iact SPads for its input data. By reading the address overhead of the
/// data SPad, the required data of the weight SPad memory is read and both values are sent to the
/// multiplier submodule. After multiplication, the corresponding Psum SPad is read and added. The
/// result is written back to the Psum SPad. 
///
/// Parameters:
///    IS_TOPLEVEL             - Decides, wether modul is topmodul or not
///    CREATE_VCD              - Decides, wether a vcd-file should be created
///    PARALLEL_MACS           - Number of MAC operations that are performed in parallel
///    DATA_IACT_BITWIDTH      - Width of input activation data
///    DATA_WGHT_BITWIDTH      - Width of weight data
///    DATA_PSUM_BITWIDTH      - Width of partial sum data, used in internal accumulator
///    DATA_IACT_IGNORE_ZEROS  - Number of zeros that can be ignored in sparse input activation data
///    DATA_WGHT_IGNORE_ZEROS  - Number of zeros that can be ignored in sparse weight data
///    IACT_DATA_ADDR          - Number of input activation data words in SPad
///    IACT_ADDR_ADDR          - Number of input activation adresses in SPad
///    WGHT_DATA_ADDR          - Number of weight data words in SPad
///    WGHT_ADDR_ADDR          - Number of weight adresses in SPad
///    PSUM_ADDR               - Number of partial sum data words in SPad
///    TRANS_BITWIDTH_IACT     - Width of iact input port 
///    TRANS_BITWIDTH_WGHT     - Width of weight input port
///    NUM_GLB_IACT            - Number of input activation global buffers
///   
/// Ports:
///    iact_select_i          - Select routing of input mux (see mux_iact), i.e. which input activation data is used)
///    iact_data_i            - Input activation data (includes actual iact data + overhead for ignoring zeros + address)
///    iact_enable_i          - Enable input activation data transfer
///    iact_ready_o           - Ready signal for input activation data transfer
///    wght_data_i            - Weight data (includes actual weight data + overhead for ignoring zeros + address)
///    wght_enable_i          - Enable weight data transfer
///    wght_ready_o           - Ready signal for weight data transfer
///    psum_data_i            - Partial sum and bias input data
///    psum_enable_i          - Enable partial sum data input transfer
///    psum_ready_o           - Ready signal for partial sum data input transfer
///    psum_data_o            - Partial sum output data
///    psum_enable_o          - Enable partial sum data output transfer
///    psum_ready_i           - Ready signal for partial sum data output transfer
///    compute_i              - Trigger computation
///    enable_stream_i        - Enable signal of data stream for parameters
///    data_stream_i          - Data stream for parameters
///

module PE 
#(

  parameter         IS_TOPLEVEL             = 1,
  parameter         CREATE_VCD              = 0,

  parameter integer PARALLEL_MACS           = 2,

  parameter integer DATA_IACT_BITWIDTH      = 8,
  parameter integer DATA_WGHT_BITWIDTH      = 8,
  parameter integer DATA_PSUM_BITWIDTH      = 20,
  parameter integer DATA_IACT_IGNORE_ZEROS  = 4,
  parameter integer DATA_WGHT_IGNORE_ZEROS  = 4,

  parameter integer IACT_DATA_ADDR          = 16,
  parameter integer IACT_ADDR_ADDR          = 9,
  
  parameter integer WGHT_DATA_ADDR          = 96,
  parameter integer WGHT_ADDR_ADDR          = 16,

  parameter integer PSUM_ADDR               = 32,

  parameter integer TRANS_BITWIDTH_IACT     = 24, // 3 * 8 bit data OR 2 * 12 bit data OR 6 * 4 bit addresses
  parameter integer TRANS_BITWIDTH_WGHT     = 24, // 3 * 8 bit weight OR 2 * 12 bit weight OR 3 * 8 bit addresses

  parameter integer NUM_GLB_IACT            = 3,

  // local parameters
  localparam integer IACT_ADDR_DATA          = $clog2(IACT_DATA_ADDR),
  localparam integer WGHT_ADDR_DATA          = $clog2(WGHT_DATA_ADDR),

  localparam integer IACT_ADDR_ADDR_BITWIDTH = $clog2(IACT_ADDR_ADDR),
  localparam integer IACT_ADDR_DATA_BITWIDTH = $clog2(IACT_ADDR_DATA),

  localparam integer IACT_DATA_DATA          = DATA_IACT_BITWIDTH + DATA_IACT_IGNORE_ZEROS,
  localparam integer IACT_DATA_ADDR_BITWIDTH = $clog2(IACT_DATA_ADDR),
  localparam integer IACT_DATA_DATA_BITWIDTH = $clog2(IACT_DATA_DATA),

  localparam integer WGHT_DATA_DATA          = (DATA_WGHT_BITWIDTH + DATA_WGHT_IGNORE_ZEROS) * PARALLEL_MACS,
  localparam integer WGHT_ADDR_ADDR_BITWIDTH = $clog2(WGHT_ADDR_ADDR),
  localparam integer WGHT_ADDR_DATA_BITWIDTH = $clog2(WGHT_ADDR_DATA),

  localparam integer WGHT_DATA_ADDR_BITWIDTH = $clog2(WGHT_DATA_ADDR),
  localparam integer WGHT_DATA_DATA_BITWIDTH = $clog2(WGHT_DATA_DATA),

  localparam integer PSUM_DATA               = DATA_PSUM_BITWIDTH,
  localparam integer PSUM_ADDR_BITWIDTH      = $clog2(PSUM_ADDR),
  localparam integer PSUM_DATA_BITWIDTH      = $clog2(PSUM_DATA),
  localparam integer PSUM_WORDS_PER_TRANSFER = PARALLEL_MACS,
  localparam integer TRANS_BITWIDTH_PSUM     = DATA_PSUM_BITWIDTH * PSUM_WORDS_PER_TRANSFER

) ( 
  input                                              clk_i,
  input                                              rst_ni,
  input       [$clog2(NUM_GLB_IACT)-1:0]             iact_select_i,
  input       [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] iact_data_i,
  input       [NUM_GLB_IACT-1:0]                     iact_enable_i,
  output      [NUM_GLB_IACT-1:0]                     iact_ready_o,
  input       [TRANS_BITWIDTH_WGHT-1:0]              wght_data_i,
  input                                              wght_enable_i,
  output reg                                         wght_ready_o,
  input       [TRANS_BITWIDTH_PSUM-1:0]              psum_data_i,
  input                                              psum_enable_i,
  output                                             psum_ready_o,
  output      [TRANS_BITWIDTH_PSUM-1:0]              psum_data_o,
  output reg                                         psum_enable_o,
  input                                              psum_ready_i,
  input                                              compute_i,
  input                                              enable_stream_i,
  input       [7:0]                                  data_stream_i
);

  reg [$clog2(16)-1:0]                        current_state_computing;

  reg [IACT_ADDR_DATA-1 : 0]                  iact_addr_SPad_data_r;
  reg [IACT_DATA_DATA-1 : 0]                  iact_data_SPad_data_r;
  reg [WGHT_ADDR_DATA-1 : 0]                  wght_addr_SPad_data_r;
  reg [WGHT_DATA_DATA-1 : 0]                  wght_data_SPad_data_r;
  reg [IACT_ADDR_ADDR_BITWIDTH-1:0]           iact_addr_SPad_addr;
  reg [IACT_DATA_ADDR_BITWIDTH-1:0]           iact_data_SPad_addr;
  wire [WGHT_ADDR_ADDR_BITWIDTH-1:0]           wght_addr_SPad_addr;
  wire [WGHT_DATA_ADDR_BITWIDTH-1:0]           wght_data_SPad_addr;
  wire [DATA_IACT_BITWIDTH-1:0]                iact_data_spad_pay;
  wire [DATA_IACT_IGNORE_ZEROS-1:0]            iact_data_spad_oh;
  reg [DATA_IACT_IGNORE_ZEROS-1:0]            iact_oh_delay_1;
  reg [DATA_IACT_IGNORE_ZEROS-1:0]            iact_oh_delay_2;
  wire [DATA_WGHT_BITWIDTH-1:0]                wght_data_spad_pay_1;
  wire [DATA_WGHT_IGNORE_ZEROS-1:0]            wght_data_spad_oh_1;
  wire [DATA_WGHT_BITWIDTH-1:0]                wght_data_spad_pay_2;
  wire [DATA_WGHT_IGNORE_ZEROS-1:0]            wght_data_spad_oh_2;
  reg                                         psum_data_SPad_en_a_r;
  reg                                         psum_data_SPad_en_b_r;
  reg                                         psum_data_SPad_en_a_w;
  reg                                         psum_data_SPad_en_b_w;
  reg                                         psum_select;
  reg                                         psum_enable;
  reg [TRANS_BITWIDTH_PSUM/2-1:0]             psum_data_1_delay;
  reg [TRANS_BITWIDTH_PSUM/2-1:0]             psum_data_2_delay;
  reg                                         mux_iact_ready;
  reg                                         adder_1_en;
  reg                                         adder_2_en;
  reg                                         fast_cycle;
  reg                                         next_iact;
  reg                                         next_iact2;
  reg                                         computing;
  reg [TRANS_BITWIDTH_IACT-1:0]               mux_iact_a_o_w;
  reg                                         mux_iact_b_o_w;
  wire                                        mux_iact_c_i_w;
  reg [WGHT_ADDR_DATA-1 : 0]                  wght_addr_current;
  reg [IACT_ADDR_DATA-1 : 0]                  iact_addr_current;
  reg [IACT_ADDR_DATA-1 : 0]                  iact_addr_count;
  reg [DATA_IACT_BITWIDTH-1 : 0]              iact_data_current_1;
  reg [DATA_IACT_BITWIDTH-1 : 0]              iact_data_current_2;
  reg [DATA_IACT_BITWIDTH-1 : 0]              iact_data_current_3;
  reg                                         iact_addr_SPad_en_r;
  reg                                         iact_data_SPad_en_r;
  reg                                         wght_addr_SPad_en_r;
  reg                                         wght_data_SPad_en_r;
  wire [PSUM_ADDR_BITWIDTH-1 : 0]              psum_spad_addr_a_r;
  wire [PSUM_ADDR_BITWIDTH-1 : 0]              psum_spad_addr_b_r;
  reg [PSUM_ADDR_BITWIDTH-1 : 0]              psum_spad_addr_a_mem;
  reg [PSUM_ADDR_BITWIDTH-1 : 0]              psum_spad_addr_b_mem;
  reg [PSUM_ADDR_BITWIDTH-1 : 0]              psum_spad_addr_a_delay;
  reg [PSUM_ADDR_BITWIDTH-1 : 0]              psum_spad_addr_b_delay;
  wire [DATA_WGHT_BITWIDTH-1 : 0]              mult_1_fac_1;
  wire [DATA_IACT_BITWIDTH-1 : 0]              mult_1_fac_2;
  wire [DATA_WGHT_BITWIDTH-1 : 0]              mult_2_fac_1;
  wire [DATA_IACT_BITWIDTH-1 : 0]              mult_2_fac_2;
  reg [PSUM_ADDR-1:0]                         used_psum_memory;
  reg                                         use_psum_1;
  reg                                         use_psum_2;
  wire [PSUM_DATA-1 : 0]                       psum_spad_data_a_o;
  wire [PSUM_DATA-1 : 0]                       psum_spad_data_b_o;
  wire [DATA_PSUM_BITWIDTH-1 : 0]              adder_1_summand_1;
  wire [DATA_PSUM_BITWIDTH-1 : 0]              adder_2_summand_1;
  wire [DATA_PSUM_BITWIDTH-1 : 0]              adder_1_summand_2;
  wire [DATA_PSUM_BITWIDTH-1 : 0]              adder_2_summand_2;
  wire [DATA_PSUM_BITWIDTH-1 : 0]              mult_1_o_w;
  wire [DATA_PSUM_BITWIDTH-1 : 0]              mult_2_o_w;
  wire [DATA_PSUM_BITWIDTH-1 : 0]              adder_1_o_w;
  wire [DATA_PSUM_BITWIDTH-1 : 0]              adder_2_o_w;
  wire [IACT_ADDR_ADDR_BITWIDTH-1 : 0]         first_spad_iact_addr_w;
  wire [IACT_ADDR_DATA-1 : 0]                  first_spad_iact_data_w;
  wire                                         first_spad_iact_en_w;
  wire                                         fifo_iact_addr_spad_empty_w;
  wire                                         fifo_iact_addr_spad_full_w;
  wire [IACT_DATA_ADDR_BITWIDTH-1 : 0]         second_spad_iact_addr_w;
  wire [IACT_DATA_DATA-1 : 0 ]                 second_spad_iact_data_w;
  wire                                         second_spad_iact_en_w;
  wire [WGHT_ADDR_ADDR_BITWIDTH-1 : 0]         first_spad_wght_addr_w;
  wire [WGHT_ADDR_DATA-1 : 0]                  first_spad_wght_data_w;
  wire                                         first_spad_wght_en_w;
  wire [WGHT_DATA_ADDR_BITWIDTH-1 : 0]         second_spad_wght_addr_w;
  wire [WGHT_DATA_DATA-1 : 0 ]                 second_spad_wght_data_w;
  wire                                         second_spad_wght_en_w;
  reg [DATA_PSUM_BITWIDTH-1 : 0]              psum_spad_data_a_i;
  reg [DATA_PSUM_BITWIDTH-1 : 0]              psum_spad_data_b_i;
  reg [PSUM_ADDR_BITWIDTH-1 : 0]              psum_spad_addr_a_w;
  reg [PSUM_ADDR_BITWIDTH-1 : 0]              psum_spad_addr_b_w;
  reg                                         reuse_psum_spad_a;
  reg                                         reuse_psum_spad_b;
  reg [DATA_PSUM_BITWIDTH-1 : 0]              reused_data_a;
  reg [DATA_PSUM_BITWIDTH-1 : 0]              reused_data_b;
  wire                                        reuse_adder_data_a2a;
  wire                                        reuse_adder_data_a2b;
  wire                                        reuse_adder_data_b2a;
  wire                                        reuse_adder_data_b2b;
  reg                                         wght_addr_use_vec;
  reg [WGHT_ADDR_ADDR_BITWIDTH-1:0]           wght_addr_vec;
  reg                                         wght_data_use_vec;
  reg [WGHT_DATA_ADDR_BITWIDTH-1:0]           wght_data_vec;
  reg [WGHT_DATA_ADDR_BITWIDTH-1:0]           wght_data_start;
  reg [WGHT_DATA_ADDR_BITWIDTH-1:0]           wght_data_end;
  reg [WGHT_DATA_ADDR_BITWIDTH-1:0]           wght_data_end_pre;
  reg [WGHT_DATA_ADDR_BITWIDTH-1:0]           wght_data_start_pre;
  reg                                         wght_start_set;
  reg                                         wght_end_set;
  reg                                         next_end_set;
  wire [3 : 0]                                 first_spad_words_iact;
  wire [4 : 0]                                 second_spad_words_iact;
  wire [4 : 0]                                 first_spad_words_wght;
  wire [6 : 0]                                 second_spad_words_wght;
  reg                                         values_valid;
  wire                                         psum_data_SPad_en_a_w_i;
  wire                                         psum_data_SPad_en_b_w_i;
  reg                                         data_mode_reg;
  reg  [$clog2(DATA_PSUM_BITWIDTH)-1: 0]      fraction_bit_reg;
  reg  [1 : 0]                                current_state_stream;

  //Necessary for creating VCD, if this is top
  `ifdef COCOTB_SIM
  initial begin
      if (CREATE_VCD == 1) begin
        $dumpfile ("sim_build/PE.vcd");
        $dumpvars (0, PE);
      end
    end
  `endif

  //State machine
  //Writing parameters to PEs
  enum bit [1:0]{
    FIRST_PARAMS  = 0,
    SECOND_PARAMS = 1,
    THIRD_PARAMS  = 2,
    FOURTH_PARAMS = 3
  } fsm_mode_stream;
  // State machine
  // IDLE: State for loading data into SPADs, wait for continue signal
  // LOADING_1-5: Preperation steps for computation
  // CALCULATING: Operative step for computation. Use of MAC-Operations
  // WAIT_TO_SEND_PSUM: Sets ready signal to 1, if input is 1. Waits for enable singal to proceed
  // SEND_PSUM: Continues outputstream of stored psum. Resumes to IDLE, when finished.
  enum bit [$clog2(16)-1:0]{
    IDLE               = 0,
    LOADING_1          = 1,
    LOADING_2          = 2,
    LOADING_3          = 3,
    LOADING_4          = 4,
    LOADING_5          = 5,
    CALCULATING        = 6,
    WAIT_TO_SEND_PSUM  = 7,
    SEND_PSUM          = 8
  } fsm_mode_computing;

  assign mux_iact_c_i_w = mux_iact_ready;
  assign {iact_data_spad_oh,iact_data_spad_pay} = iact_data_SPad_data_r;
  assign {wght_data_spad_oh_2,wght_data_spad_pay_2,wght_data_spad_oh_1,wght_data_spad_pay_1} = wght_data_SPad_data_r;
  assign psum_data_o = {adder_2_o_w,adder_1_o_w};
  assign wght_addr_SPad_addr = wght_addr_use_vec ? wght_addr_vec : iact_data_spad_oh;
  assign wght_data_SPad_addr = wght_data_use_vec ? wght_data_vec : wght_addr_SPad_data_r;
  assign mult_1_fac_1 = wght_data_spad_pay_1;
  assign mult_2_fac_1 = wght_data_spad_pay_2;
  assign mult_1_fac_2 = iact_data_current_3;
  assign mult_2_fac_2 = iact_data_current_3;
  assign reuse_adder_data_a2a = (psum_spad_addr_a_delay == psum_spad_addr_a_w) & (current_state_computing != SEND_PSUM);
  assign reuse_adder_data_a2b = (psum_spad_addr_b_delay == psum_spad_addr_a_w) & (current_state_computing != SEND_PSUM);
  assign reuse_adder_data_b2a = (psum_spad_addr_a_delay == psum_spad_addr_b_w) & (current_state_computing != SEND_PSUM);
  assign reuse_adder_data_b2b = (psum_spad_addr_b_delay == psum_spad_addr_b_w) & (current_state_computing != SEND_PSUM);
  assign adder_1_summand_1 =
                 !use_psum_1 ? 0 :
                (reuse_adder_data_a2a ? adder_1_o_w :
                (reuse_adder_data_b2a ? adder_2_o_w :
                (reuse_psum_spad_a ? reused_data_a :
                 psum_spad_data_a_o)));
  assign adder_2_summand_1 =
                 !use_psum_2 ? 0 :
                (reuse_adder_data_b2b ? adder_2_o_w :
                (reuse_adder_data_a2b ? adder_1_o_w :
                (reuse_psum_spad_b ? reused_data_b :
                 psum_spad_data_b_o)));
  assign psum_spad_data_a_i = adder_1_o_w;
  assign psum_spad_data_b_i = adder_2_o_w;
  assign psum_spad_addr_a_r = ((current_state_computing == WAIT_TO_SEND_PSUM) | (current_state_computing == SEND_PSUM)) ? 
                                psum_spad_addr_a_mem : 
                                wght_data_spad_oh_1 + psum_spad_addr_a_mem;
  assign psum_spad_addr_b_r = ((current_state_computing == WAIT_TO_SEND_PSUM) | (current_state_computing == SEND_PSUM)) ?
                                psum_spad_addr_b_mem :
                                wght_data_spad_oh_1 + wght_data_spad_oh_2 + psum_spad_addr_b_mem;
  assign psum_data_SPad_en_a_w_i = !psum_data_SPad_en_a_w ? 0 :
                                      !psum_data_SPad_en_a_r ? 1 :
                                       (psum_spad_addr_a_w == psum_spad_addr_a_r) ?  0 :
                                       (psum_spad_addr_a_w != psum_spad_addr_b_r) ?  1 : 0;
  assign psum_data_SPad_en_b_w_i = !psum_data_SPad_en_b_w ? 0 :
                                      !psum_data_SPad_en_b_r ? 1 :
                                       (psum_spad_addr_b_w == psum_spad_addr_b_r) ?  0 :
                                       (psum_spad_addr_b_w != psum_spad_addr_a_r) ?  1 : 0;
  assign psum_ready_o = psum_ready_i & psum_select;

  // FSM to control the operation of PE
  always @(posedge clk_i, negedge rst_ni) begin
    // Reset
    if (!rst_ni) begin
      data_mode_reg        <= 0;
      fraction_bit_reg     <= 0;
      current_state_stream <= 0;
    end else begin
      case (current_state_stream)
        FIRST_PARAMS : begin
          if (enable_stream_i) begin
            current_state_stream <= SECOND_PARAMS;
          end
        end
        SECOND_PARAMS : begin
          if (enable_stream_i) begin
            current_state_stream <= THIRD_PARAMS;
          end else begin
            current_state_stream <= FIRST_PARAMS;
          end
        end
        THIRD_PARAMS : begin
          if (enable_stream_i) begin
            current_state_stream <= FOURTH_PARAMS;
          end else begin
            current_state_stream <= FIRST_PARAMS;
          end
        end
        FOURTH_PARAMS : begin
          if (enable_stream_i) begin
            current_state_stream <= FIRST_PARAMS;
          end else begin
            current_state_stream <= FIRST_PARAMS;
          end
        end
        default : begin
        end
      endcase
    end
  end

  always @(posedge clk_i, negedge rst_ni) begin
    // Reset
    if (!rst_ni) begin
      current_state_computing   <= IDLE;
      iact_addr_SPad_addr       <= 0;
      iact_addr_SPad_en_r       <= 0;
      iact_addr_current         <= 0;
      iact_addr_count           <= 0;
      iact_data_SPad_addr       <= 0;
      iact_data_SPad_en_r       <= 0;
      wght_addr_SPad_en_r       <= 0;
      wght_addr_use_vec         <= 1;
      wght_data_use_vec         <= 1;
      wght_addr_vec             <= 0;
      wght_data_vec             <= 0;
      wght_data_start           <= 0;
      wght_start_set            <= 0;
      wght_data_end             <= 0;
      wght_data_end_pre         <= 0;
      wght_data_start_pre       <= 0;
      wght_end_set              <= 0;
      next_end_set              <= 0;
      wght_data_SPad_en_r       <= 0;
      psum_data_SPad_en_a_r     <= 0;
      psum_data_SPad_en_b_r     <= 0;
      psum_data_SPad_en_a_w     <= 0;
      psum_data_SPad_en_b_w     <= 0;
      iact_data_current_1       <= 0;
      iact_data_current_2       <= 0;
      iact_data_current_3       <= 0;
      iact_oh_delay_1           <= 0;
      iact_oh_delay_2           <= 0;
      computing                 <= 0;
      fast_cycle                <= 0;
      next_iact                 <= 0;
      next_iact2                <= 0;
      used_psum_memory          <= 0;
      use_psum_1                <= 0;
      use_psum_2                <= 0;
      adder_1_en                <= 0;
      adder_2_en                <= 0;
      mux_iact_ready            <= 1;
      wght_ready_o              <= 1;
      psum_select               <= 1;
      psum_data_1_delay         <= 0;
      psum_data_2_delay         <= 0;
      psum_enable               <= 0;
      psum_enable_o             <= 0;
      psum_spad_addr_a_mem      <= 0;
      psum_spad_addr_b_mem      <= 1;
      reuse_psum_spad_a         <= 0;
      reuse_psum_spad_b         <= 0;
      reused_data_a             <= 0;
      reused_data_b             <= 0;
      values_valid              <= 0;
      psum_spad_addr_a_delay    <= 0;
      psum_spad_addr_b_delay    <= 1;
      psum_spad_addr_a_w        <= 0;
      psum_spad_addr_b_w        <= 1;
    end else begin  
      {psum_data_2_delay,psum_data_1_delay} <= psum_data_i;
      if (psum_ready_i) begin
        psum_enable                         <= psum_enable_i;
      end else begin
        psum_enable                         <= 0;
      end
      psum_enable_o                         <= psum_enable;
      iact_oh_delay_1                       <= iact_data_spad_oh;
      iact_oh_delay_2                       <= iact_oh_delay_1;
      case (current_state_computing)
        IDLE : begin
          mux_iact_ready         <= 1;
          iact_addr_current      <= 0;
          iact_addr_count        <= 0;
          wght_ready_o           <= 1;
          wght_addr_vec          <= 0;
          wght_data_vec          <= 0;
          wght_data_start        <= 0;
          wght_start_set         <= 0;
          wght_data_end          <= 0;
          wght_data_end_pre      <= 0;
          wght_data_start_pre    <= 0;
          wght_end_set           <= 0;
          next_end_set           <= 0;
          //Read first values
          fast_cycle             <= 0;
          next_iact              <= 0;
          next_iact2             <= 0;
          iact_addr_SPad_addr    <= 0;
          iact_addr_SPad_en_r    <= 0;

          iact_data_SPad_addr    <= 0;
          iact_data_SPad_en_r    <= 0;
          iact_oh_delay_1        <= 0;
          iact_oh_delay_2        <= 0;
          wght_addr_SPad_en_r    <= 0;

          wght_addr_use_vec      <= 1;
          wght_data_use_vec      <= 1;
          wght_data_SPad_en_r    <= 0;

          psum_data_SPad_en_a_r  <= computing;
          psum_data_SPad_en_b_r  <= computing;
          psum_data_SPad_en_a_w  <= 0;
          psum_data_SPad_en_b_w  <= 0;

          computing              <= 0;
          values_valid           <= 0;

          psum_spad_addr_a_delay <= 0;
          psum_spad_addr_b_delay <= 1;
          psum_spad_addr_a_w     <= 0;
          psum_spad_addr_b_w     <= 1;
          psum_spad_addr_a_mem   <= 0;
          psum_spad_addr_b_mem   <= 1;
          adder_1_en             <= 0;
          adder_2_en             <= 0;
          psum_select            <= 0;
          reuse_psum_spad_a      <= 0;
          reuse_psum_spad_b      <= 0;
          reused_data_a          <= 0;
          reused_data_b          <= 0;
          use_psum_1             <= 0;
          use_psum_2             <= 0;
          used_psum_memory       <= 0;
          if (psum_enable_i) begin
            current_state_computing    <= SEND_PSUM;
            adder_1_en       <= 1;
            adder_2_en       <= 1;
            psum_select      <= 1;
            use_psum_1       <= 0;
            use_psum_2       <= 0;
            used_psum_memory <= 0;
          end
          if (compute_i) begin 
            //Start off
            current_state_computing         <= LOADING_1;
            mux_iact_ready        <= 0;
            wght_ready_o          <= 0;
            
            iact_addr_SPad_addr   <= 0;
            iact_addr_SPad_en_r   <= 1;

            iact_data_SPad_addr   <= 0;
            iact_data_SPad_en_r   <= 1;
            psum_data_SPad_en_a_r <= 0;
            psum_data_SPad_en_b_r <= 0;
            wght_addr_use_vec     <= 0;
            wght_data_use_vec     <= 0;
            use_psum_1            <= 0;
            use_psum_2            <= 0;
            used_psum_memory      <= 0;
          end
        end

        LOADING_1 : begin
          //Get first WGHT Addr Address
          current_state_computing       <= LOADING_2;
          iact_data_SPad_addr <= iact_data_SPad_addr + 1;
          wght_addr_SPad_en_r <= 1;
          iact_addr_SPad_addr <= iact_addr_SPad_addr + 1;
          psum_select         <= 0;
        end

        LOADING_2 : begin
          //Get first WGHT Data Address
          current_state_computing       <= LOADING_3;
          if (iact_addr_SPad_data_r == 0) begin
            if (iact_addr_SPad_addr == 4) begin
              current_state_computing         <= WAIT_TO_SEND_PSUM;
              computing             <= 0;
              psum_data_SPad_en_a_r <= 0;
              psum_data_SPad_en_b_r <= 0;
              psum_data_SPad_en_a_w <= 1;
              psum_data_SPad_en_b_w <= 1;
              values_valid          <= 0;
            end else begin
              current_state_computing       <= LOADING_1;
              iact_addr_SPad_addr <= iact_addr_SPad_addr + 1;
              iact_addr_SPad_en_r <= 1;
            end
          end else begin
            wght_data_SPad_en_r <= 1;
            wght_addr_vec       <= iact_data_spad_oh + 1;
            wght_addr_use_vec   <= 1;
            iact_data_current_1 <= iact_data_spad_pay;
            iact_data_SPad_addr <= iact_data_SPad_addr + 1;
            iact_addr_current   <= iact_addr_SPad_data_r;
            iact_addr_SPad_en_r <= 0;
            iact_addr_SPad_addr <= iact_addr_SPad_addr - 1;
          end
        end

        LOADING_3 : begin
          current_state_computing       <= LOADING_4;
          iact_data_SPad_addr <= iact_data_SPad_addr + 1;
          wght_addr_use_vec   <= 1;
          if (wght_addr_vec == iact_data_spad_oh) begin
            wght_addr_vec <= wght_addr_vec + 1;
          end else begin
            wght_addr_vec <= iact_data_spad_oh;
          end
          iact_data_current_1 <= iact_data_spad_pay;
          iact_data_current_2 <= iact_data_current_1;
          wght_data_start     <= wght_addr_SPad_data_r;
          iact_addr_SPad_en_r <= 0;
        end

        LOADING_4 : begin
          current_state_computing       <= LOADING_5;
          wght_data_end       <= wght_addr_SPad_data_r;
          wght_addr_use_vec   <= 1;
          iact_data_current_1 <= iact_data_spad_pay;
          iact_data_current_2 <= iact_data_current_1;
          iact_data_current_3 <= iact_data_current_2;
          wght_data_use_vec   <= 1;
          wght_data_vec       <= wght_data_start;
          if (iact_oh_delay_1 == iact_oh_delay_2 + 1) begin
            wght_data_end       <= wght_addr_SPad_data_r;
            wght_data_start_pre <= wght_addr_SPad_data_r;
            wght_start_set      <= 1;
          end else begin
            wght_addr_vec <= wght_addr_vec + 1;
          end
          iact_addr_SPad_en_r <= 0;
          if (iact_addr_current == 1) begin
            iact_addr_SPad_addr <= iact_addr_SPad_addr + 1;
          end
        end

        LOADING_5 : begin
          current_state_computing  <= CALCULATING;
          wght_start_set <= 1;
          computing      <= 1;
          if (wght_data_end > wght_data_start) begin
            values_valid       <= 1;
          end
          wght_data_vec      <= wght_data_vec + 1;
          if (wght_start_set) begin
            wght_data_end_pre <= wght_addr_SPad_data_r;
            wght_end_set      <= 1;
            if (iact_oh_delay_1 >= iact_oh_delay_2 + 1) begin
              wght_addr_vec     <= iact_oh_delay_1;
            end
          end else begin
            wght_data_start_pre<= wght_addr_SPad_data_r;
            fast_cycle         <= 1;
            wght_addr_vec     <= iact_oh_delay_1;
          end
          iact_addr_count     <= 1;
          iact_addr_SPad_en_r <= 0;
          if ((iact_addr_SPad_data_r == 0) & (iact_addr_current == 0)) begin
            iact_addr_SPad_en_r <= 1;
          end
        end

        CALCULATING : begin
          //Defaulting Values
          iact_addr_SPad_en_r   <= 0;
          iact_data_SPad_en_r   <= !mux_iact_ready;
          wght_addr_SPad_en_r   <= 1;
          wght_data_SPad_en_r   <= 1;
          psum_data_SPad_en_a_r <= computing;
          psum_data_SPad_en_b_r <= computing;
          psum_data_SPad_en_a_w <= 0;
          psum_data_SPad_en_b_w <= 0;
          reuse_psum_spad_a     <= 0;
          reuse_psum_spad_b     <= 0;
          reused_data_a         <= 0;
          reused_data_b         <= 0;
          computing             <= 1;
          values_valid          <= 1;
          wght_addr_use_vec     <= 1;
          wght_data_use_vec     <= 1;
          fast_cycle            <= 0;
          next_iact             <= 0;
          next_iact2            <= 0;
          psum_spad_addr_a_mem  <= psum_spad_addr_b_r + 1;
          psum_spad_addr_b_mem  <= psum_spad_addr_b_r + 2;
          if (wght_data_vec < (second_spad_words_wght - 1)) begin
            wght_data_vec <= wght_data_vec + 1;
          end else begin
            mux_iact_ready <= 1;
          end

          if (!next_iact || fast_cycle) begin
            if (wght_start_set) begin
              if (!wght_end_set) begin
                wght_data_end_pre <= wght_addr_SPad_data_r;
                wght_end_set      <= 1;
              end
            end else begin
              wght_data_start_pre <= wght_addr_SPad_data_r;
              wght_start_set <= 1;
              if ((first_spad_words_wght - 1) > wght_addr_vec) begin
                wght_addr_vec <= iact_oh_delay_1;
              end
            end
          end

          if ((wght_data_end <= wght_data_SPad_addr + 1) && !next_iact) begin
            wght_data_start <= wght_data_start_pre;
            wght_end_set    <= 0;
            wght_start_set  <= 0;
            if (wght_start_set) begin
              wght_data_start <= wght_data_start_pre;
              if (wght_data_vec < (second_spad_words_wght - 1)) begin
                wght_data_vec   <= wght_data_start_pre;
              end
            end
            if (wght_end_set) begin
              wght_data_end       <= wght_data_end_pre;
              wght_data_start_pre <= wght_addr_SPad_data_r;
            end else begin
              wght_data_end       <= wght_addr_SPad_data_r;
            end
            if (iact_oh_delay_1  <= iact_oh_delay_2 + 1) begin
              if ((first_spad_words_wght - 1) > wght_addr_vec) begin
                wght_addr_vec <= wght_addr_vec + 1;
              end
              if (wght_end_set) begin
                wght_data_start_pre <= wght_data_end_pre;
              end else begin
                wght_data_start_pre <= wght_addr_SPad_data_r;
              end
            end else begin

              if ((first_spad_words_wght - 1) > wght_addr_vec) begin
                wght_addr_vec <= iact_oh_delay_1;
              end
              wght_start_set <= 0;
            end
            if ((first_spad_words_wght - 1) > wght_addr_vec) begin
              wght_addr_vec <= wght_addr_vec + 1;
            end
            fast_cycle          <= 1;
            iact_data_SPad_addr <= iact_data_SPad_addr + 1;
            next_iact           <= 1;
            iact_addr_count     <= iact_addr_count + 1;
            if (((32'(iact_addr_count) + 1) >= 32'(iact_addr_current)) & ((32'(iact_addr_SPad_addr)+1) < first_spad_words_iact)) begin
              iact_addr_SPad_addr <= iact_addr_SPad_addr + 1;
              iact_addr_SPad_en_r <= 1;
            end
          end

          if (next_iact) begin
            next_iact2           <= iact_addr_SPad_en_r;
            iact_data_current_1  <= iact_data_spad_pay;
            iact_data_current_2  <= iact_data_current_1;
            iact_data_current_3  <= iact_data_current_2;
            psum_spad_addr_a_mem <= 0;
            psum_spad_addr_b_mem <= 1;
          end
          // Check valid values
          if (next_iact2) begin
            iact_addr_current <= iact_addr_SPad_data_r;
          end
          if (wght_data_end <= wght_data_vec) begin
            values_valid <= 0;
          end
          //Reuse Values of PSUM SPad
          if (((32'(32'(iact_addr_count)) == 32'(32'(iact_addr_current)+1)) | (iact_addr_count == 0)) & (next_iact)) begin
            current_state_computing          <= WAIT_TO_SEND_PSUM;
            mux_iact_ready         <= 1;
            iact_data_current_3    <= 0;
            computing              <= 0;
            psum_data_SPad_en_a_r  <= 0;
            psum_data_SPad_en_b_r  <= 0;
            psum_data_SPad_en_a_w  <= 1;
            psum_data_SPad_en_b_w  <= 1;
            values_valid           <= 0;
          end else begin
            psum_data_SPad_en_a_w <= 1;
            psum_data_SPad_en_b_w <= 1;
          end

          //Duplicated Data in adders
          if (psum_spad_addr_a_r == psum_spad_addr_a_w) begin
            reuse_psum_spad_a <= 1;
            reused_data_a     <= adder_1_o_w;
          end
          if (psum_spad_addr_b_r == psum_spad_addr_b_w) begin
            reuse_psum_spad_b <= 1;
            reused_data_b     <= adder_2_o_w;
          end
          if (psum_spad_addr_a_r == psum_spad_addr_b_w) begin
            reuse_psum_spad_a <= 1;
            reused_data_a     <= adder_2_o_w;
          end
          if (psum_spad_addr_b_r == psum_spad_addr_a_w) begin
            reuse_psum_spad_b <= 1;
            reused_data_b     <= adder_1_o_w;
          end

          adder_1_en <= 1;
          adder_2_en <= 1;
          if (used_psum_memory[(psum_spad_addr_a_r)]  == 1) begin 
            use_psum_1 <= 1;
          end else begin
            use_psum_1 <= 0;
            used_psum_memory[(psum_spad_addr_a_r)] <= 1;
          end
          if (used_psum_memory[(psum_spad_addr_b_r)]  == 1) begin 
            use_psum_2 <= 1;
          end else begin
            use_psum_2 <= 0;
            used_psum_memory[(psum_spad_addr_b_r)] <= 1;
          end
          psum_spad_addr_a_delay <= psum_spad_addr_a_r;
          psum_spad_addr_b_delay <= psum_spad_addr_b_r;
          psum_spad_addr_a_w     <= psum_spad_addr_a_delay;
          psum_spad_addr_b_w     <= psum_spad_addr_b_delay;
        end

        WAIT_TO_SEND_PSUM : begin
          //Read first values
          iact_addr_SPad_addr    <= 0;
          iact_addr_SPad_en_r    <= 0;

          iact_data_SPad_addr    <= 0;
          iact_data_SPad_en_r    <= 0;

          wght_addr_SPad_en_r    <= 0;
          wght_data_SPad_en_r    <= 0;

          psum_data_SPad_en_a_r  <= 0;
          psum_data_SPad_en_b_r  <= 0;
          psum_data_SPad_en_a_w  <= 1;
          psum_data_SPad_en_b_w  <= 1;
          psum_spad_addr_a_mem   <= 0;
          psum_spad_addr_b_mem   <= 1;
          psum_spad_addr_a_delay <= psum_spad_addr_a_r;
          psum_spad_addr_b_delay <= psum_spad_addr_b_r;
          psum_spad_addr_a_w     <= psum_spad_addr_a_delay;
          psum_spad_addr_b_w     <= psum_spad_addr_b_delay;
          adder_1_en             <= computing;
          adder_2_en             <= computing;

          reuse_psum_spad_a      <= 0;
          reuse_psum_spad_b      <= 0;
          reused_data_a          <= 0;
          reused_data_b          <= 0;

          //Reuse Values of PSUM SPad
          if (psum_spad_addr_a_r == psum_spad_addr_a_w) begin
            reuse_psum_spad_a <= 1;
            reused_data_a     <= adder_1_o_w;
          end
          if (psum_spad_addr_b_r == psum_spad_addr_b_w) begin
            reuse_psum_spad_b <= 1;
            reused_data_b     <= adder_2_o_w;
          end
          if (psum_spad_addr_a_r == psum_spad_addr_b_w) begin
            reuse_psum_spad_a <= 1;
            reused_data_a     <= adder_2_o_w;
          end
          if (psum_spad_addr_b_r == psum_spad_addr_a_w) begin
            reuse_psum_spad_b <= 1;
            reused_data_b     <= adder_1_o_w;
          end

          if (psum_select) begin
            psum_data_SPad_en_a_w <= 0;
            psum_data_SPad_en_b_w <= 0;
            psum_spad_addr_a_mem <= 0;
            psum_spad_addr_b_mem <= 1;
            psum_spad_addr_a_w   <= 2;
            psum_spad_addr_b_w   <= 3;
          end
          if (psum_enable_i) begin
            adder_1_en            <= 1;
            adder_2_en            <= 1;
            current_state_computing         <= SEND_PSUM;
            psum_data_SPad_en_a_w <= 1;
            psum_data_SPad_en_b_w <= 1;
            psum_spad_addr_a_mem  <= psum_spad_addr_a_r + 2;
            psum_spad_addr_b_mem  <= psum_spad_addr_b_r + 2;
            psum_spad_addr_a_w    <= psum_spad_addr_a_r;
            psum_spad_addr_b_w    <= psum_spad_addr_b_r;
            if (used_psum_memory[(psum_spad_addr_a_r)]  == 1) begin 
              use_psum_1    <= 1;
              used_psum_memory[(psum_spad_addr_a_r)]  <= 0;
            end else begin
              use_psum_1    <= 0;
            end
            if (used_psum_memory[(psum_spad_addr_b_r)]  == 1) begin 
              use_psum_2    <= 1;
              used_psum_memory[(psum_spad_addr_b_r)]  <= 0;
            end else begin
              use_psum_2    <= 0;
            end
          end else begin
            if (used_psum_memory[(psum_spad_addr_a_r)]  == 1) begin 
              use_psum_1 <= 1;
            end else begin
              use_psum_1 <= 0;
              used_psum_memory[(psum_spad_addr_a_r)] <= 1;
            end
            if (used_psum_memory[(psum_spad_addr_b_r)]  == 1) begin 
              use_psum_2 <= 1;
            end else begin
              use_psum_2 <= 0;
              used_psum_memory[(psum_spad_addr_b_r)] <= 1;
            end
          end
          computing   <= 0;
          psum_select <= !computing;
        end

        SEND_PSUM : begin
          psum_spad_addr_a_mem <= psum_spad_addr_a_r + 2;
          psum_spad_addr_b_mem <= psum_spad_addr_b_r + 2;
          psum_spad_addr_a_w   <= psum_spad_addr_a_r;
          psum_spad_addr_b_w   <= psum_spad_addr_b_r;
          adder_1_en           <= 1;
          adder_2_en           <= 1;
          psum_select          <= !computing;
          if (!psum_enable_i) begin
            current_state_computing <= IDLE;
            adder_1_en    <= 0;
            adder_2_en    <= 0;
          end
          if (used_psum_memory[(psum_spad_addr_a_r)]  == 1) begin 
            use_psum_1                              <= 1;
            used_psum_memory[(psum_spad_addr_a_r)]  <= 0;
          end else begin
            use_psum_1 <= 0;
          end
          if (used_psum_memory[(psum_spad_addr_b_r)]  == 1) begin 
            use_psum_2                              <= 1;
            used_psum_memory[(psum_spad_addr_b_r)]  <= 0;
          end else begin
            use_psum_2 <= 0;
          end
        end
        default : begin
        end
      endcase
    end
  end

  // SPad for IACT_ADDR
  SPad_SP #(
    .DATA_WIDTH(IACT_ADDR_DATA),
    .ADDR_WIDTH(IACT_ADDR_ADDR_BITWIDTH)

    ,.Implementation("pe_iact_addr")
  ) iact_addr_SPad ( 
    .clk_i (clk_i), 
    .re_i  (iact_addr_SPad_en_r & !first_spad_iact_en_w),
    .we_i  (first_spad_iact_en_w), 
    .addr_i(iact_addr_SPad_addr | first_spad_iact_addr_w),
    .data_i(first_spad_iact_data_w),
    .data_o(iact_addr_SPad_data_r)
  );

  // SPad for IACT_DATA
  SPad_SP #(
    .DATA_WIDTH(IACT_DATA_DATA),
    .ADDR_WIDTH(IACT_DATA_ADDR_BITWIDTH)

    ,.Implementation("pe_iact_data")
  ) iact_data_SPad ( 
    .clk_i (clk_i), 
    .re_i  (iact_data_SPad_en_r & !second_spad_iact_en_w),
    .we_i  (second_spad_iact_en_w), 
    .addr_i(iact_data_SPad_addr | second_spad_iact_addr_w),
    .data_i(second_spad_iact_data_w),
    .data_o(iact_data_SPad_data_r)
  );

  // SPad for WGHT_ADDR
  SPad_SP #(
    .DATA_WIDTH(WGHT_ADDR_DATA),
    .ADDR_WIDTH(WGHT_ADDR_ADDR_BITWIDTH)

    ,.Implementation("pe_weight_addr")
  ) weight_addr_SPad ( 
    .clk_i (clk_i), 
    .re_i  (wght_addr_SPad_en_r & !first_spad_wght_en_w),
    .we_i  (first_spad_wght_en_w), 
    .addr_i(wght_addr_SPad_addr | first_spad_wght_addr_w),
    .data_i(first_spad_wght_data_w),
    .data_o(wght_addr_SPad_data_r)
  );

  // SPad for WGHT_DATA
  SPad_SP #(
    .DATA_WIDTH(WGHT_DATA_DATA),
    .ADDR_WIDTH(WGHT_DATA_ADDR_BITWIDTH)

    ,.Implementation("pe_weight_data")
  ) weight_data_SPad ( 
    .clk_i (clk_i), 
    .re_i  (wght_data_SPad_en_r & !second_spad_wght_en_w),
    .we_i  (second_spad_wght_en_w), 
    .addr_i(wght_data_SPad_addr | second_spad_wght_addr_w),
    .data_i(second_spad_wght_data_w),
    .data_o(wght_data_SPad_data_r)
  );

  // SPad for PSUM
  SPad_DP_RW #(
    .DATA_WIDTH(PSUM_DATA),
    .ADDR_WIDTH(PSUM_ADDR_BITWIDTH)

    ,.Implementation("pe_psum")
  ) psum_SPad (
    .clk_i     (clk_i),
    .re_a_i    (psum_data_SPad_en_a_r || psum_enable_i),
    .re_b_i    (psum_data_SPad_en_b_r || psum_enable_i),
    .we_a_i    (psum_data_SPad_en_a_w_i),
    .we_b_i    (psum_data_SPad_en_b_w_i),
    .addr_r_a_i(psum_spad_addr_a_r),
    .addr_r_b_i(psum_spad_addr_b_r),
    .addr_w_a_i(psum_spad_addr_a_w),
    .addr_w_b_i(psum_spad_addr_b_w),
    .data_a_i  (psum_spad_data_a_i),
    .data_b_i  (psum_spad_data_b_i),
    .data_a_o  (psum_spad_data_a_o),
    .data_b_o  (psum_spad_data_b_o)
  );
  
  // Mux to select the correct IACT data from GLB
  mux_iact #(
    .WIDTH  (TRANS_BITWIDTH_IACT),
    .I_COUNT(NUM_GLB_IACT)
  ) mux_iact ( 
    .a_i  (iact_data_i),
    .b_i  (iact_enable_i),
    .c_i  (mux_iact_c_i_w),
    .sel_i(iact_select_i),
    .a_o  (mux_iact_a_o_w),
    .b_o  (mux_iact_b_o_w),
    .c_o  (iact_ready_o)
  );

  // Parallel to serial converter for weight data and address
  // e.g., converts a 24-bit parallel weight word to multiple 8-bit/4-bit pairs of data and address
  data_pipeline #(
    .DATA_WIDTH      (TRANS_BITWIDTH_WGHT),
    .FIRST_SPAD_ADDR (WGHT_ADDR_ADDR),
    .FIRST_SPAD_DATA (WGHT_ADDR_DATA),
    .SECOND_SPAD_ADDR(WGHT_DATA_ADDR),
    .SECOND_SPAD_DATA(WGHT_DATA_DATA)
  ) wght_data_handler ( 
    .clk_i             (clk_i),
    .rst_ni            (rst_ni),
    .compute_i         (compute_i),

    .data_i            (wght_data_i),
    .enable_i          (wght_enable_i),

    .first_spad_words_o (first_spad_words_wght),
    .second_spad_words_o (second_spad_words_wght),

    .first_spad_addr_o (first_spad_wght_addr_w),
    .first_spad_data_o (first_spad_wght_data_w),
    .first_spad_en_o   (first_spad_wght_en_w),

    .second_spad_addr_o(second_spad_wght_addr_w),
    .second_spad_data_o(second_spad_wght_data_w),
    .second_spad_en_o  (second_spad_wght_en_w)
  );

  // Parallel to serial converter for iact data and address
  // e.g., converts a 24-bit parallel iact word to multiple 8-bit/4-bit pairs of data and address
  data_pipeline #(
    .DATA_WIDTH      (TRANS_BITWIDTH_IACT),
    .FIRST_SPAD_ADDR (IACT_ADDR_ADDR),
    .FIRST_SPAD_DATA (IACT_ADDR_DATA),
    .SECOND_SPAD_ADDR(IACT_DATA_ADDR),
    .SECOND_SPAD_DATA(IACT_DATA_DATA)
  ) iact_data_handler ( 
    .clk_i             (clk_i),
    .rst_ni            (rst_ni),
    .compute_i         (compute_i),

    .data_i            (mux_iact_a_o_w),
    .enable_i          (mux_iact_b_o_w),

    .first_spad_words_o (first_spad_words_iact),
    .second_spad_words_o (second_spad_words_iact),

    .first_spad_addr_o (first_spad_iact_addr_w),
    .first_spad_data_o (first_spad_iact_data_w),
    .first_spad_en_o   (first_spad_iact_en_w),

    .second_spad_addr_o(second_spad_iact_addr_w),
    .second_spad_data_o(second_spad_iact_data_w),
    .second_spad_en_o  (second_spad_iact_en_w)
  );

  // do the first parallel multiplication of weight and iact
  multiplier #(
    .DATA_WIDTH_FAC1(DATA_WGHT_BITWIDTH),
    .DATA_WIDTH_FAC2(DATA_IACT_BITWIDTH),
    .DATA_WIDTH_PROD(DATA_PSUM_BITWIDTH)
  ) multiplier_1 (
    .clk_i          (clk_i),
    .rst_ni         (rst_ni),
    .multiplier_en_i(values_valid),
    .factor_1       (mult_1_fac_1), 
    .factor_2       (mult_1_fac_2),
    .product        (mult_1_o_w),
    .fraction_bit_i (fraction_bit_reg)
  );

  // do the second parallel multiplication of weight and iact
  multiplier #(
    .DATA_WIDTH_FAC1(DATA_WGHT_BITWIDTH),
    .DATA_WIDTH_FAC2(DATA_IACT_BITWIDTH),
    .DATA_WIDTH_PROD(DATA_PSUM_BITWIDTH)
  ) multiplier_2 (
    .clk_i          (clk_i),
    .rst_ni         (rst_ni),
    .multiplier_en_i(values_valid),
    .factor_1       (mult_2_fac_1), 
    .factor_2       (mult_2_fac_2),
    .product        (mult_2_o_w),
    .fraction_bit_i (fraction_bit_reg)
  );

  // do the first parallel addition of product and psum 
  adder #(
    .DATA_WIDTH_SUM(DATA_PSUM_BITWIDTH)
  ) adder_1 (
    .clk_i      (clk_i),
    .rst_ni     (rst_ni),
    .summand_1_i(adder_1_summand_1), 
    .summand_2_i(adder_1_summand_2),
    .sum_o      (adder_1_o_w),
    .adder_en_i (adder_1_en)
  );

  // do the second parallel addition of product and psum
  adder #(
    .DATA_WIDTH_SUM(DATA_PSUM_BITWIDTH)
  ) adder_2 (
    .clk_i      (clk_i),
    .rst_ni     (rst_ni),
    .summand_1_i(adder_2_summand_1), 
    .summand_2_i(adder_2_summand_2),
    .sum_o      (adder_2_o_w),
    .adder_en_i (adder_2_en)
  );

  // mux to select input to adder
  // (this can be either the data of psum SPAD or psum from another PE/router)
  mux2 #(
    .DATA_WIDTH(TRANS_BITWIDTH_PSUM)
  ) mux_psum (
    .a_in ({psum_data_2_delay,psum_data_1_delay}),
    .b_in ({mult_2_o_w,mult_1_o_w}),
    .sel_i(psum_select), 
    .y_o  ({adder_2_summand_2,adder_1_summand_2})
  );

endmodule
