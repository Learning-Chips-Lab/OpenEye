// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

// Simplified PE with the same interface as PE.v.
// Sparsity is removed; pipeline is simplified to IDLE -> LOADING_1 -> CALCULATING ->
// WAIT_TO_SEND_PSUM -> SEND_PSUM.  One MAC per cycle, single-ported psum SPad.
// SYSTOLIC_GEMM_EN overrides iact source from pass-through when iact_pass_enable_i is high.

module PE_simple #(

    parameter IS_TOPLEVEL = 1,
    parameter SERIAL      = 1,
    parameter CREATE_VCD  = 0,

    parameter PE_X = 0,
    parameter PE_Y = 0,

    parameter integer PARALLEL_MACS = 2,

    parameter integer SPARSITY_EN      = 1,
    parameter integer USE_DSP          = 0,
    parameter integer SYSTOLIC_GEMM_EN = 0,

    parameter integer DATA_IACT_BITWIDTH     = 8,
    parameter integer DATA_WGHT_BITWIDTH     = 8,
    parameter integer DATA_PSUM_BITWIDTH     = 20,
    parameter integer DATA_IACT_OVERHEAD     = 4,
    parameter integer DATA_WGHT_IGNORE_ZEROS = 4,

    parameter integer IACT_DATA_ADDR = 16,
    parameter integer IACT_ADDR_ADDR = 9,

    parameter integer WGHT_DATA_ADDR = 96,
    parameter integer WGHT_ADDR_ADDR = 16,

    parameter integer PSUM_ADDR = 32,

    parameter integer TRANS_BITWIDTH_IACT = 24,
    parameter integer TRANS_BITWIDTH_WGHT = 24,

    parameter integer NUM_GLB_IACT = 3,

    localparam integer IACT_ADDR_DATA          = $clog2(IACT_DATA_ADDR),
    localparam integer WGHT_ADDR_DATA          = $clog2(WGHT_DATA_ADDR),
    localparam integer IACT_ADDR_ADDR_BITWIDTH = $clog2(IACT_ADDR_ADDR),
    localparam integer IACT_DATA_ADDR_BITWIDTH = $clog2(IACT_DATA_ADDR),

    // Dense only: no overhead bits
    localparam integer IACT_DATA_DATA          = DATA_IACT_BITWIDTH,
    localparam integer IACT_DATA_DATA_BITWIDTH = $clog2(IACT_DATA_DATA),

    // Dense only: PARALLEL_MACS weights per word (no ignore_zeros)
    localparam integer WGHT_DATA_DATA          = DATA_WGHT_BITWIDTH * PARALLEL_MACS,
    localparam integer WGHT_ADDR_ADDR_BITWIDTH = $clog2(WGHT_ADDR_ADDR),
    localparam integer WGHT_ADDR_DATA_BITWIDTH = $clog2(WGHT_ADDR_DATA),
    localparam integer WGHT_DATA_ADDR_BITWIDTH = $clog2(WGHT_DATA_ADDR),
    localparam integer WGHT_DATA_DATA_BITWIDTH = $clog2(WGHT_DATA_DATA),

    localparam integer PSUM_DATA               = DATA_PSUM_BITWIDTH,
    localparam integer PSUM_ADDR_BITWIDTH      = $clog2(PSUM_ADDR),
    localparam integer PSUM_DATA_BITWIDTH      = $clog2(PSUM_DATA),
    localparam integer PSUM_WORDS_PER_TRANSFER = (SERIAL ? 1 : PARALLEL_MACS),
    localparam integer TRANS_BITWIDTH_PSUM     = DATA_PSUM_BITWIDTH * PSUM_WORDS_PER_TRANSFER,
    localparam integer VALUES_OF_IACTS         = $rtoi($ceil(TRANS_BITWIDTH_IACT / DATA_IACT_BITWIDTH))

) (
    input                                              clk_i,
    input                                              rst_ni,
    input      [           $clog2(NUM_GLB_IACT+1)-1:0] iact_select_i,
    input      [ TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] iact_data_i,
    input      [                     NUM_GLB_IACT-1:0] iact_enable_i,
    output     [                     NUM_GLB_IACT-1:0] iact_ready_o,
    input      [              TRANS_BITWIDTH_WGHT-1:0] wght_data_i,
    input                                              wght_enable_i,
    output reg                                         wght_ready_o,
    input      [              TRANS_BITWIDTH_PSUM-1:0] psum_data_i,
    input                                              psum_enable_i,
    output                                             psum_ready_o,
    output     [              TRANS_BITWIDTH_PSUM-1:0] psum_data_o,
    output reg                                         psum_enable_o,
    input                                              psum_ready_i,
    input                                              compute_i,
    input                                              enable_stream_i,
    input      [                                 11:0] data_stream_i,
    input      [          DATA_IACT_BITWIDTH-1:0]      iact_pass_data_i,
    input                                              iact_pass_enable_i,
    output                                             iact_pass_ready_o,
    output     [          DATA_IACT_BITWIDTH-1:0]      iact_pass_data_o,
    output                                             iact_pass_enable_o,
    input                                              iact_pass_ready_i
);

  // =========================================================================
  // FSM state encoding
  // =========================================================================
  localparam [3:0]
    IDLE             = 4'd0,
    LOADING_1        = 4'd1,
    CALCULATING      = 4'd2,
    WAIT_TO_SEND_PSUM = 4'd3,
    SEND_PSUM        = 4'd4;

  localparam [1:0]
    FIRST_PARAMS  = 2'd0,
    SECOND_PARAMS = 2'd1,
    THIRD_PARAMS  = 2'd2,
    FOURTH_PARAMS = 2'd3;

  // =========================================================================
  // Scratch-pad memories (simple Verilog arrays)
  // =========================================================================
  reg [DATA_IACT_BITWIDTH-1:0]  iact_data_spad [0:IACT_DATA_ADDR-1];
  reg [WGHT_DATA_DATA-1:0]      wght_data_spad [0:WGHT_DATA_ADDR-1];
  reg [DATA_PSUM_BITWIDTH-1:0]  psum_spad      [0:PSUM_ADDR-1];

  // =========================================================================
  // Configuration registers
  // =========================================================================
  reg [1:0]                          current_state_stream;
  reg [IACT_DATA_ADDR_BITWIDTH-1:0]  iact_addr_max_reg;
  reg [WGHT_DATA_ADDR_BITWIDTH-1:0]  wght_addr_max_reg;
  reg [4:0]                          filters_reg_M0;
  reg [3:0]                          channel_reg_C0;
  reg [2:0]                          stride_reg;
  reg                                data_mode_reg;
  reg [$clog2(DATA_PSUM_BITWIDTH)-1:0] fraction_bit_reg;
  reg [3:0]                          iact_x_line_repetitions;

  // =========================================================================
  // Compute FSM state
  // =========================================================================
  reg [3:0] current_state_computing;

  // =========================================================================
  // Data-loaded flags
  // =========================================================================
  reg iact_set;
  reg wght_set;

  // =========================================================================
  // Iact SPad loading
  // =========================================================================
  // 24-bit bus carries up to VALUES_OF_IACTS bytes; unpack sequentially.
  reg [IACT_DATA_ADDR_BITWIDTH-1:0] iact_spad_waddr;
  reg [7:0]                          iact_bus_pos;   // byte position within current bus word
  reg                                iact_ready_r;
  reg [IACT_DATA_ADDR_BITWIDTH-1:0]  iact_load_cnt;

  wire [TRANS_BITWIDTH_IACT-1:0] sel_iact_data;
  wire                            sel_iact_enable;

  assign sel_iact_data   = (iact_select_i == 0) ? {TRANS_BITWIDTH_IACT{1'b0}}
                         : iact_data_i[(iact_select_i-1)*TRANS_BITWIDTH_IACT +: TRANS_BITWIDTH_IACT];
  assign sel_iact_enable = (iact_select_i == 0) ? 1'b0
                         : iact_enable_i[iact_select_i-1];

  genvar gi;
  generate
    for (gi = 0; gi < NUM_GLB_IACT; gi = gi + 1) begin : gen_iact_ready
      assign iact_ready_o[gi] = (iact_select_i == (gi + 1)) ? iact_ready_r : 1'b0;
    end
  endgenerate

  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      iact_spad_waddr <= 0;
      iact_bus_pos    <= 0;
      iact_load_cnt   <= 0;
      iact_set        <= 0;
      iact_ready_r    <= 1;
    end else begin
      if (current_state_computing == IDLE && !iact_set) begin
        if (sel_iact_enable && iact_ready_r) begin
          iact_data_spad[iact_spad_waddr] <=
            sel_iact_data[iact_bus_pos*DATA_IACT_BITWIDTH +: DATA_IACT_BITWIDTH];
          iact_spad_waddr <= iact_spad_waddr + 1;
          iact_load_cnt   <= iact_load_cnt + 1;

          if (iact_load_cnt == iact_addr_max_reg) begin
            iact_set     <= 1;
            iact_ready_r <= 0;
            iact_bus_pos <= 0;
          end else if (iact_bus_pos == VALUES_OF_IACTS - 1) begin
            iact_bus_pos <= 0;
          end else begin
            iact_bus_pos <= iact_bus_pos + 1;
          end
        end
      end else if (current_state_computing == IDLE && iact_set) begin
        // Waiting for wght_set and compute_i — nothing to do
      end

      // Reset after psum send completes (going back to IDLE for next layer)
      if (current_state_computing == SEND_PSUM && !psum_enable_i) begin
        iact_set        <= 0;
        iact_spad_waddr <= 0;
        iact_bus_pos    <= 0;
        iact_load_cnt   <= 0;
        iact_ready_r    <= 1;
      end
    end
  end

  // =========================================================================
  // Weight SPad loading
  // =========================================================================
  // Each TRANS_BITWIDTH_WGHT word packs PARALLEL_MACS * DATA_WGHT_BITWIDTH bits.
  reg [WGHT_DATA_ADDR_BITWIDTH-1:0] wght_spad_waddr;
  reg [WGHT_DATA_ADDR_BITWIDTH-1:0] wght_load_cnt;

  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wght_spad_waddr <= 0;
      wght_load_cnt   <= 0;
      wght_set        <= 0;
      wght_ready_o    <= 1;
    end else begin
      if (current_state_computing == IDLE && !wght_set) begin
        if (wght_enable_i && wght_ready_o) begin
          wght_data_spad[wght_spad_waddr] <= wght_data_i[WGHT_DATA_DATA-1:0];
          wght_spad_waddr <= wght_spad_waddr + 1;
          wght_load_cnt   <= wght_load_cnt + 1;
          if (wght_load_cnt == wght_addr_max_reg) begin
            wght_set     <= 1;
            wght_ready_o <= 0;
          end
        end
      end

      if (current_state_computing == SEND_PSUM && !psum_enable_i) begin
        wght_set        <= 0;
        wght_spad_waddr <= 0;
        wght_load_cnt   <= 0;
        wght_ready_o    <= 1;
      end
    end
  end

  // =========================================================================
  // Configuration streaming FSM
  // =========================================================================
  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      current_state_stream    <= FIRST_PARAMS;
      stride_reg              <= 0;
      wght_addr_max_reg       <= 0;
      filters_reg_M0          <= 0;
      channel_reg_C0          <= 0;
      iact_addr_max_reg       <= 0;
      data_mode_reg           <= 0;
      fraction_bit_reg        <= 0;
      iact_x_line_repetitions <= 0;
    end else begin
      case (current_state_stream)
        FIRST_PARAMS: if (enable_stream_i) begin
          stride_reg            <= data_stream_i[11:9];
          wght_addr_max_reg     <= data_stream_i[8:1];
          current_state_stream  <= SECOND_PARAMS;
        end
        SECOND_PARAMS: if (enable_stream_i) begin
          filters_reg_M0        <= data_stream_i[11:7];
          channel_reg_C0        <= data_stream_i[6:3];
          current_state_stream  <= THIRD_PARAMS;
        end
        THIRD_PARAMS: if (enable_stream_i) begin
          iact_addr_max_reg     <= data_stream_i[IACT_DATA_ADDR_BITWIDTH:1];
          data_mode_reg         <= data_stream_i[0];
          current_state_stream  <= FOURTH_PARAMS;
        end
        FOURTH_PARAMS: if (enable_stream_i) begin
          fraction_bit_reg        <= data_stream_i[5:1];
          iact_x_line_repetitions <= data_stream_i[9:6];
          current_state_stream    <= FIRST_PARAMS;
        end
        default: current_state_stream <= FIRST_PARAMS;
      endcase
    end
  end

  // =========================================================================
  // Compute FSM + MAC datapath
  // =========================================================================
  reg [IACT_DATA_ADDR_BITWIDTH-1:0] iact_addr_cur;
  reg [WGHT_DATA_ADDR_BITWIDTH-1:0] wght_addr_cur;
  reg [PSUM_ADDR_BITWIDTH-1:0]      psum_send_addr;

  // One-cycle pipeline: read data from SPad this cycle, compute MAC next cycle.
  reg [DATA_IACT_BITWIDTH-1:0]  iact_val_pipe;
  reg [DATA_WGHT_BITWIDTH-1:0]  wght_val_pipe;
  reg [PSUM_ADDR_BITWIDTH-1:0]  psum_wr_addr;
  reg                            mac_valid;

  // iact source mux: systolic pass-through or SPad
  wire [DATA_IACT_BITWIDTH-1:0] iact_src =
      (SYSTOLIC_GEMM_EN && iact_pass_enable_i) ? iact_pass_data_i
                                               : iact_data_spad[iact_addr_cur];

  // Weight source: lower DATA_WGHT_BITWIDTH bits of the packed word
  wire [DATA_WGHT_BITWIDTH-1:0] wght_src = wght_data_spad[wght_addr_cur][DATA_WGHT_BITWIDTH-1:0];

  // Accumulated psum (read-then-add)
  wire [DATA_PSUM_BITWIDTH-1:0] psum_cur = psum_spad[psum_wr_addr];

  // MAC
  wire signed [DATA_IACT_BITWIDTH-1:0]  iact_signed = $signed(iact_val_pipe);
  wire signed [DATA_WGHT_BITWIDTH-1:0]  wght_signed = $signed(wght_val_pipe);
  wire signed [DATA_PSUM_BITWIDTH-1:0]  mac_out     = iact_signed * wght_signed;

  integer p;
  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      current_state_computing <= IDLE;
      iact_addr_cur   <= 0;
      wght_addr_cur   <= 0;
      psum_send_addr  <= 0;
      mac_valid       <= 0;
      psum_enable_o   <= 0;
      iact_val_pipe   <= 0;
      wght_val_pipe   <= 0;
      psum_wr_addr    <= 0;
      for (p = 0; p < PSUM_ADDR; p = p + 1)
        psum_spad[p] <= 0;
    end else begin
      mac_valid     <= 0;
      psum_enable_o <= 0;

      // Commit previous MAC result into psum SPad
      if (mac_valid) begin
        psum_spad[psum_wr_addr] <= psum_cur + mac_out;
      end

      case (current_state_computing)

        // -------------------------------------------------------------------
        IDLE: begin
          if (compute_i && iact_set && wght_set) begin
            // Zero psum SPad before a new compute pass
            for (p = 0; p < PSUM_ADDR; p = p + 1)
              psum_spad[p] <= 0;
            iact_addr_cur <= 0;
            wght_addr_cur <= 0;
            current_state_computing <= LOADING_1;
          end
        end

        // -------------------------------------------------------------------
        // LOADING_1: latch first iact/wght so the pipeline has valid data
        //            in CALCULATING on the following cycle.
        LOADING_1: begin
          iact_val_pipe <= iact_src;
          wght_val_pipe <= wght_src;
          psum_wr_addr  <= iact_addr_cur % PSUM_ADDR;
          mac_valid     <= 1;
          current_state_computing <= CALCULATING;
        end

        // -------------------------------------------------------------------
        CALCULATING: begin
          // mac_valid from previous cycle already wrote to psum_spad above.
          // Advance pointers and latch next values.
          if (iact_addr_cur == iact_addr_max_reg) begin
            // Last element was just latched and will fire as mac_valid next cycle.
            iact_val_pipe <= iact_src;
            wght_val_pipe <= wght_src;
            psum_wr_addr  <= iact_addr_cur % PSUM_ADDR;
            mac_valid     <= 1;
            current_state_computing <= WAIT_TO_SEND_PSUM;
          end else begin
            iact_addr_cur <= iact_addr_cur + 1;
            wght_addr_cur <= wght_addr_cur + 1;
            iact_val_pipe <= iact_src;
            wght_val_pipe <= wght_src;
            psum_wr_addr  <= iact_addr_cur % PSUM_ADDR;
            mac_valid     <= 1;
          end
        end

        // -------------------------------------------------------------------
        // WAIT_TO_SEND_PSUM: final mac_valid will commit via top-of-always.
        //                    Wait for psum_enable_i from upstream PE.
        WAIT_TO_SEND_PSUM: begin
          psum_send_addr <= 0;
          if (psum_enable_i)
            current_state_computing <= SEND_PSUM;
        end

        // -------------------------------------------------------------------
        SEND_PSUM: begin
          psum_enable_o <= 1;
          if (psum_ready_i) begin
            if (!psum_enable_i) begin
              // Upstream deasserted — stop sending and go back to IDLE
              psum_enable_o <= 0;
              current_state_computing <= IDLE;
            end else if (psum_send_addr == PSUM_ADDR - 1) begin
              psum_enable_o <= 0;
              current_state_computing <= IDLE;
            end else begin
              psum_send_addr <= psum_send_addr + 1;
            end
          end
        end

        default: current_state_computing <= IDLE;
      endcase
    end
  end

  // =========================================================================
  // Psum output: add incoming partial sum from upstream PE
  // =========================================================================
  wire [DATA_PSUM_BITWIDTH-1:0] psum_local = psum_spad[psum_send_addr];
  wire [DATA_PSUM_BITWIDTH-1:0] psum_accum = psum_local + psum_data_i[DATA_PSUM_BITWIDTH-1:0];

  assign psum_data_o  = {{(TRANS_BITWIDTH_PSUM-DATA_PSUM_BITWIDTH){1'b0}}, psum_accum};
  assign psum_ready_o = (current_state_computing == WAIT_TO_SEND_PSUM ||
                         current_state_computing == SEND_PSUM);

  // =========================================================================
  // Systolic iact pass-through
  // =========================================================================
  generate
    if (SYSTOLIC_GEMM_EN) begin : gen_systolic
      reg [DATA_IACT_BITWIDTH-1:0] iact_pass_data_reg;
      reg                           iact_pass_enable_reg;
      always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
          iact_pass_data_reg   <= 0;
          iact_pass_enable_reg <= 0;
        end else begin
          iact_pass_data_reg   <= iact_pass_data_i;
          iact_pass_enable_reg <= iact_pass_enable_i;
        end
      end
      assign iact_pass_data_o   = iact_pass_data_reg;
      assign iact_pass_enable_o = iact_pass_enable_reg;
      assign iact_pass_ready_o  = iact_pass_ready_i;
    end else begin : gen_no_systolic
      assign iact_pass_data_o   = {DATA_IACT_BITWIDTH{1'b0}};
      assign iact_pass_enable_o = 1'b0;
      assign iact_pass_ready_o  = 1'b0;
    end
  endgenerate

  // =========================================================================
  // data_set (convenience wire matching PE.v interface)
  // =========================================================================
  wire data_set = iact_set && wght_set;

  // ============================================================================
  // FST Waveform Dump Configuration (for CocoTB simulation)
  // ============================================================================
`ifndef NO_TRACE
  reg[1023:0] fst_path;
  initial begin
    if (IS_TOPLEVEL) begin
      if ($value$plusargs("FST_PATH=%s", fst_path)) begin
        $dumpfile(fst_path);
        $dumpvars(0, PE_simple);
      end else begin
        $dumpfile("PE_simple.fst");
        $dumpvars(0, PE_simple);
      end
    end
  end
`endif

endmodule
