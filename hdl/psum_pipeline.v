// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1

`timescale 1ns / 1ps

/// Module: psum_pipeline
///
/// Extracted partial-sum FSM for OpenEye_FPGA. It owns the PSUM sequencing,
/// maintains the psum staging-buffer address and control signals, and drives
/// the DMA output / quantization write-back paths.
module psum_pipeline #(
    parameter CLUSTER_ROWS = 2,
    parameter CLUSTER_COLUMNS = 2,
    parameter NUM_GLB_IACT = 3,
    parameter NUM_GLB_PSUM = 4,
    parameter NUM_GLB_WGHT = 3,
    parameter CLUSTERS = CLUSTER_COLUMNS * CLUSTER_ROWS,
    parameter PARALLEL_MACS = 2,
    parameter TRANS_WORDS = 8,
    parameter TRANS_BITWIDTH_PSUM = 32,
    parameter BUFFER_WIDTH = 10,
    parameter DMA_BITWIDTH = 64,
    parameter ROUTER_MODES_PSUM = 3,
    parameter QUANT_AMOUNT = 32,
    parameter DATA_PSUM_BITWIDTH = 32,
    parameter PSUM_PER_PE = 32,
    parameter PSUM_CYCLES_ONE_WORD_ALL_CELLS = 2
) (
    input wire clk_i,
    input wire rst_n,

    input wire [DMA_BITWIDTH-1:0] data_dma_i_reg,
    input wire [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_buffer_data_r,
    input wire enable_dma_i_reg,
    input wire ready_dma_i,

    input wire [CLUSTERS*NUM_GLB_PSUM-1:0] psum_ready_o_reg,
    input wire [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_o_w,
    input wire [CLUSTERS*NUM_GLB_PSUM-1:0] psum_enable_o,
    input wire [ROUTER_MODES_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] router_mode_psum,
    input wire [CLUSTERS*NUM_GLB_WGHT-1:0] wght_enable_i_reg,
    input wire [CLUSTERS*NUM_GLB_IACT-1:0] iact_enable_i_oep_w,
    input wire compute_reg,
    input wire status_reg_enable_reg,
    input wire [3:0] fsm_current_state,

    input wire [11:0] psum_size_x,
    input wire [7:0] psum_size_y,
    input wire [7:0] iact_x_line_repetitions,
    input wire [4:0] kernels_per_calc,
    input wire [7:0] needed_wght_cycles,
    input wire [$clog2(CLUSTER_ROWS+1)-1:0] needed_y_cls_reg,
    input wire [7:0] needed_psum_storage_cycles_reg,
    input wire [7:0] iact_channel_max_cycles,
    input wire [3:0] iact_channels_per_pe_next_layer,
    input wire [7:0] output_cycles,
    input wire [5:0] filters,
    input wire [7:0] psum_x_with_add_up,
    input wire [3:0] iteration_for_kernels,
    input wire [17:0] needed_cycles,
    input wire [15:0] trans_cycles_psum,
    input wire [7:0] iact_size_y,
    input wire [11:0] iact_size_x,
    input wire       fully_connected_layer,
    input wire       store_in_psum,
    input wire       send_data_out,

    input wire [8*QUANT_AMOUNT-1:0] quant_offset_flat,
    input wire [7*QUANT_AMOUNT-1:0] quant_exp_flat,
    input wire [25*QUANT_AMOUNT-1:0] quant_mant_flat,
    input wire [16:0] fsm_psum_limit,

    output reg [CLUSTERS*NUM_GLB_PSUM/2-1:0] psum_buffer_en_r,
    output reg [CLUSTERS*NUM_GLB_PSUM/2-1:0] psum_buffer_en_w,
    output     [BUFFER_WIDTH*CLUSTERS*NUM_GLB_PSUM/2-1:0] psum_buffer_addr,
    output reg [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_buffer_data_w,
    output reg [TRANS_BITWIDTH_PSUM*CLUSTERS*NUM_GLB_PSUM-1:0] psum_data_i_reg,
    output reg [CLUSTERS*NUM_GLB_PSUM-1:0] psum_enable_i_reg,
    output reg [CLUSTERS*NUM_GLB_PSUM-1:0] psum_ready_i_reg,
    output     [8*TRANS_WORDS-1:0] quantized_value_flat,
    output reg [DMA_BITWIDTH-1:0] data_dma_o,
    output reg enable_dma_o,
    output reg last_data_o,
    output reg [3:0] fsm_psum_last_state,
    output reg [3:0] fsm_psum_current_state,
    output reg [15:0] fsm_psum_cycle,
    output reg psum_transmitted,
    output reg psum_router_set_reg,
    output reg start_new_cycle,
    output reg last_data_reg,
    output reg [BUFFER_WIDTH-1:0] psum_cnt,
    output reg [7:0] current_filter
);

  localparam PSUM_IDLE                 = 0;
  localparam WAIT_TO_SEND_READY_SIGNAL = 1;
  localparam CALCULATE_PSUM            = 2;
  localparam PSUM_GET_RESULTS          = 3;
  localparam WAIT_FOR_SENDING_RESULTS  = 4;
  localparam PSUM_SEND_RESULTS         = 5;
  localparam SEND_PSUM_TO_IACT         = 6;

  localparam GET_PARAMETERS =  4'd1;
  localparam GET_BIAS = 4'd5;

  reg [7:0] storage_cycles;
  reg [7:0] psum_cycle_buffer_0;
  reg [7:0] psum_cycle_buffer_1;
  reg [7:0] psum_cycle_buffer_2;
  reg [7:0] psum_cycle_buffer_3;
  reg [7:0] psum_cycle_buffer_4;
  reg [3:0] sending_cluster_rows;
  reg [11:0] pcb_1;
  reg [11:0] pcb_2;
  reg [11:0] pcb_3;
  reg [7:0] psum_cycle_inc_0;
  reg [7:0] psum_cycle_inc_1;
  reg [7:0] psum_cycle_inc_2;
  reg [7:0] psum_cycle_inc_3;
  reg [7:0] psum_cycle_inc_4;
  reg [7:0] psum_cycle_limit_0;
  reg [7:0] psum_cycle_limit_1;
  reg [7:0] psum_cycle_limit_2;
  reg [7:0] psum_cycle_limit_3;
  reg [7:0] psum_cycle_limit_4;
  reg [11:0] pcb_inc_1;
  reg [11:0] pcb_inc_2;
  reg [11:0] pcb_inc_3;
  reg [3:0] fsm_psum_row_offset;
  reg [7:0] iact_channel_counter_reg;
  reg [7:0] finished_cycles_psum;
  reg [7:0] psum_buffer_addr_storage;
  reg [BUFFER_WIDTH-1:0] psum_buffer_addr_array [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0][NUM_GLB_PSUM/2-1:0];
  reg [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0][NUM_GLB_PSUM/2-1:0] psum_buffer_en_w_int;
  reg [CLUSTER_COLUMNS-1:0][CLUSTER_ROWS-1:0][NUM_GLB_PSUM/2-1:0] psum_buffer_en_r_int;
  reg [7:0] fsm_x_cl_psum;
  reg [7:0] fsm_x_cl_psum_q1;
  reg [7:0] fsm_x_cl_psum_q2;
  reg [7:0] fsm_x_cl_psum_q3;
  reg [7:0] fsm_y_cl_psum_offset;
  reg [7:0] fsm_y_cl_psum;
  reg [7:0] fsm_y_cl_psum_q1;
  reg [7:0] fsm_y_cl_psum_q2;
  reg [7:0] fsm_y_cl_psum_q3;
  reg [7:0] fsm_psum_r;
  reg [7:0] fsm_psum_r_q;
  reg [7:0] fsm_x_cl_psum_reg;
  reg [7:0] fsm_y_cl_psum_reg;
  reg [DMA_BITWIDTH-1:0] data_dma_o_q1;
  reg [DMA_BITWIDTH-1:0] data_dma_o_q2;
  reg [2:0] data_dma_o_counter;
  reg results_ready;
  reg [7:0] psum_cycle_limit;
  reg [7:0] psum_cycle_count;
  reg [7:0] psum_cycle_buffer;

  integer g_psum, b_psum, cc_psum, cr_psum;

  wire [8-1:0] quant_offset[QUANT_AMOUNT-1:0];
  wire [7-1:0] quant_exp[QUANT_AMOUNT-1:0];
  wire [25-1:0] quant_mant[QUANT_AMOUNT-1:0];
  reg [8-1:0] quantized_value[TRANS_WORDS-1:0];

  generate
    genvar i_addr, j_addr, g_addr, quant_flat_gen;
    for (quant_flat_gen = 0; quant_flat_gen < QUANT_AMOUNT; quant_flat_gen=quant_flat_gen+1) begin : gen_flat_quant_wires
      assign quant_offset[quant_flat_gen] = quant_offset_flat[8*quant_flat_gen+:8];
      assign quant_exp[quant_flat_gen] = quant_exp_flat[7*quant_flat_gen+:7];
      assign quant_mant[quant_flat_gen] = quant_mant_flat[25*quant_flat_gen+:25];
    end
    for (quant_flat_gen = 0; quant_flat_gen < TRANS_WORDS; quant_flat_gen=quant_flat_gen+1) begin : gen_trans_quant_wires
      assign quantized_value_flat[8*quant_flat_gen+:8] = quantized_value[quant_flat_gen];
    end
    for (i_addr = 0; i_addr < CLUSTER_COLUMNS; i_addr = i_addr + 1) begin : ADDR_X
      for (j_addr = 0; j_addr < CLUSTER_ROWS; j_addr = j_addr + 1) begin : ADDR_Y
        for (g_addr = 0; g_addr < NUM_GLB_PSUM/2; g_addr = g_addr + 1) begin : ADDR_GLB
          assign psum_buffer_addr[(i_addr*BUFFER_WIDTH*CLUSTER_ROWS*NUM_GLB_PSUM/2)+(j_addr*BUFFER_WIDTH*NUM_GLB_PSUM/2)+(g_addr*BUFFER_WIDTH)+:BUFFER_WIDTH] = psum_buffer_addr_array[i_addr][j_addr][g_addr];
        end
      end
    end
  endgenerate

  always @(posedge clk_i or negedge rst_n) begin
    if (!rst_n) begin
      psum_transmitted            <= 0;
      fsm_psum_cycle              <= 0;
      fsm_psum_last_state         <= PSUM_IDLE;
      fsm_psum_current_state      <= PSUM_IDLE;
      for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
        for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
          for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
            psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= ~0;
          end
        end
      end
      psum_buffer_addr_storage <= 0;
      psum_enable_i_reg           <= 0;
      psum_ready_i_reg            <= 0;
      storage_cycles              <= 0;
      psum_router_set_reg         <= 1;
      iact_channel_counter_reg    <= 0;
      results_ready               <= 0;
      psum_cnt                    <= 0;
      start_new_cycle             <= 0;
      enable_dma_o                <= 0;
      data_dma_o                  <= 0;
      last_data_reg               <= 0;
      fsm_x_cl_psum               <= 0;
      fsm_y_cl_psum               <= 0;
      fsm_y_cl_psum_offset        <= 0;
      fsm_psum_r                  <= 0;
      fsm_psum_r_q                <= 0;
      psum_buffer_en_r            <= 0;
      last_data_o                 <= 0;
      current_filter              <= 0;
      psum_cycle_buffer_0         <= 0;
      psum_cycle_buffer_1         <= 0;
      psum_cycle_buffer_2         <= 0;
      psum_cycle_buffer_3         <= 0;
      psum_cycle_buffer_4         <= 0;
      sending_cluster_rows        <= 0;
      pcb_1                       <= 0;
      pcb_2                       <= 0;
      pcb_3                       <= 0;
      psum_cycle_inc_0            <= 0;
      psum_cycle_inc_1            <= 0;
      psum_cycle_inc_2            <= 0;
      psum_cycle_inc_3            <= 0;
      psum_cycle_inc_4            <= 0;
      psum_cycle_limit_0          <= 0;
      psum_cycle_limit_1          <= 0;
      psum_cycle_limit_2          <= 0;
      psum_cycle_limit_3          <= 0;
      psum_cycle_limit_4          <= 0;
      pcb_inc_1                   <= 0;
      pcb_inc_2                   <= 0;
      pcb_inc_3                   <= 0;
      fsm_psum_row_offset         <= 0;
      fsm_x_cl_psum_q1            <= 0;
      fsm_x_cl_psum_q2            <= 0;
      fsm_x_cl_psum_q3            <= 0;
      psum_data_i_reg             <= 0;
      psum_cycle_count            <= 0;
      for (cr_psum = 0; cr_psum < TRANS_WORDS; cr_psum = cr_psum + 1) begin
        quantized_value[cr_psum] <= 0;
      end
      finished_cycles_psum        <= 0;
      psum_buffer_data_w          <= 0;
      data_dma_o_q1               <= 0;
      data_dma_o_q2               <= 0;
      data_dma_o_counter          <= 0;
    end else begin
      case (fsm_psum_current_state)
        PSUM_IDLE: begin
          psum_buffer_en_w <= 0;
          psum_enable_i_reg   <= 0;
          last_data_reg       <= 0;
          current_filter      <= 0;
          psum_cycle_buffer_0 <= 0;
          psum_data_i_reg     <= 0;
          if (ready_dma_i == 1) begin
            enable_dma_o <= 0;
            last_data_o  <= 0;
          end
          if (GET_PARAMETERS == fsm_current_state) begin
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                  psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= ~0;
                end
              end
            end
            fsm_x_cl_psum <= 0;
          end
          if (GET_BIAS == fsm_current_state) begin
            if (enable_dma_i_reg) begin
              fsm_psum_cycle <= fsm_psum_cycle + 1;
              psum_cycle_count <= psum_cycle_count + 1;
              if (fsm_psum_cycle == trans_cycles_psum - 1) begin
                fsm_psum_cycle <= 0;
                psum_cnt       <= psum_buffer_addr_array[0][0][0] + 1;
              end
              // Dense/FC bias loading (DensePsumStreamMapper.get_psum_stream)
              // sends one bias word per DMA cycle, cycling through all
              // CLUSTER_COLUMNS x-clusters' values for the current filter
              // index before moving to the next filter - unlike Conv's
              // packed-wide-word scheme that needs
              // PSUM_CYCLES_ONE_WORD_ALL_CELLS cycles to gather one advance's
              // worth of data. Write the current cycle's word into just its
              // cc slot (broadcast across cr/g, which all share one bias
              // value per filter) and advance the shared address only after
              // CLUSTER_COLUMNS cycles, so trans_cycles_psum can equal the
              // actual DMA stream length the test harness sends (fixed
              // lockstep schedule, no flow control) instead of being inflated
              // by PSUM_CYCLES_ONE_WORD_ALL_CELLS and starving later
              // sections.
              if (fully_connected_layer) begin
                for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                  for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                    psum_buffer_data_w[fsm_x_cl_psum*CLUSTER_ROWS*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+cr_psum*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+g_psum*TRANS_BITWIDTH_PSUM*PARALLEL_MACS+:TRANS_BITWIDTH_PSUM*PARALLEL_MACS] <= data_dma_i_reg[TRANS_BITWIDTH_PSUM*PARALLEL_MACS-1:0];
                  end
                end
                psum_buffer_en_w <= 0;
                fsm_x_cl_psum <= fsm_x_cl_psum + 1;
                if (fsm_x_cl_psum == CLUSTER_COLUMNS - 1) begin
                  fsm_x_cl_psum <= 0;
                  for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                    for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                      for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                        psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_addr_array[cc_psum][cr_psum][g_psum] + 1;
                      end
                    end
                  end
                  psum_buffer_en_w <= ~0;
                end
              end else begin
                psum_buffer_data_w[0+:DMA_BITWIDTH] <= data_dma_i_reg;
                for (g_psum = 0; g_psum < PSUM_CYCLES_ONE_WORD_ALL_CELLS - 1; g_psum = g_psum + 1) begin
                  psum_buffer_data_w[DMA_BITWIDTH*(1+g_psum)+:DMA_BITWIDTH] <= psum_buffer_data_w[DMA_BITWIDTH*g_psum+:DMA_BITWIDTH];
                end
                psum_buffer_en_w <= 0;
                if (psum_cycle_count == PSUM_CYCLES_ONE_WORD_ALL_CELLS - 1) begin
                  psum_cycle_count <= 0;
                  for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                    for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                      for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                        psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_addr_array[cc_psum][cr_psum][g_psum] + 1;
                      end
                    end
                  end
                  psum_buffer_en_w <= ~0;
                end
              end
            end
          end
          psum_transmitted    <= 0;
          fsm_psum_last_state <= PSUM_IDLE;
          storage_cycles      <= 0;
          if (compute_reg) begin
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                  psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= 0;
                end
              end
            end
            psum_buffer_data_w     <= 0;
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
          psum_buffer_en_r <= {(NUM_GLB_PSUM*CLUSTERS/2){1'd1}};
          if (fsm_psum_cycle == 16) begin
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
          psum_buffer_en_r <= {(NUM_GLB_PSUM*CLUSTERS/2){1'd1}};
          if ((results_ready == 0) & (psum_ready_i_reg != 0)) begin
            results_ready = 1;
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                  results_ready = results_ready & (psum_ready_o_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum*2] |
                   ((router_mode_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM * 2 + 2] == 0) & (CLUSTERS != 1)));
                end
              end
            end
          end
          if (results_ready & (psum_ready_i_reg != 0)) begin
            fsm_psum_cycle  <= fsm_psum_cycle + 1;
            psum_data_i_reg <= psum_buffer_data_r;
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                  if (router_mode_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM * 2 + 2] == 1 | (CLUSTERS == 1)) begin
                    psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_addr_array[cc_psum][cr_psum][g_psum] + 1;
                  end
                  // g_psum indexes GLB pairs (loop bound NUM_GLB_PSUM/2), so
                  // the stride into the full NUM_GLB_PSUM-wide router field
                  // must be g_psum*ROUTER_MODES_PSUM*2, matching every other
                  // use of this bit-2 check in this state (lines above/below).
                  if ((fsm_psum_cycle != 0) & ((router_mode_psum[(cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM) + (cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM) + (g_psum * ROUTER_MODES_PSUM * 2) + 2] == 1) | (CLUSTERS == 1))) begin
                    psum_enable_i_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum * 2] <= 1;
                    psum_enable_i_reg[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum * 2 + 1] <= 1;
                  end
                end
              end
            end
            if (fsm_psum_cycle > {{10{1'd0}},filters}) begin
              psum_buffer_en_r    <= 0;
              fsm_psum_last_state    <= CALCULATE_PSUM;
              fsm_psum_current_state <= PSUM_GET_RESULTS;
              results_ready           <= 0;
              fsm_psum_cycle         <= 0;
              psum_enable_i_reg      <= 0;
              for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                  for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                    if ((router_mode_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM * 2 + 2] == 1) | (CLUSTERS == 1)) begin
                      psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_addr_storage;
                    end
                  end
                end
              end
            end
          end
        end

        PSUM_GET_RESULTS: begin
          // psum_enable_i must stay held high for the whole streaming burst
          // (PE.v SEND_PSUM only advances/streams while its psum_enable_i
          // is asserted, and returns to IDLE the instant it sees it low).
          // The two exit branches below explicitly clear psum_enable_i_reg
          // once the burst is actually done; do not clear it every cycle
          // here or the compute core streams exactly one result then stops.
          results_ready      <= 1;
          psum_ready_i_reg   <= psum_ready_i_reg;
          psum_buffer_data_w <= psum_data_o_w;
          for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
            for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
              for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                if (psum_buffer_en_w[cc_psum*NUM_GLB_PSUM/2*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM/2+g_psum]) begin
                  psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_addr_array[cc_psum][cr_psum][g_psum] + 1;
                end
                results_ready = results_ready & (psum_enable_o[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum * 2] |
                ((router_mode_psum[cc_psum * CLUSTER_ROWS * NUM_GLB_PSUM * ROUTER_MODES_PSUM + cr_psum * NUM_GLB_PSUM * ROUTER_MODES_PSUM + g_psum * ROUTER_MODES_PSUM * 2 + 2] == 0) & (CLUSTERS != 1)));
                if (psum_enable_o[cc_psum*NUM_GLB_PSUM*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM+g_psum * 2] != 0) begin
                  psum_buffer_en_w[cc_psum*NUM_GLB_PSUM/2*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM/2+g_psum] <= 1;
                end else begin
                  psum_buffer_en_w[cc_psum*NUM_GLB_PSUM/2*CLUSTER_ROWS+cr_psum*NUM_GLB_PSUM/2+g_psum] <= 0;
                end
              end
            end
          end
          if (results_ready) begin
            psum_router_set_reg <= 0;
            fsm_psum_cycle      <= fsm_psum_cycle + 1;
          end
          if (fsm_psum_cycle[$clog2(PSUM_PER_PE+1 )-1:0] >= filters) begin
            psum_transmitted       <= 1;
            if ((finished_cycles_psum == needed_cycles - 1)) begin
              for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                  for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                    psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= 0;
                  end
                end
              end
              fsm_psum_last_state    <= PSUM_GET_RESULTS;
              fsm_psum_current_state <= WAIT_FOR_SENDING_RESULTS;
              psum_ready_i_reg       <= 0;
              fsm_psum_cycle         <= 0;
              psum_buffer_en_w       <= 0;
              psum_buffer_en_r       <= ~0;
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
              psum_buffer_en_w    <= 0;
              if (storage_cycles == needed_psum_storage_cycles_reg - 1) begin
                storage_cycles <= 0;
                psum_buffer_addr_storage <= psum_buffer_addr_storage + {{6{1'd0}}, filters};
                for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                  for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                    for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                      psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_addr_storage + {{6{1'd0}}, filters};
                    end
                  end
                end
              end else begin
                storage_cycles <= storage_cycles + 1;
                for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                  for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                    for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                      psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_addr_storage;
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
          psum_cycle_buffer_0 <= 0;
          if (send_data_out) begin
            if (fsm_psum_cycle == 1) begin
              fsm_psum_cycle         <= 0;
              fsm_psum_current_state <= PSUM_SEND_RESULTS;
              psum_buffer_en_r       <= {(NUM_GLB_PSUM/2*CLUSTER_ROWS*CLUSTER_COLUMNS){1'd1}};
            end
          end else begin
            fsm_y_cl_psum          <= 0;
            if (fsm_psum_cycle >= 2) begin
              if (store_in_psum == 0) begin
                fsm_psum_current_state    <= SEND_PSUM_TO_IACT;
                if (CLUSTER_COLUMNS * NUM_GLB_PSUM < 8) begin
                  psum_cycle_inc_0     <= 2 * CLUSTER_COLUMNS * NUM_GLB_PSUM * iact_x_line_repetitions;
                  sending_cluster_rows <= 2;
                end else begin
                  psum_cycle_inc_0     <= CLUSTER_COLUMNS * NUM_GLB_PSUM * iact_x_line_repetitions;
                  sending_cluster_rows <= 1;
                end
                if (psum_size_x <= NUM_GLB_PSUM) begin
                  psum_cycle_inc_0     <= iact_x_line_repetitions;
                  sending_cluster_rows <= 1;
                end
              end else begin
                fsm_psum_current_state <= PSUM_IDLE;
              end
              fsm_psum_cycle       <= 0;
              psum_cycle_buffer_0  <= 0;
              psum_cycle_buffer_1  <= 0;
              psum_cycle_buffer_2  <= 0;
              psum_cycle_buffer_3  <= 0;
              psum_cycle_buffer_4  <= 0;
              pcb_1                <= 0;
              pcb_2                <= 0;
              pcb_3                <= 0;
              psum_cycle_inc_1     <= 1;
              psum_cycle_inc_2     <= kernels_per_calc;
              psum_cycle_inc_3     <= 1;
              psum_cycle_inc_4     <= 1;
              pcb_inc_1            <= 1;
              pcb_inc_2            <= (filters * iact_x_line_repetitions * needed_wght_cycles);
              pcb_inc_3            <= iteration_for_kernels;
              psum_cycle_limit_0   <= kernels_per_calc * psum_x_with_add_up;
              if (iact_size_x <= NUM_GLB_PSUM) begin
                psum_cycle_limit_0   <= iact_channels_per_pe_next_layer;
              end
              psum_cycle_limit_1   <= iact_x_line_repetitions - 1;
              psum_cycle_limit_2   <= iact_channels_per_pe_next_layer;
              psum_cycle_limit_3   <= psum_size_y  - 1;
              psum_cycle_limit_4   <= kernels_per_calc - 1;
              if (psum_size_x <= NUM_GLB_PSUM) begin
                psum_cycle_limit_4   <= 1;
              end
              fsm_psum_row_offset  <= 0;
            end
          end
          fsm_psum_r    <= 0;
          fsm_y_cl_psum <= 0;
          fsm_x_cl_psum <= 0;
        end

        PSUM_SEND_RESULTS: begin
          psum_buffer_en_r <= 0;
          if (ready_dma_i == 1) begin
            if (data_dma_o_counter != 0) begin
              data_dma_o_counter <= data_dma_o_counter -1;
            end
            data_dma_o_counter <= 0;
            data_dma_o_q1      <= 0;
            data_dma_o_q2      <= data_dma_o_q1;
            data_dma_o_counter <= 0;
            fsm_psum_r_q       <= fsm_psum_r;
            fsm_x_cl_psum_q1   <= fsm_x_cl_psum;
            fsm_x_cl_psum_q2   <= fsm_x_cl_psum_q1;
            fsm_x_cl_psum_q3   <= fsm_x_cl_psum_q2;
            fsm_y_cl_psum_q1   <= fsm_y_cl_psum;
            if (fsm_psum_r | fsm_x_cl_psum | fsm_y_cl_psum | fsm_psum_cycle) begin
              enable_dma_o <= 1;
            end
            data_dma_o <= psum_buffer_data_r[(fsm_x_cl_psum_q1*CLUSTER_ROWS*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM)+(fsm_y_cl_psum_q1*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM)+(fsm_psum_r_q*PARALLEL_MACS*TRANS_BITWIDTH_PSUM)+:TRANS_BITWIDTH_PSUM * PARALLEL_MACS];
            if (data_dma_o_counter == 2) begin
              data_dma_o <= data_dma_o_q2;
            end
            for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
              for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                  if ((fsm_psum_r == g_psum) & (fsm_x_cl_psum == cc_psum) & (fsm_y_cl_psum == cr_psum)) begin
                    psum_buffer_en_r[cc_psum*CLUSTER_ROWS*NUM_GLB_PSUM/2+cr_psum*NUM_GLB_PSUM/2+g_psum] <= 1;
                    psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_addr_array[cc_psum][cr_psum][g_psum] + 1;
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
                // FC mode: CALCULATE_PSUM/PSUM_GET_RESULTS gate all bias-feed
                // and result-capture activity on router_mode_psum bit 2, which
                // dense_mapper.py's write_router_psum() sets to 1 only on
                // cluster row 0 (mode 5, "chain start") - row 0 is the port the
                // hardware actually drives psum_enable_o/psum_data_o_w on for
                // the whole chain, confirmed by simulation trace (psum_ready_o,
                // psum_enable_o and psum_buffer_en_w all read active only for
                // cr=0 throughout CALCULATE_PSUM/PSUM_GET_RESULTS). So results
                // live in row 0's bank, not CLUSTER_ROWS-1; keep fsm_y_cl_psum
                // at 0 here (matching the write side) instead of sweeping.
                fsm_y_cl_psum <= fully_connected_layer ? 0 : (fsm_y_cl_psum + needed_y_cls_reg);
                if ((fsm_y_cl_psum >= CLUSTER_ROWS - needed_y_cls_reg) | fully_connected_layer) begin
                  fsm_y_cl_psum <= 0;
                  fsm_psum_cycle <= fsm_psum_cycle + 1;
                  if (fsm_psum_cycle == (needed_wght_cycles * filters * output_cycles) - 1) begin
                    fsm_psum_cycle      <= 0;
                    for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                      for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                        for (g_psum = 0; g_psum < NUM_GLB_PSUM/2; g_psum = g_psum + 1) begin
                          psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= 0;
                        end
                      end
                    end
                    psum_buffer_en_r <= 0;
                    last_data_reg       <= 1;
                  end
                end
              end
            end
            if (last_data_reg) begin
              psum_buffer_en_r    <= 0;
              last_data_o            <= 1;
              last_data_reg          <= 0;
              finished_cycles_psum   <= 0;
              fsm_psum_last_state    <= PSUM_SEND_RESULTS;
              fsm_psum_current_state <= PSUM_IDLE;
              fsm_psum_r             <= 0;
              fsm_y_cl_psum          <= 0;
              fsm_x_cl_psum          <= 0;
            end
          end else begin
            if (data_dma_o_counter != 2) begin
              data_dma_o_counter <= data_dma_o_counter + 1;
              data_dma_o_q1      <= psum_buffer_data_r[(fsm_x_cl_psum_q1*CLUSTER_ROWS*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM)+(fsm_y_cl_psum_q1*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM)+(fsm_psum_r_q*PARALLEL_MACS*TRANS_BITWIDTH_PSUM)+:TRANS_BITWIDTH_PSUM * PARALLEL_MACS];
              data_dma_o_q2      <= data_dma_o_q1;
            end
          end
        end

        SEND_PSUM_TO_IACT: begin
          fsm_y_cl_psum_q1 <= fsm_y_cl_psum;
          fsm_y_cl_psum_q2 <= fsm_y_cl_psum_q1;
          fsm_y_cl_psum_q3 <= fsm_y_cl_psum_q2;
          fsm_x_cl_psum_q1 <= fsm_x_cl_psum;
          fsm_x_cl_psum_q2 <= fsm_x_cl_psum_q1;
          fsm_x_cl_psum_q3 <= fsm_x_cl_psum_q2;
          if (fsm_psum_cycle >= 3) begin
            if (!fully_connected_layer | (CLUSTERS == 1)) begin
              if (CLUSTER_COLUMNS*NUM_GLB_PSUM>= 8) begin
                for (cr_psum = 0; cr_psum < TRANS_WORDS/2; cr_psum = cr_psum + 1) begin
                  quantized_value[2*cr_psum]     <= (quant_mant[current_filter] *
                  (psum_buffer_data_r[((fsm_x_cl_psum_q3+(cr_psum/2))*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM)+(fsm_y_cl_psum_q3*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM)+
                  ((cr_psum%2)*TRANS_BITWIDTH_PSUM*PARALLEL_MACS)+:TRANS_BITWIDTH_PSUM]
                  + quant_offset[current_filter]))
                  >>> quant_exp[current_filter];
                  quantized_value[2*cr_psum + 1] <= (quant_mant[current_filter] *
                  (psum_buffer_data_r[((fsm_x_cl_psum_q3+(cr_psum/2))*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM)+(fsm_y_cl_psum_q3*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM)+
                  ((cr_psum%2)*TRANS_BITWIDTH_PSUM*PARALLEL_MACS+TRANS_BITWIDTH_PSUM)+:TRANS_BITWIDTH_PSUM]
                  + quant_offset[current_filter]))
                  >>> quant_exp[current_filter];
                end
              end else begin
                if (CLUSTERS*NUM_GLB_PSUM > TRANS_WORDS) begin
                  for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
                    for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
                      quantized_value[(TRANS_WORDS/2)*cr_psum+2*cc_psum]     <= (quant_mant[current_filter] *
                      (psum_buffer_data_r[cc_psum*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+(cr_psum+(fsm_y_cl_psum_q3))*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+:TRANS_BITWIDTH_PSUM]
                      + quant_offset[current_filter]))
                      >>> quant_exp[current_filter];
                      quantized_value[(TRANS_WORDS/2)*cr_psum+2*cc_psum + 1] <= (quant_mant[current_filter] *
                      (psum_buffer_data_r[cc_psum*TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+(cr_psum+(fsm_y_cl_psum_q3))*TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM+TRANS_BITWIDTH_PSUM+:TRANS_BITWIDTH_PSUM]
                      + quant_offset[current_filter]))
                      >>> quant_exp[current_filter];
                    end
                  end
                end else begin
                  for (cc_psum = 0; cc_psum < TRANS_WORDS; cc_psum = cc_psum + 1) begin
                    quantized_value[cc_psum]     <= (quant_mant[current_filter] *
                    (psum_buffer_data_r[cc_psum*TRANS_BITWIDTH_PSUM+:TRANS_BITWIDTH_PSUM]
                    + quant_offset[current_filter]))
                    >>> quant_exp[current_filter];
                  end
                end
              end
            end else begin
              quantized_value[0] <= (quant_mant[current_filter] *
              (psum_buffer_data_r[0+:TRANS_BITWIDTH_PSUM]
              + quant_offset[current_filter]))
              >>> quant_exp[current_filter];
              if (CLUSTER_ROWS == 2) begin
                quantized_value[1] <= (quant_mant[current_filter] *
                (psum_buffer_data_r[TRANS_BITWIDTH_PSUM*CLUSTER_ROWS*NUM_GLB_PSUM+:TRANS_BITWIDTH_PSUM]
                + quant_offset[current_filter]))
                >>> quant_exp[current_filter];
              end
            end
          end
          for (cc_psum = 0; cc_psum < CLUSTER_COLUMNS; cc_psum = cc_psum + 1) begin
            for (cr_psum = 0; cr_psum < CLUSTER_ROWS; cr_psum = cr_psum + 1) begin
              for (g_psum = 0; g_psum < (NUM_GLB_PSUM/2); g_psum = g_psum + 1) begin
                if (((cr_psum >= fsm_y_cl_psum) & (cr_psum < fsm_y_cl_psum+sending_cluster_rows)) | (CLUSTERS*NUM_GLB_PSUM <= 8)) begin
                  if (psum_size_x > NUM_GLB_PSUM | (cc_psum== fsm_x_cl_psum)) begin
                    psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= psum_buffer_addr_array[cc_psum][cr_psum][g_psum] + filters * needed_wght_cycles;
                    if (psum_cycle_buffer_1 == 0) begin
                      psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= pcb_1;
                      if (psum_cycle_buffer_2 == 0)  begin
                        psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= pcb_2;
                        if (psum_cycle_buffer_3 == 0) begin
                          psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= pcb_3;
                        end
                      end
                    end
                  end
                end
              end
            end
          end
          psum_cycle_buffer_0 <= psum_cycle_buffer_0 + psum_cycle_inc_0;
          if ((psum_cycle_buffer_0 + psum_cycle_inc_0 >= psum_cycle_limit_0) | (iact_channels_per_pe_next_layer == 1)) begin
            psum_cycle_buffer_0 <= 0;
            psum_cycle_buffer_1 <= psum_cycle_buffer_1 + psum_cycle_inc_1;
            if (psum_cycle_buffer_1 == psum_cycle_limit_1) begin
              psum_cycle_buffer_1 <= 0;
              pcb_1               <= pcb_1 + pcb_inc_1;
              psum_cycle_buffer_2 <= psum_cycle_buffer_2 + psum_cycle_inc_2;
              if (psum_cycle_buffer_2 + psum_cycle_inc_2>= psum_cycle_limit_2) begin
                psum_cycle_buffer_2 <= 0;
                pcb_1               <= pcb_2 + pcb_inc_2;
                pcb_2               <= pcb_2 + pcb_inc_2;
                psum_cycle_buffer_3 <= psum_cycle_buffer_3 + psum_cycle_inc_3;
                if ((psum_cycle_buffer_3 == psum_cycle_limit_3) ) begin
                  psum_cycle_buffer_3 <= 0;
                  pcb_1               <= pcb_3;
                  pcb_2               <= pcb_3;
                  psum_cycle_buffer_4 <= psum_cycle_buffer_4 + psum_cycle_inc_4;
                  if (psum_cycle_buffer_4 == psum_cycle_limit_4) begin
                    psum_cycle_buffer_4 <= 0;
                    pcb_1               <= pcb_3 + pcb_inc_3;
                    pcb_2               <= pcb_3 + pcb_inc_3;
                    pcb_3               <= pcb_3 + pcb_inc_3;
                  end
                end
              end
            end
          end
          if (psum_size_x > NUM_GLB_PSUM * CLUSTER_COLUMNS) begin
            if (iact_channels_per_pe_next_layer == 4) begin
              fsm_y_cl_psum <= fsm_y_cl_psum + sending_cluster_rows;
              if (fsm_y_cl_psum + sending_cluster_rows >= CLUSTER_ROWS | ((fsm_y_cl_psum+sending_cluster_rows)*(NUM_GLB_PSUM*CLUSTER_COLUMNS) >= kernels_per_calc * psum_x_with_add_up)) begin
                fsm_y_cl_psum <= 0;
              end
            end else begin
              if ((psum_cycle_buffer_3 == psum_size_y  - 1)) begin
                fsm_y_cl_psum <= fsm_y_cl_psum + sending_cluster_rows;
                if (fsm_y_cl_psum + sending_cluster_rows >= CLUSTER_ROWS | ((fsm_y_cl_psum+sending_cluster_rows)*(NUM_GLB_PSUM*CLUSTER_COLUMNS) >= kernels_per_calc * psum_x_with_add_up)) begin
                  fsm_y_cl_psum <= 0;
                end
              end
            end
          end else begin
            if (psum_size_x <= 4) begin
              fsm_x_cl_psum <= fsm_x_cl_psum + 1;
              if (fsm_x_cl_psum == CLUSTER_COLUMNS - 1) begin
                fsm_x_cl_psum <= 0;
                fsm_y_cl_psum <= fsm_y_cl_psum + 1;
                if (fsm_y_cl_psum ==fsm_y_cl_psum_offset +  kernels_per_calc/4 - 1) begin
                  fsm_y_cl_psum <= fsm_y_cl_psum_offset;
                  if (psum_cycle_buffer_3 == psum_size_y  - 1) begin
                    fsm_y_cl_psum        <= fsm_y_cl_psum + 1;
                    fsm_y_cl_psum_offset <= fsm_y_cl_psum + 1;
                    if (fsm_y_cl_psum + 1 >= CLUSTER_ROWS) begin
                      fsm_y_cl_psum        <= 0;
                      fsm_y_cl_psum_offset <= 0;
                    end
                  end
                end
              end
            end else begin
              if (!fully_connected_layer) begin
                if (psum_cycle_buffer_3 == psum_size_y - 1) begin
                  fsm_y_cl_psum <= fsm_y_cl_psum + sending_cluster_rows;
                  if (fsm_y_cl_psum + sending_cluster_rows>= CLUSTER_ROWS) begin
                    fsm_y_cl_psum <= 0;
                  end
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
            fsm_y_cl_psum_offset        <= 0;
            fsm_x_cl_psum               <= 0;
            psum_cycle_buffer_0         <= 0;
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
              psum_buffer_addr_array[cc_psum][cr_psum][g_psum] <= ~0;
            end
          end
        end
        psum_enable_i_reg           <= 0;
        psum_ready_i_reg            <= 0;
        fsm_psum_last_state         <= PSUM_SEND_RESULTS;
        fsm_psum_current_state      <= PSUM_IDLE;
        fsm_psum_cycle              <= 0;
        psum_buffer_addr_storage    <= 0;
        storage_cycles              <= 0;
        psum_router_set_reg         <= 1;
        iact_channel_counter_reg    <= 0;
        results_ready               <= 0;
        psum_cnt                    <= 0;
        start_new_cycle             <= 0;
        enable_dma_o                <= 0;
        data_dma_o                  <= 0;
        data_dma_o_q1               <= 0;
        data_dma_o_q2               <= 0;
        data_dma_o_counter          <= 0;
        last_data_reg               <= 0;
        fsm_x_cl_psum               <= 0;
        fsm_y_cl_psum               <= 0;
        psum_buffer_en_r            <= 0;
        finished_cycles_psum        <= 0;
        fsm_psum_r                  <= 0;
        fsm_x_cl_psum               <= 0;
      end
    end
  end

endmodule
