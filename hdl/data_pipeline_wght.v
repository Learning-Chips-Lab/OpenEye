// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: data_pipeline_wght
///
/// Weight Data Pipeline for the OpenEye neural network accelerator.
/// Manages the flow of weight data through a dual scratchpad memory system,
/// optimizing data organization for neural network weight distribution.
///
/// Description:
///   This specialized pipeline handles weight data distribution in the OpenEye
///   accelerator. It features a dual-SPAD architecture optimized for weight
///   storage and distribution patterns common in neural network operations.
///   The module supports configurable data widths and memory organizations
///   to accommodate different neural network architectures.
///
/// Operation Modes:
///   1. Data Loading:
///      - Receives weight data in configurable width chunks
///      - Manages data distribution across dual SPADs
///      - Handles address generation and data formatting
///
///   2. Compute Mode:
///      - Coordinates weight distribution during computation
///      - Manages SPAD address spaces efficiently
///      - Handles pipeline flushing and state resets
///
///   3. Data Organization:
///      - Primary SPAD: Stores weight metadata and addressing
///      - Secondary SPAD: Contains actual weight values
///      - Supports configurable data widths and organizations
///
/// Parameters:
///   DATA_WIDTH                - Width of input data bus
///   FIRST_SPAD_ADDR          - Address depth of first SPAD
///   FIRST_SPAD_DATA          - Data width per entry in first SPAD
///   SECOND_SPAD_ADDR         - Address depth of second SPAD
///   SECOND_SPAD_DATA         - Data width per entry in second SPAD
///   SECOND_PAYLOAD_WIDTH     - Actual weight data width in second SPAD
///   FIRST_SPAD_ADDR_BITWIDTH - Address bits for first SPAD (auto-calculated)
///   SECOND_SPAD_ADDR_BITWIDTH- Address bits for second SPAD (auto-calculated)
///   FIRST_SPAD_DATA_CYCLE    - Data cycles for first SPAD (derived)
///   SECOND_SPAD_DATA_CYCLE   - Data cycles for second SPAD (derived)
///   SECOND_OVERHEAD_WIDTH    - Overhead data width in second SPAD
///
/// Ports:
///   Clock and Control:
///     clk_i              - System clock
///     rst_ni            - Asynchronous reset (active low)
///     compute_i         - Triggers compute mode/reset
///     enable_i          - Enables data processing
///
///   Data Interface:
///     data_i[DATA_WIDTH-1:0] - Input weight data
///
///   First SPAD (Metadata) Interface:
///     first_spad_words_o     - Valid word count in first SPAD
///     first_spad_max_i       - Maximum address limit
///     first_spad_addr_o      - Current address
///     first_spad_data_o      - Metadata output
///     first_spad_en_o        - Write enable
///
///   Second SPAD (Weight Data) Interface:
///     second_spad_words_o    - Valid word count
///     second_spad_addr_o     - Current address
///     second_spad_data_o     - Weight data output
///     second_spad_en_o       - Write enable
///
/// Implementation Notes:
///   - Features efficient weight data organization
///   - Supports variable precision weights
///   - Includes metadata management
///   - Handles pipeline stalling and flushing
///   - Provides address space management
///   - Implements cycle-accurate timing control
///   - Manages synchronization between SPADs
///   - Supports dynamic data flow control
///   - Handles boundary conditions and overflows
///   - Implements efficient reset mechanisms
///

module data_pipeline_wght #(
    parameter DATA_WIDTH                = 24,
    parameter FIRST_SPAD_ADDR           = 16,
    parameter FIRST_SPAD_DATA           = 4,
    parameter SECOND_SPAD_ADDR          = 16,
    parameter SECOND_SPAD_DATA          = 12,
    parameter SECOND_PAYLOAD_WIDTH      = 8,
    parameter FIRST_SPAD_ADDR_BITWIDTH  = $clog2(FIRST_SPAD_ADDR),
    parameter SECOND_SPAD_ADDR_BITWIDTH = $clog2(SECOND_SPAD_ADDR),
    parameter FIRST_SPAD_DATA_CYCLE     = DATA_WIDTH / FIRST_SPAD_DATA,
    parameter SECOND_SPAD_DATA_CYCLE    = DATA_WIDTH / SECOND_SPAD_DATA,
    parameter SECOND_OVERHEAD_WIDTH     = SECOND_SPAD_DATA - SECOND_PAYLOAD_WIDTH
) (
    input clk_i,
    input rst_ni,
    input compute_i,
    //input                                         data_mode, Insert later

    input      [                DATA_WIDTH-1 : 0] data_i,
    input                                         enable_i,

    output reg [ $clog2(FIRST_SPAD_ADDR+1)-1 : 0] first_spad_words_o,
    input      [ $clog2(FIRST_SPAD_ADDR+1)-1 : 0] first_spad_max_i,
    output reg [$clog2(SECOND_SPAD_ADDR+1)-1 : 0] second_spad_words_o,

    output reg [  FIRST_SPAD_ADDR_BITWIDTH-1 : 0] first_spad_addr_o,
    output reg [           FIRST_SPAD_DATA-1 : 0] first_spad_data_o,
    output reg                                    first_spad_en_o,

    output reg [ SECOND_SPAD_ADDR_BITWIDTH-1 : 0] second_spad_addr_o,
    output reg [          SECOND_SPAD_DATA-1 : 0] second_spad_data_o,
    output reg                                    second_spad_en_o
);

  reg                                             enable_delay;
  reg         [   FIRST_SPAD_ADDR_BITWIDTH-1 : 0] first_spad_addr_delay;
  reg         [            FIRST_SPAD_DATA-1 : 0] first_spad_data_delay;
  reg  signed [            FIRST_SPAD_DATA-1 : 0] data_storage_1;  // Temporary storage for data
  reg  signed [           SECOND_SPAD_DATA-1 : 0] data_storage_2;  // Temporary storage for data
  reg         [            FIRST_SPAD_DATA-1 : 0] temp_acc_overhead;  // Temporary storage for data
  reg         [     $clog2(SECOND_SPAD_ADDR) : 0] address_temp_2;  // Temporary address storage
  reg         [$clog2(FIRST_SPAD_DATA_CYCLE) : 0] cycle_counter;   // Cycle counter for data loading
  reg         [$clog2(SECOND_OVERHEAD_WIDTH)-1:0] overhead_reg;
  reg         [$clog2(SECOND_OVERHEAD_WIDTH)-1:0] overhead_delay_reg;
  reg         [$clog2(SECOND_OVERHEAD_WIDTH)  :0] overhead_new_calc_reg;
  reg         [                              3:0] cycle_max_reg;
  reg                                             compute_delay;
  reg                                             compute_sent;
  reg                                             over_ending;
  wire        [             SECOND_SPAD_DATA-1:0] input_words_w [0:2-1];
  wire        [                              3:0] overhead_w;
  wire signed [                              7:0] next_channel_counter;
  reg         [                              7:0] overhead_pos;
  wire        [                              1:0] missingvalue;
  wire        [  $clog2(FIRST_SPAD_ADDR+1)-1 : 0] filters_w;
  wire        [                           23 : 0] premade_spad_2_output;
  wire        [                            3 : 0] overhead_output;
  wire        [                            3 : 0] overhead_next_word;

  assign filters_w = first_spad_max_i == 0 ? 16 : first_spad_max_i; 

  assign missingvalue = (overhead_pos + input_words_w[0][SECOND_PAYLOAD_WIDTH+:SECOND_OVERHEAD_WIDTH] < filters_w*2) ? 2 : 1;
  assign next_channel_counter = first_spad_data_o + ((temp_acc_overhead + overhead_w)/2) - data_storage_1 - filters_w;
  genvar w_gen;
  for (w_gen = 0; w_gen < 2; w_gen = w_gen + 1) begin
    assign input_words_w[w_gen] = data_i[(12*w_gen)+:12];
  end
  assign overhead_w = input_words_w[0][SECOND_PAYLOAD_WIDTH+:SECOND_OVERHEAD_WIDTH] + input_words_w[1][SECOND_PAYLOAD_WIDTH+:SECOND_OVERHEAD_WIDTH];
  assign overhead_next_word = input_words_w[0][11:8];
  assign premade_spad_2_output = {data_storage_2[23:12],overhead_output,data_storage_2[7:0]};
  assign overhead_output = (data_storage_2[11:8]!= 0) & overhead_delay_reg + data_storage_2[11:8] >= filters_w ? data_storage_2[11:8] + overhead_delay_reg - filters_w - over_ending: data_storage_2[11:8];


  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin  // Reset
      enable_delay          <= 0;
      first_spad_words_o    <= 0;
      second_spad_words_o   <= 0;
      first_spad_addr_o     <= 0;
      first_spad_addr_delay <= 0;
      first_spad_data_o     <= 0;
      first_spad_data_delay <= 0;
      first_spad_en_o       <= 0;
      second_spad_addr_o    <= 0;
      second_spad_en_o      <= 0;
      data_storage_1        <= 0;
      data_storage_2        <= 0;
      temp_acc_overhead     <= 0;
      address_temp_2        <= 0;
      cycle_counter         <= 0;
      second_spad_data_o    <= 0;
      overhead_reg          <= 0;
      overhead_delay_reg    <= 0;
      overhead_new_calc_reg <= 0;
      compute_delay         <= 0;
      compute_sent          <= 0;
      overhead_pos          <= 0;
      over_ending           <= 0;
    end else begin
      first_spad_en_o   <= 0;
      first_spad_data_o <= 0;
      enable_delay      <= enable_i;
      first_spad_en_o   <= 0;
      second_spad_en_o  <= 0;
      data_storage_2    <= data_i;
      over_ending       <= 0;

      if (enable_delay & (data_storage_2 != 0)) begin
        first_spad_en_o       <= 1;
        second_spad_en_o      <= 1;
        second_spad_data_o    <= premade_spad_2_output;
        if (overhead_new_calc_reg >= filters_w) begin
          first_spad_addr_o  <= first_spad_addr_o + 1;
          first_spad_words_o <= first_spad_addr_o + 2;
        end
        if (second_spad_data_o != 0) begin
          second_spad_addr_o    <= second_spad_addr_o + 1;
        end
        first_spad_data_o     <= first_spad_data_delay;
        second_spad_words_o   <= second_spad_addr_o + 2;
      end else begin
        second_spad_addr_o <= 0;
      end

      if (enable_i == 1 & (data_i != 0)) begin
        compute_sent       <= 0;
        overhead_delay_reg <= overhead_reg;
        if (compute_sent) begin
          second_spad_words_o <= 0;
        end
        cycle_counter <= cycle_counter + 1;
        if (cycle_counter == 0) begin
          cycle_counter <= 0;
        end
        address_temp_2        <= address_temp_2 + 1;
        first_spad_data_delay <= first_spad_data_delay + 1'd1;
        overhead_reg          <= overhead_reg + 2 + overhead_w;
        overhead_new_calc_reg <= overhead_reg;
        if (overhead_reg + overhead_next_word >= filters_w) begin
          overhead_reg <= 2 + overhead_reg + overhead_w - filters_w;
          if (overhead_reg == filters_w + 1) begin
            overhead_reg <= 2 + overhead_reg + overhead_w - filters_w - 1;
            over_ending  <= 1;
          end
          if (overhead_reg < filters_w) begin
            overhead_new_calc_reg <= overhead_reg + 2 + overhead_next_word;
          end
        end
        temp_acc_overhead     <= temp_acc_overhead + overhead_w;
        overhead_pos          <= overhead_pos + 2;
        if (address_temp_2 == SECOND_SPAD_ADDR - 1) begin
          address_temp_2 <= 0;
          cycle_counter  <= 0;
        end
        if (next_channel_counter >=  0) begin
          data_storage_1    <= first_spad_data_delay;
          overhead_pos      <= 0;
          if (data_storage_1 >= 0) begin
            temp_acc_overhead     <= 0;
            first_spad_addr_delay <= first_spad_addr_delay + 1;
          end
        end
      end else begin
        data_storage_1     <= -filters_w;
        overhead_pos       <= filters_w;
        
      end
      compute_delay <= compute_i;
      if (compute_i) begin  // Reset SPAD addresses and state of module
        second_spad_en_o      <= 0;
        first_spad_data_delay <= 0;
        address_temp_2        <= 0;
        cycle_counter         <= 0;
        second_spad_data_o    <= 0;
        overhead_reg          <= 0;
        overhead_new_calc_reg <= 0;
        overhead_delay_reg    <= 0;
        compute_sent          <= 1;
        overhead_pos          <= 0;
      end
      if (compute_delay) begin
        data_storage_1        <= 0;
        data_storage_2        <= 0;
        first_spad_en_o       <= 0;
        first_spad_addr_delay <= 0;
        first_spad_addr_o     <= 0;
      end
    end
  end

endmodule
