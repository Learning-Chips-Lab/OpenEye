// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: data_pipeline_iact
///
/// The Input Activation Data Pipeline manages the flow of input activation data
/// in the OpenEye neural network accelerator. It orchestrates data movement between
/// two ScratchPad (SPAD) memories with different organizations and purposes.
///
/// Description:
///   This module implements a sophisticated data pipeline for handling input activations,
///   featuring dual scratch pad memories with different data organizations. It includes
///   compression-like functionality by only storing non-zero values and their positions,
///   optimizing memory usage for sparse neural networks.
///
/// Operation Phases:
///   1. Data Reception:
///      - Receives wide input data words
///      - Processes data in configurable cycles based on SPAD parameters
///   2. Data Processing:
///      - Detects non-zero values
///      - Manages transmission counters for data positioning
///      - Handles overflow conditions
///   3. Memory Management:
///      - Controls dual SPAD interfaces
///      - Maintains separate address spaces
///      - Tracks valid data words
///
/// Parameters:
///   DATA_WIDTH                - Width of input data bus
///   FIRST_SPAD_ADDR          - Address depth of first SPAD
///   FIRST_SPAD_DATA          - Data width for first SPAD entries
///   SECOND_SPAD_ADDR         - Address depth of second SPAD
///   SECOND_SPAD_DATA         - Data width for second SPAD entries
///   SECOND_PAYLOAD_WIDTH     - Width of actual payload in second SPAD
///   FIRST_SPAD_ADDR_BITWIDTH - Address width for first SPAD (auto-calculated)
///   SECOND_SPAD_ADDR_BITWIDTH- Address width for second SPAD (auto-calculated)
///   FIRST_SPAD_DATA_CYCLE    - Data cycles for first SPAD (derived)
///   SECOND_SPAD_DATA_CYCLE   - Data cycles for second SPAD (derived)
///   SECOND_OVERHEAD_WIDTH    - Width of overhead data in second SPAD
///
/// Ports:
///   Clock and Control:
///     clk_i              - System clock input
///     rst_ni            - Asynchronous reset (active low)
///     compute_i         - Triggers computation/reset cycle
///     enable_i          - Enables data processing
///
///   Data Interface:
///     data_i[DATA_WIDTH-1:0] - Input data word
///
///   First SPAD Interface:
///     first_spad_words_o     - Number of valid words in first SPAD
///     first_spad_max_i       - Maximum address limit for first SPAD
///     first_spad_addr_o      - Address output to first SPAD
///     first_spad_data_o      - Data output to first SPAD
///     first_spad_en_o        - First SPAD write enable
///
///   Second SPAD Interface:
///     second_spad_words_o    - Number of valid words in second SPAD
///     second_spad_addr_o     - Address output to second SPAD
///     second_spad_data_o     - Data output to second SPAD
///     second_spad_en_o       - Second SPAD write enable
///
/// Implementation Notes:
///   - Uses dual-stage buffering for data processing
///   - Implements sparse data optimization
///   - Features automatic overflow protection
///   - Supports variable data widths and organizations
///   - Maintains word counts for both SPADs
///   - Uses transmission counters for precise timing
///   - Handles asynchronous reset conditions
///   - Processes data only when enabled
///   - Supports computation mode for pipeline flushing
///

module data_pipeline_iact #(
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
    input                                         clk_i,
    input                                         rst_ni,
    input                                         compute_i,
    //input                                         data_mode, Insert later

    input      [                DATA_WIDTH-1 : 0] data_i,
    input                                         enable_i,
    input     [                            3 : 0] iact_x_line_repetitions_i,

    output reg [ $clog2(FIRST_SPAD_ADDR+1)-1 : 0] first_spad_words_o,
    input      [   $clog2(FIRST_SPAD_ADDR)-1 : 0] first_spad_max_i,
    output reg [$clog2(SECOND_SPAD_ADDR+1)-1 : 0] second_spad_words_o,

    output reg [  FIRST_SPAD_ADDR_BITWIDTH-1 : 0] first_spad_addr_o,
    output reg [           FIRST_SPAD_DATA-1 : 0] first_spad_data_o,
    output reg                                    first_spad_en_o,

    output reg [ SECOND_SPAD_ADDR_BITWIDTH-1 : 0] second_spad_addr_o,
    output     [          SECOND_SPAD_DATA-1 : 0] second_spad_data_o,
    output reg                                    second_spad_en_o
);

  reg  [            FIRST_SPAD_DATA-1 : 0] data_storage_1;  // Temporary storage for data
  reg  [                 DATA_WIDTH-1 : 0] data_storage_2;  // Temporary storage for data
  reg  [     $clog2(SECOND_SPAD_ADDR) : 0] address_temp_2;  // Temporary address storage
  reg  [$clog2(FIRST_SPAD_DATA_CYCLE) : 0] cycle_counter;   // Cycle counter for data loading
  reg  [                          4-1 : 0] transmission_counter;   // Cycle counter for data loading
  reg  [                          4-1 : 0] transmission_counter_delay;   // Cycle counter for data loading
  reg  [      SECOND_OVERHEAD_WIDTH-1 : 0] overhead_reg;
  reg                                      enable_delay_reg;
  reg  [       SECOND_PAYLOAD_WIDTH-1 : 0] payload_reg;
  wire [                 DATA_WIDTH-1 : 0] current_data;
  reg                                      compute_sent;
  reg                                      uneven_ending;
  reg                                      uneven_ending_storage;
  reg  [                            3 : 0] uneven_counter;

  assign current_data = data_storage_2 >> SECOND_SPAD_DATA;
  assign second_spad_data_o = {
    transmission_counter_delay, payload_reg[SECOND_PAYLOAD_WIDTH-1 : 0]
  };  // Assign data to the second SPAD output

  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin  // Reset
      first_spad_words_o         <= 0;
      second_spad_words_o        <= 0;
      first_spad_data_o          <= 0;
      first_spad_addr_o          <= 0;
      first_spad_en_o            <= 0;
      second_spad_addr_o         <= 0;
      second_spad_en_o           <= 0;
      data_storage_1             <= 0;
      data_storage_2             <= 0;
      address_temp_2             <= 0;
      cycle_counter              <= 0;
      payload_reg                <= 0;
      overhead_reg               <= 0;
      enable_delay_reg           <= 0;
      transmission_counter       <= 0;
      transmission_counter_delay <= 0;
      compute_sent               <= 0;
      uneven_ending              <= 0;
      uneven_ending_storage      <= 0;
      uneven_counter             <= 0;
    end else begin
      first_spad_en_o   <= 0;
      second_spad_en_o  <= 0;
      enable_delay_reg <= enable_i;
      if (enable_i == 1) begin
        first_spad_en_o <= 1;
        compute_sent    <= 0;
        uneven_ending   <= 0;
        if (compute_sent) begin
          second_spad_words_o <= 0;
        end
        cycle_counter              <= cycle_counter + 1;
        transmission_counter       <= transmission_counter + 1;
        transmission_counter_delay <= transmission_counter;
        if (transmission_counter >= (data_storage_1 + first_spad_max_i)) begin
          data_storage_1    <= data_storage_1 + first_spad_max_i;
          first_spad_addr_o <= first_spad_addr_o + 1;
        end
        if (cycle_counter == SECOND_SPAD_DATA_CYCLE - 1) begin
          cycle_counter <= 0;
        end
        second_spad_addr_o  <= address_temp_2[SECOND_SPAD_ADDR_BITWIDTH-1:0];
        data_storage_2      <= current_data;

        if (((cycle_counter == 0) & (data_i[SECOND_PAYLOAD_WIDTH-1:0] != 0)) | ((cycle_counter != 0) & (current_data != 0))) begin
          second_spad_words_o <= second_spad_words_o + 1;
          first_spad_data_o   <= overhead_reg + 1'd1;
          overhead_reg        <= overhead_reg + 1;
          second_spad_en_o    <= 1;
          address_temp_2      <= address_temp_2 + 1;
          if (address_temp_2 == SECOND_SPAD_ADDR - 1) begin
            address_temp_2 <= 0;
            cycle_counter  <= 0;
          end
        end
        if ((cycle_counter == 0) & (data_i != 0)) begin
          payload_reg       <= data_i[SECOND_PAYLOAD_WIDTH-1 : 0];
          data_storage_2    <= data_i;
        end
        if ((cycle_counter != 0) & (current_data != 0)) begin
          payload_reg         <= current_data[SECOND_PAYLOAD_WIDTH-1 : 0];
        end
        if (uneven_ending) begin
          second_spad_words_o <= second_spad_words_o + 1;
          payload_reg         <= data_i[SECOND_SPAD_DATA + SECOND_PAYLOAD_WIDTH-1 : SECOND_SPAD_DATA];
          second_spad_en_o    <= 1;
          address_temp_2      <= address_temp_2 + 1;
          overhead_reg        <= overhead_reg + 1;
          first_spad_data_o   <= overhead_reg + 1'd1;
        end 

      end else begin
        if (enable_delay_reg == 1) begin
          first_spad_en_o     <= 1;
          first_spad_words_o <= first_spad_addr_o + 1;
        end
      end
      if (compute_i) begin  // Reset SPAD addresses and state of module
        first_spad_en_o            <= 0;
        second_spad_en_o           <= 0;
        data_storage_1             <= 0;
        data_storage_2             <= 0;
        first_spad_data_o          <= 0;
        address_temp_2             <= 0;
        first_spad_addr_o          <= 0;
        payload_reg                <= 0;
        overhead_reg               <= 0;
        transmission_counter       <= 0;
        transmission_counter_delay <= 0;
        second_spad_addr_o         <= 0;
        second_spad_words_o        <= 0;
        compute_sent               <= 1;
        cycle_counter              <= 0;
        //Not Fully-Connected
        if (first_spad_max_i < 4) begin
          uneven_counter <= uneven_counter + 1;
          if (uneven_counter == iact_x_line_repetitions_i - 1) begin
            uneven_counter <= 0;
          end
          if (uneven_counter == iact_x_line_repetitions_i - 1) begin
            uneven_ending_storage <= 1 - uneven_ending_storage;
            cycle_counter <= 1 - uneven_ending_storage;
            uneven_ending <= 1 - uneven_ending_storage;
          end else begin
            cycle_counter <= uneven_ending_storage;
            uneven_ending <= uneven_ending_storage;
          end
        end
        if (compute_sent) begin
          cycle_counter  <= 0;
          uneven_ending  <= 0;
          uneven_counter <= 0;
        end
      end
    end
  end

endmodule
