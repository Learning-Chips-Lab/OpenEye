// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: data_pipeline
///
/// The data_pipeline module manages the flow of data through a processing system.
/// It handles the loading of addresses and data into two separate ScratchPad (SPAD) memories. 
/// memories, transitioning between these states based on the FSM mode.
/// The module is designed for sequential operations, processing data in cycles.
///
/// Parameters:
///    DATA_WIDTH:                 Bit-width of input data
///    FIRST_SPAD_ADDR:            Number of addresses in the first SPAD
///    FIRST_SPAD_DATA:            Amount of data stored in one address of the first SPAD
///    SECOND_SPAD_ADDR:           Number of addresses in the second SPAD
///    SECOND_SPAD_DATA:           Amount of data stored in one address of the second SPAD
///    FIRST_SPAD_ADDR_BITWIDTH:   Bit-width required for addressing the first SPAD
///    SECOND_SPAD_ADDR_BITWIDTH:  Bit-width required for addressing the second SPAD
///    FIRST_SPAD_DATA_CYCLE:      Number of cycles required to load data into the first SPAD
///    SECOND_SPAD_DATA_CYCLE:     Number of cycles required to load data into the second SPAD
///
/// Ports:
///    clk_i:                      Clock signal input
///    rst_ni:                     Asynchronous reset input (active low)
///    compute_i:                  Signal to reset addresses and state of the module
///    data_i:                     Input data to be processed and stored
///    enable_i:                   Enable signal for data processing
///    first_spad_words_o:         Number of words stored in the first SPAD
///    second_spad_words_o:        Number of words stored in the second SPAD
///    first_spad_addr_o:          Address output for the first SPAD
///    first_spad_data_o:          Data output from the first SPAD
///    first_spad_en_o:            Enable signal for the first SPAD
///    second_spad_addr_o:         Address output for the second SPAD
///    second_spad_data_o:         Data output from the second SPAD
///    second_spad_en_o:           Enable signal for the second SPAD
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
    input clk_i,
    input rst_ni,
    input compute_i,
    //input                                         data_mode, Insert later

    input [DATA_WIDTH-1 : 0] data_i,
    input                    enable_i,

    output reg [ $clog2(FIRST_SPAD_ADDR+1)-1 : 0] first_spad_words_o,
    input      [   $clog2(FIRST_SPAD_ADDR)-1 : 0] first_spad_max_i,
    output reg [$clog2(SECOND_SPAD_ADDR+1)-1 : 0] second_spad_words_o,

    output reg [FIRST_SPAD_ADDR_BITWIDTH-1 : 0] first_spad_addr_o,
    output reg [         FIRST_SPAD_DATA-1 : 0] first_spad_data_o,
    output reg                                  first_spad_en_o,

    output reg [SECOND_SPAD_ADDR_BITWIDTH-1 : 0] second_spad_addr_o,
    output     [         SECOND_SPAD_DATA-1 : 0] second_spad_data_o,
    output reg                                   second_spad_en_o
);

  reg  [            FIRST_SPAD_DATA-1 : 0] data_storage_1;  // Temporary storage for data
  reg  [                 DATA_WIDTH-1 : 0] data_storage_2;  // Temporary storage for data
  reg  [     $clog2(SECOND_SPAD_ADDR) : 0] address_temp_2;  // Temporary address storage
  reg  [$clog2(FIRST_SPAD_DATA_CYCLE) : 0] cycle_counter;  // Cycle counter for data loading
  reg  [      SECOND_OVERHEAD_WIDTH-1 : 0] overhead_reg;
  reg  [      SECOND_OVERHEAD_WIDTH-1 : 0] overhead_delay_reg;
  reg  [       SECOND_PAYLOAD_WIDTH-1 : 0] payload_reg;
  wire [                 DATA_WIDTH-1 : 0] current_data;

  assign current_data = data_storage_2 >> SECOND_SPAD_DATA;
  assign second_spad_data_o = {
    overhead_delay_reg, payload_reg[SECOND_PAYLOAD_WIDTH-1 : 0]
  };  // Assign data to the second SPAD output



  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin  // Reset
      first_spad_words_o  <= 0;
      second_spad_words_o <= 0;
      first_spad_data_o   <= 0;
      first_spad_addr_o   <= 0;
      first_spad_en_o     <= 0;
      second_spad_addr_o  <= 0;
      second_spad_en_o    <= 0;
      data_storage_1      <= 0;
      data_storage_2      <= 0;
      address_temp_2      <= 0;
      cycle_counter       <= 0;
      payload_reg         <= 0;
      overhead_reg        <= 0;
      overhead_delay_reg  <= 0;
    end else begin
      first_spad_en_o   <= 0;
      first_spad_data_o <= 0;
      if (enable_i == 1) begin
        first_spad_en_o     <= 1;
        second_spad_en_o    <= 1;
        second_spad_addr_o  <= address_temp_2[SECOND_SPAD_ADDR_BITWIDTH-1:0];
        //second_spad_words_o <= ($clog2(SECOND_SPAD_ADDR+1))'(32'(second_spad_addr_o) + 2);
        second_spad_words_o <= second_spad_addr_o + 2;

        address_temp_2      <= address_temp_2 + 1;
        cycle_counter       <= cycle_counter + 1;

        payload_reg         <= current_data[SECOND_PAYLOAD_WIDTH-1 : 0];
        data_storage_2      <= current_data;
        first_spad_data_o   <= overhead_reg + 1'd1;
        overhead_reg        <= overhead_reg + 1;
        overhead_delay_reg  <= overhead_reg;

        if (cycle_counter == 0) begin
          //payload_reg       <= SECOND_PAYLOAD_WIDTH'(data_i);
          payload_reg       <= data_i[SECOND_PAYLOAD_WIDTH-1 : 0];
          data_storage_2    <= data_i;
          first_spad_data_o <= overhead_reg + 1;
        end
        if (cycle_counter == SECOND_SPAD_DATA_CYCLE[$clog2(FIRST_SPAD_DATA_CYCLE):0] - 1) begin
          cycle_counter <= 0;
        end
        if (address_temp_2 == SECOND_SPAD_ADDR - 1) begin
          address_temp_2 <= 0;
          cycle_counter  <= 0;
        end
        if (first_spad_data_o >= (data_storage_1 + first_spad_max_i)) begin
          data_storage_1    <= data_storage_1 + first_spad_max_i;
          first_spad_addr_o <= first_spad_addr_o + 1;
        end
      end else begin
        second_spad_en_o   <= 0;
        second_spad_addr_o <= 0;
        if (second_spad_addr_o != 0) begin
          first_spad_words_o <= first_spad_addr_o + 1;
        end
      end
      if (compute_i) begin  // Reset SPAD addresses and state of module
        first_spad_en_o     <= 0;
        second_spad_en_o    <= 0;
        data_storage_1      <= 0;
        data_storage_2      <= 0;
        first_spad_data_o   <= 0;
        address_temp_2      <= 0;
        cycle_counter       <= 0;
        first_spad_addr_o   <= 0;
        second_spad_words_o <= 0;
        payload_reg         <= 0;
        overhead_reg        <= 0;
        overhead_reg        <= 0;
      end
    end
  end

endmodule
