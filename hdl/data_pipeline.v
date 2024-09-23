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

module data_pipeline 
#(
    parameter DATA_WIDTH                = 1,
    parameter FIRST_SPAD_ADDR           = 1,
    parameter FIRST_SPAD_DATA           = 1,
    parameter SECOND_SPAD_ADDR          = 1,
    parameter SECOND_SPAD_DATA          = 1,
    parameter FIRST_SPAD_ADDR_BITWIDTH  = $clog2(FIRST_SPAD_ADDR),
    parameter SECOND_SPAD_ADDR_BITWIDTH = $clog2(SECOND_SPAD_ADDR),
    parameter FIRST_SPAD_DATA_CYCLE     = DATA_WIDTH / FIRST_SPAD_DATA,
    parameter SECOND_SPAD_DATA_CYCLE    = DATA_WIDTH / SECOND_SPAD_DATA
) (
    input                                        clk_i,
    input                                        rst_ni,
    input                                        compute_i,

    input      [DATA_WIDTH-1 : 0]                data_i,
    input                                        enable_i,

    output reg [$clog2(FIRST_SPAD_ADDR+1)-1 : 0]  first_spad_words_o, 
    output reg [$clog2(SECOND_SPAD_ADDR+1)-1 : 0] second_spad_words_o, 

    output reg [FIRST_SPAD_ADDR_BITWIDTH-1 : 0]  first_spad_addr_o,
    output     [FIRST_SPAD_DATA-1 : 0]           first_spad_data_o,
    output reg                                   first_spad_en_o,

    output reg [SECOND_SPAD_ADDR_BITWIDTH-1 : 0] second_spad_addr_o,
    output     [SECOND_SPAD_DATA-1 : 0]          second_spad_data_o,
    output reg                                   second_spad_en_o
);

  enum bit [0:0] {  // Enum for Modes
    LOADING_ADDR     = 0,  // Mode to load addresses into SPAD
    LOADING_DATA     = 1   // Mode to load data into SPAD
  } fsm_mode;

  reg                                     current_state;  // Current state of the FSM
  reg                                     fsm_state;  // FSM state register
  reg [DATA_WIDTH-1 : 0]                  data_storage;  // Temporary storage for data
  reg [$clog2(SECOND_SPAD_ADDR) : 0]      address_temp;  // Temporary address storage
  reg [$clog2(FIRST_SPAD_DATA_CYCLE) : 0] cycle_counter;  // Cycle counter for data loading
  genvar i;
  for(i=0; i<FIRST_SPAD_DATA; i=i+1) begin
    assign first_spad_data_o[i] = data_storage[i];  // Assign data to the first SPAD output
  end

  for(i=0; i<SECOND_SPAD_DATA; i=i+1) begin
    assign second_spad_data_o[i] = data_storage[i];  // Assign data to the second SPAD output
  end

  always@(*) begin:fsm_state_comb
    fsm_state = current_state;  // Combinational logic to update FSM state
  end


  always@(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin // Reset
      current_state         <= LOADING_ADDR;
      first_spad_words_o    <= 0;
      second_spad_words_o   <= 0;
      first_spad_addr_o     <= 0;
      first_spad_en_o       <= 0;
      second_spad_addr_o    <= 0;
      second_spad_en_o      <= 0;
      data_storage          <= 0;
      address_temp          <= 0;
      cycle_counter         <= 0;
    end
    else begin
      case(fsm_state)
        LOADING_ADDR : begin // First state, start loading the first SPAD
          second_spad_en_o  <= 0;
          second_spad_addr_o <= 0;
          if (enable_i == 1) begin
            first_spad_en_o   <= 1;
            first_spad_addr_o <= address_temp[FIRST_SPAD_ADDR_BITWIDTH-1:0];
            address_temp      <= address_temp + 1;
            cycle_counter     <= cycle_counter + 1;
            data_storage      <= data_storage >> FIRST_SPAD_DATA;
            if (cycle_counter == 0) begin
              data_storage  <= data_i;
              if ((data_i == 0) & (first_spad_data_o != 0)) begin
                first_spad_words_o <= ($clog2(FIRST_SPAD_ADDR+1)-1)'(first_spad_addr_o) + 1;
                current_state      <= LOADING_DATA;
                address_temp       <= 0;
                cycle_counter      <= 0;
              end
            end
            else begin
              if (((data_storage >> FIRST_SPAD_DATA) == 0) & (first_spad_data_o != 0)) begin
                first_spad_words_o <= ($clog2(FIRST_SPAD_ADDR+1)-1)'(first_spad_addr_o) + 1;
                current_state      <= LOADING_DATA;
                address_temp       <= 0;
                cycle_counter      <= 0;
              end
            end

            if (cycle_counter == (FIRST_SPAD_DATA_CYCLE[$clog2(FIRST_SPAD_DATA_CYCLE):0]-1)) begin
              cycle_counter <= 0;
            end
            if (address_temp == FIRST_SPAD_ADDR - 1) begin
              first_spad_words_o <= ($clog2(FIRST_SPAD_ADDR+1)-1)'(first_spad_addr_o) + 1;
              current_state      <= LOADING_DATA;
              address_temp       <= 0;
              cycle_counter      <= 0;
            end
          end
          else begin
            first_spad_en_o   <= 0;
            first_spad_addr_o <= 0;
          end
        end
        LOADING_DATA : begin // Second state, start loading the second SPAD
          first_spad_en_o <= 0;
          first_spad_addr_o <= 0;
          if (enable_i == 1) begin
            second_spad_en_o    <= 1;
            second_spad_addr_o  <= address_temp[SECOND_SPAD_ADDR_BITWIDTH-1:0];
            second_spad_words_o <= ($clog2(SECOND_SPAD_ADDR+1))'(32'(second_spad_addr_o) + 2);

            address_temp        <= address_temp + 1;
            cycle_counter       <= cycle_counter + 1;

            data_storage        <= data_storage >> SECOND_SPAD_DATA;

            if (cycle_counter == 0) begin
              data_storage  <= data_i;
            end

            if (cycle_counter == SECOND_SPAD_DATA_CYCLE[$clog2(FIRST_SPAD_DATA_CYCLE):0] - 1) begin
              cycle_counter <= 0;
            end
            if (address_temp == SECOND_SPAD_ADDR - 1) begin
              if (enable_i == 1) begin
                current_state <= LOADING_ADDR;
              end
              address_temp    <= 0;
              cycle_counter   <= 0;
            end
          end else begin
            second_spad_en_o   <= 0;
            second_spad_addr_o <= 0;
          end
        end
        default : begin
          current_state <= LOADING_ADDR;  // Default state if none match
        end
      endcase
      if (compute_i) begin // Reset SPAD addresses and state of module
        current_state         <= LOADING_ADDR;
        data_storage          <= 0;
        address_temp          <= 0;
        cycle_counter         <= 0;
      end
    end
  end
    
endmodule
