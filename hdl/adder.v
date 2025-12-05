// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: adder
/// 
/// A synchronous, registered adder module with enable control and reset functionality.
/// This module is part of the OpenEye neural network accelerator's arithmetic units.
///
/// Description:
///   The adder performs signed addition of two input operands with configurable bit width.
///   It includes synchronous operation with clock edge triggering, active-low reset,
///   and an enable signal for operation control. The output is registered and updated
///   on the positive clock edge when enabled.
///
/// Operation:
///   - When enabled (adder_en_i = 1): sum_o = summand_1_i + summand_2_i
///   - When disabled (adder_en_i = 0): sum_o = 0
///   - On reset (rst_ni = 0): sum_o = RESET_VALUE
///
/// Parameters:
///   DATA_WIDTH_SUM:   Bitwidth of the summands and sum (default: 20)
///   RESET_VALUE:      Value loaded into sum_o during reset (default: 0)
///
/// Ports:
///   clk_i:            Clock input, positive edge triggered
///   rst_ni:           Active low asynchronous reset
///   adder_en_i:       Adder enable signal (1: active, 0: reset output to 0)
///   summand_1_i:      First signed input operand [DATA_WIDTH_SUM-1:0]
///   summand_2_i:      Second signed input operand [DATA_WIDTH_SUM-1:0]
///   sum_o:            Registered signed sum output [DATA_WIDTH_SUM-1:0]
///
/// Timing:
///   - All outputs are registered and updated on the positive edge of clk_i
///   - Reset is asynchronous and active-low
///   - One clock cycle latency from input to output when enabled
///

module adder #(
    parameter DATA_WIDTH_SUM = 20,
    parameter RESET_VALUE = 0
) (
    input                                  clk_i,
    input                                  rst_ni,
    input                                  adder_en_i,
    input  signed     [DATA_WIDTH_SUM-1:0] summand_1_i,
    input  signed     [DATA_WIDTH_SUM-1:0] summand_2_i,
    output reg signed [DATA_WIDTH_SUM-1:0] sum_o
);

  always @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin  // Reset
      sum_o <= RESET_VALUE;
    end else begin
      if (adder_en_i) begin
        sum_o <= summand_1_i + summand_2_i;
      end else begin  // Set to 0, if `adder_en_i` is low
        sum_o <= 0;
      end
    end
  end
endmodule
