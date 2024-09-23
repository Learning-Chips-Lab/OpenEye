// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: adder
///
/// A simple registered adder module.
///
/// The adder computes the sum of `summand_1_i` and `summand_2_i` when `adder_en_i` is high.
/// The result is stored in `sum_o` and is updated on the rising edge of `clk_i`.
/// The result is reset to 0 when `rst_ni` is low.
///
/// Parameters:
///   DATA_WIDTH_SUM:   Bitwidth of the sum output
///
/// Ports:
///   clk_i:            Clock input
///   rst_ni:           Active low reset input
///   adder_en_i:       Adder enable input
///   summand_1_i:      First summand input
///   summand_2_i:      Second summand input
///   sum_o:            Sum output
///

module adder 
#( 
  parameter DATA_WIDTH_SUM = 20
) (
  input                             clk_i,
  input                             rst_ni,
  input                             adder_en_i,
  input signed [DATA_WIDTH_SUM-1:0] summand_1_i,
  input signed [DATA_WIDTH_SUM-1:0] summand_2_i,
  output reg signed [DATA_WIDTH_SUM-1:0] sum_o
);

  always@(posedge clk_i, negedge rst_ni) begin
    if(!rst_ni) begin // Reset
      sum_o<= 0;
    end else begin
      if(adder_en_i) begin
        sum_o <= summand_1_i + summand_2_i;
      end else begin // Keep output, if `adder_en_i` is low
        sum_o <= sum_o;
      end
    end
  end
endmodule
