// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: multiplier
///
/// Multiplier, that takes two factors and calculates its product. Can be enabled via the port
/// `multiplier_en_i`. Fixed-Point Operations are enabled by using the port `fraction_bit_i`
///
/// Parameters:
///    DATA_WIDTH_FAC1  - Bitwidth of first factor
///    DATA_WIDTH_FAC2  - Bitwidth of second factor
///    DATA_WIDTH_PROD  - Bitwidth of product
///    Q_BITWIDTH       - Bitwidth of `fraction_bit_i`
///   
/// Ports:
///    multiplier_en_i - Enable Port, activates the multiplication operation
///    factor_1        - Data input, first factor
///    factor_2        - Data input, second factor
///    product         - Data output, product
///    fraction_bit_i  - Position of point, 0 means Integer Value
///


module multiplier 
#( 
  parameter DATA_WIDTH_FAC1 = 8,
  parameter DATA_WIDTH_FAC2 = 8,
  parameter DATA_WIDTH_PROD = 20,
  parameter Q_BITWIDTH      = $clog2(DATA_WIDTH_PROD)
) (
  input                                   clk_i,
  input                                   rst_ni,
  input                                   multiplier_en_i,
  input signed      [DATA_WIDTH_FAC1-1:0] factor_1,
  input signed      [DATA_WIDTH_FAC2-1:0] factor_2,
  output reg signed [DATA_WIDTH_PROD-1:0] product,
  input             [Q_BITWIDTH-1:0]      fraction_bit_i
);

  always@(posedge clk_i, negedge rst_ni) begin
    if(!rst_ni) begin ///Reset
      product <= 0;
    end else begin
      if(multiplier_en_i) begin
        product <= factor_1 * factor_2;
      end else begin
        product <= 0;
      end
    end
  end
endmodule
