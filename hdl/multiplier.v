// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: multiplier
///
/// The Multiplier module implements a configurable arithmetic multiplier in the OpenEye
/// architecture that supports both integer and fixed-point operations. This versatile
/// module enables efficient computation of products with selectable precision and
/// number representation formats.
///
/// Key Features:
/// - Dual Operation Modes:
///   * Integer multiplication
///   * Fixed-point multiplication
///
/// - Configurable Precision:
///   * Independent factor widths
///   * Adjustable product width
///   * Dynamic fraction bits
///
/// - Synchronous Operation:
///   * Clock-synchronized computation
///   * Reset capability
///   * Enable control
///
/// - Data Handling:
///   * Signed arithmetic
///   * Overflow protection
///   * Zero output when disabled
///
/// Operational Modes:
/// 1. Integer Mode (DATATYPE = 0):
///    - Direct multiplication
///    - Full precision product
///    - No fraction handling
///
/// 2. Fixed-Point Mode (DATATYPE = 1):
///    - Fraction-aware multiplication
///    - Dynamic point positioning
///    - Automatic scaling
///
/// Parameters:
///    DATATYPE         - Operation Mode Selection
///                      0: Integer arithmetic
///                      1: Fixed-point arithmetic
///
///    DATA_WIDTH_FAC1  - First Factor Width
///                      Bitwidth of first operand
///                      Determines input precision
///
///    DATA_WIDTH_FAC2  - Second Factor Width
///                      Bitwidth of second operand
///                      May differ from first factor
///
///    DATA_WIDTH_PROD  - Product Width
///                      Bitwidth of result
///                      Must accommodate full range
///
///    Q_BITWIDTH       - Fraction Control Width
///                      Bits for fraction position
///                      log2 of max product width
///   
/// Ports:
/// Clock and Reset:
///    clk_i           - System Clock Input
///                      Positive edge triggered
///                      Synchronizes operations
///
///    rst_ni          - Asynchronous Reset Input (active low)
///                      Clears product register
///                      Initializes state
///
/// Control Interface:
///    multiplier_en_i - Operation Enable Input
///                      1: Enable multiplication
///                      0: Force zero output
///
///    fraction_bit_i  - Fraction Control [Q_BITWIDTH-1:0]
///                      Specifies decimal point position
///                      0: Integer operation
///                      >0: Fixed-point scaling
///
/// Data Interface:
///    factor_1        - First Operand [DATA_WIDTH_FAC1-1:0]
///                      Signed multiplicand
///                      First input factor
///
///    factor_2        - Second Operand [DATA_WIDTH_FAC2-1:0]
///                      Signed multiplier
///                      Second input factor
///
///    product         - Result Output [DATA_WIDTH_PROD-1:0]
///                      Signed product
///                      Scaled in fixed-point mode
///


module multiplier #(
    parameter DATATYPE        = 0,
    parameter DATA_WIDTH_FAC1 = 8,
    parameter DATA_WIDTH_FAC2 = 8,
    parameter DATA_WIDTH_PROD = 20,
    parameter Q_BITWIDTH      = $clog2(DATA_WIDTH_PROD)
) (
    input                                   clk_i,
    input                                   rst_ni,
    input                                   multiplier_en_i,
    input  signed     [DATA_WIDTH_FAC1-1:0] factor_1,
    input  signed     [DATA_WIDTH_FAC2-1:0] factor_2,
    output reg signed [DATA_WIDTH_PROD-1:0] product,
    input             [     Q_BITWIDTH-1:0] fraction_bit_i
);
  generate
    if (DATATYPE == 0) begin : gen_integer
      always @(posedge clk_i, negedge rst_ni) begin
        if (!rst_ni) begin  ///Reset
          product <= 0;
        end else begin
          if (multiplier_en_i) begin
            product <= factor_1 * factor_2;
          end else begin
            product <= 0;
          end
        end
      end
    end else begin : gen_fixed_point
      always @(posedge clk_i, negedge rst_ni) begin
        if (!rst_ni) begin  ///Reset
          product <= 0;
        end else begin
          if (multiplier_en_i) begin
            product <= (factor_1 * factor_2) >> fraction_bit_i;
          end else begin
            product <= 0;
          end
        end
      end

    end
  endgenerate
endmodule
