// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: mux2
///
/// Multiplexer that takes two inputs and can select one and connect it to an output,
/// depending on port `sel_i`.
///
/// Parameters:
///    DATA_WIDTH  - Width of data used by data ports
///   
/// Ports:
///    a_in  - Inputs port `i`, if `sel_i` is 1
///    b_in  - Inputs port `i`, if `sel_i` is 0
///    sel_i - Port for selection
///    y_o   - Output data port
////
module mux2 
#( 
  parameter DATA_WIDTH = 20
) (
  input [DATA_WIDTH-1:0] a_in,
  input [DATA_WIDTH-1:0] b_in,
  input                  sel_i,
  output[DATA_WIDTH-1:0] y_o
);

  assign y_o = sel_i ? a_in : b_in;

endmodule
