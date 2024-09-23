// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: demux2
///
/// Demultiplexer, that takes one input and can select one from two outputs via the port `sel_i`
///
/// Parameters:
///    DATA_WIDTH  - Bitwidth of input and output data
///   
/// Ports:
///    a_out  - Outputs port `i`, if `sel_i` is 1, else outputs 0
///    b_out  - Outputs port `i`, if `sel_i` is 0, else outputs 0
///    sel_i  - Port for selection
///    i      - Input data port
///

module demux2 
#( 
  parameter DATA_WIDTH = 1
) (
  input                       clk_i,
  input                       rst_ni,
  output reg [DATA_WIDTH-1:0] a_out,
  output reg [DATA_WIDTH-1:0] b_out,
  input                       sel_i,
  input      [DATA_WIDTH-1:0] i
);

  always @(posedge clk_i, negedge rst_ni) begin
    //Reset
    if (!rst_ni) begin
      a_out = 0;
      b_out = 0;
    end else begin
      if (sel_i) begin 
        a_out = i;
        b_out = 0;
      end else begin
        a_out = 0;
        b_out = i;
      end
    end

  end

endmodule
