// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: mux_iact
///
/// mux_iact is a module that essentially works like a mux, but for three different data paths.
/// Port A (`a_i` and `a_o`) contains data of a given width, while ports B and C contain control
/// Signals
///
/// Parameters:
///    WIDTH    - Bitwidth of data port A
///    I_COUNT  - Amount of different data paths
///   
/// Ports:
///    a_i   - Data input A, packed array of 3
///    b_i   - Data input B, packed array of 3
///    c_o   - Data output C, packed array of 3
///    sel_i - Port for selecting one port of packed array
///    a_o   - Data input A
///    b_o   - Data input B
///    c_i   - Data output C
///

module mux_iact 
#( 
    parameter WIDTH = 20,
    parameter I_COUNT = 3
) (
    input      [WIDTH*I_COUNT-1:0]  a_i,
    input      [I_COUNT-1:0]        b_i,
    output reg [I_COUNT-1:0]        c_o,
    input      [$clog2(I_COUNT+1)-1:0]sel_i,
    output reg [WIDTH-1:0]          a_o,
    output reg                      b_o,
    input                           c_i
);
integer j;
genvar i;

  wire [WIDTH-1:0] a_w [0:(I_COUNT)-1];
  for(i=0; i<I_COUNT; i=i+1)begin
    assign a_w[i] = a_i[(WIDTH*(i+1))-1:(WIDTH*i)];
  end
  always @(*) begin : configure_mux
    a_o = a_w[$clog2(I_COUNT)'(sel_i)];
    b_o = b_i[$clog2(I_COUNT)'(sel_i)];
    if(sel_i == I_COUNT)begin
      a_o = 0;
      b_o = 0;
    end
    for(j=0; j<I_COUNT; j=j+1)begin
      c_o[j] = c_i;
    end
  end

endmodule
