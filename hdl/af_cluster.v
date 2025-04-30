// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: af_cluster
///
/// The (a)ctivation(f)unction_cluster is a modul for creating the neccessary non-linear function.
/// Currently there is only ReLU implemented, which can be activated by setting mode_i to 1.
/// The ready and data signal gets passed on.
///
/// Parameters:
///    DATA_BITWIDTH - Bitwidth of data words
///    MODES      - Amount of MODES in this module, currently 2 (ReLU and nothing)
///   
/// Ports:
///    mode_i     - Mode, that chooses operation.
///    enable_i   - Enable Port, just gets delayed to output
///    enable_o   - Output for Enable signal, signals valid data
///    ready_i    - Ready signal, just gets delayed to output
///    ready_o    - Output for ready signal
///    data_i     - Data Port In, consists of two data packages
///    data_o     - Data Port Out, consists of two data packages
///

module af_cluster #(
    parameter integer ADVANCED_WIDTH = 64,
    parameter SERIAL = 1'b1,
    parameter integer PARALLEL_MACS = 2,
    parameter integer DATA_BITWIDTH = 20,
    parameter integer MODES = 4,
    parameter [ADVANCED_WIDTH-1:0] DIVISOR = {ADVANCED_WIDTH{1'b0}} + 35,
    parameter [ADVANCED_WIDTH-1:0] DIVIDEND = {{(ADVANCED_WIDTH - 32) {1'b0}}, 32'd3435973837}
) (
    input  [$clog2(MODES)-1:0] mode_i,
    output                     ready_o,
    input  [DATA_BITWIDTH-1:0] data_i,
    input                      enable_i,
    input                      ready_i,
    output [DATA_BITWIDTH-1:0] data_o,
    output                     enable_o
);
  localparam integer NUM_DATA = (SERIAL == 1) ? 1 : PARALLEL_MACS;
  localparam integer MODE_BITS = $clog2(MODES);

  wire [ DATA_BITWIDTH-2:0] psum                  [NUM_DATA-1:0];
  wire                      sign                  [NUM_DATA-1:0];
  wire [ DATA_BITWIDTH-1:0] data_out              [NUM_DATA-1:0];
  wire [ADVANCED_WIDTH-1:0] result_leaky          [NUM_DATA-1:0];
  wire [ DATA_BITWIDTH-1:0] truncated_result_leaky[NUM_DATA-1:0];

  genvar data_pos;
  for (data_pos = 0; data_pos < NUM_DATA; data_pos = data_pos + 1) begin
    assign result_leaky[data_pos]           = (psum[data_pos] * DIVIDEND) >> DIVISOR;
    assign truncated_result_leaky[data_pos] = result_leaky[data_pos][DATA_BITWIDTH-1:0];
    if (SERIAL == 1) begin : gen_serial
      assign sign[data_pos]                   = data_i[((1+data_pos) * DATA_BITWIDTH)-1]; //'data_i' gets split in two seperate data blocks
      assign psum[data_pos]                   = data_i[((1+data_pos) * DATA_BITWIDTH)-2 : data_pos * DATA_BITWIDTH]; //'data_i' gets split in two seperate data blocks
      assign data_o[(DATA_BITWIDTH*(1+data_pos))-1:DATA_BITWIDTH*data_pos] = data_out[data_pos]; // Concatenate both data blocks into one output
    end else begin : gen_parallel
      assign sign[data_pos]                   = data_i[((1+data_pos) * DATA_BITWIDTH/NUM_DATA)-1]; //'data_i' gets split in two seperate data blocks
      assign psum[data_pos]                   = data_i[((1+data_pos) * DATA_BITWIDTH/NUM_DATA)-2 : data_pos * DATA_BITWIDTH/NUM_DATA]; //'data_i' gets split in two seperate data blocks
      assign data_o[(DATA_BITWIDTH/NUM_DATA*(1+data_pos))-1:DATA_BITWIDTH/NUM_DATA*data_pos] = data_out[data_pos]; // Concatenate both data blocks into one output
    end
    assign data_out[data_pos] =
        (mode_i == 0) ? {sign[data_pos],psum[data_pos]} :
        (mode_i == 1) ? sign[data_pos] ? {DATA_BITWIDTH{1'b0}} : {1'b0,psum[data_pos]} :
        (mode_i == 2) ? (sign[data_pos] ? truncated_result_leaky[data_pos] : {1'b0, psum[data_pos]})  : //Approximation of 0.1 multiplier, if below 0
        {DATA_BITWIDTH{1'b0}};

  end

  assign ready_o  = ready_i;  // Pass on ready signal
  assign enable_o = enable_i;  // Pass on enable signal

endmodule
