// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: af_cluster
///
/// The Activation Function Cluster implements configurable non-linear activation functions
/// for the OpenEye neural network accelerator. It supports multiple operation modes including
/// pass-through, ReLU (Rectified Linear Unit), and LeakyReLU with configurable slope.
///
/// Description:
///   This module processes input data through selectable activation functions, supporting
///   both serial and parallel operation modes. It features parameterized bit widths and
///   configurable LeakyReLU parameters for flexible neural network implementations.
///
/// Operation Modes (mode_i):
///   0: Pass-through - Data passes unchanged
///   1: ReLU - max(0, x)
///   2: LeakyReLU - max(0.1x, x)
///   3: Zero - Output is forced to 0
///
/// Parameters:
///   ADVANCED_WIDTH  - Extended precision for LeakyReLU calculations (default: 64)
///   SERIAL         - Operation mode selection (1: serial, 0: parallel)
///   PARALLEL_MACS  - Number of parallel MAC units (default: 2)
///   DATA_BITWIDTH  - Width of input/output data words
///   MODES         - Number of supported activation modes (default: 4)
///   DIVISOR       - LeakyReLU slope divisor parameter (default: 35)
///   DIVIDEND      - LeakyReLU slope dividend parameter (default: 3435973837)
///
/// Ports:
///   mode_i[MODE_BITS-1:0] - Activation function selection
///   enable_i              - Input data valid signal
///   enable_o             - Output data valid signal (delayed enable_i)
///   ready_i              - Upstream ready signal
///   ready_o             - Downstream ready signal (mirrors ready_i)
///   data_i[DATA_BITWIDTH-1:0] - Input data vector
///   data_o[DATA_BITWIDTH-1:0] - Output data vector
///
/// Implementation Notes:
///   - Supports both serial and parallel data processing
///   - LeakyReLU uses fixed-point multiplication for slope calculation
///   - All operations are combinatorial with registered handshaking signals
///   - Data width is configurable through parameters
///   - Sign bit handling varies by activation function
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
