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

module af_cluster 
#( 
  parameter integer            DATA_BITWIDTH   = 40,
  parameter integer            MODES           = 2
) (
  input                          clk_i,
  input                          rst_ni,
  input  [$clog2(MODES)-1 : 0]   mode_i,

  output                         ready_o,
  input  [DATA_BITWIDTH-1:0]     data_i,
  input                          enable_i,

  input                          ready_i,
  output [DATA_BITWIDTH-1:0]     data_o,
  output                         enable_o

);

  localparam integer HALF_DATA_BITS = DATA_BITWIDTH/2;

wire [HALF_DATA_BITS-2:0] ms_psum;
wire [HALF_DATA_BITS-2:0] ls_psum;
wire                      ms_sign;
wire                      ls_sign;

wire [HALF_DATA_BITS-1:0] ms_data_out;
wire [HALF_DATA_BITS-1:0] ls_data_out;
assign {ms_sign,ms_psum,ls_sign,ls_psum} = data_i; //'data_i' gets split in two seperate data blocks

// If `mode_i`is set, use ReLU operator by reading `ms_sign`and `ls_sign`
assign ms_data_out = mode_i ? (ms_sign ? 0 : {1'b0,ms_psum}) : {ms_sign,ms_psum};
assign ls_data_out = mode_i ? (ls_sign ? 0 : {1'b0,ls_psum}) : {ls_sign,ls_psum};

assign ready_o = ready_i; // Pass on ready signal
assign data_o = {ms_data_out,ls_data_out}; // Concatenate both data blocks into one output
assign enable_o = enable_i; // Pass on enable signal

endmodule
