// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: bano_cluster
///
/// The (ba)tch(no)rmalization_cluster is a modul for realizing the batch normalization by binary
/// shifting the input data. The ready and data signal gets passed on.
///
/// Parameters:
///    DATA_BITWIDTH - Bitwidth of data words
///   
/// Ports:
///    enable_i  - Enable Port, just gets delayed to output
///    enable_o  - Output for Enable signal, signals valid data
///    ready_i   - Ready signal, just gets delayed to output
///    ready_o   - Output for ready signal
///    data_i    - Data Port In, consists of two data packages
///    data_o    - Data Port Out, consists of two data packages
///

module bano_cluster 
#(
  parameter          SERIAL          = 1'b1,
  parameter  integer PARALLEL_MACS   = 2,
  parameter  integer DATA_BITWIDTH   = 20,
  parameter  integer BN_OFFSET_BITS  = 8,
  localparam integer NUM_DATA        = SERIAL ? 1 : PARALLEL_MACS
) (
  input  [BN_OFFSET_BITS-1:0]         bn_offset_i,
  output                              ready_o,
  input  [DATA_BITWIDTH-1:0] data_i,
  input                               enable_i,
  input                               ready_i,
  output [DATA_BITWIDTH-1:0] data_o,
  output                              enable_o

);

assign ready_o = ready_i;
assign enable_o = enable_i;
genvar data_pos;
for (data_pos = 0;data_pos < NUM_DATA; data_pos = data_pos + 1) begin
  assign data_o[(DATA_BITWIDTH*(1+data_pos))-1:DATA_BITWIDTH*data_pos] = (data_i[(DATA_BITWIDTH*(1+data_pos))-1:DATA_BITWIDTH*data_pos]>>bn_offset_i);
end


endmodule
