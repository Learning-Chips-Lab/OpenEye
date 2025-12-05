// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: bano_cluster
///
/// The Batch Normalization (BANO) Cluster implements efficient batch normalization
/// through binary shifting operations for the OpenEye neural network accelerator.
///
/// Description:
///   This module performs simplified batch normalization by right-shifting input data
///   by a configurable amount. This approach provides an efficient hardware implementation
///   of scaling operations commonly used in neural networks. The module supports both
///   serial and parallel data processing modes.
///
/// Operation:
///   - Performs right shift: data_o = data_i >> bn_offset_i
///   - Supports multiple parallel data streams in parallel mode
///   - Maintains data validity through enable/ready handshaking
///
/// Parameters:
///   SERIAL          - Operation mode (1: serial, 0: parallel)
///   PARALLEL_MACS   - Number of parallel MAC units (default: 2)
///   DATA_BITWIDTH   - Width of input/output data words
///   BN_OFFSET_BITS  - Bit width of the shift amount control (default: 8)
///
/// Ports:
///   bn_offset_i[BN_OFFSET_BITS-1:0] - Shift amount control input
///   enable_i                        - Input data valid signal
///   enable_o                       - Output data valid signal (mirrors enable_i)
///   ready_i                        - Upstream ready signal
///   ready_o                       - Downstream ready signal (mirrors ready_i)
///   data_i[DATA_BITWIDTH-1:0]      - Input data vector
///   data_o[DATA_BITWIDTH-1:0]      - Shifted output data vector
///
/// Implementation Notes:
///   - Purely combinatorial data path
///   - Data width remains constant (no precision loss beyond shifting)
///   - In parallel mode, processes NUM_DATA parallel streams
///   - NUM_DATA is determined by SERIAL and PARALLEL_MACS parameters
///   - Handshaking signals (ready/enable) are passed through without modification
///   - All operations are performed on signed data
///

module bano_cluster #(
    parameter         SERIAL         = 1'b1,
    parameter integer PARALLEL_MACS  = 2,
    parameter integer DATA_BITWIDTH  = 20,
    parameter integer BN_OFFSET_BITS = 8
) (
    input  [BN_OFFSET_BITS-1:0] bn_offset_i,
    output                      ready_o,
    input  [ DATA_BITWIDTH-1:0] data_i,
    input                       enable_i,
    input                       ready_i,
    output [ DATA_BITWIDTH-1:0] data_o,
    output                      enable_o

);
  localparam integer NUM_DATA = SERIAL ? 1 : PARALLEL_MACS;

  assign ready_o  = ready_i;
  assign enable_o = enable_i;
  genvar data_pos;
  for (data_pos = 0; data_pos < NUM_DATA; data_pos = data_pos + 1) begin
    assign data_o[(DATA_BITWIDTH*(1+data_pos))-1:DATA_BITWIDTH*data_pos] = (data_i[(DATA_BITWIDTH*(1+data_pos))-1:DATA_BITWIDTH*data_pos]>>bn_offset_i);
  end
endmodule
