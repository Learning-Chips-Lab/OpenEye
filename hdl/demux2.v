// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: demux2
///
/// A parameterized 1-to-2 demultiplexer for the OpenEye neural network accelerator.
/// Routes input data to one of two outputs based on a selection signal.
///
/// Description:
///   This module implements a simple but efficient 1-to-2 demultiplexer with
///   configurable data width. It uses combinational logic to route the input
///   to either output based on the selection signal, with the unused output
///   being driven to zero.
///
/// Operation:
///   sel_i = 1: a_out = i, b_out = 0
///   sel_i = 0: a_out = 0, b_out = i
///
/// Parameters:
///   DATA_WIDTH - Width of input and output data buses (default: 1)
///               Supports arbitrary width for flexible data routing
///
/// Ports:
///   Outputs:
///     a_out[DATA_WIDTH-1:0] - First output channel
///                             Active when sel_i = 1
///     b_out[DATA_WIDTH-1:0] - Second output channel
///                             Active when sel_i = 0
///
///   Inputs:
///     sel_i                 - Selection control signal
///                            1: Route to a_out
///                            0: Route to b_out
///     i[DATA_WIDTH-1:0]    - Input data bus
///
/// Implementation Notes:
///   - Pure combinational logic implementation
///   - No clock or reset required
///   - Zero latency operation
///   - Inactive output is explicitly zeroed
///   - Uses efficient bitwise operations
///   - Supports any positive data width
///   - No internal state or registers
///

module demux2 #(
    parameter DATA_WIDTH = 1
) (
    output [DATA_WIDTH-1:0] a_out,
    output [DATA_WIDTH-1:0] b_out,
    input                   sel_i,
    input  [DATA_WIDTH-1:0] i
);

  assign a_out = sel_i & (i);
  assign b_out = (!sel_i) & (i);

endmodule
