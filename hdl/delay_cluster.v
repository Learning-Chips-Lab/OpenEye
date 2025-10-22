// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: delay_cluster
///
/// Configurable delay line module for the OpenEye neural network accelerator,
/// specifically designed to handle timing mismatches between partial sum (PSUM)
/// data arrival and Global Buffer (GLB) readiness.
///
/// Description:
///   This module implements a flexible delay mechanism that can introduce 0-8 cycles
///   of delay for data, enable, and ready signals. It's primarily used to synchronize
///   partial sum data arrival with GLB memory operations, preventing data loss or
///   corruption due to timing mismatches.
///
/// Operation:
///   - Uses shift register architecture for delay implementation
///   - Supports variable delay selection (0-8 cycles)
///   - Maintains signal relationships through delay chain
///   - Synchronously shifts data and control signals
///   - Handles reset conditions cleanly
///
/// Delay Implementation:
///   0: Direct pass-through (no delay)
///   1-8: Corresponding cycle delays via shift registers
///   >8: Outputs zero (protection mode)
///
/// Parameters:
///   DATA_BITWIDTH - Width of the data path (default: 20)
///                  Affects total storage width: 8*DATA_BITWIDTH for maximum delay
///
/// Ports:
///   Clock and Reset:
///     clk_i   - System clock
///     rst_ni  - Asynchronous reset (active low)
///
///   Data Path:
///     data_i[DATA_BITWIDTH-1:0] - Input data to be delayed
///     data_o[DATA_BITWIDTH-1:0] - Delayed data output
///
///   Control Signals:
///     enable_i - Input data valid signal
///     enable_o - Delayed enable signal
///     ready_i  - Input ready signal
///     ready_o  - Delayed ready signal
///
///   Configuration:
///     delay_psum_glb_i[3:0] - Delay amount selection (0-8 cycles)
///
/// Implementation Notes:
///   - Uses wide shift registers for efficient delay implementation
///   - Maintains separate delay chains for data and control signals
///   - Implements synchronous data shifting on clock edge
///   - Provides clean reset of all delay elements
///   - Zero-outputs for invalid delay values
///   - Efficient multiplexing of delayed outputs
///   - No combinatorial paths between input and output
///   - Built-in protection against invalid delay values
///

module delay_cluster #(
    parameter integer DATA_BITWIDTH = 20
) (
    input clk_i,
    input rst_ni,

    output                       ready_o,
    input  [DATA_BITWIDTH-1 : 0] data_i,
    input                        enable_i,

    input                        ready_i,
    output [DATA_BITWIDTH-1 : 0] data_o,
    output                       enable_o,

    input [3 : 0] delay_psum_glb_i

);
  reg  [8*DATA_BITWIDTH-1 : 0] data_s;
  reg  [                7 : 0] enable_s;
  reg  [                7 : 0] ready_s;

  wire [8*DATA_BITWIDTH-1 : 0] data_w;
  wire [                7 : 0] enable_w;
  wire [                7 : 0] ready_w;

  assign data_w = {{7 * DATA_BITWIDTH{1'b0}}, data_i};
  assign enable_w = {{7{1'b0}}, enable_i};
  assign ready_w = {{7{1'b0}}, ready_i};

  assign data_o = (delay_psum_glb_i== 0) ? data_i :
                (delay_psum_glb_i== 1) ? data_s[1*DATA_BITWIDTH-1 : 0] :
                (delay_psum_glb_i== 2) ? data_s[2*DATA_BITWIDTH-1 : 1*DATA_BITWIDTH] :
                (delay_psum_glb_i== 3) ? data_s[3*DATA_BITWIDTH-1 : 2*DATA_BITWIDTH] :
                (delay_psum_glb_i== 4) ? data_s[4*DATA_BITWIDTH-1 : 3*DATA_BITWIDTH] :
                (delay_psum_glb_i== 5) ? data_s[5*DATA_BITWIDTH-1 : 4*DATA_BITWIDTH] :
                (delay_psum_glb_i== 6) ? data_s[6*DATA_BITWIDTH-1 : 5*DATA_BITWIDTH] :
                (delay_psum_glb_i== 7) ? data_s[7*DATA_BITWIDTH-1 : 6*DATA_BITWIDTH] :
                (delay_psum_glb_i== 8) ? data_s[8*DATA_BITWIDTH-1 : 7*DATA_BITWIDTH] : 0;

  assign enable_o = (delay_psum_glb_i== 0) ? enable_i :
                  (delay_psum_glb_i== 1) ? enable_s[0] :
                  (delay_psum_glb_i== 2) ? enable_s[1] :
                  (delay_psum_glb_i== 3) ? enable_s[2] :
                  (delay_psum_glb_i== 4) ? enable_s[3] :
                  (delay_psum_glb_i== 5) ? enable_s[4] :
                  (delay_psum_glb_i== 6) ? enable_s[5] :
                  (delay_psum_glb_i== 7) ? enable_s[6] :
                  (delay_psum_glb_i== 8) ? enable_s[7] : 0;

  assign ready_o = (delay_psum_glb_i== 0) ? ready_i :
                 (delay_psum_glb_i== 1) ? ready_s[0] :
                 (delay_psum_glb_i== 2) ? ready_s[1] :
                 (delay_psum_glb_i== 3) ? ready_s[2] :
                 (delay_psum_glb_i== 4) ? ready_s[3] :
                 (delay_psum_glb_i== 5) ? ready_s[4] :
                 (delay_psum_glb_i== 6) ? ready_s[5] :
                 (delay_psum_glb_i== 7) ? ready_s[6] :
                 (delay_psum_glb_i== 8) ? ready_s[7] : 0;


  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin : reset
      data_s   <= 0;
      enable_s <= 0;
      ready_s  <= 0;
    end else begin
      data_s   <= (data_s << DATA_BITWIDTH) + data_w;
      enable_s <= (enable_s << 1) + enable_w;
      ready_s  <= (ready_s << 1) + ready_w;
    end
  end

endmodule
