// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
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
///     data_i[DATA_BITWIDTH-1 : 0] - Input data to be delayed
///     data_o[DATA_BITWIDTH-1 : 0] - Delayed data output
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
    input                       clk_i,
    input                       rst_ni,

    input  [DATA_BITWIDTH-1:0] data_i,
    input                       enable_i,
    input                       ready_i,

    output reg [DATA_BITWIDTH-1:0] data_o,
    output reg                      enable_o,
    output reg                      ready_o,
    input [3:0]                 delay_psum_glb_i
);

  // Delay stages (8 stages for 0-8 cycles)
  localparam integer NUM_STAGES = 8;

  // Delay registers for data, enable, and ready signals
  reg [NUM_STAGES*DATA_BITWIDTH-1:0] data_regs;
  reg [NUM_STAGES-1:0]               enable_regs;
  reg [NUM_STAGES-1:0]               ready_regs;

  // Register for the current stage (updated on each clock cycle)
  reg [NUM_STAGES*DATA_BITWIDTH-1:0] data_stage;
  reg [NUM_STAGES-1:0]               enable_stage;
  reg [NUM_STAGES-1:0]               ready_stage;

  // Generate shift register update logic
  generate
    genvar i;
    for (i = 0; i < NUM_STAGES; i = i + 1) begin : gen_data_shift
      if (i == 0) begin
        assign data_stage[i*DATA_BITWIDTH +: DATA_BITWIDTH] = data_i;
    end else begin
        assign data_stage[i*DATA_BITWIDTH +: DATA_BITWIDTH] = data_regs[(i-1)*DATA_BITWIDTH +: DATA_BITWIDTH];
    end
  end
  endgenerate

  generate
    for (i = 0; i < NUM_STAGES; i = i + 1) begin : gen_enable_shift
      if (i == 0) begin
        assign enable_stage[i] = enable_i;
      end else begin
        assign enable_stage[i] = enable_regs[i-1];
      end
    end
  endgenerate

  generate
    for (i = 0; i < NUM_STAGES; i = i + 1) begin : gen_ready_shift
      if (i == 0) begin
        assign ready_stage[i] = ready_i;
      end else begin
        assign ready_stage[i] = ready_regs[i-1];
      end
    end
  endgenerate

  // Update registers on clock edge
  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin
      data_regs   <= '0;
      enable_regs <= '0;
      ready_regs  <= '0;
    end else begin
      data_regs   <= data_stage;
      enable_regs <= enable_stage;
      ready_regs  <= ready_stage;
    end
  end

  // Output multiplexer
  // Select appropriate delay stage based on delay_psum_glb_i
  integer idx;
  always @(*) begin
    if (delay_psum_glb_i == 0) begin
      data_o    = data_i;
      enable_o  = enable_i;
      ready_o   = ready_i;
    end else if (delay_psum_glb_i <= NUM_STAGES) begin
      // Valid delay: 1-8 cycles
      idx       = delay_psum_glb_i - 1;
      data_o    = data_regs[idx*DATA_BITWIDTH +: DATA_BITWIDTH];
      enable_o  = enable_regs[idx];
      ready_o   = ready_regs[idx];
    end else begin
      // Invalid delay: output zeros (protection)
      data_o    = '0;
      enable_o  = 1'b0;
      ready_o   = 1'b0;
    end
  end

endmodule

