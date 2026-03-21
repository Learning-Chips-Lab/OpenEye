// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: RAM_SP_generic
///
/// A synthesizable, technology-independent implementation of a single-port RAM module.
/// This module provides a generic RTL description that can be mapped to standard cells
/// when dedicated SRAM macros are not available or not desired. It supports both
/// combinational and registered read paths through configuration.
///
/// Architecture Overview:
/// - Memory Organization:
///   * True single-port architecture
///   * Configurable depth and width
///   * Single address space for reads/writes
///
/// - Access Modes:
///   * Synchronous write operations
///   * Configurable read path (combinational/registered)
///   * Mutually exclusive read/write control
///
/// - Implementation Features:
///   * Synthesizable RTL description
///   * Standard cell mapping compatible
///   * Power optimization support
///   * Configurable timing paths
///
/// Parameters:
/// Memory Configuration:
///    AddrWidth        - Address width in bits
///                       Determines memory depth as 2^AddrWidth words
///    DataWidth        - Data word width in bits
///                       Defines the size of each memory location
///
/// Timing Control:
///    Pipelined        - Output register enable
///                       0: Combinational read path
///                       1: Registered read output (improved timing)
///
/// Derived Parameters:
///    Depth            - Total memory depth (2^AddrWidth)
///                       Automatically calculated
///
/// Ports:
/// Clock Interface:
///    clk             - System clock input
///                      All operations synchronized to rising edge
///
/// Control Interface:
///    cen             - Clock/chip enable (active low)
///                      Controls memory array access
///    rdwen           - Read/write control (active low for write)
///                      1: Read operation
///                      0: Write operation
///
/// Data Interface:
///    a               - Address input [AddrWidth-1:0]
///                      Specifies target memory location
///    d               - Write data input [DataWidth-1:0]
///                      Data to be written during write operations
///    q               - Read data output [DataWidth-1:0]
///                      Data read from memory (timing based on Pipelined)
///

module RAM_SP_generic #(
    parameter AddrWidth = 12,
    parameter DataWidth = 8,
    parameter Pipelined = 0
) (
    input  wire                 clk,
    input  wire                 cen,
    input  wire                 rdwen,
    input  wire [AddrWidth-1:0] a,
    input  wire [DataWidth-1:0] d,
    output reg  [DataWidth-1:0] q
);
  localparam Depth = 2 ** AddrWidth;

  reg [DataWidth-1:0] mem    [0:Depth-1];
  reg [DataWidth-1:0] memout;

  generate
    if (Pipelined) begin : gen_pipelined
      always @(posedge clk) q <= memout;
    end else begin : gen_not_pipelined
      always @* q = memout;
    end
  endgenerate

  always @(posedge clk) begin
    if (!cen && !rdwen) begin
      mem[a] <= d;
    end
  end

  always @(posedge clk) begin
    if (!cen && rdwen) begin
      memout <= mem[a];
    end
  end

  /// Implementation Notes:
  /// Memory Architecture:
  ///   - Storage Array:
  ///     * Two-dimensional array of registers
  ///     * Width: DataWidth bits
  ///     * Depth: 2^AddrWidth words
  ///     * Packed array structure for synthesis
  ///
  ///   - Read Path:
  ///     * Two-stage design
  ///     * Internal read register (memout)
  ///     * Optional output register (q)
  ///     * Configurable timing path
  ///
  /// Access Control:
  ///   - Write Operations:
  ///     * Synchronous write on clock edge
  ///     * Requires cen and rdwen both low
  ///     * Single cycle write latency
  ///     * No write mask/byte enable
  ///
  ///   - Read Operations:
  ///     * Initiated when cen low and rdwen high
  ///     * One or two cycle latency based on Pipelined
  ///     * Read-during-write returns old data
  ///
  /// Timing Considerations:
  ///   - Setup Requirements:
  ///     * Address stable before clock edge
  ///     * Data stable before clock edge
  ///     * Control signals stable before clock edge
  ///
  ///   - Clock-to-Output:
  ///     * Combinational mode: Through two levels of logic
  ///     * Pipelined mode: Registered output path
  ///
  /// Power Optimization:
  ///   - Clock Gating:
  ///     * Natural gating through cen
  ///     * Separate read/write power domains
  ///
  ///   - Memory Access:
  ///     * Single port reduces dynamic power
  ///     * Inactive state when cen high
  ///
  /// Synthesis Guidelines:
  ///   - Array Inference:
  ///     * Memory array should infer registers
  ///     * Tools may apply optimization
  ///     * May map to RAM primitives
  ///
  ///   - Timing Optimization:
  ///     * Use Pipelined=1 for timing closure
  ///     * Consider floorplanning impact
  ///     * Address setup time critical
  ///
  /// Verification Notes:
  ///   - Initial State:
  ///     * Memory contents undefined at start
  ///     * No built-in initialization
  ///     * System must handle reset state
  ///
  ///   - Access Checking:
  ///     * No bounds checking on address
  ///     * No protection against X propagation
  ///     * System must ensure valid addresses
  ///
endmodule
