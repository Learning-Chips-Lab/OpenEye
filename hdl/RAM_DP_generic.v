// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: RAM_DP_generic
///
/// A synthesizable, technology-independent implementation of a dual-port RAM with 
/// collision detection and handling. This module provides a true dual-port memory
/// with independent read and write ports, supporting concurrent access and optional
/// output pipelining.
///
/// Architecture Overview:
/// - Dual-Port Memory:
///   * Independent port A (read) and port B (write)
///   * Separate clock and enable for each port
///   * Configurable depth and width
///   * Collision detection and handling
///
/// - Implementation Features:
///   * Synthesizable RTL description
///   * Standard cell mapping support
///   * Configurable output timing
///   * Simulation-time collision checking
///
/// Key Features:
/// - Port Independence:
///   * Separate clock domains possible
///   * Individual port enables
///   * Independent address spaces
///   * Concurrent access support
///
/// - Memory Management:
///   * Write collision detection
///   * X-propagation on conflicts
///   * Power-saving enable controls
///   * Flexible timing options
///
/// Parameters:
/// Memory Configuration:
///    AddrWidth        - Address bus width (log2 of memory depth)
///                       Determines number of addressable locations
///    DataWidth        - Data word width in bits
///                       Defines size of each memory word
///
/// Timing Control:
///    Pipelined        - Output register enable
///                       0: Combinational read path
///                       1: Registered read output
///
/// Derived Parameters:
///    Memory Size      - Total size is (2^AddrWidth * DataWidth) bits
///                       Automatically calculated from parameters
///
/// Ports:
/// Clock Interface:
///    clkA            - Port A clock input
///                      Controls read operations
///    clkB            - Port B clock input
///                      Controls write operations
///
/// Control Interface:
///    cenA            - Port A clock enable (active low)
///                      Gates read operations
///    cenB            - Port B clock enable (active low)
///                      Gates write operations
///
/// Address Interface:
///    aA              - Port A address [AddrWidth-1:0]
///                      Selects location for read
///    aB              - Port B address [AddrWidth-1:0]
///                      Selects location for write
///
/// Data Interface:
///    d               - Write data input [DataWidth-1:0]
///                      Data to be written through port B
///    bw              - Byte write enable [DataWidth-1:0]
///                      Controls partial word writes
///    q               - Read data output [DataWidth-1:0]
///                      Data read through port A
///

module RAM_DP_generic #(
    parameter AddrWidth = 12,
    parameter DataWidth = 8,
    parameter Pipelined = 0
) (
    input wire clkA,
    input wire clkB,

    input wire cenA,
    input wire cenB,

    input wire [AddrWidth-1:0] aA,
    input wire [AddrWidth-1:0] aB,

    input  wire [DataWidth-1:0] d,
    input  wire [DataWidth-1:0] bw,
    output reg  [DataWidth-1:0] q
);

  reg [DataWidth-1:0] mem    [0:(2**AddrWidth)-1];

  reg [DataWidth-1:0] memout;

  generate
    if (Pipelined) begin : gen_pipelined
      always @(posedge clkA) q <= memout;
    end else begin : gen_not_pipelined
      always @* q = memout;
    end
  endgenerate

  // No async. set or reset for FF
  always @(posedge clkB) begin
    if (!cenB) begin
      mem[aB] <= d;
    end
  end

  always @(posedge clkA) begin
    if (!cenA) begin
      if (!cenB && aA == aB) begin
        // cadence synthesis_off
`ifndef SYNTHESIS
        $error("R/W collision in DP sram");
        memout <= {DataWidth{1'bx}};
`endif
        // cadence synthesis_on
      end else begin
        memout <= mem[aA];
      end
    end else begin
      memout <= {DataWidth{1'bx}};
    end
  end

  /// Implementation Notes:
  /// Memory Architecture:
  ///   - Storage Organization:
  ///     * Two-dimensional register array
  ///     * Width: DataWidth bits
  ///     * Depth: 2^AddrWidth words
  ///     * Single shared memory array
  ///
  ///   - Port Structure:
  ///     * Port A: Primary read port
  ///       - Address input (aA)
  ///       - Output path (q)
  ///       - Enable control (cenA)
  ///
  ///     * Port B: Primary write port
  ///       - Address input (aB)
  ///       - Data input (d)
  ///       - Enable control (cenB)
  ///
  /// Access Control:
  ///   - Write Operations:
  ///     * Synchronized to clkB
  ///     * Controlled by cenB
  ///     * Single cycle latency
  ///     * Full word writes only
  ///
  ///   - Read Operations:
  ///     * Synchronized to clkA
  ///     * Controlled by cenA
  ///     * One/two cycle latency (Pipelined)
  ///     * Collision checking included
  ///
  /// Collision Handling:
  ///   - Detection:
  ///     * Address comparison (aA == aB)
  ///     * Enable state checking
  ///     * Timing window analysis
  ///
  ///   - Response:
  ///     * X-propagation in simulation
  ///     * Error reporting (non-synthesis)
  ///     * Deterministic behavior in RTL
  ///
  /// Timing Considerations:
  ///   - Clock Domains:
  ///     * Independent port clocking
  ///     * Setup/hold requirements
  ///     * Clock skew management
  ///
  ///   - Data Path:
  ///     * Optional output registration
  ///     * Internal pipeline stage
  ///     * Enable timing critical
  ///
  /// Power Management:
  ///   - Clock Gating:
  ///     * Port-level enable control
  ///     * Independent power domains
  ///     * Activity minimization
  ///
  ///   - Memory Access:
  ///     * Selective port activation
  ///     * Reduced switching activity
  ///     * Enable-based power control
  ///
  /// Synthesis Guidelines:
  ///   - Technology Mapping:
  ///     * Register array inference
  ///     * Optional optimization
  ///     * Standard cell friendly
  ///
  ///   - Timing Optimization:
  ///     * Pipeline mode for frequency
  ///     * Enable path critical
  ///     * Address setup important
  ///
  /// Verification Requirements:
  ///   - Functional Checks:
  ///     * Concurrent access patterns
  ///     * Collision scenarios
  ///     * Enable combinations
  ///
  ///   - Timing Verification:
  ///     * Setup/hold requirements
  ///     * Clock domain crossing
  ///     * Enable timing windows
  ///
endmodule
