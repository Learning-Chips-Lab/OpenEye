// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: RAM_DP
///
/// The RAM_DP (Random Access Memory Dual Port) module provides a technology-independent wrapper
/// for dual-port memory implementations in the OpenEye neural network accelerator. It offers
/// separate read and write ports with independent address buses, enabling concurrent access
/// while abstracting the underlying memory implementation.
///
/// Architecture Overview:
/// - Dual-Port Memory:
///   * Independent read and write ports
///   * Separate address buses for each port
///   * Concurrent read/write capability
///   * Configurable memory organization
///
/// - Implementation Flexibility:
///   * Technology-independent interface
///   * Configurable timing behavior
///   * Synthesis-friendly design
///   * Optional output pipelining
///
/// Key Features:
/// - Independent Port Operation:
///   * Simultaneous read/write access
///   * Separate port enables
///   * Independent address spaces
///   * Full data width access
///
/// - Memory Management:
///   * No reset signal (intentional)
///   * Power-saving inactive state
///   * Configurable implementation
///   * Flexible timing options
///
/// Parameters:
/// Memory Organization:
///    AddrWidth        - Address bus width (log2 of memory depth)
///                       Defines number of addressable locations
///    DataWidth        - Data bus width in bits
///                       Specifies word size for read/write operations
///
/// Performance Options:
///    Pipelined        - Output register enable
///                       0: Combinational read path
///                       1: Registered read output
///
/// Implementation Control:
///    Implementation   - Technology mapping hint
///                       Guides physical implementation selection
///
/// Ports:
/// Clock Interface:
///    clk_i           - System clock input
///                      All operations synchronized to rising edge
///
/// Control Interface:
///    rd_en_i         - Read port enable (active high)
///                      Controls read operations
///    wr_en_i         - Write port enable (active high)
///                      Controls write operations
///
/// Address Interface:
///    addr_r_i        - Read address [AddrWidth-1:0]
///                      Selects location for read operations
///    addr_w_i        - Write address [AddrWidth-1:0]
///                      Selects location for write operations
///
/// Data Interface:
///    data_i          - Write data input [DataWidth-1:0]
///                      Data to be written when wr_en_i active
///    data_o          - Read data output [DataWidth-1:0]
///                      Data from location specified by addr_r_i
///

module RAM_DP #(
    parameter AddrWidth = 16,
    parameter DataWidth = 32,
    parameter Pipelined = 0,

    // Hint for physical Implementation.
    parameter Implementation = 0
) (
    input wire clk_i,
    // We deliberately leave out a reset here so that it is obvious 
    // that memory content is undefined prior to writing to it

    // read enable, active high
    input wire rd_en_i,
    // write enable, active high
    // when both rd_en and wr_en are deasserted, the RAM is inactive
    input wire wr_en_i,

    // address input
    input wire [AddrWidth-1:0] addr_r_i,
    input wire [AddrWidth-1:0] addr_w_i,

    // data input
    input wire [DataWidth-1:0] data_i,

    // data output
    output wire [DataWidth-1:0] data_o
);


  RAM_DP_generic #(
      .AddrWidth(AddrWidth),
      .DataWidth(DataWidth),
      .Pipelined(Pipelined)
  ) impl (
      .clkA(clk_i),
      .clkB(clk_i),

      .cenA(!(rd_en_i)),
      .cenB(!(wr_en_i)),

      .aA(addr_r_i),
      .aB(addr_w_i),

      .d (data_i),
      .bw({DataWidth{1'b1}}),
      .q (data_o)
  );

  /// Implementation Notes:
  /// Memory Architecture:
  ///   - Port Organization:
  ///     * Port A: Read operations
  ///       - Independent address (addr_r_i)
  ///       - Independent enable (rd_en_i)
  ///       - Data output path (data_o)
  ///
  ///     * Port B: Write operations
  ///       - Independent address (addr_w_i)
  ///       - Independent enable (wr_en_i)
  ///       - Data input path (data_i)
  ///
  /// Interface Mapping:
  ///   - Clock Distribution:
  ///     * Single clock (clk_i) used for both ports
  ///     * Synchronous operation
  ///     * No clock domain crossing
  ///
  ///   - Control Signals:
  ///     * Active-high enables
  ///     * Inverted internally for cenA/cenB
  ///     * Full word access (bw tied high)
  ///
  /// Timing Behavior:
  ///   - Write Operations:
  ///     * Single cycle latency
  ///     * Data captured on rising edge
  ///     * Address setup before clock
  ///
  ///   - Read Operations:
  ///     * One or two cycle latency (Pipelined)
  ///     * Address setup before clock
  ///     * Data hold time guaranteed
  ///
  /// Power Management:
  ///   - Static Power:
  ///     * Both ports inactive when enables low
  ///     * Implementation-dependent optimization
  ///
  ///   - Dynamic Power:
  ///     * Active only during operations
  ///     * Independent port activity
  ///
  /// Technology Mapping:
  ///   - Implementation Options:
  ///     * Maps to RAM_DP_generic by default
  ///     * Can be overridden for specific tech
  ///     * Preserves functional behavior
  ///
  /// Integration Guidelines:
  ///   - Reset Handling:
  ///     * No reset input provided
  ///     * Initial contents undefined
  ///     * System must handle initialization
  ///
  ///   - Access Patterns:
  ///     * Support concurrent R/W
  ///     * No write collision detection
  ///     * System must manage conflicts
  ///
  /// Verification Notes:
  ///   - Port Independence:
  ///     * Test concurrent access
  ///     * Verify no port interference
  ///     * Check timing requirements
  ///
  ///   - Memory Contents:
  ///     * Verify write operations
  ///     * Check read consistency
  ///     * Test enable functionality
  ///
endmodule
