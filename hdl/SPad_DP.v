// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: SPad_DP
///
/// The Dual-Port Scratchpad (SPad_DP) provides a high-performance, dual-port memory 
/// interface in the OpenEye architecture. This module abstracts the underlying RAM 
/// implementation while enabling simultaneous read and write operations for enhanced 
/// data throughput.
///
/// Key Features:
/// - True Dual-Port Access:
///   * Simultaneous read and write operations
///   * Independent address spaces for read/write
///   * High-bandwidth data transfer
///
/// - Configurable Architecture:
///   * Parameterized data and address widths
///   * Flexible memory organization
///   * Implementation strategy selection
///
/// - Reset-Free Design:
///   * Intentionally omits reset functionality
///   * Enables SRAM-based implementations
///   * Preserves data persistence
///
/// - Access Constraints:
///   * Read/Write Address Collision Prevention:
///     When both read (re_i) and write (we_i) are active,
///     read address (addr_r_i) and write address (addr_w_i)
///     must be different to ensure data integrity
///
/// Architectural Role:
/// The SPad_DP serves as:
/// 1. High-bandwidth local storage
/// 2. Double-buffered data cache
/// 3. Parallel access memory buffer
/// 4. Performance-critical data storage
///
/// Parameters:
///    DATA_WIDTH        - Data Path Configuration
///                        Width of data words
///                        Determines storage granularity
///
///    ADDR_WIDTH       - Memory Organization
///                        Address space size (log2)
///                        Total capacity: 2^ADDR_WIDTH words
///
///    Implementation   - Hardware Strategy
///                        0: Default implementation
///                        Other values select specialized implementations
///                        Enables target-specific optimizations
///   
/// Ports:
/// Clock Interface:
///    clk_i           - System Clock Input
///                      Positive edge triggered
///                      Synchronizes all operations
///
/// Write Interface:
///    we_i           - Write Enable Input (active high)
///                     Controls write operations
///                     1: Write enabled
///                     0: Write disabled
///
///    addr_w_i       - Write Address Bus [ADDR_WIDTH-1:0]
///                     Specifies write location
///                     Must differ from addr_r_i during concurrent access
///
///    data_i         - Write Data Bus [DATA_WIDTH-1:0]
///                     Data to be written
///                     Sampled on rising edge when we_i is high
///
/// Read Interface:
///    re_i           - Read Enable Input (active high)
///                     Controls read operations
///                     1: Read enabled
///                     0: Read disabled
///
///    addr_r_i       - Read Address Bus [ADDR_WIDTH-1:0]
///                     Specifies read location
///                     Must differ from addr_w_i during concurrent access
///
///    data_o         - Read Data Bus [DATA_WIDTH-1:0]
///                     Output data from specified read address
///                     Updates on rising edge when re_i is high
///

module SPad_DP #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 10
) (
    input wire clk_i,

    /* 
   * Remove reset port to make it clear that the
   * contents of the spad cannot simply be reset.
   * This is to make it possible to implement this SPAD
   * as an SRAM.
   */
    //input                         rst_ni,

    input  wire                  re_i,
    input  wire                  we_i,
    input  wire [ADDR_WIDTH-1:0] addr_r_i,
    input  wire [ADDR_WIDTH-1:0] addr_w_i,
    input  wire [DATA_WIDTH-1:0] data_i,
    output wire [DATA_WIDTH-1:0] data_o
);

  RAM_DP #(
      .AddrWidth(ADDR_WIDTH),
      .DataWidth(DATA_WIDTH),
      .Pipelined(0)
  ) ram (
      .clk_i   (clk_i),
      .rd_en_i (re_i),
      .wr_en_i (we_i),
      .addr_r_i(addr_r_i),
      .addr_w_i(addr_w_i),
      .data_i  (data_i),
      .data_o  (data_o)
  );
endmodule
