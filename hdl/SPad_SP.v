// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: SPad_SP
///
/// The Single-Port Scratchpad (SPad_SP) is a fundamental memory component in the OpenEye
/// architecture that provides high-speed, single-port access to local data storage. It serves
/// as an abstraction layer over the physical RAM implementation, offering a unified interface
/// for temporary data storage within processing elements.
///
/// Key Features:
/// - Single-Port Access:
///   * Mutually exclusive read/write operations
///   * Simplified control logic
///   * Resource-efficient implementation
///
/// - Configurable Architecture:
///   * Parameterized data and address widths
///   * Flexible memory depth
///   * Selectable implementation strategies
///
/// - Reset-Free Design:
///   * No reset port by design
///   * Maintains data persistence
///   * Explicit initialization required
///
/// Architectural Role:
/// The SPad_SP module serves as:
/// 1. Local storage for processing elements
/// 2. Temporary buffer for intermediate results
/// 3. Resource-efficient memory solution
/// 4. Building block for larger memory hierarchies
///
/// Parameters:
///    DATA_WIDTH        - Memory Configuration
///                        Width of each data word
///                        Determines storage granularity
///
///    ADDR_WIDTH       - Memory Organization
///                        Address space size (log2)
///                        Defines total memory depth (2^ADDR_WIDTH words)
///
///    Implementation   - Implementation Strategy
///                        0: Default implementation
///                        Other values select alternative implementations
///                        Allows optimization for different targets
///   
/// Ports:
/// Clock Interface:
///    clk_i           - System Clock Input
///                      Positive edge triggered
///
/// Control Interface:
///    we_i           - Write Enable Input (active high)
///                     1: Write operation enabled
///                     0: Write operation disabled
///
///    re_i           - Read Enable Input (active high)
///                     1: Read operation enabled
///                     0: Read operation disabled
///
/// Address Interface:
///    addr_i         - Address Bus [ADDR_WIDTH-1:0]
///                     Specifies target memory location
///                     Valid range: 0 to 2^ADDR_WIDTH-1
///
/// Data Interface:
///    data_i         - Write Data Bus [DATA_WIDTH-1:0]
///                     Data to be written during write operations
///
///    data_o         - Read Data Bus [DATA_WIDTH-1:0]
///                     Data output during read operations
///                     Maintains last read value when inactive
///

module SPad_SP #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 10,
    parameter Implementation = 0
) (
    input clk_i,

    // Remove reset port to make it clear that the
    // contents of the spad cannot simply be reset
    //input                         rst_ni,

    input                   re_i,
    input                   we_i,
    input  [ADDR_WIDTH-1:0] addr_i,
    input  [DATA_WIDTH-1:0] data_i,
    output [DATA_WIDTH-1:0] data_o
);

  RAM_SP #(
      .AddrWidth(ADDR_WIDTH),
      .DataWidth(DATA_WIDTH),
      .Pipelined(0),

      .Implementation(Implementation)
  ) ram (
      .clk_i  (clk_i),
      .rd_en_i(re_i),
      .wr_en_i(we_i),
      .addr_i (addr_i),
      .data_i (data_i),
      .data_o (data_o)
  );

endmodule
