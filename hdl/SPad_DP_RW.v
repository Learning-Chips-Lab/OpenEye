// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: SPad_DP_RW
/// 
/// The Dual-Port Read/Write Scratchpad (SPad_DP_RW) implements a sophisticated dual-port
/// memory architecture in OpenEye that supports simultaneous read and write operations
/// on both ports. This true dual-port design enables maximum flexibility in data access
/// patterns while maintaining high throughput.
///
/// Key Features:
/// - True Dual-Port Architecture:
///   * Two independent read/write ports (A and B)
///   * Simultaneous read/write on both ports
///   * Independent address spaces per operation
///
/// - Flexible Access Patterns:
///   * Read-while-write support
///   * Write-while-write capability
///   * Read-while-read operations
///
/// - Implementation Abstraction:
///   * Standard cell based design
///   * Reset-free interface
///   * Configurable parameters
///
/// - Access Constraints:
///   * Address Collision Prevention:
///     When multiple ports are active, their respective
///     addresses must be different to ensure data integrity
///
/// Architectural Role:
/// The SPad_DP_RW serves as:
/// 1. High-performance local buffer
/// 2. Multi-threaded data cache
/// 3. Parallel processing memory
/// 4. Flexible data exchange buffer
///
/// Parameters:
///    DATA_WIDTH        - Storage Configuration
///                        Width of each data word
///                        Determines data granularity
///
///    ADDR_WIDTH       - Memory Organization
///                        Address space size (log2)
///                        Total capacity: 2^ADDR_WIDTH words
///
///    Implementation   - Hardware Strategy
///                        0: Default standard cell implementation
///                        Other values for specialized implementations
///                        Enables technology-specific optimizations
///   
/// Ports:
/// Clock Interface:
///    clk_i           - System Clock Input
///                      Positive edge triggered
///                      Synchronizes all port operations
///
/// Port A Interface:
///    re_a_i         - Port A Read Enable (active high)
///                     1: Enable read from addr_r_a_i
///                     0: Disable read operation
///
///    we_a_i         - Port A Write Enable (active high)
///                     1: Enable write to addr_w_a_i
///                     0: Disable write operation
///
///    addr_r_a_i     - Port A Read Address [ADDR_WIDTH-1:0]
///                     Target address for read operations
///                     Must not conflict with other active addresses
///
///    addr_w_a_i     - Port A Write Address [ADDR_WIDTH-1:0]
///                     Target address for write operations
///                     Must not conflict with other active addresses
///
///    data_a_i       - Port A Write Data [DATA_WIDTH-1:0]
///                     Data to be written when we_a_i is high
///
///    data_a_o       - Port A Read Data [DATA_WIDTH-1:0]
///                     Data output when re_a_i is high
///
/// Port B Interface:
///    re_b_i         - Port B Read Enable (active high)
///                     1: Enable read from addr_r_b_i
///                     0: Disable read operation
///
///    we_b_i         - Port B Write Enable (active high)
///                     1: Enable write to addr_w_b_i
///                     0: Disable write operation
///
///    addr_r_b_i     - Port B Read Address [ADDR_WIDTH-1:0]
///                     Target address for read operations
///                     Must not conflict with other active addresses
///
///    addr_w_b_i     - Port B Write Address [ADDR_WIDTH-1:0]
///                     Target address for write operations
///                     Must not conflict with other active addresses
///
///    data_b_i       - Port B Write Data [DATA_WIDTH-1:0]
///                     Data to be written when we_b_i is high
///
///    data_b_o       - Port B Read Data [DATA_WIDTH-1:0]
///                     Data output when re_b_i is high
///

module SPad_DP_RW #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 10,
    parameter Implementation = 0
) (
    input wire clk_i,

    /*
   * the DPRW spad is implemented in standard cells which makes
   * it resettable in theory.
   * To keep the interface opaque, we still don't allow it
   */
    //input                           rst_ni,

    input wire re_a_i,
    input wire re_b_i,

    input wire we_a_i,
    input wire we_b_i,

    input wire [ADDR_WIDTH-1:0] addr_r_a_i,
    input wire [ADDR_WIDTH-1:0] addr_r_b_i,
    input wire [ADDR_WIDTH-1:0] addr_w_a_i,
    input wire [ADDR_WIDTH-1:0] addr_w_b_i,

    input wire [DATA_WIDTH-1:0] data_a_i,
    input wire [DATA_WIDTH-1:0] data_b_i,

    output wire [DATA_WIDTH-1:0] data_a_o,
    output wire [DATA_WIDTH-1:0] data_b_o
);


  RAM_DP_RW #(
      .AddrWidth(ADDR_WIDTH),
      .DataWidth(DATA_WIDTH),
      .Pipelined(0)
  ) ram (
      .clk_i     (clk_i),
      .re_a_i    (re_a_i),
      .re_b_i    (re_b_i),
      .we_a_i    (we_a_i),
      .we_b_i    (we_b_i),
      .addr_r_a_i(addr_r_a_i),
      .addr_r_b_i(addr_r_b_i),
      .addr_w_a_i(addr_w_a_i),
      .addr_w_b_i(addr_w_b_i),
      .data_a_i  (data_a_i),
      .data_b_i  (data_b_i),
      .data_a_o  (data_a_o),
      .data_b_o  (data_b_o)
  );
endmodule
