// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: RAM_SP
///
/// The RAM_SP (Random Access Memory Single Port) module provides a unified, technology-independent
/// interface for single-port memory implementations in the OpenEye neural network accelerator.
/// It serves as an abstraction layer that can be mapped to different physical memory macros
/// while maintaining consistent behavior and interface conventions.
///
/// Architecture Overview:
/// - Single-Port Memory:
///   * Combined read/write port with shared address bus
///   * Mutually exclusive read/write operations
///   * Configurable data and address widths
///
/// - Implementation Flexibility:
///   * Generic RTL implementation by default
///   * Conditional inclusion of technology-specific macros
///   * Support for pipelined operation
///   * Configurable memory organization
///
/// Key Features:
/// - Synchronous Operation:
///   * Clock-synchronized read/write access
///   * No asynchronous reset (intentional design choice)
///   * Undefined initial state management
///
/// - Access Control:
///   * Independent read/write enables
///   * Power-saving inactive state
///   * Write priority arbitration
///
/// - Configuration Options:
///   * Adjustable memory depth
///   * Variable data width
///   * Optional output pipelining
///   * Implementation hints for synthesis
///
/// Parameters:
/// Memory Organization:
///    AddrWidth        - Address bus width (log2 of memory depth)
///                       Determines the number of addressable locations
///    DataWidth        - Data bus width in bits
///                       Defines the size of each memory word
///
/// Performance Options:
///    Pipelined        - Enable registered outputs for improved timing
///                       0: Combinational read path
///                       1: Registered read path
///
/// Implementation Control:
///    Implementation   - Synthesis directive for physical implementation
///                       Used to guide technology mapping
///
/// Ports:
/// Clock Interface:
///    clk_i           - System clock input
///                      All operations are synchronized to rising edge
///
/// Control Interface:
///    rd_en_i         - Read enable (active high)
///                      Must be deasserted during writes
///    wr_en_i         - Write enable (active high)
///                      Takes precedence over reads
///
/// Data Interface:
///    addr_i          - Address input [AddrWidth-1:0]
///                      Shared for both reads and writes
///    data_i          - Write data input [DataWidth-1:0]
///                      Data to be written when wr_en_i is active
///    data_o          - Read data output [DataWidth-1:0]
///                      Valid when rd_en_i was active in previous cycle
///

module RAM_SP 
#(
    parameter AddrWidth = 16,
    parameter DataWidth = 32,
    parameter Pipelined = 0
    
) (
    input   wire                        clk_i,
    // We deliberately leave out a reset here so that it is obvious 
    // that memory content is undefined prior to writing to it

    // read enable, active high
    input   wire                        rd_en_i,
    // write enable, active high
    // when both rd_en and wr_en are deasserted, the RAM is inactive
    input   wire                        wr_en_i,

    // address input
    input   wire    [AddrWidth-1:0]     addr_i,

    // data input
    input   wire    [DataWidth-1:0]     data_i,

    // data output
    output  wire    [DataWidth-1:0]     data_o
);

    wire                    clk;
    wire                    cen;
    wire                    rdwen;
    wire  [AddrWidth-1:0]   addr;
    wire  [DataWidth-1:0]   d;
    wire  [DataWidth-1:0]   q;
    
    assign clk          = clk_i;
    assign cen          = !(rd_en_i || wr_en_i);
    assign rdwen        = !wr_en_i;
    assign addr         = addr_i;
    assign d            = data_i;

    assign data_o = q;

`ifdef OPENEYE_RAM_SPECIFIC
    generate
        // specific SRAM Macros / blocks for synthesis or simulation can be included here
        //`include "impl/OpenRAM/SP_OpenRAM_specific.vh"
    else begin
`endif
        RAM_SP_generic #(
            .AddrWidth          (AddrWidth),
            .DataWidth          (DataWidth),
            .Pipelined          (Pipelined)
        ) impl (
            .clk                (clk),
            .cen                (cen),
            .rdwen              (rdwen),
            .a                  (addr),
            .d                  (d),
            .q                  (q)
        );
`ifdef OPENEYE_RAM_SPECIFIC
    end
    endgenerate
`endif

    /// Implementation Notes:
    /// Memory Architecture:
    ///   - Single-Port Design:
    ///     * One shared port for both read and write operations
    ///     * Common address bus for both operations
    ///     * Mutually exclusive access control
    ///
    ///   - Signal Mapping:
    ///     * Internal clock (clk) directly maps to system clock
    ///     * Chip enable (cen) derived from read/write enables
    ///     * Read/write control (rdwen) prioritizes writes
    ///     * Direct address and data path mapping
    ///
    /// Access Protocol:
    ///   - Write Operation:
    ///     * Assert wr_en_i
    ///     * Present valid addr_i and data_i
    ///     * Data captured on rising clock edge
    ///     * Previous data at addr_i is overwritten
    ///
    ///   - Read Operation:
    ///     * Assert rd_en_i
    ///     * Present valid addr_i
    ///     * Data appears on data_o
    ///     * Timing depends on Pipelined parameter
    ///
    /// Technology Mapping:
    ///   - Conditional Implementation:
    ///     * OPENEYE_RAM_SPECIFIC macro controls mapping
    ///     * Default to generic RTL implementation
    ///     * Support for technology-specific variants
    ///
    ///   - Generic Implementation:
    ///     * Behavioral memory model
    ///     * Synthesizable RTL description
    ///     * Standard cell mapping support
    ///
    /// Performance Considerations:
    ///   - Clock Domain:
    ///     * Single clock domain operation
    ///     * No clock gating implemented
    ///     * Simple timing closure
    ///
    ///   - Access Timing:
    ///     * Write: One cycle latency
    ///     * Read: One/Two cycle latency (based on Pipelined)
    ///     * No concurrent read/write support
    ///
    /// Power Management:
    ///   - Active Control:
    ///     * Memory active when rd_en_i or wr_en_i asserted
    ///     * Inactive state when both deasserted
    ///     * Chip enable used for power control
    ///
    /// Integration Guidelines:
    ///   - Reset Handling:
    ///     * No reset input provided
    ///     * Initial memory contents undefined
    ///     * System must handle initialization
    ///
    ///   - Technology Mapping:
    ///     * Implementation parameter guides synthesis
    ///     * Vendor-specific mapping possible
    ///     * Memory compiler integration ready
    ///
endmodule
