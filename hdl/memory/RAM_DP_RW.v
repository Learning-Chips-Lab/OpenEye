// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

/// Module: RAM_DP_RW
///
/// (R)andom(A)ccess(M)emory_(D)ual(P)ort_(R)ead(W)rite is a wrapper module to provide a unified
/// interface for a dual port RAM that hides the underlying implementation details.
/// It represents a true dual port RAM with two (one clock, two ports, each with both read and write capability).
/// If there is no SRAM implementation of such a memory, it will get mapped to standard cells.
///
/// Parameters:
///    AddrWidth             - Amount of storageable data words in clog2
///    DataWidth             - Length of data words
///    Pipelined             - Enable pipelining
///    Implementation         - Select the implementation of the RAM
///
/// Ports:
///    clk_i             - Clock input
///    re_a_i            - Read enable port A
///    re_b_i            - Read enable port B
///    we_a_i            - Write enable port A
///    we_b_i            - Write enable port B
///    addr_r_a_i        - Address port for `data_a_o`
///    addr_r_b_i        - Address port for `data_b_o`
///    addr_w_a_i        - Address port for `data_a_i`
///    addr_w_b_i        - Address port for `data_b_i`
///    data_a_i          - Data port in A
///    data_b_i          - Data port in B
///    data_a_o          - Data port out A
///    data_b_o          - Data port out B
///

module RAM_DP_RW 
#(
    parameter AddrWidth = 16,
    parameter DataWidth = 32,
    parameter Pipelined = 0,
    
    // Hint for physical implementation.
    parameter Implementation = "generic"
) (
    // Common clock
    input   wire                        clk_i,
    // We deliberately leave out a reset here so that it is obvious 
    // that memory content is undefined prior to writing to it

    // read enable, active high
    input   wire                        re_a_i,
    input   wire                        re_b_i,
    // write enable, active high
    // when both rd_en and wr_en are deasserted, the RAM is inactive
    input   wire                        we_a_i,
    input   wire                        we_b_i,

    // address input
    input   wire    [AddrWidth-1:0]     addr_r_a_i,
    input   wire    [AddrWidth-1:0]     addr_r_b_i,
    input   wire    [AddrWidth-1:0]     addr_w_a_i,
    input   wire    [AddrWidth-1:0]     addr_w_b_i,

    // data input
    input   wire    [DataWidth-1:0]     data_a_i,
    input   wire    [DataWidth-1:0]     data_b_i,

    // data output
    output  wire    [DataWidth-1:0]     data_a_o,
    output  wire    [DataWidth-1:0]     data_b_o
);


    RAM_DP_RW_generic #(
        .AddrWidth          (AddrWidth),
        .DataWidth          (DataWidth),
        .Pipelined          (Pipelined)
    ) impl (
        .clk_i              (clk_i),
        .rd_en_a_i          (re_a_i),
        .rd_en_b_i          (re_b_i),
        .wr_en_a_i          (we_a_i),
        .wr_en_b_i          (we_b_i),
        .addr_r_a_i         (addr_r_a_i),
        .addr_r_b_i         (addr_r_b_i),
        .addr_w_a_i         (addr_w_a_i),
        .addr_w_b_i         (addr_w_b_i),
        .data_a_i           (data_a_i),
        .data_b_i           (data_b_i),
        .data_a_o           (data_a_o),
        .data_b_o           (data_b_o)
    );

endmodule