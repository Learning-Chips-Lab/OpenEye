// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

/// Module: RAM_DP
///
/// (R)andom(A)ccess(M)emory_(D)ual(P)ort is a wrapper module to provide a unified
/// interface for a dual port RAM that hides the underlying implementation details.
///
/// Parameters:
///    AddrWidth             - Amount of storageable data words in clog2
///    DataWidth             - Length of data words
///    Pipelined             - Enable pipelining
///    Implementation         - Select the implementation of the RAM
///
/// Ports:
///    clk_i             - Clock input
///    rd_en_i           - Read enable port
///    wr_en_i           - Write enable port
///    addr_r_i          - Address port for `data_o`
///    addr_w_i          - Address port for `data_i`
///    data_i            - Data port in
///    data_o            - Data port out
///

module RAM_DP 
#(
    parameter AddrWidth = 16,
    parameter DataWidth = 32,
    parameter Pipelined = 0,
    
    // Hint for physical Implementation.
    parameter Implementation = 0
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
    input   wire    [AddrWidth-1:0]     addr_r_i,
    input   wire    [AddrWidth-1:0]     addr_w_i,

    // data input
    input   wire    [DataWidth-1:0]     data_i,

    // data output
    output  wire    [DataWidth-1:0]     data_o
);


    RAM_DP_generic #(
        .AddrWidth          (AddrWidth),
        .DataWidth          (DataWidth),
        .Pipelined          (Pipelined)
    ) impl (
        .clkA               (clk_i),
        .clkB               (clk_i),

        .cenA               (!(rd_en_i)),
        .cenB               (!(wr_en_i)),

        .aA                 (addr_r_i),
        .aB                 (addr_w_i),

        .d                  (data_i),
        .bw                 ({DataWidth{1'b1}}),
        .q                  (data_o)
    );

endmodule