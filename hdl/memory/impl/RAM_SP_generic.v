// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

/// Module: RAM_SP_generic
///
/// (R)andom(A)ccess(M)emory_(S)ingle(P)ort_generic is a generic implementation of a single port RAM.
/// It represents a true single port RAM with one clock and one port with both read and write capability.
/// If there is no SRAM implementation of such a memory, it will get mapped to standard cells.
///
/// Parameters:
///    AddrWidth             - Amount of storageable data words in clog2
///    DataWidth             - Length of data words
///    Pipelined             - Enable pipelining
///
/// Ports:
///    clk             - Clock input
///    cen             - Clock enable
///    rdwen           - Read enable
///    a               - Address port
///    d               - Data port in
///    q               - Data port out
///

module RAM_SP_generic 
#(
    parameter AddrWidth = 12,
    parameter DataWidth = 8,
    parameter Pipelined = 0
) (
    input  wire                     clk,
    input  wire                     cen,
    input  wire                     rdwen,
    input  wire  [AddrWidth-1:0]    a,
    input  wire  [DataWidth-1:0]    d,
    output reg   [DataWidth-1:0]    q
);
    reg     [DataWidth-1:0]     mem[(2**AddrWidth)-1];
    reg     [DataWidth-1:0]     memout;

    generate
    if (Pipelined) begin 
        always @(posedge clk) q <= memout;
    end else begin
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
endmodule