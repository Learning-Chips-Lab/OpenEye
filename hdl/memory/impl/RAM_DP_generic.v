// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

/// Module: RAM_DP_generic
///
/// (R)andom(A)ccess(M)emory_(D)ual(P)ort_generic is a generic implementation of a dual port RAM.
/// It represents a true dual port RAM with two (one clock, two ports, each with both read and write capability).
/// If there is no SRAM implementation of such a memory, it will get mapped to standard cells.
///
/// Parameters:
///    AddrWidth             - Amount of storageable data words in clog2
///    DataWidth             - Length of data words
///    Pipelined             - Enable pipelining
///
/// Ports:
///    clkA             - Clock input for port A
///    clkB             - Clock input for port B
///    cenA             - Clock enable for port A
///    cenB             - Clock enable for port B
///    aA               - Address port for port A
///    aB               - Address port for port B
///    d                - Data port in
///    bw               - Data port in for write
///    q                - Data port out
///

module RAM_DP_generic 
#(
    parameter AddrWidth = 12,
    parameter DataWidth = 8,
    parameter Pipelined = 0
) (
    input  wire                     clkA,
    input  wire                     clkB,

    input  wire                     cenA,
    input  wire                     cenB,

    input  wire  [AddrWidth-1:0]    aA,
    input  wire  [AddrWidth-1:0]    aB,

    input  wire  [DataWidth-1:0]    d,
    input  wire  [DataWidth-1:0]    bw,
    output reg   [DataWidth-1:0]     q
);

    reg     [DataWidth-1:0]     mem[0:(2**AddrWidth)-1];

    reg     [DataWidth-1:0]     memout;

    generate
    if (Pipelined) begin 
        always @(posedge clkA) q <= memout;
    end else begin
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
                $error("R/W collision in DP sram");
                memout <= 'hx;
                // cadence synthesis_on
            end else begin
                memout <= mem[aA];
            end
        end else begin
            memout <= {DataWidth{1'bx}};
        end
    end
endmodule