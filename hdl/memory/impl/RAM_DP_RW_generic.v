// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

/// Module: RAM_DP_RW_generic
///
/// (R)andom(A)ccess(M)emory_(D)ual(P)ort_(R)ead(W)rite_(G)eneric is a wrapper module to provide a unified
/// interface for a dual port RAM that hides the underlying implementation details.
/// It represents a true dual port RAM with two (one clock, two ports, each with both read and write capability).
/// If there is no SRAM implementation of such a memory, it will get mapped to standard cells.
///
/// Parameters:
///    AddrWidth             - Amount of storageable data words in clog2
///    DataWidth             - Length of data words
///    Pipelined             - Enable pipelining
///
/// Ports:
///    clk_i             - Clock input
///    rd_en_a_i         - Read enable port A
///    rd_en_b_i         - Read enable port B
///    wr_en_a_i         - Write enable port A
///    wr_en_b_i         - Write enable port B
///    addr_r_a_i        - Address port for `data_a_o`
///    addr_r_b_i        - Address port for `data_b_o`
///    addr_w_a_i        - Address port for `data_a_i`
///    addr_w_b_i        - Address port for `data_b_i`
///    data_a_i          - Data port in A
///    data_b_i          - Data port in B
///    data_a_o          - Data port out A
///    data_b_o          - Data port out B
///

module RAM_DP_RW_generic 
#(
    parameter AddrWidth = 16,
    parameter DataWidth = 32,
    parameter Pipelined = 0
) (
    // Common clock
    input   wire                        clk_i,
    // We deliberately leave out a reset here so that it is obvious 
    // that memory content is undefined prior to writing to it

    // read enable, active high
    input   wire                        rd_en_a_i,
    input   wire                        rd_en_b_i,
    // write enable, active high
    // when both rd_en and wr_en are deasserted, the RAM is inactive
    input   wire                        wr_en_a_i,
    input   wire                        wr_en_b_i,

    // address inputs
    input   wire    [AddrWidth-1:0]     addr_r_a_i,
    input   wire    [AddrWidth-1:0]     addr_r_b_i,
    input   wire    [AddrWidth-1:0]     addr_w_a_i,
    input   wire    [AddrWidth-1:0]     addr_w_b_i,

    // data inputs
    input   wire    [DataWidth-1:0]     data_a_i,
    input   wire    [DataWidth-1:0]     data_b_i,

    // data outputs
    output  reg     [DataWidth-1:0]     data_a_o,
    output  reg     [DataWidth-1:0]     data_b_o
);

localparam Depth = 2 ** AddrWidth;

reg [DataWidth-1:0] mem [Depth];

reg [DataWidth-1:0] memout_a;
reg [DataWidth-1:0] memout_b;

// write process
always @(posedge clk_i) begin
    if (wr_en_a_i) begin
        mem[addr_w_a_i] <= data_a_i;
    end

    if (wr_en_b_i) begin
        mem[addr_w_b_i] <= data_b_i;
    end
end

//read process
always @(posedge clk_i) begin
    if (rd_en_a_i) begin
        memout_a <= mem[addr_r_a_i];

        // cadence synthesis_off
        // synopsys translate_off
        // In standard cell memory, concurrent read and write access should not lead to undefined behavior,
        // but it's probably still a good idea not to allow it
        if (wr_en_a_i && addr_r_a_i == addr_w_a_i) begin
            $error("Collision between Port A/Read and Port A/Write");
            memout_a <= {DataWidth{1'bx}};
        end

        if (wr_en_b_i && addr_r_a_i == addr_w_b_i) begin
            $error("Collision between Port A/Read and Port B/Write");
            memout_a <= {DataWidth{1'bx}};
        end
        // synopsys translate_on
        // cadence synthesis_on
    end

    if (rd_en_b_i) begin
        memout_b <= mem[addr_r_b_i];

        // cadence synthesis_off
        // synopsys translate_off
        if (wr_en_a_i && addr_r_b_i == addr_w_a_i) begin
            $error("Collision between Port B/Read and Port A/Write");
            memout_b <= {DataWidth{1'bx}};
        end

        if (wr_en_b_i && addr_r_b_i == addr_w_b_i) begin
            $error("Collision between Port B/Read and Port B/Write");
            memout_b <= {DataWidth{1'bx}};
        end
        // synopsys translate_on
        // cadence synthesis_on
    end
end


// output
generate
    if (Pipelined) begin 
        always @(posedge clk_i) begin
            data_a_o <= memout_a;
            data_b_o <= memout_b;
        end
    end else begin
        always @(*) begin
            data_a_o = memout_a;
            data_b_o = memout_b;
        end
    end
endgenerate
endmodule