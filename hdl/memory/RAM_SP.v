// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

/// Module: RAM_SP
///
/// (R)andom(A)ccess(M)emory_(S)ingle(P)ort is a wrapper module to provide a unified
/// interface for a single port RAM that hides the underlying implementation details.
/// This will get directly or indirectly mapped to an actual RAM macro in the physical flow.
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
///    addr_i            - Address port
///    data_i            - Data port in
///    data_o            - Data port out
///

module RAM_SP 
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

endmodule