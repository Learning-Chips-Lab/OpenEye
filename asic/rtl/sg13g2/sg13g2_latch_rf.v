// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Latch-based register file for the PE scratchpads (SPad_SP / SPad_DP),
/// selected with `OPENEYE_LATCH_RF` (see hdl/SPad_SP.v, hdl/SPad_DP.v).
///
/// Why: in p2_15 the PE scratchpads were 393k enable-DFFs
/// (sg13g2_dfrbpq_1 48.99 um^2 + enable mux2 18.14 um^2 = ~67 um^2/bit);
/// weight_data_SPad alone (48 PE x 256 x 24) was 58 % of all flops.
/// A storage bit here is one sg13g2_dlhq_1 (30.84 um^2) plus one
/// sg13g2_lgcp_1 clock gate (27.22 um^2) per word -> ~32 um^2/bit for
/// 24-bit words, no enable mux, and the clock tree sees one ICG per word
/// instead of DataWidth flop sinks.
///
/// Write scheme (cycle-equivalent to RAM_SP_generic / RAM_DP_generic):
///   cycle k   : we_i/waddr_i/wdata_i presented. The word's ICG latches
///               en = we_i & (waddr_i == w) while clk is LOW (late cycle k).
///   posedge k+1: wdata_q <= wdata_i (one shared write-data register,
///               no enable). gclk[w] rises -> word latches transparent
///               during the HIGH phase of cycle k+1, loading wdata_q.
///   negedge k+1: gclk[w] falls, word holds the new value.
///   A read captured at posedge k+1 still sees the old value (same as the
///   flop RAM, write lands at posedge k+1 via NBA); a read captured at
///   posedge k+2 sees the new value. wdata_q is stable from posedge k+1
///   to posedge k+2, so the latch closing edge (negedge k+1) has no hold
///   race against wdata_q; the latch D path is a half-cycle path.
///
/// Read: registered (memout, one cycle), q = memout -- the Pipelined=0
/// behavior of the generic RAMs. Read/write of the same word in the same
/// cycle is a collision in both generic models ($error / X) and is not
/// supported here either (the read may return the transient value).
///
/// Cells: with `OPENEYE_SG13G2_CELLS` (ASIC flow) the PDK cells
/// sg13g2_lgcp_1 / sg13g2_dlhq_1 are instantiated explicitly, so yosys
/// infers no latches and CTS sees real integrated clock gates. Without it
/// (RTL simulation) behaviorally equivalent models are used.

module oe_icg (
    input  wire clk_i,
    input  wire en_i,
    output wire gclk_o
);
`ifdef OPENEYE_SG13G2_CELLS
    sg13g2_lgcp_1 u_icg (.CLK(clk_i), .GATE(en_i), .GCLK(gclk_o));
`else
    // enable latch transparent while clk is low, AND with clk
    reg en_l;
    /* verilator lint_off LATCH */
    always @(*) if (!clk_i) en_l = en_i;
    /* verilator lint_on LATCH */
    assign gclk_o = clk_i & en_l;
`endif
endmodule

module oe_dlatch (
    input  wire gate_i,
    input  wire d_i,
    output wire q_o
);
`ifdef OPENEYE_SG13G2_CELLS
    sg13g2_dlhq_1 u_lat (.GATE(gate_i), .D(d_i), .Q(q_o));
`else
    reg q;
    /* verilator lint_off LATCH */
    always @(*) if (gate_i) q = d_i;
    /* verilator lint_on LATCH */
    assign q_o = q;
`endif
endmodule

/// One write port, one read port (separate addresses). SPad_SP connects
/// both addresses to addr_i and gates re_i with !we_i (write wins, like
/// RAM_SP); SPad_DP uses them independently.
module oe_latch_rf_1w1r #(
    parameter AddrWidth = 8,
    parameter DataWidth = 24
) (
    input  wire                 clk_i,
    input  wire                 we_i,
    input  wire [AddrWidth-1:0] waddr_i,
    input  wire [DataWidth-1:0] wdata_i,
    input  wire                 re_i,
    input  wire [AddrWidth-1:0] raddr_i,
    output wire [DataWidth-1:0] rdata_o
);
    localparam Depth = 2 ** AddrWidth;

    reg  [DataWidth-1:0] wdata_q;
    reg  [DataWidth-1:0] memout;
    // flat storage vector: word w = mem_q[w*DataWidth +: DataWidth]
    wire [Depth*DataWidth-1:0] mem_q;

    always @(posedge clk_i) wdata_q <= wdata_i;

    genvar w, b;
    generate
        for (w = 0; w < Depth; w = w + 1) begin : g_word
            wire gclk;
            oe_icg u_icg (
                .clk_i (clk_i),
                .en_i  (we_i && (waddr_i == w)),
                .gclk_o(gclk)
            );
`ifdef OPENEYE_SG13G2_CELLS
            // ASIC: one sg13g2_dlhq_1 per bit
            for (b = 0; b < DataWidth; b = b + 1) begin : g_bit
                oe_dlatch u_lat (
                    .gate_i(gclk),
                    .d_i   (wdata_q[b]),
                    .q_o   (mem_q[w*DataWidth+b])
                );
            end
`else
            // RTL sim: one behavioral word latch (same function as the
            // per-bit cells, ~DataWidth x fewer instances/events -- the
            // full-chip cocotb regression has ~400k latch bits)
            reg [DataWidth-1:0] word_q;
            /* verilator lint_off LATCH */
            always @(*) if (gclk) word_q = wdata_q;
            /* verilator lint_on LATCH */
            assign mem_q[w*DataWidth +: DataWidth] = word_q;
`endif
        end
    endgenerate

    always @(posedge clk_i) begin
        if (re_i) memout <= mem_q[raddr_i*DataWidth +: DataWidth];
    end

    assign rdata_o = memout;
endmodule
