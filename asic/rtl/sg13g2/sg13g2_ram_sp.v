// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: sg13g2_ram_sp
///
/// Technology-specific single-port RAM for IHP SG13G2, built from the PDK
/// RM_IHPSG13_1P SRAM macros. Drop-in replacement for RAM_SP_generic with
/// the identical interface (clk, cen, rdwen, a, d, q, wm), selected through
/// the OPENEYE_RAM_SPECIFIC ifdef in RAM_SP.v.
///
/// Supported configurations (DataWidth, AddrWidth, WriteMaskWidth):
///   (64,  10, 1)  BUFFER_A iact buffer        1 x 1024x64
///   (288, 11, 1)  wght buffer (24*4*3)        4 x 2048x64 + 1 x 2048x32
///   (40,  12, 1)  psum buffer (20*2)          2 x 4096x16 + 1 x 4096x8
///   (72,  10, 6)  iact_stream_constructor RAM 1 x 1024x64 + 1 x 1024x8
///
/// Behavioral mapping (must match RAM_SP_generic):
///   - write:  !cen && !rdwen, per-subword enable wm (active high)
///   - read:   !cen && rdwen, registered, 1 cycle (memout)
///   - q:      memout (Pipelined=0) or registered memout (Pipelined=1)
///   - read and write are mutually exclusive by construction
///
/// Macro interface notes (RM_IHPSG13_1P, see PDK verilog model):
///   - A_MEN active high, A_WEN/A_REN active high, A_BM active high
///     write-enable bit mask (BM[i]=1 -> bit i is written)
///   - reads are registered (A_DOUT valid one cycle after the read)
///   - A_DLY must be tied to 1 (simulation model checks this)
///   - BIST port is unused and tied off

module sg13g2_ram_sp #(
    parameter AddrWidth      = 10,
    parameter DataWidth      = 64,
    parameter Pipelined      = 1,
    parameter WriteMaskWidth = 1
) (
    input  wire                 clk,
    input  wire                 cen,
    input  wire                 rdwen,
    input  wire [AddrWidth-1:0] a,
    input  wire [DataWidth-1:0] d,
    output wire [DataWidth-1:0] q,
    input  wire [WriteMaskWidth-1:0] wm
);

    localparam MWW = (WriteMaskWidth == 1) ? 0 : (DataWidth / WriteMaskWidth);

    wire rd = !cen & rdwen;
    wire wr = !cen & !rdwen;

    wire a_men = rd | wr;
    wire a_wen = wr;
    wire a_ren = rd;   // rd and wr are mutually exclusive

    wire [DataWidth-1:0] dout;

    // ------------------------------------------------------------------
    // write mask expansion: one mask bit per subword -> per-bit A_BM
    // (A_BM active high write-enable). WriteMaskWidth=1: full-word write.
    // ------------------------------------------------------------------
    wire [DataWidth-1:0] bm;
    genvar b;
    generate
        if (WriteMaskWidth == 1) begin : g_full
            assign bm = {DataWidth{1'b1}};
        end else begin : g_expanded
            for (b = 0; b < DataWidth; b = b + 1) begin : g_bit
                assign bm[b] = wm[b / MWW];
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // macro slices per configuration
    // ------------------------------------------------------------------
    localparam CFG =
        (DataWidth == 64  && AddrWidth == 10) ? 0 :
        (DataWidth == 288 && AddrWidth == 11) ? 1 :
        (DataWidth == 40  && AddrWidth == 12) ? 2 :
        (DataWidth == 72  && AddrWidth == 10) ? 3 : -1;

    wire [63:0] dout_w64_0, dout_w64_1, dout_w64_2, dout_w64_3;
    wire [31:0] dout_w32_0;
    wire [15:0] dout_w16_0, dout_w16_1;
    wire [ 7:0] dout_w8_0;

    generate
        case (CFG)
            // (64, 10): BUFFER_A iact buffer
            0: begin : g_cfg1024x64
                RM_IHPSG13_1P_1024x64_c2_bm_bist u_mac (
                    .A_CLK(clk), .A_MEN(a_men), .A_WEN(a_wen), .A_REN(a_ren),
                    .A_ADDR(a), .A_DIN(d), .A_DLY(1'b1),
                    .A_DOUT(dout_w64_0), .A_BM(bm),
                    .A_BIST_CLK(1'b0), .A_BIST_EN(1'b0), .A_BIST_MEN(1'b0),
                    .A_BIST_WEN(1'b0), .A_BIST_REN(1'b0),
                    .A_BIST_ADDR(10'b0), .A_BIST_DIN(64'b0), .A_BIST_BM(64'b0)
                );
                assign dout = dout_w64_0;
            end

            // (288, 11): wght buffer = 4 x 64 + 1 x 32
            1: begin : g_cfg2048x288
                RM_IHPSG13_1P_2048x64_c2_bm_bist u_mac0 (
                    .A_CLK(clk), .A_MEN(a_men), .A_WEN(a_wen), .A_REN(a_ren),
                    .A_ADDR(a), .A_DIN(d[287:224]), .A_DLY(1'b1),
                    .A_DOUT(dout_w64_0), .A_BM(bm[287:224]),
                    .A_BIST_CLK(1'b0), .A_BIST_EN(1'b0), .A_BIST_MEN(1'b0),
                    .A_BIST_WEN(1'b0), .A_BIST_REN(1'b0),
                    .A_BIST_ADDR(11'b0), .A_BIST_DIN(64'b0), .A_BIST_BM(64'b0)
                );
                RM_IHPSG13_1P_2048x64_c2_bm_bist u_mac1 (
                    .A_CLK(clk), .A_MEN(a_men), .A_WEN(a_wen), .A_REN(a_ren),
                    .A_ADDR(a), .A_DIN(d[223:160]), .A_DLY(1'b1),
                    .A_DOUT(dout_w64_1), .A_BM(bm[223:160]),
                    .A_BIST_CLK(1'b0), .A_BIST_EN(1'b0), .A_BIST_MEN(1'b0),
                    .A_BIST_WEN(1'b0), .A_BIST_REN(1'b0),
                    .A_BIST_ADDR(11'b0), .A_BIST_DIN(64'b0), .A_BIST_BM(64'b0)
                );
                RM_IHPSG13_1P_2048x64_c2_bm_bist u_mac2 (
                    .A_CLK(clk), .A_MEN(a_men), .A_WEN(a_wen), .A_REN(a_ren),
                    .A_ADDR(a), .A_DIN(d[159:96]), .A_DLY(1'b1),
                    .A_DOUT(dout_w64_2), .A_BM(bm[159:96]),
                    .A_BIST_CLK(1'b0), .A_BIST_EN(1'b0), .A_BIST_MEN(1'b0),
                    .A_BIST_WEN(1'b0), .A_BIST_REN(1'b0),
                    .A_BIST_ADDR(11'b0), .A_BIST_DIN(64'b0), .A_BIST_BM(64'b0)
                );
                RM_IHPSG13_1P_2048x64_c2_bm_bist u_mac3 (
                    .A_CLK(clk), .A_MEN(a_men), .A_WEN(a_wen), .A_REN(a_ren),
                    .A_ADDR(a), .A_DIN(d[95:32]), .A_DLY(1'b1),
                    .A_DOUT(dout_w64_3), .A_BM(bm[95:32]),
                    .A_BIST_CLK(1'b0), .A_BIST_EN(1'b0), .A_BIST_MEN(1'b0),
                    .A_BIST_WEN(1'b0), .A_BIST_REN(1'b0),
                    .A_BIST_ADDR(11'b0), .A_BIST_DIN(64'b0), .A_BIST_BM(64'b0)
                );
                RM_IHPSG13_1P_2048x32_c2_bm_bist u_mac4 (
                    .A_CLK(clk), .A_MEN(a_men), .A_WEN(a_wen), .A_REN(a_ren),
                    .A_ADDR(a), .A_DIN(d[31:0]), .A_DLY(1'b1),
                    .A_DOUT(dout_w32_0), .A_BM(bm[31:0]),
                    .A_BIST_CLK(1'b0), .A_BIST_EN(1'b0), .A_BIST_MEN(1'b0),
                    .A_BIST_WEN(1'b0), .A_BIST_REN(1'b0),
                    .A_BIST_ADDR(11'b0), .A_BIST_DIN(32'b0), .A_BIST_BM(32'b0)
                );
                assign dout = {dout_w64_0, dout_w64_1, dout_w64_2, dout_w64_3, dout_w32_0};
            end

            // (40, 12): psum buffer = 2 x 16 + 1 x 8
            2: begin : g_cfg4096x40
                RM_IHPSG13_1P_4096x16_c3_bm_bist u_mac0 (
                    .A_CLK(clk), .A_MEN(a_men), .A_WEN(a_wen), .A_REN(a_ren),
                    .A_ADDR(a), .A_DIN(d[39:24]), .A_DLY(1'b1),
                    .A_DOUT(dout_w16_0), .A_BM(bm[39:24]),
                    .A_BIST_CLK(1'b0), .A_BIST_EN(1'b0), .A_BIST_MEN(1'b0),
                    .A_BIST_WEN(1'b0), .A_BIST_REN(1'b0),
                    .A_BIST_ADDR(12'b0), .A_BIST_DIN(16'b0), .A_BIST_BM(16'b0)
                );
                RM_IHPSG13_1P_4096x16_c3_bm_bist u_mac1 (
                    .A_CLK(clk), .A_MEN(a_men), .A_WEN(a_wen), .A_REN(a_ren),
                    .A_ADDR(a), .A_DIN(d[23:8]), .A_DLY(1'b1),
                    .A_DOUT(dout_w16_1), .A_BM(bm[23:8]),
                    .A_BIST_CLK(1'b0), .A_BIST_EN(1'b0), .A_BIST_MEN(1'b0),
                    .A_BIST_WEN(1'b0), .A_BIST_REN(1'b0),
                    .A_BIST_ADDR(12'b0), .A_BIST_DIN(16'b0), .A_BIST_BM(16'b0)
                );
                RM_IHPSG13_1P_4096x8_c3_bm_bist u_mac2 (
                    .A_CLK(clk), .A_MEN(a_men), .A_WEN(a_wen), .A_REN(a_ren),
                    .A_ADDR(a), .A_DIN(d[7:0]), .A_DLY(1'b1),
                    .A_DOUT(dout_w8_0), .A_BM(bm[7:0]),
                    .A_BIST_CLK(1'b0), .A_BIST_EN(1'b0), .A_BIST_MEN(1'b0),
                    .A_BIST_WEN(1'b0), .A_BIST_REN(1'b0),
                    .A_BIST_ADDR(12'b0), .A_BIST_DIN(8'b0), .A_BIST_BM(8'b0)
                );
                assign dout = {dout_w16_0, dout_w16_1, dout_w8_0};
            end

            // (72, 10): iact_stream_constructor = 1 x 64 + 1 x 8
            3: begin : g_cfg1024x72
                RM_IHPSG13_1P_1024x64_c2_bm_bist u_mac0 (
                    .A_CLK(clk), .A_MEN(a_men), .A_WEN(a_wen), .A_REN(a_ren),
                    .A_ADDR(a), .A_DIN(d[71:8]), .A_DLY(1'b1),
                    .A_DOUT(dout_w64_0), .A_BM(bm[71:8]),
                    .A_BIST_CLK(1'b0), .A_BIST_EN(1'b0), .A_BIST_MEN(1'b0),
                    .A_BIST_WEN(1'b0), .A_BIST_REN(1'b0),
                    .A_BIST_ADDR(10'b0), .A_BIST_DIN(64'b0), .A_BIST_BM(64'b0)
                );
                RM_IHPSG13_1P_1024x8_c2_bm_bist u_mac1 (
                    .A_CLK(clk), .A_MEN(a_men), .A_WEN(a_wen), .A_REN(a_ren),
                    .A_ADDR(a), .A_DIN(d[7:0]), .A_DLY(1'b1),
                    .A_DOUT(dout_w8_0), .A_BM(bm[7:0]),
                    .A_BIST_CLK(1'b0), .A_BIST_EN(1'b0), .A_BIST_MEN(1'b0),
                    .A_BIST_WEN(1'b0), .A_BIST_REN(1'b0),
                    .A_BIST_ADDR(10'b0), .A_BIST_DIN(8'b0), .A_BIST_BM(8'b0)
                );
                assign dout = {dout_w64_0, dout_w8_0};
            end

            default: begin : g_err
                // unsupported configuration: keep q at X so any use fails loudly
                assign dout = {DataWidth{1'bx}};
            end
        endcase
    endgenerate

    generate
        if (Pipelined) begin : g_pipe
            reg [DataWidth-1:0] q_r;
            always @(posedge clk) q_r <= dout;
            assign q = q_r;
        end else begin : g_nopipe
            assign q = dout;
        end
    endgenerate

endmodule
