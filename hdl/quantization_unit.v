// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

// -----------------------------------------------------------------------
// quantization_unit_pipelined
//
// Pipelined version of quantization_unit. Computes:
//
//   quantized_value = (quant_mant * (psum_data + quant_offset)) >>> shift
//
// over 3 clock cycles instead of one big combinational path:
//
//   Stage 1 (S1): register inputs, compute (psum_data + quant_offset)
//   Stage 2 (S2): register sum, compute multiplication with quant_mant
//   Stage 3 (S3): register product, apply arithmetic right-shift -> output
//
// Splitting add / multiply / shift into separate stages shortens the
// critical path (helpful once TRANS_WORDS instances run in parallel)
// at the cost of a fixed 3-cycle latency and a valid_out handshake.
//
// Intermediate widths are widened by one bit for the add stage and by
// MANT_WIDTH bits for the multiply stage to avoid overflow.
// -----------------------------------------------------------------------
module quantization_unit #(
    parameter PSUM_WIDTH   = 32,   // width of one PSUM word (psum_buffer_data_r slice)
    parameter MANT_WIDTH   = 8,    // width of quant_mant
    parameter OFFSET_WIDTH = 16,   // width of quant_offset
    parameter SHIFT_WIDTH  = 5,    // width of current_shift
    parameter OUT_WIDTH    = 8    // width of quantized_value
) (
    input  wire                           clk_i,
    input  wire                           rst_n,          // asynchronous, active-low reset

    input  wire signed [MANT_WIDTH-1:0]   quant_mant,      // quantization multiplier (per filter)
    input  wire signed [PSUM_WIDTH-1:0]   psum_data,       // raw accumulated PSUM word
    input  wire signed [OFFSET_WIDTH-1:0] quant_offset,    // quantization offset (per filter)
    input  wire        [SHIFT_WIDTH-1:0]  current_shift,   // arithmetic right-shift amount

    output reg  signed [OUT_WIDTH-1:0]    quantized_value // quantized result, valid when valid_out = 1
);

    // Internal widths sized to avoid overflow in the intermediate steps.
    localparam SUM_WIDTH  = ((PSUM_WIDTH > OFFSET_WIDTH) ? PSUM_WIDTH : OFFSET_WIDTH) + 1;
    localparam PROD_WIDTH = SUM_WIDTH + MANT_WIDTH;

    // ---------------------------------------------------------------
    // Stage 1: sum = psum_data + quant_offset
    // ---------------------------------------------------------------
    reg signed [SUM_WIDTH-1:0]   sum_s1;
    reg signed [MANT_WIDTH-1:0]  mant_s1;
    reg        [SHIFT_WIDTH-1:0] shift_s1;
    reg                          valid_s1;

    always @(posedge clk_i or negedge rst_n) begin
        if (!rst_n) begin
            sum_s1   <= {SUM_WIDTH{1'b0}};
            mant_s1  <= {MANT_WIDTH{1'b0}};
            shift_s1 <= {SHIFT_WIDTH{1'b0}};
        end else begin
            sum_s1   <= psum_data + quant_offset;
            mant_s1  <= quant_mant;
            shift_s1 <= current_shift;
        end
    end

    // ---------------------------------------------------------------
    // Stage 2: prod = mant_s1 * sum_s1
    // ---------------------------------------------------------------
    reg signed [PROD_WIDTH-1:0]  prod_s2;
    reg        [SHIFT_WIDTH-1:0] shift_s2;
    reg                          valid_s2;

    always @(posedge clk_i or negedge rst_n) begin
        if (!rst_n) begin
            prod_s2  <= {PROD_WIDTH{1'b0}};
            shift_s2 <= {SHIFT_WIDTH{1'b0}};
            valid_s2 <= 1'b0;
        end else begin
            prod_s2  <= mant_s1 * sum_s1;
            shift_s2 <= shift_s1;
        end
    end

    // ---------------------------------------------------------------
    // Stage 3: quantized_value = prod_s2 >>> shift_s2
    // ---------------------------------------------------------------
    always @(posedge clk_i or negedge rst_n) begin
        if (!rst_n) begin
            quantized_value <= {OUT_WIDTH{1'b0}};
        end else begin
            quantized_value <= prod_s2 >>> shift_s2;
        end
    end

endmodule