// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: PE_IO_debug
///
/// Debug wrapper for the PE module that exposes internal data signals decomposed into
/// separate components for easier debugging and signal probing.
///
/// This wrapper redirects all parameters, inputs and outputs to the PE module while
/// providing internal signals that decompose the packed data buses:
/// - iact_data: 3*8 bit signals (24 bits total)
/// - wght_data: 3*8 bit signals (24 bits total)
/// - psum_data: Decomposed into 8+4 bit signals for 12-bit values or 8-bit signals for 8-bit values

module PE_IO_debug #(
    parameter IS_TOPLEVEL = 1,
    parameter SERIAL      = 1,
    parameter CREATE_VCD  = 0,

    parameter integer PARALLEL_MACS = 2,

    parameter integer DATA_IACT_BITWIDTH     = 8,
    parameter integer DATA_WGHT_BITWIDTH     = 8,
    parameter integer DATA_PSUM_BITWIDTH     = 20,
    parameter integer DATA_IACT_OVERHEAD     = 4,
    parameter integer DATA_WGHT_IGNORE_ZEROS = 4,

    parameter integer IACT_DATA_ADDR = 16,
    parameter integer IACT_ADDR_ADDR = 9,

    parameter integer WGHT_DATA_ADDR = 96,
    parameter integer WGHT_ADDR_ADDR = 16,

    parameter integer PSUM_ADDR = 32,

    parameter integer TRANS_BITWIDTH_IACT = 24,
    parameter integer TRANS_BITWIDTH_WGHT = 24,

    parameter integer NUM_GLB_IACT = 3,

    // local parameters
    localparam integer IACT_ADDR_DATA = $clog2(IACT_DATA_ADDR),
    localparam integer WGHT_ADDR_DATA = $clog2(WGHT_DATA_ADDR),

    localparam integer IACT_ADDR_ADDR_BITWIDTH = $clog2(IACT_ADDR_ADDR),
    localparam integer IACT_ADDR_DATA_BITWIDTH = $clog2(IACT_ADDR_DATA),

    localparam integer IACT_DATA_DATA          = DATA_IACT_BITWIDTH + DATA_IACT_OVERHEAD,
    localparam integer IACT_DATA_ADDR_BITWIDTH = $clog2(IACT_DATA_ADDR),
    localparam integer IACT_DATA_DATA_BITWIDTH = $clog2(IACT_DATA_DATA),

    localparam integer WGHT_DATA_DATA          = (DATA_WGHT_BITWIDTH + DATA_WGHT_IGNORE_ZEROS) * PARALLEL_MACS,
    localparam integer WGHT_ADDR_ADDR_BITWIDTH = $clog2(WGHT_ADDR_ADDR),
    localparam integer WGHT_ADDR_DATA_BITWIDTH = $clog2(WGHT_ADDR_DATA),

    localparam integer WGHT_DATA_ADDR_BITWIDTH = $clog2(WGHT_DATA_ADDR),
    localparam integer WGHT_DATA_DATA_BITWIDTH = $clog2(WGHT_DATA_DATA),

    localparam integer PSUM_DATA = DATA_PSUM_BITWIDTH,
    localparam integer PSUM_ADDR_BITWIDTH = $clog2(PSUM_ADDR),
    localparam integer PSUM_DATA_BITWIDTH = $clog2(PSUM_DATA),
    localparam integer PSUM_WORDS_PER_TRANSFER = (SERIAL ? 1 : PARALLEL_MACS),
    localparam integer TRANS_BITWIDTH_PSUM = DATA_PSUM_BITWIDTH * PSUM_WORDS_PER_TRANSFER,
    localparam integer VALUES_OF_IACTS = $rtoi($ceil(TRANS_BITWIDTH_IACT / DATA_IACT_BITWIDTH))
) (
    input                                             clk_i,
    input                                             rst_ni,
    input      [          $clog2(NUM_GLB_IACT+1)-1:0] iact_select_i,
    input      [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] iact_data_i,
    input      [                    NUM_GLB_IACT-1:0] iact_enable_i,
    output     [                    NUM_GLB_IACT-1:0] iact_ready_o,
    input      [             TRANS_BITWIDTH_WGHT-1:0] wght_data_i,
    input                                             wght_enable_i,
    output                                            wght_ready_o,
    input      [             TRANS_BITWIDTH_PSUM-1:0] psum_data_i,
    input                                             psum_enable_i,
    output                                            psum_ready_o,
    output     [             TRANS_BITWIDTH_PSUM-1:0] psum_data_o,
    output                                            psum_enable_o,
    input                                             psum_ready_i,
    input                                             compute_i,
    input                                             enable_stream_i,
    input      [                                11:0] data_stream_i
);

  // ============================================================================
  // Internal Debug Signals - Decomposed Data Buses
  // ============================================================================

  // Decomposed iact_data signals for each global buffer
  // Each buffer has 24-bit data decomposed into 3x8-bit signals
  wire [7:0] iact_data_glb0_byte0;
  wire [7:0] iact_data_glb0_byte1;
  wire [7:0] iact_data_glb0_byte2;

  wire [7:0] iact_data_glb0_data0;
  wire [3:0] iact_data_glb0_addr0;
  wire [7:0] iact_data_glb0_data1;
  wire [3:0] iact_data_glb0_addr1;

  wire [7:0] iact_data_glb1_byte0;
  wire [7:0] iact_data_glb1_byte1;
  wire [7:0] iact_data_glb1_byte2;

  wire [7:0] iact_data_glb1_data0;
  wire [3:0] iact_data_glb1_addr0;
  wire [7:0] iact_data_glb1_data1;
  wire [3:0] iact_data_glb1_addr1;

  wire [7:0] iact_data_glb2_byte0;
  wire [7:0] iact_data_glb2_byte1;
  wire [7:0] iact_data_glb2_byte2;

  wire [7:0] iact_data_glb2_data0;
  wire [3:0] iact_data_glb2_addr0;
  wire [7:0] iact_data_glb2_data1;
  wire [3:0] iact_data_glb2_addr1;

  // Decomposed wght_data signals
  // 24-bit data decomposed into 3x8-bit signals
  wire [7:0] wght_data_byte0;
  wire [7:0] wght_data_byte1;
  wire [7:0] wght_data_byte2;

  wire [7:0] wght_data_data0;
  wire [3:0] wght_data_addr0;
  wire [7:0] wght_data_data1;
  wire [3:0] wght_data_addr1;

  // Decomposed psum_data_i/psum_data_o signals
  // For PARALLEL_MACS=2 and DATA_PSUM_BITWIDTH=20: 40 bits total (2x20-bit values)
  wire [19:0] psum_data_i_data0;
  wire [19:0] psum_data_i_data1;

  wire [19:0] psum_data_o_data0;
  wire [19:0] psum_data_o_data1;

  // Decomposed data_stream_i signal (12-bit decomposed into 8+4 bits)
  wire [7:0] data_stream_byte;
  wire [3:0] data_stream_nibble;

  // ============================================================================
  // Signal Decomposition Assignments
  // ============================================================================

  // IACT data decomposition - Global Buffer 0
  assign iact_data_glb0_byte0 = iact_data_i[7:0];
  assign iact_data_glb0_byte1 = iact_data_i[15:8];
  assign iact_data_glb0_byte2 = iact_data_i[23:16];

  // IACT data decomposition - Global Buffer 0, addresses and data
  assign iact_data_glb0_data0 = iact_data_i[7:0];
  assign iact_data_glb0_addr0 = iact_data_i[11:8];
  assign iact_data_glb0_data1 = iact_data_i[19:12];
  assign iact_data_glb0_addr1 = iact_data_i[23:20];

  // IACT data decomposition - Global Buffer 1
  assign iact_data_glb1_byte0 = iact_data_i[31:24];
  assign iact_data_glb1_byte1 = iact_data_i[39:32];
  assign iact_data_glb1_byte2 = iact_data_i[47:40];

  // IACT data decomposition - Global Buffer 1, addresses and data
  assign iact_data_glb1_data0 = iact_data_i[31:24];
  assign iact_data_glb1_addr0 = iact_data_i[35:32];
  assign iact_data_glb1_data1 = iact_data_i[39:32];
  assign iact_data_glb1_addr1 = iact_data_i[43:40];

  // IACT data decomposition - Global Buffer 2
  assign iact_data_glb2_byte0 = iact_data_i[55:48];
  assign iact_data_glb2_byte1 = iact_data_i[63:56];
  assign iact_data_glb2_byte2 = iact_data_i[71:64];

  // IACT data decomposition - Global Buffer 2, addresses and data
  assign iact_data_glb2_data0 = iact_data_i[55:48];
  assign iact_data_glb2_addr0 = iact_data_i[59:56];
  assign iact_data_glb2_data1 = iact_data_i[63:56];
  assign iact_data_glb2_addr1 = iact_data_i[67:64];

  // WGHT data decomposition
  assign wght_data_byte0 = wght_data_i[7:0];
  assign wght_data_byte1 = wght_data_i[15:8];
  assign wght_data_byte2 = wght_data_i[23:16];

  assign wght_data_data0 = wght_data_i[7:0];
  assign wght_data_addr0 = wght_data_i[11:8];
  assign wght_data_data1 = wght_data_i[19:12];
  assign wght_data_addr1 = wght_data_i[23:20];

  // PSUM input data decomposition
  assign psum_data_i_data0   = psum_data_i[19:0];
  assign psum_data_i_data1   = psum_data_i[39:20];

  // PSUM output data decomposition
  assign psum_data_o_data0   = psum_data_o[19:0];
  assign psum_data_o_data1   = psum_data_o[39:20];

  // Data stream decomposition
  assign data_stream_byte   = data_stream_i[7:0];
  assign data_stream_nibble = data_stream_i[11:8];

  // ============================================================================
  // PE Module Instantiation
  // ============================================================================

  PE #(
    .IS_TOPLEVEL(IS_TOPLEVEL),
    .SERIAL(SERIAL),
    .PARALLEL_MACS(PARALLEL_MACS),
    .DATA_IACT_BITWIDTH(DATA_IACT_BITWIDTH),
    .DATA_WGHT_BITWIDTH(DATA_WGHT_BITWIDTH),
    .DATA_PSUM_BITWIDTH(DATA_PSUM_BITWIDTH),
    .DATA_IACT_OVERHEAD(DATA_IACT_OVERHEAD),
    .DATA_WGHT_IGNORE_ZEROS(DATA_WGHT_IGNORE_ZEROS),
    .IACT_DATA_ADDR(IACT_DATA_ADDR),
    .IACT_ADDR_ADDR(IACT_ADDR_ADDR),
    .WGHT_DATA_ADDR(WGHT_DATA_ADDR),
    .WGHT_ADDR_ADDR(WGHT_ADDR_ADDR),
    .PSUM_ADDR(PSUM_ADDR),
    .TRANS_BITWIDTH_IACT(TRANS_BITWIDTH_IACT),
    .TRANS_BITWIDTH_WGHT(TRANS_BITWIDTH_WGHT),
    .NUM_GLB_IACT(NUM_GLB_IACT)
  ) pe_inst (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .iact_select_i(iact_select_i),
    .iact_data_i(iact_data_i),
    .iact_enable_i(iact_enable_i),
    .iact_ready_o(iact_ready_o),
    .wght_data_i(wght_data_i),
    .wght_enable_i(wght_enable_i),
    .wght_ready_o(wght_ready_o),
    .psum_data_i(psum_data_i),
    .psum_enable_i(psum_enable_i),
    .psum_ready_o(psum_ready_o),
    .psum_data_o(psum_data_o),
    .psum_enable_o(psum_enable_o),
    .psum_ready_i(psum_ready_i),
    .compute_i(compute_i),
    .enable_stream_i(enable_stream_i),
    .data_stream_i(data_stream_i)
  );

endmodule
