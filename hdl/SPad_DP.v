// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: SPad_DP
///
/// (S)cratchPad_(D)ual(P)ort is a wrapper module to provide a unified interface
/// for a dual port scratchpad memory that hides the underlying implementation
/// details. The module is used to store data words and read them out 
/// simultaneously. Important Notice: When `we_i` and `re_i` are 1, `addr_w_i`
/// and `addr_r_i` need to be different.
///
/// Parameters:
///    ADDR_WIDTH             - Amount of storageable data words in clog2
///    DATA_WIDTH             - Bitwidth of data words
///    Implementation         - Select the implementation of the SPAD
///
/// Ports:
///    we_i             - Write enable port
///    re_i             - Read enable port
///    addr_w_i         - Address port for `data_i`
///    addr_r_i         - Address port for `data_o`
///    data_i           - Data port in
///    data_o           - Data port out
///

module SPad_DP
#( 
  parameter DATA_WIDTH = 8,
  parameter ADDR_WIDTH = 10,
  
  // Hint for physical implementation.
  parameter Implementation = 0
) ( 
  input  wire                     clk_i,

  /* 
   * Remove reset port to make it clear that the
   * contents of the spad cannot simply be reset.
   * This is to make it possible to implement this SPAD
   * as an SRAM.
   */
  //input                         rst_ni,
  
  input  wire                     re_i,
  input  wire                     we_i,
  input  wire  [ADDR_WIDTH-1:0]   addr_r_i,
  input  wire  [ADDR_WIDTH-1:0]   addr_w_i,
  input  wire  [DATA_WIDTH-1:0]   data_i,
  output wire  [DATA_WIDTH-1:0]   data_o
);

  RAM_DP #(
    .AddrWidth      (ADDR_WIDTH),
    .DataWidth      (DATA_WIDTH),
    .Pipelined      (0)
  ) ram (
    .clk_i          (clk_i),
    .rd_en_i        (re_i),
    .wr_en_i        (we_i),
    .addr_r_i       (addr_r_i),
    .addr_w_i       (addr_w_i),
    .data_i         (data_i),
    .data_o         (data_o)
  );
endmodule
