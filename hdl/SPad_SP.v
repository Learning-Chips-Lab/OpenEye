// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: SPad_SP
///
/// (S)cratchPad_(S)ingle(P)ort is a wrapper module to provide a unified
/// interface for a dual port scratchpad memory that hides the underlying
/// implementation details.
///
/// Parameters:
///    ADDR_WIDTH             - Amount of storageable data words in clog2
///    DATA_WIDTH             - Length of data words
///    Implementation         - Select the implementation of the SPAD
///   
/// Ports:
///    we_i             - Write enable port
///    re_i             - Read enable port
///    addr_i           - Address port
///    data_i           - Data port in
///    data_o           - Data port out
///

module SPad_SP
#( 
  parameter DATA_WIDTH = 8,
  parameter ADDR_WIDTH = 10,
  
  // Hint for physical implementation.
  parameter Implementation = 0
) ( 
  input                           clk_i,

  // Remove reset port to make it clear that the
  // contents of the spad cannot simply be reset
  //input                         rst_ni,

  input                      re_i,
  input                      we_i,
  input   [ADDR_WIDTH-1:0]   addr_i,
  input   [DATA_WIDTH-1:0]   data_i,
  output  [DATA_WIDTH-1:0]   data_o
);
  
  RAM_SP #(
    .AddrWidth      (ADDR_WIDTH),
    .DataWidth      (DATA_WIDTH),
    .Pipelined      (0),
    
    .Implementation (Implementation)
  ) ram (
    .clk_i          (clk_i),
    .rd_en_i        (re_i),
    .wr_en_i        (we_i),
    .addr_i         (addr_i),
    .data_i         (data_i),
    .data_o         (data_o)
  );
  
endmodule
