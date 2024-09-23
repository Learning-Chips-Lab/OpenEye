// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: SPad_SP_RW
/// 
/// (S)cratchPad_(D)ual(P)ort is a wrapper module to provide a unified interface
/// for a dual port scratchpad memory that hides the underlying implementation
/// details. The module is used to store data words and read them out
/// simultaneously. As this is a true dual port, there are two types of
/// input/output pairs (A and B). Important notice: When writing or reading
/// ports are 1, their address ports need different values.
///
/// Parameters:
///    ADDR_WIDTH             - Amount of storageable data words in clog2
///    DATA_WIDTH             - Bitwidth of data words
///    Implementation         - Select the implementation of the SPAD
///   
/// Ports:
///    re_a_i           - Read enable port A
///    re_b_i           - Read enable port B
///    we_a_i           - Write enable port A
///    we_b_i           - Write enable port B
///    addr_r_a_i       - Address port for `data_a_o`
///    addr_r_b_i       - Address port for `data_b_o`
///    addr_w_a_i       - Address port for `data_a_i`
///    addr_w_b_i       - Address port for `data_b_i`
///    data_a_i         - Data port in A
///    data_b_i         - Data port in B
///    data_a_o         - Data port out A
///    data_b_o         - Data port out B
///

module SPad_DP_RW 
#( 
  parameter DATA_WIDTH = 8,
  parameter ADDR_WIDTH = 10,
  
  // Hint for physical implementation
  parameter Implementation = 0
) ( 
  input  wire                       clk_i,

  /*
   * the DPRW spad is implemented in standard cells which makes
   * it resettable in theory.
   * To keep the interface opaque, we still don't allow it
   */
  //input                           rst_ni,
  
  input  wire                       re_a_i,
  input  wire                       re_b_i,

  input  wire                       we_a_i,
  input  wire                       we_b_i,

  input  wire   [ADDR_WIDTH-1:0]    addr_r_a_i,
  input  wire   [ADDR_WIDTH-1:0]    addr_r_b_i,
  input  wire   [ADDR_WIDTH-1:0]    addr_w_a_i,
  input  wire   [ADDR_WIDTH-1:0]    addr_w_b_i,

  input  wire   [DATA_WIDTH-1:0]    data_a_i,
  input  wire   [DATA_WIDTH-1:0]    data_b_i,

  output wire   [DATA_WIDTH-1:0]    data_a_o,
  output wire   [DATA_WIDTH-1:0]    data_b_o
);


  RAM_DP_RW 
  #(
    .AddrWidth        (ADDR_WIDTH),
    .DataWidth        (DATA_WIDTH),
    .Pipelined        (0)
  ) ram (
    .clk_i            (clk_i),
    .re_a_i           (re_a_i),
    .re_b_i           (re_b_i),
    .we_a_i           (we_a_i),
    .we_b_i           (we_b_i),
    .addr_r_a_i       (addr_r_a_i),
    .addr_r_b_i       (addr_r_b_i),
    .addr_w_a_i       (addr_w_a_i),
    .addr_w_b_i       (addr_w_b_i),
    .data_a_i         (data_a_i),
    .data_b_i         (data_b_i),
    .data_a_o         (data_a_o),
    .data_b_o         (data_b_o)
  );
endmodule
