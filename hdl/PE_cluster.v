// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: PE_cluster
///
/// OpenEye Processing Element (PE) Cluster. The PE cluster connects the GLB, router and data
/// Multiplexer with the Processing Elements. Data is transferred using a hand-shake protocol.
/// The Multiplexer and Demultiplexer are configured by 'iact_choose_i' and 'psum_choose_i'. 
///
/// Parameters:
///   IS_TOPLEVEL            - Decides, wether modul is topmodul or not
///   PARALLEL_MACS          - Number of MAC operations that are performed in parallel
///   TOP_CLUSTER            - Decides, wether this cluster is the topmost cluster
///   DATA_IACT_BITWIDTH     - Width of input activation data
///   DATA_WGHT_BITWIDTH     - Width of weight data
///   DATA_PSUM_BITWIDTH     - Width of partial sum data, used in internal accumulator
///   DATA_IACT_IGNORE_ZEROS - Number of zeros that can be ignored in sparse input activation data
///   DATA_WGHT_IGNORE_ZEROS - Number of zeros that can be ignored in sparse weight data
///   TRANS_BITWIDTH_IACT    - Width of iact input port 
///   TRANS_BITWIDTH_WGHT    - Width of weight input port
///   TRANS_BITWIDTH_PSUM    - Width of partial sum input port
///   NUM_GLB_IACT           - Number of input activation global buffers
///   IACT_ADDR_WORDS        - Number of words in IACT ADDR SPAD
///   IACT_DATA_WORDS        - Number of words in IACT DATA SPAD
///   WGHT_ADDR_WORDS        - Number of words in WGHT ADDR SPAD
///   WGHT_DATA_WORDS        - Number of words in WGHT DATA SPAD
///   PSUM_WORDS             - Number of words in PSUM SPAD
///   PE_ROWS                - Amount of rows of process elements
///   PE_COLUMNS             - Amount of columns of process elements
///   PES                    - Amount of process elements
///   
/// Ports:
///   iact_choose_i           - Specify the iact MUX and DEMUX for DATA, ENABLE, READY
///   psum_choose_i           - Specify the psum MUX and DEMUX for DATA, ENABLE, READY
///   compute_i               - Trigger computation
///   pe_iact_data            - Input activation data (includes actual iact data + overhead for ignoring zeros + address)
///   pe_iact_enable          - Enable input activation data transfer
///   pe_iact_ready           - Ready signal for input activation data transfer
///   pe_wght_data            - Weight data (includes actual weight data + overhead for ignoring zeros + address)
///   pe_wght_enable          - Enable weight data transfer
///   pe_wght_ready           - Ready signal for weight data transfer
///   pe_psum_data_i          - Partial sum and bias input data
///   pe_psum_enable_i        - Enable partial sum data input transfer
///   pe_psum_ready_o         - Ready signal for partial sum data input transfer
///   pe_psum_data_o          - Partial sum output data
///   pe_psum_enable_o        - Enable partial sum data output transfer
///   pe_psum_ready_i         - Ready signal for partial sum data output transfer
///   pe_router_psum_data_i   - Partial sum and bias input data via PSUM Router
///   pe_router_psum_enable_i - Enable partial sum data input transfer via PSUM Router
///   pe_router_psum_ready_o  - Ready signal for partial sum data input transfer via PSUM Router
///   pe_router_psum_data_o   - Partial sum output data via PSUM Router
///   pe_router_psum_enable_o - Enable partial sum data output transfer via PSUM Router
///   pe_router_psum_ready_i  - Ready signal for partial sum data output transfer via PSUM Router
///   enable_stream_i         - Enable signal of data stream for parameters
///   data_stream_i           - Data stream for parameters
///

module PE_cluster 
#(
  //Set parameters
  parameter IS_TOPLEVEL                = 1,
  parameter PARALLEL_MACS              = 2,
  parameter TOP_CLUSTER                = 1,
  parameter DATA_IACT_BITWIDTH         = 8,
  parameter DATA_PSUM_BITWIDTH         = 20,
  parameter DATA_WGHT_BITWIDTH         = 8,
  parameter DATA_IACT_IGNORE_ZEROS     = 4,
  parameter DATA_WGHT_IGNORE_ZEROS     = 4,
  parameter TRANS_BITWIDTH_IACT        = 24,
  parameter TRANS_BITWIDTH_WGHT        = 24,
  parameter TRANS_BITWIDTH_PSUM        = 40,
  parameter NUM_GLB_IACT               = 3,
  parameter IACT_ADDR_WORDS            = 9,
  parameter IACT_DATA_WORDS            = 16,
  parameter WGHT_ADDR_WORDS            = 16,
  parameter WGHT_DATA_WORDS            = 96,
  parameter PSUM_WORDS                 = 32,
  parameter PE_ROWS                    = 3,
  parameter PE_COLUMNS                 = 4,
  localparam PES                       = PE_ROWS * PE_COLUMNS

) ( 
  input                                         clk_i,
  input                                         rst_ni,
  input  [$clog2(NUM_GLB_IACT)*PES-1:0]         iact_choose_i,
  input  [PE_COLUMNS-1:0]                       psum_choose_i,
  input  [PES-1:0]                              compute_i,
  
  input  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] pe_iact_data,
  input  [NUM_GLB_IACT-1:0]                     pe_iact_enable,
  output [NUM_GLB_IACT-1:0]                     pe_iact_ready,
  
  input  [TRANS_BITWIDTH_WGHT*PE_ROWS-1:0]      pe_wght_data,
  input  [PE_ROWS-1:0]                          pe_wght_enable,
  output [PE_ROWS-1:0]                          pe_wght_ready,
  
  input  [PE_COLUMNS-1:0]                       pe_psum_ready_i,
  input  [TRANS_BITWIDTH_PSUM*PE_COLUMNS-1:0]   pe_psum_data_i,
  input  [PE_COLUMNS-1:0]                       pe_psum_enable_i,

  output [PE_COLUMNS-1:0]                       pe_psum_ready_o,
  output [TRANS_BITWIDTH_PSUM*PE_COLUMNS-1:0]   pe_psum_data_o,
  output [PE_COLUMNS-1:0]                       pe_psum_enable_o,

  input  [PE_COLUMNS-1:0]                       pe_router_psum_ready_i,
  input  [TRANS_BITWIDTH_PSUM*PE_COLUMNS-1:0]   pe_router_psum_data_i,
  input  [PE_COLUMNS-1:0]                       pe_router_psum_enable_i,

  output [PE_COLUMNS-1:0]                       pe_router_psum_ready_o,
  output [TRANS_BITWIDTH_PSUM*PE_COLUMNS-1:0]   pe_router_psum_data_o,
  output [PE_COLUMNS-1:0]                       pe_router_psum_enable_o,

  input                                         enable_stream_i,
  input  [7:0]                                  data_stream_i
);

///#######################
///Reset synchronization
///#######################
wire rst_n;

RST_SYNC rst_sync_pe_cluster (
  .clk_i        (clk_i),
  .rst_ni       (rst_ni),
  .rst_no       (rst_n)
);

wire [PE_ROWS*PE_COLUMNS-1:0]                 wght_ready_temp;
wire [PE_COLUMNS*PE_ROWS*NUM_GLB_IACT-1:0]    iact_ready_temp;

genvar i,j,g;
  generate 
    for(i=0; i<PE_COLUMNS; i=i+1) begin : gen_X
      for(j=0; j<PE_ROWS; j=j+1) begin : gen_Y
        wire [TRANS_BITWIDTH_PSUM-1 : 0] psum_data_i_w;
        wire                             psum_enable_i_w;
        wire                             psum_ready_i_w;
        wire [TRANS_BITWIDTH_PSUM-1 : 0] psum_data_o_w;
        wire                             psum_enable_o_w;
        wire                             psum_ready_o_w;
        wire [NUM_GLB_IACT-1:0]          iact_ready_o_w;
        wire                             wght_ready_o_w;
          PE #(
            .IS_TOPLEVEL(0),
            .PARALLEL_MACS(PARALLEL_MACS),
            .DATA_IACT_BITWIDTH(DATA_IACT_BITWIDTH),
            .DATA_WGHT_BITWIDTH(DATA_WGHT_BITWIDTH),
            .DATA_PSUM_BITWIDTH(DATA_PSUM_BITWIDTH),
            .DATA_IACT_IGNORE_ZEROS(DATA_IACT_IGNORE_ZEROS),
            .DATA_WGHT_IGNORE_ZEROS(DATA_WGHT_IGNORE_ZEROS),
            .IACT_DATA_ADDR(IACT_DATA_WORDS),
            .IACT_ADDR_ADDR(IACT_ADDR_WORDS),
            .WGHT_DATA_ADDR(WGHT_DATA_WORDS),
            .WGHT_ADDR_ADDR(WGHT_ADDR_WORDS),
            .PSUM_ADDR(PSUM_WORDS),
            .TRANS_BITWIDTH_IACT(TRANS_BITWIDTH_IACT),
            .TRANS_BITWIDTH_WGHT(TRANS_BITWIDTH_WGHT),
            .NUM_GLB_IACT(NUM_GLB_IACT)
          )pe(
            .clk_i(clk_i),
            .rst_ni(rst_n),
            .iact_select_i(iact_choose_i[(i+j*PE_COLUMNS+1)*$clog2(NUM_GLB_IACT)-1:(i+j*PE_COLUMNS)*$clog2(NUM_GLB_IACT)]),
            .compute_i(compute_i[i+j*PE_COLUMNS]),

            .iact_data_i(pe_iact_data),
            .iact_enable_i(pe_iact_enable),
            .iact_ready_o(iact_ready_o_w),

            .wght_data_i(pe_wght_data[(j+1)*TRANS_BITWIDTH_WGHT-1:j*TRANS_BITWIDTH_WGHT]),
            .wght_enable_i(pe_wght_enable[j]),
            .wght_ready_o(wght_ready_o_w),

            .psum_data_i(psum_data_i_w),
            .psum_enable_i(psum_enable_i_w),
            .psum_ready_o(psum_ready_o_w),
            .psum_data_o(psum_data_o_w),
            .psum_enable_o(psum_enable_o_w),
            .psum_ready_i(psum_ready_i_w),
            .enable_stream_i(enable_stream_i),
            .data_stream_i(data_stream_i)
          );
      end
    end
  
  // MUX and DEMUX for PSUM Signals
  // MUX PSUM DATA
  genvar k;
  for(k=0; k<PE_COLUMNS; k=k+1) begin : psum_data_mux
    wire [TRANS_BITWIDTH_PSUM-1 : 0] out_w;
    mux2 #(
        .DATA_WIDTH(TRANS_BITWIDTH_PSUM)
    )psum_mux(
        .a_in (pe_router_psum_data_i[(k+1)*TRANS_BITWIDTH_PSUM-1:k*TRANS_BITWIDTH_PSUM]),
        .b_in (pe_psum_data_i[(k+1)*TRANS_BITWIDTH_PSUM-1:k*TRANS_BITWIDTH_PSUM]),
        .sel_i(psum_choose_i[k]),
        .y_o  (out_w)
    );
  end
  /// MUX PSUM ENABLE
  for(k=0; k<PE_COLUMNS; k=k+1) begin : psum_enable_mux
    wire out_w;
    mux2 #(
        .DATA_WIDTH(1)
    )psum_mux(
        .a_in (pe_router_psum_enable_i[k]),
        .b_in (pe_psum_enable_i[k]),
        .sel_i(psum_choose_i[k]),
        .y_o  (out_w)
    );
  end
  /// DEMUX PSUM READY
  for(k=0; k<PE_COLUMNS; k=k+1) begin : psum_ready_demux
    wire in_w;
    demux2 #(
        .DATA_WIDTH(1)
    )psum_demux(
        .a_out (pe_router_psum_ready_o[k]),
        .b_out (pe_psum_ready_o[k]),
        .sel_i (psum_choose_i[k]),
        .i     (in_w),
        .clk_i (clk_i),
        .rst_ni(rst_ni)
    );
  end

  /// PE Connections
  for(i=0; i<PE_COLUMNS; i=i+1) begin
    for(j=0; j<PE_ROWS; j=j+1) begin

    for(g=0; g<NUM_GLB_IACT; g=g+1) begin
      assign  iact_ready_temp [i+j*PE_COLUMNS+PES*g]= gen_X[i].gen_Y[j].iact_ready_o_w[g];
    end

    assign  wght_ready_temp [i+PE_COLUMNS*j]= gen_X[i].gen_Y[j].wght_ready_o_w;

      if(j == 0)begin
        if (PE_ROWS != 1) begin /// Check, wether there are more than one row
          assign gen_X[i].gen_Y[j+1].psum_ready_i_w = gen_X[i].gen_Y[j].psum_ready_o_w;
        end else begin
          assign psum_ready_demux[i].in_w = gen_X[i].gen_Y[j].psum_ready_o_w;
          assign gen_X[i].gen_Y[j].psum_enable_i_w = psum_enable_mux[i].out_w;
          assign gen_X[i].gen_Y[j].psum_data_i_w = psum_data_mux[i].out_w;
        end

        assign gen_X[i].gen_Y[j].psum_ready_i_w = ((TOP_CLUSTER == 1) ? pe_router_psum_ready_i[j] : pe_router_psum_ready_i[j] | pe_psum_ready_i[j]);
        assign pe_psum_enable_o[i]        = gen_X[i].gen_Y[j].psum_enable_o_w;
        assign pe_psum_data_o[(i+1)*TRANS_BITWIDTH_PSUM-1:i*TRANS_BITWIDTH_PSUM]= gen_X[i].gen_Y[j].psum_data_o_w;
        assign pe_router_psum_enable_o[i] = gen_X[i].gen_Y[j].psum_enable_o_w;
        assign pe_router_psum_data_o[(i+1)*TRANS_BITWIDTH_PSUM-1:i*TRANS_BITWIDTH_PSUM] = gen_X[i].gen_Y[j].psum_data_o_w;
      end else begin
        if(j == PE_ROWS - 1)begin /// Check wether os bottom row
          assign psum_ready_demux[i].in_w = gen_X[i].gen_Y[j].psum_ready_o_w;
          assign gen_X[i].gen_Y[j-1].psum_enable_i_w = gen_X[i].gen_Y[j].psum_enable_o_w;
          assign gen_X[i].gen_Y[j-1].psum_data_i_w = gen_X[i].gen_Y[j].psum_data_o_w;

          assign gen_X[i].gen_Y[j].psum_enable_i_w = psum_enable_mux[i].out_w;
          assign gen_X[i].gen_Y[j].psum_data_i_w = psum_data_mux[i].out_w;
        end else begin
          assign gen_X[i].gen_Y[j+1].psum_ready_i_w = gen_X[i].gen_Y[j].psum_ready_o_w;
          assign gen_X[i].gen_Y[j-1].psum_enable_i_w = gen_X[i].gen_Y[j].psum_enable_o_w;
          assign gen_X[i].gen_Y[j-1].psum_data_i_w = gen_X[i].gen_Y[j].psum_data_o_w;
        end
      end
    end
  end
  ///Connect ready ports
  for(g=0; g<NUM_GLB_IACT; g=g+1) begin
    assign pe_iact_ready[g] = ((~iact_ready_temp[(g+1)*PES-1:g*PES]) == {PES{1'b0}});
  end
  for(g=0; g<PE_ROWS; g=g+1) begin
    assign pe_wght_ready[g] = ((~wght_ready_temp[g*PE_ROWS+PE_COLUMNS-1:g*PE_ROWS]) == {PE_COLUMNS{1'b0}});
  end

  endgenerate

endmodule