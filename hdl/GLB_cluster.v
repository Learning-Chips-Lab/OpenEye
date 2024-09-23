// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: GLB_cluster
///
/// The GLB_Cluster (GLobal_Buffer_Cluster) contains the memory for the partial sums / biases and
/// input activations. With the router Cluster to configure the accessibility between other clusters
/// and the process element arrays. can be configured. The data to be computed is stored in the RAMs
/// (SinglePort) and is iterated through the over. Weights are pushed directly to the process elements.
///
///
/// Parameters:
///    DATA_IACT_BITWIDTH         - Width of input activation data word
///    DATA_WGHT_BITWIDTH         - Width of weight data word
///    DATA_PSUM_BITWIDTH         - Width of partial sum data word
///    TRANS_BITWIDTH_IACT        - Width of iact input port and RAM
///    TRANS_BITWIDTH_WGHT        - Width of weight input port
///    TRANS_BITWIDTH_PSUM        - Width of partial sum input port and RAM
///    NUM_GLB_IACT               - Number of input activation global buffers
///    NUM_GLB_WGHT               - Number of rows of PEs in cluster (No WGHT GLBs)
///    NUM_GLB_PSUM               - Number of partial sum global buffers
///    IACT_MEM_ADDR_WORDS        - Number of words in IACT GLB
///    PSUM_MEM_ADDR_WORDS        - Number of words in PSUM GLB
///    IACT_MEM_ADDR_BITS         - Width of words in IACT GLB
///    PSUM_MEM_ADDR_BITS         - Width of words in PSUM GLB
/// 
/// Ports:
///    data_write_enable_i        - Wether data should be read or written to RAMs. 1 is WR, 0 is RD
///
///    ext_mem_iact_addr_i        - Set addr port for IACT GLB
///    ext_mem_iact_data_i        - Data from top modul, IACT
///    ext_mem_iact_enable_i      - Enable from top modul, IACT
///    ext_mem_iact_ready_o       - Ready to top modul, IACT
///    ext_mem_wght_data_i        - Data from top modul, WGHT
///    ext_mem_wght_enable_i      - Enable from top modul, WGHT
///    ext_mem_wght_ready_o       - Ready to top modul, WGHT
///    ext_mem_psum_addr_i        - Set addr port for PSUM GLB
///    ext_mem_psum_data_i        - Data from top modul, PSUM
///    ext_mem_psum_enable_i      - Enable from top modul, PSUM
///    ext_mem_psum_ready_o       - Ready to top modul, PSUM
///    ext_mem_psum_data_o        - Data to top modul, PSUM
///    ext_mem_psum_enable_o      - Enable to top modul, PSUM
///    ext_mem_psum_ready_i       - Ready from top modul, PSUM
///
///    router_cluster_iact_data_o   - Data connection from IACT Cluster
///    router_cluster_iact_enable_o - Enable connection from IACT Cluster
///    router_cluster_iact_ready_i  - Ready connection to IACT Cluster
///    router_cluster_wght_data_o   - Data connection from WGHT Cluster
///    router_cluster_wght_enable_o - Enable connection from WGHT Cluster
///    router_cluster_wght_ready_i  - Ready connection to WGHT Cluster
///    router_cluster_psum_data_o   - Data connection from PSUM Cluster
///    router_cluster_psum_enable_o - Enable connection from PSUM Cluster
///    router_cluster_psum_ready_i  - Ready connection to PSUM Cluster
///    router_cluster_psum_data_i   - Data connection to PSUM Cluster
///    router_cluster_psum_enable_i - Enable connection to PSUM Cluster
///    router_cluster_psum_ready_o  - Ready connection from PSUM Cluster
///

module GLB_cluster 
#( 
  parameter integer DATA_IACT_BITWIDTH  = 8,
  parameter integer DATA_WGHT_BITWIDTH  = 8,
  parameter integer DATA_PSUM_BITWIDTH  = 20,

  parameter integer TRANS_BITWIDTH_IACT = 24,
  parameter integer TRANS_BITWIDTH_WGHT = 24,
  parameter integer TRANS_BITWIDTH_PSUM = 40,
    
  parameter integer NUM_GLB_IACT        = 3,
  parameter integer NUM_GLB_WGHT        = 3,
  parameter integer NUM_GLB_PSUM        = 4,

  parameter integer IACT_MEM_ADDR_WORDS = 512,
  parameter integer PSUM_MEM_ADDR_WORDS = 384,

  parameter integer IACT_MEM_ADDR_BITS  = $clog2(IACT_MEM_ADDR_WORDS),
  parameter integer PSUM_MEM_ADDR_BITS  = $clog2(PSUM_MEM_ADDR_WORDS)
) (
  input                             clk_i,
  input                             rst_ni,
  input                             data_write_enable_iact_i,
  input                             data_write_enable_i,
  ///Ext. Memory
  /////////////////////////////////////////
  ///IACT
  input  [DATA_IACT_BITWIDTH*NUM_GLB_IACT-1:0] ext_mem_iact_data_i,
  input  [IACT_MEM_ADDR_BITS*NUM_GLB_IACT-1:0] ext_mem_iact_addr_i,
  input  [NUM_GLB_IACT-1:0]                    ext_mem_iact_enable_i,
  output [NUM_GLB_IACT-1:0]                    ext_mem_iact_ready_o,
  ///WGHT
  input  [DATA_WGHT_BITWIDTH*NUM_GLB_WGHT-1:0] ext_mem_wght_data_i,
  input  [NUM_GLB_WGHT-1:0]                    ext_mem_wght_enable_i,
  output [NUM_GLB_WGHT-1:0]                    ext_mem_wght_ready_o,
  ///PSUM
  output [DATA_PSUM_BITWIDTH*NUM_GLB_PSUM-1:0] ext_mem_psum_data_o,
  output [NUM_GLB_PSUM-1:0]                    ext_mem_psum_enable_o,
  input  [NUM_GLB_PSUM-1:0]                    ext_mem_psum_ready_i,

  input  [DATA_PSUM_BITWIDTH*NUM_GLB_PSUM-1:0] ext_mem_psum_data_i,
  input  [PSUM_MEM_ADDR_BITS*NUM_GLB_PSUM-1:0] ext_mem_psum_addr_i,
  input  [NUM_GLB_PSUM-1:0]                    ext_mem_psum_enable_i,
  output [NUM_GLB_PSUM-1:0]                    ext_mem_psum_ready_o,
  ///Router Cluster
  /////////////////////////////////////////
  ///IACT
  output [DATA_IACT_BITWIDTH*NUM_GLB_IACT-1:0] router_cluster_iact_data_o,
  output reg [NUM_GLB_IACT-1:0]                router_cluster_iact_enable_o,
  input  [NUM_GLB_IACT-1:0]                    router_cluster_iact_ready_i,
  ///WGHT
  output [DATA_WGHT_BITWIDTH*NUM_GLB_WGHT-1:0] router_cluster_wght_data_o,
  output [NUM_GLB_WGHT-1:0]                    router_cluster_wght_enable_o,
  input  [NUM_GLB_WGHT-1:0]                    router_cluster_wght_ready_i,
  ///PSUM-READ    
  output [DATA_PSUM_BITWIDTH*NUM_GLB_PSUM-1:0] router_cluster_psum_data_o,
  output reg [NUM_GLB_PSUM-1:0]                router_cluster_psum_enable_o,
  input  [NUM_GLB_PSUM-1:0]                    router_cluster_psum_ready_i,
  ///PSUM-WRITE
  input  [DATA_PSUM_BITWIDTH*NUM_GLB_PSUM-1:0] router_cluster_psum_data_i,
  input  [NUM_GLB_PSUM-1:0]                    router_cluster_psum_enable_i,
  output [NUM_GLB_PSUM-1:0]                    router_cluster_psum_ready_o

);

reg [NUM_GLB_IACT-1:0] router_cluster_iact_enable_o_delay_1;
reg [NUM_GLB_PSUM-1:0] router_cluster_psum_enable_o_delay_1;
reg [NUM_GLB_IACT-1:0] router_cluster_iact_enable_o_delay_2;
reg [NUM_GLB_PSUM-1:0] router_cluster_psum_enable_o_delay_2;

genvar glb_cluster,bit_counter;
generate
  ///IACT GLB Storages
  for(glb_cluster = 0; glb_cluster < NUM_GLB_IACT; glb_cluster = glb_cluster + 1) begin : gen_iact

    wire  [TRANS_BITWIDTH_IACT-1:0] iact_glb_data_in_w;
    wire  [TRANS_BITWIDTH_IACT-1:0] iact_glb_data_out_w;
    wire  [IACT_MEM_ADDR_BITS-1:0]  iact_glb_addr_in_w;
    wire                            iact_glb_we_in_w;

    RAM_SP #(
      .DataWidth(TRANS_BITWIDTH_IACT),
      .AddrWidth(IACT_MEM_ADDR_BITS),
      .Pipelined(1)

      ,.Implementation(2) // GLB_IACT = 2
    )iact_glb( 
      .clk_i(clk_i), 
      .rd_en_i(iact_glb_we_in_w & !data_write_enable_iact_i),
      .wr_en_i(iact_glb_we_in_w & data_write_enable_iact_i), 
      .addr_i(iact_glb_addr_in_w),
      .data_i(iact_glb_data_in_w),
      .data_o(iact_glb_data_out_w)
    );
  end

  ///PSUM GLB Storages
  for(glb_cluster = 0; glb_cluster < NUM_GLB_PSUM; glb_cluster = glb_cluster + 1) begin : gen_psum

        wire  [TRANS_BITWIDTH_PSUM-1:0] psum_glb_data_in_w;
        wire  [TRANS_BITWIDTH_PSUM-1:0] psum_glb_data_out_w;
        wire  [PSUM_MEM_ADDR_BITS-1:0]  psum_glb_addr_in_w;
        wire                            psum_glb_re_in_w;
        wire                            psum_glb_we_in_w;

    RAM_SP #(
      .DataWidth(TRANS_BITWIDTH_PSUM),
      .AddrWidth(PSUM_MEM_ADDR_BITS),
      .Pipelined(1)

      ,.Implementation(1) // GLB_PSUM = 1
    )psum_glb( 
      .clk_i(clk_i), 
      .rd_en_i(psum_glb_re_in_w),
      .wr_en_i(psum_glb_we_in_w), 
      .addr_i(psum_glb_addr_in_w),
      .data_i(psum_glb_data_in_w),
      .data_o(psum_glb_data_out_w)
    );
  end
endgenerate

assign router_cluster_wght_data_o = ext_mem_wght_data_i;
assign router_cluster_wght_enable_o = ext_mem_wght_enable_i;
assign ext_mem_wght_ready_o = router_cluster_wght_ready_i;

for(glb_cluster = 0; glb_cluster < NUM_GLB_IACT; glb_cluster = glb_cluster + 1) begin
  assign gen_iact[glb_cluster].iact_glb_we_in_w = ext_mem_iact_enable_i[glb_cluster];
  assign router_cluster_iact_enable_o_delay_1[glb_cluster] = ext_mem_iact_enable_i[glb_cluster] & (!data_write_enable_iact_i);
  for(bit_counter = 0; bit_counter < IACT_MEM_ADDR_BITS; bit_counter = bit_counter + 1) begin
    assign gen_iact[glb_cluster].iact_glb_addr_in_w[bit_counter] = ext_mem_iact_addr_i[IACT_MEM_ADDR_BITS*glb_cluster+bit_counter];
  end
  for(bit_counter = 0; bit_counter < DATA_IACT_BITWIDTH; bit_counter = bit_counter + 1) begin
    assign gen_iact[glb_cluster].iact_glb_data_in_w[bit_counter] = ext_mem_iact_data_i[DATA_IACT_BITWIDTH*glb_cluster+bit_counter];
    assign router_cluster_iact_data_o[DATA_IACT_BITWIDTH*glb_cluster+bit_counter] = gen_iact[glb_cluster].iact_glb_data_out_w[bit_counter];
  end
end

assign ext_mem_iact_ready_o = router_cluster_iact_ready_i;

for(glb_cluster = 0; glb_cluster < NUM_GLB_PSUM; glb_cluster = glb_cluster + 1) begin
  assign gen_psum[glb_cluster].psum_glb_re_in_w = (ext_mem_psum_enable_i[glb_cluster] & !data_write_enable_i);
  assign gen_psum[glb_cluster].psum_glb_we_in_w = (ext_mem_psum_enable_i[glb_cluster] & data_write_enable_i) | (router_cluster_psum_enable_i[glb_cluster] & (!data_write_enable_i));
  assign router_cluster_psum_enable_o_delay_1[glb_cluster] = ext_mem_psum_enable_i[glb_cluster] & (!data_write_enable_i);

  for(bit_counter = 0; bit_counter < PSUM_MEM_ADDR_BITS; bit_counter = bit_counter + 1) begin
    assign gen_psum[glb_cluster].psum_glb_addr_in_w[bit_counter] = ext_mem_psum_addr_i[PSUM_MEM_ADDR_BITS*glb_cluster+bit_counter];
  end
  for(bit_counter = 0; bit_counter < DATA_PSUM_BITWIDTH; bit_counter = bit_counter + 1) begin
    assign ext_mem_psum_data_o[DATA_PSUM_BITWIDTH*glb_cluster+bit_counter] = gen_psum[glb_cluster].psum_glb_data_out_w[bit_counter];
    assign router_cluster_psum_data_o[DATA_PSUM_BITWIDTH*glb_cluster+bit_counter] = gen_psum[glb_cluster].psum_glb_data_out_w[bit_counter];
    assign gen_psum[glb_cluster].psum_glb_data_in_w[bit_counter] = ext_mem_psum_data_i[DATA_PSUM_BITWIDTH*glb_cluster+bit_counter] | router_cluster_psum_data_i[DATA_PSUM_BITWIDTH*glb_cluster+bit_counter];
  end
end

assign ext_mem_psum_enable_o = router_cluster_psum_enable_i;


assign router_cluster_psum_ready_o = ext_mem_psum_ready_i;
assign ext_mem_psum_ready_o = router_cluster_psum_ready_i;

integer g;

always@(posedge clk_i, negedge rst_ni) begin
  if(!rst_ni)begin ///Reset
    router_cluster_iact_enable_o_delay_2 <= 0;
    router_cluster_iact_enable_o         <= 0;
    router_cluster_psum_enable_o_delay_2 <= 0;
    router_cluster_psum_enable_o         <= 0;
  end else begin
    ///Push Enable signals
    for(g = 0; g < NUM_GLB_IACT; g = g + 1)begin
      router_cluster_iact_enable_o_delay_2[g] <= router_cluster_iact_enable_o_delay_1[g];
      router_cluster_iact_enable_o[g] <= router_cluster_iact_enable_o_delay_2[g];
    end
    for(g = 0; g < NUM_GLB_PSUM; g = g + 1)begin
      router_cluster_psum_enable_o_delay_2[g] <= router_cluster_psum_enable_o_delay_1[g];
      router_cluster_psum_enable_o[g] <= router_cluster_psum_enable_o_delay_2[g];
    end
  end
end

endmodule
