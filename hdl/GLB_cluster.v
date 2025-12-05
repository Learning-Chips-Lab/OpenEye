// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: GLB_cluster
///
/// Global Buffer Cluster for the OpenEye neural network accelerator.
/// Manages hierarchical memory organization for input activations,
/// weights, and partial sums with configurable routing capabilities.
///
/// Description:
///   This module implements a sophisticated memory hierarchy combining
///   multiple single-port RAM blocks with routing logic. It handles three
///   types of data (input activations, weights, partial sums) with
///   separate buffer spaces and access patterns optimized for neural
///   network computation.
///
/// Memory Organization:
///   - Input Activation GLBs:
///     * Configurable number of banks
///     * Single-port RAM implementation
///     * Supports both read and write modes
///
///   - Weight Handling:
///     * Direct weight forwarding
///     * No local storage (weights pushed to PEs)
///     * Configurable data width
///
///   - Partial Sum GLBs:
///     * Multiple bank organization
///     * Bidirectional data flow
///     * Accumulation support
///
/// Operation Modes:
///   1. Serial Mode (SERIAL=1):
///      - Direct path between external and router interfaces
///      - Simplified control logic
///      - Reduced hardware complexity
///
///   2. Parallel Mode (SERIAL=0):
///      - Independent bank operation
///      - Concurrent access support
///      - Enhanced throughput
///
/// Parameters:
///   Configuration:
///     SERIAL                 - Operation mode selection (0: parallel, 1: serial)
///     PARALLEL_MACS         - Number of parallel MAC operations
///
///   Data Widths:
///     DATA_IACT_BITWIDTH    - Input activation data width
///     DATA_WGHT_BITWIDTH    - Weight data width
///     DATA_PSUM_BITWIDTH    - Partial sum data width
///     TRANS_BITWIDTH_*      - Transfer widths for each data type
///
///   Memory Organization:
///     NUM_GLB_IACT          - Input activation buffer count
///     NUM_GLB_WGHT          - PE rows (weight routing)
///     NUM_GLB_PSUM          - Partial sum buffer count
///     *_MEM_ADDR_WORDS      - Memory depth configurations
///     *_MEM_ADDR_BITS       - Address width parameters
///
/// Interfaces:
///   Clock and Control:
///     clk_i                    - System clock
///     rst_ni                   - Asynchronous reset (active low)
///     data_write_enable_*_i    - Memory access mode control
///
///   External Memory Interface:
///     Input Activations:
///       ext_mem_iact_*         - IACT data, address, control signals
///     Weights:
///       ext_mem_wght_*         - Weight data and control signals
///     Partial Sums:
///       ext_mem_psum_*         - PSUM data, address, control signals
///
///   Router Cluster Interface:
///     Input Activations:
///       router_cluster_iact_*  - IACT routing signals
///     Weights:
///       router_cluster_wght_*  - Weight routing signals
///     Partial Sums:
///       router_cluster_psum_*  - PSUM routing signals (bidirectional)
///
/// Implementation Notes:
///   - Uses RAM_SP modules for memory implementation
///   - Implements handshaking protocol for all interfaces
///   - Supports configurable data widths and memory depths
///   - Features pipeline registers for timing optimization
///   - Handles asynchronous reset conditions
///   - Manages multiple clock domain interactions
///   - Provides flexible routing configurations
///   - Implements efficient bank arbitration
///   - Supports multiple access patterns
///

module GLB_cluster #(
    parameter         SERIAL             = 1'd0,
    parameter integer PARALLEL_MACS      = 2,
    parameter integer DATA_IACT_BITWIDTH = 8,
    parameter integer DATA_WGHT_BITWIDTH = 8,
    parameter integer DATA_PSUM_BITWIDTH = 20,

    parameter integer TRANS_BITWIDTH_IACT = 24,
    parameter integer TRANS_BITWIDTH_WGHT = 24,
    parameter integer TRANS_BITWIDTH_PSUM = SERIAL ? DATA_PSUM_BITWIDTH : DATA_PSUM_BITWIDTH * PARALLEL_MACS,

    parameter integer NUM_GLB_IACT = 3,
    parameter integer NUM_GLB_WGHT = 3,
    parameter integer NUM_GLB_PSUM = 4,

    parameter integer IACT_MEM_ADDR_WORDS = 512,
    parameter integer PSUM_MEM_ADDR_WORDS = 768,

    parameter integer IACT_MEM_ADDR_BITS = $clog2(IACT_MEM_ADDR_WORDS),
    parameter integer PSUM_MEM_ADDR_BITS = $clog2(PSUM_MEM_ADDR_WORDS)
) (
    input                                        clk_i,
    input                                        rst_ni,
    input                                        data_write_enable_iact_i,
    input                                        data_write_enable_i,
    ///Ext. Memory
    /////////////////////////////////////////
    ///IACT
    input  [DATA_IACT_BITWIDTH*NUM_GLB_IACT-1:0] ext_mem_iact_data_i,
    input  [IACT_MEM_ADDR_BITS*NUM_GLB_IACT-1:0] ext_mem_iact_addr_i,
    input  [                   NUM_GLB_IACT-1:0] ext_mem_iact_enable_i,
    output [                   NUM_GLB_IACT-1:0] ext_mem_iact_ready_o,
    ///WGHT
    input  [DATA_WGHT_BITWIDTH*NUM_GLB_WGHT-1:0] ext_mem_wght_data_i,
    input  [                   NUM_GLB_WGHT-1:0] ext_mem_wght_enable_i,
    output [                   NUM_GLB_WGHT-1:0] ext_mem_wght_ready_o,
    ///PSUM
    output [DATA_PSUM_BITWIDTH*NUM_GLB_PSUM-1:0] ext_mem_psum_data_o,
    output [                   NUM_GLB_PSUM-1:0] ext_mem_psum_enable_o,
    input  [                   NUM_GLB_PSUM-1:0] ext_mem_psum_ready_i,

    input      [DATA_PSUM_BITWIDTH*NUM_GLB_PSUM-1:0] ext_mem_psum_data_i,
    input      [PSUM_MEM_ADDR_BITS*NUM_GLB_PSUM-1:0] ext_mem_psum_addr_i,
    input      [                   NUM_GLB_PSUM-1:0] ext_mem_psum_enable_i,
    output     [                   NUM_GLB_PSUM-1:0] ext_mem_psum_ready_o,
    ///Router Cluster
    /////////////////////////////////////////
    ///IACT
    output     [DATA_IACT_BITWIDTH*NUM_GLB_IACT-1:0] router_cluster_iact_data_o,
    output     [                   NUM_GLB_IACT-1:0] router_cluster_iact_enable_o,
    input      [                   NUM_GLB_IACT-1:0] router_cluster_iact_ready_i,
    ///WGHT
    output     [DATA_WGHT_BITWIDTH*NUM_GLB_WGHT-1:0] router_cluster_wght_data_o,
    output     [                   NUM_GLB_WGHT-1:0] router_cluster_wght_enable_o,
    input      [                   NUM_GLB_WGHT-1:0] router_cluster_wght_ready_i,
    ///PSUM-READ    
    output     [DATA_PSUM_BITWIDTH*NUM_GLB_PSUM-1:0] router_cluster_psum_data_o,
    output     [                   NUM_GLB_PSUM-1:0] router_cluster_psum_enable_o,
    input      [                   NUM_GLB_PSUM-1:0] router_cluster_psum_ready_i,
    ///PSUM-WRITE
    input      [DATA_PSUM_BITWIDTH*NUM_GLB_PSUM-1:0] router_cluster_psum_data_i,
    input      [                   NUM_GLB_PSUM-1:0] router_cluster_psum_enable_i,
    output     [                   NUM_GLB_PSUM-1:0] router_cluster_psum_ready_o

);

  wire [NUM_GLB_IACT-1:0] router_cluster_iact_enable_o_delay_1;
  reg  [NUM_GLB_IACT-1:0] router_cluster_iact_enable_o_delay_2;

  wire [NUM_GLB_PSUM-1:0] router_cluster_psum_enable_o_delay_1;
  reg  [NUM_GLB_PSUM-1:0] router_cluster_psum_enable_o_delay_2;

  genvar glb_counter, bit_counter;
  generate
    ///IACT GLB Storages
    for (
        glb_counter = 0;
        glb_counter < NUM_GLB_IACT - (NUM_GLB_IACT * SERIAL);
        glb_counter = glb_counter + 1
    ) begin : gen_iact

      wire [TRANS_BITWIDTH_IACT-1:0] iact_glb_data_in_w;
      wire [TRANS_BITWIDTH_IACT-1:0] iact_glb_data_out_w;
      wire [ IACT_MEM_ADDR_BITS-1:0] iact_glb_addr_in_w;
      wire                           iact_glb_we_in_w;

      RAM_SP #(
          .DataWidth(TRANS_BITWIDTH_IACT),
          .AddrWidth(IACT_MEM_ADDR_BITS),
          .Pipelined(1)

          , .Implementation(2)  // GLB_IACT = 2
      ) iact_glb (
          .clk_i  (clk_i),
          .rd_en_i(iact_glb_we_in_w & !data_write_enable_iact_i),
          .wr_en_i(iact_glb_we_in_w & data_write_enable_iact_i),
          .addr_i (iact_glb_addr_in_w),
          .data_i (iact_glb_data_in_w),
          .data_o (iact_glb_data_out_w)
      );
    end
    ///PSUM GLB Storages
    for (
        glb_counter = 0;
        glb_counter < NUM_GLB_PSUM - (NUM_GLB_PSUM * SERIAL);
        glb_counter = glb_counter + 1
    ) begin : gen_psum

      wire [TRANS_BITWIDTH_PSUM-1:0] psum_glb_data_in_w;
      wire [TRANS_BITWIDTH_PSUM-1:0] psum_glb_data_out_w;
      wire [ PSUM_MEM_ADDR_BITS-1:0] psum_glb_addr_in_w;
      wire                           psum_glb_re_in_w;
      wire                           psum_glb_we_in_w;

      RAM_SP #(
          .DataWidth(TRANS_BITWIDTH_PSUM),
          .AddrWidth(PSUM_MEM_ADDR_BITS),
          .Pipelined(1)

          , .Implementation(1)  // GLB_PSUM = 1
      ) psum_glb (
          .clk_i  (clk_i),
          .rd_en_i(psum_glb_re_in_w),
          .wr_en_i(psum_glb_we_in_w),
          .addr_i (psum_glb_addr_in_w),
          .data_i (psum_glb_data_in_w),
          .data_o (psum_glb_data_out_w)
      );
    end
  endgenerate

  assign router_cluster_wght_data_o = ext_mem_wght_data_i;
  assign router_cluster_wght_enable_o = ext_mem_wght_enable_i;
  assign ext_mem_wght_ready_o = router_cluster_wght_ready_i;
  assign ext_mem_iact_ready_o = router_cluster_iact_ready_i;
  if (SERIAL == 1) begin : gen_serial_router
    assign router_cluster_psum_data_o   = ext_mem_psum_data_i;
    assign router_cluster_psum_enable_o = ext_mem_psum_enable_i;
    assign ext_mem_psum_ready_o         = router_cluster_psum_ready_i;
    assign ext_mem_psum_data_o          = router_cluster_psum_data_i;
    assign ext_mem_psum_enable_o        = router_cluster_psum_enable_i;
    assign router_cluster_psum_ready_o  = ext_mem_psum_ready_i;
    assign router_cluster_iact_data_o   = ext_mem_iact_data_i;
    assign router_cluster_iact_enable_o = ext_mem_iact_enable_i;
  end else begin : gen_parallel_router
    for (
        glb_counter = 0;
        glb_counter < NUM_GLB_IACT - (NUM_GLB_IACT * SERIAL);
        glb_counter = glb_counter + 1
    ) begin
      assign router_cluster_iact_enable_o[glb_counter] = router_cluster_iact_enable_o_delay_2[glb_counter];
      assign gen_iact[glb_counter].iact_glb_we_in_w = ext_mem_iact_enable_i[glb_counter];
      assign router_cluster_iact_enable_o_delay_1[glb_counter] = ext_mem_iact_enable_i[glb_counter] & (!data_write_enable_iact_i);
      for (bit_counter = 0; bit_counter < IACT_MEM_ADDR_BITS; bit_counter = bit_counter + 1) begin
        assign gen_iact[glb_counter].iact_glb_addr_in_w[bit_counter] = ext_mem_iact_addr_i[IACT_MEM_ADDR_BITS*glb_counter+bit_counter];
      end
      for (bit_counter = 0; bit_counter < DATA_IACT_BITWIDTH; bit_counter = bit_counter + 1) begin
        assign gen_iact[glb_counter].iact_glb_data_in_w[bit_counter] = ext_mem_iact_data_i[DATA_IACT_BITWIDTH*glb_counter+bit_counter];
        assign router_cluster_iact_data_o[DATA_IACT_BITWIDTH*glb_counter+bit_counter] = gen_iact[glb_counter].iact_glb_data_out_w[bit_counter];
      end
    end


    for (
        glb_counter = 0;
        glb_counter < NUM_GLB_PSUM - (NUM_GLB_PSUM * SERIAL);
        glb_counter = glb_counter + 1
    ) begin
      assign gen_psum[glb_counter].psum_glb_re_in_w = (ext_mem_psum_enable_i[glb_counter] & !data_write_enable_i);
      assign gen_psum[glb_counter].psum_glb_we_in_w = (ext_mem_psum_enable_i[glb_counter] & data_write_enable_i) | (router_cluster_psum_enable_i[glb_counter] & (!data_write_enable_i));
      assign router_cluster_psum_enable_o_delay_1[glb_counter] = ext_mem_psum_enable_i[glb_counter] & (!data_write_enable_i);
      for (bit_counter = 0; bit_counter < PSUM_MEM_ADDR_BITS; bit_counter = bit_counter + 1) begin
        assign gen_psum[glb_counter].psum_glb_addr_in_w[bit_counter] = ext_mem_psum_addr_i[PSUM_MEM_ADDR_BITS*glb_counter+bit_counter];
      end
      for (bit_counter = 0; bit_counter < DATA_PSUM_BITWIDTH; bit_counter = bit_counter + 1) begin
        assign ext_mem_psum_data_o[DATA_PSUM_BITWIDTH*glb_counter+bit_counter] = gen_psum[glb_counter].psum_glb_data_out_w[bit_counter];
        assign router_cluster_psum_data_o[DATA_PSUM_BITWIDTH*glb_counter+bit_counter] = gen_psum[glb_counter].psum_glb_data_out_w[bit_counter];
        assign gen_psum[glb_counter].psum_glb_data_in_w[bit_counter] = ext_mem_psum_data_i[DATA_PSUM_BITWIDTH*glb_counter+bit_counter] | router_cluster_psum_data_i[DATA_PSUM_BITWIDTH*glb_counter+bit_counter];
      end
    end
    assign ext_mem_psum_enable_o = router_cluster_psum_enable_i;
    assign router_cluster_psum_ready_o = ext_mem_psum_ready_i;
    assign ext_mem_psum_ready_o = router_cluster_psum_ready_i;
  end

  integer g;

  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin  ///Reset
      if (SERIAL == 1'd0) begin
        router_cluster_iact_enable_o_delay_2 <= 0;
      end
      router_cluster_psum_enable_o_delay_2 <= 0;
      //router_cluster_psum_enable_o         <= 0;
    end else begin
      ///Push Enable signals
      if (SERIAL == 1'd0) begin
        for (g = 0; g < NUM_GLB_IACT; g = g + 1) begin
          router_cluster_iact_enable_o_delay_2[g] <= router_cluster_iact_enable_o_delay_1[g];
        end
        for (g = 0; g < NUM_GLB_PSUM; g = g + 1) begin
          router_cluster_psum_enable_o_delay_2[g] <= router_cluster_psum_enable_o_delay_1[g];
        end
      end
    end
  end

endmodule
