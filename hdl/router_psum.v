// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: router_psum
///
/// The Partial Sum Router (router_psum) is a critical component in the OpenEye architecture
/// responsible for managing the flow of partial sum data between Processing Element (PE) 
/// clusters and the Global Buffer (GLB). This router implements a flexible routing topology
/// that supports various accumulation patterns needed for neural network computations.
///
/// Key Features:
/// - Three-Port Router Architecture:
///   * Port 0: GLB Interface (bidirectional for partial sum storage/retrieval)
///   * Port 1: Inter-cluster Connection (vertical routing between clusters)
///   * Port 2: PE Cluster Interface (partial sum input/output)
///
/// - Configurable Routing Modes:
///   * Single Cluster Mode: Direct PE-GLB communication
///   * Loop Configuration: Supports chained partial sum accumulation
///   * Multi-cluster Accumulation: Enables vertical partial sum aggregation
///
/// - Handshake-based Flow Control:
///   * Ready/Enable Protocol: Matches PE cluster and GLB interfaces
///   * Backpressure Support: Prevents data overflow
///   * Synchronization: Ensures reliable data transfer
///
/// Architectural Role:
/// The router facilitates efficient partial sum accumulation patterns by:
/// 1. Collecting partial sums from PE clusters
/// 2. Supporting vertical accumulation across multiple clusters
/// 3. Managing data flow to/from the Partial Sum GLB
/// 4. Enabling flexible accumulation schemes for different layer types
///
/// Parameters:
///    DATA_WIDTH          - Partial Sum Data Path Width
///                         Determines precision of partial sum values
///                         Typically wider than activation/weight paths
///                         to accommodate accumulation growth
///   
/// Ports:
/// Configuration Interface:
///    router_mode_i [2:0] - Router Configuration Control
///                         [2] GLB Direction Control:
///                           1: Accept data from GLB
///                           0: Send data to GLB
///                         [1:0] Cluster Position Configuration:
///                           00: Single Cluster Mode (no inter-cluster connection)
///                           01: Topmost Cluster in accumulation loop
///                           10: Middle Cluster in accumulation loop
///                           11: Bottom Cluster in accumulation loop
///                         Note: Mode '100' functions same as '000'
///
/// GLB Interface (Port 0):
///    ready_src_port_0   - GLB Source Ready Signal
///                         Indicates GLB can provide data
///    data_src_port_0    - GLB Source Data Bus [DATA_WIDTH-1:0]
///                         Partial sum data from GLB
///    enable_src_port_0  - GLB Source Valid Signal
///                         Marks valid data from GLB
///    ready_dst_port_0   - GLB Destination Ready Input
///                         Indicates GLB can accept data
///    data_dst_port_0    - GLB Destination Data [DATA_WIDTH-1:0]
///                         Partial sum data to GLB
///    enable_dst_port_0  - GLB Destination Valid
///                         Marks valid data to GLB
///
/// Inter-cluster Interface (Port 1):
///    ready_src_port_1   - Cluster Source Ready Signal
///                         Flow control from adjacent cluster
///    data_src_port_1    - Cluster Source Data [DATA_WIDTH-1:0]
///                         Partial sums from adjacent cluster
///    enable_src_port_1  - Cluster Source Valid Signal
///                         Valid data from adjacent cluster
///    ready_dst_port_1   - Cluster Destination Ready
///                         Flow control to adjacent cluster
///    data_dst_port_1    - Cluster Destination Data [DATA_WIDTH-1:0]
///                         Partial sums to adjacent cluster
///    enable_dst_port_1  - Cluster Destination Valid
///                         Valid data to adjacent cluster
///
/// PE Cluster Interface (Port 2):
///    ready_src_port_2   - PE Source Ready Signal
///                         Flow control from PE cluster
///    data_src_port_2    - PE Source Data Bus [DATA_WIDTH-1:0]
///                         Partial sums from PE cluster
///    enable_src_port_2  - PE Source Valid Signal
///                         Valid data from PE cluster
///    ready_dst_port_2   - PE Destination Ready
///                         Flow control to PE cluster
///    data_dst_port_2    - PE Destination Data [DATA_WIDTH-1:0]
///                         Partial sums to PE cluster
///    enable_dst_port_2  - PE Destination Valid
///                         Valid data to PE cluster

module router_psum #(
    parameter integer DATA_WIDTH = 20
) (
    input [2:0] router_mode_i,

    ///SRC Port 0
    output                  ready_src_port_0,
    input  [DATA_WIDTH-1:0] data_src_port_0,
    input                   enable_src_port_0,

    ///SRC Port 1
    output                  ready_src_port_1,
    input  [DATA_WIDTH-1:0] data_src_port_1,
    input                   enable_src_port_1,

    ///SRC Port 2
    output                  ready_src_port_2,
    input  [DATA_WIDTH-1:0] data_src_port_2,
    input                   enable_src_port_2,

    ///DST Port 0
    input                   ready_dst_port_0,
    output [DATA_WIDTH-1:0] data_dst_port_0,
    output                  enable_dst_port_0,

    ///DST Port 1
    input                   ready_dst_port_1,
    output [DATA_WIDTH-1:0] data_dst_port_1,
    output                  enable_dst_port_1,

    ///DST Port 2
    input                   ready_dst_port_2,
    output [DATA_WIDTH-1:0] data_dst_port_2,
    output                  enable_dst_port_2
);

  ///Status Signals in Router
  ////////////////////////////////////////
  wire [1:0] l_status;
  wire       h_status;

  assign {h_status, l_status} = router_mode_i;

  ///Signals in Router
  ////////////////////////////////////////

  wire e00;
  wire e01;
  wire e02;
  wire e10;
  wire e11;
  wire e12;
  wire e20;
  wire e21;
  wire e22;
  wire r00;
  wire r01;
  wire r02;
  wire r10;
  wire r11;
  wire r12;
  wire r20;
  wire r21;
  wire r22;

  ///Destination Port: Data
  ////////////////////////////////////////

  assign data_dst_port_0 = ({DATA_WIDTH{e00}}               & data_src_port_0)
                         | ({DATA_WIDTH{~e00 & e10}}        & data_src_port_1)
                         | ({DATA_WIDTH{~e00 & ~e10 & e20}} & data_src_port_2);

  assign data_dst_port_1 = ({DATA_WIDTH{e01}}               & data_src_port_0)
                         | ({DATA_WIDTH{~e01 & e11}}        & data_src_port_1)
                         | ({DATA_WIDTH{~e01 & ~e11 & e21}} & data_src_port_2);

  assign data_dst_port_2 = ({DATA_WIDTH{e02}}               & data_src_port_0)
                         | ({DATA_WIDTH{~e02 & e12}}        & data_src_port_1)
                         | ({DATA_WIDTH{~e02 & ~e12 & e22}} & data_src_port_2);

  ///Destination Port: Enable
  ////////////////////////////////////////

  assign enable_dst_port_0 = e00 | e10 | e20;
  assign enable_dst_port_1 = e01 | e11 | e21;
  assign enable_dst_port_2 = e02 | e12 | e22;

  ///Source Port: Ready
  ////////////////////////////////////////

  assign ready_src_port_0 = r01 & r02;
  assign ready_src_port_1 = r10 & r12;
  assign ready_src_port_2 = r20 & r21 & r22;

  ///Source Port: Enable
  ////////////////////////////////////////

  assign e00 = (router_mode_i[2:0] == 3'd0) ? 0 :
               (router_mode_i[2:0] == 3'd1) ? 0 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? 0 :
               (router_mode_i[2:0] == 3'd5) ? 0 :
               (router_mode_i[2:0] == 3'd6) ? 0 :
               (router_mode_i[2:0] == 3'd7) ? 0 : 0;
  assign e01 = (router_mode_i[2:0] == 3'd0) ? enable_src_port_0 :
               (router_mode_i[2:0] == 3'd1) ? 0 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? enable_src_port_0 :
               (router_mode_i[2:0] == 3'd5) ? 0 :
               (router_mode_i[2:0] == 3'd6) ? 0 :
               (router_mode_i[2:0] == 3'd7) ? enable_src_port_0 : 0;
  assign e02 = (router_mode_i[2:0] == 3'd0) ? 0 :
               (router_mode_i[2:0] == 3'd1) ? 0 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? 0 :
               (router_mode_i[2:0] == 3'd5) ? enable_src_port_0 :
               (router_mode_i[2:0] == 3'd6) ? enable_src_port_0 :
               (router_mode_i[2:0] == 3'd7) ? 0 : 0;

  assign e10 = (router_mode_i[2:0] == 3'd0) ? enable_src_port_1 :
               (router_mode_i[2:0] == 3'd1) ? 0 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? enable_src_port_1 :
               (router_mode_i[2:0] == 3'd5) ? enable_src_port_1 :
               (router_mode_i[2:0] == 3'd6) ? 0 :
               (router_mode_i[2:0] == 3'd7) ? 0 : 0;
  assign e11 = (router_mode_i[2:0] == 3'd0) ? 0 :
               (router_mode_i[2:0] == 3'd1) ? 0 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? 0 :
               (router_mode_i[2:0] == 3'd5) ? 0 :
               (router_mode_i[2:0] == 3'd6) ? 0 :
               (router_mode_i[2:0] == 3'd7) ? 0 : 0;
  assign e12 = (router_mode_i[2:0] == 3'd0) ? 0 :
               (router_mode_i[2:0] == 3'd1) ? enable_src_port_1 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? 0 :
               (router_mode_i[2:0] == 3'd5) ? 0 :
               (router_mode_i[2:0] == 3'd6) ? 0 :
               (router_mode_i[2:0] == 3'd7) ? 0 : 0;

  assign e20 = (router_mode_i[2:0] == 3'd0) ? 0 :
               (router_mode_i[2:0] == 3'd1) ? 0 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? 0 :
               (router_mode_i[2:0] == 3'd5) ? 0 :
               (router_mode_i[2:0] == 3'd6) ? enable_src_port_2 :
               (router_mode_i[2:0] == 3'd7) ? enable_src_port_2 : 0;
  assign e21 = (router_mode_i[2:0] == 3'd0) ? 0 :
               (router_mode_i[2:0] == 3'd1) ? 0 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? enable_src_port_2 :
               (router_mode_i[2:0] == 3'd4) ? 0 :
               (router_mode_i[2:0] == 3'd5) ? 0 :
               (router_mode_i[2:0] == 3'd6) ? 0 :
               (router_mode_i[2:0] == 3'd7) ? 0 : 0;
  assign e22 = (router_mode_i[2:0] == 3'd0) ? 0 :
               (router_mode_i[2:0] == 3'd1) ? 0 :
               (router_mode_i[2:0] == 3'd2) ? enable_src_port_2 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? 0 :
               (router_mode_i[2:0] == 3'd5) ? 0 :
               (router_mode_i[2:0] == 3'd6) ? 0 :
               (router_mode_i[2:0] == 3'd7) ? 0 : 0;

  ///Destination Port: Ready
  ////////////////////////////////////////

  assign r00 = (router_mode_i[2:0] == 3'd0) ? 1 :
               (router_mode_i[2:0] == 3'd1) ? 0 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? 1 :
               (router_mode_i[2:0] == 3'd5) ? 1 :
               (router_mode_i[2:0] == 3'd6) ? 1 :
               (router_mode_i[2:0] == 3'd7) ? 1 : 0;
  assign r10 = (router_mode_i[2:0] == 3'd0) ? ready_dst_port_0 :
               (router_mode_i[2:0] == 3'd1) ? 1 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? ready_dst_port_0 :
               (router_mode_i[2:0] == 3'd5) ? ready_dst_port_0 :
               (router_mode_i[2:0] == 3'd6) ? 0 :
               (router_mode_i[2:0] == 3'd7) ? 0 : 0;
  assign r20 = (router_mode_i[2:0] == 3'd0) ? 0 :
               (router_mode_i[2:0] == 3'd1) ? 0 :
               (router_mode_i[2:0] == 3'd2) ? 1 :
               (router_mode_i[2:0] == 3'd3) ? 1 :
               (router_mode_i[2:0] == 3'd4) ? 0 :
               (router_mode_i[2:0] == 3'd5) ? 0 :
               (router_mode_i[2:0] == 3'd6) ? ready_dst_port_0 :
               (router_mode_i[2:0] == 3'd7) ? ready_dst_port_0 : 0;

  assign r01 = (router_mode_i[2:0] == 3'd0) ? ready_dst_port_1 :
               (router_mode_i[2:0] == 3'd1) ? 0 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? ready_dst_port_1 :
               (router_mode_i[2:0] == 3'd5) ? 1 :
               (router_mode_i[2:0] == 3'd6) ? 1 :
               (router_mode_i[2:0] == 3'd7) ? ready_dst_port_1 : 0;
  assign r11 = (router_mode_i[2:0] == 3'd0) ? 1 :
               (router_mode_i[2:0] == 3'd1) ? 1 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? 1 :
               (router_mode_i[2:0] == 3'd5) ? 1 :
               (router_mode_i[2:0] == 3'd6) ? 0 :
               (router_mode_i[2:0] == 3'd7) ? 0 : 0;
  assign r21 = (router_mode_i[2:0] == 3'd0) ? 0 :
               (router_mode_i[2:0] == 3'd1) ? 0 :
               (router_mode_i[2:0] == 3'd2) ? 1 :
               (router_mode_i[2:0] == 3'd3) ? ready_dst_port_1 :
               (router_mode_i[2:0] == 3'd4) ? 0 :
               (router_mode_i[2:0] == 3'd5) ? 0 :
               (router_mode_i[2:0] == 3'd6) ? 1 :
               (router_mode_i[2:0] == 3'd7) ? 1 : 0;

  assign r02 = (router_mode_i[2:0] == 3'd0) ? 1 :
               (router_mode_i[2:0] == 3'd1) ? 1 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? 1 :
               (router_mode_i[2:0] == 3'd5) ? ready_dst_port_2 :
               (router_mode_i[2:0] == 3'd6) ? ready_dst_port_2 :
               (router_mode_i[2:0] == 3'd7) ? 1 : 0;
  assign r12 = (router_mode_i[2:0] == 3'd0) ? 1 :
               (router_mode_i[2:0] == 3'd1) ? ready_dst_port_2 :
               (router_mode_i[2:0] == 3'd2) ? 0 :
               (router_mode_i[2:0] == 3'd3) ? 0 :
               (router_mode_i[2:0] == 3'd4) ? 1 :
               (router_mode_i[2:0] == 3'd5) ? 1 :
               (router_mode_i[2:0] == 3'd6) ? 0 :
               (router_mode_i[2:0] == 3'd7) ? 0 : 0;
  assign r22 = (router_mode_i[2:0] == 3'd0) ? 0 :
               (router_mode_i[2:0] == 3'd1) ? 0 :
               (router_mode_i[2:0] == 3'd2) ? ready_dst_port_2 :
               (router_mode_i[2:0] == 3'd3) ? 1 :
               (router_mode_i[2:0] == 3'd4) ? 0 :
               (router_mode_i[2:0] == 3'd5) ? 0 :
               (router_mode_i[2:0] == 3'd6) ? 1 :
               (router_mode_i[2:0] == 3'd7) ? 1 : 0;
endmodule
