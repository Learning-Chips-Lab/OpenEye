// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: router_wght_3P
///
/// The 3-Port Weight Router (router_wght_3P) is an enhanced version of the weight router
/// that supports three-way weight distribution in the OpenEye architecture. It extends
/// the basic weight router functionality to enable more complex weight sharing patterns
/// and efficient weight distribution across multiple PE clusters.
///
/// Key Features:
/// - Three-Port Architecture:
///   * Port 0: Primary Weight GLB interface
///   * Port 1: First horizontal routing path
///   * Port 2: Second horizontal routing path
///
/// - Enhanced Weight Distribution:
///   * Multi-directional weight broadcasting
///   * Parallel weight streaming support
///   * Configurable three-way routing paths
///
/// - Advanced Flow Control:
///   * Three-way handshake protocol
///   * Independent port backpressure
///   * Synchronized multi-port transfers
///
/// Architectural Role:
/// The router enables sophisticated weight distribution by:
/// 1. Supporting complex weight sharing patterns
/// 2. Enabling parallel weight distribution
/// 3. Facilitating weight broadcast to multiple clusters
/// 4. Providing flexible weight routing topologies
///
/// Parameters:
///    DATA_WIDTH          - Weight Data Path Width
///                         Defines the precision of weight values
///                         Typically 8-bit for standard neural networks
///                         Configurable for different quantization schemes
///   
/// Ports:
/// Configuration Interface:
///    router_mode_i [1:0] - Router Mode Control
///                         00: Single cluster mode (GLB to PE only)
///                         01: Primary horizontal broadcast
///                         10: Secondary horizontal broadcast
///                         11: Dual horizontal broadcast
///
/// Primary Interface (Port 0):
///    ready_src_port_0   - GLB Source Ready Signal
///                         Flow control for Weight GLB interface
///    data_src_port_0    - GLB Source Data Bus [DATA_WIDTH-1:0]
///                         Weight data from GLB
///    enable_src_port_0  - GLB Source Valid Signal
///                         Indicates valid weight data from GLB
///    ready_dst_port_0   - PE Cluster Ready Input
///                         Flow control from PE cluster
///    data_dst_port_0    - PE Cluster Data [DATA_WIDTH-1:0]
///                         Weight data to PE cluster
///    enable_dst_port_0  - PE Cluster Valid
///                         Valid signal to PE cluster
///
/// First Horizontal Interface (Port 1):
///    ready_src_port_1   - First Horizontal Source Ready
///                         Flow control from first adjacent router
///    data_src_port_1    - First Horizontal Data Input [DATA_WIDTH-1:0]
///                         Weight data from first adjacent router
///    enable_src_port_1  - First Horizontal Source Valid
///                         Valid data from first adjacent router
///    ready_dst_port_1   - First Horizontal Destination Ready
///                         Flow control to first adjacent router
///    data_dst_port_1    - First Horizontal Data Output [DATA_WIDTH-1:0]
///                         Weight data to first adjacent router
///    enable_dst_port_1  - First Horizontal Destination Valid
///                         Valid data to first adjacent router
///
/// Second Horizontal Interface (Port 2):
///    ready_src_port_2   - Second Horizontal Source Ready
///                         Flow control from second adjacent router
///    data_src_port_2    - Second Horizontal Data Input [DATA_WIDTH-1:0]
///                         Weight data from second adjacent router
///    enable_src_port_2  - Second Horizontal Source Valid
///                         Valid data from second adjacent router
///    ready_dst_port_2   - Second Horizontal Destination Ready
///                         Flow control to second adjacent router
///    data_dst_port_2    - Second Horizontal Data Output [DATA_WIDTH-1:0]
///                         Weight data to second adjacent router
///    enable_dst_port_2  - Second Horizontal Destination Valid
///                         Valid data to second adjacent router
///Fertig

module router_wght_3P #(
    parameter integer DATA_WIDTH = 8
) (
    input [1:0] router_mode_i,

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

    ///DST Port 1
    input                   ready_dst_port_2,
    output [DATA_WIDTH-1:0] data_dst_port_2,
    output                  enable_dst_port_2
);
  ///Signals in Router
  ////////////////////////////////////////Fertig

  wire e00;
  wire e01;
  wire e02;
  wire e10;
  wire e11;
  wire e12;
  wire r00;
  wire r01;
  wire r02;
  wire r10;
  wire r11;
  wire r12;

  ///Destination Port: Data
  ////////////////////////////////////////Fertig

  assign data_dst_port_0 = ({DATA_WIDTH{e00}}            & data_src_port_0)
                         | ({DATA_WIDTH{~e00 & e10}} & data_src_port_1)
                         | ({DATA_WIDTH{~e00 & ~e10 & e20}} & data_src_port_1);

  assign data_dst_port_1 = (router_mode_i == 0) ? 0 :
                           (router_mode_i == 1) ? ({DATA_WIDTH{e01}}            & data_src_port_0) :
                           (router_mode_i == 2) ? 0 :
                           (router_mode_i == 3) ? ({DATA_WIDTH{e01}}            & data_src_port_0) :
                           0;

  assign data_dst_port_2 = (router_mode_i == 0) ? 0 :
                           (router_mode_i == 1) ? 0 :
                           (router_mode_i == 2) ? ({DATA_WIDTH{e01}}            & data_src_port_0) :
                           (router_mode_i == 3) ? ({DATA_WIDTH{e01}}            & data_src_port_0) :
                           0;

  ///Destination Port: Enable
  ////////////////////////////////////////

  assign enable_dst_port_0 = e00 | e10 | e20;
  assign enable_dst_port_1 = (router_mode_i == 0) ? 0 :  
                           (router_mode_i == 1) ? 0 :
                           (router_mode_i == 2) ? ({DATA_WIDTH{e01}}            & data_src_port_0) :
                           (router_mode_i == 3) ? ({DATA_WIDTH{e01}}            & data_src_port_0) :
                           0;
  assign enable_dst_port_2 = router_mode_i ? 0 : e01;

  ///Source Port: Ready
  ////////////////////////////////////////Fertig

  assign ready_src_port_0 = r00 & r01 & r02;
  assign ready_src_port_1 = r10 & r11 & r12;
  assign ready_src_port_2 = r00 & r01 & r22;

  ///Source Port: Enable
  ////////////////////////////////////////

  assign e00 = router_mode_i ? 0 : enable_src_port_0;
  assign e01 = router_mode_i ? 0 : enable_src_port_0;

  assign e10 = router_mode_i ? enable_src_port_1 : 0;
  assign e11 = router_mode_i ? 0 : 0;


  ///Destination Port: Ready
  ////////////////////////////////////////

  assign r00 = router_mode_i ? 1 : ready_dst_port_0;
  assign r10 = router_mode_i ? ready_dst_port_0 : 1;

  assign r01 = router_mode_i ? 1 : ready_dst_port_1;
  assign r11 = router_mode_i ? 1 : 1;

endmodule
