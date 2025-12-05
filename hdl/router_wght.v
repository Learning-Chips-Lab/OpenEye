// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: router_wght
///
/// The Weight Router (router_wght) is a specialized routing module in the OpenEye architecture
/// designed for efficient distribution of neural network weights. It implements a streamlined
/// 2-port routing topology optimized for weight data flow from the Weight GLB to PE clusters,
/// with support for horizontal weight sharing across clusters.
///
/// Key Features:
/// - Two-Port Architecture:
///   * Port 0: Primary interface for Weight GLB connection and PE cluster
///   * Port 1: Horizontal routing interface for weight sharing
///
/// - Efficient Weight Distribution:
///   * Direct GLB to PE Cluster routing
///   * Horizontal weight broadcasting support
///   * Configurable data paths for weight reuse
///
/// - Handshake-based Flow Control:
///   * Ready/Enable synchronization protocol
///   * Matches GLB and PE cluster interfaces
///   * Prevents weight data overflow
///
/// Architectural Role:
/// The router enables efficient weight distribution by:
/// 1. Providing direct paths from Weight GLB to PE clusters
/// 2. Supporting horizontal weight sharing for parameter efficiency
/// 3. Implementing synchronized weight data transfer
/// 4. Enabling flexible weight distribution patterns
///
/// Parameters:
///    DATA_WIDTH          - Weight Data Path Width
///                         Defines the precision of weight values
///                         Typically 8-bit for standard neural networks
///                         Configurable for different quantization schemes
///   
/// Ports:
/// Configuration Interface:
///    router_mode_i      - Router Mode Control
///                         1: Accept data from horizontal router (weight sharing)
///                         0: Accept data from Weight GLB
///
/// GLB/Primary Interface (Port 0):
///    ready_src_port_0   - Primary Source Ready Signal
///                         Flow control for Weight GLB interface
///    data_src_port_0    - Primary Source Data Bus [DATA_WIDTH-1:0]
///                         Weight data from GLB
///    enable_src_port_0  - Primary Source Valid Signal
///                         Indicates valid weight data from GLB
///    ready_dst_port_0   - Primary Destination Ready
///                         Flow control to PE cluster
///    data_dst_port_0    - Primary Destination Data [DATA_WIDTH-1:0]
///                         Weight data to PE cluster
///    enable_dst_port_0  - Primary Destination Valid
///                         Marks valid weight data to PE cluster
///
/// Horizontal Interface (Port 1):
///    ready_src_port_1   - Horizontal Source Ready Signal
///                         Flow control from adjacent router
///    data_src_port_1    - Horizontal Source Data [DATA_WIDTH-1:0]
///                         Weight data from adjacent router
///    enable_src_port_1  - Horizontal Source Valid Signal
///                         Valid data from adjacent router
///    ready_dst_port_1   - Horizontal Destination Ready
///                         Flow control to adjacent router
///    data_dst_port_1    - Horizontal Destination Data [DATA_WIDTH-1:0]
///                         Weight data to adjacent router
///    enable_dst_port_1  - Horizontal Destination Valid
///                         Valid data to adjacent router
///

module router_wght #(
    parameter integer DATA_WIDTH = 8
) (
    input router_mode_i,

    ///SRC Port 0
    output                  ready_src_port_0,
    input  [DATA_WIDTH-1:0] data_src_port_0,
    input                   enable_src_port_0,

    ///SRC Port 1
    output                  ready_src_port_1,
    input  [DATA_WIDTH-1:0] data_src_port_1,
    input                   enable_src_port_1,

    ///DST Port 0
    input                   ready_dst_port_0,
    output [DATA_WIDTH-1:0] data_dst_port_0,
    output                  enable_dst_port_0,

    ///DST Port 1
    input                   ready_dst_port_1,
    output [DATA_WIDTH-1:0] data_dst_port_1,
    output                  enable_dst_port_1
);
  ///Signals in Router
  ////////////////////////////////////////

  wire e00;
  wire e01;
  wire e10;
  wire e11;
  wire r00;
  wire r01;
  wire r10;
  wire r11;

  ///Destination Port: Data
  ////////////////////////////////////////

  assign data_dst_port_0 = ({DATA_WIDTH{e00}}            & data_src_port_0)
                         | ({DATA_WIDTH{~e00 & e10}} & data_src_port_1);

  assign data_dst_port_1 = router_mode_i ? 0 : ({DATA_WIDTH{e01}} & data_src_port_0);

  ///Destination Port: Enable
  ////////////////////////////////////////

  assign enable_dst_port_0 = e00 | e10;
  assign enable_dst_port_1 = router_mode_i ? 0 : e01;

  ///Source Port: Ready
  ////////////////////////////////////////

  assign ready_src_port_0 = r00 & r01;
  assign ready_src_port_1 = r10 & r11;

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
