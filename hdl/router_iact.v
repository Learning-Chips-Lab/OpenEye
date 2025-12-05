// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: router_iact
///
/// The Input Activation Router (router_iact) is a key component in the OpenEye neural network 
/// accelerator's data distribution network. It implements a configurable routing fabric that 
/// manages the movement of input activation data between clusters, global buffers (GLBs), and 
/// processing elements (PEs).
///
/// Architecture Overview:
/// - Routing Topology:
///   * 5-Port Router Design:
///     - Center Port (C): Connects to PE cluster or GLB
///     - North Port (N): Links to upper cluster
///     - East/West Port (E/W): Links to adjacent cluster
///     - South Port (S): Links to lower cluster
///
/// - Communication Protocol:
///   * Handshake-based Flow Control:
///     - Ready/Enable signaling
///     - Back-pressure support
///     - Deadlock prevention
///
/// - Routing Configuration:
///   * Source Selection:
///     - GLB input (Center)
///     - North cluster
///     - East/West cluster
///     - South cluster
///
///   * Destination Control:
///     - PE cluster routing
///     - Inter-cluster forwarding
///     - Multi-cast capability
///
/// Key Features:
/// - Flexible Data Movement:
///   * One-to-Many Distribution
///   * Configurable Paths
///   * Priority-based Routing
///   * Flow Control Support
///
/// - Network Integration:
///   * Cluster-level Connectivity
///   * GLB Data Distribution
///   * PE Array Interface
///   * Mesh Network Support
///
/// Performance Features:
/// - Low-latency Routing
/// - Configurable Bandwidth
/// - Back-pressure Handling
/// - Deadlock Avoidance
///
/// Parameters:
/// Configuration:
///    LEFT_CLUSTER        - Network Position Parameter
///                         0: Right side cluster position
///                         1: Left side cluster position
///                         Affects East/West routing logic
///
///    DATA_WIDTH         - Data Path Configuration
///                         Specifies width of all data ports
///                         Determines activation data precision
///   
/// Ports:
/// Control Interface:
///    router_mode_i [5:0] - Router Configuration Control
///                         [5:4] Source Port Selection:
///                           00: GLB/Center input
///                           01: North cluster input
///                           10: East/West cluster input
///                           11: South cluster input
///                         [3:0] Destination Enable:
///                           Bit 0: Enable Center/PE routing
///                           Bit 1: Enable North routing
///                           Bit 2: Enable East/West routing
///                           Bit 3: Enable South routing
///
/// Center (Port 0) Interface:
///    ready_src_port_0   - GLB/PE Ready Input (active high)
///                         Indicates source can accept back-pressure
///    data_src_port_0    - GLB/PE Data Input [DATA_WIDTH-1:0]
///                         Carries activation data from GLB
///    enable_src_port_0  - GLB/PE Valid Input (active high)
///                         Indicates valid data from GLB
///
/// North (Port 1) Interface:
///    ready_src_port_1   - North Ready Input
///                         Flow control from north cluster
///    data_src_port_1    - North Data Input [DATA_WIDTH-1:0]
///                         Data from north cluster
///    enable_src_port_1  - North Valid Input
///                         Valid signal from north cluster
///
/// East/West (Port 2) Interface:
///    ready_src_port_2   - East/West Ready Input
///                         Flow control from adjacent cluster
///    data_src_port_2    - East/West Data Input [DATA_WIDTH-1:0]
///                         Data from adjacent cluster
///    enable_src_port_2  - East/West Valid Input
///                         Valid signal from adjacent cluster
///
/// South (Port 3) Interface:
///    ready_src_port_3   - South Ready Input
///                         Flow control from south cluster
///    data_src_port_3    - South Data Input [DATA_WIDTH-1:0]
///                         Data from south cluster
///    enable_src_port_3  - South Valid Input
///                         Valid signal from south cluster
///
/// PE Cluster Output Interface:
///    ready_dst_port_0   - PE Ready Input
///                         Flow control from PE cluster
///    data_dst_port_0    - PE Data Output [DATA_WIDTH-1:0]
///                         Data to PE cluster
///    enable_dst_port_0  - PE Valid Output
///                         Valid signal to PE cluster
///
/// North Output Interface:
///    ready_dst_port_1   - North Ready Input
///                         Flow control from north output
///    data_dst_port_1    - North Data Output [DATA_WIDTH-1:0]
///                         Data to north cluster
///    enable_dst_port_1  - North Valid Output
///                         Valid signal to north cluster
///
/// East/West Output Interface:
///    ready_dst_port_2   - East/West Ready Input
///                         Flow control from adjacent output
///    data_dst_port_2    - East/West Data Output [DATA_WIDTH-1:0]
///                         Data to adjacent cluster
///    enable_dst_port_2  - East/West Valid Output
///                         Valid signal to adjacent cluster
///
/// South Output Interface:
///    ready_dst_port_3   - South Ready Input
///                         Flow control from south output
///    data_dst_port_3    - South Data Output [DATA_WIDTH-1:0]
///                         Data to south cluster
///    enable_dst_port_3  - South Valid Output
///                         Valid signal to south cluster

module router_iact #(
    parameter         LEFT_CLUSTER = 0,
    parameter integer DATA_WIDTH   = 8
) (
    input  [           5:0] router_mode_i,
    ///SRC Port C
    output                  ready_src_port_0,
    input  [DATA_WIDTH-1:0] data_src_port_0,
    input                   enable_src_port_0,
    ///SRC Port N
    output                  ready_src_port_1,
    input  [DATA_WIDTH-1:0] data_src_port_1,
    input                   enable_src_port_1,
    ///SRC Port EW
    output                  ready_src_port_2,
    input  [DATA_WIDTH-1:0] data_src_port_2,
    input                   enable_src_port_2,
    ///SRC Port S
    output                  ready_src_port_3,
    input  [DATA_WIDTH-1:0] data_src_port_3,
    input                   enable_src_port_3,
    ///DST Port C
    input                   ready_dst_port_0,
    output [DATA_WIDTH-1:0] data_dst_port_0,
    output                  enable_dst_port_0,
    ///DST Port N
    input                   ready_dst_port_1,
    output [DATA_WIDTH-1:0] data_dst_port_1,
    output                  enable_dst_port_1,
    ///DST Port EW
    input                   ready_dst_port_2,
    output [DATA_WIDTH-1:0] data_dst_port_2,
    output                  enable_dst_port_2,
    ///DST Port S
    input                   ready_dst_port_3,
    output [DATA_WIDTH-1:0] data_dst_port_3,
    output                  enable_dst_port_3
);
  ///Signals in Router
  ////////////////////////////////////////
  wire e00;
  wire e01;
  wire e02;
  wire e03;
  wire e10;
  wire e11;
  wire e12;
  wire e13;
  wire e20;
  wire e21;
  wire e22;
  wire e23;
  wire e30;
  wire e31;
  wire e32;
  wire e33;
  wire r00;
  wire r01;
  wire r02;
  wire r03;
  wire r10;
  wire r11;
  wire r12;
  wire r13;
  wire r20;
  wire r21;
  wire r22;
  wire r23;
  wire r30;
  wire r31;
  wire r32;
  wire r33;

  ///Destination Port: Data
  ////////////////////////////////////////
  ///Center
  assign data_dst_port_0 = ({DATA_WIDTH{e00}}                      & data_src_port_0)
                         | ({DATA_WIDTH{~e00 & e10}}               & data_src_port_1)
                         | ({DATA_WIDTH{~e00 & ~e10 & e20}}        & data_src_port_2)
                         | ({DATA_WIDTH{~e00 & ~e10 & ~e20 & e30}} & data_src_port_3);
  ///North
  assign data_dst_port_1 = ({DATA_WIDTH{e01}} & data_src_port_0)
      //| ({DATA_WIDTH{~e01 & e11}}               & data_src_port_1)
      | ({DATA_WIDTH{~e01 & ~e11 & e21}}        & data_src_port_2)
                         | ({DATA_WIDTH{~e01 & ~e11 & ~e21 & e31}} & data_src_port_3);
  ///East/West
  assign data_dst_port_2 = ({DATA_WIDTH{e02}}                      & data_src_port_0)
                         | ({DATA_WIDTH{~e02 & e12}}               & data_src_port_1)
      //| ({DATA_WIDTH{~e02 & ~e12 & e22}}        & data_src_port_2)
      | ({DATA_WIDTH{LEFT_CLUSTER}} & ({DATA_WIDTH{~e02 & ~e12 & ~e22 & e32}} & data_src_port_3));
  ///South
  assign data_dst_port_3 = ({DATA_WIDTH{e03}}                      & data_src_port_0)
                         | ({DATA_WIDTH{~e03 & e13}}               & data_src_port_1)
                         | ({DATA_WIDTH{LEFT_CLUSTER}} & ({DATA_WIDTH{~e03 & ~e13 & e23}}        & data_src_port_2));
  //| ({DATA_WIDTH{~e03 & ~e13 & ~e23 & e33}} & data_src_port_3);

  ///Destination Port: Enable
  ////////////////////////////////////////

  assign enable_dst_port_0 = e00 | e10 | e20 | e30;
  assign enable_dst_port_1 = e01 | e21 | e31;
  assign enable_dst_port_2 = e02 | e12 | e32;
  assign enable_dst_port_3 = e03 | e13 | e23;

  ///Source Port: Ready
  ////////////////////////////////////////

  assign ready_src_port_0 = r00 & r01 & r02 & r03;
  assign ready_src_port_1 = r10 & r12 & r13;
  assign ready_src_port_2 = r20 & r21 & r23;
  assign ready_src_port_3 = r30 & r31 & r32;

  ///Source Port: Enable
  ////////////////////////////////////////

  assign e00 = (2'd0 == router_mode_i[5:4]) & router_mode_i[0] ? enable_src_port_0 : 0;
  assign e01 = (2'd0 == router_mode_i[5:4]) & router_mode_i[1] ? enable_src_port_0 : 0;
  assign e02 = (2'd0 == router_mode_i[5:4]) & router_mode_i[2] ? enable_src_port_0 : 0;
  assign e03 = (2'd0 == router_mode_i[5:4]) & router_mode_i[3] ? enable_src_port_0 : 0;

  assign e10 = (2'd1 == router_mode_i[5:4]) & router_mode_i[0] ? enable_src_port_1 : 0;
  assign e11 = (2'd1 == router_mode_i[5:4]) & router_mode_i[1] ? 0 : 0;
  assign e12 = (2'd1 == router_mode_i[5:4]) & router_mode_i[2] ? enable_src_port_1 : 0;
  assign e13 = (2'd1 == router_mode_i[5:4]) & router_mode_i[3] ? enable_src_port_1 : 0;

  assign e20 = (2'd2 == router_mode_i[5:4]) & router_mode_i[0] ? enable_src_port_2 : 0;
  assign e21 = (2'd2 == router_mode_i[5:4]) & router_mode_i[1] ? enable_src_port_2 : 0;
  assign e22 = (2'd2 == router_mode_i[5:4]) & router_mode_i[2] ? 0 : 0;
  assign e23 =  (2'd2 == router_mode_i[5:4]) & router_mode_i[3] ? enable_src_port_2 & LEFT_CLUSTER : 0;

  assign e30 = (2'd3 == router_mode_i[5:4]) & router_mode_i[0] ? enable_src_port_3 : 0;
  assign e31 = (2'd3 == router_mode_i[5:4]) & router_mode_i[1] ? enable_src_port_3 : 0;
  assign e32 =  (2'd3 == router_mode_i[5:4]) & router_mode_i[2] ? enable_src_port_3 & LEFT_CLUSTER: 0;
  assign e33 = (2'd3 == router_mode_i[5:4]) & router_mode_i[3] ? 0 : 0;

  ///Destination Port: Ready
  ////////////////////////////////////////

  assign r00 = (2'd0 == router_mode_i[5:4]) ? (router_mode_i[0] ? ready_dst_port_0 : 1) : ready_dst_port_0;
  assign r10 = (2'd1 == router_mode_i[5:4]) ? (router_mode_i[0] ? ready_dst_port_0 : 1) : (2'd0 == router_mode_i[5:4]) ? ready_dst_port_0 : 1;
  assign r20 = (2'd2 == router_mode_i[5:4]) ? (router_mode_i[0] ? ready_dst_port_0 : 1) : 1;
  assign r30 = (2'd3 == router_mode_i[5:4]) ? (router_mode_i[0] ? ready_dst_port_0 : 1) : (2'd0 == router_mode_i[5:4]) ? ready_dst_port_0 : 1;

  assign r01 = (2'd0 == router_mode_i[5:4]) ? (router_mode_i[1] ? ready_dst_port_1 : 1) : (2'd1 == router_mode_i[5:4]) ? ready_dst_port_1 : 1;
  assign r11 = (2'd1 == router_mode_i[5:4]) ? (router_mode_i[1] ? 0 : 0) : ready_dst_port_1;
  assign r21 = (2'd2 == router_mode_i[5:4]) ? (router_mode_i[1] ? ready_dst_port_1 : 1) : 1;
  assign r31 = (2'd3 == router_mode_i[5:4]) ? (router_mode_i[1] ? ready_dst_port_1 : 1) : (2'd1 == router_mode_i[5:4]) ? ready_dst_port_1 : 1;

  assign r02 = (2'd0 == router_mode_i[5:4]) ? (router_mode_i[2] ? ready_dst_port_2 : 1) : (2'd2 == router_mode_i[5:4]) ? ready_dst_port_2 : 1;
  assign r12 = (2'd1 == router_mode_i[5:4]) ? (router_mode_i[2] ? ready_dst_port_2 : 1) : (2'd2 == router_mode_i[5:4]) ? ready_dst_port_2 : 1;
  assign r22 = (2'd2 == router_mode_i[5:4]) ? (router_mode_i[2] ? 0 : 0) : ready_dst_port_2;
  assign r32 = (2'd3 == router_mode_i[5:4]) ? (router_mode_i[2] ? 0 : 1) : (2'd2 == router_mode_i[5:4]) ? ready_dst_port_2 : 1;

  assign r03 = (2'd0 == router_mode_i[5:4]) ? (router_mode_i[3] ? ready_dst_port_3 : 1) : (2'd3 == router_mode_i[5:4]) ? ready_dst_port_3 : 1;
  assign r13 = (2'd1 == router_mode_i[5:4]) ? (router_mode_i[3] ? ready_dst_port_3 : 1) :  (2'd3 == router_mode_i[5:4]) ? ready_dst_port_3 : 1;
  assign r23 = (2'd2 == router_mode_i[5:4]) ? (router_mode_i[3] ? 0 : 1) : 1;
  assign r33 = (2'd3 == router_mode_i[5:4]) ? (router_mode_i[3] ? 0 : 0) : ready_dst_port_3;
endmodule
