// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: router_psum
///
/// Router_partial_sum (router_psum) contains the necessary router to distribute the weights in the
/// the OpenEye. A source can be either the top of the PE cluster, the cluster above the
/// current cluster or the PSUM GLB. The target can be either the lower end of the PE cluster,
/// the cluster below it, or the PSUM GLB. Its basic communication protocol is the same "hand-shake"
/// method method used by the Input Activation Router and the Weight Router.
///
/// Parameters:
///    DATA_WIDTH             - WIDTH of data ports
///   
/// Ports:
///    router_mode_i          - Configurs the router. MSB 1 means it accepts and gets data from the
///                             GLB. MSB 0 menans it accepts and sends data to GLB.
///                             LSB '00' indicates single Cluster w/o connection to other clusters
///                             LSB '01' indicates topmost Cluster of a loop
///                             LSB '10' indicates middle-part Cluster of a loop
///                             LSB '11' indicates bottommost Cluster of a loop
///                             '100' and '000' result in same configuration
///    ready_src_port_0       - Ready Port for the source Port 0 (Top Modul or GLB)
///    data_src_port_0        - Data Port for the source Port 0 (Top Modul or GLB)
///    enable_src_port_0      - Enable Port for the source Port 0 (Top Modul or GLB)
///    ready_src_port_1       - Ready Port for the source Port 1 (Other router)
///    data_src_port_1        - Data Port for the source Port 1 (Other router)
///    enable_src_port_1      - Enable Port for the source Port 1 (Other router)
///    ready_src_port_2       - Ready Port for the source Port 2 (PE Cluster)
///    data_src_port_2        - Data Port for the source Port 2 (PE Cluster)
///    enable_src_port_2      - Enable Port for the source Port 2 (PE Cluster)
///    ready_dst_port_0       - Ready Port for the destination Port 0 (Top Modul or GLB)
///    data_dst_port_0        - Data Port for the destination Port 0 (Top Modul or GLB)
///    enable_dst_port_0      - Enable Port for the destination Port 0 (Top Modul or GLB)
///    ready_dst_port_1       - Ready Port for the destination Port 1 (Other router)
///    data_dst_port_1        - Data Port for the destination Port 1 (Other router)
///    enable_dst_port_1      - Enable Port for the destination Port 1 (Other router)
///    ready_dst_port_2       - Ready Port for the destination Port 2 (PE Cluster)
///    data_dst_port_2        - Data Port for the destination Port 2 (PE Cluster)
///    enable_dst_port_2      - Enable Port for the destination Port 2 (PE Cluster)

module router_psum
#(
  parameter integer DATA_WIDTH     = 20
) (
  input   [2:0]            router_mode_i,
  
  ///SRC Port 0
  output                   ready_src_port_0,
  input   [DATA_WIDTH-1:0] data_src_port_0,
  input                    enable_src_port_0,
  
  ///SRC Port 1
  output                   ready_src_port_1,
  input   [DATA_WIDTH-1:0] data_src_port_1,
  input                    enable_src_port_1,
  
  ///SRC Port 2
  output                   ready_src_port_2,
  input   [DATA_WIDTH-1:0] data_src_port_2,
  input                    enable_src_port_2,
  
  ///DST Port 0
  input                    ready_dst_port_0,
  output  [DATA_WIDTH-1:0] data_dst_port_0,
  output                   enable_dst_port_0,
  
  ///DST Port 1
  input                    ready_dst_port_1,
  output  [DATA_WIDTH-1:0] data_dst_port_1,
  output                   enable_dst_port_1,
  
  ///DST Port 2
  input                    ready_dst_port_2,
  output  [DATA_WIDTH-1:0] data_dst_port_2,
  output                   enable_dst_port_2
);

  ///Status Signals in Router
  ////////////////////////////////////////
  wire [1:0] l_status;
  wire       h_status;

  assign {h_status,l_status} = router_mode_i;

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
