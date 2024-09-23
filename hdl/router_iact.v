// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: router_iact
///
/// Typically it connects one source to up to four destinations. A source can be either another
/// cluster horizontal/vertical to the current cluster or an Iact GLB. A destination can also be
/// another cluster horizontal/vertical to the current cluster or the process element cluster in the 
/// current cluster. Its basic communication protocol is the same "hand-shake" method used in the
/// used in the weight router and the partial sum router. It typically connects one source to up to
/// four destinations. 
///
/// Parameters:
///    LEFT_CLUSTER         - Indicates, if cluster is on the left side
///    DATA_WIDTH           - WIDTH of data ports
///   
/// Ports:
///    router_mode_i        - Configurs the router.
///                           MSB Describes source, LSB describes destination
///                           MSB '00' means it accepts data from the GLB "Center"
///                           MSB '01' means it accepts data from top cluster "North"
///                           MSB '10' means it accepts data from opposite cluster "West/East"
///                           MSB '11' means it accepts data from bottom cluster "South"
///                           LSB 'XXX1' sends data to PE cluster "Center"
///                           LSB 'XXX1' sends data to top cluster "North"
///                           LSB 'XXX1' sends data to opposite cluster "West/East"
///                           LSB 'XXX1' sends data to bottom cluster "South"
///    ready_src_port_0     - Ready Port for the source Port 0 (Top Modul or GLB, "Center")
///    data_src_port_0      - Data Port for the source Port 0 (Top Modul or GLB, "Center")
///    enable_src_port_0    - Enable Port for the source Port 0 (Top Modul or GLB, "Center")
///    ready_src_port_1     - Ready Port for the source Port 1 (Router above, "North")
///    data_src_port_1      - Data Port for the source Port 1 (Router above, "North")
///    enable_src_port_1    - Enable Port for the source Port 1 (Router above, "North")
///    ready_src_port_2     - Ready Port for the source Port 2 (Router opposite, "West/East")
///    data_src_port_2      - Data Port for the source Port 2 (Router opposite, "West/East")
///    enable_src_port_2    - Enable Port for the source Port 2 (Router opposite, "West/East")
///    ready_src_port_3     - Ready Port for the source Port 3 (Router below, "South")
///    data_src_port_3      - Data Port for the source Port 3 (Router below, "South")
///    enable_src_port_3    - Enable Port for the source Port 3 (Router below, "South")
///    ready_dst_port_0     - Ready Port for the destination Port 0 (PE Cluster, "Center")
///    data_dst_port_0      - Data Port for the destination Port 0 (PE Cluster, "Center")
///    enable_dst_port_0    - Enable Port for the destination Port 0 (PE Cluster, "Center")
///    ready_dst_port_1     - Ready Port for the destination Port 1 (Router above, "North")
///    data_dst_port_1      - Data Port for the destination Port 1 (Router above, "North")
///    enable_dst_port_1    - Enable Port for the destination Port 1 (Router above, "North")
///    ready_dst_port_2     - Ready Port for the destination Port 2 (Router opposite, "West/East")
///    data_dst_port_2      - Data Port for the destination Port 2 (Router opposite, "West/East")
///    enable_dst_port_2    - Enable Port for the destination Port 2 (Router opposite, "West/East")
///    ready_dst_port_3     - Ready Port for the destination Port 3 (Router below, "South")
///    data_dst_port_3      - Data Port for the destination Port 3 (Router below, "South")
///    enable_dst_port_3    - Enable Port for the destination Port 3 (Router below, "South")

module router_iact
#(
  parameter         LEFT_CLUSTER    = 0,
  parameter integer DATA_WIDTH      = 8
) (
  input [5:0]router_mode_i,
  ///SRC Port C
  output                  ready_src_port_0,
  input [DATA_WIDTH-1:0]  data_src_port_0,
  input                   enable_src_port_0,
  ///SRC Port N
  output                  ready_src_port_1,
  input [DATA_WIDTH-1:0]  data_src_port_1,
  input                   enable_src_port_1,
  ///SRC Port EW
  output                  ready_src_port_2,
  input [DATA_WIDTH-1:0]  data_src_port_2,
  input                   enable_src_port_2,
  ///SRC Port S
  output                  ready_src_port_3,
  input [DATA_WIDTH-1:0]  data_src_port_3,
  input                   enable_src_port_3,
  ///DST Port C
  input                   ready_dst_port_0,
  output[DATA_WIDTH-1:0]  data_dst_port_0,
  output                  enable_dst_port_0,
  ///DST Port N
  input                   ready_dst_port_1,
  output[DATA_WIDTH-1:0]  data_dst_port_1,
  output                  enable_dst_port_1,
  ///DST Port EW
  input                   ready_dst_port_2,
  output[DATA_WIDTH-1:0]  data_dst_port_2,
  output                  enable_dst_port_2,
  ///DST Port S
  input                   ready_dst_port_3,
  output[DATA_WIDTH-1:0]  data_dst_port_3,
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
  assign data_dst_port_1 = ({DATA_WIDTH{e01}}                      & data_src_port_0)
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

  assign e00 =  (2'd0 == router_mode_i[5:4]) & router_mode_i[0] ? enable_src_port_0 : 0;
  assign e01 =  (2'd0 == router_mode_i[5:4]) & router_mode_i[1] ? enable_src_port_0 : 0;
  assign e02 =  (2'd0 == router_mode_i[5:4]) & router_mode_i[2] ? enable_src_port_0 : 0;
  assign e03 =  (2'd0 == router_mode_i[5:4]) & router_mode_i[3] ? enable_src_port_0 : 0;

  assign e10 =  (2'd1 == router_mode_i[5:4]) & router_mode_i[0] ? enable_src_port_1 : 0;
  assign e11 =  (2'd1 == router_mode_i[5:4]) & router_mode_i[1] ? 0 : 0;
  assign e12 =  (2'd1 == router_mode_i[5:4]) & router_mode_i[2] ? enable_src_port_1 : 0;
  assign e13 =  (2'd1 == router_mode_i[5:4]) & router_mode_i[3] ? enable_src_port_1 : 0;

  assign e20 =  (2'd2 == router_mode_i[5:4]) & router_mode_i[0] ? enable_src_port_2 : 0;
  assign e21 =  (2'd2 == router_mode_i[5:4]) & router_mode_i[1] ? enable_src_port_2 : 0;
  assign e22 =  (2'd2 == router_mode_i[5:4]) & router_mode_i[2] ? 0 : 0;
  assign e23 =  (2'd2 == router_mode_i[5:4]) & router_mode_i[3] ? enable_src_port_2 & LEFT_CLUSTER : 0;

  assign e30 =  (2'd3 == router_mode_i[5:4]) & router_mode_i[0] ? enable_src_port_3 : 0;
  assign e31 =  (2'd3 == router_mode_i[5:4]) & router_mode_i[1] ? enable_src_port_3 : 0;
  assign e32 =  (2'd3 == router_mode_i[5:4]) & router_mode_i[2] ? enable_src_port_3 & LEFT_CLUSTER: 0;
  assign e33 =  (2'd3 == router_mode_i[5:4]) & router_mode_i[3] ? 0 : 0;

  ///Destination Port: Ready
  ////////////////////////////////////////

  assign r00 = (2'd0 == router_mode_i[5:4]) ? (router_mode_i[0] ? ready_dst_port_0 : 1) : 1;
  assign r10 = (2'd1 == router_mode_i[5:4]) ? (router_mode_i[0] ? ready_dst_port_0 : 1) : 1;
  assign r20 = (2'd2 == router_mode_i[5:4]) ? (router_mode_i[0] ? ready_dst_port_0 : 1) : 1;
  assign r30 = (2'd3 == router_mode_i[5:4]) ? (router_mode_i[0] ? ready_dst_port_0 : 1) : 1;

  assign r01 = (2'd0 == router_mode_i[5:4]) ? (router_mode_i[1] ? ready_dst_port_1 : 1) : 1;
  assign r11 = (2'd1 == router_mode_i[5:4]) ? (router_mode_i[1] ? 0 : 0) : 1;
  assign r21 = (2'd2 == router_mode_i[5:4]) ? (router_mode_i[1] ? ready_dst_port_1 : 1) : 1;
  assign r31 = (2'd3 == router_mode_i[5:4]) ? (router_mode_i[1] ? ready_dst_port_1 : 1) : 1;

  assign r02 = (2'd0 == router_mode_i[5:4]) ? (router_mode_i[2] ? ready_dst_port_2 : 1) : 1;
  assign r12 = (2'd1 == router_mode_i[5:4]) ? (router_mode_i[2] ? ready_dst_port_2 : 1) : 1;
  assign r22 = (2'd2 == router_mode_i[5:4]) ? (router_mode_i[2] ? 0 : 0) : 1;
  assign r32 = (2'd3 == router_mode_i[5:4]) ? (router_mode_i[2] ? 0 : 1) : 1;

  assign r03 = (2'd0 == router_mode_i[5:4]) ? (router_mode_i[3] ? ready_dst_port_3 : 1) : 1;
  assign r13 = (2'd1 == router_mode_i[5:4]) ? (router_mode_i[3] ? ready_dst_port_3 : 1) : 1;
  assign r23 = (2'd2 == router_mode_i[5:4]) ? (router_mode_i[3] ? 0 : 1) : 1;
  assign r33 = (2'd3 == router_mode_i[5:4]) ? (router_mode_i[3] ? 0 : 0) : 1;
endmodule
