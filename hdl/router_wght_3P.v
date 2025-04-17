// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: router_wght
///
/// Router_weights (router_wght) contains the necessary router for distributing the weights in the
/// OpenEye. A source can be either a horizontal cluster or the external connection to the Wght
/// ports. A destination can be either the horizontal cluster or the PE cluster.
/// communication protocol is the same "hand-shake" method used in the Input Activation Router
/// and the Partial Sum Router.
///
/// Parameters:
///    DATA_WIDTH             - WIDTH of data ports
///   
/// Ports:
///    router_mode_i          - Configurs the router. 1 means it accepts data from other router.
///                             0 menans it accepts data from top modul
///    ready_src_port_0       - Ready Port for the source Port 0 (Top Modul or GLB)
///    data_src_port_0        - Data Port for the source Port 0 (Top Modul or GLB)
///    enable_src_port_0      - Enable Port for the source Port 0 (Top Modul or GLB)
///    ready_src_port_1       - Ready Port for the source Port 1 (Other router)
///    data_src_port_1        - Data Port for the source Port 1 (Other router)
///    enable_src_port_1      - Enable Port for the source Port 1 (Other router)
///    ready_dst_port_0       - Ready Port for the destination Port 0 (Top Modul or GLB)
///    data_dst_port_0        - Data Port for the destination Port 0 (Top Modul or GLB)
///    enable_dst_port_0      - Enable Port for the destination Port 0 (Top Modul or GLB)
///    ready_dst_port_1       - Ready Port for the destination Port 1 (Other router)
///    data_dst_port_1        - Data Port for the destination Port 1 (Other router)
///    enable_dst_port_1      - Enable Port for the destination Port 1 (Other router)
///Fertig

module router_wght_3P
#(
  parameter integer DATA_WIDTH   = 8
) (
  input  [1:0]            router_mode_i,
  
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
