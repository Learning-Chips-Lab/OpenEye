// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: delay_cluster
///
/// Module for delaying input signals. This is necessary because data may arrive at the PSUM GLB
/// while the GLB is still reading data from its memory.
///
/// Parameters:
///    DATA_BITWIDTH  - Length of data
///   
/// Ports:
///    ready_o          - Outputs port `i`, if `sel_i` is 1, else outputs 0
///    data_i           - Input port `i`, if `sel_i` is 0, else outputs 0
///    enable_i         - Input port
///    ready_i          - Input port data port
///    data_o           - Output port data port
///    enable_o         - Output portdata port
///    delay_psum_glb_i - Input port data port
///

module delay_cluster 
#( 
  parameter integer            DATA_BITWIDTH   = 20
) (
  input                            clk_i,
  input                            rst_ni,

  output reg                       ready_o,
  input      [DATA_BITWIDTH-1 : 0] data_i,
  input                            enable_i,

  input                            ready_i,
  output     [DATA_BITWIDTH-1 : 0] data_o,
  output reg                       enable_o,

  input  reg [3 : 0]               delay_psum_glb_i

);
reg [8*DATA_BITWIDTH-1 : 0] data_s;
reg [7 : 0]                 enable_s;
reg [7 : 0]                 ready_s;


assign data_o = (delay_psum_glb_i== 0) ? data_i :
                (delay_psum_glb_i== 1) ? data_s[1*DATA_BITWIDTH-1 : 0] :
                (delay_psum_glb_i== 2) ? data_s[2*DATA_BITWIDTH-1 : 1*DATA_BITWIDTH] :
                (delay_psum_glb_i== 3) ? data_s[3*DATA_BITWIDTH-1 : 2*DATA_BITWIDTH] :
                (delay_psum_glb_i== 4) ? data_s[4*DATA_BITWIDTH-1 : 3*DATA_BITWIDTH] :
                (delay_psum_glb_i== 5) ? data_s[5*DATA_BITWIDTH-1 : 4*DATA_BITWIDTH] :
                (delay_psum_glb_i== 6) ? data_s[6*DATA_BITWIDTH-1 : 5*DATA_BITWIDTH] :
                (delay_psum_glb_i== 7) ? data_s[7*DATA_BITWIDTH-1 : 6*DATA_BITWIDTH] :
                (delay_psum_glb_i== 8) ? data_s[8*DATA_BITWIDTH-1 : 7*DATA_BITWIDTH] : 0;

assign enable_o = (delay_psum_glb_i== 0) ? enable_i :
                  (delay_psum_glb_i== 1) ? enable_s[0] :
                  (delay_psum_glb_i== 2) ? enable_s[1] :
                  (delay_psum_glb_i== 3) ? enable_s[2] :
                  (delay_psum_glb_i== 4) ? enable_s[3] :
                  (delay_psum_glb_i== 5) ? enable_s[4] :
                  (delay_psum_glb_i== 6) ? enable_s[5] :
                  (delay_psum_glb_i== 7) ? enable_s[6] :
                  (delay_psum_glb_i== 8) ? enable_s[7] : 0;

assign ready_o = (delay_psum_glb_i== 0) ? ready_i :
                 (delay_psum_glb_i== 1) ? ready_s[0] :
                 (delay_psum_glb_i== 2) ? ready_s[1] :
                 (delay_psum_glb_i== 3) ? ready_s[2] :
                 (delay_psum_glb_i== 4) ? ready_s[3] :
                 (delay_psum_glb_i== 5) ? ready_s[4] :
                 (delay_psum_glb_i== 6) ? ready_s[5] :
                 (delay_psum_glb_i== 7) ? ready_s[6] :
                 (delay_psum_glb_i== 8) ? ready_s[7] : 0;


 always@(posedge clk_i, negedge rst_ni) begin
    if(!rst_ni) begin: reset
      data_s   <= 0;
      enable_s <= 0;
      ready_s  <= 0;
    end else begin
      data_s   <= 8*DATA_BITWIDTH'(data_s << DATA_BITWIDTH) + 8*DATA_BITWIDTH'(data_i);
      enable_s <= 8'(data_s << 1) + 8'(enable_i);
      ready_s  <= 8'(ready_s << 1) + 8'(ready_i);
    end
 end

endmodule
