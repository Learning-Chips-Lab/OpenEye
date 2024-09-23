// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

module router_wght_tb();

    localparam integer DATA_WIDTH = 8;
    localparam integer FULL_PORTS = 2;
    localparam integer TEST_STEP = 5;                              //only for testbench 



  ///////////// Module A  
  reg                           router_mode_i_A;    // input router mode
  //SRC Port 0
  reg                           ready_src_port_0_A; // output
  reg  [DATA_WIDTH-1:0]         data_src_port_0_A;  // input
  reg                           enable_src_port_0_A;// input
  //DST Port 0
  reg                           ready_dst_port_0_A; // input
  reg [DATA_WIDTH-1:0]          data_dst_port_0_A;  // output
  reg                           enable_dst_port_0_A;// output

  ////////////// Module B
  reg                           router_mode_i_B;    // input router mode
  //SRC Port 0
  reg                           ready_src_port_0_B; // output
  reg  [DATA_WIDTH-1:0]         data_src_port_0_B;  // input
  reg                           enable_src_port_0_B;// input
  //DST Port 0
  reg                           ready_dst_port_0_B; // input
  reg [DATA_WIDTH-1:0]          data_dst_port_0_B;  // ouput
  reg                           enable_dst_port_0_B;// output

  //Cross Cennection Module A and Module B
  reg  [DATA_WIDTH-1:0]         data_src_1_A_to_dst_1_B;
  reg  [DATA_WIDTH-1:0]         data_src_1_B_to_dst_1_A;

  reg                           ready_src_1_A_to_dst_1_B;
  reg                           ready_src_1_B_to_dst_1_A;

  reg                           enable_src_1_A_to_dst_1_B;
  reg                           enable_src_1_B_to_dst_1_A;

  //Testbench only
  reg                           result_in_step[0:TEST_STEP-1];      //Result vector //only for testbench
  reg                           computation_was_right = 1;          //Result flag //only for testbench


    router_wght #(
        .DATA_WIDTH(DATA_WIDTH),
        .FULL_PORTS(FULL_PORTS)

        ) router_wght_A (

        .router_mode_i(router_mode_i_A),

        .ready_src_port_0(ready_src_port_0_A),
        .data_src_port_0(data_src_port_0_A),
        .enable_src_port_0(enable_src_port_0_A),

        .ready_src_port_1(ready_src_1_A_to_dst_1_B),
        .data_src_port_1(data_src_1_A_to_dst_1_B),
        .enable_src_port_1(enable_src_1_A_to_dst_1_B),

        .ready_dst_port_0(ready_dst_port_0_A),
        .data_dst_port_0(data_dst_port_0_A),
        .enable_dst_port_0(enable_dst_port_0_A),

        .ready_dst_port_1(ready_src_1_B_to_dst_1_A),
        .data_dst_port_1(data_src_1_B_to_dst_1_A),
        .enable_dst_port_1(enable_src_1_B_to_dst_1_A));




    router_wght  #(
        .DATA_WIDTH(DATA_WIDTH),
        .FULL_PORTS(FULL_PORTS)

        )  router_wght_B (

        .router_mode_i(router_mode_i_B),

        .ready_src_port_0(ready_src_port_0_B),
        .data_src_port_0(data_src_port_0_B),
        .enable_src_port_0(enable_src_port_0_B),

        .ready_src_port_1(ready_src_1_B_to_dst_1_A),
        .data_src_port_1(data_src_1_B_to_dst_1_A),
        .enable_src_port_1(enable_src_1_B_to_dst_1_A),

        .ready_dst_port_0(ready_dst_port_0_B),
        .data_dst_port_0(data_dst_port_0_B),
        .enable_dst_port_0(enable_dst_port_0_B),

        .ready_dst_port_1(ready_src_1_A_to_dst_1_B),
        .data_dst_port_1(data_src_1_A_to_dst_1_B),
        .enable_dst_port_1(enable_src_1_A_to_dst_1_B));

 
    // main test block
    initial begin

    // open, write and save results in .vcd file
        $dumpfile("router_wght_tb.vcd");
        $dumpvars(0,router_wght_tb);

    // set result vector to 0
        for(int j=0; j<TEST_STEP; j=j+1)begin
        result_in_step[j] = 0;        
        end


        //Testcase 0: set all Inputs to Zero
        router_mode_i_A         = 0;
        data_src_port_0_A       = 0;
        enable_src_port_0_A     = 0;
        ready_dst_port_0_A      = 0;

        router_mode_i_B         = 0;
        data_src_port_0_B       = 0;
        enable_src_port_0_B     = 0;
        ready_dst_port_0_B      = 0;

        #10
        result_in_step [0] = (data_dst_port_0_A == 0 && data_dst_port_0_B == 0 ) ? 1 : 0; //save result


        //Testcase 1: no ready and no enable are set, output for both modules shall be zero
        #10
        router_mode_i_A         = 0;
        data_src_port_0_A       = 10;
        enable_src_port_0_A     = 0;
        ready_dst_port_0_A      = 0;

        router_mode_i_B         = 0;
        data_src_port_0_B       = 20;
        enable_src_port_0_B     = 0;
        ready_dst_port_0_B      = 0;
        
        #10
        result_in_step [1] = (data_dst_port_0_A == 0 && data_dst_port_0_B == 0 ) ? 1 : 0; //save result


        //Testcase 2: ready und enable are set, Output A = 10, Output B = 20
        #10
        router_mode_i_A         = 0;
        data_src_port_0_A       = 10;
        enable_src_port_0_A     = 1;
        ready_dst_port_0_A      = 1;

        router_mode_i_B         = 0;
        data_src_port_0_B       = 20;
        enable_src_port_0_B     = 1;
        ready_dst_port_0_B      = 1;

        #10
        result_in_step [2] = (data_dst_port_0_A == 10 && data_dst_port_0_B == 20 ) ? 1 : 0; //save result

        //Testcase 3: Module B is in slave mode, send out value of module A
        #10
        router_mode_i_A         = 0;
        data_src_port_0_A       = 10;
        enable_src_port_0_A     = 1;
        ready_dst_port_0_A      = 1;

        router_mode_i_B         = 1;
        data_src_port_0_B       = 20;
        enable_src_port_0_B     = 1;
        ready_dst_port_0_B      = 1;

        #10
        result_in_step [3] = (data_dst_port_0_A == 10 && data_dst_port_0_B == 10 ) ? 1 : 0; //save result

        //Testcase 4: Module A is in slave mode, send out value of module B
        #10
        router_mode_i_A         = 1;
        data_src_port_0_A       = 10;
        enable_src_port_0_A     = 1;
        ready_dst_port_0_A      = 1;

        router_mode_i_B         = 0;
        data_src_port_0_B       = 20;
        enable_src_port_0_B     = 1;
        ready_dst_port_0_B      = 1;

        #10
        result_in_step [4] = (data_dst_port_0_A == 20 && data_dst_port_0_B == 20 ) ? 1 : 0; //save result


        
        //Flag if all Testcases are passed
        #10
        for (int i = 0; i < TEST_STEP && (!(computation_was_right == 0)) ; i = i + 1) begin
            if (result_in_step[i] == 0) begin
                computation_was_right = 0;
            end
        end
             
        //END set all to zero 
        #10
        router_mode_i_A         = 0;
        data_src_port_0_A       = 0;
        enable_src_port_0_A     = 0;
        ready_dst_port_0_A      = 0;

        router_mode_i_B         = 0;
        data_src_port_0_B       = 0;
        enable_src_port_0_B     = 0;
        ready_dst_port_0_B      = 0;


        $display("Computation was right? Answer:  ", computation_was_right);

    end

endmodule