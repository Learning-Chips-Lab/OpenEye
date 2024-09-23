// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

module af_cluster_tb();

  localparam integer            DATA_BITWIDTH   = 20;
  localparam integer            MODES           = 2;

  localparam integer TEST_STEP = 6;                         //only for testbench
  


  reg                           clk_i;
  reg                           rst_ni;
  reg   [$clog2(MODES)-1 : 0]   mode_i;

  reg                           ready_o;
  reg   [DATA_BITWIDTH-1 : 0]   data_i;
  reg                           enable_i;

  reg                           ready_i;
  reg   [DATA_BITWIDTH-1 : 0]   data_o;
  reg                           enable_o;

  reg                         result_in_step[0:TEST_STEP-1];      //Result vector //only for testbench
  reg                         computation_was_right = 1;          //Result flag //only for testbench


    af_cluster #(

        .DATA_BITWIDTH(DATA_BITWIDTH),
        .MODES(MODES)

    ) af_cluster(

        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .mode_i(mode_i),
        .ready_o(ready_o),
        .data_i(data_i),
        .enable_i(enable_i),
        .ready_i(ready_i),
        .data_o(data_o),
        .enable_o(enable_o));

 
    // main test block
    initial begin
        $dumpfile("af_cluster_tb.vcd");
        $dumpvars(0, af_cluster_tb);

    // set result vector to 0
    for(int j=0; j<TEST_STEP; j=j+1)begin
    result_in_step[j] = 0;        
    end

    // set inputs to zero
    clk_i       = 0;
    rst_ni      = 0;
    mode_i      = 0;
    data_i      = 0;
    enable_i    = 0;
    ready_i     = 0;

    //Testcase: 0 -> enable_o = 1, ready_o = 1, data_o = 0
    #10
    data_i      = 0;
    enable_i    = 1;
    ready_i     = 1;

    #10
    result_in_step [0] = (enable_o == 1 && ready_o == 1 && data_o == 0) ? 1 : 0; //save result

    //Testcase: 1 -> enable_o = 0, ready_o = 0, data_o = 100
    #10
    data_i      = 100;
    enable_i    = 0;
    ready_i     = 0; 

    #10
    result_in_step [1] = (enable_o == 0 && ready_o == 0 && data_o == 100) ? 1 : 0; //save result

    //Testcase: 2 -> enable_o = 0, ready_o = 0, data_o = 0// data_i to 1048575 (max Value of 2^20-1)
    #10
    data_i      = 1048575;
    enable_i    = 0;
    ready_i     = 0;

    #10
    result_in_step [2] = (enable_o == 0 && ready_o == 0 && data_o == 0) ? 1 : 0; //save result

    //Testcase: 3 -> enable_o = 0, ready_o = 0, data_o = 524287// data_i to 524287 (max Value of 2^19-1)
    #10
    data_i      = 524287;
    enable_i    = 0;
    ready_i     = 0;

    #10
    result_in_step [3] = (enable_o == 0 && ready_o == 0 && data_o == 524287) ? 1 : 0; //save result


    //Testcase: 4 -> enable_o = 0, ready_o = 0, data_o = 0// data_i to 524288 (max Value of 2^19)
    #10
    data_i      = 524288;
    enable_i    = 0;
    ready_i     = 0;

    #10
    result_in_step [4] = (enable_o == 0 && ready_o == 0 && data_o == 0) ? 1 : 0; //save result

    //Testcase: 5 -> enable_o = 0, ready_o = 0, data_o = 101//
    #10
    data_i      = 101;
    enable_i    = 0;
    ready_i     = 0;

    #10
    result_in_step [5] = (enable_o == 0 && ready_o == 0 && data_o == 101) ? 1 : 0; //save result

    //Flag if all Testcases are passed
        #10
        for (int i = 0; i < TEST_STEP && (!(computation_was_right == 0)) ; i = i + 1) begin
            if (result_in_step[i] == 0) begin
                computation_was_right = 0;
            end
        end

   //END, set everything to zero
    #10
    data_i      = 0;
    enable_i    = 0;
    ready_i     = 0;

    $display("Computation was right? Answer:  ", computation_was_right);


    end

endmodule