// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

module demux2_tb();

    localparam integer DATA_WIDTH = 1;
    localparam integer TEST_STEP = 4;       //only for testbench

    
    
    reg     [DATA_WIDTH-1:0]    a_out_tb;
    reg     [DATA_WIDTH-1:0]    b_out_tb;
    reg                         sel_i_tb;
    reg     [DATA_WIDTH-1:0]    i_tb;
    reg                         result_in_step[0:TEST_STEP-1];      //Result vector //only for testbench
    reg                         computation_was_right = 1;          //Result flag //only for testbench


    demux2 #(
        .DATA_WIDTH(DATA_WIDTH)

    ) demux2 (
        .sel_i(sel_i_tb),
        .i(i_tb),
        .a_out(a_out_tb),
        .b_out(b_out_tb));

 
    // main test block
    initial begin
        $dumpfile("demux2_tb.vcd");
        $dumpvars(0, demux2_tb);
    
    // set result vector to 0
        for(int j=0; j<TEST_STEP; j=j+1)begin
        result_in_step[j] = 0;        
        end
    // set inputs to 0
        i_tb = 0;
        sel_i_tb = 0;

    //Testcase: 0 -> a_out = 0, b_out = 1
        #10
        i_tb = 1;
        sel_i_tb=0;

        #10
        result_in_step [0] = (a_out_tb == 0 && b_out_tb == 1) ? 1 : 0; //save result

    //Testcase: 1 -> a_out = 1, b_out = 0
        #10
        i_tb = 1;
        sel_i_tb=1;

        #10
        result_in_step [1] = (a_out_tb == 1 && b_out_tb == 0) ? 1 : 0; //save result

    //Testcase: 2 -> a_out = 0, b_out = 1
        #10
        i_tb = 1048577;
        sel_i_tb=0;

        #10
        result_in_step [2] = (a_out_tb == 0 && b_out_tb == 1) ? 1 : 0; //save result

    //Testcase: 3 -> a_out = 1, b_out = 0
        #10
        i_tb = 1048577;
        sel_i_tb=1;

        #10
        result_in_step [3] = (a_out_tb == 1 && b_out_tb == 0) ? 1 : 0; //save result

    //Flag if all Testcases are passed
        #10
        for (int i = 0; i < TEST_STEP && (!(computation_was_right == 0)) ; i = i + 1) begin
            if (result_in_step[i] == 0) begin
                computation_was_right = 0;
            end
        end

    //END set everything to 0
        i_tb = 0;
        sel_i_tb = 0;

    $display("Computation was right? Answer:  ", computation_was_right);
      

    end







endmodule