// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

module mux2_tb();

    localparam integer DATA_WIDTH = 20;
    localparam integer TEST_STEP = 4;                              //only for testbench 

    
    reg     [DATA_WIDTH-1:0]    a_in_tb;                            //input a
    reg     [DATA_WIDTH-1:0]    b_in_tb;                            //input b
    reg                         sel_i_tb;                           //input selection
    reg     [DATA_WIDTH-1:0]    y_o_tb;                             //output
    reg                         result_in_step[0:TEST_STEP-1];      //Result vector //only for testbench
    reg                         computation_was_right = 1;          //Result flag //only for testbench


    mux2  #(
        .DATA_WIDTH(DATA_WIDTH)
     ) 

    mux2 (

        .sel_i(sel_i_tb),
        .a_in(a_in_tb),
        .b_in(b_in_tb),
        .y_o(y_o_tb));

 
    // main test block
    initial begin

    // open, write and save results in .vcd file
        $dumpfile("mux2_tb.vcd");
        $dumpvars(0, mux2_tb);


    // set result vector to 0
        for(int j=0; j<TEST_STEP; j=j+1)begin
        result_in_step[j] = 0;        
        end
    // set inputs to 0
        a_in_tb = 0;
        b_in_tb = 0;
        sel_i_tb = 0;

    //Testcase: 0 -> y_o shall be 0
        #10
        result_in_step [0] = (y_o_tb == 0) ? 1 : 0; //save result


    //Testcase: 1 -> y_o shall be 10
        #10 
        a_in_tb = 11;
        b_in_tb = 10;
        sel_i_tb = 0;
        #10
        result_in_step [1] = (y_o_tb == 10) ? 1 : 0; //save result


    //Testcase: 2 -> y_o shall be 11
        #10 
        a_in_tb = 11;
        b_in_tb = 10;
        sel_i_tb = 1;
        #10
        result_in_step [2] = (y_o_tb == 11) ? 1 : 0; //save result


    //Testcase: 3 -> y_o shall be 0
        #10 
        a_in_tb = 0;
        b_in_tb = 10;
        sel_i_tb = 1;
        #10
        result_in_step [3] = (y_o_tb == 0) ? 1 : 0; //save result


    //Flag if all Testcases are passed
        #10
        for (int i = 0; i < TEST_STEP && (!(computation_was_right == 0)) ; i = i + 1) begin
            if (result_in_step[i] == 0) begin
                computation_was_right = 0;
            end
        end

    // END set all to 0
        #10 
        a_in_tb = 0;
        b_in_tb = 0;
        sel_i_tb = 0;

        $display("Computation was right? Answer:  ", computation_was_right);
        

    end
    

endmodule