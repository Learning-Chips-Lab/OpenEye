// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

module mux_iact_tb();

    localparam integer WIDTH = 20;
    localparam integer I_COUNT = 3;

    localparam integer TEST_STEP = 7;                         //only for testbench verification

    reg     [WIDTH-1:0]                 a_i[0:I_COUNT-1];
    reg                                 b_i[0:I_COUNT-1];
    wire                                c_o[0:I_COUNT-1];
    reg     [$clog2(I_COUNT)-1:0]       sel_i;
    wire    [WIDTH-1:0]                 a_o;
    wire                                b_o;
    reg                                 c_i;

    reg                                 result_in_step[0:TEST_STEP-1];      //Result vector //only for testbench verification
    reg                                 computation_was_right = 1;          //Result flag //only for testbench verification



    mux_iact #(

        .WIDTH(WIDTH),
        .I_COUNT(I_COUNT)
           
        ) Mux_iact (

        .a_i(a_i),
        .b_i(b_i),
        .c_i(c_i),
        .a_o(a_o),
        .b_o(b_o),
        .c_o(c_o),
        .sel_i(sel_i));

 
    // main test block
    initial begin
        $dumpfile("mux_iact_tb.vcd");
        $dumpvars(0, mux_iact_tb);


    // set result vector to 0
    for(int j=0; j<TEST_STEP; j=j+1)begin
    result_in_step[j] = 1;        
    end

    // set everything to zero
    sel_i = 0;
    c_i = 0;
    for(int k=0; k<I_COUNT; k=k+1)begin
        a_i[k] = 0;
        b_i[k] = 0;        
    end

a_i[0]= 1;
b_i[0]= 1;

a_i[1]= 8'b00110011;
b_i[1]= 0;

a_i[2]= 5;
b_i[2]= 1;

//select 0
sel_i = 0;

#10
result_in_step [0] = (a_o == 1 && b_o == 1 ) ? 1 : 0; //save result


//select 1
#10
sel_i = 1;

#10
result_in_step [1] = (a_o == 8'b00110011 && b_o == 0 ) ? 1 : 0; //save result


//select 2
#10
sel_i = 2;

#10
result_in_step [2] = (a_o == 5 && b_o == 1 ) ? 1 : 0; //save result


//select 3 => a_o and b_o shall be zero
#10
sel_i = 3;

#10
result_in_step [3] = (a_o == 0 && b_o == 0 ) ? 1 : 0; //save result


// c_o[0] shall be 0, all other c_o[] shall be 1
#10
sel_i = 0;
c_i = 0;

#10
result_in_step [4] = (c_o[0] == 0 && c_o[1] == 1 && c_o[2] == 1 ) ? 1 : 0; //save result

// c_o[1] shall be 0, all other c_o[] shall be 1
#10
sel_i = 1;
c_i = 0;

#10
result_in_step [5] = (c_o[0] == 1 && c_o[1] == 0 && c_o[2] == 1 ) ? 1 : 0; //save result

// c_o[2] shall be 0, all other c_o[] shall be 1
#10
sel_i = 2;
c_i = 0;

#10
result_in_step [6] = (c_o[0] == 1 && c_o[1] == 1 && c_o[2] == 0 ) ? 1 : 0; //save result

//Flag if all Testcases are passed
    #10
    for (int i = 0; i < TEST_STEP && (!(computation_was_right == 0)) ; i = i + 1) begin
        if (result_in_step[i] == 0) begin
            computation_was_right = 0;
        end
    end

// END
#10

a_i[0]= 0;
b_i[0]= 0;

a_i[1]= 0;
b_i[1]= 0;

a_i[2]= 0;
b_i[2]= 0;

c_i = 0;


$display("Computation was right? Answer:  ", computation_was_right);



    end

endmodule