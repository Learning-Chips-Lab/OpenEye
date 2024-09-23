// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

module multiplier_tb();

    localparam integer DATA_WIDTH_FAC1 = 8;
    localparam integer DATA_WIDTH_FAC2 = 8;
    localparam integer DATA_WIDTH_PROD = 20;
    localparam integer Q_BITWIDTH      = $clog2(DATA_WIDTH_PROD);

    localparam integer TEST_STEP = 20;                         //only for testbench verification


    reg                                 clk_i_tb;
    reg                                 rst_ni_tb;
    reg                                 multiplier_en_i_tb;
    reg signed  [DATA_WIDTH_FAC1-1:0]   factor1_tb;
    reg signed  [DATA_WIDTH_FAC2-1:0]   factor2_tb;
    wire signed [DATA_WIDTH_PROD-1:0]   product_tb;
    reg         [Q_BITWIDTH-1:0]        fraction_bit_i;

    reg                                 result_in_step[0:TEST_STEP-1];      //Result vector //only for testbench verification
    reg                                 computation_was_right = 1;          //Result flag //only for testbench verification

    integer j;
    integer k;

    multiplier #(

        .DATA_WIDTH_FAC1(DATA_WIDTH_FAC1),
        .DATA_WIDTH_FAC2(DATA_WIDTH_FAC2),
        .DATA_WIDTH_PROD(DATA_WIDTH_PROD),
        .Q_BITWIDTH(Q_BITWIDTH)

    ) multiplier(

        .clk_i(clk_i_tb),
        .rst_ni(rst_ni_tb),
        .multiplier_en_i(multiplier_en_i_tb),
        .factor_1(factor1_tb),
        .factor_2(factor2_tb),
        .product(product_tb),
        .fraction_bit_i(fraction_bit_i));

 
    // main test block
    initial begin
        $dumpfile("multiplier_tb.vcd");
        $dumpvars(0, multiplier_tb);

    // set result vector to 0
    for(int j=0; j<TEST_STEP; j=j+1)begin
    result_in_step[j] = 0;        
    end

//Testcase: 0 -> set everything to zero
    clk_i_tb = 0;
    rst_ni_tb = 0;
    multiplier_en_i_tb = 0;
    factor1_tb = 0;
    factor2_tb = 0;

#10
    clk_i_tb = 1;
    rst_ni_tb = 0;
    multiplier_en_i_tb = 0;
    factor1_tb = 0;
    factor2_tb = 0;

#10
    result_in_step [0] = (product_tb == 0) ? 1 : 0; //save result

//Testcase: 1 -> enable off
#10 
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 0;
    factor1_tb = 10;
    factor2_tb = 10;

#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 0;
    factor1_tb = 10;
    factor2_tb = 10;

#10
    result_in_step [1] = (product_tb == 0) ? 1 : 0; //save result

//Testcase: 2 -> enable on
#10 
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 10;
    factor2_tb = 10;

#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 10;
    factor2_tb = 10;
#10
    result_in_step [2] = (product_tb == 100) ? 1 : 0; //save result

//Testcase: 3 -> reset low
#10 
    clk_i_tb = 0;
    rst_ni_tb = 0;
    multiplier_en_i_tb = 1;
    factor1_tb = 10;
    factor2_tb = 10;

#10
    clk_i_tb = 1;
    rst_ni_tb = 0;
    multiplier_en_i_tb = 1;
    factor1_tb = 10;
    factor2_tb = 10;
#10
    result_in_step [3] = (product_tb == 0) ? 1 : 0; //save result

    //Testcase: 4
    //Wert pos. grade pos. grade =100
 #10   
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 10;
    factor2_tb = 10;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 10;
    factor2_tb = 10;
#10
    result_in_step [4] = (product_tb == 100) ? 1 : 0; //save result


    //Testcase: 5
    //Wert pos. ungrade pos. ungrade =121
#10
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 11;
    factor2_tb = 11;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 11;
    factor2_tb = 11;

#10
    result_in_step [5] = (product_tb == 121) ? 1 : 0; //save result

    //Testcase: 6
    //Wert pos grade pos ungrade =110
#10    
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 10;
    factor2_tb = 11;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 10;
    factor2_tb = 11;
#10
    result_in_step [6] = (product_tb == 110) ? 1 : 0; //save result

    //Testcase: 7
    //Wert pos ungrade pos grade =110
#10    
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 11;
    factor2_tb = 10;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 11;
    factor2_tb = 10;
#10
    result_in_step [7] = (product_tb == 110) ? 1 : 0; //save result

    //Testcase: 8
    //Wert neg. grade neg. grade =100
#10    
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -10;
    factor2_tb = -10;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -10;
    factor2_tb = -10;
#10
    result_in_step [8] = (product_tb == 100) ? 1 : 0; //save result

    //Testcase: 9
    //Wert neg. ungrade neg. ungrade =121
#10    
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -11;
    factor2_tb = -11;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -11;
    factor2_tb = -11;
#10
    result_in_step [9] = (product_tb == 121) ? 1 : 0; //save result

    //Testcase: 10
    //Wert neg grade neg ungrade =110
#10    
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -10;
    factor2_tb = -11;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -10;
    factor2_tb = -11;

#10
    result_in_step [10] = (product_tb == 110) ? 1 : 0; //save result

    //Testcase: 11
    //Wert neg ungrade neg grade =110
#10
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -11;
    factor2_tb = -10;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -11;
    factor2_tb = -10;
#10
    result_in_step [11] = (product_tb == 110) ? 1 : 0; //save result

    //Testcase: 12
    //Wert pos. grade neg. grade =-100
#10    
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 10;
    factor2_tb = -10;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 10;
    factor2_tb = -10;
#10
    result_in_step [12] = (product_tb == -100) ? 1 : 0; //save result

    //Testcase: 13
    //Wert pos. ungrade neg. ungrade =-121
#10    
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 11;
    factor2_tb = -11;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 11;
    factor2_tb = -11;
#10
    result_in_step [13] = (product_tb == -121) ? 1 : 0; //save result

    //Testcase: 14
    //Wert pos grade neg ungrade =-110
#10    
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 10;
    factor2_tb = -11;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 10;
    factor2_tb = -11;
#10
    result_in_step [14] = (product_tb == -110) ? 1 : 0; //save result

    //Testcase: 15
    //Wert pos ungrade neg ungrade =-121
#10    
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 11;
    factor2_tb = -11;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = 11;
    factor2_tb = -11;
#10
    result_in_step [15] = (product_tb == -121) ? 1 : 0; //save result

    //Testcase: 16
    //Wert neg. grade pos. grade =-100
#10    
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -10;
    factor2_tb = 10;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -10;
    factor2_tb = 10;
#10
    result_in_step [16] = (product_tb == -100) ? 1 : 0; //save result

    //Testcase: 17
    //Wert neg. ungrade pos. ungrade =-121
#10    
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -11;
    factor2_tb = 11;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -11;
    factor2_tb = 11;
#10
    result_in_step [17] = (product_tb == -121) ? 1 : 0; //save result

    //Testcase: 18
    //Wert neg grade pos ungrade =-110
#10    
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -10;
    factor2_tb = 11;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -10;
    factor2_tb = 11;
#10
    result_in_step [18] = (product_tb == -110) ? 1 : 0; //save result

    //Testcase: 19
    //Wert neg ungrade pos grade =-110
#10    
    clk_i_tb = 0;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -11;
    factor2_tb = 10;
#10
    clk_i_tb = 1;
    rst_ni_tb = 1;
    multiplier_en_i_tb = 1;
    factor1_tb = -11;
    factor2_tb = 10;
#10
    result_in_step [19] = (product_tb == -110) ? 1 : 0; //save result



//Flag if all Testcases are passed
    #10
    for (int i = 0; i < TEST_STEP && (!(computation_was_right == 0)) ; i = i + 1) begin
        if (result_in_step[i] == 0) begin
            computation_was_right = 0;
        end
    end

//17 ENDE
//RESET =0
#10    
    clk_i_tb = 0;
    rst_ni_tb = 0;
    multiplier_en_i_tb = 1;
    factor1_tb = -11;
    factor2_tb = 10;
#10
    clk_i_tb = 1;
    rst_ni_tb = 0;
    multiplier_en_i_tb = 1;
    factor1_tb = -11;
    factor2_tb = 10;


    $display("Computation was right? Answer:  ", computation_was_right);


    

    end

/*    for(j=0; j<2**factor1_tb; j=j+1)begin
        factor1_tb = j;
        for(k=0;k<2**factor2_tb;k=k+1)begin
            factor2_tb = k;
            end
    end*/

endmodule