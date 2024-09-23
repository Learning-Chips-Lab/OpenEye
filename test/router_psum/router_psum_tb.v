// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps
module router_psum_tb();

    localparam integer DATA_WIDTH = 20;
    localparam integer FULL_PORTS = 3;

    localparam integer TEST_STEP = 5;               //only for testbench 
    localparam integer ROWS = 4;
    string             result;

//Testbench only
  reg                           result_in_step[0:TEST_STEP-1];      //Result vector //only for testbench
  reg                           computation_was_right = 1;          //Result flag //only for testbench
  int a,b,c,d;


  genvar iact_y;
  generate
      for(iact_y = 0; iact_y <  ROWS; iact_y = iact_y + 1) begin : gen_yy

  reg  [2:0]                    router_mode_i;              // input router mode
  //SRC Port 0 GLB
  reg                           ready_src_port_0;           // output
  reg  [DATA_WIDTH-1:0]         data_src_port_0;            // input
  reg                           enable_src_port_0;          // input
  //DST Port 0 GLB
  reg                           ready_dst_port_0;           // input
  reg  [DATA_WIDTH-1:0]         data_dst_port_0;            // output
  reg                           enable_dst_port_0;          // output

  //SRC Port 1 PE
  reg                           ready_src_port_1;           // output
  reg  [DATA_WIDTH-1:0]         data_src_port_1;            // input
  reg                           enable_src_port_1;          // input
  //DST Port 1 PE 
  reg                           ready_dst_port_1;           // input
  reg  [DATA_WIDTH-1:0]         data_dst_port_1;            // output
  reg                           enable_dst_port_1;          // output

  //SRC Port 2 Neighbour
  reg                           ready_src_port_2;           // output
  reg  [DATA_WIDTH-1:0]         data_src_port_2;            // input
  reg                           enable_src_port_2;          // input
  //DST Port 2 Neighbour
  reg                           ready_dst_port_2;           // input
  reg  [DATA_WIDTH-1:0]         data_dst_port_2;            // output
  reg                           enable_dst_port_2;          // output



  //PE
  reg  [DATA_WIDTH-1:0]         In_PE_data_1;
  reg  [DATA_WIDTH-1:0]         In_PE_data_2;
  reg  [DATA_WIDTH-1:0]         Out_PE_data;

  reg                           In_PE_enable_1;
  reg                           In_PE_enable_2;
  reg                           Out_PE_enable;

  wire                          In_PE_ready_1;
  wire                          In_PE_ready_2;
  reg                           Out_PE_ready;





///////////// Module instantiation 
         router_psum #(
          .DATA_WIDTH(DATA_WIDTH),
          .FULL_PORTS(FULL_PORTS)

          ) router_psum (

         .router_mode_i(router_mode_i),
  
         .ready_src_port_0(ready_src_port_0),      //output
         .data_src_port_0(data_src_port_0),        //input
         .enable_src_port_0(enable_src_port_0),    //input

         .ready_dst_port_0(ready_dst_port_0),      //input
         .data_dst_port_0(data_dst_port_0),        //output
         .enable_dst_port_0(enable_dst_port_0),    //output

         .ready_src_port_1(ready_src_port_1),      //output
         .data_src_port_1(data_src_port_1),        //input
         .enable_src_port_1(enable_src_port_1),    //input

         .ready_dst_port_1(ready_dst_port_1),      //input
         .data_dst_port_1(data_dst_port_1),        //output
         .enable_dst_port_1(enable_dst_port_1),    //output

         .ready_src_port_2(ready_src_port_2),      //output
         .data_src_port_2(data_src_port_2),        //input
         .enable_src_port_2(enable_src_port_2),    //input

         .ready_dst_port_2(ready_dst_port_2),      //input
         .data_dst_port_2(data_dst_port_2),        //output
         .enable_dst_port_2(enable_dst_port_2)     //output

         );
      end


////////  Router -> PE
      assign  gen_yy[0].In_PE_ready_1   = gen_yy[0].ready_src_port_1;
      assign  gen_yy[0].In_PE_data_1    = gen_yy[0].data_dst_port_1;
      assign  gen_yy[0].In_PE_enable_1  = gen_yy[0].enable_dst_port_1;

      assign  gen_yy[1].In_PE_ready_1   = gen_yy[1].ready_src_port_1;
      assign  gen_yy[1].In_PE_data_1    = gen_yy[1].data_dst_port_1;
      assign  gen_yy[1].In_PE_enable_1  = gen_yy[1].enable_dst_port_1;

      assign  gen_yy[2].In_PE_ready_1   = gen_yy[2].ready_src_port_1;
      assign  gen_yy[2].In_PE_data_1    = gen_yy[2].data_dst_port_1;
      assign  gen_yy[2].In_PE_enable_1  = gen_yy[2].enable_dst_port_1;

      assign  gen_yy[3].In_PE_ready_1   = gen_yy[3].ready_src_port_1;
      assign  gen_yy[3].In_PE_data_1    = gen_yy[3].data_dst_port_1;
      assign  gen_yy[3].In_PE_enable_1  = gen_yy[3].enable_dst_port_1;

////////  PE -> Router
      assign  gen_yy[0].ready_dst_port_1    =   gen_yy[0].Out_PE_ready;
      assign  gen_yy[0].data_src_port_1     =   gen_yy[0].Out_PE_data;
      assign  gen_yy[0].enable_src_port_1   =   gen_yy[0].Out_PE_enable;

      assign  gen_yy[1].ready_dst_port_1    =   gen_yy[1].Out_PE_ready;
      assign  gen_yy[1].data_src_port_1     =   gen_yy[1].Out_PE_data;
      assign  gen_yy[1].enable_src_port_1   =   gen_yy[1].Out_PE_enable;     

      assign  gen_yy[2].ready_dst_port_1    =   gen_yy[2].Out_PE_ready;
      assign  gen_yy[2].data_src_port_1     =   gen_yy[2].Out_PE_data;
      assign  gen_yy[2].enable_src_port_1   =   gen_yy[2].Out_PE_enable;

      assign  gen_yy[3].ready_dst_port_1    =   gen_yy[3].Out_PE_ready;
      assign  gen_yy[3].data_src_port_1     =   gen_yy[3].Out_PE_data;
      assign  gen_yy[3].enable_src_port_1   =   gen_yy[3].Out_PE_enable;

////////  PE -> PE
      assign  gen_yy[0].In_PE_ready_2       =   gen_yy[1].Out_PE_ready;
      assign  gen_yy[0].In_PE_data_2        =   gen_yy[1].Out_PE_data;
      assign  gen_yy[0].In_PE_enable_2      =   gen_yy[1].Out_PE_enable;

      assign  gen_yy[1].In_PE_ready_2       =   gen_yy[2].Out_PE_ready;
      assign  gen_yy[1].In_PE_data_2        =   gen_yy[2].Out_PE_data;
      assign  gen_yy[1].In_PE_enable_2      =   gen_yy[2].Out_PE_enable;

      assign  gen_yy[2].In_PE_ready_2       =   gen_yy[3].Out_PE_ready;
      assign  gen_yy[2].In_PE_data_2        =   gen_yy[3].Out_PE_data;
      assign  gen_yy[2].In_PE_enable_2      =   gen_yy[3].Out_PE_enable;




          genvar yyy;
            for (yyy=0; yyy<ROWS; yyy=yyy+1) begin

      //alles außer letzte Reihe
      if(yyy != ROWS - 1)begin
          assign gen_yy[yyy].ready_dst_port_2         = gen_yy[yyy + 1].ready_src_port_2;
          assign gen_yy[yyy + 1].data_src_port_2      = gen_yy[yyy].data_dst_port_2;
          assign gen_yy[yyy + 1].enable_src_port_2    = gen_yy[yyy].enable_dst_port_2;
      end
            end
      endgenerate

//MUX PE
always @* begin
  if (gen_yy[0].router_mode_i == 0 ||
      gen_yy[0].router_mode_i == 3 || 
      gen_yy[0].router_mode_i == 4 || 
      gen_yy[0].router_mode_i == 7) begin

      gen_yy[0].Out_PE_enable   <= gen_yy[0].In_PE_enable_1;
      gen_yy[0].Out_PE_data     <= gen_yy[0].In_PE_data_1;
      gen_yy[0].Out_PE_ready    <= gen_yy[0].In_PE_ready_1;
  end
  else begin
      gen_yy[0].Out_PE_enable   <= gen_yy[0].In_PE_enable_2;
      gen_yy[0].Out_PE_data     <= gen_yy[0].In_PE_data_2;
      gen_yy[0].Out_PE_ready    <= gen_yy[0].In_PE_ready_2;
  end

  if (gen_yy[1].router_mode_i == 0 ||
      gen_yy[1].router_mode_i == 3 || 
      gen_yy[1].router_mode_i == 4 || 
      gen_yy[1].router_mode_i == 7) begin

      gen_yy[1].Out_PE_enable   <= gen_yy[1].In_PE_enable_1;
      gen_yy[1].Out_PE_data     <= gen_yy[1].In_PE_data_1;
      gen_yy[1].Out_PE_ready    <= gen_yy[1].In_PE_ready_1;
  end
  else begin
      gen_yy[1].Out_PE_enable   <= gen_yy[1].In_PE_enable_2;
      gen_yy[1].Out_PE_data     <= gen_yy[1].In_PE_data_2;
      gen_yy[1].Out_PE_ready    <= gen_yy[1].In_PE_ready_2;
  end

  if (gen_yy[2].router_mode_i == 0 ||
      gen_yy[2].router_mode_i == 3 || 
      gen_yy[2].router_mode_i == 4 || 
      gen_yy[2].router_mode_i == 7) begin

      gen_yy[2].Out_PE_enable   <= gen_yy[2].In_PE_enable_1;
      gen_yy[2].Out_PE_data     <= gen_yy[2].In_PE_data_1;
      gen_yy[2].Out_PE_ready    <= gen_yy[2].In_PE_ready_1;
      end
  else begin
      gen_yy[2].Out_PE_enable   <= gen_yy[2].In_PE_enable_2;
      gen_yy[2].Out_PE_data     <= gen_yy[2].In_PE_data_2;
      gen_yy[0].Out_PE_ready    <= gen_yy[2].In_PE_ready_2; 
  end

  if (gen_yy[3].router_mode_i == 0 ||
      gen_yy[3].router_mode_i == 3 || 
      gen_yy[3].router_mode_i == 4 || 
      gen_yy[3].router_mode_i == 7) begin

      gen_yy[3].Out_PE_enable   <= gen_yy[3].In_PE_enable_1;
      gen_yy[3].Out_PE_data     <= gen_yy[3].In_PE_data_1;
      gen_yy[3].Out_PE_ready    <= gen_yy[3].In_PE_ready_1;
  end
  else begin
      gen_yy[3].Out_PE_enable   <= gen_yy[3].In_PE_enable_2;
      gen_yy[3].Out_PE_data     <= gen_yy[3].In_PE_data_2;
      gen_yy[3].Out_PE_ready    <= gen_yy[3].In_PE_ready_2;
  end
end

    initial begin

    // open, write and save results in .vcd file
        $dumpfile("router_psum_tb.vcd");
        $dumpvars(0,router_psum_tb);

    // set result vector to 0
        for(int j=0; j<TEST_STEP; j=j+1)begin
        result_in_step[j] = 0;        
        end



gen_yy[0].router_mode_i = 0;
gen_yy[1].router_mode_i = 0;
gen_yy[2].router_mode_i = 0;
gen_yy[3].router_mode_i = 0;

gen_yy[0].ready_dst_port_0    = 0;
gen_yy[0].data_src_port_0     = 0;
gen_yy[0].enable_src_port_0   = 0;

gen_yy[1].ready_dst_port_0    = 0;
gen_yy[1].data_src_port_0     = 0;
gen_yy[1].enable_src_port_0   = 0;

gen_yy[2].ready_dst_port_0    = 0;
gen_yy[2].data_src_port_0     = 0;
gen_yy[2].enable_src_port_0   = 0;

gen_yy[3].ready_dst_port_0    = 0;
gen_yy[3].data_src_port_0     = 0;
gen_yy[3].enable_src_port_0   = 0;


//Initial Test
      #10
      result_in_step [0] =    (gen_yy[0].data_dst_port_0 == 0 
                              && gen_yy[1].data_dst_port_0 == 0
                              && gen_yy[2].data_dst_port_0 == 0
                              && gen_yy[3].data_dst_port_0 == 0) ? 1 : 0; //save result


//Test 1
    #10
    gen_yy[0].router_mode_i = 0;

    #10
    gen_yy[0].ready_dst_port_0    = 1;
    gen_yy[0].data_src_port_0     = 25;
    gen_yy[0].enable_src_port_0   = 1;

    #10
    result_in_step [1] = (      gen_yy[0].ready_src_port_0  == 1
                            &&  gen_yy[0].data_dst_port_0   == 25
                            &&  gen_yy[0].enable_dst_port_0 == 1    ) ? 1 : 0; //save result



//Test 2
    #10
    gen_yy[0].router_mode_i = 5;
    gen_yy[1].router_mode_i = 2;
    gen_yy[2].router_mode_i = 3;

    #10
    gen_yy[0].ready_dst_port_0    = 1;
    gen_yy[0].data_src_port_0     = 5;
    gen_yy[0].enable_src_port_0   = 1;

    #10
    result_in_step [2] = (      gen_yy[0].ready_dst_port_0  == 1
                            &&  gen_yy[0].data_dst_port_0   == 5
                            &&  gen_yy[0].enable_dst_port_0 == 1 ) ? 1 : 0; //save result




//Test 3

    #10
    gen_yy[1].router_mode_i = 6;
    gen_yy[2].router_mode_i = 3;
    gen_yy[0].router_mode_i = 1;

    #10
    gen_yy[0].ready_dst_port_0    = 0;
    gen_yy[0].data_src_port_0     = 0;
    gen_yy[0].enable_src_port_0   = 0;

    #10
    gen_yy[1].ready_dst_port_0    = 1;
    gen_yy[1].data_src_port_0     = 10;
    gen_yy[1].enable_src_port_0   = 1;

    #10
    result_in_step [3] = (      gen_yy[1].ready_dst_port_0  == 1
                            &&  gen_yy[1].data_dst_port_0   == 10
                            &&  gen_yy[1].enable_dst_port_0 == 1  ) ? 1 : 0; //save result

//Test 4

    #10
    gen_yy[1].router_mode_i = 7;
    gen_yy[0].router_mode_i = 1;

    #10
    gen_yy[1].ready_dst_port_0    = 1;
    gen_yy[1].data_src_port_0     = 15;
    gen_yy[1].enable_src_port_0   = 1;

    #10
    result_in_step [4] = (      gen_yy[1].ready_dst_port_0  == 1
                            &&  gen_yy[1].data_dst_port_0   == 15
                            &&  gen_yy[1].enable_dst_port_0 == 1  ) ? 1 : 0; //save result




        //Flag if all Testcases are passed
        #10
        for (int i = 0; i < TEST_STEP && (!(computation_was_right == 0)) ; i = i + 1) begin
            if (result_in_step[i] == 0) begin
                computation_was_right = 0;
            end
        end

        #10
        if(computation_was_right == 0)
          result = "   FAILED";
        else
          result = "   PASSED";


      #10
      $display("Result0:", result_in_step[0]);
      $display("Result1:", result_in_step[1]);
      $display("Result1:", result_in_step[2]);
      $display("Result1:", result_in_step[3]);
      $display("Result1:", result_in_step[4]);
      $display("Result:",  result);





    end


endmodule