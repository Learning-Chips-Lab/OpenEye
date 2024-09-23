// timescale// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

//router modes

module router_iact_tb();

    localparam integer DATA_WIDTH = 8;
    localparam integer FULL_PORTS = 4;
    localparam integer NUM_MODULES = 6;

    localparam integer COLUMNS = 2;
    localparam integer ROWS = 4;


    localparam integer TEST_STEP = 8;               //only for testbench 
    string             result;

    //Testbench only
    reg                           result_in_step[0:TEST_STEP-1];      //Result vector //only for testbench
    reg                           computation_was_right = 1;          //Result flag //only for testbench


  genvar iact_y;
  genvar iact_x;
  generate
    for(iact_x = 0; iact_x < COLUMNS; iact_x = iact_x + 1) begin : gen_xx
      for(iact_y = 0; iact_y <  ROWS; iact_y = iact_y + 1) begin : gen_yy

///////////// Module IN/OUT
        reg [$clog2(FULL_PORTS)+FULL_PORTS-1:0]router_mode_i;

  //SRC Port 0
         reg                   ready_src_port_0;
         reg [DATA_WIDTH-1:0]  data_src_port_0;
         reg                   enable_src_port_0;
  //SRC Port 1
         reg                   ready_src_port_1;
         reg [DATA_WIDTH-1:0]  data_src_port_1;
         reg                   enable_src_port_1;
  //SRC Port 2
         reg                   ready_src_port_2;
         reg [DATA_WIDTH-1:0]  data_src_port_2;
         reg                   enable_src_port_2;
  //SRC Port 3
         reg                   ready_src_port_3;
         reg [DATA_WIDTH-1:0]  data_src_port_3;
         reg                   enable_src_port_3;
  //DST Port 0
         reg                   ready_dst_port_0;
         reg [DATA_WIDTH-1:0]  data_dst_port_0;
         reg                   enable_dst_port_0;
  //DST Port 1
         reg                   ready_dst_port_1;
         reg [DATA_WIDTH-1:0]  data_dst_port_1;
         reg                   enable_dst_port_1;
  //DST Port 2
         reg                   ready_dst_port_2;
         reg [DATA_WIDTH-1:0]  data_dst_port_2;
         reg                   enable_dst_port_2;
  //DST Port 3
         reg                   ready_dst_port_3;
         reg [DATA_WIDTH-1:0]  data_dst_port_3;
         reg                   enable_dst_port_3;




  ///////////// Module instantiation 
         router_iact #(
          .DATA_WIDTH(DATA_WIDTH),
          .FULL_PORTS(FULL_PORTS)

          ) router_iact (

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
         .enable_dst_port_2(enable_dst_port_2),    //output

         .ready_src_port_3(ready_src_port_3),      //output
         .data_src_port_3(data_src_port_3),        //input
         .enable_src_port_3(enable_src_port_3),    //input

         .ready_dst_port_3(ready_dst_port_3),      //input
         .data_dst_port_3(data_dst_port_3),        //output
         .enable_dst_port_3(enable_dst_port_3)     //output


         );
        end
     end


  genvar ccr,ccc;
  for (ccc=0; ccc<COLUMNS; ccc=ccc+1) begin
    for (ccr=0; ccr<ROWS; ccr=ccr+1) begin

      // Side Connection

      assign gen_xx[(ccc + 1) % COLUMNS].gen_yy[ccr].ready_dst_port_2            = gen_xx[ccc].gen_yy[ccr].ready_src_port_2;
      assign gen_xx[(ccc + 1) % COLUMNS].gen_yy[ccr].data_src_port_2             = gen_xx[ccc].gen_yy[ccr].data_dst_port_2;
      assign gen_xx[(ccc + 1) % COLUMNS].gen_yy[ccr].enable_src_port_2           = gen_xx[ccc].gen_yy[ccr].enable_dst_port_2;

      //all except last
      if(ccr != ROWS - 1)begin
          assign gen_xx[ccc].gen_yy[ccr].ready_dst_port_3         = gen_xx[ccc].gen_yy[ccr + 1].ready_src_port_1;
          assign gen_xx[ccc].gen_yy[ccr + 1].data_src_port_1      = gen_xx[ccc].gen_yy[ccr].data_dst_port_3;
          assign gen_xx[ccc].gen_yy[ccr + 1].enable_src_port_1    = gen_xx[ccc].gen_yy[ccr].enable_dst_port_3;
      end

      //all except first row
      if(ccr != 0)begin
          assign gen_xx[ccc].gen_yy[ccr].ready_dst_port_1         = gen_xx[ccc].gen_yy[ccr - 1].ready_src_port_3;
          assign gen_xx[ccc].gen_yy[ccr - 1].data_src_port_3      = gen_xx[ccc].gen_yy[ccr].data_dst_port_1;
          assign gen_xx[ccc].gen_yy[ccr - 1].enable_src_port_3    = gen_xx[ccc].gen_yy[ccr].enable_dst_port_1;
      end

/*
  //     assign gen_xx[0].gen_yy[0].data_src_port_2                     = 0;
  //     assign gen_xx[0].gen_yy[0].enable_src_port_2                   = 0;
       assign gen_xx[0].gen_yy[0].ready_dst_port_3                    = 0;

       assign gen_xx[COLUMNS].gen_yy[0].data_src_port_2                = 0;
       assign gen_xx[COLUMNS].gen_yy[0].enable_src_port_2              = 0;
       assign gen_xx[COLUMNS].gen_yy[0].ready_dst_port_3               = 0;

       assign gen_xx[0].gen_yy[ROWS].data_src_port_3                  = 0;
       assign gen_xx[0].gen_yy[ROWS].enable_src_port_3                = 0;
       assign gen_xx[0].gen_yy[ROWS].ready_dst_port_2                 = 0;

       assign gen_xx[COLUMNS].gen_yy[ROWS].data_src_port_3             = 0;
       assign gen_xx[COLUMNS].gen_yy[ROWS].enable_src_port_3           = 0;
       assign gen_xx[COLUMNS].gen_yy[ROWS].ready_dst_port_2            = 0;
*/
    end
  end


endgenerate

    // main test block
    initial begin
 // open, write and save results in .vcd file
        $dumpfile("router_iact_tb.vcd");
        $dumpvars(0,router_iact_tb);

    // set result vector to 0
        for(int j=0; j<TEST_STEP; j=j+1)begin
        result_in_step[j] = 0;        
        end


//Testcase 0: set all Inputs to Zero
   /*   
        for(int g=0; g<COLUMNS; g=g+1)begin
          for(int h=0; h<ROWS; h=h+1)begin


       gen_xx[g].gen_yy[h].router_mode_i          = 0;
       gen_xx[g].gen_yy[h].ready_dst_port_0       = 0;
       gen_xx[g].gen_yy[h].data_src_port_0        = 0;
       gen_xx[g].gen_yy[h].enable_src_port_0      = 0;
       end
        end 
     */

//Initialization

       gen_xx[0].gen_yy[0].router_mode_i          = 6'b000000;
       gen_xx[0].gen_yy[0].ready_dst_port_0       = 0;
       gen_xx[0].gen_yy[0].data_src_port_0        = 0;
       gen_xx[0].gen_yy[0].enable_src_port_0      = 0;

       gen_xx[0].gen_yy[1].router_mode_i          = 6'b000000;
       gen_xx[0].gen_yy[1].ready_dst_port_0       = 0;
       gen_xx[0].gen_yy[1].data_src_port_0        = 0;
       gen_xx[0].gen_yy[1].enable_src_port_0      = 0;

       gen_xx[0].gen_yy[2].router_mode_i          = 6'b000000;
       gen_xx[0].gen_yy[2].ready_dst_port_0       = 0;
       gen_xx[0].gen_yy[2].data_src_port_0        = 0;
       gen_xx[0].gen_yy[2].enable_src_port_0      = 0;

       gen_xx[0].gen_yy[3].router_mode_i          = 6'b000000;
       gen_xx[0].gen_yy[3].ready_dst_port_0       = 0;
       gen_xx[0].gen_yy[3].data_src_port_0        = 0;
       gen_xx[0].gen_yy[3].enable_src_port_0      = 0;

       gen_xx[1].gen_yy[0].router_mode_i          = 6'b000000;
       gen_xx[1].gen_yy[0].ready_dst_port_0       = 0;
       gen_xx[1].gen_yy[0].data_src_port_0        = 0;
       gen_xx[1].gen_yy[0].enable_src_port_0      = 0;

       gen_xx[1].gen_yy[1].router_mode_i          = 6'b000000;
       gen_xx[1].gen_yy[1].ready_dst_port_0       = 0;
       gen_xx[1].gen_yy[1].data_src_port_0        = 0;
       gen_xx[1].gen_yy[1].enable_src_port_0      = 0;

       gen_xx[1].gen_yy[2].router_mode_i          = 6'b000000;
       gen_xx[1].gen_yy[2].ready_dst_port_0       = 0;
       gen_xx[1].gen_yy[2].data_src_port_0        = 0;
       gen_xx[1].gen_yy[2].enable_src_port_0      = 0;

       gen_xx[1].gen_yy[3].router_mode_i          = 6'b000000;
       gen_xx[1].gen_yy[3].ready_dst_port_0       = 0;
       gen_xx[1].gen_yy[3].data_src_port_0        = 0;
       gen_xx[1].gen_yy[3].enable_src_port_0      = 0;






// Step 0: Module 0-0 get Data from GLB and send it to PE of 0-0
       #10
       gen_xx[0].gen_yy[0].router_mode_i          = 6'b000001;
       gen_xx[0].gen_yy[0].ready_dst_port_0       = 1;
       gen_xx[0].gen_yy[0].data_src_port_0        = 42;
       gen_xx[0].gen_yy[0].enable_src_port_0      = 1;

        #10
        result_in_step [0] =       (gen_xx[0].gen_yy[0].ready_src_port_0     == 1 
                                   && gen_xx[0].gen_yy[0].data_dst_port_0    == 42 
                                   && gen_xx[0].gen_yy[0].enable_dst_port_0  == 1                                  
                                   ) ? 1 : 0; //save result


// Step 1: Modul 0-0 send to all outputs
       #10
       gen_xx[0].gen_yy[0].router_mode_i          = 6'b001111;
       gen_xx[0].gen_yy[0].ready_dst_port_0       = 1;
       gen_xx[0].gen_yy[0].data_src_port_0        = 42;
       gen_xx[0].gen_yy[0].enable_src_port_0      = 1;



        #10
        result_in_step [1] =       (
                                      gen_xx[0].gen_yy[0].data_dst_port_0    == 42 
                                   && gen_xx[0].gen_yy[0].enable_dst_port_0  == 1

                                   && gen_xx[0].gen_yy[0].data_dst_port_1    == 42 
                                   && gen_xx[0].gen_yy[0].enable_dst_port_1  == 1

                                   && gen_xx[0].gen_yy[0].data_dst_port_2    == 42 
                                   && gen_xx[0].gen_yy[0].enable_dst_port_2  == 1

                                   && gen_xx[0].gen_yy[0].data_dst_port_3    == 42 
                                   && gen_xx[0].gen_yy[0].enable_dst_port_3  == 1

                                   ) ? 1 : 0; //save result




// Step 2: Module 0-0 get Data from GLB and send it to PE Side (1-0) 
       #10
       gen_xx[0].gen_yy[0].router_mode_i          = 6'b000100;
       gen_xx[0].gen_yy[0].ready_dst_port_0       = 0;
       gen_xx[0].gen_yy[0].data_src_port_0        = 42;
       gen_xx[0].gen_yy[0].enable_src_port_0      = 1;

       gen_xx[1].gen_yy[0].router_mode_i          = 6'b100001;
       gen_xx[1].gen_yy[0].ready_dst_port_0       = 1;

        #10
        result_in_step [2] =       (gen_xx[0].gen_yy[0].ready_src_port_0     == 1 
                                   && gen_xx[1].gen_yy[0].data_dst_port_0    == 42 
                                   && gen_xx[1].gen_yy[0].enable_dst_port_0  == 1                                  
                                   ) ? 1 : 0; //save result


// Step 3: Module 0-0 get Data from GLB and send it to PE South (0-1) 
       #10
       gen_xx[0].gen_yy[0].router_mode_i          = 6'b001000;
       gen_xx[0].gen_yy[0].ready_dst_port_0       = 0;
       gen_xx[0].gen_yy[0].data_src_port_0        = 42;
       gen_xx[0].gen_yy[0].enable_src_port_0      = 1;

       gen_xx[0].gen_yy[1].router_mode_i          = 6'b010001;
       gen_xx[0].gen_yy[1].ready_dst_port_0       = 1;

        #10
        result_in_step [3] =       (gen_xx[0].gen_yy[0].ready_src_port_0     == 1 
                                   && gen_xx[0].gen_yy[1].data_dst_port_0    == 42 
                                   && gen_xx[0].gen_yy[1].enable_dst_port_0  == 1                                  
                                   ) ? 1 : 0; //save result



       
// Step 4: Module 0-1 get Data from GLB and send it to PE North (0-0)
       #10
       gen_xx[0].gen_yy[0].router_mode_i          = 6'b110001;
       gen_xx[0].gen_yy[1].router_mode_i          = 6'b000010; 
       //01


       gen_xx[0].gen_yy[1].data_src_port_0        = 42;
       gen_xx[0].gen_yy[1].enable_src_port_0      = 1;

       gen_xx[0].gen_yy[0].ready_dst_port_0 = 1;

        #10
        result_in_step [4] =       (gen_xx[0].gen_yy[1].ready_src_port_0     == 1 
                                   && gen_xx[0].gen_yy[0].data_dst_port_0    == 42 
                                   && gen_xx[0].gen_yy[0].enable_dst_port_0  == 1                                  
                                   ) ? 1 : 0; //save result

       #10
       gen_xx[0].gen_yy[1].data_src_port_0        = 0;



// Step 5: Module 0-0 get Data from GLB and send it to PE of  1-0
//                                          send it to 0-1 and further to PE of 1-1
       #10
       gen_xx[0].gen_yy[0].router_mode_i          = 6'b001100; 
       //01
       gen_xx[0].gen_yy[1].router_mode_i          = 6'b010100;
       //01
       gen_xx[1].gen_yy[0].router_mode_i          = 6'b100001;
       //11
       gen_xx[1].gen_yy[1].router_mode_i          = 6'b100001;

       gen_xx[0].gen_yy[0].data_src_port_0        = 42;
       gen_xx[0].gen_yy[0].enable_src_port_0      = 1;

       gen_xx[1].gen_yy[1].ready_dst_port_0 = 1;
       gen_xx[1].gen_yy[0].ready_dst_port_0 = 1;

        #10
        result_in_step [5] =       (gen_xx[0].gen_yy[0].ready_src_port_0     == 1 
                                   && gen_xx[1].gen_yy[1].data_dst_port_0    == 42 
                                   && gen_xx[1].gen_yy[1].enable_dst_port_0  == 1           
                                   && gen_xx[1].gen_yy[0].data_dst_port_0    == 42 
                                   && gen_xx[1].gen_yy[0].enable_dst_port_0  == 1                        
                                   ) ? 1 : 0; //save result






// Step 6: Module 1-0 get Data from GLB and send it through 1-1 > 0-1 > 0-0 to PE of 1-0

       #10
       gen_xx[0].gen_yy[0].router_mode_i          = 6'b000100; 
       //01
       gen_xx[1].gen_yy[0].router_mode_i          = 6'b101001;




        #10
        result_in_step [6] =       (//gen_xx[0].gen_yy[0].ready_src_port_0     == 1 
                                      gen_xx[1].gen_yy[0].data_dst_port_0    == 42 
                                   && gen_xx[1].gen_yy[0].enable_dst_port_0  == 1
                                   && gen_xx[1].gen_yy[0].data_dst_port_3    == 0
                                   && gen_xx[1].gen_yy[0].enable_dst_port_3  == 0                             
                                   ) ? 1 : 0; //save result

        #10

              $display("gen_xx[1].gen_yy[0].data_dst_port_3:", gen_xx[1].gen_yy[0].data_dst_port_3);  
              $display("gen_xx[1].gen_yy[0].enable_dst_port_3:", gen_xx[1].gen_yy[0].enable_dst_port_3);                         


// Step 7: Module 0-0 get Data from GLB and send it through 0-1 > 1-1 > 1-2 > 1-3 > 0-3 to PE of 0-2

       #10

       gen_xx[1].gen_yy[1].data_src_port_0        = 42;
       gen_xx[1].gen_yy[1].enable_src_port_0      = 1;

       gen_xx[1].gen_yy[1].router_mode_i          = 6'b000010; 
       //01
       gen_xx[1].gen_yy[0].router_mode_i          = 6'b110101;



        #10
        result_in_step [7] =       (
                                      gen_xx[1].gen_yy[0].data_dst_port_0    == 42 
                                   && gen_xx[1].gen_yy[0].enable_dst_port_0  == 1
                                   && gen_xx[1].gen_yy[0].data_dst_port_2  == 0      
                                   && gen_xx[1].gen_yy[0].enable_dst_port_2  == 0                                        
                                   ) ? 1 : 0; //save result

#10
              
              $display("gen_xx[1].gen_yy[0].data_dst_port_2:", gen_xx[1].gen_yy[0].data_dst_port_2);  
              $display("gen_xx[1].gen_yy[0].enable_dst_port_2:", gen_xx[1].gen_yy[0].enable_dst_port_2);


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


      $display("Result0:", result_in_step[0]);
      $display("Result1:", result_in_step[1]);
      $display("Result2:", result_in_step[2]);
      $display("Result3:", result_in_step[3]);
      $display("Result4:", result_in_step[4]);
      $display("Result5:", result_in_step[5]);
      $display("Result6:", result_in_step[6]);
      $display("Result7:", result_in_step[7]);


       $display("");
              $display("");
                     $display("");
       #10
       $display("Result:", result);

    end
    

endmodule 