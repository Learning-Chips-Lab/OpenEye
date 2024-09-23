// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

module PE_cluster_tb();
  //Set parameter integers
  parameter integer DATA_BITWIDTH_IACT = 24;
  parameter integer DATA_BITWIDTH_PSUM = 20;
  parameter integer DATA_BITWIDTH_WGHT = 24;
  //Set Number of Router Clusters per Instance, will at a later stage be calculated, is dependent on PE Elements
  parameter integer NUM_GLB_IACT = 3;
  parameter integer NUM_GLB_WGHT = 3;
  parameter integer NUM_GLB_PSUM = 4;
  //Set Number of PE elements
  parameter integer PE_ROWS = NUM_GLB_WGHT;
  parameter integer PE_COLUMNS = NUM_GLB_PSUM;

  parameter integer IACT_MTRX_X = 6;
  parameter integer IACT_MTRX_Y = 3;
  parameter integer WGHT_MTRX_X = 3;
  parameter integer WGHT_MTRX_Y = 3;


  reg                           clk;
  reg                           reset_n;
  reg                           data_mode;
  reg [$clog2(NUM_GLB_IACT)-1:0]iact_choose              [0:PE_COLUMNS-1] [0:PE_ROWS-1];
  reg                           psum_choose              [0:NUM_GLB_PSUM-1];
  reg                           compute_i                [0:PE_COLUMNS-1] [0:PE_ROWS-1];
  
  reg [DATA_BITWIDTH_IACT-1:0]  pe_iact_data             [0:NUM_GLB_IACT-1];
  reg                           pe_iact_enable           [0:NUM_GLB_IACT-1];
  wire                          pe_iact_ready            [0:NUM_GLB_IACT-1];
  
  reg [DATA_BITWIDTH_WGHT-1:0]  pe_wght_data             [0:NUM_GLB_WGHT-1];
  reg                           pe_wght_enable           [0:NUM_GLB_WGHT-1];
  wire                          pe_wght_ready            [0:NUM_GLB_WGHT-1];
  
  reg                           pe_psum_ready_in         [0:NUM_GLB_PSUM-1];
  reg  [DATA_BITWIDTH_PSUM-1:0] pe_psum_data_in          [0:NUM_GLB_PSUM-1];
  reg                           pe_psum_enable_in        [0:NUM_GLB_PSUM-1];

  wire                          pe_psum_ready_out        [0:NUM_GLB_PSUM-1];
  wire [DATA_BITWIDTH_PSUM-1:0] pe_psum_data_out         [0:NUM_GLB_PSUM-1];
  wire                          pe_psum_enable_out       [0:NUM_GLB_PSUM-1];

  reg                           pe_router_psum_ready_in  [0:NUM_GLB_PSUM-1];
  reg [DATA_BITWIDTH_PSUM-1:0]  pe_router_psum_data_in   [0:NUM_GLB_PSUM-1];
  reg                           pe_router_psum_enable_in [0:NUM_GLB_PSUM-1];

  wire                          pe_router_psum_ready_out [0:NUM_GLB_PSUM-1];
  wire [DATA_BITWIDTH_PSUM-1:0] pe_router_psum_data_out  [0:NUM_GLB_PSUM-1];
  wire                          pe_router_psum_enable_out[0:NUM_GLB_PSUM-1];
  //Choose mode for the Router Clusters
  reg [1:0]router_mode_iact;
  reg      router_mode_wght;
  reg [1:0]router_mode_psum;

  reg [$clog2(DATA_BITWIDTH_PSUM)-1:0]fraction_bit;

  PE_cluster #(
    .TOP_CLUSTER(1),
    .TRANS_BITWIDTH_IACT(DATA_BITWIDTH_IACT),
    .TRANS_BITWIDTH_WGHT(DATA_BITWIDTH_WGHT),
    .TRANS_BITWIDTH_PSUM(DATA_BITWIDTH_PSUM),
    .NUM_GLB_IACT(NUM_GLB_IACT),
    .NUM_GLB_WGHT(NUM_GLB_WGHT),
    .NUM_GLB_PSUM(NUM_GLB_PSUM),
    .PE_ROWS(PE_ROWS),
    .PE_COLUMNS(PE_COLUMNS)
  )pe_cluster(
    .clk_i                    (clk),
    .rst_ni                   (reset_n),
    .data_mode_i              (data_mode),
    .iact_choose_i            (iact_choose),
    .psum_choose_i            (psum_choose),
    .compute_i                (compute_i),

    .pe_iact_data             (pe_iact_data),
    .pe_iact_enable           (pe_iact_enable),
    .pe_iact_ready            (pe_iact_ready),
    
    .pe_wght_data             (pe_wght_data),
    .pe_wght_enable           (pe_wght_enable),
    .pe_wght_ready            (pe_wght_ready),
    
    .pe_psum_ready_in         (pe_psum_ready_in),
    .pe_psum_data_in          (pe_psum_data_in),
    .pe_psum_enable_in        (pe_psum_enable_in),

    .pe_psum_ready_out        (pe_psum_ready_out),
    .pe_psum_data_out         (pe_psum_data_out),
    .pe_psum_enable_out       (pe_psum_enable_out),

    .pe_router_psum_ready_in  (pe_router_psum_ready_in),
    .pe_router_psum_data_in   (pe_router_psum_data_in),
    .pe_router_psum_enable_in (pe_router_psum_enable_in),

    .pe_router_psum_ready_out (pe_router_psum_ready_out),
    .pe_router_psum_data_out  (pe_router_psum_data_out),
    .pe_router_psum_enable_out(pe_router_psum_enable_out),
    .fraction_bit_i           (fraction_bit)
  );
    
  integer clk_prd = 10;
  //Do clock
  always begin
    clk = 0; #(clk_prd/2);
    clk = 1; #(clk_prd/2);
    //0.1GHz
  end
  
  reg     [11:0]iact_handler         [5:0];
  reg     [11:0]wght_handler         [5:0];
	integer       file_2                              = 0;
	integer       reading_cycle2                      = 0;
	integer       line_count_wght                     = 0;
	integer       line_count_iact                     = 0;
  reg     [7:0] temp_storage_8_bit2  [0:2];
  reg     [11:0]temp_storage_12_bit2 [0:1];
  reg     [11:0]temp_storage_weight1 [0:191][0:15];
  reg     [11:0]temp_storage_weight2 [0:191][0:15];
  reg     [11:0]temp_storage_weight3 [0:191][0:15];
  reg     [11:0]temp_storage_weight4 [0:191][0:15];
  reg     [11:0]temp_storage_iact1    [0:5][0:15];
  reg     [11:0]temp_storage_iact2    [0:5][0:15];
	integer       args1;
	integer       args2;
	integer       file_1                              = 0;
	integer       reading_cycle1                      = 0;
  reg     [3:0] temp_storage_4_bit1  [0:5];
  reg     [11:0]temp_storage_12_bit1 [0:1];
  integer c,d;
  initial begin
    $dumpfile("OpenEye_v2.vcd");
    $dumpvars;
    data_mode<= '0;
    fraction_bit   <= 0;
    for (c=0; c<PE_COLUMNS; c=c+1) for (d=0; d<PE_ROWS; d=d+1) iact_choose [c][d] <= 0;
    for (c=0; c<NUM_GLB_PSUM; c=c+1) psum_choose [c]                              <= 'b0;

    for (c=0; c<NUM_GLB_IACT; c=c+1) pe_iact_data [c]                             <= 'b0;
    for (c=0; c<NUM_GLB_IACT; c=c+1) pe_iact_enable [c]                           <= 'b0;
    for (c=0; c<6; c=c+1) for (d=0; d<16; d=d+1) temp_storage_iact1 [c][d]        <= 'b0;
    for (c=0; c<6; c=c+1) for (d=0; d<16; d=d+1) temp_storage_iact2 [c][d]        <= 'b0;

    for (c=0; c<NUM_GLB_WGHT; c=c+1) pe_wght_data [c]                             <= 'b0;
    for (c=0; c<NUM_GLB_WGHT; c=c+1) pe_wght_enable [c]                           <= 'b0;
    for (c=0; c<192; c=c+1) for (d=0; d<16; d=d+1) temp_storage_weight1 [c][d]    <= 'b0;
    for (c=0; c<192; c=c+1) for (d=0; d<16; d=d+1) temp_storage_weight2 [c][d]    <= 'b0;
    for (c=0; c<192; c=c+1) for (d=0; d<16; d=d+1) temp_storage_weight3 [c][d]    <= 'b0;
    for (c=0; c<192; c=c+1) for (d=0; d<16; d=d+1) temp_storage_weight4 [c][d]    <= 'b0;

    for (c=0; c<NUM_GLB_PSUM; c=c+1) pe_psum_ready_in [c]                         <= 'b0;
    for (c=0; c<NUM_GLB_PSUM; c=c+1) pe_psum_data_in [c]                          <= 'b0;
    for (c=0; c<NUM_GLB_PSUM; c=c+1) pe_psum_enable_in [c]                        <= 'b0;

    for (c=0; c<NUM_GLB_PSUM; c=c+1) pe_router_psum_ready_in [c]                  <= 'b0;
    for (c=0; c<NUM_GLB_PSUM; c=c+1) pe_router_psum_data_in [c]                   <= 'b0;
    for (c=0; c<NUM_GLB_PSUM; c=c+1) pe_router_psum_enable_in [c]                 <= 'b0;


    for (c=0; c<PE_COLUMNS; c=c+1) for (d=0; d<PE_ROWS; d=d+1) compute_i [c][d] <= 0;
    reset_n = 0; #30;
    reset_n = 1;
    #5
    pe_wght_enable[0] <= 1;
    pe_wght_enable[1] <= 1;
    pe_wght_enable[2] <= 1;
    file_2 = $fopen("../PE_cluster_design/addr_wght.txt","r");
    while (!$feof(file_2))begin
      while((!$feof(file_2)) & (reading_cycle2 < 3))begin
        args1 = $fscanf(file_2,"%d\n",temp_storage_8_bit2[reading_cycle2]);
        $display("Writing value %0d",temp_storage_8_bit2[reading_cycle2]);
        reading_cycle2++;
      end
      pe_wght_data[0] <= {temp_storage_8_bit2[2],temp_storage_8_bit2[1],temp_storage_8_bit2[0]};
      pe_wght_data[1] <= {temp_storage_8_bit2[2],temp_storage_8_bit2[1],temp_storage_8_bit2[0]};
      pe_wght_data[2] <= {temp_storage_8_bit2[2],temp_storage_8_bit2[1],temp_storage_8_bit2[0]};
      reading_cycle2 <= 0;
      #(reading_cycle2 * clk_prd);
    end
    $fclose(file_2);
    file_2 = $fopen("../PE_cluster_design/data_wght1.csv","r");
    while (!$feof(file_2))begin
      args1 = $fscanf(file_2,"%d,%d,%d\n",wght_handler[0],wght_handler[1],wght_handler[2]);
      temp_storage_weight1[0][line_count_wght] = wght_handler[0];
      temp_storage_weight1[1][line_count_wght] = wght_handler[1];
      temp_storage_weight1[2][line_count_wght] = wght_handler[2];
      $display("line:  %0d",line_count_wght);


      line_count_wght = line_count_wght + 1;
    end
    line_count_wght = 0;
		$fclose(file_2);
    file_2 = $fopen("../PE_cluster_design/data_wght2.csv","r");
    while (!$feof(file_2))begin
      args1 = $fscanf(file_2,"%d,%d,%d\n",wght_handler[0],wght_handler[1],wght_handler[2]);
      temp_storage_weight2[0][line_count_wght] = wght_handler[0];
      temp_storage_weight2[1][line_count_wght] = wght_handler[1];
      temp_storage_weight2[2][line_count_wght] = wght_handler[2];
      $display("line:  %0d",line_count_wght);


      line_count_wght = line_count_wght + 1;
    end
    line_count_wght = 0;
		$fclose(file_2);
    file_2 = $fopen("../PE_cluster_design/data_wght3.csv","r");
    while (!$feof(file_2))begin
      args1 = $fscanf(file_2,"%d,%d,%d\n",wght_handler[0],wght_handler[1],wght_handler[2]);
      temp_storage_weight3[0][line_count_wght] = wght_handler[0];
      temp_storage_weight3[1][line_count_wght] = wght_handler[1];
      temp_storage_weight3[2][line_count_wght] = wght_handler[2];
      $display("line:  %0d",line_count_wght);


      line_count_wght = line_count_wght + 1;
    end
    line_count_wght = 0;
		$fclose(file_2);
    file_2 = $fopen("../PE_cluster_design/data_wght4.csv","r");
    while (!$feof(file_2))begin
      args1 = $fscanf(file_2,"%d,%d,%d\n",wght_handler[0],wght_handler[1],wght_handler[2]);
      temp_storage_weight4[0][line_count_wght] = wght_handler[0];
      temp_storage_weight4[1][line_count_wght] = wght_handler[1];
      temp_storage_weight4[2][line_count_wght] = wght_handler[2];
      $display("line:  %0d",line_count_wght);


      line_count_wght = line_count_wght + 1;
    end
		$fclose(file_2);
    line_count_wght = 0;
    while((line_count_wght < 96))begin
      pe_wght_enable[0] <= 1;
      pe_wght_enable[1] <= 1;
      pe_wght_enable[2] <= 1;
      if((line_count_wght %2) != 1)begin
        pe_wght_data[0]   <= {temp_storage_weight2[line_count_wght/2][0],temp_storage_weight1[line_count_wght/2][0]};
        pe_wght_data[1]   <= {temp_storage_weight2[line_count_wght/2][1],temp_storage_weight1[line_count_wght/2][1]};
        pe_wght_data[2]   <= {temp_storage_weight2[line_count_wght/2][2],temp_storage_weight1[line_count_wght/2][2]};
      end else begin
        pe_wght_data[0]   <= {temp_storage_weight4[line_count_wght/2][0],temp_storage_weight3[line_count_wght/2][0]};
        pe_wght_data[1]   <= {temp_storage_weight4[line_count_wght/2][1],temp_storage_weight3[line_count_wght/2][1]};
        pe_wght_data[2]   <= {temp_storage_weight4[line_count_wght/2][2],temp_storage_weight3[line_count_wght/2][2]};
      end
      #(2 * clk_prd);
      
      line_count_wght    = line_count_wght + 1;
      reading_cycle2    = 0;
    end
    pe_wght_enable[0] <= 0;
    pe_wght_enable[1] <= 0;
    pe_wght_enable[2] <= 0;
    for (c=0; c<PE_COLUMNS; c=c+1) for (d=0; d<PE_ROWS; d=d+1) compute_i [c][d]        <= 1;
    #10
    for (c=0; c<PE_COLUMNS; c=c+1) for (d=0; d<PE_ROWS; d=d+1) compute_i [c][d]        <= 0;
    psum_choose[0] <= 1;
    #1000
    pe_router_psum_ready_in[0] <= 1;
    #2000
    while(!pe_router_psum_ready_out[0])begin
      #10;
    end
    pe_router_psum_enable_in[0] <= 1;
    #160
    pe_router_psum_enable_in[0] <= 0;
    #100
    #10 $finish;
  end
	
  initial begin
    
    #35
    pe_iact_enable[0] <= 1;
    pe_iact_enable[1] <= 1;
    pe_iact_enable[2] <= 1;
		file_1 = $fopen("../PE_cluster_design/addr_iact.txt","r");		
    while (!$feof(file_1))begin
      while((!$feof(file_1)) & (reading_cycle1 < 6))begin
        args1 = $fscanf(file_1,"%d\n",temp_storage_4_bit1[reading_cycle1]);
			  $display("Writing value %0d",temp_storage_4_bit1[reading_cycle1]);
        reading_cycle1++;
      end
      pe_iact_data[0] <= {temp_storage_4_bit1[5],temp_storage_4_bit1[4],temp_storage_4_bit1[3],temp_storage_4_bit1[2],temp_storage_4_bit1[1],temp_storage_4_bit1[0]};
      reading_cycle1 <= 0;
      #(reading_cycle1 * clk_prd);
    end
		$fclose(file_1);
    for (c=0; c<PE_COLUMNS; c=c+1) for (d=0; d<PE_ROWS; d=d+1) iact_choose [c][d] <= 3;
		file_1 = $fopen("../PE_cluster_design/data_iact1.csv","r");

    while (!$feof(file_1))begin
      args1 = $fscanf(file_1,"%d,%d,%d,%d,%d,%d\n",iact_handler[0],iact_handler[1],iact_handler[2],iact_handler[3],iact_handler[4],iact_handler[5]);
      temp_storage_iact1[0][line_count_iact] = iact_handler[0];
      temp_storage_iact1[1][line_count_iact] = iact_handler[1];
      temp_storage_iact1[2][line_count_iact] = iact_handler[2];
      temp_storage_iact1[3][line_count_iact] = iact_handler[3];
      temp_storage_iact1[4][line_count_iact] = iact_handler[4];
      temp_storage_iact1[5][line_count_iact] = iact_handler[5];


      line_count_iact = line_count_iact + 1;
    end		
    line_count_iact = 0;
		$fclose(file_1);
    
    file_1 = $fopen("../PE_cluster_design/data_iact2.csv","r");

    while (!$feof(file_1))begin
      args1 = $fscanf(file_1,"%d,%d,%d,%d,%d,%d\n",iact_handler[0],iact_handler[1],iact_handler[2],iact_handler[3],iact_handler[4],iact_handler[5]);
      temp_storage_iact2[0][line_count_iact] = iact_handler[0];
      temp_storage_iact2[1][line_count_iact] = iact_handler[1];
      temp_storage_iact2[2][line_count_iact] = iact_handler[2];
      temp_storage_iact2[3][line_count_iact] = iact_handler[3];
      temp_storage_iact2[4][line_count_iact] = iact_handler[4];
      temp_storage_iact2[5][line_count_iact] = iact_handler[5];


      line_count_iact = line_count_iact + 1;
    end
		$fclose(file_1);
    for (int iact_cycle=0; iact_cycle<2; iact_cycle=iact_cycle+1)begin
      for (int needed_cycles=0; needed_cycles<8; needed_cycles=needed_cycles+1)begin
        for (c=0; c<PE_COLUMNS; c=c+1)begin
          for (d=0; d<PE_ROWS; d=d+1)begin
            if((c+d) == (iact_cycle*3))begin
              iact_choose [c][d] <= 0;
            end else if((c+d) == (1+iact_cycle*3))begin
              iact_choose [c][d] <= 1;
            end else if((c+d) == (2+iact_cycle*3))begin
              iact_choose [c][d] <= 2;
            end else begin
              iact_choose [c][d] <= 3;
            end
          end
        end
          pe_iact_data[0] <= {temp_storage_iact2[iact_cycle*3][needed_cycles],temp_storage_iact1[iact_cycle*3][needed_cycles]};
          pe_iact_data[1] <= {temp_storage_iact2[1+iact_cycle*3][needed_cycles],temp_storage_iact1[1+iact_cycle*3][needed_cycles]};
          pe_iact_data[2] <= {temp_storage_iact2[2+iact_cycle*3][needed_cycles],temp_storage_iact1[2+iact_cycle*3][needed_cycles]};

        #(2 * clk_prd);
      end
    end
    pe_iact_enable[0] <= 0;
    pe_iact_enable[1] <= 0;
    pe_iact_enable[2] <= 0;
		$fclose(file_1);
  end
endmodule