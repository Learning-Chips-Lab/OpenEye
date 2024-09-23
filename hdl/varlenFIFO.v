// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: varlenFIFO
///
/// VariableLengthFirstInFirstOut (varlenFIFO) is a FIFO memory that can store data in 
/// words of different lengths. varlenFIFO contains a memory that can be accessed by read and write.
/// These operations can even happen simultaneously. It then creates memory via pointers.
///
/// Parameters:
///    DEPTH                  - Amount of storageable data words
///    DATA_WIDTH             - Length of data words
///   
/// Ports:
///    wr_en                  - Write Enable Port
///    rd_en                  - Read Enable Port
///    new_stream_i           - Port for resetting pointers and therefore deleting storage
///    data_i                 - Data Port In
///    data_o                 - Data Port Out
///    empty                  - Port for signaling, that storage is empty
///    full                   - Port for signaling, that storage is full
///

module varlenFIFO 
#(
  parameter DATA_WIDTH = 8,
  parameter DEPTH = 16
)(
  input wire                    clk_i,
  input wire                    rst_ni,
  input wire                    wr_en,
  input wire                    rd_en,
  input wire                    new_stream_i,
  input wire [DATA_WIDTH-1:0]   data_i,
  output reg [DATA_WIDTH-1:0]   data_o,
  output reg                    empty,
  output reg                    full
);

  reg [DATA_WIDTH-1:0]       flat_help_var; // Needed for flattening packed arrays
  reg [DATA_WIDTH*DEPTH-1:0] memory;

  reg [$clog2(DEPTH)-1:0]    write_ptr; // Write pointer address
  reg [$clog2(DEPTH)-1:0]    read_ptr; // Read pointer address
  reg [$clog2(DEPTH):0]      count;


  // Read & Write process
  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin  // Reset
      read_ptr      <= 0;
      count         <= 0;
      data_o        <= 0;
      write_ptr     <= 0;
      flat_help_var <= 0;
      empty         <= 1;
      full          <= 0;
    end
    else begin
      if (rd_en) begin // Reading process
        if (count > 0) begin // Check, if there is data to read
          flat_help_var = 0;
          for (int b=0; b<DATA_WIDTH; b=b+1) begin
            flat_help_var[b] = memory[read_ptr * DATA_WIDTH + b];
          end
          data_o        <= flat_help_var;
          flat_help_var  = 0;
          read_ptr      <= read_ptr + 1;
          count         <= count - 1;
          full          <= 0;
          if ((count - 1) == 0) begin
            empty <= 0;
          end
        end
        else begin
          data_o <= data_i;
        end
      end

      if (wr_en && count < DEPTH) begin // Writing process
        flat_help_var = data_i;
        for (int b=0; b<DATA_WIDTH; b=b+1) begin
          memory[write_ptr * DATA_WIDTH + b] = flat_help_var[b];
        end
        flat_help_var      = 0;
        write_ptr         <= write_ptr + 1;
        count             <= count + 1;
        empty             <= 0;
        if ((count + 1) == DEPTH) begin
          full <= 1;
        end
      end

      if (rd_en && wr_en) begin // If Read and Write occur simultaneous
        count <= count;
        empty <= empty;
        full  <= full;
      end

      if (new_stream_i) begin // If new stream is sent
        write_ptr <= 0;
        read_ptr  <= 0;
        count     <= 0;
        empty     <= 1;
        full      <= 0;
      end
    end
  end

endmodule
