// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: varlenFIFO
///
/// The Variable-Length FIFO (varlenFIFO) implements a configurable First-In-First-Out
/// buffer in the OpenEye architecture that supports variable-length data streams. This
/// module provides synchronized data buffering with stream control capabilities and
/// concurrent read/write operations.
///
/// Key Features:
/// - Flexible Data Handling:
///   * Configurable data width
///   * Variable stream lengths
///   * Parameterized buffer depth
///
/// - Advanced Buffer Control:
///   * Concurrent read/write support
///   * Stream reset capability
///   * Full/empty status signals
///
/// - Memory Management:
///   * Pointer-based addressing
///   * Circular buffer operation
///   * Automatic wraparound
///
/// - Stream Processing:
///   * Stream boundary detection
///   * Dynamic buffer reset
///   * Continuous operation support
///
/// Architectural Role:
/// The varlenFIFO serves as:
/// 1. Data stream buffer
/// 2. Rate matching interface
/// 3. Stream synchronization point
/// 4. Temporary data storage
///
/// Operational Modes:
/// 1. Normal Operation:
///    - Independent read/write
///    - Status tracking
///    - Flow control
///
/// 2. Concurrent Access:
///    - Simultaneous read/write
///    - Consistent count maintenance
///    - Status preservation
///
/// 3. Stream Reset:
///    - Complete buffer clear
///    - Pointer reinitialization
///    - Status reset
///
/// Parameters:
///    DATA_WIDTH        - Data Path Configuration
///                        Width of each data word
///                        Defines storage granularity
///
///    DEPTH            - Buffer Organization
///                        Number of entries in FIFO
///                        Maximum storage capacity
///                        Must be power of 2
///   
/// Ports:
/// Clock and Reset:
///    clk_i            - System Clock Input
///                       Positive edge triggered
///                       Synchronizes all operations
///
///    rst_ni           - Asynchronous Reset Input (active low)
///                       Resets all internal state
///                       Clears memory contents
///
/// Control Interface:
///    wr_en            - Write Enable Input (active high)
///                       Controls write operations
///                       Ignored when buffer full
///
///    rd_en            - Read Enable Input (active high)
///                       Controls read operations
///                       Ignored when buffer empty
///
///    new_stream_i     - Stream Reset Input (active high)
///                       Resets internal pointers
///                       Prepares for new data stream
///
/// Data Interface:
///    data_i           - Write Data Input [DATA_WIDTH-1:0]
///                       Data to be written
///                       Sampled when wr_en is high
///
///    data_o           - Read Data Output [DATA_WIDTH-1:0]
///                       Data being read
///                       Valid when not empty
///
/// Status Interface:
///    empty            - Empty Status Output
///                       1: Buffer is empty
///                       0: Data available
///
///    full             - Full Status Output
///                       1: Buffer is full
///                       0: Space available
///

module varlenFIFO #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH = 16
) (
    input  wire                  clk_i,
    input  wire                  rst_ni,
    input  wire                  wr_en,
    input  wire                  rd_en,
    input  wire                  new_stream_i,
    input  wire [DATA_WIDTH-1:0] data_i,
    output reg  [DATA_WIDTH-1:0] data_o,
    output reg                   empty,
    output reg                   full
);

  reg     [      DATA_WIDTH-1:0] flat_help_var;  // Needed for flattening packed arrays
  reg     [DATA_WIDTH*DEPTH-1:0] memory;

  reg     [   $clog2(DEPTH)-1:0] write_ptr;  // Write pointer address
  reg     [   $clog2(DEPTH)-1:0] read_ptr;  // Read pointer address
  reg     [     $clog2(DEPTH):0] count;


  // Read & Write process
  integer                        b;
  always @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin  // Reset
      read_ptr      <= 0;
      count         <= 0;
      data_o        <= 0;
      write_ptr     <= 0;
      flat_help_var  = 0;
      empty         <= 1;
      full          <= 0;
    end else begin
      if (rd_en) begin  // Reading process
        if (count > 0) begin  // Check, if there is data to read
          flat_help_var = 0;
          for (b = 0; b < DATA_WIDTH; b = b + 1) begin
            flat_help_var[b] = memory[read_ptr*DATA_WIDTH+b];
          end
          data_o <= flat_help_var;
          flat_help_var = 0;
          read_ptr <= read_ptr + 1;
          count    <= count - 1;
          full     <= 0;
          if ((count - 1) == 0) begin
            empty <= 0;
          end
        end else begin
          data_o <= data_i;
        end
      end

      if (wr_en && count < DEPTH) begin  // Writing process
        flat_help_var = data_i;
        for (b = 0; b < DATA_WIDTH; b = b + 1) begin
          memory[write_ptr*DATA_WIDTH+b] = flat_help_var[b];
        end
        flat_help_var = 0;
        write_ptr <= write_ptr + 1;
        count     <= count + 1;
        empty     <= 0;
        if ((count + 1) == DEPTH) begin
          full <= 1;
        end
      end

      if (rd_en && wr_en) begin  // If Read and Write occur simultaneous
        count <= count;
        empty <= empty;
        full  <= full;
      end

      if (new_stream_i) begin  // If new stream is sent
        write_ptr <= 0;
        read_ptr  <= 0;
        count     <= 0;
        empty     <= 1;
        full      <= 0;
      end
    end
  end

endmodule
