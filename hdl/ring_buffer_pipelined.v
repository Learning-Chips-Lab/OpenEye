// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

// -----------------------------------------------------------------------
// ring_buffer_pipelined
//
// Pipelined circular buffer / ring counter utilizing a dual-port RAM IP
// (RAM_DP) with registered output (Pipelined = 1).
//
// Operation & Latency:
//   When ready_i is asserted:
//     1. The current input data (data_i) is written to the RAM at wr_ptr.
//     2. A read operation is triggered from the RAM at rd_ptr.
//     3. Both wr_ptr and rd_ptr are advanced (wrapping around to 0 when
//        limit_i is reached).
//
// Because RAM_DP is configured with Pipelined = 1 (1 cycle memory read +
// 1 cycle output pipeline register), the corresponding read data arrives
// at data_o with a fixed 2-cycle latency. The valid_o signal tracks this
// latency shift register to qualify data_o.
// -----------------------------------------------------------------------
module ring_buffer_pipelined #(
    parameter ADDR_WIDTH = 8,        // Address bus width (log2 of maximum capacity)
    parameter DATA_WIDTH = 8         // Data bus width in bits
) (
    input  wire                   clk_i,
    input  wire                   rst_n,    // Asynchronous, active-low reset (for pointers & handshake)

    // Dynamic configuration
    input  wire [ADDR_WIDTH-1:0]  limit_i,  // Upper index boundary (wraps to 0 when reached)

    // Handshake & Data Interface
    input  wire                   ready_i,  // Trigger: store data_i and fetch next entry
    input  wire [DATA_WIDTH-1:0]  data_i,   // Input data word to write
    output wire [DATA_WIDTH-1:0]  data_o,   // Output data word from memory
    output reg                    valid_o   // Output valid (asserted 2 cycles after ready_i)
);

    // -------------------------------------------------------------------
    // Internal Ring Pointers
    // -------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0] wr_ptr;
    reg [ADDR_WIDTH-1:0] rd_ptr;

    // Write pointer tracking
    always @(posedge clk_i or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
        end else if (ready_i) begin
            wr_ptr <= wr_ptr + 1'b1;
            if (wr_ptr == limit_i) begin
                wr_ptr <= {ADDR_WIDTH{1'b0}};
            end
        end
    end

    // Read pointer tracking
    always @(posedge clk_i or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= 2;
        end else if (ready_i) begin
            rd_ptr <= rd_ptr + 1'b1;
            if (rd_ptr == limit_i) begin
                rd_ptr <= {ADDR_WIDTH{1'b0}};
            end
        end
    end

    // -------------------------------------------------------------------
    // Handshake Pipeline Tracking (2 Cycles for Pipelined RAM_DP)
    // -------------------------------------------------------------------
    reg valid_s1;

    always @(posedge clk_i or negedge rst_n) begin
        if (!rst_n) begin
            valid_s1 <= 1'b0;
            valid_o  <= 1'b0;
        end else begin
            valid_o <= ready_i;
        end
    end

    // -------------------------------------------------------------------
    // Dual-Port RAM IP Instantiation
    // -------------------------------------------------------------------
    RAM_DP #(
        .AddrWidth (ADDR_WIDTH),
        .DataWidth (DATA_WIDTH),
        .Pipelined (0)               // 1 = Registered read output (2 cycles read latency)
    ) ram_inst (
        .clk_i    (clk_i),
        .rd_en_i  (1'd1),
        .wr_en_i  (ready_i),
        .addr_r_i (rd_ptr),
        .addr_w_i (wr_ptr),
        .data_i   (data_i),
        .data_o   (data_o)
    );

endmodule