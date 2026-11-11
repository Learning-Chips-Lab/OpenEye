// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

// -----------------------------------------------------------------------
// ring_buffer_pipelined (1-Cycle Latency Version)
// -----------------------------------------------------------------------
module ring_buffer_pipelined #(
    parameter ADDR_WIDTH = 8,        // Address bus width
    parameter DATA_WIDTH = 8         // Data bus width in bits
) (
    input  wire                  clk_i,
    input  wire                  rst_n,      // Asynchronous, active-low reset

    // Dynamic configuration
    input  wire [ADDR_WIDTH-1:0] limit_i,    // Upper index boundary
    input  wire                  set_pointer_start_i,

    // Handshake & Data Interface
    input  wire                  ready_i,    // Trigger: store data_i and fetch next entry
    input  wire [DATA_WIDTH-1:0] data_i,     // Input data word
    output wire [DATA_WIDTH-1:0] data_o,     // Output data word from memory
    output reg                   valid_o     // Output valid (asserted 1 cycle after ready_i)
);

    // -------------------------------------------------------------------
    // Internal Ring Pointers
    // -------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0] wr_ptr;
    reg [ADDR_WIDTH-1:0] rd_ptr;

    // Write pointer tracking
    always @(posedge clk_i or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= {ADDR_WIDTH{1'b0}};
        end else begin
            if (set_pointer_start_i) begin
                wr_ptr <= {ADDR_WIDTH{1'b0}};
            end else if (ready_i) begin
                if (wr_ptr >= limit_i) begin
                    wr_ptr <= {ADDR_WIDTH{1'b0}};
                end else begin
                    wr_ptr <= wr_ptr + 1'b1;
                end
            end
        end
    end

    // Read pointer tracking
    always @(posedge clk_i or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= {{ADDR_WIDTH-1{1'b0}}, 1'b1}; // Initialize to 1
        end else begin
            if (set_pointer_start_i) begin
                rd_ptr <= {{ADDR_WIDTH-1{1'b0}}, 1'b1};
                if (limit_i == 0) begin
                    rd_ptr <= {ADDR_WIDTH{1'b0}};
                end
            end else if (ready_i) begin
                rd_ptr <= rd_ptr + 1'b1;
                if (rd_ptr >= limit_i) begin
                    rd_ptr <= {ADDR_WIDTH{1'b0}};
                end
            end
        end
    end

    // -------------------------------------------------------------------
    // Handshake Pipeline Tracking (1 Cycle Latency)
    // -------------------------------------------------------------------
    always @(posedge clk_i or negedge rst_n) begin
        if (!rst_n) begin
            valid_o <= 1'b0;
        end else begin
            valid_o <= ready_i;
        end
    end

    // -------------------------------------------------------------------
    // Forwarding / Write-Through Bypass Logic (Optional support for same-cycle Read/Write collision)
    // -------------------------------------------------------------------
    reg                  bypass_en;
    reg [DATA_WIDTH-1:0] bypass_data;
    wire [DATA_WIDTH-1:0] ram_data_o;

    always @(posedge clk_i or negedge rst_n) begin
        if (!rst_n) begin
            bypass_en   <= 1'b0;
            bypass_data <= {DATA_WIDTH{1'b0}};
        end else begin
            // If writing to the exact same address we are currently reading from
            bypass_en   <= ready_i && (rd_ptr == wr_ptr);
            bypass_data <= data_i;
        end
    end

    assign data_o = bypass_en ? bypass_data : ram_data_o;

    // -------------------------------------------------------------------
    // Dual-Port RAM IP Instantiation (Pipelined = 0 -> 1 Cycle Read Latency)
    // -------------------------------------------------------------------
    RAM_DP #(
        .AddrWidth (ADDR_WIDTH),
        .DataWidth (DATA_WIDTH),
        .Pipelined (0)                // 0 = 1 cycle read latency (Unregistered output)
    ) ram_inst (
        .clk_i    (clk_i),
        .rd_en_i  (ready_i),          // Read enable directly tied to ready_i
        .wr_en_i  (ready_i & (limit_i != 0)),
        .addr_r_i (rd_ptr),
        .addr_w_i (wr_ptr),
        .data_i   (data_i),
        .data_o   (ram_data_o)
    );

endmodule