// This file is part of the OpenEye project.
// All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: PE_ice_wrapper
///
/// A simplified wrapper around the PE module designed for iCE40 FPGA implementation.
/// This wrapper provides a minimal 8-bit interface that multiplexes access to the
/// various PE input and output ports through control signals.
/// It is intended for testing and demonstration purposes on iCE40 FPGAs and debugging
/// via limited I/O pins.
///
/// Interface:
/// - data_i[7:0]: 8-bit data input, multiplexed to various PE inputs based on control_i
/// - control_i[7:0]: 8-bit control input that selects which PE port to access
/// - data_o[7:0]: 8-bit data output, multiplexed from various PE outputs based on control_i
/// - clk_i: Clock input
/// - rst_i: Active-high reset input

module PE_ice_wrapper (
    input  wire [7:0] data_i,
    input  wire [7:0] control_i,
    output reg  [7:0] data_o,
    input  wire       clk_i,
    input  wire       rst_i
);

    // Control signal decoding
    // control_i[3:0] - selects input port mapping
    // control_i[7:4] - selects output port mapping
    wire [3:0] input_select  = control_i[3:0];
    wire [3:0] output_select = control_i[7:4];

    // PE module signals
    wire                      pe_clk;
    wire                      pe_rst_n;
    wire [1:0]                pe_iact_select;
    wire [71:0]               pe_iact_data;     // 3 * 24 bits for NUM_GLB_IACT=3
    wire [2:0]                pe_iact_enable;
    wire [2:0]                pe_iact_ready;
    wire [23:0]               pe_wght_data;
    wire                      pe_wght_enable;
    wire                      pe_wght_ready;
    wire [19:0]               pe_psum_data_i;
    wire                      pe_psum_enable_i;
    wire                      pe_psum_ready_i;
    wire                      pe_psum_ready_o;
    wire [19:0]               pe_psum_data_o;
    wire                      pe_psum_enable_o;
    wire                      pe_compute;
    wire                      pe_enable_stream;
    wire [11:0]               pe_data_stream;

    // Internal registers for PE inputs
    reg [71:0]  iact_data_reg;
    reg [2:0]   iact_enable_reg;
    reg [1:0]   iact_select_reg;
    reg [23:0]  wght_data_reg;
    reg         wght_enable_reg;
    reg [19:0]  psum_data_i_reg;
    reg         psum_enable_i_reg;
    reg         psum_ready_i_reg;
    reg         compute_reg;
    reg         enable_stream_reg;
    reg [11:0]  data_stream_reg;

    // Connect clock and reset (convert active-high reset to active-low)
    assign pe_clk = clk_i;
    assign pe_rst_n = ~rst_i;

    // Map internal registers to PE inputs
    assign pe_iact_data      = iact_data_reg;
    assign pe_iact_enable    = iact_enable_reg;
    assign pe_iact_select    = iact_select_reg;
    assign pe_wght_data      = wght_data_reg;
    assign pe_wght_enable    = wght_enable_reg;
    assign pe_psum_data_i    = psum_data_i_reg;
    assign pe_psum_enable_i  = psum_enable_i_reg;
    assign pe_psum_ready_i   = psum_ready_i_reg;
    assign pe_compute        = compute_reg;
    assign pe_enable_stream  = enable_stream_reg;
    assign pe_data_stream    = data_stream_reg;

    // Input multiplexing - update internal registers based on input_select
    always @(posedge clk_i) begin
        if (rst_i) begin
            iact_data_reg      <= 72'h0;
            iact_enable_reg    <= 3'h0;
            iact_select_reg    <= 2'h0;
            wght_data_reg      <= 24'h0;
            wght_enable_reg    <= 1'b0;
            psum_data_i_reg    <= 20'h0;
            psum_enable_i_reg  <= 1'b0;
            psum_ready_i_reg   <= 1'b0;
            compute_reg        <= 1'b0;
            enable_stream_reg  <= 1'b0;
            data_stream_reg    <= 12'h0;
        end else begin
            case (input_select)
                4'h0: iact_data_reg[7:0]   <= data_i;      // IACT data byte 0
                4'h1: iact_data_reg[15:8]  <= data_i;      // IACT data byte 1
                4'h2: iact_data_reg[23:16] <= data_i;      // IACT data byte 2
                4'h3: iact_data_reg[31:24] <= data_i;      // IACT data byte 3
                4'h4: wght_data_reg[7:0]   <= data_i;      // WGHT data byte 0
                4'h5: wght_data_reg[15:8]  <= data_i;      // WGHT data byte 1
                4'h6: wght_data_reg[23:16] <= data_i;      // WGHT data byte 2
                4'h7: psum_data_i_reg[7:0] <= data_i;      // PSUM input byte 0
                4'h8: psum_data_i_reg[15:8] <= data_i;     // PSUM input byte 1
                4'h9: psum_data_i_reg[19:16] <= data_i[3:0]; // PSUM input byte 2 (only 4 bits)
                4'hA: data_stream_reg[7:0]  <= data_i;     // Data stream byte 0
                4'hB: data_stream_reg[11:8] <= data_i[3:0]; // Data stream byte 1 (only 4 bits)
                4'hC: begin
                    iact_enable_reg   <= data_i[2:0];
                    iact_select_reg   <= data_i[4:3];
                    wght_enable_reg   <= data_i[5];
                    psum_enable_i_reg <= data_i[6];
                    psum_ready_i_reg  <= data_i[7];
                end
                4'hD: begin
                    compute_reg       <= data_i[0];
                    enable_stream_reg <= data_i[1];
                end
                default: begin
                    // No change for undefined selects
                end
            endcase
        end
    end

    // Output multiplexing - select which PE output to route to data_o
    always @(*) begin
        case (output_select)
            4'h0: data_o = {5'b0, pe_iact_ready};          // IACT ready signals
            4'h1: data_o = {7'b0, pe_wght_ready};          // WGHT ready signal
            4'h2: data_o = {7'b0, pe_psum_ready_o};        // PSUM ready output
            4'h3: data_o = pe_psum_data_o[7:0];            // PSUM data output byte 0
            4'h4: data_o = pe_psum_data_o[15:8];           // PSUM data output byte 1
            4'h5: data_o = {4'b0, pe_psum_data_o[19:16]};  // PSUM data output byte 2 (only 4 bits)
            4'h6: data_o = {7'b0, pe_psum_enable_o};       // PSUM enable output
            default: data_o = 8'h00;
        endcase
    end

    // Instantiate the PE module
    PE #(
        .IS_TOPLEVEL(1),
        .SERIAL(0),
        .CREATE_VCD(0),
        .PE_X(0),
        .PE_Y(0),
        .PARALLEL_MACS(2),
        .DATA_IACT_BITWIDTH(8),
        .DATA_WGHT_BITWIDTH(8),
        .DATA_PSUM_BITWIDTH(20),
        .DATA_IACT_OVERHEAD(4),
        .DATA_WGHT_IGNORE_ZEROS(4),
        .IACT_DATA_ADDR(16),
        .IACT_ADDR_ADDR(9),
        .WGHT_DATA_ADDR(96),
        .WGHT_ADDR_ADDR(16),
        .PSUM_ADDR(32),
        .TRANS_BITWIDTH_IACT(24),
        .TRANS_BITWIDTH_WGHT(24),
        .NUM_GLB_IACT(3)
    ) pe_inst (
        .clk_i(pe_clk),
        .rst_ni(pe_rst_n),
        .iact_select_i(pe_iact_select),
        .iact_data_i(pe_iact_data),
        .iact_enable_i(pe_iact_enable),
        .iact_ready_o(pe_iact_ready),
        .wght_data_i(pe_wght_data),
        .wght_enable_i(pe_wght_enable),
        .wght_ready_o(pe_wght_ready),
        .psum_data_i(pe_psum_data_i),
        .psum_enable_i(pe_psum_enable_i),
        .psum_ready_i(pe_psum_ready_i),
        .psum_ready_o(pe_psum_ready_o),
        .psum_data_o(pe_psum_data_o),
        .psum_enable_o(pe_psum_enable_o),
        .compute_i(pe_compute),
        .enable_stream_i(pe_enable_stream),
        .data_stream_i(pe_data_stream)
    );

endmodule
