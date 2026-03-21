// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: PE_AXI_wrapper
///
/// AXI-Stream wrapper for the PE (Processing Element) module providing simplified
/// interface for debugging and testing. This wrapper multiplexes a single AXI-Stream
/// input to feed the iact, wght, psum, and data_stream inputs of the PE, and provides
/// a single AXI-Stream output for psum data.
///
/// Interface Mapping:
/// - Input Mode Selection (axi_input_mode_i):
///   * 2'b00: Route to iact_data_i (input activations)
///   * 2'b01: Route to wght_data_i (weights)
///   * 2'b10: Route to psum_data_i (partial sums)
///   * 2'b11: Route to data_stream_i (configuration parameters)
///
/// - Output: psum_data_o always routed to AXI-Stream output
///
/// AXI-Stream Signals:
/// - Input:  s_axis_tdata, s_axis_tvalid, s_axis_tready
/// - Output: m_axis_tdata, m_axis_tvalid, m_axis_tready
///
/// Parameters:
///    TRANS_BITWIDTH_IACT - Bit width of input activation interface (default: 24)
///    TRANS_BITWIDTH_WGHT - Bit width of weight interface (default: 24)
///    TRANS_BITWIDTH_PSUM - Bit width of partial sum interface (computed from PE params)
///    NUM_GLB_IACT        - Number of global iact buffers (default: 3)
///
///    All other parameters match the PE module defaults

module PE_AXI_wrapper #(
    // PE Configuration Parameters
    parameter IS_TOPLEVEL = 1,
    parameter SERIAL      = 0,
    parameter CREATE_VCD  = 0,

    parameter PE_X = 0,
    parameter PE_Y = 0,

    parameter integer PARALLEL_MACS = 2,

    // Data Width Parameters
    parameter integer DATA_IACT_BITWIDTH     = 8,
    parameter integer DATA_WGHT_BITWIDTH     = 8,
    parameter integer DATA_PSUM_BITWIDTH     = 20,
    parameter integer DATA_IACT_OVERHEAD     = 4,
    parameter integer DATA_WGHT_IGNORE_ZEROS = 4,

    // Memory Parameters
    parameter integer IACT_DATA_ADDR = 16,
    parameter integer IACT_ADDR_ADDR = 9,
    parameter integer WGHT_DATA_ADDR = 96,
    parameter integer WGHT_ADDR_ADDR = 16,
    parameter integer PSUM_ADDR      = 32,

    // Transfer Interface Bitwidths
    parameter integer TRANS_BITWIDTH_IACT = 24,
    parameter integer TRANS_BITWIDTH_WGHT = 24,
    parameter integer NUM_GLB_IACT        = 3,

    // Local parameters - computed from PE parameters
    localparam integer PSUM_WORDS_PER_TRANSFER = (SERIAL ? 1 : PARALLEL_MACS),
    localparam integer TRANS_BITWIDTH_PSUM = DATA_PSUM_BITWIDTH * PSUM_WORDS_PER_TRANSFER,

    // AXI-Stream data width - fixed at 32 bits
    localparam integer AXI_DATA_WIDTH = 32
) (
    // Clock and Reset
    input                            clk_i,
    input                            rst_ni,

    // AXI-Stream Slave Interface (Input)
    input      [AXI_DATA_WIDTH-1:0]  s_axis_tdata,
    input                            s_axis_tvalid,
    output                           s_axis_tready,

    // AXI-Stream Master Interface (Output - Partial Sums)
    output     [AXI_DATA_WIDTH-1:0]  m_axis_tdata,
    output                           m_axis_tvalid,
    input                            m_axis_tready,

    // Control Signals
    input      [1:0]                 axi_input_mode_i,  // 00=iact, 01=wght, 10=psum, 11=data_stream
    input      [$clog2(NUM_GLB_IACT+1)-1:0] iact_select_i,     // Which iact buffer to use (for mode 00)
    input                            compute_i          // Start computation signal
);

    // ============================================================================
    // Internal Signal Declarations
    // ============================================================================

    // Demultiplexed enable signals
    wire iact_enable_mux;
    wire wght_enable_mux;
    wire psum_in_enable_mux;
    wire data_stream_enable_mux;

    // Demultiplexed ready signals
    wire iact_ready_mux;
    wire wght_ready_mux;
    wire psum_in_ready_mux;

    // PE interface signals
    wire [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0] pe_iact_data;
    wire [NUM_GLB_IACT-1:0]                      pe_iact_enable;
    wire [NUM_GLB_IACT-1:0]                      pe_iact_ready;

    wire [TRANS_BITWIDTH_WGHT-1:0]               pe_wght_data;
    wire                                         pe_wght_enable;
    wire                                         pe_wght_ready;

    wire [TRANS_BITWIDTH_PSUM-1:0]               pe_psum_data_i;
    wire                                         pe_psum_enable_i;
    wire                                         pe_psum_ready_o;

    wire [TRANS_BITWIDTH_PSUM-1:0]               pe_psum_data_o;
    wire                                         pe_psum_enable_o;

    wire [11:0]                                  pe_data_stream;
    wire                                         pe_enable_stream;

    // ============================================================================
    // Input Mode Demultiplexing
    // ============================================================================

    // Generate enable signals based on input mode
    assign iact_enable_mux       = (axi_input_mode_i == 2'b00) ? s_axis_tvalid : 1'b0;
    assign wght_enable_mux       = (axi_input_mode_i == 2'b01) ? s_axis_tvalid : 1'b0;
    assign psum_in_enable_mux    = (axi_input_mode_i == 2'b10) ? s_axis_tvalid : 1'b0;
    assign data_stream_enable_mux = (axi_input_mode_i == 2'b11) ? s_axis_tvalid : 1'b0;

    // Multiplex ready signal based on input mode
    assign s_axis_tready = (axi_input_mode_i == 2'b00) ? iact_ready_mux :
                          (axi_input_mode_i == 2'b01) ? wght_ready_mux :
                          (axi_input_mode_i == 2'b10) ? psum_in_ready_mux :
                          1'b1;  // Always ready for data_stream (config params)

    // ============================================================================
    // PE Input Connections with Bitwidth Conversion
    // ============================================================================

    // IACT Interface - replicate single input to all NUM_GLB_IACT inputs
    // Only the selected input (via iact_select_i) will be active
    // Handle bitwidth conversion: pad with zeros if AXI width > IACT width, truncate if smaller
    genvar i;
    generate
        for (i = 0; i < NUM_GLB_IACT; i = i + 1) begin : gen_iact_inputs
            if (TRANS_BITWIDTH_IACT <= AXI_DATA_WIDTH) begin
                // IACT fits in AXI data width - use lower bits
                assign pe_iact_data[TRANS_BITWIDTH_IACT*(i+1)-1 : TRANS_BITWIDTH_IACT*i] =
                       s_axis_tdata[TRANS_BITWIDTH_IACT-1:0];
            end else begin
                // IACT larger than AXI - truncate (take lower AXI_DATA_WIDTH bits)
                assign pe_iact_data[TRANS_BITWIDTH_IACT*(i+1)-1 : TRANS_BITWIDTH_IACT*i] =
                       {{(TRANS_BITWIDTH_IACT-AXI_DATA_WIDTH){1'b0}}, s_axis_tdata};
            end
            assign pe_iact_enable[i] = iact_enable_mux;
        end
    endgenerate

    // Use ready from selected iact input
    assign iact_ready_mux = (iact_select_i > 0 && iact_select_i <= NUM_GLB_IACT) ?
                           pe_iact_ready[iact_select_i-1] : 1'b0;

    // WGHT Interface - handle bitwidth conversion
    generate
        if (TRANS_BITWIDTH_WGHT <= AXI_DATA_WIDTH) begin
            // WGHT fits in AXI data width - use lower bits
            assign pe_wght_data = s_axis_tdata[TRANS_BITWIDTH_WGHT-1:0];
        end else begin
            // WGHT larger than AXI - pad with zeros
            assign pe_wght_data = {{(TRANS_BITWIDTH_WGHT-AXI_DATA_WIDTH){1'b0}}, s_axis_tdata};
        end
    endgenerate
    assign pe_wght_enable = wght_enable_mux;
    assign wght_ready_mux = pe_wght_ready;

    // PSUM Input Interface - handle bitwidth conversion
    generate
        if (TRANS_BITWIDTH_PSUM <= AXI_DATA_WIDTH) begin
            // PSUM fits in AXI data width - use lower bits
            assign pe_psum_data_i = s_axis_tdata[TRANS_BITWIDTH_PSUM-1:0];
        end else begin
            // PSUM larger than AXI - pad with zeros
            assign pe_psum_data_i = {{(TRANS_BITWIDTH_PSUM-AXI_DATA_WIDTH){1'b0}}, s_axis_tdata};
        end
    endgenerate
    assign pe_psum_enable_i = psum_in_enable_mux;
    assign psum_in_ready_mux = pe_psum_ready_o;

    // Data Stream Interface (configuration parameters - always 12 bits)
    assign pe_data_stream   = s_axis_tdata[11:0];
    assign pe_enable_stream = data_stream_enable_mux;

    // ============================================================================
    // PE Output Connections (PSUM to AXI-Stream Master) with Bitwidth Conversion
    // ============================================================================

    // Handle output bitwidth conversion - pad PSUM data to 32 bits if needed
    generate
        if (TRANS_BITWIDTH_PSUM <= AXI_DATA_WIDTH) begin
            // PSUM fits in AXI data width - pad upper bits with zeros
            assign m_axis_tdata = {{(AXI_DATA_WIDTH-TRANS_BITWIDTH_PSUM){1'b0}}, pe_psum_data_o};
        end else begin
            // PSUM larger than AXI - truncate to lower AXI_DATA_WIDTH bits
            assign m_axis_tdata = pe_psum_data_o[AXI_DATA_WIDTH-1:0];
        end
    endgenerate

    assign m_axis_tvalid = pe_psum_enable_o;
    // PE's psum_ready_i comes from AXI master interface

    // ============================================================================
    // PE Instantiation
    // ============================================================================

    PE #(
        .IS_TOPLEVEL(IS_TOPLEVEL),
        .SERIAL(SERIAL),
        .CREATE_VCD(CREATE_VCD),
        .PE_X(PE_X),
        .PE_Y(PE_Y),
        .PARALLEL_MACS(PARALLEL_MACS),
        .DATA_IACT_BITWIDTH(DATA_IACT_BITWIDTH),
        .DATA_WGHT_BITWIDTH(DATA_WGHT_BITWIDTH),
        .DATA_PSUM_BITWIDTH(DATA_PSUM_BITWIDTH),
        .DATA_IACT_OVERHEAD(DATA_IACT_OVERHEAD),
        .DATA_WGHT_IGNORE_ZEROS(DATA_WGHT_IGNORE_ZEROS),
        .IACT_DATA_ADDR(IACT_DATA_ADDR),
        .IACT_ADDR_ADDR(IACT_ADDR_ADDR),
        .WGHT_DATA_ADDR(WGHT_DATA_ADDR),
        .WGHT_ADDR_ADDR(WGHT_ADDR_ADDR),
        .PSUM_ADDR(PSUM_ADDR),
        .TRANS_BITWIDTH_IACT(TRANS_BITWIDTH_IACT),
        .TRANS_BITWIDTH_WGHT(TRANS_BITWIDTH_WGHT),
        .NUM_GLB_IACT(NUM_GLB_IACT)
    ) pe_inst (
        .clk_i(clk_i),
        .rst_ni(rst_ni),

        // IACT Interface
        .iact_select_i(iact_select_i),
        .iact_data_i(pe_iact_data),
        .iact_enable_i(pe_iact_enable),
        .iact_ready_o(pe_iact_ready),

        // WGHT Interface
        .wght_data_i(pe_wght_data),
        .wght_enable_i(pe_wght_enable),
        .wght_ready_o(pe_wght_ready),

        // PSUM Interface
        .psum_data_i(pe_psum_data_i),
        .psum_enable_i(pe_psum_enable_i),
        .psum_ready_o(pe_psum_ready_o),
        .psum_data_o(pe_psum_data_o),
        .psum_enable_o(pe_psum_enable_o),
        .psum_ready_i(m_axis_tready),

        // Control
        .compute_i(compute_i),
        .enable_stream_i(pe_enable_stream),
        .data_stream_i(pe_data_stream)
    );

endmodule
