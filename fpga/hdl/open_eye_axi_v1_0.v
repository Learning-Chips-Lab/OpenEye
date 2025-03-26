`timescale 1ns / 1ps

module open_eye_axi_v1_0 #
(
    // Parameters of Axi Slave Bus Interface dma_i
    parameter integer C_dma_i_TDATA_WIDTH	= 64,

    // Parameters of Axi Master Bus Interface dma_o
    parameter integer C_dma_o_TDATA_WIDTH	= 64,
    parameter integer C_dma_o_START_COUNT	= 32
)
(
    // Users to add ports here
    input wire clk,
    input wire rstn,
    // User ports ends
    // Do not modify the ports beyond this line

    // Ports of Axi Slave Bus Interface dma_i
    input wire  dma_i_aclk,
    input wire  dma_i_aresetn,
    output wire  dma_i_tready,
    input wire [C_dma_i_TDATA_WIDTH-1 : 0] dma_i_tdata,
    input wire [(C_dma_i_TDATA_WIDTH/8)-1 : 0] dma_i_tstrb,
    input wire  dma_i_tlast,
    input wire  dma_i_tvalid,

    // Ports of Axi Master Bus Interface dma_o
    input wire  dma_o_aclk,
    input wire  dma_o_aresetn,
    output wire  dma_o_tvalid,
    output wire [C_dma_o_TDATA_WIDTH-1 : 0] dma_o_tdata,
    output wire [(C_dma_o_TDATA_WIDTH/8)-1 : 0] dma_o_tstrb,
    output wire  dma_o_tlast,
    input wire  dma_o_tready
);

// Add user logic here

OpenEye_FPGA #(
) open_eye_wrapper_inst (
    .clk_i          (clk),
    .rst_ni         (rstn),
    
    .ready_dma_o    (dma_i_tready),
    .data_dma_i     (dma_i_tdata),
    .enable_dma_i   (dma_i_tvalid),
    
    .ready_dma_i    (dma_o_tready),
    .data_dma_o     (dma_o_tdata),
    .enable_dma_o   (dma_o_tvalid),
    .last_data_o    (dma_o_tlast)
);

// User logic ends

endmodule
