
`timescale 1 ns / 1 ps

	module open_eye_mt_v1_0 #
	(

		// Users to add parameters here
		parameter integer CLUSTER_ROWS	= 4,
		parameter integer NUM_GLB_IACT	= 3,
		parameter integer NUM_GLB_WGHT	= 3,
		parameter integer NUM_GLB_PSUM	= 3,

		// User parameters ends
		// Do not modify the parameters beyond this line


		// Parameters of Axi Slave Bus Interface cfg_reg
		parameter integer C_cfg_reg_DATA_WIDTH	= 32,
		parameter integer C_cfg_reg_ADDR_WIDTH	= 6,

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


		// Ports of Axi Slave Bus Interface cfg_reg
		input wire  cfg_reg_aclk,
		input wire  cfg_reg_aresetn,
		input wire [C_cfg_reg_ADDR_WIDTH-1 : 0] cfg_reg_awaddr,
		input wire [2 : 0] cfg_reg_awprot,
		input wire  cfg_reg_awvalid,
		output wire  cfg_reg_awready,
		input wire [C_cfg_reg_DATA_WIDTH-1 : 0] cfg_reg_wdata,
		input wire [(C_cfg_reg_DATA_WIDTH/8)-1 : 0] cfg_reg_wstrb,
		input wire  cfg_reg_wvalid,
		output wire  cfg_reg_wready,
		output wire [1 : 0] cfg_reg_bresp,
		output wire  cfg_reg_bvalid,
		input wire  cfg_reg_bready,
		input wire [C_cfg_reg_ADDR_WIDTH-1 : 0] cfg_reg_araddr,
		input wire [2 : 0] cfg_reg_arprot,
		input wire  cfg_reg_arvalid,
		output wire  cfg_reg_arready,
		output wire [C_cfg_reg_DATA_WIDTH-1 : 0] cfg_reg_rdata,
		output wire [1 : 0] cfg_reg_rresp,
		output wire  cfg_reg_rvalid,
		input wire  cfg_reg_rready,

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
// Instantiation of Axi Bus Interface cfg_reg
	open_eye_mt_v1_0_cfg_reg # ( 
		.C_S_AXI_DATA_WIDTH(C_cfg_reg_DATA_WIDTH),
		.C_S_AXI_ADDR_WIDTH(C_cfg_reg_ADDR_WIDTH)
	) open_eye_mt_v1_0_cfg_reg_inst (
		.S_AXI_ACLK(cfg_reg_aclk),
		.S_AXI_ARESETN(cfg_reg_aresetn),
		.S_AXI_AWADDR(cfg_reg_awaddr),
		.S_AXI_AWPROT(cfg_reg_awprot),
		.S_AXI_AWVALID(cfg_reg_awvalid),
		.S_AXI_AWREADY(cfg_reg_awready),
		.S_AXI_WDATA(cfg_reg_wdata),
		.S_AXI_WSTRB(cfg_reg_wstrb),
		.S_AXI_WVALID(cfg_reg_wvalid),
		.S_AXI_WREADY(cfg_reg_wready),
		.S_AXI_BRESP(cfg_reg_bresp),
		.S_AXI_BVALID(cfg_reg_bvalid),
		.S_AXI_BREADY(cfg_reg_bready),
		.S_AXI_ARADDR(cfg_reg_araddr),
		.S_AXI_ARPROT(cfg_reg_arprot),
		.S_AXI_ARVALID(cfg_reg_arvalid),
		.S_AXI_ARREADY(cfg_reg_arready),
		.S_AXI_RDATA(cfg_reg_rdata),
		.S_AXI_RRESP(cfg_reg_rresp),
		.S_AXI_RVALID(cfg_reg_rvalid),
		.S_AXI_RREADY(cfg_reg_rready)
	);

	// Add user logic here

    OpenEye_FPGA #(
	.CLUSTER_ROWS(CLUSTER_ROWS),
	.NUM_GLB_IACT(NUM_GLB_IACT),
	.NUM_GLB_WGHT(NUM_GLB_WGHT),
	.NUM_GLB_PSUM(NUM_GLB_PSUM)
	
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