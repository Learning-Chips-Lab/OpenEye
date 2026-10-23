// Thin top for gate-level simulation of the synthesized OpenEye_Asic netlist.
// Maps the pad-level ports back to the OpenEye_FPGA signal names so the
// existing cocotb testbench (test/cocotb_fpga/OpenEye_FPGA_tb.py) can be
// reused unchanged with TOPLEVEL=OpenEye_Asic_gatesim_top.

module OpenEye_Asic_gatesim_top (
    clk_i,
    rst_ni,
    data_dma_i,
    enable_dma_i,
    ready_dma_i,
    ready_dma_o,
    data_dma_o,
    enable_dma_o,
    last_data_o
);

    input          clk_i;
    input          rst_ni;
    input  [63:0]  data_dma_i;
    input          enable_dma_i;
    input          ready_dma_i;
    output         ready_dma_o;
    output [63:0]  data_dma_o;
    output         enable_dma_o;
    output         last_data_o;

    OpenEye_Asic u_dut (
        .clk_pad           (clk_i),
        .rst_n_pad         (rst_ni),
        .data_dma_i_pad    (data_dma_i),
        .enable_dma_i_pad  (enable_dma_i),
        .ready_dma_i_pad   (ready_dma_i),
        .ready_dma_o_pad   (ready_dma_o),
        .data_dma_o_pad    (data_dma_o),
        .enable_dma_o_pad  (enable_dma_o),
        .last_data_o_pad   (last_data_o)
    );

endmodule
