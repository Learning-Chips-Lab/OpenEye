// Liveness probe for the gate-level netlist: after reset the DMA master
// handshake (ready_dma_o) must come up, like in the RTL simulation.
`timescale 1ns/100ps

module gatesim_probe_tb;

    reg         clk_i;
    reg         rst_ni;
    reg  [63:0] data_dma_i;
    reg         enable_dma_i;
    reg         ready_dma_i;
    wire        ready_dma_o;
    wire [63:0] data_dma_o;
    wire        enable_dma_o;
    wire        last_data_o;

    OpenEye_Asic_gatesim_top u_dut (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .data_dma_i     (data_dma_i),
        .enable_dma_i   (enable_dma_i),
        .ready_dma_i    (ready_dma_i),
        .ready_dma_o    (ready_dma_o),
        .data_dma_o     (data_dma_o),
        .enable_dma_o   (enable_dma_o),
        .last_data_o    (last_data_o)
    );

    initial clk_i = 0;
    always #5 clk_i = ~clk_i;

    initial begin
        rst_ni = 0;
        data_dma_i = 0;
        enable_dma_i = 0;
        ready_dma_i = 1;
        #200;
        rst_ni = 1;
        $display("t=%0t reset released", $time);
        #3000;
        $display("t=%0t ready_dma_o=%b enable_dma_o=%b last_data_o=%b data_dma_o=%h",
                 $time, ready_dma_o, enable_dma_o, last_data_o, data_dma_o);
        if (ready_dma_o === 1'b1)
            $display("PROBE: DUT is ALIVE (ready_dma_o asserted)");
        else
            $display("PROBE: DUT appears DEAD (ready_dma_o=%b)", ready_dma_o);
        $finish;
    end

endmodule
