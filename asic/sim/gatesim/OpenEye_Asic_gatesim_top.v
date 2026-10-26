// Thin top for gate-level simulation of the synthesized OpenEye_Asic netlist.
// Maps the pad-level ports back to the OpenEye_FPGA signal names so the
// existing cocotb testbench (test/cocotb_fpga/OpenEye_FPGA_tb.py) can be
// reused unchanged with TOPLEVEL=OpenEye_Asic_gatesim_top.
//
// The /*verilator public_on*/ marks are REQUIRED together with --flatten
// (see run_verilator.sh / QWEN.md). Verilator 5.052 generates a VPI var
// table only when at least one public variable exists (build p2_10: no
// marks -> no table -> cocotb init fatal "Can not find root handle").
// The marks alone are NOT sufficient: without --flatten the wrapper
// survives as its own class while $root holds a mirror copy of these 9
// ports that is re-synced on EVERY clock edge (ico_sequent), so cocotb
// writes to this scope are clobbered (symptom: outputs read 0x0). With
// --flatten everything is inlined into the single $root class and this
// module's scope view binds its var table directly to $root storage, so
// the cocotb root handle (COCOTB_TOPLEVEL = this module name) exposes
// the LIVE port variables. Verified end-to-end (comb write without a
// clock edge + FF path via posedge/NBA) with the P15b probe
// (asic/sim/gatesim/vpi_probes/, 2026-10-25).
//
// FILE ORDER: this wrapper is the LAST Verilog file on the Verilator
// command line (see run_verilator.sh). Verilator 5.052 auto-marks a
// module public iff it is parsed after the --top-module definition, and
// public inlined cells materialize per-instance VPI scopes — with the
// ~2.1M netlist instances that is the 3.19M-scope codegen explosion
// (build 6, vpi_probes/p18/).

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

    /*verilator public_on*/ input          clk_i;
    /*verilator public_on*/ input          rst_ni;
    /*verilator public_on*/ input  [63:0]  data_dma_i;
    /*verilator public_on*/ input          enable_dma_i;
    /*verilator public_on*/ input          ready_dma_i;
    /*verilator public_on*/ output         ready_dma_o;
    /*verilator public_on*/ output [63:0]  data_dma_o;
    /*verilator public_on*/ output         enable_dma_o;
    /*verilator public_on*/ output         last_data_o;

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
