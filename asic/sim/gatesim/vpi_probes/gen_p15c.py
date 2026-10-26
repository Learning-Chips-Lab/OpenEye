#!/usr/bin/env python3
"""P15c: P15b + ONE `initial #0` statement (forces VM_TIMING=1, exactly like
the real build where the 33 RM SRAM instances carry an `initial #0` sim-only
sanity check). Closes the last structural corner:
  --flatten + public_on marks + VM_TIMING=1 + cocotb-driven clock.
Expected: same single-class structure as P15b (mirror copies synced ONLY at
firstIteration), comb write propagation without a clock edge, FF path via
posedge/NBA.
"""
import sys

M = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
out = sys.argv[2] if len(sys.argv) > 2 else "big15c.v"

with open(out, "w") as f:
    f.write("// P15c probe: M=%d, one initial #0 (VM_TIMING=1)\n" % M)
    f.write(
        "module big_leaf (input i, input j, output o);\n"
        "  wire k;\n"
        "  and (k, i, j);\n"
        "  assign o = k;\n"
        "endmodule\n"
        "module bignet (input clk, input rst_ni, input [9:0] di, output [9:0] dout);\n"
        "  // the ONLY delay construct in the design (like the RM SRAM\n"
        "  // `initial #0` sanity checks in the real PDK models)\n"
        "  initial #0;\n"
        "  reg [9:0] q;\n"
        "  always @(posedge clk or negedge rst_ni) "
        "if (!rst_ni) q <= 0; else q <= di;\n"
    )
    for i in range(M):
        f.write("  wire g%d;\n" % i)
        if i == 0:
            f.write("  big_leaf u%04d (.i(di[0]), .j(clk), .o(g%d));\n" % (i, i))
        else:
            f.write("  big_leaf u%04d (.i(g%d), .j(clk), .o(g%d));\n" % (i, i - 1, i))
    f.write("  assign dout[0] = g%d & q[0];\n" % (M - 1))
    f.write("  assign dout[9:1] = di[9:1];\n")
    f.write("endmodule\n")
    f.write(
        "\n"
        "module bigtop (\n"
        "  /*verilator public_on*/ input clk_i,\n"
        "  /*verilator public_on*/ input rst_ni,\n"
        "  /*verilator public_on*/ input [63:0] data_dma_i,\n"
        "  /*verilator public_on*/ input enable_dma_i,\n"
        "  /*verilator public_on*/ input ready_dma_i,\n"
        "  /*verilator public_on*/ output ready_dma_o,\n"
        "  /*verilator public_on*/ output [63:0] data_dma_o,\n"
        "  /*verilator public_on*/ output enable_dma_o,\n"
        "  /*verilator public_on*/ output last_data_o\n"
        ");\n"
        "  bignet u_net (.clk(clk_i), .rst_ni(rst_ni), .di(data_dma_i[9:0]), .dout(data_dma_o[9:0]));\n"
        "  wire rdy;\n"
        "  big_leaf u_rdy (.i(data_dma_i[0]), .j(enable_dma_i), .o(rdy));\n"
        "  assign ready_dma_o  = rdy;\n"
        "  assign data_dma_o[63:10] = data_dma_i[63:10];\n"
        "  assign enable_dma_o = ready_dma_i & rst_ni;\n"
        "  assign last_data_o  = data_dma_o[0];\n"
        "endmodule\n"
    )
print("wrote", out, "M=", M)
