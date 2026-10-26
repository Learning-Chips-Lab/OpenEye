#!/usr/bin/env python3
"""P15d: PDK-faithful specify probe (run in-container).

Extracts the REAL PDK cell modules (sg13g2_dfrbpq_1, sg13g2_and2_1,
sg13g2_IOPadIn) verbatim and builds two variants:
  raw: PDK modules as-is (specify blocks, specify-only-driven delayed_*)
  tf:  same modules through asic/scripts/pdk_specify_strip.py
       (specify stripped, assign delayed_X = X aliases)
plus the same 9-port /*verilator public_on*/ wrapper (plus one ffq0_o
tap) + FF/AND chains as P15b, and the p15d_tb cocotb tb.

File order matches the fixed real build: primitives file FIRST, then the
cells+top file (cells defined before the top module -> non-public -> 2
scopes expected for BOTH variants).

Expectations (the verdict):
  raw -> 2 scopes, comb PASS, FF path FAIL (delayed_CLK/delayed_D driven
         only by the ignored specify block -> undriven -> folds to 0 in
         2-state -> the FF never clocks)
  tf  -> 2 scopes, comb PASS, FF path PASS (assign delayed_X = X aliases
         restore the function; all PDK specify delays are 0.0)

Usage: gen_p15d.py [NF=10] [NG=10]
"""
import re
import subprocess
import sys

STDC = "/foss/pdks/ihp-sg13g2/libs.ref/sg13g2_stdcell/verilog/sg13g2_stdcell.v"
IOF = "/foss/pdks/ihp-sg13g2/libs.ref/sg13g2_io/verilog/sg13g2_io.v"
STRIP = "/foss/designs/OpenEye/asic/scripts/pdk_specify_strip.py"

NF = int(sys.argv[1]) if len(sys.argv) > 1 else 10  # dfrbpq FF chain
NG = int(sys.argv[2]) if len(sys.argv) > 2 else 10  # and2 chain


def extract(text, name):
    m = re.search(r"(?ms)^module\s+%s\b.*?^endmodule" % re.escape(name), text)
    if not m:
        sys.exit("module %s not found" % name)
    return m.group(0) + "\n\n"


def name_prims(text):
    """same rename as run_verilator.sh perl step (unnamed ihp_* instances)"""
    n = 0

    def repl(mm):
        nonlocal n
        n += 1
        return "%s%s u_prim_%d%s (" % (mm.group(1), mm.group(2), n, mm.group(3))

    return re.sub(r"(?m)^([ \t]*ihp_)([a-z0-9_]+)([ \t]*)\(", repl, text)


def write_design(fname, cells):
    t = []
    t.append("module p15dtop (\n")
    t.append("  /*verilator public_on*/ input clk_i,\n")
    t.append("  /*verilator public_on*/ input rst_ni,\n")
    t.append("  /*verilator public_on*/ input [63:0] data_dma_i,\n")
    t.append("  /*verilator public_on*/ input enable_dma_i,\n")
    t.append("  /*verilator public_on*/ input ready_dma_i,\n")
    t.append("  /*verilator public_on*/ output ready_dma_o,\n")
    t.append("  /*verilator public_on*/ output [63:0] data_dma_o,\n")
    t.append("  /*verilator public_on*/ output enable_dma_o,\n")
    t.append("  /*verilator public_on*/ output last_data_o,\n")
    t.append("  /*verilator public_on*/ output ffq0_o\n")
    t.append(");\n")
    # FF chain through real dfrbpq cells: Q[i] -> D[i+1], D[0] = data_dma_i[0]
    t.append("  wire ffq0;\n")
    t.append("  sg13g2_dfrbpq_1 u_ff0000 (.Q(ffq0), .CLK(clk_i), "
             ".D(data_dma_i[0]), .RESET_B(rst_ni));\n")
    for i in range(1, NF):
        t.append("  wire ffq%d;\n" % i)
        t.append("  sg13g2_dfrbpq_1 u_ff%04d (.Q(ffq%d), .CLK(clk_i), "
                 ".D(ffq%d), .RESET_B(rst_ni));\n" % (i, i, i - 1))
    # and2 chain through real and2 cells
    t.append("  wire gg0;\n")
    t.append("  sg13g2_and2_1 u_g0000 (.X(gg0), .A(data_dma_i[0]), "
             ".B(enable_dma_i));\n")
    for i in range(1, NG):
        t.append("  wire gg%d;\n" % i)
        t.append("  sg13g2_and2_1 u_g%04d (.X(gg%d), .A(gg%d), "
                 ".B(data_dma_i[%d]));\n" % (i, i, i - 1, i % 10))
    # one real IOPadIn (specify block, pure passthrough function)
    t.append("  wire padin_out;\n")
    t.append("  sg13g2_IOPadIn u_pad (.pad(data_dma_i[1]), .p2c(padin_out), "
             ".vdd(), .vss(), .iovdd(), .iovss());\n")
    t.append("  assign ready_dma_o  = gg%d;\n" % (NG - 1))
    t.append("  assign data_dma_o   = {data_dma_i[63:1], ffq%d};\n" % (NF - 1))
    t.append("  assign enable_dma_o = ready_dma_i & rst_ni;\n")
    t.append("  assign last_data_o  = padin_out;\n")
    t.append("  assign ffq0_o       = ffq0;\n")
    t.append("endmodule\n")
    with open(fname, "w") as f:
        f.write(cells + "".join(t))


stdc = open(STDC).read()
io = open(IOF).read()

raw_cells = name_prims(
    extract(stdc, "sg13g2_dfrbpq_1")
    + extract(stdc, "sg13g2_and2_1")
    + extract(io, "sg13g2_IOPadIn")
)
with open("p15d_cells_raw.v", "w") as f:
    f.write(raw_cells)
subprocess.run([sys.executable, STRIP,
                "p15d_cells_raw.v", "p15d_cells_tf.v"], check=True)

write_design("p15d_raw.v", raw_cells)
write_design("p15d_tf.v", open("p15d_cells_tf.v").read())
print("wrote p15d_raw.v / p15d_tf.v (NF=%d NG=%d)" % (NF, NG))
