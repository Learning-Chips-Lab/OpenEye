#!/usr/bin/env python3
import yaml
from pathlib import Path

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
YAML_FILE = "regmap.yaml"
VERILOG_PARAMS_OUT = "regmap_params.vh"
PYTHON_OUT = "regmap_pack.py"
DMA_STORAGE_OUT = "../../hdl/dma_storage.v"

# ------------------------------------------------------------
# Load YAML
# ------------------------------------------------------------
with open(YAML_FILE, "r") as f:
    config = yaml.safe_load(f)

dma_bitwidth = int(config["dma_bitwidth"])
registers = config["registers"]

# ------------------------------------------------------------
# Calculate transmission splitting & positions
# ------------------------------------------------------------
transmissions = []
current_trans = []
bitpos = 0
trans_id = 0

for reg in registers:
    width = int(reg["width"])
    if bitpos + width > dma_bitwidth:  # start new transmission
        transmissions.append(current_trans)
        current_trans = []
        bitpos = 0
        trans_id += 1
    reg_entry = {
        "name": reg["name"],
        "width": width,
        "trans": trans_id,
        "pos": bitpos
    }
    current_trans.append(reg_entry)
    bitpos += width

if current_trans:
    transmissions.append(current_trans)

num_transmissions = len(transmissions)

# ------------------------------------------------------------
# 1) Write Verilog parameter file
# ------------------------------------------------------------
with open(VERILOG_PARAMS_OUT, "w") as vf:
    vf.write("// Auto-generated from {}\n\n".format(YAML_FILE))
    vf.write(f"parameter DMA_BITWIDTH = {dma_bitwidth};\n")
    vf.write(f"parameter TRANSMISSIONS = {num_transmissions};\n\n")

    for tid, regs in enumerate(transmissions):
        vf.write(f"// Transmission {tid} Offsets\n")
        for i, reg in enumerate(regs):
            pname = f"PARAMETER_POS_{tid}_{i}"
            if i == 0:
                vf.write(f"parameter {pname} = 0;\n")
            else:
                prev_name = f"PARAMETER_POS_{tid}_{i-1}"
                prev_width = regs[i-1]['width']
                vf.write(f"parameter {pname} = {prev_name} + {prev_width};\n")
        vf.write("\n")

print(f"[OK] Verilog parameters written to {VERILOG_PARAMS_OUT}.")

# ------------------------------------------------------------
# 2) Write Python pack/unpack
# ------------------------------------------------------------
with open(PYTHON_OUT, "w") as pf:
    pf.write("# Auto-generated from {}\n".format(YAML_FILE))
    pf.write(f"DMA_BITWIDTH = {dma_bitwidth}\n")
    pf.write(f"TRANSMISSIONS = {num_transmissions}\n\n")

    pf.write("REGISTERS = [\n")
    for regs in transmissions:
        for reg in regs:
            pf.write(f"    {{'name': '{reg['name']}', 'width': {reg['width']}, 'trans': {reg['trans']}, 'pos': {reg['pos']}}},\n")
    pf.write("]\n\n")

    pf.write("def pack_registers(values):\n")
    pf.write("    words = [0] * TRANSMISSIONS\n")
    pf.write("    for reg in REGISTERS:\n")
    pf.write("        val = values[reg['name']]\n")
    pf.write("        mask = (1 << reg['width']) - 1\n")
    pf.write("        words[reg['trans']] |= (val & mask) << reg['pos']\n")
    pf.write("    return words\n\n")

    pf.write("def unpack_registers(words):\n")
    pf.write("    values = {}\n")
    pf.write("    for reg in REGISTERS:\n")
    pf.write("        mask = (1 << reg['width']) - 1\n")
    pf.write("        values[reg['name']] = (words[reg['trans']] >> reg['pos']) & mask\n")
    pf.write("    return values\n")

print(f"[OK] Python pack/unpack written to {PYTHON_OUT}.")

# ------------------------------------------------------------
# 3) Write dma_storage.v
# ------------------------------------------------------------
with open(DMA_STORAGE_OUT, "w") as dv:
    dv.write("// Auto-generated from {}\n\n".format(YAML_FILE))
    dv.write("`include \"regmap_params.vh\"\n\n")
    dv.write("module dma_storage (\n")
    dv.write("    input  wire                  clk_i,\n")
    dv.write("    input  wire                  rst_ni,\n")
    dv.write("    input  wire                  write_en,\n")
    dv.write("    input  wire [$clog2(TRANSMISSIONS)-1:0] write_addr,\n")
    dv.write(f"    input  wire [DMA_BITWIDTH-1:0] dma_data_i,\n")
    for regs in transmissions:
        for reg in regs:
            dv.write(f"    output reg [{reg['width']-1}:0] {reg['name']},\n")
    dv.seek(dv.tell()-2)  # Remove last comma
    dv.write("\n);\n\n")

    dv.write("always @(posedge clk_i, negedge rst_ni) begin\n")
    dv.write("    if (!rst_ni) begin  ///Reset\n")
    for regs in transmissions:
        for reg in regs:
            dv.write(f"        {reg['name']} <= {reg['width']}'d0;\n")
    dv.write("    end else if (write_en) begin\n")
    dv.write("        case (write_addr)\n")
    for tid, regs in enumerate(transmissions):
        dv.write(f"            {tid}: begin\n")
        for i, reg in enumerate(regs):
            pname = f"PARAMETER_POS_{tid}_{i}"
            dv.write(f"                {reg['name']} <= dma_data_i[{pname} + {reg['width'] - 1} : {pname}];\n")
        dv.write("            end\n")
    dv.write("        endcase\n")
    dv.write("    end\n")
    dv.write("end\n\nendmodule\n")

print(f"[OK] dma_storage module written to {DMA_STORAGE_OUT}.")