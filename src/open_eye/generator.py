#!/usr/bin/env python3
"""DMA Register Map Generator for OpenEye Neural Network Accelerator.

This module generates the DMA register map infrastructure from a YAML specification.
It produces three critical files that enable efficient parameter transfer to hardware:

1. **regmap_params.vh** - Verilog parameter definitions for bit positions
2. **regmap_pack.py** - Python pack/unpack functions for software
3. **dma_storage.v** - Pipelined Verilog module that receives and unpacks DMA transmissions

Optimized for ultra-low LUT footprint using a hardware shift-pipeline architecture.
"""
import yaml
from pathlib import Path
import sys
import os
import math

# Context for evaluating expressions in YAML (e.g., "ceil(log2(CLUSTER_ROWS))")
context = {
    "dma_bitwidth": 64,           # DMA bus width in bits
    "iact_buffer_size": 4096,     # Input activation buffer size
    "num_pes": 14,                # Number of processing elements

    # Math functions for width calculations
    "ceil": math.ceil,
    "log2": math.log2,
    "int": int,
    "max": max
}


def eval_width(width, ctx):
    """Evaluate register width, supporting both literal integers and expressions."""
    if isinstance(width, int):
        return max(width, 1)
    elif isinstance(width, str):
        return max(int(eval(width, {"__builtins__": {}}, ctx)), 1)
    else:
        raise TypeError("Invalid width type")


def create_regmap_params_vh_file(regmap_yaml_path, output_vh_path=None, output_v_path=None, output_py_path=None):
    """Generate DMA register map files from YAML specification."""

    if output_vh_path is None:
        output_vh_path = os.path.join(os.path.dirname(regmap_yaml_path), "include")
    Path(output_vh_path).mkdir(parents=True, exist_ok=True)
    if output_v_path is None:
        output_v_path = os.path.join(os.path.dirname(regmap_yaml_path))
    Path(output_v_path).mkdir(parents=True, exist_ok=True)
    if output_py_path is None:
        output_py_path = Path(__file__).resolve().parent
    Path(output_py_path).mkdir(parents=True, exist_ok=True)

    YAML_FILE = os.path.join(regmap_yaml_path, "regmap.yaml")
    VERILOG_PARAMS_OUT = os.path.join(output_vh_path, "regmap_params.vh")
    PYTHON_OUT = os.path.join(output_py_path, "regmap_pack.py")
    DMA_STORAGE_OUT = os.path.join(output_v_path, "dma_storage.v")

    # Load YAML
    with open(YAML_FILE, "r") as f:
        config = yaml.safe_load(f)

    dma_bitwidth = int(config["dma_bitwidth"])
    registers = config["registers"]
    env_ctx = dict(os.environ)
    comb_ctx = context | env_ctx
    for reg in registers:
        reg["width"] = eval_width(reg["width"], comb_ctx)

    transmissions = []      
    current_trans = []      
    bitpos = 0              
    trans_id = 0            

    for reg in registers:
        width = int(reg["width"])

        if bitpos + width > dma_bitwidth:
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

    # 1) Write Verilog parameter file (regmap_params.vh)
    with open(VERILOG_PARAMS_OUT, "w+") as vf:
        vf.write("`ifndef REGMAP_PARAMS_VH\n`define REGMAP_PARAMS_VH\n\n")
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
        vf.write("`endif // REGMAP_PARAMS_VH\n")

    print(f"[OK] Verilog parameters written to {VERILOG_PARAMS_OUT}.")

    # 2) Write Python pack/unpack module (regmap_pack.py)
    with open(PYTHON_OUT, "w+") as pf:
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
        pf.write("        val = values.get(reg['name'], 0)\n")
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

    # 3) Write ULTRA-LOW-LUT dma_storage.v module (Pipelined Shift Structure)
    with open(DMA_STORAGE_OUT, "w+") as dv:
        dv.write("// Auto-generated from {}\n//\n".format(YAML_FILE))
        dv.write("// DMA Storage Module - Pipelined Shift Register Architecture\n//\n")
        dv.write("// Optimized for minimum LUT footprint. Eliminates the address decoder completely.\n")
        dv.write("// Data is shifted sequentially into a configuration pipe register chain.\n\n")
        dv.write("`include \"regmap_params.vh\"\n\n")

        # Module header
        dv.write("module dma_storage (\n")
        dv.write("    input  wire                  clk_i,\n")
        dv.write("    input  wire                  rst_ni,           // Active-low reset\n")
        dv.write("    input  wire                  write_en,         // Write enable from DMA\n")
        dv.write(f"    input  wire [DMA_BITWIDTH-1:0] dma_data_i,  // Packed 64-bit data\n")

        # Output ports changed to WIRE!
        for regs in transmissions:
            for reg in regs:
                dv.write(f"    output wire [{reg['width']-1}:0] {reg['name']},\n")
        dv.seek(dv.tell()-2)  # Last comma removal
        dv.write("\n);\n\n")

        # Internal Pipeline Register Chain
        dv.write("  // Flat pipeline chain register to capture all transmissions\n")
        dv.write("  reg [(DMA_BITWIDTH * TRANSMISSIONS)-1:0] pipe_chain_reg;\n\n")

        # Pipelined Shift Block
        dv.write("  // Sequential Shift Register Logic\n")
        dv.write("  always @(posedge clk_i, negedge rst_ni) begin\n")
        dv.write("      if (!rst_ni) begin\n")
        dv.write("          pipe_chain_reg <= {(DMA_BITWIDTH * TRANSMISSIONS){1'b0}};\n")
        dv.write("      end else if (write_en) begin\n")
        dv.write("          // Shift old data left, shift new data into the LSB stage\n")
        dv.write("          pipe_chain_reg <= {pipe_chain_reg[(DMA_BITWIDTH * (TRANSMISSIONS-1))-1:0], dma_data_i};\n")
        dv.write("      end\n")
        dv.write("  end\n\n")

        # Continuous Assignments for Zero-LUT Bit Slicing
        dv.write("  // Continuous Assignments: Fixed wire routing from respective pipeline stages\n")
        dv.write("  // Since Transmission 0 enters first, it gets shifted up to the highest stage.\n")
        for tid, regs in enumerate(transmissions):
            dv.write(f"\n  // --- Extraction for Transmission {tid} ---\n")
            for i, reg in enumerate(regs):
                pname = f"PARAMETER_POS_{tid}_{i}"
                # Calculation of the exact bit offset within the global shift register array
                dv.write(f"  assign {reg['name']} = pipe_chain_reg[((TRANSMISSIONS - 1 - {tid}) * DMA_BITWIDTH) + {pname} + {reg['width']-1} : ((TRANSMISSIONS - 1 - {tid}) * DMA_BITWIDTH) + {pname}];\n")

        dv.write("\nendmodule\n")

    print(f"[OK] Pipelined dma_storage module written to {DMA_STORAGE_OUT}.")


if __name__ == "__main__":
    try:
        path = sys.argv[1]
    except:
        path = os.getcwd()

    try:
        output_pathvh = sys.argv[2]
    except:
        output_pathvh = path

    try:
        output_pathv = sys.argv[3]
    except:
        output_pathv = path

    try:
        output_pathpy = sys.argv[4]
    except:
        output_pathpy = None

    create_regmap_params_vh_file(path, output_pathvh, output_pathv, output_pathpy)