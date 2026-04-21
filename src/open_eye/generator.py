#!/usr/bin/env python3
"""DMA Register Map Generator for OpenEye Neural Network Accelerator.

This module generates the DMA register map infrastructure from a YAML specification.
It produces three critical files that enable efficient parameter transfer to hardware:

1. **regmap_params.vh** - Verilog parameter definitions for bit positions
2. **regmap_pack.py** - Python pack/unpack functions for software
3. **dma_storage.v** - Verilog module that receives and unpacks DMA transmissions

Background
----------
Each neural network layer requires ~40 configuration parameters (strides, kernel sizes,
buffer addresses, skip flags, etc.). Sending these individually over DMA would require
40+ bus transactions per layer, with significant overhead.

Solution
--------
This generator implements a bit-packing protocol that groups parameters into 64-bit
DMA transmissions. Parameters are packed sequentially into as few 64-bit words as
possible, typically resulting in 4 transmissions per layer instead of 40+.

The generated infrastructure handles:
- Automatic bit position calculation
- Transmission boundary detection (when 64 bits are full)
- Parameter extraction on the Verilog side
- Pack/unpack utilities on the Python side

Example
-------
Given this YAML specification::

    dma_bitwidth: 64
    registers:
      - {name: wght_cycles_reg, width: 8}
      - {name: stride_x_reg, width: 3}
      - {name: stride_y_reg, width: 3}

The generator produces:
- Verilog parameters defining bit positions (0, 8, 11, ...)
- Python functions to pack values into 64-bit words
- Verilog module to extract values from DMA data

See Also
--------
- regmap.yaml : YAML specification for register layout
- regmap_params.vh : Generated Verilog parameters
- regmap_pack.py : Generated Python pack/unpack functions
- dma_storage.v : Generated Verilog unpacking module
"""
import yaml
from pathlib import Path
import open_eye.generic_test_utils as gtu
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
    """Evaluate register width, supporting both literal integers and expressions.

    Some register widths depend on hardware parameters and are specified as
    expressions in the YAML file. For example:
        width: ceil(log2(int(CLUSTER_ROWS)+1))

    This function evaluates such expressions using the provided context.

    Parameters
    ----------
    width : int or str
        Either a literal integer width or a string expression to evaluate
    ctx : dict
        Context dictionary providing variables and functions for evaluation

    Returns
    -------
    int
        Evaluated width in bits

    Raises
    ------
    TypeError
        If width is neither int nor str

    Examples
    --------
    >>> eval_width(8, {})
    8
    >>> eval_width("ceil(log2(4))", {"ceil": math.ceil, "log2": math.log2})
    2
    """
    if isinstance(width, int):
        return max(width,1)
    elif isinstance(width, str):
        # Evaluate expression with restricted builtins (for security)
        return max(int(eval(width, {"__builtins__": {}}, ctx)),1)
    else:
        raise TypeError("Invalid width type")

def create_regmap_params_vh_file(regmap_yaml_path, output_vh_path=None, output_v_path=None, output_py_path=None):
    """Generate DMA register map files from YAML specification.

    This is the main generator function that reads a regmap.yaml file and produces
    three output files that implement the DMA register map protocol.

    See module docstring for detailed information about the bit-packing protocol.

    Parameters
    ----------
    regmap_yaml_path : str
        Path to directory containing regmap.yaml
    output_vh_path : str, optional
        Output path for regmap_params.vh
    output_v_path : str, optional
        Output path for dma_storage.v
    output_py_path : str, optional
        Output path for regmap_pack.py
    """

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------
    if output_vh_path is None:
        output_vh_path = os.path.join(os.path.dirname(regmap_yaml_path), "include")

    if output_v_path is None:
        output_v_path = os.path.join(os.path.dirname(regmap_yaml_path))
    if output_py_path is None:
        output_py_path = Path(__file__).resolve().parent

    YAML_FILE = os.path.join(regmap_yaml_path, "regmap.yaml")
    VERILOG_PARAMS_OUT = os.path.join(output_vh_path, "regmap_params.vh")
    PYTHON_OUT = os.path.join(output_py_path, "regmap_pack.py")
    DMA_STORAGE_OUT = os.path.join(output_v_path, "dma_storage.v")

    # ------------------------------------------------------------
    # Load YAML
    # ------------------------------------------------------------
    with open(YAML_FILE, "r") as f:
        config = yaml.safe_load(f)

    dma_bitwidth = int(config["dma_bitwidth"])
    registers = config["registers"]
    env_ctx = dict(os.environ)
    comb_ctx = context | env_ctx
    for reg in registers:
        reg["width"] = eval_width(reg["width"], comb_ctx)
    # ------------------------------------------------------------
    # Calculate transmission splitting & bit positions
    # ------------------------------------------------------------
    # This is the core bit-packing algorithm. It groups registers into
    # 64-bit transmissions, starting a new transmission whenever adding
    # the next register would exceed 64 bits.
    #
    # Algorithm:
    # 1. Start with transmission 0, bit position 0
    # 2. For each register:
    #    a. Check if current_position + width <= DMA_BITWIDTH
    #    b. If yes: pack at current position, advance position
    #    c. If no: start new transmission (trans++, pos=0)
    # 3. Record {name, width, trans, pos} for each register
    #
    # Result: List of transmissions, each containing list of registers
    #         with their bit positions within that transmission

    transmissions = []      # List of transmissions (each is a list of reg_entries)
    current_trans = []      # Registers being packed into current transmission
    bitpos = 0              # Current bit position within transmission (0-63)
    trans_id = 0            # Current transmission number (0, 1, 2, ...)

    for reg in registers:
        width = int(reg["width"])

        # Check if adding this register would exceed DMA width
        if bitpos + width > dma_bitwidth:
            # Save current transmission and start new one
            transmissions.append(current_trans)
            current_trans = []
            bitpos = 0
            trans_id += 1

        # Add register to current transmission with its bit position
        reg_entry = {
            "name": reg["name"],      # e.g., 'wght_cycles_reg'
            "width": width,           # e.g., 8 (bits)
            "trans": trans_id,        # e.g., 0 (first transmission)
            "pos": bitpos             # e.g., 0 (starting bit within transmission)
        }
        current_trans.append(reg_entry)
        bitpos += width  # Advance to next available bit position

    # Don't forget the last transmission
    if current_trans:
        transmissions.append(current_trans)

    num_transmissions = len(transmissions)

    # ------------------------------------------------------------
    # 1) Write Verilog parameter file (regmap_params.vh)
    # ------------------------------------------------------------
    # This file defines parameters used in dma_storage.v to extract values
    # from packed DMA transmissions. Each parameter defines a bit position.
    #
    # Generated format:
    #   parameter PARAMETER_POS_<trans>_<index> = <bit_position>;
    #
    # Example:
    #   parameter PARAMETER_POS_0_0 = 0;      // First param at bit 0
    #   parameter PARAMETER_POS_0_1 = 8;      // Second param at bit 8
    #
    # These are used in dma_storage.v like:
    #   wght_cycles_reg <= dma_data_i[PARAMETER_POS_0_0 + 7 : PARAMETER_POS_0_0];
    #                                    ↑ extracts bits [7:0]

    with open(VERILOG_PARAMS_OUT, "w+") as vf:
        # Header guard
        vf.write("`ifndef REGMAP_PARAMS_VH\n`define REGMAP_PARAMS_VH\n\n")
        vf.write("// Auto-generated from {}\n\n".format(YAML_FILE))

        # DMA configuration parameters
        vf.write(f"parameter DMA_BITWIDTH = {dma_bitwidth};\n")
        vf.write(f"parameter TRANSMISSIONS = {num_transmissions};\n\n")

        # Generate bit position parameters for each transmission
        for tid, regs in enumerate(transmissions):
            vf.write(f"// Transmission {tid} Offsets\n")
            for i, reg in enumerate(regs):
                pname = f"PARAMETER_POS_{tid}_{i}"
                if i == 0:
                    # First parameter in transmission starts at bit 0
                    vf.write(f"parameter {pname} = 0;\n")
                else:
                    # Subsequent parameters: position = prev_position + prev_width
                    prev_name = f"PARAMETER_POS_{tid}_{i-1}"
                    prev_width = regs[i-1]['width']
                    vf.write(f"parameter {pname} = {prev_name} + {prev_width};\n")
            vf.write("\n")

        # Footer
        vf.write("`endif // REGMAP_PARAMS_VH\n")

    print(f"[OK] Verilog parameters written to {VERILOG_PARAMS_OUT}.")

    # ------------------------------------------------------------
    # 2) Write Python pack/unpack module (regmap_pack.py)
    # ------------------------------------------------------------
    # This file provides Python functions to pack/unpack parameter dictionaries
    # to/from 64-bit DMA words. Used by layer_parameters.py to prepare data
    # for DMA transfer.
    #
    # Generated functions:
    #   pack_registers(values)   - Dict[str, int] -> List[int]
    #   unpack_registers(words)  - List[int] -> Dict[str, int]
    #
    # Pack example:
    #   values = {'wght_cycles_reg': 9, 'stride_x_reg': 1, ...}
    #   words = pack_registers(values)  # [0x..., 0x..., 0x..., 0x...]
    #
    # Unpack example:
    #   words = [0x123456789ABCDEF0, ...]
    #   values = unpack_registers(words)  # {'wght_cycles_reg': 9, ...}

    with open(PYTHON_OUT, "w") as pf:
        pf.write("# Auto-generated from {}\n".format(YAML_FILE))
        pf.write(f"DMA_BITWIDTH = {dma_bitwidth}\n")
        pf.write(f"TRANSMISSIONS = {num_transmissions}\n\n")

        # Generate REGISTERS list containing all register metadata
        pf.write("REGISTERS = [\n")
        for regs in transmissions:
            for reg in regs:
                pf.write(f"    {{'name': '{reg['name']}', 'width': {reg['width']}, 'trans': {reg['trans']}, 'pos': {reg['pos']}}},\n")
        pf.write("]\n\n")

        # Generate pack_registers function
        pf.write("def pack_registers(values):\n")
        pf.write("    \"\"\"Pack parameter dictionary into DMA transmission words.\n")
        pf.write("    \n")
        pf.write("    Parameters\n")
        pf.write("    ----------\n")
        pf.write("    values : dict\n")
        pf.write("        Parameter values keyed by register name\n")
        pf.write("    \n")
        pf.write("    Returns\n")
        pf.write("    -------\n")
        pf.write("    list of int\n")
        pf.write("        64-bit DMA words (one per transmission)\n")
        pf.write("    \"\"\"\n")
        pf.write("    words = [0] * TRANSMISSIONS\n")
        pf.write("    for reg in REGISTERS:\n")
        pf.write("        val = values[reg['name']]\n")
        pf.write("        mask = (1 << reg['width']) - 1\n")
        pf.write("        words[reg['trans']] |= (val & mask) << reg['pos']\n")
        pf.write("    return words\n\n")

        # Generate unpack_registers function
        pf.write("def unpack_registers(words):\n")
        pf.write("    \"\"\"Unpack DMA words into parameter dictionary.\n")
        pf.write("    \n")
        pf.write("    Parameters\n")
        pf.write("    ----------\n")
        pf.write("    words : list of int\n")
        pf.write("        64-bit DMA words (one per transmission)\n")
        pf.write("    \n")
        pf.write("    Returns\n")
        pf.write("    -------\n")
        pf.write("    dict\n")
        pf.write("        Parameter values keyed by register name\n")
        pf.write("    \"\"\"\n")
        pf.write("    values = {}\n")
        pf.write("    for reg in REGISTERS:\n")
        pf.write("        mask = (1 << reg['width']) - 1\n")
        pf.write("        values[reg['name']] = (words[reg['trans']] >> reg['pos']) & mask\n")
        pf.write("    return values\n")

    print(f"[OK] Python pack/unpack written to {PYTHON_OUT}.")

    # ------------------------------------------------------------
    # 3) Write dma_storage.v module
    # ------------------------------------------------------------
    # This Verilog module receives DMA transmissions and extracts individual
    # parameters using the bit positions defined in regmap_params.vh.
    #
    # Module interface:
    #   - Inputs: clk, rst, write_en, write_addr[trans_id], dma_data_i[63:0]
    #   - Outputs: Individual parameter registers (wght_cycles_reg, etc.)
    #
    # Operation:
    #   1. On write_en, check which transmission (write_addr)
    #   2. Extract all parameters for that transmission using bit slicing
    #   3. Store in output registers for use by other modules
    #
    # Bit slicing example:
    #   wght_cycles_reg <= dma_data_i[PARAMETER_POS_0_0 + 7 : PARAMETER_POS_0_0];
    #                                     ↑ this evaluates to [7:0]

    with open(DMA_STORAGE_OUT, "w") as dv:
        dv.write("// Auto-generated from {}\n".format(YAML_FILE))
        dv.write("//\n")
        dv.write("// DMA Storage Module - Receives and unpacks layer configuration parameters\n")
        dv.write("//\n")
        dv.write("// This module receives packed 64-bit DMA transmissions and extracts individual\n")
        dv.write("// configuration parameters using bit slicing. Each layer requires multiple DMA\n")
        dv.write("// transmissions to configure all parameters.\n")
        dv.write("//\n")
        dv.write("// Operation:\n")
        dv.write("//   1. CPU/DMA controller writes to this module with write_en=1\n")
        dv.write("//   2. write_addr selects which transmission (0 to TRANSMISSIONS-1)\n")
        dv.write("//   3. dma_data_i contains the packed 64-bit word\n")
        dv.write("//   4. Parameters are extracted and stored in output registers\n")
        dv.write("//   5. Other hardware modules read these registers for layer execution\n\n")
        dv.write("`include \"regmap_params.vh\"\n\n")

        # Module header
        dv.write("module dma_storage (\n")
        dv.write("    input  wire                  clk_i,\n")
        dv.write("    input  wire                  rst_ni,           // Active-low reset\n")
        dv.write("    input  wire                  write_en,         // Write enable from DMA\n")
        dv.write("    input  wire [$clog2(TRANSMISSIONS)-1:0] write_addr,  // Transmission ID\n")
        dv.write(f"    input  wire [DMA_BITWIDTH-1:0] dma_data_i,  // Packed 64-bit data\n")

        # Output ports (one per register)
        for regs in transmissions:
            for reg in regs:
                dv.write(f"    output reg [{reg['width']-1}:0] {reg['name']},\n")
        dv.seek(dv.tell()-2)  # Remove last comma
        dv.write("\n);\n\n")

        # Sequential logic
        dv.write("// Sequential logic: Store DMA data on write_en\n")
        dv.write("always @(posedge clk_i, negedge rst_ni) begin\n")
        dv.write("    if (!rst_ni) begin\n")
        dv.write("        // Reset all registers to 0\n")
        for regs in transmissions:
            for reg in regs:
                dv.write(f"        {reg['name']} <= {reg['width']}'d0;\n")
        dv.write("    end else if (write_en) begin\n")
        dv.write("        // Extract parameters based on transmission ID\n")
        dv.write("        case (write_addr)\n")

        # Generate case for each transmission
        for tid, regs in enumerate(transmissions):
            dv.write(f"            {tid}: begin  // Transmission {tid}\n")
            for i, reg in enumerate(regs):
                pname = f"PARAMETER_POS_{tid}_{i}"
                # Generate bit slice: dma_data_i[high:low]
                dv.write(f"                {reg['name']} <= dma_data_i[{pname} + {reg['width'] - 1} : {pname}];\n")
            dv.write("            end\n")
        dv.write("        endcase\n")
        dv.write("    end\n")
        dv.write("end\n\n")
        dv.write("endmodule\n")

    print(f"[OK] dma_storage module written to {DMA_STORAGE_OUT}.")


if __name__ == "__main__":

    # ------------------------------------------------------------
    # Path to .vh-file
    # ------------------------------------------------------------
    
    # first argument: path to regmap.yaml 
    try:
        path = sys.argv[1]
    except:
        path = os.getcwd()

    # second argument: output path (optional)
    try:
        output_pathvh = sys.argv[2]
    except:
        output_pathvh = path

    # third argument: output path (optional)
    try:
        output_pathv = sys.argv[3]
    except:
        output_pathv = path
    # third argument: output path (optional)
    try:
        output_pathpy = sys.argv[4]
    except:
        output_pathpy = None
    

    # ------------------------------------------------------------
    # Create regmap_params.vh file
    # ------------------------------------------------------------
    create_regmap_params_vh_file(path, output_pathvh, output_pathv, output_pathpy)

