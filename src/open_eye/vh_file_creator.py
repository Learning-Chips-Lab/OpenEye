# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Verilog header file generator for OpenEye accelerator configuration.

This module provides functionality to create and manage Verilog header (.vh) files
that define parameters for the OpenEye neural network accelerator hardware.

Features:
- Automatic parameter file generation
- File change detection and management
- Environment variable integration
- Compilation trigger control
"""

from pathlib import Path
import os
import sys
from typing import Optional, Union
import open_eye.open_eye_parameters as oep
import open_eye.generic_test_utils as gtu
import json

directory = (os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
hdl_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir, "hdl")


def are_files_identical(file1_path: str, file2_path: str) -> bool:
    """Compare two files for identical content.

    Args:
        file1_path: Path to first file
        file2_path: Path to second file

    Returns:
        bool: True if files are identical, False otherwise or on error
    """
    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            file1_content = file1.read()
            file2_content = file2.read()
            return file1_content == file2_content
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"An unknown error occurred: {e}")
        return False


# Suffix of the generated .vh file, per top-level module. PE.v and PE_cluster.v
# `include their own parameter file whenever they are compiled, no matter which
# module sits on top, so those two files are always generated as well.
VH_SUFFIX_BY_TOPLEVEL = {
    "OpenEye_FPGA": "_FPGA",
    # OpenEye_Parallel.v only includes parameters.vh under USE_EXTERNAL_PARAMS,
    # which the tests do not define; it is listed so the PE and PE_cluster
    # headers below are generated for it.
    "OpenEye_Parallel": "",
    "PE_cluster": "_PE_cluster",
    "PE": "_PE",
}


def create_vh_file(openeye_parameter: object, filename: str = 'parameters.vh',
                   toplevel: Optional[str] = None) -> None:
    """Create a Verilog header file with OpenEye parameters.

    Args:
        openeye_parameter: OpenEye parameter configuration object
        filename: Output .vh file path
        toplevel: Module the file is written for (from TOPLEVEL env when None).
            "OpenEye_FPGA" gets the full accelerator configuration, every other
            module gets the PE-level parameters only.

    Generated Parameters:
        - CLUSTER_ROWS: Number of cluster rows (Y dimension)
        - NUM_GLB_IACT: Number of global input activation buffers
        - NUM_GLB_PSUM: Number of global partial sum buffers
        - NUM_GLB_WGHT: Number of global weight buffers
    """
    toplevel = toplevel or gtu.load_env_to_variable("TOPLEVEL", "")
    if toplevel == "OpenEye_FPGA":
        gtu.delete_files_in_directory('demo/')
        try:
            with open("shared_config.json", "r") as f:
                config_data = json.load(f)
                TRANSMISSIONS = config_data.get("TRANSMISSIONS", 8)
        except FileNotFoundError:
            TRANSMISSIONS = 8
        with open(filename, 'w') as txt_file:
            txt_file.write(f"parameter CLUSTER_ROWS  = {openeye_parameter.Clusters_Y},\n")
            txt_file.write(f"parameter CLUSTER_COLUMNS  = {openeye_parameter.Clusters_X},\n")
            txt_file.write(f"parameter NUM_GLB_IACT  = {openeye_parameter.NUM_GLB_IACT},\n")
            txt_file.write(f"parameter NUM_GLB_PSUM  = {openeye_parameter.NUM_GLB_PSUM},\n")
            txt_file.write(f"parameter NUM_GLB_WGHT = {openeye_parameter.NUM_GLB_WGHT},\n")
            txt_file.write(f"parameter IACT_RAM_CELLS = {openeye_parameter.IACT_RAM_CELLS},\n")
            txt_file.write(f"parameter BUFFER_WIDTH = {openeye_parameter.BUFFER_WIDTH},\n")
            txt_file.write(f"parameter BUFFER_WIDTH_WGHT = {openeye_parameter.BUFFER_WIDTH_WGHT},\n")
            txt_file.write(f"parameter BUFFER_WIDTH_PSUM = {openeye_parameter.BUFFER_WIDTH_PSUM},\n")
            txt_file.write(f"parameter BRANCHES = {openeye_parameter.BRANCHES},\n")
            txt_file.write(f"parameter QUANT_AMOUNT = {openeye_parameter.QUANT_AMOUNT},\n")
            txt_file.write(f"parameter DATA_PSUM_BITWIDTH = {openeye_parameter.DATA_PSUM_BITWIDTH},\n")
            txt_file.write(f"parameter TRANS_WORDS = {openeye_parameter.TRANS_WORDS},\n")
            txt_file.write(f"parameter DMA_BITWIDTH = {openeye_parameter.DMA_BITWIDTH},\n")
            txt_file.write(f"parameter PARALLEL_MACS = {openeye_parameter.PARALLEL_MACS},\n")
            txt_file.write(f"parameter SPARSITY_EN = {openeye_parameter.SPARSITY_EN},\n")
            txt_file.write(f"parameter TRANS_BITWIDTH_IACT = {openeye_parameter.IACT_Trans_Bitwidth},\n")
            txt_file.write(f"parameter TRANS_BITWIDTH_WGHT = {openeye_parameter.WGHT_Trans_Bitwidth},\n")
            txt_file.write(f"parameter TRANSMISSIONS = {TRANSMISSIONS},\n")
    else:
        with open(filename, 'w') as txt_file:
            txt_file.write(f"parameter PARALLEL_MACS = {openeye_parameter.PARALLEL_MACS},\n")
            txt_file.write(f"parameter SPARSITY_EN = {openeye_parameter.SPARSITY_EN},\n")


def _update_vh_file(openeye_parameter: object, file_path_vh: str, file_path_hdl: str,
                    toplevel: str) -> None:
    """Write one parameter file, recompile-triggering the module only on change.

    The file is written to a temporary name first and compared with the current
    one. If nothing changed the temporary is dropped, otherwise it replaces the
    old file and the corresponding .v file is touched so the simulator rebuilds.
    """
    suffix = VH_SUFFIX_BY_TOPLEVEL[toplevel]
    pre_param_path = os.path.join(file_path_vh, "pre_parameters" + suffix + ".vh")
    param_path = os.path.join(file_path_vh, "parameters" + suffix + ".vh")

    create_vh_file(openeye_parameter, pre_param_path, toplevel=toplevel)

    if are_files_identical(pre_param_path, param_path):
        os.remove(pre_param_path)
        print(f"Same vh-file for {toplevel}. Do not recompile")
        return

    if os.path.exists(param_path):
        os.remove(param_path)
    os.rename(pre_param_path, param_path)

    verilog_path = os.path.join(file_path_hdl, f"{toplevel}.v")
    print(f"Touching {verilog_path}")
    os.utime(verilog_path, None)
    print(f"Different vh-file for {toplevel}, updated vh-file")


def create_vh_file_from_envvars(
    file_path_vh: Optional[str] = None,
    file_path_hdl: str = str(hdl_dir),
    toplevel: Optional[str] = None
) -> None:
    """Create Verilog header file using environment variable configuration.

    Args:
        file_path_vh: Path for .vh files (from VH_PATH env or None)
        file_path_hdl: Path to HDL source (from HDL_PATH env or default)
        toplevel: Top-level module name (from TOPLEVEL env or None)

    Environment Variables:
        - VH_PATH: Directory for .vh files
        - HDL_PATH: Directory containing HDL source
        - TOPLEVEL: Top-level module name
    """
    file_path_vh = file_path_vh or gtu.load_env_to_variable("VH_PATH", os.getcwd())
    file_path_hdl = file_path_hdl or gtu.load_env_to_variable("HDL_PATH", hdl_dir)
    toplevel = toplevel or gtu.load_env_to_variable("TOPLEVEL", "")
    print(toplevel)

    if toplevel not in VH_SUFFIX_BY_TOPLEVEL:
        raise ValueError(
            f"No parameter file defined for toplevel {toplevel!r}. "
            f"Known toplevels: {sorted(VH_SUFFIX_BY_TOPLEVEL)}"
        )

    openeye_parameter = oep.get_oep(serial=False)

    # Every module in the compile that includes a parameter header unconditionally
    # needs that header on disk, not just the top-level one. PE.v and PE_cluster.v
    # always do; OpenEye_FPGA.v is part of the full-design source lists even when
    # OpenEye_Parallel is on top.
    also = ["PE_cluster", "PE"]
    if toplevel in ("OpenEye_Parallel", "OpenEye_FPGA"):
        also.append("OpenEye_FPGA")
    for module in dict.fromkeys([toplevel] + also):
        _update_vh_file(openeye_parameter, file_path_vh, file_path_hdl, module)


if __name__ == "__main__":
    create_vh_file_from_envvars()