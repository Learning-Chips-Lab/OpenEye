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


def create_vh_file(openeye_parameter: object, filename: str = 'parameters.vh') -> None:
    """Create a Verilog header file with OpenEye parameters.

    Args:
        openeye_parameter: OpenEye parameter configuration object
        filename: Output .vh file path

    Generated Parameters:
        - CLUSTER_ROWS: Number of cluster rows (Y dimension)
        - NUM_GLB_IACT: Number of global input activation buffers
        - NUM_GLB_PSUM: Number of global partial sum buffers
        - NUM_GLB_WGHT: Number of global weight buffers
    """
    gtu.delete_files_in_directory('demo/')
    with open(filename, 'w') as txt_file:
        txt_file.write(f"parameter CLUSTER_ROWS  = {openeye_parameter.Clusters_Y},\n")
        txt_file.write(f"parameter NUM_GLB_IACT  = {openeye_parameter.NUM_GLB_IACT},\n")
        txt_file.write(f"parameter NUM_GLB_PSUM  = {openeye_parameter.NUM_GLB_PSUM},\n")
        txt_file.write(f"parameter NUM_GLB_WGHT = {openeye_parameter.NUM_GLB_WGHT},\n")
        txt_file.write(f"parameter RAM_CELLS = {openeye_parameter.RAM_CELLS},\n")
        txt_file.write(f"parameter BUFFER_WIDTH = {openeye_parameter.BUFFER_WIDTH},\n")
        txt_file.write(f"parameter BRANCHES = {openeye_parameter.BRANCHES},\n")
        txt_file.write(f"parameter QUANT_AMOUNT = {openeye_parameter.QUANT_AMOUNT},\n")
        txt_file.write(f"parameter DATA_PSUM_BITWIDTH = {openeye_parameter.DATA_PSUM_BITWIDTH},\n")
        


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

    # Create parameter file
    openeye_parameter = oep.get_oep(serial=False)
    pre_param_path = os.path.join(file_path_vh, "pre_parameters.vh")
    param_path = os.path.join(file_path_vh, "parameters.vh")
    
    create_vh_file(openeye_parameter, pre_param_path)
    
    if are_files_identical(pre_param_path, param_path):
        os.remove(pre_param_path)
        print("Same vh-file. Do not recompile")
    else:
        if os.path.exists(param_path):
            os.remove(param_path)
        os.rename(pre_param_path, param_path)
        
        if toplevel:
            verilog_path = os.path.join(file_path_hdl, f"{toplevel}.v")
            print(f"Touching {verilog_path}")
            os.utime(verilog_path, None)
        print("Different vh-file, updated vh-file")


if __name__ == "__main__":
    create_vh_file_from_envvars()