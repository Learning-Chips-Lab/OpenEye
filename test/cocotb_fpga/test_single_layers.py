# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import logging
import os
import sys
import pytest
import cocotb_test.simulator
import subprocess
from pathlib import Path

logger = logging.getLogger("cocotb")
directory = (os.path.abspath(os.getcwd()))

import open_eye.test_utils_main as ptu
import open_eye.vh_file_creator as vh_file_creator
import open_eye.generator as generator
from open_eye import hdl_dir, test_dir, open_eye_dir

##########################################################################################
# Clock Configuration
clk_cycle = 20
clk_cycle_unit = "ns"

clk_delay_in = 100
clk_delay_unit_in = "ps"

clk_delay_out = 100
clk_delay_unit_out = "ps"

sim = "verilator"
waves_output = True
##########################################################################################


@pytest.mark.parametrize("LOGGER_LEVEL", [0])

@pytest.mark.parametrize("NUM_FILTERS", [8])
@pytest.mark.parametrize("STRIDE", [1])
@pytest.mark.parametrize("KERNEL_SIZE_X", [3])
@pytest.mark.parametrize("KERNEL_SIZE_Y", [3])
@pytest.mark.parametrize("INPUT_SIZE_X", [4])
@pytest.mark.parametrize("INPUT_SIZE_Y", [1])
@pytest.mark.parametrize("INPUT_CHANNELS", [4])
@pytest.mark.parametrize("USE_SPARSE_IACTS", [0])
@pytest.mark.parametrize("USE_SPARSE_WGHTS", [0])
@pytest.mark.parametrize("USE_RANDOM_VALUES", [1])

#@pytest.mark.parametrize("CLUSTER_ROWS", [1,2,4,6,8])
#@pytest.mark.parametrize("CLUSTER_COLUMNS", [1,2])
@pytest.mark.parametrize("CLUSTER_ROWS", [1])
@pytest.mark.parametrize("CLUSTER_COLUMNS", [1])
@pytest.mark.parametrize("NUM_GLB_IACT", [1])
@pytest.mark.parametrize("NUM_GLB_PSUM", [4])
@pytest.mark.parametrize("NUM_GLB_WGHT", [3])
@pytest.mark.parametrize("DMA_BITWIDTH", [32])
#@pytest.mark.parametrize("SPARSITY_EN", [0,1])
@pytest.mark.parametrize("SPARSITY_EN", [0])
#@pytest.mark.parametrize("PARALLEL_MACS", [1,2])
@pytest.mark.parametrize("PARALLEL_MACS", [1])


def test_single_conv_layer(
    request,
    NUM_FILTERS, STRIDE, KERNEL_SIZE_X, KERNEL_SIZE_Y, INPUT_SIZE_X, INPUT_SIZE_Y, INPUT_CHANNELS,
    USE_SPARSE_IACTS, USE_SPARSE_WGHTS, USE_RANDOM_VALUES,
    CLUSTER_ROWS, CLUSTER_COLUMNS, DMA_BITWIDTH, SPARSITY_EN, NUM_GLB_IACT, NUM_GLB_PSUM, NUM_GLB_WGHT, PARALLEL_MACS, LOGGER_LEVEL
):
    os.environ["NUM_GLB_IACT"] = str(NUM_GLB_IACT)
    os.environ["CLUSTER_COLUMNS"] = str(CLUSTER_COLUMNS)
    os.environ["CLUSTER_ROWS"] = str(CLUSTER_ROWS)
    os.environ["NUM_GLB_PSUM"] = str(NUM_GLB_PSUM)
    os.environ["NUM_GLB_WGHT"] = str(NUM_GLB_WGHT)
    os.environ["DMA_BITWIDTH"] = str(DMA_BITWIDTH)
    os.environ["SPARSITY_EN"] = str(SPARSITY_EN)
    os.environ["PARALLEL_MACS"] = str(PARALLEL_MACS)
    
    layer = os.environ.get("LAYER", "VGG-16")
    
    os.environ["LAYER"] = layer

    test_name = f"test_fpga__data_{layer}_xr{CLUSTER_ROWS}xc{CLUSTER_COLUMNS}_macs{PARALLEL_MACS}_spen{SPARSITY_EN}"
    
    # Lokales Build-Verzeichnis (Verilator-Kompilierung bleibt schnell lokal)
    build_dir = Path(test_dir) / ".temp" / test_name
    build_dir.mkdir(parents=True, exist_ok=True)
    build_dir_str = str(build_dir)

    # Zielverzeichnis für die .fst-Waveform bestimmen und Symlink anlegen
    custom_target_base = request.config.getoption("--target-dir")
    if custom_target_base and waves_output:
        wave_dir = Path(custom_target_base) / test_name
        wave_dir.mkdir(parents=True, exist_ok=True)
        remote_fst = wave_dir / "OpenEye_FPGA.fst"
        local_fst = build_dir / "OpenEye_FPGA.fst"

        # Vorhandene Datei oder alten Symlink im lokalen Build-Ordner entfernen
        if local_fst.exists() or local_fst.is_symlink():
            local_fst.unlink()
        local_fst.symlink_to(remote_fst)

    if sim == "verilator":
        compile_args = [
            "-Wall",
            "-Wno-fatal",
            "-O3",
            "--x-assign", "fast",
            "--x-initial", "fast",
            "-CFLAGS", "-O3 -march=native",
            f"-I{build_dir_str}",
        ]
        sim_args = []
        if waves_output:
            compile_args.extend([
                "--trace-fst",
                "--trace-structs",
            ])
            sim_args.append("--trace-fst")
    else:
        compile_args = ["-g2012"]
        sim_args = []

    dut = 'OpenEye_FPGA'
    module = 'OpenEye_FPGA_tb'
    toplevel = dut
    verilog_sources = ptu.get_verilog_sources(str(hdl_dir))

    regmap_dir = os.path.join(test_dir, 'cocotb_fpga')
    subprocess.run(['python', os.path.join(open_eye_dir, 'generator.py'), regmap_dir, build_dir_str], check=True)

    vh_file_creator.create_vh_file_from_envvars(build_dir_str, str(hdl_dir) + "/", toplevel="OpenEye_FPGA")
    generator.create_regmap_params_vh_file(regmap_dir, build_dir_str, build_dir_str)

    verilog_sources = [
        src for src in verilog_sources
        if os.path.basename(src) != "dma_storage.v"
    ]
    verilog_sources.append(os.path.join(build_dir_str, "dma_storage.v"))
        
    extra_env = {
        "CLOCK_LEN": str(clk_cycle),
        "CLOCK_UNIT": clk_cycle_unit,
        "CLOCK_DELAY_INPUT": str(clk_delay_in),
        "CLOCK_DELAY_UNIT_INPUT": clk_delay_unit_in,
        "CLOCK_DELAY_OUTPUT": str(clk_delay_out),
        "CLOCK_DELAY_UNIT_OUTPUT": clk_delay_unit_out,
        "LAYER": layer,
        "NUM_FILTERS": str(NUM_FILTERS),
        "STRIDE": str(STRIDE),
        "KERNEL_SIZE_X": str(KERNEL_SIZE_X),
        "KERNEL_SIZE_Y": str(KERNEL_SIZE_Y),
        "INPUT_SIZE_X": str(INPUT_SIZE_X),
        "INPUT_SIZE_Y": str(INPUT_SIZE_Y),
        "INPUT_CHANNELS": str(INPUT_CHANNELS),
        "USE_SPARSE_IACTS": str(USE_SPARSE_IACTS),
        "USE_SPARSE_WGHTS": str(USE_SPARSE_WGHTS),
        "USE_RANDOM_VALUES": str(USE_RANDOM_VALUES),
        "CLUSTER_ROWS": str(CLUSTER_ROWS),
        "CLUSTER_COLUMNS": str(CLUSTER_COLUMNS),
        "PARALLEL_MACS": str(PARALLEL_MACS),
        "NUM_GLB_IACT": str(NUM_GLB_IACT),
        "NUM_GLB_WGHT": str(NUM_GLB_WGHT),
        "NUM_GLB_PSUM": str(NUM_GLB_PSUM),
        "LOGGER_LEVEL": str(LOGGER_LEVEL),
        "DMA_BITWIDTH": str(DMA_BITWIDTH),
        "SPARSITY_EN": str(SPARSITY_EN),
        "COCOTB_LOG_FILE_PATH": os.path.join(build_dir_str, "cocotb_sim.log"),
        "COCOTB_TRACE": "0"
    }

    defines = {
        "USE_INTERNAL_PARAMS_PE": "1",
        "USE_INTERNAL_PARAMS_PE_cluster": "1"
    }
    if not waves_output:
        defines["NO_TRACE"] = "TRUE"

    results = cocotb_test.simulator.run(
        python_search=[str(test_dir), str(Path(test_dir) / "cocotb_fpga")],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=build_dir_str,
        sim_dir=build_dir_str,  # isoliert demo/, results.xml und tmp-Dateien pro Trial
        testcase='start_test_fpga',
        defines=defines,
        parameters={"MAX_POOLING": 1, "AVERAGE_POOLING": 0},
        force_compile=True,
        waves=waves_output,
        simulator=sim,
        compile_args=compile_args,
        sim_args=sim_args,
        extra_env=extra_env
    )

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])