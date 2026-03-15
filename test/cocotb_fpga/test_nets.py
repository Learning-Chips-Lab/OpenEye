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
import tensorflow as tf
import math

logger = logging.getLogger("cocotb")

directory = (os.path.abspath(os.getcwd()))

import open_eye.test_utils_main as ptu
import open_eye.vh_file_creator as vh_file_creator
import open_eye.generator as generator
from open_eye import hdl_dir, test_dir, open_eye_dir



##########################################################################################
# TODO: put everything below in a common function
clk_cycle = 20
clk_cycle_unit = "ns"

clk_delay_in = 100
clk_delay_unit_in = "ps"

clk_delay_out = 100
clk_delay_unit_out = "ps"

##########################################################################################

#@pytest.fixture(scope="session")
def create_tf_model(model_path):
    model = tf.keras.models.Sequential()
    channels = 1
    x_axis = 28
    y_axis = 28
    filters = 16
    pool_x_axis = 2
    pool_y_axis = 2
    strides = (1,1)
    model.add(tf.keras.layers.Conv2D(filters, (3, 3), padding="SAME", input_shape=(x_axis, y_axis, channels), strides = strides))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (pool_x_axis, pool_y_axis), strides=(pool_x_axis,pool_y_axis), padding="valid"))
    channels = filters
    x_axis   = math.ceil(x_axis/pool_x_axis)
    y_axis   = math.ceil(y_axis/pool_y_axis)
    filters  = 32
    model.add(tf.keras.layers.Conv2D(filters, (3, 3), padding="SAME", input_shape=(x_axis, y_axis, channels), strides = strides))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (pool_x_axis, pool_y_axis), strides=(pool_x_axis,pool_y_axis), padding="valid"))
    channels = filters
    x_axis   = math.ceil(x_axis/pool_x_axis)
    y_axis   = math.ceil(y_axis/pool_y_axis)
    filters  = 32
    model.add(tf.keras.layers.Conv2D(filters, (3, 3), padding="SAME", input_shape=(x_axis, y_axis, channels), strides = strides))
    model.add(tf.keras.layers.Flatten())
    output_size  = 32
    model.add(tf.keras.layers.Dense(units=output_size, use_bias = True))
    output_size  = 10
    model.add(tf.keras.layers.Dense(units=output_size, use_bias = True))
    model = model.save(model_path)
    

@pytest.mark.parametrize("USE_SPARSE_IACTS", [0])
@pytest.mark.parametrize("USE_SPARSE_WGHTS", [0])
@pytest.mark.parametrize("USE_RANDOM_VALUES", [1])
@pytest.mark.parametrize("CLUSTER_ROWS", [8])
@pytest.mark.parametrize("NUM_GLB_IACT", [3])
@pytest.mark.parametrize("NUM_GLB_PSUM", [4])
@pytest.mark.parametrize("NUM_GLB_WGHT", [3])
@pytest.mark.parametrize("LOGGER_LEVEL", [0])
@pytest.mark.parametrize("MODEL_PATH", ["MNIST"])
def test_mnist(
    USE_SPARSE_IACTS, USE_SPARSE_WGHTS, USE_RANDOM_VALUES,
    CLUSTER_ROWS, NUM_GLB_IACT, NUM_GLB_PSUM, NUM_GLB_WGHT, LOGGER_LEVEL, MODEL_PATH,
    request
):
    #create_tf_model()
    os.environ["NUM_GLB_IACT"] = str(NUM_GLB_IACT)
    os.environ["CLUSTER_ROWS"] = str(CLUSTER_ROWS)
    # NodeID aus pytest, als eindeutiger Ordnername
    nodeid = request.node.nodeid.replace("::", "_").replace("/", "_").replace("[","_").replace("]","_")
    target_dir = os.path.join(test_dir, '.temp/' + nodeid)
    os.makedirs(target_dir, exist_ok=True)
    model_path = target_dir + "MNIST.h5"
    create_tf_model(model_path)
    dut = 'OpenEye_FPGA'
    module = 'OpenEye_FPGA_tb'
    toplevel = dut
    verilog_sources = ptu.get_verilog_sources(hdl_dir)


    regmap_dir = os.path.join(test_dir, 'cocotb_fpga')
    result = subprocess.run(['python', os.path.join(open_eye_dir, 'generator.py'), regmap_dir,target_dir])

    vh_file_creator.create_vh_file_from_envvars(target_dir, hdl_dir + "/", toplevel="OpenEye_FPGA")
    generator.create_regmap_params_vh_file(os.path.join(test_dir, "cocotb_fpga"), target_dir, target_dir)

    results = cocotb_test.simulator.run(
        python_search=[test_dir],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=target_dir,
        testcase='model_test',
        defines={"NO_TRACE": "TRUE"},
        force_compile=True,
        waves=True,
        simulator="icarus",
        extra_env={
            "CLOCK_LEN": str(clk_cycle),
            "CLOCK_UNIT": clk_cycle_unit,
            "CLOCK_DELAY_INPUT": str(clk_delay_in),
            "CLOCK_DELAY_UNIT_INPUT": clk_delay_unit_in,
            "CLOCK_DELAY_OUTPUT": str(clk_delay_out),
            "CLOCK_DELAY_UNIT_OUTPUT": clk_delay_unit_out,
            "USE_SPARSE_IACTS": str(USE_SPARSE_IACTS),
            "USE_SPARSE_WGHTS": str(USE_SPARSE_WGHTS),
            "USE_RANDOM_VALUES": str(USE_RANDOM_VALUES),
            "CLUSTER_ROWS": str(CLUSTER_ROWS),
            "NUM_GLB_IACT": str(NUM_GLB_IACT),
            "NUM_GLB_WGHT": str(NUM_GLB_WGHT),
            "NUM_GLB_PSUM": str(NUM_GLB_PSUM),
            "LOGGER_LEVEL": str(LOGGER_LEVEL),
            "COCOTB_TRACE": "1",
            "MODEL_PATH" : model_path
        }
    )

    

if __name__ == '__main__':
    test_mnist(USE_SPARSE_IACTS=0, USE_SPARSE_WGHTS=0, USE_RANDOM_VALUES=1,
        CLUSTER_ROWS=4, NUM_GLB_IACT=1, NUM_GLB_PSUM=4, NUM_GLB_WGHT=3, LOGGER_LEVEL=0, MODEL_PATH="MNIST.h5",
        request=pytest.fixture(lambda: None)()
    )
