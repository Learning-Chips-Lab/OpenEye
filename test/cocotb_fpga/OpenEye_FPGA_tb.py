# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import sys
import os
import cocotb
from cocotb.clock import Clock
import test_utils.test_utils_main as ptu
import test_utils.rtl_test_utils as rtl_test_utils
import test_utils.timing_parameters as tp
import test_utils.generic_test_utils as gtu
import test_utils.DRAM as DRAM
import test_utils.time_stamper as time_stamper
import test_utils.open_eye_parameters as oep
import test_utils.layer_parameters as lp
import test_utils.simple_layer_operations as slo
import test_utils.layer_execution_state as les
import test_utils.data_create as data_create
import test_utils.tflite2model as tflite2model

os.environ["CLOCK_LEN"] = "10"
os.environ["CLOCK_UNIT"] = "ns"
os.environ["CLOCK_DELAY_INPUT"] = "100"
os.environ["CLOCK_DELAY_UNIT_INPUT"] = "ps"

os.environ["CLOCK_DELAY_OUTPUT"] = "100"
os.environ["CLOCK_DELAY_UNIT_OUTPUT"] = "ps"

time_printer = time_stamper.time_stamper()

tests_dir = os.path.abspath(os.path.dirname(__file__))
hdl_dir = (os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, "hdl")))

import logging

import sys
directory = (os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.insert(1, directory)


logger = logging.getLogger("cocotb")

try:
    log_level = int(os.getenv("LOGGER_LEVEL"))
except:
    logger.warning("Logger Level not given. Setting to INFO.")
    log_level = logging.INFO
logger.setLevel(logging.INFO)


def envvars_to_vars():    
    # Get variables that are used for the execution of the test
    only_files = gtu.load_env_to_variable("ONLY_FILES", 0)
    layer_mode = gtu.load_env_to_variable("LAYER", "Convolution")
    filters = gtu.load_env_to_variable("NUM_FILTERS", 4)
    kernelsize = gtu.load_env_to_variable("KERNEL_SIZE", 3)
    inputsize_x = gtu.load_env_to_variable("INPUT_SIZE_X", 64)
    inputsize_y = gtu.load_env_to_variable("INPUT_SIZE_Y", 1)
    outputsize = gtu.load_env_to_variable("OUTPUT_SIZE", 1)
    strides = gtu.load_env_to_variable("STRIDE", 1),gtu.load_env_to_variable("STRIDE", 1)
    channels = gtu.load_env_to_variable("INPUT_CHANNELS", 4)
    sparse_iacts = gtu.load_env_to_variable("USE_SPARSE_IACTS", 0)
    sparse_wghts = gtu.load_env_to_variable("USE_SPARSE_WEIGHTS", 0)
    return only_files, layer_mode, filters, kernelsize, inputsize_x, inputsize_y, outputsize, strides, channels, sparse_iacts, sparse_wghts

@cocotb.test()
async def model_test(dut):
    """ Test the DUT with a given DNN model.

    Load a trained DNN model (in TFLite format) 
    and simulate the execution using the OpenEye FPGA wrapper.
    """

    layer_es = les.LayerExecutionState()
    serial = 1
    clk_cycle = int(os.environ["CLOCK_LEN"])
    clk_cycle_unit = os.environ["CLOCK_UNIT"]

    clk_delay_in = int(os.environ["CLOCK_DELAY_INPUT"])
    clk_delay_unit_in = os.environ["CLOCK_DELAY_UNIT_INPUT"]

    clk_delay_out = int(os.environ["CLOCK_DELAY_OUTPUT"])
    clk_delay_unit_out = os.environ["CLOCK_DELAY_UNIT_OUTPUT"]

    ptp = tp.PortTimingParameters()
    ptp.initiate_params(clk_cycle, clk_cycle_unit, clk_delay_in, clk_delay_unit_in, clk_delay_out, clk_delay_unit_out)
    
    tflite_model_path = os.environ["MODEL_PATH"]
    model = tflite2model.create_model_from_tflite(tflite_model_path=tflite_model_path)

    only_files, layer_mode, filters, kernelsize, \
        inputsize_x, inputsize_y, outputsize, strides, \
        channels, sparse_iacts, sparse_wghts = envvars_to_vars()

    await execute_model(dut, only_files, sparse_iacts, sparse_wghts, layer_es, serial, ptp, model)


@cocotb.test()
async def single_layer_test(dut):
    """Simulate a single layer.
    
    Simulate a single layer using the OpenEye FPGA wrapper.
    """

    only_files, layer_mode, filters, kernelsize, inputsize_x, inputsize_y, outputsize, strides, channels, sparse_iacts, sparse_wghts = envvars_to_vars()


    try:
        use_random = int((os.environ("USE_RANDOM_VALUES")))
        print("try use random")
    except:
        use_random = 1
        logger.debug("USE_RANDOM_VALUES set to one")
        print("except use random")

    layer_es = les.LayerExecutionState()
    serial = 1
    clk_cycle = int(os.environ["CLOCK_LEN"])
    clk_cycle_unit = os.environ["CLOCK_UNIT"]

    clk_delay_in = int(os.environ["CLOCK_DELAY_INPUT"])
    clk_delay_unit_in = os.environ["CLOCK_DELAY_UNIT_INPUT"]

    clk_delay_out = int(os.environ["CLOCK_DELAY_OUTPUT"])
    clk_delay_unit_out = os.environ["CLOCK_DELAY_UNIT_OUTPUT"]

    ptp = tp.PortTimingParameters()
    ptp.initiate_params(clk_cycle, clk_cycle_unit, clk_delay_in, clk_delay_unit_in, clk_delay_out, clk_delay_unit_out)
    
    # Create a test model    
    gtu.select_gpu(0)

    #Here If-Condition test, wether use model or single Layer
    if(use_random):
        model = data_create.create_layer(layer_mode, filters, kernelsize, inputsize_x, inputsize_y, strides, channels, outputsize)
    else:
        model = tflite2model.create_model_from_tflite(use_random)
    #load_model_function
    
    await execute_model(dut, only_files, sparse_iacts, sparse_wghts, layer_es, serial, ptp, model)


async def execute_model(dut, only_files, sparse_iacts, sparse_wghts, layer_es, serial, ptp, model):
    openeye_parameter = oep.get_oep(serial)
    time_printer.timestamp("OpenEye parameters set. ", logger)

    gtu.delete_files_in_directory('demo/')

    if (only_files == 0) :
        # Start the clock
        clk = Clock(dut.clk_i, ptp.clk_cycle, units=ptp.clk_cycle_unit)
        cocotb.start_soon(clk.start())
        dut._log.info("Clock is %s " + ptp.clk_cycle_unit, ptp.clk_cycle)
        # reset the DUT
        await cocotb.start_soon(rtl_test_utils.reset_all_signals(ptp, dut, openeye_parameter.SERIAL))

    # Process the layers of the model one after another
    max_layers = len(model.layers)
    layer_parameters = [0 for _ in range(len(model.layers))]
    for layer_number, layer in reversed(list(enumerate(model.layers))):
        layer_parameters[max_layers - layer_number - 1] = lp.LayerParameters(layer_parameters, layer, openeye_parameter, layer_number, max_layers)
    layer_parameters = list(reversed(layer_parameters))
    # Create the OpenEye parameters and the DRAM given the model
    dram = DRAM.DRAMContents(model, layer_parameters)
    time_printer.timestamp("Initialized DRAM. ", logger)
    dram.write_initial_data_to_dram(model, layer_parameters, sparse_iacts, sparse_wghts)
    for layer_number, layer in enumerate(model.layers):
        # TODO: After refactoring LayerParameters, it is nicer to use the constructor 
        time_printer.timestamp("Layer parameters created. ", logger)
        calculated_results = ptu.collect_results(layer_number, layer_parameters[layer_number], dram, openeye_parameter.SERIAL)
        output_order = ptu.make_ref(openeye_parameter, layer_parameters[layer_number], layer_number, dram, calculated_results)
        if(logging.DEBUG >= log_level):
            time_printer.timestamp("Reference data created. ", logger)
        dram_layer_content = [dram.fmap[layer_number], dram.weights[layer_number], dram.bias[layer_number]]
        time_printer.timestamp("Start creating stream. " , logger)
        stream = ptu.write_stream(openeye_parameter, layer_parameters[layer_number], dram_layer_content, sparse_iacts, sparse_wghts)
        time_printer.timestamp("Streams set. " , logger)
        for _ in range(1):
            for layer_repetition in range(layer_parameters[layer_number].needed_total_transmissions):
                gtu.create_stream_file(stream[layer_repetition],layer_number,layer_repetition)
                if (only_files == 0) :
                    logger.info("Send stream.")
                    await cocotb.start_soon(rtl_test_utils.send_stream(ptp, dut, stream[layer_repetition], openeye_parameter, layer_parameters[layer_number], layer_repetition))
                    logger.info("Stream is sent.")
                    if (layer_number == max_layers - 1) :
                        await cocotb.start_soon(rtl_test_utils.await_enable_signal(ptp, dut))
                        if("Depthwise" in str(layer_parameters[layer_number].layer_name)):
                            await cocotb.start_soon(rtl_test_utils.compare_stream_Dw(ptp, dut, layer_number, layer_repetition, layer_parameters[layer_number], openeye_parameter, layer_es, dram, log_level))
                        elif("Conv" in str(layer_parameters[layer_number].layer_name)):
                            await cocotb.start_soon(rtl_test_utils.compare_stream_Conv(ptp, dut, layer_number, layer_repetition, layer_parameters[layer_number], openeye_parameter, layer_es, dram, log_level, output_order))
                        elif("Dense" in str(layer_parameters[layer_number].layer_name)):
                            await cocotb.start_soon(rtl_test_utils.compare_stream_Dense(ptp, dut, layer_number, layer_repetition, layer_parameters[layer_number], openeye_parameter, layer_es, dram, log_level))
                        elif("Pooling" in str(layer_parameters[layer_number].layer_name)):
                            await cocotb.start_soon(rtl_test_utils.compare_stream_Pooling(ptp, dut, layer_number, layer_repetition, layer_parameters[layer_number], openeye_parameter, layer_es, dram, log_level))
                        if(logging.DEBUG >= log_level):
                            assert gtu.check_results('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_ref.txt',\
                                                    'demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/output.txt')
                        assert ptu.compare_dram_with_ref(layer_parameters[layer_number], calculated_results, dram.fmap[1 + layer_number])
                    else :
                        await cocotb.start_soon(rtl_test_utils.await_ready_signal(ptp, dut))
                        time_printer.timestamp("Ready signal detected. Start new stream " , logger)
                        dram.fmap[1 + layer_number] = ptu.fill_dram_with_ref(calculated_results, dram.fmap[1 + layer_number], layer_parameters[layer_number])
                    if (layer_parameters[layer_number].layer_name != "Pooling") :
                        slo.batchnorm_output(layer_parameters[layer_number], 1, layer_number, dram)
                    if (layer_number != max_layers - 1) :
                        assert rtl_test_utils.compare_iact_storage(ptp, dut, dram.fmap[1 + layer_number], openeye_parameter)
                else :
                    dram.fmap[1 + layer_number] = ptu.fill_dram_with_ref(calculated_results, dram.fmap[1 + layer_number], layer_parameters[layer_number])
                    if (layer_parameters[layer_number].layer_name != "Pooling") :
                        slo.batchnorm_output(layer_parameters[layer_number], 1, layer_number, dram)

                
    if (only_files == 0) :
        assert dut.rst_ni.value == 1, "rst_ni is not 1!"