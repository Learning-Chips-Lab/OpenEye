# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import sys
import os
import time
directory = (os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import cocotb_parallel.parallel_test_utils as ptu
import test_utils.rtl_test_utils as rtl_test_utils
import test_utils.timing_parameters as tp
import test_utils.generic_test_utils as gtu
import test_utils.DRAM as DRAM
import test_utils.open_eye_parameters as oep
import test_utils.layer_parameters as lp
import test_utils.simple_layer_operations as slo
import test_utils.layer_execution_state as les
import test_utils.data_create as data_create
import test_utils.tflite2model as tflite2model
import test_utils.stream_dicts as str_dic

os.environ["CLOCK_LEN"] = "10"
os.environ["CLOCK_UNIT"] = "ns"
os.environ["CLOCK_DELAY_INPUT"] = "100"
os.environ["CLOCK_DELAY_UNIT_INPUT"] = "ps"

os.environ["CLOCK_DELAY_OUTPUT"] = "100"
os.environ["CLOCK_DELAY_UNIT_OUTPUT"] = "ps"

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

@cocotb.test()
async def single_layer_test(dut):
    """ Test the DUT with a given DNN model.

    This function tests the DUT with a given DNN model.
    """
    # Get variables that are used for the execution of the test
    try:
        layer_mode = os.getenv("LAYER")
    except:
        logger.error("LAYER not given.")
    try:
        filters = int((os.getenv("NUM_FILTERS")))
    except:
        logger.debug("NUM_FILTERS not set")
        filters = 1

    try:
        kernelsize = int((os.getenv("KERNEL_SIZE")))
    except:
        logger.debug("KERNEL_SIZE not set")

    try:
        inputsize = int((os.getenv("INPUT_SIZE")))
    except:
        logger.debug("INPUT_SIZE not set")

    try:
        outputsize = int((os.getenv("OUTPUT_SIZE")))
    except:
        outputsize = 1
        logger.debug("OUTPUT_SIZE not set")

    try:
        strides = (int((os.getenv("STRIDE"))),int((os.getenv("STRIDE"))))
    except:
        logger.debug("STRIDE not set")

    try:
        channels = int((os.getenv("INPUT_CHANNELS")))
    except:
        logger.debug("INPUT_CHANNELS not set")


    try:
        use_random = int((os.getenv("USE_RANDOM_VALUES")))
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

    time_last_check = time.time()
    time_currently = time.time()
    time_elapsed = time_currently - time_last_check

    ptp = tp.PortTimingParameters()
    ptp.initiate_params(clk_cycle, clk_cycle_unit, clk_delay_in, clk_delay_unit_in, clk_delay_out, clk_delay_unit_out)
    
    # Create a test model    
    gtu.select_gpu(1)

    #Here If-Condition test, wether use model or single Layer
    if(use_random):
        model = data_create.create_layer(layer_mode, filters, kernelsize, inputsize, strides, channels, outputsize)
    else:
        model = tflite2model.create_model_from_tflite(use_random)
    #load_model_function
    
    # Create the OpenEye parameters and the DRAM given the model
    dram = DRAM.DRAMContents(model)
    time_currently = time.time()
    time_elapsed = time_currently - time_last_check
    time_last_check = time.time()
    logger.debug("Initialize DRAM. " + str(time_elapsed))
    dram.write_initial_data_to_dram(model)

    openeye_parameter = oep.create_vh_file(serial)
    time_currently = time.time()
    time_elapsed = time_currently - time_last_check
    time_last_check = time.time()
    logger.debug("OpenEye parameters set. " + str(time_elapsed))

    # Start the clock
    clk = Clock(dut.clk_i, ptp.clk_cycle, units=ptp.clk_cycle_unit)
    cocotb.start_soon(clk.start())
    dut._log.info("Clock is %s " + ptp.clk_cycle_unit, ptp.clk_cycle)
    # reset the DUT
    await cocotb.start_soon(rtl_test_utils.reset_all_signals(ptp, dut, openeye_parameter.SERIAL))

    # Process the layers of the model one after another
    for layer_number, layer in enumerate(model.layers):

        # TODO: After refactoring LayerParameters, it is nicer to use the constructor 
        # layer_parameters = ptu.LayerParameters(model.layers[layer_number], openeye_parameter)
        if("Pooling" in str(layer)):
            slo.pool(dram, layer, layer_number)
        elif("Flat" in str(layer)):
            slo.flat(dram, layer, layer_number)
        else:
            layer_parameters = lp.LayerParameters(layer, openeye_parameter)
            time_currently = time.time()
            time_elapsed = time_currently - time_last_check
            time_last_check = time.time()
            logger.info("Layer parameters created. " + str(time_elapsed))
            calculated_results = ptu.collect_results(layer, layer_number, layer_parameters, dram)
            if(logging.DEBUG >= log_level):
                ptu.make_ref(openeye_parameter, layer_parameters, layer, layer_number, dram, calculated_results)
                time_currently = time.time()
                time_elapsed = time_currently - time_last_check
                time_last_check = time.time()
                logger.info("Reference data created. " + str(time_elapsed))

            dram_layer_content = [dram.fmap[layer_number], dram.weights[layer_number], dram.bias[layer_number]]
            time_currently = time.time()
            time_elapsed = time_currently - time_last_check
            time_last_check = time.time()
            logger.info("Start creating stream. " + str(time_elapsed))
            stream = ptu.write_stream(openeye_parameter, layer_parameters, layer, dram_layer_content)
            
            time_currently = time.time()
            time_elapsed = time_currently - time_last_check
            time_last_check = time.time()
            logger.info("Streams set. " + str(time_elapsed))

            for layer_repetition in range(layer_parameters.needed_total_transmissions):
                
                logger.info("Send stream.")
                await cocotb.start_soon(rtl_test_utils.send_stream(ptp, dut, stream[layer_repetition], openeye_parameter, layer_parameters))
                logger.info("Stream is sent.")
                if("Depthwise" in str(layer)):
                    await cocotb.start_soon(rtl_test_utils.await_and_compare_stream_Dw(ptp, dut, layer_number, model, layer_repetition, layer_parameters, openeye_parameter, layer_es, dram, log_level))
                elif("Conv" in str(layer)):
                    await cocotb.start_soon(rtl_test_utils.await_and_compare_stream_Conv(ptp, dut, layer_number, model, layer_repetition, layer_parameters, openeye_parameter, layer_es, dram, log_level))
                elif("Dense" in str(layer)):
                    await cocotb.start_soon(rtl_test_utils.await_and_compare_stream_Dense(ptp, dut, layer_number, model, layer_repetition, layer_parameters, openeye_parameter, layer_es, dram, log_level))
                if(logging.DEBUG >= log_level):
                    assert gtu.check_results('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_ref.txt',\
                                            'demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/output.txt')
                    
            assert ptu.compare_dram_with_ref(layer, calculated_results, dram.fmap[1 + layer_number])

        slo.batchnorm_output(layer, 512, layer_number, dram)

    assert dut.rst_ni.value == 1, "rst_ni is not 1!"