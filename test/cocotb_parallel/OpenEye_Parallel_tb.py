# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import sys
import os
directory = (os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
sys.path.extend([directory, os.path.dirname(os.path.realpath(__file__))])
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import test.test_utils.test_utils_main as ptu
import open_eye.rtl_test_utils as rtl_test_utils
import open_eye.timing_parameters as tp
import open_eye.generic_test_utils as gtu
import open_eye.DRAM as DRAM
import open_eye.time_stamper as time_stamper
import open_eye.open_eye_parameters as oep
import open_eye.layer_parameters as lp
import open_eye.simple_layer_operations as slo
import open_eye.layer_execution_state as les
import open_eye.data_create as data_create
import open_eye.tflite2model as tflite2model
import open_eye.stream_dicts as strdic
from cocotb.triggers import Timer

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
global status_thread, iact_thread, wght_thread, psum_thread
status_thread = []
iact_thread = []
wght_thread = []
psum_thread = []

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
    except:
        use_random = 1
        logger.debug("USE_RANDOM_VALUES set to one")

    try:
        sparse_iacts = int((os.getenv("USE_SPARSE_IACTS")))
    except:
        sparse_iacts = 0
        logger.debug("No sparsety for wghts set")
    try:
        sparse_wghts = int((os.getenv("USE_SPARSE_WEIGHTS")))
    except:
        sparse_wghts = 0
        logger.debug("No sparsety for wghts set")
    
    layer_es = les.LayerExecutionState()
    serial = False
    clk_cycle = int(os.environ["CLOCK_LEN"])
    clk_cycle_unit = os.environ["CLOCK_UNIT"]

    clk_delay_in = int(os.environ["CLOCK_DELAY_INPUT"])
    clk_delay_unit_in = os.environ["CLOCK_DELAY_UNIT_INPUT"]

    clk_delay_out = int(os.environ["CLOCK_DELAY_OUTPUT"])
    clk_delay_unit_out = os.environ["CLOCK_DELAY_UNIT_OUTPUT"]

    time_printer = time_stamper.time_stamper()

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
    time_printer.timestamp("Initialized DRAM. ", logger)
    dram.write_initial_data_to_dram(model, sparse_iacts, sparse_wghts)
    time_printer.timestamp("DRAM Initialized. ", logger)

    openeye_parameter = oep.get_oep(serial)
    time_printer.timestamp("OpenEye parameters set. ", logger)

    # Start the clock
    clk = Clock(dut.clk_i, ptp.clk_cycle, units=ptp.clk_cycle_unit)
    cocotb.start_soon(clk.start())
    dut._log.info("Clock is %s " + ptp.clk_cycle_unit, ptp.clk_cycle)
    # reset the DUT
    await cocotb.start_soon(rtl_test_utils.reset_all_signals(ptp, dut, openeye_parameter.SERIAL))
    time_printer.timestamp("All signals resetted. ", logger)

    # Process the layers of the model one after another
    max_layers = len(model.layers)
    for layer_number, layer in enumerate(model.layers):
        if("Pooling" in str(layer)):
            slo.pool(dram, layer, layer_number)
        elif("Flat" in str(layer)):
            slo.flat(dram, layer, layer_number)
        else:
            layer_parameters = lp.LayerParameters(layer, openeye_parameter, layer_number, max_layers)
            time_printer.timestamp("Layer parameters created. ", logger)
            calculated_results = ptu.collect_results(layer, layer_number, layer_parameters, dram, openeye_parameter.SERIAL)
            output_order = ptu.make_ref(openeye_parameter, layer_parameters, layer, layer_number, dram, calculated_results)
            if(logging.DEBUG >= log_level):
                time_printer.timestamp("Reference data created. ", logger)

            dram_layer_content = [dram.fmap[layer_number], dram.weights[layer_number], dram.bias[layer_number]]
            time_printer.timestamp("Start creating stream. ", logger)
            stream = ptu.write_stream(openeye_parameter, layer_parameters, layer, dram_layer_content, sparse_iacts, sparse_wghts)
            
            time_printer.timestamp("Streams set. ", logger)
            for layer_repetition in range(layer_parameters.needed_total_transmissions):
                layer_thread = calculate_layer(ptp, dut, stream, openeye_parameter, layer_parameters, layer_repetition, model, layer_es, dram, log_level, layer_number, layer, output_order)
                await layer_thread
                if(logging.DEBUG >= log_level):
                    assert gtu.check_results('demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/dma_stream_ref.txt',\
                                'demo/layer_' + str(layer_number) + '_' + str(layer_repetition) + '/output.txt')
            assert ptu.compare_dram_with_ref(layer, calculated_results, dram.fmap[1 + layer_number])
        slo.batchnorm_output(layer, 512, layer_number, dram)

    assert dut.rst_ni.value == 1, "rst_ni is not 1!"

async def calculate_layer(ptp, dut, stream, oep, lp, layer_repetition, model, layer_es, dram, log_level, layer_number, layer, output_order):
    global status_thread, iact_thread, wght_thread, psum_thread
    logger.info("Send stream.")
    status_thread = cocotb.start_soon(rtl_test_utils.send_stream(ptp, dut, stream[layer_repetition], oep, lp, layer_repetition))
    await status_thread
    if(stream[layer_repetition][strdic.stream_parallel_dict["status"]][strdic.status_dict["skipPsum"]] != 1):
        psum_thread = cocotb.start_soon(rtl_test_utils.write_bias(ptp, dut, stream[layer_repetition][strdic.stream_parallel_dict["psum"]], oep, lp))
    if (layer_repetition != 0) :
        await iact_thread
        await wght_thread
    # start the transmission of the data
    else :
        iact_thread = cocotb.start_soon(rtl_test_utils.write_iact(ptp, dut, stream[layer_repetition][strdic.stream_parallel_dict["iact"]], oep, lp))
        wght_thread = cocotb.start_soon(rtl_test_utils.write_wght(ptp, dut, stream[layer_repetition][strdic.stream_parallel_dict["wght"]], oep, lp))
    # wait until all transmission are finished
    await iact_thread
    await wght_thread
    if(stream[layer_repetition][strdic.stream_parallel_dict["status"]][strdic.status_dict["skipPsum"]] != 1):
        await psum_thread
    logger.info("Stream is sent.")
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.compute_i), 1))
    await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    cocotb.start_soon(rtl_test_utils.set_input(ptp,(dut.compute_i), 0))
    await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    await Timer(ptp.clk_cycle, units=ptp.clk_cycle_unit)
    if (layer_repetition != (lp.needed_total_transmissions-1)) :
        wght_thread = cocotb.start_soon(rtl_test_utils.write_wght(ptp, dut, stream[layer_repetition + 1][strdic.stream_parallel_dict["wght"]], oep, lp))
        iact_thread = cocotb.start_soon(rtl_test_utils.write_iact(ptp, dut, stream[layer_repetition + 1][strdic.stream_parallel_dict["iact"]], oep, lp))
    await cocotb.start_soon(rtl_test_utils.await_ready_signal(ptp, dut, layer_number, model, layer_repetition, lp, oep, layer_es, dram, log_level, stream[layer_repetition]))
    
    
    #if (layer_repetition != (lp.needed_total_transmissions-1)) :
    #    iact_thread = cocotb.start_soon(rtl_test_utils.write_iact(ptp, dut, stream[layer_repetition + 1][strdic.stream_parallel_dict["iact"]], oep, lp))
    if("Depthwise" in str(layer)):
        await cocotb.start_soon(rtl_test_utils.compare_stream_Dw(ptp, dut, layer_number, model, layer_repetition, lp, oep, layer_es, dram, log_level, stream[layer_repetition], output_order))
    elif("Conv" in str(layer)):
        await cocotb.start_soon(rtl_test_utils.compare_stream_Conv(ptp, dut, layer_number, model, layer_repetition, lp, oep, layer_es, dram, log_level, stream[layer_repetition], output_order))
    elif("Dense" in str(layer)):
        await cocotb.start_soon(rtl_test_utils.compare_stream_Dense(ptp, dut, layer_number, model, layer_repetition, lp, oep, layer_es, dram, log_level, stream[layer_repetition]))