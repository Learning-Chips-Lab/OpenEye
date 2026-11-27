# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

import copy
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cocotb
import numpy as np
import tensorflow as tf
from cocotb.clock import Clock
from cocotb.triggers import SimTimeoutError, Timer, ValueChange, with_timeout
from cocotb.utils import get_sim_time

import open_eye.DRAM as DRAM
import open_eye.data_create as data_create
import open_eye.generic_test_utils as gtu
import open_eye.layer_execution_state as les
import open_eye.layer_parameters as lp
import open_eye.onnx2model as o2m
import open_eye.open_eye_parameters as oep
import open_eye.rtl_test_utils as rtl_test_utils
import open_eye.simple_layer_operations as slo
import open_eye.test_utils_main as tum
import open_eye.time_stamper as time_stamper
import open_eye.timing_parameters as tp


class CustomSimTimeFormatter(logging.Formatter):
    """Custom logging formatter that prepends simulation time in nanoseconds."""

    def format(self, record):
        try:
            sim_time = get_sim_time("ns")
            sim_time_str = f"{sim_time:10.2f}ns"
        except Exception:
            sim_time_str = "  -.--ns"

        msg = super().format(record)
        return f"{sim_time_str} {msg}"


os.environ.setdefault("CLOCK_LEN", "10")
os.environ.setdefault("CLOCK_UNIT", "ns")
os.environ.setdefault("CLOCK_DELAY_INPUT", "100")
os.environ.setdefault("CLOCK_DELAY_UNIT_INPUT", "ps")
os.environ.setdefault("CLOCK_DELAY_OUTPUT", "100")
os.environ.setdefault("CLOCK_DELAY_UNIT_OUTPUT", "ps")

time_printer = time_stamper.time_stamper()

tests_dir = os.path.abspath(os.path.dirname(__file__))
hdl_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, "hdl"))
_raw_log_level = os.getenv("LOGGER_LEVEL")
log_level = int(_raw_log_level) if _raw_log_level else logging.INFO
logger = logging.getLogger("cocotb")


# ==============================================================================
# Data Structures for Decoupled Precomputation & Simulation
# ==============================================================================


@dataclass
class LayerArtifact:
    """Precomputed reference data, streams, and routing metadata for a single layer."""

    layer_number: int
    layer_name: str
    lpar: Any
    is_output_layer: bool
    next_layers: list[str]
    target_slots: list[tuple[str, int]] = field(default_factory=list)
    ref_results: Any = None
    quantized_results: Any = None
    output_order: Any = None
    streams: list[Any] = field(default_factory=list)


@dataclass
class ModelExecutionPlan:
    """Complete precomputed execution bundle for an entire DNN model."""

    openeye_parameter: Any
    layer_parameters: dict[str, Any]
    rel_layers: dict[str, dict[str, list[str]]]
    dram: DRAM.DRAMContents
    artifacts: list[LayerArtifact]


# ==============================================================================
# Environment, Model & Logging Setup
# ==============================================================================


async def setup_file_logging():
    """Configure optional file logging with simulation timestamps."""
    global logger
    log_path = os.getenv("COCOTB_LOG_FILE_PATH")
    if not log_path:
        return

    log_path = os.path.abspath(log_path)
    logger = logging.getLogger("cocotb")

    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)

    fh = logging.FileHandler(log_path, mode="w")
    raw_level = os.getenv("LOGGER_LEVEL")
    level = int(raw_level) if raw_level else logging.INFO
    fh.setLevel(level)
    fh.setFormatter(CustomSimTimeFormatter("%(levelname)-8s %(name)-20s %(message)s"))

    logger.addHandler(fh)
    logger.info("File logging started at sim time %s ns", cocotb.utils.get_sim_time("ns"))


def envvars_to_vars():
    """Load test configuration variables from environment variables."""
    only_files = gtu.load_env_to_variable("ONLY_FILES", 0)
    layer_mode = gtu.load_env_to_variable("LAYER", "Convolution")
    filters = gtu.load_env_to_variable("NUM_FILTERS", 4)
    kernelsize_x = gtu.load_env_to_variable("KERNEL_SIZE_X", 3)
    kernelsize_y = gtu.load_env_to_variable("KERNEL_SIZE_Y", 3)
    inputsize_x = gtu.load_env_to_variable("INPUT_SIZE_X", 64)
    inputsize_y = gtu.load_env_to_variable("INPUT_SIZE_Y", 1)
    outputsize = gtu.load_env_to_variable("OUTPUT_SIZE", 1)
    stride_val = gtu.load_env_to_variable("STRIDE", 1)
    strides = (stride_val, stride_val)
    channels = gtu.load_env_to_variable("INPUT_CHANNELS", 4)
    sparse_iacts = gtu.load_env_to_variable("USE_SPARSE_IACTS", 0)
    sparse_wghts = gtu.load_env_to_variable(
        "USE_SPARSE_WGHTS", gtu.load_env_to_variable("USE_SPARSE_WEIGHTS", 0)
    )
    return (
        only_files,
        layer_mode,
        filters,
        kernelsize_x,
        kernelsize_y,
        inputsize_x,
        inputsize_y,
        outputsize,
        strides,
        channels,
        sparse_iacts,
        sparse_wghts,
    )


def _create_port_timing_params() -> tp.PortTimingParameters:
    """Initialize PortTimingParameters from clock environment variables."""
    ptp = tp.PortTimingParameters()
    ptp.initiate_params(
        int(os.environ["CLOCK_LEN"]),
        os.environ["CLOCK_UNIT"],
        int(os.environ["CLOCK_DELAY_INPUT"]),
        os.environ["CLOCK_DELAY_UNIT_INPUT"],
        int(os.environ["CLOCK_DELAY_OUTPUT"]),
        os.environ["CLOCK_DELAY_UNIT_OUTPUT"],
    )
    return ptp


def truncate_model(model):
    """Strip host-only reshape/recurrent wrapper layers not mapped to OpenEye hardware."""
    layers = model.layers if hasattr(model, "layers") else model
    return [
        layer
        for layer in layers
        if not isinstance(layer, (tf.keras.layers.Bidirectional, tf.keras.layers.Flatten))
    ]


def build_layer_topology(model):
    """Build directed graph maps ('Previous_Layer', 'Next_Layer') across hardware-mapped layers."""
    previous_layers = {layer.name: [] for layer in model.layers}
    next_layers = {layer.name: [] for layer in model.layers}

    for layer in model.layers:
        inputs = layer.input if isinstance(layer.input, list) else [layer.input]
        for tensor in inputs:
            if hasattr(tensor, "_keras_history"):
                prev_layer = tensor._keras_history[0]
                next_layers.setdefault(prev_layer.name, [])
                previous_layers.setdefault(prev_layer.name, [])
                previous_layers[layer.name].append(prev_layer.name)
                next_layers[prev_layer.name].append(layer.name)

    # Flatten is a pure reshape that never reaches the hardware flow: connect its neighbours directly.
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Flatten):
            continue
        name = layer.name
        prevs = previous_layers.pop(name, [])
        nexts = next_layers.pop(name, [])
        for prev in prevs:
            next_layers[prev] = [n for n in next_layers[prev] if n != name] + nexts
        for nxt in nexts:
            previous_layers[nxt] = [q for q in previous_layers[nxt] if q != name] + prevs

    return {"Previous_Layer": previous_layers, "Next_Layer": next_layers}


def build_edge_to_slot_map(rel_layers):
    edge_to_slot = {}
    for target_layer, prev_layers in rel_layers["Previous_Layer"].items():
        for slot_idx, src_layer in enumerate(prev_layers):
            edge_to_slot[(src_layer, target_layer)] = slot_idx
    return edge_to_slot


def load_benchmark_model(
    layer_mode,
    filters,
    kernelsize_x,
    kernelsize_y,
    inputsize_x,
    inputsize_y,
    strides,
    channels,
    outputsize,
):
    """Build or load the Keras model according to environment settings."""
    try:
        use_random = int(os.environ["USE_RANDOM_VALUES"])
    except Exception:
        use_random = 1
        logger.debug("USE_RANDOM_VALUES set to 1")

    gtu.select_gpu(0)

    if use_random:
        model = data_create.create_net(
            layer_mode,
            filters,
            kernelsize_x,
            kernelsize_y,
            inputsize_x,
            inputsize_y,
            strides,
            channels,
            outputsize,
        )
    else:
        current_dir = Path(__file__).resolve().parent
        target_dir = current_dir.parent.parent / "models"
        if str(target_dir) not in sys.path:
            sys.path.append(str(target_dir))
        from mnist_conv_net import SimpleMNISTConvNet

        model = SimpleMNISTConvNet()
        model = o2m.get_model(model)

    rel_layers = build_layer_topology(model)
    trunc_model = truncate_model(model)

    if os.environ.get("OPENEYE_EXPECT_LAYERS"):
        expected_layers = int(os.environ["OPENEYE_EXPECT_LAYERS"])
        assert len(trunc_model) == expected_layers, (
            f"{layer_mode}: expected {expected_layers} layers, got {len(trunc_model)}"
        )

    return trunc_model, rel_layers


def _is_standard_conv(name):
    return "conv" in str(name) and "depthwise" not in str(name)


# ==============================================================================
# Debug & Logging Helpers
# ==============================================================================


FSM_STATES = {
    0: "IDLE",
    1: "GET_PARAMETERS",
    2: "GET_ROUTER_CONFIG",
    3: "GET_IACT",
    4: "GET_WGHT",
    5: "GET_BIAS",
    6: "GET_QUANTIZE",
    7: "START_CONVERTER",
    8: "CONVERT_IACT",
    9: "WAIT_CYCLE",
    10: "WAIT_FOR_RESULTS",
    11: "RECEIVE_PSUMS_TO_IACT",
    12: "MAXPOOLING_READ",
    13: "MAXPOOLING_SEND",
    14: "MAXPOOLING_WAIT",
}


def decode_state(signal_val) -> str:
    if not signal_val.is_resolvable:
        return f"UNRESOLVED ({signal_val.binstr})"
    val_int = int(signal_val)
    return FSM_STATES.get(val_int, f"UNKNOWN ({val_int})")


async def monitor_fsm(dut, signal, name="FSM"):
    dut._log.info("[%s] Start-State: %s", name, decode_state(signal.value))
    while True:
        await ValueChange(signal)
        dut._log.info("[%s] -> %s", name, decode_state(signal.value))


def _start_debug_monitors(ptp, dut, openeye_parameter):
    """Spawn optional RTL signal/buffer monitors controlled by environment variables."""
    env = os.environ

    if env.get("OPENEYE_PROBE_FSM"):
        cocotb.start_soon(rtl_test_utils.probe_fsm_states(ptp, dut, oep=openeye_parameter))
        cocotb.start_soon(rtl_test_utils.monitor_iact_handoff(ptp, dut))
    elif env.get("DUMP_PSUM_BUFFERS"):
        cocotb.start_soon(rtl_test_utils.monitor_iact_handoff(ptp, dut))

    if env.get("TRACE_CONVERTER_GATE"):
        cocotb.start_soon(rtl_test_utils.trace_converter_gate(ptp, dut, openeye_parameter, col=0, row=0))
        cocotb.start_soon(rtl_test_utils.trace_converter_gate(ptp, dut, openeye_parameter, col=0, row=1))

    if env.get("DUMP_IACT_BUFFER"):
        async def _dump_buf():
            for _ in range(400):
                await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
            rtl_test_utils.dump_iact_buffer_values(dut, openeye_parameter)

        cocotb.start_soon(_dump_buf())

    if env.get("DUMP_CLUSTER_IACT"):
        cocotb.start_soon(rtl_test_utils.monitor_cluster_iact(ptp, dut, openeye_parameter))
        cocotb.start_soon(rtl_test_utils.monitor_pe_iact(ptp, dut, openeye_parameter))
        cocotb.start_soon(rtl_test_utils.monitor_converter_store(ptp, dut, openeye_parameter))

    if env.get("TRACE_PSUM_CAPTURE"):
        cocotb.start_soon(rtl_test_utils.trace_psum_capture(ptp, dut, openeye_parameter))
        cocotb.start_soon(rtl_test_utils.trace_bias_load(ptp, dut, openeye_parameter))

    if env.get("TRACE_POOLING"):
        cocotb.start_soon(rtl_test_utils.trace_pooling(ptp, dut, max_lines=4000))

    if env.get("TRACE_ALL_PE"):
        cocotb.start_soon(rtl_test_utils.trace_all_pe_states(ptp, dut, openeye_parameter, max_lines=4000))

    simple_monitors = (
        ("TRACE_CONV_WRITEBACK", rtl_test_utils.trace_conv_writeback),
        ("TRACE_RAM_WRITES", rtl_test_utils.trace_ram_writes),
        ("TRACE_WB_EVENTS", rtl_test_utils.trace_wb_events),
    )
    for flag, fn in simple_monitors:
        if env.get(flag):
            cocotb.start_soon(fn(ptp, dut))

    oep_monitors = (
        ("TRACE_CONVERT_WINDOW", rtl_test_utils.trace_convert_window),
        ("DUMP_CONVERT_LIMITS", rtl_test_utils.dump_convert_iact_limits),
        ("TRACE_IACT_LANES", rtl_test_utils.trace_iact_lanes),
        ("TRACE_PSUM_SLICES", rtl_test_utils.trace_psum_capture_slices),
        ("TRACE_PE_CALC", rtl_test_utils.trace_pe_calc_loop),
        ("PROBE_PSUM_STREAM", rtl_test_utils.probe_psum_stream),
        ("TRACE_CONVERTER", rtl_test_utils.trace_converter),
        ("TRACE_PE_SPADS", rtl_test_utils.trace_pe_spads),
        ("TRACE_PE_WADDR", rtl_test_utils.trace_pe_wght_addrs),
        ("TRACE_PE_WGHT", rtl_test_utils.trace_pe_wght_writes),
        ("TRACE_PE_PASSES", rtl_test_utils.trace_pe_pass_counts),
        ("TRACE_PE_IACT", rtl_test_utils.trace_pe_iact),
        ("TRACE_PE_PSUM", rtl_test_utils.trace_pe_psum),
    )
    for flag, fn in oep_monitors:
        if env.get(flag):
            cocotb.start_soon(fn(ptp, dut, openeye_parameter))


def _log_layer_summaries(layer_parameters):
    """Log compact per-layer parameter summaries and optional full PAGU parameter dumps."""
    if os.environ.get("DUMP_LAYER_PARAMS"):
        extra_keys = {
            "iact_x_lines",
            "iact_x_line_repetitions",
            "iact_words_per_compute",
            "needed_iact_buffer_words",
            "needed_Iact_writes",
            "channel_div_trans",
            "used_channels",
            "diff_iact_layer",
            "needed_wght_cycles",
            "iact_repetitions_per_write",
            "iact_converter_max_cycles",
            "buffer_cycles_for_x_iact",
            "iact_converter_buffer_addr_max_cycles",
            "lower_bound",
            "upper_bound",
            "used_Y_cluster",
            "y_lines_per_calculation",
            "different_kernels_per_calculation",
            "padding_y",
            "padding_x",
            "strideY",
            "used_X_cluster",
            "iact_x_add_up",
            "psum_size_y",
            "psum_size_x",
            "filters",
            "fsm_psum_limit",
            "used_psum_per_PE",
            "iact_size_x",
            "iact_size_y",
            "diff_iact_layer_next_layer",
            "psum_storage_cycles",
        }
        for n, lpar in enumerate(layer_parameters.values()):
            fields = sorted(
                k
                for k in vars(lpar)
                if k.startswith(("iact_read_", "iact_write_", "psum_pagu_")) or k in extra_keys
            )
            logger.info("layerparams %d: %s", n, " ".join(f"{k}={getattr(lpar, k)}" for k in fields))

    for lpar in layer_parameters.values():
        logger.info(
            "layer %s: in=%sx%s ch=%s filters=%s send_values_out=%s "
            "store_in_psum=%s transmissions=%s output_cycles=%s psum_output_words=%s "
            "used_Y_cluster=%s used_X_cluster=%s",
            lpar.layer_name,
            getattr(lpar, "iact_size_x", "?"),
            getattr(lpar, "iact_size_y", "?"),
            getattr(lpar, "channels", "?"),
            getattr(lpar, "filters", "?"),
            lpar.send_values_out,
            getattr(lpar, "store_in_psum", "?"),
            getattr(lpar, "needed_total_transmissions", "?"),
            getattr(lpar, "output_cycles", "?"),
            getattr(lpar, "psum_output_words", "?"),
            getattr(lpar, "used_Y_cluster", "?"),
            getattr(lpar, "used_X_cluster", "?"),
        )


def _apply_debug_dram_overrides(dram: DRAM.DRAMContents, model, layer_parameters):
    """Apply vectorized debug overrides to NumPy-backed DRAM buffers when requested via env vars."""
    env = os.environ
    first_layer = model[0].name

    if env.get("OPENEYE_ZERO_IACTS"):
        dram.fmap[first_layer][0].fill(0)
        logger.info("OPENEYE_ZERO_IACTS: layer-0 input zeroed")

    if env.get("OPENEYE_CONST_IACTS"):
        const_iact = int(env["OPENEYE_CONST_IACTS"])
        dram.fmap[first_layer][0].fill(const_iact)
        logger.info("OPENEYE_CONST_IACTS: layer-0 input set to %d", const_iact)

    if env.get("OPENEYE_CONST_WGHTS"):
        const_wght = int(env["OPENEYE_CONST_WGHTS"])
        for w in dram.weights.values():
            w.fill(const_wght)
        logger.info("OPENEYE_CONST_WGHTS: all weights set to %d", const_wght)

    if env.get("OPENEYE_ASYMMETRIC_KERNEL"):
        for params in layer_parameters.values():
            if not _is_standard_conv(params.layer_name):
                continue
            w = dram.weights[params.layer_name]
            ky, kx = w.shape[2], w.shape[3]
            pattern = (1 + np.arange(kx)[None, :] + kx * np.arange(ky)[:, None]).astype(DRAM.DRAM_DTYPE)
            w[:] = pattern[None, None, :, :]
        logger.info("OPENEYE_ASYMMETRIC_KERNEL: weight[y][x] = 1 + x + kernel_width*y")

    if env.get("OPENEYE_MIX_WEIGHTS"):
        for params in layer_parameters.values():
            if not _is_standard_conv(params.layer_name):
                continue
            w = dram.weights[params.layer_name]
            c_dim, f_dim = w.shape[0], w.shape[1]
            pattern = (1 + np.arange(c_dim)[:, None] + 5 * np.arange(f_dim)[None, :]).astype(DRAM.DRAM_DTYPE)
            w[:] = pattern[:, :, None, None]
        logger.info("OPENEYE_MIX_WEIGHTS: weight[c][f] = 1 + c + 5*f")

    if env.get("OPENEYE_RAMP_IACTS"):
        arr = dram.fmap[first_layer][0]
        idx = np.arange(arr.size, dtype=DRAM.DRAM_DTYPE)
        if env.get("OPENEYE_RAMP_MULT"):
            mult = int(env["OPENEYE_RAMP_MULT"])
            vals = ((idx * mult) % 61) + 1
        else:
            vals = (idx % 100) + 1
        arr[:] = vals.reshape(arr.shape)
        logger.info("OPENEYE_RAMP_IACTS: layer-0 input set to 1..N in flat order")

    if env.get("OPENEYE_BIAS_STEP"):
        step = int(env["OPENEYE_BIAS_STEP"])
        for name, bias in dram.bias.items():
            if bias.size > 0:
                dram.bias[name] = 1 + step * np.arange(bias.size, dtype=DRAM.DRAM_DTYPE)
        logger.info("OPENEYE_BIAS_STEP: bias[f] = 1 + %d*f", step)

    if env.get("DUMP_INPUT_IACTS"):
        try:
            flat = np.asarray(dram.fmap[first_layer][0]).ravel()
            logger.info("layer-0 input activations (first 34): %s", flat[:34].tolist())
        except Exception as exc:
            logger.info("could not flatten layer-0 input (%s)", type(exc).__name__)


# ==============================================================================
# PHASE A: Pure-Python / NumPy Precomputation of All References & Streams
# ==============================================================================


def precompute_model_artifacts(
    model,
    rel_layers,
    sparse_iacts: int,
    sparse_wghts: int,
    serial: int = 1,
    output_dir: str = "demo/",
    write_files: bool = True,
) -> ModelExecutionPlan:
    """
    Generate all layer parameters, NumPy DRAM contents, golden reference outputs,
    and DMA bitstreams upfront before starting any RTL simulation.
    """
    openeye_parameter = oep.get_oep(serial)
    time_printer.timestamp("OpenEye parameters set.", logger)

    if write_files:
        os.makedirs(output_dir, exist_ok=True)
        gtu.delete_files_in_directory(output_dir)

    # 1. Reverse topological pass: propagate downstream channel requirements upstream
    layer_parameters = {}
    for layer_number, layer in reversed(list(enumerate(model))):
        temp_params = lp.LayerParameters(
            layer_parameters, layer, openeye_parameter, rel_layers, layer_number
        )
        layer_parameters[temp_params.layer_name] = temp_params

    _log_layer_summaries(layer_parameters)

    # 2. Initialize NumPy-backed DRAM and populate initial activations/weights/biases
    seed = int(os.environ["OPENEYE_SEED"]) if os.environ.get("OPENEYE_SEED") else None
    dram = DRAM.DRAMContents(model, layer_parameters, rel_layers, seed=seed)
    time_printer.timestamp("Initialized DRAM.", logger)
    dram.write_initial_data_to_dram(
        model, layer_parameters, sparse_iacts, sparse_wghts, rel_layers, seed=seed
    )
    _apply_debug_dram_overrides(dram, model, layer_parameters)

    edge_to_slot = build_edge_to_slot_map(rel_layers)

    # Fan-out initial input activations into branch slots if needed
    if model and "input" in model[0].name:
        input_name = model[0].name
        for output_layer in rel_layers["Next_Layer"][input_name]:
            target_slot = edge_to_slot[(input_name, output_layer)]
            if 0 < target_slot < len(dram.fmap[output_layer]):
                dram.fmap[output_layer][target_slot] = tum.fill_dram_with_ref(
                    dram.fmap[input_name][0],
                    dram.fmap[output_layer][target_slot],
                    layer_parameters[input_name],
                    layer_parameters[output_layer],
                )

    # 3. Forward topological pass: compute golden references & DMA streams for ALL layers
    artifacts: list[LayerArtifact] = []
    for layer_number, layer in enumerate(model):
        layer_name = layer.name
        lpar = layer_parameters[layer_name]
        next_layers = rel_layers["Next_Layer"][layer_name]
        is_output_layer = len(next_layers) == 0

        calculated_results = tum.collect_results(lpar, dram, openeye_parameter.SERIAL)

        output_order = None
        streams = []
        if lpar.use_layer_on_hw:
            dram_layer_content = [
                dram.fmap[layer_name],
                dram.weights[layer_name],
                dram.bias[layer_name],
            ]
            output_order = tum.make_ref(
                openeye_parameter, lpar, layer_name, dram, calculated_results
            )
            streams = tum.write_stream(
                openeye_parameter, lpar, dram_layer_content, 0, 0
            )
            if write_files:
                for layer_repetition in range(lpar.needed_total_transmissions):
                    gtu.create_stream_file(streams[layer_repetition], lpar, layer_repetition)

        # Keep an immutable snapshot of pre-quantized reference values for output comparison
        ref_snapshot = copy.deepcopy(calculated_results)

        quantized_results = calculated_results
        target_slots: list[tuple[str, int]] = []
        if ("input" not in layer_name) and not is_output_layer:
            if (
                "pooling" not in layer_name
                and "add" not in layer_name
                and "concat" not in layer_name
            ):
                quantized_results = slo.quantize_dict(calculated_results, lpar)

            for output_layer in next_layers:
                target_slot = edge_to_slot[(layer_name, output_layer)]
                target_slots.append((output_layer, target_slot))
                dram.fmap[output_layer][target_slot] = tum.fill_dram_with_ref(
                    quantized_results,
                    dram.fmap[output_layer][target_slot],
                    lpar,
                    layer_parameters[output_layer],
                )

        artifacts.append(
            LayerArtifact(
                layer_number=layer_number,
                layer_name=layer_name,
                lpar=lpar,
                is_output_layer=is_output_layer,
                next_layers=list(next_layers),
                target_slots=target_slots,
                ref_results=ref_snapshot,
                quantized_results=quantized_results,
                output_order=output_order,
                streams=streams,
            )
        )

    time_printer.timestamp(
        f"Precomputed all {len(artifacts)} layer references and streams.", logger
    )
    return ModelExecutionPlan(
        openeye_parameter=openeye_parameter,
        layer_parameters=layer_parameters,
        rel_layers=rel_layers,
        dram=dram,
        artifacts=artifacts,
    )


# ==============================================================================
# PHASE B: RTL Simulation Execution Using Precomputed Plan
# ==============================================================================


async def run_rtl_simulation(dut, plan: ModelExecutionPlan, layer_es, ptp):
    """Stream precomputed DMA packets into the DUT and verify outputs/intermediate buffers."""
    openeye_parameter = plan.openeye_parameter
    dram = plan.dram
    rel_layers = plan.rel_layers

    clk = Clock(dut.clk_i, ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    cocotb.start_soon(clk.start())
    dut._log.info("Clock is %s %s", ptp.clk_cycle, ptp.clk_cycle_unit)

    await cocotb.start_soon(
        rtl_test_utils.reset_all_signals(ptp, dut, openeye_parameter.SERIAL)
    )
    cocotb.start_soon(monitor_fsm(dut, dut.fsm_current_state))
    _start_debug_monitors(ptp, dut, openeye_parameter)

    for art in plan.artifacts:
        lpar = art.lpar
        layer_name = art.layer_name
        layer_number = art.layer_number
        l_type = str(lpar.layer_name)

        for layer_repetition in range(lpar.needed_total_transmissions):
            if lpar.use_layer_on_hw:
                logger.info(
                    "Send stream No. %d with the name: %s", layer_number + 1, layer_name
                )
                fc_write_check = None
                if (
                    os.environ.get("OPENEYE_CHECK_FC_WRITES")
                    and not os.environ.get("OPENEYE_SKIP_FC_WRITE_CHECK")
                    and "dense" in l_type
                ):
                    fc_write_check = cocotb.start_soon(
                        rtl_test_utils.check_fc_activation_writes(
                            dut, openeye_parameter, lpar, dram.fmap[layer_name]
                        )
                    )

                await cocotb.start_soon(
                    rtl_test_utils.send_stream(
                        ptp,
                        dut,
                        art.streams[layer_repetition],
                        openeye_parameter,
                        lpar,
                        layer_repetition,
                    )
                )
                logger.info("Stream is sent.")

                if art.is_output_layer:
                    await cocotb.start_soon(rtl_test_utils.await_enable_signal(ptp, dut))
                    if fc_write_check is not None:
                        assert fc_write_check.done(), "FC activation RAM writes incomplete"
                        await fc_write_check

                    if "depthwise" in l_type:
                        await cocotb.start_soon(
                            rtl_test_utils.compare_stream_Dw(
                                ptp,
                                dut,
                                layer_number,
                                layer_repetition,
                                lpar,
                                openeye_parameter,
                                layer_es,
                                dram,
                                log_level,
                                art.output_order,
                                rel_layers,
                            )
                        )
                    elif "conv" in l_type:
                        await cocotb.start_soon(
                            rtl_test_utils.compare_stream_Conv(
                                ptp,
                                dut,
                                layer_number,
                                layer_repetition,
                                lpar,
                                openeye_parameter,
                                layer_es,
                                dram,
                                log_level,
                                art.output_order,
                                rel_layers,
                            )
                        )
                    elif "dense" in l_type:
                        await cocotb.start_soon(
                            rtl_test_utils.compare_stream_Dense(
                                ptp,
                                dut,
                                layer_number,
                                layer_repetition,
                                lpar,
                                openeye_parameter,
                                layer_es,
                                dram,
                                log_level,
                                rel_layers,
                            )
                        )
                    elif "pooling" in l_type:
                        await cocotb.start_soon(
                            rtl_test_utils.compare_stream_Pooling(
                                ptp,
                                dut,
                                layer_number,
                                layer_repetition,
                                lpar,
                                openeye_parameter,
                                layer_es,
                                dram,
                                log_level,
                                rel_layers,
                            )
                        )

                    if os.environ.get("DUMP_PSUM_BUFFERS") and "dense" in l_type:
                        out_buf = dram.fmap["Output"][0]
                        for f in range(len(art.ref_results)):
                            logger.info(
                                "dense f=%2d ref=%s dut=%s",
                                f,
                                art.ref_results[f],
                                out_buf[f],
                            )

                    dram_ok = tum.compare_dram_with_ref(
                        lpar, art.ref_results, dram.fmap["Output"][0]
                    )
                    file_ok = True
                    if logging.DEBUG >= log_level:
                        file_ok = gtu.check_results(
                            openeye_parameter,
                            f"demo/layer_{l_type}/dma_stream_ref.txt",
                            f"demo/layer_{l_type}/output.txt",
                        )
                    assert dram_ok and file_ok, (
                        f"result check failed: values {'ok' if dram_ok else 'differ'}, "
                        f"reference file {'ok' if file_ok else 'differs'}"
                    )
                else:
                    await cocotb.start_soon(rtl_test_utils.await_ready_signal(ptp, dut))
                    time_printer.timestamp("Ready signal detected. Start new stream", logger)

            # Intermediate activation buffer verification against precomputed quantized reference
            if (
                ("input" not in layer_name)
                and not art.is_output_layer
                and "flatten" not in layer_name
                and "concat" not in layer_name
            ):
                for output_layer, target_slot in art.target_slots:
                    logger.info(
                        "Iact check context: layer %d (%s) -> %s [slot %d], repetition %d of %d",
                        layer_number,
                        layer_name,
                        output_layer,
                        target_slot,
                        layer_repetition,
                        lpar.needed_total_transmissions,
                    )
                    assert rtl_test_utils.compare_iact_storage(
                        ptp,
                        dut,
                        art.quantized_results,
                        openeye_parameter,
                        lpar,
                        slot_idx=target_slot,
                    )

    assert dut.rst_ni.value == 1, "rst_ni is not 1!"


async def execute_model(
    dut, only_files, sparse_iacts, sparse_wghts, layer_es, serial, ptp, model, rel_layers
):
    """Precompute all model references/streams first, then run RTL simulation if requested."""
    plan = precompute_model_artifacts(
        model=model,
        rel_layers=rel_layers,
        sparse_iacts=sparse_iacts,
        sparse_wghts=sparse_wghts,
        serial=serial,
        output_dir="demo/",
        write_files=True,
    )

    if only_files == 0:
        await run_rtl_simulation(dut, plan, layer_es, ptp)


# ==============================================================================
# Entry Points (Cocotb & Standalone Precomputation)
# ==============================================================================


def run_precompute_only(output_dir: str = "demo/") -> ModelExecutionPlan:
    """Standalone entry point to generate all references and stream files without RTL sim."""
    logging.basicConfig(level=log_level, format="%(levelname)-8s %(name)-20s %(message)s")
    (
        _,
        layer_mode,
        filters,
        kernelsize_x,
        kernelsize_y,
        inputsize_x,
        inputsize_y,
        outputsize,
        strides,
        channels,
        sparse_iacts,
        sparse_wghts,
    ) = envvars_to_vars()

    trunc_model, rel_layers = load_benchmark_model(
        layer_mode,
        filters,
        kernelsize_x,
        kernelsize_y,
        inputsize_x,
        inputsize_y,
        strides,
        channels,
        outputsize,
    )
    return precompute_model_artifacts(
        model=trunc_model,
        rel_layers=rel_layers,
        sparse_iacts=sparse_iacts,
        sparse_wghts=sparse_wghts,
        serial=1,
        output_dir=output_dir,
        write_files=True,
    )


# @cocotb.test()
async def model_test(dut):
    """Test the DUT with a given trained DNN model."""
    only_files, *_, sparse_iacts, sparse_wghts = envvars_to_vars()
    layer_es = les.LayerExecutionState()
    serial = 1
    ptp = _create_port_timing_params()

    tf_model_path = os.environ["MODEL_PATH"]
    model = tf.keras.models.load_model(tf_model_path)
    rel_layers = build_layer_topology(model)
    trunc_model = truncate_model(model)

    await execute_model(
        dut, only_files, sparse_iacts, sparse_wghts, layer_es, serial, ptp, trunc_model, rel_layers
    )


@cocotb.test()
async def start_test_fpga(dut):
    """Main cocotb test entry point for FPGA verification."""
    await setup_file_logging()
    timeout_time = 15000000000
    timeout_unit = "ns"

    try:
        await with_timeout(single_layer_test(dut), timeout_time, timeout_unit)
    except SimTimeoutError:
        dut._log.error("Test did not finish in time!")
        raise


async def single_layer_test(dut):
    """Simulate a single layer or benchmark model using the OpenEye FPGA wrapper."""
    (
        only_files,
        layer_mode,
        filters,
        kernelsize_x,
        kernelsize_y,
        inputsize_x,
        inputsize_y,
        outputsize,
        strides,
        channels,
        sparse_iacts,
        sparse_wghts,
    ) = envvars_to_vars()

    layer_es = les.LayerExecutionState()
    serial = 1
    ptp = _create_port_timing_params()

    trunc_model, rel_layers = load_benchmark_model(
        layer_mode,
        filters,
        kernelsize_x,
        kernelsize_y,
        inputsize_x,
        inputsize_y,
        strides,
        channels,
        outputsize,
    )

    await execute_model(
        dut, only_files, sparse_iacts, sparse_wghts, layer_es, serial, ptp, trunc_model, rel_layers
    )