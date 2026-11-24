# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Main test utilities for the OpenEye neural network accelerator.

This module provides high-level test utilities and orchestration for testing
the OpenEye neural network accelerator implementation:

- Stream generation and mapping for different layer types
- Vectorized NumPy reference data generation and validation
- File I/O for reference CSVs and DMA stream verification
- NumPy-backed DRAM state updates and comparison
"""

import os
import logging
import math
import numpy as np

from open_eye.pooling_mapper import PoolingMapper
from open_eye.dense_mapper import DenseMapper
from open_eye.gemm_mapper import GemmMapper
from open_eye.conv_mapper import ConvMapper
from open_eye.dw_mapper import DWMapper
import open_eye.generic_test_utils as gtu
import open_eye.layer_execution_state as les

logger = logging.getLogger("cocotb")


def max_parallel_jobs():
    """Return configured worker limit (override: OPENEYE_MAX_PROCS)."""
    try:
        limit = int(os.environ.get("OPENEYE_MAX_PROCS", "0"))
    except ValueError:
        limit = 0
    if limit < 1:
        limit = max(1, (os.cpu_count() or 2) // 2)
    return limit


def run_jobs_bounded(make_process, count):
    """Run `count` helper processes, keeping at most max_parallel_jobs() alive."""
    limit = max_parallel_jobs()
    running = []
    for index in range(count):
        while len(running) >= limit:
            for job in running:
                job.join(0.05)
            running = [job for job in running if job.is_alive()]
        proc = make_process(index)
        proc.start()
        running.append(proc)
    for job in running:
        job.join()


def get_verilog_sources(hdl_dir):
    """Gather all Verilog/SystemVerilog source files from a directory tree."""
    verilog_sources = []
    for root, _, files in os.walk(hdl_dir):
        for fname in files:
            if fname.endswith((".v", ".sv")):
                verilog_sources.append(os.path.join(root, fname))
    return verilog_sources


def _build_layer_stream(params, layer_params, dram_layer_content, layer_repetition, sparse_iacts, sparse_wghts):
    """Instantiate the layer-specific mapper and generate the stream for one repetition."""
    lname = str(layer_params.layer_name)
    if "depthwise" in lname:
        mapper = DWMapper(params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts)
    elif "conv" in lname:
        mapper = ConvMapper(params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts)
    elif "gemm" in lname:
        mapper = GemmMapper(params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts)
    elif "dense" in lname:
        if getattr(params, "DATAFLOW", "row_stationary") == "output_stationary":
            mapper = GemmMapper(params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts)
        else:
            mapper = DenseMapper(params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts)
    elif "pooling" in lname:
        mapper = PoolingMapper(params, layer_params, layer_repetition, dram_layer_content, sparse_iacts, sparse_wghts)
    else:
        raise ValueError(f"Unsupported layer type for stream generation: {lname}")

    mapper.make_stream()
    return mapper.get_stream()


def write_stream_layer_mp(params, layer_params, dram_layer_content, return_dict, layer_repetition, sparse_iacts, sparse_wghts):
    """Compatibility wrapper for generating a single repetition's stream into a mapping."""
    return_dict[layer_repetition] = _build_layer_stream(
        params, layer_params, dram_layer_content, layer_repetition, sparse_iacts, sparse_wghts
    )


def write_stream(params, layer_params, dram_layer_content, sparse_iacts, sparse_wghts):
    """Generate hardware streams for all repetitions of a layer."""
    return {
        rep: _build_layer_stream(params, layer_params, dram_layer_content, rep, sparse_iacts, sparse_wghts)
        for rep in range(layer_params.needed_total_transmissions)
    }


# ==============================================================================
# Reference Generation & File Output
# ==============================================================================


def make_ref(params, layer_params, layer_name, dram, calculated_results):
    """Generate weight, activation, partial-sum, and DMA stream reference files."""
    write_weight_file(layer_params, layer_name, dram)
    logger.info("All weight-files written")

    write_iact_file(layer_params, layer_name, dram)
    logger.info("All iact-files written")

    logger.info("All results calculated")

    write_psum_file(layer_params, layer_name, dram, calculated_results)
    logger.info("All psum-files written")

    output_order = []
    lname = str(layer_params.layer_name)

    if "depthwise" in lname or "conv" in lname:
        cluster_order = [
            a + b
            for a in range(layer_params.used_Y_cluster)
            for b in range(0, params.Clusters_Y, layer_params.used_Y_cluster)
        ]
        ref_path = f"demo/layer_{layer_name}/dma_stream_ref.txt"
        with gtu.open_or_create_file(ref_path) as file_dma_ref:
            for layer_repetition in range(layer_params.needed_total_transmissions):
                if params.SERIAL:
                    calculate_conv_serial(params, layer_params, calculated_results, file_dma_ref)
                    output_order.append([])
                else:
                    coords = calculate_conv_parallel(
                        params, layer_params, calculated_results, cluster_order, layer_repetition, file_dma_ref
                    )
                    output_order.append(coords)
                logger.info(
                    "Stream %d / %d calculated.",
                    layer_repetition,
                    layer_params.needed_total_transmissions,
                )

    elif "dense" in lname or "gemm" in lname or "matmul" in lname:
        ref_path = f"demo/layer_{layer_name}/dma_stream_ref.txt"
        if params.SERIAL:
            per_column = math.ceil(layer_params.used_psum_per_PE)
            mask = (1 << params.DATA_PSUM_BITWIDTH) - 1
            fmt_spec = f"0{params.DMA_BITWIDTH}b"
            with gtu.open_or_create_file(ref_path) as file_dma_ref:
                lines = []
                for repetition in range(layer_params.needed_total_transmissions):
                    tile = (repetition // layer_params.iact_transmissions_pe) % layer_params.psum_transmissions_pe
                    offset = tile * params.Clusters_X * per_column
                    for psum_repetition in range(layer_params.needed_psum_transmissions):
                        base = offset + psum_repetition * per_column * params.Clusters_X
                        for address in range(per_column):
                            for column in range(params.Clusters_X):
                                idx = base + column * per_column + address
                                value = calculated_results[idx] if idx < len(calculated_results) else 0
                                lines.append(format(int(value) & mask, fmt_spec))
                if lines:
                    file_dma_ref.write("\n".join(lines) + "\n")
        else:
            zero_str = gtu.to_twos_complement_string(0, params.DATA_PSUM_BITWIDTH)
            with gtu.open_or_create_file(ref_path) as file_dma_ref:
                lines = []
                for layer_repetition in range(layer_params.needed_total_transmissions):
                    rep_cycle = layer_repetition // layer_params.iact_transmissions_pe
                    for psum_pe in range(math.ceil(layer_params.used_psum_per_PE / 2)):
                        for cl_y in range(params.Clusters_Y):
                            for cl_x in range(params.Clusters_X):
                                part_a = zero_str
                                part_b = zero_str
                                for counter in range(params.PARALLEL_MACS):
                                    out_idx = (
                                        counter
                                        + 2 * psum_pe
                                        + cl_x * layer_params.used_psum_per_PE
                                        + cl_y * layer_params.used_psum_per_PE * params.Clusters_X
                                        + rep_cycle
                                        * layer_params.used_psum_per_PE
                                        * params.Clusters_X
                                        * params.Clusters_Y
                                    )
                                    val = calculated_results[out_idx] if out_idx < len(calculated_results) else 0
                                    encoded = gtu.to_twos_complement_string(val, params.DATA_PSUM_BITWIDTH)
                                    if counter == 0:
                                        part_b = encoded
                                    else:
                                        part_a = encoded
                                lines.append(part_a + part_b)
                if lines:
                    file_dma_ref.write("\n".join(lines) + "\n")

    logger.info("Reference Output calculated.")
    return output_order


def _format_2d_csv(grid_2d, width: int = 5) -> str:
    """Format a 2D array/sequence into semicolon-delimited fixed-width CSV lines."""
    return "\n".join("".join(f"{int(val):{width}d};" for val in row) for row in grid_2d) + "\n"


def write_weight_file(layer_params, layer_name, dram):
    """Write reference weight matrices to CSV files."""
    lname = str(layer_params.layer_name)
    weights = dram.weights[layer_name]

    if "dense" in lname or "gemm" in lname or "matmul" in lname:
        # weights shape: (filters, iact_size_x) -> output rows: iact_size_x, cols: filters
        grid = np.asarray(weights)[: layer_params.filters, : layer_params.iact_size_x].T
        with gtu.open_or_create_file(f"demo/layer_{layer_name}/weight/wght_ref_0.csv") as fh:
            fh.write(_format_2d_csv(grid, width=5))

    elif "depthwise" in lname:
        # weights shape: (channels, kx, ky) -> output rows: y, cols: x
        w_arr = np.asarray(weights)
        for c in range(layer_params.channels):
            grid = w_arr[c, : layer_params.kernel_size[0], : layer_params.kernel_size[1]].T
            with gtu.open_or_create_file(f"demo/layer_{layer_name}/weight/wght_ref_{c}.csv") as fh:
                fh.write(_format_2d_csv(grid, width=5))

    elif "conv" in lname:
        # weights shape: (channels, filters, ky, kx)
        w_arr = np.asarray(weights)
        for c in range(layer_params.channels):
            for f in range(layer_params.filters):
                grid = w_arr[c, f, : layer_params.kernel_size[1], : layer_params.kernel_size[0]]
                with gtu.open_or_create_file(f"demo/layer_{layer_name}/weight/wght_ref_{c}_{f}.csv") as fh:
                    fh.write(_format_2d_csv(grid, width=5))


def write_iact_file(layer_params, layer_name, dram):
    """Write reference input activations to CSV files."""
    lname = str(layer_params.layer_name)
    fmap_0 = np.asarray(dram.fmap[layer_name][0])

    if "dense" in lname or "gemm" in lname or "matmul" in lname:
        with gtu.open_or_create_file(f"demo/layer_{layer_name}/iact/iact_ref_0.csv") as fh:
            fh.write("\n".join(str(int(v)) for v in fmap_0[: layer_params.iact_size_x]) + "\n")

    elif "pooling" in lname:
        in_x, in_y, in_c = layer_params.input_shape[1], layer_params.input_shape[2], layer_params.input_shape[3]
        for c in range(in_c):
            grid = fmap_0[c, :in_x, :in_y].T
            with gtu.open_or_create_file(f"demo/layer_{layer_name}/iact/iact_ref_{c}.csv") as fh:
                fh.write(_format_2d_csv(grid, width=5))

    else:
        in_x, in_y, in_c = layer_params.input_shape[1], layer_params.input_shape[2], layer_params.input_shape[3]
        kx, ky = layer_params.kernel_size[0], layer_params.kernel_size[1]
        pad_x_before = math.floor(kx / 2)
        pad_x_after = max(0, math.ceil(kx / 2) - 1)
        pad_y_before = math.floor(ky / 2)
        pad_y_after = max(0, math.ceil(ky / 2) - 1)

        for c in range(in_c):
            padded = np.pad(
                fmap_0[c, :in_x, :in_y],
                ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after)),
                mode="constant",
                constant_values=0,
            )
            with gtu.open_or_create_file(f"demo/layer_{layer_name}/iact/iact_ref_{c}.csv") as fh:
                fh.write(_format_2d_csv(padded.T, width=5))


def write_psum_file(layer_params, layer_name, dram, calculated_results):
    """Write expected layer outputs (partial sums) to CSV files."""
    lname = str(layer_params.layer_name)

    if "dense" in lname or "gemm" in lname or "matmul" in lname:
        with gtu.open_or_create_file(f"demo/layer_{layer_name}/psum/psum_ref_0.csv") as fh:
            fh.write("\n".join(str(int(calculated_results[x])) for x in range(layer_params.filters)) + "\n")

    elif "depthwise" in lname:
        for f in range(layer_params.filters):
            grid = np.asarray(calculated_results[f])[: layer_params.psum_size_x, : layer_params.psum_size_y]
            with gtu.open_or_create_file(f"demo/layer_{layer_name}/psum/psum_ref_{f}.csv") as fh:
                fh.write(_format_2d_csv(grid, width=8))

    elif "conv" in lname:
        out_x, out_y, out_c = layer_params.output_shape[1], layer_params.output_shape[2], layer_params.output_shape[3]
        for f in range(out_c):
            grid = np.asarray(calculated_results[f])[:out_x, :out_y].T
            with gtu.open_or_create_file(f"demo/layer_{layer_name}/psum/psum_ref_{f}.csv") as fh:
                fh.write(_format_2d_csv(grid, width=8))

    elif "pooling" in lname:
        with gtu.open_or_create_file(f"demo/layer_{layer_name}/psum/psum_ref_0.csv") as fh:
            fh.write("\n".join(str(calculated_results[x]) for x in range(layer_params.output_shape[3])) + "\n")


# ==============================================================================
# Vectorized Reference Computation
# ==============================================================================


def collect_results(layer_params, dram, serial):
    """Compute golden reference outputs for a layer using vectorized NumPy operations."""
    lname = str(layer_params.layer_name)
    layer_key = layer_params.layer_name

    if "dense" in lname or "gemm" in lname or "matmul" in lname:
        w = np.asarray(dram.weights[layer_key], dtype=np.int64)[: layer_params.filters, : layer_params.iact_size_x]
        x_vec = np.asarray(dram.fmap[layer_key][0], dtype=np.int64)[: layer_params.iact_size_x]
        b = np.asarray(dram.bias[layer_key], dtype=np.int64)[: layer_params.filters]
        res = w @ x_vec + b
        return {x: int(res[x]) for x in range(layer_params.filters)}

    if "depthwise" in lname:
        out_x = layer_params.psum_size_x
        out_y = layer_params.psum_size_y
        channels = layer_params.channels
        filters = layer_params.filters
        kx, ky = layer_params.kernel_size[0], layer_params.kernel_size[1]
        sx, sy = layer_params.strideX, layer_params.strideY

        fmap = np.asarray(dram.fmap[layer_key][0], dtype=np.int64)
        weights = np.asarray(dram.weights[layer_key], dtype=np.int64)
        bias = np.asarray(dram.bias[layer_key], dtype=np.int64)

        out = np.zeros((layer_params.output_shape[3], out_x, out_y), dtype=np.int64)
        max_in_x = out_x * sx
        max_in_y = out_y * sy
        ox_off = math.floor(kx / 2)
        oy_off = math.floor((ky - 1) / 2)

        for x in range(-math.floor(kx / 2), math.ceil(kx / 2)):
            j_min = max(0, math.ceil(-x / sx))
            j_max = min(out_x, math.ceil((max_in_x - x) / sx))
            if j_min >= j_max:
                continue
            wx = x + ox_off
            in_x_slice = slice(x + j_min * sx, x + j_max * sx, sx)

            for y in range(-math.floor(ky / 2), math.ceil(ky / 2)):
                i_min = max(0, math.ceil(-y / sy))
                i_max = min(out_y, math.ceil((max_in_y - y) / sy))
                if i_min >= i_max:
                    continue
                wy = y + oy_off
                in_y_slice = slice(y + i_min * sy, y + i_max * sy, sy)

                out[:channels, j_min:j_max, i_min:i_max] += (
                    fmap[:channels, in_x_slice, in_y_slice] * weights[:channels, wy, wx][:, None, None]
                )

        out[:filters] += bias[:filters, None, None]
        return out.tolist()

    if "conv" in lname:
        in_x = layer_params.input_shape[1]
        in_y = layer_params.input_shape[2]
        in_c = layer_params.input_shape[3]
        out_x = layer_params.output_shape[1]
        out_y = layer_params.output_shape[2]
        out_f = layer_params.output_shape[3]
        filters = layer_params.filters
        kx, ky = layer_params.kernel_size[0], layer_params.kernel_size[1]
        sx, sy = layer_params.strideX, layer_params.strideY

        fmap = np.asarray(dram.fmap[layer_key][0], dtype=np.int64)
        weights = np.asarray(dram.weights[layer_key], dtype=np.int64)
        bias = np.asarray(dram.bias[layer_key], dtype=np.int64)

        out = np.broadcast_to(bias[:out_f, None, None], (out_f, out_x, out_y)).copy()
        ox_off = math.floor(kx / 2)
        oy_off = math.floor((ky - 1) / 2)

        for x in range(-math.floor(kx / 2), math.ceil(kx / 2)):
            j_min = max(0, math.ceil(-x / sx))
            j_max = min(out_x, math.ceil((in_x - x) / sx))
            wx = x + ox_off

            for y in range(-math.floor(ky / 2), math.ceil(ky / 2)):
                i_min = max(0, math.ceil(-y / sy))
                i_max = min(out_y, math.ceil((in_y - y) / sy))
                wy = y + oy_off

                if j_min < j_max and i_min < i_max:
                    fmap_patch = fmap[
                        :in_c,
                        x + j_min * sx : x + j_max * sx : sx,
                        y + i_min * sy : y + i_max * sy : sy,
                    ]
                    w_tap = weights[:in_c, :filters, wy, wx]
                    out[:filters, j_min:j_max, i_min:i_max] += np.einsum("cf,cji->fji", w_tap, fmap_patch)

                if not serial:
                    oob_mask = np.ones((out_x, out_y), dtype=bool)
                    if j_min < j_max and i_min < i_max:
                        oob_mask[j_min:j_max, i_min:i_max] = False
                    if np.any(oob_mask):
                        oob_sum = weights[:in_c, :filters, wx, wy].sum(axis=0)
                        out[:filters][:, oob_mask] += oob_sum[:, None]

        return {f: out[f].tolist() for f in range(out_f)}

    if "pooling" in lname:
        out_x = layer_params.output_shape[1]
        out_y = layer_params.output_shape[2]
        out_f = layer_params.output_shape[3]
        fmap = np.asarray(dram.fmap[layer_key][0], dtype=np.int64)

        if layer_params.pooling_mode == 0:
            # 2x2 Max Pooling
            s00 = fmap[:out_f, 0 : 2 * out_x : 2, 0 : 2 * out_y : 2]
            s01 = fmap[:out_f, 0 : 2 * out_x : 2, 1 : 2 * out_y + 1 : 2]
            s10 = fmap[:out_f, 1 : 2 * out_x + 1 : 2, 0 : 2 * out_y : 2]
            s11 = fmap[:out_f, 1 : 2 * out_x + 1 : 2, 1 : 2 * out_y + 1 : 2]
            out = np.maximum(np.maximum(s00, s01), np.maximum(s10, s11))
            return out.tolist()

        # Global Average Pooling
        in_x = layer_params.input_shape[1]
        in_y = layer_params.input_shape[2]
        area = in_x * in_y
        channel_means = fmap[:out_f, :in_x, :in_y].sum(axis=(1, 2)) // area
        out = np.broadcast_to(channel_means[:, None, None], (out_f, out_x, out_y)).copy()
        return out.tolist()

    if "add" in lname:
        out_x = layer_params.output_shape[1]
        out_y = layer_params.output_shape[2]
        out_f = layer_params.output_shape[3]
        slots = [np.asarray(slot, dtype=np.int64)[:out_f, :out_x, :out_y] for slot in dram.fmap[layer_key]]
        return np.sum(slots, axis=0).tolist()

    if "concat" in lname:
        slots = [np.asarray(slot, dtype=np.int64) for slot in dram.fmap[layer_key]]
        return np.concatenate(slots, axis=0).tolist()

    return []


def calculate_dense_results_mp(x, layer_params, layer_name, dram, calculated_results, return_dict):
    """Compatibility helper for single-output Dense reference calculation."""
    w = np.asarray(dram.weights[layer_name][x], dtype=np.int64)[: layer_params.iact_size_x]
    inp = np.asarray(dram.fmap[layer_name][0], dtype=np.int64)[: layer_params.iact_size_x]
    return_dict[x] = int(calculated_results + np.dot(w, inp) + dram.bias[layer_name][x])


def refresh_position(x_cor, y_cor, filter, y_line_counter, kernel_counter, layer_params, position):
    return x_cor, y_cor, filter, y_line_counter, kernel_counter


def calculate_conv_serial(params, layer_params, calculated_results, file_dma_ref):
    """Write expected serial DMA output words for Conv2D / Depthwise Conv2D layers."""
    words_per_transmission = params.DMA_BITWIDTH // 32
    filter_cycles = (layer_params.filters // layer_params.used_psum_per_PE) // layer_params.different_kernels_per_calculation
    les.x_start = 0
    zero_word = gtu.to_twos_complement_string(0, params.DATA_PSUM_BITWIDTH)
    lines = []

    for refresh in range(layer_params.needed_refreshes_mx[0][0] // layer_params.diff_iact_layer):
        les.y_start = (
            (refresh // filter_cycles) // layer_params.iact_x_line_repetitions
        ) * layer_params.y_lines_per_calculation
        les.f_start = (
            layer_params.used_psum_per_PE * (refresh % filter_cycles) * layer_params.different_kernels_per_calculation
        )

        filt = les.f_start
        x_cor = les.x_start
        for _ in range(layer_params.used_psum_per_PE):
            kernel_counter = 0
            x_cor = les.x_start
            y_cor = les.y_start
            for cl_y in range(params.Clusters_Y):
                for _ in range(params.Clusters_X):
                    if x_cor >= layer_params.psum_size_x + layer_params.add_up:
                        if kernel_counter < layer_params.different_kernels_per_calculation - 1:
                            kernel_counter += 1
                            x_cor = les.x_start
                            filt += 1
                    for _ in range(0, params.Psum_Routers, words_per_transmission):
                        temp_string = ""
                        for _ in range(words_per_transmission):
                            if cl_y % layer_params.used_Y_cluster == 0:
                                if kernel_counter < layer_params.different_kernels_per_calculation:
                                    if (x_cor < layer_params.psum_size_x) and (y_cor < layer_params.psum_size_y):
                                        try:
                                            val = calculated_results[filt][x_cor][y_cor]
                                            temp_string = (
                                                gtu.to_twos_complement_string(val, params.DATA_PSUM_BITWIDTH)
                                                + temp_string
                                            )
                                        except Exception:
                                            temp_string = zero_word + temp_string
                                    else:
                                        temp_string = zero_word + temp_string
                                    x_cor += 1
                            else:
                                temp_string = zero_word + temp_string
                        lines.append(temp_string.zfill(params.DMA_BITWIDTH))
            filt += 1

        if filt >= layer_params.filters:
            if x_cor >= layer_params.psum_size_x + layer_params.psum_add_up:
                les.x_start = 0
            else:
                les.x_start = x_cor

    if lines:
        file_dma_ref.write("\n".join(lines) + "\n")


def calculate_dwconv_serial(params, layer_params, calculated_results, file_dma_ref):
    """Write expected serial DMA output words for Depthwise Conv2D layers."""
    calculate_conv_serial(params, layer_params, calculated_results, file_dma_ref)


def calculate_conv_parallel(params, layer_params, calculated_results, cluster_order, layer_repetition, file_dma_ref):
    """Write expected parallel DMA output words and return coordinate mapping."""
    coordinates = []
    layer_repetition_cycle = math.floor(layer_repetition / layer_params.iact_transmissions_pe)
    refresh_lower = math.ceil(layer_params.needed_refreshes_mx[layer_repetition][1] / layer_params.used_Y_cluster)
    refresh_upper = math.ceil(layer_params.needed_refreshes_mx[layer_repetition][2] / layer_params.used_Y_cluster)
    zero_word = gtu.to_twos_complement_string(0, params.DATA_PSUM_BITWIDTH)
    lines = []

    for refresh in range(refresh_lower, refresh_upper):
        match layer_params.single_cluster_computation:
            case 1:
                step = math.ceil(layer_params.filters / layer_params.needed_wght_transmissions / params.Clusters / 2)
            case 2:
                step = math.ceil(layer_params.filters / layer_params.needed_wght_transmissions / params.Clusters_Y / 2)
            case _:
                step = math.ceil(layer_params.filters / layer_params.needed_wght_transmissions / 2)

        rep_mod = layer_repetition_cycle % layer_params.needed_wght_transmissions
        psum_pe_lower = rep_mod * step
        psum_pe_upper = (rep_mod + 1) * step

        width_stride = layer_params.output_shape[1] + layer_params.add_up

        for psum_pe in range(psum_pe_lower, psum_pe_upper):
            for cl_y in cluster_order:
                for cl_x in range(params.Clusters_X):
                    for router in range(params.Psum_Routers):
                        if layer_params.computing_mx[cl_x][cl_y][0][router] != 1:
                            continue
                        partial_result_a = zero_word
                        partial_result_b = zero_word
                        for counter in range(params.PARALLEL_MACS):
                            match layer_params.single_cluster_computation:
                                case 1:
                                    linear = router + (refresh * params.PEs_X)
                                    x_cor = int(linear % width_stride)
                                    y_cor = int(linear / width_stride)
                                    filt = (
                                        2 * psum_pe
                                        + counter
                                        + (2 * psum_pe_upper * (cl_y * params.Clusters_X + cl_x))
                                    )
                                case 2:
                                    x_cor = int(
                                        ((cl_x * params.PEs_X) + router + (refresh * (params.PEs_X * params.Clusters_X)))
                                        % width_stride
                                    )
                                    y_cor = int(
                                        (router + (refresh * (params.PEs_X * params.Clusters_X))) / width_stride
                                    )
                                    filt = 2 * psum_pe + counter + (2 * psum_pe_upper * cl_y)
                                case _:
                                    linear = (
                                        router
                                        + cl_x * params.PEs_X
                                        + math.floor(cl_y / layer_params.used_Y_cluster)
                                        * params.Clusters_X
                                        * params.PEs_X
                                        + ((cl_y % layer_params.used_Y_cluster) + refresh * layer_params.used_Y_cluster)
                                        * (params.Clusters_Y * params.Clusters_X * params.PEs_X / layer_params.used_Y_cluster)
                                    )
                                    x_cor = int(linear % width_stride)
                                    y_cor = int(linear / width_stride)
                                    filt = 2 * psum_pe + counter

                            coordinates.append([filt, x_cor, y_cor])
                            if (x_cor < layer_params.output_shape[1]) and (y_cor < layer_params.output_shape[2]):
                                try:
                                    encoded = gtu.to_twos_complement_string(
                                        calculated_results[filt][x_cor][y_cor], params.DATA_PSUM_BITWIDTH
                                    )
                                    if counter == 0:
                                        partial_result_b = encoded
                                    else:
                                        partial_result_a = encoded
                                except Exception:
                                    pass

                        lines.append(partial_result_a + partial_result_b)

    if lines:
        file_dma_ref.write("\n".join(lines) + "\n")
    return coordinates


# ==============================================================================
# Result Verification & DRAM Propagation
# ==============================================================================


def compare_dram_with_ref(layer_params, ref_output, dram):
    """Compare hardware output in DRAM against reference data using vectorized NumPy checks."""
    logger.info("Results are checked.")
    name = str(layer_params.layer_name)

    if "conv" in name:
        num_fmaps = len(ref_output)
        ref_arr = np.asarray([ref_output[f] for f in range(num_fmaps)], dtype=np.int64)
        dram_arr = np.asarray(dram[:num_fmaps], dtype=np.int64)

        if np.array_equal(ref_arr, dram_arr):
            return True

        diff_mask = ref_arr != dram_arr
        failed_fmaps = np.flatnonzero(diff_mask.any(axis=(1, 2))).tolist()
        logger.error(
            "%d of %d feature maps differ from the reference (first: %s)",
            len(failed_fmaps),
            num_fmaps,
            failed_fmaps[:8],
        )
        mismatch_coords = np.argwhere(diff_mask)
        for f, x, y in mismatch_coords[:64]:
            logger.error(
                "Difference found at f = %d, x = %d, y = %d: reference %s, DUT %s",
                int(f),
                int(x),
                int(y),
                ref_arr[f, x, y],
                dram_arr[f, x, y],
            )
        return False

    if "dense" in name or "gemm" in name or "matmul" in name or "pooling" in name:
        num_out = len(ref_output)
        ref_arr = np.asarray([ref_output[f] for f in range(num_out)], dtype=np.int64)
        dram_arr = np.asarray(dram[:num_out], dtype=np.int64)

        if np.array_equal(ref_arr, dram_arr):
            return True

        mismatches = np.flatnonzero(ref_arr != dram_arr).tolist()
        logger.error("%d of %d outputs differ from the reference", len(mismatches), num_out)
        for f in mismatches[:8]:
            logger.error("  f = %d: reference %s, DUT %s", f, ref_arr[f], dram_arr[f])
        return False

    return True


def fill_dram_with_ref(ref_output, dram, current_layer_params, next_layer_params):
    """Propagate reference layer outputs into the next layer's NumPy DRAM input buffer."""
    logger.info("Results are transmitted.")
    curr_name = str(current_layer_params.layer_name)
    next_name = str(next_layer_params.layer_name)
    is_next_dense = any(k in next_name for k in ("dense", "gemm", "matmul"))

    if "conv" in curr_name:
        num_f = len(ref_output)
        ref_arr = np.asarray([ref_output[f] for f in range(num_f)], dtype=dram.dtype)
        f_dim, x_dim, y_dim = ref_arr.shape

        if any(k in next_name for k in ("conv", "pooling", "add", "concat")):
            dram[:f_dim, :x_dim, :y_dim] += ref_arr
        elif is_next_dense:
            ix = current_layer_params.iact_size_x
            iy = current_layer_params.iact_size_y
            f_idx = np.arange(f_dim)[:, None, None]
            x_idx = np.arange(x_dim)[None, :, None]
            y_idx = np.arange(y_dim)[None, None, :]
            pos = (f_idx % 2) + (2 * x_idx) + (ix * 2 * y_idx) + (ix * iy * 2 * (f_idx // 2))
            dram[pos.ravel()] += ref_arr.ravel()

    elif "pooling" in curr_name:
        ref_arr = np.asarray(ref_output, dtype=dram.dtype)
        f_dim, x_dim, y_dim = ref_arr.shape

        if is_next_dense:
            f_idx = np.arange(f_dim)[:, None, None]
            x_idx = np.arange(x_dim)[None, :, None]
            y_idx = np.arange(y_dim)[None, None, :]
            pos = (
                (f_idx % 2)
                + ((f_idx // 2) * (2 * x_dim * y_dim))
                + ((x_idx % 4) * 2)
                + ((x_idx // 4) * 8)
                + (y_idx * (2 * x_dim))
            )
            dram[pos.ravel()] += ref_arr.ravel()
        else:
            dram[:f_dim, :x_dim, :y_dim] += ref_arr

    elif "dense" in curr_name or "gemm" in curr_name or "matmul" in curr_name:
        num_f = len(ref_output)
        ref_arr = np.asarray([ref_output[f] for f in range(num_f)], dtype=dram.dtype)
        dram[:num_f] += ref_arr

    elif "add" in curr_name:
        ref_arr = np.asarray(ref_output, dtype=dram.dtype)
        f_dim, x_dim, y_dim = ref_arr.shape
        dram[:f_dim, :x_dim, :y_dim] += ref_arr

    elif "concat" in curr_name:
        ref_arr = np.asarray(ref_output, dtype=dram.dtype)
        if dram.shape == ref_arr.shape:
            dram[:] = ref_arr
        else:
            dram = ref_arr

    return dram