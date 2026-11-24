# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Layer operation utilities for the OpenEye neural network accelerator.

This module provides host-side reference implementations of basic neural network
layer operations operating on NumPy-backed DRAM buffers:
- Pooling layers (max and average pooling)
- Flattening operations
- Quantization / requantization
"""

import numpy as np


def pool(dram, layer, layer_number):
    """Perform 2D pooling on feature maps in ``dram.fmap[layer_number][0]``.

    Writes the pooled output into ``dram.fmap[layer_number + 1][0]``.
    """
    out_x = layer.output.shape[1]
    out_y = layer.output.shape[2]
    out_c = layer.output.shape[3]
    stride_x = layer.strides[0]
    stride_y = layer.strides[1]

    src = np.asarray(dram.fmap[layer_number][0], dtype=np.int64)
    dst = dram.fmap[layer_number + 1][0]

    # Build sliding/strided window view of shape (out_c, out_x, out_y, stride_x, stride_y)
    windows = np.lib.stride_tricks.sliding_window_view(
        src[:out_c], (stride_x, stride_y), axis=(1, 2)
    )[:, ::stride_x, ::stride_y, :, :][:, :out_x, :out_y, :, :]

    layer_str = str(layer)
    if "Average" in layer_str:
        area = stride_x * stride_y
        dst[:out_c, :out_x, :out_y] = (windows.sum(axis=(3, 4)) // area).astype(dst.dtype)
    elif "Max" in layer_str:
        dst[:out_c, :out_x, :out_y] = windows.max(axis=(3, 4)).astype(dst.dtype)


def flat(dram, layer, layer_number):
    """Flatten 3D feature maps (channels, width, height) into a 1D vector."""
    in_x = layer.input.shape[1]
    in_y = layer.input.shape[2]
    in_c = layer.input.shape[3]

    src = np.asarray(dram.fmap[layer_number][0])[:in_c, :in_x, :in_y]
    # Preserve channel-major, y-then-x traversal order
    flattened = src.transpose(0, 2, 1).ravel()

    dst = dram.fmap[layer_number + 1][0]
    dst[: flattened.size] = flattened.astype(dst.dtype)


def _quantize_array(values, mult: int, shift_bits: int) -> np.ndarray:
    """Apply fixed-point multiplier and floor bit-shift to an integer array."""
    arr = np.asarray(values, dtype=np.int64)
    scaled = arr * int(mult)
    if shift_bits >= 0:
        return scaled >> int(shift_bits)
    return np.floor(scaled / (2.0 ** int(shift_bits))).astype(np.int64)


def quantize_dict(calculated_results, layer_parameters):
    """Quantize per-filter feature maps using each filter's (multiplier, shift) pair."""
    if isinstance(calculated_results, dict):
        return {
            f: _quantize_array(
                matrix,
                int(layer_parameters.quantize[f][0]),
                int(layer_parameters.quantize[f][1]),
            ).tolist()
            for f, matrix in calculated_results.items()
        }

    arr = np.asarray(calculated_results, dtype=np.int64)
    out = np.empty_like(arr)
    for f in range(arr.shape[0]):
        out[f] = _quantize_array(
            arr[f],
            int(layer_parameters.quantize[f][0]),
            int(layer_parameters.quantize[f][1]),
        )
    return out.tolist()