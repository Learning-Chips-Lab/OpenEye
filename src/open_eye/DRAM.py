# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""DRAM simulation module for OpenEye accelerator testing.

This module provides the DRAMContents class which simulates DRAM memory storage
for the OpenEye neural network accelerator using contiguous NumPy arrays. It manages
storage and initialization of weights, biases, and feature maps (activations) across
Conv2D, Depthwise Convolution, Dense, GEMM, Pooling, Flatten, Add, and Concat layers.

Memory Layout (all arrays use ``DRAM_DTYPE = np.int32``):
    - ``fmap[layer_name]``: ``list[np.ndarray]`` (one array per input/branch slot)
        * Conv / Pooling / Flatten / Input / Add / Concat: ``(channels, width, height)``
        * Dense / GEMM: ``(features,)``
    - ``weights[layer_name]``: ``np.ndarray``
        * Depthwise Conv2D: ``(channels, kernel_x, kernel_y)``
        * Standard Conv2D: ``(channels, filters, kernel_y, kernel_x)``
        * Dense / GEMM: ``(filters, iact_size_x)``
    - ``bias[layer_name]``: ``np.ndarray``
        * Conv2D / Depthwise / Dense / GEMM: ``(filters,)``
"""

import numpy as np

DRAM_DTYPE = np.int32
_NONZERO_CHOICES = np.array([-1, 1], dtype=DRAM_DTYPE)


def _replace_zeros_with_unit_signs(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace all exact zero entries in ``arr`` in-place with random -1 or +1."""
    zero_mask = arr == 0
    num_zeros = int(np.count_nonzero(zero_mask))
    if num_zeros > 0:
        arr[zero_mask] = rng.choice(_NONZERO_CHOICES, size=num_zeros)
    return arr


class DRAMContents:
    """NumPy-backed DRAM contents storage for OpenEye accelerator tests."""

    def __init__(self, model, layer_parameters, rel_layers, seed: int | None = None) -> None:
        """Allocate zero-initialized NumPy storage buffers for all layers in the model."""
        self._rng = np.random.default_rng(seed)

        dram_fmap: dict[str, list[np.ndarray]] = {}
        dram_weights: dict[str, np.ndarray] = {}
        dram_bias: dict[str, np.ndarray] = {}

        next_layers_map = rel_layers["Next_Layer"]
        prev_layers_map = rel_layers["Previous_Layer"]

        for name, layer_param in layer_parameters.items():
            lname = str(name)

            # === WEIGHT & BIAS INITIALIZATION ===
            if "depthwise" in lname:
                dram_weights[name] = np.zeros(
                    (layer_param.channels, layer_param.kernel_size[0], layer_param.kernel_size[1]),
                    dtype=DRAM_DTYPE,
                )
                dram_bias[name] = np.zeros(layer_param.filters, dtype=DRAM_DTYPE)

            elif "conv" in lname:
                dram_weights[name] = np.zeros(
                    (
                        layer_param.channels,
                        layer_param.filters,
                        layer_param.kernel_size[1],
                        layer_param.kernel_size[0],
                    ),
                    dtype=DRAM_DTYPE,
                )
                dram_bias[name] = np.zeros(layer_param.filters, dtype=DRAM_DTYPE)

            elif "dense" in lname or "gemm" in lname or "matmul" in lname:
                dram_weights[name] = np.zeros(
                    (layer_param.filters, layer_param.iact_size_x),
                    dtype=DRAM_DTYPE,
                )
                dram_bias[name] = np.zeros(layer_param.filters, dtype=DRAM_DTYPE)

            elif "flat" in lname:
                dram_weights[name] = np.zeros(1, dtype=DRAM_DTYPE)
                dram_bias[name] = np.empty(0, dtype=DRAM_DTYPE)

            elif "pooling" in lname:
                dram_weights[name] = np.zeros(1, dtype=DRAM_DTYPE)
                dram_bias[name] = np.zeros(layer_param.output_shape[2], dtype=DRAM_DTYPE)

            elif "input" in lname or "add" in lname or "concat" in lname:
                dram_weights[name] = np.empty(0, dtype=DRAM_DTYPE)
                dram_bias[name] = np.empty(0, dtype=DRAM_DTYPE)

            # === FEATURE MAP INITIALIZATION ===
            if "conv" in lname or "pooling" in lname:
                dram_fmap[name] = [
                    np.zeros(
                        (layer_param.channels, layer_param.iact_size_x, layer_param.iact_size_y),
                        dtype=DRAM_DTYPE,
                    )
                ]

            elif "dense" in lname or "gemm" in lname or "matmul" in lname:
                dram_fmap[name] = [np.zeros(layer_param.iact_size_x, dtype=DRAM_DTYPE)]

            elif "flat" in lname:
                dram_fmap[name] = [
                    np.zeros(
                        (layer_param.channels, layer_param.input_shape[1], layer_param.input_shape[2]),
                        dtype=DRAM_DTYPE,
                    )
                ]

            elif "input" in lname:
                first_next = layer_parameters[next_layers_map[name][0]]
                in_shape = first_next.input_shape
                dram_fmap[name] = [
                    np.zeros((in_shape[3], in_shape[1], in_shape[2]), dtype=DRAM_DTYPE)
                ]

            elif "add" in lname or "concat" in lname:
                dram_fmap[name] = [
                    np.zeros(
                        (
                            layer_parameters[prev_name].output_shape[3],
                            layer_parameters[prev_name].output_shape[1],
                            layer_parameters[prev_name].output_shape[2],
                        ),
                        dtype=DRAM_DTYPE,
                    )
                    for prev_name in prev_layers_map[name]
                ]

            # === FINAL OUTPUT FEATURE MAP ===
            if len(next_layers_map[name]) == 0:
                out_shape = layer_param.output_shape
                if "conv" in lname:
                    dram_fmap["Output"] = [
                        np.zeros((layer_param.filters, out_shape[1], out_shape[2]), dtype=DRAM_DTYPE)
                    ]
                elif "pooling" in lname:
                    dram_fmap["Output"] = [
                        np.zeros((layer_param.channels, out_shape[1], out_shape[2]), dtype=DRAM_DTYPE)
                    ]
                elif "dense" in lname or "gemm" in lname or "matmul" in lname:
                    dram_fmap["Output"] = [np.zeros(layer_param.filters, dtype=DRAM_DTYPE)]

        self.fmap = dram_fmap
        self.weights = dram_weights
        self.bias = dram_bias

    def write_initial_data_to_dram(
        self,
        model,
        layer_parameters,
        sparse_iacts: bool,
        sparse_wghts: bool,
        rel_layers,
        seed: int | None = None,
    ) -> None:
        """Populate DRAM with vectorized quantized weights, biases, and initial activations."""
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        layers_seq = model.layers if hasattr(model, "layers") else model
        prev_layers_map = rel_layers["Previous_Layer"]

        for name, layer_param in layer_parameters.items():
            lname = str(layer_param.layer_name)
            layer_idx = layer_param.layer_number_in_model

            # === WEIGHT & BIAS LOADING ===
            if "depthwise" in lname:
                keras_weights = layers_seq[layer_idx].weights
                # Keras shape: (kx, ky, channels, depth_multiplier) -> (channels, kx, ky)
                raw_w = np.asarray(keras_weights[0], dtype=np.float64)[..., 0]
                quant_w = np.floor(127.0 * np.transpose(raw_w, (2, 0, 1))).astype(DRAM_DTYPE)
                self.weights[name] = _replace_zeros_with_unit_signs(quant_w, rng)

                if len(keras_weights) > 1:
                    raw_b = np.asarray(keras_weights[1], dtype=np.float64).ravel()
                    quant_b = np.floor(raw_b).astype(DRAM_DTYPE)
                    if layer_param.kernel_size[0] != 1:
                        self.bias[name][: quant_b.size] = quant_b[: layer_param.filters]
                    else:
                        self.bias[name][0] = quant_b[0]

            elif "conv" in lname:
                keras_weights = layers_seq[layer_idx].weights
                # Keras shape: (kx, ky, in_c, out_f) -> (in_c, out_f, ky, kx)
                raw_w = np.asarray(keras_weights[0], dtype=np.float64)
                quant_w = np.floor(127.0 * np.transpose(raw_w, (2, 3, 1, 0))).astype(DRAM_DTYPE)

                if sparse_wghts:
                    c_idx = np.arange(quant_w.shape[0])[:, None, None, None]
                    f_idx = np.arange(quant_w.shape[1])[None, :, None, None]
                    y_idx = np.arange(quant_w.shape[2])[None, None, :, None]
                    x_idx = np.arange(quant_w.shape[3])[None, None, None, :]
                    sparse_mask = ((c_idx + f_idx + x_idx + y_idx) & 1) == 0

                    _replace_zeros_with_unit_signs(quant_w, rng)
                    quant_w[sparse_mask] = 0
                else:
                    _replace_zeros_with_unit_signs(quant_w, rng)

                self.weights[name] = quant_w
                # Hardware test pattern: constant bias = 1 per filter
                self.bias[name].fill(1)

            elif "dense" in lname or "gemm" in lname or "matmul" in lname:
                quant_w = rng.integers(
                    -64, 63, size=(layer_param.filters, layer_param.iact_size_x), dtype=DRAM_DTYPE
                )
                if sparse_wghts:
                    f_idx = np.arange(quant_w.shape[0])[:, None]
                    c_idx = np.arange(quant_w.shape[1])[None, :]
                    sparse_mask = ((c_idx + f_idx) & 1) == 0

                    _replace_zeros_with_unit_signs(quant_w, rng)
                    quant_w[sparse_mask] = 0
                else:
                    _replace_zeros_with_unit_signs(quant_w, rng)

                self.weights[name] = quant_w
                self.bias[name] = np.arange(1, layer_param.filters + 1, dtype=DRAM_DTYPE)

            # === INPUT FEATURE MAP INITIALIZATION (FIRST LAYER) ===
            if prev_layers_map[name] == ["input_layer"]:
                str_name = str(name)
                if "conv" in str_name:
                    shape = (
                        layer_param.input_shape[3],
                        layer_param.input_shape[1],
                        layer_param.input_shape[2],
                    )
                    iacts = rng.integers(-32, 31, size=shape, dtype=DRAM_DTYPE)
                    if sparse_iacts:
                        c_idx = np.arange(shape[0])[:, None, None]
                        x_idx = np.arange(shape[1])[None, :, None]
                        y_idx = np.arange(shape[2])[None, None, :]
                        sparse_mask = ((c_idx + x_idx + y_idx) & 1) == 0

                        _replace_zeros_with_unit_signs(iacts, rng)
                        iacts[sparse_mask] = 0
                    else:
                        _replace_zeros_with_unit_signs(iacts, rng)

                    self.fmap[name][0] = iacts

                elif "dense" in lname or "gemm" in lname or "matmul" in lname:
                    iacts = rng.integers(-16, 15, size=(layer_param.iact_size_x,), dtype=DRAM_DTYPE)
                    self.fmap[name][0] = _replace_zeros_with_unit_signs(iacts, rng)