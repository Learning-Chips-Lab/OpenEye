# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""DRAM simulation module for OpenEye accelerator testing.

This module provides the DRAMContents class which simulates DRAM memory storage
for the OpenEye neural network accelerator. It manages storage and initialization
of weights, biases, and feature maps (activations) across various layer types
including Conv2D, Depthwise Convolution, Dense, Pooling, and Flatten layers.

Key Features:
    - Automatic storage allocation based on layer parameters
    - INT8 quantization of model weights and biases
    - Support for sparse weight and activation patterns for testing
    - Random initialization of input data for simulation

Typical Usage:
    >>> dram = DRAMContents(model, layer_parameters)
    >>> dram.write_initial_data_to_dram(model, layer_parameters,
    ...                                 sparse_iacts=False, sparse_wghts=False)
"""

import math
import numpy as np

class DRAMContents(object):
    """DRAM contents storage for intermediate data during OpenEye accelerator tests.

    This class simulates DRAM memory used by the OpenEye accelerator during simulation.
    It manages storage for feature maps, weights, and biases across different neural
    network layers, supporting Conv2D, Depthwise, Dense, Pooling, and Flatten layers.

    The DRAM contents persist intermediate data between:
    - Different layers in the network
    - Different repetitions/iterations of the same layer
    - Input feature maps and output activations

    Attributes:
        fmap (list): List of feature maps for each layer, storing input/output activations.
                     Structure varies by layer type:
                     - Conv/Pooling: 3D arrays [channels][height][width]
                     - Dense: 1D arrays [features]
        weights (list): List of weight tensors for each layer. Structure varies by layer type:
                        - Conv: 4D arrays [input_channels][filters][kernel_h][kernel_w]
                        - Depthwise: 3D arrays [channels][kernel_h][kernel_w]
                        - Dense: 2D arrays [output_features][input_features]
        bias (list): List of bias vectors for each layer, one value per output channel/filter.

    """

    def __init__(self, model, layer_parameters, seed: int | None = None) -> None:
        """Initialize DRAM storage structures for all layers in the model.

        Creates appropriately shaped data structures for feature maps, weights, and
        biases based on the layer types and parameters. All values are initialized to zero.

        Args:
            model: The neural network model containing layer configurations.
            layer_parameters (list): List of layer parameter objects, each containing:
                - layer_name: Type of layer (Conv, Depthwise, Dense, Pooling, Flatten)
                - input_shape: Input tensor dimensions
                - output_shape: Output tensor dimensions
                - kernel_size: Convolution kernel dimensions (if applicable)
                - filters: Number of output filters (if applicable)
            seed (int | None): Optional RNG seed for deterministic initialization of any
                random values (weights for Dense layers and input activations). If None,
                randomness is not seeded.

        """
        # Set RNG seed if provided for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Initialize empty lists to hold data structures for each layer
        dram_fmap = []      # Feature maps (activations)
        dram_weights = []   # Weight tensors
        dram_bias = []      # Bias vectors

        # Iterate through all layers to create appropriately shaped storage structures
        for i in range(len(layer_parameters)):
            # === WEIGHT INITIALIZATION ===
            # Create weight storage based on layer type

            if "Depthwise" in str(layer_parameters[i].layer_name):
                # Depthwise Conv: separate kernel per input channel
                # Shape: [input_channels][kernel_height][kernel_width]
                dram_weights.append([[[0 for l in range(layer_parameters[i].kernel_size[1])]
                                    for k in range(layer_parameters[i].kernel_size[0])]
                                    for j in range(layer_parameters[i].input_shape[3])])
            elif "Conv" in str(layer_parameters[i].layer_name):
                # Standard Conv2D: full kernel for each input-output channel pair
                # Shape: [input_channels][output_filters][kernel_height][kernel_width]
                dram_weights.append([[[[0 for m in range(layer_parameters[i].kernel_size[1])]
                                    for l in range(layer_parameters[i].kernel_size[0])]
                                    for k in range(layer_parameters[i].filters)]
                                    for j in range(layer_parameters[i].input_shape[3])])
            elif "Dense" in str(layer_parameters[i].layer_name):
                # Fully connected layer: weight matrix
                # Shape: [output_features][input_features]
                dram_weights.append([[0 for m in range(layer_parameters[i].input_shape[3])]
                                    for l in range(layer_parameters[i].output_shape[3])])
            elif "Flat" in str(layer_parameters[i].layer_name):
                # Flatten layer: no weights needed (placeholder)
                dram_weights.append([0])
            elif "Pooling" in str(layer_parameters[i].layer_name):
                # Pooling layer: no weights needed (placeholder)
                dram_weights.append([0])

            # === BIAS INITIALIZATION ===
            # Create bias storage: one bias value per output channel/filter

            if "Depthwise" in str(layer_parameters[i].layer_name):
                # Depthwise Conv: one bias per kernel (typically matches kernel height)
                dram_bias.append([0 for k in range(layer_parameters[i].kernel_size[0])])
            elif "Conv" in str(layer_parameters[i].layer_name):
                # Standard Conv2D: one bias per output filter
                dram_bias.append([0 for m in range(layer_parameters[i].filters)])
            elif "Dense" in str(layer_parameters[i].layer_name):
                # Dense layer: one bias per output feature
                dram_bias.append([0 for m in range(layer_parameters[i].output_shape[3])])
            elif "Pooling" in str(layer_parameters[i].layer_name):
                # Pooling layer: one bias per output channel (width dimension)
                dram_bias.append([0 for m in range(layer_parameters[i].output_shape[2])])

            # === FEATURE MAP INITIALIZATION ===
            # Create input feature map storage for each layer

            if "Conv" in str(layer_parameters[i].layer_name):
                # Conv layer input: 3D tensor [channels][height][width]
                dram_fmap.append([[[0 for l in range(layer_parameters[i].input_shape[2])]
                                for k in range(layer_parameters[i].input_shape[1])]
                                for j in range(layer_parameters[i].input_shape[3])])
            elif "Dense" in str(layer_parameters[i].layer_name):
                # Dense layer input: 1D vector [features]
                dram_fmap.append([0 for j in range(layer_parameters[i].input_shape[3])])
            elif "Flat" in str(layer_parameters[i].layer_name):
                # Flatten layer input: 3D tensor (pre-flattening shape)
                dram_fmap.append([[[0 for l in range(layer_parameters[i].input_shape[2])]
                                for k in range(layer_parameters[i].input_shape[1])]
                                for j in range(layer_parameters[i].input_shape[3])])
            elif "Pooling" in str(layer_parameters[i].layer_name):
                # Pooling layer input: 3D tensor [channels][height][width]
                dram_fmap.append([[[0 for l in range(layer_parameters[i].input_shape[2])]
                                for k in range(layer_parameters[i].input_shape[1])]
                                for j in range(layer_parameters[i].input_shape[3])])

        # Add output feature map storage for the final layer
        for i in [len(layer_parameters)-1]:
            if "Conv" in str(layer_parameters[i].layer_name):
                # Conv output: 3D tensor [channels][height][width]
                dram_fmap.append([[[0 for l in range(layer_parameters[i].output_shape[2])]
                                    for k in range(layer_parameters[i].output_shape[1])]
                                    for j in range(layer_parameters[i].output_shape[3])])
            elif "Dense" in str(layer_parameters[i].layer_name):
                # Dense output: 1D vector [features]
                dram_fmap.append([0 for l in range(layer_parameters[i].output_shape[3])])

        # Assign initialized structures to instance attributes
        self.fmap = dram_fmap
        self.weights = dram_weights
        self.bias = dram_bias

    def write_initial_data_to_dram(self, model, layer_parameters, sparse_iacts, sparse_wghts, seed: int | None = None):
        """Populate DRAM with initial weights, biases, and input feature maps from the model.

        This method loads the trained model parameters into the DRAM simulation storage.
        Weights are quantized to 8-bit integers (INT8 format, range -128 to 127).
        Optionally creates sparse tensors by zeroing out selected elements.

        Args:
            model: The trained neural network model with layer weights and biases.
            layer_parameters (list): List of layer parameter objects describing each layer.
            sparse_iacts (bool): If True, creates sparse input activations by zeroing
                                 elements at positions where (c+x+y) % 2 == 0.
            sparse_wghts (bool): If True, creates sparse weights by zeroing elements
                                 based on positional indices.
            seed (int | None): Optional RNG seed to make random initialization deterministic.
                If provided, it overrides any seed set in the constructor for this method call.

        Note:
            - Weights are quantized by multiplying by 127 and flooring to nearest integer
            - Zero weights are replaced with random values (-1 or 1) to avoid true zeros
            - Sparse patterns use modulo arithmetic on spatial/channel indices

        """
        # Optionally set RNG seed for deterministic behavior of this method
        if seed is not None:
            np.random.seed(seed)
        # === WEIGHT LOADING ===
        # Load and quantize weights from the trained model for each layer
        for l in range(len(layer_parameters)):
            if "Depthwise" in str(layer_parameters[l].layer_name):
                # Load depthwise convolution weights
                # Iterate: channels -> kernel_height -> kernel_width
                for c in range(layer_parameters[l].input_shape[3]):
                    for x in range(layer_parameters[l].kernel_size[0]):
                        for y in range(layer_parameters[l].kernel_size[1]):
                            # Quantize weight to INT8: multiply by 127 and floor
                            self.weights[l][c][x][y] = int(math.floor(float(127*model.layers[l].weights[0][x][y][c])))
                            # Replace zeros with random -1 or 1 to avoid true zero weights
                            if (self.weights[l][c][x][y] == 0):
                                self.weights[l][c][x][y] = int(np.random.choice([-1, 1]))
            elif "Conv" in str(layer_parameters[l].layer_name):
                # Load standard Conv2D weights
                # Iterate: input_channels -> output_filters -> kernel_height -> kernel_width
                for c in range(layer_parameters[l].input_shape[3]):
                    for f in range(layer_parameters[l].filters):
                        for x in range(layer_parameters[l].kernel_size[0]):
                            for y in range(layer_parameters[l].kernel_size[1]):
                                # Quantize weight to INT8
                                self.weights[l][c][f][x][y] = int(math.floor(float(127*model.layers[l].weights[0][x][y][c][f])))
                                # Apply sparsity pattern if requested: zero out elements where sum of indices is even
                                if (sparse_wghts & (((c+f+x+y) % 2) == 0)):
                                    self.weights[l][c][f][x][y] = 0
                                else:
                                    # Replace zeros with random -1 or 1
                                    if (self.weights[l][c][f][x][y] == 0):
                                        self.weights[l][c][f][x][y] = int(np.random.choice([-1, 1]))
            elif "Dense" in str(layer_parameters[l].layer_name):
                # Load Dense (fully connected) layer weights
                # Note: Uses random weights instead of model weights
                # Iterate: input_features -> output_features
                for c in range(layer_parameters[l].input_shape[3]):
                    for x in range(layer_parameters[l].output_shape[3]):
                        # Generate random INT8 weights for Dense layer
                        self.weights[l][x][c] = np.random.randint(-128, 127)
                        # Apply sparsity pattern if requested
                        if (sparse_wghts & (((c+l+x) % 2) == 0)):
                            self.weights[l][x][c] = 0
                        else:
                            # Replace zeros with random -1 or 1
                            if (self.weights[l][x][c] == 0):
                                self.weights[l][x][c] = int(np.random.choice([-1, 1]))

        # === BIAS LOADING ===
        # Load and quantize bias values from the trained model
        for l in range(len(layer_parameters)):
            if "Depthwise" in str(layer_parameters[l].layer_name):
                # Load depthwise convolution biases
                if (layer_parameters[l].kernel_size[0] != 1):
                    # Multiple bias values (one per kernel row)
                    for x in range(layer_parameters[l].kernel_size[0]):
                        self.bias[l][x] = int(math.floor(float(model.layers[l].weights[1][x])))
                else:
                    # Single bias value for 1x1 kernels
                    self.bias[l][0] = int(math.floor(float(model.layers[l].weights[1])))
            elif "Conv" in str(layer_parameters[l].layer_name):
                # Load Conv2D biases: one bias per output filter
                for x in range(layer_parameters[l].filters):
                    self.bias[l][x] = int(math.floor(float(model.layers[l].weights[1][x])))
                    # Alternative: self.bias[l][x] = x * (-1)  # Uncomment for test pattern
            elif "Dense" in str(layer_parameters[l].layer_name):
                # Load Dense layer biases: one bias per output feature
                # Note: Uses sequential test values (c+1) instead of model biases
                for c in range(layer_parameters[l].output_shape[3]):
                    # Alternative: self.bias[l][c] = int(math.floor(float(model.layers[l].weights[1][c])))
                    self.bias[l][c] = int(c + 1)  # Test pattern: 1, 2, 3, ...

        # === INPUT FEATURE MAP INITIALIZATION ===
        # Initialize input feature maps for the first layer with random values

        if "Conv" in str(layer_parameters[0].layer_name):
            # Initialize 3D input feature map for Conv layers
            # Iterate: channels -> height -> width
            for c in range(layer_parameters[0].input_shape[3]):
                for x in range(layer_parameters[0].input_shape[1]):
                    for y in range(layer_parameters[0].input_shape[2]):
                        # Generate random INT8 activation values
                        self.fmap[0][c][x][y] = np.random.randint(-128, 127)
                        self.fmap[0][c][x][y] = np.random.randint(-32, 31)
                        # Apply sparsity pattern if requested: zero elements where sum is even
                        if (sparse_iacts & (((c+x+y) % 2) == 0)):
                            self.fmap[0][c][x][y] = 0
                        else:
                            # Replace zeros with random -1 or 1
                            if (self.fmap[0][c][x][y] == 0):
                                self.fmap[0][c][x][y] = int(np.random.choice([-1, 1]))
        elif "Dense" in str(layer_parameters[0].layer_name):
            # Initialize 1D input feature map for Dense layers
            for c in range(layer_parameters[0].input_shape[3]):
                # Generate random INT8 activation values (no sparsity for Dense)
                self.fmap[0][c] = np.random.randint(-128, 127)