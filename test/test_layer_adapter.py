#!/usr/bin/env python3
# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Test script to verify KerasLayerAdapter provides all required attributes.

This script tests that the KerasLayerAdapter correctly exposes all attributes
needed by LayerParameters for each layer type.
"""

import numpy as np
from layer_adapter import KerasLayerAdapter


def test_conv2d_layer():
    """Test Conv2D layer has all required attributes."""
    print("\n" + "="*70)
    print("Testing Conv2D Layer Attributes")
    print("="*70)

    # Create a mock Conv2D layer
    class MockConv2D:
        def __init__(self):
            self.name = 'conv2d'
            self.input_shape = (1, 28, 28, 1)
            self.output_shape = (1, 28, 28, 32)
            self.weights = [
                np.random.randn(3, 3, 1, 32),  # kernel
                np.random.randn(32)             # bias
            ]
            self.kernel_size = (3, 3)
            self.strides = (1, 1)
            self.filters = 32
            self.relu = True
            self.batchnorm = False
            self.quantization_factor = [(1234567, 10)] * 32
            self.zero_point = 0
            self.store_in_psum = 0
            self.skip_psum = 0

    mock_layer = MockConv2D()
    adapted = KerasLayerAdapter(mock_layer)

    # Required attributes for Conv2D
    required_attrs = [
        'name',
        'input', 'output',  # Tensor adapters
        'kernel', 'weights',  # Weight data
        'kernel_size', 'strides', 'filters', 'padding',  # Conv params
        'relu', 'batchnorm',  # Activations
        'quantization_factor', 'zero_point',  # Quantization
        'store_in_psum', 'skip_psum'  # Control flags
    ]

    print("\nChecking Conv2D attributes:")
    all_present = True
    for attr in required_attrs:
        has_attr = hasattr(adapted, attr)
        status = "✅" if has_attr else "❌"
        value = getattr(adapted, attr, "MISSING")
        print(f"  {status} {attr:25s} = {str(value)[:50]}")
        if not has_attr:
            all_present = False

    # Test shape access
    print(f"\n  Input shape:  {adapted.input.shape}")
    print(f"  Output shape: {adapted.output.shape}")
    print(f"  Kernel shape: {adapted.kernel.shape}")

    if all_present:
        print("\n✅ Conv2D layer has ALL required attributes!")
    else:
        print("\n❌ Conv2D layer is MISSING some attributes!")

    return all_present


def test_dense_layer():
    """Test Dense layer has all required attributes."""
    print("\n" + "="*70)
    print("Testing Dense Layer Attributes")
    print("="*70)

    # Create a mock Dense layer
    class MockDense:
        def __init__(self):
            self.name = 'dense'
            self.input_shape = (1, 128)
            self.output_shape = (1, 10)
            self.weights = [
                np.random.randn(128, 10),  # weight matrix
                np.random.randn(10)         # bias
            ]
            self.quantization_factor = [(2345678, 12)] * 10
            self.zero_point = 0

    mock_layer = MockDense()
    adapted = KerasLayerAdapter(mock_layer)

    # Required attributes for Dense
    required_attrs = [
        'name',
        'input', 'output',  # Tensor adapters
        'kernel', 'weights',  # Weight data
        'units', 'use_bias',  # Dense params
        'quantization_factor', 'zero_point'  # Quantization
    ]

    print("\nChecking Dense attributes:")
    all_present = True
    for attr in required_attrs:
        has_attr = hasattr(adapted, attr)
        status = "✅" if has_attr else "❌"
        value = getattr(adapted, attr, "MISSING")
        print(f"  {status} {attr:25s} = {str(value)[:50]}")
        if not has_attr:
            all_present = False

    # Test shape access
    print(f"\n  Input shape:  {adapted.input.shape}")
    print(f"  Output shape: {adapted.output.shape}")
    print(f"  Kernel shape: {adapted.kernel.shape}")

    if all_present:
        print("\n✅ Dense layer has ALL required attributes!")
    else:
        print("\n❌ Dense layer is MISSING some attributes!")

    return all_present


def test_maxpooling_layer():
    """Test MaxPooling2D layer has all required attributes."""
    print("\n" + "="*70)
    print("Testing MaxPooling2D Layer Attributes")
    print("="*70)

    # Create a mock MaxPooling layer
    class MockMaxPooling:
        def __init__(self):
            self.name = 'max_pooling2d'
            self.input_shape = (1, 28, 28, 32)
            self.output_shape = (1, 14, 14, 32)
            self.pool_size = (2, 2)
            self.strides = (2, 2)

    mock_layer = MockMaxPooling()
    adapted = KerasLayerAdapter(mock_layer)

    # Required attributes for MaxPooling
    required_attrs = [
        'name',
        'input', 'output',  # Tensor adapters
        'pool_size', 'strides',  # Pooling params
        'weights'  # Should be empty list
    ]

    print("\nChecking MaxPooling2D attributes:")
    all_present = True
    for attr in required_attrs:
        has_attr = hasattr(adapted, attr)
        status = "✅" if has_attr else "❌"
        value = getattr(adapted, attr, "MISSING")
        print(f"  {status} {attr:25s} = {str(value)[:50]}")
        if not has_attr:
            all_present = False

    # Test shape access
    print(f"\n  Input shape:  {adapted.input.shape}")
    print(f"  Output shape: {adapted.output.shape}")
    print(f"  Pool size:    {adapted.pool_size}")

    if all_present:
        print("\n✅ MaxPooling2D layer has ALL required attributes!")
    else:
        print("\n❌ MaxPooling2D layer is MISSING some attributes!")

    return all_present


def test_maxpooling_inferred():
    """Test MaxPooling2D with inferred pool_size."""
    print("\n" + "="*70)
    print("Testing MaxPooling2D Layer (Inferred Pool Size)")
    print("="*70)

    # Create a mock MaxPooling layer without explicit pool_size
    class MockMaxPoolingNoSize:
        def __init__(self):
            self.name = 'max_pooling2d'
            self.input_shape = (1, 28, 28, 32)
            self.output_shape = (1, 14, 14, 32)
            # No pool_size attribute - should be inferred

    mock_layer = MockMaxPoolingNoSize()
    adapted = KerasLayerAdapter(mock_layer)

    print(f"\n  Input shape:  {adapted.input.shape}")
    print(f"  Output shape: {adapted.output.shape}")
    print(f"  Inferred pool_size: {adapted.pool_size}")
    print(f"  Inferred strides:   {adapted.strides}")

    # Verify correct inference
    expected_pool = (2, 2)
    if adapted.pool_size == expected_pool:
        print(f"\n✅ Pool size correctly inferred as {expected_pool}")
        return True
    else:
        print(f"\n❌ Pool size incorrectly inferred! Expected {expected_pool}, got {adapted.pool_size}")
        return False


def test_flatten_layer():
    """Test Flatten layer."""
    print("\n" + "="*70)
    print("Testing Flatten Layer Attributes")
    print("="*70)

    # Create a mock Flatten layer
    class MockFlatten:
        def __init__(self):
            self.name = 'flatten'
            self.input_shape = (1, 7, 7, 64)
            self.output_shape = (1, 3136)

    mock_layer = MockFlatten()
    adapted = KerasLayerAdapter(mock_layer)

    print(f"\n  Name:         {adapted.name}")
    print(f"  Input shape:  {adapted.input.shape}")
    print(f"  Output shape: {adapted.output.shape}")
    print(f"  Has weights:  {len(adapted.weights) > 0}")

    if adapted.name == 'flatten' and len(adapted.weights) == 0:
        print("\n✅ Flatten layer correctly configured!")
        return True
    else:
        print("\n❌ Flatten layer has issues!")
        return False


def test_attribute_access_patterns():
    """Test various attribute access patterns used in code."""
    print("\n" + "="*70)
    print("Testing Attribute Access Patterns")
    print("="*70)

    # Create a mock Conv2D layer
    class MockConv2D:
        def __init__(self):
            self.name = 'conv2d'
            self.input_shape = (1, 28, 28, 1)
            self.output_shape = (1, 28, 28, 32)
            self.weights = [
                np.random.randn(3, 3, 1, 32),
                np.random.randn(32)
            ]
            self.kernel_size = (3, 3)
            self.strides = (1, 1)
            self.filters = 32

    mock_layer = MockConv2D()
    adapted = KerasLayerAdapter(mock_layer)

    # Test common access patterns
    patterns = {
        "layer.name": lambda: adapted.name,
        "layer.input.shape": lambda: adapted.input.shape,
        "layer.output.shape": lambda: adapted.output.shape,
        "layer.kernel.shape": lambda: adapted.kernel.shape,
        "layer.kernel_size": lambda: adapted.kernel_size,
        "layer.strides": lambda: adapted.strides,
        "layer.filters": lambda: adapted.filters,
        "hasattr(layer, 'name')": lambda: hasattr(adapted, 'name'),
        "hasattr(layer, 'filters')": lambda: hasattr(adapted, 'filters'),
        "layer.get_weights()": lambda: len(adapted.get_weights()),
    }

    print("\nTesting access patterns:")
    all_work = True
    for pattern, accessor in patterns.items():
        try:
            result = accessor()
            print(f"  ✅ {pattern:30s} → {str(result)[:40]}")
        except Exception as e:
            print(f"  ❌ {pattern:30s} → ERROR: {e}")
            all_work = False

    if all_work:
        print("\n✅ All access patterns work correctly!")
    else:
        print("\n❌ Some access patterns failed!")

    return all_work


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("LAYER ADAPTER ATTRIBUTE TEST SUITE")
    print("="*70)
    print("\nThis test verifies that KerasLayerAdapter provides all attributes")
    print("required by LayerParameters for each layer type.")

    results = {
        "Conv2D": test_conv2d_layer(),
        "Dense": test_dense_layer(),
        "MaxPooling2D": test_maxpooling_layer(),
        "MaxPooling2D (inferred)": test_maxpooling_inferred(),
        "Flatten": test_flatten_layer(),
        "Access Patterns": test_attribute_access_patterns(),
    }

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")

    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED - Please review above output")
    print("="*70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
