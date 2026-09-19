# This file is part of the OpenEye project.
# SPDX-License-Identifier: SHL-2.1
"""Independent checks for the convolution milestone's model and DMA oracle."""
from io import StringIO
from types import SimpleNamespace

import pytest

from open_eye import data_create, test_utils_main as tum


@pytest.mark.parametrize("mode,count", [("Convolution_Single", 1), ("Convolution_Stack", 2)])
def test_conv_model_layer_count(mode, count):
    model = data_create.create_layer(mode, 4, 3, 3, 8, 1, (1, 1), 4, 1)
    assert len(model.layers) == count
    assert tuple(model.layers[-1].output.shape) == (None, 8, 1, 4)


@pytest.mark.parametrize("psum_width", [20, 32])
def test_conv_dma_reference_packing(psum_width):
    params = SimpleNamespace(DMA_BITWIDTH=64, DATA_PSUM_BITWIDTH=psum_width,
                             Clusters_X=2, Clusters_Y=2, Psum_Routers=4)
    layer = SimpleNamespace(filters=4, used_psum_per_PE=4,
                            different_kernels_per_calculation=1,
                            iact_size_x=8, iact_size_y=1, psum_size_x=8,
                            psum_size_y=1, add_up=0, psum_add_up=0,
                            y_lines_per_calculation=1, used_Y_cluster=1,
                            needed_refreshes_mx=[[1]], diff_iact_layer=1,
                            iact_x_line_repetitions=1)
    # Distinct signed values at every coordinate catch dropped bits, missing
    # values, and confusion between filters and spatial positions.
    values = [[(-1 if x % 2 else 1) * (100*f + x + 1) for x in range(8)]
              for f in range(4)]
    results = [[[v] for v in row] for row in values]
    output = StringIO()
    tum.calculate_conv_serial(params, layer, results, output)
    lines = output.getvalue().splitlines()
    assert len(lines) == 32
    assert all(len(line) == 64 for line in lines)
    mask = (1 << psum_width) - 1
    expected = []
    for row in values:
        expected.extend((row[x] & mask) | ((row[x+1] & mask) << psum_width)
                        for x in range(0, 8, 2))
        expected.extend([0] * 4)  # Unused second cluster row.
    assert [int(line, 2) for line in lines] == expected
