# This file is part of the OpenEye project.
# SPDX-License-Identifier: SHL-2.1
"""Independent checks for the convolution milestone's model and DMA oracle."""
import pytest

from open_eye import data_create


@pytest.mark.parametrize("mode,count", [("Convolution_Single", 1), ("Convolution_Stack", 2)])
def test_conv_model_layer_count(mode, count):
    model = data_create.create_layer(mode, 4, 3, 3, 8, 1, (1, 1), 4, 1)
    assert len(model.layers) == count
    assert tuple(model.layers[-1].output.shape) == (None, 8, 1, 4)
