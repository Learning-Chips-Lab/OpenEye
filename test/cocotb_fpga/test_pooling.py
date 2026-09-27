# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
"""Small conv -> 2x2 max-pool -> conv nets through the FPGA flow.

The pooled output is written back into the iact buffer and read by the next
conv layer, as in the MNIST net, but on an 8-pixel-wide input that simulates in
about a minute. The RTL builds either max or average pooling (compile-time
MAX_POOLING / AVERAGE_POOLING, default average), so these tests select max.
"""
import pytest

import test_conv_const as tc


@pytest.mark.parametrize("width", [8, 6])
@pytest.mark.parametrize("height", [2, 4])
def test_conv_maxpool_conv(width, height, request, monkeypatch):
    """Ramp input distinguishes pixels, so a misplaced or wrong pooled pixel fails.

    Width 6 pools to an odd width of 3, so a pooled row fills only half of its
    last two-pixel word (the MNIST net pools 14x14 down to 7x7).
    """
    monkeypatch.setattr(tc, "INPUT_SIZE_X", width)
    real_run = tc.cocotb_test.simulator.run

    def run(**kw):
        # LOGGER_LEVEL above DEBUG skips the raw DMA-file comparison. For an
        # output width that is not a multiple of the lane count the file holds
        # zeros in the unused lanes while the hardware computes them from the
        # padded input; the decoded output values are still checked.
        kw["extra_env"] = dict(kw["extra_env"], OPENEYE_EXPECT_LAYERS="3", LOGGER_LEVEL="20")
        kw["parameters"] = {"MAX_POOLING": 1, "AVERAGE_POOLING": 0}
        return real_run(**kw)

    monkeypatch.setattr(tc.cocotb_test.simulator, "run", run)
    tc._run_conv_const("Conv_Pool_Conv", 8, 2, num_glb_iact=1, input_channels=2,
                       request=request, input_height=height, ramp_iacts=True)
