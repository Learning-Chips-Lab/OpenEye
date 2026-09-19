# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Independent check of the Dense reference used by FPGA regressions."""
from types import SimpleNamespace

import numpy as np
import pytest
from open_eye import test_utils_main as tum


@pytest.mark.parametrize("k,n", [(4, 4), (8, 8), (4, 8), (8, 4), (16, 16), (31, 8), (63, 8)])
def test_dense_reference(k, n, monkeypatch):
    rng = np.random.default_rng(2026)
    activations = rng.integers(-128, 128, size=k, dtype=np.int64)
    weights = rng.integers(-128, 128, size=(n, k), dtype=np.int64)
    bias = rng.integers(-100, 101, size=n, dtype=np.int64)
    activations[0] = 0
    weights[0, :] = 0
    layer = SimpleNamespace(layer_name="Dense", iact_size_x=k, filters=n)
    dram = SimpleNamespace(weights=[None, weights], fmap=[None, activations], bias=[None, bias])

    def no_manager():
        pytest.fail("Dense reference unexpectedly started a multiprocessing manager")

    monkeypatch.setattr(tum.mp, "Manager", no_manager)
    actual = tum.collect_results(1, layer, dram, serial=True)
    assert [actual[i] for i in range(n)] == (weights @ activations + bias).tolist()
