# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""Weight-sparsity sweep for PE_cluster, at one fixed shape and seed.

Why this exists
---------------
test_PE_CLUSTER.py's optional extended sweep has 32256 parametrisations, so the sparse failures are
scattered through a matrix nobody runs in full and the *shape* of the fault is
invisible. Its default is now a small smoke suite with pinned reproducers.
Holding shape and seed fixed and sweeping only SPARSE_WGHT turns the
same bug into a monotone curve, which is far more diagnostic than a list of
failing ids:

    PARALLEL_MACS = 2, SPARSE_WGHT  0..30 %   pass
    PARALLEL_MACS = 2, SPARSE_WGHT 40..60 %   fail
    PARALLEL_MACS = 1, every level            pass

Measured 2026-09-16: 3 of 14 fail, the threshold sitting exactly at 40 %. At
40 % a single psum position is wrong in 3 of 4 PE columns; by 60 % every
position is wrong, and the DUT always comes out *short* rather than corrupted.
Severity rising with the density of zeros is the signature of the weight
zero-skip path rather than the MACs.

Note this shape does not reproduce the SimTimeoutError hangs that appear in the
full matrix (those need PARALLEL_MACS=1 together with a larger IACTSIZE_X);
here PARALLEL_MACS=1 passes throughout. Whether the hangs share this root cause
is still open - do not assume it.

Iact sparsity is held at 0 on purpose: one of the hanging cases in the full
matrix has SPARSE_IACT=0, so weight zeros alone are sufficient to trigger a
failure, and mixing both in would only blur the curve.

The whole sweep runs in well under a minute, so it is usable as a fix-verify
loop, unlike the full matrix.

Debug switches that pair with this test (see PE_cluster_tb.py):
  OPENEYE_PSUM_TERMS=1   on a mismatch, print every product feeding that psum
                         as (weight, iact, product, pe_y), and flag any single
                         term or mis-pairing that explains the difference.
  The SPAD encoder overflow guard runs unconditionally and reports any zero run
  that does not fit its overhead field - an undecodable stream would make any
  downstream psum comparison meaningless.

Usage:
    pytest test_PE_CLUSTER_sparse.py -v
    OPENEYE_PSUM_TERMS=1 pytest test_PE_CLUSTER_sparse.py -v -k "60-2"
"""

import logging
import os
import sys

import pytest

# Same path setup as test_PE_CLUSTER.py: the cocotb testbench and its helper
# are imported by bare module name, so this directory has to be importable.
sys.path.extend([os.path.abspath(os.getcwd()),
                 os.path.dirname(os.path.realpath(__file__))])
tests_dir = os.path.abspath(os.path.dirname(__file__))

import pe_cluster_test_utils as pctu
from cluster_simulator import run_cluster_simulation
from open_eye import hdl_dir, test_dir, vh_file_creator

logger = logging.getLogger("cocotb")

clk_cycle          = 10
clk_cycle_unit     = "ns"
clk_delay_in       = 100
clk_delay_unit_in  = "ps"
clk_delay_out      = 100
clk_delay_unit_out = "ps"

# One shape, one seed. These are the values from the smallest known failure in
# the full matrix (test_pe_cluster_conv[1-2-4-40-20-6-2-2]), reduced to the
# point where a single psum position fails and the arithmetic is still small
# enough to check by hand.
IACTSIZE_X  = 2
IACTSIZE_Y  = 2
WGHTSIZE_X  = 6
SEED        = 4
SPARSE_IACT = 0


@pytest.mark.parametrize("SPARSE_WGHT", [0, 10, 20, 30, 40, 50, 60])
@pytest.mark.parametrize("PARALLEL_MACS", [1, 2])
def test_pe_cluster_weight_sparsity(SPARSE_WGHT, PARALLEL_MACS, request):
    """One point on the weight-sparsity curve; SPARSITY_EN is always 1.

    SPARSITY_EN=0 is covered by the dense cases in test_PE_CLUSTER.py - with
    zero-skipping off the weight stream carries every value and this sweep
    would measure nothing.
    """
    toplevel = "PE_cluster"
    module   = "PE_cluster_tb"
    verilog_sources = pctu.get_verilog_sources(hdl_dir)

    nodeid = request.node.nodeid.replace("::", "_").replace("/", "_") \
                                .replace("[", "_").replace("]", "_")
    target_dir = os.path.join(test_dir, ".temp", nodeid)
    os.makedirs(target_dir, exist_ok=True)

    run_cluster_simulation(
        python_search=[tests_dir],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=target_dir,
        testcase="start_test_pe",
        defines={"NO_TRACE": "TRUE"},
        simulator="icarus",
        extra_env={
            "CLOCK_LEN":               str(clk_cycle),
            "CLOCK_UNIT":              clk_cycle_unit,
            "CLOCK_DELAY_INPUT":       str(clk_delay_in),
            "CLOCK_DELAY_UNIT_INPUT":  clk_delay_unit_in,
            "CLOCK_DELAY_OUTPUT":      str(clk_delay_out),
            "CLOCK_DELAY_UNIT_OUTPUT": clk_delay_unit_out,
            "IACTSIZE_X":    str(IACTSIZE_X),
            "IACTSIZE_Y":    str(IACTSIZE_Y),
            "WGHTSIZE_X":    str(WGHTSIZE_X),
            "WGHTSIZE_Y":    str(IACTSIZE_X * IACTSIZE_Y),
            "SPARSE_IACT":   str(SPARSE_IACT),
            "SPARSE_WGHT":   str(SPARSE_WGHT),
            "PARALLEL_MACS": str(PARALLEL_MACS),
            "SPARSITY_EN":   "1",
            "SEED":          str(SEED),
            "COCOTB_TRACE":  "1",
        },
    )
