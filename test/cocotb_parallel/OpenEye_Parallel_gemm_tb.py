# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""
Output-stationary (GEMM) dataflow plumbing test for OpenEye_Parallel.

Verifies the gemm_mode configuration path that was added for the
output-stationary dataflow:

  gemm_mode_i (port)
    -> gemm_mode_i_reg / gemm_mode_reg (latched with status_reg_enable_i)
      -> OpenEye_Cluster.gemm_mode_i (all clusters)
        -> PE_cluster.gemm_mode_i
          -> per-PE iact_sel_w override: PE row j reads iact GLB bank j

The numerical correctness of the output-stationary GEMM datapath itself is
covered by the PE_cluster level tests
(test/cocotb_PE_cluster/test_PE_CLUSTER_GEMM.py, approaches 1-3); this test
covers the configuration plumbing through the OpenEye_Parallel hierarchy.

Two checks per scenario:
  1. gemm_mode_reg follows gemm_mode_i only while status_reg_enable_i is set
     (config-latch semantics identical to data_mode_i).
  2. With gemm_mode=1 every sampled PE(row j) sees iact_sel_w == j in every
     cluster; with gemm_mode=0 iact_sel_w follows iact_choose_i again.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer

clk_cycle = int(os.environ.get("CLOCK_LEN", "10"))
clk_cycle_unit = os.environ.get("CLOCK_UNIT", "ns")


async def _cycles(dut, n):
    for _ in range(n):
        await Timer(clk_cycle, unit=clk_cycle_unit)


async def _apply_reset(dut):
    dut.rst_ni.value = 0
    # Drive all control inputs to benign defaults
    dut.compute_i.value = 0
    dut.status_reg_enable_i.value = 0
    dut.data_mode_i.value = 0
    dut.gemm_mode_i.value = 0
    dut.fraction_bit_i.value = 0
    dut.needed_cycles_i.value = 0
    dut.needed_x_cls_i.value = 0
    dut.needed_y_cls_i.value = 0
    dut.needed_iact_cycles_i.value = 0
    dut.filters_i.value = 0
    dut.iact_size_x_i.value = 0
    dut.iact_addr_len_i.value = 0
    dut.wght_addr_len_i.value = 0
    dut.bano_cluster_mode_i.value = 0
    dut.af_cluster_mode_i.value = 0
    dut.pooling_cluster_mode_i.value = 0
    dut.delay_psum_glb_i.value = 0
    dut.input_activations_i.value = 0
    dut.stride_x_i.value = 0
    dut.stride_y_i.value = 0
    dut.kernel_per_pe_cluster_i.value = 0
    dut.iact_x_line_repetitions_i.value = 0
    dut.kernel_size_x_i.value = 0
    dut.kernel_size_y_i.value = 0
    dut.compute_mask_i.value = 0
    dut.iact_choose_i.value = 0
    dut.psum_choose_i.value = 0
    dut.router_mode_iact_i.value = 0
    dut.router_mode_wght_i.value = 0
    dut.router_mode_psum_i.value = 0
    dut.needed_psum_storage_cycles_i.value = 0
    dut.needed_iact_channel_cycles_i.value = 0
    dut.psum_transmitted_i.value = 0
    dut.iact_data_i.value = 0
    dut.iact_enable_i.value = 0
    dut.wght_data_i.value = 0
    dut.wght_enable_i.value = 0
    dut.psum_data_i.value = 0
    dut.psum_enable_i.value = 0
    dut.psum_ready_i.value = 0
    await _cycles(dut, 5)
    dut.rst_ni.value = 1
    await _cycles(dut, 5)


async def _latch_config(dut, gemm_mode):
    """Latch a configuration exactly like the mapper flow does."""
    dut.gemm_mode_i.value = gemm_mode
    dut.status_reg_enable_i.value = 1
    await _cycles(dut, 4)
    dut.status_reg_enable_i.value = 0
    # keep the port value stable one more cycle, then drop it to prove the
    # latched register (not the port) drives the clusters
    await _cycles(dut, 1)
    dut.gemm_mode_i.value = 0
    await _cycles(dut, 2)


def _sample_iact_sel(dut, cl_x, cl_y, pe_x, pe_y):
    """Resolve the per-PE iact_sel_w wire inside the generate hierarchy."""
    cluster = dut.gen_x[cl_x].gen_y[cl_y].OpenEye_Cluster
    pe_scope = cluster.pe_cluster.gen_X[pe_x].gen_Y[pe_y]
    return int(pe_scope.iact_sel_w.value)


@cocotb.test()
async def gemm_mode_plumbing_test(dut):
    """Check gemm_mode propagation from the port into every PE cluster."""
    cocotb.start_soon(Clock(dut.clk_i, clk_cycle, unit=clk_cycle_unit).start())
    await _apply_reset(dut)

    cluster_cols = int(dut.CLUSTER_COLUMNS.value)
    cluster_rows = int(dut.CLUSTER_ROWS.value)
    pe_cols = int(dut.PE_COLUMNS.value)
    pe_rows = int(dut.PE_ROWS.value)

    # ------------------------------------------------------------------
    # 1. Row-stationary default: iact_sel_w follows iact_choose_i
    # ------------------------------------------------------------------
    # Drive a recognizable non-GEMM pattern: every PE selects bank 1
    sel_bits = (int(dut.NUM_GLB_IACT.value) + 1).bit_length() - 1
    pes = pe_cols * pe_rows
    pattern = 0
    for pe in range(pes * cluster_cols * cluster_rows):
        pattern |= 1 << (pe * sel_bits)
    dut.iact_choose_i.value = pattern
    await _latch_config(dut, gemm_mode=0)

    assert int(dut.gemm_mode_reg.value) == 0, "gemm_mode_reg must stay 0 in RS mode"
    for cl_x in range(cluster_cols):
        for cl_y in range(cluster_rows):
            for pe_y in range(pe_rows):
                sel = _sample_iact_sel(dut, cl_x, cl_y, 0, pe_y)
                assert sel == 1, (
                    f"RS mode: cluster({cl_x},{cl_y}) PE row {pe_y} iact_sel_w={sel}, "
                    "expected 1 (from iact_choose_i)")

    # ------------------------------------------------------------------
    # 2. Output-stationary mode: PE row j must read iact GLB bank j
    # ------------------------------------------------------------------
    await _latch_config(dut, gemm_mode=1)

    assert int(dut.gemm_mode_reg.value) == 1, \
        "gemm_mode_reg did not latch gemm_mode_i during status_reg_enable_i"
    for cl_x in range(cluster_cols):
        for cl_y in range(cluster_rows):
            for pe_x in range(pe_cols):
                for pe_y in range(pe_rows):
                    sel = _sample_iact_sel(dut, cl_x, cl_y, pe_x, pe_y)
                    assert sel == pe_y, (
                        f"OS mode: cluster({cl_x},{cl_y}) PE({pe_x},{pe_y}) "
                        f"iact_sel_w={sel}, expected row index {pe_y}")

    # ------------------------------------------------------------------
    # 3. Back to row-stationary: override must disappear again
    # ------------------------------------------------------------------
    await _latch_config(dut, gemm_mode=0)
    assert int(dut.gemm_mode_reg.value) == 0, "gemm_mode_reg must clear again"
    sel = _sample_iact_sel(dut, 0, 0, 0, 2)
    assert sel == 1, f"RS restore: iact_sel_w={sel}, expected 1"
