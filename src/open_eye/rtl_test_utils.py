# This file is part of the OpenEye project.
# © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.

"""RTL test utilities for the OpenEye project.

This module provides utility functions for testing RTL designs using cocotb:
- Setting input signals with proper timing
- Resetting DUT signals
- Writing input activations, weights and biases
- Comparing output streams against NumPy-backed DRAM buffers
- Managing DMA transfers and hardware diagnostics
"""

import logging
import math
import os
import random
import cocotb
import numpy as np
from cocotb.triggers import FallingEdge, RisingEdge, Timer
import open_eye.stream_dicts as strdic

logger = logging.getLogger("cocotb")


async def set_input(port_timings, signal, new_value, multiple_dim=False, array_index=None, array_max_index=None):
    """Set an input signal value with proper timing."""
    await Timer(port_timings.clk_delay_in, unit=port_timings.clk_delay_unit_in)
    if multiple_dim and array_index is not None:
        if cocotb.SIM_NAME == "Icarus Verilog":
            rev_max = list(reversed(array_max_index or []))
            rev_idx = list(reversed(array_index))
            index_of_signal = 0
            stride = 1
            for current_index in range(len(rev_idx)):
                index_of_signal += rev_idx[current_index] * stride
                stride *= rev_max[current_index]
            signal = signal[index_of_signal]
        else:
            for idx in array_index:
                signal = signal[idx]
    if isinstance(new_value, (np.integer, np.bool_)):
        new_value = int(new_value)
    signal.value = new_value


async def reset_all_signals(ptp, dut, serial):
    """Reset all signals of the DUT to a known initial state."""
    cocotb.start_soon(set_input(ptp, dut.rst_ni, 0))

    if serial == 0:
        cocotb.start_soon(set_input(ptp, dut.compute_i, 0))
        cocotb.start_soon(set_input(ptp, dut.wght_data_i, 0))
        cocotb.start_soon(set_input(ptp, dut.wght_enable_i, 0))
        cocotb.start_soon(set_input(ptp, dut.iact_data_i, 0))
        cocotb.start_soon(set_input(ptp, dut.iact_enable_i, 0))
        cocotb.start_soon(set_input(ptp, dut.psum_data_i, 0))
        cocotb.start_soon(set_input(ptp, dut.psum_enable_i, 0))
        cocotb.start_soon(set_input(ptp, dut.psum_ready_i, 0))
        cocotb.start_soon(set_input(ptp, dut.status_reg_enable_i, 0))
        cocotb.start_soon(set_input(ptp, dut.data_mode_i, 0))
        cocotb.start_soon(set_input(ptp, dut.fraction_bit_i, 0))
        cocotb.start_soon(set_input(ptp, dut.needed_cycles_i, 0))
        cocotb.start_soon(set_input(ptp, dut.needed_x_cls_i, 0))
        cocotb.start_soon(set_input(ptp, dut.needed_y_cls_i, 0))
        cocotb.start_soon(set_input(ptp, dut.needed_iact_cycles_i, 0))
        cocotb.start_soon(set_input(ptp, dut.filters_i, 0))
        cocotb.start_soon(set_input(ptp, dut.wght_addr_len_i, 0))
        cocotb.start_soon(set_input(ptp, dut.bano_cluster_mode_i, 0))
        cocotb.start_soon(set_input(ptp, dut.af_cluster_mode_i, 0))
        cocotb.start_soon(set_input(ptp, dut.pooling_cluster_mode_i, 0))
        cocotb.start_soon(set_input(ptp, dut.input_activations_i, 0))
        cocotb.start_soon(set_input(ptp, dut.iact_write_addr_t_i, 0))
        cocotb.start_soon(set_input(ptp, dut.iact_write_data_t_i, 0))
        cocotb.start_soon(set_input(ptp, dut.kernel_per_pe_cluster_i, 0))
        cocotb.start_soon(set_input(ptp, dut.compute_mask_i, 0))
        cocotb.start_soon(set_input(ptp, dut.router_mode_iact_i, 0))
        cocotb.start_soon(set_input(ptp, dut.router_mode_wght_i, 0))
        cocotb.start_soon(set_input(ptp, dut.router_mode_psum_i, 0))
        for opt_sig in ("delay_psum_glb_i", "gemm_mode_i", "raw_wght_i"):
            if hasattr(dut, opt_sig):
                cocotb.start_soon(set_input(ptp, getattr(dut, opt_sig), 0))
    else:
        cocotb.start_soon(set_input(ptp, dut.data_dma_i, 0))
        cocotb.start_soon(set_input(ptp, dut.enable_dma_i, 0))
    cocotb.start_soon(set_input(ptp, dut.ready_dma_i, 0))

    await Timer(ptp.clk_cycle, ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp, dut.rst_ni, 1))

    for _ in range(3):
        await Timer(ptp.clk_cycle, ptp.clk_cycle_unit)


async def send_stream(ptp, dut, stream, oep, lp, layer_repetition):
    """Send the configuration and data stream to the DUT."""
    if oep.SERIAL == 0:
        status_sec = stream[strdic.stream_parallel_dict["status"]]
        sdict = strdic.status_dict

        cocotb.start_soon(set_input(ptp, dut.status_reg_enable_i, 1))
        cocotb.start_soon(set_input(ptp, dut.data_mode_i, status_sec[sdict["data_mode"]]))

        if hasattr(dut, "gemm_mode_i"):
            gemm_mode_value = status_sec[sdict["gemm_mode"]]
            if gemm_mode_value == []:
                gemm_mode_value = 0
            cocotb.start_soon(set_input(ptp, dut.gemm_mode_i, gemm_mode_value))

        if hasattr(dut, "raw_wght_i"):
            raw_wght_value = 1 if (getattr(lp, "fully_connected", 0) or getattr(lp, "gemm_mode", 0)) else 0
            cocotb.start_soon(set_input(ptp, dut.raw_wght_i, raw_wght_value))
        cocotb.start_soon(set_input(ptp, dut.fraction_bit_i, status_sec[sdict["realfactor"]]))

        cocotb.start_soon(set_input(ptp, dut.needed_cycles_i, status_sec[sdict["needed_refreshes"]]))
        cocotb.start_soon(set_input(ptp, dut.needed_x_cls_i, status_sec[sdict["used_X_cluster"]]))
        cocotb.start_soon(set_input(ptp, dut.needed_y_cls_i, status_sec[sdict["used_Y_cluster"]]))
        cocotb.start_soon(set_input(ptp, dut.needed_iact_cycles_i, status_sec[sdict["needed_Iact_writes"]]))

        cocotb.start_soon(set_input(ptp, dut.filters_i, status_sec[sdict["used_psum_per_PE"]]))
        cocotb.start_soon(set_input(ptp, dut.wght_addr_len_i, status_sec[sdict["used_wght_addr_per_PE"]]))

        cocotb.start_soon(set_input(ptp, dut.bano_cluster_mode_i, 0))
        cocotb.start_soon(set_input(ptp, dut.af_cluster_mode_i, status_sec[sdict["autofunction"]]))
        cocotb.start_soon(set_input(ptp, dut.pooling_cluster_mode_i, status_sec[sdict["poolingmode"]]))
        cocotb.start_soon(set_input(ptp, dut.delay_psum_glb_i, status_sec[sdict["psum_delay"]]))

        cocotb.start_soon(set_input(ptp, dut.input_activations_i, status_sec[sdict["used_iact_per_PE"]]))
        cocotb.start_soon(set_input(ptp, dut.iact_write_addr_t_i, status_sec[sdict["iact_addr_len"]]))
        cocotb.start_soon(set_input(ptp, dut.iact_write_data_t_i, status_sec[sdict["iact_data_len"]]))
        cocotb.start_soon(set_input(ptp, dut.kernel_per_pe_cluster_i, status_sec[sdict["kernel_per_pe_cluster"]]))

        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp, dut.compute_mask_i, status_sec[sdict["usePEs"]]))
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        def _pack_router_modes(router_cfg, num_glb, router_bits):
            packed = 0
            for cl_x in range(oep.Clusters_X):
                for cl_y in range(oep.Clusters_Y):
                    for router in range(num_glb):
                        shift = router_bits * (router + num_glb * (cl_y + oep.Clusters_Y * cl_x))
                        packed |= int(router_cfg[cl_x][cl_y][router]) << shift
            return packed

        cocotb.start_soon(
            set_input(
                ptp,
                dut.router_mode_iact_i,
                _pack_router_modes(status_sec[sdict["router_iact"]], oep.NUM_GLB_IACT, oep.Iact_Router_Bits),
            )
        )
        cocotb.start_soon(
            set_input(
                ptp,
                dut.router_mode_wght_i,
                _pack_router_modes(status_sec[sdict["router_wght"]], oep.NUM_GLB_WGHT, oep.Wght_Router_Bits),
            )
        )
        cocotb.start_soon(
            set_input(
                ptp,
                dut.router_mode_psum_i,
                _pack_router_modes(status_sec[sdict["router_psum"]], oep.NUM_GLB_PSUM, oep.Psum_Router_Bits),
            )
        )

        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp, dut.status_reg_enable_i, 0))
    else:
        pdict = strdic.stream_parallel_dict
        logger.info("DMA stream sections (words each): %s", [len(section) for section in stream])
        for name in ("trans_cycles_iact", "trans_cycles_wght", "trans_cycles_psum"):
            try:
                logger.info("DUT expects %s = %d", name, int(getattr(dut, name).value))
            except Exception:
                pass

        cocotb.start_soon(set_input(ptp, dut.enable_dma_i, 1))

        for word in stream[pdict["status"]]:
            cocotb.start_soon(set_input(ptp, dut.data_dma_i, int(word)))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        for name in ("trans_cycles_iact", "trans_cycles_wght", "trans_cycles_psum"):
            try:
                logger.info("DUT expects %s = %d", name, int(getattr(dut, name).value))
            except Exception as exc:
                logger.info("DUT %s unreadable (%s)", name, type(exc).__name__)
        try:
            logger.info("DUT compute_mask_reg = %s", hex(int(dut.compute_mask_reg.value)))
        except Exception as exc:
            logger.info("DUT compute_mask_reg unreadable (%s)", type(exc).__name__)
        for name in ("send_data_out", "store_in_psum", "fully_connected_layer"):
            try:
                logger.info("DUT %s = %d", name, int(getattr(dut, name).value))
            except Exception as exc:
                logger.info("DUT %s unreadable (%s)", name, type(exc).__name__)
        logger.info(
            "Host will send: iact %d, wght %d, psum %d, quantize %d words",
            len(stream[pdict["iact"]]),
            len(stream[pdict["wght"]]),
            len(stream[pdict["psum"]]),
            len(stream[pdict["quantize"]]),
        )

        for word in stream[pdict["iact"]]:
            cocotb.start_soon(set_input(ptp, dut.data_dma_i, int(word)))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        for word in stream[pdict["wght"]]:
            cocotb.start_soon(set_input(ptp, dut.data_dma_i, int(word)))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        psum_sec = stream[pdict["psum"]]
        index = 0
        while index < len(psum_sec):
            cocotb.start_soon(set_input(ptp, dut.data_dma_i, int(psum_sec[index])))
            if random.random() < 1.0:
                index += 1
                cocotb.start_soon(set_input(ptp, dut.enable_dma_i, 1))
            else:
                cocotb.start_soon(set_input(ptp, dut.enable_dma_i, 0))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        for word in stream[pdict["quantize"]]:
            cocotb.start_soon(set_input(ptp, dut.data_dma_i, int(word)))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp, dut.enable_dma_i, 0))
        cocotb.start_soon(set_input(ptp, dut.ready_dma_i, 1))


# Diagnostic state shared with post-mortem checkers
fsm_states_seen = set()
fsm_probe_samples = [0]
fsm_state_trace = []


async def trace_pe_iact(ptp, dut, oep, max_lines=2000):
    """Per-cycle iact trace at one PE (opt-in: TRACE_PE_IACT=cx,cy,col,row[,t0,t1])."""
    try:
        parts = [float(v) for v in os.environ.get("TRACE_PE_IACT", "0,0,0,0").split(",")]
        cx, cy, col, row = (int(v) for v in parts[:4])
        t0, t1 = (parts[4], parts[5]) if len(parts) >= 6 else (0.0, 1e12)
        pe = (
            dut.OpenEye_Parallel.gen_x[cx].gen_y[cy].OpenEye_Cluster.pe_cluster.gen_X[col].gen_Y[row].pe
        )
        sp = pe.iact_data_SPad
    except Exception as exc:
        logger.error("trace_pe_iact: PE not reachable (%s)", type(exc).__name__)
        return

    def txt(sig):
        try:
            return str(sig.value)
        except Exception:
            return "?"

    lines = 0
    last_state = None
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        now = cocotb.utils.get_sim_time("ns")
        if now < t0:
            continue
        if now > t1:
            return
        cur_state = txt(pe.current_state_computing)
        cur_key = (cur_state, txt(pe.psum_enable_i), txt(pe.compute_i), txt(pe.psum_ready_o))
        if cur_key != last_state:
            logger.info(
                "pestate t=%s st=%s compute_i=%s psum_en_i=%s psum_rdy_i=%s psum_rdy_o=%s data_set=%s words_iact=%s words_wght=%s",
                cocotb.utils.get_sim_time("ns"),
                cur_state,
                txt(pe.compute_i),
                txt(pe.psum_enable_i),
                txt(pe.psum_ready_i),
                txt(pe.psum_ready_o),
                txt(pe.data_set),
                txt(pe.second_spad_words_iact),
                txt(pe.second_spad_words_wght),
            )
            last_state = cur_key
        en = txt(pe.iact_enable_i)
        if "1" not in en:
            continue
        logger.info(
            "peiact t=%s st=%s sel=%s en=%s rdy=%s data=%s | spad we=%s addr=%s d=%s",
            cocotb.utils.get_sim_time("ns"),
            txt(pe.current_state_computing),
            txt(pe.iact_select_i),
            en,
            txt(pe.iact_ready_o),
            txt(pe.iact_data_i),
            txt(sp.we_i),
            txt(sp.addr_i),
            txt(sp.data_i),
        )
        lines += 1


async def trace_all_pe_states(ptp, dut, oep, max_lines=600):
    """Per-change trace of every PE's FSM state and psum_ready_o (opt-in: TRACE_ALL_PE=t0,t1)."""
    try:
        t0, t1 = [float(v) for v in os.environ.get("TRACE_ALL_PE", "0,1e9").split(",")]
    except ValueError:
        t0, t1 = 0.0, 1e9
    pes = {}
    clusters = {}
    for cx in range(oep.Clusters_X):
        for cy in range(oep.Clusters_Y):
            try:
                base = dut.OpenEye_Parallel.gen_x[cx].gen_y[cy].OpenEye_Cluster.pe_cluster
                pes[(cx, cy)] = [
                    base.gen_X[c].gen_Y[r].pe for c in range(oep.PEs_X) for r in range(oep.PEs_Y)
                ]
                clusters[(cx, cy)] = dut.OpenEye_Parallel.gen_x[cx].gen_y[cy].OpenEye_Cluster
            except Exception as exc:
                logger.error("trace_all_pe_states: cluster (%d,%d) not reachable (%s)", cx, cy, type(exc).__name__)
                return

    def one(pe):
        try:
            st = int(pe.current_state_computing.value)
        except ValueError:
            return "x"
        try:
            rdy = str(pe.psum_ready_o.value)
        except Exception:
            rdy = "?"
        return "%x%s" % (st, rdy)

    last = {}
    lines = 0
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        now = cocotb.utils.get_sim_time("ns")
        if now < t0:
            continue
        if now > t1:
            return
        for key, lst in pes.items():
            cl = clusters[key]

            def bits(name):
                try:
                    return str(getattr(cl, name).value)
                except Exception:
                    return "?"

            cur = (
                " ".join(one(pe) for pe in lst)
                + " | pe_en_out=%s ext_en=%s rdy_in=%s dly_rdy=%s pe_rdy_out=%s top_rdy=%s"
                % (
                    bits("pe_router_psum_enable_out"),
                    bits("ext_mem_psum_enable_o"),
                    bits("pe_router_psum_ready_in"),
                    bits("delay_cluster_ready_out"),
                    bits("pe_router_psum_ready_out"),
                    bits("ready_src_top_psum"),
                )
            )
            if last.get(key) != cur:
                last[key] = cur
                logger.info("allpe t=%s c(%d,%d) %s", now, key[0], key[1], cur)
                lines += 1


async def trace_converter(ptp, dut, oep, max_lines=260):
    """Per-cycle trace of one iact converter (opt-in: TRACE_CONVERTER=cx,cy)."""
    try:
        cx, cy = [int(v) for v in os.environ.get("TRACE_CONVERTER", "0,0").split(",")]
        cv = dut.IACT_CONVERTER_X[cx].IACT_CONVERTER_Y[cy].iact_stream_constructor
        cell0 = dut.BUFFER_A[0].iact_layer_buffer
    except Exception as exc:
        logger.error("trace_converter: converter not reachable (%s)", type(exc).__name__)
        return

    def txt(sig):
        try:
            return str(sig.value)
        except Exception:
            return "?"

    def hexv(sig):
        t = txt(sig)
        if t and set(t) <= {"0", "1"}:
            return hex(int(t, 2))
        return "X" if ("x" in t.lower() or "z" in t.lower()) else t

    lines = 0
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        try:
            st = int(dut.fsm_current_state.value)
        except ValueError:
            continue
        if st < 8:
            continue
        active = (
            "1" in txt(cv.enable_store)
            or "1" in txt(cv.ram_wr_en_q)
            or "1" in txt(cv.ram_rd_en)
            or "1" in txt(cv.iact_enable_o)
            or "1" in txt(cell0.rd_en_i)
        )
        if not active:
            continue
        logger.info(
            "convtrace t=%s st=%d enc=%s | A0 rd=%s addr=%s q=%s | win=%s store=%s | "
            "wr=%s waddr=%s | rd=%s raddr=%s rdata=%s | en_o=%s data_o=%s",
            cocotb.utils.get_sim_time("ns"),
            st,
            hexv(cv.fsm_enc_current_state),
            txt(cell0.rd_en_i),
            hexv(cell0.addr_i),
            hexv(cell0.data_o),
            hexv(cv.storage_i),
            txt(cv.enable_store),
            txt(cv.ram_wr_en_q),
            hexv(cv.ram_wr_addr_q),
            txt(cv.ram_rd_en),
            hexv(cv.ram_rd_addr),
            hexv(cv.ram_data_o),
            txt(cv.iact_enable_o),
            hexv(cv.iact_data_o),
        )
        lines += 1


def dump_pe_occupancy(dut, oep, depth=32):
    """Log how many of the first `depth` words each PE has written."""
    for cx in range(oep.Clusters_X):
        for cy in range(oep.Clusters_Y):
            cells = []
            for col in range(oep.PEs_X):
                for row in range(oep.PEs_Y):
                    try:
                        pe = (
                            dut.OpenEye_Parallel.gen_x[cx]
                            .gen_y[cy]
                            .OpenEye_Cluster.pe_cluster.gen_X[col]
                            .gen_Y[row]
                            .pe
                        )
                    except Exception:
                        cells.append("c%dr%d:?" % (col, row))
                        continue
                    counts = []
                    for mem in (
                        lambda: pe.iact_data_SPad,
                        lambda: pe.weight_data_SPad,
                        lambda: pe.gen_serial_psum_spad.gen_psum_spad[0].psum_SPad,
                    ):
                        try:
                            m = mem().ram.impl.mem
                            n = 0
                            for a in range(depth):
                                try:
                                    int(m[a].value)
                                    n += 1
                                except ValueError:
                                    pass
                                except IndexError:
                                    break
                            counts.append(str(n))
                        except Exception:
                            counts.append("?")
                    cells.append("c%dr%d:%s" % (col, row, "/".join(counts)))
            logger.error("occupancy cluster(%d,%d) iact/wght/psum0: %s", cx, cy, " ".join(cells))


def _written_words(mem, depth):
    """Count words among the first `depth` addresses that are not X."""
    n = 0
    for a in range(depth):
        try:
            int(mem[a].value)
            n += 1
        except ValueError:
            pass
        except IndexError:
            break
    return n


def dump_iact_path(dut, oep, depth=64):
    """Log written-word counts along the iact path: BUFFER_A, then iact GLBs."""
    cells = []
    for cell in range(oep.IACT_RAM_CELLS):
        try:
            cells.append(_written_words(dut.BUFFER_A[cell].iact_layer_buffer.impl.mem, depth))
        except Exception:
            cells.append("?")
    logger.error("iact path BUFFER_A written words per cell (first %d addrs): %s", depth, cells)
    if oep.SERIAL:
        return
    for cx in range(oep.Clusters_X):
        for cy in range(oep.Clusters_Y):
            glbs = []
            for g in range(oep.NUM_GLB_IACT):
                try:
                    glb = (
                        dut.OpenEye_Parallel.gen_x[cx]
                        .gen_y[cy]
                        .OpenEye_Cluster.glb_cluster.gen_iact[g]
                        .iact_glb.impl.mem
                    )
                    glbs.append(_written_words(glb, depth))
                except Exception as exc:
                    glbs.append("?(%s)" % type(exc).__name__)
            logger.error("iact path cluster(%d,%d) iact GLB written words: %s", cx, cy, glbs)


iact_handoff_counts = {"enable": {}, "handshake": {}, "choose": {}, "cycles": 0}
cluster_iact_counts = {}


async def monitor_cluster_iact(ptp, dut, oep):
    """Count iact handshakes at each hop inside every OpenEye_Cluster."""
    stages = (
        ("ext", "ext_mem_iact_enable_i", "ext_mem_iact_ready_o"),
        ("glb", "glb_cluster_iact_enable", "glb_cluster_iact_ready"),
        ("pe", "pe_iact_enable", "pe_iact_ready"),
    )
    clusters = []
    for cx in range(oep.Clusters_X):
        for cy in range(oep.Clusters_Y):
            try:
                clusters.append(((cx, cy), dut.OpenEye_Parallel.gen_x[cx].gen_y[cy].OpenEye_Cluster))
            except Exception:
                pass
            for name, _, _ in stages:
                cluster_iact_counts[(cx, cy, name)] = 0

    while True:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        for key, cl in clusters:
            for name, en_sig, rdy_sig in stages:
                try:
                    en = int(getattr(cl, en_sig).value)
                    rdy = int(getattr(cl, rdy_sig).value)
                except Exception:
                    continue
                if en & rdy:
                    cluster_iact_counts[(key[0], key[1], name)] += bin(en & rdy).count("1")
                    if name == "pe":
                        try:
                            data = int(getattr(cl, "pe_iact_data").value)
                        except Exception:
                            continue
                        key_nz = (key[0], key[1], "pe_nonzero")
                        cluster_iact_counts.setdefault(key_nz, 0)
                        if data:
                            cluster_iact_counts[key_nz] += 1


pe_iact_counts = {}
converter_store_counts = {}
conv_array_seen = {}


async def monitor_converter_store(ptp, dut, oep):
    """Count the store-enable pulses each iact converter receives."""
    for cx in range(oep.Clusters_X):
        for cy in range(oep.Clusters_Y):
            converter_store_counts[(cx, cy)] = 0
    while True:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        try:
            val = int(dut.conv_array_reg.value)
            conv_array_seen[val] = conv_array_seen.get(val, 0) + 1
        except Exception:
            pass
        try:
            enc = int(dut.iact_converter_enc_enable.value)
            arr = int(dut.conv_array_reg.value)
            cyc = int(dut.iact_converter_cycles.value)
            mx = int(dut.iact_converter_max_cycles.value)
        except Exception:
            continue
        converter_store_counts["enc_enable_cycles"] = converter_store_counts.get("enc_enable_cycles", 0) + (
            1 if enc else 0
        )
        converter_store_counts["cycles_max"] = max(converter_store_counts.get("cycles_max", 0), cyc)
        converter_store_counts["max_cycles"] = mx
        if not enc or mx == 0 or cyc > mx - 1:
            continue
        for cx in range(oep.Clusters_X):
            for cy in range(oep.Clusters_Y):
                if (arr >> (cx + cy * oep.Clusters_X)) & 1:
                    converter_store_counts[(cx, cy)] += 1


def report_converter_store():
    """Log the per-converter store-enable counts and the conv_array_reg values."""
    if not converter_store_counts:
        return
    logger.error("iact converter store-enable pulses per cluster (reconstructed):")
    for key in sorted(k for k in converter_store_counts if isinstance(k, tuple)):
        logger.error("  cluster(col=%d,row=%d): en_store=%d", key[0], key[1], converter_store_counts[key])
    logger.error(
        "  enc_enable high for %d cycles; iact_converter_cycles reached %d, limit max_cycles=%d",
        converter_store_counts.get("enc_enable_cycles", 0),
        converter_store_counts.get("cycles_max", 0),
        converter_store_counts.get("max_cycles", 0),
    )
    if conv_array_seen:
        top = sorted(conv_array_seen.items(), key=lambda kv: -kv[1])[:8]
        logger.error(
            "  conv_array_reg values (value: cycles): %s",
            ", ".join("0x%x: %d" % (v, n) for v, n in top),
        )


async def monitor_pe_iact(ptp, dut, oep, pe_col=0):
    """Per PE: count enable, and enable & ready, on the lane it selects."""
    pes = []
    for cx in range(oep.Clusters_X):
        for cy in range(oep.Clusters_Y):
            for row in range(oep.PEs_Y):
                try:
                    pe = (
                        dut.OpenEye_Parallel.gen_x[cx]
                        .gen_y[cy]
                        .OpenEye_Cluster.pe_cluster.gen_X[pe_col]
                        .gen_Y[row]
                        .pe
                    )
                except Exception:
                    continue
                pes.append(((cx, cy, row), pe))
                pe_iact_counts[(cx, cy, row)] = {"sel": None, "en": 0, "hs": 0}

    while True:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        for key, pe in pes:
            try:
                sel = int(pe.iact_select_i.value)
                en = int(pe.iact_enable_i.value)
                rdy = int(pe.iact_ready_o.value)
            except Exception:
                continue
            rec = pe_iact_counts[key]
            rec["sel"] = sel
            if sel >= oep.NUM_GLB_IACT:
                continue
            if (en >> sel) & 1:
                rec["en"] += 1
                if (rdy >> sel) & 1:
                    rec["hs"] += 1


def report_pe_iact():
    """Log the per-PE iact lane counts gathered by monitor_pe_iact."""
    if not pe_iact_counts:
        return
    logger.error("per-PE iact on selected lane (sel=lane, en=valid cycles, hs=accepted):")
    for key in sorted(pe_iact_counts):
        r = pe_iact_counts[key]
        logger.error(
            "  cluster(%d,%d) row %d: sel=%s en=%d hs=%d",
            key[0],
            key[1],
            key[2],
            r["sel"],
            r["en"],
            r["hs"],
        )


def dump_compute_mask(dut, oep):
    """Log the PE compute mask as the array actually sees it, per cluster."""
    try:
        logger.error("compute_mask_i = %s", str(dut.OpenEye_Parallel.compute_mask_i.value))
    except Exception:
        logger.error("compute_mask_i unreadable")
    for cx in range(oep.Clusters_X):
        for cy in range(oep.Clusters_Y):
            try:
                cl = dut.OpenEye_Parallel.gen_x[cx].gen_y[cy]
                logger.error("  cluster(%d,%d) compute_cluster_i_w=%s", cx, cy, str(cl.compute_cluster_i_w.value))
            except Exception as exc:
                logger.error(
                    "  cluster(%d,%d) compute_cluster_i_w unreadable (%s)", cx, cy, type(exc).__name__
                )


def report_cluster_iact():
    """Log the per-cluster, per-hop iact handshake counts."""
    if not cluster_iact_counts:
        return
    logger.error("per-cluster iact handshakes (enable & ready, summed over lanes):")
    seen = sorted({(cx, cy) for cx, cy, _ in cluster_iact_counts})
    for cx, cy in seen:
        logger.error(
            "  cluster(%d,%d) ext=%s glb=%s pe=%s pe_nonzero_cycles=%s",
            cx,
            cy,
            cluster_iact_counts.get((cx, cy, "ext"), 0),
            cluster_iact_counts.get((cx, cy, "glb"), 0),
            cluster_iact_counts.get((cx, cy, "pe"), 0),
            cluster_iact_counts.get((cx, cy, "pe_nonzero"), 0),
        )


async def monitor_iact_handoff(ptp, dut):
    """Count iact valid and handshake cycles per bit at the OpenEye_Parallel input."""

    def read(name):
        try:
            return int(getattr(dut, name).value)
        except Exception:
            return None

    while True:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        iact_handoff_counts["cycles"] += 1
        en = read("iact_enable_i_oep_w")
        if not en:
            continue
        rdy = read("iact_ready_o_oep_w") or 0
        bit = 0
        while en >> bit:
            if (en >> bit) & 1:
                iact_handoff_counts["enable"][bit] = iact_handoff_counts["enable"].get(bit, 0) + 1
                if (rdy >> bit) & 1:
                    iact_handoff_counts["handshake"][bit] = iact_handoff_counts["handshake"].get(bit, 0) + 1
            bit += 1
        ch = read("iact_choose_i_oep_w")
        if ch is not None:
            key = (hex(ch), hex(en))
            if key in iact_handoff_counts["choose"] or len(iact_handoff_counts["choose"]) < 32:
                iact_handoff_counts["choose"][key] = iact_handoff_counts["choose"].get(key, 0) + 1


def report_iact_handoff(oep=None):
    c = iact_handoff_counts
    logger.error(
        "iact handoff over %d cycles: enable cycles per bit %s, handshakes per bit %s",
        c["cycles"],
        dict(sorted(c["enable"].items())),
        dict(sorted(c["handshake"].items())),
    )
    for (ch, en), n in sorted(c["choose"].items(), key=lambda kv: -kv[1]):
        line = "  %d valid cycles with enable=%s choose=%s" % (n, en, ch)
        if oep is not None:
            w = max(1, (oep.NUM_GLB_IACT).bit_length())
            pes = oep.PEs_X * oep.PEs_Y
            v = int(ch, 16)
            parts = []
            for cl in range(oep.Clusters_X * oep.Clusters_Y):
                sel = []
                for pe in range(pes):
                    f = (v >> ((cl * pes + pe) * w)) & ((1 << w) - 1)
                    if f != oep.NUM_GLB_IACT:
                        sel.append("c%dr%d<-g%d" % (pe % oep.PEs_X, pe // oep.PEs_X, f))
                parts.append("cl%d[%s]" % (cl, ",".join(sel) if sel else "none"))
            line += " -> " + " ".join(parts)
        logger.error(line)


async def probe_fsm_states(ptp, dut, stall_report_after=20000, oep=None):
    """Record every main-FSM state the DUT enters and report stalls."""
    previous = None
    stuck_for = 0
    reported = False
    try:
        fail_after = int(os.environ.get("OPENEYE_FAIL_ON_STALL", "0"))
    except ValueError:
        fail_after = 0
    if fail_after:
        stall_report_after = min(stall_report_after, fail_after)
    progress_key = None
    idle_for = 0
    while True:
        fsm_probe_samples[0] += 1
        try:
            state = int(dut.fsm_current_state.value)
        except ValueError:
            state = None
        if state is not None:
            fsm_states_seen.add(state)
            if state != previous:
                if len(fsm_state_trace) < 200:
                    fsm_state_trace.append((str(cocotb.utils.get_sim_time("ns")), state))
                fields = []
                for name in (
                    "fsm_psum_current_state",
                    "send_data_out",
                    "store_in_psum",
                    "trans_cycles_iact",
                    "trans_cycles_psum",
                ):
                    try:
                        fields.append("%s=%d" % (name, int(getattr(dut, name).value)))
                    except Exception:
                        fields.append("%s=?" % name)
                logger.info(
                    "FSM state -> %d at %s ns (%s)",
                    state,
                    cocotb.utils.get_sim_time("ns"),
                    ", ".join(fields),
                )
                stuck_for = 0
                reported = False
            else:
                stuck_for += 1
                if stuck_for == stall_report_after and not reported:
                    reported = True
                    counters = []
                    for name in (
                        "fsm_cycle",
                        "fsm_psum_cycle",
                        "fsm_psum_current_state",
                        "trans_cycles_iact",
                        "trans_cycles_wght",
                        "trans_cycles_psum",
                        "current_cycle",
                        "needed_cycles",
                        "single_iteration",
                    ):
                        try:
                            counters.append("%s=%d" % (name, int(getattr(dut, name).value)))
                        except Exception:
                            pass
                    logger.error(
                        "FSM stuck in state %d for %d cycles: %s", state, stuck_for, ", ".join(counters)
                    )
                    try:
                        pp = dut.psum_pipeline_inst
                        for name in ("psum_ready_i_reg", "psum_enable_i_reg", "results_ready", "finished_cycles_psum"):
                            try:
                                logger.error("  %s = %s", name, str(getattr(pp, name).value))
                            except Exception:
                                logger.error("  %s unreadable", name)
                        try:
                            logger.error("  psum_choose_i_reg = %s", str(dut.psum_choose_i_reg.value))
                        except Exception:
                            logger.error("  psum_choose_i_reg unreadable")
                        for cx in range(oep.Clusters_X):
                            for cy in range(oep.Clusters_Y):
                                try:
                                    cl = dut.OpenEye_Parallel.gen_x[cx].gen_y[cy].OpenEye_Cluster
                                    logger.error(
                                        "  cluster(%d,%d) pe_router_psum_ready_out=%s enable_in=%s",
                                        cx,
                                        cy,
                                        str(cl.pe_router_psum_ready_out.value),
                                        str(cl.pe_router_psum_enable_in.value),
                                    )
                                    states = [
                                        str(cl.pe_cluster.gen_X[x].gen_Y[y].pe.current_state_computing.value)
                                        for y in range(oep.PEs_Y)
                                        for x in range(oep.PEs_X)
                                    ]
                                    logger.error("  cluster(%d,%d) PE states by row: %s", cx, cy, states)
                                    pe = cl.pe_cluster.gen_X[0].gen_Y[0].pe
                                    logger.error(
                                        "  cluster(%d,%d) first PE psum_enable_i=%s psum_select=%s",
                                        cx,
                                        cy,
                                        str(pe.psum_enable_i.value),
                                        str(pe.psum_select.value),
                                    )
                                except Exception as exc:
                                    logger.error(
                                        "  cluster(%d,%d) psum ready unreadable (%s)",
                                        cx,
                                        cy,
                                        type(exc).__name__,
                                    )
                        for name in ("psum_enable_o", "psum_ready_o_reg", "router_mode_psum"):
                            logger.error("  %s = %s", name, str(getattr(pp, name).value))
                        for name in ("iact_ready_o_oep_w", "iact_enable_i_oep_w"):
                            logger.error("  %s = %s", name, str(getattr(dut, name).value))
                        for name in ("router_mode_iact", "router_mode_wght", "router_mode_psum"):
                            try:
                                logger.error("  DUT %s = %s", name, str(getattr(dut, name).value))
                            except Exception:
                                pass
                    except Exception as exc:
                        logger.error("  psum handshake vectors unreadable (%s)", type(exc).__name__)
                    report_iact_handoff(oep)
                    if oep is not None:
                        dump_iact_path(dut, oep)
                        dump_pe_occupancy(dut, oep)
            previous = state
        if fail_after:
            key = [state]
            for name in ("fsm_psum_current_state", "fsm_psum_cycle"):
                try:
                    key.append(int(getattr(dut, name).value))
                except Exception:
                    key.append(None)
            key = tuple(key)
            if key == progress_key:
                idle_for += 1
                if idle_for == fail_after:
                    if not reported:
                        logger.error("FSM stuck (main state, psum state, psum cycle) = %s", key)
                    raise AssertionError(
                        "No FSM progress for %d cycles (main state, psum state, psum cycle) = %s; "
                        "see the stall diagnostics above" % (fail_after, key)
                    )
            else:
                progress_key = key
                idle_for = 0
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)


def _ram_word(mem_entry):
    """Read a whole RAM word as an int, or None if it holds X/Z."""
    try:
        return int(mem_entry.value)
    except ValueError:
        return None


def _signed(value, bits):
    """Reinterpret an unsigned field of the given width as two's complement."""
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def _to_ram_format(datapoints, params, layer_params):
    """Pack flat activation bytes into IACT buffer RAM words."""
    if datapoints is None:
        return None

    iact_bw = int(params.IACT_Bitwidth)
    word_bw = int(params.IACT_RAM_CELLS_WORD_BITWIDTH)
    iacts_per_word = max(1, word_bw // iact_bw)
    arr = np.asarray(datapoints, dtype=np.int64).ravel()
    if arr.size == 0:
        return []

    elem_mask = (1 << iact_bw) - 1
    byte_vals = (arr & elem_mask).astype(np.uint64)
    rem = byte_vals.size % iacts_per_word
    if rem:
        byte_vals = np.pad(byte_vals, (0, iacts_per_word - rem), constant_values=0)

    grouped = byte_vals.reshape(-1, iacts_per_word)
    shifts = np.arange(iacts_per_word, dtype=np.uint64) * np.uint64(iact_bw)
    packed = np.bitwise_or.reduce(grouped << shifts, axis=1)
    return [int(w) for w in packed.tolist()]


def _iact_buffer_words(oep, layer_params, iact_ref):
    """Interleave filter pairs in (Y, X, pair_slot) order for IACT buffer comparison."""
    out_shape = getattr(layer_params, "output_shape", None)
    if out_shape is not None and len(out_shape) >= 4 and int(out_shape[3]) > 0:
        num_blocks = int(out_shape[3])
    elif isinstance(iact_ref, dict):
        num_blocks = len(iact_ref)
    else:
        num_blocks = int(getattr(layer_params, "filters", 0) or np.asarray(iact_ref).shape[0])

    if isinstance(iact_ref, dict):
        arr = np.asarray([iact_ref[i] for i in range(num_blocks)], dtype=np.int64)
    else:
        arr = np.asarray(iact_ref, dtype=np.int64)[:num_blocks]

    if arr.ndim < 3 or arr.size == 0:
        return arr.ravel()

    if arr.shape[0] % 2 != 0:
        arr = np.pad(arr, ((0, 1), (0, 0), (0, 0)), constant_values=0)

    # arr shape: (padded_blocks, X, Y) -> (padded_blocks // 2, 2, X, Y) -> (padded_blocks // 2, Y, X, 2)
    paired = arr.reshape(arr.shape[0] // 2, 2, arr.shape[1], arr.shape[2])
    return paired.transpose(0, 3, 2, 1).ravel()

def iact_cells_per_group(oep):
    """Number of RAM cells one full buffer write cycle spans."""
    return math.ceil((oep.IACT_RAM_CELLS * oep.IACT_RAM_CELLS_WORD_BITWIDTH) / oep.DMA_BITWIDTH)


def compare_iact_storage(ptp, dut, iact_ref, oep, layer_params, slot_idx=0):
    """Compare the iact double-buffer contents with reference activations."""
    logger.info("Iact storages are checked.")
    if isinstance(iact_ref, dict):
        first_item = next(iter(iact_ref.values()), [])
        shape = (len(iact_ref),) + np.asarray(first_item).shape
    else:
        try:
            shape = np.asarray(iact_ref).shape
        except Exception:
            shape = "?"
    logger.info(
        "Iact reference: shape %s, used_channels %s, IACT_RAM_CELLS %s",
        shape,
        getattr(layer_params, "filters", "?"),
        oep.IACT_RAM_CELLS,
    )

    expected = _to_ram_format(_iact_buffer_words(oep, layer_params, iact_ref), oep, layer_params)

    if expected is None:
        logger.warning(
            "No iact buffer layout known for layer '%s'; skipping the check.",
            getattr(layer_params, "layer_name", "?"),
        )
        return True

    cells_per_group = iact_cells_per_group(oep)
    try:
        depth = len(dut.BUFFER_A[0].iact_layer_buffer.impl.mem)
    except TypeError:
        depth = 1 << oep.BUFFER_WIDTH_IACT
    logger.info(
        "Iact buffer check: %d expected words, %d cells per group, depth %d (layer '%s')",
        len(expected),
        cells_per_group,
        depth,
        getattr(layer_params, "layer_name", "?"),
    )
    if len(expected) > cells_per_group * depth:
        logger.error(
            "Expected iact image (%d words) exceeds the buffer (%d cells x %d addresses); "
            "check the layout parameters for layer '%s'.",
            len(expected),
            cells_per_group,
            depth,
            getattr(layer_params, "layer_name", "?"),
        )
        return False

    branch_idx = int(layer_params.choose_iact_storage_output) * (1 << (int(oep.BUFFER_WIDTH_IACT) - 1))
    error_found = False
    errors_logged = 0
    for index, ref_word in enumerate(expected):
        cell = index % oep.IACT_RAM_CELLS
        addr = branch_idx + (index // oep.IACT_RAM_CELLS)
        dut_word = _ram_word(dut.BUFFER_A[cell].iact_layer_buffer.impl.mem[addr])
        if dut_word != ref_word:
            error_found = True
            errors_logged += 1
            if errors_logged <= 16:
                logger.error("Error found in Iact storage; word: %d cell: %d addr: %d", index, cell, addr)
                logger.error(
                    "Ref-Value: 0x%x DUT-Value: %s",
                    ref_word,
                    "X (never written)" if dut_word is None else hex(dut_word),
                )
    if errors_logged > 16:
        logger.error("... and %d further Iact storage mismatches.", errors_logged - 16)

    if os.environ.get("DUMP_IACT_IMAGE"):
        rows = (len(expected) + oep.IACT_RAM_CELLS - 1) // oep.IACT_RAM_CELLS + 2
        for addr in range(rows):
            for cell in range(oep.IACT_RAM_CELLS):
                index = addr * oep.IACT_RAM_CELLS + cell
                ref_word = expected[index] if index < len(expected) else None
                dut_word = _ram_word(dut.BUFFER_A[cell].iact_layer_buffer.impl.mem[branch_idx + addr])

                def fmt(w):
                    return (
                        "--------"
                        if w is None
                        else " ".join("%02x" % ((w >> (8 * i)) & 0xFF) for i in range(8))
                    )

                logger.info(
                    "iactimg addr=%d cell=%d ref=[%s] dut=[%s]%s",
                    addr,
                    cell,
                    fmt(ref_word),
                    fmt(dut_word),
                    "" if ref_word == dut_word else " <<",
                )
    if error_found:
        _log_iact_value_comparison(dut, oep, expected, cells_per_group, half_offset=branch_idx)
        _log_iact_buffer_occupancy(dut, oep)

    return not error_found


def _log_iact_value_comparison(dut, oep, expected, cells_per_group, half_offset=0):
    """Say whether the buffer holds the right values in the wrong order."""

    def to_bytes(word):
        return [(word >> (8 * i)) & 0xFF for i in range(8)]

    actual_words, missing = [], 0
    for index in range(len(expected)):
        word = _ram_word(
            dut.BUFFER_A[index % oep.IACT_RAM_CELLS].iact_layer_buffer.impl.mem[
                half_offset + (index // oep.IACT_RAM_CELLS)
            ]
        )
        if word is None:
            missing += 1
        else:
            actual_words.append(word)

    exact = sum(1 for e, a in zip(expected, actual_words) if e == a)
    comparable = expected[: len(actual_words)]
    expected_bytes = sorted(b for w in comparable for b in to_bytes(w))
    actual_bytes = sorted(b for w in actual_words for b in to_bytes(w))
    same_multiset = expected_bytes == actual_bytes
    logger.error(
        "Iact value comparison: %d/%d words match exactly, %d of %d expected "
        "words unwritten; over the %d written words the byte multisets %s",
        exact,
        len(expected),
        missing,
        len(expected),
        len(actual_words),
        "MATCH - right values, wrong placement" if same_multiset else "DIFFER - the values themselves are wrong",
    )


def _log_iact_buffer_occupancy(dut, oep, max_addr=64):
    """Report where the iact buffer actually holds data, to localise a mismatch."""
    for name in (
        "send_data_out",
        "fsm_current_state",
        "choose_iact_buffer",
        "iact_channels_per_pe_next_layer",
        "fsm_cycle",
        "skipIact_reg",
        "max_pooling",
        "FSM_CEIL_IACT_RTR_CCLS",
        "FSM_CEIL_WGHT_RTR_CCLS",
        "FSM_CEIL_PSUM_RTR_CCLS",
        "CLUSTERS",
        "NUM_GLB_IACT",
        "NUM_GLB_WGHT",
        "NUM_GLB_PSUM",
        "fsm_psum_cycle",
        "trans_cycles_psum",
        "fsm_psum_current_state",
        "filters",
        "needed_cycles",
        "finished_cycles_psum",
    ):
        try:
            logger.error("Iact buffer context: %s = %s", name, int(getattr(dut, name).value))
        except Exception as exc:
            logger.error("Iact buffer context: %s unavailable (%s)", name, type(exc).__name__)

    if fsm_states_seen:
        logger.error(
            "Iact buffer context: main FSM states visited = %s over %d samples "
            "(RECEIVE_PSUMS_TO_IACT is 11)",
            sorted(fsm_states_seen),
            fsm_probe_samples[0],
        )
        logger.error("Iact buffer context: FSM transitions (ns, state) = %s", fsm_state_trace)

    written = []
    for cell in range(oep.IACT_RAM_CELLS):
        for addr in range(max_addr):
            try:
                word = int(dut.BUFFER_A[cell].iact_layer_buffer.impl.mem[addr].value)
            except (ValueError, IndexError):
                continue
            written.append((cell, addr, word))
    if not written:
        logger.error(
            "Iact buffer occupancy: no written word in any of %d cells over the first %d addresses.",
            oep.IACT_RAM_CELLS,
            max_addr,
        )
        return
    cells = sorted({c for c, _, _ in written})
    logger.error(
        "Iact buffer occupancy: %d written words, cells %s, first entries %s",
        len(written),
        cells[:8],
        [(c, a, hex(w)) for c, a, w in written[:4]],
    )


async def write_iact(ptp, dut, stream, oep, lp):
    """Write input activations over the parallel interface."""
    if lp.skipIact != 1:
        while dut.iact_ready_o.value == 0:
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        for position in range(len(stream[0][0][0])):
            iact_enable_signal = 0
            iact_transmission = 0
            for x_cluster in range(oep.Clusters_X):
                for y_cluster in range(oep.Clusters_Y):
                    for router in range(oep.NUM_GLB_IACT):
                        try:
                            lane = router + y_cluster * oep.NUM_GLB_IACT + x_cluster * oep.NUM_GLB_IACT * oep.Clusters_Y
                            iact_transmission += int(stream[x_cluster][y_cluster][router][position]) << (
                                lane * oep.IACT_Trans_Bitwidth
                            )
                            iact_enable_signal += 1 << lane
                        except Exception:
                            pass
            cocotb.start_soon(set_input(ptp, dut.iact_data_i, iact_transmission))
            cocotb.start_soon(set_input(ptp, dut.iact_enable_i, iact_enable_signal))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp, dut.iact_data_i, 0))
        cocotb.start_soon(set_input(ptp, dut.iact_enable_i, 0))


async def write_wght(ptp, dut, stream, oep, lp):
    """Write weights over the parallel interface."""
    if lp.skipWght != 1:
        all_ready = (1 << (oep.Clusters_X * oep.Clusters_Y * oep.NUM_GLB_WGHT)) - 1
        while dut.wght_ready_o.value != all_ready:
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        cocotb.start_soon(set_input(ptp, dut.wght_enable_i, all_ready))
        for position in range(len(stream[0][0][0])):
            wght_enable_signal = 0
            wght_transmission = 0
            for x_cluster in range(oep.Clusters_X):
                for y_cluster in range(oep.Clusters_Y):
                    for router in range(oep.NUM_GLB_WGHT):
                        try:
                            lane = router + y_cluster * oep.NUM_GLB_WGHT + x_cluster * oep.NUM_GLB_WGHT * oep.Clusters_Y
                            wght_transmission += int(stream[x_cluster][y_cluster][router][position]) << (
                                lane * oep.WGHT_Trans_Bitwidth
                            )
                            wght_enable_signal += 1 << lane
                        except Exception:
                            pass
            cocotb.start_soon(set_input(ptp, dut.wght_data_i, wght_transmission))
            cocotb.start_soon(set_input(ptp, dut.wght_enable_i, wght_enable_signal))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp, dut.wght_data_i, 0))
        cocotb.start_soon(set_input(ptp, dut.wght_enable_i, 0))


async def write_bias(ptp, dut, stream, oep, lp):
    """Write bias values over the parallel psum interface."""
    cocotb.start_soon(
        set_input(ptp, dut.psum_enable_i, (1 << (oep.Clusters_X * oep.Clusters_Y * oep.NUM_GLB_PSUM)) - 1)
    )
    for position in range(len(stream[0][0][0])):
        psum_transmission = 0
        for x_cluster in range(oep.Clusters_X):
            for y_cluster in range(oep.Clusters_Y):
                for router in range(oep.NUM_GLB_PSUM):
                    lane = router + y_cluster * oep.NUM_GLB_PSUM + x_cluster * oep.NUM_GLB_PSUM * oep.Clusters_Y
                    psum_transmission += int(stream[x_cluster][y_cluster][router][position]) << (
                        lane * oep.PSUM_Trans_Bitwidth
                    )
        cocotb.start_soon(set_input(ptp, dut.psum_data_i, psum_transmission))
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp, dut.psum_data_i, 0))
    cocotb.start_soon(set_input(ptp, dut.psum_enable_i, 0))


async def await_enable_signal(ptp, dut):
    """Wait for DMA enable signal from the DUT."""
    cocotb.start_soon(set_input(ptp, dut.ready_dma_i, 1))
    while dut.enable_dma_o.value != 1:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)


def _signal_is(handle, expected):
    """True when a signal reads exactly `expected`; False while it holds X/Z."""
    try:
        return int(handle.value) == expected
    except ValueError:
        return False


async def await_ready_signal(ptp, dut, settle_cycles=5000, max_wait_cycles=120000000, report_every=200000):
    """Wait until the DUT finishes the current layer and asserts ready_dma_o again."""
    for _ in range(settle_cycles):
        if not _signal_is(dut.ready_dma_o, 1):
            break
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    else:
        logger.warning(
            "ready_dma_o stayed high for %d cycles%s; the completion "
            "wait below cannot be trusted for this layer.",
            settle_cycles,
            _fsm_state_note(dut),
        )

    for cycle in range(max_wait_cycles):
        if _signal_is(dut.ready_dma_o, 1):
            return
        if cycle and cycle % report_every == 0:
            logger.info("Still waiting for the layer to finish after %d cycles%s", cycle, _fsm_state_note(dut))
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

    raise TimeoutError(
        "ready_dma_o never came back high within {} cycles{} - the accelerator "
        "did not finish the layer.".format(max_wait_cycles, _fsm_state_note(dut))
    )


def _fsm_state_note(dut):
    """' (main FSM in state N)' when readable, else an empty string."""
    try:
        return " (main FSM in state {})".format(int(dut.fsm_current_state.value))
    except Exception:
        return ""


async def _compare_stream_spatial(
    ptp,
    dut,
    layer_number,
    layer_repetition,
    layer_parameters,
    oep,
    les,
    dram,
    login_level,
    output_order,
    rel_layers,
    x_wrap_limit: int,
):
    """Shared output stream collector for Conv2D and Depthwise Conv2D layers."""
    debug_enabled = logging.DEBUG >= login_level
    txt_file = None
    storage_file = None
    if debug_enabled:
        layer_dir = f"demo/layer_{layer_parameters.layer_name}"
        os.makedirs(layer_dir, exist_ok=True)
        txt_file = open(f"{layer_dir}/output.txt", "w")
        storage_file = open(f"{layer_dir}/storage_input.txt", "w")

    logger.debug("PRE")
    logger.debug(
        "f: %s x: %s y: %s f_corner_start: %s y_corner_start: %s x_corner_start: %s\n",
        les.f_start,
        les.x_start,
        les.y_start,
        les.f_corner_start,
        les.y_corner_start,
        les.x_corner_start,
    )

    f = 0
    x = 0
    y = 0
    les.current_position = 0
    next_layers = rel_layers["Next_Layer"][layer_parameters.layer_name]
    dram_string = next_layers[0] if next_layers else "Output"
    target_fmap = dram.fmap[dram_string][0]
    f_bound, x_bound, y_bound = target_fmap.shape
    psum_bits = int(oep.DATA_PSUM_BITWIDTH)
    psum_mask = (1 << psum_bits) - 1

    if debug_enabled:
        storage_file.write(
            f" f_corner_start: {les.f_corner_start} y_corner_start: {les.y_corner_start} x_corner_start: {les.x_corner_start}\n"
        )

    if oep.SERIAL == 0:
        if (layer_repetition % layer_parameters.iact_transmissions_pe) == (layer_parameters.iact_transmissions_pe - 1):
            cocotb.start_soon(send_enable_conv(ptp, dut, layer_parameters, layer_repetition, oep))
            while dut.psum_enable_o.value == 0:
                await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
            dut._log.info("Output Stream started")
            assert dut.psum_enable_o.value != 0, "psum is not 1!"
            cluster_order = [
                a + b
                for a in range(layer_parameters.used_Y_cluster)
                for b in range(0, oep.Clusters_Y, layer_parameters.used_Y_cluster)
            ]
            rep_coords = output_order[layer_repetition]
            rep_len = len(rep_coords)
            while dut.psum_enable_o.value != 0:
                for y_cluster in reversed(cluster_order):
                    for x_cluster in reversed(range(oep.Clusters_X)):
                        for router in reversed(range(oep.NUM_GLB_PSUM)):
                            if (
                                layer_parameters.computing_mx[oep.Clusters_X - x_cluster - 1][
                                    oep.Clusters_Y - y_cluster - 1
                                ][0][oep.NUM_GLB_PSUM - router - 1]
                                == 1
                            ):
                                lower_limit = (
                                    x_cluster * oep.Clusters_Y * oep.NUM_GLB_PSUM * oep.PSUM_Trans_Bitwidth
                                    + y_cluster * oep.NUM_GLB_PSUM * oep.PSUM_Trans_Bitwidth
                                    + router * oep.PSUM_Trans_Bitwidth
                                )
                                upper_limit = lower_limit + psum_bits
                                if debug_enabled:
                                    outputvalue = dut.psum_data_o.value[lower_limit:upper_limit]
                                    txt_file.write(bin(outputvalue)[2:].zfill(oep.PSUM_Trans_Bitwidth) + "\n")
                                for i in range(oep.PARALLEL_MACS):
                                    if les.current_position < rep_len:
                                        f, x, y = rep_coords[les.current_position][:3]
                                    les.current_position += 1
                                    if debug_enabled:
                                        storage_file.write(f"f: {f} x: {x} y: {y}\n")
                                    if 0 <= f < f_bound and 0 <= x < x_bound and 0 <= y < y_bound:
                                        try:
                                            raw_val = int(
                                                dut.psum_data_o.value[
                                                    lower_limit + psum_bits * (1 - i) : upper_limit - psum_bits * i
                                                ]
                                            )
                                            target_fmap[f, x, y] = _signed(raw_val & psum_mask, psum_bits)
                                        except Exception:
                                            pass
                await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp, dut.psum_enable_i, 0))
        dut._log.info("Output Stream finished")
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cocotb.start_soon(set_input(ptp, dut.status_reg_enable_i, 1))
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    else:
        words = oep.DMA_BITWIDTH // 32
        dut._log.info("Output Stream started")
        used_clusters_per_calc = math.ceil(layer_parameters.iact_size_x / 4) * 4
        values_per_transmission = math.ceil(
            layer_parameters.different_kernels_per_calculation * used_clusters_per_calc / 2
        )
        transmissions_per_cycle = (oep.Clusters_Y * oep.Clusters_X * oep.PEs_X) // 2
        current_cycle = 0
        read_data = 1
        chance = 100
        x_repetitions = layer_parameters.iact_x_line_repetitions
        x_block_width = layer_parameters.iact_x_add_up // max(x_repetitions, 1)
        x_block_base = 0
        while dut.enable_dma_o.value == 1:
            if read_data:
                try:
                    dma_word = int(dut.data_dma_o.value)
                except Exception:
                    if debug_enabled:
                        txt_file.close()
                        storage_file.close()
                    logger.error("Error reading DMA output word (X/Z detected)")
                    raise Exception("X detected.")

                if debug_enabled:
                    txt_file.write(format(dma_word, f"0{oep.DMA_BITWIDTH}b") + "\n")

                if current_cycle < values_per_transmission:
                    for i in range(words):
                        if debug_enabled:
                            storage_file.write(f"f: {f} x: {x} y: {y}\n")
                        if 0 <= f < f_bound and 0 <= x < x_bound and 0 <= y < y_bound:
                            raw_val = (dma_word >> (psum_bits * i)) & psum_mask
                            target_fmap[f, x, y] = _signed(raw_val, psum_bits)
                        x += 1
                    if x_repetitions > 1:
                        if x - x_block_base >= x_block_width:
                            f += 1
                            x = x_block_base
                            if f == layer_parameters.filters:
                                f = 0
                                x_block_base += x_block_width
                                x = x_block_base
                                if x_block_base >= layer_parameters.iact_x_add_up:
                                    x_block_base = 0
                                    x = 0
                                    y += 1
                                    if y >= layer_parameters.iact_size_y:
                                        y = 0
                    else:
                        if x >= x_wrap_limit:
                            x = 0
                            f += 1
                            if f == layer_parameters.filters:
                                f = 0
                                y += 1
                                if y >= layer_parameters.iact_size_y:
                                    y = 0
                else:
                    if debug_enabled:
                        for _ in range(words):
                            storage_file.write("Empty storage line.\n")
                current_cycle += 1
                if current_cycle == transmissions_per_cycle:
                    current_cycle = 0

            if random.randint(1, 100) <= chance:
                read_data = 1
                cocotb.start_soon(set_input(ptp, dut.ready_dma_i, 1))
            else:
                read_data = 0
                cocotb.start_soon(set_input(ptp, dut.ready_dma_i, 0))
            await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

        cocotb.start_soon(set_input(ptp, dut.ready_dma_i, 0))

    if math.floor(
        layer_repetition % (layer_parameters.iact_transmissions_pe * layer_parameters.needed_wght_transmissions)
    ) == (layer_parameters.iact_transmissions_pe * layer_parameters.needed_wght_transmissions - 1):
        les.y_corner_start = y
        les.x_corner_start = x
    if debug_enabled:
        txt_file.close()
        storage_file.write(f"f: {f} x: {x} y: {y}\n")
        storage_file.write(
            f" f_corner_start: {les.f_corner_start} y_corner_start: {les.y_corner_start} x_corner_start: {les.x_corner_start}\n"
        )
        storage_file.write(
            f"iact_transmissions_pe: {layer_parameters.iact_transmissions_pe} needed_wght_transmissions: {layer_parameters.needed_wght_transmissions}\n"
        )
        storage_file.close()
        logger.debug("POST")
        logger.debug(
            "f: %s x: %s y: %s f_corner_start: %s y_corner_start: %s x_corner_start: %s\n",
            f,
            x,
            y,
            les.f_corner_start,
            les.y_corner_start,
            les.x_corner_start,
        )


async def compare_stream_Conv(
    ptp, dut, layer_number, layer_repetition, layer_parameters, oep, les, dram, login_level, output_order, rel_layers
):
    """Compare convolutional layer output stream with expected results."""
    await _compare_stream_spatial(
        ptp,
        dut,
        layer_number,
        layer_repetition,
        layer_parameters,
        oep,
        les,
        dram,
        login_level,
        output_order,
        rel_layers,
        x_wrap_limit=layer_parameters.iact_x_add_up,
    )


async def compare_stream_Dw(
    ptp, dut, layer_number, layer_repetition, layer_parameters, oep, les, dram, login_level, output_order, rel_layers
):
    """Compare depthwise convolution layer output stream with expected results."""
    await _compare_stream_spatial(
        ptp,
        dut,
        layer_number,
        layer_repetition,
        layer_parameters,
        oep,
        les,
        dram,
        login_level,
        output_order,
        rel_layers,
        x_wrap_limit=layer_parameters.iact_size_x,
    )


psum_stream_counts = {}


async def probe_psum_stream(ptp, dut, oep):
    """Count how many psum values each router actually emits."""
    while True:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        try:
            value = int(dut.psum_enable_o.value)
        except ValueError:
            continue
        for bit in range(oep.Clusters * oep.NUM_GLB_PSUM):
            if (value >> bit) & 1:
                psum_stream_counts[bit] = psum_stream_counts.get(bit, 0) + 1


async def trace_conv_writeback(ptp, dut, max_lines=160):
    """Trace psum selection, quantization and activation writes after settling."""
    pp = dut.psum_pipeline_inst

    def value(signal):
        try:
            return int(signal.value)
        except ValueError:
            return "X"

    def words(signal, width):
        bits = str(signal.value)
        return [
            int(bits[max(0, end - width) : end], 2)
            if all(b in "01" for b in bits[max(0, end - width) : end])
            else "X"
            for end in range(len(bits), 0, -width)
        ]

    lines = 0
    while lines < max_lines:
        await FallingEdge(dut.clk_i)
        state = value(pp.fsm_psum_current_state)
        if state not in (3, 4, 6) and not (state == 0 and value(dut.fsm_current_state) == 11):
            continue
        logger.info(
            "convwrite st=%s cyc=%s select=%s addr=%s rd=%s pre=%s "
            "router=%s pen=%s pwen=%s pren=%s pdata=%s "
            "filter=%s mant=%s shift=%s offset=%s quant=%s wen=%s wdata=%s",
            state,
            value(pp.fsm_psum_cycle),
            value(pp.psum_cluster_select),
            words(pp.psum_buffer_addr, int(dut.BUFFER_WIDTH_PSUM.value)),
            words(pp.psum_buffer_data_r, len(pp.pre_quantized_value[0])),
            [value(pp.pre_quantized_value[i]) for i in range(len(pp.pre_quantized_value))],
            value(pp.router_mode_psum),
            value(pp.psum_enable_o),
            value(pp.psum_buffer_en_w),
            value(pp.psum_buffer_en_r),
            words(pp.psum_buffer_data_w, len(pp.pre_quantized_value[0])),
            value(pp.current_filter),
            value(pp.gen_quant_unit[0].u_quant_unit.quant_mant),
            value(pp.current_shift),
            value(pp.gen_quant_unit[0].u_quant_unit.quant_offset),
            words(pp.quantized_value_flat, 8),
            [value(dut.iact_buffer_en_w[i]) for i in range(len(dut.iact_buffer_en_w))],
            words(dut.iact_buffer_data_w, 64),
        )
        lines += 1


async def trace_wb_events(ptp, dut, max_lines=300):
    """Trace the interlayer write-back counters (opt-in: TRACE_WB_EVENTS)."""
    pp = dut.psum_pipeline_inst

    def value(signal):
        try:
            return int(signal.value)
        except ValueError:
            return "X"

    lines = 0
    last = None
    while lines < max_lines:
        await FallingEdge(dut.clk_i)
        if value(dut.fsm_current_state) != 11:
            continue
        en = [value(dut.iact_buffer_en_w[i]) for i in range(len(dut.iact_buffer_en_w))]
        key = (
            value(pp.fsm_psum_current_state),
            value(dut.iact_to_psum_x_pos_counter),
            value(dut.iact_to_psum_storage_counter),
            value(dut.iact_to_psum_trans_counter),
            tuple(en),
        )
        if key == last:
            continue
        last = key
        try:
            addr0 = hex(int(dut.iact_buffer_addr_reg[0].value))
        except Exception:
            addr0 = "?"
        logger.info(
            "wbtrace addr0=%s t=%s psum_st=%s psum_cyc=%s fsm_cyc=%s xpos=%s trans=%s stor=%s shifting=%s en_w=%s",
            addr0,
            cocotb.utils.get_sim_time("ns"),
            value(pp.fsm_psum_current_state),
            value(pp.fsm_psum_cycle),
            value(dut.fsm_cycle),
            value(dut.iact_to_psum_x_pos_counter),
            value(dut.iact_to_psum_trans_counter),
            value(dut.iact_to_psum_storage_counter),
            value(dut.iact_to_psum_start_shifting),
            en,
        )
        lines += 1


async def trace_ram_writes(ptp, dut, max_lines=200):
    """Log every clock in which any iact RAM write enable is set (opt-in: TRACE_RAM_WRITES)."""

    def value(sig):
        try:
            return int(sig.value)
        except Exception:
            return "?"

    lines = 0
    while lines < max_lines:
        await FallingEdge(dut.clk_i)
        en = [value(dut.iact_buffer_en_w[i]) for i in range(len(dut.iact_buffer_en_w))]
        if 1 not in en:
            continue
        try:
            wdata = hex(int(str(dut.iact_buffer_data_w.value)[-64:], 2))
        except Exception:
            wdata = "?"
        logger.info(
            "ramwrite t=%s fsm=%s last=%s addr0=%s choose=%s en=%s wdata_cell0=%s",
            cocotb.utils.get_sim_time("ns"),
            value(dut.fsm_current_state),
            value(dut.fsm_last_state),
            value(dut.iact_buffer_addr_reg[0]),
            value(dut.choose_iact_buffer),
            en,
            wdata,
        )
        lines += 1


async def trace_pe_spads(ptp, dut, oep):
    """Print operand SPAD contents of cluster (0,0) column 0 whenever they change."""
    parts = [float(v) for v in os.environ.get("TRACE_PE_SPADS", "0,1e9").split(",")]
    t0, t1 = parts[0], parts[1]
    col = int(parts[2]) if len(parts) > 2 else 0
    pes = []
    try:
        for row in range(oep.PEs_Y):
            pes.append(
                dut.OpenEye_Parallel.gen_x[0].gen_y[0].OpenEye_Cluster.pe_cluster.gen_X[col].gen_Y[row].pe
            )
    except Exception as exc:
        logger.error("trace_pe_spads: PE not reachable (%s)", type(exc).__name__)
        return

    def payloads(mem, width, count):
        out = []
        for addr in range(count):
            try:
                w = int(mem[addr].value)
            except ValueError:
                out.append(None)
                continue
            except IndexError:
                break
            v = w & 0xFF
            out.append(v - 256 if v & 0x80 else v)
        return out

    last = {}
    while True:
        await Timer(100, unit="ns")
        now = cocotb.utils.get_sim_time("ns")
        if now < t0:
            continue
        if now > t1:
            return
        for row, pe in enumerate(pes):
            try:
                iv = payloads(pe.iact_data_SPad.ram.impl.mem, 8, 24)
                wv = payloads(pe.weight_data_SPad.ram.impl.mem, 8, 28)
            except Exception:
                continue
            try:
                wa = []
                wmem = pe.gen_wght_addr_spad.weight_addr_SPad.ram.impl.mem
                for a in range(20):
                    try:
                        wa.append(int(wmem[a].value))
                    except ValueError:
                        wa.append(None)
            except Exception:
                wa = None
            try:
                ia = []
                for a in range(8):
                    try:
                        ia.append(int(pe.gen_iact_addr_spad.iact_addr_SPad.ram.impl.mem[a].value))
                    except ValueError:
                        ia.append(None)
            except Exception:
                ia = None
            key = (tuple(iv), tuple(wv), tuple(wa or ()), tuple(ia or ()))
            if last.get(row) != key:
                last[row] = key
                logger.info("spads t=%s row=%d iact=%s wght=%s waddr_spad=%s iaddr_spad=%s", now, row, iv, wv, wa, ia)


async def trace_pe_wght_addrs(ptp, dut, oep):
    """Per compute pass, log weight-data SPAD addresses one PE reads."""
    parts = [float(v) for v in os.environ.get("TRACE_PE_WADDR", "0,0,0,1").split(",")]
    cx, cy, col, row = (int(v) for v in parts[:4])
    t0, t1 = (parts[4], parts[5]) if len(parts) >= 6 else (0.0, 1e12)
    try:
        pe = (
            dut.OpenEye_Parallel.gen_x[cx].gen_y[cy].OpenEye_Cluster.pe_cluster.gen_X[col].gen_Y[row].pe
        )
    except Exception as exc:
        logger.error("trace_pe_wght_addrs: PE not reachable (%s)", type(exc).__name__)
        return

    def iact_payloads():
        out = []
        try:
            mem = pe.iact_data_SPad.ram.impl.mem
            for a in range(8):
                try:
                    out.append(int(mem[a].value) & 0xFF)
                except ValueError:
                    out.append(None)
        except Exception:
            pass
        return out

    in_calc = False
    addrs = []
    passno = 0
    start = 0
    snap = None
    while True:
        await FallingEdge(dut.clk_i)
        now = cocotb.utils.get_sim_time("ns")
        if now < t0:
            continue
        if now > t1:
            return
        try:
            st = int(pe.current_state_computing.value)
        except ValueError:
            continue
        if st == 6:
            if not in_calc:
                in_calc = True
                addrs = []
                start = now
                snap = iact_payloads()
            try:
                if int(pe.wght_data_SPad_en_r.value):
                    addrs.append(int(pe.wght_data_SPad_addr.value))
            except (ValueError, AttributeError):
                pass
        elif in_calc:
            in_calc = False
            logger.info(
                "waddr pass=%d t=%s iact=%s waddrs(min,max,n)=%s uniq=%s",
                passno,
                start,
                snap,
                (min(addrs), max(addrs), len(addrs)) if addrs else None,
                sorted(set(addrs))[:40],
            )
            passno += 1


async def trace_pe_wght_writes(ptp, dut, oep):
    """Log weight/iact SPAD writes and compute pulses of one PE."""
    parts = [float(v) for v in os.environ.get("TRACE_PE_WGHT", "0,0,0,1").split(",")]
    cx, cy, col, row = (int(v) for v in parts[:4])
    t0, t1 = (parts[4], parts[5]) if len(parts) >= 6 else (0.0, 1e12)
    try:
        pe = (
            dut.OpenEye_Parallel.gen_x[cx].gen_y[cy].OpenEye_Cluster.pe_cluster.gen_X[col].gen_Y[row].pe
        )
        ws = pe.weight_data_SPad
        isp = pe.iact_data_SPad
    except Exception as exc:
        logger.error("trace_pe_wght_writes: PE not reachable (%s)", type(exc).__name__)
        return

    def val(sig):
        try:
            return int(sig.value)
        except ValueError:
            return None

    try:
        wa = getattr(pe, "gen_wght_addr_spad").weight_addr_SPad
    except Exception as exc:
        wa = None
        logger.info("pw: weight addr SPad not reachable (%s)", type(exc).__name__)
    lines = 0
    while lines < 4000:
        await FallingEdge(dut.clk_i)
        now = cocotb.utils.get_sim_time("ns")
        if now < t0:
            continue
        if now > t1:
            return
        if wa is not None and val(wa.we_i) == 1:
            logger.info("pw t=%s WA addr=%s data=%s", now, val(wa.addr_i), val(wa.data_i))
            lines += 1
        if val(pe.wght_addr_SPad_en_r) == 1 and val(pe.current_state_computing) in (1, 2, 3, 4, 5):
            logger.info(
                "pw t=%s RA st=%s addr=%s data_r=%s",
                now,
                val(pe.current_state_computing),
                val(pe.wght_addr_SPad_addr),
                val(pe.wght_addr_SPad_data_r),
            )
            lines += 1
        if val(pe.compute_i) == 1:
            logger.info("pw t=%s compute_i state=%s", now, val(pe.current_state_computing))
            lines += 1
        if val(ws.we_i) == 1:
            d = val(ws.data_i)
            logger.info("pw t=%s W addr=%s pay0=%s", now, val(ws.addr_i), None if d is None else (d & 0xFF))
            lines += 1
        if val(isp.we_i) == 1:
            d = val(isp.data_i)
            logger.info(
                "pw t=%s I addr=%s pay=%s oh=%s",
                now,
                val(isp.addr_i),
                None if d is None else (d & 0xFF),
                None if d is None else (d >> 8),
            )
            lines += 1


async def trace_pe_pass_counts(ptp, dut, oep, max_lines=200000):
    """Count how many times each PE completes a compute pass (opt-in: TRACE_PE_PASSES=t0,t1)."""
    try:
        t0, t1 = [float(v) for v in os.environ.get("TRACE_PE_PASSES", "0,1e9").split(",")]
    except ValueError:
        t0, t1 = 0.0, 1e9
    pes = {}
    for cx in range(oep.Clusters_X):
        for cy in range(oep.Clusters_Y):
            try:
                base = dut.OpenEye_Parallel.gen_x[cx].gen_y[cy].OpenEye_Cluster.pe_cluster
                for c in range(oep.PEs_X):
                    for r in range(oep.PEs_Y):
                        pes[(cx, cy, c, r)] = base.gen_X[c].gen_Y[r].pe
            except Exception as exc:
                logger.error("trace_pe_pass_counts: cluster (%d,%d) not reachable (%s)", cx, cy, type(exc).__name__)
                return

    def state(pe):
        try:
            return int(pe.current_state_computing.value)
        except ValueError:
            return None

    counts = {k: 0 for k in pes}
    last = {k: None for k in pes}
    lines = 0
    while lines < max_lines:
        await FallingEdge(dut.clk_i)
        now = cocotb.utils.get_sim_time("ns")
        if now < t0:
            continue
        if now > t1:
            break
        for key, pe in pes.items():
            st = state(pe)
            if st == 8 and last[key] != 8:
                counts[key] += 1
            last[key] = st
        lines += 1
    for key in sorted(counts):
        cx, cy, c, r = key
        logger.info("passcount c(%d,%d) col=%d row=%d passes=%d", cx, cy, c, r, counts[key])


async def trace_pooling(ptp, dut, max_lines=400):
    """Trace the max-pooling FSM (opt-in: TRACE_POOLING=1 or TRACE_POOLING=t0,t1)."""
    try:
        t0, t1 = [float(v) for v in os.environ.get("TRACE_POOLING", "0,1e12").split(",")]
    except ValueError:
        t0, t1 = 0.0, 1e12

    def value(sig):
        try:
            return int(sig.value)
        except Exception:
            return "?"

    def hexv(sig):
        try:
            t = str(sig.value)
            return hex(int(t, 2)) if set(t) <= {"0", "1"} else t
        except Exception:
            return "?"

    def bytes_w(sig):
        try:
            t = str(sig.value)
            if not set(t) <= {"0", "1"}:
                return "X..."
            v = int(t, 2)
            return " ".join("%02x" % ((v >> (8 * i)) & 0xFF) for i in range(len(t) // 8))
        except Exception:
            return "?"

    def arr(name, n):
        out = []
        for i in range(n):
            try:
                out.append(value(getattr(dut, name)[i]))
            except Exception:
                out.append("?")
        return out

    lines = 0
    in_pool = False
    while lines < max_lines:
        await FallingEdge(dut.clk_i)
        now = cocotb.utils.get_sim_time("ns")
        if now < t0:
            continue
        if now > t1:
            return
        st = value(dut.fsm_current_state)
        if st not in (12, 13, 14):
            in_pool = False
            continue
        if not in_pool:
            in_pool = True
            logger.info("poolstart t=%s", cocotb.utils.get_sim_time("ns"))
            for addr in range(4):
                logger.info(
                    "poolram addr=%d %s",
                    addr,
                    " | ".join(
                        ("--------" if w is None else " ".join("%02x" % ((w >> (8 * i)) & 0xFF) for i in range(8)))
                        for w in (_ram_word(dut.BUFFER_A[c].iact_layer_buffer.impl.mem[addr]) for c in range(4))
                    ),
                )
        rb = dut.ring_buffer_pipelined_inst
        logger.info(
            "pooltrace t=%s st=%s cyc=%s cnv=%s fin=%s sel=%s sel2=%s addr0=%s rd=%s "
            "s1=%s s2=%s old=%s new=%s en_w=%s wr=%s rdp=%s rdy=%s spts=%s lim=%s wbus=[%s]",
            cocotb.utils.get_sim_time("ns"),
            st,
            value(dut.fsm_cycle),
            value(dut.iact_converter_cycles),
            value(dut.finished_cycles_iact),
            value(dut.select_ram_counter),
            value(dut.select_ram_counter2),
            hexv(dut.iact_buffer_addr_reg[0]),
            hexv(dut.iact_buffer_data_r),
            arr("pooling_stage_1", 8),
            arr("pooling_stage_2", 4),
            arr("pooling_buffer_old", 4),
            arr("pooling_buffer_new", 4),
            [value(dut.iact_buffer_en_w[i]) for i in range(len(dut.iact_buffer_en_w))],
            value(rb.wr_ptr),
            value(rb.rd_ptr),
            value(dut.pooling_buffer_enable),
            value(dut.set_pointer_start),
            value(rb.limit_i),
            bytes_w(dut.iact_buffer_data_w),
        )
        lines += 1


async def trace_psum_capture(ptp, dut, oep, max_lines=400):
    """Per-cycle trace of the psum feed and capture (opt-in: TRACE_PSUM_CAPTURE)."""
    pp = dut.psum_pipeline_inst
    try:
        aw = int(dut.BUFFER_WIDTH_PSUM.value)
    except Exception:
        aw = 14

    def field(sig, lo, width, signed=None):
        txt = str(sig.value)
        bits = txt[len(txt) - lo - width : len(txt) - lo]
        if any(c not in "01" for c in bits):
            return "X"
        v = int(bits, 2)
        return _signed(v & ((1 << signed) - 1), signed) if signed else v

    lines = 0
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        try:
            state = int(pp.fsm_psum_current_state.value)
        except ValueError:
            continue
        if state not in (2, 3):
            continue
        logger.info(
            "psumtrace t=%s st=%d rr=%s rdy_o=%s rdy_i=%s cyc=%s addr=%s rd=%s data_i=%s en_i=%s en_o=%s wen=%s",
            cocotb.utils.get_sim_time("ns"),
            state,
            str(pp.results_ready.value),
            str(pp.psum_ready_o_reg.value),
            str(pp.psum_ready_i_reg.value),
            field(pp.fsm_psum_cycle, 0, 8),
            field(pp.psum_buffer_addr, 0, aw),
            field(pp.psum_buffer_data_r, 0, 32, 20),
            field(pp.psum_data_i_reg, 0, 32, 20),
            field(pp.psum_enable_i_reg, 0, 1),
            field(pp.psum_enable_o, 0, 1),
            field(pp.psum_buffer_en_w, 0, 1),
        )
        lines += 1


async def trace_pe_psum(ptp, dut, oep, max_lines=500):
    """Per-cycle psum SPAD port trace of one PE (opt-in: TRACE_PE_PSUM)."""
    pe = (
        dut.OpenEye_Parallel.gen_x[0]
        .gen_y[0]
        .OpenEye_Cluster.pe_cluster.gen_X[0]
        .gen_Y[oep.PEs_Y - 1]
        .pe
    )
    spads = [pe.gen_serial_psum_spad.gen_psum_spad[l].psum_SPad for l in range(oep.PARALLEL_MACS)]

    def val(sig, signed_bits=None):
        try:
            v = int(sig.value)
        except ValueError:
            return "X"
        return _signed(v & ((1 << signed_bits) - 1), signed_bits) if signed_bits else v

    lines = 0
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        we = [val(sp.we_i) for sp in spads]
        pen = val(pe.psum_enable_i)
        if pen != 1 and not any(w == 1 for w in we):
            continue
        lanes = [
            "L%d we=%s aw=%s d=%s ar=%s" % (l, we[l], val(sp.addr_w_i), val(sp.data_i, 20), val(sp.addr_r_i))
            for l, sp in enumerate(spads)
        ]
        logger.info(
            "petrace t=%s st=%s pen=%s pin=%s | %s",
            cocotb.utils.get_sim_time("ns"),
            val(pe.current_state_computing),
            pen,
            val(pe.psum_data_i, 20),
            " | ".join(lanes),
        )
        lines += 1


async def trace_bias_load(ptp, dut, oep, max_lines=120):
    """Trace psum-buffer writes while the main FSM is in GET_BIAS (state 5)."""
    pp = dut.psum_pipeline_inst
    ng = oep.NUM_GLB_PSUM
    cr = oep.Clusters_Y
    try:
        aw = int(dut.BUFFER_WIDTH_PSUM.value)
    except Exception:
        aw = 14
    pairs = (ng + 1) // 2

    def field(sig, lo, width, signed=None):
        txt = str(sig.value)
        bits = txt[len(txt) - lo - width : len(txt) - lo]
        if any(c not in "01" for c in bits):
            return "X"
        v = int(bits, 2)
        return _signed(v & ((1 << signed) - 1), signed) if signed else v

    lines = 0
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        try:
            st = int(dut.fsm_current_state.value)
        except ValueError:
            continue
        if st != 5:
            continue
        logger.info(
            "biastrace t=%s dma_in=%s en_w=%s addr[x0,x1]=%s,%s data_w[x0,x1]=%s,%s",
            cocotb.utils.get_sim_time("ns"),
            field(pp.data_dma_i_reg, 0, 32, 20),
            field(pp.psum_buffer_en_w, 0, oep.Clusters_X * cr * pairs),
            field(pp.psum_buffer_addr, 0, aw),
            field(pp.psum_buffer_addr, cr * pairs * aw, aw),
            field(pp.psum_buffer_data_w, 0, 32, 20),
            field(pp.psum_buffer_data_w, 32 * cr * ng, 32, 20),
        )
        lines += 1


async def trace_converter_gate(ptp, dut, oep, col=0, row=0, max_lines=80):
    """Trace the per-cluster acceptance gate inside one iact converter."""
    try:
        inst = dut.IACT_CONVERTER_X[col].IACT_CONVERTER_Y[row].iact_stream_constructor
    except Exception as exc:
        logger.error("converter gate: instance unreachable (%s)", type(exc).__name__)
        return
    tag = "c%dr%d" % (col, row)
    names = (
        "fsm_current_state",
        "enable_write_to_storage",
        "x_pos_in_w_cycle",
        "x_range_lower_bound",
        "x_start",
        "iact_values_per_cluster_transmit",
        "x_pos_inc",
        "fsm_row_offset",
        "ram_wr_en",
        "fsm_cycle",
        "needed_iact_buffer_words_i",
        "fully_connected_i",
    )
    lines, prev = 0, None
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        vals = []
        for n in names:
            try:
                vals.append(int(getattr(inst, n).value))
            except Exception:
                vals.append(-1)
        if vals == prev:
            continue
        prev = vals
        logger.error("convgate[%s] %s", tag, " ".join("%s=%d" % (n, v) for n, v in zip(names, vals)))
        lines += 1


def dump_iact_buffer_values(dut, oep, cells=4, addrs=6):
    """Print the activations actually stored in BUFFER_A."""
    logger.error("BUFFER_A contents (payload bytes per cell):")
    for cell in range(cells):
        vals = []
        try:
            mem = dut.BUFFER_A[cell].iact_layer_buffer.impl.mem
        except Exception as exc:
            logger.error("  cell %d unreachable (%s)", cell, type(exc).__name__)
            continue
        for addr in range(addrs):
            try:
                word = int(mem[addr].value)
            except ValueError:
                vals.append("X")
                continue
            except IndexError:
                break
            vals.append("/".join(str((word >> (b * 8)) & 0xFF) for b in range(8)))
        logger.error("  cell %d: %s", cell, " | ".join(vals))


async def trace_convert_window(ptp, dut, oep, max_lines=40):
    """Trace the CONVERT_IACT sliding window: current_x/current_y and the gate."""
    names = (
        "fsm_cycle",
        "current_x",
        "current_y",
        "iact_size_x",
        "iact_size_y",
        "x_bound",
        "relative_pos",
        "current_pos_in_iact_glb",
        "iact_channel_sending_cycle1",
        "iact_channel_sending_cycle2",
        "iact_channels_counter",
    )
    lines = 0
    prev = None
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        try:
            if int(dut.fsm_current_state.value) != 8:
                continue
        except ValueError:
            continue
        vals = []
        for n in names:
            try:
                v = int(getattr(dut, n).value)
                if n in ("current_x", "current_y") and v > (1 << 11):
                    v -= 1 << 12
                vals.append(v)
            except Exception:
                vals.append(-1)
        if vals == prev:
            continue
        prev = vals
        logger.error("convwin %s", " ".join("%s=%d" % (n, v) for n, v in zip(names, vals)))
        lines += 1


async def dump_convert_iact_limits(ptp, dut, oep):
    """Log the counters that bound CONVERT_IACT once the state exits."""
    names = (
        "iact_glb_writing_cycles",
        "iact_channel_max_cycles",
        "iact_size_c",
        "iact_converter_max_cycles",
        "channel_div_trans",
        "iact_x_add_up",
        "iact_size_x",
        "iact_size_y",
        "iact_x_per_cluster",
        "iact_values_per_cluster_transmit",
        "needed_iact_buffer_words",
        "iact_words_per_compute",
        "fully_connected_layer",
    )
    seen = False
    last_cycle = 0
    while True:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        try:
            state = int(dut.fsm_current_state.value)
        except ValueError:
            continue
        if state == 8:
            seen = True
            try:
                last_cycle = int(dut.fsm_cycle.value)
            except Exception:
                pass
            continue
        if seen and state != 8:
            vals = []
            for n in names:
                try:
                    vals.append("%s=%d" % (n, int(getattr(dut, n).value)))
                except Exception:
                    vals.append("%s=?" % n)
            logger.error("CONVERT_IACT ended at fsm_cycle=%d; %s", last_cycle, " ".join(vals))
            return


async def trace_iact_lanes(ptp, dut, oep, cluster=0, max_words=40, give_up_after=1200):
    """Record the activation word each iact GLB lane delivers, per cluster."""
    lanes = {}
    cycles = 0
    try:
        width = int(dut.TRANS_BITWIDTH_IACT.value)
    except Exception:
        width = oep.IACT_WOH_Bitwidth
    pay_bits = oep.IACT_Bitwidth
    while True:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        cycles += 1
        if cycles >= give_up_after:
            break
        try:
            en = int(dut.iact_enable_i_oep_w.value)
            rdy = int(dut.iact_ready_o_oep_w.value)
            data = int(dut.iact_data_i_oep_w.value)
        except Exception:
            continue
        hs = en & rdy
        if not hs:
            continue
        for g in range(oep.NUM_GLB_IACT):
            bit = cluster * oep.NUM_GLB_IACT + g
            if not ((hs >> bit) & 1):
                continue
            word = (data >> (bit * width)) & ((1 << width) - 1)
            val = _signed(word & ((1 << pay_bits) - 1), pay_bits)
            if val == 0:
                continue
            seq = lanes.setdefault(g, [])
            if len(seq) < max_words:
                seq.append(val)
        if all(len(v) >= max_words for v in lanes.values()) and len(lanes) == oep.NUM_GLB_IACT:
            break
    if not lanes:
        logger.error("iact lane trace: no accepted transfers seen in %d cycles", cycles)
    for g in sorted(lanes):
        logger.error("iact lane g=%d into cluster %d (%d words): %s", g, cluster, len(lanes[g]), lanes[g])


async def trace_psum_capture_slices(ptp, dut, oep, max_lines=40):
    """Log psum_data_o_w split per cluster on every capture cycle."""
    tp = oep.DATA_PSUM_BITWIDTH
    try:
        width = int(dut.TRANS_BITWIDTH_PSUM.value)
    except Exception:
        width = tp
    lines = 0
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        try:
            enable = int(dut.psum_enable_o.value)
            if not enable:
                continue
            en = int(dut.psum_buffer_en_w.value)
            data = int(dut.psum_data_o_w.value)
            state = int(dut.fsm_psum_current_state.value)
        except Exception:
            continue
        parts = []
        for cr in range(oep.Clusters_Y):
            for cc in range(oep.Clusters_X):
                base = cc * oep.NUM_GLB_PSUM + cr * oep.Clusters_X * oep.NUM_GLB_PSUM
                val = _signed((data >> (base * width)) & ((1 << tp) - 1), tp)
                parts.append("c(%d,%d)=%d" % (cc, cr, val))
        logger.error("capture psum_state=%d en_w=0x%x psum_enable_o=0x%x %s", state, en, enable, " ".join(parts))
        lines += 1


async def trace_pe_calc_loop(ptp, dut, oep, cl_x=0, cl_y=0, pe_row=0, pe_col=0, max_lines=120):
    """Trace one PE's sparse CALCULATING loop."""
    try:
        fsm = (
            dut.OpenEye_Parallel.gen_x[cl_x]
            .gen_y[cl_y]
            .OpenEye_Cluster.pe_cluster.gen_X[pe_col]
            .gen_Y[pe_row]
            .pe.gen_sparse_fsm
        )
    except Exception as exc:
        logger.error("trace_pe_calc_loop: PE unreachable (%s)", type(exc).__name__)
        return
    names = (
        "current_state_computing",
        "iact_addr_current",
        "iact_addr_count",
        "wght_data_vec",
        "wght_data_end",
        "wght_data_start",
        "wght_data_SPad_addr",
        "values_valid",
        "computing",
    )
    lines = 0
    prev = None
    while lines < max_lines:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
        vals = []
        for n in names:
            try:
                vals.append(int(getattr(fsm, n).value))
            except Exception:
                vals.append(-1)
        if vals == prev:
            continue
        prev = vals
        if vals[0] == 0 and vals[8] == 0:
            continue
        logger.error("calc %s", " ".join("%s=%d" % (n, v) for n, v in zip(names, vals)))
        lines += 1


def dump_pe_iact_config(dut, oep, pe_col=0):
    """Print each PE's iact loop bounds and how many activations it holds."""
    logger.error("PE iact config and SPAD payload counts:")
    for cl_y in range(oep.Clusters_Y):
        for cl_x in range(oep.Clusters_X):
            for pe_row in range(oep.PEs_Y):
                try:
                    pe = (
                        dut.OpenEye_Parallel.gen_x[cl_x]
                        .gen_y[cl_y]
                        .OpenEye_Cluster.pe_cluster.gen_X[pe_col]
                        .gen_Y[pe_row]
                        .pe
                    )
                except Exception as exc:
                    logger.error(
                        "  cluster(%d,%d) row %d unreachable (%s)", cl_x, cl_y, pe_row, type(exc).__name__
                    )
                    continue
                fields = []
                for name in ("iact_addr_max_reg", "channel_reg_C0", "filters_reg_M0"):
                    try:
                        fields.append("%s=%d" % (name, int(getattr(pe, name).value)))
                    except Exception:
                        fields.append("%s=?" % name)
                nz = 0
                try:
                    mem = pe.iact_data_SPad.ram.impl.mem
                    pay_bits = int(dut.DATA_IACT_BITWIDTH.value)
                    for addr in range(len(mem)):
                        try:
                            word = int(mem[addr].value)
                        except ValueError:
                            continue
                        except IndexError:
                            break
                        if word & ((1 << pay_bits) - 1):
                            nz += 1
                except Exception as exc:
                    fields.append("spad=?(%s)" % type(exc).__name__)
                addrs = []
                try:
                    amem = pe.gen_iact_addr_spad.iact_addr_SPad.ram.impl.mem
                    for addr in range(8):
                        try:
                            addrs.append(int(amem[addr].value))
                        except ValueError:
                            addrs.append(None)
                        except IndexError:
                            break
                except Exception as exc:
                    addrs = ["?(%s)" % type(exc).__name__]
                vals = []
                try:
                    mem = pe.iact_data_SPad.ram.impl.mem
                    pay_bits = int(dut.DATA_IACT_BITWIDTH.value)
                    for addr in range(len(mem)):
                        try:
                            word = int(mem[addr].value)
                        except ValueError:
                            vals.append(None)
                            continue
                        except IndexError:
                            break
                        vals.append(_signed(word & ((1 << pay_bits) - 1), pay_bits))
                except Exception:
                    pass
                logger.error(
                    "  cluster(%d,%d) row %d: %s nonzero_payloads=%d iact_addr_spad=%s payloads=%s",
                    cl_x,
                    cl_y,
                    pe_row,
                    " ".join(fields),
                    nz,
                    addrs,
                    vals,
                )


def dump_pe_psum_values(dut, oep, pe_col=0, max_addr=8):
    """Print each PE's psum SPAD values."""
    logger.error("PE psum SPAD values (const operands: value == products accumulated):")
    for cl_y in range(oep.Clusters_Y):
        for cl_x in range(oep.Clusters_X):
            for pe_row in range(oep.PEs_Y):
                try:
                    pe = (
                        dut.OpenEye_Parallel.gen_x[cl_x]
                        .gen_y[cl_y]
                        .OpenEye_Cluster.pe_cluster.gen_X[pe_col]
                        .gen_Y[pe_row]
                        .pe
                    )
                    mem = pe.gen_serial_psum_spad.gen_psum_spad[0].psum_SPad.ram.impl.mem
                except Exception as exc:
                    logger.error(
                        "  cluster(%d,%d) row %d unreachable (%s)", cl_x, cl_y, pe_row, type(exc).__name__
                    )
                    continue
                vals = []
                for addr in range(max_addr):
                    try:
                        vals.append(_signed(int(mem[addr].value), oep.DATA_PSUM_BITWIDTH))
                    except ValueError:
                        vals.append(None)
                    except IndexError:
                        break
                logger.error("  cluster(%d,%d) row %d: %s", cl_x, cl_y, pe_row, vals)


def dump_pe_psum_spads(dut, oep, pe_col=0, max_addr=20):
    """Print the operand and psum SPADs of PEs inside the FPGA design."""
    for cl_x in range(oep.Clusters_X):
        for cl_y in range(oep.Clusters_Y):
            for pe_row in range(oep.PEs_Y):
                try:
                    pe = (
                        dut.OpenEye_Parallel.gen_x[cl_x]
                        .gen_y[cl_y]
                        .OpenEye_Cluster.pe_cluster.gen_X[pe_col]
                        .gen_Y[pe_row]
                        .pe
                    )
                except Exception as exc:
                    logger.info("PE [%d][%d][row %d] not reachable (%s)", cl_x, cl_y, pe_row, type(exc).__name__)
                    return
                occ = []
                for label, path in (("iact", "iact_data_SPad"), ("wght", "weight_data_SPad")):
                    try:
                        m = getattr(pe, path).ram.impl.mem
                        written = 0
                        for addr in range(64):
                            try:
                                int(m[addr].value)
                                written += 1
                            except ValueError:
                                pass
                            except IndexError:
                                break
                        occ.append("%s=%d" % (label, written))
                    except Exception as exc:
                        occ.append("%s=?(%s)" % (label, type(exc).__name__))
                logger.info(
                    "PE[x=%d][y=%d][col=%d][row=%d] written operand words (of first 64): %s",
                    cl_x,
                    cl_y,
                    pe_col,
                    pe_row,
                    " ".join(occ),
                )
                if os.environ.get("DUMP_WGHT_WORDS") and cl_y == 0 and pe_row == oep.PEs_Y - 1:
                    for path in ("weight_data_SPad", "gen_wght_addr_spad.weight_addr_SPad"):
                        try:
                            m = pe
                            for part in path.split("."):
                                m = getattr(m, part)
                            m = m.ram.impl.mem
                        except Exception as exc:
                            logger.info("PE[x=%d] %s not reachable (%s)", cl_x, path, type(exc).__name__)
                            continue
                        dec = []
                        for addr in range(28):
                            try:
                                w = int(m[addr].value)
                            except ValueError:
                                dec.append("X")
                                continue
                            except IndexError:
                                break
                            if path == "weight_data_SPad":
                                lanes = []
                                for lane in range(oep.PARALLEL_MACS):
                                    f = (w >> (12 * lane)) & 0xFFF
                                    pay = f & 0xFF
                                    pay = pay - 256 if pay & 0x80 else pay
                                    lanes.append("%d:%d" % (f >> 8, pay))
                                dec.append("/".join(lanes))
                            else:
                                dec.append(str(w))
                        logger.info("PE[x=%d][row=%d] %s = %s", cl_x, pe_row, path, " ".join(dec))
                for lane in range(oep.PARALLEL_MACS):
                    try:
                        mem = pe.gen_serial_psum_spad.gen_psum_spad[lane].psum_SPad.ram.impl.mem
                    except Exception as exc:
                        logger.info("PE [%d] lane %d SPAD not reachable (%s)", cl_x, lane, type(exc).__name__)
                        continue
                    words = []
                    for addr in range(max_addr):
                        try:
                            v = int(mem[addr].value)
                            words.append(v - (1 << 20) if v >> 19 else v)
                        except ValueError:
                            words.append("X")
                        except IndexError:
                            break
                    if any(w != "X" for w in words):
                        logger.info(
                            "PE[x=%d][y=%d][col=%d][row=%d] psum_SPad lane %d = %s",
                            cl_x,
                            cl_y,
                            pe_col,
                            pe_row,
                            lane,
                            words,
                        )


def dump_psum_buffers(dut, oep, max_addr=None):
    """Print the FPGA psum buffer RAM contents (even and odd GLB lanes)."""
    if max_addr is None:
        max_addr = int(os.environ.get("DUMP_PSUM_ADDRS", "8"))
    for cc in range(oep.Clusters_X):
        for cr in range(oep.Clusters_Y):
            for g in range((oep.NUM_GLB_PSUM + 1) // 2):
                try:
                    mem = dut.PSUM_RAM_X[cc].PSUM_RAM_Y[cr].PSUM_RAM_GLB[g].psum_buffer.impl.mem
                except Exception as exc:
                    logger.info("psum buffer [%d][%d][%d] not reachable (%s)", cc, cr, g, type(exc).__name__)
                    return
                bw = oep.DATA_PSUM_BITWIDTH
                lo_half, hi_half = [], []
                for addr in range(max_addr):
                    try:
                        word = int(mem[addr].value)
                    except ValueError:
                        lo_half.append("X")
                        hi_half.append("X")
                        continue
                    except IndexError:
                        break
                    lo_half.append(_signed(word & ((1 << bw) - 1), bw))
                    hi_half.append(_signed((word >> bw) & ((1 << bw) - 1), bw))
                logger.info("psum_buffer[x=%d][y=%d][glb=%d] even=%s", cc, cr, g, lo_half)
                logger.info("psum_buffer[x=%d][y=%d][glb=%d] odd =%s", cc, cr, g, hi_half)


async def check_fc_activation_writes(dut, params, layer, activations):
    """Check the FPGA's converter RAM writes against the FC input vector."""
    channels = layer.used_iact_per_PE
    banks = params.NUM_GLB_IACT
    pe_rows = params.PEs_Y
    rows = params.Clusters_Y
    writes_per_run = pe_rows * channels // 2
    expected_count = writes_per_run * layer.needed_wght_transmissions
    converters = []
    for column in range(params.Clusters_X):
        for row in range(rows):
            converter = dut.IACT_CONVERTER_X[column].IACT_CONVERTER_Y[row].iact_stream_constructor
            converters.append((column, row, converter))
    counts = {(column, row): 0 for column, row, _ in converters}
    while any(count < expected_count for count in counts.values()):
        await RisingEdge(dut.clk_i)
        for column, row, converter in converters:
            key = (column, row)
            for bank in range(banks):
                buffer = converter.BUFFER[bank]
                if hasattr(buffer, "SUBWORD"):
                    memories = [buffer.SUBWORD[slot].iact_buffer_SP for slot in range(2)]
                    if not int(memories[0].wr_en_i.value):
                        continue
                    assert int(memories[1].wr_en_i.value), "FC partial pair write"
                    assert int(memories[0].addr_i.value) == int(memories[1].addr_i.value)
                else:
                    memory = buffer.iact_buffer_SP
                    if not int(memory.wr_en_i.value):
                        continue
                counts[key] += 1
    logger.info("FC activation RAM writes checked: %s", counts)


async def compare_stream_Dense(
    ptp, dut, layer_number, layer_repetition, layer_parameters, oep, les, dram, login_level, rel_layers
):
    """Await the Dense/FC DMA output stream and store results into DRAM."""
    debug_enabled = logging.DEBUG >= login_level
    txt_file = None
    storage_file = None
    if debug_enabled:
        layer_dir = f"demo/layer_{layer_parameters.layer_name}"
        os.makedirs(layer_dir, exist_ok=True)
        txt_file = open(f"{layer_dir}/output.txt", "w")
        storage_file = open(f"{layer_dir}/storage_input.txt", "w")

    columns = oep.Clusters_X
    per_column = math.ceil(layer_parameters.used_psum_per_PE)
    tile = (layer_repetition // layer_parameters.iact_transmissions_pe) % layer_parameters.psum_transmissions_pe
    offset = tile * columns * per_column
    les.x = offset
    expected_words = columns * per_column * layer_parameters.output_cycles
    assert layer_parameters.psum_output_words == expected_words, (
        "Dense DMA output count disagrees with column/address layout"
    )
    beat = 0
    f = offset
    dut._log.info("Output Stream started")
    next_layers = rel_layers["Next_Layer"][layer_parameters.layer_name]
    dram_string = next_layers[0] if next_layers else "Output"
    target_vec = np.asarray(dram.fmap[dram_string][0]).reshape(-1)
    psum_bits = int(oep.DATA_PSUM_BITWIDTH)
    psum_mask = (1 << psum_bits) - 1

    while dut.enable_dma_o.value == 1:
        assert beat < expected_words, "Dense DMA emitted too many words"
        word = int(dut.data_dma_o.value)
        f = offset + (beat % columns) * per_column + beat // columns
        value = _signed(word & psum_mask, psum_bits)
        if f < target_vec.size:
            target_vec[f] = value
        else:
            assert value == 0, "Dense DMA padding output is nonzero"
        if debug_enabled:
            txt_file.write(format(word, f"0{oep.DMA_BITWIDTH}b") + "\n")
            storage_file.write(f"beat: {beat}, f: {f}\n")
        if os.environ.get("DUMP_PSUM_BUFFERS"):
            logger.info("Dense DMA beat=%d f=%d value=%d", beat, f, value)
        beat += 1
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    assert beat == expected_words, f"Dense DMA emitted {beat} words, expected {expected_words}"

    if os.environ.get("DUMP_PSUM_BUFFERS"):
        report_iact_handoff(oep)
        report_cluster_iact()
        report_pe_iact()
        report_converter_store()
        dump_compute_mask(dut, oep)
        dump_iact_path(dut, oep)
        dump_pe_occupancy(dut, oep)
        dump_pe_psum_spads(dut, oep)
        dump_pe_psum_values(dut, oep)
        dump_pe_iact_config(dut, oep)
        dump_psum_buffers(dut, oep)
        if psum_stream_counts:
            logger.info("psum_enable_o asserted cycles per bit: %s", dict(sorted(psum_stream_counts.items())))

    cocotb.start_soon(set_input(ptp, dut.ready_dma_i, 0))

    if debug_enabled:
        txt_file.close()
        storage_file.write(f"f: {f}\n")
        storage_file.close()
        logger.debug("POST")
        logger.debug("f: %s\n", f)


async def compare_stream_Pooling(
    ptp, dut, layer_number, layer_repetition, layer_parameters, oep, les, dram, login_level, rel_layers
):
    """Await the pooling output stream completion."""
    debug_enabled = logging.DEBUG >= login_level
    txt_file = None
    storage_file = None
    if debug_enabled:
        layer_dir = f"demo/layer_{layer_number}"
        os.makedirs(layer_dir, exist_ok=True)
        txt_file = open(f"{layer_dir}/output.txt", "w")
        storage_file = open(f"{layer_dir}/storage_input.txt", "w")

    f = 0
    dut._log.info("Output Stream started")
    while dut.enable_dma_o.value == 1:
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

    cocotb.start_soon(set_input(ptp, dut.ready_dma_i, 0))

    if debug_enabled:
        txt_file.close()
        storage_file.write(f"f: {f}\n")
        storage_file.close()
        logger.debug("POST")
        logger.debug("f: %s\n", f)


async def send_enable_conv(ptp, dut, layer_params, layer_repetition, oep):
    """Send enable signals for convolution layer operation."""
    cocotb.start_soon(
        set_input(ptp, dut.psum_enable_i, (1 << (oep.Clusters_X * oep.Clusters_Y * oep.NUM_GLB_PSUM)) - 1)
    )

    match layer_params.single_cluster_computation:
        case 1:
            cycles = math.ceil(
                (layer_params.filters * layer_params.output_shape[1] * layer_params.output_shape[2])
                / 2
                / (oep.Clusters_X * oep.Clusters_Y * oep.NUM_GLB_PSUM)
            )
        case 2:
            cycles = math.ceil(
                (layer_params.filters * layer_params.output_shape[1] * layer_params.output_shape[2])
                / (2 * oep.Clusters_X * oep.Clusters_Y * oep.NUM_GLB_PSUM)
            )
        case _:
            cycles = math.ceil(layer_params.filters / layer_params.needed_wght_transmissions / 2) * math.ceil(
                layer_params.needed_refreshes_mx[layer_repetition][0] / layer_params.used_Y_cluster
            )

    for _ in range(int(cycles)):
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)

    cocotb.start_soon(set_input(ptp, dut.psum_enable_i, 0))


async def send_enable_dw(ptp, dut, layer_params, layer_repetition, oep):
    """Send enable signals for depthwise convolution layer operation."""
    cocotb.start_soon(
        set_input(ptp, dut.psum_enable_i, (1 << (oep.Clusters_X * oep.Clusters_Y * oep.NUM_GLB_PSUM)) - 1)
    )
    cycles = math.ceil(layer_params.needed_refreshes_mx[layer_repetition][0] / layer_params.used_Y_cluster / 2)
    for _ in range(int(cycles)):
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp, dut.psum_enable_i, 0))


async def send_enable_dense(ptp, dut, layer_params, layer_repetition, oep):
    """Send enable signals for dense (fully-connected) layer operation."""
    cocotb.start_soon(
        set_input(ptp, dut.psum_enable_i, (1 << (oep.Clusters_X * oep.Clusters_Y * oep.NUM_GLB_PSUM)) - 1)
    )
    for _ in range(math.ceil(layer_params.used_psum_per_PE / 2)):
        await Timer(ptp.clk_cycle, unit=ptp.clk_cycle_unit)
    cocotb.start_soon(set_input(ptp, dut.psum_enable_i, 0))