#!/usr/bin/env python3
"""Generate asic/rtl/sg13g2/openeye.json (LibreLane Chip-flow config).

Reads the pad order from asic/rtl/sg13g2/pad_order.json (keep in sync with
gen_top.py) and fills in the SRAM macro views from the PDK.

Run:  python3 asic/scripts/gen_config.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
# macro_placement is a sibling script in asic/scripts (not a package)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from macro_placement import merge_into_macros  # noqa: E402
# The flow runs inside the iic-osic-tools container, where the repo is
# mounted at /foss/designs (see asic/scripts/env.sh) -- all config paths
# must be container paths.
CTR_REPO = "/foss/designs/OpenEye"
PDK = "/foss/pdks/ihp-sg13g2"
SRAM = f"{PDK}/libs.ref/sg13g2_sram"
OUT = REPO / "asic" / "rtl" / "sg13g2" / "openeye.json"

# the six RM_IHPSG13_1P macro types used by sg13g2_ram_sp.v
MACRO_TYPES = [
    "RM_IHPSG13_1P_1024x64_c2_bm_bist",
    "RM_IHPSG13_1P_1024x8_c2_bm_bist",
    "RM_IHPSG13_1P_2048x64_c2_bm_bist",
    "RM_IHPSG13_1P_2048x32_c2_bm_bist",
    "RM_IHPSG13_1P_4096x16_c3_bm_bist",
    "RM_IHPSG13_1P_4096x8_c3_bm_bist",
]

# corner wildcard -> SRAM liberty suffix
CORNERS = {
    "*_typ_1p20V_25C": "typ_1p20V_25C",
    "*_fast_1p32V_m40C": "fast_1p32V_m55C",
    "*_slow_1p08V_125C": "slow_1p08V_125C",
}

STDCELL = {
    "*_typ_1p20V_25C": f"{PDK}/libs.ref/sg13g2_stdcell/lib/sg13g2_stdcell_typ_1p20V_25C.lib.gz",
    "*_fast_1p32V_m40C": f"{PDK}/libs.ref/sg13g2_stdcell/lib/sg13g2_stdcell_fast_1p32V_m40C.lib.gz",
    "*_slow_1p08V_125C": f"{PDK}/libs.ref/sg13g2_stdcell/lib/sg13g2_stdcell_slow_1p08V_125C.lib.gz",
}

IOLIB = {
    "*_typ_1p20V_25C": f"{PDK}/libs.ref/sg13g2_io/lib/sg13g2_io_typ_1p2V_3p3V_25C.lib.gz",
    "*_fast_1p32V_m40C": f"{PDK}/libs.ref/sg13g2_io/lib/sg13g2_io_fast_1p32V_3p6V_m40C.lib.gz",
    "*_slow_1p08V_125C": f"{PDK}/libs.ref/sg13g2_io/lib/sg13g2_io_slow_1p08V_3p0V_125C.lib.gz",
}

HDL = f"{CTR_REPO}/hdl"

# RTL for synthesis (the .vh files staged into hdl/ resolve via relative
# include paths, no -I needed): chip top + RAM wrapper + pad blackbox +
# accelerator (incl. generated dma_storage.v).
VERILOG_FILES = [
    f"{CTR_REPO}/asic/rtl/sg13g2/OpenEye_Asic.v",
    f"{CTR_REPO}/asic/rtl/sg13g2/sg13g2_ram_sp.v",
    f"{CTR_REPO}/asic/rtl/sg13g2_io_blackbox.v",
] + sorted(f"{HDL}/{f}" for f in [
    "OpenEye_FPGA.v", "psum_pipeline.v", "dma_storage.v", "varlenFIFO.v",
    "OpenEye_Parallel.v", "OpenEye_Cluster.v", "GLB_cluster.v", "router_iact.v",
    "router_wght.v", "router_psum.v", "PE_cluster.v", "PE.v", "dsp_unit.v",
    "adder.v", "adder_tree.v", "delay_cluster.v", "data_pipeline.v",
    "data_pipeline_iact.v", "data_pipeline_wght.v", "multiplier.v", "mux2.v",
    "demux2.v", "mux_iact.v", "SPad_DP.v", "SPad_DP_RW.v", "SPad_SP.v",
    "quantization_unit.v", "RST_SYNC.v", "RAM_DP.v", "RAM_DP_RW.v", "RAM_SP.v",
    "RAM_DP_generic.v", "RAM_DP_RW_generic.v", "ring_buffer_pipelined.v",
    "RAM_SP_generic.v", "iact_stream_constructor.v", "router_configurator.v",
])


def main() -> None:
    pad_order = json.loads((REPO / "asic/rtl/sg13g2/pad_order.json").read_text())

    macros = {}
    for m in MACRO_TYPES:
        macros[m] = {
            "gds": f"{SRAM}/gds/{m}.gds",
            "lef": f"{SRAM}/lef/{m}.lef",
            "lib": {
                corner: [f"{SRAM}/lib/{m}_{sfx}.lib"]
                for corner, sfx in CORNERS.items()
            },
        }

    # Fixed R0 positions for the 33 SRAM instances. LibreLane's
    # Odb.ManualMacroPlacement only acts on MACROS.<cell>.instances and the
    # flow never places macros on its own -> without this, GeneratePDN dies
    # with [PDN-0235] Design has unplaced macros (happened in p2_11).
    merge_into_macros(macros)

    # CELL_LIBS replaces the PDK dict entirely -> stdcell + io + SRAMs
    cell_libs = {}
    for corner in CORNERS:
        sfx = CORNERS[corner]
        cell_libs[corner] = [
            STDCELL[corner],
            IOLIB[corner],
        ] + [f"{SRAM}/lib/{m}_{sfx}.lib" for m in MACRO_TYPES]

    config = {
        "meta": {
            "flow": "Chip",
        },
        "DESIGN_NAME": "OpenEye_Asic",
        "VERILOG_FILES": VERILOG_FILES,
        "VERILOG_DEFINES": [
            "USE_INTERNAL_PARAMS_PE",
            "USE_INTERNAL_PARAMS_PE_cluster",
            "OPENEYE_RAM_SPECIFIC",
            # yosys cannot resolve $value$plusargs (Icarus-only, used by the
            # FPGA debug waveform dump in OpenEye_FPGA.v)
            "NO_TRACE",
        ],
        "CLOCK_PORT": "clk_pad",
        # 2026-10-27 user decision: 50 MHz (20 ns) variant for the first
        # closed physical baseline. 10 ns is not closable on this netlist
        # (worst path ~19.6 ns nom_typ pre-resizer, wire-dominated
        # PE-fabric cross-chip net; 270k violating endpoints). See QWEN.md.
        "CLOCK_PERIOD": 20,
        # base.sdc: pad in/out delay = period * IO_DELAY_CONSTRAINT / 100
        # (default 20.0). At 20 ns the default would give 4.0 ns input
        # delay -- below the worst hold requirement (5.173 ns, DMA
        # input-capture regs, clock latency 4.95 ns). 32 -> 6.4 ns covers
        # it with ~1.5 ns margin. Assumption: external synchronous driver
        # at 50 MHz with <= 6 ns clock-to-pad -- verify against board spec.
        "IO_DELAY_CONSTRAINT": 32,
        # LibreLane 3.1 bug (librelane/steps/openroad.py, get_command):
        #     threads = str(self.config["OPENROAD_THREADS"]) or str(_get_process_limit())
        # str(None) == "None" is a truthy string, so the fallback never
        # fires: every OpenROAD step runs with "-threads None", OpenROAD
        # logs ORD-0032 and falls back to a single thread. All runtimes
        # recorded so far are single-threaded (p2_15 GRT 14:42 h,
        # post-CTS resizer 10:40 h, CTS 4:13 h; DetailedRouting would
        # take days). 48 threads on the 56-core host leaves headroom;
        # RAM is fine (openroad ~23 GB, host 251 GB).
        "OPENROAD_THREADS": 48,
        "FP_SIZING": "absolute",
        # 10.5 mm die: p2_14 floorplan metrics (measured): 44.7 mm^2
        # stdcell + 8.6 mm^2 SRAM macros = 53.2 mm^2 on the 97.2 mm^2 core
        # -> ~55 % utilization (stdcell + macros, pads excluded). 143 pads
        # at 80 um pitch (2.9 mm span, centered) leave ample room.
        "DIE_AREA": [0, 0, 10500, 10500],
        "LEFT_MARGIN_MULT": 667,
        "RIGHT_MARGIN_MULT": 667,
        "TOP_MARGIN_MULT": 85,
        "BOTTOM_MARGIN_MULT": 85,
        "PAD_SPACING_MULTIPLE": 80,
        "CELL_LIBS": cell_libs,
        "MACROS": macros,
        "EXTRA_VERILOG_MODELS": [
            f"{CTR_REPO}/asic/rtl/sg13g2_io_blackbox.v",
            # SRAM liberty files are not readable by yosys in non-`-lib`
            # mode -> clean blackbox models for lint + synthesis
            f"{CTR_REPO}/asic/rtl/sg13g2/sg13g2_sram_blackbox.v",
        ],
        "EXTRA_LEFS": [f"{CTR_REPO}/asic/rtl/bondpad_70x70.lef"],
        "EXTRA_GDS": [f"{CTR_REPO}/asic/rtl/bondpad_70x70.gds"],
        "PDN_CORE_RING": True,
        "PDN_CORE_RING_CONNECT_TO_PADS": True,
        "PDN_MACRO_CONNECTIONS": [
            ".* VPWR VGND vdd vss",
            ".* VPWR VGND iovdd iovss",
        ],
        # SRAM vddarray/vssarray pins have no PDK power net (gap, see README);
        # RUN_IRDROP_REPORT stays off until the pad-power gap is resolved.
        "RUN_IRDROP_REPORT": False,
        # p2_12 resizer incident (2026-10-27): step 36 ran 11 h+ with no stop
        # condition. Phase structure (from the run logs): the SETUP phase
        # gives up on its own after ~2.7 h (RSZ-0062; WNS -7.66 sum-of-
        # corners / -9.58 nom_typ -- structural, wire-dominated, not
        # closable by buffering), the HOLD phase is the long tail (THS
        # -27,166 -> -220 in ~8.5 h at a decaying rate; WHS flat at -4.064 on
        # the DMA input-capture regs _29536xx_/D -- a clock-latency vs
        # input-delay mismatch, also not buffer-fixable). Uncapped, the hold
        # tail grinds for days.
        # 1 % TNS cap = the knee of the measured curve. All four resizer TNS
        # knobs are capped (postCTS + postGRT x setup + hold): the caps bound
        # run time only -- the HoldViolations gate fails on WHS with or
        # without them. postCTS uses PL_RESIZER_*, postGRT GRT_RESIZER_*.
        "PL_RESIZER_SETUP_REPAIR_TNS_PCT": 1,
        "PL_RESIZER_HOLD_REPAIR_TNS_PCT": 1,
        "GRT_RESIZER_SETUP_REPAIR_TNS_PCT": 1,
        "GRT_RESIZER_HOLD_REPAIR_TNS_PCT": 1,
        # p2_14 GRT failure (2026-10-29 17:02): [GRT-0116] "Global routing
        # finished with congestion" after 50+50+50 iterations (~13.8 h).
        # Analysis of the final congestion report: GLOBAL usage is only
        # 30.3 % (24.2M / 80M resources) -- not a density problem, but
        # local overflow in the dense PE-fabric/SRAM regions (max 143
        # overflow units in one M3-H cell, 108 M2-V; 216,531 total
        # overflow = 0.9 % of demand, spread over many cells). Metal1
        # carries the PDN (routing starts at Metal2), so there is no
        # spare layer, and GRT already disabled the NDR on net781646 to
        # fight the congestion. 150 iterations could not drive the
        # residual to zero. With 6 usable layers at 30 % global usage
        # the residual is well inside what the detailed router (zroute)
        # can detour away -> let GRT finish and let DRT absorb it:
        "GRT_ALLOW_CONGESTION": True,
        "PAD_SOUTH": pad_order["SOUTH"],
        "PAD_EAST": pad_order["EAST"],
        "PAD_NORTH": pad_order["NORTH"],
        "PAD_WEST": pad_order["WEST"],
    }

    OUT.write_text(json.dumps(config, indent=4) + "\n")
    n_inst = sum(len(v.get("instances", {})) for v in macros.values())
    print(f"[OK] wrote {OUT} "
          f"({sum(len(v) for v in pad_order.values())} pads, {len(macros)} "
          f"macro types, {n_inst} placed macro instances)")


if __name__ == "__main__":
    main()
