#!/usr/bin/env python3
"""Generate asic/rtl/sg13g2/openeye.json (LibreLane Chip-flow config).

Reads the pad order from asic/rtl/sg13g2/pad_order.json (keep in sync with
gen_top.py) and fills in the SRAM macro views from the PDK.

Run:  python3 asic/scripts/gen_config.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
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
        "CLOCK_PERIOD": 10,
        "FP_SIZING": "absolute",
        # 10.5 mm die: post-synthesis area is 35.4 mm^2 logic (70 % PE-pad
        # latches) + 11.1 mm^2 SRAM = 46.5 mm^2 cells; the core
        # (~9.86 mm x 9.86 mm) gives ~48 % utilization. 143 pads at 80 um
        # pitch (2.9 mm span, centered) leave ample room.
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
        "PAD_SOUTH": pad_order["SOUTH"],
        "PAD_EAST": pad_order["EAST"],
        "PAD_NORTH": pad_order["NORTH"],
        "PAD_WEST": pad_order["WEST"],
    }

    OUT.write_text(json.dumps(config, indent=4) + "\n")
    print(f"[OK] wrote {OUT} "
          f"({sum(len(v) for v in pad_order.values())} pads, {len(macros)} macro types)")


if __name__ == "__main__":
    main()
