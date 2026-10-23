#!/usr/bin/env python3
"""Generate asic/rtl/sg13g2/OpenEye_Asic.v (chip top with pad ring) and
asic/rtl/sg13g2/pad_order.json (LibreLane PAD_* config lists).

Each pad gets a unique leaf instance name: the LibreLane PAD_SOUTH/EAST/
NORTH/WEST config lists reference pads by leaf name, so generate-loop
instances with repeated names cannot be used.

Pad layout (143 pads = 135 signal + 8 power, one vdd/vss pair per side):
    west  (36): clk, rst_n, data_dma_i[0:31]
    south (36): data_dma_i[32:63], enable_dma_i, ready_dma_i
    north (36): data_dma_o[0:31], enable_dma_o, ready_dma_o
    east  (35): data_dma_o[32:63], last_data_o

Run:  python3 asic/scripts/gen_top.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT_V = REPO / "asic" / "rtl" / "sg13g2" / "OpenEye_Asic.v"
OUT_JSON = REPO / "asic" / "rtl" / "sg13g2" / "pad_order.json"

HEADER = """\
// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: OpenEye_Asic
///
/// ASIC chip top for IHP SG13G2: OpenEye_FPGA compute core plus an explicit
/// sg13g2_io pad ring (Caravel-style, placed by OpenROAD.PadRing via the
/// PAD_SOUTH/EAST/NORTH/WEST LibreLane config lists, see pad_order.json).
///
/// GENERATED FILE -- do not edit by hand. Regenerate with:
///   python3 asic/scripts/gen_top.py
///
/// Pad map (143 pads = 135 signal + 8 power, non-inverting, 3.3V-tolerant IO):
///   inputs  (69): clk, rst_n (active low),
///                 data_dma_i[63:0], enable_dma_i, ready_dma_i
///   outputs (66): data_dma_o[63:0], ready_dma_o, enable_dma_o, last_data_o
///
/// Power (single-rail bring-up setup; the production board uses a separate
/// 3.3V IOVDD rail for the pad ring):
///   vdd/vss     = VPWR/VGND (1.2V core)
///   iovdd/iovss = tied to the core rails here
"""

# side -> list of pad instance names (power pads interleaved at the ends)
PAD_ORDER = {
    "WEST":  ["u_vdd0", "u_vss0"]
             + [f"u_dma_i_in{i}" for i in range(0, 32)]
             + ["u_clk_in", "u_rstn_in"],
    "SOUTH":  ["u_vdd1", "u_vss1"]
             + [f"u_dma_i_in{i}" for i in range(32, 64)]
             + ["u_eni_in", "u_rdyi_in"],
    "NORTH": ["u_vdd2", "u_vss2"]
             + [f"u_dma_o_out{i}" for i in range(0, 32)]
             + ["u_eno_out", "u_rdyo_out"],
    "EAST":  ["u_vdd3", "u_vss3"]
             + [f"u_dma_o_out{i}" for i in range(32, 64)]
             + ["u_last_out"],
}


def pad_in(name: str, pad_sig: str, core_sig: str) -> str:
    return (
        f"    sg13g2_IOPadIn {name} (\n"
        f"        .pad({pad_sig}), .p2c({core_sig}),\n"
        f"        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)\n"
        f"    );\n"
    )


def pad_out(name: str, pad_sig: str, core_sig: str) -> str:
    return (
        f"    sg13g2_IOPadOut30mA {name} (\n"
        f"        .c2p({core_sig}), .pad({pad_sig}),\n"
        f"        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)\n"
        f"    );\n"
    )


def pad_vdd(name: str) -> str:
    return (
        f"    sg13g2_IOPadVdd {name} (\n"
        f"        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)\n"
        f"    );\n"
    )


def pad_vss(name: str) -> str:
    return (
        f"    sg13g2_IOPadVss {name} (\n"
        f"        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)\n"
        f"    );\n"
    )


def main() -> None:
    lines = [HEADER, """
module OpenEye_Asic (
    input  wire        clk_pad,
    input  wire        rst_n_pad,
    input  wire [63:0] data_dma_i_pad,
    input  wire        enable_dma_i_pad,
    input  wire        ready_dma_i_pad,
    output wire        ready_dma_o_pad,
    output wire [63:0] data_dma_o_pad,
    output wire        enable_dma_o_pad,
    output wire        last_data_o_pad
);

    // ------------------------------------------------------------------
    // core interface wires
    // ------------------------------------------------------------------
    wire        clk_i;
    wire        rst_ni;
    wire [63:0] data_dma_i;
    wire        enable_dma_i;
    wire        ready_dma_i;
    wire        ready_dma_o;
    wire [63:0] data_dma_o;
    wire        enable_dma_o;
    wire        last_data_o;

    // single-rail power (see header)
    wire VPWR;
    wire VGND;

    // ------------------------------------------------------------------
    // compute core
    // ------------------------------------------------------------------
    OpenEye_FPGA u_openeye (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .ready_dma_o  (ready_dma_o),
        .data_dma_i   (data_dma_i),
        .enable_dma_i (enable_dma_i),
        .ready_dma_i  (ready_dma_i),
        .data_dma_o   (data_dma_o),
        .enable_dma_o (enable_dma_o),
        .last_data_o  (last_data_o)
    );

    // ------------------------------------------------------------------
    // power pads (4 x Vdd, 4 x Vss, one pair per side)
    // ------------------------------------------------------------------
"""]
    for i in range(4):
        lines.append(pad_vdd(f"u_vdd{i}"))
        lines.append(pad_vss(f"u_vss{i}"))

    lines.append("""
    // ------------------------------------------------------------------
    // input pads
    // ------------------------------------------------------------------
""")
    lines.append(pad_in("u_clk_in", "clk_pad", "clk_i"))
    lines.append(pad_in("u_rstn_in", "rst_n_pad", "rst_ni"))
    lines.append(pad_in("u_eni_in", "enable_dma_i_pad", "enable_dma_i"))
    lines.append(pad_in("u_rdyi_in", "ready_dma_i_pad", "ready_dma_i"))
    for i in range(64):
        lines.append(pad_in(f"u_dma_i_in{i}", f"data_dma_i_pad[{i}]", f"data_dma_i[{i}]"))

    lines.append("""
    // ------------------------------------------------------------------
    // output pads
    // ------------------------------------------------------------------
""")
    lines.append(pad_out("u_rdyo_out", "ready_dma_o_pad", "ready_dma_o"))
    lines.append(pad_out("u_eno_out", "enable_dma_o_pad", "enable_dma_o"))
    lines.append(pad_out("u_last_out", "last_data_o_pad", "last_data_o"))
    for i in range(64):
        lines.append(pad_out(f"u_dma_o_out{i}", f"data_dma_o_pad[{i}]", f"data_dma_o[{i}]"))

    lines.append("endmodule\n")

    OUT_V.parent.mkdir(parents=True, exist_ok=True)
    OUT_V.write_text("".join(lines))

    n = sum(len(v) for v in PAD_ORDER.values())
    names = {nm for v in PAD_ORDER.values() for nm in v}
    assert len(names) == n, "duplicate pad names"
    OUT_JSON.write_text(json.dumps(
        {k: PAD_ORDER[k] for k in ("SOUTH", "EAST", "NORTH", "WEST")},
        indent=2) + "\n")
    print(f"[OK] wrote {OUT_V} ({n} pads)")
    print(f"[OK] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
