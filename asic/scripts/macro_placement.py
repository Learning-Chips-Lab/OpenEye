#!/usr/bin/env python3
"""Deterministic SRAM macro placement for the OpenEye ASIC target.

LibreLane 3.1's OpenROAD Chip flow does NOT place macros on its own:
`Odb.ManualMacroPlacement` only acts on `MACROS.<cell>.instances` from the
config (p2_11 failed at `OpenROAD.GeneratePDN` with
`[PDN-0235] Design has unplaced macros` because the config had no
instances). This module computes fixed positions for all 41 RM instances
of the 2x2 target and is consumed by gen_config.py.

Floorplan v2 (2026-11-01, after the p2_15 post-GRT timing collapse)
-------------------------------------------------------------------
Every RM_IHPSG13_1P macro has ALL signal pins (incl. A_CLK) on Metal2 at
its BOTTOM edge (LEF y=0) and blocks Metal1..Metal4 over its whole area.
v1 stacked 33 macros R0 in three rows at the core bottom, so the pin edges
faced 3.7 / 18.3 / 21.7 um wide channels (core edge / the row below);
~5,400 pins had to escape sideways through 15 um gaps. GRT overflowed
locally (Metal4 124k, Metal2 20k at 17 % / 51 % usage) and detoured the
macro nets by millimetres: post-GRT setup WNS -193.9 ns -> -73.8 ns,
hold -30.9 ns (late A_CLK), all worst endpoints on macro A_DIN/A_ADDR.

v2 rules (asserted below):
  1. Only two macro rows, along the BOTTOM and the TOP core edge, one macro
     deep. The pin edge of every macro faces the core interior:
       bottom row: orientation MX (mirrored, pins on top),
       top row:    orientation R0 (pins on the bottom).
     The pin-less back side faces the core edge (10 um margin).
  2. Nothing is placed in front of a pin edge: the whole core interior
     (>= 8 mm) is the pin-access channel.
  3. >= 30 um between neighbouring macros (PDN straps, halo).
  4. Locality: each cluster's macros (2 psum GLBs + its iact_stream_
     constructor RAM) sit together; the shared buffers sit in the middle
     of their row (wght_buffer bottom, BUFFER_A top). With fixed macros
     as anchors, global placement pulls each cluster's logic next to its
     macro group:
       bottom: [psum X0Y0 g0 g1][iact X0Y0][wght x5][iact X1Y0][psum X1Y0 g0 g1]
       top:    [psum X0Y1 g0 g1][iact X0Y1][BUFFER_A x4][iact X1Y1][psum X1Y1 g0 g1]

Positions are the lower-left corner of the placed (oriented) bounding
box, as dbInst.setLocation() expects. Instance names are ODB (unescaped)
names; placers.py hard-fails on any name mismatch, so a stale list cannot
silently misplace. The 8 iact_stream_constructor instances
(g_cfg1024x24) are new in v2 -- verify them against the synthesized
netlist of the first v2 run (QWEN.md TODO plan).
"""

# (width, height) in um, from the PDK LEF SIZE entries
DIMS = {
    "RM_IHPSG13_1P_1024x64_c2_bm_bist": (784.48, 336.46),
    "RM_IHPSG13_1P_1024x16_c2_bm_bist": (236.8, 336.46),
    "RM_IHPSG13_1P_1024x8_c2_bm_bist": (118.4, 336.46),
    "RM_IHPSG13_1P_2048x64_c2_bm_bist": (784.48, 626.7),
    "RM_IHPSG13_1P_2048x32_c2_bm_bist": (416.64, 626.7),
    "RM_IHPSG13_1P_4096x16_c3_bm_bist": (416.64, 618.3),
    "RM_IHPSG13_1P_4096x8_c3_bm_bist": (236.8, 618.3),
}

CORE = (320.16, 321.3, 10179.84, 10175.76)  # x0 y0 x1 y1 (p2_11 metrics)

EDGE_MARGIN = 10.0  # back side of the macro to the core edge
GAP = 30.0          # between neighbouring macros in a row (minimum)

W = "u_openeye.wght_buffer.g_specific.impl.g_cfg2048x288"
P = ("u_openeye.PSUM_RAM_X[{x}].PSUM_RAM_Y[{y}].PSUM_RAM_GLB[{g}]"
     ".psum_buffer.g_specific.impl.g_cfg4096x40")
I = ("u_openeye.IACT_CONVERTER_X[{x}].IACT_CONVERTER_Y[{y}]"
     ".iact_stream_constructor.BUFFER[0].iact_buffer_SP.g_specific.impl"
     ".g_cfg1024x24")
B = ("u_openeye.BUFFER_A[{i}].iact_layer_buffer.g_specific.impl."
     "g_cfg1024x64.u_mac")

C2048X64 = "RM_IHPSG13_1P_2048x64_c2_bm_bist"
C2048X32 = "RM_IHPSG13_1P_2048x32_c2_bm_bist"
C4096X16 = "RM_IHPSG13_1P_4096x16_c3_bm_bist"
C4096X8 = "RM_IHPSG13_1P_4096x8_c3_bm_bist"
C1024X64 = "RM_IHPSG13_1P_1024x64_c2_bm_bist"
C1024X16 = "RM_IHPSG13_1P_1024x16_c2_bm_bist"
C1024X8 = "RM_IHPSG13_1P_1024x8_c2_bm_bist"


def _psum(x, y):
    """Both psum GLBs of cluster (x, y), each GLB's three macros adjacent."""
    seq = []
    for g in (0, 1):
        base = P.format(x=x, y=y, g=g)
        seq += [(C4096X16, base + ".u_mac0"), (C4096X16, base + ".u_mac1"),
                (C4096X8, base + ".u_mac2")]
    return seq


def _iact(x, y):
    base = I.format(x=x, y=y)
    return [(C1024X16, base + ".u_mac0"), (C1024X8, base + ".u_mac1")]


def _row_sequences():
    """Ordered (cell, instance) pairs: (bottom row, top row), left to right."""
    wght = [(C2048X64, f"{W}.u_mac{i}") for i in range(4)]
    wght.append((C2048X32, f"{W}.u_mac4"))
    buf_a = [(C1024X64, B.format(i=i)) for i in range(4)]
    bottom = _psum(0, 0) + _iact(0, 0) + wght + _iact(1, 0) + _psum(1, 0)
    top = _psum(0, 1) + _iact(0, 1) + buf_a + _iact(1, 1) + _psum(1, 1)
    return bottom, top


def _floor5(v: float) -> float:
    return (v // 5) * 5


def _ceil5(v: float) -> float:
    return -((-v) // 5) * 5


def compute_placement() -> dict:
    """Return {instance_name: {"location": [x, y], "orientation": ...}}."""
    bottom, top = _row_sequences()
    assert len(bottom) == 21 and len(top) == 20
    core_x0, core_y0, core_x1, core_y1 = CORE
    core_w = core_x1 - core_x0

    placed = []  # (x, y, w, h, orient, name)
    for seq, side in ((bottom, "bottom"), (top, "top")):
        widths = [DIMS[c][0] for c, _ in seq]
        # spread the slack evenly: GAP is the minimum, the row is centered
        # (5 um per macro reserved for the _ceil5 snapping drift)
        slack = core_w - 2 * EDGE_MARGIN - sum(widths) - 5 * len(seq)
        gap = slack / (len(seq) - 1)
        assert gap >= GAP, f"{side} row too wide: gap {gap:.1f} < {GAP}"
        cursor = core_x0 + EDGE_MARGIN
        for cell, name in seq:
            w, h = DIMS[cell]
            px = _ceil5(cursor)
            if side == "bottom":
                py, orient = _ceil5(core_y0 + EDGE_MARGIN), "MX"
            else:
                py, orient = _floor5(core_y1 - EDGE_MARGIN - h), "R0"
            placed.append((px, py, w, h, orient, name))
            cursor += w + gap

    # legality: inside the core, no overlap, >= GAP between neighbours
    for px, py, w, h, _o, name in placed:
        assert px >= core_x0 + EDGE_MARGIN - 1e-6, name
        assert px + w <= core_x1 - EDGE_MARGIN + 1e-6, f"{name}: right"
        assert py >= core_y0 + EDGE_MARGIN - 1e-6, f"{name}: bottom"
        assert py + h <= core_y1 - EDGE_MARGIN + 1e-6, f"{name}: top"
    for i, (ax, ay, aw, ah, _ao, an) in enumerate(placed):
        for bx, by, bw, bh, _bo, bn in placed[i + 1:]:
            if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
                raise AssertionError(f"overlap: {an} / {bn}")
            if ay == by:  # same row
                gap_x = max(bx - (ax + aw), ax - (bx + bw))
                assert gap_x >= GAP - 5, f"gap {gap_x:.1f} < {GAP}: {an} / {bn}"

    # pin edges face the interior with an open channel: the bottom row's
    # top edge (pins) and the top row's bottom edge (pins) are > 8 mm apart
    bottom_pins = max(py + h for px, py, w, h, o, _n in placed if o == "MX")
    top_pins = min(py for px, py, w, h, o, _n in placed if o == "R0")
    assert top_pins - bottom_pins > 8000, (bottom_pins, top_pins)

    out = {}
    for px, py, _w, _h, orient, name in placed:
        assert name not in out, f"duplicate instance {name}"
        out[name] = {"location": [px, py], "orientation": orient}
    assert len(out) == 41, f"expected 41 instances, got {len(out)}"
    return out


def merge_into_macros(macros: dict) -> dict:
    """Add the computed instances to the MACROS config dict (in place)."""
    bottom, top = _row_sequences()
    name2cell = {name: cell for seq in (bottom, top) for cell, name in seq}
    for name, loc in compute_placement().items():
        macros[name2cell[name]].setdefault("instances", {})[name] = loc
    return macros


if __name__ == "__main__":
    import json
    p = compute_placement()
    print(json.dumps(p, indent=1))
    print(f"#{len(p)} instances; bottom row MX, top row R0")
