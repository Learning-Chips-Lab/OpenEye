#!/usr/bin/env python3
"""Deterministic SRAM macro placement for the OpenEye ASIC target.

LibreLane 3.1's OpenROAD Chip flow does NOT place macros on its own:
`Odb.ManualMacroPlacement` only acts on `MACROS.<cell>.instances` from the
config (p2_11 failed at `OpenROAD.GeneratePDN` with
`[PDN-0235] Design has unplaced macros` because the config had no
instances). This module computes fixed R0 positions for all 33 RM
instances of the 2x2 target and is consumed by gen_config.py.

Layout: three rows along the bottom of the core (core bbox from the p2_11
run: 320.16 321.3 -> 10179.84 10175.76 um), tallest macros first:

  row 1 (y=325):  wght_buffer group   4 x 2048x64 + 1 x 2048x32  (h 626.7)
  row 2 (y=970):  psum buffers        16 x 4096x16 + 8 x 4096x8  (h 618.3)
  row 3 (y=1610): BUFFER_A group      4 x 1024x64                (h 336.46)

Spacing: 15 um between macros in a row, 20 um between rows; all positions
are snapped to the 5 um grid that placers.py's gridify() would apply
anyway, and the assertions below verify the layout stays legal after that
snapping (>= 10 um gap, inside the core).

Instance names are the exact ODB (unescaped) names of the p2_11 run,
extracted from the padring DEF; placers.py hard-fails on any name mismatch,
so a stale list cannot silently misplace.
"""

# (width, height) in um, from the PDK LEF SIZE entries
DIMS = {
    "RM_IHPSG13_1P_1024x64_c2_bm_bist": (784.48, 336.46),
    "RM_IHPSG13_1P_1024x8_c2_bm_bist": (118.4, 336.46),
    "RM_IHPSG13_1P_2048x64_c2_bm_bist": (784.48, 626.7),
    "RM_IHPSG13_1P_2048x32_c2_bm_bist": (416.64, 626.7),
    "RM_IHPSG13_1P_4096x16_c3_bm_bist": (416.64, 618.3),
    "RM_IHPSG13_1P_4096x8_c3_bm_bist": (236.8, 618.3),
}

CORE = (320.16, 321.3, 10179.84, 10175.76)  # x0 y0 x1 y1 (p2_11 metrics)

W = "u_openeye.wght_buffer.g_specific.impl.g_cfg2048x288"
P = ("u_openeye.PSUM_RAM_X[{x}].PSUM_RAM_Y[{y}].PSUM_RAM_GLB[{g}]"
     ".psum_buffer.g_specific.impl.g_cfg4096x40")

C2048X64 = "RM_IHPSG13_1P_2048x64_c2_bm_bist"
C2048X32 = "RM_IHPSG13_1P_2048x32_c2_bm_bist"
C4096X16 = "RM_IHPSG13_1P_4096x16_c3_bm_bist"
C4096X8 = "RM_IHPSG13_1P_4096x8_c3_bm_bist"
C1024X64 = "RM_IHPSG13_1P_1024x64_c2_bm_bist"


def _row_sequences():
    """Ordered (cell, instance) pairs, row by row."""
    row1 = [(C2048X64, f"{W}.u_mac{i}") for i in range(4)]
    row1.append((C2048X32, f"{W}.u_mac4"))
    # keep each psum GLB's three macros adjacent (16, 16, 8-bit)
    row2 = []
    for x in (0, 1):
        for y in (0, 1):
            for g in (0, 1):
                row2.append((C4096X16, P.format(x=x, y=y, g=g) + ".u_mac0"))
                row2.append((C4096X16, P.format(x=x, y=y, g=g) + ".u_mac1"))
                row2.append((C4096X8, P.format(x=x, y=y, g=g) + ".u_mac2"))
    row3 = [(C1024X64,
             f"u_openeye.BUFFER_A[{i}].iact_layer_buffer.g_specific.impl."
             f"g_cfg1024x64.u_mac") for i in range(4)]
    return row1, row2, row3


def _round5(v: float) -> float:
    """placers.py gridify: round to the 5 um grid."""
    return round(v / 5) * 5


def compute_placement() -> dict:
    """Return {instance_name: {"location": [x, y], "orientation": "R0"}}."""
    row1, row2, row3 = _row_sequences()
    assert len(row1) == 5 and len(row2) == 24 and len(row3) == 4

    x0 = 325  # first column (>= core x0, on the 5 um grid)
    y1 = 325  # bottom of row 1 (>= core y0, on the 5 um grid)
    y2 = _round5(y1 + DIMS[C2048X64][1] + 20)
    y3 = _round5(y2 + DIMS[C4096X16][1] + 20)
    gap = 15.0
    core_x0, core_y0, core_x1, core_y1 = CORE

    placed = []  # (x, y, w, h, name)
    for y, seq in ((y1, row1), (y2, row2), (y3, row3)):
        cursor = x0
        for cell, name in seq:
            w, h = DIMS[cell]
            px = _round5(cursor)
            assert px >= core_x0 + 1, name
            assert px + w <= core_x1 - 1, f"{name}: overflows core right"
            assert y >= core_y0 + 1 and y + h <= core_y1 - 1, \
                f"{name}: outside core"
            placed.append((px, y, w, h, name))
            cursor += w + gap

    # no overlap after gridify snapping; >= 10 um gap in a row
    for i, (ax, ay, aw, ah, an) in enumerate(placed):
        for bx, by, bw, bh, bn in placed[i + 1:]:
            overlap_x = ax < bx + bw and bx < ax + aw
            overlap_y = ay < by + bh and by < ay + ah
            if overlap_x and overlap_y:
                raise AssertionError(f"overlap: {an} / {bn}")
            if ay == by and overlap_x and abs(ax - bx) < max(aw, bw):
                gap_x = min(abs(ax - (bx + bw)), abs(bx - (ax + aw)))
                if ax != bx and gap_x < 10:
                    raise AssertionError(
                        f"row gap < 10um: {an} / {bn} ({gap_x})")

    out = {}
    for px, py, _w, _h, name in placed:
        assert name not in out, f"duplicate instance {name}"
        out[name] = {"location": [px, py], "orientation": "R0"}
    assert len(out) == 33, f"expected 33 instances, got {len(out)}"
    return out


def merge_into_macros(macros: dict) -> dict:
    """Add the computed instances to the MACROS config dict (in place)."""
    row1, row2, row3 = _row_sequences()
    name2cell = {name: cell for seq in (row1, row2, row3)
                 for cell, name in seq}
    for name, loc in compute_placement().items():
        macros[name2cell[name]].setdefault("instances", {})[name] = loc
    return macros


if __name__ == "__main__":
    import json
    p = compute_placement()
    print(json.dumps(p, indent=1))
    print(f"#{len(p)} instances; rows y=325/970/1610")
