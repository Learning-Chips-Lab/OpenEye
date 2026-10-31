#!/usr/bin/env python3
"""List the placed SRAM macros (RM_IHPSG13_*) of a LibreLane run's DEF.

Read-only. Used for Phase D: compare p2_16's macro placement against
the p2_15 baseline (three bottom rows, PSUM row in cluster-grid order).

Usage:
  python3 macro_positions.py <path-to-OpenEye_Asic.def> [--detail]

Output: one line per macro (group, cell, x, y in mm, orientation),
sorted by group then x; then a per-group/per-cell summary and the
distinct x/y coordinate sets. Coordinates come from the DEF COMPONENT
entries:  "- <inst> <cell> + FIXED ( <x> <y> ) <orient> ;" with
UNITS DISTANCE MICRONS 1000 (i.e. values are in nm).
"""
import argparse
import re
import sys
from collections import Counter

PAT = re.compile(
    r"^\s*-\s+(\S+)\s+(RM_IHPSG13\w+)\s+\+\s+FIXED\s+\(\s*(\d+)\s+(\d+)\s*\)\s+(\w+)\s*;"
)


def group_of(inst: str) -> str:
    if "PSUM_RAM" in inst:
        return "PSUM"
    if "wght_buffer" in inst:
        return "WGHT"
    return "IACT"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("def_file")
    ap.add_argument("--detail", action="store_true",
                    help="print full instance paths, not just group")
    args = ap.parse_args()

    rows = []
    with open(args.def_file, "rb") as f:
        for line in f:
            if b"RM_IHPSG13" not in line or b"FIXED" not in line:
                continue
            m = PAT.match(line.decode())
            if m:
                inst, cell, x, y, o = m.groups()
                rows.append((inst, cell, int(x) / 1e6, int(y) / 1e6, o))

    if not rows:
        print(f"no RM_IHPSG13 macros found in {args.def_file}", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (group_of(r[0]), r[2], r[3]))
    for inst, cell, x, y, o in rows:
        key = inst if args.detail else group_of(inst)
        print(f"{key:5s} {cell:32s} x={x:8.3f}mm y={y:8.3f}mm {o}")

    print()
    print("per group:", dict(Counter(group_of(r[0]) for r in rows)))
    print("per cell: ", dict(Counter(r[1] for r in rows)))
    print("distinct x (mm):", sorted({round(r[2], 3) for r in rows}))
    print("distinct y (mm):", sorted({round(r[3], 3) for r in rows}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
