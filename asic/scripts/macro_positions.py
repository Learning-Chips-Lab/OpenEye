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
    if "BUFFER_A" in inst:
        return "BUFFER_A"
    return "IACT"


def unescape(inst: str) -> str:
    """DEF COMPONENT names are ODB-escaped (\\[ \\] etc.); the placement
    JSON uses the unescaped names. Only brackets are special in our names."""
    return inst.replace("\\", "")


def check_against(rows, json_path: str) -> list:
    """Compare DEF macros against macro_placement.py's JSON (location in
    um, orientation). Returns a list of mismatch strings (empty = all OK)."""
    import json
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from macro_placement import _row_sequences
    name2cell = {name: cell for seq in _row_sequences() for cell, name in seq}
    with open(json_path) as f:
        exp = json.load(f)
    mism = []
    seen = set()
    for inst, cell, x, y, o in rows:
        e = exp.get(inst)
        if e is None:
            mism.append(f"{inst}: in DEF but NOT in placement JSON")
            continue
        seen.add(inst)
        ex, ey = (v / 1000 for v in e["location"])  # um -> mm
        if abs(x - ex) > 0.0005 or abs(y - ey) > 0.0005:
            mism.append(f"{inst}: pos ({x:.3f},{y:.3f}) != ({ex:.3f},{ey:.3f}) mm")
        if o != e["orientation"]:
            mism.append(f"{inst}: orientation {o} != {e['orientation']}")
        want = name2cell.get(inst)
        if want and cell != want:
            mism.append(f"{inst}: cell {cell} != {want}")
    for inst in sorted(set(exp) - seen):
        mism.append(f"{inst}: in placement JSON but NOT in DEF")
    return mism


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("def_file")
    ap.add_argument("--detail", action="store_true",
                    help="print full instance paths, not just group")
    ap.add_argument("--check", metavar="PLACEMENT_JSON",
                    help="compare against macro_placement.py's JSON "
                         "(location um + orientation per instance); "
                         "exit 1 on any mismatch")
    args = ap.parse_args()

    rows = []
    with open(args.def_file, "rb") as f:
        for line in f:
            if b"RM_IHPSG13" not in line or b"FIXED" not in line:
                continue
            m = PAT.match(line.decode())
            if m:
                inst, cell, x, y, o = m.groups()
                rows.append((unescape(inst), cell, int(x) / 1e6, int(y) / 1e6, o))

    if not rows:
        print(f"no RM_IHPSG13 macros found in {args.def_file}", file=sys.stderr)
        return 1

    if args.check:
        import json
        with open(args.check) as f:
            exp = json.load(f)
        mism = check_against(rows, args.check)
        n_mx = sum(1 for r in rows if r[4] == "MX")
        n_ro = sum(1 for r in rows if r[4] == "R0")
        others = sorted({r[4] for r in rows} - {"MX", "R0"})
        print(f"check: {len(rows)} DEF macros vs {len(exp)} placed; "
              f"orientations: MX={n_mx} R0={n_ro}"
              + (f" other={others}" if others else ""))
        if mism:
            print(f"MISMATCHES ({len(mism)}):")
            for m in mism:
                print("  ", m)
            return 1
        print("check: ALL MATCH")
        return 0

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
