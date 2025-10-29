#!/usr/bin/env python3
import json, sys
from collections import Counter

if len(sys.argv) < 3:
    print("Usage: hier_stat.py <design.json> <top_module>")
    sys.exit(1)

json_path, top = sys.argv[1], sys.argv[2]

with open(json_path) as f:
    design = json.load(f)
modules = design["modules"]

def count_cells(mod):
    return Counter(cell["type"] for cell in mod.get("cells", {}).values())

def walk(name, indent=0):
    if name not in modules:
        return
    mod = modules[name]
    counts = count_cells(mod)
    total = sum(counts.values())
    print("  " * indent + f"{name}: {total} cells")
    for t, c in sorted(counts.items()):
        print("  " * (indent + 1) + f"{t}: {c}")
    for cell in mod.get("cells", {}).values():
        celltype = cell["type"]
        if celltype in modules:
            walk(celltype, indent + 1)

walk(top)
