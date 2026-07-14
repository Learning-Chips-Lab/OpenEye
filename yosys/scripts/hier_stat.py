#!/usr/bin/env python3
import json
import sys
from collections import defaultdict

if len(sys.argv) < 3:
    print("Usage: hier_stat.py <OpenEye_FPGA.json> <top_module>")
    sys.exit(1)

json_path, top_module_name = sys.argv[1], sys.argv[2]

with open(json_path) as f:
    design = json.load(f)

modules = design.get("modules", {})

# Da das Design geflattet ist, nehmen wir das Top-Modul direkt
if top_module_name not in modules:
    # Falls der Name in Yosys mit Backslash deklariert ist
    top_module_name = "\\" + top_module_name
    if top_module_name not in modules:
        print(sys.argv[2], "not found in JSON.")
        sys.exit(1)

top_mod = modules[top_module_name]

# Hierarchische Töpfe für die Zellzählung
# Struktur: hierarchy_stats["Modulpfad"]["Zelltyp"] = Anzahl
hierarchy_stats = defaultdict(lambda: defaultdict(int))

for cell_name, cell_data in top_mod.get("cells", {}).items():
    cell_type = cell_data["type"]
    
    # Yosys behält bei geflatteten Zellen den alten Pfad im Namen (getrennt durch Punkte oder Backslashes)
    # Beispiel: "OpenEye_Parallel.gen_x[0].pe_cluster.pe_0.gatemate_lut_xyz"
    name_parts = cell_name.lstrip('\\').split('.')
    
    if len(name_parts) > 1:
        # Rekonstruiere den Modulpfad ohne den konkreten Hardware-Zellnamen am Ende
        module_path = " / ".join(name_parts[:-1])
    else:
        module_path = "Top Level Ports / Logic"
        
    hierarchy_stats[module_path][cell_type] += 1

# Ausgabe sortiert nach Modulpfaden
print(f"=== Hierarchical Resource Report for {sys.argv[2]} (Flattened) ===")
for mod_path, cells in sorted(hierarchy_stats.items()):
    total_cells = sum(cells.values())
    print(f"\nModule: {mod_path} ({total_cells} total primitives)")
    for c_type, c_count in sorted(cells.items()):
        print(f"  {c_type.lstrip('$')}: {c_count}")