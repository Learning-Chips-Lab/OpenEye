#!/usr/bin/env python3
import sys, json

def print_module_tree(modules, name, indent=0, visited=None):
    if visited is None: visited = set()
    if name in visited: return
    visited.add(name)
    mod = modules.get(name)
    if not mod: return
    print("  " * indent + f"{name}")
    for cell_name, cell in mod.get("cells", {}).items():
        ctype = cell["type"]
        if ctype in modules:
            print_module_tree(modules, ctype, indent + 1, visited)

def main():
    if len(sys.argv) < 2:
        print("Usage: format_stat.py <hierarchy.json>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        data = json.load(f)
    modules = data["modules"]
    top = data.get("top") or list(modules.keys())[0]
    print_module_tree(modules, top)

if __name__ == "__main__":
    main()
