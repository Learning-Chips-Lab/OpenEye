#!/usr/bin/env python3
"""One-shot RTL transform for OpenEye_Parallel.v: replace hierarchical
references into the gen_x[cc].gen_y[cr] generate scopes with plain
top-level 2D wire arrays (Yosys cannot resolve hierarchical paths into
generate scopes; iverilog/Verilator can).

Run:  python3 asic/scripts/hoist_parallel_wires.py
"""
import re
from pathlib import Path

F = Path(__file__).resolve().parent.parent.parent / "hdl" / "OpenEye_Parallel.v"

HIER = re.compile(r"gen_x\[([^\]]+)\]\.gen_y\[([^\]]+)\]\.([a-z_0-9]+)")


def main() -> None:
    lines = F.read_text().splitlines(keepends=True)
    text = "".join(lines)

    # 1) collect all signals referenced hierarchically (file-wide)
    refs = HIER.findall(text)
    signals = sorted(set(s for _, _, s in refs))
    print(f"hierarchical signals: {len(signals)}")

    # 2) locate the gen_y scope (balanced begin/end)
    start = next(i for i, ln in enumerate(lines) if "begin : gen_y" in ln)
    depth = 0
    end = start
    for i in range(start, len(lines)):
        depth += len(re.findall(r"\bbegin\b", lines[i]))
        depth -= len(re.findall(r"\bend\b", lines[i]))
        if depth == 0 and i > start:
            end = i
            break
    scope_text = "".join(lines[start:end + 1])
    print(f"gen_y scope: lines {start+1}..{end+1}")

    # 3) process the scope: drop the hoisted declarations, then map every
    #    bare reference to the 2D array element
    decls = []
    scope_proc = scope_text
    for s in signals:
        m = re.search(
            r"^[ \t]*wire[ \t]*(\[[^\]]+\])?[ \t]*" + re.escape(s) + r"[ \t]*;",
            scope_proc, re.M)
        assert m, f"no wire declaration for {s} in gen_y scope"
        width = (m.group(1) or "").strip().lstrip("[").rstrip("]")
        scope_proc = scope_proc.replace(m.group(0), "", 1)
        decls.append(f"    wire {'[' + width + '] ' if width else ''}{s} "
                     f"[0:CLUSTER_COLUMNS-1][0:CLUSTER_ROWS-1];")
    for s in signals:
        scope_proc = re.sub(r"(?<![a-z_0-9.])" + re.escape(s) + r"(?![a-z_0-9])",
                            f"{s}[clusters_x][clusters_y]", scope_proc)

    # 4) splice the processed scope back, then rewrite the remaining
    #    hierarchical references (the cross-wiring block) file-wide
    assert text.count(scope_text) == 1
    text = text.replace(scope_text, scope_proc, 1)
    text = HIER.sub(lambda m: f"{m.group(3)}[{m.group(1)}][{m.group(2)}]", text)

    # 5) insert the module-scope declarations before the grid generate
    anchor = "  genvar clusters_y;\n"
    assert anchor in text
    block = (
        "\n  // Per-cluster interconnect wires, hoisted to module scope so the\n"
        "  // cross-cluster wiring can reference them without hierarchical\n"
        "  // paths into the gen_x/gen_y generate scopes (which Yosys cannot\n"
        "  // resolve; iverilog/Verilator can).\n"
        + "\n".join(decls) + "\n"
    )
    text = text.replace(anchor, block + anchor, 1)

    F.write_text(text)
    print(f"[OK] hoisted {len(signals)} wires to module scope")


if __name__ == "__main__":
    main()
