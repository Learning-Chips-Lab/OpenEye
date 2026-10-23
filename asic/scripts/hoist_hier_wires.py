#!/usr/bin/env python3
"""Generic one-shot transform: hoist generate-scope-local wires that are
referenced hierarchically (LABEL_OUT[<expr>].LABEL_IN[<expr>].<sig>) into
module-scope 2D wire arrays, so Yosys can elaborate the design (it cannot
resolve hierarchical paths into generate scopes).

Usage:
  python3 hoist_hier_wires.py <file.v> <outer_label> <inner_label> <i_var> <j_var> <dim_i> <dim_j>
Example:
  python3 hoist_hier_wires.py hdl/PE_cluster.v gen_X gen_Y i j PE_COLUMNS PE_ROWS
"""
import re
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    outer, inner, iv, jv = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    dim_i, dim_j = sys.argv[6], sys.argv[7]
    text = path.read_text()

    hier = re.compile(
        outer + r"\[([^\]]+)\]\." + inner + r"\[([^\]]+)\]\.([a-z_0-9]+)")
    signals = sorted(set(m.group(3) for m in hier.finditer(text)))
    print(f"{path}: hierarchical signals: {len(signals)} {signals}")

    lines = text.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if f"begin : {inner}" in l)
    depth = 0
    end = start
    for i in range(start, len(lines)):
        depth += len(re.findall(r"\bbegin\b", lines[i]))
        depth -= len(re.findall(r"\bend\b", lines[i]))
        if depth == 0 and i > start:
            end = i
            break
    scope_text = "".join(lines[start:end + 1])
    print(f"  inner scope: lines {start+1}..{end+1}")

    decls = []
    scope_proc = scope_text
    for s in signals:
        m = re.search(
            r"^[ \t]*wire[ \t]*(\[[^\]]+\])?[ \t]*" + re.escape(s) + r"[ \t]*;",
            scope_proc, re.M)
        assert m, f"no wire declaration for {s} in {inner} scope"
        width = (m.group(1) or "").strip().lstrip("[").rstrip("]")
        scope_proc = scope_proc.replace(m.group(0), "", 1)
        decls.append(f"  wire {'['+width+'] ' if width else ''}{s} "
                     f"[0:{dim_i}-1][0:{dim_j}-1];")
    for s in signals:
        scope_proc = re.sub(r"(?<![a-z_0-9.])" + re.escape(s) + r"(?![a-z_0-9])",
                            f"{s}[{iv}][{jv}]", scope_proc)

    assert text.count(scope_text) == 1
    text = text.replace(scope_text, scope_proc, 1)
    text = hier.sub(lambda m: f"{m.group(3)}[{m.group(1)}][{m.group(2)}]", text)

    # insert the declarations right before the outer for-loop line
    block = ("  // Interconnect wires hoisted to module scope (Yosys cannot\n"
             "  // resolve hierarchical paths into generate scopes).\n"
             + "\n".join(decls) + "\n")
    idx = next(i for i, l in enumerate(lines) if f"begin : {outer}" in l)
    line_start = text.rfind("\n", 0, text.index(lines[idx])) + 1
    text = text[:line_start] + block + "\n" + text[line_start:]

    path.write_text(text)
    print(f"  [OK] hoisted {len(decls)} wires")


if __name__ == "__main__":
    main()
