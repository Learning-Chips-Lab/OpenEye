#!/usr/bin/env python3
"""Rename shared loop counters `a`/`b` per always block in OpenEye_FPGA.v.

Background (see asic/sim/gatesim/forloop_tests/README.md): yosys `proc`
unrolls the constant-bounded for loops but leaves the module-scope counter
integers as dead FFs. When 4 always blocks drive the same `a`/`b`,
`write_verilog` emits several always blocks for one FF -> 64 "multiple
conflicting drivers" check errors -> LibreLane step 8 aborts. Giving each
always block its own counter names removes the conflict (same pattern as
commit c6125c5). The netlist is unchanged: the counters are dead either way
and are removed by the flow's opt passes.

Usage (from the repo root):
  python3 asic/scripts/rename_loops_fpga.py            # dry run: report + planned replacements
  python3 asic/scripts/rename_loops_fpga.py --apply    # write hdl/OpenEye_FPGA.v
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(REPO, "hdl", "OpenEye_FPGA.v")

def main():
    apply = "--apply" in sys.argv
    with open(PATH) as f:
        lines = f.readlines()

    # --- find always-block spans by begin/end depth -------------------------
    spans = []  # (start_line_1based, end_line_1based)
    i = 0
    while i < len(lines):
        m = re.match(r"^\s*always @\(", lines[i])
        if not m:
            i += 1
            continue
        depth = 0
        started = False
        j = i
        while j < len(lines):
            # strip comments
            code = re.sub(r"//.*", "", lines[j])
            for tok in re.finditer(r"\bbegin\b|\bend\b", code):
                if tok.group() == "begin":
                    depth += 1
                    started = True
                else:
                    depth -= 1
            if started and depth == 0:
                break
            j += 1
        spans.append((i + 1, j + 1))
        i = j + 1

    # --- which blocks use `a` / `b` as whole words --------------------------
    use = {s: set() for s in range(len(spans))}
    for s, (lo, hi) in enumerate(spans):
        for ln in lines[lo - 1:hi]:
            code = re.sub(r"//.*", "", ln)
            if re.search(r"\ba\b", code):
                use[s].add("a")
            if re.search(r"\bb\b", code):
                use[s].add("b")

    blocks = [s for s in use if use[s]]
    print(f"always blocks: {len(spans)} total")
    for s, (lo, hi) in enumerate(spans):
        mark = ""
        if s in use:
            mark = f"  uses {sorted(use[s])}"
        print(f"  block {s+1}: lines {lo}-{hi}{mark}")
    assert len(blocks) == 4, f"expected exactly 4 blocks using a/b, got {len(blocks)}"

    # --- rename per block: a->aN, b->bN (N = 1..4 in order) -----------------
    out = lines[:]
    counts = {}
    for n, s in enumerate(blocks, start=1):
        lo, hi = spans[s]
        for idx in range(lo - 1, hi):
            # keep comments untouched: split at the first '//' (string
            # slicing preserves the trailing newline; a regex with $ would
            # drop it and merge the line with the next one)
            sl = lines[idx].find("//")
            if sl >= 0:
                code, comment = lines[idx][:sl], lines[idx][sl:]
            else:
                code, comment = lines[idx], ""
            new = re.sub(r"\ba\b", f"a{n}", code)
            new = re.sub(r"\bb\b", f"b{n}", new)
            if new != code:
                counts[(n, lo, idx + 1)] = (code.strip(), new.strip())
            out[idx] = new + comment
    assert len(out) == len(lines), "line count changed!"

    # --- update the declaration ---------------------------------------------
    for idx, ln in enumerate(out):
        if re.match(r"^\s*integer\s+a\s*,\s*b\s*,\s*word\s*,\s*line\s*;\s*$", ln):
            indent = re.match(r"^\s*", ln).group()
            out[idx] = (f"{indent}integer a1, a2, a3, a4, b1, b2, b3, b4, "
                        f"word, line;\n")
            print(f"declaration at line {idx+1} -> {out[idx].strip()}")
            break
    else:
        sys.exit("declaration line `integer a, b, word, line;` not found")

    if apply:
        with open(PATH, "w") as f:
            f.writelines(out)
        print(f"APPLIED: {len(counts)} lines changed")
    else:
        print(f"dry run: {len(counts)} lines would change, e.g.:")
        for k in list(counts)[:6]:
            print(f"  L{k[2]}: {counts[k][0][:60]}  ->  {counts[k][1][:60]}")

if __name__ == "__main__":
    main()
