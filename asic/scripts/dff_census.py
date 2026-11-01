#!/usr/bin/env python3
"""Census of sequential cells per hierarchy node in a yosys write_verilog dump.

Purpose
-------
Cross-check a synthesis run's sequential-cell inventory (DFF bits, latches,
ICGs, SRAM macros) against `stat.rpt` and against a reference run, BEFORE any
opt/techmap pass — i.e. on the pre-abc dump. This is how the p2_16
`sg13g2_dfrbpq_1` = 51,883 (vs p2_15 506,219) deviation was resolved on
2026-11-01: the whole −487,392 DFF delta is exactly the latch-RF structural
change and the "other" logic is bit-identical per module between the two
runs (see QWEN.md, plan v2, F2).

Why this parser
---------------
The yosys Verilog backend encodes $dff/$adff/$sdff cells (incl. those created
by MEMORY_MAP) as `always @(edge ...)` blocks assigning `reg [W-1:0]`
signals — NOT as `$_DFF*` cells:
  - single statement:  `always @(posedge clk)\n  q <= expr;`
  - begin/end block:   `always @(posedge clk)\n  begin\n    q <= expr;\n  end`
So: FF bits per module = sum of bit widths of regs assigned in clocked
always blocks. Module-type cells (oe_dlatch, oe_icg, RM_IHPSG13_*, sub-module
insts) are counted directly. The latch/ICG/RM totals come out EXACT vs
stat.rpt, which validates the FF-bit parsing as well.

How to produce the dump (same pipeline as LibreLane step 6, stopped before
opt/techmap/abc — read the run's own define set from its
06-yosys-synthesis/COMMANDS or yosys-synthesis.log):
  yosys -c <(
    read_verilog -sv -DUSE_INTERNAL_PARAMS_PE -DUSE_INTERNAL_PARAMS_PE_cluster \
      -DOPENEYE_RAM_SPECIFIC -DOPENEYE_LATCH_RF -DOPENEYE_SG13G2_CELLS -DNO_TRACE \
      <all RTL files, same list as the flow>
    hierarchy -top OpenEye_Asic
    proc
    memory
    memory_map
    write_verilog /path/pre_abc_dump.v
  )
(For a pre-latch-RF reference run: same script minus OPENEYE_LATCH_RF,
from that run's commit — 2026-11-01 dumps: asic/runs/diag_pre_abc_p2_16.v
and asic/runs/diag_pre_abc_p2_15.v, scripts diag_ys_p2_1{6,5}.ys.)

Usage
-----
  python3 asic/scripts/dff_census.py <pre_abc_dump.v> [top]

Prints per-hierarchy-node FF bits / latch / ICG / RM, per base module
(instance-weighted — note: "$paramod$" module rows are per parameter
variant, the base name is only the suffix) and the four TOTAL_* lines.
Compare TOTAL_* against the run's stat.rpt cell counts (latches, ICGs and
RM macros must match exactly; FF bits match the post-opt stat.rpt minus
opt-removals, pre-opt they are the ground truth).
"""
import re, sys, collections

def unescape(tok):
    if not tok.startswith("\\"):
        return tok
    body = tok[1:]
    if body.endswith("\\"):
        body = body[:-1]
    return body.replace("\\", "")

MOD_RE = re.compile(r"^\s*module\s+([^\s(]+)")
REG_RE = re.compile(r"^reg\s*(?:\[(\d+):(\d+)\])?\s*(\S+)\s*;")
ALWAYS_RE = re.compile(r"^always\s*@\(([^)]*)\)\s*(.*)$")
ASSIGN_RE = re.compile(r"([^\s;()]+)\s*<=")
CELL_RE = re.compile(r"^\s*(\S+)\s+(\S+)\s*(\(.*|\s*$)")

LATCH_CELLS = ("oe_dlatch",)
ICG_CELLS = ("oe_icg",)
RM_PREFIX = "RM_"

def base_name(modname):
    n = modname
    if n.startswith("$paramod$"):
        i = n.rfind("\\")
        return unescape(n[i+1:]) if i >= 0 else n
    if n.startswith("$paramod\\"):
        parts = [unescape(p) for p in n[len("$paramod"):].split("\\")]
        return parts[0] if parts else n
    return n

def strip_comment(line):
    if "/*" in line:
        before, rest = line.split("/*", 1)
        if "*/" not in rest:
            return ""
        line = before + rest.split("*/", 1)[1]
    return line

def parse(path):
    # stage 0: module names (handles forward references)
    names = []
    for line in open(path, errors="ignore"):
        m = MOD_RE.match(line)
        if m:
            names.append(unescape(m.group(1)))
    known = set(names)

    modules = {n: {"cells": collections.Counter(), "insts": [], "regs": {},
                   "ff_bits": 0, "ff_blocks": 0} for n in names}
    cur = None
    pending = None          # awaiting single-statement body of a clocked always
    in_block = 0
    block_depth = 0
    assigned = set()

    for line in open(path, errors="ignore"):
        line = strip_comment(line)
        t = line.strip()
        if not t or t.startswith("//"):
            continue
        m = MOD_RE.match(t)
        if m:
            cur = unescape(m.group(1))
            pending = None; in_block = 0
            continue
        if t.startswith("endmodule"):
            cur = None; pending = None; in_block = 0
            continue
        if cur is None:
            continue
        am = ALWAYS_RE.match(t)
        if am and re.search(r"(pos|neg)edge", am.group(1)):
            rest = am.group(2).strip()
            if rest.endswith(";"):
                targets = {unescape(x) for x in ASSIGN_RE.findall(rest)}
                if targets:
                    modules[cur]["ff_bits"] += sum(modules[cur]["regs"].get(r, 0) for r in targets)
                    modules[cur]["ff_blocks"] += 1
            elif rest.startswith("begin"):
                in_block = 1; block_depth = 1; assigned = set()
            else:
                pending = set()
            continue
        if pending is not None:
            if t.endswith(";"):
                targets = {unescape(x) for x in ASSIGN_RE.findall(t)}
                if targets:
                    modules[cur]["ff_bits"] += sum(modules[cur]["regs"].get(r, 0) for r in targets)
                    modules[cur]["ff_blocks"] += 1
                pending = None
            continue
        if in_block:
            if t.startswith("begin"):
                block_depth += 1
            elif t.startswith("end") and not t.startswith("endmodule"):
                block_depth -= 1
                if block_depth <= 0:
                    bits = sum(modules[cur]["regs"].get(r, 0) for r in assigned)
                    modules[cur]["ff_bits"] += bits
                    modules[cur]["ff_blocks"] += 1
                    in_block = 0
            else:
                for x in ASSIGN_RE.findall(t):
                    assigned.add(unescape(x))
            continue
        m = REG_RE.match(t)
        if m:
            w = (int(m.group(1)) - int(m.group(2)) + 1) if m.group(1) is not None else 1
            modules[cur]["regs"][unescape(m.group(3))] = w
            continue
        m = CELL_RE.match(t)
        if not m:
            continue
        ctype = unescape(m.group(1))
        if ctype.startswith((".", "always", "wire", "input", "output", "assign", "(*", "function", "module")):
            continue
        if ctype in LATCH_CELLS or ctype in ICG_CELLS or ctype.startswith(RM_PREFIX):
            modules[cur]["cells"][ctype] += 1
        elif ctype in known:
            modules[cur]["insts"].append(ctype)
        # else: leaf internal cell ($add, $mux, ...) -> ignore

    return modules

def main(path, top="OpenEye_Asic"):
    modules = parse(path)
    inst_count = collections.Counter()
    sys.setrecursionlimit(100000)
    def count_insts(mod):
        for ctype in modules[mod]["insts"]:
            inst_count[ctype] += 1
            count_insts(ctype)
    if top not in modules:
        sys.exit(f"top module {top} not found; all: {sorted(modules)}")
    count_insts(top)

    seq = collections.Counter(); lat = collections.Counter(); icg = collections.Counter(); mac = collections.Counter()
    def walk(mod, path):
        info = modules[mod]
        c = info["cells"]
        if info["ff_bits"]:
            seq[path or "<top>"] += info["ff_bits"]
        nl = sum(c.get(k, 0) for k in LATCH_CELLS)
        if nl: lat[path or "<top>"] += nl
        ni = sum(c.get(k, 0) for k in ICG_CELLS)
        if ni: icg[path or "<top>"] += ni
        nm = sum(v for k, v in c.items() if k.startswith(RM_PREFIX))
        if nm: mac[path or "<top>"] += nm
        for ctype in modules[mod]["insts"]:
            p = f"{path}.{base_name(ctype)}" if path else base_name(ctype)
            walk(ctype, p)
    walk(top, "")

    out = []
    out.append(f"== FF bits per node (top 60) from {path}")
    for k, v in sorted(seq.items(), key=lambda kv: (-kv[1], kv[0]))[:60]:
        out.append(f"{v:8d}  {k}")
    out.append(f"TOTAL_FF {sum(seq.values()):,}")
    out.append("")
    out.append("== latch bits (oe_dlatch) per node (top 30)")
    for k, v in sorted(lat.items(), key=lambda kv: (-kv[1], kv[0]))[:30]:
        out.append(f"{v:8d}  {k}")
    out.append(f"TOTAL_LATCH {sum(lat.values()):,}")
    out.append("")
    out.append("== ICG (oe_icg) per node (top 30)")
    for k, v in sorted(icg.items(), key=lambda kv: (-kv[1], kv[0]))[:30]:
        out.append(f"{v:8d}  {k}")
    out.append(f"TOTAL_ICG {sum(icg.values()):,}")
    out.append("")
    out.append("== RM macros per node (top 30)")
    for k, v in sorted(mac.items(), key=lambda kv: (-kv[1], kv[0]))[:30]:
        out.append(f"{v:8d}  {k}")
    out.append(f"TOTAL_RM {sum(mac.values()):,}")
    out.append("")
    out.append("== Per base module: FF bits & latch/icg/RM (instance-weighted)")
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for mname, info in modules.items():
        n = inst_count.get(mname, 0)
        if n == 0:
            continue
        b = base_name(mname)
        c = info["cells"]
        agg[b][0] += info["ff_bits"] * n
        agg[b][1] += sum(c.get(k, 0) for k in LATCH_CELLS) * n
        agg[b][2] += sum(c.get(k, 0) for k in ICG_CELLS) * n
        agg[b][3] += sum(v for k, v in c.items() if k.startswith(RM_PREFIX)) * n
        agg[b][4] += info["ff_blocks"] * n
    tot = [0, 0, 0, 0, 0]
    for b, (ff, la, ic, rm, bl) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        tot[0] += ff; tot[1] += la; tot[2] += ic; tot[3] += rm; tot[4] += bl
        out.append(f"{ff:8,d} FF  {la:8,d} latch  {ic:6,d} icg  {rm:4d} RM  {bl:6d} blocks  {b}")
    out.append(f"{tot[0]:8,d} FF  {tot[1]:8,d} latch  {tot[2]:6,d} icg  {tot[3]:4d} RM  {tot[4]:6d} blocks  TOTAL")
    out.append("")
    out.append("== Per paramod variant (top 50 by FF bits in definition)")
    vv = sorted(((info["ff_bits"], mname, inst_count.get(mname, 0),
                  sum(info['cells'].get(k,0) for k in LATCH_CELLS),
                  sum(info['cells'].get(k,0) for k in ICG_CELLS),
                  sum(v for k,v in info['cells'].items() if k.startswith(RM_PREFIX)))
                 for mname, info in modules.items()), key=lambda t: -t[0])
    for ff, mname, n, la, ic, rm in vv[:50]:
        if ff or la or ic or rm:
            out.append(f"{ff*n:10,d}  {base_name(mname):30s} x{n:<4d} (def: {ff:>6,d} FF, {la:>5d} latch, {ic:>4d} icg, {rm} RM)")
    print("\n".join(out))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "OpenEye_Asic")
