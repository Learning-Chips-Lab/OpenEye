#!/usr/bin/env python3
"""Module-level area / power / timing report + layout plots for a finished
LibreLane run of the OpenEye ASIC (flat netlist).

The synthesized netlist is flat: only explicitly instantiated cells (latch-RF
bits, ICGs, SRAM macros, pads) keep hierarchical instance names; yosys/abc
cells are called `_123_`. Module attribution therefore works in three steps:
  1. seed: hierarchical instance name, else a hierarchical name on one of the
     cell's OUTPUT nets (registers and named wires keep their RTL names),
     else a hierarchical name on an input net;
  2. clock-tree cells (clkbuf_*/clkload*/delaybuf_*, cells driving clknet_*/
     delaynet_* nets) form their own group "Taktbaum";
  3. label propagation over the netlist graph: an unlabeled cell takes the
     majority label of the cells on its nets (nets with > MAX_FANOUT pins are
     ignored: reset/enable trees), iterated until stable. Remaining cells ->
     "nicht zugeordnet".
The attribution of the anonymous combinational logic is therefore an
estimate (graph neighbourhood), exact for named instances.

Inputs (all from the run dir): 55 fill netlist (connectivity), 56 DEF
(placement), PDK + design LEFs (cell sizes), and the dumps of
module_sta.tcl (endpoint slacks per corner, instance power typ).

Usage (inside the container, needs matplotlib):
  python3 asic/scripts/doc/module_report.py <run_dir> <sta_dump_dir> <out_dir>
Writes <out_dir>/data/*.csv, <out_dir>/img/*.png and <out_dir>/tables.md
(markdown tables included into the hand-written report).
"""
import collections
import csv
import gzip
import json
import math
import os
import re
import sys
import time

MAX_FANOUT = 40
PDK = "/foss/pdks/ihp-sg13g2/libs.ref"
OUT_PINS = {"Y", "X", "Q", "Q_N", "GCLK", "L_HI", "L_LO", "p2c", "COUT", "S"}
OUT_PIN_PREFIX = ("A_DOUT",)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------- LEF sizes
def lef_sizes(paths):
    sizes = {}
    for p in paths:
        cur = None
        with open(p, errors="replace") as f:
            for line in f:
                t = line.split()
                if not t:
                    continue
                if t[0] == "MACRO":
                    cur = t[1]
                elif t[0] == "SIZE" and cur:
                    sizes[cur] = (float(t[1]), float(t[3]))
                elif t[0] == "END" and len(t) > 1 and t[1] == cur:
                    cur = None
    return sizes


# ------------------------------------------------------------ netlist parse
def norm(name):
    name = name.strip()
    if name.startswith("\\"):
        name = name[1:]
    return name.replace("\\", "").strip()


def parse_netlist(path):
    """-> list of (inst, cell, [(pin, [nets])])"""
    with open(path) as f:
        text = f.read()
    insts = []
    skip = ("wire", "input", "output", "module", "endmodule", "assign", "inout")
    for stmt in text.split(";\n"):
        s = stmt.strip()
        if not s or s.startswith(skip) or "(" not in s:
            continue
        head, _, body = s.partition("(")
        ht = head.split(None, 1)
        if len(ht) != 2:
            continue
        cell, inst = ht[0], norm(ht[1])
        pins = []
        i = 0
        n = len(body)
        while True:
            j = body.find(".", i)
            if j < 0:
                break
            k = body.find("(", j)
            if k < 0:
                break
            pin = body[j + 1:k].strip()
            if not re.fullmatch(r"[A-Za-z_]\w*", pin):
                i = j + 1
                continue
            depth, m = 1, k + 1
            while m < n and depth:
                c = body[m]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                m += 1
            content = body[k + 1:m - 1].strip().strip("{}")
            nets = []
            for tok in content.split(","):
                tok = tok.strip()
                if tok and "'" not in tok:
                    nets.append(norm(tok))
            pins.append((pin, nets))
            i = m
        insts.append((inst, cell, pins))
    return insts


# ------------------------------------------------------------ module keys
RX_CL = re.compile(r"OpenEye_Parallel\.gen_x\[(\d+)\]\.gen_y\[(\d+)\]\.OpenEye_Cluster\.")
RX_PE = re.compile(r"pe_cluster\.gen_X\[(\d+)\]\.gen_Y\[(\d+)\]\.pe\.")
PE_SUB = [
    (re.compile(r"\.pe\.(?:gen_\w+\.)?(?:gen_psum_spad\[\d+\]\.)?(\w+_SPad)\b"), None),
    (re.compile(r"\.pe\.gen_standard_mac\.genblk1\[\d+\]\.(multiplier)_\d+"), "multiplier"),
    (re.compile(r"\.pe\.(iact_data_handler|wght_data_handler|mux_psum)\b"), None),
    (re.compile(r"\.pe\.(gen_sparse_fsm)\b"), None),
]
CL_SUB = re.compile(r"OpenEye_Cluster\.(router_(?:iact|wght|psum)|delay_cluster)")
TOP_SUB = re.compile(
    r"^u_openeye\.(wght_buffer|BUFFER_A|IACT_CONVERTER|PSUM_RAM|dma_storage|"
    r"psum_pipeline_inst|ring_buffer_pipelined_inst|router_configurator_inst|"
    r"UNPACKED_TRACES)")
TOP_NAMES = {
    "wght_buffer": "wght_buffer (GLB Gewichte)",
    "BUFFER_A": "BUFFER_A (iact layer buffer)",
    "IACT_CONVERTER": "iact_stream_constructor",
    "PSUM_RAM": "psum_buffer (GLB psum)",
    "dma_storage": "dma_storage",
    "psum_pipeline_inst": "psum_pipeline (quant+pool)",
    "ring_buffer_pipelined_inst": "ring_buffer_pipelined",
    "router_configurator_inst": "router_configurator",
    "UNPACKED_TRACES": "psum_pipeline (quant+pool)",
}


def module_key(name):
    """hierarchical name -> instance key string or None.
    Key format: 'C<x><y>/PE<X><Y>/<sub>' | 'C<x><y>/<sub>' | 'TOP/<sub>'"""
    if not name.startswith("u_openeye."):
        return None
    mc = RX_CL.search(name)
    if mc:
        cl = f"C{mc.group(1)}{mc.group(2)}"
        mp = RX_PE.search(name)
        if mp:
            pe = f"PE{mp.group(1)}{mp.group(2)}"
            for rx, fixed in PE_SUB:
                ms = rx.search(name)
                if ms:
                    return f"{cl}/{pe}/{fixed or ms.group(1)}"
            return f"{cl}/{pe}/PE-Logik"
        if ".pe_cluster." in name:
            return f"{cl}/pe_cluster-Logik"
        ms = CL_SUB.search(name)
        if ms:
            return f"{cl}/{ms.group(1)}"
        return f"{cl}/Cluster-Logik"
    if name.startswith("u_openeye.OpenEye_Parallel."):
        return "TOP/OpenEye_Parallel-Logik"
    mt = TOP_SUB.search(name)
    if mt:
        return "TOP/" + TOP_NAMES[mt.group(1)]
    return "TOP/Top-Logik (OpenEye_Asic)"


def type_key(key):
    """instance key -> (group, sub) aggregated over clusters/PEs"""
    p = key.split("/")
    if p[0].startswith("C") and len(p) == 3:
        return ("PE", p[2])
    if p[0].startswith("C"):
        return ("Cluster", p[1])
    return (p[0], p[1])


def is_clock_cell(inst, cell, out_nets):
    if inst.startswith(("clkbuf_", "clkload", "delaybuf_")):
        return True
    return any(n.startswith(("clknet_", "delaynet_")) for n in out_nets)


def cell_class(cell):
    if cell.startswith("RM_IHPSG13"):
        return "SRAM-Makro"
    if cell.startswith(("sg13g2_IOPad", "sg13g2_Filler", "sg13g2_Corner", "bondpad")):
        return "IO-Pad"
    if "dlhq" in cell or "dllr" in cell:
        return "Latch"
    if "lgcp" in cell:
        return "ICG"
    if "dfr" in cell or "sdf" in cell or "dfp" in cell:
        return "Flipflop"
    if cell.startswith(("sg13g2_buf", "sg13g2_dlygate")):
        return "Puffer/Delay"
    if cell.startswith("sg13g2_antenna"):
        return "Antennen-Diode"
    if cell.startswith(("sg13g2_tie")):
        return "Tie"
    if cell.startswith(("sg13g2_fill", "sg13g2_decap")):
        return "Fill/Decap"
    return "Kombinatorik"


# ------------------------------------------------------------ STA dumps
END_RX = re.compile(r"^(\S+)\s+\((\S+)\)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+\((MET|VIOLATED)\)")


def parse_endpoints(path):
    """-> list of (group, endpoint, slack). Handles wrapped long names."""
    res = []
    group = None
    pending = None
    with open(path) as f:
        for line in f:
            if " group " in line and ("max_delay" in line or "min_delay" in line):
                group = line.strip().split(" group ")[1]
                continue
            s = line.rstrip("\n")
            if pending is not None:
                s = pending + " " + s.strip()
                pending = None
            m = END_RX.match(s.strip())
            if m:
                res.append((group, m.group(1), float(m.group(5))))
            elif s.strip() and not s.startswith(("-", " ")) and "Endpoint" not in s \
                    and "Required" not in s and len(s.split()) == 1:
                pending = s.strip()
    return res


def main():
    run, sta_dir, out = sys.argv[1:4]
    os.makedirs(f"{out}/data", exist_ok=True)
    os.makedirs(f"{out}/img", exist_ok=True)

    lefs = [f"{PDK}/sg13g2_stdcell/lef/sg13g2_stdcell.lef",
            f"{PDK}/sg13g2_io/lef/sg13g2_io.lef",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../rtl/bondpad_70x70.lef")]
    lefs += [os.path.join(f"{PDK}/sg13g2_sram/lef", x)
             for x in os.listdir(f"{PDK}/sg13g2_sram/lef") if x.endswith(".lef")]
    sizes = lef_sizes(lefs)
    log(f"LEF: {len(sizes)} masters")

    insts = parse_netlist(f"{run}/55-openroad-fillinsertion/OpenEye_Asic.nl.v")
    log(f"netlist: {len(insts)} instances")

    # ---- graph
    idx = {}
    cells = []
    net_pins = collections.defaultdict(list)
    out_nets_of = []
    in_nets_of = []
    for i, (inst, cell, pins) in enumerate(insts):
        idx[inst] = i
        cells.append(cell)
        on, inn = [], []
        for pin, nets in pins:
            is_out = pin in OUT_PINS or pin.startswith(OUT_PIN_PREFIX)
            for n in nets:
                (on if is_out else inn).append(n)
                net_pins[n].append(i)
        out_nets_of.append(on)
        in_nets_of.append(inn)
    names = [x[0] for x in insts]
    del insts
    log(f"graph: {len(net_pins)} nets")

    # ---- seeds
    label = [None] * len(names)
    seed_kind = collections.Counter()
    for i, inst in enumerate(names):
        cell = cells[i]
        cls = cell_class(cell)
        if cls == "Fill/Decap":
            label[i] = "FILL"
            continue
        if cls == "IO-Pad" or inst.startswith(("u_clk_in", "u_rst", "u_dma", "u_pad")):
            if cls == "IO-Pad":
                label[i] = "TOP/IO-Ring (Pads, Spacer, Bondpads)"
                seed_kind["pad"] += 1
                continue
        if is_clock_cell(inst, cell, out_nets_of[i]):
            label[i] = "TOP/Taktbaum"
            seed_kind["clock"] += 1
            continue
        k = module_key(inst)
        if k:
            label[i] = k
            seed_kind["inst"] += 1
            continue
        for n in out_nets_of[i]:
            k = module_key(n)
            if k:
                label[i] = k
                seed_kind["out-net"] += 1
                break
    log(f"seeds: {dict(seed_kind)}")
    # second seed pass: hierarchical input nets (for cells still unlabeled)
    n_in = 0
    for i in range(len(names)):
        if label[i] is None:
            c = collections.Counter(module_key(n) for n in in_nets_of[i])
            c.pop(None, None)
            if c:
                label[i] = c.most_common(1)[0][0]
                n_in += 1
    log(f"seeds via input nets: {n_in}")

    # ---- propagation
    skip_labels = {"FILL", "TOP/Taktbaum", "TOP/IO-Ring (Pads, Spacer, Bondpads)"}
    for it in range(60):
        changed = 0
        new = {}
        for i in range(len(names)):
            if label[i] is not None:
                continue
            votes = collections.Counter()
            for n in out_nets_of[i] + in_nets_of[i]:
                pl = net_pins[n]
                if len(pl) > MAX_FANOUT:
                    continue
                for j in pl:
                    lj = label[j]
                    if lj is not None and lj not in skip_labels and j != i:
                        votes[lj] += 1
            if votes:
                new[i] = votes.most_common(1)[0][0]
        for i, l in new.items():
            label[i] = l
            changed += 1
        log(f"propagation iter {it}: +{changed}")
        if not changed:
            break
    unassigned = 0
    for i in range(len(names)):
        if label[i] is None:
            label[i] = "TOP/nicht zugeordnet"
            unassigned += 1
    log(f"unassigned: {unassigned}")

    # ---- placement (DEF)
    pos = {}
    die = None
    with open(f"{run}/56-odb-cellfrequencytables/OpenEye_Asic.def") as f:
        incomp = False
        for line in f:
            if not incomp:
                if line.startswith("DIEAREA"):
                    t = line.replace("(", " ").replace(")", " ").split()
                    die = (int(t[3]) / 1000, int(t[4]) / 1000)
                if line.startswith("COMPONENTS"):
                    incomp = True
                continue
            if line.startswith("END COMPONENTS"):
                break
            t = line.split()
            if len(t) > 8 and t[0] == "-" and ("PLACED" in t or "FIXED" in t):
                k = t.index("(")
                pos[norm(t[1])] = (int(t[k + 1]) / 1000, int(t[k + 2]) / 1000, t[k + 4])
    log(f"DEF: {len(pos)} placed components, die {die}")

    # ---- power
    power = {}
    pfile = f"{sta_dir}/power_typ.txt"
    if os.path.exists(pfile):
        with open(pfile) as f:
            for line in f:
                t = line.rsplit(None, 5)
                if len(t) == 6:
                    power[norm(t[0])] = (float(t[2]), float(t[3]), float(t[4]), float(t[5]))
    log(f"power entries: {len(power)}")

    # ---- per-instance-key aggregation
    agg = collections.defaultdict(lambda: collections.Counter())
    for i, inst in enumerate(names):
        l = label[i]
        if l == "FILL":
            continue
        cell = cells[i]
        w, h = sizes.get(cell, (0.0, 0.0))
        a = agg[l]
        a["area"] += w * h
        a["cells"] += 1
        a["n:" + cell_class(cell)] += 1
        a["a:" + cell_class(cell)] += w * h
        p = power.get(inst)
        if p:
            a["p_int"] += p[0]
            a["p_sw"] += p[1]
            a["p_leak"] += p[2]
            a["p_tot"] += p[3]

    # ---- timing
    tim = collections.defaultdict(dict)  # key -> {corner_kind: (wns, nvio, tns)}
    worst = {}
    for corner in ("typ", "slow", "fast"):
        for kind in ("max", "min"):
            fn = f"{sta_dir}/endpoints_{corner}_{kind}.txt"
            if not os.path.exists(fn):
                continue
            eps = parse_endpoints(fn)
            log(f"endpoints {corner} {kind}: {len(eps)}")
            per = collections.defaultdict(lambda: [math.inf, 0, 0.0, 0])
            for group, ep, slack in eps:
                inst = ep.rsplit("/", 1)[0] if "/" in ep else None
                if inst is None:
                    key = "TOP/IO-Ports"
                else:
                    j = idx.get(inst)
                    key = label[j] if j is not None else (module_key(inst) or "TOP/nicht zugeordnet")
                v = per[key]
                v[0] = min(v[0], slack)
                v[3] += 1
                if slack < 0:
                    v[1] += 1
                    v[2] += slack
                wk = (corner, kind)
                if slack < worst.get(wk, (math.inf,))[0]:
                    worst[wk] = (slack, ep, group)
            for key, v in per.items():
                tim[key][f"{corner}_{kind}"] = tuple(v)

    # ---- write CSV per instance key
    classes = ["SRAM-Makro", "Latch", "ICG", "Flipflop", "Kombinatorik",
               "Puffer/Delay", "Tie", "Antennen-Diode", "IO-Pad"]
    with open(f"{out}/data/modules_by_instance.csv", "w", newline="") as f:
        wr = csv.writer(f)
        hdr = ["key", "cells", "area_um2", "p_total_mW", "p_int_mW", "p_sw_mW",
               "p_leak_mW"] + [f"n_{c}" for c in classes] + \
              [f"{c}_{k}_{m}" for c in ("typ", "slow", "fast") for k in ("max", "min")
               for m in ("wns", "nvio", "tns", "nend")]
        wr.writerow(hdr)
        for key in sorted(agg):
            a = agg[key]
            row = [key, a["cells"], round(a["area"], 1), a["p_tot"] * 1e3,
                   a["p_int"] * 1e3, a["p_sw"] * 1e3, a["p_leak"] * 1e3]
            row += [a["n:" + c] for c in classes]
            for c in ("typ", "slow", "fast"):
                for k in ("max", "min"):
                    v = tim.get(key, {}).get(f"{c}_{k}")
                    row += list(v) if v else ["", "", "", ""]
            wr.writerow(row)

    # ---- type-level aggregation
    tagg = collections.defaultdict(lambda: collections.Counter())
    tinst = collections.defaultdict(set)
    ttim = collections.defaultdict(dict)
    for key, a in agg.items():
        tk = type_key(key)
        tagg[tk].update(a)
        tinst[tk].add(key)
    for key, d in tim.items():
        tk = type_key(key) if key in agg or "/" in key else ("TOP", key)
        for ck, v in d.items():
            cur = ttim[tk].get(ck)
            if cur is None:
                ttim[tk][ck] = list(v)
            else:
                cur[0] = min(cur[0], v[0]); cur[1] += v[1]; cur[2] += v[2]; cur[3] += v[3]

    tot_area = sum(a["area"] for a in agg.values())
    tot_p = sum(a["p_tot"] for a in agg.values())

    order_pe = ["weight_data_SPad", "psum_SPad", "weight_addr_SPad", "iact_addr_SPad",
                "iact_data_SPad", "multiplier", "iact_data_handler", "wght_data_handler",
                "mux_psum", "gen_sparse_fsm", "PE-Logik"]
    order_cl = ["router_iact", "router_wght", "router_psum", "delay_cluster",
                "pe_cluster-Logik", "Cluster-Logik"]
    rows = [("PE", s) for s in order_pe] + [("Cluster", s) for s in order_cl]
    rows += sorted(k for k in tagg if k[0] == "TOP")
    rows += [k for k in list(tagg) + list(ttim) if k not in rows]

    def f_area(x):
        return f"{x / 1e6:.3f}"

    def f_slack(v):
        if not v:
            return "–"
        w = v[0]
        s = f"{w:+.2f}"
        return f"**{s}** ({v[1]})" if w < 0 else s

    md = []
    md.append("| Modul | Inst. | Zellen | Fläche [mm²] | Anteil | davon Latch / FF / Komb. / Puffer [mm²] | Leistung typ [mW] | Anteil |")
    md.append("|---|---:|---:|---:|---:|---|---:|---:|")
    for tk in rows:
        if tk not in tagg:
            continue
        a = tagg[tk]
        name = f"PE / {tk[1]}" if tk[0] == "PE" else (
            f"Cluster / {tk[1]}" if tk[0] == "Cluster" else tk[1])
        n_inst = len(tinst[tk])
        parts = " / ".join(f_area(a["a:" + c]) for c in ("Latch", "Flipflop", "Kombinatorik", "Puffer/Delay"))
        if a["a:SRAM-Makro"]:
            parts += f" (+ SRAM {f_area(a['a:SRAM-Makro'])})"
        md.append(f"| {name} | {n_inst} | {a['cells']:,} | {f_area(a['area'])} | "
                  f"{100 * a['area'] / tot_area:.1f} % | {parts} | {a['p_tot'] * 1e3:.1f} | "
                  f"{100 * a['p_tot'] / tot_p:.1f} % |" if tot_p else
                  f"| {name} | {n_inst} | {a['cells']:,} | {f_area(a['area'])} | "
                  f"{100 * a['area'] / tot_area:.1f} % | {parts} | – | – |")
    md.append(f"| **Summe (ohne Fill/Decap)** | | {sum(a['cells'] for a in tagg.values()):,} | "
              f"{f_area(tot_area)} | 100 % | | {tot_p * 1e3:.1f} | 100 % |")
    area_md = "\n".join(md)

    md = []
    md.append("| Modul | Setup typ | Setup slow | Setup fast | Hold typ | Hold slow | Hold fast |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for tk in rows:
        if tk not in ttim:
            continue
        d = ttim[tk]
        name = f"PE / {tk[1]}" if tk[0] == "PE" else (
            f"Cluster / {tk[1]}" if tk[0] == "Cluster" else tk[1])
        md.append(f"| {name} | " + " | ".join(
            f_slack(d.get(f"{c}_{k}")) for k in ("max", "min") for c in ("typ", "slow", "fast")) + " |")
    timing_md = "\n".join(md)

    # ---- per-cluster / per-PE spread
    pe_area = collections.Counter()
    pe_pow = collections.Counter()
    cl_area = collections.Counter()
    cl_pow = collections.Counter()
    for key, a in agg.items():
        p = key.split("/")
        if p[0].startswith("C"):
            cl_area[p[0]] += a["area"]; cl_pow[p[0]] += a["p_tot"]
            if len(p) == 3:
                pe_area[p[0] + "/" + p[1]] += a["area"]; pe_pow[p[0] + "/" + p[1]] += a["p_tot"]
    md = ["| Cluster | Fläche [mm²] | Leistung typ [mW] |", "|---|---:|---:|"]
    for c in sorted(cl_area):
        md.append(f"| {c} | {f_area(cl_area[c])} | {cl_pow[c] * 1e3:.1f} |")
    if pe_area:
        v = list(pe_area.values()); pw = list(pe_pow.values())
        md.append("")
        md.append(f"PE ({len(v)} Stück): Fläche min/mittel/max = {min(v) / 1e6:.4f} / "
                  f"{sum(v) / len(v) / 1e6:.4f} / {max(v) / 1e6:.4f} mm², Leistung typ "
                  f"min/mittel/max = {min(pw) * 1e3:.2f} / {sum(pw) / len(pw) * 1e3:.2f} / "
                  f"{max(pw) * 1e3:.2f} mW.")
    cluster_md = "\n".join(md)

    worst_md = ["| Corner | Prüfung | Slack [ns] | schlechtester Endpunkt |", "|---|---|---:|---|"]
    for (c, k), (s, ep, g) in sorted(worst.items()):
        worst_md.append(f"| {c} | {'Setup' if k == 'max' else 'Hold'} ({g}) | {s:+.3f} | `{ep}` |")

    with open(f"{out}/tables.md", "w") as f:
        f.write("<!-- generated by asic/scripts/doc/module_report.py -->\n\n")
        f.write("### Fläche und Leistung nach Modul\n\n" + area_md + "\n\n")
        f.write("### Timing nach Modul (schlechtester Slack in ns, in Klammern: Anzahl verletzender Endpunkte)\n\n" + timing_md + "\n\n")
        f.write("### Schlechteste Endpunkte je Corner\n\n" + "\n".join(worst_md) + "\n\n")
        f.write("### Cluster und PEs\n\n" + cluster_md + "\n\n")
        f.write(f"Zuordnung: {dict(seed_kind)}, über Eingangsnetze {n_in}, "
                f"nicht zugeordnet {unassigned} Zellen.\n")
    json.dump({"seed": dict(seed_kind), "seed_in": n_in, "unassigned": unassigned,
               "tot_area_um2": tot_area, "tot_power_W": tot_p,
               "worst": {f"{c}_{k}": v for (c, k), v in worst.items()}},
              open(f"{out}/data/summary.json", "w"), indent=1)
    log("tables written")

    # ---- plots
    plot(out, names, cells, label, pos, sizes, power, die)
    log("done")


def plot(out, names, cells, label, pos, sizes, power, die):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    groups = {
        "PE: weight_data_SPad": "#1f77b4", "PE: psum_SPad": "#17becf",
        "PE: andere SPads": "#9edae5", "PE: Logik/MAC": "#2ca02c",
        "Router/Cluster-Logik": "#bcbd22", "GLB/Top-Logik": "#ff7f0e",
        "psum_pipeline": "#d62728", "Taktbaum": "#7f7f7f", "sonstige": "#c7c7c7",
    }

    def group_of(l):
        if l.startswith("C"):
            p = l.split("/")
            if len(p) == 3:
                if p[2] == "weight_data_SPad":
                    return "PE: weight_data_SPad"
                if p[2] == "psum_SPad":
                    return "PE: psum_SPad"
                if p[2].endswith("_SPad"):
                    return "PE: andere SPads"
                return "PE: Logik/MAC"
            return "Router/Cluster-Logik"
        if "Taktbaum" in l:
            return "Taktbaum"
        if "psum_pipeline" in l:
            return "psum_pipeline"
        if l.startswith("TOP/") and "Pads" not in l and "nicht" not in l:
            return "GLB/Top-Logik"
        return "sonstige"

    xs = collections.defaultdict(list)
    ys = collections.defaultdict(list)
    macros = []
    pads = []
    clx, cly, clc = [], [], []
    cl_col = {"C00": "#1f77b4", "C01": "#2ca02c", "C10": "#ff7f0e", "C11": "#9467bd"}
    pxy = []
    for i, n in enumerate(names):
        l = label[i]
        if l == "FILL":
            continue
        p = pos.get(n)
        if not p:
            continue
        c = cells[i]
        w, h = sizes.get(c, (0, 0))
        if p[2] in ("E", "W", "FE", "FW", "R90", "R270"):
            w, h = h, w
        if c.startswith("RM_IHPSG13"):
            macros.append((p[0], p[1], w, h, n))
            continue
        if c.startswith(("sg13g2_IOPad", "sg13g2_Filler", "sg13g2_Corner", "bondpad")) or "IO-Ring" in l:
            pads.append((p[0], p[1], w, h))
            continue
        cx, cy = p[0] + w / 2, p[1] + h / 2
        g = group_of(l)
        xs[g].append(cx); ys[g].append(cy)
        if l.startswith("C"):
            clx.append(cx); cly.append(cy); clc.append(cl_col.get(l[:3], "#000"))
        pw = power.get(n)
        if pw:
            pxy.append((cx, cy, pw[3]))

    def frame(ax, title):
        ax.set_xlim(0, die[0]); ax.set_ylim(0, die[1]); ax.set_aspect("equal")
        ax.set_xlabel("x [µm]"); ax.set_ylabel("y [µm]"); ax.set_title(title)
        for (x, y, w, h, n) in macros:
            ax.add_patch(Rectangle((x, y), w, h, fill=True, fc="#444444", ec="black", lw=0.4, alpha=0.85))
        for (x, y, w, h) in pads:
            ax.add_patch(Rectangle((x, y), w, h, fill=True, fc="#999999", ec="none"))

    from matplotlib.colors import ListedColormap, to_rgb
    BIN = 20.0
    nb = int(math.ceil(die[0] / BIN))

    def majority_map(gx, gy, keys):
        """per BIN x BIN field: index of the group with the most cells (-1 empty)"""
        cnt = np.zeros((len(keys), nb, nb), dtype=np.int32)
        for k, g in enumerate(keys):
            if gx[g]:
                H, _, _ = np.histogram2d(gx[g], gy[g], bins=[nb, nb], range=[[0, die[0]], [0, die[1]]])
                cnt[k] = H.T
        idx = cnt.argmax(axis=0)
        idx[cnt.sum(axis=0) == 0] = -1
        return idx

    def draw_map(ax, idx, cols):
        rgb = np.ones((nb, nb, 3))
        for k, col in enumerate(cols):
            rgb[idx == k] = to_rgb(col)
        ax.imshow(rgb, origin="lower", extent=[0, die[0], 0, die[1]], interpolation="nearest")

    # 1) module map
    keys = [g for g in groups if xs[g]]
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    draw_map(ax, majority_map(xs, ys, keys), [groups[g] for g in keys])
    frame(ax, "OpenEye p2_16 – vorherrschendes Modul je 20×20 µm (Makros dunkelgrau, Pads hellgrau)")
    for g, col in groups.items():
        if xs[g]:
            ax.scatter([], [], s=40, c=col, label=f"{g} ({len(xs[g]):,})")
    ax.legend(loc="upper right", fontsize=7, markerscale=1, framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{out}/img/placement_modules.png"); plt.close(fig)

    # 2) cluster map
    cx_g = collections.defaultdict(list); cy_g = collections.defaultdict(list)
    for x_, y_, c_ in zip(clx, cly, clc):
        cx_g[c_].append(x_); cy_g[c_].append(y_)
    ckeys = list(cl_col.values())
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    draw_map(ax, majority_map(cx_g, cy_g, ckeys), ckeys)
    frame(ax, "OpenEye p2_16 – Platzierung der 4 Cluster (PE-Array, Router; je 20×20 µm)")
    for k, col in cl_col.items():
        ax.scatter([], [], s=40, c=col, label=f"Cluster {k[1]},{k[2]} (gen_x, gen_y)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/img/placement_clusters.png"); plt.close(fig)

    # 3) power density
    if pxy:
        a = np.array(pxy)
        b = 100.0
        nb = int(math.ceil(die[0] / b))
        H, xe, ye = np.histogram2d(a[:, 0], a[:, 1], bins=[nb, nb], range=[[0, die[0]], [0, die[1]]],
                                   weights=a[:, 2] * 1e3)
        fig, ax = plt.subplots(figsize=(11, 10), dpi=150)
        im = ax.imshow((H.T / (b * b / 1e6)), origin="lower", extent=[0, die[0], 0, die[1]],
                       cmap="inferno", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Leistungsdichte [mW/mm²] (typ, vektorlos)")
        for (x, y, w, h, n) in macros:
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="cyan", lw=0.6))
        ax.set_title("OpenEye p2_16 – Leistungsdichte Standardzellen (100 µm Raster; Makros cyan umrandet)")
        ax.set_xlabel("x [µm]"); ax.set_ylabel("y [µm]")
        fig.tight_layout(); fig.savefig(f"{out}/img/power_density.png"); plt.close(fig)


if __name__ == "__main__":
    main()
