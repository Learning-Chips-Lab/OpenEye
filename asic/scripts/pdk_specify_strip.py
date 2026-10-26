#!/usr/bin/env python3
"""Strip specify blocks from IHP SG13G2 PDK models for Verilator functional
gate simulation.

Why (evidence: gate-sim builds 5/6 + vpi_micro probes P15d/P18, 2026-10-25):
  Verilator 5.052 ignores specify block contents (%Warning-SPECIFYIGN) and
  is 2-state: the 57 `delayed_*` wires that only the timing blocks of the
  16 FF/latch stdcell models drive would be left undriven, fold to 0, and
  break the functional behavior of every sequential cell.
  (The 3.19M-scope / ~360 GB codegen explosion of build 6 had a different
  cause: Verilator auto-marks modules parsed AFTER the --top-module
  definition public; see the asic/sim/gatesim/run_verilator.sh header.)

All IHP SG13G2 specify delays are 0.0, so the functional behavior is
preserved by aliasing the signals that only the specify block drives:
  wire delayed_X  ->  assign delayed_X = X;   (X must be a module port)
  reg notifier    ->  left alone: only driven by $width/$recrem inside the
                      specify block; the behavioral primitive models
                      (sg13g2_primitives_vlt.v) ignore the notifier input,
                      and Verilator constant-folds the undriven reg.

Usage:
  pdk_specify_strip.py <in.v> <out.v>

Sanity checks (hard abort on violation):
  - balanced specify/endspecify in the input
  - zero 'specify' keywords in the output
  - at least one specify block found (expected for a PDK model)
  - every 'delayed_X' wire has a matching port X in the same module
  - every 'delayed_X' gets exactly one alias assign
"""
import re
import sys


def fail(msg):
    sys.stderr.write("pdk_specify_strip: ERROR: %s\n" % msg)
    sys.exit(1)


def main():
    if len(sys.argv) != 3:
        fail("usage: pdk_specify_strip.py <in.v> <out.v>")
    with open(sys.argv[1]) as f:
        lines = f.readlines()

    out = []
    ports = set()
    in_specify = False
    n_blocks = 0
    all_delayed = []

    for lineno, raw in enumerate(lines, 1):
        s = raw.strip()

        if in_specify:
            # drop everything until end specify
            if re.match(r"^endspecify\b", s):
                in_specify = False
            continue

        if re.match(r"^specify\b", s):
            n_blocks += 1
            in_specify = True
            continue

        m = re.match(r"^module\s+\w+\s*\(([^)]*)\)", s)
        if m:
            ports = set(p.strip() for p in m.group(1).split(",") if p.strip())
            out.append(raw)
            continue
        if re.match(r"^endmodule\b", s):
            ports = set()
            out.append(raw)
            continue

        if re.match(r"^(input|output|inout)\b[^;]*;", s):
            for p in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s):
                if p not in ("input", "output", "inout"):
                    ports.add(p)
            out.append(raw)
            continue

        m = re.match(r"^wire\s+([^;]+);", s)
        if m:
            names = [n.strip() for n in m.group(1).split(",") if n.strip()]
            delayed = [n for n in names if n.startswith("delayed_")]
            if delayed:
                out.append(raw)
                for n in delayed:
                    base = n[len("delayed_"):]
                    if base not in ports:
                        fail("line %d: wire %s has no matching port %s "
                             "(ports=%s)" % (lineno, n, base, sorted(ports)))
                    out.append("\tassign %s = %s;  // alias for stripped timing block (0.0 delay)\n"
                               % (n, base))
                    all_delayed.append(n)
                continue

        out.append(raw)

    if in_specify:
        fail("unterminated specify block")

    text = "".join(out)
    if "specify" in text:
        fail("specify keyword survived in output")
    if n_blocks == 0:
        fail("no specify blocks found in input (unexpected for a PDK model)")

    with open(sys.argv[2], "w") as f:
        f.write(text)
    print("pdk_specify_strip: %d specify blocks stripped, "
          "%d delayed_ aliases added -> %s"
          % (n_blocks, len(all_delayed), sys.argv[2]))


if __name__ == "__main__":
    main()
