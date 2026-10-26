#!/usr/bin/env python3
"""Summarize a Verilator inlinemod-decision dot file.

Usage: dump_inline_why.py <file.dot> [keyword ...]
Prints labels for nodes whose first label line contains one of the
keywords (default: all MODULE nodes). Node color: green=inlined,
orange=kept, black=not marked doInline.
"""
import re
import sys

txt = open(sys.argv[1]).read()
kws = sys.argv[2:]

pat = re.compile(r'(\w+)\s+\[fontsize=\d+ label="((?:[^"\\]|\\.)*)"\s*,\s*color=(\w+)\s*,\s*shape=(\w+)\]')
for m in pat.finditer(txt):
    nid, lab, color, shape = m.group(1), m.group(2), m.group(3), m.group(4)
    lab = lab.replace('\\"', '"').replace('\\n', '\n')
    first = lab.split("\n")[0]
    kind = first.split(" ", 1)[0]
    if kws:
        if not any(k in lab for k in kws):
            continue
    elif kind != "MODULE":
        continue
    print("== %s [%s/%s]" % (nid, color, shape))
    print(lab)
    print()
