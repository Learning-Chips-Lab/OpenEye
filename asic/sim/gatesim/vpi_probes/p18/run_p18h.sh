#!/bin/bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
chk() { local L=$1; shift; local T=$1; shift
  rm -rf m18h_$L
  verilator --lint-only --top-module $T --dumpi-graph 6 -Mdir m18h_$L "$@" >m18h_$L.log 2>&1 || true
  local F=$(ls m18h_$L/*inlinemod-decision.dot 2>/dev/null | head -1)
  [ -z "$F" ] && { echo "$L: NO DOT: $(tail -2 m18h_$L.log | head -1)"; return; }
  python3 dump_inline_why.py "$F" a1 2>/dev/null | grep -E "^a1 |NoInlineSoft" | sed "s/^/  $L: /"
}
# h1: a1 FIRST file, top SECOND file
chk h1 topT2 big18c_a1mod.v big18c_t2top.v
# h2: top first, a1 second (control = T2 layout)
chk h2 topT2 big18c_t2top.v big18c_a1mod.v
# h3: top + a1 in file2, cell file1 empty-ish (module a2 unused) - check a1 still
# (control for "any other file present")
printf 'module a2 (input x, output y); assign y = x; endmodule\n' > big18c_a2mod.v
chk h3 topT2 big18c_a2mod.v big18c_t2top.v big18c_a1mod.v

