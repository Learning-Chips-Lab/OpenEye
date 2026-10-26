#!/bin/bash
# P18e: definitive test — does a PUBLIC LEAF module retain a per-instance VPI scope?
# T2 (a1 in own file) is a public leaf per run_p18c. Compile WITH --vpi --flatten.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
M=mprobe_p18e
rm -rf "$M"
verilator -cc --exe -j 16 --timescale 1ns/10ps --timing -Wno-fatal \
  --top-module topT2 -DFUNCTIONAL --flatten \
  -Mdir "$M" --vpi --prefix Vtop -o Vtop \
  big18c_t2top.v big18c_a1mod.v >probe_p18e.log 2>&1
grep -m1 "Built from" probe_p18e.log || { echo VERILATOR_FAILED; tail -15 probe_p18e.log; exit 1; }
echo "total scopes: $(sed -n '/VpiScopeTable\[\] = {/,/};/p' "$M/Vtop__Syms__Slow.cpp" | grep -c 'SCOPE_')"
echo "per-instance (topT2.u*) scopes: $(sed -n '/VpiScopeTable\[\] = {/,/};/p' "$M/Vtop__Syms__Slow.cpp" | grep -c 'topT2\.u' || true)"
echo "--- scope table rows:"
{ sed -n '/VpiScopeTable\[\] = {/,/};/p' "$M/Vtop__Syms__Slow.cpp" | grep -oE '"[^"]+"(, "[^"]+")+' | head; } || true
echo
echo "=== Was a1 public in this run's decision? (quick re-dump)"
rm -rf m18e_dec
verilator -cc -j 16 --timescale 1ns/10ps --timing -Wno-fatal \
  --top-module topT2 --flatten --dumpi-graph 6 -Mdir m18e_dec \
  big18c_t2top.v big18c_a1mod.v >m18e_dec.log 2>&1 || true
python3 dump_inline_why.py m18e_dec/*inlinemod-decision.dot a1 2>/dev/null | grep -E "^a1|NoInlineSoft|DoInline"
