#!/bin/bash
# P15b: --flatten + marked top + FF + no delays (non-degenerate scheduling).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
COUNT="${M:-1000}"
COCOTB_LIB_DIR="$(python3 -m cocotb_tools.config --lib-dir)"
COCOTB_CPP="$(python3 -m cocotb_tools.config --share)/lib/verilator/verilator.cpp"
M=mprobe_p15b

python3 gen_p15b.py "$COUNT" big15b.v
rm -rf "$M"
verilator -cc --exe -j 16 --timescale 1ns/10ps --timing -Wno-fatal \
  --top-module bigtop -DFUNCTIONAL --flatten \
  -Mdir "$M" --vpi --prefix Vtop -o Vtop \
  -LDFLAGS "-Wl,-rpath,${COCOTB_LIB_DIR} -L${COCOTB_LIB_DIR} -lcocotbvpi_verilator" \
  big15b.v "${COCOTB_CPP}" >probe_p15b.log 2>&1
grep -m1 "Built from" probe_p15b.log || { echo VERILATOR_FAILED; tail -8 probe_p15b.log; exit 1; }
echo "generated: $(du -sm $M | cut -f1) MB, files: $(ls $M/*.cpp | wc -l)"
echo "non-root classes: $(ls $M/Vtop*.h 2>/dev/null | grep -vcE '024root|__pch|Vtop\.h$|Syms')"
echo "scope table:"
sed -n '/VpiScopeTable\[\] = {/,/};/p' "$M/Vtop__Syms__Slow.cpp" | grep -oE '"[^"]+"' | head -6
echo "table entries: $(grep -hoE '\{"[A-Za-z_0-9]+", offsetof' $M/*.cpp | wc -l)"
echo "TOP__bigtop member (mirror instance): $(grep -c 'TOP__bigtop' $M/Vtop__Syms.h)"
make -C "$M" -f Vtop.mk -j 16 >>probe_p15b.log 2>&1
[ -x "$M/Vtop" ] || { echo LINK_FAILED; tail -5 probe_p15b.log; exit 1; }
echo "Vtop linked"
cd "$M"
TOPLEVEL_LANG=verilog COCOTB_TOPLEVEL=bigtop COCOTB_TEST_MODULES=p15b_tb \
COCOTB_TRUST_INERTIAL_WRITES=1 PYTHONPATH="$(cd .. && pwd)" \
PYGPI_PYTHON_BIN="$(python3 -m cocotb_tools.config --python-bin)" \
GPI_USERS="$(python3 -m cocotb_tools.config --libpython);$(python3 -m cocotb_tools.config --pygpi-entry-point)" \
./Vtop 2>&1 | grep -E "after write|after edge|P15B_OK|FAIL|PASS|Error" | head -8
