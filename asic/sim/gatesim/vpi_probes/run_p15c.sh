#!/bin/bash
# P15c: P15b + one `initial #0` -> VM_TIMING=1 (the REAL build corner:
# --flatten + marked top + cocotb-driven clock + timing scheduler active).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
COUNT="${M:-1000}"
COCOTB_LIB_DIR="$(python3 -m cocotb_tools.config --lib-dir)"
COCOTB_CPP="$(python3 -m cocotb_tools.config --share)/lib/verilator/verilator.cpp"
M=mprobe_p15c

python3 gen_p15c.py "$COUNT" big15c.v
rm -rf "$M"
verilator -cc --exe -j 16 --timescale 1ns/10ps --timing -Wno-fatal \
  --top-module bigtop -DFUNCTIONAL --flatten \
  -Mdir "$M" --vpi --prefix Vtop -o Vtop \
  -LDFLAGS "-Wl,-rpath,${COCOTB_LIB_DIR} -L${COCOTB_LIB_DIR} -lcocotbvpi_verilator" \
  big15c.v "${COCOTB_CPP}" >probe_p15c.log 2>&1
grep -m1 "Built from" probe_p15c.log || { echo VERILATOR_FAILED; tail -8 probe_p15c.log; exit 1; }
echo "generated: $(du -sm $M | cut -f1) MB, files: $(ls $M/*.cpp | wc -l)"
echo "VM_TIMING: $(grep '^VM_TIMING' $M/Vtop_classes.mk | awk '{print $3}')"
echo "non-root classes: $(ls $M/Vtop*.h 2>/dev/null | grep -vcE '024root|__pch|Vtop\.h$|Syms')"
echo "scope table:"
sed -n '/VpiScopeTable\[\] = {/,/};/p' "$M/Vtop__Syms__Slow.cpp" | grep -oE '"[^"]+"' | head -6
echo "table entries: $(grep -hoE '\{"[A-Za-z_0-9]+", offsetof' $M/*.cpp | wc -l)"
echo "TOP__bigtop member (mirror instance): $(grep -c 'TOP__bigtop' $M/Vtop__Syms.h || true)"
echo "--- ico_sequent guards (MUST be firstIteration-only, word [1U] bit 0):"
grep -h -B6 "_ico_sequent__TOP__0" $M/*.cpp 2>/dev/null | grep -E "if \(\[0-9A-Fa-fx]+\s*ULL? *&|VicoTriggered" | sort -u
make -C "$M" -f Vtop.mk -j 16 >>probe_p15c.log 2>&1
[ -x "$M/Vtop" ] || { echo LINK_FAILED; tail -5 probe_p15c.log; exit 1; }
echo "Vtop linked"
cd "$M"
TOPLEVEL_LANG=verilog COCOTB_TOPLEVEL=bigtop COCOTB_TEST_MODULES=p15b_tb \
COCOTB_TRUST_INERTIAL_WRITES=1 PYTHONPATH="$(cd .. && pwd)" \
PYGPI_PYTHON_BIN="$(python3 -m cocotb_tools.config --python-bin)" \
GPI_USERS="$(python3 -m cocotb_tools.config --libpython);$(python3 -m cocotb_tools.config --pygpi-entry-point)" \
./Vtop 2>&1 | grep -E "after write|after edge|P15B_OK|FAIL|PASS|Error" | head -8
