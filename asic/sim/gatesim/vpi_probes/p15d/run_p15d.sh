#!/bin/bash
# P15d: PDK-faithful specify probe (in-container, cwd $(dirname "${BASH_SOURCE[0]}")).
# raw  = PDK cell modules verbatim (specify blocks present)
# tf   = modules through pdk_specify_strip.py (specify gone, delayed_ aliases)
# Both: --flatten + 10 /*verilator public_on*/ wrapper ports + --timing,
# file order = primitives first, then cells+top (fixed real-build layout,
# all cells non-public -> 2 scopes expected for both variants),
# cocotb functional run with p15d_tb (comb + FF path via ffq0_o tap).
# Verdict: raw -> FF FAIL (delayed_* undriven, folds to 0); tf -> P15D_OK.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
COCOTB_LIB_DIR="$(python3 -m cocotb_tools.config --lib-dir)"
COCOTB_CPP="$(python3 -m cocotb_tools.config --share)/lib/verilator/verilator.cpp"
PRIMS=/foss/designs/OpenEye/asic/sim/gatesim/sg13g2_primitives_vlt.v

python3 gen_p15d.py 10 10

for V in raw tf; do
  M="mprobe_p15d_${V}"
  echo "================ P15d variant: ${V} ================"
  rm -rf "$M"
  verilator -cc --exe -j 16 --timescale 1ns/10ps --timing -Wno-fatal \
    --top-module p15dtop -DFUNCTIONAL --flatten \
    -Mdir "$M" --vpi --prefix Vtop -o Vtop \
    -LDFLAGS "-Wl,-rpath,${COCOTB_LIB_DIR} -L${COCOTB_LIB_DIR} -lcocotbvpi_verilator" \
    "${PRIMS}" "p15d_${V}.v" "${COCOTB_CPP}" >"probe_p15d_${V}.log" 2>&1
  grep -m1 "Built from" "probe_p15d_${V}.log" || { echo VERILATOR_FAILED; tail -15 "probe_p15d_${V}.log"; exit 1; }
  echo "generated: $(du -sm $M | cut -f1) MB"
  echo "SPECIFYIGN warnings: $(grep -c 'SPECIFYIGN' "probe_p15d_${V}.log" || true)"
  SYMS="$M/Vtop__Syms__Slow.cpp"
  if [ -f "$SYMS" ]; then
    echo "scope count: $(sed -n '/VpiScopeTable\[\] = {/,/};/p' "$SYMS" | grep -c 'SCOPE_')"
    echo "scope table (first 8 names):"
    { sed -n '/VpiScopeTable\[\] = {/,/};/p' "$SYMS" | grep -oE '"[^"]+"' | grep -vE '^"<null>"$' | head -8; } || true
    echo "  per-instance scopes: $(sed -n '/VpiScopeTable\[\] = {/,/};/p' "$SYMS" | grep -c '\.[a-z_0-9]*u_')"
  else
    echo "NOTE: $SYMS not found (split ctor?), listing:"
    ls "$M" | grep -i syms | head -5
  fi
  echo "table entries (offsetof count): $(grep -ho 'offsetof' $M/*.cpp 2>/dev/null | wc -l)"
  make -C "$M" -f Vtop.mk -j 16 >>"probe_p15d_${V}.log" 2>&1
  [ -x "$M/Vtop" ] || { echo LINK_FAILED; tail -5 "probe_p15d_${V}.log"; exit 1; }
  (cd "$M" && TOPLEVEL_LANG=verilog COCOTB_TOPLEVEL=p15dtop COCOTB_TEST_MODULES=p15d_tb \
    COCOTB_TRUST_INERTIAL_WRITES=1 PYTHONPATH="$(cd .. && pwd)" \
    PYGPI_PYTHON_BIN="$(python3 -m cocotb_tools.config --python-bin)" \
    GPI_USERS="$(python3 -m cocotb_tools.config --libpython);$(python3 -m cocotb_tools.config --pygpi-entry-point)" \
    ./Vtop 2>&1 | grep -E "after write|after edge|P15D_OK|FAIL|PASS|Error" | head -8) || true
done
echo "================ P15d done ================"
