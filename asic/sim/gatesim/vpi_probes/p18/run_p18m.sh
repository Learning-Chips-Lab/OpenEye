#!/bin/bash
# P18m: validate the file-ORDER fix under --vpi --flatten.
# Leaf cases (a1, P18e layout):
#   A (top first):  big18c_t2top.v big18c_a1mod.v   (current build layout)
#   B (cells first): big18c_a1mod.v big18c_t2top.v  (proposed fix)
# 2-level cases (cparent instantiates cchild):
#   C (cells first):  big18m_c2cells.v big18m_c2top.v  (proposed fix layout)
#   D (top first):    big18m_c2top.v big18m_c2cells.v  (control = current layout)
# Expect: A and D -> per-instance scopes (cells public); B and C -> none (cells non-public).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# --- 2-level fixtures: child FF leaf + parent with 2 child instances ---
cat > big18m_c2cells.v <<'EOF'
module cchild (q, v, clk, d, r, xcr);
  output reg q;
  input v, clk, d, r, xcr;
  always @(posedge clk or posedge r)
    if (r) q <= 1'b0; else q <= d;
endmodule
module cparent (Q, CLK, D, RESET_B);
  output Q;
  input CLK, D, RESET_B;
  wire int_q, int_r, xcr_0;
  not (int_r, RESET_B);
  buf (xcr_0, 0);
  cchild u_prim_1 (.q(int_q), .v(1'b0), .clk(CLK), .d(D), .r(int_r), .xcr(xcr_0));
  cchild u_prim_2 (.q(Q), .v(1'b0), .clk(CLK), .d(int_q), .r(int_r), .xcr(xcr_0));
endmodule
EOF
cat > big18m_c2top.v <<'EOF'
module top18mC (
  /*verilator public_on*/ input clk,
  /*verilator public_on*/ input d,
  /*verilator public_on*/ input r,
  /*verilator public_on*/ output [3:0] xs
);
  cparent u4_0 (.Q(xs[0]), .CLK(clk), .D(d), .RESET_B(r));
  cparent u4_1 (.Q(xs[1]), .CLK(clk), .D(d), .RESET_B(r));
  cparent u4_2 (.Q(xs[2]), .CLK(clk), .D(d), .RESET_B(r));
  cparent u4_3 (.Q(xs[3]), .CLK(clk), .D(d), .RESET_B(r));
endmodule
EOF

run_case() { # $1=label $2=top  $3...=files
  local L=$1 TOP=$2; shift 2
  local M=mprobe_p18m_$L
  rm -rf "$M"
  verilator -cc --exe -j 16 --timescale 1ns/10ps --timing -Wno-fatal \
    --top-module "$TOP" -DFUNCTIONAL --flatten \
    -Mdir "$M" --vpi --prefix Vtop -o Vtop "$@" >"probe_p18m_$L.log" 2>&1
  if ! grep -m1 "Built from" "probe_p18m_$L.log" >/dev/null; then
    echo "$L VERILATOR_FAILED"; tail -8 "probe_p18m_$L.log"; return
  fi
  local S=$M/Vtop__Syms__Slow.cpp
  echo "===== case $L (top=$TOP files: $*):"
  echo "  total scopes: $(sed -n '/VpiScopeTable\[\] = {/,/};/p' $S | grep -c SCOPE_)"
  echo "  per-instance scopes: $(sed -n '/VpiScopeTable\[\] = {/,/};/p' $S | grep -cE "\"$TOP\." || true)"
  echo "  scope rows:"
  sed -n '/VpiScopeTable\[\] = {/,/};/p' $S | grep -oE '"[A-Za-z0-9_.]+", "[A-Za-z0-9_.]+", "[^"]*"' | sed 's/^/    /'
  echo "  var-table inserts:"
  grep -oE "varsInsertFromTable\(Vtop___024root__VpiVarTable[0-9]+, [0-9]+" $S | sed 's/.*Table/    Table/'
}

run_case A topT2   big18c_t2top.v big18c_a1mod.v
run_case B topT2   big18c_a1mod.v big18c_t2top.v
run_case C top18mC big18m_c2cells.v big18m_c2top.v
run_case D top18mC big18m_c2top.v big18m_c2cells.v
