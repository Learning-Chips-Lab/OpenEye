#!/bin/bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
chk() { local L=$1; shift; local T=$1; shift
  rm -rf m18i_$L
  verilator --lint-only --top-module $T --dumpi-graph 6 -Mdir m18i_$L "$@" >m18i_$L.log 2>&1 || true
  local F=$(ls m18i_$L/*inlinemod-decision.dot 2>/dev/null | head -1)
  [ -z "$F" ] && { echo "$L: NO DOT: $(tail -2 m18i_$L.log | head -1)"; return; }
  python3 dump_inline_why.py "$F" a1 2>/dev/null | grep -E "^a1 |NoInlineSoft" | sed "s/^/  $L: /"
}
# i1: ONE file, top defined FIRST, a1 after  -> if a1 public: parse-order rule!
cat > big18i_i1.v <<'EOF'
module topI1 (
  /*verilator public_on*/ input clk,
  /*verilator public_on*/ input d,
  /*verilator public_on*/ input r,
  /*verilator public_on*/ output [1:0] xs
);
  a1 u1_0 (.q(xs[0]), .v(1'b0), .clk(clk), .d(d), .r(r), .xcr(1'b0));
  a1 u1_1 (.q(xs[1]), .v(1'b0), .clk(clk), .d(d), .r(r), .xcr(1'b0));
endmodule
module a1 (q, v, clk, d, r, xcr);
  output reg q;
  input v, clk, d, r, xcr;
  always @(posedge clk or posedge r)
    if (r) q <= 1'b0; else q <= d;
endmodule
EOF
# i2: ONE file, a1 defined FIRST, top after (control = T1 layout)
cat > big18i_i2.v <<'EOF'
module a1 (q, v, clk, d, r, xcr);
  output reg q;
  input v, clk, d, r, xcr;
  always @(posedge clk or posedge r)
    if (r) q <= 1'b0; else q <= d;
endmodule
module topI2 (
  /*verilator public_on*/ input clk,
  /*verilator public_on*/ input d,
  /*verilator public_on*/ input r,
  /*verilator public_on*/ output [1:0] xs
);
  a1 u1_0 (.q(xs[0]), .v(1'b0), .clk(clk), .d(d), .r(r), .xcr(1'b0));
  a1 u1_1 (.q(xs[1]), .v(1'b0), .clk(clk), .d(d), .r(r), .xcr(1'b0));
endmodule
EOF
chk i1 topI1 big18i_i1.v
chk i2 topI2 big18i_i2.v
