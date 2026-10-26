#!/bin/bash
# P18c: name vs file as trigger of modPublic.
#   T1: module named ihp_dff_r in the SAME file as top, direct inst
#   T2: module named a1 in its OWN file, direct inst in top
#   T3: module named ihp_dff_r in its OWN file, direct inst in top
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# T1: ihp_dff_r in same file as top
cat > big18c_t1.v <<'EOF'
module ihp_dff_r (q, v, clk, d, r, xcr);
  output reg q;
  input v, clk, d, r, xcr;
  always @(posedge clk or posedge r)
    if (r) q <= 1'b0; else q <= d;
endmodule
module topT1 (
  /*verilator public_on*/ input clk,
  /*verilator public_on*/ input d,
  /*verilator public_on*/ input r,
  /*verilator public_on*/ output [9:0] xs
);
  ihp_dff_r u1_0 (.q(xs[0]), .v(1'b0), .clk(clk), .d(d), .r(r), .xcr(1'b0));
  ihp_dff_r u1_1 (.q(xs[1]), .v(1'b0), .clk(clk), .d(d), .r(r), .xcr(1'b0));
endmodule
EOF

# T2: module a1 in own file
cat > big18c_a1mod.v <<'EOF'
module a1 (q, v, clk, d, r, xcr);
  output reg q;
  input v, clk, d, r, xcr;
  always @(posedge clk or posedge r)
    if (r) q <= 1'b0; else q <= d;
endmodule
EOF
cat > big18c_t2top.v <<'EOF'
module topT2 (
  /*verilator public_on*/ input clk,
  /*verilator public_on*/ input d,
  /*verilator public_on*/ input r,
  /*verilator public_on*/ output [9:0] xs
);
  a1 u1_0 (.q(xs[0]), .v(1'b0), .clk(clk), .d(d), .r(r), .xcr(1'b0));
  a1 u1_1 (.q(xs[1]), .v(1'b0), .clk(clk), .d(d), .r(r), .xcr(1'b0));
endmodule
EOF

# T3: module ihp_dff_r in own file
cat > big18c_t3top.v <<'EOF'
module topT3 (
  /*verilator public_on*/ input clk,
  /*verilator public_on*/ input d,
  /*verilator public_on*/ input r,
  /*verilator public_on*/ output [9:0] xs
);
  ihp_dff_r u1_0 (.q(xs[0]), .v(1'b0), .clk(clk), .d(d), .r(r), .xcr(1'b0));
  ihp_dff_r u1_1 (.q(xs[1]), .v(1'b0), .clk(clk), .d(d), .r(r), .xcr(1'b0));
endmodule
EOF

for T in "T1 topT1 big18c_t1.v" "T2 topT2 big18c_t2top.v big18c_a1mod.v" "T3 topT3 big18c_t3top.v big18b_mod.v"; do
  set -- $T
  N=$1; TOP=$2; shift 2
  rm -rf m18c_$N
  verilator -cc --top-module $TOP --timescale 1ns/10ps --timing -Wno-fatal \
    --flatten --dumpi-graph 6 -Mdir m18c_$N "$@" > m18c_$N.log 2>&1
  echo "########## $N ($@):"
  python3 dump_inline_why.py m18c_$N/*inlinemod-decision.dot 2>/dev/null | \
    awk '/^MODULE/{m=1;next} m&&/^$/{m=0} m' | grep -E "^[a-z]|NoInlineSoft" 
done
