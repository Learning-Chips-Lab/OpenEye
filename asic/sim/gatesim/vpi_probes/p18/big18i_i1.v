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
