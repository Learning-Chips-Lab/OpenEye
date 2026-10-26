module topT2 (
  /*verilator public_on*/ input clk,
  /*verilator public_on*/ input d,
  /*verilator public_on*/ input r,
  /*verilator public_on*/ output [9:0] xs
);
  a1 u1_0 (.q(xs[0]), .v(1'b0), .clk(clk), .d(d), .r(r), .xcr(1'b0));
  a1 u1_1 (.q(xs[1]), .v(1'b0), .clk(clk), .d(d), .r(r), .xcr(1'b0));
endmodule
