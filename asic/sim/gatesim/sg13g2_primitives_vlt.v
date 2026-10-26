// Behavioral stand-ins for the sg13g2_stdcell UDP primitives defined in
// the PDK's sg13g2_udp.v (Verilator has no UDP support, so this file is
// compiled instead). The PDK cell models in sg13g2_stdcell.v instantiate these
// primitives with positional port lists; the port orders below match
// exactly. Semantics transcribed from the UDP tables in
// /foss/pdks/ihp-sg13g2/libs.ref/sg13g2_stdcell/verilog/sg13g2_udp.v:
//   - dff*      : posedge-clk sampling, async active-high reset/set
//                 (ihp_dff_sr_0 = reset priority, ihp_dff_sr_1 = set priority)
//   - latch*    : transparent while clk high, async reset/set where present
//   - mux2/4/8  : x-tolerant selects (x on the select bit falls back to the
//                 pair of inputs selected by the resolved bits)
// The `v` (notifier) and `xcr` inputs are tied to constants in the cell
// models and are unused here.
//
// Functional gate-level simulation only — no timing, no X-error modeling
// beyond natural X propagation.

`timescale 1ns/10ps

module ihp_latch (q, v, clk, d);
  output reg q;
  input v, clk, d;
  always @(clk or d)
    if (clk) q = d;
endmodule

module ihp_dff_err (q, clk, d);
  output reg q;
  input clk, d;
  always @(posedge clk) q <= d;
endmodule

module ihp_dff (q, v, clk, d, xcr);
  output reg q;
  input v, clk, d, xcr;
  always @(posedge clk) q <= d;
endmodule

module ihp_dff_r_err (q, clk, d, r);
  output reg q;
  input clk, d, r;
  always @(posedge clk or posedge r)
    if (r) q <= 1'b0; else q <= d;
endmodule

module ihp_dff_r (q, v, clk, d, r, xcr);
  output reg q;
  input v, clk, d, r, xcr;
  always @(posedge clk or posedge r)
    if (r) q <= 1'b0; else q <= d;
endmodule

module ihp_dff_s_err (q, clk, d, s);
  output reg q;
  input clk, d, s;
  always @(posedge clk or posedge s)
    if (s) q <= 1'b1; else q <= d;
endmodule

module ihp_dff_s (q, v, clk, d, s, xcr);
  output reg q;
  input v, clk, d, s, xcr;
  always @(posedge clk or posedge s)
    if (s) q <= 1'b1; else q <= d;
endmodule

module ihp_dff_sr_err (q, clk, d, s, r);
  output reg q;
  input clk, d, s, r;
  always @(posedge clk or posedge s or posedge r)
    if (r) q <= 1'b0;
    else if (s) q <= 1'b1;
    else q <= d;
endmodule

module ihp_dff_sr_0 (q, v, clk, d, s, r, xcr);
  output reg q;
  input v, clk, d, s, r, xcr;
  always @(posedge clk or posedge s or posedge r)
    if (r) q <= 1'b0;
    else if (s) q <= 1'b1;
    else q <= d;
endmodule

module ihp_dff_sr_1 (q, v, clk, d, s, r, xcr);
  output reg q;
  input v, clk, d, s, r, xcr;
  always @(posedge clk or posedge s or posedge r)
    if (s) q <= 1'b1;
    else if (r) q <= 1'b0;
    else q <= d;
endmodule

module ihp_latch_r (q, v, clk, d, r);
  output reg q;
  input v, clk, d, r;
  always @(posedge r or clk or d)
    if (r) q = 1'b0;
    else if (clk) q = d;
endmodule

module ihp_latch_s (q, v, clk, d, s);
  output reg q;
  input v, clk, d, s;
  always @(posedge s or clk or d)
    if (s) q = 1'b1;
    else if (clk) q = d;
endmodule

module ihp_latch_sr_0 (q, v, clk, d, s, r);
  output reg q;
  input v, clk, d, s, r;
  always @(posedge s or posedge r or clk or d)
    if (r) q = 1'b0;
    else if (s) q = 1'b1;
    else if (clk) q = d;
endmodule

module ihp_latch_sr_1 (q, v, clk, d, s, r);
  output reg q;
  input v, clk, d, s, r;
  always @(posedge s or posedge r or clk or d)
    if (s) q = 1'b1;
    else if (r) q = 1'b0;
    else if (clk) q = d;
endmodule

module ihp_mux2 (z, a, b, s);
  output z;
  input a, b, s;
  // s=0 -> a, s=1 -> b, s=x -> a (per UDP table)
  assign z = (s == 1'b1) ? b : a;
endmodule

module ihp_mux4 (z, a, b, c, d, s0, s1);
  output z;
  input a, b, c, d, s0, s1;
  // Standard binary select {s1,s0}: 00->a, 01->b, 10->c, 11->d
  // (x-tolerant per the UDP table: x select bits resolve to 0, i.e. the
  // a/b branch)
  assign z = s1 ? (s0 ? d : c) : (s0 ? b : a);
endmodule

module ihp_mux8 (z, a, b, c, d, e, f, g, h, s0, s1, s2);
  output z;
  input a, b, c, d, e, f, g, h, s0, s1, s2;
  // select s2s1s0: 000->a 100->b 010->c 110->d 001->e 101->f 011->g 111->h
  assign z = s2 ? (s1 ? (s0 ? h : g) : (s0 ? f : e))
                         : (s1 ? (s0 ? d : c) : (s0 ? b : a));
endmodule
