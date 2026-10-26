// Minimal test: yosys 0.70 handling of constant-bounded for loops in
// clocked always blocks.
//  - tb1: single always block, module-scope integer
//  - tb2: two always blocks sharing the loop variable
//  - tb3: loop only in reset branch
module tb1 (input clk, input rst_n, input [3:0] d, output reg [3:0] q);
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      q <= 4'h0;
    else begin
      for (i = 0; i < 4; i = i + 1)
        q[i] <= d[i];
    end
  end
endmodule

module tb2 (input clk, input rst_n, input [3:0] d, output reg [3:0] qa, output reg [3:0] qb);
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) qa <= 4'h0;
    else begin
      for (i = 0; i < 4; i = i + 1)
        qa[i] <= d[i];
    end
  end
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) qb <= 4'h0;
    else begin
      for (i = 0; i < 4; i = i + 1)
        qb[i] <= ~d[i];
    end
  end
endmodule

module tb3 (input clk, input rst_n, input [3:0] d, output reg [3:0] q);
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      q <= 4'h0;
      for (i = 0; i < 4; i = i + 1)
        q[i] <= d[i];
    end
  end
endmodule
