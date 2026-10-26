// Minimal test round 2: nested for, repeated for with same var, params.
module tb4 (input clk, input rst_n, input [3:0] d, output reg [7:0] q);
  parameter N = 2, M = 2;
  integer a, b;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) q <= 8'h0;
    else begin
      for (a = 0; a < N; a = a + 1) begin
        for (b = 0; b < M; b = b + 1) begin
          q[a*2+b] <= d[b];
        end
      end
    end
  end
endmodule

module tb5 (input clk, input rst_n, input [3:0] d, output reg [3:0] qa, output reg [3:0] qb);
  parameter N = 4;
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      qa <= 4'h0;
      qb <= 4'h0;
    end else begin
      for (i = 0; i < N; i = i + 1)
        qa[i] <= d[i];
      for (i = 0; i < N; i = i + 1)
        qb[i] <= ~d[i];
    end
  end
endmodule

module tb6 (input clk, input rst_n, input [3:0] d, input [15:0] cfg,
            output reg [3:0] qa, output reg [3:0] qb, output reg [1:0] cnt);
  parameter N = 2, M = 2;
  integer a, b;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      qa <= 4'h0;
      cnt <= 0;
      for (a = 0; a < N; a = a + 1) begin
        for (b = 0; b < M; b = b + 1)
          qb[a*2+b] <= d[b];
      end
    end else begin
      for (a = 0; a < N; a = a + 1) begin
        for (b = 0; b < M; b = b + 1)
          qb[a*2+b] <= 0;
      end
      if (cfg[0]) begin
        for (a = 0; a < N; a = a + 1) begin
          qa[a] <= cfg[a+4];
          for (b = 0; b < M; b = b + 1)
            qb[a*2+b] <= cfg[a*M+b];
        end
      end
      if (qa[0])
        cnt <= cnt + 1;
    end
  end
endmodule
