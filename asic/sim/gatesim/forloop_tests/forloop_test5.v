// Round 5: bisect - which body construct prevents unrolling?
// tb11: one for, body = if (single stmt)
module tb11 (input clk, input rst_n, input en1, input [3:0] d, output reg [3:0] q);
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) q <= 4'h0;
    else begin
      for (i = 0; i < 4; i = i + 1)
        if (en1)
          q[i] <= d[i];
    end
  end
endmodule

// tb12: one for, body = if with begin/end
module tb12 (input clk, input rst_n, input en1, input [3:0] d, output reg [3:0] q);
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) q <= 4'h0;
    else begin
      for (i = 0; i < 4; i = i + 1) begin
        if (en1) begin
          q[i] <= d[i];
        end
      end
    end
  end
endmodule

// tb13: one for, unconditional body but a second non-loop if after it
module tb13 (input clk, input rst_n, input en1, input [3:0] d, output reg [3:0] q);
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) q <= 4'h0;
    else begin
      for (i = 0; i < 4; i = i + 1)
        q[i] <= d[i];
      if (en1)
        q[0] <= q[1];
    end
  end
endmodule

// tb14: for with begin/end body, unconditional single statement
module tb14 (input clk, input rst_n, input [3:0] d, output reg [3:0] q);
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) q <= 4'h0;
    else begin
      for (i = 0; i < 4; i = i + 1) begin
        q[i] <= d[i];
      end
    end
  end
endmodule
