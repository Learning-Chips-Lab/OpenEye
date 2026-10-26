// Round 4: transformed tb7/tb8 - conditions pushed INTO the loop body,
// for loops unconditional.
module tb9 (input clk, input rst_n, input en1, input en2, input [3:0] d,
            output reg [3:0] qa, output reg [3:0] qb);
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) qa <= 4'h0;
    else begin
      for (i = 0; i < 4; i = i + 1)
        if (en1)
          qa[i] <= d[i];
    end
  end
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) qb <= 4'h0;
    else begin
      for (i = 0; i < 4; i = i + 1)
        if (en2)
          qb[i] <= ~d[i];
    end
  end
endmodule

module tb10 (input clk, input rst_n, input en1, input en2, input [3:0] d,
             output reg [3:0] qa, output reg [3:0] qb);
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      qa <= 4'h0; qb <= 4'h0;
    end else begin
      for (i = 0; i < 4; i = i + 1)
        if (en1)
          qa[i] <= d[i];
      for (i = 0; i < 4; i = i + 1)
        if (en2)
          qb[i] <= ~d[i];
    end
  end
endmodule
