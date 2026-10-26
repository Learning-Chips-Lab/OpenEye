// Round 3: conditional for loops (inside if branches).
module tb7 (input clk, input rst_n, input en1, input en2, input [3:0] d,
            output reg [3:0] qa, output reg [3:0] qb);
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) qa <= 4'h0;
    else if (en1) begin
      for (i = 0; i < 4; i = i + 1)
        qa[i] <= d[i];
    end
  end
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) qb <= 4'h0;
    else if (en2) begin
      for (i = 0; i < 4; i = i + 1)
        qb[i] <= ~d[i];
    end
  end
endmodule

// same variable, two conditional loops in ONE block
module tb8 (input clk, input rst_n, input en1, input en2, input [3:0] d,
            output reg [3:0] qa, output reg [3:0] qb);
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      qa <= 4'h0; qb <= 4'h0;
    end else begin
      if (en1)
        for (i = 0; i < 4; i = i + 1)
          qa[i] <= d[i];
      if (en2)
        for (i = 0; i < 4; i = i + 1)
          qb[i] <= ~d[i];
    end
  end
endmodule
