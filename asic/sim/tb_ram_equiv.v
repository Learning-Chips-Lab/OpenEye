`timescale 1ns / 1ps
// Unit test: RAM_SP_generic vs sg13g2_ram_sp for all five ASIC-target
// configurations. Identical stimulus per config; q must match every cycle.
//   cfg0:  64 x 1024 (WM=1)  BUFFER_A
//   cfg1: 288 x 2048 (WM=1)  wght_buffer
//   cfg2:  40 x 4096 (WM=1)  psum_buffer
//   cfg3:  72 x 1024 (WM=6)  iact_stream_constructor (NUM_GLB_IACT=3)
//   cfg4:  24 x 1024 (WM=2)  iact_stream_constructor (NUM_GLB_IACT=1)
module tb_ram_equiv;

    localparam CYCLES = 200000;

    reg clk = 0;
    always #5 clk = ~clk;

    // 16-bit LFSR PRNG
    reg [15:0] lfsr = 16'hACE1;
    always @(posedge clk) lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};

    // stimulus registers (independent per config)
    reg  [9:0]  a0;  reg [63:0] d0;
    reg  [10:0] a1;  reg [287:0] d1;
    reg  [11:0] a2;  reg [39:0] d2;
    reg  [9:0]  a3;  reg [71:0] d3;  reg [5:0] wm3;
    reg  [9:0]  a4;  reg [23:0] d4;  reg [1:0] wm4;
    reg rd0, wr0, rd1, wr1, rd2, wr2, rd3, wr3, rd4, wr4;

    wire [63:0]  q0_gen, q0_mac;
    wire [287:0] q1_gen, q1_mac;
    wire [39:0]  q2_gen, q2_mac;
    wire [71:0]  q3_gen, q3_mac;
    wire [23:0]  q4_gen, q4_mac;

    RAM_SP_generic #(.AddrWidth(10), .DataWidth(64),  .Pipelined(1), .WriteMaskWidth(1)) u_gen0
        (.clk(clk), .cen(!(rd0|wr0)), .rdwen(!wr0), .a(a0), .d(d0), .q(q0_gen), .wm(1'b1));
    sg13g2_ram_sp  #(.AddrWidth(10), .DataWidth(64),  .Pipelined(1), .WriteMaskWidth(1)) u_mac0
        (.clk(clk), .cen(!(rd0|wr0)), .rdwen(!wr0), .a(a0), .d(d0), .q(q0_mac), .wm(1'b1));

    RAM_SP_generic #(.AddrWidth(11), .DataWidth(288), .Pipelined(1), .WriteMaskWidth(1)) u_gen1
        (.clk(clk), .cen(!(rd1|wr1)), .rdwen(!wr1), .a(a1), .d(d1), .q(q1_gen), .wm(1'b1));
    sg13g2_ram_sp  #(.AddrWidth(11), .DataWidth(288), .Pipelined(1), .WriteMaskWidth(1)) u_mac1
        (.clk(clk), .cen(!(rd1|wr1)), .rdwen(!wr1), .a(a1), .d(d1), .q(q1_mac), .wm(1'b1));

    RAM_SP_generic #(.AddrWidth(12), .DataWidth(40),  .Pipelined(1), .WriteMaskWidth(1)) u_gen2
        (.clk(clk), .cen(!(rd2|wr2)), .rdwen(!wr2), .a(a2), .d(d2), .q(q2_gen), .wm(1'b1));
    sg13g2_ram_sp  #(.AddrWidth(12), .DataWidth(40),  .Pipelined(1), .WriteMaskWidth(1)) u_mac2
        (.clk(clk), .cen(!(rd2|wr2)), .rdwen(!wr2), .a(a2), .d(d2), .q(q2_mac), .wm(1'b1));

    RAM_SP_generic #(.AddrWidth(10), .DataWidth(72),  .Pipelined(1), .WriteMaskWidth(6)) u_gen3
        (.clk(clk), .cen(!(rd3|wr3)), .rdwen(!wr3), .a(a3), .d(d3), .q(q3_gen), .wm(wm3));
    sg13g2_ram_sp  #(.AddrWidth(10), .DataWidth(72),  .Pipelined(1), .WriteMaskWidth(6)) u_mac3
        (.clk(clk), .cen(!(rd3|wr3)), .rdwen(!wr3), .a(a3), .d(d3), .q(q3_mac), .wm(wm3));

    RAM_SP_generic #(.AddrWidth(10), .DataWidth(24),  .Pipelined(1), .WriteMaskWidth(2)) u_gen4
        (.clk(clk), .cen(!(rd4|wr4)), .rdwen(!wr4), .a(a4), .d(d4), .q(q4_gen), .wm(wm4));
    sg13g2_ram_sp  #(.AddrWidth(10), .DataWidth(24),  .Pipelined(1), .WriteMaskWidth(2)) u_mac4
        (.clk(clk), .cen(!(rd4|wr4)), .rdwen(!wr4), .a(a4), .d(d4), .q(q4_mac), .wm(wm4));

    integer i;
    integer mismatches;
    initial mismatches = 0;

    initial begin
        rd0=0; wr0=0; a0=0; d0=0;
        rd1=0; wr1=0; a1=0; d1=0;
        rd2=0; wr2=0; a2=0; d2=0;
        rd3=0; wr3=0; a3=0; d3=0; wm3=0;
        rd4=0; wr4=0; a4=0; d4=0; wm4=0;
        repeat (5) @(posedge clk);
        for (i = 0; i < CYCLES; i = i + 1) begin
            @(posedge clk);
            #1;
            case (lfsr[1:0])
                2'b00, 2'b01: begin
                    wr0=1; rd0=0; a0=lfsr[12:3];  d0={lfsr,d0};
                    wr1=1; rd1=0; a1=lfsr[13:3];  d1={lfsr,d1};
                    wr2=1; rd2=0; a2=lfsr[14:3];  d2={lfsr,d2};
                    wr3=1; rd3=0; a3=lfsr[12:3];  d3={lfsr,d3}; wm3=lfsr[15:10];
                    wr4=1; rd4=0; a4=lfsr[12:3];  d4={lfsr,d4}; wm4=lfsr[15:14];
                end
                default: begin
                    wr0=0; rd0=1; a0=lfsr[12:3];
                    wr1=0; rd1=1; a1=lfsr[13:3];
                    wr2=0; rd2=1; a2=lfsr[14:3];
                    wr3=0; rd3=1; a3=lfsr[12:3];
                    wr4=0; rd4=1; a4=lfsr[12:3];
                end
            endcase
        end
        repeat (10) @(posedge clk);
    end

    always @(posedge clk) begin
        if (mismatches < 20) begin
            if (q0_gen !== q0_mac || q1_gen !== q1_mac ||
                q2_gen !== q2_mac || q3_gen !== q3_mac || q4_gen !== q4_mac) begin
                mismatches = mismatches + 1;
                if (q0_gen !== q0_mac) $display("MISMATCH cfg0 @%0t gen=%b mac=%b", $time, q0_gen, q0_mac);
                if (q1_gen !== q1_mac) $display("MISMATCH cfg1 @%0t gen=%b mac=%b", $time, q1_gen, q1_mac);
                if (q2_gen !== q2_mac) $display("MISMATCH cfg2 @%0t gen=%b mac=%b", $time, q2_gen, q2_mac);
                if (q3_gen !== q3_mac) $display("MISMATCH cfg3 @%0t gen=%b mac=%b", $time, q3_gen, q3_mac);
                if (q4_gen !== q4_mac) $display("MISMATCH cfg4 @%0t gen=%b mac=%b", $time, q4_gen, q4_mac);
                if (mismatches >= 20) begin
                    $display("FAIL: too many mismatches");
                    $finish;
                end
            end
        end
    end

    initial begin
        fork
            begin : watchdog
                #(20000000);
                $display("FAIL: watchdog timeout");
                $finish;
            end
            begin
                wait (i >= CYCLES);
                #100;
                if (mismatches == 0)
                    $display("PASS: %0d cycles, all 5 configs q_gen == q_mac always", CYCLES);
                else
                    $display("FAIL: %0d mismatch cycles", mismatches);
                $finish;
            end
        join
    end

endmodule
