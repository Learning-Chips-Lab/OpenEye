`timescale 1ns / 1ps
// Unit test: RAM_SP_generic vs sg13g2_ram_sp (cfg 72x1024, WM=6).
// Identical stimulus on both; q must match every cycle.
module tb_ram_equiv;

    localparam DW = 72, AW = 10, WM = 6, CYCLES = 200000;

    reg clk = 0;
    always #5 clk = ~clk;

    reg         rd, wr;
    reg  [AW-1:0]  a;
    reg  [DW-1:0]  d;
    reg  [WM-1:0]  wm;

    wire [DW-1:0] q_gen, q_mac;

    RAM_SP_generic #(
        .AddrWidth(AW), .DataWidth(DW), .Pipelined(1), .WriteMaskWidth(WM)
    ) u_gen (
        .clk(clk), .cen(!(rd | wr)), .rdwen(!wr), .a(a), .d(d), .q(q_gen), .wm(wm)
    );

    sg13g2_ram_sp #(
        .AddrWidth(AW), .DataWidth(DW), .Pipelined(1), .WriteMaskWidth(WM)
    ) u_mac (
        .clk(clk), .cen(!(rd | wr)), .rdwen(!wr), .a(a), .d(d), .q(q_mac), .wm(wm)
    );

    // 16-bit LFSR PRNG
    reg [15:0] lfsr = 16'hACE1;
    always @(posedge clk) lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};

    integer i;
    integer mismatches;
    initial mismatches = 0;

    initial begin
        rd = 0; wr = 0; a = 0; d = 0; wm = 0;
        repeat (5) @(posedge clk);
        for (i = 0; i < CYCLES; i = i + 1) begin
            @(posedge clk);
            #1;
            case (lfsr[1:0])
                2'b00, 2'b01: begin wr = 1; rd = 0; a = lfsr[12:3]; d = {lfsr, d}; wm = lfsr[15:10]; end
                default:       begin wr = 0; rd = 1; a = lfsr[AW+2:3]; end
            endcase
        end
        repeat (10) @(posedge clk);
    end

    always @(posedge clk) begin
        if (mismatches < 20) begin
            if (q_gen !== q_mac) begin
                mismatches = mismatches + 1;
                $display("MISMATCH @%0t cycle q_gen=%b q_mac=%b", $time, q_gen, q_mac);
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
                #(5000000);
                $display("FAIL: watchdog timeout");
                $finish;
            end
            begin
                wait (i >= CYCLES);
                #100;
                if (mismatches == 0) begin
                    $display("PASS: %0d cycles, q_gen == q_mac always", CYCLES);
                end else begin
                    $display("FAIL: %0d mismatches", mismatches);
                end
                $finish;
            end
        join
    end

endmodule
