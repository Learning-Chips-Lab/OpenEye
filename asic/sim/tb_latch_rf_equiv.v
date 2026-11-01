`timescale 1ns / 1ps
// Unit test: oe_latch_rf_1w1r (asic/rtl/sg13g2/sg13g2_latch_rf.v) vs the
// generic flop RAMs, wired exactly like hdl/SPad_SP.v and hdl/SPad_DP.v.
// Identical random stimulus per pair; q must match every cycle whenever
// the generic model output is defined (generic RAMs start / read
// unwritten words as X).
//   sp0: 24 x 256 (AW=8)  weight_data_SPad  (SPad_SP, RAM_SP_generic)
//   sp1:  8 x  64 (AW=6)  weight_addr_SPad  (SPad_SP)
//   sp2: 12 x  16 (AW=4)  iact_data_SPad    (SPad_SP)
//   dp0: 20 x  32 (AW=5)  psum_SPad         (SPad_DP, RAM_DP_generic)
// DP stimulus avoids same-cycle read/write of the same word (a collision
// in RAM_DP_generic, $error + X).
//
// Run (container), behavioral cell models:
//   iverilog -g2012 -o /tmp/tb_lrf asic/sim/tb_latch_rf_equiv.v \
//     asic/rtl/sg13g2/sg13g2_latch_rf.v hdl/RAM_SP_generic.v hdl/RAM_DP_generic.v
//   vvp /tmp/tb_lrf
// Real PDK cells: add -DOPENEYE_SG13G2_CELLS and
//   /foss/pdks/ihp-sg13g2/libs.ref/sg13g2_stdcell/verilog/sg13g2_stdcell.v
module tb_latch_rf_equiv;

    localparam CYCLES = 200000;

    reg clk = 0;
    always #5 clk = ~clk;

    // 32-bit LFSR PRNG
    reg [31:0] lfsr = 32'hACE1_2345;
    always @(posedge clk) lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};

    // ---------------- SP pairs ----------------
    reg rd_s0, wr_s0, rd_s1, wr_s1, rd_s2, wr_s2;
    reg [7:0] a_s0; reg [23:0] d_s0;
    reg [5:0] a_s1; reg [ 7:0] d_s1;
    reg [3:0] a_s2; reg [11:0] d_s2;
    wire [23:0] q_s0_gen, q_s0_lrf;
    wire [ 7:0] q_s1_gen, q_s1_lrf;
    wire [11:0] q_s2_gen, q_s2_lrf;

    // RAM_SP mapping: cen = !(rd||wr), rdwen = !wr (write wins)
    RAM_SP_generic #(.AddrWidth(8), .DataWidth(24), .Pipelined(0), .WriteMaskWidth(1)) u_s0_gen
        (.clk(clk), .cen(!(rd_s0||wr_s0)), .rdwen(!wr_s0), .a(a_s0), .d(d_s0), .q(q_s0_gen), .wm(1'b0));
    oe_latch_rf_1w1r #(.AddrWidth(8), .DataWidth(24)) u_s0_lrf
        (.clk_i(clk), .we_i(wr_s0), .waddr_i(a_s0), .wdata_i(d_s0),
         .re_i(rd_s0 && !wr_s0), .raddr_i(a_s0), .rdata_o(q_s0_lrf));

    RAM_SP_generic #(.AddrWidth(6), .DataWidth(8), .Pipelined(0), .WriteMaskWidth(1)) u_s1_gen
        (.clk(clk), .cen(!(rd_s1||wr_s1)), .rdwen(!wr_s1), .a(a_s1), .d(d_s1), .q(q_s1_gen), .wm(1'b0));
    oe_latch_rf_1w1r #(.AddrWidth(6), .DataWidth(8)) u_s1_lrf
        (.clk_i(clk), .we_i(wr_s1), .waddr_i(a_s1), .wdata_i(d_s1),
         .re_i(rd_s1 && !wr_s1), .raddr_i(a_s1), .rdata_o(q_s1_lrf));

    RAM_SP_generic #(.AddrWidth(4), .DataWidth(12), .Pipelined(0), .WriteMaskWidth(1)) u_s2_gen
        (.clk(clk), .cen(!(rd_s2||wr_s2)), .rdwen(!wr_s2), .a(a_s2), .d(d_s2), .q(q_s2_gen), .wm(1'b0));
    oe_latch_rf_1w1r #(.AddrWidth(4), .DataWidth(12)) u_s2_lrf
        (.clk_i(clk), .we_i(wr_s2), .waddr_i(a_s2), .wdata_i(d_s2),
         .re_i(rd_s2 && !wr_s2), .raddr_i(a_s2), .rdata_o(q_s2_lrf));

    // ---------------- DP pair ----------------
    reg rd_d0, wr_d0;
    reg [4:0] ra_d0, wa_d0; reg [19:0] d_d0;
    wire [19:0] q_d0_gen, q_d0_lrf;

    // RAM_DP mapping: cenA = !rd, cenB = !wr
    RAM_DP_generic #(.AddrWidth(5), .DataWidth(20), .Pipelined(0)) u_d0_gen
        (.clkA(clk), .clkB(clk), .cenA(!rd_d0), .cenB(!wr_d0),
         .aA(ra_d0), .aB(wa_d0), .d(d_d0), .q(q_d0_gen));
    oe_latch_rf_1w1r #(.AddrWidth(5), .DataWidth(20)) u_d0_lrf
        (.clk_i(clk), .we_i(wr_d0), .waddr_i(wa_d0), .wdata_i(d_d0),
         .re_i(rd_d0), .raddr_i(ra_d0), .rdata_o(q_d0_lrf));

    integer i;
    integer mismatches, compared;
    initial begin mismatches = 0; compared = 0; end

    // stimulus changes 1 ns after posedge (like flop outputs)
    initial begin
        rd_s0=0; wr_s0=0; a_s0=0; d_s0=0;
        rd_s1=0; wr_s1=0; a_s1=0; d_s1=0;
        rd_s2=0; wr_s2=0; a_s2=0; d_s2=0;
        rd_d0=0; wr_d0=0; ra_d0=0; wa_d0=0; d_d0=0;
        repeat (5) @(posedge clk);
        for (i = 0; i < CYCLES; i = i + 1) begin
            @(posedge clk);
            #1;
            // SP: 3/8 write, 3/8 read, 1/8 write+read (write wins), 1/8 idle
            case (lfsr[2:0])
                3'd0, 3'd1, 3'd2: begin wr_s0=1; rd_s0=0; end
                3'd3, 3'd4, 3'd5: begin wr_s0=0; rd_s0=1; end
                3'd6:             begin wr_s0=1; rd_s0=1; end
                default:          begin wr_s0=0; rd_s0=0; end
            endcase
            {wr_s1, rd_s1} = {wr_s0, rd_s0};
            {wr_s2, rd_s2} = {lfsr[3], lfsr[4] | !lfsr[3]};
            a_s0 = lfsr[12:5];  d_s0 = lfsr[31:8];
            a_s1 = lfsr[18:13]; d_s1 = lfsr[26:19];
            a_s2 = lfsr[22:19]; d_s2 = lfsr[30:19];
            // DP: independent read and write, no same-word collision
            rd_d0 = lfsr[27]; wr_d0 = lfsr[28];
            ra_d0 = lfsr[9:5]; wa_d0 = lfsr[15:11]; d_d0 = lfsr[31:12];
            if (rd_d0 && wr_d0 && ra_d0 == wa_d0) wa_d0 = wa_d0 + 1;
        end
        repeat (10) @(posedge clk);
    end

    // compare in the middle of the low phase (outputs settled)
    task automatic check(input [255:0] name, input [23:0] g, input [23:0] l, input integer w);
        reg [23:0] mask;
        begin
            mask = (24'hFFFFFF >> (24 - w));
            if (^(g & mask) !== 1'bx) begin
                compared = compared + 1;
                if ((g & mask) !== (l & mask)) begin
                    mismatches = mismatches + 1;
                    if (mismatches <= 20)
                        $display("MISMATCH %0s @%0t gen=%h lrf=%h", name, $time, g & mask, l & mask);
                end
            end
        end
    endtask

    always @(negedge clk) begin
        #2;
        check("sp0", q_s0_gen, q_s0_lrf, 24);
        check("sp1", {16'b0, q_s1_gen}, {16'b0, q_s1_lrf}, 8);
        check("sp2", {12'b0, q_s2_gen}, {12'b0, q_s2_lrf}, 12);
        check("dp0", {4'b0, q_d0_gen}, {4'b0, q_d0_lrf}, 20);
        if (mismatches > 20) begin
            $display("FAIL: too many mismatches");
            $finish;
        end
    end

    initial begin
        wait (i >= CYCLES);
        #100;
        if (mismatches == 0 && compared > CYCLES)
            $display("PASS: %0d cycles, %0d defined comparisons, latch RF == generic RAM always",
                     CYCLES, compared);
        else
            $display("FAIL: %0d mismatches, %0d comparisons", mismatches, compared);
        $finish;
    end

endmodule
