"""Check masked RAM writes and legacy unmasked behavior with Icarus."""
import shutil
import subprocess
from pathlib import Path

import pytest

from open_eye import hdl_dir


@pytest.mark.parametrize("masks", [1, 2, 3])
@pytest.mark.parametrize("pipelined", [0, 1])
def test_ram_write_mask(tmp_path, masks, pipelined):
    if not shutil.which("iverilog"):
        pytest.skip("Icarus Verilog is required")
    bench = tmp_path / "tb.v"
    bench.write_text('''
module tb;
  parameter MASKS = 2;
  parameter PIPELINED = 1;
  localparam LANE_BITS = 24 / MASKS;
  reg clk = 0;
  always #5 clk = !clk;
  reg rd = 0, wr = 0;
  reg [1:0] addr = 0;
  reg [23:0] data = 0;
  reg [MASKS-1:0] mask = 0;
  wire [23:0] result;
  reg [23:0] expected;
  integer lane;
  RAM_SP #(.AddrWidth(2), .DataWidth(24), .Pipelined(PIPELINED),
           .WriteMaskWidth(MASKS)) ram (
      .clk_i(clk), .rd_en_i(rd), .wr_en_i(wr), .addr_i(addr),
      .data_i(data), .data_o(result), .wr_mask_i(mask));
  task write;
    input [23:0] value;
    input [MASKS-1:0] lanes;
    begin
      @(negedge clk); wr = 1; rd = 0; data = value; mask = lanes;
      @(negedge clk); wr = 0;
    end
  endtask
  task check;
    input [23:0] value;
    begin
      @(negedge clk); rd = 1;
      repeat (2) @(negedge clk);
      if (result !== value) $fatal(1, "got %h expected %h", result, value);
      rd = 0;
    end
  endtask
  initial begin
    write(24'h123456, {MASKS{1'b1}});
    check(24'h123456);
    // A zero mask suppresses writes except in legacy unmasked mode.
    write(24'habcdef, 0);
    expected = MASKS == 1 ? 24'habcdef : 24'h123456;
    check(expected);
    for (lane = 0; lane < MASKS; lane = lane + 1) begin
      write(24'hfedcba, 1 << lane);
      expected[lane*LANE_BITS+:LANE_BITS] = (24'hfedcba >> (lane*LANE_BITS));
      check(expected);
    end
    // An adjacent address must not disturb the first word.
    addr = 1;
    write(24'h765432, {MASKS{1'b1}});
    check(24'h765432);
    addr = 0;
    check(expected);
    $finish;
  end
endmodule
''')
    executable = tmp_path / "ram.vvp"
    subprocess.run(["iverilog", "-g2012", "-s", "tb",
                    f"-Ptb.MASKS={masks}", f"-Ptb.PIPELINED={pipelined}",
                    "-o", str(executable), str(bench),
                    str(Path(hdl_dir) / "RAM_SP.v"),
                    str(Path(hdl_dir) / "RAM_SP_generic.v")], check=True, capture_output=True)
    subprocess.run(["vvp", str(executable)], check=True, capture_output=True, timeout=10)
