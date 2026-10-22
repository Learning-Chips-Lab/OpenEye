// Minimal blackbox models for the sg13g2_io pad cells used by the OpenEye
// ASIC top. The PDK's sg13g2_io.v cannot be parsed by OpenSTA's netlist
// reader (specify blocks), so this file is used as EXTRA_VERILOG_MODELS
// instead. Only the pad types instantiated in the design are declared.

module sg13g2_IOPadIn (pad, p2c, vdd, vss, iovdd, iovss);
  input  pad;
  output p2c;
  inout  vdd, vss, iovdd, iovss;
endmodule

module sg13g2_IOPadOut30mA (pad, c2p, vdd, vss, iovdd, iovss);
  output pad;
  input  c2p;
  inout  vdd, vss, iovdd, iovss;
endmodule

module sg13g2_IOPadVdd (vdd, vss, iovdd, iovss);
  inout vdd, vss, iovdd, iovss;
endmodule

module sg13g2_IOPadVss (vdd, vss, iovdd, iovss);
  inout vdd, vss, iovdd, iovss;
endmodule
