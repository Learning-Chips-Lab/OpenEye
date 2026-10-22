/*
 * counter -- LibreLane Chip-flow smoke test for IHP SG13G2.
 *
 * 8-bit counter core plus an explicit sg13g2_io pad ring (Caravel-style:
 * pad cells are instantiated in RTL, placed by OpenROAD.PadRing via the
 * PAD_SOUTH/EAST/NORTH/WEST config lists).
 *
 * Pad power pins are wired to the core rails in RTL (single-rail smoke
 * setup: iovdd follows VPWR; the production chip uses a separate 3.3V
 * IOVDD rail). The pad cells are blackboxed via the PDK Verilog model
 * (EXTRA_VERILOG_MODELS), since yosys liberty blackboxes do not expose
 * power pins as connectable ports. Corners/fillers (placed by the flow)
 * get their power via PDN_MACRO_CONNECTIONS.
 */

module counter #(parameter WIDTH = 8) (
  output reg [WIDTH-1:0] out_o,
  input                  clk_i,
  input                  reset_i
);

  always @(posedge clk_i or posedge reset_i) begin
    if (reset_i) out_o <= {WIDTH{1'b0}};
    else         out_o <= out_o + {{WIDTH-1{1'b0}}, 1'b1};
  end

endmodule // counter

module counter_top (
  input  wire        clk_pad,
  input  wire        rst_pad,
  output wire [7:0]  out_pad
);

  wire       clk_i;
  wire       reset_i;
  wire [7:0] out_o;
  wire       VPWR;
  wire       VGND;

  counter #(.WIDTH(8)) u_counter (
    .out_o   (out_o),
    .clk_i   (clk_i),
    .reset_i (reset_i)
  );

  // Input pads
  sg13g2_IOPadIn u_clk (
    .pad   (clk_pad),
    .p2c   (clk_i),
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );

  sg13g2_IOPadIn u_rst (
    .pad   (rst_pad),
    .p2c   (reset_i),
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );

  // Output pads
  sg13g2_IOPadOut30mA u_out0 (
    .c2p   (out_o[0]),
    .pad   (out_pad[0]),
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );
  sg13g2_IOPadOut30mA u_out1 (
    .c2p   (out_o[1]),
    .pad   (out_pad[1]),
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );
  sg13g2_IOPadOut30mA u_out2 (
    .c2p   (out_o[2]),
    .pad   (out_pad[2]),
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );
  sg13g2_IOPadOut30mA u_out3 (
    .c2p   (out_o[3]),
    .pad   (out_pad[3]),
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );
  sg13g2_IOPadOut30mA u_out4 (
    .c2p   (out_o[4]),
    .pad   (out_pad[4]),
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );
  sg13g2_IOPadOut30mA u_out5 (
    .c2p   (out_o[5]),
    .pad   (out_pad[5]),
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );
  sg13g2_IOPadOut30mA u_out6 (
    .c2p   (out_o[6]),
    .pad   (out_pad[6]),
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );
  sg13g2_IOPadOut30mA u_out7 (
    .c2p   (out_o[7]),
    .pad   (out_pad[7]),
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );

  // Power pads (power pins only, bondpads inside the macro).
  // Only iovdd exists in the liberty blackbox model; wiring it to the
  // core rail keeps the instances through synthesis. The remaining power
  // pins are connected via PDN_MACRO_CONNECTIONS in the PadRing step.
  sg13g2_IOPadVdd u_vdd1 (
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );
  sg13g2_IOPadVdd u_vdd2 (
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );
  sg13g2_IOPadVdd u_vdd3 (
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );
  sg13g2_IOPadVdd u_vdd4 (
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );
  sg13g2_IOPadVss u_vss1 (
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );
  sg13g2_IOPadVss u_vss2 (
    .vdd   (VPWR),
    .vss   (VGND),
    .iovdd (VPWR),
    .iovss (VGND)
  );

endmodule // counter_top
