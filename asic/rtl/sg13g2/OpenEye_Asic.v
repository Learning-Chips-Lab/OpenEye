// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

`timescale 1ns / 1ps

/// Module: OpenEye_Asic
///
/// ASIC chip top for IHP SG13G2: OpenEye_FPGA compute core plus an explicit
/// sg13g2_io pad ring (Caravel-style, placed by OpenROAD.PadRing via the
/// PAD_SOUTH/EAST/NORTH/WEST LibreLane config lists).
///
/// GENERATED FILE -- do not edit by hand. Regenerate with:
///   python3 asic/scripts/gen_top.py
///
/// Pad map (135 signal pads, all non-inverting, 3.3V-tolerant IO):
///   inputs  (69): clk, rst_n (active low),
///                 data_dma_i[63:0], enable_dma_i, ready_dma_i
///   outputs (66): data_dma_o[63:0], ready_dma_o, enable_dma_o, last_data_o
///
/// Power (single-rail bring-up setup; the production board uses a separate
/// 3.3V IOVDD rail for the pad ring):
///   vdd/vss     = VPWR/VGND (1.2V core)
///   iovdd/iovss = tied to the core rails here

module OpenEye_Asic (
    input  wire        clk_pad,
    input  wire        rst_n_pad,
    input  wire [63:0] data_dma_i_pad,
    input  wire        enable_dma_i_pad,
    input  wire        ready_dma_i_pad,
    output wire        ready_dma_o_pad,
    output wire [63:0] data_dma_o_pad,
    output wire        enable_dma_o_pad,
    output wire        last_data_o_pad
);

    // ------------------------------------------------------------------
    // core interface wires
    // ------------------------------------------------------------------
    wire        clk_i;
    wire        rst_ni;
    wire [63:0] data_dma_i;
    wire        enable_dma_i;
    wire        ready_dma_i;
    wire        ready_dma_o;
    wire [63:0] data_dma_o;
    wire        enable_dma_o;
    wire        last_data_o;

    // single-rail power (see header)
    wire VPWR;
    wire VGND;

    // ------------------------------------------------------------------
    // compute core
    // ------------------------------------------------------------------
    OpenEye_FPGA u_openeye (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .ready_dma_o  (ready_dma_o),
        .data_dma_i   (data_dma_i),
        .enable_dma_i (enable_dma_i),
        .ready_dma_i  (ready_dma_i),
        .data_dma_o   (data_dma_o),
        .enable_dma_o (enable_dma_o),
        .last_data_o  (last_data_o)
    );

    // ------------------------------------------------------------------
    // input pads
    // ------------------------------------------------------------------
    sg13g2_IOPadIn u_clk_in (
        .pad(clk_pad), .p2c(clk_i),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_rstn_in (
        .pad(rst_n_pad), .p2c(rst_ni),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_eni_in (
        .pad(enable_dma_i_pad), .p2c(enable_dma_i),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_rdyi_in (
        .pad(ready_dma_i_pad), .p2c(ready_dma_i),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in0 (
        .pad(data_dma_i_pad[0]), .p2c(data_dma_i[0]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in1 (
        .pad(data_dma_i_pad[1]), .p2c(data_dma_i[1]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in2 (
        .pad(data_dma_i_pad[2]), .p2c(data_dma_i[2]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in3 (
        .pad(data_dma_i_pad[3]), .p2c(data_dma_i[3]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in4 (
        .pad(data_dma_i_pad[4]), .p2c(data_dma_i[4]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in5 (
        .pad(data_dma_i_pad[5]), .p2c(data_dma_i[5]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in6 (
        .pad(data_dma_i_pad[6]), .p2c(data_dma_i[6]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in7 (
        .pad(data_dma_i_pad[7]), .p2c(data_dma_i[7]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in8 (
        .pad(data_dma_i_pad[8]), .p2c(data_dma_i[8]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in9 (
        .pad(data_dma_i_pad[9]), .p2c(data_dma_i[9]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in10 (
        .pad(data_dma_i_pad[10]), .p2c(data_dma_i[10]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in11 (
        .pad(data_dma_i_pad[11]), .p2c(data_dma_i[11]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in12 (
        .pad(data_dma_i_pad[12]), .p2c(data_dma_i[12]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in13 (
        .pad(data_dma_i_pad[13]), .p2c(data_dma_i[13]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in14 (
        .pad(data_dma_i_pad[14]), .p2c(data_dma_i[14]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in15 (
        .pad(data_dma_i_pad[15]), .p2c(data_dma_i[15]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in16 (
        .pad(data_dma_i_pad[16]), .p2c(data_dma_i[16]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in17 (
        .pad(data_dma_i_pad[17]), .p2c(data_dma_i[17]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in18 (
        .pad(data_dma_i_pad[18]), .p2c(data_dma_i[18]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in19 (
        .pad(data_dma_i_pad[19]), .p2c(data_dma_i[19]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in20 (
        .pad(data_dma_i_pad[20]), .p2c(data_dma_i[20]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in21 (
        .pad(data_dma_i_pad[21]), .p2c(data_dma_i[21]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in22 (
        .pad(data_dma_i_pad[22]), .p2c(data_dma_i[22]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in23 (
        .pad(data_dma_i_pad[23]), .p2c(data_dma_i[23]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in24 (
        .pad(data_dma_i_pad[24]), .p2c(data_dma_i[24]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in25 (
        .pad(data_dma_i_pad[25]), .p2c(data_dma_i[25]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in26 (
        .pad(data_dma_i_pad[26]), .p2c(data_dma_i[26]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in27 (
        .pad(data_dma_i_pad[27]), .p2c(data_dma_i[27]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in28 (
        .pad(data_dma_i_pad[28]), .p2c(data_dma_i[28]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in29 (
        .pad(data_dma_i_pad[29]), .p2c(data_dma_i[29]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in30 (
        .pad(data_dma_i_pad[30]), .p2c(data_dma_i[30]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in31 (
        .pad(data_dma_i_pad[31]), .p2c(data_dma_i[31]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in32 (
        .pad(data_dma_i_pad[32]), .p2c(data_dma_i[32]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in33 (
        .pad(data_dma_i_pad[33]), .p2c(data_dma_i[33]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in34 (
        .pad(data_dma_i_pad[34]), .p2c(data_dma_i[34]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in35 (
        .pad(data_dma_i_pad[35]), .p2c(data_dma_i[35]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in36 (
        .pad(data_dma_i_pad[36]), .p2c(data_dma_i[36]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in37 (
        .pad(data_dma_i_pad[37]), .p2c(data_dma_i[37]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in38 (
        .pad(data_dma_i_pad[38]), .p2c(data_dma_i[38]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in39 (
        .pad(data_dma_i_pad[39]), .p2c(data_dma_i[39]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in40 (
        .pad(data_dma_i_pad[40]), .p2c(data_dma_i[40]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in41 (
        .pad(data_dma_i_pad[41]), .p2c(data_dma_i[41]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in42 (
        .pad(data_dma_i_pad[42]), .p2c(data_dma_i[42]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in43 (
        .pad(data_dma_i_pad[43]), .p2c(data_dma_i[43]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in44 (
        .pad(data_dma_i_pad[44]), .p2c(data_dma_i[44]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in45 (
        .pad(data_dma_i_pad[45]), .p2c(data_dma_i[45]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in46 (
        .pad(data_dma_i_pad[46]), .p2c(data_dma_i[46]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in47 (
        .pad(data_dma_i_pad[47]), .p2c(data_dma_i[47]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in48 (
        .pad(data_dma_i_pad[48]), .p2c(data_dma_i[48]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in49 (
        .pad(data_dma_i_pad[49]), .p2c(data_dma_i[49]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in50 (
        .pad(data_dma_i_pad[50]), .p2c(data_dma_i[50]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in51 (
        .pad(data_dma_i_pad[51]), .p2c(data_dma_i[51]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in52 (
        .pad(data_dma_i_pad[52]), .p2c(data_dma_i[52]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in53 (
        .pad(data_dma_i_pad[53]), .p2c(data_dma_i[53]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in54 (
        .pad(data_dma_i_pad[54]), .p2c(data_dma_i[54]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in55 (
        .pad(data_dma_i_pad[55]), .p2c(data_dma_i[55]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in56 (
        .pad(data_dma_i_pad[56]), .p2c(data_dma_i[56]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in57 (
        .pad(data_dma_i_pad[57]), .p2c(data_dma_i[57]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in58 (
        .pad(data_dma_i_pad[58]), .p2c(data_dma_i[58]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in59 (
        .pad(data_dma_i_pad[59]), .p2c(data_dma_i[59]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in60 (
        .pad(data_dma_i_pad[60]), .p2c(data_dma_i[60]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in61 (
        .pad(data_dma_i_pad[61]), .p2c(data_dma_i[61]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in62 (
        .pad(data_dma_i_pad[62]), .p2c(data_dma_i[62]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadIn u_dma_i_in63 (
        .pad(data_dma_i_pad[63]), .p2c(data_dma_i[63]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );

    // ------------------------------------------------------------------
    // output pads
    // ------------------------------------------------------------------
    sg13g2_IOPadOut30mA u_rdyo_out (
        .c2p(ready_dma_o), .pad(ready_dma_o_pad),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_eno_out (
        .c2p(enable_dma_o), .pad(enable_dma_o_pad),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_last_out (
        .c2p(last_data_o), .pad(last_data_o_pad),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out0 (
        .c2p(data_dma_o[0]), .pad(data_dma_o_pad[0]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out1 (
        .c2p(data_dma_o[1]), .pad(data_dma_o_pad[1]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out2 (
        .c2p(data_dma_o[2]), .pad(data_dma_o_pad[2]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out3 (
        .c2p(data_dma_o[3]), .pad(data_dma_o_pad[3]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out4 (
        .c2p(data_dma_o[4]), .pad(data_dma_o_pad[4]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out5 (
        .c2p(data_dma_o[5]), .pad(data_dma_o_pad[5]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out6 (
        .c2p(data_dma_o[6]), .pad(data_dma_o_pad[6]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out7 (
        .c2p(data_dma_o[7]), .pad(data_dma_o_pad[7]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out8 (
        .c2p(data_dma_o[8]), .pad(data_dma_o_pad[8]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out9 (
        .c2p(data_dma_o[9]), .pad(data_dma_o_pad[9]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out10 (
        .c2p(data_dma_o[10]), .pad(data_dma_o_pad[10]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out11 (
        .c2p(data_dma_o[11]), .pad(data_dma_o_pad[11]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out12 (
        .c2p(data_dma_o[12]), .pad(data_dma_o_pad[12]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out13 (
        .c2p(data_dma_o[13]), .pad(data_dma_o_pad[13]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out14 (
        .c2p(data_dma_o[14]), .pad(data_dma_o_pad[14]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out15 (
        .c2p(data_dma_o[15]), .pad(data_dma_o_pad[15]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out16 (
        .c2p(data_dma_o[16]), .pad(data_dma_o_pad[16]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out17 (
        .c2p(data_dma_o[17]), .pad(data_dma_o_pad[17]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out18 (
        .c2p(data_dma_o[18]), .pad(data_dma_o_pad[18]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out19 (
        .c2p(data_dma_o[19]), .pad(data_dma_o_pad[19]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out20 (
        .c2p(data_dma_o[20]), .pad(data_dma_o_pad[20]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out21 (
        .c2p(data_dma_o[21]), .pad(data_dma_o_pad[21]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out22 (
        .c2p(data_dma_o[22]), .pad(data_dma_o_pad[22]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out23 (
        .c2p(data_dma_o[23]), .pad(data_dma_o_pad[23]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out24 (
        .c2p(data_dma_o[24]), .pad(data_dma_o_pad[24]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out25 (
        .c2p(data_dma_o[25]), .pad(data_dma_o_pad[25]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out26 (
        .c2p(data_dma_o[26]), .pad(data_dma_o_pad[26]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out27 (
        .c2p(data_dma_o[27]), .pad(data_dma_o_pad[27]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out28 (
        .c2p(data_dma_o[28]), .pad(data_dma_o_pad[28]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out29 (
        .c2p(data_dma_o[29]), .pad(data_dma_o_pad[29]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out30 (
        .c2p(data_dma_o[30]), .pad(data_dma_o_pad[30]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out31 (
        .c2p(data_dma_o[31]), .pad(data_dma_o_pad[31]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out32 (
        .c2p(data_dma_o[32]), .pad(data_dma_o_pad[32]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out33 (
        .c2p(data_dma_o[33]), .pad(data_dma_o_pad[33]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out34 (
        .c2p(data_dma_o[34]), .pad(data_dma_o_pad[34]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out35 (
        .c2p(data_dma_o[35]), .pad(data_dma_o_pad[35]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out36 (
        .c2p(data_dma_o[36]), .pad(data_dma_o_pad[36]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out37 (
        .c2p(data_dma_o[37]), .pad(data_dma_o_pad[37]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out38 (
        .c2p(data_dma_o[38]), .pad(data_dma_o_pad[38]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out39 (
        .c2p(data_dma_o[39]), .pad(data_dma_o_pad[39]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out40 (
        .c2p(data_dma_o[40]), .pad(data_dma_o_pad[40]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out41 (
        .c2p(data_dma_o[41]), .pad(data_dma_o_pad[41]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out42 (
        .c2p(data_dma_o[42]), .pad(data_dma_o_pad[42]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out43 (
        .c2p(data_dma_o[43]), .pad(data_dma_o_pad[43]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out44 (
        .c2p(data_dma_o[44]), .pad(data_dma_o_pad[44]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out45 (
        .c2p(data_dma_o[45]), .pad(data_dma_o_pad[45]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out46 (
        .c2p(data_dma_o[46]), .pad(data_dma_o_pad[46]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out47 (
        .c2p(data_dma_o[47]), .pad(data_dma_o_pad[47]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out48 (
        .c2p(data_dma_o[48]), .pad(data_dma_o_pad[48]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out49 (
        .c2p(data_dma_o[49]), .pad(data_dma_o_pad[49]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out50 (
        .c2p(data_dma_o[50]), .pad(data_dma_o_pad[50]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out51 (
        .c2p(data_dma_o[51]), .pad(data_dma_o_pad[51]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out52 (
        .c2p(data_dma_o[52]), .pad(data_dma_o_pad[52]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out53 (
        .c2p(data_dma_o[53]), .pad(data_dma_o_pad[53]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out54 (
        .c2p(data_dma_o[54]), .pad(data_dma_o_pad[54]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out55 (
        .c2p(data_dma_o[55]), .pad(data_dma_o_pad[55]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out56 (
        .c2p(data_dma_o[56]), .pad(data_dma_o_pad[56]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out57 (
        .c2p(data_dma_o[57]), .pad(data_dma_o_pad[57]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out58 (
        .c2p(data_dma_o[58]), .pad(data_dma_o_pad[58]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out59 (
        .c2p(data_dma_o[59]), .pad(data_dma_o_pad[59]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out60 (
        .c2p(data_dma_o[60]), .pad(data_dma_o_pad[60]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out61 (
        .c2p(data_dma_o[61]), .pad(data_dma_o_pad[61]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out62 (
        .c2p(data_dma_o[62]), .pad(data_dma_o_pad[62]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
    sg13g2_IOPadOut30mA u_dma_o_out63 (
        .c2p(data_dma_o[63]), .pad(data_dma_o_pad[63]),
        .vdd(VPWR), .vss(VGND), .iovdd(VPWR), .iovss(VGND)
    );
endmodule
