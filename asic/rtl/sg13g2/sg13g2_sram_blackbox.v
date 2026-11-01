// This file is part of the OpenEye project.
// © Fachhochschule Dortmund – University of Applied Sciences and Arts (until 2025), Universität Duisburg-Essen (since 2025).
// SPDX-License-Identifier: SHL-2.1
// For more details, see the LICENSE file in the root directory of this project.

// Clean blackbox models for the RM_IHPSG13_1P SRAM macros used by
// sg13g2_ram_sp.v. The PDK liberty files cannot be read by Yosys in
// non-`-lib` mode ("bus interfaces are only supported in -lib mode"), so
// the lint step would not see the macro modules. These models provide the
// exact signal port list (no power pins: vdd/vss/vddarray/vssarray are
// declared in the LEF/LIB and handled by OpenROAD) for both the Verilator
// lint step and the Yosys synthesis blackboxing (EXTRA_VERILOG_MODELS).
//
// Functional models for simulation: the PDK verilog files under
// libs.ref/sg13g2_sram/verilog/ with -DFUNCTIONAL (see
// asic/scripts/regress_macros.sh).

module RM_IHPSG13_1P_1024x64_c2_bm_bist (
    A_CLK, A_MEN, A_WEN, A_REN,
    A_ADDR, A_DIN, A_DLY, A_DOUT, A_BM,
    A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN,
    A_BIST_ADDR, A_BIST_DIN, A_BIST_BM
);
    input A_CLK, A_MEN, A_WEN, A_REN, A_DLY;
    input [9:0] A_ADDR;
    input [63:0] A_DIN;
    output [63:0] A_DOUT;
    input [63:0] A_BM;
    input A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN;
    input [9:0] A_BIST_ADDR;
    input [63:0] A_BIST_DIN, A_BIST_BM;
endmodule

module RM_IHPSG13_1P_1024x16_c2_bm_bist (
    A_CLK, A_MEN, A_WEN, A_REN,
    A_ADDR, A_DIN, A_DLY, A_DOUT, A_BM,
    A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN,
    A_BIST_ADDR, A_BIST_DIN, A_BIST_BM
);
    input A_CLK, A_MEN, A_WEN, A_REN, A_DLY;
    input [9:0] A_ADDR;
    input [15:0] A_DIN;
    output [15:0] A_DOUT;
    input [15:0] A_BM;
    input A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN;
    input [9:0] A_BIST_ADDR;
    input [15:0] A_BIST_DIN, A_BIST_BM;
endmodule

module RM_IHPSG13_1P_1024x8_c2_bm_bist (
    A_CLK, A_MEN, A_WEN, A_REN,
    A_ADDR, A_DIN, A_DLY, A_DOUT, A_BM,
    A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN,
    A_BIST_ADDR, A_BIST_DIN, A_BIST_BM
);
    input A_CLK, A_MEN, A_WEN, A_REN, A_DLY;
    input [9:0] A_ADDR;
    input [7:0] A_DIN;
    output [7:0] A_DOUT;
    input [7:0] A_BM;
    input A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN;
    input [9:0] A_BIST_ADDR;
    input [7:0] A_BIST_DIN, A_BIST_BM;
endmodule

module RM_IHPSG13_1P_2048x64_c2_bm_bist (
    A_CLK, A_MEN, A_WEN, A_REN,
    A_ADDR, A_DIN, A_DLY, A_DOUT, A_BM,
    A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN,
    A_BIST_ADDR, A_BIST_DIN, A_BIST_BM
);
    input A_CLK, A_MEN, A_WEN, A_REN, A_DLY;
    input [10:0] A_ADDR;
    input [63:0] A_DIN;
    output [63:0] A_DOUT;
    input [63:0] A_BM;
    input A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN;
    input [10:0] A_BIST_ADDR;
    input [63:0] A_BIST_DIN, A_BIST_BM;
endmodule

module RM_IHPSG13_1P_2048x32_c2_bm_bist (
    A_CLK, A_MEN, A_WEN, A_REN,
    A_ADDR, A_DIN, A_DLY, A_DOUT, A_BM,
    A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN,
    A_BIST_ADDR, A_BIST_DIN, A_BIST_BM
);
    input A_CLK, A_MEN, A_WEN, A_REN, A_DLY;
    input [10:0] A_ADDR;
    input [31:0] A_DIN;
    output [31:0] A_DOUT;
    input [31:0] A_BM;
    input A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN;
    input [10:0] A_BIST_ADDR;
    input [31:0] A_BIST_DIN, A_BIST_BM;
endmodule

module RM_IHPSG13_1P_4096x16_c3_bm_bist (
    A_CLK, A_MEN, A_WEN, A_REN,
    A_ADDR, A_DIN, A_DLY, A_DOUT, A_BM,
    A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN,
    A_BIST_ADDR, A_BIST_DIN, A_BIST_BM
);
    input A_CLK, A_MEN, A_WEN, A_REN, A_DLY;
    input [11:0] A_ADDR;
    input [15:0] A_DIN;
    output [15:0] A_DOUT;
    input [15:0] A_BM;
    input A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN;
    input [11:0] A_BIST_ADDR;
    input [15:0] A_BIST_DIN, A_BIST_BM;
endmodule

module RM_IHPSG13_1P_4096x8_c3_bm_bist (
    A_CLK, A_MEN, A_WEN, A_REN,
    A_ADDR, A_DIN, A_DLY, A_DOUT, A_BM,
    A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN,
    A_BIST_ADDR, A_BIST_DIN, A_BIST_BM
);
    input A_CLK, A_MEN, A_WEN, A_REN, A_DLY;
    input [11:0] A_ADDR;
    input [7:0] A_DIN;
    output [7:0] A_DOUT;
    input [7:0] A_BM;
    input A_BIST_CLK, A_BIST_EN, A_BIST_MEN, A_BIST_WEN, A_BIST_REN;
    input [11:0] A_BIST_ADDR;
    input [7:0] A_BIST_DIN, A_BIST_BM;
endmodule
