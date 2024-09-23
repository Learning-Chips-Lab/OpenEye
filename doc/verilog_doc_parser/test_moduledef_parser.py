# This file is part of the OpenEye project.
# All rights reserved. © Fachhochschule Dortmund - University of Applied Sciences and Arts.
# SPDX-License-Identifier: SHL-2.1
# For more details, see the LICENSE file in the root directory of this project.
import os
from .moduledef_parser import parse_module_def

class TestModuledefParser(object):

    def test_parse_and_compare_tester_module(self):
        
        text="""
// Blah blah blah

module tester
#(
  parameter integer WIDTH = 8
)(
  input                             clk_i,
  input                             rst_ni,
  input signed                      a_i,
  output signed [15:0]              sum_o,
  output reg                        carry_o,
  output reg signed                 overflow_o
  input wire                        wire_i,
  output reg unsigned               wire_o,
  input signed [7:0]                b_i,
  input signed [7:0]                wire_i2,
  output signed [15:0]              wire_o2,
  input reg signed [7:0]            reg_i3,
  output wire signed [WIDTH:0]      wire_o3,
  output reg signed [WIDTH-1:0]     reg_o4
);
  // Verilog code here
endmodule
"""
        module = parse_module_def(text)
        assert module.module_name == "tester"

        assert len(module.parameters) == 1
        assert "WIDTH" in module.parameters
        assert module.parameters["WIDTH"].ptype == "integer"
        assert module.parameters["WIDTH"].default_value == "8"

        assert len(module.ports) == 14
        assert "clk_i" in module.ports
        assert module.ports["clk_i"].direction == "input"
        assert module.ports["clk_i"].sign == "unsigned"
        assert module.ports["clk_i"].type == "wire" 
        assert module.ports["clk_i"].range_upper == 1
        assert module.ports["clk_i"].range_lower == 0

        assert "rst_ni" in module.ports
        assert module.ports["rst_ni"].direction == "input"
        assert module.ports["rst_ni"].sign == "unsigned"
        assert module.ports["rst_ni"].type == "wire"
        assert module.ports["rst_ni"].range_upper == 1
        assert module.ports["rst_ni"].range_lower == 0

        assert "a_i" in module.ports
        assert module.ports["a_i"].direction == "input"
        assert module.ports["a_i"].sign == "signed"
        assert module.ports["a_i"].type == "wire"
        assert module.ports["a_i"].range_upper == 1
        assert module.ports["a_i"].range_lower == 0

        assert "sum_o" in module.ports
        assert module.ports["sum_o"].direction == "output"
        assert module.ports["sum_o"].sign == "signed"
        assert module.ports["sum_o"].type == "wire"
        assert module.ports["sum_o"].range_upper == 15
        assert module.ports["sum_o"].range_lower == 0

        assert "carry_o" in module.ports
        assert module.ports["carry_o"].direction == "output"
        assert module.ports["carry_o"].sign == "unsigned"
        assert module.ports["carry_o"].type == "reg"
        assert module.ports["carry_o"].range_upper == 1
        assert module.ports["carry_o"].range_lower == 0

        assert "overflow_o" in module.ports
        assert module.ports["overflow_o"].direction == "output"
        assert module.ports["overflow_o"].sign == "signed"
        assert module.ports["overflow_o"].type == "reg"
        assert module.ports["overflow_o"].range_upper == 1
        assert module.ports["overflow_o"].range_lower == 0

        assert "wire_i2" in module.ports
        assert module.ports["wire_i2"].direction == "input"
        assert module.ports["wire_i2"].sign == "signed"
        assert module.ports["wire_i2"].type == "wire"
        assert module.ports["wire_i2"].range_upper == 7
        assert module.ports["wire_i2"].range_lower == 0

        assert "wire_o2" in module.ports
        assert module.ports["wire_o2"].direction == "output"
        assert module.ports["wire_o2"].sign == "signed"
        assert module.ports["wire_o2"].type == "wire"
        assert module.ports["wire_o2"].range_upper == 15
        assert module.ports["wire_o2"].range_lower == 0

        assert "reg_i3" in module.ports
        assert module.ports["reg_i3"].direction == "input"
        assert module.ports["reg_i3"].sign == "signed"
        assert module.ports["reg_i3"].type == "reg"
        assert module.ports["reg_i3"].range_upper == 7
        assert module.ports["reg_i3"].range_lower == 0

        assert "wire_o3" in module.ports
        assert module.ports["wire_o3"].direction == "output"
        assert module.ports["wire_o3"].sign == "signed"
        assert module.ports["wire_o3"].type == "wire"
        assert module.ports["wire_o3"].range_upper == "WIDTH"
        assert module.ports["wire_o3"].range_lower == 0

        assert "reg_o4" in module.ports
        assert module.ports["reg_o4"].direction == "output"
        assert module.ports["reg_o4"].sign == "signed"
        assert module.ports["reg_o4"].type == "reg"
        assert module.ports["reg_o4"].range_upper == "WIDTH-1"
        assert module.ports["reg_o4"].range_lower == 0


    def test_parse_af_cluster(self):
        text= \
"""
/// Module: af_cluster
///
/// The (a)ctivation(f)unction_cluster is a modul for creating the neccessary non-linear function.
/// Currently there is only ReLU implemented, which can be activated by setting mode_i to 1
/// The ready and data signal gets passed on.
///
/// Parameters:
///    DATA_WIDTH:      Bitwidth of data words
///    MODES:           Amount of MODES in this module, currently 2 (ReLU and nothing)
///   
/// Ports:
///    mode_i:          Mode, that chooses operation.
///    enable_i:        Enable Port, just gets delayed to output
///    enable_o:        Output for Enable signal, signals valid data
///    ready_i:         Ready signal, just gets delayed to output
///    ready_o:         Output for ready signal
///    data_i:          Data Port In, consists of two data packages
///    data_o:          Data Port Out, consists of two data packages
///

module af_cluster 
#( 
  parameter integer            DATA_BITWIDTH   = 40,
  parameter integer            MODES           = 2
) (
  input                          clk_i,
  input                          rst_ni,
  input  [$clog2(MODES)-1 : 0]   mode_i,

  output                         ready_o,
  input  [DATA_BITWIDTH-1:0]     data_i,
  input                          enable_i,

  input                          ready_i,
  output [DATA_BITWIDTH-1:0]     data_o,
  output                         enable_o

);
  // Verilog code here
endmodule
"""
        module = parse_module_def(text)
        assert module.module_name == "af_cluster"

    def test_parse_delay_cluster(self):
        text= \
"""

/// Module: delay_cluster
///
/// Module for delaying input signals
///
/// Parameters:
///    DATA_BITWIDTH  - Length of data
///   
/// Ports:
///    ready_o          - Outputs port `i`, if `sel_i` is 1, else outputs 0
///    data_i           - Input port `i`, if `sel_i` is 0, else outputs 0
///    enable_i         - Input port
///    ready_i          - Input port data port
///    data_o           - Output port data port
///    enable_o         - Output portdata port
///    delay_psum_glb_i - Input port data port
///

module delay_cluster 
#( 
  parameter integer            DATA_BITWIDTH   = 20
) (
  input                            clk_i,
  input                            rst_ni,

  output reg                       ready_o,
  input      [DATA_BITWIDTH-1 : 0] data_i,
  input                            enable_i,

  input                            ready_i,
  output     [DATA_BITWIDTH-1 : 0] data_o,
  output reg                       enable_o,

  input  reg [3 : 0]               delay_psum_glb_i

);
  // Verilog code here
endmodule
"""
        module = parse_module_def(text)
        assert module.module_name == "delay_cluster"


    def test_parse_simple_formula(self): 
        text= \
"""
module parameter_test 
#( 
  parameter integer            FIRST_PARAMTER       = 20,
  parameter integer            SECOND_PARAMETER     = 8,
  parameter FOURTH_PARAMETER                        = FIRST_PARAMETER * SECOND_PARAMETER,
  parameter FIFTH_PARAMETER                         = FIRST_PARAMETER + SECOND_PARAMETER,
  parameter SIXTH_PARAMETER                         = FIRST_PARAMETER - SECOND_PARAMETER,
  parameter SEVENTH_PARAMETER                       = FIRST_PARAMETER / SECOND_PARAMETER,
  parameter EIGHTH_PARAMETER                        = FIRST_PARAMETER % SECOND_PARAMETER,
  parameter NINTH_PARAMETER                         = FIRST_PARAMETER ** SECOND_PARAMETER,
  parameter TENTH_PARAMETER                         = FIRST_PARAMETER & SECOND_PARAMETER,
  parameter ELEVENTH_PARAMETER                      = FIRST_PARAMETER | SECOND_PARAMETER,
  parameter TWELFTH_PARAMETER                       = FIRST_PARAMETER ^ SECOND_PARAMETER,
  parameter THIRTEENTH_PARAMETER                    = ~FIRST_PARAMETER,
  parameter FOURTEENTH_PARAMETER                    = FIRST_PARAMETER << SECOND_PARAMETER,
  parameter FIFTEENTH_PARAMETER                     = FIRST_PARAMETER >> SECOND_PARAMETER,
  parameter SIXTEENTH_PARAMETER                     = $clog2(SECOND_PARAMETER),
  parameter SEVENTEENTH_PARAMETER                   = $clog10(SECOND_PARAMETER)
) (
  input                            clk_i,
  input                            rst_ni,

  output reg                       ready_o,
  input      [SIXTH_PARAMETER-1 : 0] data_i,
  input                            enable_i,

  input                            ready_i,
  output     [SIXTEENTH_PARAMETER-1 : 0] data_o,
  output reg                       enable_o
);
  // Verilog code here
endmodule
"""

        from pyparsing import Regex, OneOrMore, Word, alphanums, LineEnd, Suppress, Optional, Combine, Group, oneOf
        identifier = Regex(r"[a-zA-Z_][a-zA-Z0-9_\$]*")
        number = OneOrMore(Word(alphanums + r"_+-*/%<>&|^~'\".\$() "), stopOn=LineEnd())  # Allow for parameter values to be expressions

        # Parameter definition: parameter name and value
        ptype     = Optional( oneOf('integer real realtime time'), default='integer')
        param_def = Group(Suppress("parameter") + ptype("p_type") + identifier("p_name") + Optional(Suppress("=")) + Optional(Combine(number)("p_value"), default=None) + Optional(Suppress(",")))

        parse = param_def.parse_string("parameter integer            FIRST_PARAMTER       = 20,")
        assert parse[0]["p_type"] == "integer"
        assert parse[0]["p_name"] == "FIRST_PARAMTER"
        assert parse[0]["p_value"] == "20"

        parse = param_def.parse_string("  parameter FOURTH_PARAMETER                        = FIRST_PARAMETER * SECOND_PARAMETER,")
        assert parse[0]["p_type"] == "integer"
        assert parse[0]["p_name"] == "FOURTH_PARAMETER"
        assert parse[0]["p_value"] == "FIRST_PARAMETER * SECOND_PARAMETER"

        parse = param_def.parse_string("  parameter FIFTH_PARAMETER                         = FIRST_PARAMETER + SECOND_PARAMETER,")
        assert parse[0]["p_type"] == "integer"
        assert parse[0]["p_name"] == "FIFTH_PARAMETER"
        assert parse[0]["p_value"] == "FIRST_PARAMETER + SECOND_PARAMETER"

        parse = param_def.parse_string("  parameter SIXTH_PARAMETER                         = FIRST_PARAMETER - SECOND_PARAMETER,")
        assert parse[0]["p_type"] == "integer"
        assert parse[0]["p_name"] == "SIXTH_PARAMETER"
        assert parse[0]["p_value"] == "FIRST_PARAMETER - SECOND_PARAMETER"

        parse = param_def.parse_string("  parameter SEVENTH_PARAMETER                       = FIRST_PARAMETER / SECOND_PARAMETER,")
        assert parse[0]["p_type"] == "integer"
        assert parse[0]["p_name"] == "SEVENTH_PARAMETER"
        assert parse[0]["p_value"] == "FIRST_PARAMETER / SECOND_PARAMETER"

        parse = param_def.parse_string("  parameter EIGHTH_PARAMETER                        = FIRST_PARAMETER % SECOND_PARAMETER,")

        parse = param_def.parse_string("  parameter NINTH_PARAMETER                         = FIRST_PARAMETER ** SECOND_PARAMETER,")

        parse = param_def.parse_string("  parameter TENTH_PARAMETER                         = FIRST_PARAMETER & SECOND_PARAMETER,")

        parse = param_def.parse_string("  parameter ELEVENTH_PARAMETER                      = FIRST_PARAMETER | SECOND_PARAMETER,")

        parse = param_def.parse_string("  parameter TWELFTH_PARAMETER                       = FIRST_PARAMETER ^ SECOND_PARAMETER,")

        parse = param_def.parse_string("  parameter THIRTEENTH_PARAMETER                    = ~FIRST_PARAMETER,")

        parse = param_def.parse_string("  parameter FOURTEENTH_PARAMETER                    = FIRST_PARAMETER << SECOND_PARAMETER,")

        parse = param_def.parse_string("  parameter FIFTEENTH_PARAMETER                     = FIRST_PARAMETER >> SECOND_PARAMETER,")

        parse = param_def.parse_string("  parameter SIXTEENTH_PARAMETER                     = $clog2(SECOND_PARAMETER),")

        parse = param_def.parse_string("  parameter SEVENTEENTH_PARAMETER                   = $clog10(SECOND_PARAMETER)")

    def test_parse_parameter_definition(self):

        parameters_string = """#(
  parameter integer            FIRST_PARAMTER       = 20,
  parameter integer            SECOND_PARAMETER     = 8,
  parameter FOURTH_PARAMETER                        = FIRST_PARAMETER * SECOND_PARAMETER,
  parameter FIFTH_PARAMETER                         = FIRST_PARAMETER + SECOND_PARAMETER,
  parameter SIXTH_PARAMETER                         = FIRST_PARAMETER - SECOND_PARAMETER,
  parameter SEVENTH_PARAMETER                       = FIRST_PARAMETER / SECOND_PARAMETER,
  parameter EIGHTH_PARAMETER                        = FIRST_PARAMETER % SECOND_PARAMETER,
  parameter NINTH_PARAMETER                         = FIRST_PARAMETER ** SECOND_PARAMETER,
  parameter TENTH_PARAMETER                         = FIRST_PARAMETER & SECOND_PARAMETER,
  parameter ELEVENTH_PARAMETER                      = FIRST_PARAMETER | SECOND_PARAMETER,
  parameter TWELFTH_PARAMETER                       = FIRST_PARAMETER ^ SECOND_PARAMETER,
  parameter THIRTEENTH_PARAMETER                    = ~FIRST_PARAMETER,
  parameter FOURTEENTH_PARAMETER                    = FIRST_PARAMETER << SECOND_PARAMETER,
  parameter FIFTEENTH_PARAMETER                     = FIRST_PARAMETER >> SECOND_PARAMETER,
  parameter SIXTEENTH_PARAMETER                     = $clog2(SECOND_PARAMETER),
  parameter SEVENTEENTH_PARAMETER                   = $clog10(SECOND_PARAMETER)
) ("""

        
        from pyparsing import Regex, OneOrMore, Word, alphanums, LineEnd, Suppress, Optional, Combine, Group, oneOf
        identifier = Regex(r"[a-zA-Z_][a-zA-Z0-9_\$]*")
        number = OneOrMore(Word(alphanums + r"_+-*/%<>&|^~'\".\$() "), stopOn=LineEnd())  # Allow for parameter values to be expressions

        # Parameter definition: parameter name and value
        ptype     = Optional( oneOf('integer real realtime time'), default='integer')
        param_def = Group(Suppress("parameter") + ptype("p_type") + identifier("p_name") + Optional(Suppress("=")) + Optional(Combine(number)("p_value"), default=None) + Optional(Suppress(",")))

        parameters = Group(
            Suppress("#(") +
            OneOrMore(param_def, stopOn = ")") + 
            Suppress(")")
        )("parameters")

        parse = parameters.parse_string(parameters_string)
        assert len(parse.parameters) == 16

        text = """
#(
  parameter IS_TOPLEVEL                = 1,
  parameter LEFT_CLUSTER               = 0,
  parameter TOP_CLUSTER                = 0,
  parameter BOTTOM_CLUSTER             = 0,
  parameter DATA_IACT_BITWIDTH         = 8,
  parameter DATA_PSUM_BITWIDTH         = 20,
  parameter DATA_WGHT_BITWIDTH         = 8,
  parameter TRANS_BITWIDTH_IACT        = 24,
  parameter TRANS_BITWIDTH_PSUM        = 20,
  parameter TRANS_BITWIDTH_WGHT        = 24,
  parameter NUM_GLB_IACT               = 3,
  parameter NUM_GLB_WGHT               = 3,
  parameter NUM_GLB_PSUM               = 4,
  parameter PE_ROWS                    = 3,
  parameter PE_COLUMNS                 = 4,
  parameter PES                        = PE_ROWS * PE_COLUMNS,
  parameter CLUSTER_ROWS               = 8,
  parameter CLUSTER_COLUMNS             = 2,
  parameter CLUSTERS                   = CLUSTER_COLUMNS * CLUSTER_ROWS,
  parameter IACT_PER_PE                = 16,
  parameter WGHT_PER_PE                = 192,
  parameter PSUM_PER_PE                = 16,
  parameter IACT_MEM_ADDR_WORDS        = 512,
  parameter IACT_MEM_ADDR_BITS         = $clog2(IACT_MEM_ADDR_WORDS),
  parameter PSUM_MEM_ADDR_WORDS        = 384,
  parameter PSUM_MEM_ADDR_BITS         = $clog2(PSUM_MEM_ADDR_WORDS),
  parameter BANO_MODES                 = 2,
  parameter AF_MODES                   = 2
)("""

        parse = parameters.parse_string(text)
        assert len(parse.parameters) == 28


    def test_parse_ports(self):
        
        def cast_to_int(value):
            try:
                return int(value)
            except:
                return value

        from pyparsing import Regex, OneOrMore, Word, alphanums, LineEnd, Suppress, Optional, Group, oneOf, Literal, pyparsing_common, tokenMap, ZeroOrMore, dbl_slash_comment
        identifier = Regex(r"[a-zA-Z_][a-zA-Z0-9_\$]*")
        number = OneOrMore(Word(alphanums + r"_+-*/%<>&|^~'\".\$() "), stopOn=LineEnd())  # Allow for parameter values to be expressions
        number.setParseAction(tokenMap(cast_to_int))
        # Port direction: input, output, inout
        port_dir = Literal("input") | Literal("output") | Literal("inout")
        
        # Port definition: direction and identifier, optionally with width [MSB:LSB]
        port_sign = Optional( oneOf('signed unsigned'), default='unsigned')
        port_type = Optional( oneOf('wire reg logic'), default='wire')
        port_range = Optional(Group(Suppress("[") + number("upper") + Suppress(":") + number("lower") + Suppress("]")), default=[[1, 0]])
        port_def = Group(port_dir("direction") + port_type("type") + port_sign("sign") + port_range("range") + identifier("name") + Optional(Suppress(",")))

        ptype = Optional( oneOf('integer real realtime time'), default='integer')

        ports = Group(
            Suppress("(") +
            OneOrMore(port_def,  stopOn=");") +
            Suppress(");")
        )("ports")
        ports.ignore(dbl_slash_comment)

        parse = port_def.parse_string("input                                          clk_i,")
        assert parse[0]["direction"]== "input"
        assert parse[0]["type"] == "wire"
        assert parse[0]["sign"] == "unsigned"
        assert parse[0]["range"][0][0] == 1
        assert parse[0]["range"][0][1] == 0
        assert parse[0]["name"] == "clk_i"

        parse = port_def.parse_string("input rst_ni,")
        assert parse[0]["direction"]== "input"
        assert parse[0]["type"] == "wire"
        assert parse[0]["sign"] == "unsigned"
        assert parse[0]["range"][0][0] == 1
        assert parse[0]["range"][0][1] == 0
        assert parse[0]["name"] == "rst_ni"

        parse = port_def.parse_string("input signed a_i,")
        assert parse[0]["direction"]== "input"
        assert parse[0]["type"] == "wire"
        assert parse[0]["sign"] == "signed"
        assert parse[0]["range"][0][0] == 1
        assert parse[0]["range"][0][1] == 0
        assert parse[0]["name"] == "a_i"

        parse = port_def.parse_string("input data_mode_i,")
        assert parse[0]["direction"]== "input"
        assert parse[0]["type"] == "wire"
        assert parse[0]["sign"] == "unsigned"
        assert parse[0]["range"][0][0] == 1
        assert parse[0]["range"][0][1] == 0
        assert parse[0]["name"] == "data_mode_i"

        parse = port_def.parse_string("input [$clog2(NUM_GLB_IACT)*PES-1:0]           iact_choose_i,")
        assert parse[0]["direction"]== "input"
        assert parse[0]["type"] == "wire"
        assert parse[0]["sign"] == "unsigned"
        assert parse[0]["range"][0][0] == "$clog2(NUM_GLB_IACT)*PES-1"
        assert parse[0]["range"][0][1] == 0
        assert parse[0]["name"] == "iact_choose_i"

        parse = port_def.parse_string("input [NUM_GLB_PSUM-1:0] psum_choose_i,")
        assert parse[0]["direction"]== "input"
        assert parse[0]["type"] == "wire"
        assert parse[0]["sign"] == "unsigned"
        assert parse[0]["range"][0][0] == "NUM_GLB_PSUM-1"
        assert parse[0]["range"][0][1] == 0
        assert parse[0]["name"] == "psum_choose_i"

        parse = port_def.parse_string("input [PES-1:0] compute_i,")
        assert parse[0]["direction"]== "input"
        assert parse[0]["type"] == "wire"
        assert parse[0]["sign"] == "unsigned"
        assert parse[0]["range"][0][0] == "PES-1"
        assert parse[0]["range"][0][1] == 0
        assert parse[0]["name"] == "compute_i"


        text = """
(
  // Connections to external Memory
  /////////////////////////////////////////
  input  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   ext_mem_iact_data_i,
  input  [IACT_MEM_ADDR_BITS*NUM_GLB_IACT-1:0]    ext_mem_iact_addr_i,
  input  [NUM_GLB_IACT-1:0]                       ext_mem_iact_enable_i,
  output [NUM_GLB_IACT-1:0]                       ext_mem_iact_ready_o,

  input  [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0]   ext_mem_wght_data_i,
  input  [NUM_GLB_WGHT-1:0]                       ext_mem_wght_enable_i,
  output [NUM_GLB_WGHT-1:0]                       ext_mem_wght_ready_o,

  input  [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   ext_mem_psum_data_i,
  input  [PSUM_MEM_ADDR_BITS*NUM_GLB_PSUM-1:0]    ext_mem_psum_addr_i,
  input  [NUM_GLB_PSUM-1:0]                       ext_mem_psum_enable_i,
  output [NUM_GLB_PSUM-1:0]                       ext_mem_psum_ready_o,

  output [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   ext_mem_psum_data_o,
  output [NUM_GLB_PSUM-1:0]                       ext_mem_psum_enable_o,
  input  [NUM_GLB_PSUM-1:0]                       ext_mem_psum_ready_i,

  
  ///Router Ports
  /////////////////////////////////////////
  
  ///Weights
  input  [NUM_GLB_WGHT-1:0]                       router_mode_wght_i,
    
  output [NUM_GLB_WGHT-1:0]                       enable_dst_side_wght,
  output [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0]   data_dst_side_wght,
  input  [NUM_GLB_WGHT-1:0]                       ready_dst_side_wght,
    
  input  [NUM_GLB_WGHT-1:0]                       enable_src_side_wght,
  input  [TRANS_BITWIDTH_WGHT*NUM_GLB_WGHT-1:0]   data_src_side_wght,
  output [NUM_GLB_WGHT-1:0]                       ready_src_side_wght,
    
  ///Activations
  input  [6*NUM_GLB_IACT-1:0]                     router_mode_iact_i,
    
  output [NUM_GLB_IACT-1:0]                       enable_dst_side_iact,
  output [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   data_dst_side_iact,
  input  [NUM_GLB_IACT-1:0]                       ready_dst_side_iact,
    
  input  [NUM_GLB_IACT-1:0]                       enable_src_side_iact,
  input  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   data_src_side_iact,
  output [NUM_GLB_IACT-1:0]                       ready_src_side_iact,
    
  output [NUM_GLB_IACT-1:0]                       enable_dst_top_iact,
  output [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   data_dst_top_iact,
  input  [NUM_GLB_IACT-1:0]                       ready_dst_top_iact,
    
  input  [NUM_GLB_IACT-1:0]                       enable_src_top_iact,
  input  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   data_src_top_iact,
  output [NUM_GLB_IACT-1:0]                       ready_src_top_iact,
    
  output [NUM_GLB_IACT-1:0]                       enable_dst_bottom_iact,
  output [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   data_dst_bottom_iact,
  input  [NUM_GLB_IACT-1:0]                       ready_dst_bottom_iact,
    
  input  [NUM_GLB_IACT-1:0]                       enable_src_bottom_iact,
  input  [TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1:0]   data_src_bottom_iact,
  output [NUM_GLB_IACT-1:0]                       ready_src_bottom_iact,
    
  ///Psum
  input  [3*NUM_GLB_PSUM-1:0]                     router_mode_psum_i,
    
  output [NUM_GLB_PSUM-1:0]                       enable_dst_top_psum,
  output [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   data_dst_top_psum,
  input  [NUM_GLB_PSUM-1:0]                       ready_dst_top_psum,
    
  input  [NUM_GLB_PSUM-1:0]                       enable_src_top_psum,
  input  [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   data_src_top_psum,
  output [NUM_GLB_PSUM-1:0]                       ready_src_top_psum,
    
  output [NUM_GLB_PSUM-1:0]                       enable_dst_bottom_psum,
  output [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   data_dst_bottom_psum,
  input  [NUM_GLB_PSUM-1:0]                       ready_dst_bottom_psum,
    
  input  [NUM_GLB_PSUM-1:0]                       enable_src_bottom_psum,
  input  [TRANS_BITWIDTH_PSUM*NUM_GLB_PSUM-1:0]   data_src_bottom_psum,
  output [NUM_GLB_PSUM-1:0]                       ready_src_bottom_psum,
  
  input  [$clog2(DATA_PSUM_BITWIDTH)-1:0]         fraction_bit_i,

  input  [$clog2(BANO_MODES)*NUM_GLB_PSUM-1:0]    bano_cluster_mode_i,
  input  [$clog2(AF_MODES)*NUM_GLB_PSUM-1:0]      af_cluster_mode_i,
  input  [3:0]                                    delay_psum_glb_i
);
"""
  
        parse = ports.parse_string(text)
        assert len(parse.ports) == 57

    def parse_and_compare_module_name_from_moduledef(self, module_file):
        """This simple test parses the module name from the docstring of the
        given module file and compares it with the module file name.
        
        The rational is that we can be sure that the module docstring is parsed
        correctly, i.e., no error is thrown, and the module name is correctly
        extracted from the docstring.
        """
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(cur_dir, "..", "..", "hdl", module_file)
        with open(test_file, "r") as f:
            text = f.read()

        module_def = parse_module_def(text)
        assert module_def.module_name == module_file.split(".")[0]

    def test_parse_af_cluster_file(self):
        self.parse_and_compare_module_name_from_moduledef("af_cluster.v")

    def test_parse_bano_cluster_file(self):
        self.parse_and_compare_module_name_from_moduledef("bano_cluster.v")

    def test_parse_data_pipeline_file(self):
        self.parse_and_compare_module_name_from_moduledef("data_pipeline.v")

    def test_parse_delay_cluster_file(self):
        self.parse_and_compare_module_name_from_moduledef("delay_cluster.v")

    def test_parse_demux2_file(self):
        self.parse_and_compare_module_name_from_moduledef("demux2.v")

    def test_parse_multiplier_file(self):
        self.parse_and_compare_module_name_from_moduledef("multiplier.v")

    def test_parse_mux2_file(self):
        self.parse_and_compare_module_name_from_moduledef("mux2.v")

    def test_parse_mux_iact_file(self):
        self.parse_and_compare_module_name_from_moduledef("mux_iact.v")

    def test_parse_router_iact_file(self):
        self.parse_and_compare_module_name_from_moduledef("router_iact.v")

    def test_parse_router_psum_file(self):
        self.parse_and_compare_module_name_from_moduledef("router_psum.v")

    def test_parse_router_wght_file(self):
        self.parse_and_compare_module_name_from_moduledef("router_wght.v")
    
    def test_parse_varlenFIFO_file(self):
        self.parse_and_compare_module_name_from_moduledef("varlenFIFO.v")

    def test_parse_GLB_cluster_file(self):
        self.parse_and_compare_module_name_from_moduledef("GLB_cluster.v")

    def test_parse_OpenEye_Cluster_file(self):
        self.parse_and_compare_module_name_from_moduledef("OpenEye_Cluster.v")

    def test_parse_OpenEye_Parallel_file(self):
        self.parse_and_compare_module_name_from_moduledef("OpenEye_Parallel.v")

    def test_parse_OpenEye_Wrapper_file(self):
        self.parse_and_compare_module_name_from_moduledef("OpenEye_Wrapper.v")