"""Declaration regression tests, including read only checks of HDL files."""

from pathlib import Path

import pytest

from .moduledef_parser import parse_module_def, preprocess_header


PROFILE = {"USE_INTERNAL_PARAMS", "USE_INTERNAL_PARAMS_PE", "USE_INTERNAL_PARAMS_PE_cluster"}
HDL = Path(__file__).resolve().parents[2] / "hdl"


def test_quoted_parentheses_and_identifier_prefixes():
    module = parse_module_def('module M #(parameter TXT = ")", parameter W=8, '
                              'parameter integer_count=3)(input wire_ready); endmodule')
    assert module.parameters["TXT"].default_value == '")"'
    assert module.parameters["W"].default_value == "8"
    assert module.parameters["integer_count"].default_value == "3"
    assert list(module.ports) == ["wire_ready"]


@pytest.mark.parametrize("marker", ["#(", "# (", "#\n("])
def test_parameter_marker_spacing_and_quoted_commas(marker):
    module = parse_module_def(
        'module M ' + marker + 'parameter TXT="a,b ) []", parameter W=8)(input clk); endmodule'
    )
    assert module.parameters["TXT"].default_value == '"a,b ) []"'
    assert module.parameters["W"].default_value == "8"


def test_quoted_comments_escaped_quotes_and_nested_expressions():
    module = parse_module_def(r'''module M #(
        parameter TXT = "// ) \" ( /* literal */",
        parameter W = $clog2((8 + 1)), /* ) ignored */
        localparam [W-1:0] MASK = {W{1'b1}}
    )(input [($clog2(W)+1)-1:0] data); endmodule''')
    assert module.parameters["TXT"].default_value == r'"// ) \" ( /* literal */"'
    assert module.parameters["W"].default_value == "$clog2((8 + 1))"
    assert module.parameters["MASK"].kind == "localparam"
    assert module.parameters["MASK"].ptype == "[W-1:0]"
    assert module.parameters["MASK"].default_value == "{W{1'b1}}"
    assert module.ports["data"].packed_range == ("($clog2(W)+1)-1", "0")


def test_scalar_and_explicit_two_bit_range_are_distinct():
    module = parse_module_def("module M(input scalar, output reg signed [1:0] pair); endmodule")
    assert module.ports["scalar"].packed_range is None
    assert module.ports["scalar"].sign == "unsigned"
    assert module.ports["scalar"].type == "wire"
    assert module.ports["pair"].packed_range == ("1", "0")
    assert module.ports["pair"].type == "reg"
    assert module.ports["pair"].sign == "signed"


BRANCHES = '''module M #(
`ifdef INTERNAL
parameter W = 8,
`else
parameter W = 32,
`endif
localparam DOUBLE = W * 2
)(input [W-1:0] data);
`ifdef BODY_ONLY
wire opaque;
`endif
endmodule'''


def test_conditional_headers_require_explicit_profile_and_select_one_branch():
    with pytest.raises(ValueError, match="requires an explicit defines profile"):
        parse_module_def(BRANCHES)
    internal = parse_module_def(BRANCHES, {"INTERNAL"})
    external = parse_module_def(BRANCHES, set())
    assert internal.parameters["W"].default_value == "8"
    assert external.parameters["W"].default_value == "32"
    for module in (internal, external):
        assert set(module.parameters) == {"W", "DOUBLE"}
        assert module.parameters["DOUBLE"].kind == "localparam"
        assert module.parameters["DOUBLE"].default_value == "W * 2"
        assert module.ports["data"].packed_range == ("W-1", "0")
    selected = preprocess_header(BRANCHES, {"INTERNAL"})
    assert "parameter W = 32" not in selected
    assert "`ifdef BODY_ONLY\nwire opaque;\n`endif" in selected


def test_inactive_include_does_not_get_expanded():
    source = '''module M #(
`ifdef INTERNAL
parameter W=8
`else
`include "absent.vh"
`endif
)(input clk); endmodule'''
    assert parse_module_def(source, {"INTERNAL"}).parameters["W"].default_value == "8"
    with pytest.raises(ValueError, match="Unsupported `include"):
        parse_module_def(source, set())


@pytest.mark.parametrize(("source", "message"), [
    ("module M(input clk); endmodule module N(input x); endmodule", "exactly one module"),
    ("`ifdef A\nmodule M(input clk); endmodule\n`endif", "enclosing a module"),
    ("module M(input a output b); endmodule", "Unsupported port name"),
    ("module M #(parameter W=8, nonsense)(input clk); endmodule", "Unsupported parameter declaration"),
    ("module M #(parameter W=8,)(input clk); endmodule", "Unsupported parameter declaration"),
    ("module M #(parameter W=`VALUE)(input clk); endmodule", "Macro expansion"),
    ("module M #(parameter TXT=\"unterminated)(input clk); endmodule", "Unterminated string"),
    ("module M(input [7:0] data [2:0]); endmodule", "Unsupported port name"),
    ("module M(a); input a; endmodule", "Unsupported ANSI port"),
    ("module M #(parameter W=(1] + 2))(input clk); endmodule", "Unsupported parameter declaration"),
    ("module M(input [(1]+2):0] data); endmodule", "Unsupported ANSI port declaration"),
])
def test_unsupported_or_malformed_declarations_fail_specifically(source, message):
    with pytest.raises(ValueError, match=message):
        parse_module_def(source)


def test_nested_header_conditionals():
    source = '''module M #(
`ifndef EXTERNAL
`ifdef WIDE
parameter W=16
`else
parameter W=8
`endif
`else
parameter W=4
`endif
)(input clk); endmodule'''
    assert parse_module_def(source, {"WIDE"}).parameters["W"].default_value == "16"
    assert parse_module_def(source, set()).parameters["W"].default_value == "8"
    assert parse_module_def(source, {"EXTERNAL"}).parameters["W"].default_value == "4"
    assert parse_module_def(
        source, {"EXTERNAL", "WIDE"}
    ).parameters["W"].default_value == "4"


def test_actual_pe_header_defaults_and_port_widths():
    module = parse_module_def((HDL / "PE.v").read_text(encoding="utf-8"), PROFILE)
    assert module.parameters["PARALLEL_MACS"].default_value == "2"
    assert module.parameters["SPARSITY_EN"].default_value == "1"
    assert module.parameters["TRANS_BITWIDTH_PSUM"].kind == "localparam"
    assert module.parameters["TRANS_BITWIDTH_PSUM"].default_value == "DATA_PSUM_BITWIDTH * PSUM_WORDS_PER_TRANSFER"
    assert module.parameters["PARALLEL_MACS_ADDR"].ptype == "[PSUM_ADDR_BITWIDTH-1:0]"
    assert module.parameters["PARALLEL_MACS_ADDR"].default_value == "PARALLEL_MACS[PSUM_ADDR_BITWIDTH-1:0]"
    assert module.parameters["PARALLEL_MACS"].ptype == "integer"
    assert module.parameters["SERIAL"].ptype == ""
    assert module.ports["data_stream_i"].packed_range == ("11", "0")
    assert module.ports["iact_data_i"].packed_range == ("TRANS_BITWIDTH_IACT*NUM_GLB_IACT-1", "0")
    assert "IDLE" not in module.parameters


def test_actual_parallel_selected_defaults_and_expression_widths():
    source = (HDL / "OpenEye_Parallel.v").read_text(encoding="utf-8")
    module = parse_module_def(source, PROFILE)
    assert module.parameters["CLUSTER_ROWS"].default_value == "8"
    assert module.parameters["PARALLEL_MACS"].default_value == "2"
    assert module.parameters["SPARSITY_EN"].default_value == "1"
    assert module.parameters["IACT_MEM_ADDR_BITS"].kind == "localparam"
    assert module.parameters["IACT_MEM_ADDR_BITS"].default_value == "$clog2(IACT_MEM_ADDR_WORDS)"
    assert module.parameters["CLUSTERS"].default_value == "CLUSTER_COLUMNS * CLUSTER_ROWS"
    assert module.ports["kernel_per_pe_cluster_i"].packed_range == (
        "((PE_ROWS > 1) ? $clog2(PE_ROWS) : 1)-1", "0"
    )
    assert module.ports["iact_ready_o"].type == "reg"
    assert module.ports["psum_transmitted_i"].packed_range is None
    with pytest.raises(ValueError, match="Unsupported `include"):
        parse_module_def(source, PROFILE | {"USE_EXTERNAL_PARAMS"})


@pytest.mark.parametrize(("filename", "ports"), [
    ("SPad_DP.v", ["clk_i", "re_i", "we_i", "addr_r_i", "addr_w_i", "data_i", "data_o"]),
    ("SPad_DP_RW.v", ["clk_i", "re_a_i", "re_b_i", "we_a_i", "we_b_i",
                      "addr_r_a_i", "addr_r_b_i", "addr_w_a_i", "addr_w_b_i",
                      "data_a_i", "data_b_i", "data_a_o", "data_b_o"]),
])
def test_actual_scratchpad_block_comments_and_declarations(filename, ports):
    module = parse_module_def((HDL / filename).read_text(encoding="utf-8"), PROFILE)
    assert list(module.ports) == ports
    assert module.parameters["DATA_WIDTH"].default_value == "8"
    assert module.parameters["ADDR_WIDTH"].default_value == "10"
    for name, port in module.ports.items():
        assert port.type == "wire"
        assert port.sign == "unsigned"
        expected = (("ADDR_WIDTH-1", "0") if name.startswith("addr_")
                    else ("DATA_WIDTH-1", "0") if name.startswith("data_") else None)
        assert port.packed_range == expected
        assert port.direction == ("output" if name.endswith("_o") else "input")


def test_actual_ram_mixed_case_names_and_quoted_default():
    module = parse_module_def((HDL / "RAM_DP_RW.v").read_text(encoding="utf-8"), PROFILE)
    assert list(module.parameters) == ["AddrWidth", "DataWidth", "Pipelined", "Implementation"]
    assert module.parameters["AddrWidth"].default_value == "16"
    assert module.parameters["Implementation"].default_value == '"generic"'
    assert module.parameters["Implementation"].ptype == ""
    assert module.ports["data_b_o"].packed_range == ("DataWidth-1", "0")


def test_parameter_type_signedness_and_literal_default():
    module = parse_module_def("module M #(parameter logic signed [7:0] value = -8'sd1)(); endmodule")
    assert module.parameters["value"].ptype == "logic signed [7:0]"
    assert module.parameters["value"].default_value == "-8'sd1"


def test_actual_adder_defaults_and_signed_ports():
    module = parse_module_def((HDL / "adder.v").read_text(encoding="utf-8"))
    assert module.parameters["DATA_WIDTH_SUM"].default_value == "20"
    assert module.parameters["RESET_VALUE"].default_value == "0"
    assert module.parameters["DATA_WIDTH_SUM"].ptype == ""
    assert module.ports["sum_o"].type == "reg"
    assert module.ports["sum_o"].sign == "signed"
    assert module.ports["sum_o"].packed_range == ("DATA_WIDTH_SUM-1", "0")


def test_actual_corpus_declarations():
    sources = sorted(HDL.rglob("*.v"))
    assert sources, f"No Verilog files found under {HDL}"
    for source in sources:
        module = parse_module_def(
            source.read_text(encoding="utf-8"),
            PROFILE,
        )
        assert module.module_name == source.stem, str(source)
