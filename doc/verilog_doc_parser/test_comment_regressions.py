"""Structured comment boundaries and source description regressions."""

from pathlib import Path

import pytest
from pyparsing import ParseResults

from .docstring_parser import MissingDocumentationError, parse_documentation


def test_section_headers_are_not_identifiers(tmp_path): 
    source = tmp_path / "M.v"
    source.write_text(
        "/// Module: M\n"
        "/// Description stays here.\n"
        "/// Parameters:\n"
        "///   W: Actual width\n"
        "///   integer_count - Integer counter\n"
        "///   AddrWidth: Address width\n"
        "/// Ports:\n"
        "///   clk: Clock description\n"
        "///   wire_ready: Ready description\n"
        "/// Timing:\n"
        "///   - Timing notes are not a port.\n"
        "module M(input clk); endmodule\n",
        encoding="utf-8",
    )
    doc = parse_documentation(source.read_text(encoding="utf-8"))
    assert doc.parameters == {
        "W": "Actual width",
        "integer_count": "Integer counter",
        "AddrWidth": "Address width",
    }
    assert doc.ports == {"clk": "Clock description", "wire_ready": "Ready description"}
    assert doc.description == "Description stays here."
    assert isinstance(doc.results, ParseResults)
    assert doc.results["name"] == "M"
    assert isinstance(doc.results["parameters"], ParseResults)
    assert [(item["name"], item["description"].strip()) for item in doc.results["parameters"]] == list(doc.parameters.items())
    assert isinstance(doc.results["ports"], ParseResults)
    assert [(item["name"], item["description"].strip()) for item in doc.results["ports"]] == list(doc.ports.items())


def test_continuations_remain_descriptions():
    doc = parse_documentation(
        "//! Module: M\n"
        "//! Parameters:\n"
        "//!   SERIAL - Serial mode\n"
        "//!            0: Parallel\n"
        "//!            1: Serial\n"
        "//!            Default: 0\n"
        "//! Ports:\n"
        "//!   ready - Ready signal\n"
        "//!           Width: 1 bit\n"
        "//!           More detail.\n"
    )
    assert doc.parameters == {"SERIAL": "Serial mode\n0: Parallel\n1: Serial\nDefault: 0"}
    assert doc.ports == {"ready": "Ready signal\nWidth: 1 bit\nMore detail."}
    assert doc.results["parameters"][0]["description"].strip() == doc.parameters["SERIAL"]
    assert doc.results["ports"][0]["description"].strip() == doc.ports["ready"]


def test_original_compact_hyphen_separator_is_preserved():
    doc = parse_documentation(
        "/// Module: M\n"
        "/// Existing description.\n"
        "/// Ports:\n"
        "///   clk_i-Clock input\n"
    )
    assert doc.ports == {"clk_i": "Clock input"}


def test_comment_block_stops_before_later_unrelated_comments():
    doc = parse_documentation(
        "/// Module: M\n"
        "/// Ports:\n"
        "///   clk: Clock description\n"
        "module M(input clk);\n"
        "/// Ports:\n"
        "///   bogus: Unrelated body comment\n"
        "endmodule\n"
    )
    assert doc.ports == {"clk": "Clock description"}


def test_missing_marker_is_specific():
    with pytest.raises(MissingDocumentationError, match="No structured"):
        parse_documentation("// Module: M\nmodule M(input clk); endmodule")


def test_malformed_marker_is_not_missing_documentation():
    with pytest.raises(ValueError, match="Invalid structured Module: name") as exc_info:
        parse_documentation("/// Module: 123invalid\n")

    assert not isinstance(exc_info.value, MissingDocumentationError)


def test_actual_pe_descriptions_are_preserved():
    source = Path(__file__).resolve().parents[2] / "hdl" / "PE.v"
    doc = parse_documentation(source.read_text(encoding="utf-8"))
    assert doc.module_name == "PE"
    assert doc.description.startswith("The PE (Processing Element) is a fundamental computational unit")
    assert "Key aspects of PE interconnection:" in doc.description
    assert "- Bypass network: Forwards freshly computed psums to avoid SPad read latency" in doc.description
    assert doc.parameters["SERIAL"].startswith("Boolean flag to enable SERIAL processing mode\n")
    assert "SERIAL mode reduces area at the cost of 2× latency" in doc.parameters["SERIAL"]
    assert doc.parameters["USE_DSP"].startswith("MAC implementation selection\n")
    assert doc.ports["clk_i"].startswith("System clock input (positive edge triggered)\n")
    assert "All registers update on rising edge of clk_i" in doc.ports["clk_i"]
    assert "Selected by iact_select_i, routed through internal multiplexer" in doc.ports["iact_data_i"]
    assert not ({"Ports", "clk_i", "iact_data_i"} & doc.parameters.keys())


def test_actual_adder_descriptions_are_preserved():
    source = Path(__file__).resolve().parents[2] / "hdl" / "adder.v"
    doc = parse_documentation(source.read_text(encoding="utf-8"))
    assert doc.parameters == {
        "DATA_WIDTH_SUM": "Bitwidth of the summands and sum (default: 20)",
        "RESET_VALUE": "Value loaded into sum_o during reset (default: 0)",
    }
    assert doc.ports["summand_1_i"] == "First signed input operand [DATA_WIDTH_SUM-1:0]"
    assert doc.ports["rst_ni"] == "Active low asynchronous reset"
    assert "The adder performs signed addition of two input operands" in doc.description
    assert "Timing" not in doc.ports