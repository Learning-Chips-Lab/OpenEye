from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

from PIL import Image
import pytest

from . import rst_doc_generator as generator
from .docstring_parser import parse_documentation
from .moduledef_parser import parse_module_def


SOURCE = """/// Module: M
/// Existing description.
/// Parameters:
///   W: Width description
///   Derived: Derived description
/// Ports:
///   clk: Clock description
///   pair: Pair description
module M #(parameter integer W = 8, localparam integer Derived = (W + 1),
           parameter HIDDEN = 7)(input clk, output [1:0] pair);
localparam INTERNAL = 9;
endmodule
"""


def make_input(tmp_path):
    hdl = tmp_path / "hdl"
    hdl.mkdir()
    (hdl / "M.v").write_text(SOURCE, encoding="utf-8")
    return hdl


def snapshot_files(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def symbolator_success(command, **kwargs):
    output = Path(command[command.index("-o") + 1])
    selected = Path(command[command.index("-i") + 1])
    Image.new("RGB", (12, 12), "white").save(output / f"__{selected.stem}.png")
    return subprocess.CompletedProcess(command, 0, stdout="controlled diagram success\n")


@pytest.mark.parametrize("module_name", ["PE", "adder"])
def test_render_actual_source_descriptions(tmp_path, module_name):
    source = Path(__file__).resolve().parents[2] / "hdl" / f"{module_name}.v"
    copied = tmp_path / source.name
    copied.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    text = copied.read_text(encoding="utf-8")
    doc = parse_documentation(text)
    module = parse_module_def(text, defines=generator.DEFINES)
    rst = generator.generate_rst(doc, module)
    if module_name == "PE":
        assert "The PE (Processing Element) is a fundamental computational unit" in rst
        assert "        Boolean flag to enable SERIAL processing mode\n" in rst
        assert "        SERIAL mode reduces area at the cost of 2× latency\n" in rst
        assert "        All registers update on rising edge of clk_i\n" in rst
    else:
        assert "A synchronous, registered adder module with enable control and reset functionality." in rst
        assert "        Bitwidth of the summands and sum (default: 20)\n" in rst
        assert "        Registered signed sum output [DATA_WIDTH_SUM-1:0]\n" in rst


def test_source_defaults_localparam_and_explicit_widths():
    rst = generator.generate_rst(parse_documentation(SOURCE), parse_module_def(SOURCE))
    assert ".. verilog:parameter:: parameter integer W = 8\n" in rst
    assert ".. verilog:parameter:: localparam integer Derived = (W + 1)\n" in rst
    assert ".. verilog:port:: input wire unsigned clk\n" in rst
    assert ".. verilog:port:: output wire unsigned [1:0] pair\n" in rst
    assert "HIDDEN" not in rst
    assert "INTERNAL" not in rst


def test_port_list_and_coverage_follow_the_hardware(tmp_path, monkeypatch):
    hdl = make_input(tmp_path)
    source = """/// Module: M
/// Example with incomplete, out-of-order port comments.
/// Ports:
///   sum_o: Signed result.
///   clk: Clock.
///   obsolete: A stale comment.
module M(input clk, input wr_mask_i, output reg signed [3:0] sum_o);
endmodule
"""
# Use CRLF line endings to check that the report hashes the original file bytes.
    source_bytes = source.replace("\n", "\r\n").encode("utf-8")
    (hdl / "M.v").write_bytes(source_bytes)
    monkeypatch.setattr(generator.subprocess, "run", symbolator_success)
    output = tmp_path / "out"
    report = generator.generate(hdl, output)
    entry, = report["generated"]
    assert report["failed"] == []
    assert entry["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert entry["ports"] == ["clk", "wr_mask_i", "sum_o"]
    assert entry["undocumented_ports"] == ["wr_mask_i"]
    assert entry["unmatched_comment_ports"] == ["obsolete"]
    rst = (output / "M.rst").read_text(encoding="utf-8")
    assert ".. verilog:module:: module M(clk, wr_mask_i, sum_o)\n" in rst
    assert ".. verilog:port:: output reg signed [3:0] sum_o\n" in rst
    assert {item["name"] for item in entry["declarations"]} == {"clk", "sum_o"}
    assert rst.count(".. verilog:port::") == 2
    assert "obsolete" not in rst


def test_module_named_index_keeps_its_own_page(tmp_path, monkeypatch):
    hdl = make_input(tmp_path)
    (hdl / "index.v").write_text(SOURCE.replace("Module: M", "Module: index")
                                 .replace("module M #", "module index #"), encoding="utf-8")
    monkeypatch.setattr(generator.subprocess, "run", symbolator_success)
    output = tmp_path / "out"
    report = generator.generate(hdl, output)
    assert report["failed"] == []
    entries = {entry["module"]: entry for entry in report["generated"]}
    assert entries["index"]["page"] == "module-index"
    assert ".. verilog:module:: module index(clk, pair)" in (output / "module-index.rst").read_text()
    assert (output / "__index.png").is_file()
    navigation = (output / "index.rst").read_text()
    assert navigation.splitlines()[-2:] == ["    M", "    module-index"]
    assert ".. verilog:module::" not in navigation


def test_successful_generation_and_existing_output_refusal(tmp_path, monkeypatch):
    hdl = make_input(tmp_path)
    output = tmp_path / "successful"
    monkeypatch.setattr(generator.subprocess, "run", symbolator_success)
    report = generator.generate(hdl, output)
    assert [entry["module"] for entry in report["generated"]] == ["M"]
    assert report["failed"] == report["omitted"] == []
    assert report["generated"][0]["undocumented_parameters"] == ["HIDDEN"]
    previous = snapshot_files(output)
    with pytest.raises(FileExistsError):
        generator.generate(hdl, output)
    assert snapshot_files(output) == previous


def test_diagram_nonzero_exit_is_recorded(tmp_path, monkeypatch):
    hdl = make_input(tmp_path)
    output = tmp_path / "failed"
    monkeypatch.setattr(
        generator.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 19, stdout="controlled failure\n"),
    )
    report = generator.generate(hdl, output)
    assert report["generated"] == []
    assert len(report["failed"]) == 1
    assert report["failed"][0]["diagram_exit_status"] == 19
    assert report["failed"][0]["reason"] == "ValueError: Symbolator exited 19; see M.diagram.log"
    assert (output / "M.diagram.log").read_text() == "controlled failure\n"
    assert not (output / "M.rst").exists()


def test_diagram_success_without_png_is_failure(tmp_path, monkeypatch):
    hdl = make_input(tmp_path)
    monkeypatch.setattr(
        generator.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="no image\n"),
    )
    report = generator.generate(hdl, tmp_path / "failed")
    assert report["generated"] == []
    assert len(report["failed"]) == 1
    assert report["failed"][0]["reason"].startswith("FileNotFoundError:")
    assert "__M.png" in report["failed"][0]["reason"]
    assert not (tmp_path / "failed" / "M.rst").exists()


def test_output_write_failure_records_exact_error_and_preserves_previous(tmp_path, monkeypatch):
    hdl = make_input(tmp_path)
    monkeypatch.setattr(generator.subprocess, "run", symbolator_success)
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "M.rst").write_text("earlier successful output\n")
    original_write = Path.write_text

    def fail_module_write(path, *args, **kwargs):
        if path == tmp_path / "failed" / "M.rst":
            raise PermissionError("controlled RST write failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_module_write)
    report = generator.generate(hdl, tmp_path / "failed")
    assert report["generated"] == []
    assert report["failed"][0]["reason"] == "PermissionError: controlled RST write failure"
    assert (previous / "M.rst").read_text() == "earlier successful output\n"


def test_coverage_write_failure_propagates_specific_error(tmp_path, monkeypatch):
    hdl = make_input(tmp_path)
    original_write = Path.write_text
    monkeypatch.setattr(generator.subprocess, "run", symbolator_success)

    def fail_coverage_write(path, *args, **kwargs):
        if path == tmp_path / "failed" / "coverage.json":
            raise PermissionError("controlled coverage write failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_coverage_write)
    with pytest.raises(PermissionError, match="controlled coverage write failure"):
        generator.generate(hdl, tmp_path / "failed")
    assert not (tmp_path / "failed" / "coverage.json").exists()


def test_invalid_declaration_is_reported_instead_of_partial_success(tmp_path):
    hdl = make_input(tmp_path)
    (hdl / "M.v").write_text("/// Module: M\nmodule M(input clk, broken declaration); endmodule\n")
    report = generator.generate(hdl, tmp_path / "failed")
    assert report["generated"] == report["omitted"] == []
    assert len(report["failed"]) == 1
    assert report["failed"][0]["reason"].startswith("ValueError:")
    assert "broken declaration" in report["failed"][0]["reason"]


def test_unreadable_source_records_specific_failure(tmp_path, monkeypatch):
    hdl = make_input(tmp_path)
    original = Path.read_bytes

    def fail_source_read(path):
        if path == hdl / "M.v":
            raise PermissionError("controlled source read failure")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", fail_source_read)
    report = generator.generate(hdl, tmp_path / "failed")
    assert report["generated"] == report["omitted"] == []
    assert report["failed"] == [{"source": "M.v", "reason": "PermissionError: controlled source read failure"}]


@pytest.fixture
def real_symbolator():
    executable = Path(sys.executable).parent / "symbolator"
    if executable.is_file():
        return str(executable)
    found = shutil.which("symbolator")
    assert found is not None, "The documentation environment must provide real Symbolator"
    return found


def run_cli(hdl, output, *arguments):
    command = [sys.executable, generator.__file__, "--hdl-dir", str(hdl),
               "--output-dir", str(output), *arguments]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output.parent / f"{output.name}.log").write_text(result.stdout, encoding="utf-8")
    (output.parent / f"{output.name}.command.json").write_text(
        json.dumps({"argv": command, "exit_status": result.returncode}, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def test_generator_cli_success_controlled_failures_and_restoration(tmp_path, real_symbolator):
    hdl = make_input(tmp_path)
    baseline = tmp_path / "baseline"
    result = run_cli(hdl, baseline, "--symbolator", real_symbolator)
    assert result.returncode == 0, result.stdout
    before = snapshot_files(baseline)

    (hdl / "M.v").write_text("/// Module: M\nmodule M(input clk, broken declaration); endmodule\n")
    parse_output = tmp_path / "parse-failed"
    result = run_cli(hdl, parse_output, "--symbolator", real_symbolator)
    assert result.returncode == 1, result.stdout
    report = json.loads((parse_output / "coverage.json").read_text())
    assert report["generated"] == report["omitted"] == []
    assert report["failed"][0]["reason"] == "ValueError: Unsupported ANSI port declaration: broken declaration"
    assert snapshot_files(baseline) == before

    (hdl / "M.v").write_text(SOURCE, encoding="utf-8")
    diagram_output = tmp_path / "diagram-failed"
    result = run_cli(hdl, diagram_output, "--symbolator", "/bin/false")
    assert result.returncode == 1, result.stdout
    report = json.loads((diagram_output / "coverage.json").read_text())
    assert report["generated"] == report["omitted"] == []
    assert report["failed"][0]["diagram_exit_status"] == 1
    assert report["failed"][0]["reason"] == "ValueError: Symbolator exited 1; see M.diagram.log"
    assert snapshot_files(baseline) == before

    restored = tmp_path / "restored"
    result = run_cli(hdl, restored, "--symbolator", real_symbolator)
    assert result.returncode == 0, result.stdout
    report = json.loads((restored / "coverage.json").read_text())
    assert [entry["module"] for entry in report["generated"]] == ["M"]
    assert report["failed"] == report["omitted"] == []
    assert snapshot_files(baseline) == before


def test_generator_cli_partial_coverage_requires_explicit_option(tmp_path, real_symbolator):
    hdl = make_input(tmp_path)
    (hdl / "N.v").write_text("module N(input clk); endmodule\n")
    strict = tmp_path / "strict"
    result = run_cli(hdl, strict, "--symbolator", real_symbolator)
    assert result.returncode == 1, result.stdout
    report = json.loads((strict / "coverage.json").read_text())
    assert [entry["module"] for entry in report["generated"]] == ["M"]
    assert report["failed"] == []
    assert report["omitted"][0]["module"] == "N"
    assert report["omitted"][0]["reason"] == "No structured /// or //! Module: comment"
    assert report["allow_partial"] is False

    partial = tmp_path / "partial"
    result = run_cli(hdl, partial, "--symbolator", real_symbolator, "--allow-partial")
    assert result.returncode == 0, result.stdout
    report = json.loads((partial / "coverage.json").read_text())
    assert report["allow_partial"] is True
    assert report["failed"] == []
    assert [entry["module"] for entry in report["generated"]] == ["M"]
    assert [entry["module"] for entry in report["omitted"]] == ["N"]