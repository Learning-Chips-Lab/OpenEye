"""Regression checks for real HTML validation; every output lives under tmp_path."""
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from doc.validate_output import validate_output


@pytest.fixture
def rendered(tmp_path):
    generated = tmp_path / "generated"
    html = tmp_path / "html"
    generated.mkdir()
    (html / "verilog").mkdir(parents=True)
    (html / "_images").mkdir()
    declaration = {"kind": "parameter", "name": "W", "signature": "localparam integer W = 2*4",
                   "description": "Actual default width"}
    coverage = {"generated": [{"module": "M", "description": "Original module description",
                               "declarations": [declaration]}], "omitted": [], "failed": []}
    (generated / "coverage.json").write_text(json.dumps(coverage))
    Image.new("RGB", (8, 8), "white").save(generated / "__M.png")
    (html / "_images" / "__M.png").write_bytes((generated / "__M.png").read_bytes())
    (html / "index.html").write_text("<html><body>Documentation</body></html>")
    (html / "verilog" / "index.html").write_text('<html><body><a href="M.html">M</a></body></html>')
    (html / "verilog" / "M.html").write_text('''<html><body>
<dl class="verilog module"><dt id="verilog-m">module M()</dt><dd>
<p>Original module description</p><img src="../_images/__M.png">
<dl class="verilog parameter"><dt id="verilog-m-w">localparam integer W = 2*4<a class="headerlink">¶</a></dt>
<dd><p>Actual default width</p></dd></dl>
</dd></dl></body></html>''')
    return generated, html


def test_valid_output(rendered):
    assert validate_output(*rendered) == {"modules": 1, "declarations": 1, "diagrams": 1, "local_images": 1}


@pytest.mark.parametrize(("before", "after", "message"), [
    ('id="verilog-m-w"', 'id="wrong"', "Missing or ambiguous domain anchor for M.W"),
    ("2*4", "24", "Rendered declaration signature differs for M.W"),
    ("Actual default width", "Filler", "Rendered description differs for M.W"),
    ("Original module description", "Filler", "Rendered description differs for M"),
    ('<img src="../_images/__M.png">', "", "Missing required diagram reference for M"),
    ('src="../_images/__M.png"', 'src="../_images/missing.png"', "Missing referenced image"),
])
def test_specific_html_failures(rendered, before, after, message):
    generated, html = rendered
    page = html / "verilog" / "M.html"
    page.write_text(page.read_text().replace(before, after))
    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        validate_output(generated, html)


def test_corrupt_diagram(rendered):
    generated, html = rendered
    (html / "_images" / "__M.png").write_bytes(b"not a PNG")
    with pytest.raises(OSError, match="cannot identify image file"):
        validate_output(generated, html)


def test_stale_diagram(rendered):
    generated, html = rendered
    Image.new("RGB", (8, 8), "black").save(html / "_images" / "__M.png")
    with pytest.raises(ValueError, match="Rendered diagram differs from generated diagram for M"):
        validate_output(generated, html)


def test_missing_module_page(rendered):
    generated, html = rendered
    (html / "verilog" / "M.html").unlink()
    with pytest.raises(ValueError, match="Missing required module HTML: M"):
        validate_output(generated, html)


def test_failed_coverage(rendered):
    generated, html = rendered
    path = generated / "coverage.json"
    coverage = json.loads(path.read_text())
    coverage["failed"] = [{"source": "broken.v", "reason": "unsupported"}]
    path.write_text(json.dumps(coverage))
    with pytest.raises(ValueError, match="generation failures"):
        validate_output(generated, html)


@pytest.fixture
def complete_rendered(rendered):
    """Coverage includes every hardware port, including an undocumented one."""
    generated, html = rendered
    coverage_path = generated / "coverage.json"
    coverage = json.loads(coverage_path.read_text())
    entry = coverage["generated"][0]
    entry["ports"] = ["clk_i", "wr_mask_i"]
    entry["declarations"].append({"kind": "port", "name": "clk_i",
        "signature": "input unsigned clk_i", "description": "Clock input"})
    coverage_path.write_text(json.dumps(coverage))
    page = html / "verilog" / "M.html"
    page.write_text(page.read_text().replace("module M()", "module M(clk_i, wr_mask_i)").replace(
        "</dd></dl></body>", '<dl class="verilog port"><dt id="verilog-m-clk-i">input unsigned clk_i</dt>'
        '<dd><p>Clock input</p></dd></dl></dd></dl></body>'))
    return generated, html


def test_complete_hardware_port_list(complete_rendered):
    assert validate_output(*complete_rendered)["declarations"] == 2


@pytest.mark.parametrize("heading", [
    "module M(clk_i)", "module M(clk_i, wr_mask_i, extra)",
    "module M(wr_mask_i, clk_i)",
])
def test_module_heading_requires_all_ports_in_hardware_order(complete_rendered, heading):
    generated, html = complete_rendered
    page = html / "verilog" / "M.html"
    page.write_text(page.read_text().replace("module M(clk_i, wr_mask_i)", heading))
    with pytest.raises(ValueError, match="Rendered module signature differs for M"):
        validate_output(generated, html)


@pytest.mark.parametrize("case", ["valid", "wrong_navigation", "missing_page"])
def test_coverage_page_name_controls_page_and_navigation(rendered, case):
    generated, html = rendered
    coverage_path = generated / "coverage.json"
    coverage = json.loads(coverage_path.read_text())
    coverage["generated"][0].update(module="index", page="module-index")
    coverage_path.write_text(json.dumps(coverage))
    (generated / "__M.png").rename(generated / "__index.png")
    (html / "_images" / "__M.png").rename(html / "_images" / "__index.png")
    original_page = html / "verilog" / "M.html"
    original_page.write_text(original_page.read_text().replace("module M()", "module index()").replace(
        "verilog-m", "verilog-index").replace("__M.png", "__index.png"))
    original_page.rename(html / "verilog" / "module-index.html")
    if case != "wrong_navigation":
        (html / "verilog" / "index.html").write_text('<a href="module-index.html">index</a>')
    if case == "valid":
        assert validate_output(generated, html)["modules"] == 1
    elif case == "missing_page":
        (html / "verilog" / "module-index.html").unlink()
        with pytest.raises(ValueError, match="Missing required module HTML: index"):
            validate_output(generated, html)
    else:
        with pytest.raises(ValueError, match="Module absent from generated HTML index: index"):
            validate_output(generated, html)


@pytest.mark.parametrize(("before", "after", "label"), [
    ("Original module description", "Original module", "M"),
    ("Actual default width", "Actual default", "M.W"),
    ("Clock input", "Clock", "M.clk_i"),
    ("Clock input", "Clockinput", "M.clk_i"),
    ("Clock input", "Clock input is disconnected", "M.clk_i"),
])
def test_own_prose_must_be_complete_and_exact(complete_rendered, before, after, label):
    generated, html = complete_rendered
    page = html / "verilog" / "M.html"
    page.write_text(page.read_text().replace(before, after))
    with pytest.raises(ValueError, match="Rendered description differs for " + label.replace(".", r"\.")):
        validate_output(generated, html)


def test_nested_port_prose_cannot_replace_module_description(complete_rendered):
    generated, html = complete_rendered
    page = html / "verilog" / "M.html"
    page.write_text(page.read_text().replace("<p>Original module description</p>", "").replace(
        "<p>Clock input</p>", "<p>Clock input</p><p>Original module description</p>"))
    with pytest.raises(ValueError, match="Rendered description differs for M$"):
        validate_output(generated, html)


def test_prose_accepts_wrapping_lists_emphasis_and_typography(complete_rendered):
    generated, html = complete_rendered
    coverage_path = generated / "coverage.json"
    coverage = json.loads(coverage_path.read_text())
    coverage["generated"][0]["declarations"][0]["description"] = (
        'Actual **default** width.\n\n- First item\n- Second item\n\nUse "generic" -- it works...\n\nNext --- detail.')
    coverage_path.write_text(json.dumps(coverage))
    page = html / "verilog" / "M.html"
    page.write_text(page.read_text().replace("Original module description", "Original\nmodule description").replace(
        "<p>Actual default width</p>", '<p>Actual <strong>default</strong> width.</p><ul>'
        '<li>First item</li><li>Second item</li></ul><p>Use “generic” – it works…</p><p>Next — detail.</p>').replace(
        "Clock input", "Clock <em>input</em>"))
    assert validate_output(generated, html)["declarations"] == 2


@pytest.mark.parametrize(("expected", "actual", "valid"), [
    ("localparam [3:0] W = 2*4", "localparam [3 : 0] W = 2 * 4", True),
    ('localparam W = "generic"', 'localparam W = "gener ic"', False),
    ('localparam W = "two words"', 'localparam W="two words"', True),
    (r'localparam W = "say \"hi\""', r'localparam W="say \"hi\""', True),
])
def test_signature_token_and_string_boundaries(rendered, expected, actual, valid):
    generated, html = rendered
    coverage_path = generated / "coverage.json"
    coverage = json.loads(coverage_path.read_text())
    coverage["generated"][0]["declarations"][0]["signature"] = expected
    coverage_path.write_text(json.dumps(coverage))
    page = html / "verilog" / "M.html"
    page.write_text(page.read_text().replace("localparam integer W = 2*4", actual))
    if valid:
        assert validate_output(generated, html)["declarations"] == 1
    else:
        with pytest.raises(ValueError, match=r"Rendered declaration signature differs for M\.W"):
            validate_output(generated, html)


def test_signature_preserves_identifier_boundaries(complete_rendered):
    generated, html = complete_rendered
    page = html / "verilog" / "M.html"
    page.write_text(page.read_text().replace("input unsigned clk_i", "input unsignedclk_i"))
    with pytest.raises(ValueError, match=r"Rendered declaration signature differs for M\.clk_i"):
        validate_output(generated, html)


def test_real_sphinx_domain_rendering(tmp_path):
    """A real domain parse must produce the identities the validator requires."""
    source = tmp_path / "source"
    generated = source / "verilog"
    generated.mkdir(parents=True)
    (source / "conf.py").write_text("extensions = ['sphinx_verilog_domain']\nmaster_doc = 'index'\n")
    (source / "index.rst").write_text("Docs\n====\n\n.. toctree::\n\n   verilog/index\n")
    (generated / "index.rst").write_text("Modules\n=======\n\n.. toctree::\n\n   M\n")
    signature = "localparam integer W = 2*4"
    quoted_signature = 'parameter MODE = "generic"'
    description = 'Preserved **parameter** text.\n\n- First item\n- Second item\n\nUse "portable" -- it works...'
    (generated / "M.rst").write_text(
        "M\n=\n\n.. verilog:module:: module M(wire_ready, wr_mask_i)\n\n"
        "    Original module\n    description\n\n    .. image:: __M.png\n\n"
        f"    .. verilog:parameter:: {signature}\n\n"
        '        Preserved **parameter** text.\n\n        - First item\n        - Second item\n\n'
        '        Use "portable" -- it works...\n\n'
        f"    .. verilog:parameter:: {quoted_signature}\n\n"
        "        Implementation selection\n")
    Image.new("RGB", (8, 8), "white").save(generated / "__M.png")
    (generated / "coverage.json").write_text(json.dumps({"generated": [{"module": "M", "description": "Original module description",
        "ports": ["wire_ready", "wr_mask_i"],
        "declarations": [{"kind": "parameter", "name": "W", "signature": signature, "description": description},
                         {"kind": "parameter", "name": "MODE", "signature": quoted_signature,
                          "description": "Implementation selection"}]}], "failed": [], "omitted": []}))
    html = tmp_path / "html"
    result = subprocess.run([sys.executable, "-m", "sphinx", "-b", "html", str(source), str(html)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout
    assert "ERROR" not in result.stdout, result.stdout
    assert validate_output(generated, html)["declarations"] == 2
    page = html / "verilog" / "M.html"
    page.write_text(page.read_text().replace("generic", "gener ic"))
    with pytest.raises(ValueError, match=r"Rendered declaration signature differs for M\.MODE"):
        validate_output(generated, html)


def test_real_sphinx_actual_documented_prose(tmp_path):
    """Preserve real PE and adder prose through domain HTML rendering."""
    from doc.verilog_doc_parser.docstring_parser import parse_documentation
    from doc.verilog_doc_parser.moduledef_parser import parse_module_def
    from doc.verilog_doc_parser.rst_doc_generator import DEFINES, declarations, generate_rst

    source = tmp_path / "source"
    generated = source / "verilog"
    generated.mkdir(parents=True)
    (source / "conf.py").write_text("extensions = ['sphinx_verilog_domain']\nmaster_doc = 'index'\n")
    (source / "index.rst").write_text("Docs\n====\n\n.. toctree::\n\n   verilog/index\n")
    (generated / "index.rst").write_text("Modules\n=======\n\n.. toctree::\n\n   PE\n   adder\n")
    entries = []
    for module in ("PE", "adder"):
        text = (Path(__file__).resolve().parents[1] / "hdl" / f"{module}.v").read_text()
        documentation = parse_documentation(text)
        definition = parse_module_def(text, defines=DEFINES)
        (generated / f"{module}.rst").write_text(generate_rst(documentation, definition))
        Image.new("RGB", (8, 8), "white").save(generated / f"__{module}.png")
        entries.append({"module": module, "description": documentation.description,
                        "ports": list(definition.ports),
                        "declarations": declarations(documentation, definition)})
    (generated / "coverage.json").write_text(json.dumps({"generated": entries, "failed": [], "omitted": []}))
    html = tmp_path / "html"
    result = subprocess.run([sys.executable, "-m", "sphinx", "-b", "html", str(source), str(html)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout
    assert "ERROR" not in result.stdout, result.stdout
    assert validate_output(generated, html)["modules"] == 2
