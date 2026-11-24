"""Check that recorded Yosys counts appear correctly in Plotly and Sphinx HTML."""
import hashlib
import json
from pathlib import Path
import re
from shutil import copyfile
import subprocess
import sys

import pytest

from doc.source._ext.yosys_resources import make_figure


DOC = Path(__file__).resolve().parent
REPORT = DOC / "source/test/pe_dense_resources.json"
EXPECTED = {
    "CC_ADDF": 230,
    "CC_BRAM_20K": 2,
    "CC_DFF": 445,
    "CC_L2T4": 207,
    "CC_L2T5": 102,
    "CC_LUT1": 31,
    "CC_LUT2": 207,
    "CC_MULT": 3,
    "CC_MX2": 129,
    "CC_MX4": 82,
}


@pytest.fixture
def report():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def assert_chart_counts(names, values):
    assert len(names) == len(values) == len(EXPECTED)
    assert dict(zip(names, values)) == EXPECTED


def test_recorded_snapshot_and_chart_counts(report):
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == (
        "e7f905757c6d68d3d8a4f9f17db56c515d87c1caf05fedbb83873a2be446deb7"
    )
    assert report["design"]["num_cells_by_type"] == EXPECTED
    assert report["design"]["num_cells"] == 1438

    figure = make_figure(report)
    assert len(figure.data) == 1
    bars = figure.data[0]
    assert bars.orientation == "h"
    assert_chart_counts(bars.y, bars.x)
    assert sum(bars.x) == 1438


def test_design_counts_are_not_added_to_module_counts(report):
    report["modules"][r"\PE"]["num_cells_by_type"] = {"DUPLICATE": 10000}
    bars = make_figure(report).data[0]
    assert_chart_counts(bars.y, bars.x)


@pytest.mark.parametrize("counts", [
    {}, {"A": -1}, {"A": True}, {"A": 1.5}, {"A": "2"}, {1: 2},
])
def test_invalid_cell_counts_are_rejected(report, counts):
    report["design"]["num_cells_by_type"] = counts
    with pytest.raises(ValueError, match="cell counts"):
        make_figure(report)


def test_inconsistent_total_is_rejected(report):
    report["design"]["num_cells"] += 1
    with pytest.raises(ValueError, match="total"):
        make_figure(report)


def test_real_sphinx_embeds_recorded_values(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    copyfile(REPORT, source / REPORT.name)
    copyfile(DOC / "source/test/index.rst", source / "index.rst")
    (source / "conf.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(DOC / 'source/_ext')!r})\n"
        "extensions = ['yosys_resources']\n",
        encoding="utf-8",
    )

    output = tmp_path / "html"
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", "-W", str(source), str(output)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 0, result.stdout

    html = (output / "index.html").read_text(encoding="utf-8")
    assert 'id="pe-dense-resources"' in html
    assert "plotly.js v" in html
    external_scripts = re.findall(r"""<script\b[^>]*\bsrc=["']([^"']+)["']""", html)
    assert not any("plotly" in script.lower() for script in external_scripts)

    chart = re.search(r'Plotly\.newPlot\(\s*"pe-dense-resources"\s*,\s*', html)
    assert chart is not None, "Missing Plotly chart for the dense PE"
    traces, _ = json.JSONDecoder().raw_decode(html, chart.end())
    assert len(traces) == 1
    assert_chart_counts(traces[0]["y"], traces[0]["x"])

    downloads = list((output / "_downloads").rglob(REPORT.name))
    assert len(downloads) == 1, "Expected one downloadable Yosys report"
    downloaded_report = downloads[0]
    href = downloaded_report.relative_to(output).as_posix()
    assert f'href="{href}"' in html
    assert downloaded_report.read_bytes() == REPORT.read_bytes()