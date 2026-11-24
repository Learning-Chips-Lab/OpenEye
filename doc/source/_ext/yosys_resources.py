"""Render a recorded Yosys cell-count report as an inline Plotly chart."""
import json
from pathlib import Path

from docutils import nodes
import plotly.graph_objects as go
from sphinx.util.docutils import SphinxDirective


def make_figure(report):
    design = report['design']
    counts = design['num_cells_by_type']
    if not isinstance(counts, dict) or not counts or any(
            not isinstance(name, str) or type(count) is not int or count < 0
            for name, count in counts.items()):
        raise ValueError('Expected nonnegative integer cell counts by type')
    if type(design['num_cells']) is not int or sum(counts.values()) != design['num_cells']:
        raise ValueError('Cell counts do not match the reported total')

    names = sorted(counts)
    values = [counts[name] for name in names]

    figure = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        text=values,
        textposition='auto',
        hovertemplate='%{y}: %{x:d} cells<extra></extra>',
    ))
    figure.update_layout(
        title=f"Recorded dense PE: {design['num_cells']:,} GateMate cells",
        xaxis_title='Cell count', yaxis_title='Cell type',
        height=480, margin=dict(l=110, r=20, t=60, b=50))
    figure.update_xaxes(rangemode='tozero', tickformat='d')
    return figure


class YosysResources(SphinxDirective):
    required_arguments = 1

    def run(self):
        _, filename = self.env.relfn2path(self.arguments[0])
        self.env.note_dependency(filename)
        try:
            report = json.loads(Path(filename).read_text(encoding='utf-8'))
            figure = make_figure(report)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise self.error(
                f"Cannot render Yosys resources from {self.arguments[0]!r}: {error}"
            ) from error

        html = figure.to_html(full_html=False, include_plotlyjs=True,
                              div_id='pe-dense-resources', config={'responsive': True})
        return [nodes.raw('', html, format='html')]


def setup(app):
    app.add_directive('yosys-resources', YosysResources)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}