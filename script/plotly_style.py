"""Shared Plotly styling for the paper's result figures.

One consistent palette, marker map, and layout so every figure reads as one visual
family, exported to static PDF via kaleido. Replaces the earlier plotnine setup.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # noqa: F401  (re-exported for figure scripts)

PALETTE = {
    "two-stage": "#7f7f7f",
    "SPO+":      "#1f77b4",
    "IMLE":      "#9467bd",
    "SFGE":      "#2ca02c",
    "PolyStep":  "#d62728",
    "PolyStep (ours)": "#d62728",
    "BD":          "#8c564b",
    "FIG":         "#e377c2",
    "PredGNN":     "#17becf",
    "AvgTSP":      "#bcbd22",
    "DistrictNet": "#ff7f0e",
}

SYMBOL = {
    "two-stage": "circle", "SPO+": "square", "IMLE": "diamond",
    "SFGE": "triangle-up", "PolyStep": "star", "PolyStep (ours)": "star",
    "BD": "circle", "FIG": "square", "PredGNN": "triangle-up",
    "AvgTSP": "triangle-down", "DistrictNet": "diamond",
}

FONT = "Times New Roman, Times, serif"
GRID = "#E6E6E6"
AXIS = "#666666"


def finalize(fig, width, height, legend=True, base_size=15):
    """Apply the shared white/serif publication theme and fixed canvas size."""
    fig.update_layout(
        template="simple_white",
        width=width, height=height,
        font=dict(family=FONT, size=base_size, color="#1a1a1a"),
        margin=dict(l=72, r=24, t=42, b=58),
        showlegend=legend,
        legend=dict(title_text="", orientation="v", x=1.005, y=1.0,
                    xanchor="left", yanchor="top", font=dict(size=base_size - 2),
                    bgcolor="rgba(255,255,255,0.6)"),
        bargap=0.18, bargroupgap=0.08,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=0.6, zeroline=False,
                     ticks="outside", ticklen=4, linecolor=AXIS, linewidth=0.8,
                     title_standoff=8)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=0.6, zeroline=False,
                     ticks="outside", ticklen=4, linecolor=AXIS, linewidth=0.8,
                     title_standoff=8)
    return fig


def line_trace(x, y, method, err=None, show_legend=True, width=2.2, msize=8, name=None):
    """A styled line+marker trace for one method, with optional error bars."""
    return go.Scatter(
        x=list(x), y=list(y), mode="lines+markers",
        name=name or method, legendgroup=method, showlegend=show_legend,
        line=dict(color=PALETTE.get(method, "#444"), width=width),
        marker=dict(color=PALETTE.get(method, "#444"), symbol=SYMBOL.get(method, "circle"),
                    size=msize, line=dict(width=0)),
        error_y=(dict(type="data", array=list(err), visible=True, thickness=1.0,
                      width=2.5, color=PALETTE.get(method, "#444")) if err is not None else None),
    )


def bar_trace(x, y, group, color, err=None, show_legend=True):
    """A styled grouped-bar trace."""
    return go.Bar(
        x=list(x), y=list(y), name=group, legendgroup=group, showlegend=show_legend,
        marker=dict(color=color, line=dict(width=0)),
        error_y=(dict(type="data", array=list(err), visible=True, thickness=1.0, width=3,
                      color="#333") if err is not None else None),
    )


def save(fig, path):
    fig.write_image(path)
    print("wrote", path)
