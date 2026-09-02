"""Charts for the app and the README.

Palette roles: the model's score owns categorical slot 1 (blue); ground-truth
attack periods are drawn as a red status band, always with a legend entry, so
the state is never carried by colour alone; the alert threshold is a neutral
rule. Marks are thin, gridlines are solid hairlines a shade off the surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LIGHT = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink_soft": "#52514e", "grid": "#e8e7e3",
    "score": "#2a78d6", "attack": "rgba(227,73,72,0.18)", "attack_line": "#e34948",
    "alt": "#1baf7a", "warn": "#eda100",
}
DARK = {
    "surface": "#1a1a19", "ink": "#ffffff", "ink_soft": "#c3c2b7", "grid": "#333330",
    "score": "#3987e5", "attack": "rgba(230,103,103,0.22)", "attack_line": "#e66767",
    "alt": "#199e70", "warn": "#c98500",
}


def palette(dark: bool = False) -> dict:
    return DARK if dark else LIGHT


def _axes(c: dict, x_title: str = "", y_title: str = "") -> dict:
    common = dict(
        automargin=True, showgrid=True, gridcolor=c["grid"], gridwidth=1, zeroline=False,
        linecolor=c["grid"], linewidth=1, ticks="outside", tickcolor=c["grid"],
        tickfont=dict(color=c["ink_soft"], size=11),
    )
    return {"xaxis": {**common, "title_text": x_title},
            "yaxis": {**common, "title_text": y_title}}


def timeline_figure(
    times: np.ndarray,
    scores: np.ndarray,
    truth: np.ndarray,
    threshold: float,
    labels: dict[str, str],
    dark: bool = False,
):
    """Attack probability through the day, with the true attack periods shaded."""
    import plotly.graph_objects as go

    c = palette(dark)
    order = np.argsort(times)
    times, scores, truth = np.asarray(times)[order], np.asarray(scores)[order], np.asarray(truth)[order]

    # The x-axis is the window sequence, not wall-clock time. Flow volume varies
    # by orders of magnitude across the day, so on a time axis the thousand
    # windows of an attack burst collapse into a sliver while a quiet morning
    # takes half the width. One window, one step; the clock stays in the hover.
    x = np.arange(len(scores))
    clock = pd.to_datetime(times).strftime("%H:%M:%S")

    fig = go.Figure()

    # Ground-truth attack periods, as contiguous shaded runs.
    edges = np.flatnonzero(np.diff(np.r_[0, truth, 0]) != 0).reshape(-1, 2)
    for a, b in edges:
        fig.add_vrect(x0=a - 0.5, x1=min(b, len(x) - 1) + 0.5,
                      fillcolor=c["attack"], line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=[x[0]], y=[None], mode="markers", name=labels["truth"],
        marker=dict(color=c["attack_line"], size=10, symbol="square"), showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=x, y=scores, mode="lines", name=labels["score"],
        line=dict(color=c["score"], width=2), customdata=clock,
        hovertemplate="window %{x} · %{customdata}<br>p=%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(
        y=threshold, line=dict(color=c["ink_soft"], width=1, dash="dash"),
        annotation_text=labels["threshold"], annotation_position="top left",
        annotation_font=dict(color=c["ink_soft"], size=11),
    )

    fig.update_layout(
        paper_bgcolor=c["surface"], plot_bgcolor=c["surface"], hovermode="x unified",
        margin=dict(l=0, r=8, t=8, b=0), height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(color=c["ink_soft"], size=11), bgcolor="rgba(0,0,0,0)"),
        **_axes(c, labels.get("x", "window (in time order)"), labels.get("y", "P(attack)")),
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def comparison_figure(summary: pd.DataFrame, labels: dict[str, str], dark: bool = False):
    """PR-AUC per architecture with the spread across seeds as an error bar."""
    import plotly.graph_objects as go

    c = palette(dark)
    df = summary[summary.index != "prevalence_floor"]
    floor = summary.loc["prevalence_floor", "pr_auc_mean"] if "prevalence_floor" in summary.index else None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=df["pr_auc_mean"],
        error_y=dict(type="data", array=df.get("pr_auc_std"), color=c["ink_soft"], width=6),
        marker=dict(color=c["score"], line=dict(width=0)),
        text=[f"{v:.3f}" for v in df["pr_auc_mean"]], textposition="outside",
        textfont=dict(color=c["ink_soft"], size=11), name="PR-AUC",
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))
    if floor is not None:
        fig.add_hline(y=floor, line=dict(color=c["attack_line"], width=1, dash="dot"),
                      annotation_text=labels.get("floor", "prevalence floor"),
                      annotation_position="top right",
                      annotation_font=dict(color=c["ink_soft"], size=11))
    fig.update_layout(
        paper_bgcolor=c["surface"], plot_bgcolor=c["surface"], showlegend=False,
        margin=dict(l=0, r=8, t=24, b=0), height=320,
        **_axes(c, "", "PR-AUC"),
    )
    return fig
