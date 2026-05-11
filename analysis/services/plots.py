"""Plotly figure builders.

Each function returns a JSON string ready to drop into a template via
``Plotly.newPlot(..., JSON.parse(figure_json))``.
"""
from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go

from .bandwidth import BandwidthScore
from .bootstrap import ConfidenceBand


_LAYOUT_DEFAULTS = dict(
    margin=dict(l=50, r=20, t=50, b=50),
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _to_json(fig: go.Figure) -> str:
    return json.dumps(fig, cls=PlotlyJSONEncoder)


class PlotlyJSONEncoder(json.JSONEncoder):
    """Serialize numpy arrays and Plotly objects."""

    def default(self, obj):  # type: ignore[override]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if hasattr(obj, "to_plotly_json"):
            return obj.to_plotly_json()
        return super().default(obj)


def occurrence_rate_figure(
    grid: np.ndarray,
    rates_by_bandwidth: dict[float, np.ndarray],
    event_dates: np.ndarray,
    title: str = "Occurrence rate by bandwidth",
) -> str:
    """One curve per bandwidth, plus event-date rug marks."""
    fig = go.Figure()
    for bw, rate in rates_by_bandwidth.items():
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=rate,
                mode="lines",
                name=f"Bandwidth {bw:g} yr",
                hovertemplate="t=%{x:.4f}<br>rate=%{y:.4f}<extra></extra>",
            )
        )
    if event_dates.size:
        fig.add_trace(
            go.Scatter(
                x=event_dates,
                y=np.zeros_like(event_dates),
                mode="markers",
                marker=dict(symbol="line-ns-open", size=10, color="rgba(220,50,50,0.8)"),
                name="Events",
                hovertemplate="event=%{x:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Events per year",
        **_LAYOUT_DEFAULTS,
    )
    return _to_json(fig)


def confidence_band_figure(
    band: ConfidenceBand,
    bandwidth_years: float,
    title: str | None = None,
) -> str:
    """Smoothed estimate plus 80% / 90% bootstrap percentile bands."""
    title = title or f"Bootstrap confidence band (h = {bandwidth_years:g} yr)"
    fig = go.Figure()
    # 90% band (5th-95th percentiles)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([band.grid, band.grid[::-1]]),
            y=np.concatenate([band.p95, band.p05[::-1]]),
            fill="toself",
            fillcolor="rgba(120,120,120,0.18)",
            line=dict(width=0),
            hoverinfo="skip",
            name="5th–95th percentile",
        )
    )
    # 80% band (10th-90th)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([band.grid, band.grid[::-1]]),
            y=np.concatenate([band.p90, band.p10[::-1]]),
            fill="toself",
            fillcolor="rgba(120,120,120,0.32)",
            line=dict(width=0),
            hoverinfo="skip",
            name="10th–90th percentile",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=band.grid,
            y=band.p50,
            mode="lines",
            line=dict(color="rgba(40,40,40,0.6)", dash="dot"),
            name="Median bootstrap",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=band.grid,
            y=band.point_estimate,
            mode="lines",
            line=dict(color="rgb(31,119,180)", width=2.5),
            name="Smoothed estimate",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Events per year",
        **_LAYOUT_DEFAULTS,
    )
    return _to_json(fig)


def bandwidth_score_figure(
    scores: list[BandwidthScore],
    chosen: BandwidthScore,
    title: str = "Bandwidth selection (lower score is better)",
) -> str:
    """MISE-like score vs bandwidth, with the local minima highlighted."""
    bws = np.array([s.bandwidth_years for s in scores])
    vals = np.array([s.score for s in scores])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bws,
            y=vals,
            mode="lines+markers",
            name="MISE score",
            line=dict(color="rgb(50,50,50)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[chosen.bandwidth_years],
            y=[chosen.score],
            mode="markers",
            marker=dict(color="red", size=12, symbol="x"),
            name=f"Chosen h = {chosen.bandwidth_years:g} yr",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Bandwidth (years)",
        yaxis_title="MISE score",
        **_LAYOUT_DEFAULTS,
    )
    return _to_json(fig)


def multi_threshold_figure(
    grid: np.ndarray,
    rates_by_threshold: dict[str, np.ndarray],
    title: str = "Occurrence rate by threshold",
) -> str:
    """Compare rates across indicator columns (e.g. Over3 vs Over4)."""
    fig = go.Figure()
    for label, rate in rates_by_threshold.items():
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=rate,
                mode="lines",
                name=label,
                hovertemplate="t=%{x:.4f}<br>rate=%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Events per year",
        **_LAYOUT_DEFAULTS,
    )
    return _to_json(fig)


def comparison_figure(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    title: str = "Comparison",
) -> str:
    """Overlay multiple (label, x, y) series on one chart for run comparison."""
    fig = go.Figure()
    for label, x, y in series:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label))
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Events per year",
        **_LAYOUT_DEFAULTS,
    )
    return _to_json(fig)
