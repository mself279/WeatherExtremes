"""Pre-analysis summaries of a GHCN station + element.

Pure-Python: takes a raw station DataFrame (output of ``read_station_csv``)
and produces small dataclasses + Plotly figure JSON for the explore page.

Each summary returns *both* the underlying data (so it's testable without
plotly) and a figure builder that converts it to a Plotly JSON string.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .elements import ELEMENTS, ElementSpec, threshold_column_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_for_element(
    raw_df: pd.DataFrame, element: str, *, drop_quality_failed: bool = True
) -> pd.DataFrame:
    """Filter to a single element, drop quality-failed rows, parse dates."""
    df = raw_df.loc[raw_df["ELEMENT"] == element].copy()
    if df.empty:
        raise ValueError(f"No rows for element {element}.")
    if drop_quality_failed and "Q_FLAG" in df.columns:
        df = df.loc[df["Q_FLAG"].fillna("").astype(str).str.strip() == ""]
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["DATE"])
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["VALUE"])
    return df


def _expected_obs_per_year(year: int) -> int:
    """Daily observations expected per calendar year (365 or 366)."""
    return 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365


def _apply_operator(values: np.ndarray, operator: str, threshold_raw: float) -> np.ndarray:
    if operator == ">":
        return values > threshold_raw
    if operator == ">=":
        return values >= threshold_raw
    if operator == "<":
        return values < threshold_raw
    if operator == "<=":
        return values <= threshold_raw
    raise ValueError(f"Unknown operator: {operator!r}")


# ---------------------------------------------------------------------------
# 1. Data completeness
# ---------------------------------------------------------------------------


@dataclass
class CompletenessSummary:
    element: str
    years: list[int]
    counts: list[int]      # observed obs per year
    expected: list[int]    # 365 or 366 per year
    pct: list[float]       # observed / expected * 100


def data_completeness(raw_df: pd.DataFrame, element: str) -> CompletenessSummary:
    df = _filter_for_element(raw_df, element)
    yr_counts = df.groupby(df["DATE"].dt.year).size()
    if yr_counts.empty:
        return CompletenessSummary(element=element, years=[], counts=[], expected=[], pct=[])
    min_yr, max_yr = int(yr_counts.index.min()), int(yr_counts.index.max())
    years = list(range(min_yr, max_yr + 1))
    counts = [int(yr_counts.get(y, 0)) for y in years]
    expected = [_expected_obs_per_year(y) for y in years]
    pct = [round(100.0 * c / e, 1) for c, e in zip(counts, expected)]
    return CompletenessSummary(
        element=element, years=years, counts=counts, expected=expected, pct=pct
    )


# ---------------------------------------------------------------------------
# 2. Value distribution
# ---------------------------------------------------------------------------


@dataclass
class DistributionSummary:
    element: str
    spec: ElementSpec
    values_display: np.ndarray
    quantiles: dict[str, float]
    threshold_counts: dict[float, int] = field(default_factory=dict)
    n: int = 0


def value_distribution(
    raw_df: pd.DataFrame,
    element: str,
    thresholds: Sequence[float] | None = None,
    operator: str = ">",
) -> DistributionSummary:
    df = _filter_for_element(raw_df, element)
    spec = ELEMENTS[element]
    raw_vals = df["VALUE"].to_numpy(dtype=float)
    display_vals = spec.raw_to_display(raw_vals)

    qs = np.quantile(display_vals, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    quantiles = dict(zip(
        ["min", "p10", "p25", "p50", "p75", "p90", "p99", "max"], qs.tolist()
    ))

    threshold_counts: dict[float, int] = {}
    if thresholds:
        for t in thresholds:
            t_raw = spec.display_to_raw(float(t))
            mask = _apply_operator(raw_vals, operator, t_raw)
            threshold_counts[float(t)] = int(mask.sum())

    return DistributionSummary(
        element=element,
        spec=spec,
        values_display=display_vals,
        quantiles=quantiles,
        threshold_counts=threshold_counts,
        n=int(len(display_vals)),
    )


# ---------------------------------------------------------------------------
# 3. Annual extremes (max for >-default elements, min for <-default elements)
# ---------------------------------------------------------------------------


@dataclass
class AnnualExtremesSummary:
    element: str
    spec: ElementSpec
    direction: str          # "max" or "min"
    years: list[int]
    values: list[float]     # in display units
    trend_slope: float      # display units per year
    trend_intercept: float


def annual_extremes(raw_df: pd.DataFrame, element: str) -> AnnualExtremesSummary:
    df = _filter_for_element(raw_df, element)
    spec = ELEMENTS[element]
    direction = "max" if spec.default_operator in (">", ">=") else "min"
    df = df.assign(YEAR=df["DATE"].dt.year)
    if direction == "max":
        yr = df.groupby("YEAR")["VALUE"].max()
    else:
        yr = df.groupby("YEAR")["VALUE"].min()

    raw_vals = yr.to_numpy(dtype=float)
    display_vals = spec.raw_to_display(raw_vals)
    years = [int(y) for y in yr.index.tolist()]

    if len(years) >= 2:
        slope, intercept = np.polyfit(years, display_vals, 1)
    else:
        slope = 0.0
        intercept = float(display_vals[0]) if display_vals.size else 0.0

    return AnnualExtremesSummary(
        element=element,
        spec=spec,
        direction=direction,
        years=years,
        values=display_vals.tolist(),
        trend_slope=float(slope),
        trend_intercept=float(intercept),
    )


# ---------------------------------------------------------------------------
# 4. Threshold-crossing frequency per year
# ---------------------------------------------------------------------------


@dataclass
class ThresholdFrequencySummary:
    element: str
    spec: ElementSpec
    operator: str
    years: list[int]
    counts_per_threshold: dict[float, list[int]]  # threshold (display) -> per-year counts


def threshold_frequencies(
    raw_df: pd.DataFrame,
    element: str,
    thresholds: Sequence[float],
    operator: str = ">",
) -> ThresholdFrequencySummary:
    spec = ELEMENTS[element]
    df = _filter_for_element(raw_df, element)
    df = df.assign(YEAR=df["DATE"].dt.year)
    raw_vals = df["VALUE"].to_numpy(dtype=float)
    if df.empty:
        return ThresholdFrequencySummary(
            element=element, spec=spec, operator=operator,
            years=[], counts_per_threshold={float(t): [] for t in thresholds},
        )
    min_yr, max_yr = int(df["YEAR"].min()), int(df["YEAR"].max())
    years = list(range(min_yr, max_yr + 1))
    counts_per_threshold: dict[float, list[int]] = {}
    for t in thresholds:
        t_raw = spec.display_to_raw(float(t))
        mask = _apply_operator(raw_vals, operator, t_raw)
        flagged_years = df.loc[mask, "YEAR"].to_numpy()
        per_year = pd.Series(flagged_years).value_counts().to_dict()
        counts_per_threshold[float(t)] = [int(per_year.get(y, 0)) for y in years]
    return ThresholdFrequencySummary(
        element=element, spec=spec, operator=operator,
        years=years, counts_per_threshold=counts_per_threshold,
    )


# ---------------------------------------------------------------------------
# Plot builders (Plotly JSON strings)
# ---------------------------------------------------------------------------


_LAYOUT_DEFAULTS = dict(
    margin=dict(l=50, r=20, t=50, b=50),
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _bar_color(pct: float) -> str:
    if pct >= 95:
        return "rgb(46,160,67)"     # green
    if pct >= 70:
        return "rgb(252,191,73)"    # amber
    return "rgb(220,68,68)"         # red


def completeness_figure(s: CompletenessSummary) -> str:
    import plotly.graph_objects as go

    colors = [_bar_color(p) for p in s.pct]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=s.years,
        y=s.counts,
        marker_color=colors,
        customdata=s.pct,
        hovertemplate="%{x}<br>%{y} obs (%{customdata:.0f}%)<extra></extra>",
        name="Observations",
    ))
    fig.update_layout(
        title=f"Data completeness — {s.element} (% of expected daily observations)",
        xaxis_title="Year",
        yaxis_title="Observations per year",
        **_LAYOUT_DEFAULTS,
    )
    return fig.to_json()


def distribution_figure(s: DistributionSummary, operator: str) -> str:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=s.values_display,
        nbinsx=60,
        marker_color="rgb(31,119,180)",
        opacity=0.85,
        name=f"{s.element} ({s.spec.display_unit})",
    ))
    op_sym = {">": ">", ">=": "≥", "<": "<", "<=": "≤"}.get(operator, operator)
    for t, count in s.threshold_counts.items():
        fig.add_vline(
            x=float(t),
            line_dash="dash",
            line_color="red",
            annotation_text=f"{op_sym} {t:g}: {count} events",
            annotation_position="top right",
        )
    fig.update_layout(
        title=(
            f"Value distribution — {s.element} "
            f"(n={s.n:,}, log-y scale)"
        ),
        xaxis_title=f"{s.spec.label} ({s.spec.display_unit})",
        yaxis_title="Frequency (log)",
        yaxis_type="log",
        **_LAYOUT_DEFAULTS,
    )
    return fig.to_json()


def annual_extremes_figure(s: AnnualExtremesSummary) -> str:
    import plotly.graph_objects as go

    direction_label = "Maximum" if s.direction == "max" else "Minimum"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.years,
        y=s.values,
        mode="markers+lines",
        line=dict(color="rgb(31,119,180)", width=1),
        marker=dict(size=6),
        name=f"Annual {direction_label.lower()}",
    ))
    if s.years:
        trend_y = [s.trend_intercept + s.trend_slope * y for y in s.years]
        fig.add_trace(go.Scatter(
            x=s.years,
            y=trend_y,
            mode="lines",
            line=dict(color="red", dash="dash"),
            name=(
                f"Linear trend "
                f"({s.trend_slope:+.4f} {s.spec.display_unit}/yr)"
            ),
        ))
    fig.update_layout(
        title=f"Annual {direction_label.lower()} — {s.element} ({s.spec.display_unit})",
        xaxis_title="Year",
        yaxis_title=f"{direction_label} ({s.spec.display_unit})",
        **_LAYOUT_DEFAULTS,
    )
    return fig.to_json()


def threshold_frequency_figure(s: ThresholdFrequencySummary) -> str:
    import plotly.graph_objects as go

    op_sym = {">": ">", ">=": "≥", "<": "<", "<=": "≤"}.get(s.operator, s.operator)
    fig = go.Figure()
    for thr, counts in s.counts_per_threshold.items():
        thr_str = str(int(thr) if thr == int(thr) else thr)
        fig.add_trace(go.Bar(
            x=s.years,
            y=counts,
            name=f"{op_sym} {thr_str} {s.spec.display_unit}",
        ))
    fig.update_layout(
        title=f"Annual count of threshold-crossing events — {s.element}",
        xaxis_title="Year",
        yaxis_title="Days per year",
        barmode="group",
        **_LAYOUT_DEFAULTS,
    )
    return fig.to_json()
