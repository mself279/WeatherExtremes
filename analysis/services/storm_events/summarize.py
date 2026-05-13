"""Summaries + Plotly figures for the Storm Events exploration page."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Summaries (pure data, no Plotly)
# ---------------------------------------------------------------------------


@dataclass
class EventsSummary:
    n_events: int
    year_first: int | None
    year_last: int | None
    total_injuries: int
    total_deaths: int
    total_damage_property_usd: float
    total_damage_crops_usd: float
    top_counties: list[tuple[str, int]]      # (county, count) sorted desc, top 10
    annual_counts: list[tuple[int, int]]     # (year, count) full range
    annual_damage_property: list[tuple[int, float]]  # (year, USD) full range
    monthly_counts: list[tuple[int, int]]    # (month 1-12, count)
    magnitude_breakdown: dict[str, int] = field(default_factory=dict)


def summarize(df: pd.DataFrame, *, magnitude_label: str = "") -> EventsSummary:
    """Build an :class:`EventsSummary` from the parsed-events DataFrame."""
    if df.empty:
        return EventsSummary(
            n_events=0, year_first=None, year_last=None,
            total_injuries=0, total_deaths=0,
            total_damage_property_usd=0.0, total_damage_crops_usd=0.0,
            top_counties=[], annual_counts=[], annual_damage_property=[],
            monthly_counts=[], magnitude_breakdown={},
        )

    years = df["year"].dropna()
    year_first = int(years.min()) if len(years) else None
    year_last = int(years.max()) if len(years) else None

    # Annual counts across the full year range (so years with zero events show as zeros).
    if year_first is not None and year_last is not None:
        all_years = list(range(year_first, year_last + 1))
        yr_series = df["year"].astype("Int64").value_counts().to_dict()
        annual = [(y, int(yr_series.get(y, 0))) for y in all_years]
        # Annual property-damage totals over the same year range. NaN damages
        # contribute 0 so years with reports-without-damage stay at 0.
        damage_by_year = (
            df.assign(_d=df["damage_property_usd"].fillna(0.0))
              .groupby(df["year"].astype("Int64"))["_d"]
              .sum()
              .to_dict()
        )
        annual_damage = [(y, float(damage_by_year.get(y, 0.0))) for y in all_years]
    else:
        annual = []
        annual_damage = []

    monthly = []
    if "month" in df.columns:
        m_counts = df["month"].dropna().astype(int).value_counts().to_dict()
        monthly = [(m, int(m_counts.get(m, 0))) for m in range(1, 13)]

    county_counts = (
        df["cz_name"].fillna("").replace("", "(unknown)").value_counts().head(10)
    )
    top_counties = [(str(name).title(), int(count)) for name, count in county_counts.items()]

    # Magnitude breakdown depends on event type. For tornadoes the TOR_F_SCALE
    # column is the relevant categorical; otherwise we bucket the numeric
    # ``magnitude`` value.
    mag_breakdown: dict[str, int] = {}
    if df["tor_f_scale"].astype(str).str.strip().ne("").any():
        scales = df["tor_f_scale"].astype(str).str.strip().replace("", "Unknown")
        mag_breakdown = scales.value_counts().to_dict()
        mag_breakdown = {str(k): int(v) for k, v in mag_breakdown.items()}
    elif df["magnitude"].notna().any():
        # Bucket numeric magnitudes by half-integers for hail, integers for wind.
        bins = np.round(df["magnitude"].dropna() * 2) / 2.0
        bucketed = bins.astype(float).value_counts().sort_index()
        mag_breakdown = {f"{k:g}": int(v) for k, v in bucketed.items()}

    return EventsSummary(
        n_events=int(len(df)),
        year_first=year_first,
        year_last=year_last,
        total_injuries=int(df["injuries_direct"].fillna(0).sum()),
        total_deaths=int(df["deaths_direct"].fillna(0).sum()),
        total_damage_property_usd=float(df["damage_property_usd"].fillna(0).sum()),
        total_damage_crops_usd=float(df["damage_crops_usd"].fillna(0).sum()),
        top_counties=top_counties,
        annual_counts=annual,
        annual_damage_property=annual_damage,
        monthly_counts=monthly,
        magnitude_breakdown=mag_breakdown,
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

_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def annual_counts_figure(s: EventsSummary, *, title: str) -> str:
    import plotly.graph_objects as go

    if not s.annual_counts:
        return _empty_figure(title)
    years = [y for y, _ in s.annual_counts]
    counts = [c for _, c in s.annual_counts]
    fig = go.Figure(go.Bar(
        x=years, y=counts,
        marker_color="rgb(31,119,180)",
        hovertemplate="%{x}: %{y}<extra></extra>",
        name="Events per year",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Events per year",
        **_LAYOUT_DEFAULTS,
    )
    return fig.to_json()


def annual_damage_property_figure(s: EventsSummary, *, title: str) -> str:
    """Annual property damage (USD) per year. The view should only render this
    figure when ``s.total_damage_property_usd > 0``; this function will render
    an all-zero chart otherwise rather than fail."""
    import plotly.graph_objects as go

    if not s.annual_damage_property:
        return _empty_figure(title)
    years = [y for y, _ in s.annual_damage_property]
    damages = [d for _, d in s.annual_damage_property]
    fig = go.Figure(go.Bar(
        x=years, y=damages,
        marker_color="rgb(214,39,40)",
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        name="Property damage",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Property damage (USD)",
        **_LAYOUT_DEFAULTS,
    )
    return fig.to_json()


def monthly_counts_figure(s: EventsSummary, *, title: str) -> str:
    import plotly.graph_objects as go

    if not s.monthly_counts:
        return _empty_figure(title)
    months = [_MONTH_NAMES[m - 1] for m, _ in s.monthly_counts]
    counts = [c for _, c in s.monthly_counts]
    fig = go.Figure(go.Bar(
        x=months, y=counts,
        marker_color="rgb(255,127,14)",
        hovertemplate="%{x}: %{y}<extra></extra>",
        name="Events per month",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Total events",
        **_LAYOUT_DEFAULTS,
    )
    return fig.to_json()


def magnitude_breakdown_figure(s: EventsSummary, *, title: str, unit: str = "") -> str:
    import plotly.graph_objects as go

    if not s.magnitude_breakdown:
        return _empty_figure(title)
    # Try numeric sort; fall back to lexicographic.
    items = list(s.magnitude_breakdown.items())
    try:
        items.sort(key=lambda kv: float(str(kv[0]).lstrip("FE").strip()))
    except ValueError:
        items.sort(key=lambda kv: str(kv[0]))
    labels = [k for k, _ in items]
    counts = [v for _, v in items]
    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker_color="rgb(44,160,44)",
        hovertemplate="%{x}: %{y}<extra></extra>",
        name="Events by magnitude",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=f"Magnitude{(' (' + unit + ')') if unit else ''}",
        yaxis_title="Events",
        **_LAYOUT_DEFAULTS,
    )
    return fig.to_json()


def top_counties_figure(s: EventsSummary, *, title: str) -> str:
    import plotly.graph_objects as go

    if not s.top_counties:
        return _empty_figure(title)
    counties = [c for c, _ in s.top_counties][::-1]   # reverse for horizontal bar
    counts = [n for _, n in s.top_counties][::-1]
    fig = go.Figure(go.Bar(
        x=counts, y=counties, orientation="h",
        marker_color="rgb(148,103,189)",
        hovertemplate="%{y}: %{x}<extra></extra>",
        name="Top counties",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Events",
        yaxis_title="",
        **_LAYOUT_DEFAULTS,
    )
    return fig.to_json()


def _empty_figure(title: str) -> str:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(text="No data", showarrow=False, xref="paper", yref="paper",
                       x=0.5, y=0.5, font=dict(size=14, color="#888"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=title, **_LAYOUT_DEFAULTS)
    return fig.to_json()
