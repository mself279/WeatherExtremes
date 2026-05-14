"""Convert filtered Storm Events into the indicator-column CSV the KDE pipeline understands.

The pipeline expects a CSV with an ``Event_Date`` column (fractional years) and
one or more 0/1 indicator columns. For Storm Events, every event in the
filtered set is by definition an event — the indicator column is simply 1 on
days that had a matching event and 0 on days that didn't. Magnitude /
threshold filtering already happened upstream in the NCEI URL.
"""
from __future__ import annotations

import pandas as pd

from .event_types import EventTypeSpec
from .fetcher import FilterParams


def _date_to_fractional_year(d: pd.Timestamp) -> float:
    year = d.year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    frac = (d - start).total_seconds() / (end - start).total_seconds()
    return round(year + frac, 4)


def indicator_column_name(spec: EventTypeSpec, params: FilterParams) -> str:
    """Build a descriptive 0/1 indicator column name.

    Examples:
    * Tornado + tornfilter=3 → ``TornadoF3plus``
    * Hail + hailfilter=1.50 → ``Hail1p5in``
    * Thunderstorm Wind + windfilter=075 → ``ThunderstormWind75kt``
    * Flash Flood (no magnitude) → ``FlashFlood``
    """
    base = spec.label.replace(" ", "").replace("/", "")
    if spec.magnitude_field == "tornfilter" and params.tornfilter not in ("", "0"):
        return f"{base}F{params.tornfilter}plus"
    if spec.magnitude_field == "hailfilter" and params.hailfilter not in ("", "0.00"):
        v = params.hailfilter.rstrip("0").rstrip(".") or "0"
        v = v.replace(".", "p")
        return f"{base}{v}in"
    if spec.magnitude_field == "windfilter" and params.windfilter not in ("", "000"):
        try:
            kt = int(params.windfilter)
            return f"{base}{kt}kt"
        except ValueError:
            return base
    return base


def dataset_name(state: str, spec: EventTypeSpec, params: FilterParams) -> str:
    base = f"{state} {spec.label}"
    if spec.magnitude_field == "tornfilter" and params.tornfilter not in ("", "0"):
        base += f" F{params.tornfilter}+"
    elif spec.magnitude_field == "hailfilter" and params.hailfilter not in ("", "0.00"):
        v = params.hailfilter.rstrip("0").rstrip(".") or "0"
        base += f' {v}"'
    elif spec.magnitude_field == "windfilter" and params.windfilter not in ("", "000"):
        try:
            base += f" {int(params.windfilter)}kt+"
        except ValueError:
            pass
    base += f" ({params.begin_year}-{params.end_year})"
    return base


def build_indicator_csv(
    events_df: pd.DataFrame,
    *,
    begin_year: int,
    end_year: int,
    indicator_name: str,
) -> pd.DataFrame:
    """Daily grid for ``begin_year`` to ``end_year`` with a 0/1 indicator column.

    Multiple events on the same date collapse to a single 1 (the KDE pipeline
    dedups event dates anyway, so this just makes the CSV smaller).
    """
    grid = pd.date_range(f"{begin_year}-01-01", f"{end_year}-12-31", freq="D")
    out = pd.DataFrame({"DATE": grid})

    event_days = (
        events_df["begin_datetime"]
        .dropna()
        .dt.normalize()
        .drop_duplicates()
    )

    out[indicator_name] = out["DATE"].isin(event_days).astype(int)
    out["Event_Date"] = out["DATE"].apply(_date_to_fractional_year)
    return out[["Event_Date", indicator_name]]
