"""Convert raw GHCN per-station data into the indicator-column CSV format
that the existing analysis pipeline already consumes.

Output schema::

    Event_Date, <ThresholdCol>, <ThresholdCol>, ...

with one row per day in the user's selected date range. Each
``<ThresholdCol>`` is a 0/1 indicator named via
:func:`elements.threshold_column_name` — for example ``Over3``,
``Over90``, ``Under0``, ``Over70``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .elements import (
    ELEMENTS,
    ElementSpec,
    operator_apply,
    threshold_column_name,
)


@dataclass
class TransformResult:
    df: pd.DataFrame                # the indicator-column dataframe (Event_Date + Over*)
    threshold_columns: list[str]    # generated indicator column names
    metadata: dict                  # human-readable summary for storage on Dataset


def _date_to_fractional_year(d: pd.Timestamp) -> float:
    """Convert a Timestamp to a fractional year (rounded to 4 decimals).

    Matches the convention already used elsewhere in the app: a date like
    1938-04-04 becomes 1938.2548.
    """
    year = d.year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    frac = (d - start).total_seconds() / (end - start).total_seconds()
    return round(year + frac, 4)


def transform_to_indicators(
    raw_df: pd.DataFrame,
    *,
    element: str,
    thresholds: Sequence[float],
    operator: str = ">",
    start_year: int | None = None,
    end_year: int | None = None,
    drop_quality_failed: bool = True,
) -> TransformResult:
    """Filter & transform a raw GHCN station dataframe into indicator columns.

    Parameters
    ----------
    raw_df:
        Output of :func:`analysis.services.ghcn.formats.read_station_csv`.
    element:
        Element code (``"PRCP"``, ``"TMAX"``, ...). Must be in :data:`ELEMENTS`.
    thresholds:
        Threshold values in **display units** (e.g. inches, °F, mph).
    operator:
        One of ``">", ">=", "<", "<="``.
    start_year, end_year:
        Inclusive year bounds. ``None`` = use whatever the data covers.
    drop_quality_failed:
        Drop rows whose Q_FLAG is non-blank (i.e. flagged by NOAA quality control).
    """
    if element not in ELEMENTS:
        raise ValueError(f"Unsupported element: {element!r}")
    spec: ElementSpec = ELEMENTS[element]
    if not thresholds:
        raise ValueError("At least one threshold is required.")

    df = raw_df.loc[raw_df["ELEMENT"] == element].copy()
    if df.empty:
        raise ValueError(f"Station has no rows for element {element}.")

    if drop_quality_failed and "Q_FLAG" in df.columns:
        df = df.loc[df["Q_FLAG"].fillna("").astype(str).str.strip() == ""]

    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["DATE"])

    if start_year is not None:
        df = df.loc[df["DATE"].dt.year >= int(start_year)]
    if end_year is not None:
        df = df.loc[df["DATE"].dt.year <= int(end_year)]

    if df.empty:
        raise ValueError(
            f"No {element} rows remain after filtering "
            f"(quality, {start_year}–{end_year})."
        )

    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["VALUE"])

    actual_start = df["DATE"].min()
    actual_end = df["DATE"].max()

    # Build a complete daily grid so days with no observation become explicit
    # zeros (no event), not implicit NaNs that would skew the indicator.
    grid = pd.DataFrame({"DATE": pd.date_range(actual_start, actual_end, freq="D")})
    df_one_per_day = df.groupby("DATE", as_index=False)["VALUE"].max()
    merged = grid.merge(df_one_per_day, on="DATE", how="left")

    # Fractional-year Event_Date column.
    merged["Event_Date"] = merged["DATE"].apply(_date_to_fractional_year)

    # Build indicator columns directly in raw units (avoids floating-point
    # round-trip noise around the threshold edge).
    threshold_cols: list[str] = []
    events_per_threshold: dict[str, int] = {}
    for thr_display in thresholds:
        thr_raw = spec.display_to_raw(float(thr_display))
        col = threshold_column_name(operator, float(thr_display))
        merged[col] = operator_apply(operator, merged["VALUE"], thr_raw)
        threshold_cols.append(col)
        events_per_threshold[col] = int(merged[col].sum())

    # Final output: Event_Date + indicators, sorted ascending.
    out = (
        merged[["Event_Date", *threshold_cols]]
        .sort_values("Event_Date")
        .reset_index(drop=True)
    )

    # Convert min/max raw values to display units for the metadata summary.
    raw_values = merged["VALUE"].dropna().to_numpy()
    if raw_values.size:
        raw_min = float(raw_values.min())
        raw_max = float(raw_values.max())
        display_min = float(spec.raw_to_display(raw_min))
        display_max = float(spec.raw_to_display(raw_max))
    else:
        raw_min = raw_max = display_min = display_max = float("nan")

    metadata = {
        "element": element,
        "operator": operator,
        "thresholds_display": [float(t) for t in thresholds],
        "thresholds_raw": [spec.display_to_raw(float(t)) for t in thresholds],
        "display_unit": spec.display_unit,
        "raw_unit": spec.raw_unit,
        "rows": int(len(out)),
        "observations_with_value": int((~merged["VALUE"].isna()).sum()),
        "events_per_threshold": events_per_threshold,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "display_min": display_min,
        "display_max": display_max,
        "actual_start": actual_start.strftime("%Y-%m-%d"),
        "actual_end": actual_end.strftime("%Y-%m-%d"),
        "drop_quality_failed": drop_quality_failed,
    }

    return TransformResult(df=out, threshold_columns=threshold_cols, metadata=metadata)
