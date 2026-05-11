"""Annual block maxima (or minima) extraction from a GHCN station.

For v1, GEV requires a Dataset whose ``source_station`` is set — we read
the cached raw daily ``.csv.gz`` from ``media/ghcn_cache/by_station/``,
filter to the chosen element, and reduce to one extreme per calendar year.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..ghcn import fetcher as ghcn_fetcher
from ..ghcn.elements import ELEMENTS, ElementSpec
from ..ghcn.formats import read_station_csv


@dataclass(frozen=True)
class AnnualMaxima:
    element: str
    spec: ElementSpec
    direction: str        # "max" or "min"
    years: np.ndarray     # 1-D int array
    raw: np.ndarray       # in raw GHCN units (e.g. tenths of mm)
    display: np.ndarray   # in display units (inches, °F, mph)


class ExtractError(ValueError):
    """Raised when the source dataset can't yield annual maxima."""


def extract_annual_maxima(
    *,
    station_id: str,
    element: str,
    cache_dir: Path,
    direction: str = "max",
    start_year: int | None = None,
    end_year: int | None = None,
    drop_quality_failed: bool = True,
    min_obs_per_year: int = 200,
) -> AnnualMaxima:
    """Read the cached station file and compute one extreme per year.

    Parameters
    ----------
    station_id:
        GHCN station ID (11 chars). The cached ``.csv.gz`` must already be
        on disk from a previous import; if missing this will fetch it.
    element:
        GHCN element code (PRCP, TMAX, ...). Must be in :data:`ELEMENTS`.
    cache_dir:
        ``MEDIA_ROOT/ghcn_cache``.
    direction:
        ``"max"`` (default) or ``"min"``. For TMIN-style elements, the
        block extreme is the minimum of the year.
    start_year, end_year:
        Inclusive year bounds. ``None`` = use whatever the data covers.
    drop_quality_failed:
        Drop rows whose Q_FLAG is non-blank.
    min_obs_per_year:
        Skip years with fewer than this many observations (incomplete years
        skew the block extreme).
    """
    if element not in ELEMENTS:
        raise ExtractError(f"Unsupported element: {element!r}")
    if direction not in {"max", "min"}:
        raise ExtractError(f"direction must be 'max' or 'min', got {direction!r}")

    spec = ELEMENTS[element]
    raw_path = ghcn_fetcher.fetch_station(station_id, cache_dir=cache_dir)
    raw_df = read_station_csv(raw_path)
    df = raw_df.loc[raw_df["ELEMENT"] == element].copy()
    if df.empty:
        raise ExtractError(f"Station {station_id} has no {element} rows.")

    if drop_quality_failed and "Q_FLAG" in df.columns:
        df = df.loc[df["Q_FLAG"].fillna("").astype(str).str.strip() == ""]

    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["DATE"])
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["VALUE"])
    df["YEAR"] = df["DATE"].dt.year

    if start_year is not None:
        df = df.loc[df["YEAR"] >= int(start_year)]
    if end_year is not None:
        df = df.loc[df["YEAR"] <= int(end_year)]
    if df.empty:
        raise ExtractError(
            f"No {element} rows survived filtering "
            f"({start_year}–{end_year}, quality)."
        )

    yr_count = df.groupby("YEAR").size()
    yr_keep = yr_count[yr_count >= min_obs_per_year].index
    df = df.loc[df["YEAR"].isin(yr_keep)]
    if df.empty:
        raise ExtractError(
            f"No years in {start_year}–{end_year} had at least "
            f"{min_obs_per_year} observations of {element}."
        )

    if direction == "max":
        block = df.groupby("YEAR")["VALUE"].max()
    else:
        block = df.groupby("YEAR")["VALUE"].min()

    years = np.asarray(block.index, dtype=int)
    raw_vals = np.asarray(block.to_numpy(), dtype=float)
    display_vals = np.asarray(spec.raw_to_display(raw_vals), dtype=float)

    return AnnualMaxima(
        element=element,
        spec=spec,
        direction=direction,
        years=years,
        raw=raw_vals,
        display=display_vals,
    )
