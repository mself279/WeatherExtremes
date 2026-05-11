"""Tests for the GHCN exploration summary module (no plotly required)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.services.ghcn.summarize import (
    annual_extremes,
    data_completeness,
    threshold_frequencies,
    value_distribution,
)


def _synthetic_raw(
    element: str, dates: pd.DatetimeIndex, values
) -> pd.DataFrame:
    return pd.DataFrame({
        "ID": "TEST",
        "DATE": dates.strftime("%Y%m%d"),
        "ELEMENT": element,
        "VALUE": values,
        "M_FLAG": "",
        "Q_FLAG": "",
        "S_FLAG": "",
        "OBS_TIME": "",
    })


def test_completeness_marks_missing_years():
    # Two complete years 2000-2001, then a gap, then partial 2003
    dates = pd.concat([
        pd.Series(pd.date_range("2000-01-01", "2001-12-31", freq="D")),
        pd.Series(pd.date_range("2003-06-01", "2003-12-31", freq="D")),
    ])
    raw = _synthetic_raw("PRCP", pd.DatetimeIndex(dates), np.zeros(len(dates), dtype=int))
    s = data_completeness(raw, "PRCP")
    assert s.years == [2000, 2001, 2002, 2003]
    assert s.counts[2] == 0  # 2002 is the gap
    assert s.pct[0] >= 99 and s.pct[1] >= 99
    assert 50 < s.pct[3] < 65  # ~7 of 12 months in 2003


def test_value_distribution_threshold_counts_match_raw_filter():
    dates = pd.date_range("2010-01-01", "2010-12-31", freq="D")
    vals = np.zeros(len(dates), dtype=int)
    # 800 raw = ~3.15", 1500 raw = ~5.91"
    vals[10] = 800
    vals[100] = 1500
    vals[200] = 1500
    raw = _synthetic_raw("PRCP", dates, vals)
    s = value_distribution(raw, "PRCP", thresholds=[3, 5], operator=">")
    assert s.threshold_counts[3.0] == 3  # 800, 1500, 1500 all > 3"
    assert s.threshold_counts[5.0] == 2  # only the two 1500s > 5"
    assert s.n == len(dates)
    assert s.spec.display_unit == "in"


def test_annual_extremes_uses_min_for_tmin():
    dates = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    vals = np.full(len(dates), 100, dtype=int)  # 10 °C baseline
    vals[5] = -200      # 2000: -20 °C  (-4 °F)
    vals[400] = -150    # 2001: -15 °C  (5 °F)
    vals[800] = -100    # 2002: -10 °C  (14 °F)
    raw = _synthetic_raw("TMIN", dates, vals)
    s = annual_extremes(raw, "TMIN")
    assert s.direction == "min"
    assert s.years == [2000, 2001, 2002]
    # Display values are in °F: -20°C → -4°F, -15°C → 5°F, -10°C → 14°F
    assert abs(s.values[0] - (-4)) < 0.5
    assert abs(s.values[2] - 14) < 0.5


def test_annual_extremes_uses_max_for_tmax():
    dates = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    vals = np.full(len(dates), 100, dtype=int)  # 10 °C baseline
    vals[200] = 400   # 2000: 40 °C = 104 °F
    vals[500] = 380   # 2001: 38 °C = 100.4 °F
    raw = _synthetic_raw("TMAX", dates, vals)
    s = annual_extremes(raw, "TMAX")
    assert s.direction == "max"
    assert abs(s.values[0] - 104) < 1
    assert abs(s.values[1] - 100.4) < 1


def test_threshold_frequency_per_year_matches_expectation():
    # Build 3 years of TMAX, with deliberate hot days in each year
    years = [2010, 2011, 2012]
    frames = []
    for y in years:
        d = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
        v = np.full(len(d), 200, dtype=int)  # 20°C = 68°F baseline
        # Insert hot days; raw is tenths of °C
        # 95°F = 35°C = 350; 100°F = 37.78°C ≈ 378
        if y == 2010:
            v[180:185] = 360   # 5 days > 95°F, < 100°F
        elif y == 2011:
            v[180:188] = 380   # 8 days > 95°F AND > 100°F
        else:  # 2012
            v[180:182] = 360   # 2 days > 95°F
        frames.append(_synthetic_raw("TMAX", d, v))
    raw = pd.concat(frames, ignore_index=True)
    s = threshold_frequencies(raw, "TMAX", thresholds=[95, 100], operator=">")
    assert s.years == [2010, 2011, 2012]
    over_95 = s.counts_per_threshold[95.0]
    over_100 = s.counts_per_threshold[100.0]
    assert over_95 == [5, 8, 2]
    assert over_100 == [0, 8, 0]
