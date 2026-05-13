"""Parse the NCEI Storm Events CSV response.

The interactive endpoint returns a CSV with the standard Storm Events columns.
Some columns are not always present (e.g. ``TOR_F_SCALE`` only when the user
queried for tornadoes), so the parser tolerates missing columns and returns a
normalized DataFrame with a stable column set.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


_DATE_FORMATS = (
    "%d-%b-%Y %H:%M:%S",  # NCEI interactive endpoint: "11-MAY-2011 14:50:00"
    "%d-%b-%y %H:%M:%S",  # 2-digit year variant
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)
_DAMAGE_RE = re.compile(r"^\$?\s*([\d.]+)\s*([KMBT]?)$", re.IGNORECASE)
_DAMAGE_MULT = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


@dataclass
class ParsedEvents:
    df: pd.DataFrame                # normalized DataFrame; see _CANONICAL_COLS
    raw_rows: int                   # how many CSV rows we read
    columns_seen: list[str]         # original column names from NCEI


_CANONICAL_COLS = [
    "event_id",
    "event_type",
    "state",
    "state_fips",
    "cz_name",
    "cz_fips",
    "begin_datetime",
    "end_datetime",
    "year",
    "month",
    "magnitude",
    "magnitude_type",
    "tor_f_scale",
    "category",
    "injuries_direct",
    "deaths_direct",
    "damage_property_usd",
    "damage_crops_usd",
    "begin_lat",
    "begin_lon",
    "end_lat",
    "end_lon",
]


def _parse_damage(val: object) -> float:
    """Parse strings like ``$1.5M``, ``$500K``, ``$0``, ``0.00K`` to USD float."""
    if val is None:
        return float("nan")
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return float("nan")
    m = _DAMAGE_RE.match(s)
    if not m:
        return float("nan")
    base, suffix = m.group(1), m.group(2).upper()
    try:
        return float(base) * _DAMAGE_MULT[suffix]
    except (KeyError, ValueError):
        return float("nan")


def _parse_datetime_series(s: pd.Series) -> pd.Series:
    """Try a few formats; fall back to pandas inference."""
    for fmt in _DATE_FORMATS:
        out = pd.to_datetime(s, format=fmt, errors="coerce")
        if out.notna().mean() > 0.9:
            return out
    return pd.to_datetime(s, errors="coerce", utc=False)


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype("Int64")


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_events_csv(data: bytes | str) -> ParsedEvents:
    """Parse NCEI's CSV bytes into a normalized DataFrame.

    ``df`` always has the columns listed in :data:`_CANONICAL_COLS`. Columns
    NCEI didn't return are filled with NaN or empty strings.
    """
    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="replace")
    else:
        text = data

    if not text.strip():
        return ParsedEvents(df=_empty(), raw_rows=0, columns_seen=[])

    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    raw_rows = int(len(df))
    columns_seen = list(df.columns)

    # Normalize column names to UPPER_SNAKE then map to canonical set.
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

    out = pd.DataFrame(index=df.index)

    def col(*names: str) -> pd.Series:
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series([""] * len(df), index=df.index)

    out["event_id"] = col("EVENT_ID")
    out["event_type"] = col("EVENT_TYPE")
    out["state"] = col("STATE")
    out["state_fips"] = col("STATE_FIPS")
    out["cz_name"] = col("CZ_NAME", "COUNTY_NAME")
    out["cz_fips"] = col("CZ_FIPS")

    begin_raw = col("BEGIN_DATE_TIME", "BEGIN_TIME")
    end_raw = col("END_DATE_TIME", "END_TIME")
    out["begin_datetime"] = _parse_datetime_series(begin_raw)
    out["end_datetime"] = _parse_datetime_series(end_raw)
    # Derive year and month from begin_datetime; fall back to YEAR column.
    parsed_year = out["begin_datetime"].dt.year.astype("Int64")
    fallback_year = _to_int(col("YEAR"))
    out["year"] = parsed_year.where(parsed_year.notna(), fallback_year)
    out["month"] = out["begin_datetime"].dt.month.astype("Int64")

    out["magnitude"] = _to_float(col("MAGNITUDE"))
    out["magnitude_type"] = col("MAGNITUDE_TYPE")
    out["tor_f_scale"] = col("TOR_F_SCALE")
    out["category"] = col("CATEGORY")

    out["injuries_direct"] = _to_int(col("INJURIES_DIRECT"))
    out["deaths_direct"] = _to_int(col("DEATHS_DIRECT"))

    out["damage_property_usd"] = col("DAMAGE_PROPERTY").apply(_parse_damage)
    out["damage_crops_usd"] = col("DAMAGE_CROPS").apply(_parse_damage)

    out["begin_lat"] = _to_float(col("BEGIN_LAT"))
    out["begin_lon"] = _to_float(col("BEGIN_LON"))
    out["end_lat"] = _to_float(col("END_LAT"))
    out["end_lon"] = _to_float(col("END_LON"))

    # Ensure all canonical columns exist even if data was empty.
    for c in _CANONICAL_COLS:
        if c not in out.columns:
            out[c] = np.nan

    return ParsedEvents(
        df=out[_CANONICAL_COLS].copy(),
        raw_rows=raw_rows,
        columns_seen=columns_seen,
    )


def _empty() -> pd.DataFrame:
    """Empty DataFrame with the canonical column set."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in _CANONICAL_COLS})
