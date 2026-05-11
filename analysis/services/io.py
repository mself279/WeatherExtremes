"""CSV parsing for event input files.

The expected format is a header row with at least a numeric ``Event_Date``
column expressed as fractional years (e.g. ``1938.2466``) plus one or more
0/1 indicator columns (e.g. ``Over3``, ``Over4``) marking dates on which an
event occurred. The user picks which indicator column to analyze.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd


class InputFileError(ValueError):
    """Raised when an uploaded file cannot be parsed as event input."""


@dataclass(frozen=True)
class ParsedInput:
    """Result of parsing an input file."""

    raw: pd.DataFrame
    indicator_columns: list[str]


_REQUIRED_COLUMN = "Event_Date"


def parse_input(source: str | Path | IO[bytes] | IO[str]) -> ParsedInput:
    """Read a CSV and return the raw DataFrame plus the list of 0/1 columns.

    Parameters
    ----------
    source:
        File path, file-like object, or anything pandas.read_csv accepts.
    """
    try:
        df = pd.read_csv(source)
    except Exception as exc:  # pragma: no cover - pandas raises many types
        raise InputFileError(f"Could not read CSV: {exc}") from exc

    if _REQUIRED_COLUMN not in df.columns:
        raise InputFileError(
            f"Input file is missing required column '{_REQUIRED_COLUMN}'."
        )

    if not pd.api.types.is_numeric_dtype(df[_REQUIRED_COLUMN]):
        raise InputFileError(
            f"Column '{_REQUIRED_COLUMN}' must be numeric (fractional years)."
        )

    df[_REQUIRED_COLUMN] = df[_REQUIRED_COLUMN].astype(float).round(4)

    indicator_cols: list[str] = []
    for col in df.columns:
        if col == _REQUIRED_COLUMN:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            uniques = pd.unique(series.dropna())
            if len(uniques) and set(uniques.astype(int)).issubset({0, 1}):
                indicator_cols.append(col)

    return ParsedInput(raw=df, indicator_columns=indicator_cols)


def event_dates(
    df: pd.DataFrame, indicator_column: str
) -> np.ndarray:
    """Return the sorted unique fractional-year dates where the indicator is 1."""
    if indicator_column not in df.columns:
        raise InputFileError(f"Column '{indicator_column}' not found in input.")
    mask = df[indicator_column].fillna(0).astype(float) > 0
    dates = (
        df.loc[mask, _REQUIRED_COLUMN]
        .astype(float)
        .round(4)
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )
    return dates
