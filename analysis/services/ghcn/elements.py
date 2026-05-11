"""GHCN element metadata: units, conversions, default thresholds.

Each element has a ``raw`` unit (what NOAA stores in the .csv files) and a
``display`` unit (what users actually want to think in: inches, °F, mph).
The forms and templates use the display unit; conversion to raw happens
just before the threshold comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

_TENTHS_MM_TO_INCHES = 0.1 / 25.4    # 1 (tenths of mm) -> 0.003937 in
_MM_TO_INCHES = 1.0 / 25.4
_MS_TO_MPH = 2.2369362921           # 1 m/s -> 2.2369... mph


def _tenths_c_to_f(v: float) -> float:
    return (v * 0.1) * 9.0 / 5.0 + 32.0


def _f_to_tenths_c(v: float) -> float:
    return ((v - 32.0) * 5.0 / 9.0) / 0.1


def _tenths_ms_to_mph(v: float) -> float:
    return (v * 0.1) * _MS_TO_MPH


def _mph_to_tenths_ms(v: float) -> float:
    return (v / _MS_TO_MPH) / 0.1


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElementSpec:
    code: str
    label: str
    raw_unit: str
    display_unit: str
    raw_to_display: Callable[[float], float]
    display_to_raw: Callable[[float], float]
    default_thresholds: tuple[float, ...]
    default_operator: str  # ">" or "<"
    # Sensible bucket width for the GEV bucket diagnostic in display units.
    # Chosen larger than the data's storage resolution to avoid spurious
    # spikes — temperature in GHCN is stored as tenths of °C, which becomes
    # roughly 0.18°F resolution after conversion, so 2°F buckets smooth that
    # discreteness out without losing the distribution shape.
    default_bucket_width: float = 0.5


# Order matters for "preferred default" logic in the import form: the first
# element that the station has wins. Keep PRCP first because it's the canonical
# use case the app was designed around.
ELEMENTS: dict[str, ElementSpec] = {
    "PRCP": ElementSpec(
        code="PRCP",
        label="Precipitation",
        raw_unit="tenths of mm",
        display_unit="in",
        raw_to_display=lambda v: v * _TENTHS_MM_TO_INCHES,
        display_to_raw=lambda v: v / _TENTHS_MM_TO_INCHES,
        default_thresholds=(3.0, 4.0, 5.0, 6.0),
        default_operator=">",
        default_bucket_width=0.5,
    ),
    "TMAX": ElementSpec(
        code="TMAX",
        label="Maximum temperature",
        raw_unit="tenths of °C",
        display_unit="°F",
        raw_to_display=_tenths_c_to_f,
        display_to_raw=_f_to_tenths_c,
        default_thresholds=(90.0, 95.0, 100.0, 105.0),
        default_operator=">",
        default_bucket_width=2.0,
    ),
    "TMIN": ElementSpec(
        code="TMIN",
        label="Minimum temperature",
        raw_unit="tenths of °C",
        display_unit="°F",
        raw_to_display=_tenths_c_to_f,
        display_to_raw=_f_to_tenths_c,
        default_thresholds=(30.0, 20.0, 10.0, 0.0),
        default_operator="<",
        default_bucket_width=2.0,
    ),
    "TAVG": ElementSpec(
        code="TAVG",
        label="Average temperature",
        raw_unit="tenths of °C",
        display_unit="°F",
        raw_to_display=_tenths_c_to_f,
        display_to_raw=_f_to_tenths_c,
        default_thresholds=(80.0, 85.0, 90.0),
        default_operator=">",
        default_bucket_width=2.0,
    ),
    "AWND": ElementSpec(
        code="AWND",
        label="Average wind speed",
        raw_unit="tenths of m/s",
        display_unit="mph",
        raw_to_display=_tenths_ms_to_mph,
        display_to_raw=_mph_to_tenths_ms,
        default_thresholds=(50.0, 60.0, 70.0, 80.0, 90.0, 100.0),
        default_operator=">",
        default_bucket_width=5.0,
    ),
    "WSF2": ElementSpec(
        code="WSF2",
        label="Fastest 2-minute wind speed",
        raw_unit="tenths of m/s",
        display_unit="mph",
        raw_to_display=_tenths_ms_to_mph,
        display_to_raw=_mph_to_tenths_ms,
        default_thresholds=(50.0, 60.0, 70.0, 80.0, 90.0, 100.0),
        default_operator=">",
        default_bucket_width=5.0,
    ),
    "WSF5": ElementSpec(
        code="WSF5",
        label="Fastest 5-second wind speed",
        raw_unit="tenths of m/s",
        display_unit="mph",
        raw_to_display=_tenths_ms_to_mph,
        display_to_raw=_mph_to_tenths_ms,
        default_thresholds=(50.0, 60.0, 70.0, 80.0, 90.0, 100.0),
        default_operator=">",
        default_bucket_width=5.0,
    ),
    "SNOW": ElementSpec(
        code="SNOW",
        label="Snowfall",
        raw_unit="mm",
        display_unit="in",
        raw_to_display=lambda v: v * _MM_TO_INCHES,
        display_to_raw=lambda v: v / _MM_TO_INCHES,
        default_thresholds=(6.0, 12.0, 18.0, 24.0),
        default_operator=">",
        default_bucket_width=1.0,
    ),
    "SNWD": ElementSpec(
        code="SNWD",
        label="Snow depth",
        raw_unit="mm",
        display_unit="in",
        raw_to_display=lambda v: v * _MM_TO_INCHES,
        display_to_raw=lambda v: v / _MM_TO_INCHES,
        default_thresholds=(6.0, 12.0, 18.0, 24.0),
        default_operator=">",
        default_bucket_width=1.0,
    ),
}

# Operator → column name prefix (must produce valid Python identifier chars).
_OP_LABEL = {">": "Over", ">=": "AtLeast", "<": "Under", "<=": "AtMost"}

OPERATOR_CHOICES: tuple[tuple[str, str], ...] = (
    (">", "greater than (>)"),
    (">=", "greater than or equal (≥)"),
    ("<", "less than (<)"),
    ("<=", "less than or equal (≤)"),
)


def threshold_column_name(operator: str, threshold: float) -> str:
    """Build a stable column name like 'Over3' or 'Under3p5' from a threshold.

    Decimals use 'p' as a separator so the result is a valid CSV column name
    that round-trips through pandas without quoting.
    """
    if operator not in _OP_LABEL:
        raise ValueError(f"Unknown operator: {operator!r}")
    if threshold == int(threshold):
        thr_str = str(int(threshold))
    else:
        thr_str = repr(threshold).replace(".", "p").rstrip("0").rstrip("p")
        if not thr_str:
            thr_str = "0"
    if thr_str.startswith("-"):
        thr_str = "neg" + thr_str[1:]
    return f"{_OP_LABEL[operator]}{thr_str}"


def operator_apply(operator: str, series, threshold_raw: float):
    """Apply ``operator`` to a pandas Series in raw units; return a 0/1 int array."""
    import numpy as np

    if operator == ">":
        flag = series > threshold_raw
    elif operator == ">=":
        flag = series >= threshold_raw
    elif operator == "<":
        flag = series < threshold_raw
    elif operator == "<=":
        flag = series <= threshold_raw
    else:
        raise ValueError(f"Unknown operator: {operator!r}")
    # NaN comparisons -> False; cast to int 0/1.
    return np.where(flag.fillna(False), 1, 0)


def supported_codes() -> list[str]:
    return list(ELEMENTS.keys())
