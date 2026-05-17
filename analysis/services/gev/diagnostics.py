"""Goodness-of-fit and exceedance diagnostics for a fitted GEV.

The chi-square-style bucket diagnostic mirrors the workbook's "Variance
between expected and observed" computation: bin the data into equal-width
buckets, compare observed counts to ``n · ∫_bin pdf``, summarize.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .distribution import gev_cdf, gev_quantile, gev_return_level


# ---------------------------------------------------------------------------
# Bucket diagnostic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BucketDiagnostic:
    edges: np.ndarray
    observed: np.ndarray
    expected: np.ndarray
    chi_square: float          # Σ (obs - exp)^2 / exp, ignoring expected ≈ 0
    sum_squared_error: float   # Σ (obs - exp)^2 (matches the workbook's "variance")


def bucket_diagnostic(
    maxima: np.ndarray,
    location: float,
    scale: float,
    shape: float,
    *,
    bucket_width: float,
    edge_min: float | None = None,
    edge_max: float | None = None,
    direction: str = "max",
) -> BucketDiagnostic:
    """Histogram-style observed-vs-expected diagnostic.

    For ``direction == "min"``, ``location`` is the displayed (negated) value
    and the underlying GEV was fit to ``-X``. The CDF of X at edge ``e`` is
    then ``1 - gev_cdf(-e; -location, scale, shape)`` — the same negation
    correction applied in :mod:`plots.histogram_pdf_figure`.

    ``edge_min`` / ``edge_max`` default to the rounded data range.
    """
    x = np.asarray(maxima, dtype=float)
    if edge_min is None:
        edge_min = float(np.floor(x.min() / bucket_width) * bucket_width)
    if edge_max is None:
        edge_max = float(np.ceil(x.max() / bucket_width) * bucket_width)
    if edge_max <= edge_min:
        raise ValueError("edge_max must exceed edge_min")

    edges = np.arange(edge_min, edge_max + bucket_width / 2, bucket_width)
    observed, _ = np.histogram(x, bins=edges)

    if direction == "max":
        cdf_at_edges = gev_cdf(edges, location, scale, shape)
    else:
        # F_X(e) = 1 - F_{-X}(-e); underlying μ_fit = -location.
        cdf_at_edges = 1.0 - gev_cdf(-edges, -location, scale, shape)
    expected = float(len(x)) * np.diff(cdf_at_edges)

    sse = float(np.sum((observed - expected) ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        contribs = np.where(expected > 1e-9, (observed - expected) ** 2 / expected, 0.0)
    chi2 = float(np.sum(contribs))

    return BucketDiagnostic(
        edges=edges,
        observed=observed.astype(int),
        expected=expected.astype(float),
        chi_square=chi2,
        sum_squared_error=sse,
    )


# ---------------------------------------------------------------------------
# Return periods
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReturnLevels:
    return_periods: np.ndarray  # T (years)
    levels: np.ndarray          # x_T (display units)


_DEFAULT_RETURN_PERIODS = (2, 5, 10, 25, 50, 100, 200, 500, 1000)


def return_levels(
    location: float,
    scale: float,
    shape: float,
    return_periods: Sequence[float] | None = None,
    direction: str = "max",
) -> ReturnLevels:
    """T-year return level.

    For ``direction == "max"`` (default), the T-year return level is the value
    the annual maximum exceeds on average once every T years (``F(x_T) = 1−1/T``).

    For ``direction == "min"``, the user-meaningful question is the value the
    annual minimum *falls below* once every T years (``F_X(x_T) = 1/T``).
    Translated through the negation: ``x_T = −gev_quantile(1−1/T; −location, σ, ξ)``.
    """
    T = np.asarray(
        return_periods if return_periods is not None else _DEFAULT_RETURN_PERIODS,
        dtype=float,
    )
    if direction == "max":
        levels = gev_return_level(T, location, scale, shape)
    else:
        levels = -gev_quantile(1.0 - 1.0 / T, -location, scale, shape)
    return ReturnLevels(return_periods=T, levels=np.asarray(levels))


# ---------------------------------------------------------------------------
# Exceedance probability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exceedance:
    reference_year: int
    reference_value: float          # display units
    exceedance_probability: float   # P(X > reference_value)
    expected_return_period: float   # 1 / P(X > x)


def exceedance(
    location: float, scale: float, shape: float,
    reference_value: float, reference_year: int,
    direction: str = "max",
) -> Exceedance:
    """Probability the annual extreme is "more extreme" than ``reference_value``.

    For ``direction == "max"``: ``P(X > reference_value) = 1 − F(ref)``.

    For ``direction == "min"``: ``P(X < reference_value) = F_X(ref)
    = 1 − gev_cdf(−ref; −location, σ, ξ)``.

    The reported ``exceedance_probability`` is always the user-meaningful
    "probability of an event at least as extreme as the reference."
    """
    if direction == "max":
        cdf = float(gev_cdf(np.array([reference_value]), location, scale, shape)[0])
        p = max(min(1.0 - cdf, 1.0), 0.0)
    else:
        cdf_neg = float(gev_cdf(np.array([-reference_value]), -location, scale, shape)[0])
        p = max(min(1.0 - cdf_neg, 1.0), 0.0)
    rt = float("inf") if p == 0 else 1.0 / p
    return Exceedance(
        reference_year=int(reference_year),
        reference_value=float(reference_value),
        exceedance_probability=p,
        expected_return_period=rt,
    )


# ---------------------------------------------------------------------------
# Percentiles (workbook's "percentiles for the top 5 LL" → just one MLE here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FittedQuantiles:
    probabilities: np.ndarray
    values: np.ndarray  # display units


_DEFAULT_QUANTILE_PROBS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def fitted_quantiles(
    location: float, scale: float, shape: float,
    probs: Sequence[float] | None = None,
    direction: str = "max",
) -> FittedQuantiles:
    """Quantiles of the fitted distribution of the annual extreme.

    Interpretation of ``p10``, ``p90`` etc.: ``F(x_p) = p``, i.e. probability
    ``p`` of the annual extreme being at or below ``x_p``. Same convention for
    both directions; only the math changes.
    """
    p = np.asarray(probs if probs is not None else _DEFAULT_QUANTILE_PROBS, dtype=float)
    if direction == "max":
        values = gev_quantile(p, location, scale, shape)
    else:
        # F_X(x_p) = p → 1 − gev_cdf(−x_p; −location, σ, ξ) = p
        # → −x_p = gev_quantile(1−p; −location, σ, ξ).
        values = -gev_quantile(1.0 - p, -location, scale, shape)
    return FittedQuantiles(probabilities=p, values=np.asarray(values))
