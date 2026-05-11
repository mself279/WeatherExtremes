"""GEV distribution math: PDF, CDF, quantile, log-likelihood.

Convention follows Coles (2001) and most academic references: the shape
parameter ξ > 0 is Fréchet (heavy tail), ξ < 0 is Weibull (bounded above),
ξ = 0 is the Gumbel limit. SciPy's ``genextreme`` uses the opposite sign
convention (its ``c`` = -ξ), which is why we don't simply call it directly.

Pure numpy — no scipy dependency.
"""
from __future__ import annotations

import numpy as np

# Threshold below which we treat ξ as zero and use the Gumbel limit.
_SHAPE_EPS = 1e-10


def _z(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    return (np.asarray(x, dtype=float) - loc) / scale


def gev_logpdf(
    x: np.ndarray, loc: float, scale: float, shape: float
) -> np.ndarray:
    """Log probability density of GEV(μ=loc, σ=scale, ξ=shape).

    Returns ``-inf`` outside the support.
    """
    x = np.atleast_1d(x).astype(float)
    if scale <= 0:
        return np.full_like(x, -np.inf)

    z = _z(x, loc, scale)
    out = np.full_like(x, -np.inf, dtype=float)

    if abs(shape) < _SHAPE_EPS:
        out = -np.log(scale) - z - np.exp(-z)
        return out

    t = 1.0 + shape * z
    valid = t > 0
    if np.any(valid):
        t_v = t[valid]
        log_t = np.log(t_v)
        out[valid] = -np.log(scale) - (1.0 + 1.0 / shape) * log_t - t_v ** (-1.0 / shape)
    return out


def gev_pdf(
    x: np.ndarray, loc: float, scale: float, shape: float
) -> np.ndarray:
    """Probability density."""
    return np.exp(gev_logpdf(x, loc, scale, shape))


def gev_cdf(
    x: np.ndarray, loc: float, scale: float, shape: float
) -> np.ndarray:
    """Cumulative distribution function.

    For ξ > 0 (Fréchet), values below the lower support boundary map to 0.
    For ξ < 0 (Weibull), values above the upper support boundary map to 1.
    """
    x = np.atleast_1d(x).astype(float)
    if scale <= 0:
        return np.full_like(x, np.nan)

    z = _z(x, loc, scale)

    if abs(shape) < _SHAPE_EPS:
        return np.exp(-np.exp(-z))

    t = 1.0 + shape * z
    out = np.empty_like(x, dtype=float)
    valid = t > 0
    out[valid] = np.exp(-(t[valid] ** (-1.0 / shape)))
    if shape > 0:
        # Below lower bound (μ - σ/ξ): F = 0
        out[~valid] = 0.0
    else:
        # Above upper bound: F = 1
        out[~valid] = 1.0
    return out


def gev_quantile(
    p: np.ndarray, loc: float, scale: float, shape: float
) -> np.ndarray:
    """Inverse CDF: x such that F(x) = p.

    p must be in (0, 1).
    """
    p = np.atleast_1d(p).astype(float)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("Quantile probabilities must be in (0, 1).")

    if abs(shape) < _SHAPE_EPS:
        return loc - scale * np.log(-np.log(p))

    return loc + scale * ((-np.log(p)) ** (-shape) - 1.0) / shape


def gev_return_level(
    return_period_years: np.ndarray | float,
    loc: float,
    scale: float,
    shape: float,
) -> np.ndarray:
    """T-year return level: the value exceeded on average once every T years.

    For annual maxima, F(x_T) = 1 - 1/T, so x_T = quantile(1 - 1/T).
    """
    T = np.atleast_1d(return_period_years).astype(float)
    if np.any(T <= 1):
        raise ValueError("return_period_years must be > 1")
    return gev_quantile(1.0 - 1.0 / T, loc, scale, shape)


def gev_log_likelihood(
    maxima: np.ndarray, loc: float, scale: float, shape: float
) -> float:
    """Sum of log-PDFs. Returns ``-inf`` if any observation is outside the support."""
    x = np.asarray(maxima, dtype=float)
    if scale <= 0:
        return -np.inf

    z = _z(x, loc, scale)
    n = len(x)

    if abs(shape) < _SHAPE_EPS:
        return -n * np.log(scale) - float(np.sum(z)) - float(np.sum(np.exp(-z)))

    t = 1.0 + shape * z
    if np.any(t <= 0):
        return -np.inf
    log_t = np.log(t)
    return float(
        -n * np.log(scale)
        - (1.0 + 1.0 / shape) * np.sum(log_t)
        - np.sum(t ** (-1.0 / shape))
    )
