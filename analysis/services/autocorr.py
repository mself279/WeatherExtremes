"""Tau-based declustering for clustered extreme events.

Sequences of consecutive event flags are treated as belonging to a single
cluster of dependent extremes. ``decluster`` keeps only the first event of
each cluster of length ``tau + 1``.

Reference: Mudelsee, "Climate Time Series Analysis", Ch. 6.4.
"""
from __future__ import annotations

import numpy as np


def estimate_tau(indicator: np.ndarray, max_lag: int = 14) -> list[int]:
    """Estimate decorrelation lag tau from autocorrelation of the indicator.

    Returns the unique integer tau values across lags 1..max_lag.
    """
    if indicator.size <= max_lag + 1:
        return [0]

    indicator = indicator.astype(float)
    sd = indicator.std()
    if sd == 0:
        return [0]

    taus: set[int] = set()
    for lag in range(1, max_lag + 1):
        a = indicator[lag:]
        b = indicator[:-lag]
        if a.std() == 0 or b.std() == 0:
            continue
        rho = float(np.corrcoef(a, b)[0, 1])
        if rho <= 0:
            continue
        log_rho = np.log(rho)
        if log_rho == 0:
            continue
        tau = max(int(-1.0 / log_rho), 0)
        taus.add(tau)

    return sorted(taus) or [0]


def decluster(indicator: np.ndarray, tau: int) -> np.ndarray:
    """Keep the first event of each cluster; drop events within ``tau`` of it.

    A 0 indicator resets the cluster. Returns a new 0/1 array of the same
    length as ``indicator``.
    """
    if tau < 0:
        raise ValueError("tau must be >= 0")
    out = np.zeros_like(indicator, dtype=int)
    series_count = 0
    for i, flag in enumerate(indicator.astype(int)):
        if flag == 1 and series_count == 0:
            series_count = 1
        elif flag == 1 and series_count == tau + 1:
            series_count = 1
        elif flag == 1 and series_count < tau + 1:
            series_count += 1
        else:  # flag == 0
            series_count = 0
        out[i] = 1 if series_count == 1 and flag == 1 else 0
    return out
