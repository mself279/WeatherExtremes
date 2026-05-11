"""Bootstrap confidence bands for the occurrence-rate estimator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kde import occurrence_rate, reflect_events


@dataclass(frozen=True)
class ConfidenceBand:
    grid: np.ndarray
    point_estimate: np.ndarray
    p05: np.ndarray
    p10: np.ndarray
    p25: np.ndarray
    p50: np.ndarray
    p75: np.ndarray
    p90: np.ndarray
    p95: np.ndarray


def bootstrap_rates(
    grid: np.ndarray,
    event_dates: np.ndarray,
    bandwidth_years: float,
    iterations: int,
    rng: np.random.Generator | None = None,
    *,
    reflect_for_bias: bool = False,
    observation_start: float | None = None,
    observation_end: float | None = None,
) -> np.ndarray:
    """Resample event dates with replacement and recompute the rate each time.

    When ``reflect_for_bias`` is True, each bootstrap sample is reflected at
    the observation boundaries (Mudelsee bias correction) before the KDE so
    the confidence band stays well-defined near the edges. ``observation_start``
    and ``observation_end`` must then be provided.

    Returns an ``(iterations, len(grid))`` array of bootstrap rate curves.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if event_dates.size == 0:
        return np.zeros((iterations, len(grid)))
    if reflect_for_bias and (observation_start is None or observation_end is None):
        raise ValueError(
            "observation_start/observation_end required when reflect_for_bias=True"
        )

    rng = rng or np.random.default_rng()
    n = event_dates.size
    bias_window = 3.0 * bandwidth_years if reflect_for_bias else 0.0

    out = np.empty((iterations, len(grid)), dtype=float)
    for i in range(iterations):
        sample = rng.choice(event_dates, size=n, replace=True)
        if reflect_for_bias:
            sample = reflect_events(
                sample,
                float(observation_start),  # type: ignore[arg-type]
                float(observation_end),    # type: ignore[arg-type]
                bias_window,
            )
        out[i] = occurrence_rate(grid, sample, bandwidth_years, delete_one=False)
    return out


def confidence_band(
    grid: np.ndarray,
    point_estimate: np.ndarray,
    bootstrap_curves: np.ndarray,
) -> ConfidenceBand:
    """Reduce bootstrap curves to per-point percentiles."""
    quantiles = np.quantile(
        bootstrap_curves,
        q=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        axis=0,
    )
    return ConfidenceBand(
        grid=grid,
        point_estimate=point_estimate,
        p05=quantiles[0],
        p10=quantiles[1],
        p25=quantiles[2],
        p50=quantiles[3],
        p75=quantiles[4],
        p90=quantiles[5],
        p95=quantiles[6],
    )
